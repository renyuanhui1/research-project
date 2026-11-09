#!/usr/bin/env python3
"""三机多目标视觉伺服：共享一份 DINO，三路感知/控制状态相互隔离。"""

import argparse
import asyncio
import copy
import datetime
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import commentjson
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import projectairsim
from projectairsim import Drone, World

import sys

# Allow both direct script execution and package imports.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()


from common import sensors as ce
from control import servo_closed_loop as servo
from common.sensors import decode_image
from perception.extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_CONFIG_DIR = project_path(PROFILE["simulator"]["scene"]).parent


class Mailbox:
    """单架无人机的最新传感器消息；不同无人机不会覆盖彼此。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.messages = {}

    def callback(self, tag):
        def receive(*items):
            with self.lock:
                self.messages[tag] = items[-1]
        return receive

    def get(self, tag):
        with self.lock:
            return self.messages.get(tag)


class MultiCameraViewer:
    """把三路 Chase RGB 组成一个实时横向窗口；仅在显式启用时创建 GUI。"""

    WINDOW = "Drone1 | Drone2 | Drone3 - Chase RGB"
    COLORS = {
        "Drone1": (255, 170, 0),
        "Drone2": (13, 184, 255),
        "Drone3": (184, 31, 255),
    }

    def __init__(self, enabled, runtimes):
        self.enabled = bool(enabled)
        self.names = [runtime.name for runtime in runtimes]
        self.frames = {}
        self.labels = {}
        if self.enabled:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW, 1440, 306)

    def update(self, runtime, rgb, label):
        if not self.enabled:
            return
        self.frames[runtime.name] = np.asarray(rgb).copy()
        self.labels[runtime.name] = str(label)

    def render(self):
        if not self.enabled:
            return
        panels = []
        for name in self.names:
            rgb = self.frames.get(name)
            if rgb is None:
                bgr = np.zeros((270, 480, 3), dtype=np.uint8)
            else:
                bgr = cv2.resize(rgb[..., :3], (480, 270), interpolation=cv2.INTER_AREA)
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            panel = cv2.copyMakeBorder(
                bgr, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
            color = self.COLORS.get(name, (255, 255, 255))
            text = f"{name}  {self.labels.get(name, 'WAIT_FRAME')}"
            cv2.putText(panel, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            panels.append(panel)
        cv2.imshow(self.WINDOW, np.hstack(panels))
        cv2.waitKey(1)

    def close(self):
        if self.enabled:
            try:
                cv2.destroyWindow(self.WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass


@dataclass
class TargetModel:
    mean: torch.Tensor
    std: torch.Tensor
    full: torch.Tensor
    roi: torch.Tensor | None


class SharedFingerprintBank:
    """一份 DINO + 每个任务各自的标准化统计和模板原型。"""

    def __init__(self, runtimes, device):
        self.device = device
        first = runtimes[0].args
        self.input_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.input_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.dino = load_model("dinov2_vits14", device,
                               weights=str(first.dino_weights), repo_dir=str(first.repo_dir))
        self.models = {}
        stats_cache = {}

        for runtime in runtimes:
            args = runtime.args
            stats_key = str(args.stats_episode.resolve())
            if stats_key not in stats_cache:
                with h5py.File(args.stats_episode, "r") as handle:
                    features = self.encode(handle["rgb"][:])
                mean = features.reshape(-1, features.shape[-1]).mean(0).to(device)
                std = features.reshape(-1, features.shape[-1]).std(0).clamp_min(1e-6).to(device)
                stats_cache[stats_key] = (mean, std)
            mean, std = stats_cache[stats_key]

            bgr = cv2.imread(str(args.template))
            if bgr is None:
                raise SystemExit(f"{runtime.name}: 读不到模板 {args.template}")
            template = np.ascontiguousarray(bgr[:, :, ::-1])[None]
            zt = self.encode(template)[0].to(device)
            zt = (zt - mean) / std
            patch_count = zt.shape[0]
            grid = int(round(patch_count ** 0.5))
            if grid * grid != patch_count:
                raise SystemExit(f"{runtime.name}: DINO patch 数不是方形网格 {patch_count}")
            if not hasattr(self, "grid"):
                self.grid = grid
                axis = torch.linspace(-1.0, 1.0, grid, device=device)
                yy, xx = torch.meshgrid(axis, axis, indexing="ij")
                self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
            elif self.grid != grid:
                raise SystemExit("三机模板的 DINO 网格大小不一致")

            full = F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0)
            roi = None
            roi_count = 0
            if args.template_roi is not None:
                x0, y0, x1, y1 = args.template_roi
                centers = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
                vv, uu = torch.meshgrid(centers, centers, indexing="ij")
                mask = ((uu.reshape(-1) >= x0) & (uu.reshape(-1) <= x1)
                        & (vv.reshape(-1) >= y0) & (vv.reshape(-1) <= y1))
                if not mask.any():
                    raise SystemExit(f"{runtime.name}: template_roi 没覆盖任何 patch")
                roi = F.normalize(F.normalize(zt[mask], dim=-1).mean(0), dim=0)
                roi_count = int(mask.sum())
            self.models[runtime.name] = TargetModel(mean, std, full, roi)
            detail = "" if roi is None else f", ROI {roi_count} patches"
            print(f"{runtime.name}: 指纹就绪 grid={grid}{detail}")

    @torch.no_grad()
    def encode(self, images, batch_size=32):
        output = []
        for start in range(0, len(images), batch_size):
            tensor = to_input_tensor(images[start:start + batch_size], 224,
                                     self.input_mean, self.input_std, self.device)
            output.append(self.dino.forward_features(tensor)["x_norm_patchtokens"].float().cpu())
        return torch.cat(output, 0)

    @torch.no_grad()
    def query(self, batch):
        """batch: [(runtime, rgb224)]，一次前向后返回各自的指纹观测。"""
        images = np.stack([rgb for _, rgb in batch])
        tensor = to_input_tensor(images, 224, self.input_mean, self.input_std, self.device)
        features = self.dino.forward_features(tensor)["x_norm_patchtokens"].float()
        results = {}
        for index, (runtime, _) in enumerate(batch):
            args = runtime.args
            target = self.models[runtime.name]
            z = (features[index] - target.mean) / target.std
            proto = target.roi if runtime.roi_active and target.roi is not None else target.full
            similarity = torch.matmul(F.normalize(z, dim=-1), proto)
            weight = torch.softmax(similarity / args.target_softmax_temp, dim=-1)
            center = torch.matmul(weight, self.patch_xy)
            mass = torch.sigmoid(
                (similarity - args.target_mass_thresh) * args.target_mass_sharpness).mean()
            results[runtime.name] = (
                float(center[0]), float(center[1]), float(mass), float(similarity.max()),
                similarity.cpu().numpy(),
            )
        return results


@dataclass
class Runtime:
    name: str
    priority: int
    args: argparse.Namespace
    target_ned: tuple
    approach_standoff: float
    crossing_entry_ned: tuple
    crossing_exit_ned: tuple
    drone: object = None
    mailbox: Mailbox = field(default_factory=Mailbox)
    status: str = "INIT"
    stop_reason: str = "timeout"
    step: int = 0
    last_rgb_ts: int = -1
    last_frame_wall: float = 0.0
    last_wall: float | None = None
    last_contact_depth_ts: int = -1
    filtered: dict = field(default_factory=lambda: {
        key: None for key in ("cx", "cy", "mass", "peak", "dist")})
    lost_frames: int = 0
    lock_streak: int = 0
    tracking_acquired: bool = False
    contact_count: int = 0
    roi_active: bool = False
    roi_switch_count: int = 0
    pending: list = field(default_factory=list)
    log: dict = field(default_factory=lambda: {
        key: [] for key in ("step", "t", "pose", "sig", "act", "vf", "slow",
                            "rgb", "sim", "dist", "seg", "control")})
    loop_started: float = 0.0
    coordination_started: float = 0.0
    coordination_step: int = 0
    coordination_last_rgb_ts: int = -1
    chase_last_rgb_ts: int = -1
    coordination_log: list = field(default_factory=list)


@dataclass
class Candidate:
    runtime: Runtime
    rgb: np.ndarray
    seg_image: np.ndarray
    sim: np.ndarray
    pose: tuple
    raw_signal: tuple
    filtered_signal: tuple
    distance_raw: float | None
    distance_filtered: float | None
    seg_valid: bool
    target_pixels: int
    frame_dt: float
    locked: bool
    creep_allowed: bool
    slow: float
    command: list
    stop_now: str | None
    roi_used: bool
    contact_count: int
    safety_hold: bool = False


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_mission(path, max_steps_override=None):
    mission_path = Path(path).expanduser().resolve()
    data = commentjson.loads(mission_path.read_text(encoding="utf-8"))
    common = data.get("common", {})
    agents = data.get("agents", [])
    if len(agents) != 3:
        raise SystemExit(f"任务必须恰好配置 3 架无人机，当前 {len(agents)}")

    crossing_raw = data.get("crossing", {})

    def point2(value, label):
        if value is None or len(value) != 2:
            raise SystemExit(f"{label} 必须是 [north, east]")
        return tuple(float(component) for component in value)

    crossing = {
        "center_ned": point2(crossing_raw.get("center_ned"), "crossing.center_ned"),
        "radius": float(crossing_raw.get("radius", 0.0)),
        "altitude": float(crossing_raw.get("altitude", 0.0)),
        "velocity": float(crossing_raw.get("velocity", 0.0)),
    }
    if any(crossing[key] <= 0 for key in ("radius", "altitude", "velocity")):
        raise SystemExit("crossing.radius/altitude/velocity 必须为正数")

    parser = servo.build_parser()
    defaults = servo.parse_args([])
    runtimes = []
    names = set()
    priorities = set()
    for item in agents:
        args = copy.deepcopy(defaults)
        merged = {**common, **item}
        for key, value in merged.items():
            if key in ("name", "priority", "target_ned", "approach_standoff",
                       "crossing_entry_ned", "crossing_exit_ned"):
                continue
            if not hasattr(args, key):
                raise SystemExit(f"{item.get('name', '?')}: 未知控制参数 {key}")
            setattr(args, key, value)
        args.scene = data["scene"]
        args.sim_config_dir = DEFAULT_CONFIG_DIR
        args.template = resolve_path(args.template, PROJECT_ROOT)
        args.stats_episode = resolve_path(args.stats_episode, PROJECT_ROOT)
        args.repo_dir = resolve_path(args.repo_dir, PROJECT_ROOT)
        args.dino_weights = resolve_path(args.dino_weights, PROJECT_ROOT)
        args.template_roi = None if args.template_roi is None else tuple(args.template_roi)
        target_ned = item.get("target_ned", args.face_ned)
        if target_ned is None or len(target_ned) != 2:
            raise SystemExit(f"{item.get('name', '?')}: 必须配置 target_ned=[north,east]")
        target_ned = tuple(float(value) for value in target_ned)
        args.face_ned = target_ned
        args.seg_color = (tuple(args.seg_color) if args.seg_color is not None
                          else servo.SEG_COLORS_BGR.get(args.target))
        args.out_dir = ""
        if max_steps_override is not None:
            args.max_steps = max_steps_override
        servo.validate_args(parser, args)

        name = str(item["name"])
        priority = int(item.get("priority", len(runtimes)))
        if name in names or priority in priorities:
            raise SystemExit("无人机 name 和 priority 必须各自唯一")
        names.add(name)
        priorities.add(priority)
        for required in (args.template, args.stats_episode, args.dino_weights):
            if not Path(required).is_file():
                raise SystemExit(f"{name}: 文件不存在 {required}")
        approach_standoff = float(item.get(
            "approach_standoff", common.get("approach_standoff", 60.0)))
        if approach_standoff <= 0:
            raise SystemExit(f"{name}: approach_standoff 必须为正数")
        crossing_entry = point2(item.get("crossing_entry_ned"),
                                f"{name}.crossing_entry_ned")
        crossing_exit = point2(item.get("crossing_exit_ned"),
                               f"{name}.crossing_exit_ned")
        center = np.asarray(crossing["center_ned"], dtype=float)
        start = np.asarray(crossing_entry, dtype=float)
        end = np.asarray(crossing_exit, dtype=float)
        segment = end - start
        length_sq = float(segment @ segment)
        if length_sq <= 0:
            raise SystemExit(f"{name}: crossing 入口和出口不能相同")
        fraction = float(np.clip(((center - start) @ segment) / length_sq, 0.0, 1.0))
        distance_to_segment = float(np.linalg.norm(start + fraction * segment - center))
        if distance_to_segment > crossing["radius"]:
            raise SystemExit(
                f"{name}: crossing 线段距中心 {distance_to_segment:.1f}m，未穿过半径 "
                f"{crossing['radius']:.1f}m 的交叉区")
        runtimes.append(Runtime(
            name=name, priority=priority, args=args, target_ned=target_ned,
            approach_standoff=approach_standoff,
            crossing_entry_ned=crossing_entry, crossing_exit_ned=crossing_exit))

    safety = data.get("safety", {})
    hold_distance = float(safety.get("hold_distance", 8.0))
    emergency_distance = float(safety.get("emergency_distance", 5.0))
    prediction_seconds = float(safety.get("prediction_seconds", 1.0))
    if not 0 < emergency_distance < hold_distance or prediction_seconds <= 0:
        raise SystemExit("安全参数须满足 0 < emergency_distance < hold_distance 且预测时间 > 0")
    return (mission_path, data, runtimes,
            (hold_distance, emergency_distance, prediction_seconds), crossing)


def body_to_ned(command, yaw):
    forward, right, down, _ = command
    return np.array([
        math.cos(yaw) * forward - math.sin(yaw) * right,
        math.sin(yaw) * forward + math.cos(yaw) * right,
        down,
    ], dtype=float)


def build_candidate(runtime, message, rgb, observation):
    args = runtime.args
    now = time.monotonic()
    rgb_ts = int(message.get("time_stamp", -1))
    runtime.last_rgb_ts = rgb_ts
    runtime.last_frame_wall = now
    frame_dt = args.dt if runtime.last_wall is None else now - runtime.last_wall
    runtime.last_wall = now

    cx, cy, mass, peak, similarity = observation
    n, e, down, yaw = servo.get_pose_yaw(runtime.drone)
    for key, value in (("cx", cx), ("cy", cy), ("mass", mass), ("peak", peak)):
        runtime.filtered[key] = servo.ema_update(runtime.filtered[key], value, args.ema_alpha)
    cx_f, cy_f = runtime.filtered["cx"], runtime.filtered["cy"]
    mass_f, peak_f = runtime.filtered["mass"], runtime.filtered["peak"]
    locked = peak_f >= args.control_min_peak
    runtime.lock_streak = runtime.lock_streak + 1 if locked else 0
    runtime.tracking_acquired |= runtime.lock_streak >= args.lock_confirm_frames
    runtime.lost_frames = 0 if locked else runtime.lost_frames + 1

    depth_message = runtime.mailbox.get("depth")
    seg_message = runtime.mailbox.get("seg")

    def fresh(sensor_message):
        if sensor_message is None or rgb_ts < 0:
            return False
        return abs(int(sensor_message.get("time_stamp", -1)) - rgb_ts) / 1e9 \
            <= args.max_sensor_skew

    trust_depth = peak_f >= args.dist_min_peak and fresh(depth_message)
    seg_valid = False
    target_pixels = 0
    distance_raw = None
    if trust_depth and args.no_seg_depth:
        distance_raw = servo.sample_depth(depth_message, cx_f, cy_f, half=args.depth_half)
    elif trust_depth and fresh(seg_message):
        distance_raw, seg_valid, target_pixels = servo.sample_target_depth(
            depth_message, seg_message, cx_f, cy_f, args.seg_color,
            half=args.depth_half, color_tolerance=args.seg_color_tolerance,
            min_pixels=args.depth_min_pixels)
    if distance_raw is not None:
        runtime.filtered["dist"] = servo.ema_update(
            runtime.filtered["dist"], distance_raw, args.depth_ema_alpha)
    distance_filtered = runtime.filtered["dist"] if distance_raw is not None else None

    roi_used = runtime.roi_active
    target_model_has_roi = args.template_roi is not None
    activate_roi = False
    if target_model_has_roi and not runtime.roi_active:
        switch_hit = distance_filtered is not None and locked \
            and distance_filtered <= args.roi_switch_dist
        runtime.roi_switch_count = runtime.roi_switch_count + 1 if switch_hit else 0
        activate_roi = runtime.roi_switch_count >= args.roi_switch_confirm_frames

    creep_allowed = not runtime.tracking_acquired or runtime.lost_frames <= args.lost_creep_frames
    if locked:
        depth_mode = distance_filtered is not None \
            and (args.stop_dist is not None or args.slow_dist > 0)
        mass_for_control = 0.0 if depth_mode else mass_f
        forward, right, velocity_down, yaw_rate, slow, _ = servo.compute_body_cmd(
            args, cx_f, cy_f, mass_for_control)
    elif creep_allowed:
        forward, right, velocity_down, yaw_rate, slow = args.lost_forward, 0.0, 0.0, 0.0, 1.0
    else:
        forward, right, velocity_down, yaw_rate, slow = 0.0, 0.0, 0.0, 0.0, 0.0

    scale = servo.depth_speed_scale(distance_filtered, args.slow_dist,
                                    args.stop_dist, args.slow_min)
    if distance_filtered is not None and args.slow_dist > 0:
        forward *= scale
        right *= scale
        velocity_down *= scale

    depth_ts = -1 if depth_message is None else int(depth_message.get("time_stamp", -1))
    contact_hit = args.stop_dist is not None and distance_filtered is not None \
        and distance_filtered <= args.stop_dist and locked \
        and depth_ts != runtime.last_contact_depth_ts
    if contact_hit:
        runtime.contact_count += 1
        runtime.last_contact_depth_ts = depth_ts
    elif depth_ts != runtime.last_contact_depth_ts:
        runtime.contact_count = 0
        runtime.last_contact_depth_ts = depth_ts
    if runtime.contact_count > 0:
        forward = right = velocity_down = yaw_rate = 0.0

    stop_now = None
    if args.stop_dist is not None and runtime.contact_count >= args.stop_confirm_frames:
        stop_now = "contact"
    elif args.stop_alt is not None and -down <= args.stop_alt:
        stop_now = "altitude"
    elif args.stop_alt is None and args.stop_dist is None and mass_f >= args.mass_stop:
        stop_now = "mass"
    elif runtime.step + 1 >= args.max_steps:
        stop_now = "timeout"

    if seg_message is None:
        seg_image = np.zeros_like(rgb)
    else:
        seg_image = cv2.resize(ce.decode(seg_message)[..., :3], rgb.shape[1::-1],
                               interpolation=cv2.INTER_NEAREST)

    candidate = Candidate(
        runtime=runtime, rgb=rgb, seg_image=seg_image, sim=similarity,
        pose=(n, e, down, yaw), raw_signal=(cx, cy, mass, peak),
        filtered_signal=(cx_f, cy_f, mass_f, peak_f),
        distance_raw=distance_raw, distance_filtered=distance_filtered,
        seg_valid=seg_valid, target_pixels=target_pixels, frame_dt=frame_dt,
        locked=locked, creep_allowed=creep_allowed, slow=slow,
        command=[forward, right, velocity_down, yaw_rate], stop_now=stop_now,
        roi_used=roi_used, contact_count=runtime.contact_count)

    if activate_roi:
        runtime.roi_active = True
        runtime.roi_switch_count = 0
        for key in runtime.filtered:
            runtime.filtered[key] = None
        runtime.lock_streak = 0
        runtime.lost_frames = 0
        runtime.contact_count = 0
        print(f"{runtime.name}[{runtime.step}] FULL→ROI；下一帧重置 EMA")
    return candidate


def apply_safety(candidates, runtimes, safety):
    """按当前位置与 1 秒预测位置做简单优先级仲裁。"""
    hold_distance, emergency_distance, prediction_seconds = safety
    active = [runtime for runtime in runtimes if runtime.status in ("TRACK", "ARRIVED")]
    poses = {runtime.name: np.asarray(servo.get_pose_yaw(runtime.drone)[:3], dtype=float)
             for runtime in active}
    by_name = {candidate.runtime.name: candidate for candidate in candidates}
    velocities = {}
    for runtime in active:
        candidate = by_name.get(runtime.name)
        velocities[runtime.name] = (np.zeros(3) if candidate is None
                                    else body_to_ned(candidate.command, candidate.pose[3]))

    for index, left in enumerate(active):
        for right in active[index + 1:]:
            current = float(np.linalg.norm(poses[left.name] - poses[right.name]))
            predicted_left = poses[left.name] + prediction_seconds * velocities[left.name]
            predicted_right = poses[right.name] + prediction_seconds * velocities[right.name]
            predicted = float(np.linalg.norm(predicted_left - predicted_right))
            if current < emergency_distance:
                for runtime in active:
                    candidate = by_name.get(runtime.name)
                    if candidate is not None:
                        candidate.command[:] = [0.0, 0.0, 0.0, 0.0]
                        candidate.safety_hold = True
                print(f"SAFETY emergency {left.name}/{right.name}: {current:.1f}m，全体悬停")
            elif predicted < hold_distance:
                lower = max((left, right), key=lambda runtime: runtime.priority)
                candidate = by_name.get(lower.name)
                if candidate is not None:
                    candidate.command[:] = [0.0, 0.0, 0.0, 0.0]
                    candidate.safety_hold = True
                    print(f"SAFETY hold {lower.name}: {left.name}/{right.name} 预测 {predicted:.1f}m")


def record_candidate(candidate):
    runtime = candidate.runtime
    args = runtime.args
    n, e, down, yaw = candidate.pose
    cx, cy, mass, peak = candidate.raw_signal
    cx_f, cy_f, mass_f, peak_f = candidate.filtered_signal
    forward, right, velocity_down, yaw_rate = candidate.command
    log = runtime.log
    log["step"].append(runtime.step)
    log["t"].append(time.monotonic() - runtime.loop_started)
    log["pose"].append([n, e, down, yaw])
    log["sig"].append([cx, cy, mass, peak])
    log["act"].append([forward, right, velocity_down, yaw_rate])
    log["vf"].append(forward)
    log["slow"].append(candidate.slow)
    log["rgb"].append(candidate.rgb.copy())
    log["sim"].append(candidate.sim)
    log["dist"].append(-1.0 if candidate.distance_raw is None else candidate.distance_raw)
    log["control"].append([
        cx_f, cy_f, mass_f, peak_f,
        -1.0 if candidate.distance_filtered is None else candidate.distance_filtered,
        float(candidate.locked), float(candidate.seg_valid), candidate.frame_dt,
        candidate.contact_count, float(candidate.target_pixels), float(candidate.roi_used),
    ])
    log["seg"].append(candidate.seg_image)
    state = "SAFE_HOLD" if candidate.safety_hold else (
        "LOCK" if candidate.locked else ("CREEP" if candidate.creep_allowed else "HOLD"))
    distance = "n/a" if candidate.distance_filtered is None else f"{candidate.distance_filtered:.1f}"
    print(f"{runtime.name}[{runtime.step}] {state} alt={-down:.1f} "
          f"center=({cx_f:+.2f},{cy_f:+.2f}) peak={peak_f:.3f} dist={distance} "
          f"vf={forward:.2f} vd={velocity_down:.2f}")
    runtime.step += 1


async def submit_command(candidate):
    runtime = candidate.runtime
    args = runtime.args
    forward, right, velocity_down, yaw_rate = candidate.command
    task = await runtime.drone.move_by_velocity_body_frame_async(
        v_forward=forward, v_right=right, v_down=velocity_down,
        duration=args.dt * args.cmd_duration_factor, yaw=yaw_rate, yaw_is_rate=True)
    runtime.pending.append(task)


async def hold(runtime, duration=0.25):
    task = await runtime.drone.move_by_velocity_body_frame_async(
        v_forward=0.0, v_right=0.0, v_down=0.0,
        duration=duration, yaw=0.0, yaw_is_rate=True)
    runtime.pending.append(task)


def update_chase_viewer(runtimes, viewer, labels):
    """读取各机最新 Chase 帧；显示链路与 FrontCamera 控制链路相互独立。"""
    if not viewer.enabled:
        return
    updated = False
    for runtime in runtimes:
        message = runtime.mailbox.get("chase_rgb")
        timestamp = -1 if message is None else int(message.get("time_stamp", -1))
        if message is None or timestamp == runtime.chase_last_rgb_ts:
            continue
        runtime.chase_last_rgb_ts = timestamp
        viewer.update(runtime, decode_image(message)[0], labels.get(runtime.name, runtime.status))
        updated = True
    if updated:
        viewer.render()


def capture_coordination(runtimes, bank, phases, metrics, viewer):
    """记录坐标协调阶段的新相机帧、DINO 响应、位姿和机间距离。"""
    update_chase_viewer(runtimes, viewer, phases)
    poses = {
        runtime.name: np.asarray(servo.get_pose_yaw(runtime.drone)[:3], dtype=float)
        for runtime in runtimes
    }
    for index, left in enumerate(runtimes):
        for right in runtimes[index + 1:]:
            distance = float(np.linalg.norm(poses[left.name] - poses[right.name]))
            if distance < metrics["min_separation_m"]:
                metrics["min_separation_m"] = distance
                metrics["min_separation_pair"] = [left.name, right.name]
            if "CROSS" in phases.values() and distance < metrics["crossing_min_separation_m"]:
                metrics["crossing_min_separation_m"] = distance
                metrics["crossing_min_separation_pair"] = [left.name, right.name]

    ready = []
    for runtime in runtimes:
        message = runtime.mailbox.get("rgb")
        timestamp = -1 if message is None else int(message.get("time_stamp", -1))
        if message is None or timestamp == runtime.coordination_last_rgb_ts:
            continue
        runtime.coordination_last_rgb_ts = timestamp
        decoded = decode_image(message)[0]
        rgb = cv2.resize(decoded, (224, 224), interpolation=cv2.INTER_AREA)
        ready.append((runtime, rgb))
    if not ready:
        return

    observations = bank.query(ready)
    now = time.monotonic()
    for runtime, rgb in ready:
        cx, cy, mass, peak, similarity = observations[runtime.name]
        pose = servo.get_pose_yaw(runtime.drone)
        runtime.coordination_log.append({
            "step": runtime.coordination_step,
            "t": now - runtime.coordination_started,
            "pose": np.asarray(pose, dtype=np.float32),
            "rgb": rgb.copy(),
            "sim": np.asarray(similarity, dtype=np.float32),
            "signal": (cx, cy, mass, peak),
            "phase": phases.get(runtime.name, "COORD"),
        })
        runtime.coordination_step += 1


async def monitor_motion(tasks, runtimes, bank, phases, metrics, timeout, viewer):
    """等待一组移动任务，同时持续记录三机协调阶段。"""
    if not tasks:
        capture_coordination(runtimes, bank, phases, metrics, viewer)
        return
    group = asyncio.gather(*tasks)
    deadline = time.monotonic() + timeout
    while not group.done():
        capture_coordination(runtimes, bank, phases, metrics, viewer)
        if time.monotonic() >= deadline:
            group.cancel()
            raise TimeoutError(f"协调移动阶段超过 {timeout:.0f}s")
        await asyncio.sleep(0.02)
    await group
    capture_coordination(runtimes, bank, phases, metrics, viewer)


def save_coordination(out_dir, runtime, grid, crossing):
    """保存坐标协调阶段，供 RViz 在视觉闭环前连续回放。"""
    if not runtime.coordination_log:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coordination.csv", "w", encoding="utf-8") as handle:
        handle.write("step,t,n,e,d,alt,yaw,phase,cx,cy,mass,peak\n")
        for row in runtime.coordination_log:
            n, e, down, yaw = row["pose"]
            cx, cy, mass, peak = row["signal"]
            handle.write(
                f"{row['step']},{row['t']:.3f},{n:.3f},{e:.3f},{down:.3f},{-down:.3f},"
                f"{yaw:.4f},{row['phase']},{cx:.4f},{cy:.4f},{mass:.4f},{peak:.4f}\n")

    dump = out_dir / "coordination_dump"
    dump.mkdir(exist_ok=True)
    center = np.asarray(crossing["center_ned"], dtype=np.float32)
    for index, row in enumerate(runtime.coordination_log):
        _, _, mass, peak = row["signal"]
        np.savez_compressed(
            dump / f"step_{index:04d}.npz",
            step=index, pose=row["pose"], rgb=row["rgb"], sim=row["sim"], grid=grid,
            dt=np.float32(0.2), mass=np.float32(mass), peak=np.float32(peak),
            phase=np.asarray(row["phase"]), crossing_center=center,
            crossing_radius=np.float32(crossing["radius"]),
            crossing_altitude=np.float32(crossing["altitude"]),
        )
    print(f"  协调回放已存: {dump} ({len(runtime.coordination_log)} 帧)")


async def run_mission(cli, mission_path, mission_data, runtimes, safety, crossing):
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, mission={mission_path}")
    from common.runtime import configure_client_log
    configure_client_log()
    client = projectairsim.ProjectAirSimClient(address=cli.address)
    world = None
    run_root = None
    bank = None
    viewer = MultiCameraViewer(cli.show_cameras, runtimes)
    coordination_metrics = {
        "passage_order": [],
        "wait_seconds": {},
        "min_separation_m": float("inf"),
        "min_separation_pair": [],
        "crossing_min_separation_m": float("inf"),
        "crossing_min_separation_pair": [],
    }
    try:
        client.connect()
        print("已连接 ProjectAirSim")
        world = World(client, mission_data["scene"],
                      sim_config_path=str(DEFAULT_CONFIG_DIR.resolve()), delay_after_load_sec=2)
        ce.setup_segmentation(world, {"StaticMeshActor_14", *[
            runtime.args.target for runtime in runtimes]})

        for runtime in runtimes:
            runtime.drone = Drone(client, world, runtime.name)
            camera = runtime.args.camera
            client.subscribe(runtime.drone.sensors[camera]["scene_camera"],
                             runtime.mailbox.callback("rgb"))
            client.subscribe(runtime.drone.sensors[camera]["depth_camera"],
                             runtime.mailbox.callback("depth"))
            client.subscribe(runtime.drone.sensors[camera]["segmentation_camera"],
                             runtime.mailbox.callback("seg"))
            if viewer.enabled:
                client.subscribe(runtime.drone.sensors["Chase"]["scene_camera"],
                                 runtime.mailbox.callback("chase_rgb"))
            print(f"{runtime.name}: 目标={runtime.args.target} 分割色={runtime.args.seg_color}")

        bank = SharedFingerprintBank(runtimes, device)

        stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_root = Path(cli.out_dir).expanduser().resolve() / f"three_aircraft_{stamp}"
        coordination_started = time.monotonic()
        for runtime in runtimes:
            runtime.coordination_started = coordination_started

        for runtime in runtimes:
            if not (runtime.drone.enable_api_control() and runtime.drone.arm()):
                raise RuntimeError(f"{runtime.name}: enable_api_control/arm 失败")
            runtime.status = "ARMED"
        takeoff_tasks = [await runtime.drone.takeoff_async() for runtime in runtimes]
        phases = {runtime.name: "TAKEOFF" for runtime in runtimes}
        await monitor_motion(takeoff_tasks, runtimes, bank, phases,
                             coordination_metrics, timeout=45.0, viewer=viewer)
        print("三机起飞完成")

        # 先进入同一协调高度，便于在 RViz 中直观看到交叉区预约与等待。
        altitude_tasks = []
        for runtime in runtimes:
            north, east, _, _ = servo.get_pose_yaw(runtime.drone)
            altitude_tasks.append(await runtime.drone.move_to_position_async(
                north=north, east=east, down=-abs(crossing["altitude"]), velocity=8.0))
        phases = {runtime.name: "CLIMB_COORD" for runtime in runtimes}
        await monitor_motion(altitude_tasks, runtimes, bank, phases,
                             coordination_metrics, timeout=45.0, viewer=viewer)
        print(f"三机到达 {crossing['altitude']:.0f}m 协调高度")

        entry_tasks = []
        for runtime in runtimes:
            north, east = runtime.crossing_entry_ned
            entry_tasks.append(await runtime.drone.move_to_position_async(
                north=north, east=east, down=-abs(crossing["altitude"]),
                velocity=crossing["velocity"]))
            print(f"{runtime.name}: 交叉入口=({north:.1f},{east:.1f})")
        phases = {runtime.name: "QUEUE_APPROACH" for runtime in runtimes}
        await monitor_motion(entry_tasks, runtimes, bank, phases,
                             coordination_metrics, timeout=45.0, viewer=viewer)
        print("三机到达交叉区等待点")

        queue_ready = time.monotonic()
        passed = set()
        for selected in sorted(runtimes, key=lambda runtime: runtime.priority):
            phases = {}
            for runtime in runtimes:
                if runtime is selected:
                    phases[runtime.name] = "CROSS"
                elif runtime.name in passed:
                    phases[runtime.name] = "CLEAR"
                else:
                    phases[runtime.name] = "WAIT"
            coordination_metrics["wait_seconds"][selected.name] = round(
                time.monotonic() - queue_ready, 3)
            north, east = selected.crossing_exit_ned
            print(f"交叉区放行 {selected.name}；其余未放行无人机 WAIT")
            crossing_task = await selected.drone.move_to_position_async(
                north=north, east=east, down=-abs(crossing["altitude"]),
                velocity=crossing["velocity"])
            await monitor_motion([crossing_task], runtimes, bank, phases,
                                 coordination_metrics, timeout=30.0, viewer=viewer)
            passed.add(selected.name)
            coordination_metrics["passage_order"].append(selected.name)
            print(f"{selected.name}: 已离开交叉区 → ({north:.1f},{east:.1f})")

        # 交叉后各自进入目标航路，同时调整到视觉终端所需高度层。
        approach_tasks = []
        for runtime in runtimes:
            north, east, _, _ = servo.get_pose_yaw(runtime.drone)
            target = np.asarray(runtime.target_ned, dtype=float)
            current = np.asarray([north, east], dtype=float)
            delta = target - current
            distance = float(np.linalg.norm(delta))
            if distance > runtime.approach_standoff:
                waypoint = target - delta / distance * runtime.approach_standoff
            else:
                waypoint = current
            approach_tasks.append(await runtime.drone.move_to_position_async(
                north=float(waypoint[0]), east=float(waypoint[1]),
                down=-abs(runtime.args.start_altitude), velocity=10.0))
            print(f"{runtime.name}: 目标航路接近点=({waypoint[0]:.1f},{waypoint[1]:.1f}), "
                  f"高度 {runtime.args.start_altitude:.0f}m")
        phases = {runtime.name: "TARGET_TRANSIT" for runtime in runtimes}
        await monitor_motion(approach_tasks, runtimes, bank, phases,
                             coordination_metrics, timeout=90.0, viewer=viewer)
        print("三机到达视觉接近点")

        face_tasks = []
        for runtime in runtimes:
            north, east, _, _ = servo.get_pose_yaw(runtime.drone)
            target_north, target_east = runtime.target_ned
            face_tasks.append(await runtime.drone.rotate_to_yaw_async(
                yaw=math.atan2(target_east - east, target_north - north)))
        phases = {runtime.name: "ORIENT_TARGET" for runtime in runtimes}
        await monitor_motion(face_tasks, runtimes, bank, phases,
                             coordination_metrics, timeout=30.0, viewer=viewer)
        print("三机初始朝向完成，进入视觉闭环")

        now = time.monotonic()
        for runtime in runtimes:
            runtime.status = "TRACK"
            runtime.last_frame_wall = now
            runtime.loop_started = now

        while any(runtime.status == "TRACK" for runtime in runtimes):
            now = time.monotonic()
            ready = []
            for runtime in runtimes:
                if runtime.status != "TRACK":
                    continue
                message = runtime.mailbox.get("rgb")
                timestamp = -1 if message is None else int(message.get("time_stamp", -1))
                if message is not None and timestamp != runtime.last_rgb_ts:
                    decoded = decode_image(message)[0]
                    rgb = cv2.resize(decoded, (224, 224),
                                     interpolation=cv2.INTER_AREA)
                    ready.append((runtime, message, rgb))
                elif now - runtime.last_frame_wall > runtime.args.frame_timeout:
                    runtime.status = "FAILED"
                    runtime.stop_reason = "frame_timeout"
                    await hold(runtime)
                    print(f"{runtime.name}: RGB 超时，转 FAILED/HOLD；其他无人机继续")

            update_chase_viewer(
                runtimes, viewer, {runtime.name: runtime.status for runtime in runtimes})
            if not ready:
                await asyncio.sleep(0.002)
                continue

            observations = bank.query([(runtime, rgb) for runtime, _, rgb in ready])
            candidates = [build_candidate(runtime, message, rgb, observations[runtime.name])
                          for runtime, message, rgb in ready]
            apply_safety(candidates, runtimes, safety)

            for candidate in candidates:
                runtime = candidate.runtime
                if candidate.stop_now is not None:
                    candidate.command[:] = [0.0, 0.0, 0.0, 0.0]
                record_candidate(candidate)
                await submit_command(candidate)
                if candidate.stop_now is not None:
                    runtime.stop_reason = candidate.stop_now
                    runtime.status = "ARRIVED" if candidate.stop_now != "timeout" else "FAILED"
                    print(f"{runtime.name}: {runtime.status}, reason={runtime.stop_reason}")

            # 已到达/失败的无人机持续零速保位，直到其余任务结束。
            for runtime in runtimes:
                if runtime.status in ("ARRIVED", "FAILED"):
                    await hold(runtime)

        print("三机任务结束，状态：" + ", ".join(
            f"{runtime.name}={runtime.status}/{runtime.stop_reason}" for runtime in runtimes))
        await asyncio.sleep(max(0.0, cli.hold_after))
    finally:
        viewer.close()
        for runtime in runtimes:
            if runtime.pending:
                await asyncio.gather(*runtime.pending, return_exceptions=True)
            if runtime.drone is not None:
                try:
                    runtime.drone.disarm()
                    runtime.drone.disable_api_control()
                except Exception as error:
                    print(f"{runtime.name}: 释放控制异常 {error}")
        client.disconnect()
        print("已断开连接")

        if run_root is not None and bank is not None and not cli.no_save:
            run_root.mkdir(parents=True, exist_ok=True)
            summary = {
                "mission": str(mission_path),
                "scene": mission_data["scene"],
                "crossing": coordination_metrics,
                "agents": {},
            }
            for runtime in runtimes:
                runtime.args.drone_name = runtime.name
                save_coordination(run_root / runtime.name, runtime, bank.grid, crossing)
                servo.save_run(run_root / runtime.name, runtime.log, runtime.args,
                               runtime.status == "ARRIVED", bank.grid,
                               stop_reason=runtime.stop_reason)
                summary["agents"][runtime.name] = {
                    "target": runtime.args.target,
                    "status": runtime.status,
                    "stop_reason": runtime.stop_reason,
                    "frames": len(runtime.log["step"]),
                }
            (run_root / "mission_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"多机数据已存: {run_root}")


def parse_cli():
    parser = argparse.ArgumentParser(description="三机多目标视觉伺服")
    parser.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--address", default=PROFILE["simulator"]["address"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs/multi"))
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖任务文件中三机共同的 max_steps")
    parser.add_argument("--hold-after", type=float, default=3.0,
                        help="全部结束后保持零速的秒数")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-cameras", action="store_true",
                        help="实时横向显示 Drone1/2/3 三路第三人称 Chase RGB")
    parser.add_argument("--validate-only", action="store_true",
                        help="只校验任务/文件/参数，不连接仿真、不加载 DINO")
    return parser.parse_args()


def main():
    cli = parse_cli()
    mission_path, mission_data, runtimes, safety, crossing = load_mission(
        cli.mission, max_steps_override=cli.max_steps)
    print("任务校验通过：")
    for runtime in runtimes:
        print(f"  {runtime.name} → {runtime.args.target}, target={runtime.target_ned}, "
              f"cross={runtime.crossing_entry_ned}→{runtime.crossing_exit_ned}, "
              f"standoff={runtime.approach_standoff:.0f}m, alt={runtime.args.start_altitude:.1f}m, "
              f"template={runtime.args.template.name}")
    print(f"  safety hold/emergency/predict={safety}")
    print(f"  crossing center/radius/alt/velocity={crossing}")
    if cli.validate_only:
        return
    asyncio.run(run_mission(cli, mission_path, mission_data, runtimes, safety, crossing))


if __name__ == "__main__":
    main()
