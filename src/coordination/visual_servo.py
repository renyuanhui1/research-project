"""Multi-UAV Stage-2 real visual servo controllers (DINOv2 RGB-D approach + RGB-D roof-opening entry).

Implements 1A serialized GPU visual servoing for Batch-2 inspection UAVs without reading
target ground-truth coordinates during closed-loop control:
- `dino_oblique`: Builds a target prototype directly from Stage-1 handover imagery using
  the project's frozen DINOv2 ViT-S/14 encoder and runs `servo_closed_loop` body-frame
  velocity control with RGB-D depth range gating and 3-frame stop confirmation.
- `opening_nadir`: Uses `opening_geometry.opening_candidates` on live nadir RGB-D depth to
  horizontally center over a verified hangar roof breach, verify the vertical free shaft,
  and descend smoothly to interior capture altitude.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from common.paths import PROJECT_ROOT, project_path
from common.sensors import decode as decode_depth_msg, decode_image
from control import servo_closed_loop as sc
from perception.extract_dino_features import IMAGENET_MEAN, IMAGENET_STD, load_model, to_input_tensor
from perception.opening_geometry import opening_candidates
from perception.rgbd_entry import CV_TO_SIM


DEFAULT_DINO_WEIGHTS = PROJECT_ROOT / "weights" / "dinov2_vits14_pretrain.pth"
DEFAULT_DINO_REPO = PROJECT_ROOT / "third_party" / "dinov2"
DEFAULT_STATS_EPISODE = PROJECT_ROOT / "data" / "templates" / "airport_7890" / "oblique" / "stats-episode.h5"


def _pitch_to_quat(pitch_deg: float) -> dict[str, float]:
    half = math.radians(pitch_deg) / 2.0
    return {"w": math.cos(half), "x": 0.0, "y": math.sin(half), "z": 0.0}


def _camera_pose(pitch_deg: float, x: float = 0.0, y: float = 0.0, z: float = 0.3) -> dict[str, Any]:
    return {
        "translation": {"x": x, "y": y, "z": z},
        "rotation": _pitch_to_quat(pitch_deg),
        "frame_id": "DEFAULT_FRAME",
    }


def _central_guard_depth(depth_arr: np.ndarray) -> float | None:
    """Return 2nd-percentile valid depth in central 40% crop for obstacle clearance guard."""
    d = np.asarray(depth_arr, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape[:2]
    crop = d[int(h * 0.3) : int(h * 0.7), int(w * 0.3) : int(w * 0.7)]
    valid = crop[np.isfinite(crop) & (crop > 0.2) & (crop < 500.0)]
    if valid.size < 20:
        return None
    return float(np.percentile(valid, 2))


def _quat_to_rot(msg: dict[str, Any] | None) -> np.ndarray | None:
    if not msg or not all(("rot_" + k) in msg for k in ("w", "x", "y", "z")):
        return None
    q = np.array([float(msg["rot_" + k]) for k in ("w", "x", "y", "z")], dtype=float)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or not (0.9 < norm < 1.1):
        return None
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def compute_nadir_opening_step(
    depth_arr: np.ndarray,
    *,
    fov_deg: float = 90.0,
    min_layer_gap_m: float = 6.0,
    min_clearance_m: float = 1.6,
    gain_xy: float = 0.28,
    max_xy_mps: float = 0.85,
    align_tol_m: float = 0.85,
    cam_rot: np.ndarray | None = None,
    drone_yaw_rad: float = 0.0,
) -> dict[str, Any]:
    """Pure perception + pitch/roll-compensated horizontal alignment step for nadir roof-opening servo.

    When `cam_rot` (world-from-camera rotation matrix from live message quaternion) is supplied,
    the aperture pixel ray is de-tilted into gravity-aligned world horizontal coordinates and
    rotated into the drone's yaw-aligned body frame, preventing quadrotor pitch/roll oscillation.
    """
    d = np.asarray(depth_arr, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape[:2]
    # Keep down=None so opening_candidates selects the distance-transform maximum-clearance
    # aperture (away from interior roof beams); pitch/roll tilt is compensated below via cam_rot.
    geom = opening_candidates(d, fov_degrees=fov_deg, min_layer_gap_m=min_layer_gap_m, down=None)
    out: dict[str, Any] = {
        "status": geom.get("status", "unknown"),
        "detected": False,
        "aligned": False,
        "shaft_clear": False,
        "forward_mps": 0.0,
        "right_mps": 0.0,
        "err_forward_m": None,
        "err_right_m": None,
        "err_horiz_m": None,
        "cx": 0.0,
        "cy": 0.0,
        "clearance_m": 0.0,
        "near_layer_depth_m": geom.get("near_layer_depth_m"),
        "far_layer_depth_m": geom.get("far_layer_depth_m"),
        "center_p10_depth_m": None,
        "visible_gap_mask": None,
    }
    cands = geom.get("candidates") or []
    if not cands:
        return out

    # Prefer largest unobstructed clearance radius inside the roof breach
    best = max(cands, key=lambda c: float(c.get("projected_clearance_m", 0.0)))
    clearance = float(best["projected_clearance_m"])
    px, py = best["aperture_point_depth_xy"]
    focal = float(geom["focal_px"])
    near_depth = float(geom["near_layer_depth_m"])

    if cam_rot is not None:
        # De-tilt pixel ray into gravity-aligned world NED, then rotate into body-yaw frame
        ray_sim = np.array([1.0, (float(px) + 0.5 - w * 0.5) / focal, (float(py) + 0.5 - h * 0.5) / focal], dtype=float)
        ray_ned = cam_rot @ ray_sim
        if ray_ned[2] > 0.2:
            delta_ned = ray_ned * (near_depth / float(ray_ned[2]))
            err_n, err_e = float(delta_ned[0]), float(delta_ned[1])
            cyaw, syaw = math.cos(drone_yaw_rad), math.sin(drone_yaw_rad)
            err_forward = cyaw * err_n + syaw * err_e
            err_right = -syaw * err_n + cyaw * err_e
        else:
            err_right = (float(px) + 0.5 - w * 0.5) * near_depth / focal
            err_forward = -(float(py) + 0.5 - h * 0.5) * near_depth / focal
    else:
        # Level nadir (-90 deg pitch) camera: +u -> body +right, -v -> body +forward
        err_right = (float(px) + 0.5 - w * 0.5) * near_depth / focal
        err_forward = -(float(py) + 0.5 - h * 0.5) * near_depth / focal
    err_horiz = math.hypot(err_forward, err_right)

    cx = (float(px) + 0.5) / w * 2.0 - 1.0
    cy = (float(py) + 0.5) / h * 2.0 - 1.0

    vx = gain_xy * err_forward
    vy = gain_xy * err_right
    spd = math.hypot(vx, vy)
    if spd > max_xy_mps and spd > 1e-6:
        vx *= max_xy_mps / spd
        vy *= max_xy_mps / spd

    # Verify vertical free shaft directly under the drone (at true gravity-nadir pixel if tilted)
    if cam_rot is not None:
        nadir_sim = cam_rot.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        if nadir_sim[0] > 0.2:
            nu = int(round(nadir_sim[1] / nadir_sim[0] * focal + w * 0.5 - 0.5))
            nv = int(round(nadir_sim[2] / nadir_sim[0] * focal + h * 0.5 - 0.5))
        else:
            nu, nv = w // 2, h // 2
    else:
        nu, nv = w // 2, h // 2
    nu = max(16, min(w - 17, nu))
    nv = max(16, min(h - 17, nv))
    half_y, half_x = max(12, h // 14), max(12, w // 14)
    center_patch = d[max(0, nv - half_y) : min(h, nv + half_y), max(0, nu - half_x) : min(w, nu + half_x)]
    valid_center = center_patch[np.isfinite(center_patch) & (center_patch > 0.2) & (center_patch < 500.0)]
    center_p10 = float(np.percentile(valid_center, 10)) if valid_center.size >= 20 else None
    shaft_clear = bool(
        center_p10 is not None
        and clearance >= min_clearance_m
        and center_p10 >= near_depth + max(5.0, min_layer_gap_m * 0.8)
    )
    aligned = bool(err_horiz <= align_tol_m and clearance >= min_clearance_m)

    out.update(
        detected=True,
        aligned=aligned,
        shaft_clear=shaft_clear,
        forward_mps=float(vx),
        right_mps=float(vy),
        err_forward_m=round(err_forward, 3),
        err_right_m=round(err_right, 3),
        err_horiz_m=round(err_horiz, 3),
        cx=float(np.clip(cx, -1.0, 1.0)),
        cy=float(np.clip(cy, -1.0, 1.0)),
        clearance_m=round(clearance, 2),
        center_p10_depth_m=round(center_p10, 2) if center_p10 is not None else None,
        visible_gap_mask=best.get("visible_gap_mask"),
    )
    return out


def foreground_patch_mask(depth_arr: np.ndarray, grid: int, max_depth_m: float) -> np.ndarray:
    """Build a (grid, grid) boolean mask keeping patches within `max_depth_m` foreground depth."""
    d = np.asarray(depth_arr, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    clean = np.where(np.isfinite(d) & (d > 0.2) & (d < 500.0), d, 999.0).astype(np.float32)
    # Local min filter preserves thin foreground structures (e.g., tower pillar) under subsampling
    eroded = cv2.erode(clean, np.ones((7, 7), np.uint8))
    sampled = cv2.resize(eroded, (grid, grid), interpolation=cv2.INTER_NEAREST)
    return (sampled > 0.2) & (sampled <= float(max_depth_m))


class SharedDinoMatcher:
    """Single shared DINOv2 ViT-S/14 encoder for 1A serialized multi-UAV visual servoing."""

    def __init__(
        self,
        *,
        weights_path: Path | str = DEFAULT_DINO_WEIGHTS,
        repo_dir: Path | str = DEFAULT_DINO_REPO,
        stats_episode: Path | str = DEFAULT_STATS_EPISODE,
        image_size: int = 448,
        softmax_temp: float = 0.005,
        mass_thresh: float = 0.5,
        mass_sharpness: float = 15.0,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.softmax_temp = float(softmax_temp)
        self.mass_thresh = float(mass_thresh)
        self.mass_sharpness = float(mass_sharpness)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        torch.set_num_threads(2)
        self.dino = load_model(
            "dinov2_vits14",
            self.device,
            weights=str(project_path(weights_path)),
            repo_dir=str(project_path(repo_dir)),
        )

        stats_path = project_path(stats_episode)
        with h5py.File(stats_path, "r") as f:
            srgb = f["rgb"][:]
        zf = self._encode(srgb, bs=2)
        self.zm = zf.reshape(-1, zf.shape[-1]).mean(0).to(self.device)
        self.zs = zf.reshape(-1, zf.shape[-1]).std(0).clamp_min(1e-6).to(self.device)
        num_patches = int(zf.shape[1])
        self.grid = int(round(num_patches ** 0.5))
        if self.grid * self.grid != num_patches:
            raise RuntimeError(f"Non-square DINO patch grid: {num_patches}")

        xs = torch.linspace(-1.0, 1.0, self.grid, device=self.device)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode(self, rgb_batch: np.ndarray, bs: int = 2) -> torch.Tensor:
        outs = []
        for i in range(0, len(rgb_batch), bs):
            x = to_input_tensor(rgb_batch[i : i + bs], self.image_size, self.mean, self.std, self.device)
            outs.append(self.dino.forward_features(x)["x_norm_patchtokens"].float().cpu())
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def build_prototype(
        self,
        template_image_path: Path | str,
        roi: tuple[float, float, float, float] | list[float] | None = None,
    ) -> tuple[torch.Tensor, int]:
        bgr = cv2.imread(str(template_image_path))
        if bgr is None:
            raise FileNotFoundError(f"Template image not found: {template_image_path}")
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        zt = self._encode(rgb[None], bs=1)[0].to(self.device)
        zt = (zt - self.zm) / self.zs

        if roi is not None:
            x0, y0, x1, y1 = [float(v) for v in roi]
            axis = (torch.arange(self.grid, device=self.device, dtype=torch.float32) + 0.5) / self.grid
            vv, uu = torch.meshgrid(axis, axis, indexing="ij")
            mask = (
                (uu.reshape(-1) >= x0)
                & (uu.reshape(-1) <= x1)
                & (vv.reshape(-1) >= y0)
                & (vv.reshape(-1) <= y1)
            )
            if not bool(mask.any()):
                raise ValueError(f"Template ROI {roi} covers 0 patches on {self.grid}x{self.grid} grid")
            selected = zt[mask]
            patch_count = int(mask.sum().item())
        else:
            selected = zt
            patch_count = int(zt.shape[0])

        proto = F.normalize(F.normalize(selected, dim=-1).mean(0), dim=0)
        return proto, patch_count

    @torch.no_grad()
    def query(
        self,
        rgb: np.ndarray,
        proto: torch.Tensor,
        *,
        depth_arr: np.ndarray | None = None,
        max_depth_m: float | None = None,
    ) -> tuple[float, float, float, float, np.ndarray]:
        x = to_input_tensor(rgb[None], self.image_size, self.mean, self.std, self.device)
        z = self.dino.forward_features(x)["x_norm_patchtokens"][0].float()
        z = (z - self.zm) / self.zs
        sim = torch.matmul(F.normalize(z, dim=-1), proto)
        sim_for_ctrl = sim
        if depth_arr is not None and max_depth_m is not None:
            fg = foreground_patch_mask(depth_arr, self.grid, max_depth_m)
            if int(fg.sum()) >= 3:
                fg_t = torch.from_numpy(fg.reshape(-1)).to(self.device)
                sim_for_ctrl = torch.where(fg_t, sim, torch.full_like(sim, -1e4))
        w = torch.softmax(sim_for_ctrl / self.softmax_temp, dim=-1)
        center = torch.matmul(w, self.patch_xy)
        mass = torch.sigmoid((sim_for_ctrl - self.mass_thresh) * self.mass_sharpness).mean()
        peak = sim_for_ctrl.max()
        sim_2d = sim_for_ctrl.clamp_min(-1.0).reshape(self.grid, self.grid).detach().cpu().numpy().astype(np.float32)
        return float(center[0]), float(center[1]), float(mass), float(peak), sim_2d


class ServoFrameBuffer:
    """Thread-safe synchronized RGB + Depth frame subscriber for one active servo UAV.

    When `recorder` is provided, RGB frames are tapped via `recorder.set_front_listener`
    so we never double-subscribe or unsubscribe the recorder's `FrontCamera/scene_camera`.
    """

    def __init__(self, client: Any, drone: Any, target_id: str, recorder: Any | None = None) -> None:
        self._client = client
        self._drone = drone
        self._target_id = target_id
        self._recorder = recorder
        self._lock = threading.Lock()
        self._rgb_cache: dict[int, dict[str, Any]] = {}
        self._depth_cache: dict[int, dict[str, Any]] = {}
        self._topics: list[str] = []

    def _push(self, cache: dict[int, dict[str, Any]], msg: dict[str, Any]) -> None:
        ts = int(msg.get("time_stamp", 0))
        with self._lock:
            cache[ts] = msg
            while len(cache) > 16:
                cache.pop(min(cache))

    def start(self) -> None:
        front = self._drone.sensors["FrontCamera"]
        if self._recorder is not None and hasattr(self._recorder, "set_front_listener"):
            self._recorder.set_front_listener(self._target_id, lambda msg: self._push(self._rgb_cache, msg))
        else:
            rgb_topic = front["scene_camera"]
            self._client.subscribe(rgb_topic, lambda _t, m: self._push(self._rgb_cache, m))
            self._topics.append(rgb_topic)

        depth_topic = front["depth_camera"]
        self._client.subscribe(depth_topic, lambda _t, m: self._push(self._depth_cache, m))
        self._topics.append(depth_topic)

    def stop(self) -> None:
        if self._recorder is not None and hasattr(self._recorder, "set_front_listener"):
            self._recorder.set_front_listener(self._target_id, None)
        for topic in self._topics:
            try:
                self._client.unsubscribe(topic)
            except Exception:
                pass
        self._topics.clear()

    async def next_pair(
        self,
        after_rgb_ts: int,
        timeout_s: float = 3.0,
        max_skew_ns: int = 350_000_000,
    ) -> tuple[int, dict[str, Any], int, dict[str, Any]] | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._rgb_cache and self._depth_cache:
                    for rt in sorted((ts for ts in self._rgb_cache if ts > after_rgb_ts), reverse=True):
                        dt = min(self._depth_cache, key=lambda dts: abs(dts - rt))
                        if abs(dt - rt) <= max_skew_ns:
                            return rt, self._rgb_cache[rt], dt, self._depth_cache[dt]
            await asyncio.sleep(0.01)
        return None


async def _hover_zero(drone: Any) -> None:
    try:
        task = await drone.move_by_velocity_body_frame_async(0.0, 0.0, 0.0, duration=0.3, yaw=0.0, yaw_is_rate=True)
        await asyncio.wait_for(task, timeout=5.0)
    except Exception:
        pass
    try:
        task = await drone.hover_async()
        await asyncio.wait_for(task, timeout=10.0)
    except Exception:
        pass


async def run_dino_visual_servo(
    *,
    client: Any,
    drone: Any,
    drone_id: str,
    target_id: str,
    matcher: SharedDinoMatcher,
    template_image_path: Path | str,
    servo_cfg: dict[str, Any],
    collision_flags: dict[str, int],
    recorder: Any | None = None,
) -> dict[str, Any]:
    """Run real closed-loop DINOv2 + RGB-D oblique visual servo approach for one UAV."""
    roi = servo_cfg.get("template_roi")
    proto, patch_count = matcher.build_prototype(template_image_path, roi=roi)

    pitch_deg = float(servo_cfg.get("camera_pitch_deg", -30.0))
    stop_dist = float(servo_cfg.get("stop_dist_m", 5.8))
    slow_dist = float(servo_cfg.get("slow_dist_m", 24.0))
    close_range_m = float(servo_cfg.get("close_range_m", 12.0))
    close_speed_mps = float(servo_cfg.get("close_speed_mps", 0.75))
    depth_gate_margin_m = float(servo_cfg.get("depth_gate_margin_m", 15.0))
    v_forward = float(servo_cfg.get("v_forward_mps", 2.2))
    v_down = float(servo_cfg.get("v_down_mps", 0.7))
    k_yaw = float(servo_cfg.get("k_yaw", 0.85))
    yaw_max = float(servo_cfg.get("yaw_max_rps", 0.45))
    k_vert = float(servo_cfg.get("k_vert", 1.2))
    allow_climb = bool(servo_cfg.get("allow_climb", False))
    min_peak = float(servo_cfg.get("min_peak", 0.25))
    min_alt_m = float(servo_cfg.get("min_alt_m", 5.0))
    max_alt_m = float(servo_cfg.get("max_alt_m", 60.0))
    depth_guard_m = float(servo_cfg.get("depth_guard_m", 2.5))
    confirm_target = int(servo_cfg.get("confirm_frames", 3))
    max_duration_s = float(servo_cfg.get("max_duration_s", 55.0))
    ema_alpha_xy = float(servo_cfg.get("ema_alpha_xy", 0.35))
    ema_alpha_d = float(servo_cfg.get("ema_alpha_depth", 0.5))

    ctrl_args = SimpleNamespace(
        nadir=False,
        mass_stop=0.95,
        v_forward=v_forward,
        v_down=v_down,
        k_vert=k_vert,
        k_yaw=k_yaw,
        yaw_max=yaw_max,
        k_xy=1.0,
        center_gate=0.35,
    )

    drone.set_camera_pose("FrontCamera", _camera_pose(pitch_deg))
    await asyncio.sleep(0.6)

    buf = ServoFrameBuffer(client, drone, target_id=target_id, recorder=recorder)
    buf.start()
    col_baseline = collision_flags.get(drone_id, 0)
    t0 = time.monotonic()
    last_rgb_ts = -1
    last_depth_ts = -1
    xf: float | None = None
    yf: float | None = None
    df: float | None = None
    confirm_count = 0
    lost_count = 0
    close_latched = False
    prev_cmd = np.zeros(3, dtype=float)
    prev_tick = time.monotonic()
    steps: list[dict[str, Any]] = []
    stop_reason = "duration_limit"

    try:
        while True:
            now = time.monotonic()
            if now - t0 >= max_duration_s:
                stop_reason = "duration_limit"
                break

            pair = await buf.next_pair(last_rgb_ts, timeout_s=3.0)
            if pair is None:
                stop_reason = "frame_timeout"
                break
            rt, rgb_msg, dt, depth_msg = pair
            last_rgb_ts = rt

            rgb, _ = decode_image(rgb_msg)
            depth_arr = np.asarray(decode_depth_msg(depth_msg), dtype=np.float32)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[..., 0]

            # Depth-gated foreground lock prevents distant twin structures from hijacking the target
            gate_max_d = (df + depth_gate_margin_m) if (df is not None and df < 45.0) else None
            cx, cy, mass, peak, sim_2d = matcher.query(
                rgb, proto, depth_arr=depth_arr, max_depth_m=gate_max_d
            )
            locked = peak >= min_peak
            lost_count = 0 if locked else (lost_count + 1)

            xf = sc.ema_update(xf, cx, ema_alpha_xy) if locked else (xf if xf is not None else 0.0)
            yf = sc.ema_update(yf, cy, ema_alpha_xy) if locked else (yf if yf is not None else 0.0)

            raw_dist = sc.sample_depth_array(depth_arr, xf, yf, half=4) if locked else None
            # Reject sudden far-background depth jumps once locked onto a foreground target
            if raw_dist is not None and df is not None and df < 45.0 and raw_dist > df + depth_gate_margin_m:
                raw_dist = None
            if raw_dist is not None:
                df = sc.ema_update(df, raw_dist, ema_alpha_d)
            cur_dist = df if (locked and raw_dist is not None) else None
            guard_d = _central_guard_depth(depth_arr)

            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            alt_m = -float(pos["z"])

            f, r, d_cmd, yaw_rate, _, _ = sc.compute_body_cmd(ctrl_args, xf, yf, 0.0)
            # Attenuate forward speed when target is far off horizontal center
            f *= max(0.0, 1.0 - abs(xf) / 0.55)
            if allow_climb:
                d_cmd = float(np.clip(k_vert * yf, -1.0, 1.4))
            else:
                d_cmd = float(np.clip(d_cmd, 0.0, 1.6))

            if not locked or cur_dist is None:
                f = r = d_cmd = yaw_rate = 0.0
            else:
                eff_dist = min(cur_dist, guard_d) if guard_d is not None else cur_dist
                if eff_dist <= close_range_m:
                    close_latched = True
                scale = sc.depth_speed_scale(eff_dist, slow_dist, stop_dist, 0.22)
                f *= scale
                r *= scale
                d_cmd *= scale
                if close_latched:
                    spd = math.sqrt(f * f + r * r + d_cmd * d_cmd)
                    if spd > close_speed_mps:
                        f *= close_speed_mps / spd
                        r *= close_speed_mps / spd
                        d_cmd *= close_speed_mps / spd
                if alt_m <= min_alt_m + 0.8 and d_cmd > 0.0:
                    d_cmd = 0.0

            # Smooth slew-rate limit on body velocity
            tick_dt = max(0.04, min(time.monotonic() - prev_tick, 0.4))
            max_delta = 1.8 * tick_dt
            cmd_vec = np.clip(np.array([f, r, d_cmd], dtype=float), prev_cmd - max_delta, prev_cmd + max_delta)
            f, r, d_cmd = float(cmd_vec[0]), float(cmd_vec[1]), float(cmd_vec[2])
            prev_cmd = cmd_vec
            prev_tick = time.monotonic()

            if dt != last_depth_ts:
                range_hit = bool(
                    locked
                    and (
                        (cur_dist is not None and cur_dist <= stop_dist)
                        or (guard_d is not None and guard_d <= stop_dist)
                    )
                )
                confirm_count = (confirm_count + 1) if range_hit else 0
                last_depth_ts = dt
            if confirm_count > 0:
                f = r = d_cmd = yaw_rate = 0.0
                prev_cmd[:] = 0.0

            phase_cn = "停机确认" if confirm_count > 0 else ("视觉抵近" if locked else "失锁悬停")
            if recorder is not None:
                recorder.set_status(
                    target_id,
                    f"视觉伺服·{phase_cn}({cur_dist:.1f}m)" if cur_dist is not None else f"视觉伺服·{phase_cn}",
                )
                recorder.set_servo_overlay(
                    target_id,
                    {
                        "mode": "dino",
                        "badge": "实时视觉伺服",
                        "phase_cn": phase_cn,
                        "sim_2d": sim_2d,
                        "cx": float(xf),
                        "cy": float(yf),
                        "peak": float(peak),
                        "depth_m": float(cur_dist) if cur_dist is not None else None,
                        "stop_dist_m": stop_dist,
                        "cmd_fwd": f,
                        "cmd_right": r,
                        "cmd_down": d_cmd,
                        "cmd_yaw": yaw_rate,
                        "confirm": confirm_count,
                        "confirm_target": confirm_target,
                        "locked": locked,
                    },
                )

            step_row = {
                "step": len(steps),
                "t_s": round(time.monotonic() - t0, 3),
                "alt_m": round(alt_m, 2),
                "cx": round(float(xf), 3),
                "cy": round(float(yf), 3),
                "peak": round(float(peak), 3),
                "depth_m": round(float(cur_dist), 2) if cur_dist is not None else None,
                "guard_depth_m": round(float(guard_d), 2) if guard_d is not None else None,
                "cmd_body": [round(f, 2), round(r, 2), round(d_cmd, 2), round(yaw_rate, 3)],
                "confirm": confirm_count,
                "locked": locked,
            }
            steps.append(step_row)

            # Safety guards & stop criteria
            if collision_flags.get(drone_id, 0) > col_baseline:
                stop_reason = "collision_guard"
                break
            if confirm_count >= confirm_target:
                stop_reason = "range_confirmed"
                break
            if guard_d is not None and guard_d < depth_guard_m:
                stop_reason = "depth_guard"
                break
            if alt_m <= min_alt_m:
                stop_reason = "altitude_guard"
                break
            if alt_m > max_alt_m:
                stop_reason = "max_altitude_guard"
                break
            if lost_count >= 15:
                stop_reason = "tracking_lost"
                break

            # Fire velocity pulse non-blockingly so next perception tick overrides seamlessly
            await drone.move_by_velocity_body_frame_async(
                f, r, d_cmd, duration=0.45, yaw=yaw_rate, yaw_is_rate=True
            )

        await _hover_zero(drone)
    finally:
        buf.stop()
        if recorder is not None:
            recorder.set_servo_overlay(target_id, None)

    final_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    return {
        "mode": "dino_oblique",
        "target_id": target_id,
        "drone_id": drone_id,
        "template_image": str(template_image_path),
        "template_roi": roi,
        "template_patches": patch_count,
        "stop_reason": stop_reason,
        "succeeded": stop_reason in ("range_confirmed", "depth_guard", "altitude_guard"),
        "duration_s": round(time.monotonic() - t0, 2),
        "steps": len(steps),
        "final_depth_m": steps[-1]["depth_m"] if steps else None,
        "final_peak": steps[-1]["peak"] if steps else None,
        "final_pose_ned": [round(float(final_pos["x"]), 2), round(float(final_pos["y"]), 2), round(float(final_pos["z"]), 2)],
        "trace": steps,
    }


async def run_opening_visual_servo(
    *,
    client: Any,
    drone: Any,
    drone_id: str,
    target_id: str,
    servo_cfg: dict[str, Any],
    collision_flags: dict[str, int],
    recorder: Any | None = None,
) -> dict[str, Any]:
    """Run real closed-loop RGB-D roof-opening alignment and verified vertical descent."""
    min_layer_gap_m = float(servo_cfg.get("min_layer_gap_m", 6.0))
    min_clearance_m = float(servo_cfg.get("min_clearance_m", 1.6))
    align_tol_m = float(servo_cfg.get("align_tol_m", 0.85))
    align_confirm_target = int(servo_cfg.get("align_confirm_frames", 3))
    gain_xy = float(servo_cfg.get("gain_xy", 0.28))
    max_xy_mps = float(servo_cfg.get("max_xy_mps", 0.85))
    descend_mps = float(servo_cfg.get("descend_mps", 2.0))
    target_interior_alt_m = float(servo_cfg.get("target_interior_alt_m", 15.0))
    min_floor_guard_m = float(servo_cfg.get("min_floor_guard_m", 5.0))
    max_duration_s = float(servo_cfg.get("max_duration_s", 55.0))

    drone.set_camera_pose("FrontCamera", _camera_pose(-90.0))
    await asyncio.sleep(0.8)

    buf = ServoFrameBuffer(client, drone, target_id=target_id, recorder=recorder)
    buf.start()
    col_baseline = collision_flags.get(drone_id, 0)
    t0 = time.monotonic()
    last_rgb_ts = -1
    align_hits = 0
    lost_count = 0
    align_alt_m = -float(drone.get_ground_truth_kinematics()["pose"]["position"]["z"])
    phase = "align"  # align -> descend
    steps: list[dict[str, Any]] = []
    stop_reason = "duration_limit"

    try:
        while True:
            if time.monotonic() - t0 >= max_duration_s:
                stop_reason = "duration_limit"
                break

            pair = await buf.next_pair(last_rgb_ts, timeout_s=3.0)
            if pair is None:
                stop_reason = "frame_timeout"
                break
            rt, _rgb_msg, _dt, depth_msg = pair
            last_rgb_ts = rt

            depth_arr = np.asarray(decode_depth_msg(depth_msg), dtype=np.float32)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[..., 0]

            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            ori = kin["pose"]["orientation"]
            alt_m = -float(pos["z"])
            drone_yaw = math.atan2(
                2.0 * (float(ori["w"]) * float(ori["z"]) + float(ori["x"]) * float(ori["y"])),
                1.0 - 2.0 * (float(ori["y"]) ** 2 + float(ori["z"]) ** 2),
            )
            cam_rot = _quat_to_rot(depth_msg)
            guard_d = _central_guard_depth(depth_arr)

            res = compute_nadir_opening_step(
                depth_arr,
                min_layer_gap_m=min_layer_gap_m,
                min_clearance_m=min_clearance_m,
                gain_xy=gain_xy,
                max_xy_mps=max_xy_mps,
                align_tol_m=align_tol_m,
                cam_rot=cam_rot,
                drone_yaw_rad=drone_yaw,
            )

            f = r = d_cmd = 0.0
            if phase == "align":
                # Keep altitude steady during horizontal visual centering
                d_cmd = float(np.clip(0.35 * (alt_m - align_alt_m), -0.5, 0.5))
                if res["detected"]:
                    lost_count = 0
                    f = res["forward_mps"]
                    r = res["right_mps"]
                    if res["aligned"] and res["shaft_clear"]:
                        align_hits += 1
                    else:
                        align_hits = 0
                    if align_hits >= align_confirm_target:
                        phase = "descend"
                        f = r = 0.0
                        d_cmd = descend_mps * 0.6
                else:
                    lost_count += 1
                    align_hits = 0
                phase_cn = f"破口对准 {align_hits}/{align_confirm_target}" if res["detected"] else "搜索屋顶破口"
            else:
                # Descending through the verified vertical free shaft
                if res["detected"] and alt_m > target_interior_alt_m + 10.0:
                    # Keep gentle horizontal centering while roof rim is still in FOV
                    f = float(np.clip(res["forward_mps"] * 0.45, -0.5, 0.5))
                    r = float(np.clip(res["right_mps"] * 0.45, -0.5, 0.5))
                rem_alt = max(0.0, alt_m - target_interior_alt_m)
                d_cmd = float(np.clip(rem_alt * 0.6, 0.55, descend_mps))
                phase_cn = f"垂直穿洞入室({alt_m:.1f}m)"

            if recorder is not None:
                recorder.set_status(target_id, f"视觉伺服·{phase_cn}")
                recorder.set_servo_overlay(
                    target_id,
                    {
                        "mode": "opening",
                        "badge": "实时视觉伺服",
                        "phase_cn": phase_cn,
                        "gap_mask": res["visible_gap_mask"],
                        "cx": float(res["cx"]),
                        "cy": float(res["cy"]),
                        "err_horiz_m": res["err_horiz_m"],
                        "clearance_m": res["clearance_m"],
                        "depth_m": res["center_p10_depth_m"] or guard_d,
                        "cmd_fwd": f,
                        "cmd_right": r,
                        "cmd_down": d_cmd,
                        "cmd_yaw": 0.0,
                        "confirm": align_confirm_target if phase == "descend" else align_hits,
                        "confirm_target": align_confirm_target,
                        "locked": bool(res["detected"] or phase == "descend"),
                    },
                )

            steps.append(
                {
                    "step": len(steps),
                    "t_s": round(time.monotonic() - t0, 3),
                    "phase": phase,
                    "alt_m": round(alt_m, 2),
                    "detected": res["detected"],
                    "err_horiz_m": res["err_horiz_m"],
                    "clearance_m": res["clearance_m"],
                    "near_layer_depth_m": res["near_layer_depth_m"],
                    "center_p10_depth_m": res["center_p10_depth_m"],
                    "guard_depth_m": round(float(guard_d), 2) if guard_d is not None else None,
                    "cmd_body": [round(f, 2), round(r, 2), round(d_cmd, 2)],
                    "align_hits": align_hits,
                }
            )

            if collision_flags.get(drone_id, 0) > col_baseline:
                stop_reason = "collision_guard"
                break
            if phase == "descend" and alt_m <= target_interior_alt_m + 0.6:
                stop_reason = "interior_reached"
                break
            if phase == "descend" and guard_d is not None and guard_d < min_floor_guard_m:
                stop_reason = "floor_depth_guard"
                break
            if phase == "align" and lost_count >= 15:
                stop_reason = "opening_lost"
                break

            await drone.move_by_velocity_body_frame_async(f, r, d_cmd, duration=0.45, yaw=0.0, yaw_is_rate=True)

        await _hover_zero(drone)
    finally:
        buf.stop()
        if recorder is not None:
            recorder.set_servo_overlay(target_id, None)

    final_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    return {
        "mode": "opening_nadir",
        "target_id": target_id,
        "drone_id": drone_id,
        "stop_reason": stop_reason,
        "succeeded": stop_reason in ("interior_reached", "floor_depth_guard"),
        "duration_s": round(time.monotonic() - t0, 2),
        "steps": len(steps),
        "entered_interior": phase == "descend" and stop_reason in ("interior_reached", "floor_depth_guard"),
        "final_alt_m": round(-float(final_pos["z"]), 2),
        "final_pose_ned": [round(float(final_pos["x"]), 2), round(float(final_pos["y"]), 2), round(float(final_pos["z"]), 2)],
        "trace": steps,
    }
