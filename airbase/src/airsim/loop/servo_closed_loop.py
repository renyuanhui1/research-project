"""servo_closed_loop.py —— 方案A: 指纹视觉伺服闭环(不用世界模型)。

用已验证单调的模板指纹信号, 实时驱动无人机飞向目标:
  每步: 取帧 → DINO编码 → 标准化 → 指纹响应 → 得 center(目标方位)/mass(大小)/peak
不依赖 predictor。用于验证"指纹信号能不能实时把无人机领到目标跟前"。

两种模式:
  默认斜视(相机 -50°, 当前主用): --face-ned 一次性对准 → cx 转向 + 前进 +
           **cy 垂直闭环(视线角控制)** 让目标保持画面中心 → 沿视线接触; mass 够大停。
  --nadir  俯视伺服(相机须 -90°, 备选): --goto-ned 开环到目标上空 → cx/cy 双轴水平闭环 +
           居中才降, --stop-alt 按高度判接触。

标准化 z_mean/z_std 从 --stats-episode(录好的接近 h5)启动时算, 与离线判据同源。
注意: 换相机角度/缩放飞机后, 模板和 stats-episode 必须重做(同视角同尺度)。

用法(UE 前台, 有 GPU) —— 斜视主线:
  python airbase/src/airsim/servo_closed_loop.py \
      --template airbase/pictures/尾翼.jpg \
      --stats-episode airbase/outputs/recordings/approach/airbase_tgt1_50m.h5 \
      --face-ned -64.2 -18.5 --start-altitude 40
"""

import argparse
import asyncio
import datetime
import math
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import projectairsim
from projectairsim import Drone, World

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import（如 loop/servo_closed_loop → wm/decode_check）：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
import collect_episode as ce          # 复用 decode()：深度可能是 16UC1 或 32FC1
from decode_check import decode_image
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = _AIRBASE_ROOT
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"

_lock = threading.Lock()
_msgs = {}                       # tag → 最新一帧（rgb / depth / seg）

# ProjectAirSim 分割相机输出的是调色板 BGR，不是原始 ID。下列颜色来自本场景
# 07-30/08-01 的实测记录；--seg-color 可覆盖，避免场景改动后把旧颜色当真。
SEG_COLORS_BGR = {
    "StaticMeshActor_16": (96, 138, 60),
    "StaticMeshActor_18": (247, 61, 22),
    "StaticMeshActor_20": (4, 154, 16),
    "StaticMeshActor_22": (190, 68, 107),
    "StaticMeshActor_24": (131, 43, 106),
}


def make_cb(tag):
    def _cb(*args):
        with _lock:
            _msgs[tag] = args[-1]
    return _cb


cb = make_cb("rgb")              # 兼容旧调用（PX4 版有自己的 cb，不受影响）


def get_latest(tag="rgb"):
    with _lock:
        return _msgs.get(tag)


def reset_messages():
    """清掉上次场景/上次运行遗留的消息，避免把旧帧当作新场景首帧。"""
    with _lock:
        _msgs.clear()


async def wait_new_message(tag, last_ts=-1, timeout=2.0, poll=0.002):
    """等待指定传感器的新时间戳；控制循环由 RGB 新帧驱动，不再靠固定 sleep。"""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        msg = get_latest(tag)
        if msg is not None and int(msg.get("time_stamp", -1)) != last_ts:
            return msg
        await asyncio.sleep(poll)
    raise TimeoutError(f"等待 {tag} 新帧超时 {timeout:.1f}s")


def ema_update(previous, current, alpha):
    """一阶指数滤波；alpha 是当前观测权重，首帧直接采用当前值。"""
    current = float(current)
    return current if previous is None else float(alpha * current + (1.0 - alpha) * previous)


def make_target_mask(seg_bgr, color_bgr, tolerance=0, dilate_px=0):
    """从调色板 BGR 分割图提取目标 Actor 掩膜。"""
    seg = np.asarray(seg_bgr)[..., :3].astype(np.int16)
    color = np.asarray(color_bgr, np.int16).reshape(1, 1, 3)
    mask = np.all(np.abs(seg - color) <= int(tolerance), axis=-1).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask.astype(bool)


def sample_depth_array(depth, cx, cy, half=3, target_mask=None, min_pixels=3):
    """在归一化目标点附近采深度；给掩膜时只接受目标 Actor 像素。"""
    d = np.asarray(depth, np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape
    u = int(round((float(cx) + 1.0) * 0.5 * (w - 1)))
    v = int(round((float(cy) + 1.0) * 0.5 * (h - 1)))
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    valid = np.isfinite(d[v0:v1, u0:u1]) & (d[v0:v1, u0:u1] > 0.1) \
        & (d[v0:v1, u0:u1] < 60000.0)
    if target_mask is not None:
        m = np.asarray(target_mask, bool)
        if m.shape != d.shape:
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid &= m[v0:v1, u0:u1]
    values = d[v0:v1, u0:u1][valid]
    if values.size < int(min_pixels):
        return None
    return float(np.median(values))


def depth_speed_scale(distance, slow_dist, stop_dist, slow_min):
    """距离减速比例：slow_dist 外全速，stop_dist 处降到 slow_min。"""
    if distance is None or slow_dist <= 0:
        return 1.0
    stop = 0.0 if stop_dist is None else float(stop_dist)
    span = max(float(slow_dist) - stop, 1e-6)
    ratio = (float(distance) - stop) / span
    return float(np.clip(ratio, float(slow_min), 1.0))


def get_pose_yaw(drone):
    k = drone.get_ground_truth_kinematics()["pose"]
    p = k["position"]; o = k["orientation"]
    w, x, y, z = o["w"], o["x"], o["y"], o["z"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(p["x"]), float(p["y"]), float(p["z"]), yaw


class Fingerprint:
    """DINO 编码 + 模板指纹, 返回 center/mass/peak。标准化统计量来自 stats-episode。"""

    def __init__(self, args, device):
        self.dev = device
        self.args = args
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.dino = load_model("dinov2_vits14", device,
                               weights=str(args.dino_weights), repo_dir=str(args.repo_dir))
        # 标准化统计量: 从 stats-episode 帧算
        with h5py.File(args.stats_episode, "r") as f:
            srgb = f["rgb"][:]
        zf = self._encode(srgb)                      # (N,P,D)
        self.zm = zf.reshape(-1, zf.shape[-1]).mean(0).to(device)
        self.zs = zf.reshape(-1, zf.shape[-1]).std(0).clamp_min(1e-6).to(device)
        # 模板 proto：默认保留全模板；可选 ROI 原型只在显式传 --template-roi 时创建。
        tbgr = cv2.imread(str(args.template))
        if tbgr is None:
            raise SystemExit(f"读不到模板: {args.template}")
        zt = self._encode(np.ascontiguousarray(tbgr[:, :, ::-1])[None])[0].to(device)
        zt = (zt - self.zm) / self.zs
        P = zf.shape[1]; self.grid = int(round(P ** 0.5))
        if self.grid * self.grid != P:
            raise SystemExit(f"DINO patch 数不是方形网格: {P}")
        xs = torch.linspace(-1.0, 1.0, self.grid, device=device)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
        self.protos = {
            "full": F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0),
        }
        self.roi_patch_count = 0
        if args.template_roi is not None:
            x0, y0, x1, y1 = args.template_roi
            axis = (torch.arange(self.grid, device=device, dtype=torch.float32) + 0.5) / self.grid
            vv, uu = torch.meshgrid(axis, axis, indexing="ij")
            mask = ((uu.reshape(-1) >= x0) & (uu.reshape(-1) <= x1)
                    & (vv.reshape(-1) >= y0) & (vv.reshape(-1) <= y1))
            if not mask.any():
                raise SystemExit(f"--template-roi {args.template_roi} 没覆盖任何模板 patch")
            self.protos["roi"] = F.normalize(
                F.normalize(zt[mask], dim=-1).mean(0), dim=0)
            self.roi_patch_count = int(mask.sum())
        detail = (f", ROI={args.template_roi} ({self.roi_patch_count} patches)"
                  if self.has_roi else "")
        print(f"指纹就绪: proto D={self.protos['full'].shape[0]}, grid={self.grid}{detail}")

    @property
    def has_roi(self):
        return "roi" in self.protos

    @torch.no_grad()
    def _encode(self, rgb, bs=32):
        out = []
        for i in range(0, len(rgb), bs):
            x = to_input_tensor(rgb[i:i + bs], self.args.image_size, self.mean, self.std, self.dev)
            out.append(self.dino.forward_features(x)["x_norm_patchtokens"].float().cpu())
        return torch.cat(out, 0)

    @torch.no_grad()
    def query(self, rgb224, use_roi=False):
        """rgb224:(224,224,3) uint8 → (center_x, center_y, mass, peak)。"""
        x = to_input_tensor(rgb224[None], self.args.image_size, self.mean, self.std, self.dev)
        z = self.dino.forward_features(x)["x_norm_patchtokens"][0].float()
        z = (z - self.zm) / self.zs
        proto = self.protos["roi" if use_roi and self.has_roi else "full"]
        sim = torch.matmul(F.normalize(z, dim=-1), proto)           # (P,) 即热力图
        w = torch.softmax(sim / self.args.target_softmax_temp, dim=-1)
        center = torch.matmul(w, self.patch_xy)                     # (2,)
        mass = torch.sigmoid(
            (sim - self.args.target_mass_thresh) * self.args.target_mass_sharpness).mean()
        peak = sim.max()
        return (float(center[0]), float(center[1]), float(mass), float(peak),
                sim.cpu().numpy())


def compute_body_cmd(args, cx, cy, mass, yaw=None):
    """指纹信号 → **机体系**速度设定值。返回 (forward, right, down, yaw_rate[rad/s], slow, vf)。

    08-01 从 servo_closed_loop_px4.py 搬回这里作为**单一实现** —— 原先两个脚本各写一份
    等价的控制律，改一处忘另一处只是时间问题。PX4 版现在 import 它。
    `yaw` 参数保留只为兼容旧调用，机体系下用不到（世界系版本才需要它做旋转）。
    """
    slow = max(0.0, 1.0 - mass / args.mass_stop)          # 越近越慢
    if args.nadir:
        # 俯视: cx/cy 双轴闭环压目标到画面中心; 越居中降得越快。机体系直接给 forward/right。
        forward = -args.k_xy * cy
        right = args.k_xy * cx
        sp = math.hypot(forward, right)
        if sp > args.v_forward:
            forward *= args.v_forward / sp
            right *= args.v_forward / sp
        err = math.hypot(cx, cy)
        down = args.v_down * max(0.0, 1.0 - err / args.center_gate)
        yaw_rate = 0.0
        vf = math.hypot(forward, right)
    else:
        # 斜视(-50°): 前进 + cy 垂直闭环(视线角控制, 让目标保持画面中心, 沿视线接触) + cx 转向。
        vf = args.v_forward * slow
        forward = vf
        right = 0.0
        down = max(0.0, args.v_down * slow + args.k_vert * cy)
        yaw_rate = float(np.clip(args.k_yaw * cx, -args.yaw_max, args.yaw_max))
    return forward, right, down, yaw_rate, slow, vf


def sample_depth(msg, cx, cy, half=3):
    """在指纹重心处采深度（米）。cx/cy 是 [-1,1] 归一化画面坐标；取邻域中位数抗噪。

    深度是**自我中心的相对测量**，真机装个深度相机就有 —— 与"目标坐标"那种外部先验不同，
    用它不破坏"无度量、目标坐标未知"的设定（台账 §2026-07-31 已界定）。
    返回 None 表示该处无有效深度（天空哨兵/越界）。
    """
    return sample_depth_array(ce.decode(msg), cx, cy, half=half, min_pixels=1)


def sample_target_depth(depth_msg, seg_msg, cx, cy, color_bgr, half=12,
                        color_tolerance=0, min_pixels=3):
    """只在目标 Actor 分割掩膜内取深度。

    返回 (distance, seg_valid, target_pixels)。分割只负责确认像素属于目标主体，
    部位方向仍由 DINO 指纹给出；不能把整机掩膜冒充机头/机翼定位。
    """
    if depth_msg is None or seg_msg is None or color_bgr is None:
        return None, False, 0
    depth = ce.decode(depth_msg)
    seg = ce.decode(seg_msg)[..., :3]
    dh, dw = depth.shape[:2]
    if seg.shape[:2] != (dh, dw):
        seg = cv2.resize(seg, (dw, dh), interpolation=cv2.INTER_NEAREST)
    mask = make_target_mask(seg, color_bgr, tolerance=color_tolerance)
    n_target = int(mask.sum())
    if n_target == 0:
        return None, False, 0
    dist = sample_depth_array(depth, cx, cy, half=half, target_mask=mask,
                              min_pixels=min_pixels)
    return dist, dist is not None, n_target


def save_run(out_dir, log, args, reached, grid, stop_reason=None):
    """存 signals.csv(标量诊断) + run.h5(每帧 rgb) + viz_dump/(rviz 可直接播: 轨迹+指纹热力图)。"""
    if not log["step"]:
        print("无记录可存(未进入闭环)"); return
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = np.array(log["pose"], np.float32)   # n,e,d,yaw
    sig = np.array(log["sig"], np.float32)     # cx,cy,mass,peak
    act = np.array(log["act"], np.float32)     # vn,ve,vd,yaw_rate
    tsec = np.array(log["t"], np.float32)
    with h5py.File(out_dir / "run.h5", "w") as f:
        f.create_dataset("rgb", data=np.stack(log["rgb"]).astype(np.uint8),
                         compression="gzip", compression_opts=4)
        f.create_dataset("pose", data=pose)
        f.create_dataset("sig", data=sig)
        f.create_dataset("action", data=act)
        f.create_dataset("t", data=tsec)
        f.attrs["pose_layout"] = "n,e,d,yaw"
        f.attrs["sig_layout"] = "cx,cy,mass,peak"
        f.attrs["action_layout"] = "v_forward,v_right,v_down,yaw_rate"   # 08-01 改机体系
        f.attrs["template"] = str(args.template)
        f.attrs["stats_episode"] = str(args.stats_episode)
        f.attrs["template_roi"] = (-1.0, -1.0, -1.0, -1.0) \
            if args.template_roi is None else args.template_roi
        f.attrs["roi_switch_dist"] = args.roi_switch_dist
        f.attrs["roi_switch_confirm_frames"] = args.roi_switch_confirm_frames
        f.attrs["controller"] = "servo_v2"
        if hasattr(args, "drone_name"):
            f.attrs["drone_name"] = str(args.drone_name)
        f.attrs["target"] = str(args.target)
        f.attrs["seg_depth_enabled"] = not bool(args.no_seg_depth)
        f.attrs["seg_color_bgr"] = (-1, -1, -1) if args.seg_color is None else args.seg_color
        f.attrs["ema_alpha"] = args.ema_alpha
        f.attrs["depth_ema_alpha"] = args.depth_ema_alpha
        f.attrs["control_min_peak"] = args.control_min_peak
        f.attrs["dist_min_peak"] = args.dist_min_peak
        f.attrs["stop_dist"] = -1.0 if args.stop_dist is None else args.stop_dist
        f.attrs["slow_dist"] = args.slow_dist
        f.attrs["stop_confirm_frames"] = args.stop_confirm_frames
        f.attrs["cmd_duration_factor"] = args.cmd_duration_factor
        f.attrs["mass_stop"] = args.mass_stop
        f.attrs["reached"] = bool(reached)
        # reached 单独看没有意义: 它把"到接触高度"和"目标充满视野"混为一谈。
        # stop_reason 才说得清是哪一种 —— evaluator 据此区分真接触与高度停止。
        f.attrs["stop_reason"] = stop_reason or ("mass" if reached else "timeout")
        f.attrs["action_frame"] = "body"       # 与采集/世界模型一致
        if log.get("dist"):
            f.create_dataset("dist", data=np.array(log["dist"], np.float32))   # 重心处深度(米), -1=无效
        if log.get("control"):
            f.create_dataset("control", data=np.array(log["control"], np.float32))
            f.attrs["control_layout"] = (
                "cx_f,cy_f,mass_f,peak_f,dist_f,locked,seg_valid,frame_dt,contact_count,"
                "target_pixels,roi_active"
            )   # 旧 run: 9列; 09-15 加 target_pixels; 09-16 加 roi_active
        if log.get("seg"):
            f.create_dataset("seg", data=np.stack(log["seg"]).astype(np.uint8),
                             compression="gzip", compression_opts=4)
    cols = ["step", "t", "n", "e", "d", "alt", "yaw", "cx", "cy", "mass", "peak",
            "vf", "vn", "ve", "vd", "yaw_rate", "slow", "cx_f", "cy_f", "mass_f",
            "peak_f", "dist_f", "locked", "seg_valid", "frame_dt", "contact_count",
            "target_pixels", "roi_active"]
    with open(out_dir / "signals.csv", "w") as f:
        f.write(f"# template={args.template} mass_stop={args.mass_stop} reached={bool(reached)}\n")
        f.write(",".join(cols) + "\n")
        for i in range(len(log["step"])):
            n, e, d, yaw = pose[i]; cx, cy, mass, peak = sig[i]; vn, ve, vd, yr = act[i]
            ctrl = log["control"][i] if log.get("control") else [np.nan] * 11
            f.write(",".join(map(str, [
                log["step"][i], f"{tsec[i]:.2f}", f"{n:.2f}", f"{e:.2f}", f"{d:.2f}", f"{-d:.2f}",
                f"{yaw:.3f}", f"{cx:.3f}", f"{cy:.3f}", f"{mass:.3f}", f"{peak:.3f}",
                f"{log['vf'][i]:.3f}", f"{vn:.3f}", f"{ve:.3f}", f"{vd:.3f}", f"{yr:.3f}",
                f"{log['slow'][i]:.3f}", *[f"{float(v):.4f}" for v in ctrl]])) + "\n")
    if grid is not None and log["sim"]:   # rviz 直接可播的每帧 npz(轨迹+热力图)
        vdir = out_dir / "viz_dump"; vdir.mkdir(exist_ok=True)
        for i in range(len(log["step"])):
            mass_f, peak_f = log["control"][i][2:4] if log.get("control") else sig[i, 2:4]
            np.savez_compressed(
                vdir / f"step_{i:04d}.npz",
                step=int(log["step"][i]), pose=pose[i], rgb=log["rgb"][i].astype(np.uint8),
                sim=np.asarray(log["sim"][i], np.float32), grid=grid, dt=args.dt,
                mass=np.float32(mass_f), peak=np.float32(peak_f))
        print(f"  rviz: ros2 launch src/mpc/plan_viz.launch.py dump_dir:={vdir}")
    print(f"运行数据已存: {out_dir}  ({len(log['step'])} 步, run.h5 + signals.csv + viz_dump/)")


async def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    log = {k: [] for k in ("step", "t", "pose", "sig", "act", "vf", "slow",
                           "rgb", "sim", "dist", "seg", "control")}
    reached = False
    stop_reason = "timeout"          # 跑满 max_steps 而没触发任何判据
    grid = None
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None; api_on = False
    pending = []
    reset_messages()
    try:
        client.connect(); print("已连接")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        if not args.no_seg_depth:
            ce.setup_segmentation(world)
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], make_cb("rgb"))
        # 深度进距离控制；分割只确认该深度属于目标主体，不负责区分机头/机翼等部位。
        for sensor, tag in (("depth_camera", "depth"), ("segmentation_camera", "seg")):
            try:
                client.subscribe(drone.sensors[args.camera][sensor], make_cb(tag))
            except Exception as err:
                print(f"未订阅 {sensor}（{err}）—— 该路数据缺失，相关判据自动退化")
        seg_color = tuple(args.seg_color) if args.seg_color is not None \
            else SEG_COLORS_BGR.get(args.target)
        args.seg_color = seg_color
        if not args.no_seg_depth:
            if seg_color is None:
                print(f"目标 {args.target} 没有已标定分割颜色；深度不会用于接触判据，请传 --seg-color B G R")
            else:
                print(f"目标分割约束: {args.target} BGR={seg_color}")
        fp = Fingerprint(args, device)
        grid = fp.grid

        assert drone.enable_api_control() and drone.arm(); api_on = True
        await (await drone.takeoff_async())
        n0, e0, d0 = get_pose_yaw(drone)[:3]
        print(f"爬升到 {args.start_altitude:.0f}m ...")
        await (await drone.move_to_position_async(
            north=n0, east=e0, down=-abs(args.start_altitude), velocity=10.0))
        if args.goto_ned is not None:              # 俯视: 开环飞到目标上空, 之后纯视觉
            gn, ge = args.goto_ned
            print(f"开环飞往目标上空 ({gn:.1f},{ge:.1f}) ...")
            await (await drone.move_to_position_async(
                north=gn, east=ge, down=-abs(args.start_altitude), velocity=10.0))
        if args.face_ned is not None:              # 前视: 仅初始化朝向, 让目标进画面
            fn, fe = args.face_ned
            nn, ee, _, _ = get_pose_yaw(drone)
            await asyncio.wait_for(await drone.rotate_to_yaw_async(
                yaw=math.atan2(fe - ee, fn - nn)), timeout=30.0)
        # 等一张属于当前场景的新帧。
        try:
            await wait_new_message("rgb", timeout=10.0)
        except TimeoutError:
            print("没收到帧"); return

        print(f"=== 进入视觉伺服闭环 v2 ({1 / args.dt:.1f}Hz, "
              f"cmd={args.dt * args.cmd_duration_factor:.2f}s) ===")
        t_loop = time.monotonic()
        last_rgb_ts = -1
        last_wall = None
        last_contact_depth_ts = -1
        filtered = {k: None for k in ("cx", "cy", "mass", "peak", "dist")}
        lost_frames = 0
        lock_streak = 0
        tracking_acquired = False
        contact_count = 0
        roi_active = False
        roi_switch_count = 0
        for step in range(args.max_steps):
            msg = await wait_new_message("rgb", last_rgb_ts, timeout=args.frame_timeout)
            rgb_ts = int(msg.get("time_stamp", -1))
            last_rgb_ts = rgb_ts
            now = time.monotonic()
            frame_dt = args.dt if last_wall is None else now - last_wall
            last_wall = now
            rgb = decode_image(msg)[0]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            cx, cy, mass, peak, sim = fp.query(rgb, use_roi=roi_active)
            n, e, dz, yaw = get_pose_yaw(drone)

            filtered["cx"] = ema_update(filtered["cx"], cx, args.ema_alpha)
            filtered["cy"] = ema_update(filtered["cy"], cy, args.ema_alpha)
            filtered["mass"] = ema_update(filtered["mass"], mass, args.ema_alpha)
            filtered["peak"] = ema_update(filtered["peak"], peak, args.ema_alpha)
            cx_f, cy_f = filtered["cx"], filtered["cy"]
            mass_f, peak_f = filtered["mass"], filtered["peak"]
            locked = peak_f >= args.control_min_peak
            lock_streak = lock_streak + 1 if locked else 0
            tracking_acquired = tracking_acquired or lock_streak >= args.lock_confirm_frames
            lost_frames = 0 if locked else lost_frames + 1

            dmsg = get_latest("depth")
            smsg = get_latest("seg")
            # 消息跨流偏差过大时宁可不用深度，也不把旧掩膜套到新画面上。
            def fresh(sensor_msg):
                if sensor_msg is None or rgb_ts < 0:
                    return False
                return abs(int(sensor_msg.get("time_stamp", -1)) - rgb_ts) / 1e9 \
                    <= args.max_sensor_skew

            trust_depth = peak_f >= args.dist_min_peak and fresh(dmsg)
            seg_valid = False
            target_pixels = 0
            dist_t = None
            if trust_depth and args.no_seg_depth:
                dist_t = sample_depth(dmsg, cx_f, cy_f, half=args.depth_half)
            elif trust_depth and fresh(smsg):
                dist_t, seg_valid, target_pixels = sample_target_depth(
                    dmsg, smsg, cx_f, cy_f, seg_color, half=args.depth_half,
                    color_tolerance=args.seg_color_tolerance,
                    min_pixels=args.depth_min_pixels,
                )
            if dist_t is not None:
                filtered["dist"] = ema_update(filtered["dist"], dist_t, args.depth_ema_alpha)
            dist_f = filtered["dist"] if dist_t is not None else None

            # 全模板负责远段截获；可信目标距离连续进入阈值后，一次性切到局部部位原型。
            # 切换在本帧记录/控制之后生效，下一帧清空方向与距离 EMA，避免两种原型混滤。
            activate_roi_after_frame = False
            if fp.has_roi and not roi_active:
                roi_switch_hit = dist_f is not None and locked \
                    and dist_f <= args.roi_switch_dist
                roi_switch_count = roi_switch_count + 1 if roi_switch_hit else 0
                activate_roi_after_frame = roi_switch_count >= args.roi_switch_confirm_frames

            # 信号可靠时按正常伺服控制；失锁初期沿原航向低速爬行，持续失锁则悬停。
            if locked:
                depth_mode = dist_f is not None and (args.stop_dist is not None or args.slow_dist > 0)
                mass_for_control = 0.0 if depth_mode else mass_f
                fwd, rgt, vd, yaw_rate, slow, _ = compute_body_cmd(
                    args, cx_f, cy_f, mass_for_control)
            else:
                creep_allowed = not tracking_acquired or lost_frames <= args.lost_creep_frames
            if not locked and creep_allowed:
                fwd, rgt, vd, yaw_rate, slow = args.lost_forward, 0.0, 0.0, 0.0, 1.0
            elif not locked:
                fwd, rgt, vd, yaw_rate, slow = 0.0, 0.0, 0.0, 0.0, 0.0

            range_scale = depth_speed_scale(dist_f, args.slow_dist, args.stop_dist, args.slow_min)
            if dist_f is not None and args.slow_dist > 0:
                # 保持视线方向比例，只让偏航继续全速修正。
                fwd *= range_scale; rgt *= range_scale; vd *= range_scale

            # 接触必须是可靠目标深度连续命中，不能用同一张旧深度重复计数。
            depth_ts = -1 if dmsg is None else int(dmsg.get("time_stamp", -1))
            contact_hit = args.stop_dist is not None and dist_f is not None \
                and dist_f <= args.stop_dist and locked and depth_ts != last_contact_depth_ts
            if contact_hit:
                contact_count += 1
                last_contact_depth_ts = depth_ts
            elif depth_ts != last_contact_depth_ts:
                contact_count = 0
                last_contact_depth_ts = depth_ts
            if contact_count > 0:       # 候选接触期间先悬停，等独立新深度帧确认
                fwd = rgt = vd = yaw_rate = 0.0

            stop_now = None
            if args.stop_dist is not None and contact_count >= args.stop_confirm_frames:
                stop_now = "contact"
            elif args.stop_alt is not None and -dz <= args.stop_alt:
                stop_now = "altitude"
            elif args.stop_alt is None and args.stop_dist is None and mass_f >= args.mass_stop:
                stop_now = "mass"

            # 逐步记录(含触发停止的这一帧)
            log["step"].append(step); log["t"].append(time.monotonic() - t_loop)
            log["pose"].append([n, e, dz, yaw]); log["sig"].append([cx, cy, mass, peak])
            log["act"].append([fwd, rgt, vd, yaw_rate]); log["vf"].append(fwd); log["slow"].append(slow)
            log["rgb"].append(rgb.copy()); log["sim"].append(sim)
            log["dist"].append(-1.0 if dist_t is None else dist_t)
            log["control"].append([
                cx_f, cy_f, mass_f, peak_f, -1.0 if dist_f is None else dist_f,
                float(locked), float(seg_valid), frame_dt, contact_count,
                float(target_pixels),          # 目标掩膜像素数: 标定 --stop-dist 时要看
                float(roi_active),              # 0=全模板，1=模板 ROI
            ])
            if smsg is None:
                log["seg"].append(np.zeros_like(rgb))
            else:
                log["seg"].append(cv2.resize(ce.decode(smsg)[..., :3], rgb.shape[1::-1],
                                             interpolation=cv2.INTER_NEAREST))

            state = "LOCK" if locked else ("CREEP" if creep_allowed else "HOLD")
            print(f"[{step}] {state} alt={-dz:5.1f} center=({cx_f:+.2f},{cy_f:+.2f}) "
                  f"proto={'ROI' if roi_active else 'FULL'} mass={mass_f:.3f} peak={peak_f:.3f} "
                  f"dist={'  n/a' if dist_f is None else f'{dist_f:5.1f}'} "
                  f"seg={int(seg_valid)}({target_pixels}) | vf={fwd:.2f} vd={vd:.2f} "
                  f"yaw_rate={yaw_rate:+.2f} dt={frame_dt:.3f}")

            if stop_now is not None:
                reached, stop_reason = True, stop_now
                print(f"[{step}] 停止判据={stop_now}，连续深度确认={contact_count}")
                await (await drone.move_by_velocity_body_frame_async(
                    v_forward=0.0, v_right=0.0, v_down=0.0,
                    duration=max(args.dt, 0.2), yaw=0.0, yaw_is_rate=True))
                break

            if activate_roi_after_frame:
                roi_active = True
                roi_switch_count = 0
                for key in ("cx", "cy", "mass", "peak", "dist"):
                    filtered[key] = None
                lock_streak = 0
                lost_frames = 0
                contact_count = 0
                print(f"[{step}] 原型切换 FULL→ROI：连续 {args.roi_switch_confirm_frames} 帧 "
                      f"dist≤{args.roi_switch_dist:.1f}m；下一帧重置 EMA")

            # **机体系**；只提交任务、不等动作结束。下一帧的 DINO 推理期间该指令仍持续。
            # ProjectAirSim 的新 SingleTaskCall 会抢占旧任务，CancelledError 属正常；
            # duration 加余量，避免新帧/DINO稍慢时出现无控制空档。
            pending.append(await drone.move_by_velocity_body_frame_async(
                v_forward=fwd, v_right=rgt, v_down=vd,
                duration=args.dt * args.cmd_duration_factor,
                yaw=yaw_rate, yaw_is_rate=True))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        print("=== 闭环结束 ===", "已贴近 ✅" if reached else "跑满步数(未达停止阈值)")
    finally:
        if api_on and drone is not None:
            try:
                drone.disarm(); drone.disable_api_control()
            except Exception as e:
                print(f"释放控制异常(可忽略): {e}")
        client.disconnect(); print("已断开连接")
        if args.out_dir:
            tmpl = Path(args.template).stem
            stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
            save_run(Path(args.out_dir) / f"{tmpl}_{stamp}", log, args, reached, grid,
                     stop_reason=stop_reason)


def build_parser():
    base = PROJECT_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--scene", default="scene_airbase.jsonc")
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default="FrontCamera")
    p.add_argument("--target", default=ce.DEFAULT_TARGET,
                   help="目标 Actor；用于运行时分割 ID 覆盖和目标主体深度校验")
    p.add_argument("--template", type=Path, default=base / "pictures/尾翼.jpg")
    p.add_argument("--template-roi", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="模板归一化 ROI；不传则始终使用全模板。机头已验证区域: 0.20 0.68 0.80 1.00")
    p.add_argument("--roi-switch-dist", type=float, default=50.0,
                   help="有 --template-roi 时，可信目标深度连续低于此值后从全模板切到 ROI")
    p.add_argument("--roi-switch-confirm-frames", type=int, default=3,
                   help="FULL→ROI 切换所需连续可信距离帧数")
    p.add_argument("--stats-episode", type=Path,
                   default=base / "outputs/recordings/approach/airbase_tgt1_50m.h5")
    p.add_argument("--repo-dir", type=Path, default=base / "dinov2")
    p.add_argument("--dino-weights", type=Path, default=base / "weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--nadir", action="store_true",
                   help="俯视伺服模式(相机须 -90°): cx/cy 水平双闭环+居中才降")
    p.add_argument("--goto-ned", type=float, nargs=2, default=None, metavar=("N", "E"),
                   help="俯视: 先开环飞到该 NED 上空再进闭环")
    p.add_argument("--face-ned", type=float, nargs=2, default=None, metavar=("N", "E"),
                   help="前视: 初始一次性朝向(让目标进画面); 之后纯视觉")
    p.add_argument("--start-altitude", type=float, default=50.0)
    p.add_argument("--out-dir", default=str(base / "outputs/runs/servo"),
                   help="运行数据落盘根目录(每次跑自动建 <模板名>_<时间>/ 子目录存 run.h5+signals.csv); 传空关闭")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    # 控制增益
    p.add_argument("--v-forward", type=float, default=3.0, help="水平速度上限")
    p.add_argument("--v-down", type=float, default=1.2)
    p.add_argument("--k-yaw", type=float, default=0.8)
    p.add_argument("--yaw-max", type=float, default=0.6)
    p.add_argument("--k-vert", type=float, default=1.5,
                   help="斜视: cy(目标垂直偏差)→下降率增益(视线角控制, 让目标保持画面中心)")
    p.add_argument("--k-xy", type=float, default=3.0, help="俯视: 画面偏差→水平速度增益")
    p.add_argument("--center-gate", type=float, default=0.5,
                   help="俯视: 偏差超此值不下降, 以内线性放开下降")
    p.add_argument("--mass-stop", type=float, default=0.20, help="前视: mass 超过即判定贴近停止")
    p.add_argument("--control-min-peak", type=float, default=0.12,
                   help="低于此 peak 不使用指纹误差转向；先低速直行，持续失锁后悬停")
    p.add_argument("--ema-alpha", type=float, default=0.4,
                   help="cx/cy/mass/peak EMA 的当前帧权重，1=不滤波")
    p.add_argument("--depth-ema-alpha", type=float, default=0.5,
                   help="有效目标深度 EMA 的当前帧权重")
    p.add_argument("--lost-creep-frames", type=int, default=5,
                   help="已经稳定截获后，失锁时允许低速直行的帧数，之后悬停")
    p.add_argument("--lost-forward", type=float, default=0.5,
                   help="首次截获前/短暂失锁时的直行速度；不偏航、不下降")
    p.add_argument("--lock-confirm-frames", type=int, default=3,
                   help="连续多少帧达到 control-min-peak 才认为曾稳定截获")
    p.add_argument("--dist-min-peak", type=float, default=0.12,
                   help="指纹 peak 低于此值时不信任重心, 深度判据自动失效(防误判接触)。"
                        "默认与控制门控 --control-min-peak 对齐: 同一个 peak_f 不该一个说"
                        "\"可以转向\"、另一个说\"不能信距离\"。原值 0.35 会把末段深度整段关掉"
                        "(8 次实飞实测可用率 26%% → 0.12 时 86%%, 见 EXPERIMENTS.md §2026-09-15)")
    p.add_argument("--stop-dist", type=float, default=None,
                   help="接触判据: 指纹重心处深度 ≤ 此值(米)即判接触。比 --stop-alt/--mass-stop 都直接")
    p.add_argument("--slow-dist", type=float, default=0.0,
                   help=">0 时按到目标的距离线性减速(dist/slow_dist)，替代只按 mass 减速")
    p.add_argument("--slow-min", type=float, default=0.15, help="距离减速的下限系数")
    p.add_argument("--stop-confirm-frames", type=int, default=3,
                   help="接触深度需由多少张独立深度帧连续确认")
    p.add_argument("--no-seg-depth", action="store_true",
                   help="不使用目标 Actor 分割约束深度（部署无分割时的退化模式）")
    p.add_argument("--seg-color", type=int, nargs=3, default=None, metavar=("B", "G", "R"),
                   help="目标分割调色板 BGR；默认按 --target 使用本场景实测颜色")
    p.add_argument("--seg-color-tolerance", type=int, default=0,
                   help="分割 BGR 逐通道容差；调色板图通常用 0")
    p.add_argument("--depth-half", type=int, default=12,
                   help="指纹点周围深度采样窗口半径（像素，原始深度分辨率）")
    p.add_argument("--depth-min-pixels", type=int, default=3,
                   help="窗口内至少多少个目标掩膜深度像素才算有效")
    p.add_argument("--max-sensor-skew", type=float, default=0.25,
                   help="RGB 与 depth/seg 允许的最大时间差（秒；分割实测约滞后一帧）")
    p.add_argument("--stop-alt", type=float, default=None,
                   help="俯视: 高度降到此值即判定接触停止(俯视 mass 早饱和用高度判; 如接触≈14.6m 设 16)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.1,
                   help="控制/相机标称周期；当前 FrontCamera 为 10Hz")
    p.add_argument("--cmd-duration-factor", type=float, default=2.0,
                   help="速度指令保活时长=dt×该系数，用于覆盖下一帧等待和DINO推理")
    p.add_argument("--frame-timeout", type=float, default=2.0,
                   help="等待下一张 RGB 的超时秒数")
    # 指纹参数(与离线判据一致)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p


def validate_args(p, args):
    for name in ("ema_alpha", "depth_ema_alpha"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            p.error(f"--{name.replace('_', '-')} 必须在 (0,1]，当前 {value}")
    if args.dt <= 0 or args.cmd_duration_factor <= 0:
        p.error("--dt 和 --cmd-duration-factor 必须为正数")
    if args.stop_confirm_frames < 1 or args.lock_confirm_frames < 1 or args.depth_min_pixels < 1:
        p.error("--stop-confirm-frames、--lock-confirm-frames 和 --depth-min-pixels 必须 ≥1")
    if args.roi_switch_confirm_frames < 1:
        p.error("--roi-switch-confirm-frames 必须 ≥1")
    if args.template_roi is not None:
        x0, y0, x1, y1 = args.template_roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            p.error("--template-roi 必须满足 0 ≤ x0 < x1 ≤ 1 且 0 ≤ y0 < y1 ≤ 1")
        if args.roi_switch_dist <= 0:
            p.error("使用 --template-roi 时 --roi-switch-dist 必须为正数")
    if args.slow_dist > 0 and args.stop_dist is not None and args.slow_dist <= args.stop_dist:
        p.error("--slow-dist 必须大于 --stop-dist，才能形成有效减速区间")
    if not 0.0 <= args.slow_min <= 1.0:
        p.error("--slow-min 必须在 [0,1]")
    return args


def parse_args(argv=None):
    p = build_parser()
    return validate_args(p, p.parse_args(argv))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
