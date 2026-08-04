"""servo_closed_loop.py —— 方案A: 指纹视觉伺服闭环(不用世界模型)。

用已验证单调的模板指纹信号, 实时驱动无人机飞向目标:
  每步: 取帧 → DINO编码 → 标准化 → 指纹响应 → 得 center(目标方位)/mass(大小)/peak
不依赖 predictor。用于验证"指纹信号能不能实时把无人机领到目标跟前"。

两种模式:
  默认斜视(相机 -35°, 当前主用): --face-ned 一次性对准 → cx 转向 + 前进 +
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
# 原来写死 parents[2]，脚本一挪进子目录就指错，README 为此专门警告过「不要把脚本挪进子目录」。
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


def make_cb(tag):
    def _cb(*args):
        with _lock:
            _msgs[tag] = args[-1]
    return _cb


cb = make_cb("rgb")              # 兼容旧调用（PX4 版有自己的 cb，不受影响）


def get_latest(tag="rgb"):
    with _lock:
        return _msgs.get(tag)


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
        # 模板 proto
        tbgr = cv2.imread(str(args.template))
        if tbgr is None:
            raise SystemExit(f"读不到模板: {args.template}")
        zt = self._encode(np.ascontiguousarray(tbgr[:, :, ::-1])[None])[0].to(device)
        zt = (zt - self.zm) / self.zs
        self.proto = F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0)
        P = zf.shape[1]; self.grid = int(round(P ** 0.5))
        xs = torch.linspace(-1.0, 1.0, self.grid, device=device)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
        print(f"指纹就绪: proto D={self.proto.shape[0]}, grid={self.grid}")

    @torch.no_grad()
    def _encode(self, rgb, bs=32):
        out = []
        for i in range(0, len(rgb), bs):
            x = to_input_tensor(rgb[i:i + bs], self.args.image_size, self.mean, self.std, self.dev)
            out.append(self.dino.forward_features(x)["x_norm_patchtokens"].float().cpu())
        return torch.cat(out, 0)

    @torch.no_grad()
    def query(self, rgb224):
        """rgb224:(224,224,3) uint8 → (center_x, center_y, mass, peak)。"""
        x = to_input_tensor(rgb224[None], self.args.image_size, self.mean, self.std, self.dev)
        z = self.dino.forward_features(x)["x_norm_patchtokens"][0].float()
        z = (z - self.zm) / self.zs
        sim = torch.matmul(F.normalize(z, dim=-1), self.proto)      # (P,) 即热力图
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
    d = ce.decode(msg)[..., 0].astype(np.float32)
    h, w = d.shape
    u = int(round((cx + 1) * 0.5 * (w - 1)))
    v = int(round((cy + 1) * 0.5 * (h - 1)))
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    patch = d[v0:v1, u0:u1]
    ok = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 60000.0)]
    return float(np.median(ok)) if ok.size else None


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
        f.attrs["mass_stop"] = args.mass_stop
        f.attrs["reached"] = bool(reached)
        # reached 单独看没有意义: 它把"到接触高度"和"目标充满视野"混为一谈。
        # stop_reason 才说得清是哪一种 —— evaluator 据此区分真接触与高度停止。
        f.attrs["stop_reason"] = stop_reason or ("mass" if reached else "timeout")
        f.attrs["action_frame"] = "body"       # 与采集/世界模型一致
        if log.get("dist"):
            f.create_dataset("dist", data=np.array(log["dist"], np.float32))   # 重心处深度(米), -1=无效
        if log.get("seg"):
            f.create_dataset("seg", data=np.stack(log["seg"]).astype(np.uint8),
                             compression="gzip", compression_opts=4)
    cols = ["step", "t", "n", "e", "d", "alt", "yaw", "cx", "cy", "mass", "peak",
            "vf", "vn", "ve", "vd", "yaw_rate", "slow"]
    with open(out_dir / "signals.csv", "w") as f:
        f.write(f"# template={args.template} mass_stop={args.mass_stop} reached={bool(reached)}\n")
        f.write(",".join(cols) + "\n")
        for i in range(len(log["step"])):
            n, e, d, yaw = pose[i]; cx, cy, mass, peak = sig[i]; vn, ve, vd, yr = act[i]
            f.write(",".join(map(str, [
                log["step"][i], f"{tsec[i]:.2f}", f"{n:.2f}", f"{e:.2f}", f"{d:.2f}", f"{-d:.2f}",
                f"{yaw:.3f}", f"{cx:.3f}", f"{cy:.3f}", f"{mass:.3f}", f"{peak:.3f}",
                f"{log['vf'][i]:.3f}", f"{vn:.3f}", f"{ve:.3f}", f"{vd:.3f}", f"{yr:.3f}",
                f"{log['slow'][i]:.3f}"])) + "\n")
    if grid is not None and log["sim"]:   # rviz 直接可播的每帧 npz(轨迹+热力图)
        vdir = out_dir / "viz_dump"; vdir.mkdir(exist_ok=True)
        for i in range(len(log["step"])):
            np.savez_compressed(
                vdir / f"step_{i:04d}.npz",
                step=int(log["step"][i]), pose=pose[i], rgb=log["rgb"][i].astype(np.uint8),
                sim=np.asarray(log["sim"][i], np.float32), grid=grid, dt=args.dt)
        print(f"  rviz: ros2 launch src/mpc/plan_viz.launch.py dump_dir:={vdir}")
    print(f"运行数据已存: {out_dir}  ({len(log['step'])} 步, run.h5 + signals.csv + viz_dump/)")


async def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    log = {k: [] for k in ("step", "t", "pose", "sig", "act", "vf", "slow",
                           "rgb", "sim", "dist", "seg")}
    reached = False
    stop_reason = "timeout"          # 跑满 max_steps 而没触发任何判据
    grid = None
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None; api_on = False
    try:
        client.connect(); print("已连接")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], make_cb("rgb"))
        # 深度进控制(距离减速/接触判据), 分割只进评价器。缺任一路不影响主流程: 取不到就退回老判据。
        for sensor, tag in (("depth_camera", "depth"), ("segmentation_camera", "seg")):
            try:
                client.subscribe(drone.sensors[args.camera][sensor], make_cb(tag))
            except Exception as err:
                print(f"未订阅 {sensor}（{err}）—— 该路数据缺失，相关判据自动退化")
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
        # 等一帧
        t0 = time.monotonic()
        while get_latest("rgb") is None and time.monotonic() - t0 < 10:
            await asyncio.sleep(0.1)
        if get_latest("rgb") is None:
            print("没收到帧"); return

        print("=== 进入视觉伺服闭环 ===")
        t_loop = time.monotonic()
        for step in range(args.max_steps):
            msg = get_latest("rgb")
            rgb = decode_image(msg)[0]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            cx, cy, mass, peak, sim = fp.query(rgb)
            n, e, dz, yaw = get_pose_yaw(drone)

            fwd, rgt, vd, yaw_rate, slow, vf = compute_body_cmd(args, cx, cy, mass)
            dmsg = get_latest("depth")
            # 指纹不可信时重心是噪声，采出来的深度是画面里随便某个点 —— 实测 08-01：
            # 20m 外 mass≈0.001、peak≈0.1~0.3，dist 在 44~60m 之间乱跳。
            # 不门控的话，飞过任何近处物体都会被误判成"接触"。
            trust = peak >= args.dist_min_peak
            dist_t = None if (dmsg is None or not trust) else sample_depth(dmsg, cx, cy)
            if dist_t is not None and args.slow_dist > 0:     # 按**到目标的距离**减速，而非按 mass
                vf_scale = min(1.0, max(args.slow_min, dist_t / args.slow_dist))
                fwd *= vf_scale; rgt *= vf_scale; vd *= vf_scale

            # 逐步记录(含触发停止的这一帧)
            log["step"].append(step); log["t"].append(time.monotonic() - t_loop)
            log["pose"].append([n, e, dz, yaw]); log["sig"].append([cx, cy, mass, peak])
            log["act"].append([fwd, rgt, vd, yaw_rate]); log["vf"].append(vf); log["slow"].append(slow)
            log["rgb"].append(rgb.copy()); log["sim"].append(sim)
            log["dist"].append(-1.0 if dist_t is None else dist_t)
            smsg = get_latest("seg")                          # 只为评价器留真值，不进控制
            if smsg is not None:
                log["seg"].append(cv2.resize(ce.decode(smsg)[..., :3], rgb.shape[1::-1],
                                             interpolation=cv2.INTER_NEAREST))

            # 停止判据。**分清是哪一种** —— 老版只有一个 reached=True，把"到接触高度"
            # 和"目标充满视野"混为一谈，评价时说不清到底接触没接触（台账 07-30 待办）。
            if args.stop_dist is not None and dist_t is not None and dist_t <= args.stop_dist:
                print(f"[{step}] 目标距离={dist_t:.1f}m ≤{args.stop_dist} → 接触判定, 停")
                reached, stop_reason = True, "contact"; break
            if args.stop_alt is not None and -dz <= args.stop_alt:
                print(f"[{step}] alt={-dz:.1f}≤{args.stop_alt} → 到达接触高度, 停")
                reached, stop_reason = True, "altitude"; break
            if args.stop_alt is None and args.stop_dist is None and mass >= args.mass_stop:
                print(f"[{step}] mass={mass:.3f}≥{args.mass_stop} → 判定贴近目标, 停")
                reached, stop_reason = True, "mass"; break

            print(f"[{step}] alt={-dz:5.1f} center=({cx:+.2f},{cy:+.2f}) mass={mass:.3f} "
                  f"peak={peak:.3f} dist={'  n/a' if dist_t is None else f'{dist_t:5.1f}'} | "
                  f"vf={fwd:.2f} vd={vd:.2f} yaw_rate={yaw_rate:+.2f}")
            # **机体系**：与采集(action_frame="body")和世界模型一致，A/B 两条件才可比。
            await (await drone.move_by_velocity_body_frame_async(
                v_forward=fwd, v_right=rgt, v_down=vd, duration=args.dt,
                yaw=yaw_rate, yaw_is_rate=True))
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


def parse_args():
    base = PROJECT_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--scene", default="scene_airbase.jsonc")
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default="FrontCamera")
    p.add_argument("--template", type=Path, default=base / "pictures/尾翼.jpg")
    p.add_argument("--stats-episode", type=Path,
                   default=base / "outputs/recordings/approach/airbase_tgt1_100m.h5")
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
    p.add_argument("--dist-min-peak", type=float, default=0.35,
                   help="指纹 peak 低于此值时不信任重心, 深度判据自动失效(防误判接触)")
    p.add_argument("--stop-dist", type=float, default=None,
                   help="接触判据: 指纹重心处深度 ≤ 此值(米)即判接触。比 --stop-alt/--mass-stop 都直接")
    p.add_argument("--slow-dist", type=float, default=0.0,
                   help=">0 时按到目标的距离线性减速(dist/slow_dist)，替代只按 mass 减速")
    p.add_argument("--slow-min", type=float, default=0.15, help="距离减速的下限系数")
    p.add_argument("--stop-alt", type=float, default=None,
                   help="俯视: 高度降到此值即判定接触停止(俯视 mass 早饱和用高度判; 如接触≈14.6m 设 16)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.3)
    # 指纹参数(与离线判据一致)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
