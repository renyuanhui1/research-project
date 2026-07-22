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

from decode_check import decode_image
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"

_latest = None
_lock = threading.Lock()


def cb(*args):
    global _latest
    with _lock:
        _latest = args[-1]


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


def save_run(out_dir, log, args, reached, grid):
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
        f.attrs["action_layout"] = "vn,ve,vd,yaw_rate"
        f.attrs["template"] = str(args.template)
        f.attrs["stats_episode"] = str(args.stats_episode)
        f.attrs["mass_stop"] = args.mass_stop
        f.attrs["reached"] = bool(reached)
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
    log = {k: [] for k in ("step", "t", "pose", "sig", "act", "vf", "slow", "rgb", "sim")}
    reached = False
    grid = None
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None; api_on = False
    try:
        client.connect(); print("已连接")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], cb)
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
        while _latest is None and time.monotonic() - t0 < 10:
            await asyncio.sleep(0.1)
        if _latest is None:
            print("没收到帧"); return

        print("=== 进入视觉伺服闭环 ===")
        t_loop = time.monotonic()
        for step in range(args.max_steps):
            with _lock:
                msg = _latest
            rgb = decode_image(msg)[0]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            cx, cy, mass, peak, sim = fp.query(rgb)
            n, e, dz, yaw = get_pose_yaw(drone)

            slow = max(0.0, 1.0 - mass / args.mass_stop)      # 越近越慢
            if args.nadir:
                # 俯视视线角控制: 画面上方=机体前方, 画面右=机体右。
                # cx/cy 双轴闭环压目标到画面中心; 越居中降得越快, 偏了先纠平再降。
                vfwd = -args.k_xy * cy
                vrt = args.k_xy * cx
                sp = math.hypot(vfwd, vrt)
                if sp > args.v_forward:
                    vfwd *= args.v_forward / sp; vrt *= args.v_forward / sp
                vn = vfwd * math.cos(yaw) - vrt * math.sin(yaw)
                ve = vfwd * math.sin(yaw) + vrt * math.cos(yaw)
                # 俯视下 mass 早早饱和, 不能拿它门控下降; 只由"是否居中"门控, 高度到 stop-alt 才停
                err = math.hypot(cx, cy)
                vd = args.v_down * max(0.0, 1.0 - err / args.center_gate)
                vf = math.hypot(vfwd, vrt); yaw_rate = 0.0
            else:
                # 斜视(-35°): cx 转向 + 前进 + cy 垂直闭环(视线角控制)。
                # cy>0=目标沉到画面下方→多降; cy≈0=居中→只匀速; 让目标保持画面中心=沿视线接触,
                # 不再像旧版开环匀速降(那样目标沉出画面→落在目标前方地面)。
                vf = args.v_forward * slow
                vn = vf * math.cos(yaw); ve = vf * math.sin(yaw)
                vd = max(0.0, args.v_down * slow + args.k_vert * cy)
                yaw_rate = float(np.clip(args.k_yaw * cx, -args.yaw_max, args.yaw_max))
            # 逐步记录(含触发停止的这一帧)
            log["step"].append(step); log["t"].append(time.monotonic() - t_loop)
            log["pose"].append([n, e, dz, yaw]); log["sig"].append([cx, cy, mass, peak])
            log["act"].append([vn, ve, vd, yaw_rate]); log["vf"].append(vf); log["slow"].append(slow)
            log["rgb"].append(rgb.copy()); log["sim"].append(sim)

            if args.stop_alt is not None and -dz <= args.stop_alt:
                print(f"[{step}] alt={-dz:.1f}≤{args.stop_alt} → 到达接触高度, 停"); reached = True; break
            if args.stop_alt is None and mass >= args.mass_stop:
                print(f"[{step}] mass={mass:.3f}≥{args.mass_stop} → 判定贴近目标, 停"); reached = True; break

            print(f"[{step}] alt={-dz:5.1f} center=({cx:+.2f},{cy:+.2f}) mass={mass:.3f} "
                  f"peak={peak:.3f} | vf={vf:.2f} vd={vd:.2f} yaw_rate={yaw_rate:+.2f}")
            await (await drone.move_by_velocity_async(
                v_north=vn, v_east=ve, v_down=vd, duration=args.dt,
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
            save_run(Path(args.out_dir) / f"{tmpl}_{stamp}", log, args, reached, grid)


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
