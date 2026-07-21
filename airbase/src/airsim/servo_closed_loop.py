"""servo_closed_loop.py —— 方案A: 指纹视觉伺服闭环(不用世界模型)。

用已验证单调的模板指纹信号, 实时驱动无人机飞向目标:
  每步: 取帧 → DINO编码 → 标准化 → 指纹响应 → 得 center(目标方位)/mass(大小)/peak
        → 转向对准(yaw) + 前进 + 下降; mass 够大即判定贴近、停。
不依赖 predictor。用于验证"指纹信号能不能实时把无人机领到目标跟前"。

标准化 z_mean/z_std 从 --stats-episode(录好的接近 h5)启动时算, 与离线判据同源。
初始朝向用 --face-ned 仅做一次性对准(让目标进画面), 之后纯靠视觉。

用法(UE 前台, 有 GPU):
  python airbase/src/airsim/servo_closed_loop.py --face-ned -64.2 -18.5
"""

import argparse
import asyncio
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
        P = zf.shape[1]; grid = int(round(P ** 0.5))
        xs = torch.linspace(-1.0, 1.0, grid, device=device)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
        print(f"指纹就绪: proto D={self.proto.shape[0]}, grid={grid}")

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
        sim = torch.matmul(F.normalize(z, dim=-1), self.proto)      # (P,)
        w = torch.softmax(sim / self.args.target_softmax_temp, dim=-1)
        center = torch.matmul(w, self.patch_xy)                     # (2,)
        mass = torch.sigmoid(
            (sim - self.args.target_mass_thresh) * self.args.target_mass_sharpness).mean()
        peak = sim.max()
        return float(center[0]), float(center[1]), float(mass), float(peak)


async def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
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

        assert drone.enable_api_control() and drone.arm(); api_on = True
        await (await drone.takeoff_async())
        n0, e0, d0 = get_pose_yaw(drone)[:3]
        print(f"爬升到 {args.start_altitude:.0f}m ...")
        await (await drone.move_to_position_async(
            north=n0, east=e0, down=-abs(args.start_altitude), velocity=10.0))
        if args.face_ned is not None:              # 仅初始化朝向, 让目标进画面
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
        reached = False
        for step in range(args.max_steps):
            with _lock:
                msg = _latest
            rgb = decode_image(msg)[0]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            cx, cy, mass, peak = fp.query(rgb)
            _, _, dz, yaw = get_pose_yaw(drone)

            if mass >= args.mass_stop:
                print(f"[{step}] mass={mass:.3f}≥{args.mass_stop} → 判定贴近目标, 停"); reached = True; break

            slow = max(0.0, 1.0 - mass / args.mass_stop)      # 越近越慢
            vf = args.v_forward * slow
            vn = vf * math.cos(yaw); ve = vf * math.sin(yaw)
            vd = args.v_down * slow                            # 下降, 近了收
            yaw_rate = float(np.clip(args.k_yaw * cx, -args.yaw_max, args.yaw_max))
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
    p.add_argument("--face-ned", type=float, nargs=2, default=None, metavar=("N", "E"),
                   help="初始一次性朝向(让目标进画面); 之后纯视觉")
    p.add_argument("--start-altitude", type=float, default=50.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    # 控制增益
    p.add_argument("--v-forward", type=float, default=3.0)
    p.add_argument("--v-down", type=float, default=1.2)
    p.add_argument("--k-yaw", type=float, default=0.8)
    p.add_argument("--yaw-max", type=float, default=0.6)
    p.add_argument("--mass-stop", type=float, default=0.20, help="mass 超过即判定贴近停止(离线20m处约0.15)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.3)
    # 指纹参数(与离线判据一致)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
