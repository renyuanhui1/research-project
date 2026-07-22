"""grab_nadir_frames.py —— 俯视(相机-90°)下, 悬到目标正上方垂直下降逐高度抓帧。

给俯视伺服/世界模型准备两样东西, 一次搞定:
  1) stats-episode(run.h5: rgb + pose[n,e,d,yaw]) —— 供伺服标准化 z_mean/z_std 用;
  2) 若干高度处的 png —— 供你裁"俯视整机模板"(50m 飞机须≥40m 高才拍得全, 从 ~45m 帧裁)。

与 record_approach 的区别: 那个飞斜线(只在低空才到目标正上方, 整机溢出);
本工具在目标正上方**垂直**降, 每个高度都是正俯视整机, 才能裁到全貌模板。

用法(UE 前台, 相机须已改 -90°):
  python airbase/src/airsim/grab_nadir_frames.py --target-ned -64.2 -18.5 \
      --start-alt 100 --end-alt 15
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

import projectairsim
from projectairsim import Drone, World

from decode_check import decode_image

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


async def grab(drone, dt=0.3):
    """等一帧新图并解码返回 (H,W,3) uint8。"""
    t0 = time.monotonic()
    while _latest is None and time.monotonic() - t0 < 5:
        await asyncio.sleep(0.05)
    with _lock:
        msg = _latest
    return decode_image(msg)[0]


async def main(args):
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None; api_on = False
    rgbs, poses = [], []
    try:
        client.connect(); print("已连接")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], cb)
        assert drone.enable_api_control() and drone.arm(); api_on = True
        await (await drone.takeoff_async())

        tn, te = args.target_ned
        # 先飞到目标正上方(起始高度)
        print(f"飞到目标正上方 ({tn:.1f},{te:.1f}) @ {args.start_alt:.0f}m ...")
        await (await drone.move_to_position_async(
            north=tn, east=te, down=-abs(args.start_alt), velocity=10.0))
        await asyncio.sleep(1.0)

        # 逐高度垂直下降抓帧
        alts = np.arange(args.start_alt, args.end_alt - 1e-6, -abs(args.step_alt))
        args.out_h5.parent.mkdir(parents=True, exist_ok=True)
        pic_dir = PROJECT_ROOT / "pictures"; pic_dir.mkdir(exist_ok=True)
        for a in alts:
            await (await drone.move_to_position_async(
                north=tn, east=te, down=-abs(float(a)), velocity=3.0))
            await asyncio.sleep(0.4)
            rgb = await grab(drone)
            n, e, d, yaw = get_pose_yaw(drone)
            rgbs.append(cv2.resize(rgb, (args.save_w, args.save_h), interpolation=cv2.INTER_AREA))
            poses.append([n, e, d, yaw])
            # 关键高度另存 png 供裁模板
            if any(abs(a - h) < abs(args.step_alt) / 2 for h in args.png_alts):
                p = pic_dir / f"nadir_{int(round(a)):03d}m.png"
                cv2.imwrite(str(p), rgb[:, :, ::-1])
                print(f"  alt={a:5.1f}m  抓帧+存 {p.name}")
            else:
                print(f"  alt={a:5.1f}m  抓帧")

        rgbs = np.stack(rgbs).astype(np.uint8)
        with h5py.File(args.out_h5, "w") as f:
            f.create_dataset("rgb", data=rgbs, compression="gzip", compression_opts=4)
            f.create_dataset("pose", data=np.array(poses, np.float32))
            f.attrs["pose_layout"] = "n,e,d,yaw"
            f.attrs["kind"] = "nadir_descent"
            f.attrs["target_ned"] = args.target_ned
        print(f"\n已存 stats-episode: {args.out_h5}  ({len(rgbs)} 帧)")
        print(f"裁模板: 用 pictures/nadir_*m.png 里 ~45m 那张裁整架飞机 → pictures/俯视整机.jpg")
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
    p.add_argument("--target-ned", type=float, nargs=2, required=True, metavar=("N", "E"))
    p.add_argument("--start-alt", type=float, default=100.0)
    p.add_argument("--end-alt", type=float, default=15.0)
    p.add_argument("--step-alt", type=float, default=5.0, help="每降多少米抓一帧")
    p.add_argument("--png-alts", type=float, nargs="+", default=[50, 45, 40, 30, 20],
                   help="这些高度另存 png 供裁模板")
    p.add_argument("--save-w", type=int, default=640)
    p.add_argument("--save-h", type=int, default=360)
    p.add_argument("--out-h5", type=Path,
                   default=base / "outputs/recordings/approach/airbase_tgt1_nadir.h5")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
