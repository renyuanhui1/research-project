"""探针：确认 FrontCamera 深度流的 topic 名 + 数据格式/单位。

只做一件事：连上仿真 → 打印相机下所有 topic key → 订到"深度"那路 → 抓一帧 →
把 encoding/dtype/形状/通道数/min/max/中心像素值全打印出来。
目的是在写实时建图前，先确认深度 topic 叫什么、单位是不是米，不硬编码瞎猜。

跑（在服务器连宿主机仿真）：
  python src/airsim/probe_depth.py --address 192.168.31.178
需先把 robot_..._sensors.jsonc 里 FrontCamera 的 image-type 2（深度）
capture-enabled/streaming-enabled 改成 true（已改）。
"""
import argparse
import asyncio
import threading
from pathlib import Path

import numpy as np
import projectairsim
from projectairsim import Drone, World

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_SCENE = "scene_drone_sensors.jsonc"
DEFAULT_CAMERA = "FrontCamera"

_lock = threading.Lock()
_latest = None


def _cb(topic, msg):
    global _latest
    with _lock:
        _latest = msg


def _get():
    with _lock:
        return _latest


def _dtype_from_encoding(enc):
    """按 ROS 风格 encoding 名推 numpy dtype，如 16UC1->uint16, 32FC1->float32。"""
    e = str(enc).lower()
    if "16u" in e:
        return np.uint16
    if "8u" in e:
        return np.uint8
    if "32f" in e:
        return np.float32
    return None


def describe(msg):
    """把一条深度 msg 的关键信息打印出来，按 encoding 自动选 dtype。"""
    h, w = msg.get("height"), msg.get("width")
    enc = msg.get("encoding")
    data = msg["data"]
    print(f"  encoding = {enc!r}")
    print(f"  height x width = {h} x {w}")
    print(f"  data 类型 = {type(data).__name__}, "
          f"len = {len(data) if hasattr(data, '__len__') else '?'}")

    dt = _dtype_from_encoding(enc)
    if dt is None:
        print(f"  ⚠ 不认识的 encoding={enc!r}，无法解码；原始前 16 字节: "
              f"{bytes(data[:16]) if hasattr(data, '__getitem__') else data}")
        return

    if isinstance(data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(data, dtype=dt)
    else:
        arr = np.asarray(data, dtype=dt)

    n = arr.size
    px = (h or 0) * (w or 0)
    print(f"  按 {np.dtype(dt).name} 解: 元素数={n}, H*W={px}, 通道数={n/px if px else float('nan'):.3f}")
    if not (px and n % px == 0):
        print(f"  ⚠ 仍无法整除，原始前 8 个值: {arr[:8]}")
        return

    d = arr.reshape(h, w, n // px)[..., 0].astype(np.float64)
    print(f"  原始值(整数刻度): min={d.min():.0f} max={d.max():.0f} mean={d.mean():.1f}")
    print(f"  中心像素原始值 = {d[h // 2, w // 2]:.0f}")
    print(f"  ==0 占比 = {(d == 0).mean() * 100:.1f}%   ==max 占比 = {(d == d.max()).mean() * 100:.1f}%")
    print(f"  → 若单位是毫米: min={d.min()/1000:.2f}m max={d.max()/1000:.2f}m 中心={d[h//2,w//2]/1000:.2f}m")
    print(f"  (请对照场景实际距离判断刻度：毫米? 厘米? 还是量程归一化的整数?)")


async def main(args):
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None
    api_on = False
    try:
        client.connect()
        print("已连接 ProjectAirSim")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        print(f"场景已加载: {args.scene}")
        drone = Drone(client, world, "Drone1")

        if args.camera not in drone.sensors:
            raise RuntimeError(f"没有相机 '{args.camera}'，现有: {list(drone.sensors)}")
        keys = list(drone.sensors[args.camera])
        print(f"\n=== {args.camera} 下的所有 topic key ===")
        for k in keys:
            print(f"  {k}: {drone.sensors[args.camera][k]}")

        # 找"深度"那路：key 含 depth 的优先；找不到就报出来让人工看
        depth_keys = [k for k in keys if "depth" in k.lower()]
        if not depth_keys:
            raise RuntimeError(
                f"未找到含 'depth' 的 topic key（现有 {keys}）。"
                f" 可能深度未开或命名不同，请把上面的 key 列表发我。")
        dk = depth_keys[0]
        print(f"\n订阅深度 topic: {dk} -> {drone.sensors[args.camera][dk]}")
        client.subscribe(drone.sensors[args.camera][dk], _cb)

        # 起飞让场景实时渲染出帧（相机 10Hz 需要仿真在跑）
        require = lambda name, r: print(f"{name}: {r}") or (
            None if r is True else (_ for _ in ()).throw(RuntimeError(f"{name} 失败: {r}")))
        require("enable_api_control", drone.enable_api_control()); api_on = True
        require("arm", drone.arm())
        print("起飞中...")
        await (await drone.takeoff_async())
        await asyncio.sleep(1.0)

        print("\n等待第一帧深度...")
        waited = 0.0
        while _get() is None and waited < 15.0:
            await asyncio.sleep(0.2); waited += 0.2
        msg = _get()
        if msg is None:
            raise TimeoutError("15s 内没收到深度帧，检查深度 capture/streaming 是否已开")
        print(f"收到深度帧（等待 {waited:.1f}s）:")
        describe(msg)

    finally:
        if api_on and drone is not None:
            try:
                await (await drone.land_async())
                drone.disarm(); drone.disable_api_control()
            except Exception as e:
                print(f"清理时忽略: {e}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="192.168.31.178")
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse()))
