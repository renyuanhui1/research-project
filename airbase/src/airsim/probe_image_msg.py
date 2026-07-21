"""probe_image_msg.py —— 抓一帧 FrontCamera 原始消息, 打印字段 + 行跨距自检。

目的: 排查解码条纹是不是 step(行跨距) 被忽略造成的。
只订阅一帧、打印 height/width/encoding/step/len(data) 并做整除校验, 不起飞、不存数据。

用法(UE 前台):
  python airbase/src/airsim/probe_image_msg.py --scene scene_airbase.jsonc
"""

import argparse
import threading
import time
from pathlib import Path

import projectairsim
from projectairsim import Drone, World

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"

_latest = None
_lock = threading.Lock()


def cb(*args):
    # 兼容 subscribe 回调签名: 可能是 (msg) 或 (topic, msg), 取最后一个为消息
    global _latest
    msg = args[-1]
    with _lock:
        if _latest is None:
            _latest = msg


def main(args):
    client = projectairsim.ProjectAirSimClient(address=args.address)
    try:
        client.connect()
        print("已连接")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        topic = drone.sensors[args.camera]["scene_camera"]
        client.subscribe(topic, cb)
        print(f"已订阅 {topic}, 等待一帧...")

        t0 = time.monotonic()
        while time.monotonic() - t0 < 15:
            with _lock:
                m = _latest
            if m is not None:
                break
            time.sleep(0.1)
        if _latest is None:
            print("15s 内没收到帧。"); return

        m = _latest
        h, w = int(m["height"]), int(m["width"])
        data = m["data"]
        n = len(data)
        step = m.get("step")
        print("\n=== 相机消息字段 ===")
        print("  所有 key:", list(m.keys()))
        print(f"  height={h}  width={w}  encoding={m.get('encoding')!r}  big_endian={m.get('big_endian')}")
        print(f"  step(行字节)={step}   len(data)={n}   data_type={type(data).__name__}")
        print("\n=== 行跨距自检 ===")
        print(f"  width*3 = {w*3}   width*4 = {w*4}   （step 应等于其中之一）")
        print(f"  h*w*3   = {h*w*3}   h*w*4 = {h*w*4}   （len(data) 应等于其中之一）")
        if isinstance(step, int) and step not in (w*3, w*4):
            print(f"  ⚠ step({step}) ≠ width*3/4 → 有行填充, reshape(h,w,c) 会错位 = 条纹根因!")
        if n not in (h*w*3, h*w*4) and isinstance(step, int):
            print(f"  ⚠ len(data)={n} = h*step? {n == h*step} → 需按 step 逐行切再丢弃填充")
        if step in (w*3, w*4) and n in (h*w*3, h*w*4):
            print("  ✓ step 与 len(data) 都是紧密排列, 不是行跨距问题(另找原因)")
    finally:
        client.disconnect()
        print("已断开连接")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--scene", default="scene_airbase.jsonc")
    p.add_argument("--camera", default="FrontCamera")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
