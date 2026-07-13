"""脚本 1：decode_check.py —— 解码验证（数据闭环第一步）

目的：确认能把 ProjectAirSim 相机 Pub/Sub 帧 image_msg["data"] 正确解成图像并存 png。
做法：连接 → 加载场景 → 初始化 Drone1 → 订阅 FrontCamera/scene_camera（Pub/Sub）→
      回调缓存最新帧 → 起飞飞几秒 → 存 3~5 张不同时刻的图，肉眼确认画面正确。

通过标准：存出的 png 是清晰正确的仿真画面（不是白图/花屏，红蓝不反）。
"""

import argparse
import asyncio
import threading
from pathlib import Path

import cv2
import numpy as np
import projectairsim
from projectairsim import Drone, World


# ---- 配置（魔法数集中于此，便于调）----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADDRESS = "172.21.192.1"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_SCENE = "scene_drone_sensors.jsonc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/captures/decode_check"
DEFAULT_CAMERA = "FrontCamera"
DEFAULT_FRAMES = 5          # 存几张图
DEFAULT_INTERVAL = 0.5      # 两张图之间的间隔（秒）


# ---- 共享缓存：Pub/Sub 回调把最新帧写进来 ----
_latest_lock = threading.Lock()
_latest_msg = None
_frame_count = 0
_printed_once = False


def image_callback(topic, image_msg):
    """FrontCamera/scene_camera 的回调。ProjectAirSim 传感器走 Pub/Sub，帧从这里来。

    image_msg（已确认字段）：
        time_stamp, height, width, encoding, big_endian, step, data,
        pos_x..pos_z, rot_w..rot_z, annotations
    """
    global _latest_msg, _frame_count, _printed_once
    with _latest_lock:
        _latest_msg = image_msg
        _frame_count += 1
        if not _printed_once:
            _printed_once = True
            h, w = image_msg["height"], image_msg["width"]
            data = image_msg["data"]
            print(
                "[首帧] "
                f"encoding={image_msg['encoding']!r} height={h} width={w} "
                f"step={image_msg.get('step')} len(data)={len(data)} "
                f"data_type={type(data).__name__}"
            )


def decode_image(image_msg):
    """把一条相机 image_msg 解成标准 RGB 的 numpy 数组 (H, W, 3) uint8。

    - 通道数 C 用 len(data)/(H*W) 反推校验，不硬编码。
    - 按 encoding 决定通道顺序，统一转成标准 RGB。
    返回 (rgb, info)；info 含 encoding/channels 便于打印确认。
    """
    h, w = image_msg["height"], image_msg["width"]
    data = image_msg["data"]
    enc = str(image_msg["encoding"]).lower()

    # Pub/Sub 经 msgpack 传来的是 bytes/buffer；reqrep 经 JSON 传来的是 list。
    if isinstance(data, list):
        arr = np.array(data, dtype=np.uint8)
    else:
        arr = np.frombuffer(data, dtype=np.uint8)

    n = arr.size
    pixels = h * w
    if pixels == 0 or n % pixels != 0:
        raise ValueError(f"len(data)={n} 无法被 H*W={pixels} 整除，无法反推通道数")
    channels = n // pixels

    img = arr.reshape(h, w, channels)

    # 统一转成标准 RGB
    if channels == 3:
        rgb = img[..., ::-1] if enc.startswith("bgr") else img
    elif channels == 4:
        # bgra/rgba：先取前三通道，再按需翻转
        rgb = img[..., 2::-1] if enc.startswith("bgr") else img[..., :3]
    elif channels == 1:
        rgb = np.repeat(img, 3, axis=2)
    else:
        raise ValueError(f"未预期的通道数 channels={channels} (encoding={enc})")

    return np.ascontiguousarray(rgb), {"encoding": enc, "channels": channels}


def save_rgb(rgb, file_path):
    """存 png。cv2 写盘需 BGR，所以把标准 RGB 翻回 BGR 再写。"""
    if not cv2.imwrite(str(file_path), rgb[..., ::-1]):
        raise RuntimeError(f"写图失败: {file_path}")


def require_success(name, result):
    print(f"{name}: {result}")
    if result is not True:
        raise RuntimeError(f"{name} 失败: {result}")


async def wait_first_frame(timeout=15.0):
    """等待回调收到第一帧。"""
    waited = 0.0
    while True:
        with _latest_lock:
            if _latest_msg is not None:
                return
        if waited >= timeout:
            raise TimeoutError(f"{timeout}s 内未收到任何相机帧，检查 capture-enabled / topic")
        await asyncio.sleep(0.2)
        waited += 0.2


def parse_args():
    p = argparse.ArgumentParser(description="解码验证：订阅相机帧并存 png")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    return p.parse_args()


async def main(args):
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None
    api_on = False
    try:
        client.connect()
        print("已连接 ProjectAirSim")

        world = World(
            client,
            args.scene,
            sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
            delay_after_load_sec=2,
        )
        print(f"场景已加载: {args.scene}")

        drone = Drone(client, world, "Drone1")
        print("Drone1 已初始化")

        if args.camera not in drone.sensors or "scene_camera" not in drone.sensors[args.camera]:
            raise RuntimeError(f"相机 '{args.camera}' 未配置 scene_camera")
        rgb_topic = drone.sensors[args.camera]["scene_camera"]
        client.subscribe(rgb_topic, image_callback)
        print(f"已订阅相机 topic: {rgb_topic}")

        # 起飞，确保相机在动、有帧
        require_success("enable_api_control", drone.enable_api_control())
        api_on = True
        require_success("arm", drone.arm())
        print("起飞中...")
        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        await wait_first_frame()
        print(f"已收到首帧，开始存图（共 {args.frames} 张，每 {args.interval}s 一张）")

        # 边飞边存不同时刻的图（前进 + 上升，画面有变化）
        total = max(args.frames * args.interval + 1.0, 3.0)
        move_task = await drone.move_by_velocity_async(
            v_north=1.0, v_east=0.0, v_down=-0.5, duration=total
        )

        for i in range(args.frames):
            await asyncio.sleep(args.interval)
            with _latest_lock:
                msg = _latest_msg
            rgb, info = decode_image(msg)
            path = output_dir / f"frame_{i:03d}.png"
            save_rgb(rgb, path)
            print(f"  存图 {path}  shape={rgb.shape} {info}")

        await move_task
        print(f"完成，共收到 {_frame_count} 帧，存了 {args.frames} 张图到 {output_dir}")

    finally:
        if api_on and drone is not None:
            try:
                land_task = await drone.land_async()
                await land_task
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"降落/释放控制异常（可忽略）: {err}")
        client.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
