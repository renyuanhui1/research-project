"""脚本 2：collect_episode.py —— 单条 episode 采集

目的：飞一条预设(任务相关)轨迹，同步记录 (图像, 动作, 位姿, 时间戳)，存成 HDF5。
复用脚本 1 的 decode_image()；图像解码后 resize 到 224×224 存储。

时间对齐约定（关键，否则训练数据是错的）：
    每步先取当前帧 image_t / pose_t / time_t，再下发动作 a_t，记录 (image_t, a_t, pose_t, time_t)。
    即 a_t 是"导致 image_t → image_{t+1} 的那个动作"。整条序列保持一致。

HDF5 datasets：
    rgb    (T,224,224,3) uint8
    action (T,4)         float32   # vx, vy, vz, yaw_rate
    pose   (T,7)         float32   # pos_xyz + quat_wxyz（调试/replay 用，不喂模型）
    time   (T,)          int64     # time_stamp（纳秒）
"""

import argparse
import asyncio
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import projectairsim
from projectairsim import Drone, World

from decode_check import decode_image  # 复用脚本 1 的解码（同目录）


# ---- 配置（魔法数集中于此）----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADDRESS = "172.21.192.1"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_SCENE = "scene_drone_sensors.jsonc"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/datasets/episodes/episode_000.h5"
DEFAULT_CAMERA = "FrontCamera"

STORE_HW = 224          # 存储分辨率（正方形）
DEFAULT_DT = 0.1        # 控制步长（秒）→ 10Hz
DEFAULT_STEPS = 200     # 步数 → 20s
CMD_DURATION_FACTOR = 1.5  # 指令时长 = dt × 此系数；整步内速度持续有效、无滑行空档
DEFAULT_ALTITUDE = 6.0  # 起飞后目标高度（米）

# 动作噪声标准差（叠加在标称动作上，便于多样化）
NOISE_V = 0.2           # 速度噪声 (m/s)
NOISE_YAW = 0.1         # 偏航率噪声 (rad/s)

# 起点随机化幅度
START_XY = 3.0          # 水平偏移幅度 (m)

# 任务相关的预设轨迹：分段 (vx, vy, vz, yaw_rate, steps)
# v_down 负值=上升；包含前进/爬升/横移/偏航的组合，不只飞直线。
TRAJECTORY = [
    (1.5, 0.0, 0.0, 0.0, 30),   # 直线前进
    (1.0, 0.0, -0.5, 0.0, 30),  # 前进 + 爬升
    (1.0, 1.0, 0.0, 0.3, 30),   # 前进 + 右移 + 右偏航
    (1.0, 0.0, 0.0, -0.4, 30),  # 前进 + 左偏航
    (0.8, -1.0, 0.4, 0.0, 30),  # 前进 + 左移 + 下降
    (1.5, 0.0, 0.0, 0.2, 50),   # 前进 + 缓右偏航
]


# ---- 共享缓存：Pub/Sub 回调把最新帧整条 msg 写进来 ----
_lock = threading.Lock()
_latest_msg = None


def image_callback(topic, image_msg):
    global _latest_msg
    with _lock:
        _latest_msg = image_msg


def get_latest():
    with _lock:
        return _latest_msg


def reset_cache():
    """清空最新帧缓存（批量采集在场景重置后调用，避免读到上一条的残留帧）。"""
    global _latest_msg
    with _lock:
        _latest_msg = None


def extract_pose(msg):
    """从帧自带字段取 pose = [pos_xyz, quat_wxyz]（7 维）。"""
    return np.array(
        [msg["pos_x"], msg["pos_y"], msg["pos_z"],
         msg["rot_w"], msg["rot_x"], msg["rot_y"], msg["rot_z"]],
        dtype=np.float32,
    )


def build_nominal(trajectory, steps):
    """把分段轨迹展开成每步标称动作 (steps, 4)，不足则循环、超出则截断。"""
    seq = []
    for vx, vy, vz, yr, n in trajectory:
        seq.extend([(vx, vy, vz, yr)] * n)
    if not seq:
        raise ValueError("trajectory 为空")
    while len(seq) < steps:
        seq.extend(seq)
    return np.array(seq[:steps], dtype=np.float32)


def require_success(name, result):
    print(f"{name}: {result}")
    if result is not True:
        raise RuntimeError(f"{name} 失败: {result}")


async def wait_first_frame(timeout=15.0):
    waited = 0.0
    while get_latest() is None:
        if waited >= timeout:
            raise TimeoutError(f"{timeout}s 内未收到相机帧，检查 capture-enabled / topic")
        await asyncio.sleep(0.2)
        waited += 0.2


async def wait_new_frame(last_ts, timeout=2.0, poll=0.002):
    """等待一张时间戳不同于 last_ts 的新帧，返回 (msg, waited_seconds)。

    事件驱动节拍的核心：相机配为 10Hz 时，每步在此处等到下一张新帧，
    dt 自然 = 相机周期 ≈ 0.1s（均匀）。waited≈0 说明取帧时已有新帧在排队，即 CPU/RPC 赶不上相机。
    """
    waited = 0.0
    while True:
        msg = get_latest()
        if msg is not None and int(msg["time_stamp"]) != last_ts:
            return msg, waited
        if waited >= timeout:
            raise TimeoutError("等待新相机帧超时（检查相机出帧 / capture-interval）")
        await asyncio.sleep(poll)
        waited += poll


def get_ned(drone):
    pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    return float(pos["x"]), float(pos["y"]), float(pos["z"])


async def goto_start(drone, args, rng):
    """起飞到目标高度，并做起点位姿随机化。"""
    n, e, d = get_ned(drone)
    off_n = rng.uniform(-START_XY, START_XY) if args.randomize_start else 0.0
    off_e = rng.uniform(-START_XY, START_XY) if args.randomize_start else 0.0
    task = await drone.move_to_position_async(
        north=n + off_n, east=e + off_e, down=d - args.altitude, velocity=2.0
    )
    await asyncio.wait_for(task, timeout=30.0)
    await asyncio.sleep(1.0)  # settle


async def run_episode(drone, args, trajectory=TRAJECTORY, seed=None):
    """飞一条轨迹并返回 (rgb, action, pose, time, stats)；不负责存盘（供脚本2/4 复用）。"""
    rng = np.random.default_rng(args.seed if seed is None else seed)
    nominal = build_nominal(trajectory, args.steps)

    await goto_start(drone, args, rng)
    await wait_first_frame()
    print(f"开始采集：{args.steps} 步 @ {1/args.dt:.0f}Hz")

    rgb_buf = np.empty((args.steps, STORE_HW, STORE_HW, 3), dtype=np.uint8)
    act_buf = np.empty((args.steps, 4), dtype=np.float32)
    pose_buf = np.empty((args.steps, 7), dtype=np.float32)
    time_buf = np.empty((args.steps,), dtype=np.int64)

    # 事件驱动的均匀 10Hz：相机配为 10Hz，每步等"下一张新帧"到达即记录并派发动作，
    # dt 自然 = 相机周期 ≈ 0.1s（均匀，无 64/135 双峰抖动）。
    # 派发只发送、不等执行完成；指令时长 dt×1.5 使整步内速度持续有效、无滑行空档，下一条覆盖它。
    cmd_duration = args.dt * CMD_DURATION_FACTOR
    pending = []        # 后台指令任务，循环末统一回收
    behind = 0          # 取帧时已有新帧在排队（waited≈0）→ CPU/RPC 赶不上相机的步数
    init = get_latest()
    last_ts = int(init["time_stamp"]) if init is not None else -1
    t_start = time.monotonic()

    for t in range(args.steps):
        # 1) 等下一张新帧 = image_t（事件驱动，相机周期即为 dt）
        msg, waited = await wait_new_frame(last_ts)
        if waited <= 0.0:
            behind += 1
        ts = int(msg["time_stamp"])
        last_ts = ts
        rgb = decode_image(msg)[0]
        rgb = cv2.resize(rgb, (STORE_HW, STORE_HW), interpolation=cv2.INTER_AREA)

        # 2) 计算本步动作 a_t（标称 + 噪声）
        noise = rng.normal(0.0, [NOISE_V, NOISE_V, NOISE_V, NOISE_YAW]).astype(np.float32)
        a = nominal[t] + noise

        # 3) 记录 (image_t, a_t, pose_t, time_t)
        rgb_buf[t] = rgb
        act_buf[t] = a
        pose_buf[t] = extract_pose(msg)
        time_buf[t] = ts

        # 4) 派发 a_t（只发送、不等完成），驱动 image_t → image_{t+1}
        pending.append(await drone.move_by_velocity_async(
            v_north=float(a[0]), v_east=float(a[1]), v_down=float(a[2]),
            duration=cmd_duration, yaw=float(a[3]), yaw_is_rate=True,
        ))

    elapsed = time.monotonic() - t_start
    avg_hz = args.steps / elapsed
    print(f"采集完成：实际 {elapsed:.2f}s，平均 {avg_hz:.2f}Hz，赶不上相机 {behind} 步")
    await asyncio.gather(*pending, return_exceptions=True)  # 回收后台指令任务

    stats = {"elapsed": elapsed, "avg_hz": avg_hz, "behind": behind}
    return rgb_buf, act_buf, pose_buf, time_buf, stats


def save_hdf5(out, args, rgb, act, pose, tstamp, extra_attrs=None):
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=4)
        f.create_dataset("action", data=act)
        f.create_dataset("pose", data=pose)
        f.create_dataset("time", data=tstamp)
        f.attrs["dt"] = args.dt
        f.attrs["steps"] = args.steps
        f.attrs["store_hw"] = STORE_HW
        f.attrs["camera"] = args.camera
        f.attrs["scene"] = args.scene
        f.attrs["action_layout"] = "vx,vy,vz,yaw_rate"
        f.attrs["pose_layout"] = "px,py,pz,qw,qx,qy,qz"
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
    print(f"已存 HDF5: {out}  rgb{rgb.shape} action{act.shape} pose{pose.shape}")


def parse_args():
    p = argparse.ArgumentParser(description="单条 episode 采集 → HDF5")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--altitude", type=float, default=DEFAULT_ALTITUDE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize-start", action="store_true")
    return p.parse_args()


async def main(args):
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None
    api_on = False
    try:
        client.connect()
        print("已连接 ProjectAirSim")
        world = World(
            client, args.scene,
            sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
            delay_after_load_sec=2,
        )
        print(f"场景已加载: {args.scene}")
        drone = Drone(client, world, "Drone1")
        print("Drone1 已初始化")

        if args.camera not in drone.sensors or "scene_camera" not in drone.sensors[args.camera]:
            raise RuntimeError(f"相机 '{args.camera}' 未配置 scene_camera")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], image_callback)
        print(f"已订阅: {drone.sensors[args.camera]['scene_camera']}")

        require_success("enable_api_control", drone.enable_api_control())
        api_on = True
        require_success("arm", drone.arm())
        print("起飞中...")
        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        rgb, act, pose, tstamp, _ = await run_episode(drone, args)
        save_hdf5(args.output, args, rgb, act, pose, tstamp, {"seed": args.seed})

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
