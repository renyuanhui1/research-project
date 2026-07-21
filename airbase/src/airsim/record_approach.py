"""录制"直线飞向目标"的完整接近 episode（供目标函数验证/标定用）。

与 collect_episode 的区别：不飞随机模板轨迹，而是
  起飞 → 爬到指定高度 → 原地转向目标 NED 坐标 → 匀速直线飞到目标前 stop-dist 处。
复用 collect_episode 的 run_episode/save_hdf5（事件驱动 10Hz、相同 HDF5 格式），
动作即真实下发的 NED 速度指令（带小噪声，与训练数据分布一致）。

用途：
  1) 覆盖全距离段（此前 episode 0050 只到目标前 ~19m，最后接近段无验证数据）；
  2) 飞完后从末段帧裁近景模板；
  3) 用 check_cost_monotonic 在这条 episode 上重跑单调性 + 重标定 goal-thresh。

用法（UE 前台！）：
  python src/airsim/record_approach.py                    # 默认飞向绿环 (42.3, 7.5)
  python src/airsim/record_approach.py --altitude 12      # 圆环中心偏高时调
注意：goal-ned 用 NED 米（UE cm / 100，相对出生点）。
"""

import argparse
import asyncio
import math
from pathlib import Path

import projectairsim
from projectairsim import Drone, World

from collect_episode import (
    DEFAULT_ADDRESS, DEFAULT_SCENE, DEFAULT_CONFIG_DIR, DEFAULT_CAMERA, DEFAULT_DT,
    image_callback, get_ned, require_success, run_episode, save_hdf5,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description="直线接近目标的 episode 录制")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "outputs/recordings/approach/approach_full.h5")
    p.add_argument("--goal-ned", type=float, nargs=2, default=[42.3, 7.5],
                   metavar=("N", "E"), help="目标 NED 坐标（米，UE cm/100）")
    p.add_argument("--speed", type=float, default=1.2, help="接近速度 (m/s)")
    p.add_argument("--stop-dist", type=float, default=2.0, help="停在目标前多少米")
    p.add_argument("--altitude", type=float, default=10.8,
                   help="飞行高度（米），默认与既有数据一致")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--seed", type=int, default=0)
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
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], image_callback)

        require_success("enable_api_control", drone.enable_api_control())
        api_on = True
        require_success("arm", drone.arm())
        print("起飞中...")
        await (await drone.takeoff_async())

        # 爬升到巡航高度
        n0, e0, d0 = get_ned(drone)
        task = await drone.move_to_position_async(
            north=n0, east=e0, down=-abs(args.altitude), velocity=2.0)
        await asyncio.wait_for(task, timeout=60.0)
        await asyncio.sleep(1.0)

        # 转向目标并规划直线段
        n0, e0, _ = get_ned(drone)
        gn, ge = args.goal_ned
        dist = math.hypot(gn - n0, ge - e0)
        heading = math.atan2(ge - e0, gn - n0)
        fly_dist = dist - args.stop_dist
        if fly_dist <= 0:
            raise SystemExit(f"已在目标 {dist:.1f}m 内，无需飞行")
        steps = int(fly_dist / args.speed / args.dt)
        print(f"当前 ({n0:.1f},{e0:.1f}) → 目标 ({gn:.1f},{ge:.1f})  "
              f"距离 {dist:.1f}m  航向 {math.degrees(heading):.1f}°  "
              f"飞 {fly_dist:.1f}m = {steps} 步 @ {args.speed}m/s")
        yaw_task = await drone.rotate_to_yaw_async(yaw=heading)
        await asyncio.wait_for(yaw_task, timeout=30.0)
        await asyncio.sleep(1.0)

        # 直线匀速段（NED 速度沿航向；yaw_rate=0 机头保持朝目标）
        vn = args.speed * math.cos(heading)
        ve = args.speed * math.sin(heading)
        trajectory = [(vn, ve, 0.0, 0.0, steps)]

        # run_episode 需要的字段。altitude 置 0：内部 goto_start 按"相对当前再爬升
        # altitude 米"工作，而我们已在上面爬到位，避免重复爬高。
        args.steps = steps
        args.randomize_start = False
        args.altitude = 0.0
        rgb, act, pose, tstamp, _ = await run_episode(
            drone, args, trajectory=trajectory, seed=args.seed)
        save_hdf5(args.output, args, rgb, act, pose, tstamp,
                  {"kind": "approach", "goal_ned": args.goal_ned,
                   "speed": args.speed, "stop_dist": args.stop_dist})
        nf, ef, _ = get_ned(drone)
        print(f"结束位置 ({nf:.1f},{ef:.1f})，距目标 "
              f"{math.hypot(gn - nf, ge - ef):.1f}m，已存 {args.output}")

    finally:
        if api_on and drone is not None:
            try:
                await (await drone.land_async())
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"降落/释放控制异常（可忽略）: {err}")
        client.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
