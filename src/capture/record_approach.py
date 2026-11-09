"""按显式目标坐标录制直线接近episode。

通过--goal-ned N E与--goal-alt ALT指定目标，配置--speed和--stop-dist控制采集航线。
复用capture.collect_episode的采集和HDF5保存函数。此入口用于采集图像及模板校验素材。"""

import argparse
import asyncio
import math
from pathlib import Path

import projectairsim
from projectairsim import Drone, World

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# Allow both direct script execution and package imports.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()

from capture.collect_episode import (
    DEFAULT_ADDRESS, DEFAULT_SCENE, DEFAULT_CONFIG_DIR, DEFAULT_CAMERA, DEFAULT_DT,
    image_callback, get_ned, require_success, run_episode, save_hdf5,
)



def parse_args():
    p = argparse.ArgumentParser(description="直线接近目标的 episode 录制")
    p.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--output", type=Path,
                   default=project_path(PROFILE["assets"]["stats_episode"]))
    p.add_argument("--goal-ned", type=float, nargs=2, required=True,
                   metavar=("N", "E"), help="目标水平 NED 坐标（米，UE cm/100）")
    p.add_argument("--goal-alt", type=float, required=True,
                   help="目标点离地高度（米，UE z cm/100），斜下降终点对准此高度")
    p.add_argument("--speed", type=float, default=1.2, help="接近速度 (m/s)")
    p.add_argument("--stop-dist", type=float, default=2.0, help="停在目标前多少米(3D)")
    p.add_argument("--altitude", type=float, default=10.8,
                   help="斜下降起始高度（米）")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


async def main(args):
    from common.runtime import configure_client_log
    configure_client_log()
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
            north=n0, east=e0, down=-abs(args.altitude), velocity=10.0)
        await asyncio.wait_for(task, timeout=60.0)
        await asyncio.sleep(1.0)

        # 转向目标并规划 3D 斜下降直线段（含垂直分量）
        n0, e0, d0 = get_ned(drone)
        gn, ge = args.goal_ned
        gd = -abs(args.goal_alt)              # 目标点 NED down（负=离地高度）
        dn, de, dd = gn - n0, ge - e0, gd - d0
        dist3d = math.sqrt(dn * dn + de * de + dd * dd)
        heading = math.atan2(de, dn)          # 水平航向（机头对目标）
        fly_dist = dist3d - args.stop_dist
        if fly_dist <= 0:
            raise SystemExit(f"已在目标 {dist3d:.1f}m 内，无需飞行")
        steps = int(fly_dist / args.speed / args.dt)
        print(f"当前 ({n0:.1f},{e0:.1f},高{-d0:.1f}m) → 目标 ({gn:.1f},{ge:.1f},高{-gd:.1f}m)  "
              f"3D距离 {dist3d:.1f}m  航向 {math.degrees(heading):.1f}°  "
              f"飞 {fly_dist:.1f}m = {steps} 步 @ {args.speed}m/s（斜下降）")
        yaw_task = await drone.rotate_to_yaw_async(yaw=heading)
        await asyncio.wait_for(yaw_task, timeout=30.0)
        await asyncio.sleep(1.0)

        # 3D 速度沿直线单位向量（含 v_down 下降分量；yaw_rate=0 机头保持朝目标）。
        # run_episode 已改为**机体系**下发：上面刚转向目标，故机体 forward 就是水平接近方向，
        # 省掉了原来 vn/ve 的分解（等价，且不再依赖转向精度）。
        uz = dd / dist3d
        v_forward = args.speed * math.hypot(dn, de) / dist3d
        v_down = args.speed * uz              # 正=下降
        trajectory = [(v_forward, 0.0, v_down, 0.0, steps)]

        args.steps = steps
        bufs, _ = await run_episode(drone, args, trajectory, seed=args.seed)
        save_hdf5(args.output, args, bufs, None,
                  {"kind": "approach", "goal_ned": args.goal_ned,
                   "speed": args.speed, "stop_dist": args.stop_dist})
        nf, ef, _ = get_ned(drone)
        print(f"结束位置 ({nf:.1f},{ef:.1f})，距目标 "
              f"{math.hypot(gn - nf, ge - ef):.1f}m，已存 {args.output}")

    finally:
        # 录完直接断开(不降落, 省时且避免 land 卡住)
        if api_on and drone is not None:
            try:
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"释放控制异常（可忽略）: {err}")
        client.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
