"""probe_spawn.py —— 读回真实出生位姿（机场 场景标定专用）"""

import argparse
import asyncio
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

DEFAULT_CONFIG_DIR = project_path(PROFILE["simulator"]["scene"]).parent
DEFAULT_SCENE = project_path(PROFILE["simulator"]["scene"]).name


def get_ned(drone):
    pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    return float(pos["x"]), float(pos["y"]), float(pos["z"])


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
        print(f"场景已加载: {args.scene}")
        drone = Drone(client, world, "Drone1")
        print("Drone1 已初始化")

        n, e, d = get_ned(drone)
        print("\n=== 真实出生位姿 (NED, 米) ===")
        print(f"  north(x) = {n:+.3f}")
        print(f"  east (y) = {e:+.3f}")
        print(f"  down (z) = {d:+.3f}   → 高度(相对NED原点) = {-d:+.3f} m")
        print("对照 --scene 指定文件 的 origin.xyz，判断配置数字是否=真实米数。\n")

        if args.takeoff:
            require = drone.enable_api_control() and drone.arm()
            print(f"api_control+arm: {require}")
            api_on = True
            tk = await drone.takeoff_async()
            await tk
            n2, e2, d2 = get_ned(drone)
            print(f"起飞后 NED = ({n2:+.3f}, {e2:+.3f}, {d2:+.3f}), 高度 = {-d2:+.3f} m")

        if args.touch_ground:
            # 触地测高: 不爬升, 从出生点直接缓降, z 连续几次不再增大即触地, 读那个 z。
            # 不用 land_async(出生就在地面时它判定"已着陆"会一直挂住)。
            require = drone.enable_api_control() and drone.arm()
            print(f"api_control+arm: {require}")
            api_on = True
            _, _, d0 = get_ned(drone)
            print(f"从出生点高度 {-d0:+.3f}m 直接缓降探地 ...")
            prev, stable, dg, contacted = d0, 0, d0, False
            for i in range(200):
                await (await drone.move_by_velocity_async(
                    v_north=0.0, v_east=0.0, v_down=2.0,
                    duration=0.4, yaw=0.0, yaw_is_rate=True))
                _, _, dz = get_ned(drone)
                stable = stable + 1 if (dz - prev) < 0.05 else 0
                prev, dg = dz, dz
                if i % 20 == 0:
                    print(f"  ...下降中 z={dz:+.2f} (已降 {dz-d0:+.1f}m)")
                if stable >= 3:  # 连续几次高度不变 = 触到地面
                    contacted = True
                    break
            print("→ 判定: " + ("高度不再变化=触地 ✅" if contacted else "跑满上限仍在降 ❌"))
            ng, eg, _ = get_ned(drone)
            print("\n=== 触地位姿 (NED, 米) —— 地面基准 ===")
            print(f"  north(x) = {ng:+.3f}")
            print(f"  east (y) = {eg:+.3f}")
            print(f"  down (z) = {dg:+.3f}   ← 跑道面/碰撞体所在的 NED z（地面基准）")
            print(f"  → 想离地 H 米出生, 把 origin.xyz 的 z 设为 ({dg:.3f} - H)。例 H=50 → z = {dg - 50:.3f}\n")
    finally:
        if api_on and drone is not None:
            try:
                # 触地模式已在地面, land 会挂住 → 跳过; 其余情况 land 加超时兜底
                if not args.touch_ground:
                    await asyncio.wait_for(await drone.land_async(), timeout=15)
            except Exception:
                pass
            drone.disarm()
            drone.disable_api_control()
        client.disconnect()
        print("已断开连接")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    p.add_argument("--address", default=PROFILE["simulator"]["address"])
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--takeoff", action="store_true", help="额外读起飞后位姿")
    p.add_argument("--touch-ground", action="store_true",
                   help="爬升后缓降到触地, 读触地 NED z = 地面基准(标定用)")
    p.add_argument("--climb", type=float, default=40.0,
                   help="触地测高前先爬升的高度(米), 默认40")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
