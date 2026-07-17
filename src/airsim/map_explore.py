"""覆盖建图飞行：不带找目标那套，专门为"扫清周围环境"飞。

模式：在每个停点【原地慢转一整圈】让前视相机把四周刷进来(那些扇形就是 FOV 扫地面)，
再【往前挪一段】到下个停点，循环。全程开深度，每帧反投影出世界点云落 npz，
plan_viz_node 直接出稠密点云 / 高度方块图(像探索建图)。

前置：config 里 FrontCamera 深度(image-type 2)capture/streaming 已开(见 depth_map.py)。

跑(服务器连宿主机)：
  python src/airsim/map_explore.py --address 192.168.31.178 \
      --viz-dump outputs/runs/mppi/explore01 --stops 4 --step-dist 4
本地：plan_viz_node.py --dump-dir ~/mnt/server_runs/explore01 --map-voxel 0.2 --rate 20
"""
import argparse
import asyncio
from math import pi
from pathlib import Path

import cv2
import numpy as np
import projectairsim
from projectairsim import Drone, World

import collect_episode as ce
import depth_map as dm


def build_schedule(stops, step_dist, fwd, yaw_rate, dt):
    """每步 (v_north, v_east, v_down, yaw_rate)：停点转一圈 → 往北挪 step_dist → 下一停点。"""
    n_spin = max(int((2 * pi / max(yaw_rate, 1e-3)) / dt), 1)   # 转满一圈的步数
    n_move = max(int((step_dist / max(fwd, 1e-3)) / dt), 1)
    sched = []
    for s in range(stops):
        sched += [(0.0, 0.0, 0.0, yaw_rate)] * n_spin           # 原地扫一圈
        if s < stops - 1:
            sched += [(fwd, 0.0, 0.0, 0.0)] * n_move            # 往北挪到下一停点
    return sched


async def main(args):
    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None
    api_on = False
    dtopic = None
    try:
        client.connect()
        print("已连接仿真")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        ce.reset_cache()
        client.subscribe(drone.sensors[args.camera]["scene_camera"], ce.image_callback)
        dkeys = [k for k in drone.sensors[args.camera] if "depth" in k.lower()]
        if not dkeys:
            raise SystemExit("没有深度流：config 里 FrontCamera image-type 2 的 "
                             "capture-enabled/streaming-enabled 未开")
        dtopic = drone.sensors[args.camera][dkeys[0]]
        dm.reset_depth()
        client.subscribe(dtopic, dm.depth_callback)
        print(f"已订阅深度: {dtopic}")

        ce.require_success("enable_api_control", drone.enable_api_control()); api_on = True
        ce.require_success("arm", drone.arm())
        await asyncio.wait_for(await drone.takeoff_async(), timeout=30.0)
        n, e, d = ce.get_ned(drone)
        await asyncio.wait_for(await drone.move_to_position_async(
            north=n, east=e, down=d - args.altitude, velocity=2.0), timeout=30.0)
        await asyncio.sleep(1.0)
        await ce.wait_first_frame()

        sched = build_schedule(args.stops, args.step_dist, args.fwd, args.yaw_rate, args.dt)
        vd = Path(args.viz_dump); vd.mkdir(parents=True, exist_ok=True)
        print(f"建图飞行：{args.stops} 个停点、每点转一圈、间距 {args.step_dist}m，共 {len(sched)} 步")

        last_ts = -1
        for step, (vn, ve, vdn, yr) in enumerate(sched):
            msg, _ = await ce.wait_new_frame(last_ts)
            last_ts = int(msg["time_stamp"])
            rgb_full = ce.decode_image(msg)[0]
            rgb = cv2.resize(rgb_full, (ce.STORE_HW, ce.STORE_HW), interpolation=cv2.INTER_AREA)
            pose = ce.extract_pose(msg)

            mp = {}
            depth = dm.get_depth()
            if depth is not None:
                pts, cols = dm.frame_to_world(depth, rgb_full, pose,
                                              args.map_stride, args.map_max_range)
                mp = {"pts": pts.astype(np.float32), "cols": cols}
            np.savez_compressed(vd / f"step_{step:04d}.npz",
                                step=step, pose=pose, dt=args.dt, grid=16, rgb=rgb,
                                sim=np.zeros(1, np.float32), **mp)

            await drone.move_by_velocity_async(
                v_north=vn, v_east=ve, v_down=vdn,
                duration=args.dt * args.dur_factor, yaw=yr, yaw_is_rate=True)
            if step % 20 == 0:
                print(f"  step {step:4d}/{len(sched)}  pos=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f})"
                      f"  点={len(mp['pts']) if mp else 0}")
        print("建图飞行结束")

    finally:
        if api_on and drone is not None:
            try:
                await asyncio.wait_for(await drone.land_async(), timeout=30.0)
                drone.disarm(); drone.disable_api_control()
            except Exception as ex:
                print(f"清理忽略: {ex}")
        try:
            client.unsubscribe(drone.sensors[args.camera]["scene_camera"])
            if dtopic is not None:
                client.unsubscribe(dtopic)
        except Exception:
            pass
        client.disconnect()
        print("已断开仿真")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--address", default=ce.DEFAULT_ADDRESS)
    p.add_argument("--scene", default=ce.DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=ce.DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default=ce.DEFAULT_CAMERA)
    p.add_argument("--altitude", type=float, default=ce.DEFAULT_ALTITUDE)
    p.add_argument("--dt", type=float, default=ce.DEFAULT_DT)
    p.add_argument("--dur-factor", type=float, default=1.5)
    p.add_argument("--viz-dump", required=True, help="npz 落盘目录(给 plan_viz_node)")
    p.add_argument("--stops", type=int, default=4, help="停点数(每点原地转一圈)")
    p.add_argument("--step-dist", type=float, default=4.0, help="相邻停点间距(米, 往北)")
    p.add_argument("--fwd", type=float, default=1.0, help="停点间平移速度(m/s)")
    p.add_argument("--yaw-rate", type=float, default=0.3, help="扫描转速(rad/s, 越小重叠越好)")
    p.add_argument("--map-stride", type=int, default=8, help="深度像素下采样步长")
    p.add_argument("--map-max-range", type=float, default=30.0, help="丢弃超此距离的点(米)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse()))
