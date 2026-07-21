"""脚本 4：batch_collect.py —— 批量采集多条 episode

目的：把脚本 2 放进循环，生成多条 episode（先 20~50 条跑通，再扩到几百条）。
做法：
  - 连接一次（ProjectAirSim 一次只允许一个客户端），循环内**重载场景**做干净重置，
    避免无人机跨 episode 漂移飞出地图。
  - 定义一组基础轨迹模板 + 随机化（轮询模板保证覆盖均衡；每条随机缩放速度幅值；
    seed 驱动动作噪声与起点随机化）。
  - 每条存独立 HDF5（编号命名），并写一个 manifest.json 记录元信息（模板/幅值/seed/统计）。

复用脚本 2 的：image_callback / run_episode / save_hdf5 / decode。
"""

import argparse
import asyncio
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import projectairsim
from projectairsim import Drone, World

import collect_episode as ce

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- 基础轨迹模板：(vx, vy, vz, yaw_rate, steps)；v_down 负=上升 ----
# build_nominal 会按 steps 循环/截断，故各模板分段总和不必精确等于 steps。
TEMPLATES = {
    "mixed": ce.TRAJECTORY,  # 均衡：前进/爬升/横移/偏航组合（脚本 2 的默认轨迹）
    "forward_climb": [
        (1.5, 0.0, -0.4, 0.0, 50), (1.5, 0.0, 0.3, 0.0, 50),
        (1.2, 0.0, -0.5, 0.0, 50), (1.2, 0.0, 0.2, 0.0, 50),
    ],
    "yaw_scan": [
        (1.0, 0.0, 0.0, 0.5, 40), (1.0, 0.0, 0.0, -0.5, 40),
        (1.2, 0.0, 0.0, 0.6, 40), (1.2, 0.0, 0.0, -0.6, 40),
        (1.0, 0.0, 0.0, 0.4, 40),
    ],
    "lateral": [
        (0.8, 1.2, 0.0, 0.0, 40), (0.8, -1.2, 0.0, 0.0, 40),
        (1.0, 1.0, -0.3, 0.2, 40), (1.0, -1.0, 0.3, -0.2, 40),
        (1.2, 0.0, 0.0, 0.0, 40),
    ],
}

AMP_RANGE = (0.8, 1.2)  # 每条 episode 的速度幅值随机缩放区间


def scale_traj(traj, amp):
    """按幅值因子缩放一条模板的速度/偏航率（步数不变）。"""
    return [(vx * amp, vy * amp, vz * amp, yr * amp, n) for vx, vy, vz, yr, n in traj]


def make_ep_args(args):
    """构造 run_episode / save_hdf5 需要的参数对象（复用脚本 2 字段）。"""
    return Namespace(
        scene=args.scene, camera=args.camera, dt=args.dt, steps=args.steps,
        altitude=args.altitude, randomize_start=True, seed=0,
    )


def parse_args():
    p = argparse.ArgumentParser(description="批量采集多条 episode → HDF5 + manifest")
    p.add_argument("--address", default=ce.DEFAULT_ADDRESS)
    p.add_argument("--scene", default=ce.DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=ce.DEFAULT_CONFIG_DIR)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/datasets/episodes_batch")
    p.add_argument("--camera", default=ce.DEFAULT_CAMERA)
    p.add_argument("--dt", type=float, default=ce.DEFAULT_DT)
    p.add_argument("--steps", type=int, default=ce.DEFAULT_STEPS)
    p.add_argument("--altitude", type=float, default=ce.DEFAULT_ALTITUDE)
    p.add_argument("--num", type=int, default=3, help="采集条数（先小后大）")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0, help="文件编号起始（便于续采）")
    p.add_argument("--no-warmup", action="store_true", help="跳过开跑前的相机热身")
    return p.parse_args()


async def warmup_camera(client, args, settle_ms=115.0, window=20, timeout=60.0):
    """开跑前热身：空拉相机帧（无人机不起飞），直到帧间隔稳定到 ~100ms 才返回。

    相机冷启动（GPU 着色器编译/资产流送）会让最初几十秒出帧偏慢，此处先热透，
    避免前几条 episode 的 dt 偏大、抖动大。热身是渲染级，后续重载场景仍保持。
    """
    sim_cfg = str(args.sim_config_dir.expanduser().resolve())
    world = World(client, args.scene, sim_config_path=sim_cfg, delay_after_load_sec=2)
    drone = Drone(client, world, "Drone1")
    topic = drone.sensors[args.camera]["scene_camera"]
    ce.reset_cache()
    client.subscribe(topic, ce.image_callback)
    try:
        await ce.wait_first_frame()
        intervals = []
        last = -1
        t0 = time.monotonic()
        med = float("inf")
        while time.monotonic() - t0 < timeout:
            msg, _ = await ce.wait_new_frame(last)
            ts = int(msg["time_stamp"])
            if last > 0:
                intervals.append((ts - last) / 1e6)
            last = ts
            if len(intervals) >= window:
                recent = sorted(intervals[-window:])
                med = recent[window // 2]
                if med < settle_ms:
                    print(f"相机已热身：最近 {window} 帧中位 {med:.0f}ms，用时 {time.monotonic()-t0:.1f}s")
                    return
        print(f"热身超时 {timeout:.0f}s（最近中位 {med:.0f}ms），继续采集")
    finally:
        client.unsubscribe(topic)


async def collect_one(client, args, ep_args, name, traj, seed, amp, out_path):
    """重载场景做干净重置，飞一条 episode 并存盘，返回统计。"""
    sim_cfg = str(args.sim_config_dir.expanduser().resolve())
    world = World(client, args.scene, sim_config_path=sim_cfg, delay_after_load_sec=2)
    drone = Drone(client, world, "Drone1")

    if args.camera not in drone.sensors or "scene_camera" not in drone.sensors[args.camera]:
        raise RuntimeError(f"相机 '{args.camera}' 未配置 scene_camera")
    topic = drone.sensors[args.camera]["scene_camera"]
    ce.reset_cache()
    client.subscribe(topic, ce.image_callback)

    api_on = False
    try:
        ce.require_success("enable_api_control", drone.enable_api_control())
        api_on = True
        ce.require_success("arm", drone.arm())
        takeoff_task = await drone.takeoff_async()
        await asyncio.wait_for(takeoff_task, timeout=30.0)  # 卡住门时起飞会完不成，超时即跳过该条

        rgb, act, pose, tstamp, stats = await ce.run_episode(
            drone, ep_args, trajectory=traj, seed=seed
        )
        ce.save_hdf5(out_path, ep_args, rgb, act, pose, tstamp,
                     {"seed": seed, "template": name, "amp": amp})
        return stats
    finally:
        if api_on:
            try:
                land_task = await drone.land_async()
                await asyncio.wait_for(land_task, timeout=30.0)  # 卡住门时降落会完不成，超时即放弃清理
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"降落/释放控制异常（可忽略）: {err}")
        client.unsubscribe(topic)


async def main(args):
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_args = make_ep_args(args)
    names = list(TEMPLATES)

    client = projectairsim.ProjectAirSimClient(address=args.address)
    manifest = []
    try:
        client.connect()
        print(f"已连接 ProjectAirSim，准备批量采集 {args.num} 条 → {out_dir}")
        if not args.no_warmup:
            await warmup_camera(client, args)
        for k in range(args.num):
            idx = args.start_index + k
            name = names[k % len(names)]          # 轮询模板，覆盖均衡
            seed = args.base_seed + idx
            amp = float(np.random.default_rng(seed).uniform(*AMP_RANGE))
            traj = scale_traj(TEMPLATES[name], amp)
            out_path = out_dir / f"episode_{idx:04d}.h5"
            print(f"\n=== [{k+1}/{args.num}] episode_{idx:04d} 模板={name} 幅值={amp:.2f} ===")

            # 单条容错：相机偶发卡顿等异常只跳过这一条、继续下一条（下一条重载场景常能恢复），
            # 不再让整批退出。collect_one 的 finally 已负责降落/退订清理。
            try:
                stats = await collect_one(client, args, ep_args, name, traj, seed, amp, out_path)
            except Exception as err:
                print(f"  episode_{idx:04d} 失败，跳过：{err}")
                continue
            row = {"index": idx, "file": out_path.name, "template": name,
                   "amp": round(amp, 3), "seed": seed,
                   "avg_hz": round(stats["avg_hz"], 2), "behind": stats["behind"]}
            manifest.append(row)
            print(f"  完成：{row}")
    finally:
        client.disconnect()
        print("已断开连接")

    manifest_path = out_dir / "manifest.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(existing + manifest, ensure_ascii=False, indent=2))
    print(f"\n批量完成，共 {len(manifest)} 条；manifest: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
