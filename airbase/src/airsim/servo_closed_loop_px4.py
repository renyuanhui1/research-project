"""servo_closed_loop_px4.py —— 方案A 指纹视觉伺服闭环的 **PX4 版**。

与 servo_closed_loop.py 的唯一区别是「平台层」: 飞控从 AirSim simpleflight 换成 PX4。
  · 感知(DINO 指纹)完全复用 —— 直接 import 老脚本的 Fingerprint / save_run, 老脚本一字不动。
  · 控制律照搬, 但输出改成 **机体系速度**(forward/right/down/yawspeed), 省掉 vf→vn/ve 的 yaw 旋转。
  · 取图仍由 Project AirSim 提供(渲染+FrontCamera -50°); 起飞/位姿/下发全部走 **MAVSDK 直连 PX4**。

数据流:
  ProjectAirSim(渲染) --FrontCamera--> 本脚本 --DINO 指纹--> 速度设定值
                                              |
  PX4 SITL <--MAVSDK offboard(set_velocity_body 直连 14540)--+
       └─(经 robot-config 的 px4-api 桥) 回灌姿态给渲染

前置:
  1) UE 前台加载 airbase 关卡; ProjectAirSim 载入 scene_airbase_px4.jsonc(robot-config 含 PX4 controller + FrontCamera -50°)。
  2) 另开终端启动 PX4 SITL 并连上 ProjectAirSim(与现有 px4 联调相同):
       export PX4_SIM_HOST_ADDR=172.21.192.1 && cd ~/PX4-Autopilot && make px4_sitl none_iris
     等 PX4 打印 "Simulator connected" 与 "home_set"。

用法(斜视主线, 与老脚本参数一致, 多一个 --mav-url):
  python airbase/src/airsim/servo_closed_loop_px4.py \
      --template airbase/pictures/尾翼.jpg \
      --stats-episode airbase/outputs/recordings/approach/airbase_tgt1_50m.h5 \
      --face-ned -64.2 -18.5 --start-altitude 40
"""

import argparse
import asyncio
import datetime
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import projectairsim
from projectairsim import Drone, World

from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, PositionNedYaw, OffboardError

from decode_check import decode_image
from servo_closed_loop import Fingerprint, save_run   # 复用感知与落盘, 不改老脚本

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"

# 相机帧(Project AirSim 推流) —— 与老脚本相同的最新帧缓存
_latest = None
_lock = threading.Lock()

# PX4 遥测(MAVSDK 后台任务持续刷新的最新位姿)
_pos = None          # PositionNed: north_m/east_m/down_m
_yaw_deg = None      # float, 机头朝向(度)


def cb(*args):
    global _latest
    with _lock:
        _latest = args[-1]


async def _telemetry_pos(mav):
    global _pos
    async for pv in mav.telemetry.position_velocity_ned():
        _pos = pv.position


async def _telemetry_att(mav):
    global _yaw_deg
    async for a in mav.telemetry.attitude_euler():
        _yaw_deg = a.yaw_deg


def compute_body_cmd(args, cx, cy, mass, yaw):
    """指纹信号 → 机体系速度设定值。控制律与 servo_closed_loop.py 等价, 但直接出机体系。
    返回 (forward, right, down, yaw_rate[rad/s], slow, vf)。"""
    slow = max(0.0, 1.0 - mass / args.mass_stop)      # 越近越慢
    if args.nadir:
        # 俯视: cx/cy 双轴闭环压目标到画面中心; 越居中降得越快。机体系直接给 forward/right, 无需旋转。
        forward = -args.k_xy * cy
        right = args.k_xy * cx
        sp = math.hypot(forward, right)
        if sp > args.v_forward:
            forward *= args.v_forward / sp
            right *= args.v_forward / sp
        err = math.hypot(cx, cy)
        down = args.v_down * max(0.0, 1.0 - err / args.center_gate)
        yaw_rate = 0.0
        vf = math.hypot(forward, right)
    else:
        # 斜视(-50°): 前进 + cy 垂直闭环(视线角控制, 让目标保持画面中心, 沿视线接触) + cx 转向。
        vf = args.v_forward * slow
        forward = vf
        right = 0.0
        down = max(0.0, args.v_down * slow + args.k_vert * cy)
        yaw_rate = float(np.clip(args.k_yaw * cx, -args.yaw_max, args.yaw_max))
    return forward, right, down, yaw_rate, slow, vf


async def _stream_setpoint_until(mav, sp, cond, timeout, desc, dt=0.2):
    """持续以 >2Hz 下发一个 offboard 位置设定值, 直到 cond() 成立或超时(offboard 需连续设定值流)。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        await mav.offboard.set_position_ned(sp)
        if cond():
            return True
        await asyncio.sleep(dt)
    print(f"[warn] 超时: {desc}")
    return False


async def _wait_frame():
    t0 = time.monotonic()
    while _latest is None and time.monotonic() - t0 < 10:
        await asyncio.sleep(0.1)
    return _latest is not None


async def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    log = {k: [] for k in ("step", "t", "pose", "sig", "act", "vf", "slow", "rgb", "sim")}
    reached = False
    grid = None

    # --- Project AirSim: 仅渲染 + 订阅 FrontCamera(不 enable_api_control, 控制权交给 MAVSDK) ---
    client = projectairsim.ProjectAirSimClient(address=args.address)
    mav = None
    tel_tasks = []
    offboard_on = False
    try:
        client.connect(); print("Project AirSim 已连接")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.sensors[args.camera]["scene_camera"], cb)
        fp = Fingerprint(args, device)
        grid = fp.grid

        # --- MAVSDK: 连接 PX4 ---
        mav = System()
        print(f"连接 PX4: {args.mav_url} ...")
        await mav.connect(system_address=args.mav_url)
        async for state in mav.core.connection_state():
            if state.is_connected:
                print("PX4 已连接"); break
        print("等待 PX4 位置就绪(global position / home)...")
        async for h in mav.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok:
                print("PX4 位置就绪"); break
        tel_tasks = [asyncio.ensure_future(_telemetry_pos(mav)),
                     asyncio.ensure_future(_telemetry_att(mav))]
        # 等首个位姿
        for _ in range(50):
            if _pos is not None and _yaw_deg is not None:
                break
            await asyncio.sleep(0.1)
        if _pos is None:
            print("没收到 PX4 位姿"); return

        print("arm ...")
        await mav.action.arm()
        # offboard 预热: 先给一个零设定值再 start
        await mav.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        try:
            await mav.offboard.start()
        except OffboardError as e:
            print(f"offboard.start 失败: {e._result.result}"); await mav.action.disarm(); return
        offboard_on = True

        # 爬升到目标高度(保持出生点 n/e 与当前朝向)
        n0, e0, yaw0 = _pos.north_m, _pos.east_m, _yaw_deg
        alt = abs(args.start_altitude)
        print(f"爬升到 {alt:.0f}m ...")
        await _stream_setpoint_until(
            mav, PositionNedYaw(n0, e0, -alt, yaw0),
            cond=lambda: -_pos.down_m >= alt * 0.95, timeout=40.0, desc="爬升")

        if args.goto_ned is not None:                 # 俯视: 开环飞到目标上空
            gn, ge = args.goto_ned
            print(f"开环飞往目标上空 ({gn:.1f},{ge:.1f}) ...")
            await _stream_setpoint_until(
                mav, PositionNedYaw(gn, ge, -alt, yaw0),
                cond=lambda: math.hypot(_pos.north_m - gn, _pos.east_m - ge) < 1.5,
                timeout=60.0, desc="飞往目标上空")
        if args.face_ned is not None:                 # 前视: 初始一次性朝向, 让目标进画面
            fn, fe = args.face_ned
            yaw_tgt = math.degrees(math.atan2(fe - _pos.east_m, fn - _pos.north_m))
            print(f"对准目标, yaw→{yaw_tgt:.1f}° ...")
            await _stream_setpoint_until(
                mav, PositionNedYaw(_pos.north_m, _pos.east_m, -alt, yaw_tgt),
                cond=lambda: abs((( _yaw_deg - yaw_tgt + 180) % 360) - 180) < 3.0,
                timeout=30.0, desc="对准目标")

        if not await _wait_frame():
            print("没收到相机帧"); return

        print("=== 进入视觉伺服闭环(MAVSDK offboard) ===")
        t_loop = time.monotonic()
        for step in range(args.max_steps):
            with _lock:
                msg = _latest
            rgb = decode_image(msg)[0]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            cx, cy, mass, peak, sim = fp.query(rgb)
            n, e, dz = _pos.north_m, _pos.east_m, _pos.down_m
            yaw = math.radians(_yaw_deg)

            forward, right, down, yaw_rate, slow, vf = compute_body_cmd(args, cx, cy, mass, yaw)
            # 落盘沿用世界系布局(与老脚本 signals 可对比): 机体系→NED 仅用于记录
            vn = forward * math.cos(yaw) - right * math.sin(yaw)
            ve = forward * math.sin(yaw) + right * math.cos(yaw)

            log["step"].append(step); log["t"].append(time.monotonic() - t_loop)
            log["pose"].append([n, e, dz, yaw]); log["sig"].append([cx, cy, mass, peak])
            log["act"].append([vn, ve, down, yaw_rate]); log["vf"].append(vf); log["slow"].append(slow)
            log["rgb"].append(rgb.copy()); log["sim"].append(sim)

            if args.stop_alt is not None and -dz <= args.stop_alt:
                print(f"[{step}] alt={-dz:.1f}≤{args.stop_alt} → 到达接触高度, 停"); reached = True; break
            if args.stop_alt is None and mass >= args.mass_stop:
                print(f"[{step}] mass={mass:.3f}≥{args.mass_stop} → 判定贴近目标, 停"); reached = True; break

            print(f"[{step}] alt={-dz:5.1f} center=({cx:+.2f},{cy:+.2f}) mass={mass:.3f} "
                  f"peak={peak:.3f} | fwd={forward:.2f} down={down:.2f} yaw_rate={yaw_rate:+.2f}")
            await mav.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, down, math.degrees(yaw_rate)))
            await asyncio.sleep(args.dt)
        print("=== 闭环结束 ===", "已贴近 ✅" if reached else "跑满步数(未达停止阈值)")
    finally:
        for t in tel_tasks:
            t.cancel()
        if mav is not None:
            try:
                if offboard_on:
                    await mav.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await mav.offboard.stop()
                await mav.action.hold()
            except Exception as e:
                print(f"释放 PX4 控制异常(可忽略): {e}")
        client.disconnect(); print("已断开 Project AirSim")
        if args.out_dir:
            tmpl = Path(args.template).stem
            stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
            save_run(Path(args.out_dir) / f"px4_{tmpl}_{stamp}", log, args, reached, grid)


def parse_args():
    base = PROJECT_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1", help="Project AirSim(渲染+相机)地址")
    p.add_argument("--mav-url", default="udpin://0.0.0.0:14540", help="MAVSDK 连 PX4 的地址")
    p.add_argument("--scene", default="scene_airbase_px4.jsonc")
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default="FrontCamera")
    p.add_argument("--template", type=Path, default=base / "pictures/尾翼.jpg")
    p.add_argument("--stats-episode", type=Path,
                   default=base / "outputs/recordings/approach/airbase_tgt1_100m.h5")
    p.add_argument("--repo-dir", type=Path, default=base / "dinov2")
    p.add_argument("--dino-weights", type=Path, default=base / "weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--nadir", action="store_true", help="俯视伺服(相机须 -90°): cx/cy 水平双闭环+居中才降")
    p.add_argument("--goto-ned", type=float, nargs=2, default=None, metavar=("N", "E"),
                   help="俯视: 先开环飞到该 NED 上空再进闭环")
    p.add_argument("--face-ned", type=float, nargs=2, default=None, metavar=("N", "E"),
                   help="前视: 初始一次性朝向(让目标进画面); 之后纯视觉")
    p.add_argument("--start-altitude", type=float, default=50.0)
    p.add_argument("--out-dir", default=str(base / "outputs/runs/servo"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    # 控制增益(与老脚本一致)
    p.add_argument("--v-forward", type=float, default=3.0, help="水平速度上限")
    p.add_argument("--v-down", type=float, default=1.2)
    p.add_argument("--k-yaw", type=float, default=0.8)
    p.add_argument("--yaw-max", type=float, default=0.6)
    p.add_argument("--k-vert", type=float, default=1.5, help="斜视: cy→下降率增益(视线角控制)")
    p.add_argument("--k-xy", type=float, default=3.0, help="俯视: 画面偏差→水平速度增益")
    p.add_argument("--center-gate", type=float, default=0.5, help="俯视: 偏差超此值不下降")
    p.add_argument("--mass-stop", type=float, default=0.20, help="前视: mass 超过即判定贴近停止")
    p.add_argument("--stop-alt", type=float, default=None, help="高度降到此值即判接触停止")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.3)
    # 指纹参数(与离线判据一致)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
