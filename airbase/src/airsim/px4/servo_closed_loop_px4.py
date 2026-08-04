"""servo_closed_loop_px4.py —— 方案A 指纹视觉伺服闭环的 **PX4 版**。

与 servo_closed_loop.py 的唯一区别是「平台层」: 飞控从 AirSim simpleflight 换成 PX4。
  · 感知(DINO 指纹)完全复用 —— 直接 import 老脚本的 Fingerprint / save_run, 老脚本一字不动。
  · 控制律照搬; 输出为 **世界NED 速度 + 绝对航向**(set_velocity_ned)。
    不用 yawspeed: PX4 一收到角速率就把绝对 yaw 置 NaN 自积分, 实测闭环第一拍甩头 ~300°(见 EXPERIMENTS.md)。
  · 取图仍由 Project AirSim 提供(渲染+FrontCamera -50°); 起飞/位姿/下发全部走 **MAVSDK 直连 PX4**。

数据流:
  ProjectAirSim(渲染) --FrontCamera--> 本脚本 --DINO 指纹--> 速度设定值
                                              |
  PX4 SITL <--MAVSDK offboard(set_velocity_ned 直连 14540)--+
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
from mavsdk.offboard import VelocityBodyYawspeed, VelocityNedYaw, PositionNedYaw, OffboardError

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
# 原来写死 parents[2]，脚本一挪进子目录就指错，README 为此专门警告过「不要把脚本挪进子目录」。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import（如 loop/servo_closed_loop → wm/decode_check）：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
from decode_check import decode_image
from servo_closed_loop import Fingerprint, save_run, compute_body_cmd  # 感知/落盘/控制律单一实现

PROJECT_ROOT = _AIRBASE_ROOT
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




def show_view(rgb, sim, grid, cx, cy, mass, peak, alt, step, scale=2, heat=False, lag=0.0):
    """实时窗口: 默认显示**原始机载画面**, 只叠指纹重心(绿圈)、画面中心(白十字)和数值。
    heat=True(--view-heat) 才叠指纹响应热力图(红=响应强): 绝对刻度(<0.2 不显色, 0.6 全红),
    与 rviz 的 plan_viz_node 一致 —— 不用 min-max, 否则全图弱噪声会被拉成满屏红。"""
    img = rgb.astype(np.float32)
    h, w = img.shape[:2]
    if heat and sim.size == grid * grid:
        heat = np.clip((sim.reshape(grid, grid) - 0.2) / 0.4, 0.0, 1.0)
        heat = np.kron(heat, np.ones((h // grid, w // grid), np.float32))
        img[..., 0] = img[..., 0] * (1 - 0.6 * heat) + 255 * 0.6 * heat
        img[..., 1] *= (1 - 0.5 * heat)
        img[..., 2] *= (1 - 0.5 * heat)
    bgr = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    H, W = bgr.shape[:2]
    cv2.drawMarker(bgr, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    cv2.circle(bgr, (int((cx + 1) / 2 * W), int((cy + 1) / 2 * H)), 10, (0, 255, 0), 2)
    cv2.rectangle(bgr, (0, 0), (W, 48), (0, 0, 0), -1)          # 深色底条, 浅背景上也认得清
    cv2.putText(bgr, f"step {step}  alt {alt:.1f}m", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(bgr, f"mass {mass:.3f}  peak {peak:.3f}", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(bgr, f"lag {lag:+.2f}s", (W - 110, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255) if lag > 0.5 else (0, 255, 255), 1)   # >0.5s 变红: 画面已经不实时
    cv2.imshow("airbase servo (onboard view)", bgr)
    cv2.waitKey(1)


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
            # 判据不能只看"到位": 转向还带着角速度就切到速度设定值, PX4 会在目标航向两侧持续
            # 振荡甩头(实测第一拍 +103°/s, 8s 收不住, 目标直接飞出画面)。所以要求偏差 <3° 且
            # **连续保持 yaw_settle 秒**, 确认停稳了再进闭环。复现见 px4_yaw_probe.py 的 G 段。
            t_ok = None

            def yaw_settled():
                nonlocal t_ok
                if abs(((_yaw_deg - yaw_tgt + 180) % 360) - 180) >= 3.0:
                    t_ok = None
                    return False
                t_ok = t_ok or time.monotonic()
                return time.monotonic() - t_ok >= args.yaw_settle

            await _stream_setpoint_until(
                mav, PositionNedYaw(_pos.north_m, _pos.east_m, -alt, yaw_tgt),
                cond=yaw_settled, timeout=40.0, desc="对准目标")
            print(f"航向已停稳: {_yaw_deg:.1f}°")

        if not await _wait_frame():
            print("没收到相机帧"); return

        print("=== 进入视觉伺服闭环(MAVSDK offboard) ===")
        t_loop = time.monotonic()
        ts0, w0, lag = None, None, 0.0
        for step in range(args.max_steps):
            with _lock:
                msg = _latest
            # 画面滞后: 以第 0 帧为基准, 比较墙上时钟与帧时间戳各自走过的时间。
            # 差值持续增大 = 订阅队列消费不过来, 看到的是旧画面(1920x1080 未压缩 20FPS ≈ 124MB/s,
            # 而闭环只有 ~3Hz 在取帧)。差值稳定在 0 附近 = 画面是实时的。
            ts = msg.get("time_stamp")
            if ts is not None:
                if ts0 is None:
                    ts0, w0 = ts, time.monotonic()
                lag = (time.monotonic() - w0) - (ts - ts0) / 1e9
            rgb = decode_image(msg)[0]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            cx, cy, mass, peak, sim = fp.query(rgb)
            n, e, dz = _pos.north_m, _pos.east_m, _pos.down_m
            yaw = math.radians(_yaw_deg)

            forward, right, down, yaw_rate, slow, vf = compute_body_cmd(args, cx, cy, mass, yaw)
            # 机体系→世界NED: 既是下发量, 也是落盘量(与老脚本 signals 布局一致, 可直接对比)
            vn = forward * math.cos(yaw) - right * math.sin(yaw)
            ve = forward * math.sin(yaw) + right * math.cos(yaw)
            # 航向: 以**实测**航向为基准往前推一拍。不能只发 yawspeed —— PX4 收到角速率设定值
            # 会把绝对 yaw 置 NaN 自己积分, 实测过它会在闭环第一拍甩头 ~300°, 目标直接出画面。
            # 也不对指令自积分(真机跟不上时会跑飞), 每拍重新以 _yaw_deg 为基准即可。
            yaw_cmd = _yaw_deg + math.degrees(yaw_rate) * args.dt

            log["step"].append(step); log["t"].append(time.monotonic() - t_loop)
            log["pose"].append([n, e, dz, yaw]); log["sig"].append([cx, cy, mass, peak])
            log["act"].append([vn, ve, down, yaw_rate]); log["vf"].append(vf); log["slow"].append(slow)
            log["rgb"].append(rgb.copy()); log["sim"].append(sim)

            if args.view:
                show_view(rgb, sim, grid, cx, cy, mass, peak, -dz, step,
                          heat=args.view_heat, lag=lag)

            if args.stop_alt is not None and -dz <= args.stop_alt:
                print(f"[{step}] alt={-dz:.1f}≤{args.stop_alt} → 到达接触高度, 停"); reached = True; break
            if args.stop_alt is None and mass >= args.mass_stop:
                print(f"[{step}] mass={mass:.3f}≥{args.mass_stop} → 判定贴近目标, 停"); reached = True; break

            print(f"[{step}] alt={-dz:5.1f} center=({cx:+.2f},{cy:+.2f}) mass={mass:.3f} "
                  f"peak={peak:.3f} | fwd={forward:.2f} down={down:.2f} yaw_rate={yaw_rate:+.2f} "
                  f"| 画面滞后={lag:+.2f}s")
            await mav.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, down, yaw_cmd))
            await asyncio.sleep(args.dt)
        print("=== 闭环结束 ===", "已贴近 ✅" if reached else "跑满步数(未达停止阈值)")
    finally:
        if args.view:
            cv2.destroyAllWindows()
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
                   default=base / "outputs/recordings/approach/airbase_tgt1_50m.h5")
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
    p.add_argument("--view", action="store_true",
                   help="开一个实时窗口显示原始机载画面(WSL 需要 WSLg, DISPLAY 已就绪)")
    p.add_argument("--view-heat", action="store_true",
                   help="在 --view 窗口上叠指纹响应热力图(红=响应强), 默认不叠")
    p.add_argument("--yaw-settle", type=float, default=2.0,
                   help="对准后航向需连续稳定这么多秒才进闭环(防甩头, 见 EXPERIMENTS.md)")
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
