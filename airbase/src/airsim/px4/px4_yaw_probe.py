"""px4_yaw_probe.py —— 隔离测试: PX4 offboard 下"航向保不住"到底是谁的锅。

背景: servo_closed_loop_px4.py 一进闭环, 飞机就甩头 ~300° 并偏航到东向, 目标出画面。
已排除: 对准计算(step0 yaw 正确)、旋翼混控(与厂商参考配置逐字相同)、yaw 指令形式
(角速率 / 绝对航向两种都甩)。剩下两个嫌疑: (a) 速度设定值路径本身, (b) 下发频率只有 2.9Hz。

本脚本摘掉 DINO 和相机, 只留飞控这一层(Project AirSim 仍要连并载入场景 ——
PX4 的 4560 模拟器链路靠它建立, 不加载场景 MAVSDK 等不到位置就绪), 依次跑三段并打印实测航向:
  A 位置保持 5Hz   —— 对照组, 已知稳(闭环前的爬升/对准就是这条路径)
  B 速度设定值 5Hz —— 若这里就甩 → 是速度路径的锅, 与频率无关
  C 速度设定值 2.9Hz(=闭环实际频率) —— B 稳而 C 甩 → 是频率的锅, 加快下发即可

判据: 每段结束打印 |航向偏差| 的最大值。稳 = 全程 <10°。

用法(UE 里 End PIE 重 Play + 起全新 PX4 之后):
  python airbase/src/airsim/px4_yaw_probe.py
"""
import argparse
import asyncio
import math
import time
from pathlib import Path

import projectairsim
from projectairsim import Drone, World

from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, OffboardError

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import（如 loop/servo_closed_loop → wm/decode_check）：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
PROJECT_ROOT = _AIRBASE_ROOT
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"

_pos = None
_yaw_deg = None


async def _telemetry_pos(mav):
    global _pos
    async for pv in mav.telemetry.position_velocity_ned():
        _pos = pv.position


async def _telemetry_att(mav):
    global _yaw_deg
    async for a in mav.telemetry.attitude_euler():
        _yaw_deg = a.yaw_deg


def _err(yaw_tgt):
    """航向偏差(度), 已折到 [-180,180]。"""
    return ((_yaw_deg - yaw_tgt + 180) % 360) - 180


async def _phase(name, mav, yaw_tgt, dur, hz, make_sp, block=0.0):
    """按 hz 下发 make_sp(t) 持续 dur 秒, 全程记录航向偏差, 返回最大偏差。
    block>0 时每拍下发前同步阻塞这么久, 用来模拟 DINO 推理把 asyncio 事件循环卡住。"""
    print(f"\n--- {name} ({hz:.2f}Hz, {dur:.0f}s) ---")
    dt = 1.0 / hz
    t0 = time.monotonic()
    worst, last_print, prev_yaw = 0.0, -1.0, _yaw_deg
    while (t := time.monotonic() - t0) < dur:
        if block > 0:
            time.sleep(block)          # 故意用同步 sleep: 就是要卡住事件循环
        await make_sp(t)
        e = abs(_err(yaw_tgt))
        worst = max(worst, e)
        if t - last_print >= 0.5:
            rate = (((_yaw_deg - prev_yaw + 180) % 360) - 180) / max(t - last_print, 1e-3)
            last_print, prev_yaw = t, _yaw_deg
            print(f"  t={t:4.1f}  yaw={_yaw_deg:7.1f}°  偏差={_err(yaw_tgt):+6.1f}°  "
                  f"角速度={rate:+6.1f}°/s  "
                  f"n={_pos.north_m:6.2f} e={_pos.east_m:6.2f} alt={-_pos.down_m:5.1f}")
        await asyncio.sleep(dt)
    verdict = "稳 ✅" if worst < 10 else "甩头 ❌"
    print(f"  → 最大偏差 {worst:.1f}°  {verdict}")
    return worst


async def main(args):
    # Project AirSim: 必须先连上并载入场景, PX4 的 4560 模拟器链路才建立(否则 MAVSDK 等不到位置就绪)。
    # 这里只加载场景, 不订阅相机、不 enable_api_control —— 控制权全交给 MAVSDK。
    client = projectairsim.ProjectAirSimClient(address=args.address)
    client.connect(); print("Project AirSim 已连接")
    world = World(client, args.scene,
                  sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                  delay_after_load_sec=2)
    drone = Drone(client, world, "Drone1")
    print(f"场景已载入: {args.scene}")
    try:
        await _probe(args)
    finally:
        client.disconnect(); print("Project AirSim 已断开")


async def _probe(args):
    mav = System()
    print(f"连接 PX4: {args.mav_url} ...")
    await mav.connect(system_address=args.mav_url)
    async for s in mav.core.connection_state():
        if s.is_connected:
            print("PX4 已连接"); break
    print("等待位置就绪 ...")
    async for h in mav.telemetry.health():
        if h.is_global_position_ok and h.is_home_position_ok:
            print("位置就绪"); break
    asyncio.ensure_future(_telemetry_pos(mav))
    asyncio.ensure_future(_telemetry_att(mav))
    for _ in range(50):
        if _pos is not None and _yaw_deg is not None:
            break
        await asyncio.sleep(0.1)

    # arm 重试: 位置刚就绪时 PX4 的预检往往还没过(COMMAND_DENIED)。伺服脚本因为要先加载 DINO
    # 白等了几秒才 arm, 所以撞不上; 这里没那段等待, 必须自己重试。
    print("arm ...")
    for i in range(30):
        try:
            await mav.action.arm(); break
        except Exception as e:
            if i % 5 == 0:
                print(f"  arm 被拒({type(e).__name__}), 等 PX4 预检 ... {i}s")
            await asyncio.sleep(1.0)
    else:
        print("30s 内始终 arm 不上, 放弃"); return
    await mav.offboard.set_position_ned(PositionNedYaw(_pos.north_m, _pos.east_m, -1.0, _yaw_deg))
    try:
        await mav.offboard.start()
    except OffboardError as e:
        print(f"offboard.start 失败: {e._result.result}"); await mav.action.disarm(); return

    alt = args.altitude
    yaw_tgt = args.yaw

    # 爬升 + 转向(与伺服脚本同一条路径)
    print(f"\n爬升到 {alt:.0f}m 并转到 {yaw_tgt:.1f}° ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30:
        await mav.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -alt, yaw_tgt))
        if -_pos.down_m >= alt * 0.95 and abs(_err(yaw_tgt)) < 3.0:
            break
        await asyncio.sleep(0.2)
    print(f"就位: alt={-_pos.down_m:.1f} yaw={_yaw_deg:.1f}°")

    # 沿目标方向的水平速度(与伺服闭环同量级)
    v = args.speed
    vn = v * math.cos(math.radians(yaw_tgt))
    ve = v * math.sin(math.radians(yaw_tgt))

    n0, e0 = _pos.north_m, _pos.east_m
    a = await _phase("A 位置保持 5Hz(对照组)", mav, yaw_tgt, 4.0, 5.0,
                     lambda t: mav.offboard.set_position_ned(
                         PositionNedYaw(n0, e0, -alt, yaw_tgt)))

    b = await _phase("B 速度设定值 5Hz", mav, yaw_tgt, 8.0, 5.0,
                     lambda t: mav.offboard.set_velocity_ned(
                         VelocityNedYaw(vn, ve, 0.0, yaw_tgt)))

    async def reset_yaw():
        """每段之前把航向拉回目标, 保证各段起点一致。"""
        print("\n复位航向 ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15:
            await mav.offboard.set_position_ned(
                PositionNedYaw(_pos.north_m, _pos.east_m, -alt, yaw_tgt))
            if abs(_err(yaw_tgt)) < 3.0:
                break
            await asyncio.sleep(0.2)

    await reset_yaw()
    c = await _phase("C 速度设定值 2.9Hz(=闭环实际频率)", mav, yaw_tgt, 8.0, 2.9,
                     lambda t: mav.offboard.set_velocity_ned(
                         VelocityNedYaw(vn, ve, 0.0, yaw_tgt)))

    # 以下三段: 在 C 的基础上各加一个伺服闭环才有的因素, 一次只加一个。
    await reset_yaw()
    d = await _phase("D +航向用实测值(相对指令, 同伺服)", mav, yaw_tgt, 8.0, 2.9,
                     lambda t: mav.offboard.set_velocity_ned(
                         VelocityNedYaw(vn, ve, 0.0, _yaw_deg)))

    await reset_yaw()
    e = await _phase("E +边飞边降 down=1.1", mav, yaw_tgt, 8.0, 2.9,
                     lambda t: mav.offboard.set_velocity_ned(
                         VelocityNedYaw(vn, ve, 1.1, yaw_tgt)))

    await reset_yaw()
    f = await _phase("F +每拍阻塞 0.2s(模拟 DINO 卡事件循环)", mav, yaw_tgt, 8.0, 2.9,
                     lambda t: mav.offboard.set_velocity_ned(
                         VelocityNedYaw(vn, ve, 0.0, yaw_tgt)), block=0.2)

    # G: 复刻伺服的交接时刻 —— 大角度转向"刚跨过 3° 阈值"就切速度设定值, 此时飞机仍带角速度。
    # 前面 A~F 每段之前都 reset_yaw(), 航向早已稳住、角速度为零, 所以从没复现过这个条件。
    print("\n先转到反方向(制造大角度转向) ...")
    yaw_back = ((yaw_tgt + 180 + 180) % 360) - 180
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20:
        await mav.offboard.set_position_ned(
            PositionNedYaw(_pos.north_m, _pos.east_m, -alt, yaw_back))
        if abs(_err(yaw_back)) < 3.0:
            break
        await asyncio.sleep(0.2)
    await asyncio.sleep(2.0)                      # 等它彻底停稳, 保证起点干净

    print(f"转回 {yaw_tgt:.1f}°, 一跨进 3° 立刻切速度设定值(与伺服脚本完全一致) ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20:
        await mav.offboard.set_position_ned(
            PositionNedYaw(_pos.north_m, _pos.east_m, -alt, yaw_tgt))
        if abs(_err(yaw_tgt)) < 3.0:
            break
        await asyncio.sleep(0.2)
    g = await _phase("G 转向未停稳就切速度设定值(复刻伺服交接)", mav, yaw_tgt, 8.0, 2.9,
                     lambda t: mav.offboard.set_velocity_ned(
                         VelocityNedYaw(vn, ve, 0.0, yaw_tgt)))

    print(f"\n===== 结论 =====")
    for label, w in (("A 位置保持 5Hz        ", a), ("B 速度设定值 5Hz      ", b),
                     ("C 速度设定值 2.9Hz    ", c), ("D +实测航向(相对指令) ", d),
                     ("E +边飞边降           ", e), ("F +每拍阻塞 0.2s      ", f),
                     ("G 转向未停稳就交接    ", g)):
        print(f"{label} 最大偏差 {w:6.1f}°  {'甩头 ❌' if w >= 10 else '稳 ✅'}")
    if g >= 10:
        print("→ 元凶就是 G: 对准转向没停稳就交给闭环, 旋转滑过去了。")
        print("  修法: 对准判据加'角速度≈0', 或到位后多保持 1~2s 位置设定值再进闭环。")
    else:
        print("→ G 也稳: 七个因素都排除了, 得回去把伺服闭环逐行和探针对齐再二分。")

    print("\n停 offboard, 断开(不等降落, 按仓库惯例)")
    try:
        await mav.offboard.stop()
    except OffboardError:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1", help="Project AirSim(渲染)地址")
    p.add_argument("--scene", default="scene_airbase_px4.jsonc")
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--mav-url", default="udpin://0.0.0.0:14540")
    p.add_argument("--altitude", type=float, default=40.0)
    p.add_argument("--yaw", type=float, default=-163.9, help="对准 airbase 目标的航向")
    p.add_argument("--speed", type=float, default=2.9, help="水平速度, 与伺服闭环同量级")
    asyncio.run(main(p.parse_args()))
