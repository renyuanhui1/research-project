"""probe_target_object.py —— 只读探针: 目标飞机在场景里叫什么、在哪、朝哪、多大。

为什么要它: 现在闭环结束只有 `reached=True`, 里面混着"降到 stop-alt 就停"和"mass 到阈值就停",
两者都不等于"碰到了机头"。要做 privileged evaluator(真值部位误差 / 接触判定), 前提是能拿到
目标飞机的**位姿 + 尺寸 + 朝向**。本脚本先把这三件在仿真里问出来。

只做请求式查询: 连接 → 载入场景 → list_objects / get_object_poses / get_3d_bounding_box /
get_object_scale → 打印。**不初始化 drone, 不 arm, 不起飞, 不写任何控制量。**

已核对的服务端行为(ProjectAirSim 源码):
  - get_object_pose      → 全局 NED 米; 找不到物体返回 translation=(nan,nan,nan), 不抛异常
                           (WorldSimApi.cpp:709)
  - get_object_poses     → 服务端循环调上者, 一次往返查多个名字 (WorldSimApi.cpp:728)
  - get_3d_bounding_box  → {center, size, quaternion}, center/size 已由 UnrealToNedLinear
                           转成 NED 米, size 是**全长**(Extent*2)取绝对值; 查不到返回 {}
                           (UnrealScene.cpp:620-659)
  - box_alignment        → 0=WORLD_AXIS(轴对齐), 1=OBJECT_ORIENTED(随物体朝向)

用法(UE 里已 Play, 场景可被载入即可):
  python airbase/src/airsim/probe_target_object.py
  python airbase/src/airsim/probe_target_object.py --pick "Airplane|Jet|Aircraft"   # 已知名字片段
  python airbase/src/airsim/probe_target_object.py --name BP_Airplane_2            # 只查一个
"""

import argparse
import math
from pathlib import Path

import numpy as np
import projectairsim
from projectairsim import World
from projectairsim.types import BoxAlignment

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
# 原来写死 parents[2]，脚本一挪进子目录就指错，README 为此专门警告过「不要把脚本挪进子目录」。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import（如 loop/servo_closed_loop → wm/decode_check）：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
PROJECT_ROOT = _AIRBASE_ROOT
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_SCENE = "scene_airbase.jsonc"
DEFAULT_OUT = PROJECT_ROOT / "outputs/evaluations/target_object"


def quat_to_R(q):
    """{w,x,y,z} → 3x3 旋转矩阵(把物体局部系向量转到世界 NED)。"""
    w, x, y, z = float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_yaw_deg(q):
    w, x, y, z = float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def show_bbox(world, name, tgt):
    """查并打印一个物体的有向包围盒 + 由它导出的 6 个"部位锚点"(世界 NED)。"""
    try:
        oob = world.get_3d_bounding_box(name, BoxAlignment.OBJECT_ORIENTED)
        aab = world.get_3d_bounding_box(name, BoxAlignment.WORLD_AXIS)
    except Exception as e:
        print(f"  get_3d_bounding_box 失败: {e}")
        return None
    if not oob or "center" not in oob:
        print("  get_3d_bounding_box 返回空 → 该物体没有可用网格/找不到")
        return None

    c = np.array([oob["center"]["x"], oob["center"]["y"], oob["center"]["z"]], float)
    s = np.array([oob["size"]["x"], oob["size"]["y"], oob["size"]["z"]], float)
    q = oob["quaternion"]
    R = quat_to_R(q)

    print(f"  OOB center(NED) = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})   高度 = {-c[2]:+.2f} m")
    print(f"  OOB size(全长 m) = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f})   yaw = {quat_to_yaw_deg(q):+.1f}°")
    if aab and "size" in aab:
        a = aab["size"]
        print(f"  AAB size(全长 m) = ({a['x']:.2f}, {a['y']:.2f}, {a['z']:.2f})  (轴对齐, 供对照)")
    try:
        print(f"  get_object_scale = {world.get_object_scale(name)}")
    except Exception as e:
        print(f"  get_object_scale 失败: {e}")
    if tgt is not None:
        print(f"  → 到已标定目标点 {tuple(round(v, 1) for v in tgt)} 的距离 = "
              f"{np.linalg.norm(c - tgt):.2f} m")

    # 6 个面心 = 局部轴 ±半长, 转到世界系。哪个轴最长就是机身纵轴(机头/尾)。
    order = np.argsort(-s)                       # 长→短
    axname = {0: "局部X", 1: "局部Y", 2: "局部Z"}
    print(f"  轴长排序: {axname[order[0]]}({s[order[0]]:.1f}m 最长→机身纵轴) > "
          f"{axname[order[1]]}({s[order[1]]:.1f}m) > {axname[order[2]]}({s[order[2]]:.1f}m)")
    print("  面心锚点(世界 NED, 可作部位候选):")
    for i in range(3):
        for sgn in (+1, -1):
            v = np.zeros(3); v[i] = sgn * s[i] / 2.0
            p = c + R @ v
            print(f"    {axname[i]}{'+' if sgn > 0 else '-'}  "
                  f"N={p[0]:+8.2f}  E={p[1]:+8.2f}  D={p[2]:+7.2f}  (高度 {-p[2]:+6.2f}m)")
    return c


def main(args):
    tgt = None if args.target_ned is None else np.array(
        [args.target_ned[0], args.target_ned[1], -abs(args.target_alt)], float)

    client = projectairsim.ProjectAirSimClient(address=args.address)
    try:
        client.connect()
        print("已连接 ProjectAirSim")
        world = World(
            client, args.scene,
            sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
            delay_after_load_sec=2,
        )
        print(f"场景已加载: {args.scene}\n")

        # --- 单个物体模式: 直接查, 不列表 ---
        if args.name:
            print(f"=== {args.name} ===")
            try:
                p = world.get_object_pose(args.name)
                t = p.translation
                print(f"  pose(NED) = ({t.x:+.2f}, {t.y:+.2f}, {t.z:+.2f})"
                      f"   yaw = {quat_to_yaw_deg(p.rotation):+.1f}°")
                if math.isnan(float(t.x)):
                    print("  ← NaN: 场景里没有这个名字")
            except Exception as e:
                print(f"  get_object_pose 失败: {e}")
            show_bbox(world, args.name, tgt)
            return

        # --- 列出物体 ---
        names = world.list_objects(args.regex)
        print(f"list_objects({args.regex!r}) → {len(names)} 个物体")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        namefile = args.out_dir / "objects.txt"
        namefile.write_text("\n".join(names), encoding="utf-8")
        print(f"完整名单 -> {namefile}")

        if args.pick:
            import re
            pat = re.compile(args.pick, re.I)
            hits = [n for n in names if pat.search(n)]
            print(f"\n--pick {args.pick!r} 命中 {len(hits)} 个: {hits[:20]}")
            cands = hits[:args.top]
        elif tgt is not None:
            # 不知道名字 → 用"离已标定目标点最近"来认人。一次往返查全部位姿。
            poses = world.get_object_poses(names)
            rows = []
            for n, p in zip(names, poses):
                t = p.translation
                if any(math.isnan(float(v)) for v in (t.x, t.y, t.z)):
                    continue
                c = np.array([t.x, t.y, t.z], float)
                rows.append((float(np.linalg.norm(c - tgt)), n, c))
            rows.sort(key=lambda r: r[0])
            print(f"\n有效位姿 {len(rows)}/{len(names)} 个。离目标点 "
                  f"({tgt[0]:+.1f}, {tgt[1]:+.1f}, {tgt[2]:+.1f}) 最近的 {args.top} 个:")
            for d, n, c in rows[:args.top]:
                print(f"  {d:8.2f} m  {n:<40s} NED=({c[0]:+8.2f},{c[1]:+8.2f},{c[2]:+7.2f})")
            cands = [n for _, n, _ in rows[:args.top]]
        else:
            print("\n(未给 --pick 也未给 --target-ned, 只列名单, 不查包围盒)")
            cands = []

        for n in cands:
            print(f"\n=== {n} ===")
            show_bbox(world, n, tgt)
    finally:
        client.disconnect()
        print("\n已断开连接")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--regex", default=".*", help="list_objects 的名字正则")
    p.add_argument("--pick", default=None, help="从名单里筛候选的正则(知道名字片段时用)")
    p.add_argument("--name", default=None, help="只查这一个物体, 跳过列表")
    p.add_argument("--top", type=int, default=5, help="查包围盒的候选个数")
    p.add_argument("--target-ned", type=float, nargs=2, default=[-64.2, -18.5],
                   metavar=("N", "E"), help="已标定的目标点, 用来认人; 传 None 关闭")
    p.add_argument("--target-alt", type=float, default=1.4)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
