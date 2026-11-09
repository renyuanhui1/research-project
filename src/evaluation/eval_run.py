"""eval_run.py —— 闭环运行的**特权评价器**（L0 健全性 + L1 任务判据）。纯离线，不连仿真。"""

import argparse
import glob
import json
import math
from pathlib import Path

import h5py
import numpy as np

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# Allow both direct script execution and package imports.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()


def configure_camera(run=None):
    """Use the recorded camera geometry, falling back to the selected scene profile."""
    global CAM_PITCH_DEG, CAM_HFOV_DEG, CAM_XYZ, CAM_ASPECT
    from common.profiles import scene_bundle
    path=Path(run) if run is not None else None
    if path is not None and path.suffix=='.h5':path=path.parent
    recorded=path/'runtime-scene.json' if path is not None else None
    cfg=json.loads(recorded.read_text()) if recorded is not None and recorded.exists() else scene_bundle(PROFILE)
    camera=next(s for a in cfg['actors'] for s in a['robot-config']['sensors'] if s['id']==PROFILE['simulator']['camera'])
    capture=next(c for c in camera['capture-settings'] if c['image-type']==0)
    CAM_PITCH_DEG=float(camera['origin']['rpy-deg'].split()[1])
    CAM_XYZ=tuple(map(float,camera['origin']['xyz'].split()))
    CAM_HFOV_DEG=float(capture['fov-degrees'])
    CAM_ASPECT=capture['height']/capture['width']

configure_camera()


def attr_value(value):
    """HDF5 属性转成便于打印/JSON 处理的值；向量属性保持 ndarray。"""
    array = np.asarray(value)
    return array.item() if array.size == 1 else array


def Rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def Ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def quat_to_R(q):
    """(w,x,y,z) → 3x3，把**目标局部系**向量转到世界 NED。"""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def in_view(P, pos, yaw):
    """目标点在不在画面里。返回 (是否可见, u∈[-1,1], v∈[-1,1], 前向距离)。

    与 `probe_depth_seg_flight.project()` 同一套几何（07-30 已验证：机头锚点的红十字
    确实落在机头尖端上）。这里用**归一化坐标**而不是像素，从而绕开
    "224×224 是 16:9 各向异性压扁、宽高比变了但 vFOV 没变" 那个坑。
    """
    Rb = Rz(yaw) @ Ry(math.radians(CAM_PITCH_DEG))
    fwd, rgt, dwn = Rb[:, 0], Rb[:, 1], Rb[:, 2]
    cam = np.asarray(pos, float) + Rz(yaw) @ np.array(CAM_XYZ)
    v = np.asarray(P, float) - cam
    f = float(v @ fwd)
    if f <= 0:
        return False, None, None, f                     # 在相机后方
    th = math.tan(math.radians(CAM_HFOV_DEG) / 2)
    u = (v @ rgt) / f / th
    w_ = (v @ dwn) / f / (th * CAM_ASPECT)
    return abs(u) <= 1.0 and abs(w_) <= 1.0, float(u), float(w_), f


def box_clearance(p, center, size, R):
    """点到有向包围盒**表面**的距离。>0 在盒外，<0 表示已穿进盒内（取最小穿透深度的负值）。"""
    local = R.T @ (np.asarray(p, float) - center)       # 世界 → 目标局部系
    half = np.asarray(size, float) / 2.0
    d = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(d, 0.0))
    inside = min(float(np.max(d)), 0.0)                 # 全在盒内时为负
    return outside + inside


def part_anchors(center, size, R):
    """由 OOB 导出部位锚点（世界 NED）。最长轴 = 机身纵轴，其 + 向为机头。

    锚点是包围盒近似，不是模型表面点。机头轴向须与当前资产核对。
    """
    order = int(np.argmax(size))                        # 最长轴 = 纵轴
    ax = {}
    for i in range(3):
        for sgn in (+1, -1):
            v = np.zeros(3)
            v[i] = sgn * size[i] / 2.0
            ax[(i, sgn)] = center + R @ v
    return {
        "nose": ax[(order, +1)], "tail": ax[(order, -1)], "center": np.asarray(center, float),
        "axis0+": ax[(0, +1)], "axis0-": ax[(0, -1)],
        "axis1+": ax[(1, +1)], "axis1-": ax[(1, -1)],
        "top": ax[(2, -1)], "bottom": ax[(2, +1)],      # NED: z 向下，故 -z 是顶
    }, order


def load_run(path):
    path = Path(path)
    h5 = path if path.suffix == ".h5" else path / "run.h5"
    if not h5.exists():
        raise SystemExit(f"找不到 run.h5: {h5}")
    with h5py.File(h5, "r") as f:
        pose = f["pose" if "pose" in f else "pose_ned"][:].astype(float)
        t = f["t"][:].astype(float) if "t" in f else np.arange(len(pose), dtype=float)
        act = f["action"][:].astype(float) if "action" in f else (f["action_body"][:].astype(float) if "action_body" in f else None)
        attrs = {k: attr_value(v) for k, v in f.attrs.items()}
    if pose.shape[1] == 3:
        poses=json.loads((h5.parent/'poses.json').read_text())
        if len(poses)!=len(pose):raise ValueError('poses.json length disagrees with run.h5')
        quats=np.array([[p['orientation'][k] for k in ('w','x','y','z')] for p in poses])
        pose=np.column_stack((pose,quats))
    if pose.shape[1] == 4:                              # 闭环格式 n,e,d,yaw
        pos, yaw = pose[:, :3], pose[:, 3]
    elif pose.shape[1] == 7:                            # 采集格式 px,py,pz,qw,qx,qy,qz
        pos = pose[:, :3]
        yaw = np.array([math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                   1 - 2 * (q[2] ** 2 + q[3] ** 2)) for q in pose[:, 3:]])
    else:
        raise SystemExit(f"不认识的 pose 布局: {pose.shape}")
    return h5, pos, yaw, t, act, attrs


def load_target(args):
    """从HDF5目标几何属性取OOB；文件可以只含属性，不需要图像。"""
    f = args.target_h5
    if f is None:
        cands = sorted(glob.glob(str(Path(args.dataset_dir) / "*.h5")))
        if not cands:
            raise SystemExit(f"--dataset-dir 下没有 *.h5: {args.dataset_dir}\n"
                             f"用 --target-h5 显式指定一条带目标 OOB 的 h5")
        for c in cands:                                 # 找 target_name 匹配的那条
            with h5py.File(c, "r") as h:
                if args.target is None or str(h.attrs.get("target_name")) == args.target:
                    f = c
                    break
        else:
            raise SystemExit(f"{args.dataset_dir} 里没有 target_name={args.target} 的 episode")
    with h5py.File(f, "r") as h:
        if "target_center" not in h.attrs:
            raise SystemExit(f"{f} 的 attrs 里没有 target_center（不是带目标真值的采集数据？）")
        return (str(h.attrs["target_name"]), np.asarray(h.attrs["target_center"], float),
                np.asarray(h.attrs["target_size"], float),
                np.asarray(h.attrs["target_quat"], float), f)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default=None)
    p.add_argument("run", help="闭环运行目录（含 run.h5）或 run.h5 本身")
    p.add_argument("--target-h5", default=None, help="带目标OOB属性的HDF5文件；默认按 --target 自动找")
    p.add_argument("--target", default=PROFILE["target"]["actor"], help="目标 actor 名")
    p.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "data/metadata/targets"),
                   help="包含目标几何元数据的目录，读取*.h5属性")
    p.add_argument("--part", default="nose", help="部位锚点：nose/tail/center/top/bottom/axis0±/axis1±")
    p.add_argument("--d", default="1,2,3,5", help="成功阈值（米），逗号分隔；不写死单一值")
    p.add_argument("--json", default=None, help="把结果另存为 json")
    args = p.parse_args()

    configure_camera(args.run)
    h5, pos, yaw, t, act, attrs = load_run(args.run)
    name, c, s, q, src = load_target(args)
    R = quat_to_R(q)
    anchors, long_axis = part_anchors(c, s, R)
    if args.part not in anchors:
        raise SystemExit(f"--part 只能是 {list(anchors)}")
    A = anchors[args.part]

    print(f"运行: {h5}")
    print(f"目标: {name}  OOB 中心 {np.round(c, 2)}  尺寸 {np.round(s, 2)}  (真值来自 {Path(src).name})")
    print(f"部位: {args.part} → {np.round(A, 2)}   (最长轴=局部轴{long_axis}, 长 {s[long_axis]:.2f}m)")

    # ---------- L0：轨迹本身可不可信（不依赖目标） ----------
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    alt = -pos[:, 2]
    l0 = {
        "帧数": len(pos),
        "时长_s": round(float(t[-1] - t[0]), 2),
        "平均步长_s": round(float(np.median(dt)), 4) if len(dt) else None,
        "总路程_m": round(float(step.sum()), 2),
        "单步最大位移_m": round(float(step.max()), 2) if len(step) else 0.0,
        "高度范围_m": [round(float(alt.min()), 2), round(float(alt.max()), 2)],
        "位姿含NaN": bool(~np.isfinite(pos).all()),
        "穿地(alt<0)帧数": int((alt < 0).sum()),
    }
    print("\n【L0 健全性】")
    for k, v in l0.items():
        print(f"  {k:16s} {v}")
    warn = []
    if l0["位姿含NaN"]:
        warn.append("位姿有 NaN")
    if l0["总路程_m"] < 1.0:
        warn.append("几乎没动（总路程 <1m），闭环可能没起来")
    if len(step) and step.max() > 10 * max(np.median(step), 1e-6):
        warn.append(f"单步位移最大值是中位数的 {step.max()/max(np.median(step),1e-6):.0f} 倍（疑似跳变/丢帧）")
    if l0["穿地(alt<0)帧数"]:
        warn.append(f"{l0['穿地(alt<0)帧数']} 帧高度为负")
    print("  " + ("⚠️ " + "；".join(warn) if warn else "✅ 无异常"))

    # ---------- L1：任务判据（用目标真值） ----------
    dist = np.linalg.norm(pos - A, axis=1)
    clear = np.array([box_clearance(p_, c, s, R) for p_ in pos])
    vis = np.array([in_view(c, p_, y_)[0] for p_, y_ in zip(pos, yaw)])
    i_min = int(np.argmin(dist))
    # 接触速度用位姿差分（比记录的指令更贴近实际；见 08-01 实测 位移/指令≈0.77）
    if 0 < i_min < len(pos) - 1 and len(dt):
        vmin = float(np.linalg.norm(pos[i_min + 1] - pos[i_min - 1]) / (t[i_min + 1] - t[i_min - 1]))
    else:
        vmin = float("nan")

    print("\n【L1 任务判据】")
    print(f"  到「{args.part}」最小距离   {dist.min():6.2f} m   @ 第 {i_min} 帧 / t={t[i_min]:.1f}s")
    print(f"  终点距离                {dist[-1]:6.2f} m")
    print(f"  到 OOB 表面最小间隙      {clear.min():6.2f} m" +
          ("   ⚠️ 负值 = 已穿进包围盒" if clear.min() < 0 else ""))
    print(f"  最近点处速度            {vmin:6.2f} m/s")
    print(f"  目标在视野内的帧占比      {vis.mean():6.1%}   （末 10 帧 {vis[-10:].mean():.0%}）")

    ds = [float(x) for x in args.d.split(",")]
    print(f"\n  多阈值成功判定（min_dist ≤ d）:")
    succ = {}
    for d in ds:
        ok = bool(dist.min() <= d)
        succ[d] = ok
        print(f"    d = {d:4.1f} m   {'✅ 成功' if ok else '❌ 失败'}")

    sr = attrs.get("stop_reason")
    if sr:
        note = {"contact": "深度判据，最接近真接触", "altitude": "只是降到某高度，**不等于接触**",
                "mass": "目标充满视野，可能只是近而未触", "timeout": "跑满步数未触发任何判据"}
        print(f"\n  停止原因：{sr}（{note.get(sr, '?')}）")
    elif "reached" in attrs:
        print(f"\n  参考：run.h5 里的 reached={attrs['reached']} —— "
              f"**该字段混淆高度停止与 mass 停止，不作为成功判据**（08-01 起改记 stop_reason）")

    if args.json:
        out = {"run": str(h5), "target": name, "part": args.part,
               "anchor": A.tolist(), "l0": l0, "warn": warn,
               "min_dist": float(dist.min()), "min_dist_step": i_min,
               "min_dist_t": float(t[i_min]), "final_dist": float(dist[-1]),
               "min_clearance": float(clear.min()), "speed_at_min": vmin,
               "in_view_frac": float(vis.mean()), "success": succ}
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
