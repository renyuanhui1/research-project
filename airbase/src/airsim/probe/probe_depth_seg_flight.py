"""probe_depth_seg_flight.py —— 高空拍地面目标, 验证深度相机 + 分割相机是否可用。

为什么要飞: 分割/深度在出生点(地面)看不到任何目标, 只有爬到高空、机头转向目标,
-50° 的 FrontCamera 才把地面飞机收进画面。

做法(不横向飞, 出生点原地爬升+转向, 最安全; 地形最高 37.6m, 80/140m 都有余量):
  1) 给 5 个目标 + 地面各指定**唯一**分割 ID
     —— 必须做: 默认 ID 是名字字符哈希, staticmeshactor_NN 字母全同 → 全是 83,
        飞机/地面/其他飞机同色, 掩膜直接用是废的;
  2) 起飞 → 爬到指定高度 → 转到指定航向 → 抓 RGB/深度/掩膜三帧存盘;
  3) 对每个目标, 用**已验证的投影模型**把它的包围盒中心投到像素上, 在该像素采样:
       掩膜颜色  → 回答"分割能不能区分不同飞机"
       深度原值 vs 真实几何距离 → 回答"深度是不是米制"(台账里那个 16UC1 非度量 bug)

用法(UE 前台 Play, 不需要 PX4):
  python airbase/src/airsim/probe_depth_seg_flight.py
  python airbase/src/airsim/probe_depth_seg_flight.py --views 80:-160 140:-153 80:21
"""

import argparse
import asyncio
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import projectairsim
from projectairsim import Drone, World
from projectairsim.types import BoxAlignment

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import（如 loop/servo_closed_loop → wm/decode_check）：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
PROJECT_ROOT = _AIRBASE_ROOT

# 目标名 → 分割 ID。名字唯一, 精确匹配只命中一个 actor(已实测 set 成功)。
TARGETS = {
    "StaticMeshActor_22": ("一号飞机", 200),
    "StaticMeshActor_18": ("二号飞机", 210),
    "StaticMeshActor_16": ("三号飞机", 220),
    "StaticMeshActor_20": ("四号飞机", 230),
    "StaticMeshActor_24": ("汽车",     240),
    "StaticMeshActor_14": ("地面/地形", 40),
}

_lock = threading.Lock()
_msgs = {}


def make_cb(tag):
    def cb(_topic, msg):
        with _lock:
            _msgs[tag] = msg
    return cb


def decode(msg):
    """→ (arr[h,w,c], 说明)。深度可能是 16UC1 或 32FC1, 都要认。"""
    h, w = msg["height"], msg["width"]
    enc = str(msg["encoding"]).upper()
    specs = {"BGR": (np.uint8, 3), "RGB": (np.uint8, 3), "BGRA": (np.uint8, 4),
             "RGBA": (np.uint8, 4), "8UC1": (np.uint8, 1), "16UC1": (np.uint16, 1),
             "32FC1": (np.float32, 1)}
    if enc not in specs:
        return None, f"不认识的 encoding={enc!r}"
    dt, ch = specs[enc]
    data = msg["data"]
    buf = (np.frombuffer(data, dtype=dt) if isinstance(data, (bytes, bytearray, memoryview))
           else np.asarray(data, dtype=dt))
    if buf.size != h * w * ch:
        return None, f"元素数 {buf.size} != 预期 {h*w*ch}"
    return buf.reshape(h, w, ch), f"{h}x{w}x{ch} {np.dtype(dt).name}"


def Rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def Ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def project(P, pos, yaw, W, H, pitch_deg=-50.0, hfov_deg=90.0, cam_xyz=(0.5, 0.0, 0.0)):
    """世界 NED 点 → 像素。返回 (u,v,前向距离) 或 None(在相机后方)。

    **注意**: 这里按"方形像素"推垂直视场 tv = th*(H/W), 只对**相机原生分辨率**成立
    (960x540 → vFOV 58.7°)。不要拿它算 HDF5 里存的 224x224 帧 —— 那是 16:9 被各向异性
    压成方形的, 宽高比变了但 vFOV 仍是 58.7°, 必须把 tv 写死成 0.5625*th。
    """
    Rb = Rz(yaw) @ Ry(math.radians(pitch_deg))
    fwd, rgt, dwn = Rb[:, 0], Rb[:, 1], Rb[:, 2]
    cam = pos + Rz(yaw) @ np.array(cam_xyz)
    v = np.asarray(P, float) - cam
    f = float(v @ fwd)
    if f <= 0:
        return None
    th = math.tan(math.radians(hfov_deg) / 2)
    tv = th * (H / W)                      # 同一焦距 → 垂直半视场按宽高比缩放
    u = 0.5 * (1 + (v @ rgt) / f / th) * W
    w_ = 0.5 * (1 + (v @ dwn) / f / tv) * H
    return int(round(u)), int(round(w_)), f


def get_pose(drone):
    k = drone.get_ground_truth_kinematics()["pose"]
    p = k["position"]
    q = k["orientation"]
    pos = np.array([float(p["x"]), float(p["y"]), float(p["z"])])
    w_, x_, y_, z_ = (float(q[c]) for c in "wxyz")
    yaw = math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    return pos, yaw


async def wait_fresh(tags, timeout=15.0):
    """丢弃旧帧, 等每一路都来新帧(转向/爬升后画面才是对的)。"""
    with _lock:
        base = {t: (_msgs[t]["time_stamp"] if t in _msgs else -1) for t in tags}
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        with _lock:
            if all(t in _msgs and _msgs[t]["time_stamp"] > base[t] for t in tags):
                return {t: _msgs[t] for t in tags}
        await asyncio.sleep(0.1)
    with _lock:
        return {t: _msgs[t] for t in tags if t in _msgs}


def save_and_report(got, pos, yaw, boxes, out_dir, tag_name):
    """存三张图 + 打印每个目标处的掩膜颜色/深度值 vs 真实距离。"""
    arrs = {}
    for t, msg in got.items():
        a, info = decode(msg)
        print(f"  {t:5s}: encoding={msg['encoding']!r} → {info}")
        if a is not None:
            arrs[t] = a

    if "rgb" in arrs:
        cv2.imwrite(str(out_dir / f"{tag_name}_rgb.png"), arrs["rgb"])
    if "seg" in arrs:
        s = arrs["seg"]
        cv2.imwrite(str(out_dir / f"{tag_name}_seg.png"), s if s.shape[2] != 1 else s[..., 0])
    if "depth" in arrs:
        d = arrs["depth"][..., 0]
        np.save(out_dir / f"{tag_name}_depth_raw.npy", d)       # 原值留着分析尺度
        finite = d[np.isfinite(d)]
        lo, hi = (np.percentile(finite, 1), np.percentile(finite, 99)) if finite.size else (0, 1)
        vis = np.clip((d.astype(np.float64) - lo) / max(hi - lo, 1e-9), 0, 1)
        cv2.imwrite(str(out_dir / f"{tag_name}_depth_vis.png"),
                    (vis * 255).astype(np.uint8))
        print(f"  depth 原值: min={d.min()} max={d.max()} 中位={np.median(d):.1f} "
              f"(1%~99% = {lo:.1f}~{hi:.1f})")

    if "seg" in arrs:
        s = arrs["seg"]
        flat = s.reshape(-1, s.shape[2])
        uniq, cnt = np.unique(flat, axis=0, return_counts=True)
        print(f"  掩膜里不同颜色数 = {len(uniq)}")
        for u, c in sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: -x[1])[:6]:
            print(f"      {u} : {100*c/len(flat):5.2f}%")

    H, W = (arrs.get("seg") if "seg" in arrs else arrs["rgb"]).shape[:2]
    print(f"  --- 各目标在画面里的位置 (画面 {W}x{H}) ---")
    for name, (label, seg_id, ctr) in boxes.items():
        r = project(ctr, pos, yaw, W, H)
        true_d = float(np.linalg.norm(ctr - pos))
        if r is None or not (0 <= r[0] < W and 0 <= r[1] < H):
            print(f"    {label:9s} 真实距离 {true_d:6.1f}m  → 不在画面内")
            continue
        u, v, _ = r
        msg_bits = [f"像素({u:4d},{v:4d})", f"真实距离 {true_d:6.1f}m"]
        if "seg" in arrs:
            msg_bits.append(f"掩膜颜色 {arrs['seg'][v, u].tolist()}")
        if "depth" in arrs:
            dv = float(arrs["depth"][v, u, 0])
            ratio = dv / true_d if true_d > 0 else float("nan")
            msg_bits.append(f"深度原值 {dv:.1f} (原值/真距 = {ratio:.3f})")
        print(f"    {label:9s} 指定ID={seg_id}  " + "  ".join(msg_bits))


async def main(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    client = projectairsim.ProjectAirSimClient(address=args.address)
    subs, api_on, drone = [], False, None
    try:
        client.connect()
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        print(f"场景已加载: {args.scene}\n")

        # ---- 1) 指定唯一分割 ID + 读回包围盒中心 ----
        print("=== 指定分割 ID / 读包围盒 ===")
        boxes = {}
        for name, (label, seg_id) in TARGETS.items():
            ok = world.set_segmentation_id_by_name(name, seg_id, False, True)
            back = world.get_segmentation_id_by_name(name, True)
            bb = world.get_3d_bounding_box(name, BoxAlignment.OBJECT_ORIENTED)
            ctr = (np.array([bb["center"]["x"], bb["center"]["y"], bb["center"]["z"]], float)
                   if bb and "center" in bb else None)
            print(f"  {label:9s} {name:20s} set={ok} 回读={back}"
                  + (f" 中心=({ctr[0]:+.1f},{ctr[1]:+.1f},{ctr[2]:+.1f})" if ctr is not None
                     else " 无包围盒"))
            if ctr is not None and back == seg_id and label != "地面/地形":
                boxes[name] = (label, seg_id, ctr)
        print()

        # ---- 2) 订阅三路 ----
        cam, tags = args.camera, []
        for key, tag in (("scene_camera", "rgb"), ("depth_camera", "depth"),
                         ("segmentation_camera", "seg")):
            if key in drone.sensors.get(cam, {}):
                t = drone.sensors[cam][key]
                client.subscribe(t, make_cb(tag))
                subs.append(t)
                tags.append(tag)
                print(f"已订阅 {tag:5s} {t}")
            else:
                print(f"!! {cam}/{key} 不存在 → 该路 capture-enabled 没开")
        if "seg" not in tags:
            print("\n分割那路没开, 先去 robot config 把 image-type 3 的 capture-enabled 改 true")
            return
        print()

        # ---- 3) 起飞 → 各视角抓帧 ----
        if not drone.enable_api_control():
            raise RuntimeError("enable_api_control 失败")
        api_on = True
        if not drone.arm():
            raise RuntimeError("arm 失败")
        await asyncio.wait_for(await drone.takeoff_async(), timeout=40)
        print("已起飞\n")

        for alt, yaw_deg in args.views:
            print(f"=== 视角: 高度 {alt:.0f}m, 航向 {yaw_deg:+.0f}° ===")
            await asyncio.wait_for(await drone.move_to_position_async(
                north=0.0, east=0.0, down=-abs(alt), velocity=args.climb_speed),
                timeout=180)
            await asyncio.wait_for(await drone.rotate_to_yaw_async(math.radians(yaw_deg)),
                                   timeout=60)
            await asyncio.sleep(args.settle)          # 停稳, 免得画面还在晃
            got = await wait_fresh(tags)
            if len(got) < len(tags):
                print(f"  !! 只拿到 {list(got)}, 缺 {set(tags)-set(got)}")
            pos, yaw = get_pose(drone)
            print(f"  实际位姿: NED=({pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f}) "
                  f"高度={-pos[2]:.1f}m yaw={math.degrees(yaw):+.1f}°")
            save_and_report(got, pos, yaw, boxes, out_dir,
                            f"h{int(alt)}_yaw{int(yaw_deg)}")
            print()

        print(f"图已存到 {out_dir}")
    finally:
        if api_on and drone is not None:
            try:
                await asyncio.wait_for(await drone.land_async(), timeout=120)
            except Exception as e:
                print(f"降落异常(可忽略): {e}")
            drone.disarm()
            drone.disable_api_control()
        for t in subs:
            try:
                client.unsubscribe(t)
            except Exception:
                pass
        client.disconnect()
        print("已断开")


def _view(s):
    a, y = s.split(":")
    return (float(a), float(y))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--sim-config-dir", type=Path, default=PROJECT_ROOT / "sim_config")
    p.add_argument("--scene", default="scene_airbase.jsonc")
    p.add_argument("--camera", default="FrontCamera")
    # 高度:航向。默认三个视角(出生点在 NED 0,0; 目标方位: 一号-165°, 二号-157°, 三号-142°, 汽车+21°)
    #   80:-160  → 一号(65m,近中心) + 二号(183m,上部)
    #   140:-153 → 一号 + 二号 + 三号(319m, 上边缘 372m 内)
    #   80:21    → 汽车(65m, 近中心)
    p.add_argument("--views", type=_view, nargs="+",
                   default=[(80.0, -160.0), (140.0, -153.0), (80.0, 21.0)],
                   metavar="ALT:YAW")
    p.add_argument("--climb-speed", type=float, default=8.0)
    p.add_argument("--settle", type=float, default=2.5, help="转向后停稳秒数")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "outputs/evaluations/depth_seg")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
