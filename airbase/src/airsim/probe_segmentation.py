"""probe_segmentation.py —— 分割相机 + 名字映射 能力探测(只读, 不起飞)

回答两个问题:
  1) 分割相机开一个布尔就能用吗? (订阅得到帧吗, 解得开吗, 里面有几个 ID)
  2) 能不能拿到"资产名字"? (get_segmentation_id_map / list_assets 返回什么)
顺带看 image message 的 annotations 字段里有没有现成的 2D/3D 框。
"""

import sys, threading, time
from pathlib import Path

import cv2
import numpy as np
import projectairsim
from projectairsim import Drone, World

ADDR = "172.21.192.1"
CFG = Path(__file__).resolve().parents[2] / "sim_config"
SCENE = "scene_airbase.jsonc"
OUT = Path(__file__).resolve().parents[2] / "outputs/evaluations/segmentation"

_lock = threading.Lock()
_msgs = {}


def make_cb(tag):
    def cb(topic, msg):
        with _lock:
            _msgs[tag] = msg
    return cb


def decode(msg):
    h, w = msg["height"], msg["width"]
    enc = str(msg["encoding"]).upper()
    data = msg["data"]

    specs = {
        "BGR": (np.uint8, 3),
        "RGB": (np.uint8, 3),
        "BGRA": (np.uint8, 4),
        "RGBA": (np.uint8, 4),
        "8UC1": (np.uint8, 1),
        "16UC1": (np.uint16, 1),
    }
    if enc not in specs:
        return None, f"不支持的 encoding={enc!r}"

    dtype, channels = specs[enc]
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(data, dtype=dtype)
    else:
        buf = np.asarray(data, dtype=dtype)

    expected = h * w * channels
    if buf.size != expected:
        return None, f"元素数={buf.size}，预期={expected}"

    arr = buf.reshape(h, w, channels)
    return arr, f"{h}x{w}x{channels} {np.dtype(dtype).name}"


def main():
    client = projectairsim.ProjectAirSimClient(address=ADDR)
    subs = []
    try:
        client.connect()
        print("已连接")
        world = World(client, SCENE, sim_config_path=str(CFG), delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        print("场景+Drone1 就绪\n")

        # ---- 1) 哪些相机 topic 真的存在 ----
        print("=== drone.sensors 里的相机 topic ===")
        for cam, d in drone.sensors.items():
            keys = [k for k in d if not k.endswith("_info")]
            if keys:
                print(f"  {cam}: {keys}")
        print()

        # ---- 2) 名字映射 ----
        try:
            m = world.get_segmentation_id_map()
            print(f"=== get_segmentation_id_map(): {len(m)} 条 (use-owner-name=true) ===")
            for k in sorted(m)[:40]:
                print(f"  {k:<40s} -> {m[k]}")
            if len(m) > 40:
                print(f"  ... 还有 {len(m)-40} 条")
        except Exception as e:
            print(f"get_segmentation_id_map 失败: {e}")
        print()
        try:
            a = world.list_assets(".*")
            print(f"=== list_assets('.*'): {len(a)} 个资产 ===")
            for n in sorted(a)[:60]:
                print(f"  {n}")
            if len(a) > 60:
                print(f"  ... 还有 {len(a)-60} 个")
        except Exception as e:
            print(f"list_assets 失败: {e}")
        print()
        for nm, own in (("staticmeshactor_22", True), ("StaticMeshActor_22", True)):
            try:
                print(f"get_segmentation_id_by_name({nm!r}, use_owner_name={own}) = "
                      f"{world.get_segmentation_id_by_name(nm, own)}")
            except Exception as e:
                print(f"  查询 {nm} 失败: {e}")
        print()

        # ---- 2.5) 默认 ID 会撞车(哈希只累加小写字母, staticmeshactor_NN 字母全同),
        #      所以试着给目标显式指定一个唯一 ID, 看能不能改动。 ----
        print("=== 显式指定目标分割 ID ===")
        for nm in ("StaticMeshActor_22", "staticmeshactor_22"):
            try:
                ok = world.set_segmentation_id_by_name(nm, 200, False, True)
                now = world.get_segmentation_id_by_name(nm, True)
                print(f"  set({nm!r}, 200, regex=False) -> {ok};  再查 = {now}")
                if now == 200:
                    break
            except Exception as e:
                print(f"  set({nm!r}) 异常: {e}")
        else:
            try:
                ok = world.set_segmentation_id_by_name("staticmeshactor_22", 200, True, True)
                print(f"  正则模式 set -> {ok};  再查 = "
                      f"{world.get_segmentation_id_by_name('staticmeshactor_22', True)}")
            except Exception as e:
                print(f"  正则模式异常: {e}")
        print()

        # ---- 3) 订阅分割流 (哪个相机有 segmentation_camera 就用它) ----
        cam = next((c for c, d in drone.sensors.items() if "segmentation_camera" in d),
                   "FrontCamera")
        print(f"用相机: {cam}")
        want = {"segmentation_camera": "seg", "scene_camera": "rgb", "depth_camera": "depth"}
        for key, tag in want.items():
            if key in drone.sensors.get(cam, {}):
                t = drone.sensors[cam][key]
                client.subscribe(t, make_cb(tag))
                subs.append(t)
                print(f"已订阅 {t}")
            else:
                print(f"!! {cam}/{key} 不存在 (capture-enabled 没生效?)")
        print("\n等帧 (最多 20s) ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:
            with _lock:
                if "seg" in _msgs:
                    break
            time.sleep(0.2)
        with _lock:
            got = dict(_msgs)
        if "seg" not in got:
            print("!! 20s 内没收到分割帧")
            return

        OUT.mkdir(parents=True, exist_ok=True)
        # seg 排最前: 它是本次探测的主角, 不能被别的流解码失败带崩(上次就是 depth 崩了)。
        for tag in sorted(got, key=lambda t: t != "seg"):
          msg = got[tag]
          try:
            arr, info = decode(msg)
            print(f"\n--- {tag}: encoding={msg['encoding']!r} step={msg.get('step')} → {info}")
            if arr is None:
                continue
            if tag == "depth":
                print(f"    深度数值: min={arr.min()} max={arr.max()} "
                      f"中位={int(np.median(arr))}  (若非米制, 这里会是很大的整数)")
            enc = str(msg["encoding"]).upper()
            if arr.shape[2] == 1:
                save_arr = arr[..., 0]
            elif enc == "RGB":
                save_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif enc == "RGBA":
                save_arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            else:
                save_arr = arr
            path = OUT / f"seg_probe_{tag}.png"
            if not cv2.imwrite(str(path), save_arr):
                print(f"!! 保存失败: {path}")
            if tag == "seg":
                flat = arr.reshape(-1, arr.shape[2])
                uniq, cnt = np.unique(flat, axis=0, return_counts=True)
                print(f"    不同颜色/ID 数 = {len(uniq)}  (画面里有几类物体)")
                for u, c in sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: -x[1])[:10]:
                    print(f"      {u} : {c} px ({100*c/flat.shape[0]:.1f}%)")
          except Exception as e:
            print(f"!! {tag} 处理失败(不影响其他流): {type(e).__name__}: {e}")

        ann = got["seg"].get("annotations")
        print(f"\n=== annotations 字段: type={type(ann).__name__} "
              f"len={len(ann) if hasattr(ann,'__len__') else 'n/a'}")
        if ann:
            print(f"  第一条: {ann[0]}")
        print(f"\n图已存到 {OUT}/seg_probe_*.png")
    finally:
        for t in subs:
            try:
                client.unsubscribe(t)
            except Exception:
                pass
        client.disconnect()
        print("已断开")


if __name__ == "__main__":
    main()
