"""一次性整理脚本：合并 train/test/new 三个采集目录 → 干净的最终数据集。

规则：
  - hz < 5（后台采的 3Hz）          → 废弃
  - 尾部连续冻结 >= TAIL_MIN 步      → 裁尾（保留到冻结开始后 KEEP_CONTACT 帧）
  - 其余                             → 直接用
连续重编号写入 OUT_DIR，保留属性（steps 改成新长度），生成 manifest.json。
本脚本只构建 OUT_DIR，不删除任何源文件（删除单独手工确认后再做）。
"""
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRS = [
    str(PROJECT_ROOT / "outputs/datasets/raw/episodes_train"),
    str(PROJECT_ROOT / "outputs/datasets/raw/episodes_test"),
    str(PROJECT_ROOT / "outputs/datasets/raw/episodes_new"),
]
OUT_DIR = str(PROJECT_ROOT / "outputs/datasets/episodes_dataset")
HZ_MIN = 5.0        # 低于此判为退化（3Hz）废弃
TAIL_MIN = 15       # 尾部冻结步数达到此值才裁
KEEP_CONTACT = 5    # 裁尾时在冻结开始处多留几帧接触瞬间
FROZEN_STEP = 0.02  # 单步位移(m)小于此视为冻结


def classify(path):
    with h5py.File(path) as h:
        t = h["time"][:].astype(np.int64)
        pos = h["pose"][:, :3].astype(np.float64)
    T = len(t)
    hz = 1000.0 / np.median(np.diff(t) / 1e6)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    tail = 0
    for s in step[::-1]:
        if s < FROZEN_STEP:
            tail += 1
        else:
            break
    if hz < HZ_MIN:
        return "discard", hz, T, T
    if tail >= TAIL_MIN:
        new_len = max(50, T - tail + KEEP_CONTACT)
        return "trim", hz, T, new_len
    return "keep", hz, T, T


def write_episode(src, dst, new_len):
    with h5py.File(src) as h, h5py.File(dst, "w") as o:
        for k in ("rgb", "action", "pose", "time"):
            o.create_dataset(k, data=h[k][:new_len])
        for ak, av in h.attrs.items():
            o.attrs[ak] = av
        o.attrs["steps"] = new_len  # 裁尾后实际长度


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = []
    for d in SRC_DIRS:
        files += sorted(glob.glob(os.path.join(d, "episode_*.h5")))

    manifest = []
    idx = 0
    n_keep = n_trim = n_discard = 0
    for src in files:
        kind, hz, T, new_len = classify(src)
        if kind == "discard":
            n_discard += 1
            continue
        dst = os.path.join(OUT_DIR, f"episode_{idx:04d}.h5")
        write_episode(src, dst, new_len)
        with h5py.File(dst) as h:
            tpl = h.attrs.get("template", "?")
            seed = int(h.attrs.get("seed", -1))
            amp = round(float(h.attrs.get("amp", 0)), 3)
        manifest.append({
            "index": idx, "file": os.path.basename(dst), "template": tpl,
            "amp": amp, "seed": seed, "hz": round(hz, 2),
            "frames": int(new_len), "orig_frames": int(T),
            "trimmed": kind == "trim", "source": src,
        })
        if kind == "trim":
            n_trim += 1
        else:
            n_keep += 1
        idx += 1

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"源文件 {len(files)} 条")
    print(f"  直接用 {n_keep} 条；裁尾 {n_trim} 条；废弃 {n_discard} 条")
    print(f"最终数据集 {idx} 条 → {OUT_DIR}")
    print(f"manifest: {os.path.join(OUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
