"""离线检查 target-conditioned 目标函数。

读取一条 episode 的 rgb 序列，用 plan_closed_loop 的 DINO 模板指纹目标函数逐帧打分。
用途：
  - 看 cost 是否随接近目标整体下降；
  - 看响应 peak / center / mass 是否合理；
  - 输出 csv，便于和旧 poolcos/patchmse 或人工观察对照。
"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from plan_closed_loop import Planner

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="检查 target 模板目标函数在 episode 上的逐帧响应")
    p.add_argument("--episode", type=Path,
                   default=Path(f"{base}/outputs/datasets/episodes_dataset/episode_0050.h5"))
    p.add_argument("--target-template", default=f"{base}/outputs/references/templates/tmpl_goal0050_green.png")
    p.add_argument("--repo-dir", default=f"{base}/dinov2")
    p.add_argument("--dino-weights", default=f"{base}/weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--out-csv", type=Path, default=Path(f"{base}/outputs/evaluations/target_cost/target_cost_check.csv"))
    p.add_argument("--device", default=None)
    p.add_argument("--stride", type=int, default=1, help="每隔多少帧评估一次")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--target-conf-weight", type=float, default=1.0)
    p.add_argument("--target-center-weight", type=float, default=0.8)
    p.add_argument("--target-size-weight", type=float, default=0.4)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-topk-frac", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p.parse_args()


def planner_args(args):
    return SimpleNamespace(
        repo_dir=args.repo_dir,
        dino_weights=args.dino_weights,
        ckpt=args.ckpt,
        no_fp16=args.no_fp16,
        cost_metric="target",
        target_template=args.target_template,
        v_max=2.0,
        vz_max=0.4,
        yaw_max=1.0,
        target_conf_weight=args.target_conf_weight,
        target_center_weight=args.target_center_weight,
        target_size_weight=args.target_size_weight,
        target_softmax_temp=args.target_softmax_temp,
        target_topk_frac=args.target_topk_frac,
        target_mass_thresh=args.target_mass_thresh,
        target_mass_sharpness=args.target_mass_sharpness,
    )


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    planner = Planner(planner_args(args), device)

    with h5py.File(args.episode, "r") as h:
        rgb = h["rgb"][:]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for i in range(0, len(rgb), args.stride):
            z = planner.encode(np.ascontiguousarray(rgb[i]))
            c = planner.target_components(z)
            center = c["center"]
            peak_idx = int(c["peak_idx"].item())
            rows.append({
                "frame": i,
                "cost": c["cost"].item(),
                "peak": c["peak"].item(),
                "topk_mean": c["topk_mean"].item(),
                "center_x": center[0].item(),
                "center_y": center[1].item(),
                "center_penalty": c["center_penalty"].item(),
                "mass": c["mass"].item(),
                "peak_patch_x": peak_idx % planner.grid,
                "peak_patch_y": peak_idx // planner.grid,
            })

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    costs = np.array([r["cost"] for r in rows], dtype=np.float32)
    masses = np.array([r["mass"] for r in rows], dtype=np.float32)
    peaks = np.array([r["peak"] for r in rows], dtype=np.float32)
    print(f"episode={args.episode}")
    print(f"frames={len(rgb)} evaluated={len(rows)} stride={args.stride}")
    print(f"cost first/median/last/min = {costs[0]:.3f}/{np.median(costs):.3f}/{costs[-1]:.3f}/{costs.min():.3f}")
    print(f"peak first/median/last/max = {peaks[0]:.3f}/{np.median(peaks):.3f}/{peaks[-1]:.3f}/{peaks.max():.3f}")
    print(f"mass first/median/last/max = {masses[0]:.3f}/{np.median(masses):.3f}/{masses[-1]:.3f}/{masses.max():.3f}")
    print(f"csv={args.out_csv}")
    print("first rows:")
    for r in rows[:5]:
        print(r)
    print("last rows:")
    for r in rows[-5:]:
        print(r)


if __name__ == "__main__":
    main()
