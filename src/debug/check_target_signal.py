"""check_target_signal.py —— 解耦的"生死判据"：指纹 cost 有没有信号。"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# Allow both direct script execution and package imports.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()

from perception.extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD



def _rank(a):
    """秩, **并列取平均秩**。argsort(argsort(a)) 对并列值强行分配不同秩, 会高估相关性
    (本仓库因此虚报过 ρ, 见 EXPERIMENTS.md §2026-07-29)。"""
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    """Spearman 秩相关(无 scipy 依赖): 对(并列平均)秩做 Pearson。"""
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


@torch.no_grad()
def encode_all(model, rgb, image_size, mean, std, device, bs=32):
    """rgb:(N,H,W,3) uint8 → 原始(未标准化) patch 特征 (N,P,D) float32(cpu)。"""
    feats = []
    for i in range(0, len(rgb), bs):
        x = to_input_tensor(rgb[i:i + bs], image_size, mean, std, device)
        f = model.forward_features(x)["x_norm_patchtokens"].float().cpu()
        feats.append(f)
    return torch.cat(feats, 0)


def fingerprint_cost(zn_frames, proto, patch_xy, args):
    """zn_frames:(N,P,D) 已L2归一; proto:(D,); 返回逐帧分量 dict(numpy)。"""
    sim = torch.matmul(zn_frames, proto)                      # (N,P)
    k = max(1, int(sim.shape[-1] * args.target_topk_frac))
    topk_mean = sim.topk(k, dim=-1).values.mean(dim=-1)       # (N,)
    peak = sim.max(dim=-1).values
    w = torch.softmax(sim / args.target_softmax_temp, dim=-1)
    center = torch.matmul(w, patch_xy)                        # (N,2)
    center_penalty = (center ** 2).sum(dim=-1)
    mass = torch.sigmoid(
        (sim - args.target_mass_thresh) * args.target_mass_sharpness).mean(dim=-1)
    cost = (args.target_conf_weight * (1.0 - topk_mean)
            + args.target_center_weight * center_penalty
            - args.target_size_weight * mass)
    return {k2: v.numpy() for k2, v in dict(
        cost=cost, peak=peak, topk_mean=topk_mean,
        center_penalty=center_penalty, mass=mass).items()}


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model = load_model("dinov2_vits14", device,
                       weights=str(args.dino_weights), repo_dir=str(args.repo_dir))

    # --- episode 帧 + pose ---
    with h5py.File(args.episode, "r") as f:
        rgb = f["rgb"][:]
        pose = f["pose" if "pose" in f else "pose_ned"][:]
    N = len(rgb)
    print(f"episode: {N} 帧, 分辨率 {rgb.shape[1]}x{rgb.shape[2]}")

    # --- 编码帧 + 模板 ---
    zf = encode_all(model, rgb, args.image_size, mean, std, device)   # (N,P,D)
    tbgr = cv2.imread(str(args.template))
    if tbgr is None:
        raise SystemExit(f"读不到模板: {args.template}")
    trgb = np.ascontiguousarray(tbgr[:, :, ::-1])[None]               # (1,h,w,3)
    zt = encode_all(model, trgb, args.image_size, mean, std, device)[0]  # (P,D)

    # --- 用本 episode 帧统计做标准化(全帧全patch) ---
    zm = zf.reshape(-1, zf.shape[-1]).mean(0)
    zs = zf.reshape(-1, zf.shape[-1]).std(0).clamp_min(1e-6)
    zf = (zf - zm) / zs
    zt = (zt - zm) / zs

    # --- 目标 proto: 模板 patch L2归一→均值→L2归一 ---
    proto = F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0)       # (D,)
    znf = F.normalize(zf, dim=-1)                                     # (N,P,D)

    P = zf.shape[1]
    grid = int(round(P ** 0.5))
    xs = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (P,2)

    comp = fingerprint_cost(znf, proto, patch_xy, args)

    # --- 真实 3D 距离到目标 ---
    tgt = np.array([args.target_ned[0], args.target_ned[1], -abs(args.target_alt)])
    dist = np.linalg.norm(pose[:, :3] - tgt, axis=1)
    alt = -pose[:, 2]

    # --- 单调性: cost 应随距离增大而增大 → Spearman(cost,dist) 越正越好 ---
    rho_cost = spearman(comp["cost"], dist)
    rho_peak = spearman(comp["peak"], dist)     # peak 越近越大 → 应为负
    rho_mass = spearman(comp["mass"], dist)     # mass 越近越大 → 应为负

    print("\n=== 单调性(Spearman 对真实距离) ===")
    print(f"  cost vs 距离 = {rho_cost:+.3f}   (+1=完美单调, 越正越好)")
    print(f"  peak vs 距离 = {rho_peak:+.3f}   (应为负: 越近匹配越强)")
    print(f"  mass vs 距离 = {rho_mass:+.3f}   (应为负: 越近目标越大)")
    print(f"\n  最远帧 dist={dist.max():.1f}m cost={comp['cost'][dist.argmax()]:+.3f}"
          f"  peak={comp['peak'][dist.argmax()]:.3f}")
    print(f"  最近帧 dist={dist.min():.1f}m cost={comp['cost'][dist.argmin()]:+.3f}"
          f"  peak={comp['peak'][dist.argmin()]:.3f}")

    # --- csv ---
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="") as fp:
        wcsv = csv.writer(fp)
        wcsv.writerow(["frame", "alt_m", "dist_m", "cost", "peak", "topk_mean",
                       "center_penalty", "mass"])
        for i in range(N):
            wcsv.writerow([i, f"{alt[i]:.2f}", f"{dist[i]:.2f}",
                           f"{comp['cost'][i]:.4f}", f"{comp['peak'][i]:.4f}",
                           f"{comp['topk_mean'][i]:.4f}",
                           f"{comp['center_penalty'][i]:.4f}", f"{comp['mass'][i]:.4f}"])
    print(f"\ncsv -> {args.out_csv}")

    # --- png(可选) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(dist)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(dist[order], comp["cost"][order], ".-")
        ax[0].set_xlabel("true dist to target (m)"); ax[0].set_ylabel("fingerprint cost")
        ax[0].set_title(f"cost vs dist  (Spearman={rho_cost:+.2f})"); ax[0].invert_xaxis()
        ax[1].plot(dist[order], comp["peak"][order], ".-", label="peak")
        ax[1].plot(dist[order], comp["mass"][order], ".-", label="mass")
        ax[1].set_xlabel("true dist to target (m)"); ax[1].legend()
        ax[1].set_title("peak/mass vs dist"); ax[1].invert_xaxis()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.out_png, dpi=110)
        print(f"png -> {args.out_png}")
    except Exception as e:
        print(f"(跳过画图: {e})")


def parse_args():
    base = PROJECT_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--template", type=Path, default=project_path(PROFILE["assets"]["template"]))
    p.add_argument("--repo-dir", type=Path, default=project_path(PROFILE["assets"]["dino_repo"]))
    p.add_argument("--dino-weights", type=Path,
                   default=project_path(PROFILE["assets"]["dino_weights"]))
    p.add_argument("--target-ned", type=float, nargs=2, required=True,
                   metavar=("N", "E"))
    p.add_argument("--target-alt", type=float, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    p.add_argument("--out-csv", type=Path,
                   default=base / "debug/target_signal/signal.csv")
    p.add_argument("--out-png", type=Path,
                   default=base / "debug/target_signal/signal.png")
    # 指纹 cost 参数(默认与 plan_closed_loop 一致)
    p.add_argument("--target-conf-weight", type=float, default=1.0)
    p.add_argument("--target-center-weight", type=float, default=0.8)
    p.add_argument("--target-size-weight", type=float, default=0.4)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-topk-frac", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
