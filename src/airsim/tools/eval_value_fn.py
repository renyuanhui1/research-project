"""离线评估价值函数 V(z, z_goal)：单调性 + 梯度强度随距离的衰减曲线（定位有效半径拐点）。

两个问题（都不跑 MPPI、不连仿真，纯前向，几分钟）：
  A. 单调性：goal 固定为 episode 末帧，V(z_t, z_goal) 是否随真实剩余距离（pose 只当离线尺子）
     单调下降？超出 KMAX 的段应看到饱和平台——这本身就是"有效半径"的图。
  B. 梯度强度：对每个 K，ΔV(K) = mean_t [ V(z_t, z_{t+K}) - V(z_{t+1}, z_{t+K}) ]。
     理想=1（走一步少一步）；随 K 增大衰减，拐点=子目标间距该取的上限。
     （背景：eval_mppi_dir 实测 K=8 规划 vx≈0.89 健康、K=25 只剩 0.29——幅度信号先于方向死。）

建议用 held-out 直线接近条 episode_0050（23.7m→0）；train_value 留出的是末 5 条(0048~0052)。
产物：outputs/evaluations/value_fn/ 下 csv + png。
  python src/airsim/tools/eval_value_fn.py            # 本地默认路径
  服务器需显式传 --dino-weights/--repo-dir/--episodes-dir（同 eval_mppi_dir）。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---- 内联 train_value 的两件套（与 eval_mppi_dir 相同，避免服务器依赖） ----
def pool_spatial(z, grid_out=4):
    lead = z.shape[:-2]; P, D = z.shape[-2], z.shape[-1]
    g = int(round(P ** 0.5)); bs = g // grid_out
    z = z.reshape(*lead, g, g, D)[..., :grid_out * bs, :grid_out * bs, :]
    z = z.reshape(*lead, grid_out, bs, grid_out, bs, D).mean(dim=(-4, -2))
    return z.reshape(*lead, grid_out * grid_out * D)


class ValueFn(nn.Module):
    def __init__(self, d=384, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, z, g):
        return self.net(torch.cat([z, g, g - z], dim=-1)).squeeze(-1)


def spearman(a, b):
    """Spearman ρ（手写秩相关，免 scipy 依赖）。"""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PROJECT_ROOT))
    ap.add_argument("--src-path", default=None)
    ap.add_argument("--repo-dir", default=None)
    ap.add_argument("--dino-weights", default=None)
    ap.add_argument("--ckpt", default=None, help="predictor ckpt(取 z_mean/z_std)")
    ap.add_argument("--value-fn-path", default=None)
    ap.add_argument("--episodes-dir", default=None)
    ap.add_argument("--ep", default="episode_0050.h5", help="建议 held-out 直线接近条")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 8, 12, 16, 20, 25, 30, 40, 50])
    ap.add_argument("--out-dir", default=None, help="默认 outputs/evaluations/value_fn")
    cli = ap.parse_args()

    BASE = cli.base
    src_path = cli.src_path or f"{BASE}/src/airsim"
    repo_dir = cli.repo_dir or f"{BASE}/dinov2"
    dino_w = cli.dino_weights or f"{BASE}/weights/dinov2_vits14_pretrain.pth"
    ckpt_p = cli.ckpt or f"{BASE}/weights/predictor_h5.pt"
    value_p = cli.value_fn_path or f"{BASE}/weights/value_fn.pt"
    eps_dir = cli.episodes_dir or f"{BASE}/outputs/datasets/episodes_dataset"
    out_dir = cli.out_dir or f"{BASE}/outputs/evaluations/value_fn"
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, src_path)
    from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    zm = torch.tensor(np.asarray(ck["z_mean"]), device=DEV)
    zs = torch.tensor(np.asarray(ck["z_std"]), device=DEV)
    mean = torch.tensor(IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEV).view(1, 3, 1, 1)
    dino = load_model("dinov2_vits14", DEV, weights=dino_w, repo_dir=repo_dir)
    vck = torch.load(value_p, map_location="cpu")
    vfn = ValueFn(vck["d"]).to(DEV).eval()
    vfn.load_state_dict(vck["model"])
    grid = vck.get("pool_grid", 4)
    kmax = vck.get("kmax")
    print(f"value_fn={value_p} (d={vck['d']} grid={grid} kmax={kmax})  ep={cli.ep}")

    with h5py.File(f"{eps_dir}/{cli.ep}", "r") as f:
        rgb = f["rgb"][:]
        pose = f["pose"][:, :3]
    T = len(rgb)

    # 全帧编码 → 池化 (T, grid²·384)
    P = []
    with torch.no_grad():
        for t in range(T):
            x = to_input_tensor(rgb[t][None], 224, mean, std, DEV)
            z = (dino.forward_features(x)["x_norm_patchtokens"][0].float() - zm) / zs
            P.append(pool_spatial(z, grid))
    P = torch.stack(P)  # (T, D)
    print(f"编码完成 T={T}")

    tag = f"{cli.ep.replace('.h5','')}_{os.path.basename(value_p).replace('.pt','')}"

    # ---- A. 单调性：goal=末帧 ----
    with torch.no_grad():
        v = vfn(P, P[-1].expand_as(P)).cpu().numpy()
    dist = np.linalg.norm(pose - pose[-1], axis=1)
    rho = spearman(v[:-1], dist[:-1])
    # 饱和平台估计：V 首次降到 (kmax-2) 以下的位置之前视为平台
    sat_end = int(np.argmax(v < (kmax - 2))) if (v < (kmax - 2)).any() else T
    print(f"[A 单调性] Spearman ρ(V, 真实剩余距离) = {rho:+.3f}")
    print(f"           V 范围 {v.min():.1f}~{v.max():.1f}; 前 {sat_end} 帧疑似饱和平台(V>{kmax-2})")

    # ---- B. 梯度强度 ΔV(K) ----
    rows_b = []
    print(f"[B 梯度强度] ΔV = V(z_t,g)-V(z_t+1,g), g=z_t+K (理想=1)")
    for K in cli.ks:
        if K + 1 >= T:
            continue
        idx = np.arange(0, T - K - 1)
        with torch.no_grad():
            g = P[idx + K]
            dv = (vfn(P[idx], g) - vfn(P[idx + 1], g)).cpu().numpy()
        rows_b.append((K, dv.mean(), dv.std(), np.median(dv), (dv > 0).mean()))
        print(f"  K={K:3d}: ΔV mean={dv.mean():+.3f} median={np.median(dv):+.3f} "
              f"std={dv.std():.3f} 正确方向占比={(dv > 0).mean():.2f}")

    # ---- 落盘 csv ----
    csv_p = f"{out_dir}/value_{tag}.csv"
    with open(csv_p, "w") as f:
        f.write(f"# ep={cli.ep} value_fn={value_p} kmax={kmax} rho_monotonic={rho:.3f}\n")
        f.write("section,t_or_K,V_or_dVmean,dist_or_dVstd,dVmedian,frac_positive\n")
        for t in range(T):
            f.write(f"A,{t},{v[t]:.3f},{dist[t]:.3f},,\n")
        for K, m, s, md, fp in rows_b:
            f.write(f"B,{K},{m:.4f},{s:.4f},{md:.4f},{fp:.3f}\n")
    print(f"csv 已落盘: {csv_p}")

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(dist, v, ".", ms=3)
    ax1.axhline(kmax, ls="--", c="gray", lw=0.8)
    ax1.text(dist.max() * 0.6, kmax + 0.3, f"KMAX={kmax} (saturation)", fontsize=8, c="gray")
    ax1.set_xlabel("true remaining distance (m)")
    ax1.set_ylabel("V (predicted steps-to-goal)")
    ax1.set_title(f"A. monotonicity  Spearman rho={rho:+.3f}")
    ax1.invert_xaxis()
    ks_b = [r[0] for r in rows_b]
    ax2.errorbar(ks_b, [r[1] for r in rows_b], yerr=[r[2] for r in rows_b],
                 marker="o", capsize=3)
    ax2.axhline(1.0, ls="--", c="green", lw=0.8)
    ax2.axhline(0.0, ls="-", c="gray", lw=0.8)
    if kmax:
        ax2.axvline(kmax, ls="--", c="red", lw=0.8)
        ax2.text(kmax + 0.5, 0.8, f"KMAX={kmax}", fontsize=8, c="red")
    ax2.set_xlabel("K (goal distance in steps)")
    ax2.set_ylabel("per-step dV (ideal=1)")
    ax2.set_title("B. gradient strength vs distance (knee = subgoal spacing)")
    fig.tight_layout()
    png_p = f"{out_dir}/value_{tag}.png"
    fig.savefig(png_p, dpi=130)
    print(f"png 已落盘: {png_p}")


if __name__ == "__main__":
    main()
