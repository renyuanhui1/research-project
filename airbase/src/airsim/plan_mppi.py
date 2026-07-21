"""脚本 7（阶段 A）：plan_mppi.py —— 离线潜空间 MPPI 规划验证（不连仿真）

目的：在接仿真闭环之前，先单独验证"规划逻辑"对不对。
做法：拿一条留出 episode，起点 z0=z[0]、目标 z_goal=z[H]（H 步之后的真实潜状态），
  用训好的世界模型在潜空间 rollout 一批候选动作序列，MPPI 迭代挑出让"预测终点最接近
  z_goal"的动作序列。再和两个基线比：
    - do-nothing（动作全 0）
    - true-actions（采集时真实执行的动作 a[0:H]，是连接 z0→z_goal 的"标准答案"）
  判定：MPPI 终点距离应明显小于 do-nothing，并接近 true-actions（说明规划真能朝目标走）。

全部在"训练集统计标准化"的潜空间里算（z_mean/z_std 存在 ckpt 里）。
依赖：torch + h5py + numpy（与训练同环境）。
"""
import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import torch

from train_predictor import LatentPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="离线潜空间 MPPI 规划验证")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--latent-dir", default=f"{base}/latents")
    p.add_argument("--episode-dir", default=f"{base}/outputs/datasets/episodes_dataset")
    p.add_argument("--episode", type=int, default=50, help="用哪条 episode 测（默认首条验证集）")
    p.add_argument("--horizon", type=int, default=30, help="规划步数 H；目标=z[H]")
    p.add_argument("--samples", type=int, default=2000, help="每轮采样的动作序列数")
    p.add_argument("--iters", type=int, default=10, help="MPPI 迭代轮数")
    p.add_argument("--sigma", type=float, default=0.6, help="动作采样初始标准差")
    p.add_argument("--sigma-min", type=float, default=0.05, help="sigma 退火下限，防过早收敛")
    p.add_argument("--temperature", type=float, default=0.1, help="MPPI 权重温度 lambda（越小越尖锐）")
    p.add_argument("--v-max", type=float, default=2.0, help="速度动作截断范围 ±v_max (m/s)")
    p.add_argument("--yaw-max", type=float, default=1.0, help="yaw_rate 截断范围 ±yaw_max (rad/s)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rollout(model, z0, actions):
    """z0:(P,D) 单个起点；actions:(N,H,4) → 返回每条序列的终点 (N,P,D)。批量并行 rollout。"""
    N = actions.shape[0]
    z = z0.unsqueeze(0).expand(N, -1, -1).contiguous()  # (N,P,D)
    for k in range(actions.shape[1]):
        z = model(z, actions[:, k])
    return z


def goal_dist(z_end, z_goal):
    """终点到目标的潜空间 MSE 距离；z_end:(N,P,D) 或 (P,D)。"""
    if z_end.dim() == 2:
        return ((z_end - z_goal) ** 2).mean()
    return ((z_end - z_goal.unsqueeze(0)) ** 2).mean(dim=(1, 2))  # (N,)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck.get("z_mean") is None:
        raise SystemExit("ckpt 没有 z_mean/z_std，无法在标准化潜空间规划")
    m = torch.tensor(np.asarray(ck["z_mean"]), device=device)
    s = torch.tensor(np.asarray(ck["z_std"]), device=device)
    model = LatentPredictor(ck["dim"], ck["num_patches"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # 读这条 episode 的 latent + 真实动作
    lat_files = sorted(glob.glob(str(Path(args.latent_dir) / "episode_*_dino.h5")))
    lat_path = next(f for f in lat_files if f"_{args.episode:04d}_" in Path(f).name)
    with h5py.File(lat_path, "r") as h:
        z_raw = torch.tensor(h["z"][:].astype(np.float32), device=device)
        src = str(h.attrs["source"])
    with h5py.File(Path(args.episode_dir) / src, "r") as h:
        a_true = torch.tensor(h["action"][:].astype(np.float32), device=device)

    zn = (z_raw - m) / s                       # 标准化潜空间
    H = min(args.horizon, len(zn) - 1)
    z0, z_goal = zn[0], zn[H]
    a_true_H = a_true[:H]                       # (H,4) 连接 z0→z_goal 的真值动作

    a_lo = torch.tensor([-args.v_max, -args.v_max, -args.v_max, -args.yaw_max], device=device)
    a_hi = torch.tensor([args.v_max, args.v_max, args.v_max, args.yaw_max], device=device)

    # MPPI：均值 mu(H,4) 初始 0，迭代采样→rollout→按 -cost 加权更新
    mu = torch.zeros(H, 4, device=device)
    sigma = torch.full((H, 4), args.sigma, device=device)
    with torch.no_grad():
        for it in range(args.iters):
            noise = torch.randn(args.samples, H, 4, device=device)
            acts = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise).clamp(a_lo, a_hi)
            z_end = rollout(model, z0, acts)            # (N,P,D)
            cost = goal_dist(z_end, z_goal)             # (N,)
            w = torch.softmax(-cost / args.temperature, dim=0)  # (N,)
            mu = (w.view(-1, 1, 1) * acts).sum(dim=0)   # (H,4) 加权更新均值
            # CEM 式 sigma 退火：用加权标准差收紧采样范围，带下限防过早收敛
            var = (w.view(-1, 1, 1) * (acts - mu.unsqueeze(0)) ** 2).sum(dim=0)
            sigma = var.sqrt().clamp_min(args.sigma_min)
            best = cost.min().item()
            print(f"  iter {it+1:2d}/{args.iters}  best_cost={best:.4f}  "
                  f"mean_cost={cost.mean().item():.4f}  sigma={sigma.mean().item():.3f}")

    # 三方对比：起点、do-nothing、true-actions、MPPI
    with torch.no_grad():
        d_start = goal_dist(z0, z_goal).item()
        d_zero = goal_dist(rollout(model, z0, torch.zeros(1, H, 4, device=device))[0], z_goal).item()
        d_true = goal_dist(rollout(model, z0, a_true_H.unsqueeze(0))[0], z_goal).item()
        d_mppi = goal_dist(rollout(model, z0, mu.unsqueeze(0))[0], z_goal).item()

    print(f"\nepisode={args.episode}  H={H}")
    print(f"起点到目标距离        d_start = {d_start:.4f}")
    print(f"do-nothing(动作全0)   d_zero  = {d_zero:.4f}")
    print(f"true-actions(真值动作) d_true  = {d_true:.4f}")
    print(f"MPPI 规划              d_mppi  = {d_mppi:.4f}")
    improve = (1 - d_mppi / d_zero) * 100 if d_zero > 0 else 0
    print(f"\nMPPI 相对 do-nothing 缩短 {improve:.1f}%")
    if d_mppi < d_zero * 0.7:
        print("结论: 规划有效——MPPI 找到的动作能让预测潜状态明显朝目标靠近")
    else:
        print("结论: 规划效果弱——需调 horizon/sigma/iters 或换更稳的模型")


if __name__ == "__main__":
    main()
