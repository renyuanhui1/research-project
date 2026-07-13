"""离线检验 MPPI 选出的动作方向是否合理（零仿真、无 vx-min 拐杖、自包含）。

held-out episode：起点 z_t，目标 z_{t+K}（hindsight 可达）。
内联 MPPI（纯终点代价）在世界模型上规划，比"规划平均动作 vs 真实平均动作"的方向（余弦 + 主分量符号）。
两种 metric 都用【纯终点代价】公平对比：patchmse=裸潜距；value=时间距离价值函数。

关键用法：改 --K 看近目标(8) vs 远目标(25)时 patchmse 与 value 谁的方向不崩。
  patchmse 远处崩、value 仍保方向 → 学习式代价成立(只需层级A：做扎实 value)；
  value 也崩 → 该 latent 撑不起远程判别(需层级B：重训 metric-consistent latent)。
  注：value_fn 有训练 KMAX(约25)，K 超过它会饱和，故先在 K≤25 内比。

本脚本只依赖服务器上有的纯模块：train_predictor / extract_dino_features（不碰仿真 SDK）。
  python src/airsim/tools/eval_mppi_dir.py --K 8
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


# ---- 内联 train_value 的两件套（避免依赖 train_value.py 是否在服务器）----
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PROJECT_ROOT), help="项目根目录；默认根据脚本位置自动推导")
    ap.add_argument("--src-path", default=None, help="含 train_predictor.py/extract_dino_features.py 的目录")
    ap.add_argument("--repo-dir", default=None, help="dinov2 仓库目录")
    ap.add_argument("--dino-weights", default=None, help="dinov2_vits14_pretrain.pth")
    ap.add_argument("--ckpt", default=None, help="predictor_h5.pt")
    ap.add_argument("--value-fn-path", default=None, help="value_fn.pt")
    ap.add_argument("--episodes-dir", default=None, help="含 episode_*.h5 的目录")
    ap.add_argument("--ep", default="episode_0052.h5", help="held-out episode 文件名")
    ap.add_argument("--K", type=int, default=8, help="目标在未来第几帧(=规划 horizon)")
    ap.add_argument("--metrics", nargs="+", default=["patchmse", "value"])
    ap.add_argument("--starts", nargs="+", type=int, default=[0, 20, 40, 60, 80, 100, 120, 140])
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--v-max", type=float, default=2.5)
    ap.add_argument("--vz-max", type=float, default=1.0)
    ap.add_argument("--yaw-max", type=float, default=1.2)
    ap.add_argument("--plan-repeats", type=int, default=2, help="每次规划内部连续 plan 几次收敛")
    ap.add_argument("--avg-runs", type=int, default=5, help="每个起点独立重跑几次取平均(压 MPPI 采样方差)")
    ap.add_argument("--seed", type=int, default=0, help="固定随机种子，结果可复现")
    cli = ap.parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    BASE = cli.base
    src_path = cli.src_path or f"{BASE}/src/airsim"
    repo_dir = cli.repo_dir or f"{BASE}/dinov2"
    dino_w = cli.dino_weights or f"{BASE}/weights/dinov2_vits14_pretrain.pth"
    ckpt_p = cli.ckpt or f"{BASE}/weights/predictor_h5.pt"
    value_p = cli.value_fn_path or f"{BASE}/weights/value_fn.pt"
    eps_dir = cli.episodes_dir or f"{BASE}/outputs/datasets/episodes_dataset"
    sys.path.insert(0, src_path)
    from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
    from train_predictor import LatentPredictor
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model = LatentPredictor(ck["dim"], ck["num_patches"]).to(DEV).eval()
    model.load_state_dict(ck["model"])
    zm = torch.tensor(np.asarray(ck["z_mean"]), device=DEV)
    zs = torch.tensor(np.asarray(ck["z_std"]), device=DEV)
    mean = torch.tensor(IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEV).view(1, 3, 1, 1)
    dino = load_model("dinov2_vits14", DEV, weights=dino_w, repo_dir=repo_dir)

    value_fn = value_grid = None
    if "value" in cli.metrics:
        vp = value_p
        if not os.path.exists(vp):
            print(f"[warn] 缺 {vp}，跳过 value"); cli.metrics = [m for m in cli.metrics if m != "value"]
        else:
            vck = torch.load(vp, map_location="cpu")
            value_fn = ValueFn(vck["d"]).to(DEV).eval(); value_fn.load_state_dict(vck["model"])
            value_grid = vck.get("pool_grid", 4)
            print(f"value_fn 载入: d={vck['d']} grid={value_grid} kmax={vck.get('kmax')}")

    a_lo = torch.tensor([-cli.v_max, -cli.v_max, -cli.vz_max, -cli.yaw_max], device=DEV)
    a_hi = torch.tensor([cli.v_max, cli.v_max, cli.vz_max, cli.yaw_max], device=DEV)

    @torch.no_grad()
    def encode(rgb):
        x = to_input_tensor(rgb[None], 224, mean, std, DEV)
        return (dino.forward_features(x)["x_norm_patchtokens"][0].float() - zm) / zs

    def goal_cost(zb, zg, metric):
        if metric == "value":
            zbp = pool_spatial(zb.float(), value_grid)
            zgp = pool_spatial(zg.float(), value_grid)
            return value_fn(zbp, zgp.expand_as(zbp))
        return ((zb - zg) ** 2).mean(dim=(-2, -1))          # patchmse

    @torch.no_grad()
    def plan(z0, zg, mu, metric):
        H, N = mu.shape[0], cli.samples
        sigma = torch.full((H, 4), cli.sigma, device=DEV)
        for _ in range(cli.iters):
            noise = torch.randn(N, H, 4, device=DEV)
            acts = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise).clamp(a_lo, a_hi)
            zb = z0.unsqueeze(0).expand(N, -1, -1).contiguous()
            for k in range(H):
                zb = model(zb, acts[:, k])
            cost = goal_cost(zb, zg, metric)                 # 纯终点代价
            w = torch.softmax(-cost / cli.temperature, dim=0)
            mu = (w.view(-1, 1, 1) * acts).sum(dim=0)
            var = (w.view(-1, 1, 1) * (acts - mu.unsqueeze(0)) ** 2).sum(dim=0)
            sigma = var.sqrt().clamp_min(cli.sigma_min)
        return mu

    ep_path = f"{eps_dir}/{cli.ep}"
    need = max(cli.starts) + cli.K + 2
    with h5py.File(ep_path, "r") as f:
        rgb = f["rgb"][:need]; act = f["action"][:need].astype(np.float32)
    zc = {}

    def zt(t):
        if t not in zc:
            zc[t] = encode(rgb[t])
        return zc[t]

    print(f"ep={cli.ep} K={cli.K} metrics={cli.metrics} dev={DEV}")
    for metric in cli.metrics:
        print(f"\n============ 目标函数 = {metric} (纯终点代价) ============")
        print(f"{'起点t':>6} {'真实平均动作[vx,vy,vz,yaw]':>32} {'规划平均动作':>30} {'余弦':>6} {'主符号':>6}")
        cos_all, sign_ok_n = [], 0
        for t in cli.starts:
            z0, zg = zt(t), zt(t + cli.K)
            pms = []
            for _ in range(cli.avg_runs):                       # 独立重跑取平均，压采样方差
                mu = torch.zeros(cli.K, 4, device=DEV)
                for _ in range(cli.plan_repeats):
                    mu = plan(z0, zg, mu, metric)
                pms.append(mu.mean(dim=0).cpu().numpy())
            pm = np.mean(pms, axis=0)
            rm = act[t:t + cli.K].mean(axis=0)
            cos = float(np.dot(pm, rm) / (np.linalg.norm(pm) * np.linalg.norm(rm) + 1e-9))
            cos_all.append(cos)
            dom = int(np.argmax(np.abs(rm)))
            ok = np.sign(pm[dom]) == np.sign(rm[dom]); sign_ok_n += int(ok)
            rs = "[" + ",".join(f"{x:+.2f}" for x in rm) + "]"
            ps = "[" + ",".join(f"{x:+.2f}" for x in pm) + "]"
            print(f"{t:>6} {rs:>32} {ps:>30} {cos:>6.2f} {'✓' if ok else '✗':>6}")
        print(f"  余弦均值={np.mean(cos_all):+.2f}  主分量符号对={sign_ok_n}/{len(cli.starts)}")


if __name__ == "__main__":
    main()
