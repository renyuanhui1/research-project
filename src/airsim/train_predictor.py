"""脚本 6：train_predictor.py —— 训练 action-conditioned latent predictor（世界模型核心）

学习映射：(z_t = DINO(图像_t), a_t = 动作) → z_{t+1} = DINO(图像_{t+1})

按文档：
  1. Dataset：从 latent + action 构造 (z_t, a_t, z_{t+1})（支持多步 (z_t..z_{t+k})）。
  2. 模型：在 patch latent 上、以 (z_t, a_t) 预测 z_{t+1} 的 Transformer（DINO-WM 范式）。
     先求结构最简、能跑通、loss 能降。
  3. 损失：预测 latent 与真实 latent 的 L2(MSE)。
  4. 先小数据过拟合验证（--overfit），再上全量。
通过标准：训练 loss 稳定下降；能在留出序列上做出合理的多步 latent rollout。

数据来源（HDF5）：
  - latent：脚本 5 输出 outputs/features/latents/episode_XXXX_dino.h5  → dataset z (T, P, D)
  - action：脚本 2/4 输出 episode_XXXX.h5                     → dataset action (T, 4)
  时间对齐：a_t 驱动 z_t → z_{t+1}，故样本 = (z[t], action[t], z[t+1])。

环境：统一使用 conda env `airsim`（torch/h5py/numpy/projectairsim）。
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------- Dataset ----------------
class LatentActionDataset(Dataset):
    """把 latent + action 构造成 (z_t, actions[t:t+h], z[t+1:t+h+1]) 样本。"""

    def __init__(self, latent_dir, episode_dir, horizon=1, episodes=None):
        latent_dir = Path(latent_dir)
        episode_dir = Path(episode_dir)
        files = sorted(glob.glob(str(latent_dir / "episode_*_dino.h5")))
        if episodes is not None:
            files = [files[i] for i in episodes]
        if not files:
            raise SystemExit(f"latent 目录没有 episode_*_dino.h5: {latent_dir}")

        self.zs, self.acts, self.samples = [], [], []
        self.dim = self.num_patches = None
        for fn in files:
            with h5py.File(fn, "r") as f:
                z = f["z"][:]                    # (T, P, D)
                src = str(f.attrs["source"])
            with h5py.File(episode_dir / src, "r") as f:
                a = f["action"][:]               # (T, 4)
            if len(z) != len(a):
                raise ValueError(f"{fn}: latent 帧数 {len(z)} 与 action 数 {len(a)} 不一致")
            self.dim, self.num_patches = z.shape[2], z.shape[1]
            ep = len(self.zs)
            self.zs.append(z)
            self.acts.append(a.astype(np.float32))
            for t in range(len(z) - horizon):
                self.samples.append((ep, t))
        self.horizon = horizon
        print(f"Dataset：{len(files)} 条 episode，{len(self.samples)} 个样本，"
              f"dim={self.dim} num_patches={self.num_patches} horizon={horizon}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ep, t = self.samples[i]
        h = self.horizon
        z = self.zs[ep]
        z_t = z[t].astype(np.float32)                       # (P, D)
        acts = self.acts[ep][t:t + h]                       # (h, 4)
        tgts = z[t + 1:t + 1 + h].astype(np.float32)        # (h, P, D)
        return z_t, acts, tgts


# ---------------- Model ----------------
class LatentPredictor(nn.Module):
    """以 (z_t, a_t) 预测 z_{t+1} 的最简 patch-latent Transformer（残差预测）。"""

    def __init__(self, dim, num_patches, action_dim=4, n_layers=4, n_heads=8, mlp_ratio=4):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * mlp_ratio,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, z, a):
        # z: (B, P, D)；a: (B, action_dim)
        ae = self.action_mlp(a).unsqueeze(1)   # (B,1,D) 动作条件，广播到每个 patch
        x = z + self.pos + ae
        x = self.encoder(x)
        return z + self.head(x)                # 残差：预测 z_{t+1}


# ---------------- Train / Eval ----------------
def rollout_loss(model, z0, acts, tgts, loss_fn):
    """从 z0 起按 acts 多步 rollout（预测喂回输入），对每步与 tgts 求平均 loss。"""
    z = z0
    total = 0.0
    h = acts.shape[1]
    for k in range(h):
        z = model(z, acts[:, k])
        total = total + loss_fn(z, tgts[:, k])
    return total / h


@torch.no_grad()
def rollout_eval(model, dataset, device, max_steps=20):
    """对验证集第一条 episode，从 z[0] 连续 rollout，打印每步 latent MSE（看误差累积）。"""
    model.eval()
    z = torch.from_numpy(dataset.zs[0][0:1].astype(np.float32)).to(device)  # (1,P,D)
    acts = dataset.acts[0]
    T = min(max_steps, len(acts) - 1)
    errs = []
    for k in range(T):
        a = torch.from_numpy(acts[k:k + 1]).to(device)
        z = model(z, a)
        z_true = torch.from_numpy(dataset.zs[0][k + 1:k + 2].astype(np.float32)).to(device)
        errs.append(float(((z - z_true) ** 2).mean()))
    print("  多步 rollout 每步 MSE:", " ".join(f"{e:.4f}" for e in errs[:10]),
          "..." if T > 10 else "")
    return errs


def compute_latent_stats(zs):
    """对一组 latent（每个 (T,P,D)）求每维(D)的均值/标准差，用于标准化。

    latent 存盘是 float16，直接在 float16 上对几百万个元素求和会溢出（>65504），
    必须先升 float32、并用 float64 累加，否则统计量出 inf/nan。
    """
    D = zs[0].shape[-1]
    flat = np.concatenate([z.reshape(-1, D).astype(np.float32) for z in zs], axis=0)
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)         # (D,)
    std = (flat.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)  # (D,) 加 eps 防除零
    return mean, std


def apply_latent_norm(dataset, mean, std):
    """就地把 dataset 的 latent 标准化为 (z-mean)/std。"""
    for i in range(len(dataset.zs)):
        dataset.zs[i] = ((dataset.zs[i] - mean) / std).astype(np.float32)


def save_loss_log(history, ckpt):
    """把 loss 历史写成 csv（一定写），并尽量画一张 png（没装 matplotlib 就跳过）。"""
    import csv
    ckpt = Path(ckpt)
    csv_path = ckpt.with_suffix(".loss.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse"])
        for ep, tr, va in history:
            w.writerow([ep, f"{tr:.6f}", "" if va is None else f"{va:.6f}"])
    print(f"已保存 loss 曲线数据: {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps = [h[0] for h in history]
        plt.figure(figsize=(7, 4))
        plt.plot(eps, [h[1] for h in history], label="train_mse")
        ve = [(h[0], h[2]) for h in history if h[2] is not None]
        if ve:
            plt.plot([e for e, _ in ve], [v for _, v in ve], "o-", label="val_mse")
        plt.xlabel("epoch"); plt.ylabel("MSE"); plt.yscale("log")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        png_path = ckpt.with_suffix(".loss.png")
        plt.savefig(png_path, dpi=120)
        print(f"已保存 loss 曲线图: {png_path}")
    except Exception as e:
        print(f"（未画图，可只看 csv）: {e}")


def split_episodes(latent_dir, val_num):
    n = len(sorted(glob.glob(str(Path(latent_dir) / "episode_*_dino.h5"))))
    idx = list(range(n))
    val = idx[-val_num:] if (val_num > 0 and n > val_num) else []
    train = [i for i in idx if i not in val]
    return train, val


def parse_args():
    p = argparse.ArgumentParser(description="训练 action-conditioned latent predictor")
    p.add_argument("--latent-dir", type=Path, default=PROJECT_ROOT / "outputs/features/latents")
    p.add_argument("--episode-dir", type=Path, default=PROJECT_ROOT / "outputs/datasets/episodes_dataset")
    p.add_argument("--ckpt", type=Path, default=PROJECT_ROOT / "weights/predictor_h5.pt")
    p.add_argument("--horizon", type=int, default=1, help="训练步数：1=单步，>1=多步 rollout")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--val-num", type=int, default=1, help="留出做 rollout 验证的 episode 数")
    p.add_argument("--overfit", action="store_true", help="只用 1 条 episode、不留验证，验证能否学到东西")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-normalize", action="store_true",
                   help="关闭 latent 标准化（默认开启，强烈建议保持开启）")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子(初始化+打乱)。集成：换 seed 训多个成员")
    p.add_argument("--bootstrap", type=float, default=0.0,
                   help=">0 时每个成员只训随机采样的该比例训练 episode(如0.85)，增大集成差异")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.overfit:
        train_eps, val_eps = [0], [0]
    else:
        train_eps, val_eps = split_episodes(args.latent_dir, args.val_num)
    if args.bootstrap > 0 and not args.overfit:      # 集成成员：episode 级自助采样
        rng = np.random.default_rng(args.seed)
        n_keep = max(1, int(len(train_eps) * args.bootstrap))
        train_eps = sorted(rng.choice(train_eps, size=n_keep, replace=False).tolist())
    print(f"设备={device} seed={args.seed} 训练集 episode={train_eps} 验证集 episode={val_eps}")

    train_set = LatentActionDataset(args.latent_dir, args.episode_dir,
                                    horizon=args.horizon, episodes=train_eps)
    if args.no_normalize:
        z_mean = z_std = None
        print("未做 latent 标准化（--no-normalize）")
    else:
        z_mean, z_std = compute_latent_stats(train_set.zs)  # 仅用训练集统计，不泄漏验证集
        apply_latent_norm(train_set, z_mean, z_std)
        print(f"latent 已按训练集每维均值/方差标准化（dim={len(z_mean)}）")
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    model = LatentPredictor(train_set.dim, train_set.num_patches,
                            n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params/1e6:.2f}M")

    val_set = None
    if val_eps:
        val_set = LatentActionDataset(args.latent_dir, args.episode_dir,
                                      horizon=1, episodes=val_eps)
        if z_mean is not None:
            apply_latent_norm(val_set, z_mean, z_std)  # 用训练集统计标准化验证集

    history = []  # 每个 epoch: (epoch, train_mse, val_mse)；val_mse 没评估时为空
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for z0, acts, tgts in loader:
            z0 = z0.to(device); acts = acts.to(device); tgts = tgts.to(device)
            loss = rollout_loss(model, z0, acts, tgts, loss_fn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item(); nb += 1
        train_mse = running / max(nb, 1)
        print(f"epoch {epoch:3d}/{args.epochs}  train_mse={train_mse:.5f}")
        val_mse = None
        if val_set is not None and (epoch % 10 == 0 or epoch == args.epochs):
            errs = rollout_eval(model, val_set, device)
            val_mse = float(np.mean(errs)) if errs else None
        history.append((epoch, train_mse, val_mse))

    save_loss_log(history, args.ckpt)
    torch.save({"model": model.state_dict(),
                "dim": train_set.dim, "num_patches": train_set.num_patches,
                "args": vars(args), "history": history,
                "z_mean": z_mean, "z_std": z_std}, args.ckpt)
    print(f"已保存权重: {args.ckpt}")


if __name__ == "__main__":
    main()
