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
def stack_action_history(a, k):
    """(T,4) → (T, 4k)：每步拼上最近 k 条指令 [a_t, a_{t-1}, ..., a_{t-k+1}]，开头补零。

    为什么需要（08-01 实测）：无人机这一步怎么动，取决于**当前速度**，而速度是过去十几步
    指令累积的结果（飞控速度环 ~11 步才跟上指令）——单帧图像里看不出速度。
    对 wm300 做线性回归：位移只用 a_t 时 R²=0.468，只用 a_{t-1} 反而 R²=0.576，
    用 a_t..a_{t-3} 到 0.737。即模型只拿 a_t 时，信息上限就锁死在 47%，
    学不到动作特异性（实测 model/swap 0.83~0.92，训练集上同样差 → 不是过拟合是欠信息）。

    补零 = "此前静止"，与事实相符：goto_station 到位后有 rotate + 0.8s settle。
    闭环合法性：历史指令是控制器自己发出去的，不是特权信息；MPPI 评估候选序列时，
    第 k 步的历史就是它自己前面几步的候选值，训练与部署的输入构造方式一致。
    """
    if k <= 1:
        return a.astype(np.float32)
    pad = np.zeros((k - 1, a.shape[1]), np.float32)
    p = np.concatenate([pad, a.astype(np.float32)], 0)         # (T+k-1, 4)
    return np.concatenate([p[k - 1 - j:len(p) - j] for j in range(k)], axis=1)


class LatentActionDataset(Dataset):
    """把 latent + action 构造成 (z_t, actions[t:t+h], z[t+1:t+h+1]) 样本。"""

    def __init__(self, latent_dir, episode_dir, horizon=1, episodes=None, action_hist=1):
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
            self.acts.append(stack_action_history(a, action_hist))
            for t in range(len(z) - horizon):
                self.samples.append((ep, t))
        self.horizon = horizon
        self.action_hist = action_hist
        self.action_dim = self.acts[0].shape[1]
        print(f"Dataset：{len(files)} 条 episode，{len(self.samples)} 个样本，"
              f"dim={self.dim} num_patches={self.num_patches} horizon={horizon} "
              f"action_hist={action_hist}(动作维 {self.action_dim})")

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
    """对**每条**验证 episode 从 z[0] 连续 rollout，返回逐步 MSE 的跨 episode 平均。

    08-01 修：原来写死 `dataset.zs[0]`，只跑验证集第一条 —— `--val-num` 给多少都没用，
    val_mse 永远是 n=1。实测该值在相邻评估点之间能跳 ±0.09（0.618/0.727/0.677/0.636），
    噪声压过信号，据此判早停或挑检查点都不可靠。
    """
    model.eval()
    per_ep = []
    for zs, acts in zip(dataset.zs, dataset.acts):
        T = min(max_steps, len(acts) - 1)
        if T <= 0:
            continue
        z = torch.from_numpy(zs[0:1].astype(np.float32)).to(device)     # (1,P,D)
        errs = []
        for k in range(T):
            a = torch.from_numpy(acts[k:k + 1]).to(device)
            z = model(z, a)
            z_true = torch.from_numpy(zs[k + 1:k + 2].astype(np.float32)).to(device)
            errs.append(float(((z - z_true) ** 2).mean()))
        per_ep.append(errs)
    if not per_ep:
        return []
    n = min(len(e) for e in per_ep)                  # episode 不等长时按最短对齐
    mean = np.mean(np.array([e[:n] for e in per_ep]), axis=0)
    print(f"  多步 rollout 每步 MSE（跨 {len(per_ep)} 条平均）:",
          " ".join(f"{e:.4f}" for e in mean[:10]), "..." if n > 10 else "")
    return mean.tolist()


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
    p.add_argument("--action-hist", type=int, default=1,
                   help="每步拼最近几条指令作为动作输入（1=只用 a_t，即旧行为；4=实测 R² 0.47→0.74）")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--val-num", type=int, default=1, help="留出做 rollout 验证的 episode 数")
    p.add_argument("--val-every", type=int, default=5, help="每几个 epoch 评一次验证集")
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
                                    horizon=args.horizon, episodes=train_eps,
                                    action_hist=args.action_hist)
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
                            action_dim=train_set.action_dim,
                            n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params/1e6:.2f}M")

    val_set = None
    if val_eps:
        val_set = LatentActionDataset(args.latent_dir, args.episode_dir,
                                      horizon=1, episodes=val_eps,
                                      action_hist=args.action_hist)
        if z_mean is not None:
            apply_latent_norm(val_set, z_mean, z_std)  # 用训练集统计标准化验证集

    def save_ckpt(epoch, val_mse, hist):
        torch.save({"model": model.state_dict(),
                    "dim": train_set.dim, "num_patches": train_set.num_patches,
                    # 评估/规划脚本按这两个字段重建同样的动作输入，防止训练与推理格式对不上
                    "action_hist": train_set.action_hist, "action_dim": train_set.action_dim,
                    "args": vars(args), "history": hist,
                    "epoch": epoch, "val_mse": val_mse,
                    "z_mean": z_mean, "z_std": z_std}, args.ckpt)

    history = []  # 每个 epoch: (epoch, train_mse, val_mse)；val_mse 没评估时为空
    best_val, best_epoch = float("inf"), -1
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
        if val_set is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            errs = rollout_eval(model, val_set, device)
            val_mse = float(np.mean(errs)) if errs else None
        history.append((epoch, train_mse, val_mse))
        # 存**验证最优**而非最后一轮：数据量一变，最优轮数就变，靠猜 --epochs 不可靠。
        if val_mse is not None and val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            save_ckpt(epoch, val_mse, history)
            print(f"  ↑ 新的最优 val={val_mse:.5f}，已存 {args.ckpt.name}")

    save_loss_log(history, args.ckpt)
    if best_epoch < 0:                      # 没有验证集（--overfit / --val-num 0）时存最后一轮
        save_ckpt(args.epochs, None, history)
        print(f"已保存权重（最后一轮，无验证集）: {args.ckpt}")
    else:
        print(f"已保存权重（第 {best_epoch} 轮，val={best_val:.5f}）: {args.ckpt}")


if __name__ == "__main__":
    main()
