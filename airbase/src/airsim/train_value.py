"""训练"时间距离"价值函数 V(z, z_goal) ≈ 还差几步到目标（世界模型的强尺子）。

思路(后见之明/hindsight)：同一条轨迹里 frame t 到 frame t+k 就是 k 步。
用现有数据训 V(pool(z_t), pool(z_{t+k})) ≈ k；跨集配对当"很远"(=KMAX)。
V 按定义单调、梯度强，用来替换 MPPI 里又平又弱的 DINO 潜距目标。

输入表示：对标准化 patch latent 按 patch 取均值池化成 (384,)，与闭环里预测 latent 池化一致。
产物：weights/value_fn.pt（含 MLP 权重 + 维度 + KMAX）。
缓存：outputs/cache/pooled_latents_g4.npz（池化latent，二次运行秒开）。
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, h5py
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD

BASE = str(Path(__file__).resolve().parents[2]); DEV = "cuda"


def pool_spatial(z, grid_out=4):
    """粗空间池化：z (...,P,D) 的 P=g×g patch → grid_out×grid_out 块均值 → 展平 (...,grid_out²·D)。
    保留左右/上下/远近的粗布局，让价值函数能区分"前进"与"横移"（全局平均会丢掉这个）。"""
    lead = z.shape[:-2]; P, D = z.shape[-2], z.shape[-1]
    g = int(round(P ** 0.5)); bs = g // grid_out
    z = z.reshape(*lead, g, g, D)[..., :grid_out * bs, :grid_out * bs, :]
    z = z.reshape(*lead, grid_out, bs, grid_out, bs, D).mean(dim=(-4, -2))
    return z.reshape(*lead, grid_out * grid_out * D)


def get_pooled_latents(args):
    """编码所有 episode 的帧 → 标准化后粗空间池化 (T, grid²·384)，缓存到 npz。"""
    cache = args.cache_path
    if os.path.exists(cache) and not args.reextract:
        d = np.load(cache, allow_pickle=True)
        eps = [d[k] for k in d.files]
        print(f"载入缓存池化latent: {len(eps)} 条")
        return eps
    dino = load_model("dinov2_vits14", DEV, weights=args.dino_weights, repo_dir=args.repo_dir)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    zm = torch.tensor(np.asarray(ck["z_mean"]), device=DEV)
    zs = torch.tensor(np.asarray(ck["z_std"]), device=DEV)
    mean = torch.tensor(IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEV).view(1, 3, 1, 1)
    files = sorted(glob.glob(f"{args.episodes_dir}/episode_*.h5"))
    eps = []
    with torch.no_grad():
        for i, f in enumerate(files):
            with h5py.File(f, "r") as h:
                rgb = h["rgb"][:]
            pooled = []
            for r in rgb:
                x = to_input_tensor(r[None], 224, mean, std, DEV)
                z = (dino.forward_features(x)["x_norm_patchtokens"][0].float() - zm) / zs
                pooled.append(pool_spatial(z, args.pool_grid).cpu().numpy())
            eps.append(np.stack(pooled).astype(np.float32))
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} T={len(rgb)}")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, *eps)
    print(f"池化latent已缓存: {cache}")
    return eps


class ValueFn(nn.Module):
    def __init__(self, d=384, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1))

    def forward(self, z, g):  # z,g: (...,384) -> (...,)
        return self.net(torch.cat([z, g, g - z], dim=-1)).squeeze(-1)


def build_samples(eps, KMAX, neg_frac, rng, k_per_frame=5):
    """构造 (z, goal, 距离) 三元组。正样本=同轨迹 k 步；负样本=跨集(=KMAX)。"""
    Z, G, D = [], [], []
    for ep in eps:
        T = len(ep)
        for t in range(T):
            D.append(0.0); Z.append(ep[t]); G.append(ep[t])           # 自身=0
            for k in rng.integers(1, KMAX + 1, size=k_per_frame):     # 每帧采多个k(加密)
                if t + k < T:
                    Z.append(ep[t]); G.append(ep[t + k]); D.append(float(k))
    n = len(Z); nn_neg = int(n * neg_frac)
    for _ in range(nn_neg):                                            # 跨集负样本=KMAX
        a, b = rng.integers(0, len(eps), size=2)
        if a == b: continue
        Z.append(eps[a][rng.integers(len(eps[a]))])
        G.append(eps[b][rng.integers(len(eps[b]))]); D.append(float(KMAX))
    return (torch.tensor(np.stack(Z)), torch.tensor(np.stack(G)), torch.tensor(D, dtype=torch.float32))


def build_consistency(eps, KMAX, rng, k_per_frame=3):
    """时间一致性三元组 (z_t, z_{t+1}, goal=z_{t+k})，训练时约束 V(z_t,g) ≈ 1 + V(z_{t+1},g)。
    k≥2 保证 z_{t+1} 仍在目标之前。让代价沿轨迹按步单调、平滑（消除偶发离谱动作）。"""
    Zt, Zt1, G = [], [], []
    for ep in eps:
        T = len(ep)
        for t in range(T - 1):
            for k in rng.integers(2, KMAX + 1, size=k_per_frame):
                if t + k < T:
                    Zt.append(ep[t]); Zt1.append(ep[t + 1]); G.append(ep[t + k])
    return (torch.tensor(np.stack(Zt)), torch.tensor(np.stack(Zt1)), torch.tensor(np.stack(G)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE, help="默认根；下面各路径不填时按它推")
    p.add_argument("--repo-dir", default=None, help="dinov2 仓库目录")
    p.add_argument("--dino-weights", default=None)
    p.add_argument("--ckpt", default=None, help="predictor_h5.pt(取 z_mean/z_std)")
    p.add_argument("--out", default=None, help="输出 value_fn 权重")
    p.add_argument("--episodes-dir", default=None, help="含 episode_*.h5 的目录")
    p.add_argument("--cache-path", default=None, help="池化latent缓存 npz 路径")
    p.add_argument("--pool-grid", type=int, default=4, help="粗空间池化网格(4→4×4×384=6144维)")
    p.add_argument("--kmax", type=int, default=25)
    p.add_argument("--neg-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-eps", type=int, default=5, help="留出末尾几条做验证")
    p.add_argument("--k-per-frame", type=int, default=5, help="每帧回归采样的 k 个数(加密)")
    p.add_argument("--consistency-weight", type=float, default=1.0,
                   help="时间一致性损失权重 V(z_t,g)≈1+V(z_{t+1},g)；0=关闭")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Adam 权重衰减(平滑代价面)")
    p.add_argument("--reextract", action="store_true")
    p.add_argument("--device", default="cuda", help="有缓存时训小MLP，本地可用 cpu(不吃显卡)")
    args = p.parse_args()

    b = args.base
    args.repo_dir = args.repo_dir or f"{b}/dinov2"
    args.dino_weights = args.dino_weights or f"{b}/weights/dinov2_vits14_pretrain.pth"
    args.ckpt = args.ckpt or f"{b}/weights/predictor_h5.pt"
    args.out = args.out or f"{b}/weights/value_fn.pt"
    args.episodes_dir = args.episodes_dir or f"{b}/outputs/datasets/episodes_dataset"
    args.cache_path = args.cache_path or f"{b}/outputs/cache/pooled_latents_g{args.pool_grid}.npz"

    global DEV
    DEV = args.device

    eps = get_pooled_latents(args)
    rng = np.random.default_rng(0)
    tr, va = eps[:-args.val_eps], eps[-args.val_eps:]
    Z, G, D = build_samples(tr, args.kmax, args.neg_frac, rng, args.k_per_frame)
    Z, G, D = Z.to(DEV), G.to(DEV), D.to(DEV)
    cw = args.consistency_weight
    if cw > 0:
        Zt, Zt1, Gc = build_consistency(tr, args.kmax, rng)
        Zt, Zt1, Gc = Zt.to(DEV), Zt1.to(DEV), Gc.to(DEV)
        print(f"一致性样本 {len(Gc)}  (权重={cw})")
    print(f"回归样本 {len(D)}  (KMAX={args.kmax}, k/帧={args.k_per_frame})")

    DIM = args.pool_grid ** 2 * 384
    model = ValueFn(d=DIM).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.HuberLoss()
    N = len(D)
    Nc = len(Gc) if cw > 0 else 0
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=DEV)
        tr_r, tr_c = 0.0, 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            loss_r = lossf(model(Z[idx], G[idx]), D[idx])
            loss = loss_r
            if cw > 0:                                  # 时间一致性(TD 式，目标端 detach)
                cidx = torch.randint(0, Nc, (len(idx),), device=DEV)
                v_t = model(Zt[cidx], Gc[cidx])
                with torch.no_grad():
                    tgt = 1.0 + model(Zt1[cidx], Gc[cidx])
                loss_c = lossf(v_t, tgt)
                loss = loss + cw * loss_c
                tr_c += loss_c.item() * len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_r += loss_r.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  回归huber={tr_r/N:.3f}  一致性huber={tr_c/max(N,1):.3f}")

    torch.save({"model": model.state_dict(), "d": DIM, "pool_grid": args.pool_grid, "kmax": args.kmax}, args.out)
    print(f"已保存 {args.out}")

    # 验证：留出集，V(z_t, z_last) 应随 t 单调降到 ~0
    model.eval()
    print("\n=== 验证：留出集 V(z_t, 末帧) 应随 t 单调下降 ===")
    with torch.no_grad():
        for ep in va[:3]:
            T = len(ep); g = torch.tensor(ep[-1], device=DEV)
            zs_ = torch.tensor(ep, device=DEV)
            v = model(zs_, g.expand_as(zs_)).cpu().numpy()
            idxs = list(range(0, T, max(1, T // 8)))
            print("  " + "  ".join(f"t{t}:{v[t]:.1f}" for t in idxs))
            # 单调性：与理想距离 (T-1-t) 的相关
            ideal = np.arange(T)[::-1].astype(np.float32)
            corr = np.corrcoef(v, ideal)[0, 1]
            print(f"    与理想剩余步数相关={corr:.3f} (越接近1越单调)")


if __name__ == "__main__":
    main()
