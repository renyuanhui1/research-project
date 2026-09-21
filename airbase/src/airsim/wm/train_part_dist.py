"""训练「单帧 → 到目标部位锚点的距离」回归器（airbase `StaticMeshActor_22`）。

动机见 airbase/EXPERIMENTS.md §2026-08-06 四.3：mass 碗形使减速律从不触发、末段深度自关，
需要一个不依赖指纹置信度的距离来源。本脚本**只做离线训练**，不接 servo_closed_loop。

标签来源是 loop/eval_run.py 的 part_anchors（**import，不复制**）—— 与离线评价同源，
避免像 depth_map.py 的 /1000 那样两处实现只修一份。
输入是抽好的 DINO latent（outputs/features/part_dist/<数据集>/），不现场跑 DINO 前向。

_22 的最长轴是局部轴 1，故 part_anchors 的 nose≡axis1+、tail≡axis1-，**实际只有 7 个不同点**。
训练 7 路，报表再用 PART_ALIASES 展开成 9 行。

用法：
  python train_part_dist.py                 # 100 条 _22，val 10 / train 90
  python train_part_dist.py --overfit        # 只喂 1 条，验证代码本身(V5.1)
"""
import argparse, math, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, h5py
import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
from train_value import pool_spatial                 # 与价值函数同一套池化，口径必须一致
from eval_run import quat_to_R, part_anchors, in_view

BASE = str(_AIRBASE_ROOT); DEV = "cuda"

# 7 个互异的锚点；alias 只用于报表凑满 part_anchors 的 9 个名字
PARTS = ["nose", "tail", "center", "top", "bottom", "axis0+", "axis0-"]
PART_ALIASES = {"axis1+": "nose", "axis1-": "tail"}
LONG_AXIS = 1                    # _22 的 OOB 最长轴；PART_ALIASES 成立的前提
BANDS = [0, 10, 15, 20, 30, 45, 70]


def find_pairs(latent_root, dataset_root):
    """→ [(sub, src, latent_path, episode_path)]。硬断言，绝不静默。

    追溯键是 (子目录名, attrs["source"]) 二元组 —— 单看 source 两个数据集会撞名
    （nearfield 与 randseg 有 20 个同名 episode）。
    """
    out = []
    for sub in sorted(p for p in latent_root.iterdir() if p.is_dir()):
        for lf in sorted(sub.glob("episode_*_dino.h5")):
            with h5py.File(lf, "r") as f:
                src = str(f.attrs["source"]); T = f["z"].shape[0]
            ep = dataset_root / sub.name / src
            if not ep.exists():
                raise SystemExit(f"{lf} 的 source={src} 在 {ep} 找不到")
            with h5py.File(ep, "r") as f:
                if f["pose"].shape[0] != T:
                    raise SystemExit(f"{lf} latent {T} 帧 vs pose {f['pose'].shape[0]} 帧不一致")
            out.append((sub.name, src, lf, ep))
    keys = {(s, n) for s, n, _, _ in out}
    if len(keys) != len(out):
        raise SystemExit(f"有重名 episode 被复用：{len(out)} 条只有 {len(keys)} 个唯一键")
    return out


def part_labels(ep_h5):
    """一条采集 episode → 位姿 (T,3)、7 路真值距离 (T,7) 米、7 路可观测掩膜 (T,7)。"""
    with h5py.File(ep_h5, "r") as h:
        pose = h["pose"][:].astype(np.float64)
        if str(h.attrs.get("target_name")) != "StaticMeshActor_22":
            raise SystemExit(f"{ep_h5} 的 target 不是 _22：{h.attrs.get('target_name')}")
        center = np.asarray(h.attrs["target_center"], float)
        size = np.asarray(h.attrs["target_size"], float)
        quat = np.asarray(h.attrs["target_quat"], float)
    pos = pose[:, :3]
    yaw = np.array([math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2)) for q in pose[:, 3:]])
    A, order = part_anchors(center, size, quat_to_R(quat))
    if order != LONG_AXIS:
        raise SystemExit(f"{ep_h5} 最长轴 order={order}，与别名表假设 {LONG_AXIS} 不符")
    Y = np.stack([np.linalg.norm(pos - np.asarray(A[k], float), axis=1) for k in PARTS], 1)
    vis = np.stack([np.array([bool(in_view(A[k], p, y)[0]) for p, y in zip(pos, yaw)])
                    for k in PARTS], 1)
    return pos, Y.astype(np.float32), vis


def read_pooled(lf, pool_grid):
    """读 latent（float16）→ 池化 (T, grid²·384) float32。升 float32 再算，fp16 直接求和会溢出。"""
    with h5py.File(lf, "r") as f:
        z = f["z"][:].astype(np.float32)
    return pool_spatial(torch.from_numpy(z), pool_grid).numpy()


def split_episodes(mins, val_near, val_far):
    """按每条 episode 的**最小 nose 距离**升序切分，返回 (train_idx, val_idx)。

    不能用 band 属性：far 带的 nearfield/episode_0100 最小距离 8.2m，比 near 带的
    episode_0012(8.9m) 还近 —— 按 band 分会把最近的两条分到同一组。
    也不用现成脚本的 sorted()[-N:]：末 N 条会全是 randseg 的远带，val 里 0–20m 帧数为 0。
    """
    order = sorted(range(len(mins)), key=lambda i: mins[i])
    val = list(order[:val_near])
    rest = order[val_near:]
    if val_far > 0 and rest:
        idx = np.linspace(0, len(rest) - 1, min(val_far, len(rest))).round().astype(int)
        val += [rest[i] for i in sorted(set(idx.tolist()))]
    val = sorted(set(val))
    tr = [i for i in range(len(mins)) if i not in set(val)]
    return tr, val


class PartDistNet(nn.Module):
    """池化 latent → 7 路距离。共享全部上游，只在最后一层分叉。"""
    def __init__(self, d=6144, h=256, n_out=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, n_out))

    def forward(self, z):          # (...,d) -> (...,n_out)
        return self.net(z)


def band_stat(d, v, tag):
    """打印分带帧数（含可观测帧数），用于切分自检。"""
    print(f"  {tag}:")
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (d >= lo) & (d < hi)
        print(f"    {lo:2d}-{hi:2d} m: n={int(m.sum()):5d}  n_vis={int((m & v).sum()):5d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE)
    p.add_argument("--latent-root", default=None, help="outputs/features/part_dist")
    p.add_argument("--dataset-root", default=None, help="outputs/datasets")
    p.add_argument("--out", default=None, help="输出权重")
    p.add_argument("--pool-grid", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-near", type=int, default=1, help="val 里放最近的第几条")
    p.add_argument("--val-far", type=int, default=9, help="其余里等间隔再抽几条进 val")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", action="store_true", help="只训 1 条，验证代码正确性(V5.1)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    b = args.base
    args.latent_root = args.latent_root or f"{b}/outputs/features/part_dist"
    args.dataset_root = args.dataset_root or f"{b}/outputs/datasets"
    args.out = args.out or f"{b}/weights/part_dist_22.pt"
    if args.overfit:                       # 别用 1 条训出来的垃圾模型覆盖正式权重
        args.out = str(Path(args.out).with_name(Path(args.out).stem + "_overfit.pt"))

    global DEV
    DEV = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    pairs = find_pairs(Path(args.latent_root), Path(args.dataset_root))
    print(f"找到 {len(pairs)} 条 episode（{args.latent_root}）")

    eps = []
    for sub, src, lf, ep in pairs:
        pos, Y, vis = part_labels(ep)
        Z = read_pooled(lf, args.pool_grid)
        if len(Z) != len(Y):
            raise SystemExit(f"{lf} 池化 {len(Z)} 帧 vs 标签 {len(Y)} 帧不一致")
        eps.append({"tag": f"{sub}/{src}", "Z": Z, "Y": Y, "vis": vis,
                    "min_dist": float(Y[:, 0].min())})     # 第 0 路 = nose
        print(f"  {eps[-1]['tag']:32s} T={len(Y)}  nose {Y[:,0].min():5.1f}~{Y[:,0].max():5.1f} m")

    tr_idx, va_idx = split_episodes([e["min_dist"] for e in eps], args.val_near, args.val_far)
    if args.overfit:
        tr_idx, va_idx = tr_idx[:1], tr_idx[:1]
    if not va_idx:
        raise SystemExit("val 为空")
    print(f"\n切分: train {len(tr_idx)} 条 / val {len(va_idx)} 条")
    print(f"  val = {[eps[i]['tag'] for i in va_idx]}")

    def gather(idx):
        Z = torch.tensor(np.concatenate([eps[i]["Z"] for i in idx]))
        Y = torch.tensor(np.concatenate([eps[i]["Y"] for i in idx]))
        V = np.concatenate([eps[i]["vis"] for i in idx])
        return Z, Y, V

    Ztr, Ytr, Vtr = gather(tr_idx)
    Zva, Yva, Vva = gather(va_idx)
    print(f"  train {tuple(Ztr.shape)}  val {tuple(Zva.shape)}")

    # V4：近场不能被远场淹没；0-10m 必须在某一边有帧
    print("\n=== 切分自检（nose 真值）===")
    band_stat(Ytr[:, 0].numpy(), Vtr[:, 0], "train")
    band_stat(Yva[:, 0].numpy(), Vva[:, 0], "val")
    n_val_near = int(((Yva[:, 0] >= 0) & (Yva[:, 0] < 20)).sum())
    if n_val_near < 100 and not args.overfit:
        raise SystemExit(f"val 的 0–20m 只有 {n_val_near} 帧，近场被淹没了，检查 --val-near")
    if not args.overfit:
        assert set(tr_idx) & set(va_idx) == set() and len(set(tr_idx) | set(va_idx)) == len(eps)

    # 标准化统计量只从 train 算（不泄漏），与 train_predictor 同一纪律
    z_mean = Ztr.reshape(-1, Ztr.shape[-1]).mean(0).to(DEV)
    z_std = Ztr.reshape(-1, Ztr.shape[-1]).std(0).clamp_min(1e-6).to(DEV)
    y_mean = Ytr.mean(0).to(DEV)
    y_std = Ytr.std(0).clamp_min(1e-6).to(DEV)

    Ztr_n = ((Ztr.to(DEV) - z_mean) / z_std)
    Ytr_n = ((Ytr.to(DEV) - y_mean) / y_std)
    Zva_n = ((Zva.to(DEV) - z_mean) / z_std)

    DIM = args.pool_grid ** 2 * 384
    model = PartDistNet(DIM).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.HuberLoss()
    N = len(Ytr_n)

    def mae_m(z_n, Y_m):
        """归一化空间预测 → 米制 MAE（全帧 × 全路平均）。"""
        with torch.no_grad():
            pred = model(z_n) * y_std + y_mean
        return float((pred - Y_m.to(DEV)).abs().mean())

    # null baseline：恒定输出 train 中位距离。赢不了它 = 什么都没学到
    null_mae = float((Ytr.median(0).values.to(DEV) - Yva.to(DEV)).abs().mean())

    log = []
    best = (float("inf"), None, -1)
    print(f"\n训练 {args.epochs} epoch（{N} 帧 × 7 路，null baseline val MAE={null_mae:.2f} m）")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=DEV)
        tot = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            loss = lossf(model(Ztr_n[idx]), Ytr_n[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % args.val_every == 0 or ep == 0:
            model.eval()
            tr_mae, va_mae = mae_m(Ztr_n, Ytr), mae_m(Zva_n, Yva)
            log.append((ep + 1, tr_mae, va_mae))
            flag = ""
            if va_mae < best[0]:
                best = (va_mae, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
                flag = "  <- best"
            print(f"  epoch {ep+1:3d}  huber={tot/N:.4f}  train MAE={tr_mae:6.2f} m  "
                  f"val MAE={va_mae:6.2f} m{flag}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    print(f"\n最优 epoch {best[2]}  val MAE={best[0]:.2f} m  (null baseline {null_mae:.2f} m)")
    if best[0] >= null_mae:
        print("  ❌ 没有赢过 null baseline —— 模型没学到东西")
    else:
        print(f"  ✅ 比 null baseline 好 {(1 - best[0] / null_mae) * 100:.0f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model": model.state_dict(), "d_in": DIM, "h": 256, "n_out": len(PARTS),
        "pool_grid": args.pool_grid, "parts": PARTS, "part_aliases": PART_ALIASES,
        "long_axis": LONG_AXIS,
        "y_mean": y_mean.cpu().numpy(), "y_std": y_std.cpu().numpy(),
        "z_mean": z_mean.cpu().numpy(), "z_std": z_std.cpu().numpy(),
        "target_name": "StaticMeshActor_22",
        "train_pairs": [eps[i]["tag"] for i in tr_idx],
        "val_pairs": [eps[i]["tag"] for i in va_idx],
        "epoch": best[2], "val_mae_m": best[0], "null_mae_m": null_mae,
        "args": vars(args),
    }, args.out)
    print(f"已保存 {args.out}")

    lp = Path(args.out).with_suffix(".loss.csv")
    with open(lp, "w") as f:
        f.write("epoch,train_mae_m,val_mae_m\n")
        for e, a, c in log:
            f.write(f"{e},{a:.4f},{c:.4f}\n")
    print(f"loss -> {lp}")


if __name__ == "__main__":
    main()
