"""check_correspondence_signal.py —— 远段 cost 的读出方式对比：patch 相似度 vs 几何对应。"""

import argparse
import csv
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
    """秩, **并列取平均秩**(无 scipy 依赖)。

    不能用 `argsort(argsort(a))` —— 那对并列值按出现顺序强行分配不同秩, 会系统性**高估**
    相关性。本脚本里 dino_nn / sift_n 是整数计数, 并列率高达 88%/94%, 用错会把
    ρ 从 -0.61 虚报到 -0.76(见 EXPERIMENTS.md §2026-07-29 的修正)。
    """
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0        # 并列段取平均秩
        i = j + 1
    return r


def spearman(a, b):
    """Spearman 秩相关: 对(并列平均)秩做 Pearson。"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def encode_all(model, rgb, image_size, mean, std, device, bs=32):
    """rgb:(N,H,W,3) uint8 → 未标准化 patch 特征 (N,P,D) float32(cpu)。"""
    feats = []
    for i in range(0, len(rgb), bs):
        x = to_input_tensor(rgb[i:i + bs], image_size, mean, std, device)
        feats.append(model.forward_features(x)["x_norm_patchtokens"].float().cpu())
    return torch.cat(feats, 0)


def mutual_nn(znf, znt, patch_xy, thresh):
    """互为最近邻匹配读出。znf:(N,P,D) 帧, znt:(M,D) 模板, 均已 L2 归一。

    返回 (count, conc):
      count —— 互为最近邻且相似度 ≥ thresh 的匹配对数(越多=目标越可辨认)
      conc  —— 这些匹配在画面上的聚集度 = 1/(1+匹配点到其质心的平均距离)。
               真目标的匹配会聚成一团(conc 高); 噪声匹配散布全图(conc 低)。
               这一项是 peak/mass 完全没有的信息, 也是几何读出的价值所在。
    """
    counts, concs = [], []
    for i in range(znf.shape[0]):
        s = znf[i] @ znt.T                       # (P,M) 帧patch × 模板patch
        f2t = s.argmax(dim=1)                    # 每个帧patch 的最佳模板patch
        t2f = s.argmax(dim=0)                    # 每个模板patch 的最佳帧patch
        idx = torch.arange(s.shape[0])
        mutual = t2f[f2t] == idx                 # 互为最近邻
        strong = s.max(dim=1).values >= thresh
        keep = mutual & strong
        n = int(keep.sum())
        counts.append(n)
        if n >= 2:
            xy = patch_xy[keep]                  # (n,2) 匹配点在画面上的归一坐标
            concs.append(float(1.0 / (1.0 + (xy - xy.mean(0)).norm(dim=1).mean())))
        else:
            concs.append(0.0)
    return np.array(counts, np.float32), np.array(concs, np.float32)


def sift_matches(frames, tmpl_bgr, ratio=0.75):
    """SIFT + 比值检验的良匹配数。纯几何、无语义, 作为"对应线索"的下界参考。"""
    sift = cv2.SIFT_create()
    kt, dt = sift.detectAndCompute(cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY), None)
    if dt is None or len(kt) < 2:
        print("[warn] 模板上没提到 SIFT 特征点, sift_n 全 0")
        return np.zeros(len(frames), np.float32)
    bf = cv2.BFMatcher()
    out = []
    for im in frames:
        kf, df = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
        if df is None or len(kf) < 2:
            out.append(0); continue
        good = sum(1 for m, n in bf.knnMatch(df, dt, k=2) if m.distance < ratio * n.distance)
        out.append(good)
    return np.array(out, np.float32)


def binned(dist, vals, edges):
    """按距离分箱取中位数, 返回 [(lo,hi,n,median), ...]。"""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        rows.append((lo, hi, int(m.sum()), float(np.median(vals[m])) if m.any() else float("nan")))
    return rows


def main():
    a = parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    with h5py.File(a.episode, "r") as f:
        rgb = f["rgb"][:]
        pose = f["pose" if "pose" in f else "pose_ned"][:]
    if a.max_frames and len(rgb) > a.max_frames:            # 冒烟用: 均匀抽帧
        sel = np.linspace(0, len(rgb) - 1, a.max_frames).astype(int)
        rgb, pose = rgb[sel], pose[sel]
    print(f"episode: {len(rgb)} 帧, {rgb.shape[1]}x{rgb.shape[2]}")

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model = load_model("dinov2_vits14", device,
                       weights=str(a.dino_weights), repo_dir=str(a.repo_dir))

    # 帧特征与模板无关 → 只编码一次, 多个模板复用(否则每换模板要重编码一遍)
    zf = encode_all(model, rgb, a.image_size, mean, std, device)               # (N,P,D)

    # 标准化统计量必须来自**独立的**episode: 用被测 episode 自身估 z_mean/z_std 是测试信息
    # 泄漏, 会让信号看起来比实际好(见 EXPERIMENTS.md §2026-07-29 的修正)。
    if Path(a.stats_episode).resolve() == Path(a.episode).resolve():
        print("[warn] stats-episode 与被测 episode 同源 → 存在测试信息泄漏, 结果偏乐观")
        zsrc = zf
    else:
        with h5py.File(a.stats_episode, "r") as f:
            srgb = f["rgb"][:]
        zsrc = encode_all(model, srgb, a.image_size, mean, std, device)
        print(f"标准化统计量来自: {Path(a.stats_episode).name} ({len(srgb)} 帧)")
    zm = zsrc.reshape(-1, zsrc.shape[-1]).mean(0)
    zs = zsrc.reshape(-1, zsrc.shape[-1]).std(0).clamp_min(1e-6)
    del zsrc
    zf = (zf - zm) / zs
    znf = F.normalize(zf, dim=-1)

    P = zf.shape[1]; grid = int(round(P ** 0.5))
    xs = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)           # (P,2)

    tgt = np.array([a.target_ned[0], a.target_ned[1], -abs(a.target_alt)])
    dist = np.linalg.norm(pose[:, :3] - tgt, axis=1)
    far = dist >= a.far_min
    edges = [0, 10, 20, 30, 40, 50, 70, 120]
    summary = []

    for tmpl in a.template:
        tbgr = cv2.imread(str(tmpl))
        if tbgr is None:
            print(f"[warn] 读不到模板, 跳过: {tmpl}"); continue
        trgb = np.ascontiguousarray(tbgr[:, :, ::-1])[None]
        zt = (encode_all(model, trgb, a.image_size, mean, std, device)[0] - zm) / zs
        znt = F.normalize(zt, dim=-1)

        # 读出 1: 现有指纹(基线)
        sim = torch.matmul(znf, F.normalize(znt.mean(0), dim=0))              # (N,P)
        peak = sim.max(dim=-1).values.numpy()
        mass = torch.sigmoid((sim - a.mass_thresh) * a.mass_sharpness).mean(dim=-1).numpy()
        # 读出 2: 同一份特征的互为最近邻匹配
        nn_count, nn_conc = mutual_nn(znf, znt, patch_xy, a.nn_thresh)
        # 读出 3: SIFT(可选)
        sift_n = sift_matches(rgb, tbgr) if a.with_sift else np.zeros(len(rgb), np.float32)

        series = [("peak", peak), ("mass", mass), ("dino_nn", nn_count), ("dino_conc", nn_conc)]
        if a.with_sift:
            series.append(("sift_n", sift_n))

        name = Path(tmpl).stem
        print(f"\n{'=' * 68}\n模板 {name}   episode {Path(a.episode).stem}   {len(rgb)} 帧")
        print(f"--- 单调性 Spearman(信号, 距离), 应为**负**(越近信号越强) ---")
        print(f"{'读出':12s} {'全程':>8s} {f'≥{a.far_min:.0f}m 远段':>14s}")
        row = {"模板": name}
        for k, v in series:
            r_all = spearman(v, dist)
            r_far = spearman(v[far], dist[far]) if far.sum() >= 3 else float("nan")
            print(f"{k:12s} {r_all:+8.3f} {r_far:+14.3f}")
            row[k] = r_far
        summary.append(row)
        print(f"  (远段样本 {int(far.sum())} 帧; |ρ|<0.2 ≈ 没梯度, 就是'平')")

        print(f"--- 分箱中位数 ---")
        print("距离".ljust(10) + "n".rjust(5) + "".join(k.rjust(12) for k, _ in series))
        rows0 = binned(dist, series[0][1], edges)
        for i, (lo, hi, n, _) in enumerate(rows0):
            if n == 0:
                continue
            cells = "".join(f"{binned(dist, v, edges)[i][3]:12.3f}" for _, v in series)
            print(f"{f'{lo}-{hi}m':10s}{n:5d}{cells}")

        out = a.out_dir / f"{Path(a.episode).stem}__{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "dist"] + [k for k, _ in series])
            for i in range(len(dist)):
                w.writerow([i, f"{dist[i]:.3f}"] + [f"{v[i]:.4f}" for _, v in series])
        print(f"逐帧写入 {out}")

    if len(summary) > 1:
        keys = [k for k in summary[0] if k != "模板"]
        print(f"\n{'=' * 68}\n汇总: ≥{a.far_min:.0f}m 远段 Spearman ρ(越负越好)")
        print("模板".ljust(12) + "".join(k.rjust(12) for k in keys))
        for r in summary:
            print(r["模板"].ljust(12) + "".join(f"{r[k]:+12.3f}" for k in keys))


def parse_args():
    base = PROJECT_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--template", type=Path, nargs="+",
                   default=[project_path(PROFILE["assets"]["template"])],
                   help="可给多个; 帧特征只编码一次, 多模板复用")
    p.add_argument("--stats-episode", type=Path,
                   default=project_path(PROFILE["assets"]["stats_episode"]),
                   help="标准化统计量来源, 必须与 --episode 不同(否则测试信息泄漏)")
    p.add_argument("--repo-dir", type=Path, default=project_path(PROFILE["assets"]["dino_repo"]))
    p.add_argument("--dino-weights", type=Path,
                   default=project_path(PROFILE["assets"]["dino_weights"]))
    p.add_argument("--target-ned", type=float, nargs=2, required=True,
                   metavar=("N", "E"))
    p.add_argument("--target-alt", type=float, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    p.add_argument("--max-frames", type=int, default=0, help=">0 则均匀抽帧(本地冒烟用)")
    p.add_argument("--with-sift", action="store_true", help="额外算 SIFT 匹配数(CPU, 慢)")
    p.add_argument("--nn-thresh", type=float, default=0.5,
                   help="互为最近邻匹配的相似度下限")
    p.add_argument("--far-min", type=float, default=20.0,
                   help="远段起点(m): 指纹在这以外已知全平, 重点看这一段")
    p.add_argument("--mass-thresh", type=float, default=0.35)
    p.add_argument("--mass-sharpness", type=float, default=20.0)
    p.add_argument("--out-dir", type=Path,
                   default=base / "debug/correspondence",
                   help="逐帧 CSV 落盘目录, 文件名 <episode>__<模板>.csv")
    return p.parse_args()


if __name__ == "__main__":
    main()
