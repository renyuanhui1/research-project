"""评估「到目标部位距离」回归器（airbase `StaticMeshActor_22`）。

这一版是**数据够不够的探针**，所以核心产物不是总体 MAE，而是：
  1. 分距离带误差表（逐带 n / n_vis / MAE / bias / ρ），并同时给「全帧」与「目标可见帧」两个口径
  2. train vs val 的同带对比 —— 用来分解"过拟合"还是"数据不足"
  3. 与 peak / mass / 深度 / const 基线的并排对比

标定口径的偏离（与 check_target_signal.py 不同，必须知道）：
  该脚本用**单条 episode 自己的**帧统计做标准化；这里统一用 **train 切分**的全局统计，
  与模型看到的标准化空间一致。否则基线与模型不可比，且前者对逐帧指标是同 episode 内的泄漏。

用法：
  python eval_part_dist.py --ckpt weights/part_dist_22.pt
"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch, h5py
import sys
from pathlib import Path      # 引导块自足
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
from train_part_dist import (PARTS, PART_ALIASES, BANDS, find_pairs, part_labels,
                             PartDistNet)
from train_value import pool_spatial
from check_target_signal import spearman, fingerprint_cost
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD

BASE = str(_AIRBASE_ROOT)
FP_ARGS = dict(target_topk_frac=0.08, target_softmax_temp=0.08,
               target_mass_thresh=0.35, target_mass_sharpness=20.0,
               target_conf_weight=1.0, target_center_weight=0.8, target_size_weight=0.4)


def band_of(d, bands=BANDS):
    for i, (lo, hi) in enumerate(zip(bands[:-1], bands[1:])):
        if lo <= d < hi:
            return f"{lo}-{hi}"
    return None


def quantile_lookup(tr_score, tr_dist, te_score, nbins=20):
    """把非米制基线校准成米：train 上按分数分位切箱，取箱内距离中位数当查表值。"""
    tr_score = np.asarray(tr_score, float); tr_dist = np.asarray(tr_dist, float)
    ok = np.isfinite(tr_score) & np.isfinite(tr_dist)    # 一个 NaN 就会污染全部分位数
    tr_score, tr_dist = tr_score[ok], tr_dist[ok]
    if len(tr_score) < nbins:
        return np.full(len(te_score), np.nan)
    qs = np.quantile(tr_score, np.linspace(0, 1, nbins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    centers = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (tr_score >= qs[i]) & (tr_score < qs[i + 1])
        if m.sum():
            centers[i] = np.median(tr_dist[m])
    good = ~np.isnan(centers)
    if good.sum() < 2:
        return np.full(len(te_score), np.nan)
    centers = np.interp(np.arange(nbins), np.nonzero(good)[0], centers[good])
    idx = np.clip(np.searchsorted(qs, te_score, side="right") - 1, 0, nbins - 1)
    return centers[idx]


def metrics(pred, true):
    """→ dict(MAE, MedAE, bias, RMSE, rho)。pred 可以是 None（无效帧）。"""
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() == 0:
        return dict(n=int(m.sum()), MAE=np.nan, MedAE=np.nan, bias=np.nan,
                    RMSE=np.nan, rho=np.nan)
    d = pred[m] - true[m]
    return dict(n=int(m.sum()), MAE=float(np.abs(d).mean()), MedAE=float(np.median(np.abs(d))),
                bias=float(d.mean()), RMSE=float(np.sqrt((d ** 2).mean())),
                rho=spearman(pred[m], true[m]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=f"{BASE}/weights/part_dist_22.pt")
    p.add_argument("--latent-root", default=f"{BASE}/outputs/features/part_dist")
    p.add_argument("--dataset-root", default=f"{BASE}/outputs/datasets")
    p.add_argument("--out-dir", default=f"{BASE}/outputs/evaluations/part_dist")
    p.add_argument("--template", default=f"{BASE}/pictures/机头.jpg", help="peak/mass 基线用的模板")
    p.add_argument("--no-fingerprint", action="store_true", help="跳过 peak/mass 基线")
    p.add_argument("--no-depth", action="store_true", help="跳过深度基线")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PartDistNet(ck["d_in"], ck["h"], ck["n_out"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    pool_grid = ck["pool_grid"]
    z_mean = torch.tensor(np.asarray(ck["z_mean"]), device=dev)
    z_std = torch.tensor(np.asarray(ck["z_std"]), device=dev)
    y_mean = np.asarray(ck["y_mean"]); y_std = np.asarray(ck["y_std"])
    tr_tags, va_tags = set(ck["train_pairs"]), set(ck["val_pairs"])   # "子目录/文件名"
    print(f"ckpt {args.ckpt}  最优 epoch {ck['epoch']}  val MAE={ck['val_mae_m']:.2f} m"
          f"  null={ck['null_mae_m']:.2f} m")
    print(f"train {len(tr_tags)} 条 / val {len(va_tags)} 条")

    pairs = find_pairs(Path(args.latent_root), Path(args.dataset_root))
    for sub, src, lf, ep in pairs:
        tag = f"{sub}/{src}"
        if tag not in tr_tags and tag not in va_tags:
            raise SystemExit(f"{tag} 不在 ckpt 记录的切分里 —— ckpt 与数据不匹配")

    # ---- 第一遍：train 切分的 patch 级统计量（供 peak/mass 基线标准化）----
    use_fp = not args.no_fingerprint
    if use_fp:
        s = np.zeros(384); ss = np.zeros(384); n = 0
        for sub, src, lf, ep in pairs:
            if f"{sub}/{src}" not in tr_tags:
                continue
            with h5py.File(lf, "r") as f:
                z = f["z"][:].astype(np.float32).reshape(-1, 384)
            s += z.sum(0); ss += (z ** 2).sum(0); n += len(z)
        # 指纹这条链整体放 CPU：fingerprint_cost 直接对 tensor 调 .numpy()，不接受 cuda
        pz_m = torch.tensor(s / n, dtype=torch.float32)
        pz_s = torch.tensor(np.sqrt(np.maximum(ss / n - (s / n) ** 2, 1e-12)), dtype=torch.float32)
        print(f"patch 统计量来自 train 的 {n} 帧")
        # 模板 → proto：patch 特征 L2 归一→均值→再 L2 归一（与 check_target_signal 同法）
        import cv2
        import torch.nn.functional as F
        tbgr = cv2.imread(str(args.template))
        if tbgr is None:
            raise SystemExit(f"读不到模板: {args.template}")
        dino = load_model("dinov2_vits14", dev, weights=f"{BASE}/weights/dinov2_vits14_pretrain.pth",
                          repo_dir=f"{BASE}/dinov2")
        mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        with torch.no_grad():
            trgb = np.ascontiguousarray(tbgr[:, :, ::-1])[None]
            zt = dino.forward_features(to_input_tensor(trgb, 224, mean, std, dev))["x_norm_patchtokens"][0]
            zt = ((zt.cpu() - pz_m) / pz_s)
            PROTO = F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0)
        P = zt.shape[0]; g = int(round(P ** 0.5))
        xs = torch.linspace(-1.0, 1.0, g, dtype=torch.float32)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        PATCH_XY = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        del dino

    # ---- 逐 episode：预测、真值、可观测性、基线 ----
    rec = []                                    # 每帧一条 dict
    fp_peak, fp_mass = [], []
    for k, (sub, src, lf, ep) in enumerate(pairs):
        pos, Y, vis = part_labels(ep)
        with h5py.File(lf, "r") as f:
            z16 = f["z"][:]
        z = torch.from_numpy(z16.astype(np.float32))
        with torch.no_grad():
            pooled = pool_spatial(z, pool_grid).to(dev)
            pred = (model((pooled - z_mean) / z_std) * torch.tensor(y_std, device=dev)
                    + torch.tensor(y_mean, device=dev)).cpu().numpy()
        if use_fp:
            zn = torch.nn.functional.normalize((z - pz_m) / pz_s, dim=-1)
            comp = fingerprint_cost(zn, PROTO, PATCH_XY, argparse.Namespace(**FP_ARGS))
            fp_peak.append(comp["peak"]); fp_mass.append(comp["mass"])
        # 深度基线：目标掩膜内有效深度中位数（tolerance=0，_24 的颜色与 _22 极近）
        dep = np.full(len(Y), np.nan)
        if not args.no_depth:
            with h5py.File(ep, "r") as f:
                depth = f["depth"][:]; seg = f["seg"][:]
            from servo_closed_loop import make_target_mask, SEG_COLORS_BGR
            for t in range(len(Y)):
                dm = depth[t]
                if dm.ndim == 3:
                    dm = dm[..., 0]
                mask = make_target_mask(seg[t], SEG_COLORS_BGR["StaticMeshActor_22"], tolerance=0)
                v = dm[mask & (dm > 0.1) & (dm < 60000.0)]
                if v.size >= 3:
                    dep[t] = float(np.median(v))
        for t in range(len(Y)):
            rec.append(dict(split="train" if f"{sub}/{src}" in tr_tags else "val",
                            tag=f"{sub}/{src}", frame=t, pos=pos[t].tolist(), Y=Y[t], vis=vis[t],
                            pred=pred[t], depth=dep[t]))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(pairs)}]")

    if use_fp:
        peak = np.concatenate(fp_peak); mass = np.concatenate(fp_mass)
        for i, r in enumerate(rec):
            r["peak"] = float(peak[i]); r["mass"] = float(mass[i])
    for r in rec:
        r["nose"] = float(r["Y"][0])

    tr = [r for r in rec if r["split"] == "train"]
    va = [r for r in rec if r["split"] == "val"]
    print(f"\n帧数 train={len(tr)} val={len(va)}")
    if not va:
        raise SystemExit("val 为空")

    # ---- 基线校准（只看 val；查表值从 train 拟合）----
    def calib(key):
        if key not in tr[0]:
            return None
        tr_s = np.array([r[key] for r in tr]); tr_d = np.array([r["nose"] for r in tr])
        va_s = np.array([r[key] for r in va])
        return quantile_lookup(tr_s, tr_d, va_s)

    base_val = {"const": np.full(len(va), np.median([r["nose"] for r in tr]))}
    for key in ("peak", "mass", "depth"):
        c = calib(key)
        if c is not None:
            base_val[key] = c
    base_val["pred"] = np.array([r["pred"][0] for r in va])
    base_val["true"] = np.array([r["nose"] for r in va])
    base_val["vis"] = np.array([r["vis"][0] for r in va])
    base_val["peak_raw"] = np.array([r.get("peak", np.nan) for r in va])
    base_val["mass_raw"] = np.array([r.get("mass", np.nan) for r in va])
    base_val["depth_raw"] = np.array([r.get("depth", np.nan) for r in va])

    # ---- 主表：nose 分带，模型 vs 全部基线 ----
    band_tbl = []
    print("\n" + "=" * 96)
    print("=== 到「nose」锚点的分距离带误差（米）— val 切分 ===")
    print(f"{'band':>7} {'n':>5} {'n_vis':>6} | {'模型':>7} {'MedAE':>7} {'bias':>7} {'rho':>7} | "
          f"{'const':>7} {'peak*':>7} {'mass*':>7} {'depth*':>7} {'cov':>5}")
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        b = f"{lo}-{hi}"
        m = (base_val["true"] >= lo) & (base_val["true"] < hi)
        n = int(m.sum()); nv = int((m & base_val["vis"]).sum())
        if n == 0:
            continue
        row = dict(band=b, n=n, n_vis=nv)
        cells = []
        for key in ("pred", "const", "peak", "mass", "depth"):
            if key not in base_val:
                cells.append(None); continue
            mm = metrics(base_val[key][m], base_val["true"][m])
            row[key + "_MAE"] = mm["MAE"]; row[key + "_rho"] = mm["rho"]
            if key == "pred":
                row.update(MedAE=mm["MedAE"], bias=mm["bias"], RMSE=mm["RMSE"])
            cells.append(mm)
        cov = np.isfinite(base_val["depth"][m]).mean() if "depth" in base_val else np.nan
        row["depth_cov"] = float(cov)
        band_tbl.append(row)
        tag = "❌" if n < 20 else ("⚠️" if n < 100 else "  ")
        def fmt(c):
            return f"{c['MAE']:7.2f}" if c and np.isfinite(c["MAE"]) else "   ~~~~"
        print(f"{b:>7} {n:>5} {nv:>6}{tag}| {fmt(cells[0])} {row['MedAE']:7.2f} "
              f"{row['bias']:+7.2f} {row['pred_rho']:+7.2f} | "
              + " ".join(fmt(c) for c in cells[1:]) + f" {cov:5.2f}")

    def agg(tbl, key="pred_MAE"):
        v = [(r["n"], r[key]) for r in tbl if r.get(key) is not None and np.isfinite(r[key])]
        if not v:
            return np.nan, np.nan
        nn = np.array([a for a, _ in v]); vv = np.array([b for _, b in v])
        return float((nn * vv).sum() / nn.sum()), float(vv.mean())

    w, mac = agg(band_tbl)
    print("-" * 96)
    print(f"val 总体 MAE（按帧数加权） {w:6.2f} m     宏平均（每带等权） {mac:6.2f} m")
    print(f"基线：const {agg(band_tbl,'const_MAE')[0]:6.2f} m   "
          f"peak {agg(band_tbl,'peak_MAE')[0] if 'peak_MAE' in band_tbl[0] else float('nan'):6.2f} m   "
          f"mass {agg(band_tbl,'mass_MAE')[0] if 'mass_MAE' in band_tbl[0] else float('nan'):6.2f} m   "
          f"depth {agg(band_tbl,'depth_MAE')[0] if 'depth_MAE' in band_tbl[0] else float('nan'):6.2f} m")
    print(f"可观测性：val 全语料 nose 在画面内 {base_val['vis'].mean()*100:.1f}%")
    vis = base_val["vis"]; pr = base_val["pred"]; tt = base_val["true"]
    m_vis = metrics(pr[vis], tt[vis]); m_hid = metrics(pr[~vis], tt[~vis])
    print(f"           三个口径：全帧 {metrics(pr, tt)['MAE']:.2f} m ｜ "
          f"目标可见帧 {m_vis['MAE']:.2f} m (n={m_vis['n']}) ｜ "
          f"目标不可见帧 {m_hid['MAE']:.2f} m (n={m_hid['n']})")
    print("           （可见/不可见若接近，说明模型更多靠地景、高度等旁路线索而非目标本身）")

    # ---- 过拟合 vs 数据不足：train / val 同带对比 ----
    print("\n" + "=" * 96)
    print("=== 分解：train vs val 同带 MAE（比值大=过拟合；两边都高=数据/表征不足）===")
    print(f"{'band':>7} {'train n':>8} {'trainMAE':>9} {'val n':>6} {'valMAE':>8} {'比值':>7}"
          f" {'真值均值':>9} {'预测均值':>9}")
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        tm = [r for r in tr if lo <= r["nose"] < hi]
        vm = [r for r in va if lo <= r["nose"] < hi]
        if not tm or not vm:
            continue
        tp = np.array([r["pred"][0] for r in tm]); td = np.array([r["nose"] for r in tm])
        vp = np.array([r["pred"][0] for r in vm]); vd = np.array([r["nose"] for r in vm])
        print(f"{lo:>3}-{hi:<3} {len(tm):>8} {np.abs(tp-td).mean():>9.2f} {len(vm):>6} "
              f"{np.abs(vp-vd).mean():>8.2f} {np.abs(vp-vd).mean()/max(np.abs(tp-td).mean(),1e-6):>7.1f}"
              f" {vd.mean():>9.2f} {vp.mean():>9.2f}")

    # ---- 逐 episode 单调性（带门控）----
    print("\n" + "=" * 96)
    print("=== 逐 episode 单调性（Spearman；只在 rho_true>=0.8 的 episode 上判定模型）===")
    mono = []
    for tag in sorted({r["tag"] for r in va}):
        rows = sorted([r for r in va if r["tag"] == tag], key=lambda r: r["frame"])
        d = np.array([r["nose"] for r in rows]); t = np.arange(len(rows))
        pr = np.array([r["pred"][0] for r in rows])
        # rho_true 用 -d：接近过程本身单调时它 → +1（直接用 d 会恒为负，门控永远不触发）
        row = dict(tag=tag, n=len(rows), d_start=d[0], d_end=d[-1], d_min=d.min(),
                   rho_true=spearman(-d, t), rho_model=spearman(pr, d))
        for key in ("peak", "mass"):
            if key in rows[0]:
                row["rho_" + key] = spearman(np.array([r[key] for r in rows]), d)
        if "depth" in rows[0]:
            dep = np.array([r["depth"] for r in rows])
            row["rho_depth"] = spearman(dep, d) if np.isfinite(dep).sum() > 3 else np.nan
        mono.append(row)
        print(f"  {tag:28s} rho_true={row['rho_true']:+.2f}  rho_model={row['rho_model']:+.2f}  "
              f"peak={row.get('rho_peak', float('nan')):+.2f}  mass={row.get('rho_mass', float('nan')):+.2f}  "
              f"{row['d_start']:.0f}→{row['d_end']:.0f} (min {row['d_min']:.1f})")
    gated = [r for r in mono if r["rho_true"] >= 0.8]
    rate = np.mean([r["rho_model"] >= 0.8 for r in gated]) if gated else np.nan
    print(f"\n  rho_true>=0.8 的 episode：{len(gated)}/{len(mono)} 条；"
          f"其中模型 rho_model>=0.8 占比 {rate*100:.0f}%" if gated else "  没有 rho_true>=0.8 的 episode")
    for key in ("peak", "mass"):
        if gated and ("rho_" + key) in gated[0]:
            r2 = np.mean([r["rho_" + key] >= 0.8 for r in gated])
            print(f"  同口径 {key}: {r2*100:.0f}%")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ctr = [(lo + hi) / 2 for lo, hi in zip(BANDS[:-1], BANDS[1:]) if lo < 70]
        keys = [("pred_MAE", "model"), ("const_MAE", "const"), ("peak_MAE", "peak*"),
                ("mass_MAE", "mass*"), ("depth_MAE", "depth*")]
        rows = [r for r in band_tbl if r["n"] >= 20]      # 太小的带不出线
        for k, lab in keys:
            xs = [ (r["band"].split("-")[0], r) for r in rows if r.get(k) is not None ]
            if not xs:
                continue
            ax[0].plot([int(b) for b, _ in xs], [r[k] for _, r in xs], marker="o", label=lab)
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("true distance to nose (m)"); ax[0].set_ylabel("MAE (m)")
        ax[0].set_title("MAE vs distance (val; * = calibrated baseline)")
        ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
        lbl = [r["band"] for r in band_tbl]
        x = np.arange(len(lbl))
        ax[1].bar(x - 0.2, [r["n"] for r in band_tbl], 0.4, label="n (all frames)")
        ax[1].bar(x + 0.2, [r["n_vis"] for r in band_tbl], 0.4, label="n_vis (target in view)")
        ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, rotation=30)
        ax[1].set_ylabel("frames"); ax[1].set_title("Data coverage by band (val)")
        ax[1].legend()
        fig.tight_layout(); fig.savefig(out / "err_vs_dist.png", dpi=120)
        print(f"\n图 -> {out}/err_vs_dist.png")
    except Exception as e:
        print(f"（画图跳过：{e}）")

    # ---- 落盘 ----
    import csv
    with open(out / "band_table.csv", "w", newline="") as f:
        cols = ["band", "n", "n_vis", "MedAE", "bias", "RMSE", "depth_cov"] + \
               [f"{k}_{s}" for k in ("pred", "const", "peak", "mass", "depth") for s in ("MAE", "rho")]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in band_tbl:
            wr.writerow(r)
    with open(out / "frame_pred.csv", "w", newline="") as f:
        cols = ["split", "tag", "frame", "nose", "pred_nose", "vis_nose"] + \
               [f"pred_{k}" for k in PARTS[1:]] + \
               [f"true_{k}" for k in PARTS[1:]] + ["peak", "mass", "depth"]
        wr = csv.writer(f); wr.writerow(cols)
        for r in rec:
            wr.writerow([r["split"], r["tag"], r["frame"], f"{r['nose']:.4f}",
                         f"{r['pred'][0]:.4f}", int(r["vis"][0])] +
                        [f"{v:.4f}" for v in r["pred"][1:]] +
                        [f"{v:.4f}" for v in r["Y"][1:]] +
                        [f"{r.get('peak', float('nan')):.5f}", f"{r.get('mass', float('nan')):.5f}",
                         f"{r.get('depth', float('nan')):.3f}"])
    with open(out / "episode_monotonic.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(mono[0].keys()), extrasaction="ignore")
        wr.writeheader(); wr.writerows(mono)
    summary = dict(ckpt=args.ckpt, best_epoch=ck["epoch"], ckpt_val_mae_m=ck["val_mae_m"],
                   null_mae_m=ck["null_mae_m"], val_n=len(va), train_n=len(tr),
                   weighted_MAE_m=w, macro_MAE_m=mac,
                   visible_frac=float(base_val["vis"].mean()),
                   MAE_on_visible=metrics(base_val["pred"][base_val["vis"]],
                                          base_val["true"][base_val["vis"]])["MAE"],
                   mono_gated_n=len(gated), mono_gated_total=len(mono),
                   mono_pass_rate=float(rate) if gated else None, band_table=band_tbl)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n产物 -> {out}/  (band_table.csv, frame_pred.csv, episode_monotonic.csv, summary.json)")


if __name__ == "__main__":
    main()
