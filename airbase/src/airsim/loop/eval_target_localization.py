#!/usr/bin/env python3
"""比较全模板与 ROI 模板对指定部位的离线定位误差。

读取闭环 ``run.h5`` 中保存的 RGB/位姿，不连接仿真，也不改变控制器。目标部位真值只用于
事后评价：从采集 episode 的 OOB 推导部位锚点，再用与 ``eval_run.py`` 相同的相机模型投影
到画面。两种原型共享在线控制器的 DINO、标准化统计量和 softmax 温度：

* full：模板全部 patch 的平均原型（当前在线方法）；
* roi：只平均 ``--roi x0 y0 x1 y1`` 覆盖的模板 patch。

ROI 坐标归一化到模板宽高的 [0, 1]。默认值覆盖 ``pictures/机头.jpg`` 下部的机鼻区域；
换模板时必须显式重设。输出 CSV、JSON 和 PNG，默认写入
``outputs/evaluations/target_localization/<run名>/``。
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import sys
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]

from eval_run import in_view, part_anchors, quat_to_R
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = _AIRBASE_ROOT


@torch.no_grad()
def encode_all(model, rgb, image_size, mean, std, device, batch_size):
    out = []
    for start in range(0, len(rgb), batch_size):
        x = to_input_tensor(rgb[start:start + batch_size], image_size, mean, std, device)
        out.append(model.forward_features(x)["x_norm_patchtokens"].float().cpu())
    return torch.cat(out, dim=0)


def find_target(dataset_dir, target, target_h5=None):
    candidates = [str(target_h5)] if target_h5 else sorted(
        glob.glob(str(Path(dataset_dir) / "episode_*.h5")))
    for filename in candidates:
        with h5py.File(filename, "r") as f:
            if target is not None and str(f.attrs.get("target_name")) != target:
                continue
            required = ("target_name", "target_center", "target_size", "target_quat")
            if not all(key in f.attrs for key in required):
                continue
            return (str(f.attrs["target_name"]),
                    np.asarray(f.attrs["target_center"], float),
                    np.asarray(f.attrs["target_size"], float),
                    np.asarray(f.attrs["target_quat"], float), filename)
    raise SystemExit(f"找不到带 OOB 真值的 target={target} episode: {dataset_dir}")


def make_proto(template_features, patch_uv, roi=None):
    selected = torch.ones(len(template_features), dtype=torch.bool)
    if roi is not None:
        x0, y0, x1, y1 = roi
        selected = ((patch_uv[:, 0] >= x0) & (patch_uv[:, 0] <= x1)
                    & (patch_uv[:, 1] >= y0) & (patch_uv[:, 1] <= y1))
        if not selected.any():
            raise SystemExit(f"ROI {roi} 没覆盖任何模板 patch")
    proto = F.normalize(F.normalize(template_features[selected], dim=-1).mean(0), dim=0)
    return proto, int(selected.sum())


def response(frame_features, proto, patch_xy, temperature):
    sim = torch.matmul(F.normalize(frame_features, dim=-1), proto)
    weights = torch.softmax(sim / temperature, dim=-1)
    center = torch.matmul(weights, patch_xy)
    argmax = patch_xy[sim.argmax(dim=-1)]
    return center.numpy(), argmax.numpy(), sim.max(dim=-1).values.numpy()


def summarize(name, center_error, peak_error, peak, visible, distances, lock_threshold):
    valid = visible & np.isfinite(center_error)
    locked = valid & (peak >= lock_threshold)
    result = {
        "visible_frames": int(valid.sum()),
        "median_center_error_px": float(np.median(center_error[valid])) if valid.any() else None,
        "p90_center_error_px": float(np.percentile(center_error[valid], 90)) if valid.any() else None,
        "median_peak_error_px": float(np.median(peak_error[valid])) if valid.any() else None,
        "lock_fraction_visible": float(locked.sum() / valid.sum()) if valid.any() else None,
        "median_center_error_px_when_locked": (
            float(np.median(center_error[locked])) if locked.any() else None),
    }
    print(f"{name:>4s}: center median={result['median_center_error_px']:.1f}px "
          f"p90={result['p90_center_error_px']:.1f}px  "
          f"peak median={result['median_peak_error_px']:.1f}px  "
          f"lock={result['lock_fraction_visible']:.1%}")
    for lo, hi in ((0, 15), (15, 20), (20, 30), (30, 50), (50, np.inf)):
        mask = valid & (distances >= lo) & (distances < hi)
        if mask.any():
            label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f">={lo:g}m"
            result.setdefault("distance_bins", {})[label] = {
                "frames": int(mask.sum()),
                "median_center_error_px": float(np.median(center_error[mask])),
                "lock_fraction": float((peak[mask] >= lock_threshold).mean()),
            }
    return result


def main():
    args = parse_args()
    run_h5 = args.run if args.run.suffix == ".h5" else args.run / "run.h5"
    if not run_h5.exists():
        raise SystemExit(f"找不到 run.h5: {run_h5}")
    with h5py.File(run_h5, "r") as f:
        rgb = f["rgb"][::args.stride]
        pose = f["pose"][::args.stride].astype(float)
        steps = np.arange(len(f["rgb"]))[::args.stride]
    if pose.shape[1] != 4:
        raise SystemExit(f"当前只支持闭环 n,e,d,yaw 位姿，实际 shape={pose.shape}")
    with h5py.File(args.stats_episode, "r") as f:
        stats_rgb = f["rgb"][:]

    target_name, target_center, target_size, target_quat, target_source = find_target(
        args.dataset_dir, args.target, args.target_h5)
    anchors, _ = part_anchors(target_center, target_size, quat_to_R(target_quat))
    if args.part not in anchors:
        raise SystemExit(f"--part 只能是 {list(anchors)}")
    anchor = anchors[args.part]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, frames={len(rgb)}, target={target_name}/{args.part}")
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model = load_model("dinov2_vits14", device, weights=str(args.dino_weights),
                       repo_dir=str(args.repo_dir))
    stats_features = encode_all(model, stats_rgb, args.image_size, mean, std, device,
                                args.batch_size)
    frame_features = encode_all(model, rgb, args.image_size, mean, std, device,
                                args.batch_size)
    template_bgr = cv2.imread(str(args.template))
    if template_bgr is None:
        raise SystemExit(f"读不到模板: {args.template}")
    template_rgb = np.ascontiguousarray(template_bgr[:, :, ::-1])[None]
    template_features = encode_all(model, template_rgb, args.image_size, mean, std, device,
                                   args.batch_size)[0]

    flat = stats_features.reshape(-1, stats_features.shape[-1])
    mean_features = flat.mean(0)
    std_features = flat.std(0).clamp_min(1e-6)
    frame_features = (frame_features - mean_features) / std_features
    template_features = (template_features - mean_features) / std_features

    patch_count = frame_features.shape[1]
    grid = int(round(patch_count ** 0.5))
    if grid * grid != patch_count:
        raise SystemExit(f"patch 数不是方形网格: {patch_count}")
    axis_xy = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(axis_xy, axis_xy, indexing="ij")
    patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    axis_uv = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    vv, uu = torch.meshgrid(axis_uv, axis_uv, indexing="ij")
    patch_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)

    full_proto, full_patches = make_proto(template_features, patch_uv)
    roi_proto, roi_patches = make_proto(template_features, patch_uv, args.roi)
    methods = {}
    for name, proto in (("full", full_proto), ("roi", roi_proto)):
        center, argmax, peak = response(frame_features, proto, patch_xy,
                                        args.target_softmax_temp)
        methods[name] = {"center": center, "argmax": argmax, "peak": peak}

    gt = np.full((len(pose), 2), np.nan, dtype=float)
    visible = np.zeros(len(pose), dtype=bool)
    forward = np.full(len(pose), np.nan, dtype=float)
    distances = np.linalg.norm(pose[:, :3] - anchor, axis=1)
    for index, row in enumerate(pose):
        visible[index], u, v, forward[index] = in_view(anchor, row[:3], row[3])
        if u is not None:
            gt[index] = (u, v)

    pixels_per_normalized_unit = (args.image_size - 1) / 2.0
    summaries = {}
    for name, values in methods.items():
        values["center_error"] = np.linalg.norm(values["center"] - gt, axis=1) \
            * pixels_per_normalized_unit
        values["peak_error"] = np.linalg.norm(values["argmax"] - gt, axis=1) \
            * pixels_per_normalized_unit
        summaries[name] = summarize(name, values["center_error"], values["peak_error"],
                                    values["peak"], visible, distances, args.lock_threshold)

    out_dir = args.out_dir or (
        PROJECT_ROOT / "outputs/evaluations/target_localization" / run_h5.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "localization.csv"
    columns = [steps, distances, visible.astype(int), gt[:, 0], gt[:, 1]]
    header = ["step", "anchor_dist", "visible", "gt_x", "gt_y"]
    for name, values in methods.items():
        columns.extend([values["center"][:, 0], values["center"][:, 1],
                        values["argmax"][:, 0], values["argmax"][:, 1],
                        values["peak"], values["center_error"], values["peak_error"]])
        header.extend([f"{name}_cx", f"{name}_cy", f"{name}_peak_x", f"{name}_peak_y",
                       f"{name}_score", f"{name}_center_error_px", f"{name}_peak_error_px"])
    np.savetxt(csv_path, np.column_stack(columns), delimiter=",", header=",".join(header),
               comments="", fmt="%.6g")

    report = {
        "run": str(run_h5), "template": str(args.template),
        "stats_episode": str(args.stats_episode), "target_source": str(target_source),
        "target": target_name, "part": args.part, "anchor": anchor.tolist(),
        "roi": list(args.roi), "patches": {"full": full_patches, "roi": roi_patches},
        "stride": args.stride, "lock_threshold": args.lock_threshold,
        "methods": summaries,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, values in methods.items():
            mask = visible & np.isfinite(values["center_error"])
            axes[0].scatter(distances[mask], values["center_error"][mask], s=7, alpha=.45,
                            label=name)
            axes[1].scatter(distances[mask], values["peak"][mask], s=7, alpha=.45,
                            label=name)
        axes[0].set(xlabel="true distance to part anchor (m)", ylabel="center error (px)")
        axes[1].set(xlabel="true distance to part anchor (m)", ylabel="response peak")
        for axis in axes:
            axis.invert_xaxis(); axis.grid(alpha=.25); axis.legend()
        fig.tight_layout(); fig.savefig(out_dir / "localization.png", dpi=140); plt.close(fig)
    except Exception as error:
        print(f"跳过 PNG: {error}")

    print(f"full patches={full_patches}, roi patches={roi_patches}, roi={args.roi}")
    print(f"输出: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="包含 run.h5 的运行目录，或 run.h5")
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "pictures/机头.jpg")
    parser.add_argument("--stats-episode", type=Path, default=
                        PROJECT_ROOT / "outputs/recordings/approach/airbase_tgt1_50m.h5")
    parser.add_argument("--dataset-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/datasets/wm300")
    parser.add_argument("--target-h5", type=Path, default=None)
    parser.add_argument("--target", default="StaticMeshActor_22")
    parser.add_argument("--part", default="nose")
    parser.add_argument("--roi", type=float, nargs=4, default=(0.20, 0.68, 0.80, 1.00),
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--lock-threshold", type=float, default=0.12)
    parser.add_argument("--target-softmax-temp", type=float, default=0.08)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--repo-dir", type=Path, default=PROJECT_ROOT / "dinov2")
    parser.add_argument("--dino-weights", type=Path,
                        default=PROJECT_ROOT / "weights/dinov2_vits14_pretrain.pth")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        parser.error("--roi 必须满足 0 ≤ x0 < x1 ≤ 1 且 0 ≤ y0 < y1 ≤ 1")
    if args.stride < 1 or args.batch_size < 1:
        parser.error("--stride 和 --batch-size 必须 ≥ 1")
    return args


if __name__ == "__main__":
    main()
