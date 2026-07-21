"""离线单调性对比：target 模板指纹 cost vs 旧整图 cost（poolcos / patchmse）。

在同一条"真实飞近目标"的 episode 上逐帧计算三种目标函数，
以 pose 得到的真实到目标距离为参照，检验哪种 cost 随接近目标单调下降。
这是闭环收敛的必要条件：cost 若在真实接近过程中都不单调，MPPI 无从优化。

输出：
  - 对比曲线图 png（上：三种 cost 归一化 + 真实距离 vs 帧号；下：cost vs 真实距离）
  - csv（逐帧原始值）
  - 终端打印每种 cost 与真实距离的 Spearman 相关（+1 = 完美单调，越大越好）

用法（服务器/本机有 GPU 时）：
  python src/airsim/check_cost_monotonic.py \
      --episode outputs/datasets/episodes_dataset/episode_0050.h5 \
      --target-template outputs/references/templates/tmpl_goal0050_green.png
旧 cost 的目标图默认取 episode 末帧（与 plan_closed_loop 的 record-goal 约定一致）。
"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plan_closed_loop import Planner

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="三种目标函数在真实接近轨迹上的单调性对比")
    p.add_argument("--episode", type=Path,
                   default=Path(f"{base}/outputs/datasets/episodes_dataset/episode_0050.h5"))
    p.add_argument("--target-template", default=f"{base}/outputs/references/templates/tmpl_goal0050_green.png")
    p.add_argument("--goal-image", default=None,
                   help="旧 cost 的目标图路径；缺省用 episode 末帧")
    p.add_argument("--repo-dir", default=f"{base}/dinov2")
    p.add_argument("--dino-weights", default=f"{base}/weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--out-png", type=Path, default=Path(f"{base}/outputs/evaluations/cost_monotonic/cost_monotonic.png"))
    p.add_argument("--out-csv", type=Path, default=Path(f"{base}/outputs/evaluations/cost_monotonic/cost_monotonic.csv"))
    p.add_argument("--device", default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--yaw-thresh", type=float, default=0.3,
                   help="|yaw_rate| 超过此值视为偏航段：图中标灰，且额外算剔除后的 ρ_facing")
    p.add_argument("--no-fp16", action="store_true")
    # target 指纹 cost 的分量权重（与 check_target_cost.py 同默认）
    p.add_argument("--target-conf-weight", type=float, default=1.0)
    p.add_argument("--target-center-weight", type=float, default=0.8)
    p.add_argument("--target-size-weight", type=float, default=0.4)
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-topk-frac", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    return p.parse_args()


def planner_args(args):
    return SimpleNamespace(
        repo_dir=args.repo_dir,
        dino_weights=args.dino_weights,
        ckpt=args.ckpt,
        no_fp16=args.no_fp16,
        cost_metric="target",
        target_template=args.target_template,
        v_max=2.0, vz_max=0.4, yaw_max=1.0, vx_min=None,
        use_spline=False,
        target_conf_weight=args.target_conf_weight,
        target_center_weight=args.target_center_weight,
        target_size_weight=args.target_size_weight,
        target_softmax_temp=args.target_softmax_temp,
        target_topk_frac=args.target_topk_frac,
        target_mass_thresh=args.target_mass_thresh,
        target_mass_sharpness=args.target_mass_sharpness,
    )


def spearman(a, b):
    """Spearman 秩相关，纯 numpy。"""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    planner = Planner(planner_args(args), device)

    with h5py.File(args.episode, "r") as h:
        rgb = h["rgb"][:]
        pose = h["pose"][:]
        action = h["action"][:]   # (T,4): vx,vy,vz,yaw_rate

    # 旧 cost 的目标 latent：指定图 或 episode 末帧
    if args.goal_image:
        bgr = cv2.imread(args.goal_image)
        if bgr is None:
            raise SystemExit(f"读不到 goal 图: {args.goal_image}")
        goal_rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    else:
        goal_rgb = np.ascontiguousarray(rgb[-1])
    with torch.no_grad():
        z_goal = planner.encode(goal_rgb)

    # 真实到目标距离（pose 前 3 维为位置，目标 = 末帧位置）
    goal_pos = pose[-1, :3]
    idxs = list(range(0, len(rgb), args.stride))
    real_dist = np.linalg.norm(pose[idxs, :3] - goal_pos, axis=1)

    rows = []
    with torch.no_grad():
        for i in idxs:
            z = planner.encode(np.ascontiguousarray(rgb[i]))
            tc = planner.target_components(z)
            zp, gp = z.mean(dim=-2), z_goal.mean(dim=-2)
            c_poolcos = (1 - torch.nn.functional.cosine_similarity(
                zp, gp, dim=-1)).item()
            c_patchmse = ((z - z_goal) ** 2).mean().item()
            rows.append({"frame": i, "real_dist": float(real_dist[len(rows)]),
                         "target": tc["cost"].item(), "poolcos": c_poolcos,
                         "patchmse": c_patchmse,
                         # target cost 的三个分量（加权前原始值），用于定位鼓包来源
                         "conf": 1.0 - tc["topk_mean"].item(),
                         "center_penalty": tc["center_penalty"].item(),
                         "mass": tc["mass"].item()})

    names = ["target", "poolcos", "patchmse"]
    series = {n: np.array([r[n] for r in rows]) for n in names}

    # 偏航段：机头转开时目标出画面中心/视野，cost 上涨是正确行为，
    # 但米制距离不懂朝向，会把这段误判为"非单调"→ 额外算剔除偏航段后的 ρ_facing
    yaw = np.abs(action[idxs, 3])
    facing = yaw < args.yaw_thresh

    # 单调性：cost 应随真实距离减小而减小 → 与 real_dist 的 Spearman 越接近 +1 越好
    print(f"episode={args.episode}  frames={len(rows)}  goal_dist {real_dist[0]:.2f}→{real_dist[-1]:.2f} m")
    print(f"facing 帧 {facing.sum()}/{len(facing)}（|yaw_rate|<{args.yaw_thresh}）")
    stats = {}
    for n in names:
        s = spearman(series[n], real_dist)
        s_face = spearman(series[n][facing], real_dist[facing])
        stats[n] = s
        print(f"  {n:9s} ρ_all = {s:+.3f}   ρ_facing = {s_face:+.3f}   "
              f"first/last = {series[n][0]:.4f}/{series[n][-1]:.4f}")

    # csv
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 画图（标题用英文，避免 DejaVu 缺 CJK 字形的警告）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 12))
    colors = {"target": "tab:green", "poolcos": "tab:orange", "patchmse": "tab:red"}
    frames = [r["frame"] for r in rows]

    # 偏航段标灰：验证 cost 鼓包是否与机头转开重合
    def yaw_spans(ax):
        s = None
        for i, f in enumerate(list(~facing) + [False]):
            if f and s is None:
                s = i
            elif not f and s is not None:
                ax.axvspan(frames[s], frames[min(i, len(frames) - 1)],
                           color="gray", alpha=0.15, lw=0)
                s = None
    for n in names:
        ax1.plot(frames, minmax(series[n]), color=colors[n],
                 label=f"{n} (rho={stats[n]:+.2f})")
    ax1.plot(frames, minmax(real_dist), "k--", alpha=0.6, label="real dist (norm)")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("normalized cost")
    ax1.set_title("cost vs frame (gray = yaw-away segments)")
    yaw_spans(ax1)
    ax1.legend()
    ax1.grid(alpha=0.3)

    order = np.argsort(real_dist)
    for n in names:
        ax2.plot(real_dist[order], minmax(series[n])[order], color=colors[n], label=n)
    ax2.set_xlabel("real distance to goal (m)")
    ax2.set_ylabel("normalized cost")
    ax2.set_title("cost vs real distance (ideal: monotonic)")
    ax2.invert_xaxis()   # 从远到近，向右 = 接近目标
    ax2.legend()
    ax2.grid(alpha=0.3)

    # target cost 分量分解：定位非单调鼓包来自哪一项
    comp_w = {"conf": ("tab:blue", args.target_conf_weight),
              "center_penalty": ("tab:purple", args.target_center_weight),
              "mass": ("tab:brown", -args.target_size_weight)}   # mass 是负贡献
    for cn, (col, wgt) in comp_w.items():
        vals = np.array([r[cn] for r in rows]) * wgt
        ax3.plot(frames, vals, color=col, label=f"{cn} x {wgt:+.2f}")
    ax3.plot(frames, series["target"], color="tab:green", lw=2, alpha=0.6,
             label="target total")
    ax3.set_xlabel("frame")
    ax3.set_ylabel("weighted component")
    ax3.set_title("target cost breakdown (gray = yaw-away segments)")
    yaw_spans(ax3)
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"png={args.out_png}")
    print(f"csv={args.out_csv}")


if __name__ == "__main__":
    main()
