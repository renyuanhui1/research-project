"""把 plan_closed_loop --viz-dump 落盘的 step_*.npz 里的真实 pose 画成一条 3D 轨迹。

纯 numpy + matplotlib，不吃 GPU。pose 是 NED(北,东,下)，这里转成 ENU(x=东,y=北,z=上)
与 rviz 同款朝向。轴标签用英文，免得 matplotlib 缺中文字体显示成方块。

用法：
  python src/mpc/plot_traj_dump.py --dump-dir outputs/runs/mppi/handoff03 --out traj.png
  # 本地看服务器的 run：--dump-dir ~/mnt/server_runs/handoff03
  # 加 --side 同时出高度侧视剖面(北 vs 高度)，文件名自动加 _side
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_traj(dump_dir):
    files = sorted(Path(dump_dir).glob("step_*.npz"))
    if not files:
        raise SystemExit(f"没有 step_*.npz: {dump_dir}")
    pts = []
    for f in files:
        n, e, d = np.load(f)["pose"][:3]              # NED
        pts.append((float(e), float(n), float(-d)))   # ENU: x=东, y=北, z=上
    return np.asarray(pts)                            # (T,3)


def plot_side(p, out):
    """高度侧视剖面：横=北, 竖=高度, 等比例, 直观看升/降。"""
    north, up = p[:, 1], p[:, 2]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(north, up, color="tab:blue", lw=2)
    ax.scatter(north[0],  up[0],  color="green", s=60, label=f"start (h={up[0]:.1f}m)")
    ax.scatter(north[-1], up[-1], color="red", marker="x", s=60, label=f"end (h={up[-1]:.1f}m)")
    ax.set_xlabel("North (m)")
    ax.set_ylabel("Height Up (m)")
    ax.set_title(f"side profile: height {up[0]-up[-1]:+.2f} m over {north[-1]-north[0]:.1f} m north")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"已保存侧视: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", default="traj.png")
    ap.add_argument("--side", action="store_true", help="同时出高度侧视剖面(文件名加 _side)")
    args = ap.parse_args()

    p = load_traj(args.dump_dir)
    total = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="tab:blue", lw=2, label="trajectory")
    ax.scatter(*p[0],  color="green", s=60, label="start")
    ax.scatter(*p[-1], color="red",   s=60, marker="x", label="end")
    ax.set_xlabel("X East (m)")
    ax.set_ylabel("Y North (m)")
    ax.set_zlabel("Z Up (m)")
    # 等比例：三轴用同一世界尺度(1m 在各轴视觉长度一致)，与 rviz 一致，
    # 否则 matplotlib 会把最小的那个维度拉伸撑满，夸张成"歪扭"。
    c = p.mean(axis=0)
    r = float((p.max(axis=0) - p.min(axis=0)).max()) / 2 + 1e-3
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    ax.set_title(f"actual trajectory  ({len(p)} steps, {total:.1f} m)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已保存: {args.out}  ({len(p)} 点, 累计 {total:.1f} m)")

    if args.side:
        stem = Path(args.out)
        plot_side(p, str(stem.with_name(stem.stem + "_side" + stem.suffix)))


if __name__ == "__main__":
    main()
