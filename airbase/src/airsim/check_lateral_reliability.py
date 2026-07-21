"""离线体检：世界模型对"横向/垂直"运动的预测是否可靠。

回答一个决定路线的问题：闭环里横向乱漂（同参数一次往左一次往右），
到底是①代价权重没调好（可救，调参）还是②世界模型对横向预测本身是噪声
（不可救，只能补数据重训）。

方法（不飞、不改控制逻辑）：
  1. 扫描已录的真实机载帧（viz_run*/*.npz 里的 rgb），用指纹算 center_x 定位环在左/右；
  2. 选出环明显偏一侧的帧（|center_x|>--offset）；
  3. 每帧让世界模型 rollout "往环侧移(纠偏)" vs "往反侧移"，比较终端 target cost；
  4. 若"纠偏动作"稳定更优（命中率高）→ 横向可靠，是调参问题；
     若命中率≈50% → 横向=噪声，需补横移数据重训。
  同法测垂直（center_y 上/下）。

用法：
  python src/airsim/check_lateral_reliability.py --frames-glob 'outputs/runs/mppi/run*/step_*.npz'
"""

import argparse
import glob
import sys
from types import SimpleNamespace

import numpy as np
import torch

from plan_closed_loop import Planner

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="世界模型横向/垂直预测可靠性体检")
    p.add_argument("--frames-glob", default=f"{base}/outputs/runs/mppi/run*/step_*.npz",
                   help="真实机载帧来源（npz 里含 rgb）")
    p.add_argument("--target-template", default=f"{base}/outputs/references/templates/tmpl_ring_full_onboard.png")
    p.add_argument("--repo-dir", default=f"{base}/dinov2")
    p.add_argument("--dino-weights", default=f"{base}/weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--lat-speed", type=float, default=0.6, help="测试用横向/垂直速度幅值")
    p.add_argument("--fwd", type=float, default=0.3, help="测试时叠加的前进速度")
    p.add_argument("--offset", type=float, default=0.15, help="|center| 超过多少算明显偏一侧")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--device", default=None)
    return p.parse_args()


def planner_args(a):
    return SimpleNamespace(
        repo_dir=a.repo_dir, dino_weights=a.dino_weights, ckpt=a.ckpt, no_fp16=False,
        cost_metric="target", target_template=a.target_template,
        v_max=2.0, vz_max=1.0, yaw_max=1.0, vx_min=None, use_spline=False,
        target_conf_weight=0.5, target_center_weight=1.0, target_size_weight=0.8,
        target_softmax_temp=0.08, target_topk_frac=0.08,
        target_mass_thresh=0.2, target_mass_sharpness=10.0)


def rollout_cost(pl, z0, a4, H):
    acts = torch.tensor(np.tile(a4, (H, 1)), dtype=torch.float32, device=pl.dev)
    z = z0.unsqueeze(0)
    with torch.no_grad(), pl._autocast():
        for k in range(H):
            z = pl.model(z, acts[k:k + 1])
    return pl.target_cost(z[0].float()).item()


def main():
    a = parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pl = Planner(planner_args(a), dev)

    files = sorted(glob.glob(a.frames_glob))
    if not files:
        sys.exit(f"没找到帧: {a.frames_glob}")

    lat_hits, lat_tot = 0, 0     # 横向：环偏侧帧
    ver_hits, ver_tot = 0, 0     # 垂直：环偏上下帧
    lat_margins, ver_margins = [], []

    for f in files[:: max(1, len(files) // a.max_frames)]:
        rgb = np.load(f)["rgb"]
        z0 = pl.encode(np.ascontiguousarray(rgb))
        c = pl.target_components(z0)
        cx, cy = c["center"][0].item(), c["center"][1].item()

        # 横向：环偏一侧时，纠偏动作 = vy 与 center_x 同号（前面实测：环左→west(-vy)纠偏）
        if abs(cx) > a.offset:
            toward = rollout_cost(pl, z0, [a.fwd,  np.sign(cx) * a.lat_speed, 0, 0], a.horizon)
            away = rollout_cost(pl, z0, [a.fwd, -np.sign(cx) * a.lat_speed, 0, 0], a.horizon)
            lat_hits += int(toward < away)
            lat_margins.append(away - toward)   # 正=纠偏更优（对）
            lat_tot += 1

        # 垂直：center_y 同理，纠偏 vz 与 center_y 同号
        if abs(cy) > a.offset:
            toward = rollout_cost(pl, z0, [a.fwd, 0, np.sign(cy) * a.lat_speed, 0], a.horizon)
            away = rollout_cost(pl, z0, [a.fwd, 0, -np.sign(cy) * a.lat_speed, 0], a.horizon)
            ver_hits += int(toward < away)
            ver_margins.append(away - toward)
            ver_tot += 1

    def report(name, hits, tot, margins):
        if tot == 0:
            print(f"{name}: 无偏离帧"); return
        rate = hits / tot * 100
        m = np.array(margins)
        verdict = ("可靠(是调参问题)" if rate >= 75 else
                   "噪声(需补数据重训)" if rate <= 60 else "弱信号(边界)")
        print(f"{name}: 纠偏命中 {hits}/{tot} = {rate:.0f}%  "
              f"margin 均值{m.mean():+.4f} 正比例{(m>0).mean()*100:.0f}%  → {verdict}")

    print(f"帧源={a.frames_glob}  评估帧数≈{len(files[:: max(1, len(files)//a.max_frames)])}  "
          f"horizon={a.horizon}")
    print("（纠偏命中 = 世界模型认为'往环那侧移动'比'往反侧'代价更低）")
    report("横向 vy", lat_hits, lat_tot, lat_margins)
    report("垂直 vz", ver_hits, ver_tot, ver_margins)


if __name__ == "__main__":
    main()
