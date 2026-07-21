"""离线检查 target cost 在世界模型 rollout 下是否偏好合理动作。

这不是闭环控制脚本。它回答一个更基础的问题：
  从同一帧 z0 出发，世界模型预测不同动作序列后的 target cost 排序是什么？

如果 target cost + world model 不把"真实前进/接近"排到更好，MPPI 在线也不会自发前进。
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from plan_closed_loop import Planner

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="检查 target cost 对不同动作 rollout 的排序")
    p.add_argument("--episode", type=Path,
                   default=Path(f"{base}/outputs/datasets/episodes_dataset/episode_0050.h5"))
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--target-template", default=f"{base}/outputs/references/templates/tmpl_goal0050_green.png")
    p.add_argument("--repo-dir", default=f"{base}/dinov2")
    p.add_argument("--dino-weights", default=f"{base}/weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--no-fp16", action="store_true")
    # target cost 分量权重，默认取 check_cost_monotonic 调优结果（ρ_facing=+0.82）
    p.add_argument("--target-conf-weight", type=float, default=0.5)
    p.add_argument("--target-center-weight", type=float, default=0.4)
    p.add_argument("--target-size-weight", type=float, default=1.2)
    p.add_argument("--target-mass-thresh", type=float, default=0.2)
    p.add_argument("--target-mass-sharpness", type=float, default=10.0)
    return p.parse_args()


def planner_args(args):
    return SimpleNamespace(
        repo_dir=args.repo_dir,
        dino_weights=args.dino_weights,
        ckpt=args.ckpt,
        no_fp16=args.no_fp16,
        cost_metric="target",
        target_template=args.target_template,
        v_max=2.0,
        vz_max=0.5,
        yaw_max=1.0,
        vx_min=None,
        use_spline=False,
        init_vx=0.7,
        target_conf_weight=args.target_conf_weight,
        target_center_weight=args.target_center_weight,
        target_size_weight=args.target_size_weight,
        target_softmax_temp=0.08,
        target_topk_frac=0.08,
        target_mass_thresh=args.target_mass_thresh,
        target_mass_sharpness=args.target_mass_sharpness,
        target_terminal_weight=0.7,
        target_path_weight=0.3,
        action_prior_weight=0.0,
        forward_bonus_weight=0.0,
    )


def make_actions(true_actions, horizon):
    cases = {
        "zero": np.zeros((horizon, 4), dtype=np.float32),
        "forward_0.5": np.tile([0.5, 0.0, 0.0, 0.0], (horizon, 1)).astype(np.float32),
        "forward_1.0": np.tile([1.0, 0.0, 0.0, 0.0], (horizon, 1)).astype(np.float32),
        "east_0.5": np.tile([0.0, 0.5, 0.0, 0.0], (horizon, 1)).astype(np.float32),
        "west_0.5": np.tile([0.0, -0.5, 0.0, 0.0], (horizon, 1)).astype(np.float32),
        "yaw_left": np.tile([0.5, 0.0, 0.0, -0.3], (horizon, 1)).astype(np.float32),
        "yaw_right": np.tile([0.5, 0.0, 0.0, 0.3], (horizon, 1)).astype(np.float32),
    }
    if len(true_actions) >= horizon:
        cases["dataset_true"] = true_actions[:horizon].astype(np.float32)
    return cases


@torch.no_grad()
def rollout(planner, z0, acts):
    z = z0.unsqueeze(0)
    costs = []
    comps = []
    for k in range(acts.shape[0]):
        z = planner.model(z, acts[k:k + 1])
        c = planner.target_components(z[0])
        costs.append(c["cost"].item())
        comps.append(c)
    return z[0], costs, comps


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    planner = Planner(planner_args(args), device)

    with h5py.File(args.episode, "r") as h:
        rgb = h["rgb"][args.frame]
        true_actions = h["action"][args.frame:args.frame + args.horizon]
        # dataset_true 的真实结果帧：区分"世界模型预测错"还是"cost 在该段无信号"
        rgb_future = (h["rgb"][args.frame + args.horizon]
                      if args.frame + args.horizon < h["rgb"].shape[0] else None)

    z0 = planner.encode(np.ascontiguousarray(rgb))
    c0 = planner.target_components(z0)
    print(f"episode={args.episode} frame={args.frame} horizon={args.horizon}")
    print(f"start cost={c0['cost'].item():.4f} peak={c0['peak'].item():.3f} mass={c0['mass'].item():.3f}")
    print("case             final_cost  delta    path_mean  peak   mass   center_x center_y")

    rows = []
    for name, acts_np in make_actions(true_actions, args.horizon).items():
        acts = torch.tensor(acts_np, device=device)
        zend, costs, comps = rollout(planner, z0, acts)
        cf = comps[-1]
        center = cf["center"]
        rows.append((name, costs[-1], np.mean(costs), cf, center))

    for name, final_cost, path_mean, cf, center in sorted(rows, key=lambda r: r[1]):
        print(f"{name:15s} {final_cost:10.4f} {final_cost - c0['cost'].item():7.4f}"
              f" {path_mean:10.4f} {cf['peak'].item():6.3f} {cf['mass'].item():6.3f}"
              f" {center[0].item():8.3f} {center[1].item():8.3f}")

    # 对照：dataset_true 真实执行后的真实帧（不经世界模型）。
    # 若真实帧 cost 降而上面 dataset_true 预测行升 → 世界模型预测是误差来源；
    # 若真实帧 cost 也不降 → 该距离/horizon 下 cost 本身无信号。
    if rgb_future is not None:
        zf = planner.encode(np.ascontiguousarray(rgb_future))
        cr = planner.target_components(zf)
        center = cr["center"]
        print(f"{'REAL_future':15s} {cr['cost'].item():10.4f}"
              f" {cr['cost'].item() - c0['cost'].item():7.4f} {'-':>10s}"
              f" {cr['peak'].item():6.3f} {cr['mass'].item():6.3f}"
              f" {center[0].item():8.3f} {center[1].item():8.3f}   <- 真实帧对照")


if __name__ == "__main__":
    main()
