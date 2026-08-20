"""check_action_snr.py —— L2 体检：世界模型的**动作可分辨性信噪比**。

**要回答的问题**：离线 MSE/swap 都过了，闭环却看不到效果 —— 模型到底能不能分辨
MPPI 实际在比较的那些**彼此相近的候选动作**？

判据体系里已有的两级都不回答这个：
  L0  多步 latent MSE      —— 只说"预测得准不准"，文献明确它不预测规划成功
                              （arXiv 2606.09028 / 2605.07278）
  L1  swap / zero-act      —— 比的是"真动作 vs 随机换一条**别的** episode 的动作"，
                              这是**粗粒度**判别；MPPI 要的是几百条邻近候选之间排序
  L2  本脚本               —— 在 MPPI 真实采样的动作分布下，比较

        signal = 改变动作能让预测移动多远（候选之间的 spread）
        noise  = 模型自己错多远（用真动作预测 vs 真实 latent 的误差）
        SNR    = signal / noise

  **SNR < 1 → MPPI 是在模型误差里排序，闭环必然无效，且再怎么调 cost 也救不回来。**
  这与 §2026-08-04 坑3 在 **cost 侧**算的 SNR 0.087 是同一套逻辑，只是换到**模型侧**。

两个域都报：
  - **latent 域**：终点潜向量的 spread vs 预测误差
  - **cost 域**（更决定性，因为 MPPI 排的是 cost 不是 latent）：把终点 latent 过同一套
    模板指纹读出（peak / mass），比 cost 的 spread 与 cost 的预测误差

⚠️ 候选动作在动作边界内**均匀**采样，比 MPPI 实际的高斯采样**散得更开**
   → 算出的 SNR 是**乐观上界**。若均匀采样下都 <1，结论是铁的。

⚠️ 标准化一律用 **ckpt 里的 z_mean/z_std**（与 plan_closed_loop.Planner.encode 一致），
   不是 servo 那套从 stats-episode 现算的统计量 —— 两者不可混用。

用法（held-out 默认 280~299，与 train_predictor.split_episodes 的 val 划分一致）：
  python src/airsim/loop/check_action_snr.py \
      --ckpt weights/predictor_wm300_h8e150.pt \
      --template pictures/机头.jpg --horizon 12
"""

import argparse
import glob
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
from train_predictor import LatentPredictor

PROJECT_ROOT = _AIRBASE_ROOT


def parse_args():
    base = PROJECT_ROOT
    p = argparse.ArgumentParser(description="L2：世界模型的动作可分辨性 SNR")
    p.add_argument("--ckpt", type=Path, default=base / "weights/predictor_wm300_h8e150.pt")
    p.add_argument("--episode-dir", type=Path, default=base / "outputs/datasets/wm300")
    p.add_argument("--val-num", type=int, default=20,
                   help="held-out 条数；与 train_predictor.split_episodes 一致取**最后** N 条")
    p.add_argument("--episodes", type=int, default=8, help="实际取用的 held-out 条数(从 val 里取前几条)")
    p.add_argument("--frame-stride", type=int, default=10, help="每条 episode 每隔几帧取一个起点")
    p.add_argument("--horizon", type=int, default=12, help="rollout 步数；08-04 闭环用的是 12")
    p.add_argument("--n-candidates", type=int, default=100, help="每个起点采多少条候选动作序列")
    # 动作边界：§2026-08-04 按训练分布 0.5%/99.5% 分位重标的那组（非对称）
    p.add_argument("--vx-range", type=float, nargs=2, default=[0.25, 2.57])
    p.add_argument("--vy-range", type=float, nargs=2, default=[-2.07, 2.07])
    p.add_argument("--vz-range", type=float, nargs=2, default=[-0.38, 1.28])
    p.add_argument("--yaw-range", type=float, nargs=2, default=[-0.65, 0.65])
    # cost 域读出（与 plan_closed_loop.target_components / servo 的指纹一致）
    p.add_argument("--template", type=Path, default=base / "pictures/机头.jpg")
    p.add_argument("--repo-dir", type=Path, default=base / "dinov2")
    p.add_argument("--dino-weights", type=Path, default=base / "weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--target-softmax-temp", type=float, default=0.08)
    p.add_argument("--target-mass-thresh", type=float, default=0.35)
    p.add_argument("--target-mass-sharpness", type=float, default=20.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", type=Path,
                   default=base / "outputs/evaluations/action_snr/action_snr.csv")
    return p.parse_args()


def stack_hist_window(acts, k):
    """(N,H,4) 候选 + 已有历史 → 每步模型输入 (N,H,4k)，最新在前。

    与 train_predictor.stack_action_history / plan_closed_loop.rollout_inputs 同一约定：
    第 h 步的输入是 [a_h, a_{h-1}, ..., a_{h-k+1}]，不足处由**起点之前的真实动作**补齐。
    """
    if k <= 1:
        return acts
    n, h, _ = acts.shape
    full = acts                                        # 调用方已把 pre 拼在前面
    win = full.unfold(1, k, 1).permute(0, 1, 3, 2)     # (N, H, k, 4) 旧→新
    return win.flip(2).reshape(n, h - k + 1, k * 4)    # 翻成"最新在前"再摊平


def build_inputs(pre_acts, seq_acts, k):
    """pre_acts:(k-1,4) 起点前的真实动作(旧→新)；seq_acts:(N,H,4) → (N,H,4k)。"""
    n = seq_acts.shape[0]
    if k <= 1:
        return seq_acts
    pre = pre_acts.unsqueeze(0).expand(n, -1, -1)      # (N,k-1,4) 旧→新
    full = torch.cat([pre, seq_acts], dim=1)          # (N,k-1+H,4)
    return stack_hist_window(full, k)


@torch.no_grad()
def rollout(model, z0, inputs):
    """z0:(P,D)；inputs:(N,H,A) → 每步的预测 latent，list 长 H，每个 (N,P,D)。"""
    z = z0.unsqueeze(0).expand(inputs.shape[0], -1, -1).contiguous()
    outs = []
    for h in range(inputs.shape[1]):
        z = model(z, inputs[:, h])
        outs.append(z)
    return outs


@torch.no_grad()
def fingerprint(z, proto, args):
    """(...,P,D) → (peak, mass)。与 plan_closed_loop.target_components 同一读出。"""
    sim = torch.matmul(F.normalize(z.float(), dim=-1), proto)          # (...,P)
    peak = sim.max(dim=-1).values
    mass = torch.sigmoid(
        (sim - args.target_mass_thresh) * args.target_mass_sharpness).mean(dim=-1)
    return peak, mass


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    g = np.random.default_rng(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    k = int(ck.get("action_hist", 1))
    model = LatentPredictor(ck["dim"], ck["num_patches"],
                            action_dim=int(ck.get("action_dim", 4))).to(device).eval()
    model.load_state_dict(ck["model"])
    zm = torch.tensor(np.asarray(ck["z_mean"]), device=device)
    zs = torch.tensor(np.asarray(ck["z_std"]), device=device)
    print(f"ckpt={args.ckpt.name}  action_hist={k}  dim={ck['dim']}  "
          f"patches={ck['num_patches']}  ckpt_epoch={ck.get('epoch')}  val_mse={ck.get('val_mse'):.4f}")

    dino = load_model("dinov2_vits14", device,
                      weights=str(args.dino_weights), repo_dir=str(args.repo_dir))
    imean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    istd = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(rgb_batch):
        """(N,224,224,3) uint8 → 标准化 latent (N,P,D)，与 Planner.encode 同一约定。"""
        out = []
        for i in range(0, len(rgb_batch), 32):
            x = to_input_tensor(rgb_batch[i:i + 32], 224, imean, istd, device)
            out.append(dino.forward_features(x)["x_norm_patchtokens"].float())
        return (torch.cat(out, 0) - zm) / zs

    tbgr = cv2.imread(str(args.template))
    if tbgr is None:
        raise SystemExit(f"读不到模板: {args.template}")
    zt = encode(np.ascontiguousarray(tbgr[:, :, ::-1])[None])[0]
    proto = F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0)
    print(f"模板={args.template.name}  候选={args.n_candidates}  horizon={args.horizon}")

    files = sorted(glob.glob(str(args.episode_dir / "episode_*.h5")))
    val_files = files[-args.val_num:][:args.episodes]
    print(f"held-out 取 {len(val_files)} 条: {Path(val_files[0]).name} .. {Path(val_files[-1]).name}\n")

    H = args.horizon
    lo = torch.tensor([args.vx_range[0], args.vy_range[0],
                       args.vz_range[0], args.yaw_range[0]], device=device)
    hi = torch.tensor([args.vx_range[1], args.vy_range[1],
                       args.vz_range[1], args.yaw_range[1]], device=device)

    # 逐 horizon 累积：候选间 spread、模型误差、cost 的 spread 与误差
    acc = {h: {"spread": [], "err": [], "pk_sp": [], "pk_er": [], "ms_sp": [], "ms_er": []}
           for h in range(1, H + 1)}
    n_pts = 0

    for fp_ in val_files:
        with h5py.File(fp_, "r") as f:
            rgb = f["rgb"][:]
            act = f["action"][:].astype(np.float32)
        z_all = encode(rgb)                                   # (T,P,D) 标准化
        acts_t = torch.tensor(act, device=device)
        T = len(rgb)
        for t0 in range(k - 1, T - H, args.frame_stride):
            z0 = z_all[t0]
            pre = acts_t[t0 - k + 1:t0]                       # (k-1,4) 旧→新
            true_seq = acts_t[t0:t0 + H].unsqueeze(0)         # (1,H,4)
            cand = lo + (hi - lo) * torch.rand(args.n_candidates, H, 4, device=device)

            z_true_pred = rollout(model, z0, build_inputs(pre, true_seq, k))
            z_cand = rollout(model, z0, build_inputs(pre, cand, k))

            for h in range(1, H + 1):
                zt_real = z_all[t0 + h]                       # 真实 latent
                zp = z_true_pred[h - 1][0]                    # 用真动作预测
                zc = z_cand[h - 1]                            # (N,P,D) 候选
                acc[h]["spread"].append(zc.std(dim=0).pow(2).mean().sqrt().item())
                acc[h]["err"].append((zp - zt_real).pow(2).mean().sqrt().item())

                pk_c, ms_c = fingerprint(zc, proto, args)     # (N,)
                pk_p, ms_p = fingerprint(zp, proto, args)     # 标量
                pk_r, ms_r = fingerprint(zt_real, proto, args)
                acc[h]["pk_sp"].append(pk_c.std().item())
                acc[h]["pk_er"].append(abs(pk_p.item() - pk_r.item()))
                acc[h]["ms_sp"].append(ms_c.std().item())
                acc[h]["ms_er"].append(abs(ms_p.item() - ms_r.item()))
            n_pts += 1

    print(f"统计自 {n_pts} 个起点\n")
    print("     ---------- latent 域 ----------    ---------- cost 域(peak) ----------   (mass)")
    print("  h   spread     err     SNR            spread     err      SNR             SNR")
    rows = []
    for h in range(1, H + 1):
        a = acc[h]
        sp, er = float(np.mean(a["spread"])), float(np.mean(a["err"]))
        psp, per = float(np.mean(a["pk_sp"])), float(np.mean(a["pk_er"]))
        msp, mer = float(np.mean(a["ms_sp"])), float(np.mean(a["ms_er"]))
        snr, psnr = sp / max(er, 1e-9), psp / max(per, 1e-9)
        msnr = msp / max(mer, 1e-9)
        flag = "" if psnr >= 1.0 else "  <- cost 域已淹没"
        print(f"{h:3d}  {sp:7.4f} {er:7.4f}  {snr:6.3f}          "
              f"{psp:7.4f} {per:7.4f}  {psnr:6.3f}          {msnr:6.3f}{flag}")
        rows.append([h, sp, er, snr, psp, per, psnr, msp, mer, msnr])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("h,latent_spread,latent_err,latent_snr,"
                "peak_spread,peak_err,peak_snr,mass_spread,mass_err,mass_snr\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" if i else str(v) for i, v in enumerate(r)) + "\n")
    print(f"\n已写出 {args.out_csv}")
    print("判读：cost 域 SNR < 1 ⇒ MPPI 在模型误差里排序，闭环必然无效，调 cost 也救不回来。"
          "\n      （候选是均匀采样，比 MPPI 的高斯采样更散 → 这已是乐观上界。）")


if __name__ == "__main__":
    main()
