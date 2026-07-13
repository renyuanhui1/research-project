"""离线世界模型评测（零约束零规划）：多步 rollout 预测 vs 真实 latent，带基线。
answers: 预测器有效吗(vs identity)、听命令吗(vs 零动作)、给指令走对应路吗(vs 换动作)、泛化吗(held-out vs 训练)。

在服务器跑（吃显存）：
  python src/airsim/tools/eval_wm.py --base /path/to/repo
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PROJECT_ROOT), help="项目根目录；默认根据脚本位置自动推导")
    ap.add_argument("--steps", type=int, default=30, help="rollout 步数")
    ap.add_argument("--roll-from", type=int, default=0, help="从第几帧起 rollout")
    cli = ap.parse_args()
    BASE = cli.base
    sys.path.insert(0, f"{BASE}/src/airsim")
    from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
    from train_predictor import LatentPredictor

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    STEPS, ROLL_FROM = cli.steps, cli.roll_from

    ck = torch.load(f"{BASE}/weights/predictor_h5.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    val_num = int(a.get("val_num", 1)) if isinstance(a, dict) else getattr(a, "val_num", 1)
    model = LatentPredictor(ck["dim"], ck["num_patches"]).to(DEV).eval()
    model.load_state_dict(ck["model"])
    zm = torch.tensor(np.asarray(ck["z_mean"]), device=DEV)
    zs = torch.tensor(np.asarray(ck["z_std"]), device=DEV)
    mean = torch.tensor(IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEV).view(1, 3, 1, 1)
    dino = load_model("dinov2_vits14", DEV,
                      weights=f"{BASE}/weights/dinov2_vits14_pretrain.pth", repo_dir=f"{BASE}/dinov2")

    files = sorted(glob.glob(f"{BASE}/outputs/datasets/episodes_dataset/episode_*.h5"))
    n = len(files)
    held = list(range(n - val_num, n))
    train_sample = [i for i in [0, n // 3, 2 * n // 3] if i not in held][:3]
    eval_idx = held + train_sample
    print(f"共 {n} 条, held-out={held}, 训练抽样={train_sample}, val_num={val_num}, dev={DEV}")

    @torch.no_grad()
    def encode_ep(path, nframes):
        with h5py.File(path, "r") as f:
            rgb = f["rgb"][:nframes]
            act = f["action"][:nframes].astype(np.float32)
        zl = []
        for r in rgb:
            x = to_input_tensor(r[None], 224, mean, std, DEV)
            z = (dino.forward_features(x)["x_norm_patchtokens"][0].float() - zm) / zs
            zl.append(z)
        return torch.stack(zl), torch.tensor(act, device=DEV)

    @torch.no_grad()
    def rollout(z0, acts, T):
        z = z0.unsqueeze(0)
        out = []
        for k in range(T):
            z = model(z, acts[k:k + 1])
            out.append(z[0])
        return torch.stack(out)

    need = ROLL_FROM + STEPS + 1
    cache = {i: encode_ep(files[i], need) for i in eval_idx}

    def other(i):
        return eval_idx[(eval_idx.index(i) + 1) % len(eval_idx)]

    buckets = {"model": [], "id": [], "zero": [], "swap": []}
    tag = {}
    for i in eval_idx:
        zreal, acts = cache[i]
        T = min(STEPS, len(acts) - 1 - ROLL_FROM)
        z0 = zreal[ROLL_FROM]
        tgt = zreal[ROLL_FROM + 1:ROLL_FROM + 1 + T]
        e_model = ((rollout(z0, acts[ROLL_FROM:], T) - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        e_id = ((z0.unsqueeze(0) - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        e_zero = ((rollout(z0, torch.zeros_like(acts)[ROLL_FROM:], T) - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        e_swap = ((rollout(z0, cache[other(i)][1][:T], T) - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        buckets["model"].append(e_model); buckets["id"].append(e_id)
        buckets["zero"].append(e_zero); buckets["swap"].append(e_swap)
        tag[i] = "held-out" if i in held else "训练"

    hs = [h for h in [0, 2, 4, 9, 14, 19, 29] if h < STEPS]
    print("\n每条 episode 的每步预测误差(model)与 identity 基线比:")
    print(f"{'ep':>16} {'类型':>8} " + " ".join(f"h{h+1:>2}" for h in hs))
    for j, i in enumerate(eval_idx):
        em, ei = buckets["model"][j], buckets["id"][j]
        row = " ".join(f"{em[h]/max(ei[h],1e-6):>4.2f}" for h in hs)
        print(f"{os.path.basename(files[i]):>16} {tag[i]:>8} " + row + "   (model/identity, <1=赢)")

    def avg(key, sel):
        return np.mean(np.stack([buckets[key][eval_idx.index(i)] for i in sel]), axis=0)

    for grp, sel in [("held-out", held), ("训练", train_sample)]:
        if not sel:
            continue
        m, idd, ze, sw = avg("model", sel), avg("id", sel), avg("zero", sel), avg("swap", sel)
        print(f"\n=== {grp} 平均（{len(sel)}条）每步绝对 MSE ===")
        print(f"{'指标':>10} " + " ".join(f"h{h+1:>2}" for h in hs))
        for name, arr in [("model", m), ("identity", idd), ("zero-act", ze), ("swap动作", sw)]:
            print(f"{name:>10} " + " ".join(f"{arr[h]:>5.2f}" for h in hs))
        print(f"  model/identity: " + " ".join(f"{m[h]/max(idd[h],1e-6):>5.2f}" for h in hs) + "  (<1=学到运动)")
        print(f"  model/zero-act: " + " ".join(f"{m[h]/max(ze[h],1e-6):>5.2f}" for h in hs) + "  (<1=真听命令)")
        print(f"  model/swap:     " + " ".join(f"{m[h]/max(sw[h],1e-6):>5.2f}" for h in hs) + "  (<1=给指令走对应路)")


if __name__ == "__main__":
    main()
