"""本地可行性测试：在 4060 Ti(8G,仿真已占 ~7G)上，闭环推理还剩多少显存、够不够、多快。

测三件事：
  1. 加载 DINO(本地仓库+权重)+ 世界模型到 GPU，看常驻显存；
  2. 对一帧 224×224 抽 DINO 特征，计时；
  3. 跑一次 MPPI(默认 256 条 × horizon 30)rollout，计时；
报告峰值显存 + 单步总耗时，据此判断本地一体化闭环可行否。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
from train_predictor import LatentPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", default=f"{base}/dinov2")
    p.add_argument("--dino-weights", default=f"{base}/weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--samples", type=int, default=256, help="MPPI 采样条数")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--fp16", action="store_true", help="DINO 用半精度省显存")
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()

    # 1) 加载 DINO + 世界模型
    dino = load_model("dinov2_vits14", dev, weights=args.dino_weights, repo_dir=args.repo_dir)
    if args.fp16:
        dino = dino.half()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = LatentPredictor(ck["dim"], ck["num_patches"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    if args.fp16:
        mean, std = mean.half(), std.half()
    load_mb = torch.cuda.memory_allocated() / 1e6
    print(f"模型常驻显存 ≈ {load_mb:.0f} MB")

    # 2) DINO 抽一帧特征计时（用随机图模拟相机帧）
    rgb = (np.random.rand(1, args.image_size, args.image_size, 3) * 255).astype(np.uint8)
    with torch.no_grad():
        for _ in range(3):  # 预热
            x = to_input_tensor(rgb, args.image_size, mean, std, dev)
            if args.fp16:
                x = x.half()
            z = dino.forward_features(x)["x_norm_patchtokens"]
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            x = to_input_tensor(rgb, args.image_size, mean, std, dev)
            if args.fp16:
                x = x.half()
            z = dino.forward_features(x)["x_norm_patchtokens"]
        torch.cuda.synchronize()
        dino_ms = (time.time() - t0) / 10 * 1000
    print(f"DINO 抽一帧 ≈ {dino_ms:.1f} ms  (z {tuple(z.shape)})")

    # 3) MPPI 一次 rollout 计时（z0 用刚抽的特征，float32 进世界模型）
    z0 = z[0].float()                                   # (P,D)
    P, D = z0.shape
    with torch.no_grad():
        acts = torch.randn(args.samples, args.horizon, 4, device=dev)
        def rollout():
            zb = z0.unsqueeze(0).expand(args.samples, -1, -1).contiguous()
            for k in range(args.horizon):
                zb = model(zb, acts[:, k])
            return zb
        for _ in range(2):
            rollout()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            rollout()
        torch.cuda.synchronize()
        mppi_ms = (time.time() - t0) / 5 * 1000
    print(f"MPPI {args.samples}条×{args.horizon}步 rollout ≈ {mppi_ms:.1f} ms")

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n峰值显存 ≈ {peak_mb:.0f} MB")
    print(f"单步预算(DINO+1次MPPI) ≈ {dino_ms + mppi_ms:.1f} ms → 约 {1000/(dino_ms+mppi_ms):.1f} Hz")
    print("注：闭环每步通常迭代 MPPI 数轮，实际帧率按 轮数×MPPI 估算；显存若 < 1000MB 才稳妥与仿真共存")


if __name__ == "__main__":
    main()
