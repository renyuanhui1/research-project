"""验证脚本：训练好的世界模型在验证集上到底有没有学到动力学。

判定标准（与尺度无关）：模型单步预测 MSE 应明显低于 identity 基线
（identity = 预测 z_{t+1}=z_t）。两者都在"训练集统计标准化"的空间里算才可比。

用法（默认路径按服务器目录，可命令行覆盖）：
  python verify_predictor.py
  python verify_predictor.py --ckpt .../predictor.pt --latent-dir .../latents \
      --episode-dir .../episodes_dataset --val-num 3
"""
import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import torch

from train_predictor import LatentPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="验证世界模型是否超过 identity 基线")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    p.add_argument("--latent-dir", default=f"{base}/outputs/features/latents")
    p.add_argument("--episode-dir", default=f"{base}/outputs/datasets/episodes_dataset")
    p.add_argument("--val-num", type=int, default=3, help="末尾留作验证的 episode 数（与训练一致）")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck.get("z_mean") is None:
        raise SystemExit("ckpt 里没有 z_mean/z_std，该模型训练时没开标准化，无法可比验证")
    m = torch.tensor(np.asarray(ck["z_mean"]), device=device)
    s = torch.tensor(np.asarray(ck["z_std"]), device=device)

    model = LatentPredictor(ck["dim"], ck["num_patches"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    val_files = sorted(glob.glob(str(Path(args.latent_dir) / "episode_*_dino.h5")))[-args.val_num:]
    print(f"验证集 {len(val_files)} 条: {[Path(f).name for f in val_files]}")

    mse_model, mse_id = [], []
    with torch.no_grad():
        for f in val_files:
            with h5py.File(f, "r") as h:
                z = torch.tensor(h["z"][:].astype(np.float32), device=device)
                src = str(h.attrs["source"])
            with h5py.File(Path(args.episode_dir) / src, "r") as h:
                a = torch.tensor(h["action"][:].astype(np.float32), device=device)
            zn = (z - m) / s                          # 用训练集统计标准化
            pred = model(zn[:-1], a[:-1])             # a_t 驱动 z_t→z_{t+1}
            mse_model.append(((pred - zn[1:]) ** 2).mean().item())
            mse_id.append(((zn[:-1] - zn[1:]) ** 2).mean().item())  # identity 基线

    mm, mi = float(np.mean(mse_model)), float(np.mean(mse_id))
    print(f"模型 单步 MSE   = {mm:.4f}")
    print(f"identity 基线   = {mi:.4f}")
    drop = (1 - mm / mi) * 100
    print(f"相对 identity 降低 = {drop:.1f}%")
    verdict = "通过：模型在没见过的数据上确实学到了动力学" if mm < mi * 0.9 else \
              ("勉强：略好于 identity，泛化有限" if mm < mi else "未通过：未超过 identity，没学到泛化的动力学")
    print("结论:", verdict)


if __name__ == "__main__":
    main()
