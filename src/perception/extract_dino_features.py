"""冻结DINOv2加载、图像预处理与批量特征提取。

load_model()与to_input_tensor()供在线感知复用。命令行从episode HDF5的rgb提取patch tokens，
写入指定特征目录；使用--input-dir、--output-dir、--weights和--repo-dir配置输入及本地模型。"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

# torch 延迟到 main 里导入，方便没装 torch 时也能 `python -c import` 检查语法
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# Allow both direct script execution and package imports.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()

PATCH = 14

# 模型名 → 特征维度（torch.hub facebookresearch/dinov2）
MODEL_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def parse_args():
    p = argparse.ArgumentParser(description="冻结 DINOv2 抽 latent（一次性预处理）")
    p.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/datasets/episodes_dataset")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/features/latents")
    p.add_argument("--model", default="dinov2_vits14", choices=list(MODEL_DIM))
    p.add_argument("--weights", type=Path, default=project_path(PROFILE["assets"]["dino_weights"]),
                   help="本地预训练 .pth 路径（如 dinov2_vits14_pretrain.pth）；给了就离线加载权重")
    p.add_argument("--repo-dir", type=Path, default=project_path(PROFILE["assets"]["dino_repo"]),
                   help="本地 dinov2 仓库目录（含 hubconf.py）；给了就用 source=local 离线取模型结构")
    p.add_argument("--image-size", type=int, default=224, help="送入 DINO 的边长，须为 14 的倍数")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="存盘精度")
    p.add_argument("--include-cls", action="store_true", help="额外存 CLS token（dataset 'cls'）")
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    p.add_argument("--overwrite", action="store_true", help="已存在也重抽")
    return p.parse_args()


def load_model(model_name, device, weights=None, repo_dir=None):
    """加载冻结 DINOv2。
    - repo_dir 给定：用本地 dinov2 仓库（source=local）取模型结构，离线、不连 GitHub。
    - weights 给定：从本地 .pth 加载权重（此时不联网下权重）。
    - 都不给：退回 torch.hub 联网加载（结构+权重）。
    """
    import torch
    want_pretrained = weights is None  # 没给本地权重时才让 hub 下载权重
    if repo_dir is not None:
        model = torch.hub.load(str(repo_dir), model_name, source="local",
                               pretrained=want_pretrained)
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=want_pretrained)
    if weights is not None:
        state = torch.load(str(weights), map_location=device)
        model.load_state_dict(state)
        print(f"已从本地权重加载: {weights}")
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def to_input_tensor(rgb_batch, image_size, mean, std, device):
    """rgb_batch: (B,H,W,3) uint8（标准 RGB）→ 归一化后的 (B,3,S,S) float tensor。"""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(rgb_batch).to(device).float().div_(255.0)  # (B,H,W,3)
    x = x.permute(0, 3, 1, 2).contiguous()                          # (B,3,H,W)
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    return x


def extract_episode(model, rgb, args, device, mean, std):
    """对一条 episode 的 rgb (T,H,W,3) 抽特征，返回 (z, cls)。"""
    import torch
    T = rgb.shape[0]
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    z_list, cls_list = [], []
    with torch.no_grad():
        for i in range(0, T, args.batch_size):
            batch = rgb[i:i + args.batch_size]
            x = to_input_tensor(batch, args.image_size, mean, std, device)
            feats = model.forward_features(x)
            patch = feats["x_norm_patchtokens"]          # (B, P, D)
            z_list.append(patch.cpu().numpy().astype(np_dtype))
            if args.include_cls:
                cls_list.append(feats["x_norm_clstoken"].cpu().numpy().astype(np_dtype))
    z = np.concatenate(z_list, axis=0)                   # (T, P, D)
    cls = np.concatenate(cls_list, axis=0) if cls_list else None
    return z, cls


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.image_size % PATCH != 0:
        raise ValueError(f"--image-size {args.image_size} 必须是 {PATCH} 的倍数")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(args.input_dir.expanduser().resolve() / "episode_*.h5")))
    if not files:
        raise SystemExit(f"输入目录没有 episode_*.h5: {args.input_dir}")

    print(f"设备={device} 模型={args.model}(dim={MODEL_DIM[args.model]}) "
          f"image_size={args.image_size} patch={PATCH} → {(args.image_size//PATCH)**2} 个 patch")
    model = load_model(args.model, device, weights=args.weights, repo_dir=args.repo_dir)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    for fn in files:
        stem = Path(fn).stem
        out_path = out_dir / f"{stem}_dino.h5"
        if out_path.exists() and not args.overwrite:
            print(f"跳过（已存在）: {out_path.name}")
            continue
        with h5py.File(fn, "r") as f:
            rgb = f["rgb"][:]  # (T,H,W,3) uint8
        z, cls = extract_episode(model, rgb, args, device, mean, std)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("z", data=z, compression="gzip", compression_opts=4)
            if cls is not None:
                f.create_dataset("cls", data=cls, compression="gzip", compression_opts=4)
            f.attrs["source"] = Path(fn).name
            f.attrs["model"] = args.model
            f.attrs["dim"] = MODEL_DIM[args.model]
            f.attrs["num_patches"] = z.shape[1]
            f.attrs["image_size"] = args.image_size
            f.attrs["dtype"] = args.dtype
            f.attrs["token"] = "patch+cls" if cls is not None else "patch"
        print(f"已抽: {out_path.name}  z{z.shape}{' +cls'+str(cls.shape) if cls is not None else ''}")

    print(f"\n完成，latent 写入 {out_dir}")


if __name__ == "__main__":
    main()
