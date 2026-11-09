"""冻结DINO特征、模板原型和全图匹配中心计算。"""

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from perception.extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD


class Fingerprint:
    """DINO 编码 + 模板指纹, 返回 center/mass/peak。标准化统计量来自 stats-episode。"""

    def __init__(self, args, device):
        self.dev = device
        self.args = args
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.dino = load_model("dinov2_vits14", device,
                               weights=str(args.dino_weights), repo_dir=str(args.repo_dir))
        # 标准化统计量: 从 stats-episode 帧算
        with h5py.File(args.stats_episode, "r") as f:
            srgb = f["rgb"][:]
        zf = self._encode(srgb)                      # (N,P,D)
        self.zm = zf.reshape(-1, zf.shape[-1]).mean(0).to(device)
        self.zs = zf.reshape(-1, zf.shape[-1]).std(0).clamp_min(1e-6).to(device)
        # 模板 proto：默认保留全模板；可选 ROI 原型只在显式传 --template-roi 时创建。
        tbgr = cv2.imread(str(args.template))
        if tbgr is None:
            raise SystemExit(f"读不到模板: {args.template}")
        zt = self._encode(np.ascontiguousarray(tbgr[:, :, ::-1])[None])[0].to(device)
        zt = (zt - self.zm) / self.zs
        P = zf.shape[1]; self.grid = int(round(P ** 0.5))
        if self.grid * self.grid != P:
            raise SystemExit(f"DINO patch 数不是方形网格: {P}")
        xs = torch.linspace(-1.0, 1.0, self.grid, device=device)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
        self.protos = {
            "full": F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0),
        }
        self.roi_patch_count = 0
        if args.template_roi is not None:
            x0, y0, x1, y1 = args.template_roi
            axis = (torch.arange(self.grid, device=device, dtype=torch.float32) + 0.5) / self.grid
            vv, uu = torch.meshgrid(axis, axis, indexing="ij")
            mask = ((uu.reshape(-1) >= x0) & (uu.reshape(-1) <= x1)
                    & (vv.reshape(-1) >= y0) & (vv.reshape(-1) <= y1))
            if not mask.any():
                raise SystemExit(f"--template-roi {args.template_roi} 没覆盖任何模板 patch")
            self.protos["roi"] = F.normalize(
                F.normalize(zt[mask], dim=-1).mean(0), dim=0)
            self.roi_patch_count = int(mask.sum())
        detail = (f", ROI={args.template_roi} ({self.roi_patch_count} patches)"
                  if self.has_roi else "")
        print(f"指纹就绪: proto D={self.protos['full'].shape[0]}, grid={self.grid}{detail}")

    @property
    def has_roi(self):
        return "roi" in self.protos

    @torch.no_grad()
    def _encode(self, rgb, bs=32):
        out = []
        for i in range(0, len(rgb), bs):
            x = to_input_tensor(rgb[i:i + bs], self.args.image_size, self.mean, self.std, self.dev)
            out.append(self.dino.forward_features(x)["x_norm_patchtokens"].float().cpu())
        return torch.cat(out, 0)

    @torch.no_grad()
    def query(self, rgb224, use_roi=False):
        """rgb224:(224,224,3) uint8 → (center_x, center_y, mass, peak)。"""
        x = to_input_tensor(rgb224[None], self.args.image_size, self.mean, self.std, self.dev)
        z = self.dino.forward_features(x)["x_norm_patchtokens"][0].float()
        z = (z - self.zm) / self.zs
        proto = self.protos["roi" if use_roi and self.has_roi else "full"]
        sim = torch.matmul(F.normalize(z, dim=-1), proto)           # (P,) 即热力图
        w = torch.softmax(sim / self.args.target_softmax_temp, dim=-1)
        center = torch.matmul(w, self.patch_xy)                     # (2,)
        mass = torch.sigmoid(
            (sim - self.args.target_mass_thresh) * self.args.target_mass_sharpness).mean()
        peak = sim.max()
        return (float(center[0]), float(center[1]), float(mass), float(peak),
                sim.cpu().numpy())
