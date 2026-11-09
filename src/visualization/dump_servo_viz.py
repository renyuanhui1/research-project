"""dump_servo_viz.py —— 把伺服运行 run.h5 离线转成 rviz 可播的 npz dump(轨迹 + 指纹热力图)。"""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# Allow both direct script execution and package imports.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()



def record_path(value, run_dir=None):
    """按当前场景根目录查找运行记录中的文件路径。"""
    path = Path(value)
    if path.is_absolute() and not path.exists() and PROJECT_ROOT.name in path.parts:
        index = path.parts.index(PROJECT_ROOT.name)
        candidate = PROJECT_ROOT.joinpath(*path.parts[index + 1:])
        if candidate.exists():
            return candidate
    if not path.exists() and run_dir is not None and (Path(run_dir)/path.name).exists():
        return Path(run_dir)/path.name
    return path


def encode_all(model, rgb, size, mean, std, dev, bs=32):
    from perception.extract_dino_features import to_input_tensor
    out = []
    for i in range(0, len(rgb), bs):
        x = to_input_tensor(rgb[i:i + bs], size, mean, std, dev)
        out.append(model.forward_features(x)["x_norm_patchtokens"].float().cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="Scene profile name or JSON path")
    ap.add_argument("--run-h5", required=True, help="servo_closed_loop 存的 run.h5")
    ap.add_argument("--out-dir", default=None, help="npz dump 目录(默认 run.h5 同级 viz_dump/)")
    ap.add_argument("--template", default=None, help="默认取 run.h5 属性里的模板")
    ap.add_argument("--stats-episode", default=None, help="默认取 run.h5 属性里的 stats_episode")
    ap.add_argument("--repo-dir", default=str(project_path(PROFILE["assets"]["dino_repo"])))
    ap.add_argument("--dino-weights", default=str(project_path(PROFILE["assets"]["dino_weights"])))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--dt", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import sys
    from perception.extract_dino_features import load_model, IMAGENET_MEAN, IMAGENET_STD

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_h5 = Path(a.run_h5)
    out_dir = Path(a.out_dir) if a.out_dir else run_h5.parent / "viz_dump"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(run_h5, "r") as f:
        rgb = f["rgb"][:]
        from evaluation.eval_run import load_run
        _,pos,yaw,_,_,_=load_run(run_h5)
        pose=np.column_stack((pos,yaw))
        if "control" in f:                       # v2: cx_f,cy_f,mass_f,peak_f,...
            mass_peak = f["control"][:, 2:4]
        elif "sig" in f:                         # 旧记录退化为原始 mass/peak
            mass_peak = f["sig"][:, 2:4]
        else:
            mass_peak = None
        tmpl = a.template or f.attrs.get("template") or str(project_path(PROFILE["assets"]["template"]))
        stats = a.stats_episode or f.attrs.get("stats_episode") or str(project_path(PROFILE["assets"]["stats_episode"]))
    tmpl = record_path(tmpl, run_h5.parent)
    stats = record_path(stats, run_h5.parent)
    print(f"run.h5: {len(rgb)} 帧  模板={tmpl}  stats={stats}")

    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    model = load_model("dinov2_vits14", dev, weights=a.dino_weights, repo_dir=a.repo_dir)

    # 标准化统计量: 与伺服同源(stats-episode 全帧全 patch)
    with h5py.File(stats, "r") as f:
        srgb = f["rgb"][:]
    zsf = encode_all(model, srgb, a.image_size, mean, std, dev)
    zm = zsf.reshape(-1, zsf.shape[-1]).mean(0)
    zs = zsf.reshape(-1, zsf.shape[-1]).std(0).clamp_min(1e-6)

    # 模板 proto
    tbgr = cv2.imread(str(tmpl))
    if tbgr is None:
        raise SystemExit(f"读不到模板: {tmpl}")
    zt = encode_all(model, np.ascontiguousarray(tbgr[:, :, ::-1])[None], a.image_size, mean, std, dev)[0]
    zt = (zt - zm) / zs
    proto = F.normalize(F.normalize(zt, dim=-1).mean(0), dim=0)   # (D,)

    # 逐帧: 热力图 sim = normalize(标准化 patch) · proto
    zf = encode_all(model, rgb, a.image_size, mean, std, dev)      # (T,P,D)
    zf = (zf - zm) / zs
    sim = torch.matmul(F.normalize(zf, dim=-1), proto).numpy()     # (T,P)
    P = zf.shape[1]; grid = int(round(P ** 0.5))

    for i in range(len(rgb)):
        payload = dict(step=i, pose=pose[i].astype(np.float32), rgb=rgb[i].astype(np.uint8),
                       sim=sim[i].astype(np.float32), grid=grid, dt=a.dt)
        if mass_peak is not None:
            payload.update(mass=np.float32(mass_peak[i, 0]),
                           peak=np.float32(mass_peak[i, 1]))
        np.savez_compressed(out_dir / f"step_{i:04d}.npz", **payload)
    print(f"已写 {len(rgb)} 个 npz → {out_dir}")
    print(f"rviz: ros2 launch src/visualization/matching.launch.py replay:=true dump_dir:={out_dir}")


if __name__ == "__main__":
    main()
