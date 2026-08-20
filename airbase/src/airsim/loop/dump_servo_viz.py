"""dump_servo_viz.py —— 把伺服运行 run.h5 离线转成 rviz 可播的 npz dump(轨迹 + 指纹热力图)。

不用重飞：读 servo_closed_loop 存的 run.h5(每帧 rgb + pose)，用 DINO+模板重算每帧
patch 响应热力图，连 pose 打包成 step_XXXX.npz，喂给 src/mpc/plan_viz_node.py。
指纹算法与 check_target_signal.py / servo 一致(同一 stats-episode 标准化 + 模板 proto)。

产出每帧 npz 键: pose(n,e,d,yaw; 节点取[:3]画轨迹) / rgb / sim(P,=热力图) /
grid / step / dt / mass / peak。mass/peak 优先取伺服 v2 实际控制用的滤波值。

用法(有 GPU):
  python airbase/src/airsim/dump_servo_viz.py \
      --run-h5 airbase/outputs/runs/servo/尾翼_0722_110804/run.h5
然后本地 rviz:
  ros2 launch src/mpc/plan_viz.launch.py dump_dir:=<上面打印的 out-dir>
"""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path      # 引导块自足：不依赖上面的 import 顺序
# 向上找 sim_config/ 来定位 airbase/ —— 与脚本放在第几层无关。
# 原来写死 parents[2]，脚本一挪进子目录就指错，README 为此专门警告过「不要把脚本挪进子目录」。
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
# 跨分组 import（如 loop/servo_closed_loop → wm/decode_check）：各流程目录都挂上 sys.path
sys.path[:0] = [str(_AIRBASE_SRC)] + [str(d) for d in _AIRBASE_SRC.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "."))]
PROJECT_ROOT = _AIRBASE_ROOT


def encode_all(model, rgb, size, mean, std, dev, bs=32):
    from extract_dino_features import to_input_tensor
    out = []
    for i in range(0, len(rgb), bs):
        x = to_input_tensor(rgb[i:i + bs], size, mean, std, dev)
        out.append(model.forward_features(x)["x_norm_patchtokens"].float().cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-h5", required=True, help="servo_closed_loop 存的 run.h5")
    ap.add_argument("--out-dir", default=None, help="npz dump 目录(默认 run.h5 同级 viz_dump/)")
    ap.add_argument("--template", default=None, help="默认取 run.h5 属性里的模板")
    ap.add_argument("--stats-episode", default=None, help="默认取 run.h5 属性里的 stats_episode")
    ap.add_argument("--repo-dir", default=str(PROJECT_ROOT / "dinov2"))
    ap.add_argument("--dino-weights", default=str(PROJECT_ROOT / "weights/dinov2_vits14_pretrain.pth"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--dt", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src/airsim"))
    from extract_dino_features import load_model, IMAGENET_MEAN, IMAGENET_STD

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_h5 = Path(a.run_h5)
    out_dir = Path(a.out_dir) if a.out_dir else run_h5.parent / "viz_dump"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(run_h5, "r") as f:
        rgb = f["rgb"][:]
        pose = f["pose"][:]                      # (T,4) n,e,d,yaw
        if "control" in f:                       # v2: cx_f,cy_f,mass_f,peak_f,...
            mass_peak = f["control"][:, 2:4]
        elif "sig" in f:                         # 旧记录退化为原始 mass/peak
            mass_peak = f["sig"][:, 2:4]
        else:
            mass_peak = None
        tmpl = a.template or f.attrs.get("template")
        stats = a.stats_episode or f.attrs.get("stats_episode")
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
    print(f"rviz: ros2 launch src/mpc/plan_viz.launch.py dump_dir:={out_dir}")


if __name__ == "__main__":
    main()
