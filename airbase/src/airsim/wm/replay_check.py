"""脚本 3：replay_check.py —— 采集质量验证

目的：把采好的 HDF5 放出来，肉眼确认图像序列流畅、和动作/位姿对得上、无白图错位。
做法：
  1) 数值检查：帧数==动作数==位姿数==时间数；时间戳单调递增；图像不空；位姿有变化。
  2) 把 rgb 逐帧（放大便于看）叠加当前 action / pose / time，拼成带标注的 mp4。

通过标准：能确认这条 episode 数据干净、对齐。
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def run_checks(rgb, action, pose, tstamp):
    """打印数值一致性检查，返回是否全部通过。"""
    ok = True
    n = len(rgb)
    print(f"帧数 rgb={len(rgb)} action={len(action)} pose={len(pose)} time={len(tstamp)}")
    if not (len(action) == len(pose) == len(tstamp) == n):
        print("  [FAIL] 各数组长度不一致"); ok = False
    else:
        print("  [OK] 各数组长度一致")

    diffs = np.diff(tstamp)
    if np.all(diffs > 0):
        print(f"  [OK] 时间戳单调递增（中位步长 {np.median(diffs)/1e6:.1f} ms）")
    else:
        print(f"  [FAIL] 时间戳非单调，{int((diffs<=0).sum())} 处非递增"); ok = False

    means = rgb.reshape(n, -1).mean(axis=1)
    empty = int(((means < 2) | (means > 253)).sum())
    if empty == 0:
        print(f"  [OK] 无白/黑空图（亮度均值范围 {means.min():.1f}~{means.max():.1f}）")
    else:
        print(f"  [WARN] 疑似空图 {empty} 帧（纯黑/纯白）")

    moved = float(np.linalg.norm(pose[-1, :3] - pose[0, :3]))
    qn = np.linalg.norm(pose[:, 3:7], axis=1)
    print(f"  [INFO] 位移 {moved:.2f} m；四元数模长 {qn.min():.3f}~{qn.max():.3f}（应≈1）")
    if moved < 0.1:
        print("  [WARN] 总位移过小，机体几乎没动")
    return ok


def overlay(frame_bgr, idx, n, action, pose, tstamp, t0):
    """在一帧 BGR 图上叠加文字信息。"""
    lines = [
        f"frame {idx}/{n-1}  t={(tstamp - t0)/1e9:.2f}s",
        f"act vx={action[0]:+.2f} vy={action[1]:+.2f} vz={action[2]:+.2f} yr={action[3]:+.2f}",
        f"pos x={pose[0]:+.1f} y={pose[1]:+.1f} z={pose[2]:+.1f}",
    ]
    y = 22
    for ln in lines:
        cv2.putText(frame_bgr, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return frame_bgr


def make_video(rgb, action, pose, tstamp, out_path, fps, scale):
    n = len(rgb)
    h = w = rgb.shape[1] * scale
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开 VideoWriter: {out_path}")
    t0 = int(tstamp[0])
    for i in range(n):
        bgr = rgb[i][..., ::-1]  # 存的是 RGB，转 BGR 给 cv2
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        bgr = overlay(np.ascontiguousarray(bgr), i, n, action[i], pose[i], int(tstamp[i]), t0)
        writer.write(bgr)
    writer.release()
    print(f"已写视频: {out_path}  ({n} 帧 @ {fps:.1f}fps, {w}x{h})")


def parse_args():
    p = argparse.ArgumentParser(description="回放验证 HDF5 episode")
    p.add_argument("input", type=Path, help="episode HDF5 文件")
    p.add_argument("--output", type=Path, default=None, help="输出 mp4（默认同名 .mp4）")
    p.add_argument("--fps", type=float, default=None, help="默认按 attrs 的 dt 推算")
    p.add_argument("--scale", type=int, default=2, help="放大倍数便于观看")
    return p.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.input, "r") as f:
        rgb = f["rgb"][:]
        action = f["action"][:]
        pose = f["pose"][:]
        tstamp = f["time"][:]
        dt = float(f.attrs.get("dt", 0.1))

    print(f"== 检查 {args.input} ==")
    ok = run_checks(rgb, action, pose, tstamp)

    # 默认按真实时间戳算 fps（视频时长=真实飞行时长），而非名义 1/dt
    if args.fps:
        fps = args.fps
    else:
        span_s = (int(tstamp[-1]) - int(tstamp[0])) / 1e9
        fps = (len(tstamp) - 1) / span_s if span_s > 0 else (1.0 / dt)
        print(f"按真实时间戳推算 fps={fps:.2f}（真实时长 {span_s:.2f}s）")
    out = args.output or args.input.with_suffix(".mp4")
    make_video(rgb, action, pose, tstamp, out, fps, args.scale)

    print("== 结果:", "PASS（数值检查通过，请再肉眼看视频确认对齐）" if ok else "FAIL（见上方）", "==")


if __name__ == "__main__":
    main()
