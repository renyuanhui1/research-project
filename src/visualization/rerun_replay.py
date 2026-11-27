#!/usr/bin/env python3
"""Offline Rerun (.rrd) replay and side-by-side comparison CLI.

Supports:
  - `opening`: replay an opening-alignment or opening-entry control directory
    (3D RGB-D roof point cloud, trajectory, feature/matching view, metric depth,
    raw RGB, interior snapshot, time-series curves, and control logs).
  - `compare-harbor`: side-by-side comparison of two Coastal Harbor Task 2
    indoor-entry runs (synchronized H.264 dual-view videos, 3D breach point cloud
    with wind arrows and start/end comparison, altitude/error/compensation curves,
    and event logs).

Use `--serve` to immediately host the Rerun web viewer over HTTP/gRPC
(recommended under WSL2 so rendering runs natively in the Windows browser).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from perception.depth_map import _quat_to_R  # noqa: E402


def ned_to_enu(pos: list[float] | tuple[float, float, float] | dict[str, Any]) -> list[float]:
    """Convert NED [N, E, D] (or {'x': N, 'y': E, 'z': D}) to ENU [E, N, -D]."""
    if isinstance(pos, dict):
        if "position" in pos and isinstance(pos["position"], dict):
            pos = pos["position"]
        return [float(pos["y"]), float(pos["x"]), float(-pos["z"])]
    return [float(pos[1]), float(pos[0]), float(-pos[2])]


def encode_jpeg(img_path: Path, max_width: int = 800, quality: int = 82) -> bytes | None:
    """Read an image, downscale if wider than max_width, and return JPEG bytes."""
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def sanitize_depth_m(
    depth: np.ndarray,
    *,
    min_m: float = 0.1,
    max_m: float = 120.0,
    stride: int = 1,
) -> np.ndarray:
    """Return a clean float32 metric depth array (0.0 for invalid pixels)."""
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if stride > 1:
        arr = arr[::stride, ::stride]
    valid = np.isfinite(arr) & (arr > min_m) & (arr < max_m)
    return np.where(valid, arr, 0.0).astype(np.float32)


def project_nadir_rgbd(
    rgb_path: Path,
    depth_path: Path,
    cam_pos_ned: list[float],
    cam_rot_wxyz: list[float],
    *,
    stride: int = 2,
    min_depth_m: float = 0.5,
    max_depth_m: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a 90-degree horizontal FOV pinhole RGB-D frame into ENU points."""
    raw_npz = np.load(depth_path)
    depth = (raw_npz["depth_m"] if hasattr(raw_npz, "files") and "depth_m" in raw_npz.files else raw_npz).astype(
        np.float32
    )
    if depth.ndim == 3:
        depth = depth[..., 0]

    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"Unable to read RGB image: {rgb_path}")
    rgb = rgb_bgr[:, :, ::-1]

    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    vals = depth[::stride, ::stride]
    valid = np.isfinite(vals) & (vals > min_depth_m) & (vals < max_depth_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), depth

    u = (xx[valid] + 0.5 - w / 2.0) / (w / 2.0)
    v = (yy[valid] + 0.5 - h / 2.0) / (w / 2.0)
    dirs = np.column_stack([np.ones(u.size, dtype=np.float32), u.astype(np.float32), v.astype(np.float32)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    local = dirs * vals[valid, None]

    rot = _quat_to_R(cam_rot_wxyz)
    ned = local @ rot.T + np.asarray(cam_pos_ned, dtype=np.float64)
    enu = np.column_stack([ned[:, 1], ned[:, 0], -ned[:, 2]]).astype(np.float32)
    colors = rgb[yy[valid], xx[valid]].astype(np.uint8)
    return enu, colors, depth


def _load_opening_background_points(flight_dir: Path, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Load prebuilt rviz/rgbd-map.npz if present, else project sparse RGB-D frames."""
    map_path = flight_dir / "rviz" / "rgbd-map.npz"
    if map_path.exists():
        z = np.load(map_path)
        return z["points_enu_m"].astype(np.float32), z["rgb"].astype(np.uint8)

    if not rows:
        return None
    indices = sorted(set(range(0, len(rows), 10)) | {len(rows) - 1})
    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    for idx in indices:
        row = rows[idx]
        meta = row.get("depth_metadata")
        img_rel = row.get("image")
        dep_rel = row.get("depth")
        if not (meta and img_rel and dep_rel):
            continue
        img_p = flight_dir / img_rel
        dep_p = flight_dir / dep_rel
        if not (img_p.exists() and dep_p.exists()):
            continue
        cam_pos = [meta["pos_x"], meta["pos_y"], meta["pos_z"]]
        cam_rot = [meta["rot_w"], meta["rot_x"], meta["rot_y"], meta["rot_z"]]
        pts, cols, _ = project_nadir_rgbd(img_p, dep_p, cam_pos, cam_rot, stride=4, max_depth_m=150.0)
        if pts.shape[0] > 0:
            all_pts.append(pts)
            all_rgb.append(cols)
    if not all_pts:
        return None
    pts_cat = np.concatenate(all_pts, axis=0)
    rgb_cat = np.concatenate(all_rgb, axis=0)
    _, keep = np.unique(np.floor(pts_cat / 0.35).astype(np.int32), axis=0, return_index=True)
    return pts_cat[keep], rgb_cat[keep]


def _log_video_asset(entity_path: str, mp4_path: Path) -> float:
    """Attach an MP4 asset statically and reference every frame on the `time_s` timeline."""
    asset = rr.AssetVideo(path=mp4_path)
    rr.log(entity_path, asset, static=True)
    ts_ns = asset.read_frame_timestamps_nanos()
    if len(ts_ns) > 0:
        for t_ns in ts_ns:
            rr.set_time("time_s", duration=float(t_ns) * 1e-9)
            rr.log(entity_path, rr.VideoFrameReference(nanoseconds=int(t_ns)))
        return float(ts_ns[-1]) * 1e-9
    return 0.0


def build_opening_rrd(
    flight_dir: Path,
    out_rrd: Path,
    *,
    max_image_width: int = 800,
    depth_stride: int = 2,
    depth_range: tuple[float, float] = (5.0, 75.0),
) -> dict[str, Any]:
    """Export an opening-alignment or opening-entry control run into a `.rrd` file."""
    flight_dir = Path(flight_dir).resolve()
    out_rrd = Path(out_rrd).resolve()
    out_rrd.parent.mkdir(parents=True, exist_ok=True)

    signals_path = flight_dir / "signals.jsonl"
    if not signals_path.exists():
        raise FileNotFoundError(f"Missing signals.jsonl in {flight_dir}")
    rows = [json.loads(line) for line in signals_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {signals_path}")

    manifest_path = flight_dir / "rviz" / "manifest.json"
    items: list[dict[str, Any]] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_items = manifest.get("items") or []
        if len(raw_items) == len(rows):
            items = raw_items

    bg_cloud = _load_opening_background_points(flight_dir, rows)
    full_traj = [ned_to_enu(r["pose"]) for r in rows]
    alts = [p[2] for p in full_traj]
    interior_path = flight_dir / "interior-observation.png"
    has_interior = interior_path.exists()

    tabs_2d: list[Any] = [
        rrb.Spatial2DView(origin="camera/feature_view", name="匹配 / 特征跟踪画面"),
        rrb.Spatial2DView(origin="camera/depth_m", name="深度图 (米制, 悬停读数)"),
        rrb.Spatial2DView(origin="camera/raw_rgb", name="原始机载 RGB"),
    ]
    if has_interior:
        tabs_2d.append(rrb.Spatial2DView(origin="camera/interior_snapshot", name="入室后前向实拍"))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name=f"3D 机库点云与飞行轨迹 ({alts[0]:.1f}m -> {alts[-1]:.1f}m)",
            ),
            rrb.Vertical(
                rrb.Tabs(*tabs_2d),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="metrics/altitude", name="高度与距屋面距离 (m)"),
                    rrb.TimeSeriesView(origin="metrics/error_and_speed", name="水平误差 & 速度"),
                    rrb.TextLogView(origin="logs/control", name="控制状态日志"),
                    row_shares=[1.0, 1.0, 0.85],
                ),
                row_shares=[1.2, 1.25],
            ),
            column_shares=[1.35, 1.0],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )

    rr.init(f"opening_replay_{flight_dir.parent.name}")
    rr.save(str(out_rrd), default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if bg_cloud is not None:
        rr.log("world/hangar_rgbd_map", rr.Points3D(bg_cloud[0], colors=bg_cloud[1], radii=0.16), static=True)
    rr.log(
        "world/full_trajectory_ref",
        rr.LineStrips3D([full_traj], colors=[[150, 150, 165]], radii=0.09),
        static=True,
    )
    rr.log(
        "world/waypoints",
        rr.Points3D(
            [full_traj[0], full_traj[-1]],
            colors=[[80, 225, 120], [255, 80, 160]],
            radii=0.65,
            labels=[f"Start ({alts[0]:.1f}m)", f"End ({alts[-1]:.1f}m)"],
        ),
        static=True,
    )

    if has_interior:
        interior_jpg = encode_jpeg(interior_path, max_width=960, quality=85)
        if interior_jpg is not None:
            rr.log(
                "camera/interior_snapshot",
                rr.EncodedImage(contents=interior_jpg, media_type="image/jpeg"),
                static=True,
            )

    rr.log(
        "metrics/altitude/drone_altitude_m",
        rr.SeriesLines(colors=[80, 180, 255], names="drone_altitude (m)", widths=2.2),
        static=True,
    )
    rr.log(
        "metrics/altitude/roof_remaining_m",
        rr.SeriesLines(colors=[255, 150, 40], names="roof_remaining (m, <0=室内)", widths=2.0),
        static=True,
    )
    rr.log(
        "metrics/altitude/center_floor_range_m",
        rr.SeriesLines(colors=[130, 220, 130], names="center_floor_depth (m)", widths=1.6),
        static=True,
    )
    rr.log(
        "metrics/error_and_speed/horizontal_error_m",
        rr.SeriesLines(colors=[255, 95, 95], names="horizontal_error (m)", widths=2.0),
        static=True,
    )
    rr.log(
        "metrics/error_and_speed/horizontal_speed_mps",
        rr.SeriesLines(colors=[90, 200, 255], names="horizontal_speed (m/s)", widths=1.6),
        static=True,
    )
    rr.log(
        "metrics/error_and_speed/down_speed_mps",
        rr.SeriesLines(colors=[210, 130, 255], names="down_speed (m/s)", widths=1.8),
        static=True,
    )

    t0 = float(rows[0]["time_s"])
    prev_state: str | None = None
    traj_so_far: list[list[float]] = []

    for idx, row in enumerate(rows):
        rel_t = float(row["time_s"]) - t0
        rr.set_time("step", sequence=int(row.get("step", idx)))
        rr.set_time("time_s", duration=rel_t)

        pos = full_traj[idx]
        alt_m = pos[2]
        traj_so_far.append(pos)

        err_ne = row.get("error_ne_m") or [0.0, 0.0]
        err_m = float(math.hypot(err_ne[0], err_ne[1]))
        speed_mps = float(row.get("horizontal_speed_mps") or 0.0)
        down_mps = float(row.get("down_speed_mps") or 0.0)
        state = str(row.get("state", ""))
        entry = row.get("entry") or {}
        roof_rem = entry.get("roof_remaining_m")
        floor_rng = entry.get("minimum_center_floor_range_m")

        rr.log("world/drone/trajectory", rr.LineStrips3D([traj_so_far], colors=[[255, 165, 30]], radii=0.22))
        extra_lbl = f" | roof={roof_rem:+.1f}m" if roof_rem is not None else f" | err={err_m:.2f}m"
        rr.log(
            "world/drone/position",
            rr.Points3D(
                [pos],
                colors=[[40, 145, 255]],
                radii=0.65,
                labels=[f"#{row.get('step', idx)} {state} z={alt_m:.1f}m{extra_lbl}"],
            ),
        )

        item = items[idx] if items is not None else None
        ap = item.get("projected_aperture_enu_m") if item is not None else None
        if ap is not None:
            rr.log("world/aperture/projected_point", rr.Points3D([ap], colors=[[0, 240, 240]], radii=0.65, labels=["Opening"]))
            rr.log("world/aperture/sight_ray", rr.LineStrips3D([[pos, ap]], colors=[[0, 210, 210]], radii=0.08))

        feat_path: Path | None = None
        if item is not None and item.get("image"):
            cand = Path(item["image"])
            if cand.exists():
                feat_path = cand
        if feat_path is None and row.get("diagnostic"):
            cand = flight_dir / row["diagnostic"]
            if cand.exists():
                feat_path = cand
        if feat_path is not None:
            feat_jpg = encode_jpeg(feat_path, max_width=max_image_width, quality=80)
            if feat_jpg is not None:
                rr.log("camera/feature_view", rr.EncodedImage(contents=feat_jpg, media_type="image/jpeg"))

        if row.get("image"):
            raw_jpg = encode_jpeg(flight_dir / row["image"], max_width=min(max_image_width, 640), quality=78)
            if raw_jpg is not None:
                rr.log("camera/raw_rgb", rr.EncodedImage(contents=raw_jpg, media_type="image/jpeg"))

        if row.get("depth"):
            depth_path = flight_dir / row["depth"]
            if depth_path.exists():
                depth_clean = sanitize_depth_m(np.load(depth_path), stride=depth_stride)
                rr.log(
                    "camera/depth_m",
                    rr.DepthImage(depth_clean, meter=1.0, colormap="turbo", depth_range=depth_range),
                )

        rr.log("metrics/altitude/drone_altitude_m", rr.Scalars(alt_m))
        if roof_rem is not None and math.isfinite(float(roof_rem)):
            rr.log("metrics/altitude/roof_remaining_m", rr.Scalars(float(roof_rem)))
        if floor_rng is not None and math.isfinite(float(floor_rng)):
            rr.log("metrics/altitude/center_floor_range_m", rr.Scalars(float(floor_rng)))

        rr.log("metrics/error_and_speed/horizontal_error_m", rr.Scalars(err_m))
        rr.log("metrics/error_and_speed/horizontal_speed_mps", rr.Scalars(speed_mps))
        rr.log("metrics/error_and_speed/down_speed_mps", rr.Scalars(down_mps))

        if state != prev_state or idx % 20 == 0 or idx == len(rows) - 1:
            vp = entry.get("visual_pose") or {}
            inliers = vp.get("inliers")
            extra_log = (
                f" roof_rem={roof_rem:+.1f}m down={down_mps:.2f}m/s inliers={inliers}"
                if roof_rem is not None
                else f" err={err_m:.2f}m speed={speed_mps:.2f}m/s"
            )
            rr.log(
                "logs/control",
                rr.TextLog(f"step={int(row.get('step', idx)):03d} t={rel_t:5.1f}s z={alt_m:5.1f}m [{state}]{extra_log}"),
            )
            prev_state = state

    rr.disconnect()
    return {
        "frames": len(rows),
        "out_rrd": str(out_rrd),
        "size_bytes": out_rrd.stat().st_size,
        "altitude_range_m": [round(min(alts), 2), round(max(alts), 2)],
        "has_map": bg_cloud is not None,
        "has_interior": has_interior,
    }


def _find_target_entry(summary: dict[str, Any], target_id: str | None) -> tuple[str, dict[str, Any]]:
    targets = summary.get("targets") or []
    if not targets:
        raise ValueError("No targets listed in summary.json")
    if target_id is None:
        first = targets[0]
        return str(first["target_id"]), first
    for item in targets:
        if str(item.get("target_id")) == target_id:
            return target_id, item
    raise ValueError(f"Target {target_id!r} not found in summary.json")


def build_harbor_compare_rrd(
    baseline_dir: Path,
    method_dir: Path,
    out_rrd: Path,
    *,
    target_id: str | None = None,
    baseline_label: str = "Baseline-P",
    method_label: str = "DOB-CBF-QP",
    point_stride: int = 2,
    depth_range: tuple[float, float] = (10.0, 40.0),
) -> dict[str, Any]:
    """Export a side-by-side comparison `.rrd` for two Coastal Harbor Task 2 runs."""
    baseline_dir = Path(baseline_dir).resolve()
    method_dir = Path(method_dir).resolve()
    out_rrd = Path(out_rrd).resolve()
    out_rrd.parent.mkdir(parents=True, exist_ok=True)

    base_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    meth_summary = json.loads((method_dir / "summary.json").read_text(encoding="utf-8"))

    resolved_target, base_target = _find_target_entry(base_summary, target_id)
    _, meth_target = _find_target_entry(meth_summary, resolved_target)

    base_tdir = baseline_dir / resolved_target
    meth_tdir = method_dir / resolved_target
    base_trace_obj = json.loads((base_tdir / "opening_servo_trace.json").read_text(encoding="utf-8"))
    meth_trace_obj = json.loads((meth_tdir / "opening_servo_trace.json").read_text(encoding="utf-8"))
    base_rows = base_trace_obj.get("trace") or []
    meth_rows = meth_trace_obj.get("trace") or []

    nc = meth_target.get("nadir_capture") or base_target.get("nadir_capture") or {}
    pts_enu: np.ndarray | None = None
    colors_rgb: np.ndarray | None = None
    nadir_depth: np.ndarray | None = None
    nadir_jpg = meth_tdir / "nadir_overview.jpg"
    nadir_npz = meth_tdir / "nadir_overview.npz"
    if not (nadir_jpg.exists() and nadir_npz.exists()):
        nadir_jpg = base_tdir / "nadir_overview.jpg"
        nadir_npz = base_tdir / "nadir_overview.npz"
    if nadir_jpg.exists() and nadir_npz.exists() and nc.get("cam_pos_ned") and nc.get("cam_rot_wxyz"):
        pts_enu, colors_rgb, nadir_depth = project_nadir_rgbd(
            nadir_jpg, nadir_npz, nc["cam_pos_ned"], nc["cam_rot_wxyz"], stride=point_stride
        )

    base_mp4 = base_tdir / "indoor_entry.mp4"
    meth_mp4 = meth_tdir / "indoor_entry.mp4"
    has_videos = base_mp4.exists() or meth_mp4.exists()

    base_stop = base_trace_obj.get("stop_reason", "")
    meth_stop = meth_trace_obj.get("stop_reason", "")
    base_alt_end = float(base_trace_obj.get("final_alt_m") or (base_rows[-1]["alt_m"] if base_rows else 0.0))
    meth_alt_end = float(meth_trace_obj.get("final_alt_m") or (meth_rows[-1]["alt_m"] if meth_rows else 0.0))

    left_tabs: list[Any] = [
        rrb.Spatial3DView(origin="world", name=f"3D {resolved_target} 破口点云 & 起终点对比"),
    ]
    meth_interior = meth_tdir / "interior_forward.jpg"
    base_interior = base_tdir / "interior_forward.jpg"
    if meth_interior.exists():
        left_tabs.append(rrb.Spatial2DView(origin="snapshots/interior_method", name=f"{method_label} 入室后前向实拍"))
    if base_interior.exists():
        left_tabs.append(rrb.Spatial2DView(origin="snapshots/interior_baseline", name=f"{baseline_label} 入室后前向实拍"))
    breach_overlay = meth_tdir / "nadir_breach_overlay.jpg"
    if not breach_overlay.exists():
        breach_overlay = base_tdir / "nadir_breach_overlay.jpg"
    if breach_overlay.exists():
        left_tabs.append(rrb.Spatial2DView(origin="snapshots/nadir_breach", name="俯视破口测量标注"))
    if nadir_depth is not None:
        left_tabs.append(rrb.Spatial2DView(origin="snapshots/nadir_depth", name="俯视米制深度图 (悬停读数)"))

    bottom_row = rrb.Horizontal(
        rrb.Tabs(*left_tabs),
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="metrics/altitude_m",
                name=f"高度对比 (m): {baseline_label} {base_alt_end:.1f}m vs {method_label} {meth_alt_end:.1f}m",
            ),
            rrb.TimeSeriesView(origin="metrics/error_m", name="水平对准误差对比 (m)"),
            rrb.Tabs(
                rrb.TimeSeriesView(origin="metrics/wind_comp", name="侧向抗风指令 & DOB/CBF 补偿强度"),
                rrb.TextLogView(origin="logs/compare", name="实验对照事件日志"),
            ),
            row_shares=[1.0, 1.0, 1.0],
        ),
        column_shares=[1.05, 1.25],
    )

    if has_videos:
        top_row = rrb.Horizontal(
            rrb.Spatial2DView(origin="video/baseline", name=f"{baseline_label} ({base_stop})"),
            rrb.Spatial2DView(origin="video/method", name=f"{method_label} ({meth_stop})"),
            column_shares=[1.0, 1.0],
        )
        root_layout: Any = rrb.Vertical(top_row, bottom_row, row_shares=[1.0, 1.25])
    else:
        root_layout = bottom_row

    blueprint = rrb.Blueprint(
        root_layout,
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )

    rr.init(f"harbor_compare_{resolved_target}")
    rr.save(str(out_rrd), default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if pts_enu is not None and colors_rgb is not None and pts_enu.shape[0] > 0:
        rr.log(f"world/{resolved_target}_roof", rr.Points3D(pts_enu, colors=colors_rgb, radii=0.12), static=True)

    start_ned = nc.get("drone_pos_ned")
    base_end_ned = base_trace_obj.get("final_pose_ned")
    meth_end_ned = meth_trace_obj.get("final_pose_ned")

    start_enu = ned_to_enu(start_ned) if start_ned else None
    if start_enu is not None:
        rr.log(
            "world/endpoints/start",
            rr.Points3D([start_enu], colors=[[100, 200, 255]], radii=0.65, labels=[f"共同起点 ({start_enu[2]:.1f}m)"]),
            static=True,
        )
    if base_end_ned:
        base_end_enu = ned_to_enu(base_end_ned)
        rr.log(
            "world/endpoints/baseline_final",
            rr.Points3D(
                [base_end_enu],
                colors=[[255, 75, 75]],
                radii=0.75,
                labels=[f"{baseline_label} 终态: {base_end_enu[2]:.1f}m ({base_stop})"],
            ),
            static=True,
        )
        if start_enu is not None:
            rr.log(
                "world/endpoints/displacement_baseline",
                rr.LineStrips3D([[start_enu, base_end_enu]], colors=[[255, 80, 80]], radii=0.14),
                static=True,
            )
    if meth_end_ned:
        meth_end_enu = ned_to_enu(meth_end_ned)
        rr.log(
            "world/endpoints/method_final",
            rr.Points3D(
                [meth_end_enu],
                colors=[[60, 235, 120]],
                radii=0.75,
                labels=[f"{method_label} 终态: {meth_end_enu[2]:.1f}m ({meth_stop})"],
            ),
            static=True,
        )
        if start_enu is not None:
            rr.log(
                "world/endpoints/displacement_method",
                rr.LineStrips3D([[start_enu, meth_end_enu]], colors=[[60, 235, 120]], radii=0.16),
                static=True,
            )

    wind_ned = meth_summary.get("wind_ned_mps") or base_summary.get("wind_ned_mps")
    if start_enu is not None and wind_ned and len(wind_ned) == 3:
        wind_enu = np.asarray(ned_to_enu(wind_ned), dtype=np.float64)
        wind_speed = float(np.linalg.norm(wind_enu))
        if wind_speed > 0.1:
            unit = wind_enu / wind_speed
            arrow_vec = (unit * 6.0).tolist()
            perp = np.array([-unit[1], unit[0], 0.0])
            base_origin = np.array([start_enu[0], start_enu[1], max(start_enu[2] - 6.0, 12.0)]) - unit * 12.0
            origins = [(base_origin + perp * offset).tolist() for offset in (-4.0, 0.0, 4.0)]
            rr.log(
                "world/wind_arrows",
                rr.Arrows3D(
                    origins=origins,
                    vectors=[arrow_vec] * 3,
                    colors=[[0, 210, 255]] * 3,
                    radii=0.18,
                    labels=["", f"原生侧风 {wind_speed:.1f} m/s", ""],
                ),
                static=True,
            )

    if meth_interior.exists():
        jpg = encode_jpeg(meth_interior, max_width=960, quality=84)
        if jpg is not None:
            rr.log("snapshots/interior_method", rr.EncodedImage(contents=jpg, media_type="image/jpeg"), static=True)
    if base_interior.exists():
        jpg = encode_jpeg(base_interior, max_width=960, quality=84)
        if jpg is not None:
            rr.log("snapshots/interior_baseline", rr.EncodedImage(contents=jpg, media_type="image/jpeg"), static=True)
    if breach_overlay.exists():
        jpg = encode_jpeg(breach_overlay, max_width=960, quality=84)
        if jpg is not None:
            rr.log("snapshots/nadir_breach", rr.EncodedImage(contents=jpg, media_type="image/jpeg"), static=True)
    if nadir_depth is not None:
        depth_clean = sanitize_depth_m(nadir_depth, max_m=80.0)
        rr.log(
            "snapshots/nadir_depth",
            rr.DepthImage(depth_clean, meter=1.0, colormap="turbo", depth_range=depth_range),
            static=True,
        )

    rr.log(
        "metrics/altitude_m/baseline",
        rr.SeriesLines(colors=[255, 85, 85], names=f"{baseline_label} 高度 (m)", widths=2.2),
        static=True,
    )
    rr.log(
        "metrics/altitude_m/method",
        rr.SeriesLines(colors=[60, 235, 120], names=f"{method_label} 高度 (m)", widths=2.4),
        static=True,
    )
    rr.log(
        "metrics/error_m/baseline",
        rr.SeriesLines(colors=[255, 85, 85], names=f"{baseline_label} 水平误差 (m)", widths=2.2),
        static=True,
    )
    rr.log(
        "metrics/error_m/method",
        rr.SeriesLines(colors=[60, 235, 120], names=f"{method_label} 水平误差 (m)", widths=2.4),
        static=True,
    )
    rr.log(
        "metrics/wind_comp/baseline_cmd_y",
        rr.SeriesLines(colors=[255, 110, 110], names=f"{baseline_label} 侧向指令 cmd_y (m/s)", widths=1.8),
        static=True,
    )
    rr.log(
        "metrics/wind_comp/method_cmd_y",
        rr.SeriesLines(colors=[60, 235, 120], names=f"{method_label} 侧向指令 cmd_y (m/s)", widths=2.0),
        static=True,
    )
    rr.log(
        "metrics/wind_comp/method_intervention",
        rr.SeriesLines(colors=[255, 200, 60], names=f"{method_label} DOB/CBF 补偿幅度 (m/s)", widths=2.0),
        static=True,
    )

    if base_mp4.exists():
        _log_video_asset("video/baseline", base_mp4)
    if meth_mp4.exists():
        _log_video_asset("video/method", meth_mp4)

    for tag, label, rows, stop_reason, succeeded in [
        ("baseline", baseline_label, base_rows, base_stop, bool(base_trace_obj.get("succeeded"))),
        ("method", method_label, meth_rows, meth_stop, bool(meth_trace_obj.get("succeeded"))),
    ]:
        prev_phase: str | None = None
        for idx, r in enumerate(rows):
            t_s = float(r["t_s"])
            rr.set_time("time_s", duration=t_s)
            rr.log(f"metrics/altitude_m/{tag}", rr.Scalars(float(r["alt_m"])))
            if r.get("err_horiz_m") is not None:
                rr.log(f"metrics/error_m/{tag}", rr.Scalars(float(r["err_horiz_m"])))
            cmd = r.get("cmd_body") or [0.0, 0.0, 0.0]
            rr.log(f"metrics/wind_comp/{tag}_cmd_y", rr.Scalars(float(cmd[1])))
            if tag == "method" and r.get("cbf_intervention") is not None:
                rr.log("metrics/wind_comp/method_intervention", rr.Scalars(float(r["cbf_intervention"])))

            phase = str(r.get("phase", ""))
            is_last = idx == len(rows) - 1
            if phase != prev_phase or is_last:
                suffix = f" --> {'SUCCESS' if succeeded else 'STOP'}: {stop_reason}" if is_last else ""
                rr.log(
                    "logs/compare",
                    rr.TextLog(
                        f"[{label}] t={t_s:5.1f}s phase={phase} alt={float(r['alt_m']):.2f}m "
                        f"err={r.get('err_horiz_m')} detected={r.get('detected')}{suffix}",
                        level="WARN" if (is_last and not succeeded) else "INFO",
                    ),
                )
                prev_phase = phase

    rr.disconnect()
    return {
        "target_id": resolved_target,
        "baseline_steps": len(base_rows),
        "method_steps": len(meth_rows),
        "out_rrd": str(out_rrd),
        "size_bytes": out_rrd.stat().st_size,
        "has_videos": has_videos,
        "has_point_cloud": pts_enu is not None and pts_enu.shape[0] > 0,
    }


def _find_rerun_cli() -> str:
    sibling = Path(sys.executable).resolve().parent / "rerun"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("rerun")
    if found:
        return found
    raise FileNotFoundError("Could not locate `rerun` CLI binary. Install `rerun-sdk` in the active environment.")


def serve_rrd(rrd_path: Path, *, web_port: int = 9090, grpc_port: int = 9876) -> int:
    """Host an `.rrd` recording via Rerun's HTTP web viewer + gRPC server."""
    rerun_bin = _find_rerun_cli()
    url = f"http://127.0.0.1:{web_port}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{grpc_port}%2Fproxy"
    print(f"Serving {rrd_path} in browser at: {url}", flush=True)
    proc = subprocess.run(
        [
            rerun_bin,
            "--serve-web",
            "--web-viewer-port",
            str(web_port),
            "--port",
            str(grpc_port),
            str(rrd_path),
        ],
        check=False,
    )
    return int(proc.returncode)


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--serve", action="store_true", help="导出后立即启动 Rerun 网页查看器（适合 WSL/远程浏览器）")
    parser.add_argument("--web-port", type=int, default=9090, help="网页查看器 HTTP 端口，默认 9090")
    parser.add_argument("--grpc-port", type=int, default=9876, help="Rerun gRPC 数据流端口，默认 9876")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun 离线多模态回放与实验对照工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_open = sub.add_parser("opening", help="回放洞口对准或穿洞入室控制目录（含 signals.jsonl 与 RGB-D）")
    p_open.add_argument("--flight-dir", type=Path, required=True, help="控制记录目录（包含 signals.jsonl）")
    p_open.add_argument("--out", type=Path, default=None, help="输出 .rrd 路径，默认写入 <flight-dir>/rerun/replay.rrd")
    p_open.add_argument("--max-image-width", type=int, default=800, help="2D 画面 JPEG 最大宽度像素，默认 800")
    p_open.add_argument("--depth-stride", type=int, default=2, help="深度图降采样步长，默认 2")
    p_open.add_argument("--depth-range", type=float, nargs=2, default=(5.0, 75.0), metavar=("MIN_M", "MAX_M"), help="深度图着色范围（米）")
    _add_serve_args(p_open)

    p_cmp = sub.add_parser("compare-harbor", help="同屏对比两组渔港 Task2 破口入室实验（如 Baseline vs DOB-CBF）")
    p_cmp.add_argument("--baseline-dir", type=Path, required=True, help="基线实验目录（包含 summary.json）")
    p_cmp.add_argument("--method-dir", type=Path, required=True, help="对比方法实验目录（包含 summary.json）")
    p_cmp.add_argument("--target", type=str, default=None, help="目标建筑 ID（默认自动取 summary.json 首项，如 Building_098）")
    p_cmp.add_argument("--baseline-label", type=str, default="Baseline-P", help="基线显示名称")
    p_cmp.add_argument("--method-label", type=str, default="DOB-CBF-QP", help="对比方法显示名称")
    p_cmp.add_argument("--out", type=Path, required=True, help="输出 .rrd 路径")
    p_cmp.add_argument("--point-stride", type=int, default=2, help="俯视屋顶点云投影降采样步长，默认 2")
    p_cmp.add_argument("--depth-range", type=float, nargs=2, default=(10.0, 40.0), metavar=("MIN_M", "MAX_M"), help="俯视深度图着色范围（米）")
    _add_serve_args(p_cmp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "opening":
        out_path = args.out or (args.flight_dir / "rerun" / "replay.rrd")
        summary = build_opening_rrd(
            args.flight_dir,
            out_path,
            max_image_width=args.max_image_width,
            depth_stride=args.depth_stride,
            depth_range=(float(args.depth_range[0]), float(args.depth_range[1])),
        )
    elif args.command == "compare-harbor":
        out_path = args.out
        summary = build_harbor_compare_rrd(
            args.baseline_dir,
            args.method_dir,
            out_path,
            target_id=args.target,
            baseline_label=args.baseline_label,
            method_label=args.method_label,
            point_stride=args.point_stride,
            depth_range=(float(args.depth_range[0]), float(args.depth_range[1])),
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.serve:
        return serve_rrd(Path(summary["out_rrd"]), web_port=args.web_port, grpc_port=args.grpc_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
