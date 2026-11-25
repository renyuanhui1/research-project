"""Coastal Harbor Task 3: Continuous Linear Eave Inspection & Damage Boundary Localization.

Flies continuously from s = 0 m to s = 32.2 m along a warehouse roof eave, capturing
synchronized RGB-D frames at fixed along-track spacing, then evaluates:
1. Effective coverage ratio & missed-segment count via independent camera-to-line projection
2. Continuous along-track damage localization (start/end boundary error in meters & interval IoU)
   by fusing VLM per-frame horizontal damage-span detections projected onto the reference line.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.paths import CONFIGS, project_path
from common.runtime import configure_client_log
from coordination.vlm import OllamaVLMClient, VLMImage, VLMResponse
from harbor.part_recheck import (
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_KEY,
    _await_task,
    _camera_pose,
    _capture_rgbd,
    _load_harbor_runtime_scene,
    _stage_drone_at_start,
    _strip_runtime_arrays,
    quat_wxyz_to_rot,
    ue_to_ned,
)
from harbor.recorder import HarborVideoRecorder


DEFAULT_LINE_CONFIG = CONFIGS / "missions" / "coastal_harbor_line_inspection.json"

LINE_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "roof_visible": {"type": "boolean"},
        "has_roof_damage": {"type": "boolean"},
        "damage_u_start": {"type": "number"},
        "damage_u_end": {"type": "number"},
        "damage_description": {"type": "string"},
    },
    "required": [
        "roof_visible",
        "has_roof_damage",
        "damage_u_start",
        "damage_u_end",
        "damage_description",
    ],
}

LINE_FRAME_SYSTEM_PROMPT = (
    "你是沿海渔港仓库屋檐沿线连续巡检视觉分析员。当前图像是无人机沿仓库斜屋面一侧匀速飞行时拍下的单帧画面。\n"
    "请严格依据当前画面中央屋面坡段回答：\n"
    "1. roof_visible：画面中是否可见仓库斜坡屋面与屋檐。\n"
    "2. has_roof_damage：当前画面里是否能看到明显的屋面板掀翻卷曲、屋面板缺失破洞或暴露的屋架/檩条。\n"
    "   - 如果整个可见屋面都是平整连续的深色/灰色波纹板，没有破洞或掀翻，has_roof_damage 必须为 false。\n"
    "3. damage_u_start 与 damage_u_end（0.0 到 1.0 的图像水平归一化坐标，0.0=画面最左边缘，1.0=画面最右边缘）：\n"
    "   - 若 has_roof_damage 为 true，估计画面中屋面受损区（从最左端掀翻/破口边缘到最右端破口边缘）在图像水平方向所占的起止比例，满足 0.0 <= damage_u_start < damage_u_end <= 1.0；\n"
    "   - 若 has_roof_damage 为 false，两个都填 0.0。\n"
    "4. 严格按指定 JSON Schema 输出简体中文。"
)


def line_point_ue(line_spec: dict[str, Any], s_m: float) -> np.ndarray:
    o = np.asarray(line_spec["building_origin_ue_m"], dtype=float)
    ex = np.asarray(line_spec["axis_x_ue"], dtype=float)
    ey = np.asarray(line_spec["axis_y_ue"], dtype=float)
    ex = ex / np.linalg.norm(ex)
    ey = ey / np.linalg.norm(ey)
    lx = float(line_spec["local_x_at_s0_m"]) + float(s_m)
    pt = o + lx * ex + float(line_spec["eave_y_offset_m"]) * ey
    pt[2] = float(line_spec["eave_z_ue_m"])
    return pt


def flight_point_ned(line_spec: dict[str, Any], s_m: float) -> tuple[list[float], float]:
    o = np.asarray(line_spec["building_origin_ue_m"], dtype=float)
    ex = np.asarray(line_spec["axis_x_ue"], dtype=float)
    ey = np.asarray(line_spec["axis_y_ue"], dtype=float)
    ex = ex / np.linalg.norm(ex)
    ey = ey / np.linalg.norm(ey)
    lx = float(line_spec["local_x_at_s0_m"]) + float(s_m)
    p_ue = o + lx * ex + float(line_spec["flight_standoff_y_m"]) * ey
    p_ned = [float(p_ue[0]), float(p_ue[1]), -float(line_spec["flight_z_ue_m"])]
    # Look straight toward the roof eave (-ey direction)
    yaw_deg = math.degrees(math.atan2(-float(ey[1]), -float(ey[0])))
    return p_ned, yaw_deg


def project_line_point(
    *,
    pt_ue: np.ndarray,
    cam_pos_ned: list[float] | np.ndarray,
    cam_rot: np.ndarray,
    rgb_wh: tuple[int, int] = (1920, 1080),
    fov_deg: float = 90.0,
) -> dict[str, Any]:
    w, h = int(rgb_wh[0]), int(rgb_wh[1])
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    pt_ned = np.array([float(pt_ue[0]), float(pt_ue[1]), -float(pt_ue[2])], dtype=float)
    v_cam = cam_rot.T @ (pt_ned - np.asarray(cam_pos_ned, dtype=float))
    z_cam = float(v_cam[0])
    if z_cam <= 0.25:
        return {"in_frame": False, "u_norm": None, "v_norm": None, "depth_m": z_cam, "gsd_cm_px": None}
    u_norm = (fx * (float(v_cam[1]) / z_cam) + w / 2.0) / w
    v_norm = (fx * (float(v_cam[2]) / z_cam) + h / 2.0) / h
    in_frame = (0.04 <= u_norm <= 0.96) and (0.04 <= v_norm <= 0.96)
    gsd_cm_px = (z_cam / fx) * 100.0
    return {
        "in_frame": in_frame,
        "u_norm": float(u_norm),
        "v_norm": float(v_norm),
        "depth_m": z_cam,
        "gsd_cm_px": float(gsd_cm_px),
    }


def evaluate_line_coverage_and_damage(
    *,
    frames: list[dict[str, Any]],
    line_spec: dict[str, Any],
    gt_damage_intervals: list[list[float]],
    rgb_wh: tuple[int, int] = (1920, 1080),
    fov_deg: float = 90.0,
    vote_threshold: float = 0.45,
) -> dict[str, Any]:
    """Compute line coverage ratio, missed segment count, and damage start/end localization error."""
    s_min = float(line_spec["s_min_m"])
    s_max = float(line_spec["s_max_m"])
    bin_m = float(line_spec.get("coverage_bin_m", 0.5))
    max_gsd = float(line_spec.get("max_gsd_cm_per_px", 3.0))

    n_bins = int(round((s_max - s_min) / bin_m))
    centers = [s_min + (i + 0.5) * bin_m for i in range(n_bins)]
    bin_pts = [line_point_ue(line_spec, s) for s in centers]

    seen_counts = np.zeros(n_bins, dtype=int)
    dmg_votes = np.zeros(n_bins, dtype=int)

    for fr in frames:
        cam_p = fr["capture"]["cam_pos_ned"]
        cam_rot = quat_wxyz_to_rot(*fr["capture"]["cam_rot_wxyz"])
        vlm_res = fr.get("vlm_result") or {}
        has_dmg = bool(vlm_res.get("has_roof_damage", False))
        u0 = float(vlm_res.get("damage_u_start", 0.0))
        u1 = float(vlm_res.get("damage_u_end", 0.0))
        if u1 < u0:
            u0, u1 = u1, u0

        for idx, pt_ue in enumerate(bin_pts):
            proj = project_line_point(pt_ue=pt_ue, cam_pos_ned=cam_p, cam_rot=cam_rot, rgb_wh=rgb_wh, fov_deg=fov_deg)
            if proj["in_frame"] and proj["gsd_cm_px"] is not None and proj["gsd_cm_px"] <= max_gsd:
                seen_counts[idx] += 1
                if has_dmg and proj["u_norm"] is not None and (u0 <= proj["u_norm"] <= u1):
                    dmg_votes[idx] += 1

    covered = seen_counts > 0
    coverage_ratio = float(covered.mean()) if n_bins > 0 else 0.0

    # Count contiguous uncovered segments
    missed_segments: list[list[float]] = []
    in_miss = False
    miss_start = 0
    for idx, cov in enumerate(covered):
        if not cov and not in_miss:
            in_miss = True
            miss_start = idx
        elif cov and in_miss:
            in_miss = False
            missed_segments.append([round(s_min + miss_start * bin_m, 2), round(s_min + idx * bin_m, 2)])
    if in_miss:
        missed_segments.append([round(s_min + miss_start * bin_m, 2), round(s_max, 2)])

    # Extract predicted damage intervals by vote fraction
    pred_flags = np.zeros(n_bins, dtype=bool)
    for idx in range(n_bins):
        if seen_counts[idx] > 0 and (dmg_votes[idx] / seen_counts[idx]) >= vote_threshold:
            pred_flags[idx] = True

    pred_intervals: list[list[float]] = []
    in_d = False
    d_start = 0
    for idx, flag in enumerate(pred_flags):
        if flag and not in_d:
            in_d = True
            d_start = idx
        elif not flag and in_d:
            in_d = False
            pred_intervals.append([round(s_min + d_start * bin_m, 2), round(s_min + idx * bin_m, 2)])
    if in_d:
        pred_intervals.append([round(s_min + d_start * bin_m, 2), round(s_max, 2)])

    # Compare against GT damage intervals (primary interval error + bin-level IoU)
    gt_flags = np.zeros(n_bins, dtype=bool)
    for idx, s in enumerate(centers):
        if any(a <= s <= b for a, b in gt_damage_intervals):
            gt_flags[idx] = True

    inter = int(np.logical_and(pred_flags, gt_flags).sum())
    union = int(np.logical_or(pred_flags, gt_flags).sum())
    iou = (inter / union) if union > 0 else 1.0

    start_err_m = None
    end_err_m = None
    length_err_m = None
    if gt_damage_intervals and pred_intervals:
        gt0 = gt_damage_intervals[0]
        # Pick longest predicted interval (or merged span)
        pred_main = max(pred_intervals, key=lambda iv: iv[1] - iv[0])
        start_err_m = round(abs(pred_main[0] - gt0[0]), 2)
        end_err_m = round(abs(pred_main[1] - gt0[1]), 2)
        length_err_m = round(abs((pred_main[1] - pred_main[0]) - (gt0[1] - gt0[0])), 2)

    return {
        "line_length_m": round(s_max - s_min, 2),
        "n_bins": n_bins,
        "bin_size_m": bin_m,
        "effective_coverage_ratio": round(coverage_ratio, 4),
        "missed_segment_count": len(missed_segments),
        "missed_segments_m": missed_segments,
        "mean_views_per_bin": round(float(seen_counts.mean()), 2),
        "gt_damage_intervals_m": gt_damage_intervals,
        "predicted_damage_intervals_m": pred_intervals,
        "damage_start_error_m": start_err_m,
        "damage_end_error_m": end_err_m,
        "damage_length_error_m": length_err_m,
        "damage_interval_iou": round(iou, 4),
        "bin_seen_counts": seen_counts.tolist(),
        "bin_damage_votes": dmg_votes.tolist(),
    }


def analyze_line_frame_rgbd(
    *,
    capture: dict[str, Any],
    line_spec: dict[str, Any],
    rgb_wh: tuple[int, int] = (1920, 1080),
    fov_deg: float = 90.0,
    residual_thresh_m: float = 0.45,
) -> tuple[dict[str, Any], float]:
    """Standalone onboard roof-damage span detector using RGB-D depth residuals along the reference line.

    Requires zero VLM / zero server / zero GPU memory:
    - Intact roof plane has nearly constant residual `d_meas - dist_3d` along the slope (~+0.18 m).
    - Lifted/folded roof panels protrude toward the camera (negative residual, < baseline - 0.45 m).
    - Missing roof panels expose the building interior floor (large positive residual, > baseline + 0.45 m).
    """
    t0 = time.perf_counter()
    depth = np.load(capture["npz_path"])["depth_m"].astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    dh, dw = depth.shape[:2]
    cam_p = np.asarray(capture["cam_pos_ned"], dtype=float)
    cam_rot = quat_wxyz_to_rot(*capture["cam_rot_wxyz"])

    s_min = float(line_spec["s_min_m"])
    s_max = float(line_spec["s_max_m"])
    samples: list[tuple[float, float]] = []  # (u_norm, residual_m)
    for s in np.linspace(s_min, s_max, 97):
        pt_ue = line_point_ue(line_spec, float(s))
        pr = project_line_point(pt_ue=pt_ue, cam_pos_ned=cam_p, cam_rot=cam_rot, rgb_wh=rgb_wh, fov_deg=fov_deg)
        if not pr["in_frame"] or pr["u_norm"] is None or pr["v_norm"] is None:
            continue
        dist_3d = float(np.linalg.norm(ue_to_ned(pt_ue) - cam_p))
        du = int(np.clip(round(pr["u_norm"] * dw), 2, dw - 3))
        dv = int(np.clip(round(pr["v_norm"] * dh), 3, dh - 4))
        patch = depth[dv - 3 : dv + 4, du - 2 : du + 3]
        valid = patch[np.isfinite(patch) & (patch > 1.0) & (patch < 150.0)]
        if valid.size:
            samples.append((float(pr["u_norm"]), float(np.median(valid)) - dist_3d))

    if len(samples) < 4:
        lat = (time.perf_counter() - t0) * 1000.0
        return (
            {
                "roof_visible": False,
                "has_roof_damage": False,
                "damage_u_start": 0.0,
                "damage_u_end": 0.0,
                "damage_description": "Insufficient roof depth samples.",
            },
            lat,
        )

    residuals = np.array([r for _, r in samples], dtype=float)
    # Robust intact-plane baseline: median of points closest to the mode (ignoring huge breach jumps)
    normal_like = residuals[np.abs(residuals) < 0.55]
    baseline = float(np.median(normal_like)) if normal_like.size >= 3 else float(np.median(residuals))
    flags = [abs(r - baseline) > residual_thresh_m for _, r in samples]

    # Extract contiguous anomalous runs (min 3 consecutive samples, ignoring 1-sample gable-end drops)
    runs: list[tuple[int, int]] = []
    start_idx: int | None = None
    for i, flg in enumerate(flags):
        if flg and start_idx is None:
            start_idx = i
        elif not flg and start_idx is not None:
            if i - start_idx >= 3:
                runs.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None and len(flags) - start_idx >= 3:
        runs.append((start_idx, len(flags) - 1))

    if runs:
        # Pick the dominant contiguous damage run visible in this frame
        a, b = max(runs, key=lambda ab: ab[1] - ab[0])
        us_run = [samples[k][0] for k in range(a, b + 1)]
        u0, u1, has_d = min(us_run), max(us_run), True
        desc = f"RGB-D 深度平面异常（基线 {baseline:+.2f}m，连续异常 {b - a + 1} 点）"
    else:
        u0, u1, has_d = 0.0, 0.0, False
        desc = f"RGB-D 屋面深度平稳（基线 {baseline:+.2f}m）"

    lat = (time.perf_counter() - t0) * 1000.0
    return (
        {
            "roof_visible": True,
            "has_roof_damage": has_d,
            "damage_u_start": round(float(u0), 3),
            "damage_u_end": round(float(u1), 3),
            "damage_description": desc,
        },
        lat,
    )


def analyze_line_frame(
    *,
    vlm_client: OllamaVLMClient | None,
    neutral_id: str,
    frame_idx: int,
    image_path: Path,
) -> VLMResponse:
    img = VLMImage(image_id=f"{neutral_id}-F{frame_idx:02d}", view_type="沿屋檐连续巡检帧", image_path=image_path)
    user_prompt = (
        f"巡检线编号：{neutral_id}，第 {frame_idx + 1} 帧（图像编号={img.image_id}）。\n"
        "请检测当前帧仓库斜屋面是否存在掀翻、缺失破口或露梁损伤，并给出受损区在画面水平方向 [0.0, 1.0] 的起止比例。"
    )
    if vlm_client is None:
        t0 = time.perf_counter()
        res = {
            "roof_visible": True,
            "has_roof_damage": False,
            "damage_u_start": 0.0,
            "damage_u_end": 0.0,
            "damage_description": "Mock intact roof.",
        }
        return VLMResponse(res, json.dumps(res, ensure_ascii=False), "mock", (time.perf_counter() - t0) * 1000.0)
    parsed, raw_http, latency_ms = vlm_client._call_chat(
        LINE_FRAME_SYSTEM_PROMPT, user_prompt, [img], LINE_FRAME_SCHEMA
    )
    has_dmg = bool(parsed.get("has_roof_damage", False))
    u0 = float(np.clip(float(parsed.get("damage_u_start", 0.0)), 0.0, 1.0)) if has_dmg else 0.0
    u1 = float(np.clip(float(parsed.get("damage_u_end", 0.0)), 0.0, 1.0)) if has_dmg else 0.0
    if has_dmg and u1 <= u0 + 0.02:
        u0, u1 = 0.25, 0.75
    cleaned = {
        "roof_visible": bool(parsed.get("roof_visible", True)),
        "has_roof_damage": has_dmg,
        "damage_u_start": round(u0, 3),
        "damage_u_end": round(u1, 3),
        "damage_description": str(parsed.get("damage_description", "")).strip(),
    }
    return VLMResponse(cleaned, raw_http, vlm_client.model_name, latency_ms)


def _draw_strip_summary(frames: list[dict[str, Any]], metrics: dict[str, Any], out_path: Path) -> None:
    tiles = []
    for fr in frames:
        img = cv2.imread(fr["capture"]["jpg_path"])
        if img is None:
            continue
        tile = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
        v = fr.get("vlm_result") or {}
        if v.get("has_roof_damage"):
            x0 = int(round(v["damage_u_start"] * 480))
            x1 = int(round(v["damage_u_end"] * 480))
            cv2.rectangle(tile, (x0, 34), (x1, 266), (0, 60, 255), 3)
        cv2.rectangle(tile, (0, 0), (480, 30), (20, 20, 20), -1)
        lbl = f"s={fr['s_target_m']:.1f}m dmg={v.get('has_roof_damage')} [{v.get('damage_u_start',0):.2f},{v.get('damage_u_end',0):.2f}]"
        cv2.putText(tile, lbl, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = int(math.ceil(len(tiles) / cols))
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


async def run_line_inspection(
    *,
    cfg: dict[str, Any],
    address: str,
    out_dir: Path,
    vlm_client: OllamaVLMClient | None,
    detector: str = "rgbd",
    replay_from: Path | None = None,
    record_video: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = cfg["line_spec"]
    t0 = time.monotonic()
    video_info: dict[str, Any] | None = None

    if replay_from is not None:
        prev = json.loads((replay_from / "summary.json").read_text(encoding="utf-8"))
        frames = [
            {
                "frame_index": fr["frame_index"],
                "s_target_m": fr["s_target_m"],
                "t_s": fr["t_s"],
                "capture": fr["capture"],
            }
            for fr in prev["frames"]
        ]
        flight_duration_s = float(prev.get("flight_duration_s", 0.0))
        col_delta = int(prev.get("collisions", 0))
    else:
        from projectairsim import Drone, ProjectAirSimClient, World

        configure_client_log(out_dir)
        scene_path = project_path(cfg["scene_config"])
        runtime_scene = _load_harbor_runtime_scene(scene_path, enable_depth_stream=False)
        (out_dir / "runtime-scene.json").write_text(json.dumps(runtime_scene, indent=2), encoding="utf-8")

        s_min = float(spec["s_min_m"])
        s_max = float(spec["s_max_m"])
        spacing = float(spec["sample_spacing_m"])
        n_pts = int(round((s_max - s_min) / spacing)) + 1
        s_samples = [round(s_min + (s_max - s_min) * i / max(n_pts - 1, 1), 2) for i in range(n_pts)]
        pitch_deg = float(spec["camera_pitch_deg"])
        speed_mps = float(spec["cruise_speed_mps"])

        client = ProjectAirSimClient(address)
        collision_count = 0
        frames = []

        try:
            client.connect()
            client.socket_services.recv_timeout = 60000
            world = World.__new__(World)
            world.client = client
            world.sim_instance_idx = -1
            world.load_scene(runtime_scene, delay_after_load_sec=2)
            drone = Drone(client, world, cfg["drone_id"])
            if "collision_info" in drone.robot_info:
                def _on_col(_t: str, _m: dict[str, Any]) -> None:
                    nonlocal collision_count
                    collision_count += 1
                client.subscribe(drone.robot_info["collision_info"], _on_col)

            drone.enable_api_control()
            drone.arm()
            try:
                await _await_task(lambda cb: drone.takeoff_async(timeout_sec=20, callback=cb), timeout_s=28.0)
            except Exception:
                pass

            start_ned, yaw_deg = flight_point_ned(spec, s_samples[0])
            await _stage_drone_at_start(drone, start_ned, yaw_deg, pitch_deg)
            col_before = collision_count
            t_fly = time.monotonic()
            rec: HarborVideoRecorder | None = None
            if record_video:
                rec = HarborVideoRecorder(
                    out_dir / "line_inspection.mp4",
                    title=f"任务3｜{cfg['target_id']} 沿屋檐连续巡检({detector})",
                )
                rec.set_badge(f"全长 {float(spec['s_max_m']) - float(spec['s_min_m']):.1f}m")
                rec.set_status(cfg["neutral_id"], "沿线巡检启动 s=0.0m")
                rec.attach_streams(client, drone)
                rec.start()

            try:
                for idx, s_val in enumerate(s_samples):
                    pt_ned, pt_yaw_deg = flight_point_ned(spec, s_val)
                    if rec is not None:
                        rec.set_status(cfg["neutral_id"], f"沿线飞往 s={s_val:.1f}m ({idx+1}/{len(s_samples)})")
                    if idx > 0:
                        await _await_task(
                            lambda cb, p=pt_ned: drone.move_to_position_async(
                                p[0], p[1], p[2], speed_mps, timeout_sec=18.0, callback=cb
                            ),
                            timeout_s=25.0,
                        )
                        try:
                            await _await_task(
                                lambda cb, y=math.radians(pt_yaw_deg): drone.rotate_to_yaw_async(y, timeout_sec=8.0, callback=cb),
                                timeout_s=12.0,
                            )
                        except Exception:
                            pass
                        await asyncio.sleep(0.35)

                    cap = _capture_rgbd(drone, cfg["camera_id"], out_dir / "frames" / f"frame_{idx:02d}_s{s_val:.1f}m")
                    cap_clean = _strip_runtime_arrays(cap)
                    fr_item: dict[str, Any] = {
                        "frame_index": idx,
                        "s_target_m": s_val,
                        "t_s": round(time.monotonic() - t_fly, 2),
                        "capture": cap_clean,
                    }
                    if detector == "rgbd":
                        live_res, live_lat = analyze_line_frame_rgbd(
                            capture=cap_clean,
                            line_spec=spec,
                            rgb_wh=tuple(cfg["rgb_resolution"]),
                            fov_deg=float(cfg["camera_fov_deg"]),
                        )
                        fr_item["vlm_latency_ms"] = round(live_lat, 1)
                        fr_item["vlm_result"] = live_res
                        if rec is not None:
                            rec.set_line_overlay(live_res)
                            rec.set_status(
                                cfg["neutral_id"],
                                f"s={s_val:.1f}m {'破损检测!' if live_res['has_roof_damage'] else '屋面完好'}",
                            )
                    frames.append(fr_item)
                    print(f"[LineInspect] frame {idx+1}/{len(s_samples)} s={s_val:.1f}m pos={cap['drone_pos_ned']}", flush=True)
            finally:
                if rec is not None:
                    video_info = rec.stop()

            flight_duration_s = round(time.monotonic() - t_fly, 2)
            col_delta = collision_count - col_before
        finally:
            client.disconnect()

    # Run per-frame damage-span detection (skip frames already analyzed live in rgbd mode)
    for fr in frames:
        if "vlm_result" in fr and detector == "rgbd":
            res = fr["vlm_result"]
            lat_ms = float(fr.get("vlm_latency_ms", 0.0))
            print(
                f"  [RGB-D F{fr['frame_index']:02d} s={fr['s_target_m']:.1f}m] "
                f"dmg={res['has_roof_damage']} span=[{res['damage_u_start']}, {res['damage_u_end']}] ({lat_ms:.1f}ms)",
                flush=True,
            )
            continue
        if detector == "rgbd":
            res, lat_ms = analyze_line_frame_rgbd(
                capture=fr["capture"],
                line_spec=spec,
                rgb_wh=tuple(cfg["rgb_resolution"]),
                fov_deg=float(cfg["camera_fov_deg"]),
            )
            fr["vlm_latency_ms"] = round(lat_ms, 1)
            fr["vlm_result"] = res
            print(
                f"  [RGB-D F{fr['frame_index']:02d} s={fr['s_target_m']:.1f}m] "
                f"dmg={res['has_roof_damage']} span=[{res['damage_u_start']}, {res['damage_u_end']}] ({lat_ms:.1f}ms)",
                flush=True,
            )
        else:
            resp = analyze_line_frame(
                vlm_client=vlm_client,
                neutral_id=cfg["neutral_id"],
                frame_idx=fr["frame_index"],
                image_path=Path(fr["capture"]["jpg_path"]),
            )
            fr["vlm_latency_ms"] = round(resp.latency_ms, 1)
            fr["vlm_result"] = resp.result
            print(
                f"  [VLM F{fr['frame_index']:02d} s={fr['s_target_m']:.1f}m] "
                f"dmg={resp.result['has_roof_damage']} span=[{resp.result['damage_u_start']}, {resp.result['damage_u_end']}] "
                f"({resp.latency_ms:.0f}ms)",
                flush=True,
            )

    metrics = evaluate_line_coverage_and_damage(
        frames=frames,
        line_spec=spec,
        gt_damage_intervals=cfg["evaluation_gt"]["damage_intervals_m"],
        rgb_wh=tuple(cfg["rgb_resolution"]),
        fov_deg=float(cfg["camera_fov_deg"]),
    )
    _draw_strip_summary(frames, metrics, out_dir / "line_inspection_grid.jpg")

    report = {
        "finished_at": datetime.now().isoformat(),
        "address": address,
        "target_id": cfg["target_id"],
        "detector": detector,
        "vlm_model": "none (onboard_rgbd)" if detector == "rgbd" else (vlm_client.model_name if vlm_client else "mock"),
        "flight_duration_s": flight_duration_s,
        "total_wall_time_s": round(time.monotonic() - t0, 2),
        "num_frames": len(frames),
        "collisions": int(col_delta),
        "video": video_info,
        "metrics": metrics,
        "frames": frames,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# 海港任务 3：沿线连续巡检报告",
        "",
        f"- 巡检对象：`{cfg['target_id']}`（沿屋檐 0.0 – {metrics['line_length_m']} m）",
        f"- 检测方式：`{detector}`（模型：`{report['vlm_model']}`）",
        f"- 飞行耗时：`{flight_duration_s} s`，采集帧数：`{len(frames)}`，空中碰撞：`{col_delta}`",
        f"- **有效覆盖率**：`{metrics['effective_coverage_ratio']*100:.1f}%`（平均每 0.5 m 区段被 `{metrics['mean_views_per_bin']}` 帧覆盖）",
        f"- **漏拍区段数**：`{metrics['missed_segment_count']}`（`{metrics['missed_segments_m']}`）",
        f"- **损伤区间真值**：`{metrics['gt_damage_intervals_m']}` m",
        f"- **损伤区间估计**：`{metrics['predicted_damage_intervals_m']}` m",
        f"- **起点误差 / 终点误差 / IoU**：`{metrics['damage_start_error_m']} m` / `{metrics['damage_end_error_m']} m` / `{metrics['damage_interval_iou']}`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coastal Harbor Task 3: Continuous Linear Eave Inspection")
    p.add_argument("--config", type=Path, default=DEFAULT_LINE_CONFIG)
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--detector",
        choices=("rgbd", "vlm"),
        default="rgbd",
        help="Per-frame damage span detector: 'rgbd' (standalone onboard depth residual, zero server) or 'vlm'.",
    )
    p.add_argument(
        "--replay-from",
        type=Path,
        default=None,
        help="Optional existing Task-3 run directory to replay offline without connecting to UE.",
    )
    p.add_argument("--vlm", choices=("none", "ollama", "mock"), default="none")
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    vlm_client = None
    if args.detector == "vlm" and args.vlm == "ollama":
        vlm_client = OllamaVLMClient(
            model_name=args.model,
            base_url=args.ollama_url,
            ssh_host=args.ssh_host.strip() if args.ssh_host else None,
            ssh_key=args.ssh_key.strip() if args.ssh_host else None,
            think=args.think,
            temperature=0.1,
            timeout_s=110.0,
        )
    asyncio.run(
        run_line_inspection(
            cfg=cfg,
            address=args.address,
            out_dir=args.out,
            vlm_client=vlm_client,
            detector=args.detector,
            replay_from=args.replay_from,
            record_video=args.record_video,
        )
    )


if __name__ == "__main__":
    main()
