"""Coastal Harbor Task 2: Roof Breach Measurement, Passability Decision & Indoor Entry."""

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
from coordination.visual_servo import run_opening_visual_servo
from coordination.vlm import OllamaVLMClient, VLMImage
from harbor.part_recheck import (
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_KEY,
    _await_task,
    _camera_pose,
    _capture_rgbd,
    _load_harbor_runtime_scene,
    _stage_drone_at_start,
    _strip_runtime_arrays,
)
from harbor.recorder import HarborVideoRecorder
from perception.opening_geometry import opening_candidates


DEFAULT_INDOOR_CONFIG = CONFIGS / "missions" / "coastal_harbor_indoor_entry.json"

INTERIOR_VLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "inside_building": {"type": "boolean"},
        "visible_interior_structures": {"type": "array", "items": {"type": "string"}},
        "roof_damage_confirmed": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["inside_building", "visible_interior_structures", "roof_damage_confirmed", "summary"],
}

INTERIOR_VLM_PROMPT = (
    "你是灾后无人机室内勘测评估员。你正在查看无人机从仓库屋顶破口垂直入室后拍摄的室内前视/斜视图像。\n"
    "请严格依据画面回答：\n"
    "1. inside_building：当前画面是否明确位于建筑内部空间。\n"
    "2. visible_interior_structures：列出画面中可见的室内构件（如内墙、窗户、地面、大门、屋架/梁、光线透入处等）。\n"
    "3. roof_damage_confirmed：结合室内透光/破口或屋面视图，是否证实存在屋顶破损缺失。\n"
    "4. 严格按 JSON Schema 输出简体中文。"
)


def measure_breach_from_nadir_depth(
    depth_arr: np.ndarray,
    *,
    fov_deg: float = 90.0,
    min_layer_gap_m: float = 5.0,
    min_passable_clearance_radius_m: float = 1.6,
) -> dict[str, Any]:
    """Measure roof-breach width, missing area, and largest unobstructed clearance radius from nadir RGB-D."""
    d = np.asarray(depth_arr, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    geom = opening_candidates(d, fov_degrees=fov_deg, min_layer_gap_m=min_layer_gap_m, down=None)
    cands = geom.get("candidates") or []
    if not cands:
        return {
            "detected": False,
            "status": geom.get("status", "no_opening"),
            "passable": False,
            "near_layer_depth_m": geom.get("near_layer_depth_m"),
            "far_layer_depth_m": geom.get("far_layer_depth_m"),
            "measured_width_m": 0.0,
            "measured_length_m": 0.0,
            "measured_area_m2": 0.0,
            "max_clearance_radius_m": 0.0,
            "num_sub_cells": 0,
        }

    near_d = float(geom["near_layer_depth_m"])
    focal = float(geom["focal_px"])
    m_per_px = near_d / focal

    # Combine all enclosed opening candidates belonging to the roof breach to compute total missing area & bbox
    best = max(cands, key=lambda c: float(c.get("projected_clearance_m", 0.0)))
    max_clr = float(best["projected_clearance_m"])
    union_mask = np.zeros(d.shape[:2], dtype=bool)
    for c in cands:
        m = c.get("visible_gap_mask")
        if m is not None:
            union_mask |= np.asarray(m, dtype=bool)

    ys, xs = np.where(union_mask)
    if xs.size == 0:
        span_w = span_l = area_m2 = 0.0
    else:
        dx_m = float(xs.max() - xs.min() + 1) * m_per_px
        dy_m = float(ys.max() - ys.min() + 1) * m_per_px
        span_w = min(dx_m, dy_m)
        span_l = max(dx_m, dy_m)
        area_m2 = float(union_mask.sum()) * (m_per_px ** 2)

    passable = bool(max_clr >= min_passable_clearance_radius_m)
    return {
        "detected": True,
        "status": geom.get("status", "ok"),
        "passable": passable,
        "decision": "enter" if passable else "blocked_abort",
        "near_layer_depth_m": round(near_d, 3),
        "far_layer_depth_m": round(float(geom["far_layer_depth_m"]), 3),
        "measured_width_m": round(span_w, 3),
        "measured_length_m": round(span_l, 3),
        "measured_area_m2": round(area_m2, 2),
        "max_clearance_radius_m": round(max_clr, 3),
        "max_clearance_diameter_m": round(max_clr * 2.0, 3),
        "num_sub_cells": len(cands),
        "best_aperture_uv": [round(float(v), 1) for v in best["aperture_point_depth_xy"]],
        "union_mask": union_mask,
    }


def score_breach_measurement(meas: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    w_gt = float(gt["open_bay_width_m"])
    a_gt = float(gt["open_bay_area_m2"])
    w_err = abs(float(meas["measured_width_m"]) - w_gt)
    a_err = abs(float(meas["measured_area_m2"]) - a_gt)
    return {
        "gt_width_m": w_gt,
        "gt_area_m2": a_gt,
        "width_abs_error_m": round(w_err, 3),
        "width_rel_error_pct": round(w_err / max(w_gt, 1e-6) * 100.0, 2),
        "area_abs_error_m2": round(a_err, 2),
        "area_rel_error_pct": round(a_err / max(a_gt, 1e-6) * 100.0, 2),
    }


async def run_indoor_task(
    *,
    cfg: dict[str, Any],
    address: str,
    out_dir: Path,
    mode: str,
    selected_targets: set[str] | None,
    min_clearance_override: float | None,
    vlm_client: OllamaVLMClient | None,
    record_video: bool = False,
) -> dict[str, Any]:
    from projectairsim import Drone, ProjectAirSimClient, World

    out_dir.mkdir(parents=True, exist_ok=True)
    configure_client_log(out_dir)
    scene_path = project_path(cfg["scene_config"])
    runtime_scene = _load_harbor_runtime_scene(scene_path, enable_depth_stream=(mode == "enter"))
    (out_dir / "runtime-scene.json").write_text(json.dumps(runtime_scene, indent=2), encoding="utf-8")

    policy = cfg["passability_policy"]
    min_clr = float(min_clearance_override if min_clearance_override is not None else policy["min_passable_clearance_radius_m"])
    min_gap = float(policy["min_layer_gap_m"])
    edef = cfg["entry_defaults"]

    client = ProjectAirSimClient(address)
    collision_flags: dict[str, int] = {cfg["drone_id"]: 0}
    results: list[dict[str, Any]] = []

    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        client.socket_topics.recv_buffer_size = 24
        world = World.__new__(World)
        world.client = client
        world.sim_instance_idx = -1
        world.load_scene(runtime_scene, delay_after_load_sec=2)
        drone_id = cfg["drone_id"]
        drone = Drone(client, world, drone_id)
        if "collision_info" in drone.robot_info:
            def _on_col(_t: str, _m: dict[str, Any]) -> None:
                collision_flags[drone_id] = collision_flags.get(drone_id, 0) + 1
            client.subscribe(drone.robot_info["collision_info"], _on_col)

        drone.enable_api_control()
        drone.arm()
        try:
            await _await_task(lambda cb: drone.takeoff_async(timeout_sec=20, callback=cb), timeout_s=28.0)
        except Exception:
            pass

        for tgt in cfg["targets"]:
            tid = tgt["target_id"]
            if selected_targets and tid not in selected_targets:
                continue
            tdir = out_dir / tid
            tdir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== [Task 2 Indoor] {tid} (mode={mode}, min_clr={min_clr}m) ===", flush=True)
            await _stage_drone_at_start(drone, tgt["start_hover_ned"], tgt["start_yaw_deg"], -90.0)
            nadir_cap = _capture_rgbd(drone, cfg["camera_id"], tdir / "nadir_overview")
            meas = measure_breach_from_nadir_depth(
                nadir_cap["depth_arr"],
                fov_deg=float(cfg["camera_fov_deg"]),
                min_layer_gap_m=min_gap,
                min_passable_clearance_radius_m=min_clr,
            )
            mask = meas.pop("union_mask", None)
            if mask is not None:
                # Save visualization overlay on nadir RGB
                rgb_vis = cv2.imread(nadir_cap["jpg_path"])
                if rgb_vis is not None:
                    m_up = cv2.resize(mask.astype(np.uint8) * 255, (rgb_vis.shape[1], rgb_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                    overlay = rgb_vis.copy()
                    overlay[m_up > 0] = (0, 220, 120)
                    blended = cv2.addWeighted(rgb_vis, 0.65, overlay, 0.35, 0)
                    cv2.imwrite(str(tdir / "nadir_breach_overlay.jpg"), blended, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            err = score_breach_measurement(meas, tgt["evaluation_gt"])
            print(
                f"  measured: w={meas['measured_width_m']}m l={meas['measured_length_m']}m "
                f"area={meas['measured_area_m2']}m2 max_clr={meas['max_clearance_radius_m']}m "
                f"cells={meas['num_sub_cells']} passable={meas['passable']} | "
                f"err_w={err['width_abs_error_m']}m err_a={err['area_abs_error_m2']}m2",
                flush=True,
            )

            entry_log: dict[str, Any] = {"attempted": False, "entered": False, "exited": False}
            if mode == "enter":
                if not meas["passable"]:
                    entry_log = {
                        "attempted": False,
                        "entered": False,
                        "exited": False,
                        "decision": "blocked_abort",
                        "reason": f"max_clearance_radius {meas['max_clearance_radius_m']}m < threshold {min_clr}m",
                    }
                    print(f"  -> BLOCKED_ABORT: {entry_log['reason']}", flush=True)
                else:
                    col_before = collision_flags.get(drone_id, 0)
                    t_enter = time.monotonic()
                    rec: HarborVideoRecorder | None = None
                    video_info: dict[str, Any] | None = None
                    if record_video:
                        rec = HarborVideoRecorder(
                            tdir / "indoor_entry.mp4",
                            title=f"任务2｜{tid} 破口对准与穿顶入室",
                        )
                        rec.set_badge(f"净空半径 {meas['max_clearance_radius_m']:.2f}m")
                        rec.set_status(tgt["neutral_id"], "顶视破口对准中")
                        rec.attach_streams(client, drone)
                        rec.start()
                    try:
                        servo_cfg = {
                            "min_layer_gap_m": min_gap,
                            "min_clearance_m": min_clr,
                            "align_tol_m": float(edef["align_tol_m"]),
                            "align_confirm_frames": int(edef["align_confirm_frames"]),
                            "gain_xy": float(edef["gain_xy"]),
                            "max_xy_mps": float(edef["max_xy_mps"]),
                            "descend_mps": float(edef["descend_mps"]),
                            "target_interior_alt_m": float(edef["target_interior_alt_ue_m"]),
                            "min_floor_guard_m": float(edef["min_floor_guard_m"]),
                            "max_duration_s": float(edef["max_duration_s"]),
                        }
                        sres = await run_opening_visual_servo(
                            client=client,
                            drone=drone,
                            drone_id=drone_id,
                            target_id=tgt["neutral_id"],
                            servo_cfg=servo_cfg,
                            collision_flags=collision_flags,
                            recorder=rec,
                        )
                        (tdir / "opening_servo_trace.json").write_text(
                            json.dumps(sres, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        interior_caps: list[dict[str, Any]] = []
                        vlm_interior = None
                        exited = False
                        actually_inside = bool(
                            sres["entered_interior"]
                            and float(sres["final_alt_m"]) <= float(edef["target_interior_alt_ue_m"]) + 2.5
                        )
                        if actually_inside:
                            if rec is not None:
                                rec.set_status(tgt["neutral_id"], "室内悬停取证")
                            # Capture interior forward view inside the warehouse
                            drone.set_camera_pose(cfg["camera_id"], _camera_pose(-10.0))
                            await asyncio.sleep(0.8)
                            fwd_cap = _capture_rgbd(drone, cfg["camera_id"], tdir / "interior_forward")
                            interior_caps.append({"view": "interior_forward", **_strip_runtime_arrays(fwd_cap)})

                            if vlm_client is not None:
                                img = VLMImage(
                                    image_id=f"{tgt['neutral_id']}-INT-FWD",
                                    view_type="入室前视取证",
                                    image_path=Path(fwd_cap["jpg_path"]),
                                )
                                parsed, _raw, lat = vlm_client._call_chat(
                                    INTERIOR_VLM_PROMPT,
                                    f"目标 {tgt['neutral_id']} 入室前视图像（{img.image_id}），请确认室内构件与屋顶破损。",
                                    [img],
                                    INTERIOR_VLM_SCHEMA,
                                )
                                vlm_interior = {"latency_ms": round(lat, 1), "result": parsed}

                            # Safe vertical exit straight up through the same entry shaft
                            if rec is not None:
                                rec.set_status(tgt["neutral_id"], "垂直原路退出")
                            cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
                            exit_z = -float(edef["exit_alt_ue_m"])
                            drone.set_camera_pose(cfg["camera_id"], _camera_pose(-90.0))
                            await _await_task(
                                lambda cb, x=float(cur_pos["x"]), y=float(cur_pos["y"]), z=exit_z: drone.move_to_position_async(
                                    x, y, z, 1.4, timeout_sec=35.0, callback=cb
                                ),
                                timeout_s=45.0,
                            )
                            after_exit = drone.get_ground_truth_kinematics()["pose"]["position"]
                            exited = (-float(after_exit["z"])) >= float(edef["exit_alt_ue_m"]) - 1.5
                    finally:
                        if rec is not None:
                            video_info = rec.stop()
                    col_delta = collision_flags.get(drone_id, 0) - col_before
                    entry_log = {
                        "attempted": True,
                        "decision": "enter",
                        "stop_reason": sres["stop_reason"],
                        "entered": actually_inside,
                        "exited": bool(exited),
                        "final_interior_alt_m": sres["final_alt_m"],
                        "duration_s": round(time.monotonic() - t_enter, 2),
                        "collisions": int(col_delta),
                        "video": video_info,
                        "interior_captures": interior_caps,
                        "vlm_interior": vlm_interior,
                    }
                    print(
                        f"  -> ENTRY RESULT: entered={entry_log['entered']} exited={entry_log['exited']} "
                        f"stop={sres['stop_reason']} alt={sres['final_alt_m']}m col={col_delta}",
                        flush=True,
                    )

            results.append(
                {
                    "target_id": tid,
                    "nadir_capture": _strip_runtime_arrays(nadir_cap),
                    "measurement": meas,
                    "measurement_error": err,
                    "entry": entry_log,
                }
            )
    finally:
        client.disconnect()

    report = {
        "finished_at": datetime.now().isoformat(),
        "address": address,
        "mode": mode,
        "min_passable_clearance_radius_m": min_clr,
        "targets": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coastal Harbor Task 2: Roof Breach Measurement & Indoor Entry")
    p.add_argument("--config", type=Path, default=DEFAULT_INDOOR_CONFIG)
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--mode", choices=("measure", "enter"), default="measure")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--min-clearance-m", type=float, default=None)
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
    if args.vlm == "ollama" and args.mode == "enter":
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
        run_indoor_task(
            cfg=cfg,
            address=args.address,
            out_dir=args.out,
            mode=args.mode,
            selected_targets=set(args.targets) if args.targets else None,
            min_clearance_override=args.min_clearance_m,
            vlm_client=vlm_client,
            record_video=args.record_video,
        )
    )


if __name__ == "__main__":
    main()
