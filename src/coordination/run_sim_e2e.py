"""ProjectAirSim 6-UAV two-stage end-to-end cooperative survey runner & FPS probe."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import commentjson
import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.paths import CONFIGS, RUNS, SIMULATION_DIR, project_path
from common.runtime import configure_client_log
from common.sensors import decode_image
from coordination.center import SurveyCenter
from coordination.store import utc_now_iso
from coordination.video_recorder import MultiUAVVideoRecorder, format_drone_cn, format_target_cn
from coordination.visual_servo import SharedDinoMatcher, run_dino_visual_servo, run_opening_visual_servo
from coordination.vlm import BaseVLMClient, MockVLMClient, OllamaVLMClient


_VIEW_CN = {
    "nadir": "正俯视",
    "oblique": "斜俯视",
    "closeup_oblique": "近距斜视",
    "interior_closeup": "破口入室",
    "closeup_side": "近距侧视",
}

_ACTION_CN = {
    "closeup_approach": "低空抵近精查",
    "roof_entry": "屋顶破口入室",
    "side_inspection": "侧视近距精查",
    "intact_verification": "近距完好复核",
    "none": "常规复核",
}

_SEVERITY_CN = {
    "intact": "完好",
    "none": "无明显异常",
    "suspected": "疑似受损",
    "minor": "轻度损伤",
    "moderate": "中度损伤",
    "severe": "重度损伤",
    "uncertain": "待定",
}

_VERDICT_CN = {
    "confirmed": "确认受损",
    "ruled_out": "确认完好",
    "still_uncertain": "仍存疑",
    "failed": "评估失败",
}


DEFAULT_MISSION_CONFIG = CONFIGS / "missions" / "coordination_sim_6uav.json"
DEFAULT_SSH_HOST = "ryh2025@192.168.31.237"
DEFAULT_SSH_KEY = "/home/hui/.ssh/id_ed25519_501"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 6-UAV ProjectAirSim FPS probe and/or full two-stage cooperative survey."
    )
    parser.add_argument(
        "--mission-config",
        default=str(DEFAULT_MISSION_CONFIG),
        help="Path to 6-UAV simulation mission JSON.",
    )
    parser.add_argument(
        "--address",
        default="172.21.192.1",
        help="ProjectAirSim transport address (default: 172.21.192.1 for WSL->Windows UE).",
    )
    parser.add_argument(
        "--mission-id",
        default=None,
        help="Mission ID (default: timestamped).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: runs/coordination/<mission-id>).",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only load the 6-UAV scene and measure actual camera FPS/resolution/jitter, then exit.",
    )
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=8.0,
        help="Duration in seconds for live camera FPS probe (default: 8.0).",
    )
    parser.add_argument(
        "--vlm",
        choices=["ollama", "mock"],
        default="ollama",
        help="VLM backend (default: ollama).",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:9b",
        help="Ollama model name (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP base URL.",
    )
    parser.add_argument(
        "--ssh-host",
        default=DEFAULT_SSH_HOST,
        help="SSH host running Ollama; pass '' for direct HTTP.",
    )
    parser.add_argument(
        "--ssh-key",
        default=DEFAULT_SSH_KEY,
        help="SSH private key path.",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable thinking mode on Ollama Qwen3.5 models (default: disabled).",
    )
    parser.add_argument(
        "--dispatch-policy",
        choices=["always_confirm", "only_flagged"],
        default="always_confirm",
        help="Stage-2 dispatch policy (default: always_confirm).",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record 3x2 live multi-UAV overview + 4x highlight MP4 videos (default: enabled).",
    )
    parser.add_argument(
        "--visual-servo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1A serialized GPU visual servo closed-loop control for Batch-2 terminal approach/entry (default: enabled).",
    )
    parser.add_argument(
        "--restore-default-scene",
        action="store_true",
        help="Load empty DefaultScene on exit to unload 6-UAV camera streams.",
    )
    return parser.parse_args(argv)


def _load_scene_bundle(scene_path: Path) -> dict[str, Any]:
    scene = commentjson.loads(scene_path.read_text(encoding="utf-8"))
    for actor in scene.get("actors", []):
        rc = actor.get("robot-config")
        if isinstance(rc, str):
            rc_path = scene_path.parent / rc
            if not rc_path.is_file():
                rc_path = SIMULATION_DIR / rc
            actor["robot-config"] = commentjson.loads(rc_path.read_text(encoding="utf-8"))
    return scene


def _pitch_to_quat(pitch_deg: float) -> dict[str, float]:
    half = math.radians(pitch_deg) / 2.0
    return {"w": math.cos(half), "x": 0.0, "y": math.sin(half), "z": 0.0}


def _camera_pose(pitch_deg: float, x: float = 0.0, y: float = 0.0, z: float = 0.3) -> dict[str, Any]:
    return {
        "translation": {"x": x, "y": y, "z": z},
        "rotation": _pitch_to_quat(pitch_deg),
        "frame_id": "DEFAULT_FRAME",
    }


def _build_vlm(args: argparse.Namespace) -> BaseVLMClient:
    if args.vlm == "mock":
        return MockVLMClient(model_name="mock-vlm-v1")
    ssh_host = args.ssh_host.strip() if args.ssh_host else None
    ssh_key = args.ssh_key.strip() if (ssh_host and args.ssh_key) else None
    return OllamaVLMClient(
        model_name=args.model,
        base_url=args.ollama_url,
        ssh_host=ssh_host,
        ssh_key=ssh_key,
        think=args.think,
        temperature=0.1,
    )


async def _await_drone_task(label: str, task_coro: Any, timeout_s: float = 90.0) -> Any:
    result: list[Any] = []
    task = await task_coro(lambda v: result.append(v))
    await asyncio.wait_for(task, timeout=timeout_s)
    return result[0] if result else None


async def _move_segmented(
    drone: Drone,
    target_ned: tuple[float, float, float],
    speed_mps: float,
    tol_m: float = 2.5,
) -> dict[str, float]:
    pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    start = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
    goal = np.array(target_ned, dtype=float)
    dist = float(np.linalg.norm(goal - start))
    if dist < 0.3:
        return {"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])}
    seg_len = max(15.0, speed_mps * 20.0)
    steps = max(1, int(math.ceil(dist / seg_len)))
    for i in range(1, steps + 1):
        pt = start + (goal - start) * (i / steps)
        seg_dist = float(np.linalg.norm(pt - (start if i == 1 else start + (goal - start) * ((i - 1) / steps))))
        timeout_s = max(25.0, seg_dist / max(speed_mps, 1.0) + 25.0)
        await _await_drone_task(
            "move",
            lambda cb, p=pt, t=timeout_s: drone.move_to_position_async(
                float(p[0]), float(p[1]), float(p[2]), float(speed_mps), timeout_sec=t, callback=cb
            ),
            timeout_s=timeout_s + 10.0,
        )
    final_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    actual = np.array([final_pos["x"], final_pos["y"], final_pos["z"]], dtype=float)
    err = float(np.linalg.norm(actual - goal))
    if err > tol_m:
        # One corrective hop
        await _await_drone_task(
            "move_correct",
            lambda cb: drone.move_to_position_async(
                float(goal[0]), float(goal[1]), float(goal[2]), min(float(speed_mps), 4.0), timeout_sec=30.0, callback=cb
            ),
            timeout_s=40.0,
        )
        final_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    return {"x": float(final_pos["x"]), "y": float(final_pos["y"]), "z": float(final_pos["z"])}


async def _wait_hover_level(drone: Drone, max_wait_s: float = 3.0, speed_thresh_mps: float = 0.25) -> None:
    """Wait until braking pitch/roll settles before snapping nadir/oblique images."""
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        kin = drone.get_ground_truth_kinematics()
        vel = kin.get("twist", {}).get("linear", {})
        speed = math.sqrt(float(vel.get("x", 0.0)) ** 2 + float(vel.get("y", 0.0)) ** 2 + float(vel.get("z", 0.0)) ** 2)
        if speed <= speed_thresh_mps:
            await asyncio.sleep(0.5)
            return
        await asyncio.sleep(0.2)


async def _capture_camera_png(
    drone: Drone,
    camera_id: str,
    save_path: Path,
    settle_s: float = 0.8,
) -> dict[str, Any]:
    await asyncio.sleep(settle_s)
    images = drone.get_images(camera_id, [0])
    msg = images[0]
    rgb, _ = decode_image(msg)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), rgb[:, :, ::-1])
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return {
        "path": str(save_path),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mean": round(float(rgb.mean()), 2),
        "std": round(float(rgb.std()), 2),
        "white_fraction": round(float((rgb.min(axis=2) >= 250).mean()), 4),
        "time_stamp": int(msg.get("time_stamp", 0)),
    }


def _probe_stream_fps(
    client: ProjectAirSimClient,
    drones: dict[str, Drone],
    drone_names: list[str],
    cameras: tuple[str, ...] = ("FrontCamera", "Chase"),
    duration_s: float = 8.0,
) -> dict[str, Any]:
    lock = threading.Lock()
    records: dict[str, list[tuple[float, int, tuple[int, int]]]] = {}
    topics: list[str] = []

    def make_cb(key: str):
        def _cb(_topic: str, msg: dict[str, Any]) -> None:
            now = time.monotonic()
            ts = int(msg.get("time_stamp", 0))
            shape = (int(msg.get("width", 0)), int(msg.get("height", 0)))
            with lock:
                records.setdefault(key, []).append((now, ts, shape))
        return _cb

    for name in drone_names:
        drone = drones[name]
        for cam in cameras:
            if cam in drone.sensors and "scene_camera" in drone.sensors[cam]:
                topic = drone.sensors[cam]["scene_camera"]
                key = f"{name}/{cam}/rgb"
                records[key] = []
                topics.append(topic)
                client.subscribe(topic, make_cb(key))
            if cam == "FrontCamera" and cam in drone.sensors and "depth_camera" in drone.sensors[cam]:
                topic = drone.sensors[cam]["depth_camera"]
                key = f"{name}/{cam}/depth"
                records[key] = []
                topics.append(topic)
                client.subscribe(topic, make_cb(key))

    # Warm up 1.5s then clear buffer so we measure steady-state FPS
    time.sleep(1.5)
    with lock:
        for key in records:
            records[key].clear()
    t0 = time.monotonic()
    time.sleep(duration_s)
    elapsed = time.monotonic() - t0

    for topic in topics:
        try:
            client.unsubscribe(topic)
        except Exception:
            pass

    stats: dict[str, Any] = {}
    with lock:
        for key, items in records.items():
            unique_ts = len({ts for _, ts, _ in items})
            shape = items[-1][2] if items else (0, 0)
            deltas_ms = [
                (items[i][0] - items[i - 1][0]) * 1000.0
                for i in range(1, len(items))
            ]
            stats[key] = {
                "resolution": list(shape),
                "received_frames": len(items),
                "unique_timestamps": unique_ts,
                "received_fps": round(len(items) / elapsed, 2),
                "unique_fps": round(unique_ts / elapsed, 2),
                "p50_interval_ms": round(float(np.median(deltas_ms)), 1) if deltas_ms else None,
                "p95_interval_ms": round(float(np.percentile(deltas_ms, 95)), 1) if deltas_ms else None,
            }
    return {
        "probed_drones": list(drone_names),
        "cameras": list(cameras),
        "duration_s": round(elapsed, 2),
        "streams": stats,
    }


async def _fly_round1_for_target(
    drone: Drone,
    target_cfg: dict[str, Any],
    mission_cfg: dict[str, Any],
    center: SurveyCenter,
    capture_dir: Path,
    collision_flags: dict[str, int],
    recorder: MultiUAVVideoRecorder | None = None,
) -> dict[str, Any]:
    tid = target_cfg["target_id"]
    drone_id = target_cfg["round1_drone_id"]
    cruise_z = -float(target_cfg["cruise_alt_m"])
    cruise_v = float(mission_cfg["cruise_speed_mps"])
    approach_v = float(mission_cfg["approach_speed_mps"])
    vert_v = float(mission_cfg["vertical_speed_mps"])
    home_n, home_e, _ = [float(x) for x in target_cfg["round1_home_ned"]]

    t0 = time.monotonic()
    log: dict[str, Any] = {"target_id": tid, "drone_id": drone_id, "stage": 1, "shots": [], "collisions_before": collision_flags.get(drone_id, 0)}
    if not drone.enable_api_control() or not drone.arm():
        raise RuntimeError(f"Failed to enable/arm {drone_id}")

    if recorder:
        recorder.set_status(tid, f"起飞爬升至 {int(target_cfg['cruise_alt_m'])}m")
    print(f"[{drone_id}->{tid}] Taking off & climbing to {target_cfg['cruise_alt_m']}m", flush=True)
    await _await_drone_task("takeoff", lambda cb: drone.takeoff_async(timeout_sec=25, callback=cb), timeout_s=35)
    await _move_segmented(drone, (home_n, home_e, cruise_z), vert_v)

    for shot in target_cfg["round1_shots"]:
        sn, se, sz = [float(x) for x in shot["position_ned"]]
        yaw_rad = math.radians(float(shot["yaw_deg"]))
        pitch_deg = float(shot["camera_pitch_deg"])
        view_cn = _VIEW_CN.get(shot["view_type"], shot["view_type"])
        if recorder:
            recorder.set_status(tid, f"巡航前往（{int(target_cfg['cruise_alt_m'])}m）→ {view_cn}")
        await _move_segmented(drone, (sn, se, cruise_z), cruise_v)
        if abs(sz - cruise_z) > 1.0:
            if recorder:
                recorder.set_status(tid, f"下降至观测高度 {int(-sz)}m")
            await _move_segmented(drone, (sn, se, sz), approach_v)
        await _await_drone_task("yaw", lambda cb, y=yaw_rad: drone.rotate_to_yaw_async(y, timeout_sec=20, callback=cb), timeout_s=30)
        await _await_drone_task("hover", lambda cb: drone.hover_async(callback=cb), timeout_s=15)
        await _wait_hover_level(drone)
        drone.set_camera_pose("FrontCamera", _camera_pose(pitch_deg))

        if recorder:
            recorder.set_status(tid, f"悬停采图（{view_cn}）")
        image_id = f"{tid}-R1-{shot['suffix']}"
        front_raw = capture_dir / f"{image_id}-front.png"
        chase_raw = capture_dir / f"{image_id}-chase.png"
        front_meta = await _capture_camera_png(drone, "FrontCamera", front_raw, settle_s=1.0)
        chase_meta = await _capture_camera_png(drone, "Chase", chase_raw, settle_s=0.3)

        center.submit_upload(
            target_id=tid,
            stage=1,
            drone_id=drone_id,
            image_path=front_raw,
            image_id=image_id,
            captured_at=utc_now_iso(),
            view_type=shot["view_type"],
            metadata={
                "front_capture": front_meta,
                "chase_visualization_path": str(chase_raw),
                "chase_capture": chase_meta,
                "pose_ned": drone.get_ground_truth_kinematics()["pose"]["position"],
            },
        )
        log["shots"].append({"image_id": image_id, "view_type": shot["view_type"], "front": front_meta, "chase": chase_meta})
        print(f"[{drone_id}->{tid}] Uploaded {image_id} ({shot['view_type']}, {front_meta['width']}x{front_meta['height']}, mean={front_meta['mean']})", flush=True)
        drone.set_camera_pose("FrontCamera", _camera_pose(-30.0))
        if abs(sz - cruise_z) > 1.0:
            await _move_segmented(drone, (sn, se, cruise_z), vert_v)

    if recorder:
        recorder.set_status(tid, f"返航巡航（{int(target_cfg['cruise_alt_m'])}m）")
    print(f"[{drone_id}->{tid}] Returning to base ({home_n}, {home_e})", flush=True)
    await _move_segmented(drone, (home_n, home_e, cruise_z), cruise_v)
    await _move_segmented(drone, (home_n, home_e, -4.0), vert_v)
    await _await_drone_task("land", lambda cb: drone.land_async(timeout_sec=25, callback=cb), timeout_s=35)
    drone.disarm()
    center.report_return(tid, stage=1, drone_id=drone_id, status="returned", note="Landed at home pad")
    if recorder:
        recorder.set_status(tid, "已返航降落")
    log["collisions_after"] = collision_flags.get(drone_id, 0)
    log["flight_duration_s"] = round(time.monotonic() - t0, 2)
    print(f"[{drone_id}->{tid}] Returned & landed in {log['flight_duration_s']}s", flush=True)
    return log


def _select_handover_template(handover: dict[str, Any], suffix: str) -> Path:
    ids = handover.get("round1_image_ids") or []
    paths = handover.get("round1_image_paths") or []
    for img_id, img_path in zip(ids, paths):
        if str(img_id).endswith(f"-{suffix}") and Path(img_path).is_file():
            return Path(img_path)
    for img_path in reversed(paths):
        if Path(img_path).is_file():
            return Path(img_path)
    raise FileNotFoundError(f"No valid Stage-1 handover image found for {handover.get('target_id')}")


async def _fly_round2_for_target(
    client: ProjectAirSimClient,
    drone: Drone,
    target_cfg: dict[str, Any],
    mission_cfg: dict[str, Any],
    handover: dict[str, Any],
    center: SurveyCenter,
    capture_dir: Path,
    servo_log_dir: Path,
    collision_flags: dict[str, int],
    recorder: MultiUAVVideoRecorder | None = None,
    use_visual_servo: bool = False,
    dino_matcher: SharedDinoMatcher | None = None,
    servo_lock: asyncio.Lock | None = None,
) -> dict[str, Any]:
    tid = target_cfg["target_id"]
    drone_id = handover["assigned_drone_id"]
    cruise_z = -float(target_cfg["cruise_alt_m"])
    cruise_v = float(mission_cfg["cruise_speed_mps"])
    approach_v = float(mission_cfg["approach_speed_mps"])
    vert_v = float(mission_cfg["vertical_speed_mps"])
    home_n, home_e, _ = [float(x) for x in target_cfg["round2_home_ned"]]
    servo_cfg = target_cfg.get("visual_servo") if use_visual_servo else None

    t0 = time.monotonic()
    col_before = collision_flags.get(drone_id, 0)
    log: dict[str, Any] = {
        "target_id": tid,
        "drone_id": drone_id,
        "stage": 2,
        "control_mode": f"visual_servo:{servo_cfg['mode']}" if servo_cfg else "waypoint",
        "handover_question": handover["question"],
        "recommended_action": handover["recommended_action"],
        "shots": [],
        "collisions_before": col_before,
    }
    if not drone.enable_api_control() or not drone.arm():
        raise RuntimeError(f"Failed to enable/arm {drone_id}")

    if recorder:
        recorder.set_status(tid, f"起飞爬升至 {int(target_cfg['cruise_alt_m'])}m")
    print(f"[{drone_id}->{tid}] Stage-2 dispatched ({handover['recommended_action']}), taking off to {target_cfg['cruise_alt_m']}m", flush=True)
    await _await_drone_task("takeoff", lambda cb: drone.takeoff_async(timeout_sec=25, callback=cb), timeout_s=35)
    await _move_segmented(drone, (home_n, home_e, cruise_z), vert_v)

    servo_res: dict[str, Any] | None = None
    servo_succeeded = False

    if servo_cfg is not None:
        # 1A Phase 1: All Batch-2 drones concurrently cruise to their visual-servo start points
        sn0, se0, sz0 = [float(x) for x in servo_cfg["start_ned"]]
        if recorder:
            recorder.set_status(tid, f"巡航前往伺服起点（{int(target_cfg['cruise_alt_m'])}m）")
        await _move_segmented(drone, (sn0, se0, cruise_z), cruise_v)
        if recorder:
            recorder.set_status(tid, f"下降至伺服起点（{int(-sz0)}m）")
        await _move_segmented(drone, (sn0, se0, sz0), approach_v)
        await _await_drone_task(
            "yaw",
            lambda cb, y=math.radians(float(servo_cfg.get("start_yaw_deg", 0.0))): drone.rotate_to_yaw_async(y, timeout_sec=20, callback=cb),
            timeout_s=30,
        )
        await _await_drone_task("hover", lambda cb: drone.hover_async(callback=cb), timeout_s=15)
        await _wait_hover_level(drone)
        drone.set_camera_pose("FrontCamera", _camera_pose(float(servo_cfg.get("camera_pitch_deg", -30.0))))

        # 1A Phase 2: Acquire serialized GPU visual-servo lock
        if recorder:
            recorder.set_status(tid, "就位悬停（等待视觉伺服）")
        lock = servo_lock if servo_lock is not None else asyncio.Lock()
        async with lock:
            mode = servo_cfg["mode"]
            print(f"[{drone_id}->{tid}] Starting closed-loop visual servo ({mode})...", flush=True)
            if mode == "dino_oblique":
                if dino_matcher is None:
                    raise RuntimeError("SharedDinoMatcher not initialized for dino_oblique visual servo")
                tpl_path = _select_handover_template(handover, str(servo_cfg.get("template_shot_suffix", "02")))
                servo_res = await run_dino_visual_servo(
                    client=client,
                    drone=drone,
                    drone_id=drone_id,
                    target_id=tid,
                    matcher=dino_matcher,
                    template_image_path=tpl_path,
                    servo_cfg=servo_cfg,
                    collision_flags=collision_flags,
                    recorder=recorder,
                )
            elif mode == "opening_nadir":
                servo_res = await run_opening_visual_servo(
                    client=client,
                    drone=drone,
                    drone_id=drone_id,
                    target_id=tid,
                    servo_cfg=servo_cfg,
                    collision_flags=collision_flags,
                    recorder=recorder,
                )
            else:
                raise ValueError(f"Unsupported visual_servo mode: {mode}")

            servo_succeeded = bool(servo_res.get("succeeded"))
            servo_log_dir.mkdir(parents=True, exist_ok=True)
            trace_path = servo_log_dir / f"{tid}_servo.json"
            trace_path.write_text(json.dumps(servo_res, ensure_ascii=False, indent=2), encoding="utf-8")
            log["visual_servo"] = {k: v for k, v in servo_res.items() if k != "trace"}
            log["visual_servo"]["trace_file"] = str(trace_path)
            print(
                f"[{drone_id}->{tid}] Visual servo finished: mode={mode}, reason={servo_res['stop_reason']}, "
                f"succeeded={servo_succeeded}, steps={servo_res['steps']}, dur={servo_res['duration_s']}s",
                flush=True,
            )
    else:
        approach_wps = target_cfg.get("round2_approach_waypoints_ned", [])
        if approach_wps:
            first_wp = approach_wps[0]
            if recorder:
                recorder.set_status(tid, f"巡航前往目标（{int(target_cfg['cruise_alt_m'])}m）")
            await _move_segmented(drone, (float(first_wp[0]), float(first_wp[1]), cruise_z), cruise_v)
            for wp in approach_wps:
                if recorder:
                    recorder.set_status(tid, f"抵近接近（高度 {int(-float(wp[2]))}m）")
                await _move_segmented(drone, (float(wp[0]), float(wp[1]), float(wp[2])), approach_v)

    for shot in target_cfg["round2_shots"]:
        sn, se, sz = [float(x) for x in shot["position_ned"]]
        yaw_rad = math.radians(float(shot["yaw_deg"]))
        pitch_deg = float(shot["camera_pitch_deg"])
        used_fallback = False
        view_cn = _VIEW_CN.get(shot["view_type"], shot["view_type"])

        if servo_cfg is not None and servo_succeeded:
            # Capture directly from the pose reached by closed-loop visual servo
            if servo_cfg["mode"] == "opening_nadir":
                await _await_drone_task("yaw", lambda cb, y=yaw_rad: drone.rotate_to_yaw_async(y, timeout_sec=20, callback=cb), timeout_s=30)
                await _await_drone_task("hover", lambda cb: drone.hover_async(callback=cb), timeout_s=15)
                await _wait_hover_level(drone)
                drone.set_camera_pose("FrontCamera", _camera_pose(pitch_deg))
            else:
                # Keep current visual-servo heading & camera pitch aimed squarely at the target
                await _await_drone_task("hover", lambda cb: drone.hover_async(callback=cb), timeout_s=15)
                await _wait_hover_level(drone)
        else:
            if servo_cfg is not None:
                used_fallback = True
                if recorder:
                    recorder.set_status(tid, f"安全回退就位（{view_cn}）")
                if "fallback_position_ned" in shot:
                    sn, se, sz = [float(x) for x in shot["fallback_position_ned"]]
                    pitch_deg = float(shot.get("fallback_camera_pitch_deg", pitch_deg))
            else:
                if recorder:
                    recorder.set_status(tid, f"抵近就位（{view_cn}）")
            await _move_segmented(drone, (sn, se, sz), min(approach_v, 3.0))
            if collision_flags.get(drone_id, 0) > col_before and "fallback_position_ned" in shot:
                used_fallback = True
                fn, fe, fz = [float(x) for x in shot["fallback_position_ned"]]
                pitch_deg = float(shot.get("fallback_camera_pitch_deg", pitch_deg))
                await _move_segmented(drone, (fn, fe, fz), 3.0)
            await _await_drone_task("yaw", lambda cb, y=yaw_rad: drone.rotate_to_yaw_async(y, timeout_sec=20, callback=cb), timeout_s=30)
            await _await_drone_task("hover", lambda cb: drone.hover_async(callback=cb), timeout_s=15)
            await _wait_hover_level(drone)
            drone.set_camera_pose("FrontCamera", _camera_pose(pitch_deg))

        if recorder:
            recorder.set_status(tid, f"悬停复查采图（{view_cn}）")
        image_id = f"{tid}-R2-{shot['suffix']}"
        front_raw = capture_dir / f"{image_id}-front.png"
        chase_raw = capture_dir / f"{image_id}-chase.png"
        front_meta = await _capture_camera_png(drone, "FrontCamera", front_raw, settle_s=1.0)
        chase_meta = await _capture_camera_png(drone, "Chase", chase_raw, settle_s=0.3)

        center.submit_upload(
            target_id=tid,
            stage=2,
            drone_id=drone_id,
            image_path=front_raw,
            image_id=image_id,
            captured_at=utc_now_iso(),
            task_id=handover["task_id"],
            view_type=shot["view_type"],
            metadata={
                "front_capture": front_meta,
                "chase_visualization_path": str(chase_raw),
                "chase_capture": chase_meta,
                "used_fallback": used_fallback,
                "control_mode": log["control_mode"],
                "pose_ned": drone.get_ground_truth_kinematics()["pose"]["position"],
            },
        )
        log["shots"].append(
            {"image_id": image_id, "view_type": shot["view_type"], "used_fallback": used_fallback, "front": front_meta, "chase": chase_meta}
        )
        print(f"[{drone_id}->{tid}] Uploaded {image_id} ({shot['view_type']}, {front_meta['width']}x{front_meta['height']}, mean={front_meta['mean']})", flush=True)
        drone.set_camera_pose("FrontCamera", _camera_pose(-30.0))

    if servo_cfg is not None and servo_cfg["mode"] == "opening_nadir" and servo_succeeded:
        # Climb straight vertically out of the exact clear roof opening entered by visual servo
        cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
        exit_z = -float(servo_cfg.get("vertical_exit_alt_m", 38.0))
        if recorder:
            recorder.set_status(tid, f"垂直穿洞退出（高度 {int(-exit_z)}m）")
        await _move_segmented(drone, (float(cur_pos["x"]), float(cur_pos["y"]), exit_z), min(approach_v, 3.0))
    else:
        for wp in target_cfg.get("round2_exit_waypoints_ned", []):
            if recorder:
                recorder.set_status(tid, f"退出目标区（高度 {int(-float(wp[2]))}m）")
            await _move_segmented(drone, (float(wp[0]), float(wp[1]), float(wp[2])), approach_v)

    cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    await _move_segmented(drone, (float(cur_pos["x"]), float(cur_pos["y"]), cruise_z), vert_v)
    if recorder:
        recorder.set_status(tid, f"返航巡航（{int(target_cfg['cruise_alt_m'])}m）")
    print(f"[{drone_id}->{tid}] Returning to base ({home_n}, {home_e})", flush=True)
    await _move_segmented(drone, (home_n, home_e, cruise_z), cruise_v)
    await _move_segmented(drone, (home_n, home_e, -4.0), vert_v)
    await _await_drone_task("land", lambda cb: drone.land_async(timeout_sec=25, callback=cb), timeout_s=35)
    drone.disarm()
    center.report_return(tid, stage=2, drone_id=drone_id, status="returned", note="Landed at home pad after recheck")
    if recorder:
        recorder.set_status(tid, "已返航降落")
    log["collisions_after"] = collision_flags.get(drone_id, 0)
    log["flight_duration_s"] = round(time.monotonic() - t0, 2)
    print(f"[{drone_id}->{tid}] Returned & landed in {log['flight_duration_s']}s", flush=True)
    return log


async def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    mission_cfg_path = project_path(args.mission_config)
    mission_cfg = json.loads(mission_cfg_path.read_text(encoding="utf-8"))
    scene_path = project_path(mission_cfg["scene_config"])
    scene_cfg = _load_scene_bundle(scene_path)

    mission_id = args.mission_id or datetime.now().strftime("sim6-%Y%m%d-%H%M%S")
    output_dir = project_path(args.output_dir) if args.output_dir else (RUNS / "coordination" / mission_id)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime-scene.json").write_text(json.dumps(scene_cfg, indent=2), encoding="utf-8")
    configure_client_log(output_dir)

    client = ProjectAirSimClient(address=args.address)
    sim_metrics: dict[str, Any] = {
        "mission_id": mission_id,
        "address": args.address,
        "scene_config": str(scene_path),
        "probe_only": bool(args.probe_only),
        "started_at": utc_now_iso(),
    }

    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        client.socket_topics.recv_buffer_size = 24
        world = World.__new__(World)
        world.client = client
        world.sim_instance_idx = -1
        t_load = time.monotonic()
        world.load_scene(scene_cfg, delay_after_load_sec=3)
        sim_metrics["scene_load_s"] = round(time.monotonic() - t_load, 2)
        print(f"Loaded 6-UAV scene {scene_cfg['id']} in {sim_metrics['scene_load_s']}s", flush=True)

        drone_names = [a["name"] for a in scene_cfg["actors"]]
        drones = {name: Drone(client, world, name) for name in drone_names}

        # Subscribe to collision info for all 6 drones
        collision_flags: dict[str, int] = {name: 0 for name in drone_names}

        def make_col_cb(dname: str):
            def _col(_topic: str, _msg: dict[str, Any]) -> None:
                collision_flags[dname] = collision_flags.get(dname, 0) + 1
            return _col

        for name, d in drones.items():
            if "collision_info" in d.robot_info:
                client.subscribe(d.robot_info["collision_info"], make_col_cb(name))

        # FPS probes: 1 active drone, 3 active drones (Batch 1), and all 6 drones
        print(f"Probing live FPS (1 drone / 3 drones / 6 drones, {args.probe_seconds}s each)...", flush=True)
        probe_1 = _probe_stream_fps(client, drones, drone_names[:1], duration_s=min(args.probe_seconds, 5.0))
        probe_3 = _probe_stream_fps(client, drones, drone_names[:3], duration_s=min(args.probe_seconds, 5.0))
        probe_6 = _probe_stream_fps(client, drones, drone_names, duration_s=args.probe_seconds)
        sim_metrics["fps_probes"] = {
            "single_drone_subscribed": probe_1,
            "batch1_3drones_subscribed": probe_3,
            "all_6drones_subscribed": probe_6,
        }
        (output_dir / "fps_probe.json").write_text(json.dumps(sim_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("FPS probe saved:", output_dir / "fps_probe.json", flush=True)

        if args.probe_only:
            return sim_metrics

        # Full End-to-End 2-Batch 6-UAV Survey
        vlm = _build_vlm(args)
        center = SurveyCenter(
            mission_id=mission_id,
            output_dir=output_dir,
            vlm_client=vlm,
            dispatch_policy=args.dispatch_policy,
            allow_existing=True,
            config={
                "mode": "projectairsim_6uav_e2e",
                "address": args.address,
                "scene_config": str(scene_path),
                "mission_config": str(mission_cfg_path),
                "dispatch_policy": args.dispatch_policy,
                "vlm_backend": args.vlm,
                "vlm_model": vlm.model_name,
            },
        )
        capture_dir = output_dir / "sim_captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        targets = mission_cfg["targets"]
        target_positions_ned = {
            t["target_id"]: (
                float(t["round1_shots"][0]["position_ned"][0]),
                float(t["round1_shots"][0]["position_ned"][1]),
            )
            for t in targets
        }
        target_labels = {
            t["target_id"]: str(t.get("display_name") or format_target_cn(t["target_id"]))
            for t in targets
        }
        trajectory_store: dict[str, Any] = {
            "mission_id": mission_id,
            "frame_convention": "NED -> ENU: (E, N, -D)",
            "home_ned": [0.0, 0.0, -1.0],
            "home_enu": [0.0, 0.0, 1.0],
            "targets": {
                t["target_id"]: {
                    "display_name": target_labels[t["target_id"]],
                    "ned": [
                        float(t["round1_shots"][0]["position_ned"][0]),
                        float(t["round1_shots"][0]["position_ned"][1]),
                        0.0,
                    ],
                    "enu": [
                        float(t["round1_shots"][0]["position_ned"][1]),
                        float(t["round1_shots"][0]["position_ned"][0]),
                        0.0,
                    ],
                }
                for t in targets
            },
            "drones": {
                t["round1_drone_id"]: {
                    "batch": 1,
                    "target_id": t["target_id"],
                    "display_name": format_drone_cn(t["round1_drone_id"]),
                    "samples": [],
                }
                for t in targets
            },
        }
        for t in targets:
            trajectory_store["drones"][t["round2_drone_id"]] = {
                "batch": 2,
                "target_id": t["target_id"],
                "display_name": format_drone_cn(t["round2_drone_id"]),
                "samples": [],
            }

        recorder: MultiUAVVideoRecorder | None = None
        if args.record_video and len(targets) == 3:
            recorder = MultiUAVVideoRecorder(
                output_dir=output_dir / "videos",
                target_order=[t["target_id"] for t in targets],
                fps=30,
                highlight_stride=4,
                target_labels=target_labels,
                target_positions_ned=target_positions_ned,
                home_ned=(0.0, 0.0),
            )
            recorder.start()

        mission_t0 = time.monotonic()
        pose_stage_ref: dict[str, Any] = {"stage": 1, "active": []}
        pose_stop = asyncio.Event()

        async def _pose_sampler() -> None:
            while not pose_stop.is_set():
                t_rel = round(time.monotonic() - mission_t0, 3)
                stage_now = int(pose_stage_ref["stage"])
                for dname in list(pose_stage_ref["active"]):
                    try:
                        kin = drones[dname].get_ground_truth_kinematics()
                        pos = kin["pose"]["position"]
                        vel = kin.get("twist", {}).get("linear", {})
                        n, e, d = float(pos["x"]), float(pos["y"]), float(pos["z"])
                        spd = math.sqrt(
                            float(vel.get("x", 0.0)) ** 2
                            + float(vel.get("y", 0.0)) ** 2
                            + float(vel.get("z", 0.0)) ** 2
                        )
                        trajectory_store["drones"][dname]["samples"].append(
                            {
                                "t_s": t_rel,
                                "stage": stage_now,
                                "ned": [round(n, 3), round(e, 3), round(d, 3)],
                                "enu": [round(e, 3), round(n, 3), round(-d, 3)],
                                "speed_mps": round(spd, 2),
                            }
                        )
                        if recorder is not None:
                            recorder.update_pose(dname, n, e, d, speed_mps=spd, t_s=t_rel)
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(pose_stop.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass

        sampler_task = asyncio.create_task(_pose_sampler())
        try:
            for t in targets:
                center.register_target(
                    target_id=t["target_id"],
                    round1_drone_id=t["round1_drone_id"],
                    ground_truth_label=t.get("ground_truth_label"),
                )

            # Stage 1: Batch-1 3 drones fly concurrently
            pose_stage_ref["stage"] = 1
            pose_stage_ref["active"] = [t["round1_drone_id"] for t in targets]
            if recorder:
                recorder.set_phase("第一阶段：首批三机广域采集")
                recorder.attach_batch_streams(
                    client,
                    drones,
                    {t["target_id"]: t["round1_drone_id"] for t in targets},
                )
            t_s1 = time.monotonic()
            r1_logs = await asyncio.gather(
                *[
                    _fly_round1_for_target(
                        drones[t["round1_drone_id"]], t, mission_cfg, center, capture_dir, collision_flags, recorder=recorder
                    )
                    for t in targets
                ]
            )
            sim_metrics["stage1_flight_wall_s"] = round(time.monotonic() - t_s1, 2)
            sim_metrics["stage1_logs"] = r1_logs
            pose_stage_ref["active"] = []
            if recorder:
                recorder.detach_streams()

            # VLM Stage 1 assessment for each target
            t_v1 = time.monotonic()
            handovers: dict[str, dict[str, Any]] = {}
            r1_summary_lines: list[str] = []
            r1_thumbs: dict[str, Path] = {}
            for t in targets:
                tid = t["target_id"]
                print(f"[VLM-R1] Assessing {tid} with {vlm.model_name}...", flush=True)
                res1 = center.run_initial_assessment(tid)
                r1_res = res1["assessment"]["result"] if res1["assessment"] else None
                flagged = bool(r1_res.get("needs_recheck")) if r1_res else False
                sec_task = (r1_res.get("secondary_task") or {}) if r1_res else {}
                action = sec_task.get("task_type") or (res1["recheck_task"]["recommended_action"] if res1["recheck_task"] else "none")
                t_type = (r1_res.get("target_type") or "-") if r1_res else "-"
                sus_parts = "、".join((r1_res.get("suspected_damaged_parts") or [])[:2]) if r1_res else ""
                part_txt = sus_parts or "未见明显受损"
                focus_part = sec_task.get("focus_part") or (res1["recheck_task"]["focus_region"] if res1["recheck_task"] else "目标主体")
                print(
                    f"[VLM-R1] {tid}: status={res1['status']}, type={t_type}, needs_recheck={flagged}, "
                    f"parts={part_txt}, task={action}",
                    flush=True,
                )
                r1_summary_lines.append(
                    f"{target_labels[tid]}｜种类：{t_type}｜初评：{part_txt}｜"
                    f"二次任务：{_ACTION_CN.get(action, action)}（瞄准{focus_part}）"
                )
                r1_thumbs[tid] = capture_dir / f"{tid}-R1-02-front.png"
                if recorder:
                    recorder.set_vlm_badge(tid, "初评：疑似受损" if flagged else "初评：近距复核")
                if res1["recheck_task"] is not None:
                    handovers[tid] = center.dispatch_recheck(tid, t["round2_drone_id"])
            sim_metrics["stage1_vlm_wall_s"] = round(time.monotonic() - t_v1, 2)
            if recorder:
                recorder.write_summary_card(
                    title=f"视觉语言模型初评与复查派单（{vlm.model_name}，耗时 {sim_metrics['stage1_vlm_wall_s']} 秒）",
                    lines=r1_summary_lines,
                    duration_s=3.5,
                    thumbnails=r1_thumbs,
                )

            # Stage 2: Batch-2 dispatched drones fly concurrently (with 1A serialized GPU visual servo)
            dino_matcher: SharedDinoMatcher | None = None
            servo_lock = asyncio.Lock()
            servo_log_dir = output_dir / "servo_logs"
            sim_metrics["visual_servo_enabled"] = bool(args.visual_servo)
            needs_dino = bool(
                args.visual_servo
                and any(
                    t["target_id"] in handovers
                    and (t.get("visual_servo") or {}).get("mode") == "dino_oblique"
                    for t in targets
                )
            )
            if needs_dino:
                print("[VisualServo] Loading shared DINOv2 ViT-S/14 encoder on GPU...", flush=True)
                t_dino = time.monotonic()
                dino_matcher = SharedDinoMatcher()
                sim_metrics["dino_load_s"] = round(time.monotonic() - t_dino, 2)
                print(
                    f"[VisualServo] Shared DINOv2 ready on {dino_matcher.device} "
                    f"(grid={dino_matcher.grid}x{dino_matcher.grid}) in {sim_metrics['dino_load_s']}s",
                    flush=True,
                )

            pose_stage_ref["stage"] = 2
            pose_stage_ref["active"] = [t["round2_drone_id"] for t in targets if t["target_id"] in handovers]
            if recorder and handovers:
                recorder.set_phase(
                    "第二阶段：次批三机视觉伺服复查" if args.visual_servo else "第二阶段：次批三机抵近复查"
                )
                recorder.attach_batch_streams(
                    client,
                    drones,
                    {t["target_id"]: t["round2_drone_id"] for t in targets if t["target_id"] in handovers},
                )
            t_s2 = time.monotonic()
            r2_tasks = [
                _fly_round2_for_target(
                    client,
                    drones[t["round2_drone_id"]],
                    t,
                    mission_cfg,
                    handovers[t["target_id"]],
                    center,
                    capture_dir,
                    servo_log_dir,
                    collision_flags,
                    recorder=recorder,
                    use_visual_servo=bool(args.visual_servo),
                    dino_matcher=dino_matcher,
                    servo_lock=servo_lock,
                )
                for t in targets
                if t["target_id"] in handovers
            ]
            r2_logs = await asyncio.gather(*r2_tasks) if r2_tasks else []
            sim_metrics["stage2_flight_wall_s"] = round(time.monotonic() - t_s2, 2)
            sim_metrics["stage2_logs"] = r2_logs
            pose_stage_ref["active"] = []
            if recorder:
                recorder.detach_streams()

            # VLM Stage 2 reassessment
            t_v2 = time.monotonic()
            r2_summary_lines: list[str] = []
            r2_thumbs: dict[str, Path] = {}
            for tid in handovers:
                print(f"[VLM-R2] Reassessing {tid} with {vlm.model_name}...", flush=True)
                res2 = center.run_final_assessment(tid)
                r2_res = res2["assessment"]["result"] if res2["assessment"] else None
                verdict = r2_res["verdict"] if r2_res else "failed"
                verdict_cn = _VERDICT_CN.get(verdict, verdict)
                t_type2 = (r2_res.get("target_type") or "-") if r2_res else "-"
                sev2 = _SEVERITY_CN.get((r2_res.get("damage_severity") or ""), "-") if r2_res else "-"
                parts2 = "、".join((r2_res.get("confirmed_damaged_parts") or [])[:2]) if r2_res else ""
                part2_txt = parts2 or "无受损部位"
                print(f"[VLM-R2] {tid}: status={res2['status']}, type={t_type2}, verdict={verdict}, severity={sev2}, parts={part2_txt}", flush=True)
                r2_summary_lines.append(
                    f"{target_labels[tid]}｜种类：{t_type2}｜部位：{part2_txt}｜分级：{sev2}｜结论：{verdict_cn}"
                )
                r2_thumbs[tid] = capture_dir / f"{tid}-R2-01-front.png"
                if recorder:
                    recorder.set_vlm_badge(tid, f"终评：{verdict_cn}")
            sim_metrics["stage2_vlm_wall_s"] = round(time.monotonic() - t_v2, 2)
            if recorder:
                recorder.write_summary_card(
                    title=f"视觉语言模型终评结果（{vlm.model_name}，耗时 {sim_metrics['stage2_vlm_wall_s']} 秒）",
                    lines=r2_summary_lines,
                    duration_s=4.0,
                    thumbnails=r2_thumbs,
                )
                sim_metrics["video_metrics"] = recorder.stop()
                recorder = None

            pose_stop.set()
            await sampler_task
            traj_path = output_dir / "trajectories.json"
            traj_path.write_text(json.dumps(trajectory_store, ensure_ascii=False, indent=2), encoding="utf-8")
            sim_metrics["trajectories_file"] = str(traj_path)
            sim_metrics["total_collisions"] = collision_flags
            sim_metrics["finished_at"] = utc_now_iso()

            json_path, md_path = center.export_report()
            (output_dir / "sim_metrics.json").write_text(json.dumps(sim_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"E2E Mission Finished: {mission_id}", flush=True)
            print(f"  Report:  {md_path}", flush=True)
            print(f"  JSON:    {json_path}", flush=True)
            print(f"  Metrics: {output_dir / 'sim_metrics.json'}", flush=True)
            print(f"  Trajs:   {traj_path}", flush=True)
            if "video_metrics" in sim_metrics:
                print(f"  Video RT: {sim_metrics['video_metrics']['realtime_video']}", flush=True)
                print(f"  Video 4x: {sim_metrics['video_metrics']['highlight_video_4x']}", flush=True)
            return sim_metrics
        finally:
            pose_stop.set()
            if not sampler_task.done():
                sampler_task.cancel()
            if recorder is not None:
                try:
                    recorder.stop()
                except Exception:
                    pass
            center.close()
    finally:
        if args.restore_default_scene:
            try:
                default_scene = {
                    "id": "DefaultScene",
                    "actors": [],
                    "clock": {"type": "real-time"},
                    "home-geo-point": {"latitude": 47.641468, "longitude": -122.140165, "altitude": 122.0},
                    "scene-type": "UnrealNative",
                }
                world = World.__new__(World)
                world.client = client
                world.sim_instance_idx = -1
                world.load_scene(default_scene, delay_after_load_sec=1)
                print("Restored empty DefaultScene", flush=True)
            except Exception as exc:
                print(f"Warning: failed to restore DefaultScene: {exc}", flush=True)
        try:
            client.disconnect()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run_simulation(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
