"""Offline unit tests for Coastal Harbor Task 1 geometry scoring, orbit planning, and metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from harbor.part_recheck import (
    DEFAULT_CONFIG,
    HarborPartVLM,
    camera_rot_from_yaw_pitch,
    estimate_target_center_from_depth,
    evaluate_capture_geometry,
    plan_orbit_waypoints,
    score_verdict_correct,
    summarize_trials,
)
from harbor.recorder import HarborVideoRecorder
from coordination.vlm import VLMImage


class HarborPartRecheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.scoring = self.cfg["scoring"]
        self.fish_gt = next(t["evaluation_gt"] for t in self.cfg["targets"] if t["target_id"] == "Building_071")

    def test_front_close_view_is_visible_and_clear(self) -> None:
        # Place camera 8 m in front of Building_071 awning normal, elevated 3 m, pitched -20 deg
        part_ue = np.array(self.fish_gt["part_center_ue_m"])
        n_ue = np.array(self.fish_gt["part_normal_ue"])
        cam_ue = part_ue + n_ue * 8.0 + np.array([0.0, 0.0, 3.0])
        cam_ned = np.array([cam_ue[0], cam_ue[1], -cam_ue[2]])
        yaw = math.atan2(part_ue[1] - cam_ue[1], part_ue[0] - cam_ue[0])
        cam_rot = camera_rot_from_yaw_pitch(yaw, -20.0)
        depth = np.full((360, 640), 8.5, dtype=np.float32)

        res = evaluate_capture_geometry(
            cam_pos_ned=cam_ned,
            cam_rot=cam_rot,
            depth_arr=depth,
            evaluation_gt=self.fish_gt,
            scoring_cfg=self.scoring,
        )
        self.assertTrue(res["in_frame"])
        self.assertTrue(res["front_facing"])
        self.assertFalse(res["depth_occluded"])
        self.assertTrue(res["part_visible"])
        self.assertTrue(res["part_clear"])
        self.assertLess(res["view_angle_deg"], 30.0)

    def test_backside_view_is_rejected_even_if_close(self) -> None:
        part_ue = np.array(self.fish_gt["part_center_ue_m"])
        n_ue = np.array(self.fish_gt["part_normal_ue"])
        cam_ue = part_ue - n_ue * 10.0 + np.array([0.0, 0.0, 3.0])
        cam_ned = np.array([cam_ue[0], cam_ue[1], -cam_ue[2]])
        yaw = math.atan2(part_ue[1] - cam_ue[1], part_ue[0] - cam_ue[0])
        cam_rot = camera_rot_from_yaw_pitch(yaw, -18.0)
        depth = np.full((360, 640), 6.0, dtype=np.float32)  # Back wall occludes front awning

        res = evaluate_capture_geometry(
            cam_pos_ned=cam_ned,
            cam_rot=cam_rot,
            depth_arr=depth,
            evaluation_gt=self.fish_gt,
            scoring_cfg=self.scoring,
        )
        self.assertFalse(res["front_facing"])
        self.assertTrue(res["depth_occluded"])
        self.assertFalse(res["part_visible"])
        self.assertFalse(res["part_clear"])

    def test_far_front_view_is_visible_but_not_clear(self) -> None:
        part_ue = np.array(self.fish_gt["part_center_ue_m"])
        n_ue = np.array(self.fish_gt["part_normal_ue"])
        cam_ue = part_ue + n_ue * 24.0 + np.array([0.0, 0.0, 14.0])
        cam_ned = np.array([cam_ue[0], cam_ue[1], -cam_ue[2]])
        yaw = math.atan2(part_ue[1] - cam_ue[1], part_ue[0] - cam_ue[0])
        cam_rot = camera_rot_from_yaw_pitch(yaw, -30.0)
        depth = np.full((360, 640), 27.8, dtype=np.float32)

        res = evaluate_capture_geometry(
            cam_pos_ned=cam_ned,
            cam_rot=cam_rot,
            depth_arr=depth,
            evaluation_gt=self.fish_gt,
            scoring_cfg=self.scoring,
        )
        self.assertTrue(res["part_visible"])
        self.assertFalse(res["part_clear"])

    def test_depth_only_orbit_planner_reaches_opposite_side(self) -> None:
        drone_pos = np.array([0.0, 0.0, -14.0])
        yaw = 0.0  # Facing +N
        # Slant depth 23.09 m at -30 deg -> horizontal distance ~20.0 m
        depth = np.full((360, 640), 23.094, dtype=np.float32)
        center_ne, radius = estimate_target_center_from_depth(
            drone_pos_ned=drone_pos,
            drone_yaw_rad=yaw,
            camera_pitch_deg=-30.0,
            depth_arr=depth,
        )
        self.assertAlmostEqual(float(center_ne[0]), 20.0, places=1)
        self.assertAlmostEqual(float(center_ne[1]), 0.0, places=1)
        self.assertAlmostEqual(radius, 20.0, places=1)

        wps = plan_orbit_waypoints(
            drone_pos_ned=drone_pos,
            center_ne=center_ne,
            radius_m=radius,
            delta_angle_deg=180.0,
            orbit_alt_m=15.0,
            final_alt_m=14.0,
            num_arc_steps=3,
        )
        end_pos = wps[-1]["position_ned"]
        self.assertAlmostEqual(end_pos[0], 40.0, places=1)
        self.assertAlmostEqual(end_pos[1], 0.0, places=1)
        self.assertAlmostEqual(abs(wps[-1]["yaw_rad"]), math.pi, places=2)

    def test_summary_splits_direct_and_occluded(self) -> None:
        trials = [
            {
                "target_id": "Vessel_037",
                "damage_gt": "damaged",
                "view_regime": "direct",
                "condition": "far_only",
                "best_part_clear": False,
                "best_part_visible": True,
                "verdict": "still_uncertain",
                "verdict_correct": False,
                "final_depth_m": None,
                "best_dist_to_part_m": 25.0,
                "flight_time_s": 0.0,
                "num_captures": 1,
                "collisions": 0,
            },
            {
                "target_id": "Vessel_037",
                "damage_gt": "damaged",
                "view_regime": "occluded",
                "condition": "active_viewpoint",
                "best_part_clear": True,
                "best_part_visible": True,
                "verdict": "confirmed",
                "verdict_correct": True,
                "final_depth_m": 5.8,
                "best_dist_to_part_m": 6.2,
                "flight_time_s": 28.5,
                "num_captures": 3,
                "collisions": 0,
            },
        ]
        self.assertTrue(score_verdict_correct("confirmed", "damaged"))
        self.assertTrue(score_verdict_correct("ruled_out", "intact"))
        self.assertFalse(score_verdict_correct("still_uncertain", "intact"))
        s = summarize_trials(trials, target_stop_dist_m=6.0)
        self.assertEqual(s["direct"]["far_only"]["part_clear_count"], 0)
        self.assertEqual(s["occluded"]["active_viewpoint"]["part_clear_count"], 1)
        self.assertAlmostEqual(s["occluded"]["active_viewpoint"]["mean_stop_dist_err_m"], 0.2, places=2)

    def test_standalone_vlm_none_skips_remote_server(self) -> None:
        vlm = HarborPartVLM(backend="none")
        dummy = VLMImage(image_id="IMG-1", view_type="test", image_path=Path("/tmp/nonexistent.jpg"))
        r1 = vlm.plan_viewpoint(
            neutral_id="OBJ-1", target_class_hint="商铺", focus_part_request="雨棚", image=dummy
        )
        self.assertEqual(r1.result["recommended_maneuver"], "direct_approach")
        r2 = vlm.assess_damage(
            neutral_id="OBJ-1",
            target_class_hint="商铺",
            focus_part_request="雨棚",
            images=[dummy],
            condition_label="standalone",
        )
        self.assertEqual(r2.result["verdict"], "skipped_standalone")

    def test_depth_center_estimator_uses_roi_over_background(self) -> None:
        drone_pos = np.array([0.0, 0.0, -12.0])
        depth = np.full((360, 640), 45.0, dtype=np.float32)  # Distant background
        # Foreground vessel inside ROI [0.43, 0.45, 0.59, 0.62] at slant depth 22.0 m
        depth[int(360 * 0.45) : int(360 * 0.62), int(640 * 0.43) : int(640 * 0.59)] = 22.0
        center_ne, radius = estimate_target_center_from_depth(
            drone_pos_ned=drone_pos,
            drone_yaw_rad=0.0,
            camera_pitch_deg=-30.0,
            depth_arr=depth,
            depth_center_offset_m=1.5,
            roi=[0.43, 0.45, 0.59, 0.62],
        )
        expected_r = 22.0 * math.cos(math.radians(30.0)) + 1.5
        self.assertAlmostEqual(radius, expected_r, places=1)
        self.assertAlmostEqual(float(center_ne[0]), expected_r, places=1)

    def test_harbor_video_recorder_writes_mp4(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "test_flight.mp4"
            rec = HarborVideoRecorder(mp4, title="测试录像", fps=10, encode_h264=False)
            dummy_bgr = np.full((360, 640, 3), (60, 90, 120), dtype=np.uint8)
            rec.ingest_frame_bgr("chase", dummy_bgr, timestamp=1)
            rec.ingest_frame_bgr("front", dummy_bgr, timestamp=1)
            rec.set_status("HARBOR-OBJ-1", "视觉伺服中")
            rec.set_servo_overlay(
                "HARBOR-OBJ-1",
                {
                    "mode": "dino",
                    "sim_2d": np.random.rand(16, 16).astype(np.float32),
                    "cx": 0.1,
                    "cy": -0.05,
                    "peak": 0.82,
                    "depth_m": 8.4,
                    "cmd_fwd": 1.2,
                    "cmd_right": 0.1,
                    "cmd_down": 0.4,
                    "confirm": 1,
                    "confirm_target": 3,
                    "locked": True,
                },
            )
            frame = rec._compose_frame()
            self.assertEqual(frame.shape, (416, 1280, 3))
            rec._writer.write(frame)
            rec._frames_written += 1
            info = rec.stop()
            self.assertEqual(info["frames"], 1)
            self.assertTrue(mp4.is_file() and mp4.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
