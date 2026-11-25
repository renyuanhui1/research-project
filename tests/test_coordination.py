"""Offline tests for centralized two-stage multi-UAV cooperative survey."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from coordination.center import SurveyCenter
from coordination.run_replay import parse_args, run_replay
from coordination.vlm import MockVLMClient


def _make_dummy_png(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)
    return path


class CoordinationTests(unittest.TestCase):
    def test_out_of_order_duplicates_and_label_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Intentionally embed revealing words in source paths & ground-truth labels
            a_r1 = _make_dummy_png(tmp_dir / "src" / "aircraft-a-wing-curl-damaged.png", b"a1")
            a_r2 = _make_dummy_png(tmp_dir / "src" / "aircraft-a-wing-curl-closeup.png", b"a2")
            b_r1 = _make_dummy_png(tmp_dir / "src" / "hangar-roof-hole.png", b"b1")
            b_r2 = _make_dummy_png(tmp_dir / "src" / "hangar-interior.png", b"b2")
            c_r1 = _make_dummy_png(tmp_dir / "src" / "aircraft-b-完好-intact.png", b"c1")

            vlm = MockVLMClient(
                round1_overrides={
                    "OBJ-C": {
                        "needs_recheck": False,
                        "recheck_question": "",
                        "recommended_action": "none",
                        "initial_summary": "首轮画面清晰，未见异常，无需复查。",
                    }
                },
                round2_overrides={
                    "OBJ-B": {
                        "verdict": "still_uncertain",
                        "reasoning": "补拍仍存在局部阴影遮挡，保留不确定结论。",
                    }
                },
            )
            center = SurveyCenter("m-test-1", tmp_dir / "mission", vlm, dispatch_policy="only_flagged")
            try:
                center.register_target("OBJ-A", "Scout-1", ground_truth_label="02-aircraft-a-wing-curl")
                center.register_target("OBJ-B", "Scout-2", ground_truth_label="09-hangar-910")
                center.register_target("OBJ-C", "Scout-3", ground_truth_label="05-aircraft-b-完好")

                # Out-of-order: OBJ-C arrives and finishes before OBJ-A/OBJ-B even upload
                r_c = center.submit_upload("OBJ-C", 1, "Scout-3", c_r1, "OBJ-C-R1-01")
                self.assertFalse(r_c["duplicate"])
                s1_c = center.run_initial_assessment("OBJ-C")
                self.assertIsNone(s1_c["recheck_task"])
                self.assertEqual(center.store.get_target("m-test-1", "OBJ-C")["status"], "completed_no_recheck")
                # Photo received, but Scout-3 has NOT reported return yet
                self.assertEqual(center.store.get_target("m-test-1", "OBJ-C")["round1_return_status"], "assigned")
                center.report_return("OBJ-C", 1, "Scout-3", status="returned")
                self.assertEqual(center.store.get_target("m-test-1", "OBJ-C")["round1_return_status"], "returned")

                # Duplicate uploads for OBJ-A before AND after initial assessment
                self.assertFalse(center.submit_upload("OBJ-A", 1, "Scout-1", a_r1, "OBJ-A-R1-01")["duplicate"])
                self.assertTrue(center.submit_upload("OBJ-A", 1, "Scout-1", a_r1, "OBJ-A-R1-01")["duplicate"])
                s1_a_first = center.run_initial_assessment("OBJ-A")
                self.assertTrue(s1_a_first["recheck_created"])
                self.assertTrue(center.submit_upload("OBJ-A", 1, "Scout-1", a_r1, "OBJ-A-R1-01")["duplicate"])
                s1_a_second = center.run_initial_assessment("OBJ-A")
                self.assertTrue(s1_a_second["cached"])
                self.assertFalse(s1_a_second["recheck_created"])

                # Dispatch Batch-2 for OBJ-A, duplicate R2 upload, verify upload vs return separation
                packet_a = center.dispatch_recheck("OBJ-A", "Inspect-1")
                self.assertEqual(packet_a["target_id"], "OBJ-A")
                self.assertEqual(packet_a["round1_image_ids"], ["OBJ-A-R1-01"])
                self.assertFalse(center.submit_upload("OBJ-A", 2, "Inspect-1", a_r2, "OBJ-A-R2-01")["duplicate"])
                self.assertTrue(center.submit_upload("OBJ-A", 2, "Inspect-1", a_r2, "OBJ-A-R2-01")["duplicate"])
                self.assertEqual(center.store.get_target("m-test-1", "OBJ-A")["round2_return_status"], "assigned")
                center.run_final_assessment("OBJ-A")
                center.report_return("OBJ-A", 2, "Inspect-1", status="returned")
                center.report_return("OBJ-A", 1, "Scout-1", status="returned")

                # OBJ-B gets stage-2 photo even before explicit dispatch, then finishes as still_uncertain
                center.submit_upload("OBJ-B", 1, "Scout-2", b_r1, "OBJ-B-R1-01")
                center.run_initial_assessment("OBJ-B")
                center.dispatch_recheck("OBJ-B", "Inspect-2")
                center.submit_upload("OBJ-B", 2, "Inspect-2", b_r2, "OBJ-B-R2-01")
                s2_b = center.run_final_assessment("OBJ-B")
                self.assertEqual(s2_b["assessment"]["result"]["verdict"], "still_uncertain")
                center.report_return("OBJ-B", 1, "Scout-2", status="returned")
                center.report_return("OBJ-B", 2, "Inspect-2", status="returned")

                # Verify strict label isolation in every VLM call
                forbidden_tokens = ("wing-curl", "hangar", "完好", "aircraft-a", "aircraft-b")
                for call in vlm.recorded_calls:
                    blob = json.dumps(call, ensure_ascii=False).lower()
                    for token in forbidden_tokens:
                        self.assertNotIn(token, blob, f"Leaked token {token!r} to VLM call: {blob}")

                json_path, md_path = center.export_report()
                report = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(report["summary"]["completed_targets"], 3)
                self.assertEqual(report["summary"]["recheck_tasks_created"], 2)
                self.assertEqual(report["summary"]["duplicate_uploads_ignored"], 3)
                md_text = md_path.read_text(encoding="utf-8")
                self.assertIn("OBJ-A", md_text)
                self.assertIn("still_uncertain", md_text)
                self.assertIn("无需复查", md_text)
            finally:
                center.close()

    def test_vlm_failure_is_isolated_per_target_and_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_a1 = _make_dummy_png(tmp_dir / "a1.png", b"a1")
            img_a2 = _make_dummy_png(tmp_dir / "a2.png", b"a2")
            img_b1 = _make_dummy_png(tmp_dir / "b1.png", b"b1")
            vlm = MockVLMClient(fail_targets_stage1={"OBJ-B"})
            center = SurveyCenter("m-fail-1", tmp_dir / "mission", vlm)
            try:
                center.register_target("OBJ-A", "Scout-1")
                center.register_target("OBJ-B", "Scout-2")
                center.submit_upload("OBJ-A", 1, "Scout-1", img_a1, "OBJ-A-R1-01")
                center.submit_upload("OBJ-B", 1, "Scout-2", img_b1, "OBJ-B-R1-01")

                res_b = center.run_initial_assessment("OBJ-B")
                self.assertEqual(res_b["status"], "failed")
                self.assertEqual(center.store.get_target("m-fail-1", "OBJ-B")["status"], "round1_vlm_failed")

                # OBJ-A completes unaffected
                center.run_initial_assessment("OBJ-A")
                center.dispatch_recheck("OBJ-A", "Inspect-1")
                center.submit_upload("OBJ-A", 2, "Inspect-1", img_a2, "OBJ-A-R2-01")
                center.run_final_assessment("OBJ-A")
                self.assertEqual(center.store.get_target("m-fail-1", "OBJ-A")["status"], "completed")

                partial = center.build_report()
                self.assertEqual(partial["mission"]["status"], "completed_with_pending")
                self.assertEqual(partial["summary"]["completed_targets"], 1)
                self.assertEqual(partial["summary"]["failed_targets"], 1)

                # Recover OBJ-B once VLM becomes healthy
                vlm.fail_targets_stage1.clear()
                retry_b = center.run_initial_assessment("OBJ-B")
                self.assertEqual(retry_b["status"], "ok")
                self.assertIsNotNone(retry_b["recheck_task"])
            finally:
                center.close()

    def test_always_confirm_dispatches_even_when_stage1_sees_no_damage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a_r1 = _make_dummy_png(tmp_dir / "a1.png", b"a1")
            a_r2 = _make_dummy_png(tmp_dir / "a2.png", b"a2")
            vlm = MockVLMClient(
                round1_overrides={
                    "OBJ-A": {
                        "needs_recheck": False,
                        "recheck_question": "",
                        "recommended_action": "none",
                        "uncertainties": ["正俯视无法观察机翼垂向形变与侧面蒙皮"],
                        "initial_summary": "首轮未见明显表面异常。",
                    }
                },
                round2_overrides={
                    "OBJ-A": {
                        "verdict": "confirmed",
                        "final_summary": "近距补拍发现第一轮俯视漏检的机翼上卷变形。",
                    }
                },
            )
            center = SurveyCenter("m-confirm-1", tmp_dir / "mission", vlm, dispatch_policy="always_confirm")
            try:
                center.register_target("OBJ-A", "Scout-1")
                center.submit_upload("OBJ-A", 1, "Scout-1", a_r1, "OBJ-A-R1-01")
                s1 = center.run_initial_assessment("OBJ-A")
                self.assertTrue(s1["recheck_created"])
                self.assertIsNotNone(s1["recheck_task"])
                self.assertEqual(s1["recheck_task"]["recommended_action"], "closeup_approach")
                self.assertIn("盲区", s1["recheck_task"]["question"])

                packet = center.dispatch_recheck("OBJ-A", "Inspect-1")
                self.assertEqual(packet["assigned_drone_id"], "Inspect-1")
                center.submit_upload("OBJ-A", 2, "Inspect-1", a_r2, "OBJ-A-R2-01")
                s2 = center.run_final_assessment("OBJ-A")
                self.assertEqual(s2["assessment"]["result"]["verdict"], "confirmed")

                _, md_path = center.export_report()
                self.assertIn("无损近距复核", md_path.read_text(encoding="utf-8"))
            finally:
                center.close()

    def test_robot_simulation_configs_30hz_and_multi_720p(self) -> None:
        single_cfg = json.loads(
            (ROOT / "configs" / "simulation" / "robot_quadrotor_fastphysics_sensors.jsonc").read_text(encoding="utf-8")
        )
        single_sensors = {s["id"]: s for s in single_cfg["sensors"]}
        for cam in ("FrontCamera", "Chase"):
            self.assertAlmostEqual(single_sensors[cam]["capture-interval"], 0.0333333, places=6)
        seg = next(c for c in single_sensors["FrontCamera"]["capture-settings"] if c["image-type"] == 3)
        self.assertFalse(seg["capture-enabled"])
        for unused in ("lidar1", "DistanceSensor", "Barometer", "Magnetometer", "Airspeed", "Battery"):
            self.assertFalse(single_sensors[unused]["enabled"])
        self.assertFalse(single_sensors["DistanceSensor"]["draw-debug-points"])

        multi_cfg = json.loads(
            (ROOT / "configs" / "simulation" / "robot_quadrotor_multi_720p.jsonc").read_text(encoding="utf-8")
        )
        multi_sensors = {s["id"]: s for s in multi_cfg["sensors"]}
        self.assertEqual(set(multi_sensors.keys()), {"Chase", "FrontCamera", "IMU1", "GPS"})
        for cam in ("FrontCamera", "Chase"):
            self.assertTrue(multi_sensors[cam]["enabled"])
            self.assertAlmostEqual(multi_sensors[cam]["capture-interval"], 0.0333333, places=6)
            rgb = next(c for c in multi_sensors[cam]["capture-settings"] if c["image-type"] == 0)
            self.assertEqual((rgb["width"], rgb["height"]), (1280, 720))
        depth = next(c for c in multi_sensors["FrontCamera"]["capture-settings"] if c["image-type"] == 2)
        self.assertEqual((depth["width"], depth["height"]), (640, 360))
        self.assertFalse(any(c["image-type"] == 3 for c in multi_sensors["FrontCamera"]["capture-settings"]))

    def test_multi_uav_video_recorder_writes_playable_mp4(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("cv2/numpy not available in current interpreter")
        from coordination.video_recorder import CANVAS_H, CANVAS_W, MultiUAVVideoRecorder

        with tempfile.TemporaryDirectory() as tmp:
            rec = MultiUAVVideoRecorder(
                Path(tmp) / "videos",
                ["OBJ-A", "OBJ-B", "OBJ-C"],
                fps=10,
                highlight_stride=2,
                target_positions_ned={"OBJ-A": (377.2, 228.8), "OBJ-B": (-45.5, 534.8), "OBJ-C": (47.8, 493.9)},
                encode_h264=False,
            )
            rec.set_phase("第一阶段：首批三机广域采集", {"OBJ-A": "Scout-1", "OBJ-B": "Scout-2", "OBJ-C": "Scout-3"})
            rec.set_status("OBJ-A", "悬停采图（正俯视）")
            rec.set_vlm_badge("OBJ-A", "初评：疑似受损")
            rec.update_pose("Scout-1", 0.0, -6.0, -1.0, speed_mps=0.0, t_s=0.0)
            rec.update_pose("Scout-1", 377.2, 228.8, -75.0, speed_mps=15.0, t_s=25.0)
            rec.update_pose("Inspect-2", -48.6, 531.5, -15.0, speed_mps=2.5, t_s=60.0)
            frame = np.full((360, 640, 3), (60, 120, 180), dtype=np.uint8)
            for tid in ("OBJ-A", "OBJ-B", "OBJ-C"):
                rec.ingest_frame_bgr(tid, "chase", frame, timestamp=1)
                rec.ingest_frame_bgr(tid, "front", frame, timestamp=1)
            rec.write_summary_card("视觉语言模型终评结果（Qwen3.5-9B）", ["目标A（飞机）：终评结论 = 确认受损"], duration_s=0.5)
            metrics = rec.stop()
            self.assertGreaterEqual(metrics["realtime_frames"], 5)
            self.assertEqual(metrics["resolution"], [CANVAS_W, CANVAS_H])
            cap = cv2.VideoCapture(metrics["realtime_video"])
            ok, img = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(img.shape, (CANVAS_H, CANVAS_W, 3))

    def test_real_manifest_mock_replay_stress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(
                [
                    "--vlm",
                    "mock",
                    "--stress-test",
                    "--mission-id",
                    "unit-replay-stress",
                    "--output-dir",
                    str(Path(tmp) / "replay"),
                ]
            )
            report = run_replay(args)
            self.assertEqual(report["summary"]["total_targets"], 3)
            self.assertEqual(report["summary"]["completed_targets"], 3)
            self.assertEqual(report["summary"]["recheck_tasks_created"], 3)
            self.assertGreaterEqual(report["summary"]["duplicate_uploads_ignored"], 6)
            # Verify rich semantic output fields are present in both stages and rendered report
            first = report["targets"][0]
            s1_res = first["stage1_assessment"]["result"]
            s2_res = first["stage2_assessment"]["result"]
            for key in ("target_type", "suspected_damaged_parts", "visible_damage_description", "initial_damage_level", "secondary_task"):
                self.assertIn(key, s1_res)
            for key in ("task_type", "focus_part", "desired_viewpoint", "task_instruction", "priority"):
                self.assertIn(key, s1_res["secondary_task"])
            for key in ("target_type", "damage_severity", "confirmed_damaged_parts", "damage_description", "recommended_followup"):
                self.assertIn(key, s2_res)
            md_text = (Path(tmp) / "replay" / "report.md").read_text(encoding="utf-8")
            self.assertIn("语义评估总览", md_text)
            self.assertIn("二次复查任务单与交接记录", md_text)
            self.assertIn("确认受损部位", md_text)

    def test_visual_servo_opening_step_and_overlays(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available in current interpreter")
        from coordination.video_recorder import CANVAS_H, CANVAS_W, MultiUAVVideoRecorder
        from coordination.visual_servo import compute_nadir_opening_step, foreground_patch_mask

        # 0. Foreground depth gating masks out 240m twin background while keeping 15m foreground
        d_gate = np.full((360, 640), 240.0, dtype=np.float32)
        d_gate[120:260, 290:350] = 15.0
        fg = foreground_patch_mask(d_gate, grid=32, max_depth_m=30.0)
        self.assertTrue(bool(fg[16, 16]))
        self.assertFalse(bool(fg[4, 28]))

        # 1. Synthetic nadir depth image: 16m flat roof with a 38m deep circular hole offset to top-right
        h, w = 360, 640
        yy, xx = np.mgrid[:h, :w]
        focal = w / 2.0
        ray_norm = np.sqrt(1.0 + ((xx + 0.5 - w / 2.0) / focal) ** 2 + ((yy + 0.5 - h / 2.0) / focal) ** 2)
        depth_offset = (16.0 * ray_norm).astype(np.float32)
        hole_offset = ((xx - (w // 2 + 55)) ** 2 + (yy - (h // 2 - 45)) ** 2) <= (45 ** 2)
        depth_offset[hole_offset] = (38.0 * ray_norm[hole_offset]).astype(np.float32)

        step1 = compute_nadir_opening_step(depth_offset)
        self.assertTrue(step1["detected"])
        self.assertGreater(step1["forward_mps"], 0.0)  # hole is above center -> fly forward
        self.assertGreater(step1["right_mps"], 0.0)    # hole is right of center -> fly right
        self.assertFalse(step1["aligned"])
        self.assertGreater(step1["clearance_m"], 1.6)

        # 2. Centered circular hole -> aligned + vertical shaft clear
        depth_center = (16.0 * ray_norm).astype(np.float32)
        hole_center = ((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= (55 ** 2)
        depth_center[hole_center] = (38.0 * ray_norm[hole_center]).astype(np.float32)
        step2 = compute_nadir_opening_step(depth_center)
        self.assertTrue(step2["detected"])
        self.assertTrue(step2["aligned"])
        self.assertTrue(step2["shaft_clear"])

        # 3. Mission config contains visual_servo sections for all 3 targets
        mission = json.loads((ROOT / "configs" / "missions" / "coordination_sim_6uav.json").read_text(encoding="utf-8"))
        modes = {t["target_id"]: t["visual_servo"]["mode"] for t in mission["targets"]}
        self.assertEqual(modes, {"OBJ-A": "dino_oblique", "OBJ-B": "opening_nadir", "OBJ-C": "dino_oblique"})

        # 4. Recorder renders both DINO heatmap & opening mask overlays cleanly
        with tempfile.TemporaryDirectory() as tmp:
            rec = MultiUAVVideoRecorder(
                Path(tmp) / "videos",
                ["OBJ-A", "OBJ-B", "OBJ-C"],
                fps=10,
                highlight_stride=2,
                encode_h264=False,
            )
            rec.set_phase("第二阶段：次批三机视觉伺服复查", {"OBJ-A": "Inspect-1", "OBJ-B": "Inspect-2", "OBJ-C": "Inspect-3"})
            frame = np.full((360, 640, 3), (70, 90, 110), dtype=np.uint8)
            for tid in ("OBJ-A", "OBJ-B", "OBJ-C"):
                rec.ingest_frame_bgr(tid, "chase", frame, timestamp=1)
                rec.ingest_frame_bgr(tid, "front", frame, timestamp=1)
            sim = np.zeros((32, 32), dtype=np.float32)
            sim[14:18, 14:18] = 0.85
            rec.set_servo_overlay(
                "OBJ-A",
                {
                    "mode": "dino",
                    "badge": "实时视觉伺服",
                    "phase_cn": "视觉抵近",
                    "sim_2d": sim,
                    "cx": 0.1,
                    "cy": -0.05,
                    "peak": 0.85,
                    "depth_m": 18.2,
                    "cmd_fwd": 1.6,
                    "cmd_right": 0.0,
                    "cmd_down": 0.4,
                    "confirm": 1,
                    "confirm_target": 3,
                    "locked": True,
                },
            )
            rec.set_servo_overlay(
                "OBJ-B",
                {
                    "mode": "opening",
                    "badge": "实时视觉伺服",
                    "phase_cn": "破口对准 2/3",
                    "gap_mask": step2["visible_gap_mask"],
                    "cx": step2["cx"],
                    "cy": step2["cy"],
                    "err_horiz_m": step2["err_horiz_m"],
                    "clearance_m": step2["clearance_m"],
                    "cmd_fwd": 0.1,
                    "cmd_right": -0.1,
                    "cmd_down": 0.0,
                    "confirm": 2,
                    "confirm_target": 3,
                    "locked": True,
                },
            )
            composed = rec._compose_frame()
            self.assertEqual(composed.shape, (CANVAS_H, CANVAS_W, 3))
            rec.stop()


if __name__ == "__main__":
    unittest.main()
