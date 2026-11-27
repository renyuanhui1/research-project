"""Offline unit tests for src/visualization/rerun_replay.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from visualization.rerun_replay import (  # noqa: E402
    build_harbor_compare_rrd,
    build_opening_rrd,
    ned_to_enu,
    sanitize_depth_m,
)


class TestRerunReplay(unittest.TestCase):
    def test_ned_to_enu_and_sanitize_depth(self) -> None:
        self.assertEqual(ned_to_enu([10.0, 20.0, -30.0]), [20.0, 10.0, 30.0])
        self.assertEqual(ned_to_enu({"position": {"x": -5.0, "y": 8.0, "z": -12.0}}), [8.0, -5.0, 12.0])

        raw = np.array([[np.nan, 0.05, 15.5, 200.0], [1.0, np.inf, 42.0, -3.0]], dtype=np.float32)
        clean = sanitize_depth_m(raw, min_m=0.1, max_m=120.0)
        self.assertEqual(clean.dtype, np.float32)
        np.testing.assert_allclose(clean, [[0.0, 0.0, 15.5, 0.0], [1.0, 0.0, 42.0, 0.0]])

    def test_build_opening_rrd_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            flight = tmp_path / "control"
            (flight / "frames").mkdir(parents=True)
            (flight / "rviz").mkdir(parents=True)

            np.savez(
                flight / "rviz" / "rgbd-map.npz",
                points_enu_m=np.array([[0.0, 0.0, 10.0], [1.0, 1.0, 10.0]], dtype=np.float32),
                rgb=np.array([[200, 150, 100], [100, 150, 200]], dtype=np.uint8),
            )

            rows = []
            for i in range(3):
                img = np.full((64, 80, 3), 40 * (i + 1), dtype=np.uint8)
                cv2.imwrite(str(flight / "frames" / f"{i:04d}.png"), img)
                cv2.imwrite(str(flight / "frames" / f"{i:04d}-matching.png"), img)
                np.save(flight / "frames" / f"{i:04d}-depth.npy", np.full((32, 40), 40.0 - i * 10.0, dtype=np.float32))
                rows.append(
                    {
                        "step": i,
                        "time_s": 1.0 + i * 0.5,
                        "state": "ALIGN" if i < 2 else "ENTERED",
                        "pose": {"position": {"x": 10.0, "y": 20.0 + i, "z": -(60.0 - i * 15.0)}},
                        "horizontal_speed_mps": 0.2,
                        "down_speed_mps": 0.5 if i >= 1 else 0.0,
                        "error_ne_m": [0.4 - i * 0.1, 0.2],
                        "image": f"frames/{i:04d}.png",
                        "diagnostic": f"frames/{i:04d}-matching.png",
                        "depth": f"frames/{i:04d}-depth.npy",
                        "entry": {"roof_remaining_m": 10.0 - i * 9.0, "minimum_center_floor_range_m": 25.0 - i * 5.0},
                    }
                )
            (flight / "signals.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            cv2.imwrite(str(flight / "interior-observation.png"), np.full((64, 80, 3), 180, dtype=np.uint8))

            out_rrd = tmp_path / "opening.rrd"
            res = build_opening_rrd(flight, out_rrd)
            self.assertTrue(out_rrd.exists())
            self.assertEqual(res["frames"], 3)
            self.assertGreater(res["size_bytes"], 1000)
            self.assertTrue(res["has_map"])
            self.assertTrue(res["has_interior"])
            self.assertEqual(res["altitude_range_m"], [30.0, 60.0])

    def test_build_harbor_compare_rrd_synthetic(self) -> None:
        def make_run(root: Path, *, controller: str, final_alt: float, succeeded: bool, stop_reason: str) -> None:
            tdir = root / "Building_098"
            tdir.mkdir(parents=True)
            cv2.imwrite(str(tdir / "nadir_overview.jpg"), np.full((60, 80, 3), 120, dtype=np.uint8))
            cv2.imwrite(str(tdir / "nadir_breach_overlay.jpg"), np.full((60, 80, 3), 150, dtype=np.uint8))
            np.savez(tdir / "nadir_overview.npz", depth_m=np.full((30, 40), 20.0, dtype=np.float16))
            if succeeded:
                cv2.imwrite(str(tdir / "interior_forward.jpg"), np.full((60, 80, 3), 200, dtype=np.uint8))

            summary = {
                "controller": controller,
                "wind_ned_mps": [0.0, 18.0, 0.0],
                "targets": [
                    {
                        "target_id": "Building_098",
                        "nadir_capture": {
                            "cam_pos_ned": [-773.8, -90.8, -34.0],
                            "cam_rot_wxyz": [0.7071, 0.0, -0.7071, 0.0],
                            "drone_pos_ned": [-773.5, -90.5, -34.3],
                        },
                    }
                ],
            }
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            trace = {
                "controller": controller,
                "stop_reason": stop_reason,
                "succeeded": succeeded,
                "final_alt_m": final_alt,
                "final_pose_ned": [-774.0, -80.0 if not succeeded else -94.0, -final_alt],
                "trace": [
                    {
                        "step": i,
                        "t_s": i * 0.5,
                        "phase": "align" if i == 0 else "descend",
                        "alt_m": 34.0 - (i * 10.0 if succeeded else 0.0),
                        "detected": True,
                        "err_horiz_m": 4.0 - (i * 1.5 if succeeded else -0.5),
                        "cmd_body": [0.1, -0.8, 0.2 if succeeded else 0.0],
                        "cbf_intervention": 0.9 if controller == "cbf_qp" else None,
                    }
                    for i in range(3)
                ],
            }
            (tdir / "opening_servo_trace.json").write_text(json.dumps(trace), encoding="utf-8")

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            base_dir = tmp_path / "baseline"
            meth_dir = tmp_path / "dob_cbf"
            make_run(base_dir, controller="baseline_p", final_alt=34.5, succeeded=False, stop_reason="opening_lost")
            make_run(meth_dir, controller="cbf_qp", final_alt=6.0, succeeded=True, stop_reason="interior_reached")

            out_rrd = tmp_path / "compare.rrd"
            res = build_harbor_compare_rrd(base_dir, meth_dir, out_rrd)
            self.assertTrue(out_rrd.exists())
            self.assertEqual(res["target_id"], "Building_098")
            self.assertEqual(res["baseline_steps"], 3)
            self.assertEqual(res["method_steps"], 3)
            self.assertTrue(res["has_point_cloud"])
            self.assertGreater(res["size_bytes"], 1000)


if __name__ == "__main__":
    unittest.main()
