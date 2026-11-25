"""Offline unit tests for Coastal Harbor Task 3 continuous linear inspection metrics."""

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

from harbor.line_inspection import (
    DEFAULT_LINE_CONFIG,
    evaluate_line_coverage_and_damage,
    flight_point_ned,
    line_point_ue,
    project_line_point,
)
from harbor.part_recheck import camera_rot_from_yaw_pitch


def _rot_to_quat_wxyz(r: np.ndarray) -> list[float]:
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    return [1.0, 0.0, 0.0, 0.0]


class HarborLineInspectionTests(unittest.TestCase):
    def test_synthetic_line_coverage_and_boundary_localization(self) -> None:
        cfg = json.loads(DEFAULT_LINE_CONFIG.read_text(encoding="utf-8"))
        spec = cfg["line_spec"]
        gt = cfg["evaluation_gt"]["damage_intervals_m"]
        gt_start, gt_end = gt[0]

        frames = []
        for idx, s in enumerate(np.linspace(0.0, 32.2, 15)):
            cam_ned, yaw_deg = flight_point_ned(spec, float(s))
            cam_rot = camera_rot_from_yaw_pitch(math.radians(yaw_deg), float(spec["camera_pitch_deg"]))
            # Project synthetic GT damage span into this camera frame
            us = []
            for ds in np.linspace(gt_start, gt_end, 40):
                proj = project_line_point(
                    pt_ue=line_point_ue(spec, float(ds)),
                    cam_pos_ned=cam_ned,
                    cam_rot=cam_rot,
                )
                if proj["in_frame"] and proj["u_norm"] is not None:
                    us.append(proj["u_norm"])
            has_dmg = len(us) >= 3
            vlm_res = {
                "roof_visible": True,
                "has_roof_damage": has_dmg,
                "damage_u_start": min(us) if has_dmg else 0.0,
                "damage_u_end": max(us) if has_dmg else 0.0,
            }
            frames.append(
                {
                    "frame_index": idx,
                    "s_target_m": float(s),
                    "capture": {
                        "cam_pos_ned": cam_ned,
                        "cam_rot_wxyz": _rot_to_quat_wxyz(cam_rot),
                    },
                    "vlm_result": vlm_res,
                }
            )

        m = evaluate_line_coverage_and_damage(
            frames=frames,
            line_spec=spec,
            gt_damage_intervals=gt,
        )
        self.assertEqual(m["effective_coverage_ratio"], 1.0)
        self.assertEqual(m["missed_segment_count"], 0)
        self.assertLessEqual(m["damage_start_error_m"], 0.6)
        self.assertLessEqual(m["damage_end_error_m"], 0.6)
        self.assertGreater(m["damage_interval_iou"], 0.90)


if __name__ == "__main__":
    unittest.main()
