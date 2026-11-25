"""Offline unit tests for Coastal Harbor Task 2 roof-breach measurement and passability gating."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from harbor.indoor_entry import measure_breach_from_nadir_depth, score_breach_measurement


class HarborIndoorEntryTests(unittest.TestCase):
    def test_passable_open_breach_and_beam_obstructed_breach(self) -> None:
        # Nadir perspective range map 640x360 at 20 m above roof (focal = 320 px -> 1 px = 0.0625 m)
        # Create a 5.0 m x 8.0 m opening (80 px x 128 px) with floor at 30 m
        yy, xx = np.mgrid[:360, :640]
        ray_norm = np.sqrt(1.0 + ((xx + 0.5 - 320.0) / 320.0) ** 2 + ((yy + 0.5 - 180.0) / 320.0) ** 2).astype(
            np.float32
        )
        planar_open = np.full((360, 640), 20.0, dtype=np.float32)
        planar_open[116:244, 280:360] = 30.0
        depth_open = planar_open * ray_norm
        m_open = measure_breach_from_nadir_depth(
            depth_open, fov_deg=90.0, min_layer_gap_m=5.0, min_passable_clearance_radius_m=1.6
        )
        self.assertTrue(m_open["detected"])
        self.assertTrue(m_open["passable"])
        self.assertAlmostEqual(m_open["measured_width_m"], 5.0, delta=0.2)
        self.assertAlmostEqual(m_open["measured_area_m2"], 40.0, delta=1.5)
        self.assertGreater(m_open["max_clearance_radius_m"], 2.3)

        # Now insert two crossbeams splitting the 8.0 m length into narrow strips (2.2 m each -> radius 1.1 m < 1.6 m)
        planar_blocked = planar_open.copy()
        planar_blocked[152:158, 280:360] = 20.0
        planar_blocked[194:200, 280:360] = 20.0
        depth_blocked = planar_blocked * ray_norm
        m_blk = measure_breach_from_nadir_depth(
            depth_blocked, fov_deg=90.0, min_layer_gap_m=5.0, min_passable_clearance_radius_m=1.6
        )
        self.assertTrue(m_blk["detected"])
        self.assertFalse(m_blk["passable"])
        self.assertEqual(m_blk["decision"], "blocked_abort")
        self.assertLess(m_blk["max_clearance_radius_m"], 1.5)

        err = score_breach_measurement(m_open, {"open_bay_width_m": 5.0, "open_bay_area_m2": 40.0})
        self.assertLess(err["width_abs_error_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
