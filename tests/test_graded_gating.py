"""Offline unit tests for ego-motion spatial tracking, PAVA calibration, and GradedGate."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from perception.graded_gating import pava_isotonic, DistanceBinCalibrator, SpatialTracker, GradedGate


def _cam(x=0.0, y=0.0, z=-50.0):
    return {'pos_x': x, 'pos_y': y, 'pos_z': z, 'rot_w': 1.0, 'rot_x': 0.0, 'rot_y': 0.0, 'rot_z': 0.0}


class GradedGatingTests(unittest.TestCase):
    def test_pava_isotonic_is_monotone_and_bounded(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        labels = [0, 0, 1, 0, 1, 0, 1, 1]
        kx, ky = pava_isotonic(scores, labels, smooth=1.0)
        self.assertTrue(all(kx[i] < kx[i + 1] for i in range(len(kx) - 1)))
        self.assertTrue(all(ky[i] <= ky[i + 1] + 1e-12 for i in range(len(ky) - 1)))
        self.assertTrue(all(0.0 < y < 1.0 for y in ky))

    def test_calibrator_roundtrip_and_llr_sign(self):
        scores = np.linspace(0.05, 0.95, 100)
        labels = (scores >= 0.6).astype(float)
        ranges = np.full(100, 100.0)
        cal = DistanceBinCalibrator().fit(scores, labels, ranges)
        cal2 = DistanceBinCalibrator.from_dict(cal.to_dict())
        p_hi, llr_hi, b_hi = cal2.probability_and_llr(0.9, 100.0)
        p_lo, llr_lo, b_lo = cal2.probability_and_llr(0.1, 100.0)
        self.assertEqual(b_hi, '80-120')
        self.assertEqual(b_lo, '80-120')
        self.assertGreater(p_hi, p_lo)
        self.assertGreater(llr_hi, 0.0)
        self.assertLess(llr_lo, 0.0)

    def test_spatial_tracker_uses_only_relative_motion_and_rejects_jump(self):
        tracker = SpatialTracker(max_ray_angle_deg=5.0)
        box = [860, 440, 1060, 640]
        # Target sits 100 m ahead along +X; drone moves +10 m along +X -> depth becomes 90 m, center unchanged.
        r0 = tracker.update([0.5, 0.5], box, 100.0, _cam(0.0, 0.0, -50.0), 0.0)
        r1 = tracker.update([0.5, 0.5], box, 90.0, _cam(10.0, 0.0, -50.0), 0.5)
        self.assertTrue(r0['consistent'] and r1['consistent'])
        self.assertEqual(r1['streak'], 2)
        self.assertLess(r1['ray_ang_deg'], 0.01)
        # World translation invariance: shift both frames by +500 m gives identical angle
        t2 = SpatialTracker(max_ray_angle_deg=5.0)
        t2.update([0.5, 0.5], box, 100.0, _cam(500.0, 300.0, -50.0), 0.0)
        r1_shift = t2.update([0.5, 0.5], box, 90.0, _cam(510.0, 300.0, -50.0), 0.5)
        self.assertAlmostEqual(r1['ray_ang_deg'], r1_shift['ray_ang_deg'], places=6)
        # Candidate jumps to far right corner -> inconsistent
        r2 = tracker.update([0.9, 0.5], [1650, 440, 1850, 640], 90.0, _cam(10.0, 0.0, -50.0), 1.0)
        self.assertFalse(r2['consistent'])
        self.assertEqual(r2['streak'], 0)

    def test_graded_gate_permits_far_approach_and_blocks_unconfirmed_near_handoff(self):
        scores = np.linspace(0.05, 0.95, 120)
        labels = (scores >= 0.55).astype(float)
        cal = DistanceBinCalibrator().fit(scores, labels, np.full(120, 100.0))
        gate = GradedGate(cal, {'far_consistent_frames': 2, 'confirm_llr_threshold': 2.0, 'confirm_min_mid_frames': 3})
        box = [860, 440, 1060, 640]
        # Far range: even 10 frames of high score (>80m) must NOT finalize identity_confirmed
        for i in range(10):
            g = gate.update(0.90, [0.5, 0.5], box, 110.0 - i, _cam(float(i)), i * 0.5)
        self.assertEqual(g['phase'], 'FAR_APPROACH')
        self.assertTrue(g['motion_permitted'])
        self.assertFalse(g['identity_confirmed'])
        # Jumping straight to <=45 m without mid-range confirmation must block near handoff
        g_block = gate.update(0.75, [0.5, 0.5], box, 40.0, _cam(70.0), 5.5)
        self.assertFalse(g_block['near_handoff_permitted'])
        self.assertFalse(g_block['motion_permitted'])

    def test_graded_gate_rejects_low_score_sequence_and_requires_margin_when_enabled(self):
        scores = np.linspace(0.05, 0.95, 120)
        labels = (scores >= 0.55).astype(float)
        cal = DistanceBinCalibrator().fit(scores, labels, np.full(120, 60.0))
        gate = GradedGate(cal, {'require_same_class_margin': True, 'min_same_class_logit_margin': 0.5,
                                'confirm_llr_threshold': 2.0, 'confirm_min_mid_frames': 2})
        box = [860, 440, 1060, 640]
        # High raw score in mid-range, but another same-class template scores higher -> must NOT confirm
        for i in range(4):
            out = gate.update(0.85, [0.5, 0.5], box, 60.0 - 3 * i, _cam(3.0 * i), i * 0.5,
                              same_class_other_scores=[0.90, 0.40])
        self.assertFalse(out['identity_confirmed'])
        # Absent/wrong-category low scores -> hits reject boundary even with stationary damping
        gate2 = GradedGate(cal, {'reject_llr_threshold': -1.5})
        for i in range(6):
            out2 = gate2.update(0.10, [0.5, 0.5], box, 100.0, _cam(3.0 * i), i * 0.5)
        self.assertTrue(out2['identity_rejected'])
        self.assertFalse(out2['motion_permitted'])


if __name__ == '__main__':
    unittest.main()
