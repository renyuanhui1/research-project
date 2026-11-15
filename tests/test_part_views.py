"""Geometry of the fixed capture plan and actual local-search recovery decisions."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from capture.collect_part_views import plan_views, decode_rpc_depth
from evaluation.eval_part_reacquisition import run_methods


class IndependentPartViewTests(unittest.TestCase):
    def test_rpc_depth_byte_lists_preserve_half_and_single_precision_pixel_layout(self):
        for dtype, encoding in [(np.float16, '16FC1'), (np.float32, '32FC1')]:
            expected = np.array([[1., 2.], [3., np.inf]], dtype=dtype)
            message = dict(height=2, width=2, encoding=encoding,
                           data=list(expected.tobytes()))
            np.testing.assert_array_equal(decode_rpc_depth(message), expected)
            message['data'] = expected.reshape(-1).tolist()
            np.testing.assert_array_equal(decode_rpc_depth(message), expected)

    def test_every_planned_reference_is_inside_the_fixed_camera_frustum(self):
        protocol = json.loads((ROOT/'configs/missions/f14_tail_reacquisition_validation.json').read_text())
        reference = np.array([555., 129., -5.])
        cases = plan_views(protocol, reference)
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({c['id'] for c in cases}), 18)
        pitch = math.radians(-protocol['camera_pitch_deg'])
        mount = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                          [-math.sin(pitch), 0, math.cos(pitch)]])
        for case in cases:
            yaw = math.radians(case['yaw_deg'])
            body = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            camera = np.array(case['position_ned_m'])+body@np.array([0, 0, .3])
            local = (reference-camera)@(body@mount)
            self.assertGreater(local[0], 0)
            self.assertLess(abs(local[1]/local[0]), 1.)
            self.assertLess(abs(local[2]/local[0]), 1080/1920)

    def search_fixture(self, candidate, duplicated=False):
        rng = np.random.default_rng(71)
        template = rng.integers(0, 256, (40, 24, 3), dtype=np.uint8)
        rgb = rng.integers(20, 55, (220, 300, 3), dtype=np.uint8)
        rgb[80:120, 124:148] = template
        if duplicated:
            rgb[80:120, 160:184] = template
        fp = SimpleNamespace(dev='cpu', selected_part='tail', refinement_state={'tail': {'point_px': [0, 0]}},
                             query=lambda image: None,
                             local_references={'tail': {'rgb': template, 'anchor': [12, 20]}},
                             last_diagnostics={'crop_xyxy': [0, 0, 300, 220], 'whole_peak': .7,
                                               'tail': {'local_similarity': np.full(1024, .7),
                                                        'refinement': {'coarse_candidate_px': candidate}}})
        geometry = SimpleNamespace(hits=lambda sample, point: np.linalg.norm(np.array(point)-[136, 100]) < 1.)
        sample = dict(name='synthetic', seed=None, label_kind='known_texture_anchor', depth_m=10.)
        result = run_methods(sample, rgb, 'original', fp, geometry,
                             ['cold', 'radius_3x', 'cold_then_radius_3x'])
        self.assertEqual(fp.refinement_state, {})
        return {r['method']: r for r in result}

    def test_expansion_recovers_anchor_outside_initial_window_without_history(self):
        records = self.search_fixture([136, 130])
        self.assertFalse(records['cold']['accepted'])
        self.assertTrue(records['cold_then_radius_3x']['accepted'])
        self.assertTrue(records['cold_then_radius_3x']['hits_label'])
        self.assertEqual(records['cold_then_radius_3x']['search_stage'], 'expanded')

    def test_initial_acceptance_survives_a_competing_instance_in_the_wider_window(self):
        records = self.search_fixture([136, 100], duplicated=True)
        self.assertTrue(records['cold']['accepted'])
        self.assertFalse(records['radius_3x']['accepted'])
        self.assertTrue(records['cold_then_radius_3x']['accepted'])
        self.assertTrue(records['cold_then_radius_3x']['hits_label'])
        self.assertEqual(records['cold_then_radius_3x']['search_stage'], 'initial')


if __name__ == '__main__':
    unittest.main()
