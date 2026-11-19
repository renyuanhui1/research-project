"""Ray geometry, observed-pixel provenance, matching and empty calibration semantics."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from perception.birdseye_matching import backproject, birdseye, source_point, sift_candidates, DinoMatcher
from evaluation.eval_birdseye_matching import calibrate, summarize


class BirdseyeTests(unittest.TestCase):
    def setUp(self):
        q = Rotation.from_euler('y', -90, degrees=True).as_quat()
        self.camera = dict(zip(['rot_' + k for k in 'xyzw'], q))
        self.cfg = dict(gsd_m=.2, maximum_side=128, max_ray_depth_m=500,
                        horizontal_radius_m=250, splat_radius_factor=.75)

    def test_ray_length_pixel_centers_and_nadir_axes(self):
        depth = np.full((2, 2), 3., np.float32)
        points = backproject(depth, self.camera, 90)
        np.testing.assert_allclose(np.linalg.norm(points, axis=-1), depth)
        self.assertGreater(points[0, 0, 0], 0)  # top pixel is north
        self.assertLess(points[0, 0, 1], 0)     # left pixel is west
        self.assertTrue((points[..., 2] > 0).all())

    def test_exclusion_provenance_and_horizontal_position_invariance(self):
        rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        depth = np.full((4, 4), 10., np.float32)
        excluded = np.zeros((4, 4), bool); excluded[1, 1] = True
        a = birdseye(rgb, depth, self.camera, 90, self.cfg, excluded)
        b = birdseye(rgb, depth, dict(self.camera, pos_x=12345, pos_y=-765, pos_z=-800), 90, self.cfg, excluded)
        np.testing.assert_array_equal(a['source'], b['source'])
        self.assertFalse((a['source'] == 5).any())
        y, x = np.argwhere(a['source'] >= 0)[0]; i = a['source'][y, x]
        self.assertEqual(source_point(a, [(x + .5) / a['source'].shape[1], (y + .5) / a['source'].shape[0]]),
                         [(i % 4 + .5) / 4, (i // 4 + .5) / 4])
        np.testing.assert_array_equal(a['rgb'][y, x], rgb.reshape(-1, 3)[i])

    def test_empty_depth_does_not_generate_a_prediction(self):
        view = birdseye(np.zeros((2, 2, 3), np.uint8), np.full((2, 2), np.nan), self.camera, 90, self.cfg)
        self.assertIsNone(source_point(view, [.5, .5]))

    def test_top_surface_wins_in_a_shared_orthographic_cell(self):
        depth = np.full((4, 4), 10., np.float32); depth[1, 1] = 5
        rgb = np.zeros((4, 4, 3), np.uint8); rgb[1, 1] = [90, 30, 10]
        view = birdseye(rgb, depth, self.camera, 90, dict(self.cfg, gsd_m=100))
        # These four northwest rays share one cell; the closer top surface is pixel 5.
        self.assertEqual(int(view['source'][0, 0]), 5)
        np.testing.assert_array_equal(view['rgb'][0, 0], [90, 30, 10])

    def test_dino_cannot_select_an_unsupported_peak(self):
        import torch
        matcher = object.__new__(DinoMatcher)
        matcher.cfg = dict(minimum_patch_support=.25, nms_radius_cells=0)
        support = np.ones((4, 4), bool); support[:2, :2] = False
        candidates = matcher.candidates(torch.eye(4), torch.tensor([[1., .8, .6, .4]]), support, 4)
        self.assertEqual(len(candidates), 3)
        self.assertAlmostEqual(candidates[0]['score'], .8)
        self.assertTrue(all(support[int(p['view_point_xy'][1] * 4), int(p['view_point_xy'][0] * 4)] for p in candidates))

    def test_sift_geometric_matches_survive_rotation_scale_and_distractors(self):
        rng = np.random.default_rng(73)
        points = rng.uniform(10, 90, (20, 2)).astype(np.float32)
        descriptors = rng.normal(size=(20, 128)).astype(np.float32)
        transform = cv2.getRotationMatrix2D((0, 0), 35, 1.2); transform[:, 2] = [80, 110]
        query_points = np.c_[points, np.ones(len(points))] @ transform.T
        query = (np.r_[query_points, rng.uniform(0, 300, (10, 2))].astype(np.float32),
                 np.r_[descriptors, rng.normal(size=(10, 128))].astype(np.float32))
        cfg = dict(ratio=.8, minimum_inliers=4, seed=1731, reprojection_px=3, ransac_iterations=2000)
        candidates = sift_candidates(query, (points, descriptors), (300, 300), (100, 100), cfg, 5)
        self.assertEqual(len(candidates), 1); self.assertEqual(candidates[0]['inliers'], 20)
        self.assertTrue(any(np.allclose(np.array(candidates[0]['view_point_xy']) * 300, p) for p in query_points))

    def test_failed_calibration_rejects_unbounded_scores_and_keeps_ties(self):
        bad = [dict(available=True, score=20., correct=False), dict(available=True, score=10., correct=True)]
        self.assertIsNone(calibrate(bad, .95)['threshold'])
        tied = [dict(available=True, score=20., correct=True), dict(available=True, score=20., correct=False)]
        self.assertIsNone(calibrate(tied, .95)['threshold'])
        good = [dict(available=True, score=20., correct=True), dict(available=True, score=10., correct=False)]
        self.assertEqual(calibrate(good, .95)['threshold'], 15.)

    def test_empty_acceptance_and_topk_are_separate(self):
        rows = [dict(positive=True, visible_pixels=2, available=True, score=20., correct=False, point_hits=[False, True]),
                dict(positive=False, visible_pixels=0, available=True, score=50., correct=False, point_hits=[False])]
        result = summarize(rows, [1, 3], None)
        self.assertEqual(result['coverage']['1']['rate'], 0.)
        self.assertEqual(result['coverage']['3']['rate'], 1.)
        self.assertEqual(result['accepted'], 0); self.assertIsNone(result['accepted_precision'])


if __name__ == '__main__':
    unittest.main()
