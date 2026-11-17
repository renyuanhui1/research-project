"""Metrics and image-only candidate coverage, including plateau and NMS behavior."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation.eval_template_selection import (auc, wilson, point_hit, crop_coverage,
                                                 peak_candidates, coverage_summary, summarize)
from perception.template_verification import proposal_box


class TemplateSelectionTests(unittest.TestCase):
    def test_auc_ties_and_empty_acceptance_do_not_claim_perfect_precision(self):
        self.assertEqual(auc([.5, .5], [True, False]), .5)
        self.assertEqual(auc([.9, .1], [True, False]), 1.)
        self.assertIsNone(auc([.9], [True]))
        self.assertIsNone(wilson(0, 0))
        low, high = wilson(38, 40)
        self.assertAlmostEqual(low, .83496, places=4)
        self.assertTrue(low < .95 < high)

    def test_top1_reproduces_frozen_crop_and_nms_retains_distinct_peaks(self):
        heatmap = np.full((16, 16), -10., np.float32)
        heatmap[2:4, 2:4] = 5.
        heatmap[11, 12] = 4.
        cfg = {'crop_margin': 1.5, 'crop_minimum_fraction': .12, 'crop_logit_delta': 2.}
        candidates = peak_candidates(heatmap, (100, 200), cfg, 3, .5)
        self.assertEqual(candidates[0]['box_xyxy'], proposal_box(heatmap, (100, 200), 1.5, .12, 2.))
        self.assertEqual(candidates[0]['point_xy'], [2.5 / 16, 2.5 / 16])
        self.assertTrue(any(c['point_xy'] == [12.5 / 16, 11.5 / 16] for c in candidates))
        constant = peak_candidates(np.ones((8, 8), np.float32), (100, 200), cfg, 5, .5)
        self.assertEqual(len(constant), 1)

    def test_crop_and_point_coverage_are_separate_and_use_native_label_coordinates(self):
        mask = np.zeros((10, 20), bool); mask[2:4, 8:12] = True
        self.assertTrue(point_hit(mask, [.5, .3]))
        self.assertFalse(point_hit(mask, [1., .3]))
        self.assertEqual(crop_coverage(mask, [80, 20, 100, 40], (100, 200)), .5)
        rows = [{'point_hits': [False, True], 'crop_coverages': [.5, 1.]},
                {'point_hits': [False, False], 'crop_coverages': [0., 0.]}]
        result = coverage_summary(rows, [1, 3, 5], .5)
        self.assertEqual(result['1']['point_recall'], 0.)
        self.assertEqual(result['1']['crop_recall'], .5)
        self.assertEqual(result['3']['point_recall'], .5)
        self.assertEqual(result['5']['point_recall'], .5)

    def test_invisible_positives_do_not_inflate_recall_denominator(self):
        rows = [{'positive': True, 'visible_pixels': 10, 'score': .9, 'accepted': True,
                 'verified_correct': True, 'rgb_inlier_fraction': .5},
                {'positive': True, 'visible_pixels': 0, 'score': .1, 'accepted': False,
                 'verified_correct': False, 'rgb_inlier_fraction': 0.},
                {'positive': False, 'visible_pixels': 10, 'score': .95, 'accepted': True,
                 'verified_correct': False, 'rgb_inlier_fraction': .8}]
        result = summarize(rows, 20)
        self.assertEqual(result['verified_positive_recall'], 1.)
        self.assertEqual(result['accepted_precision'], .5)
        self.assertEqual(result['negative_false_accept_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
