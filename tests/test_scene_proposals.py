"""Instance weighting, unknown pixels and fold-specific supervision exclusion."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training.train_scene_proposals import balanced_targets, score_predictions
from perception.template_matcher import localization_loss


class SceneProposalTests(unittest.TestCase):
    def make_data(self):
        original = torch.zeros(1, 4, 4); original[0, :2, :2] = 1
        other = torch.zeros(4, 4); other[3, 3] = 1
        held = torch.zeros(4, 4); held[2, 3] = 1
        return {'mask': original, 'valid': torch.ones_like(original),
                'context_targets': {0: ['other', 'held']},
                'context_mask': {(0, 'other'): other, (0, 'held'): held}}

    def test_large_and_small_instances_receive_equal_mass_without_mutation(self):
        data = self.make_data(); previous = data['mask'].clone()
        target, audit = balanced_targets(data, [0], {'other'})
        self.assertAlmostEqual(float(target[0, :2, :2].sum()), .5)
        self.assertAlmostEqual(float(target[0, 3, 3]), .5)
        self.assertEqual(float(target[0, 2, 3]), 0.)
        torch.testing.assert_close(data['mask'], previous, rtol=0, atol=0)
        self.assertEqual(audit['extra_instances_used'], 1)
        self.assertEqual(audit['extra_instances_excluded'], 1)

    def test_unknown_instance_contributes_neither_target_mass_nor_gradient(self):
        data = self.make_data(); data['valid'][0, 3, 3] = 0
        target, _ = balanced_targets(data, [0], {'other'})
        self.assertEqual(float(target[0, 3, 3]), 0.)
        self.assertAlmostEqual(float(target.sum()), 1.)
        logits = torch.zeros_like(target, requires_grad=True)
        localization_loss(logits, target, data['valid']).backward()
        self.assertEqual(float(logits.grad[0, 3, 3]), 0.)

    def test_single_instance_has_the_same_loss_and_gradient_as_original_supervision(self):
        data = self.make_data(); target, _ = balanced_targets(data, [0], set())
        a = torch.randn_like(target, requires_grad=True); b = a.detach().clone().requires_grad_(True)
        first = localization_loss(a, data['mask'], data['valid']); second = localization_loss(b, target, data['valid'])
        first.backward(); second.backward()
        torch.testing.assert_close(first, second); torch.testing.assert_close(a.grad, b.grad)

    def test_empty_supervision_does_not_create_a_positive_instance(self):
        data = {'mask': torch.zeros(1, 4, 4), 'valid': torch.ones(1, 4, 4)}
        target, audit = balanced_targets(data, [0], set())
        self.assertEqual(float(target.sum()), 0.); self.assertEqual(audit['visible_instances'], 0)

    def test_context_gain_is_split_between_fitted_and_held_appearances(self):
        data = {'records': [{'name': 'frame', 'target_id': 'main', 'rgb_shape_hw': [4, 4]}],
                'groups': {'main': {'id': 'main', 'index': 1, 'appearance_group': 'held'},
                           'other': {'id': 'other', 'index': 2, 'appearance_group': 'fit'}}}
        context = {'samples': [{'sample_name': 'frame', 'label_targets': [{'target_id': 'other', 'index': 2}]}]}
        labels = {'frame': {'ids': np.array([[1, 0], [0, 2]], np.uint16)}}
        candidates = {0: [{'point_xy': [.25, .25], 'box_xyxy': [0, 0, 2, 2]},
                          {'point_xy': [.75, .75], 'box_xyxy': [2, 2, 4, 4]}]}
        metrics, rows, _ = score_predictions(data, context, labels, candidates,
            {0: {'held_groups': ['held']}}, {'candidate_counts': [1, 3, 5],
             'minimum_crop_coverage': .5, 'area_fraction_bins': [0, .1, 1]})
        self.assertEqual(rows[0]['request_scope'], 'held'); self.assertEqual(rows[1]['request_scope'], 'fit')
        self.assertEqual(metrics['context_by_request_scope']['fit']['1']['point_recall'], 0.)
        self.assertEqual(metrics['context_by_request_scope']['fit']['3']['point_recall'], 1.)


if __name__ == '__main__':
    unittest.main()
