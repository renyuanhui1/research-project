"""Check reference geometry and target-level leakage without UE or model weights."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from capture.build_target_catalog import crop_bounds, project, resolve_splits, template_masks


class TargetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.message = dict(width=128, height=96, pos_x=0., pos_y=0., pos_z=-10.,
                            rot_w=2**-.5, rot_x=0., rot_y=-2**-.5, rot_z=0.)
        self.targets = [dict(index=i, id=str(i), appearance_group=str(i), category='hangar') for i in (1, 2, 3)]
        self.protocol = dict(train=[1], validation=[2], test=[3], disjoint_primary_meshes=True)

    def test_nadir_projection_axes_and_nondefault_fov(self):
        for fov in (60., 90.):
            uv, focal = project(np.array([[0., 0., 0.], [2., 3., 0.]]), self.message, fov)
            np.testing.assert_allclose(uv[0], [64, 48], atol=1e-10)
            np.testing.assert_allclose(uv[1], [64+.3*focal, 48-.2*focal], atol=1e-10)
        with self.assertRaisesRegex(ValueError, 'front'):
            project(np.array([[0., 0., -11.]]), self.message, 90.)

    def test_crop_clamping_preserves_geometry_and_rejects_truncation(self):
        points = np.array([[20., 10.], [100., 90.]])
        self.assertEqual(crop_bounds(points, 128, 96, 1.15), [14, 4, 106, 96])
        with self.assertRaisesRegex(ValueError, 'inside'):
            crop_bounds(np.array([[20., -1.], [100., 90.]]), 128, 96, 1.15)
        with self.assertRaisesRegex(ValueError, 'entire'):
            crop_bounds(points, 128, 96, .9)

    def test_same_mesh_alias_across_partitions_is_rejected(self):
        self.targets[2]['appearance_group'] = self.targets[0]['appearance_group']
        with self.assertRaisesRegex(ValueError, 'mesh leaks'):
            resolve_splits(self.targets, {'appearance': self.protocol})

    def test_target_overlap_or_missing_target_is_rejected(self):
        for test in ([1], []):
            protocol = dict(self.protocol, test=test)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                resolve_splits(self.targets, {'appearance': protocol})

    def test_instance_control_allows_seen_mesh_but_requires_it_in_training(self):
        protocol = dict(self.protocol, disjoint_primary_meshes=False, evaluation_category='hangar')
        for t in self.targets:
            t['appearance_group'] = 'same mesh'
        resolved = resolve_splits(self.targets, {'instance': protocol})
        self.assertEqual(resolved['instance']['test'], ['3'])
        self.targets[1]['appearance_group'] = 'unseen validation mesh'
        with self.assertRaisesRegex(ValueError, 'training instances'):
            resolve_splits(self.targets, {'instance': protocol})

    def test_held_out_category_cannot_calibrate_thresholds(self):
        self.targets[1]['category'] = 'tower'
        protocol = dict(self.protocol, held_out_category='tower')
        with self.assertRaisesRegex(ValueError, 'fitting data'):
            resolve_splits(self.targets, {'category': protocol})

    def test_category_test_reports_unseen_category_separately(self):
        self.targets[2]['category'] = 'tower'
        protocol = dict(self.protocol, held_out_category='tower')
        split = resolve_splits(self.targets, {'category': protocol})['category']
        self.assertEqual(split['test_by_scope']['unseen_category'], ['3'])
        self.assertEqual(split['category_counts']['train'], {'hangar': 1})

    def test_roof_occlusion_does_not_label_hidden_part_as_visible(self):
        vertices = np.array([[-2, -2, 0], [-2, 2, 0], [2, 2, 0], [2, -2, 0]], np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
        mesh = {'primary': (vertices, faces)}
        options = (self.message, 64., [60, 44, 68, 52], 'nose', {'radius_m': 5},
                   np.zeros(3), np.array([1., 0.]), np.zeros(3))
        whole, part = template_masks(mesh, 'primary', *options)
        self.assertEqual(np.count_nonzero(whole), 64)
        self.assertEqual(np.count_nonzero(part), 64)
        mesh['roof'] = (vertices - np.array([0, 0, 2], np.float32), faces)
        whole, part = template_masks(mesh, 'primary', *options)
        self.assertEqual(np.count_nonzero(whole), 64)
        self.assertEqual(np.count_nonzero(part), 0)


if __name__ == '__main__':
    unittest.main()
