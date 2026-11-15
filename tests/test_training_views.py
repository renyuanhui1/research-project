"""Training-view sampling and per-part mask rendering."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from capture.collect_training_views import plan_views
from capture.part_masks import PartMaskRenderer, part_membership


def cube_mesh():
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 2, 6], [0, 6, 4],
                      [1, 5, 7], [1, 7, 3], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]], np.uint32)
    return vertices, faces


class PartMaskTests(unittest.TestCase):
    def test_membership_uses_the_scene_profile_bounds(self):
        part = dict(radius_m=5., longitudinal_min_m=-5., longitudinal_max_m=-.5,
                    lateral_min_m=-5., lateral_max_m=5.)
        points = np.array([[-1., 0., 0.], [1., 0., 0.], [-.4, 0., 0.], [-1., 6., 0.]])
        inside = part_membership(points, part, [0, 0, 0], [1, 0], [0, 0, 0])
        np.testing.assert_array_equal(inside, [True, False, False, False])

    def test_render_camera_matches_the_derived_body_mount(self):
        vertices, faces = cube_mesh()
        mesh = ROOT/'tests'/'_cube.npz'
        try:
            np.savez(mesh, vertices_enu=vertices, triangles=faces)
            part = dict(radius_m=5., longitudinal_min_m=-5., longitudinal_max_m=-.5,
                        lateral_min_m=-5., lateral_max_m=5.)
            offset, mount = np.array([0., 0., .3]), np.eye(3)
            renderer = PartMaskRenderer(mesh, {'tail': part}, [0, 0, 0], [1, 0], [0, 0, 0],
                                        camera_offset=offset, camera_mount=mount,
                                        shape=(60, 80), horizontal_fov_deg=90., stride=4)
            body = [-10., 0., -.3]
            derived = renderer.render(body, mount)
            recorded = renderer.render_camera(np.array(body)+offset, mount)
            for key in derived:
                np.testing.assert_array_equal(derived[key], recorded[key])
        finally:
            mesh.unlink()

    def test_renderer_labels_the_visible_face_and_not_the_background(self):
        vertices, faces = cube_mesh()
        mesh = ROOT/'tests'/'_cube.npz'
        try:
            np.savez(mesh, vertices_enu=vertices, triangles=faces)
            part = dict(radius_m=5., longitudinal_min_m=-5., longitudinal_max_m=-.5,
                        lateral_min_m=-5., lateral_max_m=5.)
            renderer = PartMaskRenderer(mesh, {'tail': part}, [0, 0, 0], [1, 0], [0, 0, 0],
                                        camera_offset=[0, 0, 0], camera_mount=np.eye(3),
                                        shape=(120, 160), horizontal_fov_deg=90., stride=4)
            masks = renderer.render([-10, 0, 0], np.eye(3))
            self.assertEqual(masks['tail'].shape, (120, 160))
            self.assertEqual(int(masks['aircraft'][60, 80]), 1)
            self.assertEqual(int(masks['tail'][60, 80]), 1)
            self.assertEqual(int(masks['aircraft'][2, 2]), 0)
            self.assertEqual(int(masks['tail'][2, 2]), 0)
            self.assertGreater(int(masks['tail'].sum()), 0)
        finally:
            mesh.unlink()

    def test_renderer_selects_only_the_declared_part(self):
        vertices, faces = cube_mesh()
        mesh = ROOT/'tests'/'_cube.npz'
        try:
            np.savez(mesh, vertices_enu=vertices, triangles=faces)
            rear = dict(radius_m=5., longitudinal_min_m=.5, longitudinal_max_m=5.,
                        lateral_min_m=-5., lateral_max_m=5.)
            renderer = PartMaskRenderer(mesh, {'rear': rear}, [0, 0, 0], [1, 0], [0, 0, 0],
                                        camera_offset=[0, 0, 0], camera_mount=np.eye(3),
                                        shape=(60, 80), horizontal_fov_deg=90., stride=4)
            masks = renderer.render([-10, 0, 0], np.eye(3))
            self.assertEqual(int(masks['aircraft'][30, 40]), 1)
            self.assertEqual(int(masks['rear'][30, 40]), 0)
        finally:
            mesh.unlink()


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(count=25, seed=7, horizontal_range_m=[12., 60.],
                             bearing_deg=[-40., 40.], heading_offset_deg=[-20., 20.],
                             position_jitter_m=0., camera_pitch_deg=35.)
        self.reference = np.array([555., 129., -5.])

    def test_views_are_deterministic_and_cover_the_declared_bounds(self):
        forward = np.array([0., 1.])
        first = plan_views(self.protocol, self.reference, forward, np.random.default_rng(7))
        second = plan_views(self.protocol, self.reference, forward, np.random.default_rng(7))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 25)
        for case in first:
            self.assertAlmostEqual(np.linalg.norm(np.array(case['position_ned_m'])[:2]-self.reference[:2]),
                                   case['horizontal_range_m'], places=6)
            self.assertTrue(12. <= case['horizontal_range_m'] <= 60.)
            self.assertTrue(-40. <= case['bearing_deg'] <= 40.)
            self.assertTrue(-20. <= case['heading_offset_deg'] <= 20.)
            offset = np.array(case['position_ned_m'])[:2]-self.reference[:2]
            forward_distance = offset@forward
            lateral_distance = offset@np.array([-forward[1], forward[0]])
            self.assertAlmostEqual(math.degrees(math.atan2(lateral_distance, forward_distance)),
                                   case['bearing_deg'], places=6)

    def test_stratified_ranges_cover_the_near_and_far_regimes(self):
        protocol = dict(self.protocol, count=2000, range_strata=[[12., 60., .7], [60., 150., .3]])
        cases = plan_views(protocol, self.reference, np.array([0., 1.]), np.random.default_rng(7))
        ranges = np.array([case['horizontal_range_m'] for case in cases])
        self.assertLessEqual(ranges.max(), 150.)
        self.assertGreater(ranges.max(), 60.)
        near = ((ranges >= 12.) & (ranges <= 60.)).mean()
        self.assertGreater(near, .62)
        self.assertLess(near, .78)

    def test_vertical_offset_keeps_the_fixed_pitch_aimed_at_the_reference(self):
        cases = plan_views(self.protocol, self.reference, np.array([0., 1.]), np.random.default_rng(7))
        for case in cases:
            height = case['position_ned_m'][2]-self.reference[2]
            expected = -case['horizontal_range_m']*math.tan(math.radians(35.))
            self.assertAlmostEqual(height, expected, places=6)


if __name__ == '__main__':
    unittest.main()
