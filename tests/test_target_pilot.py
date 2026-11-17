"""Depth supervision, camera planning and lossless pilot storage checks without UE."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from capture.collect_target_pilot import depth_filter, plan_views, LabelRenderer, write_sample, verify_store
from common.patches import read_rgb


class TargetPilotTests(unittest.TestCase):
    def test_occluders_are_background_and_missing_or_farther_depth_is_unknown(self):
        expected = np.array([[5, 5, 5, 5, np.inf, np.inf]], np.float32)
        measured = np.array([[5, 3, 6, np.nan, 8, np.nan]], np.float32)
        result = depth_filter(expected, np.ones(expected.shape, bool), measured, .15)
        np.testing.assert_array_equal(result['assembly_mask'], [[1, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(result['part_mask'], result['assembly_mask'])
        np.testing.assert_array_equal(result['occluded_mask'], [[0, 1, 0, 0, 0, 0]])
        np.testing.assert_array_equal(result['valid_mask'], [[1, 1, 0, 0, 1, 0]])

    def test_native_half_precision_quantization_does_not_hide_near_occluders(self):
        expected = np.array([[125.02, 125.02]], np.float32)
        result = depth_filter(expected, np.ones((1, 2), bool), np.array([[125., 124.]], np.float16), .001)
        np.testing.assert_array_equal(result['assembly_mask'], [[1, 0]])
        np.testing.assert_array_equal(result['occluded_mask'], [[0, 1]])

    def test_planned_camera_points_at_reference_and_preserves_fixed_pitch(self):
        catalog = {'targets': [{'id':'plane','index':7,'category':'aircraft','reference_ned_m':[10,20,-3],
                               'region':{'forward_ne':[0,1]}}],
                   'primary_protocol':'appearance','splits':{'appearance':{'train':['plane']}}}
        protocol = {'target_indices':[7],'horizontal_distances_m':[12,35,75,125],
                    'bearings_deg':{'aircraft':[-45,0,45]},'camera_pitch_deg':35}
        cases = plan_views(catalog, protocol)
        self.assertEqual(len(cases), 12)
        for case in cases:
            pose = case['pose'];q = pose['rotation']
            rotation = Rotation.from_quat([q[k] for k in ('x','y','z','w')]).as_matrix()
            point = np.array([pose['translation'][k] for k in ('x','y','z')])
            local = (np.array([10,20,-3])-point)@rotation
            np.testing.assert_allclose(local[1:], 0, atol=1e-10)
            self.assertGreater(local[0], 0)
        catalog['splits']['appearance']['train'] = []
        with self.assertRaisesRegex(ValueError, 'training split'):
            plan_views(catalog, protocol)
        protocol['target_indices'] = [999]
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            plan_views(catalog, protocol)

    def test_real_raycast_range_labels_and_hdf5_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            vertices = np.array([[-20,-20,0],[-20,20,0],[20,20,0],[20,-20,0]],np.float32)
            mesh = directory/'plane.npz';np.savez(mesh,vertices_enu=vertices,triangles=np.array([[0,1,2],[0,2,3]],np.uint32))
            target = {'primary_actor':'plane','actors':[{'name':'plane','mesh':str(mesh)}],
                      'reference_ned_m':[0,0,0], 'region':{'name':'nose','geometry':{'radius_m':5},'origin_ned_m':[0,0,0],'forward_ne':[1,0]}}
            message = dict(pos_x=0.,pos_y=0.,pos_z=-10.,rot_w=2**-.5,rot_x=0.,rot_y=-2**-.5,rot_z=0.)
            yy,xx=np.mgrid[0:4,0:4];depth=(10*np.sqrt(1+((xx+.5-2)/2)**2+((yy+.5-2)/2)**2)).astype(np.float32)
            masks,residual=LabelRenderer(target).render(depth,message,90.,.15)
            self.assertEqual(np.count_nonzero(masks['assembly_mask']),16)
            self.assertEqual(np.count_nonzero(masks['part_mask']),4)
            self.assertLess(residual,1e-4)
            depth[0,0]=5.;depth[-1,-1]=np.nan
            masks,_=LabelRenderer(target).render(depth,message,90.,.15)
            self.assertTrue(masks['occluded_mask'][0,0]);self.assertFalse(masks['valid_mask'][-1,-1])
            rgb=np.random.default_rng(3).integers(0,256,(8,8,3),dtype=np.uint8)
            sample={'name':'sample','rgb_shape_hw':[8,8],'mask_pixels':{k:int(np.count_nonzero(v)) for k,v in masks.items()}}
            path=directory/'samples.h5'
            with h5py.File(path,'w') as store:write_sample(store,'sample',rgb,depth,masks,sample)
            verify_store(path,{'samples':[sample]})
            with h5py.File(path,'r') as store:
                np.testing.assert_array_equal(read_rgb(store,'sample'),rgb)
                np.testing.assert_array_equal(store['sample/depth_m'][()],depth.astype(np.float16))
                self.assertEqual(json.loads(store['sample'].attrs['metadata_json'])['name'],'sample')


if __name__ == '__main__':
    unittest.main()
