"""Depth-verified context identities and genuinely ambiguous mesh copies."""
from pathlib import Path
import sys
import unittest
import numpy as np
import open3d as o3d

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capture.label_target_context import ContextRenderer,unambiguous_context
from capture.collect_target_pilot import depth_filter


class TargetContextTests(unittest.TestCase):
    def test_nearest_surface_occlusion_unknown_depth_and_ignored_regions(self):
        renderer=ContextRenderer.__new__(ContextRenderer);renderer.scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        for distance in [4.,8.]:
            vertices=np.array([[distance,-4,-4],[distance,4,-4],[distance,4,4],[distance,-4,4]],np.float32)
            faces=np.array([[0,1,2],[0,2,3]],np.uint32)
            renderer.scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
        renderer.lookup=np.array([1,2],np.uint16)
        message=dict(pos_x=0,pos_y=0,pos_z=0,rot_x=0,rot_y=0,rot_z=0,rot_w=1)
        valid=np.ones((1,1),bool);ignored=np.zeros((1,1),bool)
        ids,known=renderer.render(np.array([[4.]],np.float32),message,90,.15,valid,ignored)
        self.assertEqual(int(ids[0,0]),1);self.assertTrue(known[0,0])
        ids,known=renderer.render(np.array([[2.]],np.float32),message,90,.15,valid,ignored)
        self.assertEqual(int(ids[0,0]),0);self.assertTrue(known[0,0])
        ids,known=renderer.render(np.array([[10.]],np.float32),message,90,.15,valid,ignored)
        self.assertEqual(int(ids[0,0]),0);self.assertFalse(known[0,0])
        ids,known=renderer.render(np.array([[4.]],np.float32),message,90,.15,valid,~ignored)
        self.assertEqual(int(ids[0,0]),0);self.assertFalse(known[0,0])

    def test_visible_identical_copies_do_not_create_conflicting_extra_pairs(self):
        targets={1:{'appearance_group':'main'},2:{'appearance_group':'copy'},3:{'appearance_group':'copy'},4:{'appearance_group':'unique'}}
        self.assertEqual(unambiguous_context(np.array([[1,2,3,4]]),targets[1],targets),[4])
        self.assertEqual(unambiguous_context(np.array([[1,2,0,0]]),targets[1],targets),[2])

    def test_half_precision_maximum_does_not_create_infinite_depth_tolerance(self):
        expected=np.array([[50.,50.]],np.float32);measured=np.array([[65504.,100000.]],np.float32)
        with np.errstate(all='raise'):
            masks=depth_filter(expected,np.ones((1,2),bool),measured,.15)
        self.assertFalse(masks['assembly_mask'].any());self.assertFalse(masks['valid_mask'].any())


if __name__=='__main__':unittest.main()
