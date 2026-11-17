import sys
from pathlib import Path
import unittest
import subprocess
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation.eval_dino_backbones import reference_views,roi_patches,match,point_label,repository_source


class DinoBackboneComparisonTests(unittest.TestCase):
    def test_dependency_without_git_does_not_inherit_project_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','--allow-empty','-qm','fixture'],check=True)
            repo=root/'third_party/backbone';repo.mkdir(parents=True);(repo/'model.py').write_text('x=1\n')
            provenance=repository_source(repo)
            self.assertIsNone(provenance['source_commit'])
            self.assertEqual(set(provenance['python_source_sha256']),{'model.py'})

    def test_roi_follows_square_padding_and_rotation(self):
        image=np.zeros((100,200,3),np.uint8)
        views,boxes=reference_views(image,[.5,0,1,.5],448)
        self.assertEqual(views[0].shape,(448,448,3))
        np.testing.assert_allclose(boxes[0],[.5,.25,1,.5])
        np.testing.assert_allclose(boxes[1],[.25,0,.5,.5])
        np.testing.assert_allclose(boxes[2],[0,.5,.5,.75])

    def test_invisible_or_sub_patch_roi_stays_unavailable(self):
        self.assertFalse(roi_patches(None,28).any())
        self.assertFalse(roi_patches([.501,.501,.502,.502],28).any())
        self.assertEqual(int(roi_patches([0,0,1,1],28).sum()),784)

    def test_requested_template_changes_location(self):
        tokens=torch.tensor([[[1.,0],[0,1],[-1,0],[0,-1]]])
        mean=torch.zeros(2);std=torch.ones(2)
        a=torch.tensor([[[1.,0]]]);b=-a
        center_a,peak_a,_,_=match(tokens,a,mean,std,1,.005)
        center_b,peak_b,_,_=match(tokens,b,mean,std,1,.005)
        np.testing.assert_allclose(peak_a,[[.25,.25]])
        np.testing.assert_allclose(peak_b,[[.25,.75]])
        self.assertGreater(center_b[0,1]-center_a[0,1],.9)

    def test_hit_uses_native_mask_and_visible_area(self):
        mask=np.zeros((10,20),bool);mask[3:7,8:12]=True
        inside,error=point_label(mask,[.5,.5]);self.assertTrue(inside);self.assertLess(error,0)
        inside,error=point_label(mask,[.1,.1]);self.assertFalse(inside);self.assertGreater(error,0)
        self.assertEqual(point_label(mask,[1,1]),(False,None))
        self.assertEqual(point_label(np.zeros_like(mask),[.5,.5]),(False,None))


if __name__=='__main__':unittest.main()
