import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation.eval_dino_matching import decode,patch_correspondence,correspondence_point


CFG={'temperature':.005,'soft_temperature':.02,'local_radius_cells':1,'maximum_correspondences':1000,
     'reprojection_threshold':.001,'ransac_iterations':2000,'ransac_confidence':.99}


class DinoMatchingTests(unittest.TestCase):
    def test_coordinate_change_isolated_from_probability(self):
        response=torch.tensor([[10.,-10.,-10.,-10.]])
        np.testing.assert_allclose(decode(response,'legacy',CFG),[[0,0]],atol=1e-6)
        np.testing.assert_allclose(decode(response,'cell',CFG),[[.25,.25]],atol=1e-6)
        np.testing.assert_allclose(decode(response,'peak',CFG),[[.25,.25]],atol=1e-6)

    def test_local_centroid_does_not_average_distant_equal_peak(self):
        response=torch.full((1,16),-1.);response[0,0]=1.;response[0,15]=1.
        np.testing.assert_allclose(decode(response,'cell',CFG),[[.5,.5]],atol=1e-5)
        np.testing.assert_allclose(decode(response,'local',CFG),[[.125,.125]],atol=1e-5)

    def test_template_roi_selects_corresponding_query_part_after_reflection(self):
        sim=np.zeros((9,9),np.float32)
        for y in range(3):
            for x in range(3):sim[y*3+2-x,y*3+x]=1.
        matches=patch_correspondence(sim,CFG)
        roi=np.zeros(9,bool);roi[0]=True
        for method in ['mutual','affine']:
            point,score=correspondence_point(matches,roi,method)
            np.testing.assert_allclose(point,[5/6,1/6],atol=1e-6)
            self.assertGreater(score,0)
        without_fit=patch_correspondence(sim,CFG,estimate_affine=False)
        np.testing.assert_allclose(correspondence_point(matches,roi,'mutual')[0],correspondence_point(without_fit,roi,'mutual')[0])

    def test_tied_features_do_not_create_affine_correspondence(self):
        matches=patch_correspondence(np.ones((9,9),np.float32),CFG)
        self.assertEqual(correspondence_point(matches,np.ones(9,bool),'affine'),(None,-1.000001))
        self.assertEqual(correspondence_point(matches,np.zeros(9,bool),'mutual'),(None,-1.000001))


if __name__=='__main__':unittest.main()
