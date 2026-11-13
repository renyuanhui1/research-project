"""Shared-start admission and image-coordinate checks without UE or model weights."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from common.start_state import start_state_check
from perception.part_matching import CroppedPartMatcher, crop_to_full, response_crop, roi_overlap_weights, refine_local_image
from perception.fingerprint import Fingerprint
from control.run_approach import common_preflight_passed,refined_depth_half_width
from control.servo_closed_loop import sample_depth_array


class PartMatchingTests(unittest.TestCase):
    def test_overlap_preserves_thin_ROI_between_cell_centers(self):
        weights,box=roi_overlap_weights([.49,.1,.51,.9],2,padding=0)
        np.testing.assert_allclose(weights,np.full((2,2),.25))
        self.assertAlmostEqual(float(weights.sum()),1.)
        weights,box=roi_overlap_weights([0,0,.02,.1],32)
        self.assertEqual(box[:2].tolist(),[0,0])
        self.assertTrue(np.isfinite(weights).all());self.assertGreater(np.count_nonzero(weights),0)
        with self.assertRaises(ValueError):roi_overlap_weights([.5,0,.5,1],32)

    def test_local_registration_transfers_anchor_and_rejects_missing_texture(self):
        rng=np.random.default_rng(21);template=rng.integers(0,256,(40,24,3),dtype=np.uint8)
        rgb=rng.integers(20,55,(220,300,3),dtype=np.uint8);rgb[80:120,124:148]=template
        result=refine_local_image(rgb,template,[12,20],[131,102],[0,0,300,220])
        self.assertTrue(result['accepted']);np.testing.assert_allclose(result['point_px'],[136,100],atol=.5)
        rgb[80:120,124:148]=30
        self.assertFalse(refine_local_image(rgb,template,[12,20],[131,102],[0,0,300,220])['accepted'])
        self.assertFalse(refine_local_image(rgb,np.zeros_like(template),[12,20],[131,102],[0,0,300,220])['accepted'])

    def test_local_registration_tracks_scale_and_rotation(self):
        rng=np.random.default_rng(31);template=rng.integers(0,256,(40,24,3),dtype=np.uint8)
        rgb=rng.integers(20,55,(220,300,3),dtype=np.uint8);rgb[80:140,124:160]=cv2.resize(template,(36,60))
        result=refine_local_image(rgb,template,[12,20],[140,111],[0,0,300,220],{'scale':1.5,'angle_deg':0})
        self.assertTrue(result['accepted']);np.testing.assert_allclose(result['point_px'],[142,110],atol=.5)
        transform=cv2.getRotationMatrix2D((12,20),5,1.);rgb=rng.integers(20,55,(220,300,3),dtype=np.uint8);rgb[80:120,124:148]=cv2.warpAffine(template,transform,(24,40),borderMode=cv2.BORDER_REPLICATE)
        result=refine_local_image(rgb,template,[12,20],[131,102],[0,0,300,220])
        self.assertTrue(result['accepted']);self.assertAlmostEqual(result['angle_deg'],5.)

    def test_thin_anchor_depth_does_not_take_background_median(self):
        depth=np.full((360,640),50.,np.float32);depth[120:165,309:311]=12.
        cx=2*930/1920-1;cy=2*430/1080-1
        half=refined_depth_half_width([6.3,59.3],[1920,1080],[640,360])
        self.assertEqual(half,0);self.assertEqual(sample_depth_array(depth,cx,cy,half=half,min_pixels=1),12.)
        self.assertEqual(sample_depth_array(depth,cx,cy,half=4,min_pixels=1),50.)
        self.assertEqual(refined_depth_half_width([63,593],[1920,1080],[640,360]),4)

    def test_local_registration_keeps_previous_instance_when_coarse_response_jumps(self):
        rng=np.random.default_rng(41);template=rng.integers(0,256,(40,24,3),dtype=np.uint8)
        rgb=rng.integers(20,55,(220,300,3),dtype=np.uint8)
        rgb[80:120,124:148]=template;rgb[80:120,200:224]=template
        result=refine_local_image(rgb,template,[12,20],[212,100],[0,0,300,220],{'scale':1.,'angle_deg':0.,'point_px':[136,100]})
        self.assertTrue(result['accepted']);np.testing.assert_allclose(result['point_px'],[136,100],atol=.5)
        self.assertEqual(result['search_source'],'previous_registered_point')

    def test_empty_reference_part_is_invalid_without_blocking_available_part(self):
        args=SimpleNamespace(template='unused.png',image_size=28,target_softmax_temp=.005,target_mass_thresh=.2,target_mass_sharpness=10.)
        def initialize(fp,args,device):
            fp.args=args;fp.dev=device;fp.grid=2;fp.zm=torch.zeros(3);fp.zs=torch.ones(3)
            fp.mean=torch.zeros(1,3,1,1);fp.std=torch.ones(1,3,1,1)
            fp.patch_xy=torch.tensor([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])
            fp.dino=SimpleNamespace(forward_features=lambda x:{'x_norm_patchtokens':torch.ones(1,4,3)})
        with patch.object(Fingerprint,'__init__',initialize), patch.object(CroppedPartMatcher,'_encode',return_value=torch.ones(1,4,3)), patch('perception.part_matching.cv2.imread',return_value=np.zeros((8,8,3),np.uint8)):
            fp=CroppedPartMatcher(args,'cpu',{'nose':[0,0,.5,.5],'tail':[.49,0,.51,1]},'nose')
        self.assertEqual(fp.reference_patch_counts,{'nose':1,'tail':0})
        with patch.object(Fingerprint,'query',return_value=(0.,0.,0.,.8,np.full(4,.8,np.float32))):
            result=fp.query_all(np.full((40,60,3),128,np.uint8))
        self.assertGreater(result['nose'][3],.2)
        self.assertEqual(result['tail'][3],-1.)
        self.assertEqual(fp.last_diagnostics['tail']['failure_reason'],'no_reference_patches')
        checks=[{'frame':frame,'part':name,'inside':True,'peak':result[name][3]} for name in ('nose','tail') for frame in ('reference','start')]
        self.assertTrue(common_preflight_passed(checks,'nose'))
        self.assertFalse(common_preflight_passed(checks,'tail'))

    def test_selected_preflight_still_requires_both_views_and_valid_target(self):
        checks=[{'part':part,'frame':frame,'inside':part!='tail','peak':.6} for part in ('nose','wing','tail') for frame in ('reference','start')]
        self.assertFalse(common_preflight_passed(checks))
        self.assertTrue(common_preflight_passed(checks,'nose'))
        self.assertTrue(common_preflight_passed(checks,'wing'))
        self.assertFalse(common_preflight_passed(checks,'tail'))
        self.assertFalse(common_preflight_passed(checks,'missing'))
        self.assertFalse(common_preflight_passed([v for v in checks if v['frame']=='reference'],'nose'))
        for peak in (.1,float('nan')):
            invalid=copy.deepcopy(checks)
            next(v for v in invalid if v['part']=='nose' and v['frame']=='start')['peak']=peak
            self.assertFalse(common_preflight_passed(invalid,'nose'))

    def test_crop_coordinates_map_to_original_depth_pixels(self):
        shape=(1080,1920,3);box=[600,300,1200,700]
        np.testing.assert_allclose(crop_to_full((0,0),box,shape),[-.0625,-.074074074])
        np.testing.assert_allclose(crop_to_full((-1,-1),box,shape),[2*600/1920-1,2*300/1080-1])
        np.testing.assert_allclose(crop_to_full((1,1),box,shape),[2*1200/1920-1,2*700/1080-1])

    def test_coarse_crop_rejects_missing_and_nonfinite_response(self):
        sim=np.full((32,32),-.1,np.float32)
        self.assertIsNone(response_crop(sim,(1080,1920,3)))
        sim[16,16]=np.nan
        self.assertIsNone(response_crop(sim,(1080,1920,3)))

    def test_coarse_crop_keeps_context_and_clips_at_frame_edge(self):
        sim=np.full((32,32),-.1,np.float32);sim[:4,:8]=.6
        box=response_crop(sim,(1080,1920,3))
        self.assertEqual(box[:2],[0,0])
        self.assertGreater(box[2],480)
        self.assertGreater(box[3],135)
        self.assertLessEqual(box[2],1920);self.assertLessEqual(box[3],1080)

    def test_common_pose_checks_yaw_wrap_speed_and_tilt(self):
        expected={'position_ned_m':[1,2,-32],'yaw_deg':179.9,'position_tolerance_m':.35,'yaw_tolerance_deg':.75,'speed_tolerance_mps':.08,'tilt_tolerance_deg':1}
        yaw=math.radians(-179.9)
        kin={'pose':{'position':{'x':1.,'y':2.,'z':-32.},'orientation':{'w':math.cos(yaw/2),'x':0.,'y':0.,'z':math.sin(yaw/2)}},'twist':{'linear':{'x':0.,'y':0.,'z':0.}}}
        self.assertTrue(start_state_check(kin,expected)['passed'])
        shifted=copy.deepcopy(kin);shifted['pose']['position']['x']+=.5
        self.assertFalse(start_state_check(shifted,expected)['passed'])
        moving=copy.deepcopy(kin);moving['twist']['linear']['x']=.1
        self.assertFalse(start_state_check(moving,expected)['passed'])
        tilted=copy.deepcopy(kin);tilted['pose']['orientation']={'w':math.cos(.1),'x':math.sin(.1),'y':0.,'z':0.}
        expected=dict(expected,yaw_deg=0.)
        self.assertFalse(start_state_check(tilted,expected)['passed'])


if __name__=='__main__':unittest.main()
