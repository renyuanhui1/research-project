"""Metric scale, no target location dependence and candidate-set acceptance."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import json
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from perception.metric_birdseye import metric_cloud,metric_reference,metric_search
from perception.multicandidate_verification import MultiCandidateVerifier
from evaluation.eval_birdseye_matching import calibrate
from perception.shadow_verification import ShadowVerifier, synchronized_pair


class CompletionTests(unittest.TestCase):
    def test_shadow_pair_skips_newest_misaligned_frame_without_reusing_inputs(self):
        rgb={1_000_000_000:'aligned',1_200_000_000:'too_new'}
        depth={1_020_000_000:'depth'}
        self.assertEqual(synchronized_pair(rgb,depth,-1,-1,.05,.5),('aligned','depth'))
        self.assertIsNone(synchronized_pair(rgb,depth,1_000_000_000,1_020_000_000,.05,.5))
        self.assertIsNone(synchronized_pair(rgb,depth,-1,-1,.05,.1))

    def test_stop_hold_detects_drift_after_initial_settling(self):
        import asyncio
        from unittest.mock import Mock
        from control import run_approach as control
        def kin(x):
            return dict(pose=dict(position=dict(x=x,y=0,z=0),orientation=dict(w=1,x=0,y=0,z=0)),
                        twist=dict(linear=dict(x=0,y=0,z=0),angular=dict(x=0,y=0,z=0)))
        async def command(*args,**kwargs):
            async def done():return True
            return asyncio.create_task(done())
        drone=Mock();drone.move_by_velocity_async=command
        drone.get_ground_truth_kinematics.side_effect=[kin(0)]*6+[kin(.5)]
        records=[]
        with patch.object(control,'drone',drone,create=True):
            with self.assertRaisesRegex(RuntimeError,'stop_hold_displacement_exceeded'):
                asyncio.run(control.hover(hold_s=15,tolerance_m=.35,records=records))
        self.assertEqual(records[-1]['displacement_m'],.5)
        self.assertTrue(all(r['command_result'] for r in records))

    def test_stop_hold_preserves_action_errors(self):
        import asyncio
        from unittest.mock import Mock
        from control import run_approach as control
        async def command(*args,**kwargs):raise RuntimeError('RPC unavailable')
        drone=Mock();drone.move_by_velocity_async=command
        drone.get_ground_truth_kinematics.return_value=dict(pose=dict(orientation=dict(w=1,x=0,y=0,z=0)))
        records=[]
        with patch.object(control,'drone',drone,create=True):
            with self.assertRaisesRegex(RuntimeError,'RPC unavailable'):
                asyncio.run(control.hover(records=records))
        self.assertIn('RPC unavailable',records[0]['error'])

    def test_reference_and_search_have_equal_metric_pixel_scale(self):
        cfg={'reference_long_side':100,'sift':{'maximum_features':100}}
        reference=metric_reference(np.zeros((100,200,3),np.uint8),.1,cfg)
        self.assertAlmostEqual(reference['gsd'],.2)
        self.assertEqual(reference['rgb'].shape[:2],(50,100))
        self.assertAlmostEqual(reference['rgb'].shape[1]*reference['gsd'],20.)

    def test_metric_cloud_ignores_world_translation_and_excludes_redaction(self):
        q=Rotation.from_euler('y',-90,degrees=True).as_quat();camera=dict(zip(['rot_'+k for k in 'xyzw'],q))
        rgb=np.zeros((4,4,3),np.uint8);depth=np.full((4,4),10.);excluded=np.eye(4,dtype=bool)
        cfg=dict(max_ray_depth_m=500,horizontal_radius_m=250)
        a=metric_cloud(rgb,depth,camera,90,cfg,excluded)
        b=metric_cloud(rgb,depth,dict(camera,pos_x=500,pos_y=600,pos_z=-50),90,cfg,excluded)
        np.testing.assert_array_equal(a['es'],b['es']);self.assertEqual(len(a['indices']),12)

    def test_absent_request_must_compete_against_entire_candidate_set(self):
        rows=[dict(available=True,score=.8,correct=True),dict(available=True,score=.9,correct=False)]
        self.assertIsNone(calibrate(rows,.95)['threshold'])

    def test_selection_switches_candidate_with_requested_template(self):
        import torch
        model=object.__new__(MultiCandidateVerifier)
        model.acceptance=dict(center=torch.zeros(44),scale=torch.ones(44),weight=torch.tensor([1.]+[0.]*43),bias=torch.tensor(0.),threshold=.8)
        query={'candidates':[{'point_xy':[.2,.2]},{'point_xy':[.8,.8]}]}
        a=np.zeros((2,44),np.float32);a[:,0]=[3.,-3.]
        with patch.object(model,'candidate_features',side_effect=[a,a[::-1].copy()]):
            first=model.select(query,'A');second=model.select(query,'B')
        self.assertEqual(first['chosen_rank'],1);self.assertEqual(second['chosen_rank'],2)
        self.assertTrue(first['accepted'] and second['accepted'])

    def test_shadow_confirmation_resets_on_gap_and_records_native_inputs(self):
        from unittest.mock import Mock
        import h5py
        observer=object.__new__(ShadowVerifier)
        observer.model=Mock()
        observer.model.predict.return_value=dict(accepted=True,point_xy=[.5,.5],score=.99,candidates=[],chosen_rank=1)
        observer.target_id='requested';observer.spec=dict(max_gap_s=2,confirmation_frames=3)
        observer.streak=0;observer.last_observation_time=None;observer.count=0;observer.errors=[]
        metadata=dict(time_stamp=1_000_000_000,width=4,height=4)
        with TemporaryDirectory() as directory:
            observer.directory=Path(directory)
            with patch('perception.shadow_verification.torch.cuda.synchronize'),patch('perception.shadow_verification.decode',return_value=np.ones((4,4),np.float32)):
                for t in [0.,.5,1.,4.]:observer._observe(np.zeros((4,4,3),np.uint8),metadata,metadata,{},t)
            rows=[json.loads(line) for line in (observer.directory/'shadow-records.jsonl').read_text().splitlines()]
            self.assertEqual([r['confirmed'] for r in rows],[False,False,True,False])
            self.assertEqual(observer.errors,[])
            with h5py.File(observer.directory/'shadow.h5') as store:
                self.assertEqual(len(store),4);self.assertEqual(store['000000']['depth_m'].shape,(4,4))

    def test_shadow_inference_error_is_recorded_and_breaks_confirmation(self):
        from unittest.mock import Mock
        observer=object.__new__(ShadowVerifier);observer.model=Mock()
        observer.model.predict.side_effect=RuntimeError('inference failed')
        observer.target_id='requested';observer.errors=[];observer.streak=2
        with patch('perception.shadow_verification.torch.cuda.synchronize'):
            observer._observe(np.zeros((2,2,3),np.uint8),{}, {},{},0)
        self.assertEqual(observer.streak,0);self.assertIn('inference failed',observer.errors[0])


if __name__=='__main__':unittest.main()
