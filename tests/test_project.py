"""Offline checks for scene assets, recorded runs, decoding and control feedback."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from common.profiles import load_profile, scene_bundle, load_mission
from common.sensors import decode, decode_image
from control.servo_closed_loop import depth_speed_scale, sample_depth_array, ema_update, compute_body_cmd, parse_args
from evaluation.eval_run import load_run, quat_to_R
from perception.depth_map import decode_depth

class ProjectTests(unittest.TestCase):
    def test_profile_assets_and_scene(self):
        profile=load_profile('airport')
        for key in ('template','stats_episode','preparation_assets','dino_weights','dino_repo','target_metadata'):
            self.assertTrue(project_path(profile['assets'][key]).exists(),key)
        scene=scene_bundle(profile)
        self.assertEqual(scene['actors'][0]['name'],profile['simulator']['drone'])
        sensors={s['id']:s for s in scene['actors'][0]['robot-config']['sensors']}
        self.assertEqual(len(sensors),len(scene['actors'][0]['robot-config']['sensors']))
        for key in ('FrontCamera','Chase'):
            rgb=next(c for c in sensors[key]['capture-settings'] if c['image-type']==0)
            self.assertEqual((rgb['width'],rgb['height']),(1920,1080))
        depth=next(c for c in sensors['FrontCamera']['capture-settings'] if c['image-type']==2)
        self.assertEqual((depth['width'],depth['height']),(640,360))
        self.assertEqual(load_mission(profile)['speed'],3)
        self.assertEqual(project_path('.'),ROOT)

    def test_custom_profile_keeps_explicit_paths(self):
        profile=load_profile()
        profile['simulator']['address']='127.0.0.1'
        profile['map']['root']='data/maps/another_scene'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'custom.json';path.write_text(json.dumps(profile))
            loaded=load_profile(path)
            self.assertEqual(loaded['simulator']['address'],'127.0.0.1')
            self.assertEqual(project_path(loaded['map']['root']),ROOT/'data/maps/another_scene')

    def test_all_camera_encodings_and_containers(self):
        specs={'RGB':(np.uint8,3),'BGR':(np.uint8,3),'RGBA':(np.uint8,4),'BGRA':(np.uint8,4),
               '8UC1':(np.uint8,1),'16UC1':(np.uint16,1),'16FC1':(np.float16,1),'32FC1':(np.float32,1)}
        for enc,(dtype,ch) in specs.items():
            raw=np.arange(2*3*ch,dtype=dtype).reshape(2,3,ch)
            for data in (raw.tobytes(),bytearray(raw.tobytes()),memoryview(raw.tobytes()),raw.flatten().tolist()):
                with self.subTest(encoding=enc,container=type(data).__name__):
                    msg={'height':2,'width':3,'encoding':enc,'data':data}
                    np.testing.assert_array_equal(decode(msg),raw)
                    if ch in (3,4):
                        rgb,_=decode_image(msg)
                        expected=raw[:,:,:3][:,:,::-1] if enc.startswith('BGR') else raw[:,:,:3]
                        np.testing.assert_array_equal(rgb,expected)
        with self.assertRaises(ValueError):decode({'height':2,'width':3,'encoding':'16FC1','data':[1]})

    def test_fp16_meter_depth_and_invalid_values(self):
        depth=np.array([[4.,np.inf],[0.,3.5]],dtype=np.float16)
        msg={'height':2,'width':2,'encoding':'16FC1','data':depth.tobytes()}
        values,valid=decode_depth(msg)
        np.testing.assert_array_equal(valid,[[True,False],[False,True]])
        np.testing.assert_array_equal(values,[[4,0],[0,3.5]])
        with tempfile.TemporaryDirectory() as tmp:
            from capture.collect_episode import save_hdf5
            args=SimpleNamespace(dt=.1,steps=2,camera='FrontCamera',scene='scene_airport.jsonc')
            bufs={'rgb':np.zeros((2,2,2,3),np.uint8),'depth':np.repeat(depth[None],2,axis=0),
                  'action':np.zeros((2,4)),'pose':np.zeros((2,7)),'time':np.array([0,100000000],np.int64)}
            file=Path(tmp)/'episode.h5';save_hdf5(file,args,bufs)
            with h5py.File(file) as h:
                self.assertEqual(h['depth'].dtype,np.float16)
                self.assertEqual(h.attrs['depth_unit'],'meter')

    def test_control_feedback_and_distance_scale(self):
        args=parse_args(['--nadir','--v-forward','3','--k-xy','1','--v-down','.5'])
        centered=compute_body_cmd(args,0,0,0)
        offset=compute_body_cmd(args,.4,-.2,0)
        self.assertGreater(offset[1],centered[1])
        self.assertGreater(offset[0],centered[0])
        self.assertEqual(depth_speed_scale(45,35,4,.15),1)
        self.assertEqual(depth_speed_scale(4,35,4,.15),.15)
        self.assertAlmostEqual(ema_update(.1,.5,.3),.22)
        self.assertAlmostEqual(sample_depth_array(np.full((5,5),4.),0,0),4)
        self.assertIsNone(sample_depth_array(np.full((5,5),np.inf),0,0))

    def test_airport_run_pose_and_metadata(self):
        run=ROOT/'runs/approach/close4-1793954736186'
        _,pos,yaw,t,actions,_=load_run(run)
        self.assertEqual(pos.shape,(184,3))
        self.assertEqual(len(yaw),184)
        self.assertTrue(np.isfinite(yaw).all())
        self.assertEqual(actions.shape,(184,3))
        self.assertTrue((np.diff(t)>0).all())
        for name in load_profile()['map']['actors']:
            with h5py.File(ROOT/'data/metadata/targets'/(name+'.h5')) as h:
                self.assertEqual(h.attrs['target_name'],name)
                self.assertTrue((h.attrs['target_size']>0).all())
                rot=quat_to_R(h.attrs['target_quat'])
                np.testing.assert_allclose(rot.T@rot,np.eye(3),atol=1e-6)

    def test_occupancy_images_and_axes(self):
        import cv2,yaml
        folder=ROOT/'data/maps/airport_scene/grids'
        summary=json.loads((folder/'summary.json').read_text())
        self.assertEqual({x['name'] for x in summary['targets']},set(load_profile()['map']['actors']))
        for info in summary['maps']:
            name=info['name']
            with np.load(folder/(name+'.npz')) as grid:
                cells=grid['occupancy'];origin=grid['origin_enu_m'];res=float(grid['resolution_m'])
            self.assertTrue(np.isin(cells,[-1,0,100]).all())
            config=yaml.safe_load((folder/(name+'.yaml')).read_text())
            self.assertEqual(config['resolution'],res)
            np.testing.assert_allclose(config['origin'][:2],origin[:2])
            pixels=np.full(cells.shape,205,np.uint8);pixels[cells==0]=254;pixels[cells==100]=0
            np.testing.assert_array_equal(cv2.imread(str(folder/config['image']),0),pixels[::-1])

if __name__=='__main__':unittest.main()
