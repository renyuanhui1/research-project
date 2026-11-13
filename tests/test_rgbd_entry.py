"""入室通道拒绝未知/横梁、视觉位姿方向和下降状态转换。"""
import sys
from pathlib import Path
import unittest
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from perception.rgbd_entry import CV_TO_SIM, FreeTube, VisualAnchor, entry_velocity


def depth_plane(distance,shape=(360,640)):
    h,w=shape;y,x=np.mgrid[:h,:w]
    return (distance*np.sqrt(1+((x+.5-w/2)/(w/2))**2+((y+.5-h/2)/(w/2))**2)).astype(np.float32)


def oblique_scene(angle=70, beam=False):
    """水平屋面下30m，洞内地面下70m；洞中心沿斜向视轴，半宽7m。"""
    a=np.deg2rad(angle);world=np.array([[0.,-np.sin(a),np.cos(a)],[1.,0.,0.],[0.,np.cos(a),np.sin(a)]])
    h,w=360,640;y,x=np.mgrid[:h,:w]
    rays=np.stack(((x+.5-w/2)/(w/2),(y+.5-h/2)/(w/2),np.ones((h,w))),axis=-1)
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    directions=rays@world.T;roof=directions*(30/directions[:,:,2])[:,:,None]
    hole=(abs(roof[:,:,0]-30/np.tan(a))<7)&(abs(roof[:,:,1])<7)
    if beam:hole &= ~((roof[:,:,1]>.7)&(roof[:,:,1]<1.4))
    return np.where(hole,70/directions[:,:,2],30/directions[:,:,2]).astype(np.float32)


class TubeTests(unittest.TestCase):
    def oblique_tube(self,angle=60):
        a=np.deg2rad(angle);world=np.array([[0.,-np.sin(a),np.cos(a)],[1.,0.,0.],[0.,np.cos(a),np.sin(a)]])
        down=world.T@np.array([0.,0.,1.])
        aperture=world.T@np.array([30/np.tan(a),0.,30.])
        return FreeTube(aperture,down,world,oblique=True),world

    def test_slanted_velocity_and_entire_body_sweep(self):
        tube,world=self.oblique_tube()
        state,v,_,_=entry_velocity(np.zeros(3),tube,world)
        self.assertGreater(v[0],0);self.assertGreater(v[2],0)
        self.assertAlmostEqual(v[0]/v[2],1/np.sqrt(3),places=6)
        self.assertLessEqual(np.linalg.norm(v[:2]),.250001)
        # 三维包络包含相对中心线上下1.25m机体引入的附加水平偏移。
        self.assertGreaterEqual(tube.horizontal_half_widths[0],1.25*(1+1/np.sqrt(3))-.0001)
        center=tube.axis_slope*(tube.goal_progress-.2)
        self.assertEqual(entry_velocity(center,tube,world)[0],'INTERIOR_CONFIRM')
        shifted=center+world.T@np.array([0.,1.,0.])
        self.assertEqual(entry_velocity(shifted,tube,world)[1][2],0.)

    def test_slanted_closed_loop_follows_path_without_world_endpoint(self):
        tube,world=self.oblique_tube();center=np.zeros(3)
        horizontal_travel=0.
        for _ in range(1500):
            state,velocity,error,progress=entry_velocity(center,tube,world)
            self.assertLess(np.linalg.norm(error),.36)
            if state=='INTERIOR_CONFIRM':break
            center+=world.T@velocity*.3
            horizontal_travel+=np.linalg.norm(velocity[:2])*.3
        self.assertEqual(state,'INTERIOR_CONFIRM')
        self.assertGreater(horizontal_travel,20.)
        self.assertGreater(progress,37.5)

    def test_oblique_obstacle_is_rejected_and_unknown_is_not_free(self):
        tube,_=self.oblique_tube()
        with self.assertRaisesRegex(RuntimeError,'obstacle'):
            tube.update(np.ones((360,640),np.float32)*10,np.eye(3),np.zeros(3),0,np.zeros(3))
        with self.assertRaisesRegex(RuntimeError,'invalid'):
            tube.update(np.zeros((360,640),np.float32),np.eye(3),np.zeros(3),0,np.zeros(3))

    def test_visible_slanted_body_volume_and_off_axis_beam(self):
        tube,_=self.oblique_tube(70)
        certificate=tube.update(oblique_scene(),np.eye(3),np.zeros(3),0,np.zeros(3))
        self.assertEqual(certificate['refreshed_samples'],certificate['future_samples'])
        # 中心视线仍穿过洞口，但机体侧面会撞到横梁，必须拒绝。
        with self.assertRaisesRegex(RuntimeError,'obstacle'):
            tube.update(oblique_scene(beam=True),np.eye(3),np.zeros(3),1,np.zeros(3))
        shallow,_=self.oblique_tube(60)
        with self.assertRaisesRegex(RuntimeError,'unknown'):
            shallow.update(oblique_scene(60),np.eye(3),np.zeros(3),0,np.zeros(3))

    def tube(self):
        # 光学z向下、x向东、y向南。
        world=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        return FreeTube(np.array([0.,0.,30.]),np.array([0.,0.,1.]),world)

    def test_observed_free_volume_and_goal(self):
        tube=self.tube()
        r=tube.update(depth_plane(60),np.eye(3),np.zeros(3),0,np.zeros(3))
        self.assertEqual(r['future_samples'],r['samples'])
        self.assertEqual(r['max_future_observation_age_s'],0)
        self.assertEqual(tube.goal_progress,38)

    def test_beam_and_invalid_depth_reject_the_body_volume(self):
        for value in (25.,0.,float('nan')):
            d=depth_plane(60)
            d[170:190,310:330]=value
            with self.subTest(value=value),self.assertRaises(RuntimeError):
                self.tube().update(d,np.eye(3),np.zeros(3),0,np.zeros(3))

    def test_near_body_observation_cache_expires(self):
        t=self.tube();t.update(depth_plane(60),np.eye(3),np.zeros(3),0,np.zeros(3))
        center=np.array([0.,0.,30.])
        with self.assertRaisesRegex(RuntimeError,'stale'):
            t.update(depth_plane(30),np.eye(3),-center,46,center)

    def test_entry_velocity_keeps_horizontal_loop_and_slows_at_roof(self):
        t=self.tube();world=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        state,v,e,p=entry_velocity(np.array([1.,0.,0.]),t,world)
        self.assertEqual(v[2],0)
        self.assertLess(v[1],0)
        self.assertAlmostEqual(np.linalg.norm(v[:2]),.25)
        self.assertEqual(entry_velocity(np.zeros(3),t,world)[1][2],.8)
        self.assertEqual(entry_velocity(np.array([0.,0.,26.]),t,world)[1][2],.35)
        state,v,_,_=entry_velocity(np.array([0.,0.,37.8]),t,world)
        self.assertEqual(state,'INTERIOR_CONFIRM')
        np.testing.assert_array_equal(v,[0,0,0])


class VisualTrackingTests(unittest.TestCase):
    def test_oblique_rgbd_tracks_forward_and_down_with_keyframes(self):
        rng=np.random.default_rng(93);image=np.zeros((540,960,3),np.uint8)
        for _ in range(1000):
            x,y=rng.integers([15,15],[945,525]);c=int(rng.integers(60,255))
            cv2.rectangle(image,(x-3,y-3),(x+3,y+3),(c,c,c),-1)
        angle=np.deg2rad(70);down=np.array([0.,np.cos(angle),np.sin(angle)])
        y,x=np.mgrid[:360,:640];ray=np.stack(((x+.5-320)/320,(y+.5-180)/320,np.ones((360,640))),axis=-1)
        ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
        def ranges(height):return (height/(ray@down)).astype(np.float32)
        tracker=VisualAnchor(image,ranges(70.));k=tracker.matrix
        for step in range(1,26):
            center=np.array([0.,0.,step*.6])
            homography=k@(np.eye(3)-np.outer(center,down)/70.)@np.linalg.inv(k)
            frame=cv2.warpPerspective(image,homography,(960,540))
            estimated=tracker.update(frame,ranges(70.-center@down))
            np.testing.assert_allclose(estimated,center,atol=.15)
        self.assertGreater(tracker.keyframes,1)

    def test_textureless_input_cannot_start_entry(self):
        with self.assertRaisesRegex(RuntimeError,'features'):
            VisualAnchor(np.zeros((540,960,3),np.uint8),depth_plane(60))

    def test_rgbd_pose_tracks_translation_and_descent(self):
        rng=np.random.default_rng(42)
        image=np.zeros((540,960,3),np.uint8)
        for _ in range(700):
            x,y=rng.integers([15,15],[945,525]);c=int(rng.integers(60,255))
            cv2.rectangle(image,(x-3,y-3),(x+3,y+3),(c,c,c),-1)
        tracker=VisualAnchor(image,depth_plane(60))
        for step in range(1,4):
            expected=np.array([.25*step,-.15*step,1.*step])
            scale=60/(60-expected[2]);cx,cy=479.5,269.5
            warp=np.array([[scale,0,cx*(1-scale)-480*expected[0]/(60-expected[2])],
                           [0,scale,cy*(1-scale)-480*expected[1]/(60-expected[2])]],float)
            frame=cv2.warpAffine(image,warp,(960,540))
            estimated=tracker.update(frame,depth_plane(60-expected[2]))
            np.testing.assert_allclose(estimated,expected,atol=.12)
        self.assertGreater(tracker.quality['inliers'],25)
        with self.assertRaisesRegex(RuntimeError,'RGBD_tracks'):
            tracker.update(frame,np.zeros((360,640),np.float32))

    def test_anchor_survives_large_descent_and_view_expansion(self):
        rng=np.random.default_rng(7)
        image=np.zeros((540,960,3),np.uint8)
        for _ in range(1200):
            x,y=rng.integers([300,160],[660,380]);c=int(rng.integers(60,255))
            cv2.rectangle(image,(x-2,y-2),(x+2,y+2),(c,c,c),-1)
        tracker=VisualAnchor(image,depth_plane(60))
        for descent in range(1,51):
            scale=60/(60-descent)
            warp=np.array([[scale,0,479.5*(1-scale)],[0,scale,269.5*(1-scale)]],float)
            frame=cv2.warpAffine(image,warp,(960,540))
            estimated=tracker.update(frame,depth_plane(60-descent))
            np.testing.assert_allclose(estimated,[0,0,descent],atol=.15)


if __name__=='__main__':unittest.main()
