"""洞口控制的帧独立性、坐标方向、失锁停止、收敛和保护检查。"""
import copy
import math
from pathlib import Path
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from control.run_opening_alignment import (
    Alignment, FramePairs, Limits, aperture_in_camera, horizontal_error, identity_gate, safety_check, validate_cameras)
from perception.rgbd_entry import CV_TO_SIM
from control.run_opening_alignment import rotation


def camera(w=1920, h=1080):
    return dict(width=w, height=h, encoding='16FC1' if w == 640 else 'BGR',
                rot_w=math.sqrt(.5), rot_x=0., rot_y=-math.sqrt(.5), rot_z=0.,
                pos_x=0., pos_y=0., pos_z=-65.)


def info(w, h):
    return dict(width=w, height=h, distortion_params=[0]*5,
                intrinsic_camera_matrix=[w/2, 0, w/2, 0, w/2, h/2, 0, 0, 1])


def match(n=-2., e=8.):
    return {'near_layer_depth_m': 40., 'proposed_aperture_point_rgb_xy':
            [959.5+e*960/40, 539.5-n*960/40],
            'selected': {'spatial_dino_score': .65, 'projected_aperture_radius_m': 6.,
                         'bbox_depth_xyxy': [200, 100, 440, 280]},
            'top_two_score_margin': .2}


class FramePairTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.limits = FramePairs(), Limits()

    def feed(self, tag, stamp):
        self.frames.callback(tag)('', {'time_stamp': stamp})

    def test_both_sources_must_be_independent(self):
        self.feed('rgb', 1_000_000_000)
        self.feed('depth', 1_050_000_000)
        self.assertIsNotNone(self.frames.pair(1_100_000_000, self.limits))
        self.feed('rgb', 1_100_000_000)
        self.assertIsNone(self.frames.pair(1_200_000_000, self.limits))
        self.feed('depth', 1_150_000_000)
        self.assertIsNotNone(self.frames.pair(1_200_000_000, self.limits))

    def test_skew_backlog_future_and_pre_motion_frames_rejected(self):
        for rt, dt, now, after in [(1e9, 1.2e9, 1.3e9, -1),
                                   (1e9, 1e9, 2e9, -1),
                                   (2e9, 2e9, 1e9, -1),
                                   (1e9, 1e9, 1.1e9, 1e9)]:
            f = FramePairs()
            f.callback('rgb')('', {'time_stamp': int(rt)})
            f.callback('depth')('', {'time_stamp': int(dt)})
            self.assertIsNone(f.pair(int(now), self.limits, int(after)))

    def test_old_local_arrival_rejected_even_with_fresh_sim_time(self):
        for tag in ('rgb', 'depth'):
            self.frames.frames[tag].append((time.monotonic()-2, {'time_stamp': 1_000_000_000}))
        self.assertIsNone(self.frames.pair(1_100_000_000, self.limits))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.limits = Limits()
        self.controller = Alignment(self.limits, .4)

    def test_nadir_direction_and_camera_yaw(self):
        np.testing.assert_allclose(horizontal_error(match(n=-2, e=8), camera()), [-2, 8], atol=1e-10)
        msg = camera()
        # Rz(90deg) * Ry(-90deg): image-right points south.
        msg.update(rot_w=.5, rot_x=.5, rot_y=-.5, rot_z=.5)
        np.testing.assert_allclose(horizontal_error(match(n=0, e=8), msg), [-8, 0], atol=1e-10)

    def test_acquisition_and_speed_bound(self):
        for i in range(3):
            state, velocity, _ = self.controller.update(match(), camera(), 0, i)
            self.assertLessEqual(np.linalg.norm(velocity), self.limits.max_speed_mps+1e-12)
            self.assertEqual(state, 'ACQUIRE' if i < 2 else 'ALIGN')
        self.assertLess(velocity[0], 0)
        self.assertGreater(velocity[1], 0)

    def test_weak_missing_ambiguous_and_nonfinite_matches_stop(self):
        variants = [dict(match(), selected=None), dict(match(), proposed_aperture_point_rgb_xy=None),
                    dict(match(), top_two_score_margin=.05)]
        weak = match(); weak['selected']['spatial_dino_score'] = .39
        bad = match(); bad['selected']['spatial_dino_score'] = float('nan')
        variants.extend([weak, bad])
        for item in variants:
            with self.subTest(item=item), self.assertRaises(RuntimeError):
                Alignment(self.limits, .4).update(item, camera(), 0, 0)

    def test_candidate_and_aperture_switch_stop(self):
        for item in [match(n=8), match()]:
            c = Alignment(self.limits, .4)
            c.update(match(), camera(), 0, 0)
            if item['proposed_aperture_point_rgb_xy'] == match()['proposed_aperture_point_rgb_xy']:
                item['selected']['bbox_depth_xyxy'] = [0, 0, 50, 50]
            with self.assertRaisesRegex(RuntimeError, 'jump'):
                c.update(item, camera(), 0, 1)

    def test_near_roof_and_small_gap_stop(self):
        a, b = match(), match()
        a['near_layer_depth_m'] = 10
        b['selected']['projected_aperture_radius_m'] = .2
        for item in (a, b):
            with self.assertRaises(RuntimeError):
                Alignment(self.limits, .4).update(item, camera(), 0, 0)

    def test_settling_and_independent_frame_hold_required(self):
        centered = match(n=0, e=0)
        for i in range(3):
            state, velocity, _ = self.controller.update(centered, camera(), .3, i)
        self.assertEqual(state, 'SETTLE')
        np.testing.assert_array_equal(velocity, [0, 0])
        for i in range(4):
            state, _, _ = self.controller.update(centered, camera(), 0, 3+i)
            self.assertEqual(state, 'CONFIRM')
        self.assertEqual(self.controller.update(centered, camera(), 0, 7)[0], 'ALIGNED')

    def test_ideal_horizontal_plant_converges_without_vertical_command(self):
        error = np.array([-3., 8.])
        for i in range(300):
            state, velocity, _ = self.controller.update(match(*error), camera(), 0, i*.5)
            self.assertEqual(velocity.shape, (2,))
            error -= velocity*self.limits.pulse_s
            if state == 'ALIGNED':
                break
        self.assertEqual(state, 'ALIGNED')
        self.assertLessEqual(np.linalg.norm(error), self.limits.tolerance_m)


class CameraAndSafetyTests(unittest.TestCase):
    def test_oblique_handoff_requires_valid_identity_before_context_leaves_view(self):
        m=match();m['selected']['crop_rgb_xyxy']=[300,70,1500,1000]
        checked,event=identity_gate(m,camera(),31.,True,True,.4)
        self.assertTrue(checked);self.assertEqual(event['reason'],'opening_context_near_image_boundary')
        weak=copy.deepcopy(m);weak['selected']['spatial_dino_score']=.39;weak['proposed_aperture_point_rgb_xy']=None
        with self.assertRaisesRegex(RuntimeError,'identity_lost'):identity_gate(weak,camera(),31.,True,True,.4)
        # 已交接后弱匹配不会重新激活DINO；相同边界条件下垂直模式仍按距离交接。
        self.assertEqual(identity_gate(weak,camera(),30.,False,True,.4),(False,None))
        self.assertEqual(identity_gate(m,camera(),31.,True,False,.4),(True,None))
        self.assertEqual(identity_gate(weak,camera(),24.,True,False,.4)[1]['reason'],'roof_remaining_25m')

    def test_oblique_ray_intersection_and_alignment_keep_standoff(self):
        msg=camera();msg.update(rot_w=math.cos(math.pi/6),rot_y=-math.sin(math.pi/6))
        depth=copy.deepcopy(msg);depth.update(width=640,height=360,encoding='16FC1')
        validate_cameras(msg,depth,info(1920,1080),info(640,360),oblique=True)
        with self.assertRaises(ValueError):validate_cameras(msg,depth,info(1920,1080),info(640,360))
        m=match(n=0,e=0);world=rotation(msg)@CV_TO_SIM
        point=world@aperture_in_camera(m,msg,world.T@np.array([0.,0.,1.]))
        np.testing.assert_allclose(point,[40/math.sqrt(3),0,40],atol=1e-10)
        np.testing.assert_allclose(horizontal_error(m,msg,True),[0,0],atol=1e-10)
        m['proposed_aperture_point_rgb_xy'][0]+=96
        self.assertGreater(horizontal_error(m,msg,True)[1],0)

    def test_live_intrinsics_pose_and_encoding_checked(self):
        args = [camera(), camera(640, 360), info(1920, 1080), info(640, 360)]
        validate_cameras(*args)
        cases = []
        bad = copy.deepcopy(args); bad[3]['intrinsic_camera_matrix'][0] = 400; cases.append(bad)
        bad = copy.deepcopy(args); bad[1]['pos_x'] = 1; cases.append(bad)
        bad = copy.deepcopy(args); bad[0].update(rot_w=1., rot_y=0.); cases.append(bad)
        bad = copy.deepcopy(args); bad[1]['encoding'] = '16UC1'; cases.append(bad)
        bad = copy.deepcopy(args); bad[2]['distortion_params'][0] = .1; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                validate_cameras(*bad)

    def test_altitude_drift_and_horizontal_envelope(self):
        origin = np.array([0., 0., -65.])
        for p in ([0, 0, -40], [0, 0, -110], [0, 0, -63], [16, 0, -65], [float('nan'), 0, -65]):
            with self.assertRaises(RuntimeError):
                safety_check({'pose': {'position': dict(zip(('x', 'y', 'z'), p))}}, origin, Limits())


if __name__ == '__main__':
    unittest.main()
