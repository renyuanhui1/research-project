"""The active verifier selects the actual part crop and rejection prevents motion."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from perception.active_verification import ActivePartMatcher
from perception.fingerprint import Fingerprint
from perception.graded_gating import DistanceBinCalibrator, GradedGate
from perception.part_matching import CroppedPartMatcher


class ActiveVerificationTests(unittest.TestCase):
    def observer(self, accepted=True, part_peak=.9):
        observer = object.__new__(ActivePartMatcher)
        observer.model = Mock()
        observer.model.device = SimpleNamespace(type='cpu')
        observer.model.predict.return_value = dict(accepted=accepted, point_xy=[.5, .5], score=.95,
                                                   chosen_rank=1, candidates=[dict(box_xyxy=[8, 6, 36, 24])])
        observer.part_matcher = Mock(selected_part='nose')
        observer.part_matcher.query_all.return_value = {'nose': (0., 0., 0., part_peak, np.ones(4))}
        observer.part_matcher.last_diagnostics = dict(crop_xyxy=[8, 6, 36, 24], whole_peak=.95)
        observer.args = SimpleNamespace(control_min_peak=.2)
        observer.target_id = 'requested'
        observer.grid = 2
        observer.spec = dict(confirmation_frames=3, max_gap_s=2.)
        observer.streak = 0
        observer.staged = False
        observer.near_admission = None
        observer.last_observation_time = None
        observer.count = 0
        return observer

    def test_part_crop_comes_from_model_after_three_accepts(self):
        observer = self.observer()
        rgb = np.zeros((40, 50, 3), np.uint8)
        with patch('perception.active_verification.time.monotonic', side_effect=[0., .5, 1.]):
            first = observer.query(rgb)
            second = observer.query(rgb)
            third = observer.query(rgb)
        self.assertEqual(first[3], -1.)
        self.assertEqual(second[3], -1.)
        self.assertEqual(third[3], .9)
        observer.part_matcher.query_all.assert_called_once_with(rgb, box=[8, 6, 36, 24], whole_peak=.95)
        self.assertTrue(observer.last_prediction['motion_permitted'])

    def test_rejection_stops_part_matching_and_resets_confirmation(self):
        observer = self.observer(accepted=False)
        observer.streak = 3
        result = observer.query(np.zeros((40, 50, 3), np.uint8))
        self.assertEqual(result[3], -1.)
        self.assertEqual(observer.streak, 0)
        self.assertFalse(observer.last_prediction['motion_permitted'])
        observer.part_matcher.query_all.assert_not_called()

    def test_time_gap_breaks_three_frame_confirmation(self):
        observer = self.observer()
        with patch('perception.active_verification.time.monotonic', side_effect=[0., .5, 3.]):
            for _ in range(3):
                observer.query(np.zeros((40, 50, 3), np.uint8))
        self.assertEqual(observer.streak, 1)
        observer.part_matcher.query_all.assert_not_called()

    def test_accepted_identity_still_requires_a_valid_part(self):
        observer = self.observer(part_peak=.1)
        with patch('perception.active_verification.time.monotonic', side_effect=[0., .5, 1.]):
            for _ in range(3):
                observer.query(np.zeros((40, 50, 3), np.uint8))
        self.assertTrue(observer.last_prediction['confirmed'])
        self.assertFalse(observer.last_prediction['motion_permitted'])

    def test_external_crop_bypasses_the_original_coarse_localizer(self):
        matcher = object.__new__(CroppedPartMatcher)
        matcher.args = SimpleNamespace(image_size=28, target_softmax_temp=.005)
        matcher.mean = torch.zeros(1, 3, 1, 1)
        matcher.std = torch.ones(1, 3, 1, 1)
        matcher.dev = 'cpu'
        matcher.zm = torch.zeros(2)
        matcher.zs = torch.ones(2)
        matcher.dino = Mock()
        matcher.dino.forward_features.return_value = {'x_norm_patchtokens': torch.ones(1, 4, 2)}
        matcher.part_protos = {'nose': torch.ones(2)}
        matcher.local_references = {}
        matcher.grid = 2
        matcher._response = Mock(return_value=(0., 0., 0., .9, np.ones(4)))
        rgb = np.zeros((40, 50, 3), np.uint8)
        with patch.object(Fingerprint, 'query', side_effect=AssertionError('Old coarse locator used')):
            result = matcher.query_all(rgb, box=[8, 6, 36, 24], whole_peak=.95)
        self.assertEqual(matcher.last_diagnostics['crop_xyxy'], [8, 6, 36, 24])
        self.assertAlmostEqual(result['nose'][0], -.12)
        with self.assertRaises(ValueError):
            matcher.query_all(rgb, box=[0, 0, 60, 40], whole_peak=.95)

    def test_record_keeps_prediction_inputs_and_command_dispatch(self):
        observer = self.observer(accepted=False)
        rgb = np.zeros((4, 4, 3), np.uint8)
        observer.query(rgb)
        metadata = dict(time_stamp=1, width=4, height=4)
        control_row = dict(step=0, time_s=0., forward=0., right=0., down_speed=0., yaw_rate=0.,
                           state='HOLD', command_dispatched=True)
        with TemporaryDirectory() as directory:
            observer.directory = Path(directory)
            with patch('perception.active_verification.decode', return_value=np.ones((4, 4))):
                observer.record(rgb, metadata, metadata, control_row)
            record = json.loads((observer.directory / 'active-records.jsonl').read_text())
            self.assertTrue(record['command_dispatched'])
            self.assertEqual(record['command_body'], [0., 0., 0.])
            self.assertFalse(record['prediction']['motion_permitted'])
            with h5py.File(observer.directory / 'active-inputs.h5') as stored:
                self.assertEqual(stored['000000/depth_m'].shape, (4, 4))

    def test_staged_far_uses_model_point_then_admits_original_near_matcher(self):
        observer = self.observer()
        observer.staged = True
        candidate = observer.model.predict.return_value['candidates'][0]
        candidate.update(point_xy=[.7, .4], verification_score=.95)
        observer.model.predict.return_value['point_xy'] = [.7, .4]
        rgb = np.zeros((40, 50, 3), np.uint8)
        for _ in range(3):
            far = observer.query(rgb, use_roi=False)
            observer.count += 1
        self.assertAlmostEqual(far[0], .4)
        self.assertAlmostEqual(far[1], -.2)
        observer.part_matcher.query.assert_not_called()
        observer.part_matcher.query_all.assert_not_called()
        observer.part_matcher.query.return_value = (.1, .2, 1., .8, np.ones(4))
        near = observer.query(rgb, use_roi=True)
        self.assertEqual(near[3], .8)
        self.assertEqual(observer.near_admission['model_record_index'], 2)
        self.assertEqual(observer.last_prediction['phase'], 'near_part')
        self.assertFalse(observer.last_prediction['model_inference_performed'])
        observer.query(rgb, use_roi=True)
        self.assertEqual(observer.model.predict.call_count, 3)
        self.assertEqual(observer.part_matcher.query.call_count, 2)

    def test_staged_near_rejects_missing_or_stale_model_confirmation(self):
        observer = self.observer()
        observer.staged = True
        observer.last_prediction = {}
        rgb = np.zeros((40, 50, 3), np.uint8)
        self.assertEqual(observer.query(rgb, use_roi=True)[3], -1.)
        observer.last_observation_time = 1.
        observer.last_prediction = dict(confirmed=True, motion_permitted=True)
        with patch('perception.active_verification.time.monotonic', return_value=4.):
            self.assertEqual(observer.query(rgb, use_roi=True)[3], -1.)
        observer.part_matcher.query.assert_not_called()

    def test_active_staged_preflight_checks_only_the_part_templates_it_uses(self):
        from control import run_approach as controller
        rgb = np.zeros((4, 4, 3), np.uint8)
        matcher = Mock()
        matcher.query.return_value = (0., 0., 1., .9, np.ones(4))
        options = SimpleNamespace(active_verifier='frozen.json', handoff_on_roi=False, validate_only=True)
        with TemporaryDirectory() as directory:
            out = Path(directory)
            np.savez(out / 'validation-frames.npz', far_rgb=rgb, far_bbox=np.array([.5, .5, .8, .8]),
                     near_rgb=rgb, near_bbox=np.array([-.2, -.2, .2, .2]))
            with patch.multiple(controller, common_start=False, oblique=True, options=options,
                                OUT=out, setup={}, args=SimpleNamespace(control_min_peak=.2),
                                target_part='nose', create=True), \
                    patch.object(controller, 'SmallBatchFingerprint', return_value=matcher):
                controller.matching_preflight()
        self.assertEqual(matcher.query.call_count, 1)
        self.assertTrue(matcher.query.call_args.kwargs['use_roi'])

    def _graded_observer(self, score=.75, peer_score=.30):
        observer = self.observer(accepted=False)
        observer.staged = True
        observer.model.prepare_query.return_value = dict(candidates=[dict(box_xyxy=[880, 460, 1040, 620])])
        observer.model.select.return_value = dict(
            accepted=False, point_xy=[.5, .5], score=score, chosen_rank=1,
            candidates=[dict(point_xy=[.5, .5], box_xyxy=[880, 460, 1040, 620], verification_score=score)],
        )
        observer.model.score_candidate.return_value = peer_score
        observer.peer_ids = ['peer_a']
        cal = DistanceBinCalibrator()
        cal.fit([.1, .2, .3, .4, .7, .8, .9, .95] * 8, [0, 0, 0, 0, 1, 1, 1, 1] * 8, [70.] * 64)
        observer.graded_gate = GradedGate(cal, dict(
            far_consistent_frames=2, far_min_score=.50, far_llr_threshold=-.2,
            confirm_max_range_m=80., confirm_llr_threshold=1.5, confirm_min_mid_frames=2,
            require_same_class_margin=True, min_same_class_logit_margin=.3,
        ))
        return observer

    def test_graded_far_permits_cautious_approach_without_peer_calls_and_blocks_near(self):
        observer = self._graded_observer(score=.72, peer_score=.20)
        rgb = np.zeros((40, 50, 3), np.uint8)
        meta0 = dict(pos_x=0., pos_y=0., pos_z=-60., rot_w=1., rot_x=0., rot_y=0., rot_z=0., width=1920, height=1080)
        meta1 = dict(pos_x=3., pos_y=0., pos_z=-60., rot_w=1., rot_x=0., rot_y=0., rot_z=0., width=1920, height=1080)
        with patch('perception.active_verification._sample_candidate_depth', return_value=110.0):
            r0 = observer.query(rgb, use_roi=False, rgb_metadata=meta0, depth_message={}, time_s=0.0)
            r1 = observer.query(rgb, use_roi=False, rgb_metadata=meta1, depth_message={}, time_s=1.0)
        self.assertEqual(r0[3], -1.)
        self.assertAlmostEqual(r1[3], .72)
        self.assertTrue(observer.last_prediction['motion_permitted'])
        self.assertFalse(observer.last_prediction['confirmed'])
        self.assertEqual(observer.last_prediction['speed_cap_mps'], 1.5)
        observer.model.score_candidate.assert_not_called()
        # Near handoff must still be blocked before mid-range identity confirmation
        near = observer.query(rgb, use_roi=True)
        self.assertEqual(near[3], -1.)
        observer.part_matcher.query.assert_not_called()

    def test_graded_mid_confirms_with_peer_margin_and_admits_near(self):
        observer = self._graded_observer(score=.90, peer_score=.20)
        rgb = np.zeros((40, 50, 3), np.uint8)
        observer.part_matcher.query.return_value = (.1, .2, 1., .8, np.ones(4))
        with patch('perception.active_verification._sample_candidate_depth', return_value=60.0):
            for step in range(4):
                meta = dict(pos_x=step * 3., pos_y=0., pos_z=-35., rot_w=1., rot_x=0., rot_y=0., rot_z=0., width=1920, height=1080)
                observer.query(rgb, use_roi=False, rgb_metadata=meta, depth_message={}, time_s=float(step))
                observer.count += 1
        self.assertTrue(observer.last_prediction['confirmed'])
        self.assertGreater(observer.model.score_candidate.call_count, 0)
        near = observer.query(rgb, use_roi=True)
        self.assertEqual(near[3], .8)
        self.assertEqual(observer.last_prediction['phase'], 'near_part')

    def test_graded_mid_blocks_confirmation_when_peer_beats_target(self):
        observer = self._graded_observer(score=.75, peer_score=.92)
        rgb = np.zeros((40, 50, 3), np.uint8)
        with patch('perception.active_verification._sample_candidate_depth', return_value=60.0):
            for step in range(5):
                meta = dict(pos_x=step * 3., pos_y=0., pos_z=-35., rot_w=1., rot_x=0., rot_y=0., rot_z=0., width=1920, height=1080)
                observer.query(rgb, use_roi=False, rgb_metadata=meta, depth_message={}, time_s=float(step))
        self.assertFalse(observer.last_prediction['confirmed'])
        self.assertLess(observer.last_prediction['gate']['same_class_mean_margin'], 0.0)


if __name__ == '__main__':
    unittest.main()
