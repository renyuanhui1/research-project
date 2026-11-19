"""Use the frozen candidate verifier to select the part search crop and permit motion."""
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import torch

from common.paths import project_path
from common.sensors import decode
from evaluation.eval_template_selection import read, digest
from perception.multicandidate_verification import MultiCandidateVerifier


class ActivePartMatcher:
    def __init__(self, part_matcher, frozen_path, target_id, target_actor, directory, *, staged=False):
        self.part_matcher = part_matcher
        self.args = part_matcher.args
        self.grid = part_matcher.grid
        self.directory = None if directory is None else Path(directory)
        frozen_path = project_path(frozen_path)
        frozen = read(frozen_path)
        for path, expected in (frozen['sources'] | frozen['inputs']).items():
            if digest(project_path(path)) != expected:
                raise ValueError('Frozen source differs: ' + path)
        export = frozen_path.parent / 'verifier.pt'
        if digest(export) != frozen['verifier_sha256']:
            raise ValueError('Verifier export changed')
        cfg = frozen['configuration']
        target = next(t for t in read(project_path(cfg['catalog']))['targets'] if t['id'] == target_id)
        if target['primary_actor'] != target_actor:
            raise ValueError('Verifier target and prepared aircraft disagree')
        image = cv2.imread(str(project_path(target['template']['file'])))[:, :, ::-1].copy()
        self.model = MultiCandidateVerifier(cfg, exported=export)
        self.model.add_reference(target_id, image)
        self.target_id = target_id
        self.spec = cfg['active_control']
        self.staged = staged
        self.near_admission = None
        self.streak = 0
        self.last_observation_time = None
        self.count = 0
        self.last_diagnostics = {}
        self.last_prediction = {}
        self.manifest = dict(frozen_path=str(frozen_path), frozen_sha256=digest(frozen_path),
                             verifier_sha256=digest(export), threshold=frozen['threshold'],
                             target_id=target_id, target_actor=target_actor,
                             template=target['template'], part_template=str(self.args.template),
                             part_template_sha256=digest(self.args.template),
                             part_statistics_sha256=digest(self.args.stats_episode),
                             part_reference_metadata_sha256=digest(Path(self.args.template).parent / 'template.json'),
                             part_localization='Existing template ROI matching inside the selected model crop',
                             controls_velocity=True, controls_target_crop=True, confirmation=self.spec)
        if staged:
            self.manifest.update(control_mode='far_model_then_near_part',
                                 far_localization='Frozen model candidate point',
                                 near_localization='Prepared part ROI after confirmed model handoff',
                                 controls_target_crop=False)
        if self.directory is not None:
            (self.directory / 'active-verifier-manifest.json').write_text(json.dumps(self.manifest, indent=2) + '\n')

    def query(self, rgb, use_roi=True):
        if self.staged and use_roi:
            return self.query_near(rgb)
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = self.model.predict(rgb, self.target_id)
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        observed_at = time.monotonic()
        gap = None if self.last_observation_time is None else observed_at - self.last_observation_time
        if prediction['accepted']:
            self.streak = self.streak + 1 if gap is not None and gap <= self.spec['max_gap_s'] else 1
        else:
            self.streak = 0
        self.last_observation_time = observed_at
        confirmed = self.streak >= self.spec['confirmation_frames']
        self.last_prediction = dict(prediction, inference_s=duration, consecutive_accepted=self.streak,
                                    confirmed=confirmed, selected_box_xyxy=None,
                                    motion_permitted=False, part_point_xy=None,
                                    phase='far_whole' if self.staged else 'model_crop_part',
                                    model_inference_performed=True)
        if not confirmed:
            self.last_diagnostics = dict(whole_peak=prediction['score'], crop_xyxy=None)
            return 0., 0., 0., -1., np.full(self.grid**2, -1., np.float32)
        candidate = prediction['candidates'][prediction['chosen_rank'] - 1]
        box = candidate['box_xyxy']
        if self.staged:
            point = prediction['point_xy']
            score = prediction['score']
            similarity = np.full((self.grid, self.grid), -1., np.float32)
            for item in prediction['candidates']:
                px, py = np.clip(np.asarray(item['point_xy']) * self.grid, 0, self.grid - 1).astype(int)
                similarity[py, px] = item['verification_score']
            self.last_diagnostics = dict(whole_peak=score, crop_xyxy=box)
            self.last_prediction.update(selected_box_xyxy=box, motion_permitted=True,
                                        control_point_xy=point, point_source='new_model_candidate')
            return 2 * point[0] - 1, 2 * point[1] - 1, 1., score, similarity.reshape(-1)
        result = self.part_matcher.query_all(rgb, box=box, whole_peak=prediction['score'])[
            self.part_matcher.selected_part]
        self.last_diagnostics = self.part_matcher.last_diagnostics
        cx, cy, _, peak, _ = result
        self.last_prediction.update(selected_box_xyxy=box,
                                    part_point_xy=[(cx + 1) / 2, (cy + 1) / 2],
                                    motion_permitted=peak >= self.args.control_min_peak)
        return result

    def query_near(self, rgb):
        if self.near_admission is None:
            recent = (self.last_observation_time is not None and
                      time.monotonic() - self.last_observation_time <= self.spec['max_gap_s'])
            if not recent or not self.last_prediction.get('confirmed') or not self.last_prediction.get('motion_permitted'):
                self.last_prediction = dict(self.last_prediction, motion_permitted=False,
                                            failure_reason='near_handoff_requires_recent_model_confirmation')
                return 0., 0., 0., -1., np.full(self.grid**2, -1., np.float32)
            self.near_admission = dict(model_record_index=self.count - 1,
                                       score=self.last_prediction['score'],
                                       chosen_rank=self.last_prediction['chosen_rank'],
                                       point_xy=self.last_prediction['point_xy'],
                                       admitted_monotonic=time.monotonic())
        result = self.part_matcher.query(rgb, use_roi=True)
        cx, cy, _, peak, _ = result
        self.last_diagnostics = dict(whole_peak=self.near_admission['score'], crop_xyxy=None)
        self.last_prediction = dict(self.last_prediction, phase='near_part',
                                    model_inference_performed=False, inference_s=0.,
                                    near_admission=self.near_admission, point_source='prepared_part_roi',
                                    part_point_xy=[(cx + 1) / 2, (cy + 1) / 2],
                                    motion_permitted=peak >= self.args.control_min_peak)
        return result

    def record(self, rgb, rgb_metadata, depth_message, control_row):
        row = dict(index=self.count, control_step=control_row['step'], time_s=control_row['time_s'],
                   prediction=self.last_prediction,
                   rgb_metadata={k: v for k, v in rgb_metadata.items() if k != 'data'},
                   depth_metadata={k: v for k, v in depth_message.items() if k != 'data'},
                   command_body=[control_row[k] for k in ('forward', 'right', 'down_speed')],
                   command_dispatched=control_row['command_dispatched'],
                   yaw_rate=control_row['yaw_rate'], state=control_row['state'])
        with h5py.File(self.directory / 'active-inputs.h5', 'a') as store:
            group = store.create_group(f'{self.count:06d}')
            group.create_dataset('rgb_png', data=cv2.imencode('.png', rgb[:, :, ::-1])[1], compression='gzip')
            group.create_dataset('depth_m', data=decode(depth_message).squeeze(), compression='gzip')
            group.attrs['record'] = json.dumps(row, allow_nan=False)
        with (self.directory / 'active-records.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.count += 1
