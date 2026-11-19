"""Asynchronous observation only: record RGB-D and frozen verifier predictions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import cv2
import h5py
import numpy as np
import torch
from common.paths import project_path
from common.sensors import decode, decode_image
from evaluation.eval_template_selection import read, digest
from perception.multicandidate_verification import MultiCandidateVerifier


def synchronized_pair(rgb_cache, depth_cache, last_rgb, last_depth, max_skew_s, max_age_s):
    """Newest unused pair, bounded by skew and age relative to the latest RGB."""
    if not rgb_cache or not depth_cache: return None
    newest = max(rgb_cache)
    depths = [ts for ts in depth_cache if ts > last_depth]
    if not depths: return None
    for rt in sorted(rgb_cache, reverse=True):
        if rt <= last_rgb or (newest-rt)/1e9 > max_age_s: break
        dt = min(depths, key=lambda ts: abs(ts-rt))
        if abs(dt-rt)/1e9 <= max_skew_s: return rgb_cache[rt], depth_cache[dt]
    return None


class ShadowVerifier:
    def __init__(self, frozen_path, directory, target_id=None):
        frozen_path = project_path(frozen_path); frozen = read(frozen_path); cfg = frozen['configuration']
        for path, expected in (frozen['sources'] | frozen['inputs']).items():
            if digest(project_path(path)) != expected: raise ValueError('Frozen source differs: '+path)
        export = frozen_path.parent/'verifier.pt'
        if digest(export) != frozen['verifier_sha256']: raise ValueError('Verifier export changed')
        self.model = MultiCandidateVerifier(cfg, exported=export)
        self.spec = cfg['shadow']; self.target_id = self.spec['target_id'] if target_id is None else target_id
        catalog = read(project_path(cfg['catalog'])); target = next(t for t in catalog['targets'] if t['id']==self.target_id)
        image = cv2.imread(str(project_path(target['template']['file'])))[:,:,::-1].copy()
        self.model.add_reference(self.target_id,image)
        self.directory = Path(directory); self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None; self.last_submit = -float('inf'); self.dropped = 0; self.count = 0; self.errors = []
        self.streak = 0; self.last_observation_time = None
        self.last_rgb = -1; self.last_depth = -1; self.unsynchronized = 0
        self.manifest = dict(frozen_path=str(frozen_path),frozen_sha256=digest(frozen_path),verifier_sha256=digest(export),
                             target_id=self.target_id,confirmation=self.spec,controls_velocity=False)
        (self.directory/'shadow-manifest.json').write_text(json.dumps(self.manifest,indent=2))

    def submit(self, rgb_cache, depth_cache, pose, time_s):
        if time_s-self.last_submit < self.spec['sample_interval_s']: return
        if self.future is not None and not self.future.done():
            self.dropped += 1; return
        pair = synchronized_pair(rgb_cache, depth_cache, self.last_rgb, self.last_depth,
                                 self.spec['max_sensor_skew_s'], self.spec['max_frame_age_s'])
        if pair is None:
            self.unsynchronized += 1; return
        rgb_metadata, depth_message = pair
        rgb = decode_image(rgb_metadata)[0]
        self.last_rgb = int(rgb_metadata['time_stamp']); self.last_depth = int(depth_message['time_stamp'])
        self.last_submit = time_s
        self.future = self.executor.submit(self._observe,rgb.copy(),dict(rgb_metadata),dict(depth_message),pose,time_s)

    def _observe(self,rgb,metadata,depth_message,pose,time_s):
        try:
            torch.cuda.synchronize();start=time.perf_counter()
            prediction=self.model.predict(rgb,self.target_id)
            torch.cuda.synchronize();duration=time.perf_counter()-start
            gap=None if self.last_observation_time is None else time_s-self.last_observation_time
            self.streak=(self.streak+1 if gap is not None and gap<=self.spec['max_gap_s'] else 1) if prediction['accepted'] else 0
            self.last_observation_time=time_s
            row=dict(prediction,time_s=time_s,inference_s=duration,consecutive_accepted=self.streak,
                     confirmed=self.streak>=self.spec['confirmation_frames'],pose=pose,
                     rgb_metadata={k:v for k,v in metadata.items() if k!='data'},
                     depth_metadata={k:v for k,v in depth_message.items() if k!='data'},
                     sensor_skew_s=abs(int(metadata['time_stamp'])-int(depth_message['time_stamp']))/1e9)
            with h5py.File(self.directory/'shadow.h5','a') as store:
                group=store.create_group(f'{self.count:06d}')
                group.create_dataset('rgb_png',data=cv2.imencode('.png',rgb[:,:,::-1])[1],compression='gzip')
                group.create_dataset('depth_m',data=decode(depth_message).squeeze(),compression='gzip')
                group.attrs['record']=json.dumps(row,allow_nan=False)
            with (self.directory/'shadow-records.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.count+=1
        except Exception as error:
            self.errors.append(repr(error)); self.streak=0

    def finish(self):
        self.executor.shutdown(wait=True)
        return dict(self.manifest,frames=self.count,dropped_busy=self.dropped,skipped_without_fresh_pair=self.unsynchronized,errors=self.errors,
                    status='completed' if not self.errors else 'failed',inputs_file='shadow.h5',records_file='shadow-records.jsonl')
