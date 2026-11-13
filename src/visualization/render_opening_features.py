"""离线重建红色DINO响应及绿色控制准星，供保存的洞口飞行记录回放。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from common.profiles import load_profile
from perception.fingerprint import Fingerprint
from perception.depth_map import _quat_to_R
from perception.opening_geometry import opening_candidates
from perception.rgbd_entry import VisualAnchor, CV_TO_SIM


def red_response(image,scores,box,temperature):
    """飞机回放的sqrt(softmax)红色样式，此处权重仅用于显示。"""
    x0,y0,x1,y1=box
    weights=np.exp((scores-scores.max())/temperature);weights/=weights.sum()
    heat=cv2.resize(np.sqrt(weights/weights.max()).reshape(32,32),(x1-x0,y1-y0))
    patch=image[y0:y1,x0:x1].astype(np.float32)
    patch[:,:,2]=patch[:,:,2]*(1-.6*heat)+255*.6*heat
    patch[:,:,:2]*=(1-.5*heat[:,:,None])
    image[y0:y1,x0:x1]=np.clip(patch,0,255).astype(np.uint8)


def control_pixel(row,depth):
    if not row.get('entry'):return row['match'].get('proposed_aperture_point_rgb_xy')
    m=row['rgb_metadata'];h,w=depth.shape
    if 'path_lookahead_camera_cv_m' in row['entry']:
        x,y,z=row['entry']['path_lookahead_camera_cv_m']
        if z<=.1:return None
        return [x/z*m['width']/2+(m['width']-1)/2,y/z*m['width']/2+(m['height']-1)/2]
    floor=float(np.median(depth[h//2-4:h//2+5,w//2-4:w//2+5]))
    q=np.array([m['rot_'+k] for k in ('w','x','y','z')]);q/=np.linalg.norm(q)
    target=(_quat_to_R(q)@CV_TO_SIM).T@np.r_[row['error_ne_m'],floor]
    if target[2]<=0:return None
    return [target[0]/target[2]*m['width']/2+(m['width']-1)/2,
            target[1]/target[2]*m['width']/2+(m['height']-1)/2]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--profile')
    options=p.parse_args();run=options.run_dir.resolve();out=options.out_dir.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Use an empty output directory')
    rows=[json.loads(s) for s in (run/'signals.jsonl').read_text().splitlines() if s.strip()]
    report=json.loads((run/'report.json').read_text());mission=json.loads((run/'mission.json').read_text())
    profile=load_profile(options.profile);weights=project_path(profile['assets']['dino_weights'])
    sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    if sha!=report['weights_sha256']:raise RuntimeError('Weights differ from flight recording')
    if mission['image_size']!=448:raise ValueError('Expected a 32x32 patch grid')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    fp=Fingerprint(SimpleNamespace(template=run/'opening-template.png',stats_episode=run/'reference-stats.h5',
        repo_dir=project_path(profile['assets']['dino_repo']),dino_weights=weights,
        image_size=448,template_roi=None),'cuda' if torch.cuda.is_available() else 'cpu')
    template=cv2.imread(str(run/'opening-template.png'))[:,:,::-1].copy()
    tokens=fp._encode(template[None])[0]
    reference=torch.nn.functional.normalize((tokens-fp.zm.cpu())/fp.zs.cpu(),dim=-1)
    arrays=np.zeros((len(rows),32,32),np.float32);valid=np.zeros(len(rows),bool)
    points=np.full((len(rows),2),np.nan,np.float32)
    tracker=None;items=[];score_error=pose_error=0.;red_frames=0
    for i,row in enumerate(rows):
        image=cv2.imread(str(run/row['image']))
        if image is None:raise FileNotFoundError(run/row['image'])
        rgb=image[:,:,::-1].copy();depth=np.load(run/row['depth'])
        selected=row['match'].get('selected');entry=row.get('entry')
        active=not entry or entry['DINO_identity_gate_active'];score=None
        if selected:
            box=selected['crop_rgb_xyxy'];x0,y0,x1,y1=box
            tokens=fp._encode(rgb[y0:y1,x0:x1].copy()[None])[0]
            tokens=torch.nn.functional.normalize((tokens-fp.zm.cpu())/fp.zs.cpu(),dim=-1)
            scores=(tokens*reference).sum(-1).numpy();score=float(scores.mean())
            difference=abs(score-selected['spatial_dino_score']);score_error=max(score_error,difference)
            if difference>1e-5:raise RuntimeError('DINO reconstruction differs from recording')
            arrays[i]=scores.reshape(32,32);valid[i]=True
            if active and score>=mission['min_spatial_dino_score']:
                red_response(image,scores,box,mission['softmax_temperature']);red_frames+=1
        if row['state']=='HANDOFF_VALIDATED':
            down=None
            if report.get('oblique'):
                m=row['rgb_metadata'];q=np.array([m['rot_'+k] for k in ('w','x','y','z')]);q/=np.linalg.norm(q)
                down=(_quat_to_R(q)@CV_TO_SIM).T@np.array([0.,0.,1.])
            geometry=opening_candidates(depth,down=down)
            mask=next(c['visible_gap_mask'] for c in geometry['candidates'] if c['label']==selected['candidate_label'])
            tracker=VisualAnchor(rgb,depth,mask)
        if entry:
            if tracker is None:raise RuntimeError('Missing visual handoff')
            center=tracker.update(rgb,depth)
            difference=float(np.linalg.norm(center-np.array(entry['visual_center_anchor_cv_m'])))
            pose_error=max(pose_error,difference)
            if difference>.01:raise RuntimeError('RGB-D reconstruction differs from recording')
        view=cv2.resize(image,(960,540))
        if entry and not active:
            for pixel in tracker.pixels[::max(1,len(tracker.pixels)//350)]:
                cv2.circle(view,tuple(np.rint(pixel).astype(int)),1,(50,170,50),-1)
        point=control_pixel(row,depth)
        if point is not None:
            points[i]=point;u=round(point[0]*960/rgb.shape[1]);v=round(point[1]*540/rgb.shape[0])
            if 0<=u<960 and 0<=v<540:cv2.drawMarker(view,(u,v),(0,255,0),cv2.MARKER_CROSS,17,2,cv2.LINE_AA)
        controller='DINO + DEPTH' if not entry else ('RGB-D + DINO CHECK' if active else 'RGB-D HANDOFF')
        name='step_%04d.png'%row['step']
        if not cv2.imwrite(str(out/name),view):raise RuntimeError('Cannot save feature view')
        items.append({'step':row['step'],'timestamp_ns':row['rgb_metadata']['time_stamp'],'image':name,
            'controller':controller,'dino_active':active,'spatial_dino_score':score,
            'green_marker_source':('recorded aperture pixel' if not entry else
                                  ('recorded visual path lookahead, 3m vertical progress' if report.get('oblique') else 'recorded RGB-D error projected to floor plane')),
            'caption':'RECOMPUTED RESPONSE | '+controller+(' | score %.3f'%score if score is not None and active else '')})
        if i%30==0:print('FEATURE_VIEW',i,controller,score,flush=True)
    np.savez_compressed(out/'patch-similarity.npz',spatial_patch_similarity=arrays,
                        has_candidate=valid,control_marker_original_rgb_xy=points)
    result={'status':'completed','source_run':str(run),'items':items,'weights_sha256':sha,
        'source_signals_sha256':hashlib.sha256((run/'signals.jsonl').read_bytes()).hexdigest(),
        'simulation_connected':False,'flight_commands_sent':False,'features_recomputed_after_flight':True,
        'red_overlay':'sqrt(softmax/max_softmax) of spatial patch cosine at T=0.005; display only, not control weights',
        'red_frames':red_frames,'frames_with_recomputed_DINO':int(valid.sum()),
        'maximum_recorded_score_difference':score_error,'maximum_recorded_visual_pose_difference_m':pose_error,
        'RGBD_features_recomputed_after_flight':True,'source_trajectory_and_control_unchanged':True}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('FEATURE_VIEW_DONE',len(items),'DINO',int(valid.sum()),'red',red_frames,
          'score_error',score_error,'pose_error',pose_error,flush=True)


if __name__=='__main__':main()
