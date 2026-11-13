"""Capture a stationary endpoint and independently evaluate the selected aircraft part."""
import argparse
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
import torch
from projectairsim import ProjectAirSimClient, World, Drone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from common.runtime import configure_client_log
from common.sensors import decode_image
from perception.part_matching import CroppedPartMatcher
from control.run_approach import SmallBatchFingerprint


def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--continue-tracking',action='store_true',help='Offline endpoint check initialized from the last accepted saved registration state')
    options=parser.parse_args(argv);run=options.run_dir.resolve();out=run/'evaluation'
    report=json.loads((run/'report.json').read_text());profile=json.loads((run/'scene-profile.json').read_text());setup=json.loads((run/'controller-setup.json').read_text())
    meta=json.loads((run/'template.json').read_text());ref=np.array(meta['reference_ned_m']);part_name=setup['target_part'];cfg=json.loads((run/'runtime-scene.json').read_text())
    out.mkdir(exist_ok=True);configure_client_log(out)
    if options.offline:
        inspection=json.loads((out/'inspection.json').read_text());rgb=cv2.imread(str(out/'final-part.png'))[:,:,::-1].copy()
    else:
        if (out/'inspection.json').exists():raise RuntimeError('Endpoint already captured; use --offline')
        client=ProjectAirSimClient(profile['simulator']['address']);inspection={'flight_commands_sent':False}
        try:
            client.connect();client.socket_services.recv_timeout=10000;event=threading.Event()
            def info(t,m):client.on_topic_info(t,m);event.set()
            client.subscribe('/$topics',info)
            if not event.wait(5):raise RuntimeError('Topic discovery timeout')
            world=World.__new__(World);world.client=client;world.parent_topic='/Sim/'+cfg['id'];world.sim_config=cfg
            drone=Drone(client,world,profile['simulator']['drone'])
            inspection['vehicle_kinematics']=drone.get_ground_truth_kinematics();inspection['target_pose']=world.get_object_pose(profile['target']['actor'])
            message=drone.get_images(profile['simulator']['camera'],[0])[0];rgb=decode_image(message)[0]
            inspection['camera_metadata']={k:v for k,v in message.items() if k!='data'};cv2.imwrite(str(out/'final-part.png'),rgb[:,:,::-1])
        finally:
            client.disconnect();(out/'inspection.json').write_text(json.dumps(inspection,indent=2))
    message=inspection['camera_metadata'];args=SimpleNamespace(**report['controller_args']);torch.set_num_threads(2);cv2.setNumThreads(1)
    if 'common_start' in setup:
        shared=json.loads((Path(setup['shared_reference'])/'template.json').read_text())
        fp=CroppedPartMatcher(args,'cuda',{k:v['template_roi'] for k,v in shared['parts'].items()},part_name)
    else:fp=SmallBatchFingerprint(args,'cuda')
    tracking_seed=None
    if options.continue_tracking:
        if not options.offline or not getattr(args,'thin_roi_refinement',False):
            raise ValueError('Tracking continuation requires --offline and a thin-ROI trial')
        rows=json.loads((run/'control-signals.json').read_text())
        last=next(row for row in reversed(rows) if row.get('refinement',{}).get('accepted'))
        state=last['refinement'];fp.refinement_state[part_name]={k:state[k] for k in ('scale','angle_deg','point_px')}
        tracking_seed={'control_step':last['step'],'rgb_timestamp':last['rgb_timestamp'],'state':fp.refinement_state[part_name]}
    cx,cy,mass,peak,sim=fp.query(rgb,use_roi=True)
    data=np.load(ROOT/profile['map']['root']/'meshes'/(profile['target']['actor']+'.npz'));vertices=data['vertices_enu'][:,[1,0,2]].copy();vertices[:,2]*=-1;faces=data['triangles'].astype(np.uint32)
    pose=inspection['target_pose'];origin=np.array([pose['translation'][k] for k in ('x','y','z')]);q=pose['rotation'];yaw=math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))+math.radians(profile['target']['forward_axis_angle_deg']);forward=np.array([math.cos(yaw),math.sin(yaw)]);lateral=np.array([-forward[1],forward[0]]);part=profile['parts'][part_name]
    def in_part(points):
        points=np.atleast_2d(points);valid=np.linalg.norm(points-ref,axis=1)<=part['radius_m']
        if part_name!='nose':
            longitudinal=(points[:,:2]-origin[:2])@forward;side=(points[:,:2]-origin[:2])@lateral
            valid &= (side>=part['lateral_min_m'])&(longitudinal>part['longitudinal_min_m'])&(longitudinal<part['longitudinal_max_m'])&(-points[:,2]>part['min_altitude_m'])
            if 'lateral_max_m' in part:valid &= side<=part['lateral_max_m']
            if 'max_altitude_m' in part:valid &= -points[:,2]<part['max_altitude_m']
        return valid
    selected=in_part(vertices);selected_faces=faces[selected[faces].all(1)]
    def scene(triangles):
        s=o3d.t.geometry.RaycastingScene();s.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),o3d.core.Tensor(triangles));return s
    full=scene(faces);local=scene(selected_faces);camera=np.array([message['pos_'+k] for k in ('x','y','z')]);body=np.array([inspection['vehicle_kinematics']['pose']['position'][k] for k in ('x','y','z')]);ray=rotation([message['rot_'+k] for k in ('w','x','y','z')])@np.array([1.,cx,cy*message['height']/message['width']]);ray/=np.linalg.norm(ray)
    distance=float(full.cast_rays(o3d.core.Tensor(np.r_[camera,ray][None].astype(np.float32)))['t_hit'].numpy()[0]);hit=camera+distance*ray if np.isfinite(distance) else None
    point_distance=lambda s,p:float(s.compute_distance(o3d.core.Tensor(p[None].astype(np.float32))).numpy()[0])
    camera_rotation=rotation([message['rot_'+k] for k in ('w','x','y','z')]);pts=(vertices[selected]-camera)@camera_rotation;uv=np.c_[message['width']/2+message['width']/2*pts[:,1]/pts[:,0],message['height']/2+message['width']/2*pts[:,2]/pts[:,0]]
    image=rgb[:,:,::-1].copy();cv2.rectangle(image,tuple(np.rint(uv.min(0)).astype(int)),tuple(np.rint(uv.max(0)).astype(int)),(255,255,0),2);center=(round((cx+1)*message['width']/2),round((cy+1)*message['height']/2));cv2.drawMarker(image,center,(0,255,0),cv2.MARKER_CROSS,25,2);cv2.putText(image,'Independent part check: '+part_name,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2);cv2.imwrite(str(out/'part-validation.png'),image)
    verified=hit is not None and bool(in_part(hit)[0]) and peak>=args.control_min_peak
    diagnostic=fp.last_diagnostics.get(part_name,{}) if 'common_start' in setup else {}
    result={'status':'passed' if verified and report.get('arrival_confirmed') else 'part_not_verified','part':part_name,'target_actor':profile['target']['actor'],'control_stop_reason':report.get('stop_reason'),'control_final_filtered_depth_m':report.get('final_depth_m'),'fresh_DINO_center_normalized_xy':[cx,cy],'fresh_DINO_peak':peak,'DINO_ray_hits_aircraft':hit is not None,'DINO_ray_hit_ned_m':None if hit is None else hit.tolist(),'DINO_ray_hits_configured_part':verified,'camera_ray_aircraft_range_m':distance if hit is not None else None,'body_to_aircraft_surface_m':point_distance(full,body),'body_to_selected_part_surface_m':point_distance(local,body),'body_to_reference_m':float(np.linalg.norm(body-ref)),'selected_mesh_vertices':int(selected.sum()),'selected_triangles_all_vertices_in_region':len(selected_faces),'camera_position_ned_m':camera.tolist(),'DINO_ray_direction_ned':ray.tolist(),'geometry_used_for_evaluation_only':True,'flight_commands_sent':False,'endpoint_speed_mps':float(np.linalg.norm([inspection['vehicle_kinematics']['twist']['linear'][k] for k in ('x','y','z')]))}
    result.update(point_source=diagnostic.get('point_source','dino_center'),matched_center_normalized_xy=[cx,cy],matching_ray_hits_configured_part=verified,refinement=diagnostic.get('refinement'),coarse_dino_center=diagnostic.get('coarse_center'))
    result.update(matching_initialization='saved_tracking_state' if tracking_seed else 'cold_start',tracking_seed=tracking_seed)
    if diagnostic.get('point_source')=='local_image_registration':
        for old,new in [('fresh_DINO_center_normalized_xy','fresh_matching_center_normalized_xy'),('DINO_ray_hits_aircraft','matching_ray_hits_aircraft'),('DINO_ray_hit_ned_m','matching_ray_hit_ned_m'),('DINO_ray_hits_configured_part','matching_ray_hits_configured_part'),('DINO_ray_direction_ned','matching_ray_direction_ned')]:
            result[new]=result.pop(old)
    (out/'part-surface-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
