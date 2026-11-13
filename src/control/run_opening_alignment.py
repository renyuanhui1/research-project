"""实时RGB-D洞口控制：定高水平对准，或经视觉交接与通道核验下降入室。"""
import argparse
import asyncio
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
from projectairsim import Drone, ProjectAirSimClient, World

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from common.profiles import load_profile
from common.runtime import configure_client_log
from common.sensors import decode, decode_image
from perception.fingerprint import Fingerprint
from perception.opening_matching import depth_constrained_match
from perception.opening_geometry import opening_candidates
from perception.rgbd_entry import CV_TO_SIM, VisualAnchor, FreeTube, entry_velocity


@dataclass(frozen=True)
class Limits:
    max_speed_mps: float = .35
    gain_per_s: float = .25
    pulse_s: float = .3
    max_skew_s: float = .15
    max_age_s: float = .75
    image_timeout_s: float = 3.
    min_altitude_m: float = 45.
    max_altitude_m: float = 105.
    max_altitude_drift_m: float = 1.5
    max_displacement_m: float = 15.
    min_roof_range_m: float = 15.
    min_aperture_radius_m: float = 1.
    min_score_margin: float = .1
    max_point_jump_m: float = 2.
    acquire_frames: int = 3
    aligned_frames: int = 5
    aligned_hold_s: float = 2.
    tolerance_m: float = .5
    settled_speed_mps: float = .15
    max_duration_s: float = 240.
    oblique_handoff_border_fraction: float = .08


def rotation(metadata):
    q = np.array([metadata['rot_' + k] for k in ('w', 'x', 'y', 'z')], float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or not .9 < norm < 1.1:
        raise ValueError('Invalid camera quaternion')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def validate_cameras(rgb, depth, rgb_info, depth_info, oblique=False):
    """核对针孔内参、RGB-D配准及所选控制模式的朝向。"""
    for msg, info, shape in ((rgb, rgb_info, (1920, 1080)),
                             (depth, depth_info, (640, 360))):
        w, h = msg['width'], msg['height']
        if (w, h) != shape or (info['width'], info['height']) != shape:
            raise ValueError('Unexpected live camera resolution')
        k = np.array(info['intrinsic_camera_matrix'], float).reshape(3, 3)
        expected = np.array([[w/2, 0, w/2], [0, w/2, h/2], [0, 0, 1]])
        if not np.allclose(k, expected, rtol=1e-4, atol=1e-3):
            raise ValueError('Live intrinsics disagree with 90-degree pinhole geometry')
        if not np.allclose(info['distortion_params'], 0):
            raise ValueError('Distorted camera is unsupported')
        r = rotation(msg)
        downward = r[2, 0]
        if oblique and not math.sin(math.radians(40)) <= downward <= math.sin(math.radians(80)):
            raise ValueError('Oblique mode requires 40 to 80 degrees below horizontal')
        if not oblique and downward < math.cos(math.radians(5)):
            raise ValueError('Live camera is not nadir')
    if str(depth['encoding']).upper() not in ('16FC1', '32FC1'):
        raise ValueError('Expected floating-point meter perspective depth')
    rp = np.array([rgb['pos_'+k] for k in ('x', 'y', 'z')])
    dp = np.array([depth['pos_'+k] for k in ('x', 'y', 'z')])
    if not np.isfinite(rp).all() or not np.isfinite(dp).all():
        raise ValueError('Invalid camera position metadata')
    if np.linalg.norm(rp-dp) > .15 or not np.allclose(rotation(rgb), rotation(depth), atol=.02):
        raise ValueError('RGB and depth require registration')


class FramePairs:
    """仅消费一次的独立RGB/深度帧，拒绝时间倒退、积压与过期输入。"""
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = {'rgb': deque(maxlen=4), 'depth': deque(maxlen=4)}
        self.last = {'rgb': -1, 'depth': -1}
        self.latest = {}

    def callback(self, tag):
        def receive(_, message):
            with self.lock:
                self.latest[tag] = message
                if tag in self.frames:
                    self.frames[tag].append((time.monotonic(), message))
        return receive

    def pair(self, sim_time, limits, after_sim=-1):
        with self.lock:
            choices = []
            for ra, rgb in self.frames['rgb']:
                rt = int(rgb['time_stamp'])
                if rt <= max(self.last['rgb'], after_sim):
                    continue
                for da, depth in self.frames['depth']:
                    dt = int(depth['time_stamp'])
                    if dt <= max(self.last['depth'], after_sim):
                        continue
                    age = (sim_time-min(rt, dt))/1e9
                    if (abs(rt-dt)/1e9 <= limits.max_skew_s and 0 <= age <= limits.max_age_s
                            and max(rt, dt) <= sim_time
                            and time.monotonic()-min(ra, da) <= limits.max_age_s):
                        choices.append((min(rt, dt), rgb, depth))
            if not choices:
                return None
            _, rgb, depth = max(choices, key=lambda x: x[0])
            self.last.update(rgb=int(rgb['time_stamp']), depth=int(depth['time_stamp']))
            return rgb, depth


def aperture_in_camera(match, rgb, down):
    """洞口像素射线与视觉估计的水平屋面相交；洞内地面深度不作为洞口距离。"""
    u,v=match['proposed_aperture_point_rgb_xy']
    f=rgb['width']/2
    ray=np.array([(u+.5-rgb['width']/2)/f,(v+.5-rgb['height']/2)/f,1.])
    denominator=float(ray@down)
    if denominator<=.1:raise RuntimeError('aperture_ray_not_towards_roof')
    return ray*float(match['near_layer_depth_m'])/denominator


def horizontal_error(match, rgb, oblique=False):
    """图像偏差的米制近似；仅使用近层深度与相机旋转，不使用目标或相机世界位置。"""
    if oblique:
        world=rotation(rgb)@CV_TO_SIM
        down=world.T@np.array([0.,0.,1.])
        point=aperture_in_camera(match,rgb,down)
        aim=np.array([0.,0.,float(match['near_layer_depth_m'])/down[2]])
        return (world@(point-aim))[:2]
    u, v = match['proposed_aperture_point_rgb_xy']
    d = float(match['near_layer_depth_m'])
    f = rgb['width']/2
    delta = np.array([0., (u+.5-rgb['width']/2)*d/f,
                       (v+.5-rgb['height']/2)*d/f])
    return (rotation(rgb) @ delta)[:2]


def identity_gate(match, rgb, roof_remaining, active, oblique, min_score, border_fraction=.08):
    """有效DINO帧在近景上下文出画前交接；交接后不重新激活远距模板。"""
    if not active:return False,None
    if roof_remaining<=25.:return False,{'reason':'roof_remaining_25m'}
    selected=match.get('selected')
    if (selected is None or match.get('proposed_aperture_point_rgb_xy') is None or
            not math.isfinite(selected['spatial_dino_score']) or selected['spatial_dino_score']<min_score):
        raise RuntimeError('far_approach_DINO_identity_lost')
    if oblique:
        x0,y0,x1,y1=selected['crop_rgb_xyxy'];w,h=rgb['width'],rgb['height']
        margin=min(x0/w,y0/h,(w-x1)/w,(h-y1)/h)
        if margin<=border_fraction:
            return True,{'reason':'opening_context_near_image_boundary',
                         'context_border_fraction':margin,'threshold_fraction':border_fraction,
                         'last_verified_dino_score':selected['spatial_dino_score']}
    return True,None


class Alignment:
    def __init__(self, limits, min_score, oblique=False):
        self.limits, self.min_score = limits, min_score
        self.oblique=oblique
        self.previous = None
        self.acquired = 0
        self.centered = 0
        self.centered_since = None

    def update(self, match, rgb, speed, now):
        l = self.limits
        selected = match.get('selected')
        if (selected is None or match.get('proposed_aperture_point_rgb_xy') is None
                or not math.isfinite(selected['spatial_dino_score'])
                or selected['spatial_dino_score'] < self.min_score):
            raise RuntimeError('weak_or_missing_match')
        if match.get('top_two_score_margin', 1.) < l.min_score_margin:
            raise RuntimeError('ambiguous_opening_identity')
        if not math.isfinite(match['near_layer_depth_m']) or match['near_layer_depth_m'] < l.min_roof_range_m:
            raise RuntimeError('roof_range_guard')
        if not math.isfinite(selected['projected_aperture_radius_m']) or selected['projected_aperture_radius_m'] < l.min_aperture_radius_m:
            raise RuntimeError('projected_gap_guard')
        error = horizontal_error(match, rgb, self.oblique)
        if not np.isfinite(error).all() or not math.isfinite(speed):
            raise RuntimeError('invalid_control_input')
        box = np.asarray(selected['bbox_depth_xyxy'], float)
        if self.previous is not None:
            old_error, old_box = self.previous
            overlap = np.maximum(0, np.minimum(box[2:], old_box[2:])-np.maximum(box[:2], old_box[:2])).prod()
            area = (box[2:]-box[:2]).prod() + (old_box[2:]-old_box[:2]).prod() - overlap
            if area <= 0 or overlap/area < .5 or np.linalg.norm(error-old_error) > l.max_point_jump_m:
                raise RuntimeError('opening_or_aperture_jump')
        self.previous = error, box
        self.acquired += 1
        if self.acquired < l.acquire_frames:
            return 'ACQUIRE', np.zeros(2), error
        if np.linalg.norm(error) <= l.tolerance_m:
            if speed > l.settled_speed_mps:
                self.centered, self.centered_since = 0, None
                return 'SETTLE', np.zeros(2), error
            self.centered += 1
            if self.centered_since is None:
                self.centered_since = now
            state = 'ALIGNED' if self.centered >= l.aligned_frames and now-self.centered_since >= l.aligned_hold_s else 'CONFIRM'
            return state, np.zeros(2), error
        self.centered, self.centered_since = 0, None
        velocity = l.gain_per_s * error
        velocity *= min(1., l.max_speed_mps/max(np.linalg.norm(velocity), 1e-9))
        return 'ALIGN', velocity, error


def safety_check(kin, initial, limits):
    p = np.array([kin['pose']['position'][k] for k in ('x', 'y', 'z')], float)
    if not np.isfinite(p).all():
        raise RuntimeError('invalid_vehicle_position')
    if not limits.min_altitude_m <= -p[2] <= limits.max_altitude_m:
        raise RuntimeError('altitude_guard')
    if abs(p[2]-initial[2]) > limits.max_altitude_drift_m:
        raise RuntimeError('altitude_drift_guard')
    if np.linalg.norm(p[:2]-initial[:2]) > limits.max_displacement_m:
        raise RuntimeError('displacement_guard')


def initialize_model(config, profile, out):
    source = cv2.imread(str(project_path(config['reference_image'])))
    if source is None:
        raise FileNotFoundError(config['reference_image'])
    x0, y0, x1, y1 = config['template_crop_xyxy']
    if not 0 <= x0 < x1 <= source.shape[1] or not 0 <= y0 < y1 <= source.shape[0]:
        raise ValueError('Invalid template crop')
    cv2.imwrite(str(out/'opening-template.png'), source[y0:y1, x0:x1])
    with h5py.File(out/'reference-stats.h5', 'w') as h:
        h.create_dataset('rgb', data=source[None, :, :, ::-1], compression='gzip')
    args = SimpleNamespace(template=out/'opening-template.png', stats_episode=out/'reference-stats.h5',
                           repo_dir=project_path(profile['assets']['dino_repo']),
                           dino_weights=project_path(profile['assets']['dino_weights']),
                           image_size=config['image_size'], template_roi=None)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    fp = Fingerprint(args, 'cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        tokens = fp._encode(np.ascontiguousarray(source[y0:y1, x0:x1, ::-1])[None])[0]
        tokens = torch.nn.functional.normalize((tokens-fp.zm.cpu())/fp.zs.cpu(), dim=-1)
    return fp, tokens, hashlib.sha256(args.dino_weights.read_bytes()).hexdigest()


def capture_interior(drone,out,oblique=False):
    """使用已有曝光配置采集室内前向图，随后恢复原相机安装姿态。"""
    pose={'translation':{'x':0.,'y':0.,'z':.3},
          'rotation':{'w':1.,'x':0.,'y':0.,'z':0.},'frame_id':'DEFAULT_FRAME'}
    result={}
    try:
        if not drone.set_camera_pose('FrontCamera',pose):
            raise RuntimeError('interior_camera_pose_not_confirmed')
        time.sleep(2)
        photo=drone.get_images('FrontCamera',[0])[0]
        image=decode_image(photo)[0]
        if not cv2.imwrite(str(out/'interior-observation.png'),image[:,:,::-1]):
            raise RuntimeError('interior_image_save_failed')
        result={'image':'interior-observation.png','camera':'FrontCamera',
                'metadata':{k:v for k,v in photo.items() if k!='data'},
                'temporary_relative_pose':pose,'mean':float(image.mean()),
                'white_fraction':float((image.min(2)>=250).mean()),'pixel_std':float(image.std())}
        if result['white_fraction']>.5 or result['pixel_std']<4:
            raise RuntimeError('interior_image_quality_rejected')
    finally:
        result['original_camera_pose_restored']=drone.reset_camera_pose('FrontCamera')
        if not oblique:result['nadir_camera_restored']=result['original_camera_pose_restored']
    return result


async def run(options):
    out = options.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use an empty output directory: '+str(out))
    configure_client_log(out)
    profile = load_profile(options.profile)
    config = json.loads(project_path(options.config).read_text())
    cfg = json.loads(project_path(options.runtime_scene).read_text())
    limits = Limits(max_duration_s=options.max_duration,
                    min_altitude_m=8. if options.enter else 45.,
                    max_altitude_drift_m=65. if options.enter else 1.5,
                    max_displacement_m=60. if options.oblique else 15.)
    frames = FramePairs()
    client = ProjectAirSimClient(options.address or profile['simulator']['address'])
    drone, engaged = None, False
    report = {'status': 'started', 'scope': 'horizontal_alignment_only', 'execute': options.execute,
              'flight_commands_sent': False, 'velocity_commands_sent': 0,
              'ue_target_coordinates_used_for_control': False, 'collision_truth_queries_used': False,
              'vertical_policy': 'hold initial vehicle NED Z; no planned descent',
              'normalization_source': config['reference_image'], 'query_frames_used_for_normalization': False,
              'score_threshold_calibrated_on_held_out_data': False,
              'aligned': False, 'limits': asdict(limits), 'rows': 0}
    report.update(oblique=options.oblique,
                  roof_model='horizontal plane estimated from gravity-projected RGB-D layers',
                  camera_policy='fixed oblique' if options.oblique else 'nadir',
                  entry_path='visual ray through roof aperture' if options.oblique else 'vertical through roof aperture')
    (out/'mission.json').write_text(json.dumps(config, indent=2))
    (out/'runtime-scene.json').write_text(json.dumps(cfg, indent=2))
    if options.enter:
        report.update(scope='visual_approach_and_entry',entered=False,
                      vertical_policy='RGB-D anchored relative descent, 8m beyond estimated roof plane',
                      entry_limits={'inside_beyond_roof_m':8.,'tube_half_width_m':1.25,
                                    'far_descent_speed_mps':.8,'near_descent_speed_mps':.35,
                                    'pulse_s':.8,'minimum_optical_axis_range_m':5.,
                                    'visual_altitude_disagreement_guard_m':1.5,
                                    'tube_cached_observation_max_age_s':45.,'static_scene_assumption':True})

    async def hover():
        report['flight_commands_sent'] = True
        task = await asyncio.wait_for(drone.hover_async(), 2)
        result = await asyncio.wait_for(task, 5)
        if result is not True:
            raise RuntimeError('hover_command_not_confirmed')

    def sim_time(world):
        return int(world.get_sim_time())

    try:
        for port in (8990, 8989):
            with socket.create_connection((client.address, port), timeout=2):
                pass
        fp, tokens, sha = initialize_model(config, profile, out)
        report.update(weights_sha256=sha, device=fp.dev)
        client.connect()
        client.socket_services.recv_timeout = 5000
        client.socket_topics.recv_buffer_size = 4
        event = threading.Event()
        def topics(t, m):
            client.on_topic_info(t, m)
            event.set()
        client.subscribe('/$topics', topics)
        if not event.wait(5):
            raise RuntimeError('topic_discovery_timeout')
        name = profile['simulator']['drone']
        roots = {t.split('/robots/'+name+'/')[0] for t in client.topics if '/robots/'+name+'/' in t}
        if roots != {'/Sim/'+cfg['id']}:
            raise RuntimeError('existing_drone_scene_does_not_match_runtime_snapshot')
        world = World.__new__(World)
        world.client, world.parent_topic, world.sim_config = client, next(iter(roots)), cfg
        world.home_geo_point = cfg.get('home-geo-point', {})
        drone = Drone(client, world, name)
        report.update(scene_topic=world.parent_topic, transport_address=client.address,
                      initial_kinematics=drone.get_ground_truth_kinematics(),
                      landed_state=drone.get_landed_state(), api_control_enabled=drone.is_api_control_enabled())
        initial = np.array([report['initial_kinematics']['pose']['position'][k] for k in ('x', 'y', 'z')])
        safety_check(report['initial_kinematics'], initial, limits)
        if report['landed_state'] != 1 or not report['api_control_enabled']:
            raise RuntimeError('requires_existing_airborne_drone_with_API_control')
        camera = drone.sensors['FrontCamera']
        for tag, key in [('rgb_info', 'scene_camera_info'), ('depth_info', 'depth_camera_info')]:
            if camera[key] not in client.topics:
                raise RuntimeError('missing_live_camera_topic: '+key)
            client.subscribe(camera[key], frames.callback(tag))
        info_deadline=time.monotonic()+10
        while not all(k in frames.latest for k in ('rgb_info','depth_info')):
            if time.monotonic()>info_deadline:raise RuntimeError('camera_intrinsics_timeout')
            await asyncio.sleep(.03)
        for tag,key in [('rgb','scene_camera'),('depth','depth_camera')]:
            if camera[key] not in client.topics:raise RuntimeError('missing_live_camera_topic: '+key)
            client.subscribe(camera[key],frames.callback(tag))
        client.subscribe(drone.robot_info['collision_info'], frames.callback('collision'))
        await asyncio.sleep(1)
        report['collision_baseline'] = int(frames.latest.get('collision', {}).get('time_stamp', 0))
        controller = Alignment(limits, config['min_spatial_dino_score'],options.oblique)
        tracker=tube=None
        dino_identity_active=True
        world_from_anchor=None
        handoff_vehicle_z=None
        entry_confirm=0
        entry_confirm_start=None
        if options.execute:
            engaged = True
            await hover()
        after_sim = sim_time(world)
        start = time.monotonic()
        (out/'frames').mkdir()
        with (out/'signals.jsonl').open('w', buffering=1) as log:
            while time.monotonic()-start < limits.max_duration_s:
                deadline = time.monotonic()+limits.image_timeout_s
                pair = None
                while time.monotonic() < min(deadline, start+limits.max_duration_s):
                    pair = frames.pair(sim_time(world), limits, after_sim)
                    if pair is not None:
                        break
                    await asyncio.sleep(.03)
                if pair is None:
                    raise RuntimeError('fresh_independent_RGBD_timeout')
                rgb_msg, depth_msg = pair
                rgb = decode_image(rgb_msg)[0]
                depth = decode(depth_msg)[..., 0].astype(np.float32)
                stem = 'frames/%04d' % report['rows']
                cv2.imwrite(str(out/(stem+'.png')), rgb[:, :, ::-1])
                np.save(out/(stem+'-depth.npy'), depth)
                report['last_input'] = {
                    'image': stem+'.png', 'depth': stem+'-depth.npy',
                    'rgb_metadata': {k: v for k, v in rgb_msg.items() if k != 'data'},
                    'depth_metadata': {k: v for k, v in depth_msg.items() if k != 'data'}}
                validate_cameras(rgb_msg, depth_msg, frames.latest['rgb_info'], frames.latest['depth_info'],options.oblique)
                current_down=(rotation(rgb_msg)@CV_TO_SIM).T@np.array([0.,0.,1.]) if options.oblique else None
                match, view = depth_constrained_match(fp, rgb, depth, tokens, config['min_spatial_dino_score'],down=current_down)
                report['last_match'] = match
                cv2.imwrite(str(out/(stem+'-matching.png')), view)
                kin = drone.get_ground_truth_kinematics()
                safety_check(kin, initial, limits)
                collision = frames.latest.get('collision', {})
                if int(collision.get('time_stamp', 0)) > report['collision_baseline']:
                    report['collision_event'] = collision
                    raise RuntimeError('new_collision_event')
                age = (sim_time(world)-min(rgb_msg['time_stamp'], depth_msg['time_stamp']))/1e9
                if not 0 <= age <= limits.max_age_s:
                    raise RuntimeError('RGBD_expired_during_inference')
                speed = math.hypot(kin['twist']['linear']['x'], kin['twist']['linear']['y'])
                entry_data=None
                command_down=0.
                pulse_s=limits.pulse_s
                if tracker is None:
                    state, velocity, error = controller.update(match, rgb_msg, speed, time.monotonic())
                    if options.enter and state=='ALIGNED':
                        report['aligned']=True
                        geometry=opening_candidates(depth,down=current_down)
                        selected=next(c for c in geometry['candidates'] if c['label']==match['selected']['candidate_label'])
                        tracker=VisualAnchor(rgb,depth,selected['visible_gap_mask'])
                        world_from_anchor=rotation(rgb_msg)@CV_TO_SIM
                        down=world_from_anchor.T@np.array([0.,0.,1.])
                        aperture=aperture_in_camera(match,rgb_msg,down)
                        tube=FreeTube(aperture,down,world_from_anchor,oblique=options.oblique)
                        certificate=tube.update(depth,np.eye(3),np.zeros(3),time.monotonic(),np.zeros(3))
                        handoff_vehicle_z=pos_z=float(kin['pose']['position']['z'])
                        report['handoff']={'at_step':report['rows'],'anchor_kinematics_z_ned_m':pos_z,
                            'roof_distance_from_anchor_m':tube.roof_distance,'goal_progress_m':tube.goal_progress,
                            'aperture_anchor_cv_m':aperture.tolist(),
                            'path_angle_below_horizontal_deg':float(np.rad2deg(np.arctan2(1.,np.linalg.norm((world_from_anchor@tube.axis_slope)[:2])))),
                            'visual_features':tracker.quality,'tube':certificate,'reference_image':stem+'.png'}
                        state='HANDOFF_VALIDATED'
                else:
                    center=tracker.update(rgb,depth)
                    report['last_visual_pose']={'center_anchor_cv_m':center.tolist(),'quality':tracker.quality}
                    now=time.monotonic()
                    certificate=tube.update(depth,tracker.rotation,tracker.translation,now,center)
                    state,command,error,progress=entry_velocity(center,tube,world_from_anchor)
                    velocity=command[:2];command_down=float(command[2]);pulse_s=.8
                    actual_descent=float(kin['pose']['position']['z'])-handoff_vehicle_z
                    if abs(actual_descent-progress)>1.5:
                        report['independent_safety_intervention']='visual_altitude_disagreement'
                        raise RuntimeError('visual_altitude_disagreement_guard')
                    predicted_rotation=world_from_anchor@tracker.rotation.T
                    actual_rotation=rotation(rgb_msg)@CV_TO_SIM
                    attitude_error=float(np.rad2deg(np.arccos(np.clip((np.trace(predicted_rotation.T@actual_rotation)-1)/2,-1,1))))
                    if attitude_error>3.:raise RuntimeError('visual_attitude_disagreement_guard')
                    floor=depth[depth.shape[0]//2-4:depth.shape[0]//2+5,depth.shape[1]//2-4:depth.shape[1]//2+5]
                    if not np.isfinite(floor).all() or float(np.min(floor))<5.:
                        raise RuntimeError('optical_axis_clearance_guard' if options.oblique else 'floor_clearance_guard')
                    check_identity,handoff_event=identity_gate(match,rgb_msg,tube.roof_distance-progress,
                        dino_identity_active,options.oblique,config['min_spatial_dino_score'],limits.oblique_handoff_border_fraction)
                    if check_identity:
                        if options.oblique:
                            predicted=tracker.rotation@tube.aperture+tracker.translation
                            disagreement=np.linalg.norm(aperture_in_camera(match,rgb_msg,current_down)-predicted)
                        else:
                            disagreement=np.linalg.norm(horizontal_error(match,rgb_msg)-error)
                        if disagreement>1.5:
                            raise RuntimeError('DINO_visual_anchor_disagreement')
                    if handoff_event is not None:
                        dino_identity_active=False
                        report['dino_handoff']=dict(handoff_event,at_step=report['rows'],roof_remaining_m=tube.roof_distance-progress)
                    if state=='INTERIOR_CONFIRM':
                        settled=speed<.15 and abs(kin['twist']['linear']['z'])<.15
                        entry_confirm=entry_confirm+1 if settled else 0
                        if entry_confirm==1:entry_confirm_start=now
                        if not settled:entry_confirm_start=None
                        if entry_confirm>=5 and now-entry_confirm_start>=2.:
                            state='ENTERED';report['entered']=True
                    else:
                        entry_confirm=0;entry_confirm_start=None
                    entry_data={'visual_center_anchor_cv_m':center.tolist(),'visual_descent_m':progress,
                        'roof_remaining_m':tube.roof_distance-progress,'goal_remaining_m':tube.goal_progress-progress,
                        'visual_pose':tracker.quality,'tube':certificate,
                        'independent_actual_descent_m':actual_descent,'independent_attitude_error_deg':attitude_error,
                        'minimum_optical_axis_range_m':float(np.min(floor)),
                        'path_lateral_error_m':float(np.linalg.norm(error)),
                        'DINO_identity_gate_active':check_identity}
                    if options.oblique:
                        lookahead=tube.axis_origin+tube.axis_slope*(progress+3.)
                        entry_data['path_lookahead_camera_cv_m']=(tracker.rotation@lookahead+tracker.translation).tolist()
                index = report['rows']
                row = {'step': index, 'time_s': time.monotonic()-start, 'state': state,
                       'pose': kin['pose'], 'horizontal_speed_mps': speed,
                       'rgb_metadata': {k: v for k, v in rgb_msg.items() if k != 'data'},
                       'depth_metadata': {k: v for k, v in depth_msg.items() if k != 'data'},
                       'match': match, 'error_ne_m': error.tolist(), 'velocity_ne_mps': velocity.tolist(),
                       'hold_z_ned_m': float(initial[2]), 'source_age_s': age, 'image': stem+'.png',
                       'depth': stem+'-depth.npy', 'diagnostic': stem+'-matching.png', 'velocity_sent': False}
                row.update(entry=entry_data,down_speed_mps=command_down,pulse_s=pulse_s)
                # 保存图像后再次检查帧龄；过期帧不进入速度接口。
                try:
                    age = (sim_time(world)-min(rgb_msg['time_stamp'], depth_msg['time_stamp']))/1e9
                    row['precommand_source_age_s'] = age
                    if not 0 <= age <= limits.max_age_s:
                        raise RuntimeError('RGBD_expired_before_command')
                    if options.execute and (state=='ALIGN' or
                            (entry_data is not None and state not in ('INTERIOR_CONFIRM','ENTERED'))):
                        row['velocity_sent'] = True
                        report['velocity_commands_sent'] += 1
                        if entry_data is None:
                            request=drone.move_by_velocity_z_async(float(velocity[0]),float(velocity[1]),
                                float(initial[2]),duration=pulse_s,yaw_is_rate=True,yaw=0.)
                        else:
                            request=drone.move_by_velocity_async(float(velocity[0]),float(velocity[1]),
                                command_down,duration=pulse_s,yaw_is_rate=True,yaw=0.)
                        task = await asyncio.wait_for(request,2)
                        row['command_result'] = await asyncio.wait_for(task, 3)
                        if row['command_result'] is not True:
                            raise RuntimeError('velocity_command_not_confirmed')
                        await hover()
                    after_sim = sim_time(world)
                finally:
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    report['rows'] += 1
                print('ALIGNMENT', index, state, 'error_m', np.round(error, 3).tolist(),
                      'speed', np.round(np.r_[velocity,command_down], 3).tolist(),
                      'visual_descent',None if entry_data is None else round(entry_data['visual_descent_m'],3),flush=True)
                if state=='ENTERED':
                    report['stop_reason']='entered_and_stable_visual_hover'
                    if options.execute:
                        report['interior_observation']=capture_interior(drone,out,options.oblique)
                    break
                if state == 'ALIGNED':
                    report['aligned'] = True
                    report['stop_reason'] = 'independent_frames_aligned_and_settled'
                    break
                if not options.execute and report['rows'] >= options.observe_frames:
                    report['stop_reason'] = 'observation_complete'
                    break
            else:
                report['stop_reason'] = 'duration_limit'
        report['camera_info'] = {k: frames.latest[k] for k in ('rgb_info', 'depth_info')}
        report['status'] = 'completed'
        if options.execute and options.enter and not report.get('entered'):
            report['status']='stopped'
    except BaseException as exc:
        report.update(status='stopped', stop_reason=str(exc), error=repr(exc), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report['camera_info'] = {k: frames.latest[k] for k in ('rgb_info', 'depth_info') if k in frames.latest}
        if engaged and client.state:
            try:
                await hover()
                report['final_hover_confirmed'] = True
            except Exception as exc:
                report['final_hover_error'] = repr(exc)
        if drone is not None and client.state:
            try:
                report['final_kinematics'] = drone.get_ground_truth_kinematics()
            except Exception as exc:
                report['final_state_error'] = repr(exc)
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        client.disconnect()
        print('DONE', report['status'], report.get('stop_reason'), out, flush=True)
    return 0 if report['status'] == 'completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile')
    p.add_argument('--address')
    p.add_argument('--config', default='configs/missions/hangar_opening_validation.json')
    p.add_argument('--runtime-scene', required=True, help='Snapshot describing the existing airborne drone; never loaded')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='Send bounded velocity pulses and hover commands')
    p.add_argument('--enter', action='store_true', help='Continue after alignment: RGB-D tube validation, visual tracking and entry')
    p.add_argument('--oblique', action='store_true', help='Oblique camera, gravity-projected roof geometry and slanted entry path')
    p.add_argument('--max-duration', type=float)
    p.add_argument('--observe-frames', type=int, default=8)
    options = p.parse_args()
    if options.oblique and not options.enter:p.error('--oblique requires --enter')
    ceiling=900 if options.enter else 240
    if options.max_duration is None:options.max_duration=ceiling
    if not math.isfinite(options.max_duration) or not 1 <= options.max_duration <= ceiling or options.observe_frames < 1:
        p.error('Duration exceeds this mode limit or observe-frames is not positive')
    raise SystemExit(asyncio.run(run(options)))


if __name__ == '__main__':
    main()
