"""按已知机库坐标采集洞口开发序列，保存原始RGB/深度与位姿；不执行视觉抵近。"""
import argparse
import asyncio
import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import BoxAlignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.profiles import load_profile, scene_bundle
from common.runtime import configure_client_log
from common.sensors import decode, decode_image


async def run(options):
    out = options.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Use an empty output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    configure_client_log(out)
    profile = load_profile(options.profile)
    address = options.address or profile['simulator']['address']
    client = ProjectAirSimClient(address)
    drone = None
    enabled = False
    latest = {}
    report = {'status': 'started', 'coordinate_guided_development_capture': True,
              'alignment_start_only': options.alignment_start_only,
              'visual_approach_test': False, 'target_actor': options.target, 'frames': [], 'waypoints': []}

    def save():
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def callback(tag):
        def receive(topic, message):
            latest[tag] = message
        return receive

    async def move(n, e, altitude, speed):
        result = await asyncio.wait_for(await drone.move_to_position_async(
            n, e, -altitude, speed, timeout_sec=45, yaw_is_rate=False, yaw=0), 55)
        pose = drone.get_ground_truth_kinematics()['pose']
        p = pose['position']
        error = math.sqrt((p['x']-n)**2 + (p['y']-e)**2 + (p['z']+altitude)**2)
        report['waypoints'].append({'requested_ned_m': [n,e,-altitude], 'speed_mps': speed,
                                   'result': result, 'pose': pose, 'error_m': error})
        save()
        print('MOVE', p, 'error_m', round(error, 3), flush=True)
        if error > 2:
            raise RuntimeError('Waypoint not reached')
        collision = latest.get('collision', {})
        if int(collision.get('time_stamp',0)) > report.get('collision_baseline',0):
            report['collision_event'] = collision
            raise RuntimeError('New collision event')

    async def capture(name):
        prior_rgb_timestamp=latest.get('rgb',{}).get('time_stamp',-1)
        await (await drone.hover_async())
        await asyncio.sleep(3)
        start = time.monotonic()
        while time.monotonic()-start < 10:
            if 'rgb' in latest and 'depth' in latest and latest['rgb']['time_stamp']>prior_rgb_timestamp and abs(latest['rgb']['time_stamp']-latest['depth']['time_stamp'])<=250000000:
                break
            await asyncio.sleep(.1)
        if 'rgb' not in latest or 'depth' not in latest:
            raise RuntimeError('Missing RGB/depth')
        msg, dm = latest['rgb'], latest['depth']
        if msg['time_stamp']<=prior_rgb_timestamp:
            raise RuntimeError('No fresh RGB after movement')
        if abs(msg['time_stamp']-dm['time_stamp'])>250000000:
            raise RuntimeError('RGB/depth timestamp skew exceeds 250ms')
        rgb,_ = decode_image(msg)
        if rgb.shape != (1080,1920,3):
            raise RuntimeError('Unexpected RGB resolution')
        cv2.imwrite(str(out/(name+'.png')), rgb[:,:,::-1])
        np.save(out/(name+'-depth.npy'), decode(dm)[...,0].astype(np.float32))
        report['frames'].append({'name': name, 'image': name+'.png', 'depth': name+'-depth.npy',
            'role': 'development_only', 'pose': drone.get_ground_truth_kinematics()['pose'],
            'camera_metadata': {k:v for k,v in msg.items() if k!='data'},
            'depth_metadata': {k:v for k,v in dm.items() if k!='data'},
            'image_mean': float(rgb.mean()), 'white_fraction': float((rgb.min(2)>=250).mean())})
        save()
        print('CAPTURE', name, 'mean', round(float(rgb.mean()),2), flush=True)

    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        client.socket_topics.recv_buffer_size = 12
        event = threading.Event()
        def topics(t,m):
            client.on_topic_info(t,m);event.set()
        client.subscribe('/$topics', topics)
        if not event.wait(5):
            raise RuntimeError('No topic information')
        if any('/robots/Drone1/' in t for t in client.topics):
            if not options.reset_scene:
                raise RuntimeError('A robot already exists; inspect its state before reloading with --reset-scene')
            roots={t.split('/robots/Drone1/')[0] for t in client.topics if '/robots/Drone1/' in t}
            if len(roots)!=1:raise RuntimeError('Ambiguous existing robot scene')
            report['before_explicit_scene_reset']=client.request({
                'method':roots.pop()+'/robots/'+profile['simulator']['drone']+'/GetGroundTruthKinematics',
                'params':{},'version':1.0})
        cfg = scene_bundle(profile)
        robot = cfg['actors'][0]['robot-config']
        front = next(s for s in robot['sensors'] if s['id']=='FrontCamera')
        front['origin']['rpy-deg']='0 %g 0'%(-options.camera_down_angle)
        front['gimbal']={'lock-roll':True,'lock-pitch':True,'lock-yaw':False}
        front['capture-interval']=.2
        front['capture-settings']=[c for c in front['capture-settings'] if c['image-type'] in (0,2)]
        for cap in front['capture-settings']:
            cap['streaming-enabled']=True
            if cap['image-type']==0:
                cap.update({'target-gamma':1.4,'auto-exposure-bias':2.,'auto-exposure-speed':100.,'motion-blur-amount':0.})
        for sensor in robot['sensors']:
            if sensor['id']=='DistanceSensor':sensor['draw-debug-points']=False
        (out/'runtime-scene.json').write_text(json.dumps(cfg,indent=2))
        world=World.__new__(World);world.client=client;world.sim_instance_idx=-1
        world.load_scene(cfg,delay_after_load_sec=2)
        drone=Drone(client,world,profile['simulator']['drone'])
        bounds=world.get_3d_bounding_box(options.target,BoxAlignment.WORLD_AXIS)
        center=bounds['center'];n,e=center['x'],center['y']
        top=-center['z']+bounds['size']['z']/2
        default_offset=(-(65-top)/math.tan(math.radians(options.camera_down_angle)),0.)
        offset=options.start_offset_ne if options.start_offset_ne is not None else default_offset
        if options.alignment_start_only:n,e=n+offset[0],e+offset[1]
        if 45-top<15:
            raise RuntimeError('Insufficient roof clearance for planned capture')
        minimum_altitude = 65 if options.alignment_start_only else 45
        report.update(scene_topic=world.parent_topic,target_bounds=bounds,minimum_planned_altitude_m=minimum_altitude,
                      planned_clearance_above_target_top_m=minimum_altitude-top,transport_address=address,
                      camera_down_angle_deg=options.camera_down_angle,start_offset_ne_m=list(offset),
                      starting_region_ned_m=[n,e,-65.],scene_reset_requested=options.reset_scene)
        client.subscribe(drone.robot_info['collision_info'],callback('collision'))
        if not drone.enable_api_control() or not drone.arm():raise RuntimeError('Cannot enable/arm')
        enabled=True
        report['takeoff_result']=await (await drone.takeoff_async(timeout_sec=25))
        await asyncio.sleep(1)
        report['collision_baseline']=int(latest.get('collision',{}).get('time_stamp',0))
        p=drone.get_ground_truth_kinematics()['pose']['position']
        cruise = 65 if options.alignment_start_only else 100
        await move(p['x'],p['y'],cruise,5)
        for i in range(1,7):await move(n*i/6,e*i/6,cruise,6)
        client.subscribe(drone.sensors['FrontCamera']['scene_camera'],callback('rgb'))
        client.subscribe(drone.sensors['FrontCamera']['depth_camera'],callback('depth'))
        if options.alignment_start_only:
            await capture('alignment-start-65m')
        for altitude in (() if options.alignment_start_only else (100,80,65,55,45)):
            await move(n,e,altitude,2)
            await capture('center-'+str(altitude)+'m')
        if not options.alignment_start_only:
            await move(n,e,65,2)
            for label,dn,de in [('north',6,0),('south',-6,0),('east',0,6),('west',0,-6)]:
                await move(n+dn,e+de,65,1)
                await capture('offset-'+label+'-65m')
            await move(n,e,65,1)
        report.update(status='completed',end_state='hovering at prepared start at about 65m; armed',
                      final_kinematics=drone.get_ground_truth_kinematics())
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    finally:
        if enabled and client.state:
            try:await asyncio.wait_for(await drone.hover_async(),15)
            except Exception as exc:report['hover_error']=repr(exc)
        save();client.disconnect()
        print('DONE',report['status'],out,flush=True)
    return 0 if report['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile')
    parser.add_argument('--address')
    parser.add_argument('--target',default='StaticMeshActor_2019')
    parser.add_argument('--alignment-start-only',action='store_true',
                        help='Only prepare and capture a coordinate-guided 65m alignment start; no visual control')
    parser.add_argument('--camera-down-angle',type=float,default=90.,help='Degrees below horizontal; oblique preparation requires --alignment-start-only')
    parser.add_argument('--start-offset-ne',type=float,nargs=2,help='Preparation offset from target bounds center, in N/E meters; default aims the camera towards the roof')
    parser.add_argument('--reset-scene',action='store_true',help='Explicitly inspect existing robot and reload simulation for preparation')
    parser.add_argument('--out-dir',type=Path,required=True)
    options=parser.parse_args()
    if not math.isfinite(options.camera_down_angle) or not 45<=options.camera_down_angle<=90:
        parser.error('camera-down-angle must be in [45,90]')
    if not options.alignment_start_only and (options.camera_down_angle!=90 or options.start_offset_ne is not None):
        parser.error('Oblique/offset preparation requires --alignment-start-only')
    if options.start_offset_ne is not None and not all(math.isfinite(v) and abs(v)<=60 for v in options.start_offset_ne):
        parser.error('start-offset-ne must be finite and within 60m per axis')
    raise SystemExit(asyncio.run(run(options)))


if __name__=='__main__':main()
