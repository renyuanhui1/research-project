"""Collect predetermined aircraft-part views using coordinate-guided simulation flight."""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from projectairsim import ProjectAirSimClient, World, Drone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from common.runtime import configure_client_log
from common.sensors import decode, decode_image


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_rpc_depth(message):
    """RPC byte lists retain their byte layout; numeric pixel lists stay numeric."""
    data = message['data']
    if isinstance(data, list) and len(data) != message['height']*message['width']:
        message = dict(message, data=bytes(data))
    return decode(message).squeeze()


def plan_views(protocol, reference):
    cases = []
    for distance in protocol['horizontal_ranges_m']:
        for bearing in protocol['bearings_from_front_deg']:
            angle = math.radians(bearing)
            point = np.asarray(reference)+[distance*math.sin(angle), distance*math.cos(angle),
                                           -distance*math.tan(math.radians(protocol['camera_pitch_deg']))]
            yaw = math.degrees(math.atan2(reference[1]-point[1], reference[0]-point[0]))
            for offset in protocol['heading_offsets_deg']:
                cases.append(dict(id='r%02d_b%+03d_y%+03d' % (distance, bearing, offset),
                                  position_ned_m=point.tolist(), yaw_deg=yaw+offset,
                                  horizontal_range_m=distance, bearing_deg=bearing, heading_offset_deg=offset))
    return cases


def check_route(cases, protocol, profile):
    with np.load(ROOT/profile['map']['root']/'grids/context_voxels.npz') as data:
        obstacles = data['centers_enu_m']
    # Mapping uncertainty: half a five-metre voxel diagonal plus vehicle radius.
    margin = 5*math.sqrt(3)/2+.6
    points = [protocol['cruise_ned_m']]+[c['position_ned_m'] for c in cases]+[protocol['cruise_ned_m']]
    clearances = []
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a)[[1, 0, 2]]*[1, 1, -1], np.asarray(b)[[1, 0, 2]]*[1, 1, -1]
        delta = b-a
        t = np.clip(((obstacles-a)@delta)/(delta@delta), 0, 1) if delta@delta else np.zeros(len(obstacles))
        clearances.append(float(np.linalg.norm(obstacles-(a+t[:, None]*delta), axis=1).min()-margin))
    with np.load(ROOT/profile['map']['root']/'meshes'/(profile['target']['actor']+'.npz')) as data:
        highest = float(data['vertices_enu'][:, 2].max())
    minimum_altitude = min(-p[2] for p in points)
    if min(clearances) < 3 or minimum_altitude-highest < 5:
        raise ValueError('Planned route lacks conservative mapped clearance')
    return dict(context_clearance_lower_bound_m=min(clearances), minimum_altitude_m=minimum_altitude,
                aircraft_highest_point_m=highest, limitations='Coarse static-map check, not continuous body-clearance verification')


async def collect(options, protocol, profile, setup, cfg, cases, out, report):
    client = ProjectAirSimClient(options.address)
    drone = None
    collision = {}
    collision_baseline = 0
    action_log = []
    report['actions'] = action_log

    def guard():
        if int(collision.get('time_stamp', 0)) > collision_baseline:
            raise RuntimeError('New collision event: '+json.dumps(collision))

    async def action(name, factory):
        guard()
        values = []
        task = await factory(lambda value: values.append(value))
        await asyncio.wait_for(task, 65)
        kin = drone.get_ground_truth_kinematics()
        action_log.append(dict(action=name, returns=values, kinematics=kin))
        print('ACTION', name, values, kin['pose']['position'], flush=True)
        guard()

    async def hover():
        await action('hover', lambda callback: drone.hover_async(callback=callback))

    async def move(goal):
        start = np.array([drone.get_ground_truth_kinematics()['pose']['position'][k] for k in ('x', 'y', 'z')])
        goal = np.array(goal)
        segments = max(1, int(np.ceil(np.linalg.norm(goal-start)/(protocol['speed_mps']*20))))
        for index in range(1, segments+1):
            point = start+(goal-start)*index/segments
            await action('move', lambda callback: drone.move_to_position_async(*map(float, point), protocol['speed_mps'],
                                                                               timeout_sec=50, callback=callback))
            actual = np.array([drone.get_ground_truth_kinematics()['pose']['position'][k] for k in ('x', 'y', 'z')])
            if np.linalg.norm(actual-point) > 1.5:
                raise RuntimeError('Coordinate waypoint not reached')

    async def settle(case):
        deadline = time.monotonic()+40
        target = np.array(case['position_ned_m'])
        stable = 0
        while time.monotonic() < deadline:
            guard()
            kin = drone.get_ground_truth_kinematics()
            pos = np.array([kin['pose']['position'][k] for k in ('x', 'y', 'z')])
            speed = np.linalg.norm([kin['twist']['linear'][k] for k in ('x', 'y', 'z')])
            q = kin['pose']['orientation']
            yaw = math.degrees(math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2)))
            yaw_error = abs((yaw-case['yaw_deg']+180)%360-180)
            position_ready = (np.linalg.norm(pos-target) <= protocol['position_tolerance_m'] and
                              yaw_error <= protocol['heading_tolerance_deg'])
            valid = position_ready and speed <= protocol['speed_tolerance_mps']
            stable = stable+1 if valid else 0
            if stable >= 4:
                await hover()
                await asyncio.sleep(.3)
                final = drone.get_ground_truth_kinematics()
                final_pos = np.array([final['pose']['position'][k] for k in ('x', 'y', 'z')])
                final_speed = np.linalg.norm([final['twist']['linear'][k] for k in ('x', 'y', 'z')])
                if np.linalg.norm(final_pos-target) <= protocol['position_tolerance_m'] and final_speed <= protocol['speed_tolerance_mps']:
                    return final
                stable = 0
            if position_ready and (not valid or stable == 1):
                await hover()
            elif not valid:
                velocity = np.clip((target-pos)*.7, -.5, .5)
                report.setdefault('settling_commands', []).append(dict(condition=case['id'], velocity_ned_mps=velocity.tolist()))
                await asyncio.wait_for(await drone.move_by_velocity_async(*map(float, velocity), duration=.2,
                                                                          yaw_is_rate=False, yaw=math.radians(case['yaw_deg'])), 3)
            await asyncio.sleep(.25)
        raise RuntimeError('Observation pose did not settle: '+case['id'])

    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        event = threading.Event()
        def topics(t, message):
            client.on_topic_info(t, message)
            event.set()
        client.subscribe('/$topics', topics)
        if not event.wait(8):
            raise RuntimeError('Initial topic discovery timeout')
        clocks = [t for t in client.topics if t.endswith('/clock')]
        if len(clocks) != 1:
            raise RuntimeError('Expected one active scene')
        world = World.__new__(World)
        world.client = client
        world.parent_topic = clocks[0].rsplit('/clock', 1)[0]
        actors = world.list_actors()
        target_pose = client.request(dict(method=world.parent_topic+'/GetObjectPose',
                                          params=dict(object_name=profile['target']['actor']), version=1.))
        scale = world.get_object_scale(profile['target']['actor'])
        report['preflight'] = dict(actors=actors, target_pose=target_pose, target_scale=scale,
                                   flight_commands_sent=False, scene_reloaded=False)
        (out/'report.json').write_text(json.dumps(report, indent=2))
        if actors and options.resume_from is None:
            raise RuntimeError('Scene already has actors; refusing to reload')
        if options.resume_from is not None:
            previous = read(options.resume_from/'report.json')
            if (previous['status'] != 'failed' or previous['samples'] or
                    previous['source_run'] != str(options.source_run) or
                    read(options.resume_from/'runtime-scene.json') != cfg or actors != [profile['simulator']['drone']]):
                raise RuntimeError('Resume requires this collection\'s zero-sample preparation failure and matching config')
        old = read(options.source_run/'evaluation/inspection.json')['target_pose']
        if not np.allclose([target_pose['translation'][k] for k in ('x', 'y', 'z')],
                           [old['translation'][k] for k in ('x', 'y', 'z')], atol=.01):
            raise RuntimeError('Target moved from frozen reference geometry')
        if not np.allclose(scale, [20, 20, 20], atol=.01) or target_pose['rotation'] != old['rotation']:
            raise RuntimeError('Target scale or rotation changed')
        report['inspection'] = dict(target_pose=target_pose)
        world.sim_config_path = str(ROOT/'configs/simulation')
        world.sim_instance_idx = -1
        if options.resume_from is None:
            world.load_scene(cfg, delay_after_load_sec=1)
            report['scene_reloaded'] = True
        else:
            world.sim_config = cfg
            world.home_geo_point = cfg['home-geo-point']
            report['resumed_preparation'] = str(options.resume_from)
        drone = Drone(client, world, profile['simulator']['drone'])
        if options.resume_from is not None:
            kin = drone.get_ground_truth_kinematics()
            pos = np.array([kin['pose']['position'][k] for k in ('x', 'y', 'z')])
            speed = np.linalg.norm([kin['twist']['linear'][k] for k in ('x', 'y', 'z')])
            known = [protocol['spawn_ned_m'], protocol['cruise_ned_m']]+[case['position_ned_m'] for case in cases]
            if (min(np.linalg.norm(pos-point) for point in np.array(known)) > 2 or
                    -pos[2] < .5 or speed > .08 or drone.get_landed_state() != 1):
                raise RuntimeError('Existing robot is not hovering near a known preparation waypoint')
            report['recovery_state'] = kin
        def collided(t, message):
            collision.clear()
            collision.update(message)
        client.subscribe(drone.robot_info['collision_info'], collided)
        await asyncio.sleep(.5)
        collision_baseline = int(collision.get('time_stamp', 0))
        if not drone.is_api_control_enabled() and not drone.enable_api_control():
            raise RuntimeError('API control unavailable')
        if drone.get_landed_state() != 1 and not drone.arm():
            raise RuntimeError('Arm refused')
        report['flight_commands_sent'] = True
        if drone.get_landed_state() != 1:
            await action('takeoff', lambda callback: drone.takeoff_async(callback=callback))
        deadline = time.monotonic()+10
        while drone.get_landed_state() != 1 or -drone.get_ground_truth_kinematics()['pose']['position']['z'] < .5:
            if time.monotonic() >= deadline:
                raise RuntimeError('Takeoff did not produce an airborne state')
            await asyncio.sleep(.2)
        await move(protocol['cruise_ned_m'])
        await hover()
        last_timestamp = -1
        (out/'frames').mkdir()
        for case in cases:
            guard()
            await move(case['position_ned_m'])
            await action('yaw', lambda callback: drone.rotate_to_yaw_async(math.radians(case['yaw_deg']),
                                                                          timeout_sec=20, margin=math.radians(.2), callback=callback))
            await hover()
            await settle(case)
            for frame_index in range(protocol['frames_per_condition']):
                guard()
                images = drone.get_images(profile['simulator']['camera'], [0, 2])
                rgb_message, depth_message = images[0], images[2]
                stamp = int(rgb_message['time_stamp'])
                skew = abs(stamp-int(depth_message['time_stamp']))/1e9
                if stamp <= last_timestamp or skew > protocol['max_rgb_depth_skew_s']:
                    raise RuntimeError('Duplicate/stale RGB or unsynchronized RGB-D')
                last_timestamp = stamp
                rgb = decode_image(rgb_message)[0]
                depth = decode_rpc_depth(depth_message)
                if rgb.shape != (1080, 1920, 3) or depth.shape != (360, 640):
                    raise RuntimeError('Unexpected native sensor dimensions')
                kin = await settle(case)
                name = case['id']+'_f%d' % frame_index
                rgb_path, depth_path = out/'frames'/(name+'.png'), out/'frames'/(name+'-depth.npy')
                if not cv2.imwrite(str(rgb_path), rgb[:, :, ::-1]):
                    raise RuntimeError('Cannot save RGB')
                np.save(depth_path, depth)
                valid = depth[177:183, 317:323]
                valid = valid[np.isfinite(valid) & (valid > .1)]
                record = dict(name=name, case=case, rgb=str(rgb_path.relative_to(out)), depth=str(depth_path.relative_to(out)),
                              camera_metadata={k: v for k, v in rgb_message.items() if k != 'data'},
                              depth_metadata={k: v for k, v in depth_message.items() if k != 'data'},
                              rgb_sha256=digest(rgb_path), depth_sha256=digest(depth_path),
                              center_depth_m=float(np.median(valid)) if valid.size else None,
                              rgb_depth_skew_s=skew, vehicle_kinematics=kin)
                report['samples'].append(record)
                (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                print('CAPTURE', name, 'skew_s', skew, flush=True)
            print('CONDITION_DONE', case['id'], len(report['samples']), flush=True)
        await move(protocol['cruise_ned_m'])
        await hover()
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        traceback.print_exc()
    finally:
        if drone is not None:
            try:
                await asyncio.wait_for(await drone.hover_async(), 12)
                report['final_state'] = drone.get_ground_truth_kinematics()
                report['end_state'] = 'hovering; not landed or disarmed'
            except Exception as error:
                report['hover_error'] = repr(error)
        report['collision_info'] = dict(collision)
        client.disconnect()
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/missions/aircraft_a_tail_reacquisition_validation.json')
    parser.add_argument('--address', default='172.21.192.1')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume-from', type=Path, help='Resume only this collector\'s zero-sample preparation failure')
    options = parser.parse_args(argv)
    options.source_run = options.source_run.resolve()
    if options.resume_from is not None:
        options.resume_from = options.resume_from.resolve()
    out = options.out_dir.resolve()
    protocol = read(options.protocol)
    source = options.source_run
    setup, profile = read(source/'controller-setup.json'), read(source/'scene-profile.json')
    if setup['target_part'] != 'tail' or protocol['target_part'] != 'tail':
        raise ValueError('Collection requires a tail reference')
    for name, expected in setup['shared_asset_sha256'].items():
        if digest(Path(setup['shared_reference'])/name) != expected:
            raise ValueError('Frozen reference changed')
    cases = plan_views(protocol, setup['reference_ned_m'])
    route = check_route(cases, protocol, profile)
    cfg = copy.deepcopy(read(source/'runtime-scene.json'))
    actor = next(a for a in cfg['actors'] if a['name'] == profile['simulator']['drone'])
    actor['origin'] = dict(xyz=' '.join(map(str, protocol['spawn_ned_m'])), **{'rpy-deg': '0 0 -90'})
    front = next(s for s in actor['robot-config']['sensors'] if s['id'] == profile['simulator']['camera'])
    if front['origin']['rpy-deg'] != '0 -35 0':
        raise ValueError('Frozen reference camera pitch mismatch')
    summary = dict(cases=cases, route_check=route, planned_frames=len(cases)*protocol['frames_per_condition'])
    if options.validate_only:
        print(json.dumps(summary, indent=2))
        return 0
    if out.exists() and any(out.iterdir()):
        raise ValueError('Capture directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    configure_client_log(out)
    frozen = dict(protocol=protocol, source_run=str(source), reference_sha256=setup['shared_asset_sha256'], **summary,
                  source_sha256={name: digest(ROOT/name) for name in ('src/perception/part_matching.py',
                                 'src/perception/fingerprint.py', 'src/evaluation/eval_part_reacquisition.py',
                                 'src/capture/collect_part_views.py')})
    (out/'protocol.json').write_text(json.dumps(frozen, indent=2))
    (out/'runtime-scene.json').write_text(json.dumps(cfg, indent=2))
    report = dict(status='started', source_run=str(source), samples=[], flight_commands_sent=False,
                  scene_reloaded=False, capture_policy='Predetermined coordinate flight; no visual controller',
                  protocol_sha256=digest(out/'protocol.json'))
    asyncio.run(collect(options, protocol, profile, setup, cfg, cases, out, report))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
