"""Collect randomized part views by teleporting the drone, with per-part mask labels."""
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
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from projectairsim import ProjectAirSimClient, World, Drone
from projectairsim.types import Pose, Vector3, Quaternion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from capture.collect_part_views import read, digest, decode_rpc_depth
from capture.part_masks import PartMaskRenderer
from common.runtime import configure_client_log
from common.sensors import decode_image


def plan_views(protocol, reference, forward, rng):
    """Randomized poses in the aircraft frame; the fixed camera pitch points at the reference."""
    forward = np.asarray(forward, float)
    lateral = np.array([-forward[1], forward[0]])
    strata = protocol.get('range_strata')
    if strata:
        weights = np.array([row[2] for row in strata], float)
        picks = rng.choice(len(strata), size=protocol['count'], p=weights/weights.sum())
        distances = [float(rng.uniform(*strata[index][:2])) for index in picks]
    else:
        distances = rng.uniform(*protocol['horizontal_range_m'], protocol['count'])
    cases = []
    for case_id, distance in enumerate(distances):
        bearing = float(rng.uniform(*protocol['bearing_deg']))
        heading = float(rng.uniform(*protocol['heading_offset_deg']))
        jitter = rng.normal(0, protocol.get('position_jitter_m', 0), 2)
        offset = distance*(math.cos(math.radians(bearing))*forward +
                           math.sin(math.radians(bearing))*lateral)+jitter
        point = np.asarray(reference, float)+[offset[0], offset[1],
                                              -distance*math.tan(math.radians(protocol['camera_pitch_deg']))]
        yaw = math.degrees(math.atan2(reference[1]-point[1], reference[0]-point[0]))+heading
        cases.append(dict(id='v%05d' % case_id, position_ned_m=point.tolist(), yaw_deg=yaw,
                          horizontal_range_m=distance, bearing_deg=bearing, heading_offset_deg=heading))
    return cases


def camera_frame(position, yaw_deg, offset, mount):
    yaw = math.radians(yaw_deg)
    body = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                     [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return np.asarray(position, float)+body@offset, body@mount


def check_views(cases, protocol, profile, reference, offset, mount, shape=(1080, 1920)):
    """Reject poses that are too close to mapped context or project the reference outside the image."""
    with np.load(ROOT/profile['map']['root']/'grids/context_voxels.npz') as data:
        obstacles = data['centers_enu_m']
    tree = cKDTree(obstacles)
    margin = 5*math.sqrt(3)/2+.6
    points = np.array([case['position_ned_m'] for case in cases])
    enu = points[:, [1, 0, 2]]*[1, 1, -1]
    clearance = float((tree.query(enu)[0]-margin).min())
    minimum_altitude = float(-points[:, 2].min())
    with np.load(ROOT/profile['map']['root']/'meshes'/(profile['target']['actor']+'.npz')) as data:
        highest = float(data['vertices_enu'][:, 2].max())
    if clearance < 3 or minimum_altitude-highest < 5:
        raise ValueError('Sampled poses lack conservative mapped clearance')
    h, w = shape
    focal = w/2
    for case in cases:
        position, rotation = camera_frame(case['position_ned_m'], case['yaw_deg'], offset, mount)
        camera = rotation.T@(np.asarray(reference, float)-position)
        if camera[0] <= 0:
            raise ValueError('Reference behind the camera: '+case['id'])
        u, v = w/2+focal*camera[1]/camera[0], h/2+focal*camera[2]/camera[0]
        if not (.05*w <= u <= .95*w and .05*h <= v <= .95*h):
            raise ValueError('Reference outside the central image region: '+case['id'])
    return dict(context_clearance_lower_bound_m=clearance, minimum_altitude_m=minimum_altitude,
                aircraft_highest_point_m=highest, reference_pixels_checked=len(cases),
                limitations='Per-pose static-map clearance only; teleport capture, no continuous body-clearance check')


def forward_axis(target_pose, profile):
    q = target_pose['rotation']
    yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
    yaw += math.radians(profile['target']['forward_axis_angle_deg'])
    return np.array([math.cos(yaw), math.sin(yaw)]), np.array([target_pose['translation'][k] for k in ('x', 'y', 'z')])


def quaternion(yaw_deg):
    return Quaternion({'w': math.cos(math.radians(yaw_deg)/2), 'x': 0., 'y': 0.,
                       'z': math.sin(math.radians(yaw_deg)/2)})


def body_rotation(kinematics):
    q = kinematics['pose']['orientation']
    return Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()


def body_position(kinematics):
    return np.array([kinematics['pose']['position'][k] for k in ('x', 'y', 'z')])


async def collect(options, protocol, profile, setup, cfg, cases, renderer, out, report):
    client = ProjectAirSimClient(options.address)
    drone = None
    collision = {}
    collision_baseline = 0
    camera_id = profile['simulator']['camera']

    def guard():
        if int(collision.get('time_stamp', 0)) > collision_baseline:
            raise RuntimeError('New collision event: '+json.dumps(collision))

    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        event = threading.Event()

        def topics(topic, message):
            client.on_topic_info(topic, message)
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
        if actors and options.resume_from is None:
            raise RuntimeError('Scene already has actors; refusing to reload')
        target_pose = client.request(dict(method=world.parent_topic+'/GetObjectPose',
                                          params=dict(object_name=profile['target']['actor']), version=1.))
        scale = world.get_object_scale(profile['target']['actor'])
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
            if body_position(kin)[2] > -.5 or drone.get_landed_state() != 1:
                raise RuntimeError('Resume requires the prepared drone on the ground')
            report['recovery_state'] = kin

        def collided(topic, message):
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
            await asyncio.wait_for(await drone.takeoff_async(), 65)
        deadline = time.monotonic()+10
        while drone.get_landed_state() != 1 or -body_position(drone.get_ground_truth_kinematics())[2] < .5:
            if time.monotonic() >= deadline:
                raise RuntimeError('Takeoff did not produce an airborne state')
            await asyncio.sleep(.2)

        done = {sample['name'] for sample in report['samples']}
        warmup_deadline = time.monotonic()+120
        stable, warmup_frames = 0, 0
        while stable < 3:
            rgb = decode_image(drone.get_images(camera_id, [0])[0])[0]
            warmup_frames += 1
            stable = stable+1 if rgb.mean() < 220 and rgb.std() > 5 else 0
            if time.monotonic() >= warmup_deadline:
                raise RuntimeError('Camera exposure did not settle; last frame mean=%.1f std=%.1f'
                                   % (rgb.mean(), rgb.std()))
            await asyncio.sleep(.2)
        report['exposure_warmup_frames'] = warmup_frames
        last_timestamp = -1
        for index, case in enumerate(cases):
            guard()
            name = case['id']
            if name in done:
                continue
            pose = Pose({'translation': Vector3({'x': case['position_ned_m'][0], 'y': case['position_ned_m'][1],
                                                 'z': case['position_ned_m'][2]}),
                         'rotation': quaternion(case['yaw_deg']), 'frame_id': 'DEFAULT_ID'})
            if not drone.set_pose(pose, reset_kinematics=True):
                raise RuntimeError('SetPose rejected: '+name)
            await asyncio.sleep(protocol['settle_sec'])
            kin = drone.get_ground_truth_kinematics()
            error = float(np.linalg.norm(body_position(kin)-np.array(case['position_ned_m'])))
            if error > protocol['position_tolerance_m']:
                if not drone.set_pose(pose, reset_kinematics=True):
                    raise RuntimeError('SetPose rejected on retry: '+name)
                await asyncio.sleep(protocol['settle_sec'])
                kin = drone.get_ground_truth_kinematics()
                error = float(np.linalg.norm(body_position(kin)-np.array(case['position_ned_m'])))
                if error > protocol['position_tolerance_m']:
                    raise RuntimeError('Teleported pose not reached: '+name)
            deadline = time.monotonic()+3
            while True:
                images = drone.get_images(camera_id, [0, 2])
                rgb_message, depth_message = images[0], images[2]
                stamp = int(rgb_message['time_stamp'])
                skew = abs(stamp-int(depth_message['time_stamp']))/1e9
                if stamp > last_timestamp and skew <= protocol['max_rgb_depth_skew_s']:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('No fresh synchronized RGB-D: '+name)
                await asyncio.sleep(.05)
            last_timestamp = stamp
            rgb = decode_image(rgb_message)[0]
            depth = decode_rpc_depth(depth_message)
            if rgb.shape != (1080, 1920, 3) or depth.shape != (360, 640):
                raise RuntimeError('Unexpected native sensor dimensions')
            brightness, spread = float(rgb.mean()), float(rgb.std())
            if brightness > 250 or spread < 1:
                raise RuntimeError('RGB frame is blank (mean=%.1f std=%.1f); lock Min/Max EV100 in the '
                                   'post-process volume' % (brightness, spread))
            kin = drone.get_ground_truth_kinematics()
            camera_position = np.array([rgb_message['pos_x'], rgb_message['pos_y'], rgb_message['pos_z']], float)
            camera_rotation = Rotation.from_quat([rgb_message['rot_x'], rgb_message['rot_y'],
                                                  rgb_message['rot_z'], rgb_message['rot_w']]).as_matrix()
            masks = renderer.render_camera(camera_position, camera_rotation)
            rgb_path = out/'frames'/(name+'.png')
            depth_path = out/'frames'/(name+'-depth.npy')
            if not cv2.imwrite(str(rgb_path), rgb[:, :, ::-1]):
                raise RuntimeError('Cannot save RGB')
            np.save(depth_path, depth)
            mask_records = {}
            for part, mask in masks.items():
                path = out/'masks'/(name+'-'+part+'.png')
                if not cv2.imwrite(str(path), mask*255):
                    raise RuntimeError('Cannot save mask')
                mask_records[part] = dict(path=str(path.relative_to(out)), pixels=int(mask.sum()),
                                          sha256=digest(path))
            valid = depth[177:183, 317:323]
            valid = valid[np.isfinite(valid) & (valid > .1)]
            report['samples'].append(dict(
                name=name, case=case, rgb=str(rgb_path.relative_to(out)), depth=str(depth_path.relative_to(out)),
                rgb_sha256=digest(rgb_path), depth_sha256=digest(depth_path), masks=mask_records,
                camera_metadata={k: v for k, v in rgb_message.items() if k != 'data'},
                depth_metadata={k: v for k, v in depth_message.items() if k != 'data'},
                rgb_depth_skew_s=skew, center_depth_m=float(np.median(valid)) if valid.size else None,
                teleport_error_m=error, vehicle_kinematics=kin,
                label_pose_source='camera_metadata', camera_position_ned=camera_position.tolist(),
                camera_quaternion=[float(rgb_message[k]) for k in ('rot_w', 'rot_x', 'rot_y', 'rot_z')],
                camera_to_kinematics_lag_s=(int(kin['time_stamp'])-stamp)/1e9))
            if (index+1) % protocol.get('report_every', 50) == 0:
                (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                print('CAPTURE', name, 'of', len(cases), 'tail_px', mask_records['tail']['pixels'], flush=True)
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
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/missions/f14_training_views.json')
    parser.add_argument('--address', default='172.21.192.1')
    parser.add_argument('--count', type=int, help='Override the configured sample count')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume-from', type=Path, help='Resume this collector after a failed preparation')
    options = parser.parse_args(argv)
    options.source_run = options.source_run.resolve()
    if options.resume_from is not None:
        options.resume_from = options.resume_from.resolve()
    out = options.out_dir.resolve()
    protocol = read(options.protocol)
    if options.count is not None:
        protocol['count'] = options.count
    if protocol['count'] < 1:
        raise ValueError('Sample count must be positive')
    source = options.source_run
    setup, profile = read(source/'controller-setup.json'), read(source/'scene-profile.json')
    if setup['target_part'] != protocol['target_part']:
        raise ValueError('Protocol and source run disagree on the target part')
    for name, expected in setup['shared_asset_sha256'].items():
        if digest(Path(setup['shared_reference'])/name) != expected:
            raise ValueError('Frozen reference changed')
    cfg = copy.deepcopy(read(source/'runtime-scene.json'))
    actor = next(a for a in cfg['actors'] if a['name'] == profile['simulator']['drone'])
    actor['origin'] = dict(xyz=' '.join(map(str, protocol['spawn_ned_m'])), **{'rpy-deg': '0 0 -90'})
    sensor = next(s for s in actor['robot-config']['sensors'] if s['id'] == profile['simulator']['camera'])
    if sensor['origin']['rpy-deg'] != '0 -35 0' or sensor.get('gimbal'):
        raise ValueError('Training capture requires the fixed -35 degree camera')
    offset = np.fromstring(sensor['origin']['xyz'], sep=' ')
    pitch = math.radians(-35)
    mount = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                      [-math.sin(pitch), 0, math.cos(pitch)]])
    target_pose = read(source/'evaluation/inspection.json')['target_pose']
    forward, origin = forward_axis(target_pose, profile)
    reference = np.asarray(setup['reference_ned_m'], float)
    rng = np.random.default_rng(protocol['seed'])
    cases = plan_views(protocol, reference, forward, rng)
    route = check_views(cases, protocol, profile, reference, offset, mount)
    summary = dict(cases=cases, route_check=route, planned_frames=len(cases), seed=protocol['seed'])
    if options.validate_only:
        print(json.dumps(summary, indent=2))
        return 0
    if out.exists() and any(out.iterdir()) and options.resume_from is None:
        raise ValueError('Capture directory must be empty')
    (out/'frames').mkdir(parents=True, exist_ok=True)
    (out/'masks').mkdir(parents=True, exist_ok=True)
    configure_client_log(out)
    mesh_path = ROOT/profile['map']['root']/'meshes'/(profile['target']['actor']+'.npz')
    renderer = PartMaskRenderer(mesh_path, profile['parts'], origin, forward, reference,
                                offset, mount, shape=(1080, 1920), horizontal_fov_deg=90.,
                                stride=protocol.get('mask_stride', 2))
    (out/'protocol.json').write_text(json.dumps(dict(
        protocol=protocol, source_run=str(source), reference_ned_m=reference.tolist(),
        forward_ned=forward.tolist(), aircraft_origin_ned=origin.tolist(),
        mesh_sha256=digest(mesh_path), source_sha256={name: digest(ROOT/name) for name in (
            'src/capture/collect_training_views.py', 'src/capture/part_masks.py')},
        **summary), indent=2))
    (out/'runtime-scene.json').write_text(json.dumps(cfg, indent=2))
    report = dict(status='started', source_run=str(source), samples=[], flight_commands_sent=False,
                  scene_reloaded=False, capture_policy='Teleport sampling; no visual controller',
                  protocol_sha256=digest(out/'protocol.json'))
    if options.resume_from is not None:
        report['samples'] = read(options.resume_from/'report.json')['samples']
        report['resumed_from'] = str(options.resume_from)
    asyncio.run(collect(options, protocol, profile, setup, cfg, cases, renderer, out, report))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
