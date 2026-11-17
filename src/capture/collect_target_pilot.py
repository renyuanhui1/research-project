"""Catalog-driven RGB-D sampling with a non-physics camera and depth-verified labels."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import socket
from datetime import datetime, timezone

import cv2
import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from projectairsim import ProjectAirSimClient, World, Drone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from common.view_geometry import visible_groups, projected_box, redact_rgb
from common.runtime import configure_client_log
from common.sensors import decode_image
from capture.collect_part_views import decode_rpc_depth
from capture.part_masks import load_mesh
from capture.build_target_catalog import camera_rotation, selected_region, sha256, validate_catalog


def plan_views(catalog, protocol):
    planned = []
    if not set(protocol['target_indices']) <= {t['index'] for t in catalog['targets']}:
        raise ValueError('Unknown target index')
    split = catalog['splits'][catalog['primary_protocol']]
    allowed = set().union(*(set(split[key]) for key in protocol.get('partitions', ['train'])))
    for target in catalog['targets']:
        if target['index'] not in protocol['target_indices']:
            continue
        if target['id'] not in allowed:
            raise ValueError('Target is outside the configured training split/partitions')
        if protocol.get('sampling') == 'stratified_random':
            planned.extend(random_views(target, catalog, protocol))
            continue
        reference = np.asarray(target['reference_ned_m'])
        forward = np.asarray(target['region']['forward_ne'])
        lateral = np.array([-forward[1], forward[0]])
        for distance in protocol['horizontal_distances_m']:
            for bearing in protocol['bearings_deg'][target['category']]:
                angle = math.radians(bearing)
                xy = reference[:2] + distance * (math.cos(angle) * forward + math.sin(angle) * lateral)
                position = np.r_[xy, reference[2] - distance * math.tan(math.radians(protocol['camera_pitch_deg']))]
                yaw = math.degrees(math.atan2(reference[1] - xy[1], reference[0] - xy[0]))
                q = Rotation.from_euler('ZYX', [yaw, -protocol['camera_pitch_deg'], 0], degrees=True).as_quat()
                planned.append({'name': f"t{target['index']:02d}-r{distance:g}-b{bearing:+g}",
                                'target_id': target['id'], 'category': target['category'],
                                'horizontal_distance_m': distance, 'bearing_deg': bearing,
                                'pose': {'frame_id': 'DEFAULT_FRAME',
                                         'translation': dict(zip(('x', 'y', 'z'), position.tolist())),
                                         'rotation': dict(zip(('x', 'y', 'z', 'w'), q.tolist()))}})
    if {p['target_id'] for p in planned} != {t['id'] for t in catalog['targets'] if t['index'] in protocol['target_indices']} or not planned:
        raise ValueError('No complete pilot view plan')
    return planned


def random_views(target, catalog, protocol):
    split = catalog['splits'][catalog['primary_protocol']]
    partition = next(k for k in ('train','validation','test') if target['id'] in split[k])
    later = {'train':['validation','test'], 'validation':['test'], 'test':[]}[partition]
    excluded = {t['appearance_group'] for t in catalog['targets']
                if any(t['id'] in split[k] for k in later)} if protocol['redact_later_partition_appearances'] else set()
    rng = np.random.default_rng(protocol['seed'] + target['index']*100000)
    count = protocol['samples_per_partition_target'][partition]
    strata = np.arange(count) % len(protocol['range_strata_m']);rng.shuffle(strata)
    forward = np.array(target['region']['forward_ne']);lateral = np.array([-forward[1],forward[0]])
    result = []
    for i, stratum in enumerate(strata):
        distance = float(rng.uniform(*protocol['range_strata_m'][stratum]))
        bearing = float(rng.uniform(*protocol['bearing_ranges_deg'][target['category']]))
        heading = float(rng.uniform(*protocol['heading_offsets_deg']))
        reference = np.array(target['reference_ned_m'])
        reference[2] -= float(rng.choice(protocol['reference_height_offsets_m'][target['category']]))
        angle = math.radians(bearing)
        point = reference + np.r_[distance*(math.cos(angle)*forward+math.sin(angle)*lateral),
                                  -distance*math.tan(math.radians(protocol['camera_pitch_deg']))]
        point[2] -= distance*float(rng.uniform(*protocol['vertical_jitter_ratio']))
        if target['category']=='hangar':
            point[2] = min(point[2], target['bounds_ned_m'][0][2]-protocol['minimum_roof_camera_clearance_m'])
        yaw = math.degrees(math.atan2(reference[1]-point[1],reference[0]-point[0]))+heading
        rotation = Rotation.from_euler('ZYX',[yaw,-protocol['camera_pitch_deg'],0],degrees=True)
        groups = visible_groups(catalog['targets'],point,rotation.as_matrix(),protocol['horizontal_fov_deg'],
                                protocol['rgb_wh'][0]/protocol['rgb_wh'][1])
        ignored = [t for t in catalog['targets'] if t['appearance_group'] in excluded or
                   (partition=='train' and protocol['redact_other_training_instances'] and
                    t['appearance_group']==target['appearance_group'] and t['id']!=target['id'])]
        boxes = [projected_box(t['bounds_ned_m'],point,rotation.as_matrix(),protocol['horizontal_fov_deg'],
                               *protocol['depth_wh']) for t in ignored]
        q = rotation.as_quat()
        result.append({'name':f"t{target['index']:02d}-v{i:05d}", 'target_id':target['id'],
                       'category':target['category'],'partition':partition,'range_stratum':int(stratum),
                       'horizontal_distance_m':distance,'bearing_deg':bearing,'heading_offset_deg':heading,
                       'possibly_visible_groups':sorted(groups),
                       'ignored_depth_boxes':[b for b in boxes if b is not None],
                       'pose':{'frame_id':'DEFAULT_FRAME','translation':dict(zip(('x','y','z'),point.tolist())),
                               'rotation':dict(zip(('x','y','z','w'),q.tolist()))}})
    return result


def depth_filter(expected, part_geometry, measured, tolerance):
    """Keep consistent target hits; closer foreground is background, missing/far depth is unknown."""
    measured = np.asarray(measured, np.float32)
    finite = np.isfinite(measured) & (measured > 0)
    projected = np.isfinite(expected) & (expected > 0)
    with np.errstate(invalid='ignore', over='ignore'):
        native = np.where(finite, measured, 0).astype(np.float16)
        finite &= np.isfinite(native)
        native = np.where(finite, native, 0).astype(np.float16)
        upward = np.abs(np.spacing(native)).astype(np.float32)
        downward = native.astype(np.float32)-np.nextafter(native, np.float16(0)).astype(np.float32)
        ulp = np.where(np.isfinite(upward), upward, downward)
    limit = tolerance + ulp
    difference = measured - np.where(projected, expected, 0)
    visible = projected & finite & (np.abs(difference) <= limit)
    occluded = projected & finite & (difference < -limit)
    valid = finite & (~projected | visible | occluded)
    return {'assembly_mask': visible, 'part_mask': visible & part_geometry,
            'valid_mask': valid, 'occluded_mask': occluded, 'geometry_mask': projected}


class LabelRenderer:
    def __init__(self, target):
        self.target = target
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        for actor in target['actors']:
            vertices, faces = load_mesh(project_path(actor['mesh']))
            geometry_id = self.scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
            if actor['name'] == target['primary_actor']:
                self.primary_id = geometry_id

    def render(self, depth, message, fov, tolerance):
        h, w = depth.shape
        focal = w / (2 * math.tan(math.radians(fov) / 2))
        u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
        directions = np.stack([np.ones(u.shape), (u-w/2)/focal, (v-h/2)/focal], -1).reshape(-1, 3)
        directions = directions @ camera_rotation(message).T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        position = np.array([message['pos_' + k] for k in ('x', 'y', 'z')])
        rays = np.c_[np.broadcast_to(position, directions.shape), directions].astype(np.float32)
        result = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
        expected = result['t_hit'].numpy()
        primary = np.isfinite(expected) & (result['geometry_ids'].numpy() == self.primary_id)
        part = np.zeros(len(expected), bool)
        points = position + expected[primary, None] * directions[primary]
        region = self.target['region']
        part[primary] = selected_region(points, region['name'], region['geometry'],
                                        region['origin_ned_m'], region['forward_ne'], self.target['reference_ned_m'])
        masks = depth_filter(expected.reshape(h, w), part.reshape(h, w), depth, tolerance)
        error = np.abs(depth[masks['assembly_mask']] - expected.reshape(h, w)[masks['assembly_mask']])
        return masks, None if not error.size else float(np.median(error))


def write_sample(store, name, rgb, depth, masks, metadata):
    encoded, png = cv2.imencode('.png', rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not encoded:
        raise OSError('Cannot encode RGB')
    group = store.create_group(name)
    group.create_dataset('rgb_png', data=png)
    group.create_dataset('depth_m', data=depth.astype(np.float16), compression='gzip', compression_opts=1)
    for key, mask in masks.items():
        group.create_dataset(key, data=mask.astype(np.uint8), compression='gzip', compression_opts=1)
    group.attrs['metadata_json'] = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    store.flush()


def verify_store(path, report):
    with h5py.File(path, 'r') as store:
        if set(store) != {s['name'] for s in report['samples']}:
            raise ValueError('HDF5 and sample manifest disagree')
        for sample in report['samples']:
            group = store[sample['name']]
            image = cv2.imdecode(group['rgb_png'][()], cv2.IMREAD_COLOR)
            if list(image.shape[:2]) != sample['rgb_shape_hw'] or group['depth_m'].shape != group['assembly_mask'].shape:
                raise ValueError('Image/label dimensions disagree')
            for key, count in sample['mask_pixels'].items():
                if np.count_nonzero(group[key][()]) != count:
                    raise ValueError('Saved mask pixel count changed')


def overview(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples = {}
    for sample in report['samples']:
        key = (sample['target_id'], sample.get('range_stratum', sample['horizontal_distance_m']))
        if key not in examples or sample['mask_pixels']['part_mask'] > examples[key]['mask_pixels']['part_mask']:
            examples[key] = sample
    fig, axes = plt.subplots(math.ceil(len(examples)/4), 4, figsize=(16, 9), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    with h5py.File(out/'samples.h5', 'r') as store:
        for ax, sample in zip(axes.flat, examples.values()):
            group = store[sample['name']]
            image = cv2.imdecode(group['rgb_png'][()], cv2.IMREAD_COLOR)[:, :, ::-1].copy()
            if 'ignored_mask' in group:image = redact_rgb(image,group['ignored_mask'][()])
            for key, color in [('assembly_mask', [0, 255, 0]), ('part_mask', [255, 150, 0]), ('occluded_mask', [255, 0, 0])]:
                mask = cv2.resize(group[key][()], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                image[mask] = (.7*image[mask] + .3*np.array(color)).astype(np.uint8)
            ax.imshow(image);ax.axis('off');ax.set_title(sample['name'], fontsize=10)
    fig.suptitle('RGB-D pilot: green target / orange selected part / red foreground occlusion')
    fig.tight_layout();fig.savefig(out/'overview.png', dpi=120);plt.close(fig)


def collect(protocol, config_path, catalog, cases, out, resume=False):
    cfg = copy.deepcopy(json.loads(project_path(protocol['scene_template']).read_text()))
    actor = cfg['actors'][0]
    actor['origin'] = {'xyz': ' '.join(str(v) for v in cases[0]['pose']['translation'].values()), 'rpy-deg': '0 0 0'}
    sensor = actor['robot-config']['sensors'][0]
    sensor['capture-settings'] = []
    for kind, size in [(0, protocol['rgb_wh']), (2, protocol['depth_wh'])]:
        sensor['capture-settings'].append({'image-type': kind, 'width': size[0], 'height': size[1],
            'fov-degrees': protocol['horizontal_fov_deg'], 'capture-enabled': True, 'streaming-enabled': False,
            'pixels-as-float': kind == 2, 'compress': False, 'target-gamma': 2.5})
    (out/'camera-scene.json').write_text(json.dumps(cfg, indent=2))
    targets = {t['id']: t for t in catalog['targets']}
    renderers = {}
    report = {'status': 'started', 'flight_commands_sent': False, 'sampling': 'non-physics camera poses, not closed-loop flight',
              'protocol': protocol, 'protocol_sha256': sha256(config_path),
              'catalog_sha256': sha256(project_path(protocol['catalog'])), 'samples': [],
              'planned_samples':len(cases), 'capture_target_ids':sorted({c['target_id'] for c in cases})}
    if resume and (out/'report.json').exists():
        previous=json.loads((out/'report.json').read_text())
        if previous['protocol_sha256']!=report['protocol_sha256'] or previous['catalog_sha256']!=report['catalog_sha256']:
            raise ValueError('Resume inputs changed')
        report['samples']=previous['samples']
        report['capture_segments']=previous.get('capture_segments',[])
        if previous.get('error'):report.setdefault('previous_errors',[]).append(previous['error'])
    report.setdefault('capture_segments',[]).append({'start_sample_count':len(report['samples']),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'collector_sha256':sha256(Path(__file__))})
    client = ProjectAirSimClient(protocol['address'])
    world, loaded = None, False
    try:
        for port in (8989,8990):
            with socket.create_connection((protocol['address'],port),timeout=3):pass
        client.connect();client.socket_services.recv_timeout = 15000;client.get_topic_info()
        clocks = [t for t in client.topics if t.endswith('/clock')]
        if len(clocks) != 1:
            raise RuntimeError('Expected one active scene')
        world = World.__new__(World);world.client = client;world.parent_topic = clocks[0].rsplit('/clock', 1)[0]
        if world.list_actors():
            raise RuntimeError('Scene contains actors; camera capture requires an empty actor scene')
        cfg['id'] = world.parent_topic.split('/')[-1]
        for target_id in {c['target_id'] for c in cases}:
            for item in targets[target_id]['actors']:
                pose = world.get_object_pose(item['name']);scale = world.get_object_scale(item['name'])
                actual = np.array([pose['translation'][k] for k in ('x', 'y', 'z')])
                expected = np.array(item['location_ue_m']) * [1, 1, -1]
                actual_q = np.array([pose['rotation'][k] for k in ('w', 'x', 'y', 'z')])
                if not np.allclose(actual, expected, atol=.01, rtol=0) or not np.allclose(scale, item['scale'], atol=.001, rtol=0) or abs(actual_q @ item['rotation_wxyz']) < 1-1e-6:
                    raise RuntimeError('Target transform changed: '+item['name'])
        world.load_scene(cfg, delay_after_load_sec=1);loaded = True
        drone = Drone(client, world, actor['name'])
        last_stamp = -1
        with h5py.File(out/'samples.h5', 'a' if resume else 'w') as store:
            done={s['name'] for s in report['samples']}
            if not done<=set(store):raise ValueError('Reported samples missing from HDF5')
            for name in set(store)-done:
                if name not in {c['name'] for c in cases}:raise ValueError('Unexpected sample in resume store')
                del store[name]
            for case in cases:
                if case['name'] in done:continue
                if not drone.set_pose(case['pose'], reset_kinematics=True):
                    raise RuntimeError('Observer SetPose rejected')
                time.sleep(protocol['settle_sec'])
                stable = 0;deadline = time.monotonic()+120
                while stable < 3:
                    images = drone.get_images(sensor['id'], [0, 2]);rm, dm = images[0], images[2]
                    rgb = decode_image(rm)[0];depth = decode_rpc_depth(dm)
                    stamp = int(rm['time_stamp']);skew = abs(stamp-int(dm['time_stamp']))/1e9
                    pos = np.array([rm['pos_'+k] for k in ('x','y','z')])
                    pose_error = np.linalg.norm(pos-np.array(list(case['pose']['translation'].values())))
                    actual_q = np.array([rm['rot_'+k] for k in ('x','y','z','w')])
                    desired_q = np.array([case['pose']['rotation'][k] for k in ('x','y','z','w')])
                    valid = (stamp > last_stamp and skew <= protocol['max_rgb_depth_skew_s'] and pose_error < .01 and
                             abs(actual_q@desired_q) > 1-1e-6 and rgb.mean() < 220 and rgb.std() > 5)
                    stable = stable+1 if valid else 0;last_stamp = stamp
                    if time.monotonic() > deadline:
                        raise RuntimeError('Fresh RGB-D/pose/exposure did not settle: '+case['name'])
                    time.sleep(.1)
                if rgb.shape[:2] != tuple(reversed(protocol['rgb_wh'])) or depth.shape != tuple(reversed(protocol['depth_wh'])):
                    raise RuntimeError('Sensor size disagrees with protocol')
                if not np.allclose([rm['pos_'+k] for k in ('x','y','z')], [dm['pos_'+k] for k in ('x','y','z')], atol=.01, rtol=0) or not np.allclose(camera_rotation(rm), camera_rotation(dm), atol=1e-4, rtol=0):
                    raise RuntimeError('RGB and depth cameras are not registered')
                target_id = case['target_id']
                if target_id not in renderers:
                    renderers[target_id] = LabelRenderer(targets[target_id])
                masks, residual = renderers[target_id].render(depth, dm, protocol['horizontal_fov_deg'], protocol['depth_tolerance_m'])
                if 'ignored_depth_boxes' in case:
                    ignored = np.zeros(depth.shape,bool)
                    for x0,y0,x1,y1 in case['ignored_depth_boxes']:ignored[y0:y1,x0:x1]=True
                    for key in ('valid_mask','assembly_mask','part_mask','occluded_mask'):masks[key] &= ~ignored
                    masks['ignored_mask'] = ignored
                sample = dict(case, camera_metadata={k:v for k,v in rm.items() if k!='data'},
                              depth_metadata={k:v for k,v in dm.items() if k!='data'}, rgb_depth_skew_s=skew,
                              rgb_shape_hw=list(rgb.shape[:2]), depth_shape_hw=list(depth.shape),
                              mask_pixels={k:int(np.count_nonzero(v)) for k,v in masks.items()},
                              median_consistent_depth_error_m=residual,
                              capture_segment=len(report['capture_segments'])-1)
                write_sample(store, case['name'], rgb, depth, masks, sample)
                report['samples'].append(sample)
                (out/'report.json').write_text(json.dumps(report, indent=2))
                print('CAPTURE',case['name'],'target/part/occluded',sample['mask_pixels']['assembly_mask'],sample['mask_pixels']['part_mask'],sample['mask_pixels']['occluded_mask'],flush=True)
        report['status'] = 'captured_pending_validation'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if loaded:
            try:
                empty = copy.deepcopy(cfg);empty['actors'] = []
                world.load_scene(empty, delay_after_load_sec=.2)
                report['observer_removed'] = not world.list_actors()
            except Exception as error:
                report.update(observer_removed=False, cleanup_error=repr(error))
        client.disconnect()
        (out/'report.json').write_text(json.dumps(report, indent=2))
    if not report.get('observer_removed'):
        raise RuntimeError('Observer cleanup was not confirmed')
    verify_store(out/'samples.h5', report)
    overview(out, report)
    report['storage_verified'] = True
    report['status'] = 'completed'
    report['samples_h5_sha256'] = sha256(out/'samples.h5')
    report['source_sha256'] = {name:sha256(ROOT/name) for name in ['src/capture/collect_target_pilot.py','src/capture/build_target_catalog.py','src/capture/part_masks.py','src/common/view_geometry.py']}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print('PILOT_COMPLETE',len(report['samples']),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/datasets/airport_pilot.json')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue the same frozen plan and verify completed target shards')
    options = parser.parse_args(argv);config_path = project_path(options.protocol)
    protocol = json.loads(config_path.read_text());directory = project_path(protocol['catalog']).parent
    catalog = validate_catalog(directory);cases = plan_views(catalog, protocol)
    out = project_path(protocol['output_directory'])
    if options.validate_only:
        print(json.dumps({'planned_samples':len(cases),'cases':cases},indent=2));return
    if out.exists() and any(out.iterdir()) and not options.resume:
        raise ValueError('Pilot output directory must be empty')
    out.mkdir(parents=True,exist_ok=True);configure_client_log(out)
    (out/'plan.json').write_text(json.dumps({'protocol':protocol,'cases':cases},indent=2))
    if protocol.get('shard_by_target'):
        manifest = {'status':'started','protocol':protocol,'protocol_sha256':sha256(config_path),
                    'catalog_sha256':sha256(project_path(protocol['catalog'])), 'planned_samples':len(cases),'targets':[]}
        (out/'report.json').write_text(json.dumps(manifest,indent=2))
        for target_id in dict.fromkeys(c['target_id'] for c in cases):
            directory = out/target_id;directory.mkdir(exist_ok=options.resume);configure_client_log(directory)
            selected = [c for c in cases if c['target_id']==target_id]
            previous=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else None
            if options.resume and previous and previous['status']=='completed':
                if previous['protocol_sha256']!=sha256(config_path) or previous['samples_h5_sha256']!=sha256(directory/'samples.h5'):
                    raise ValueError('Completed shard changed')
                verify_store(directory/'samples.h5',previous)
            else:
                collect(protocol,config_path,catalog,selected,directory,resume=options.resume)
            r = json.loads((directory/'report.json').read_text())
            manifest['targets'].append({'id':target_id,'directory':target_id,'samples':len(r['samples']),
                                        'status':r['status'],'samples_h5_sha256':r['samples_h5_sha256']})
            (out/'report.json').write_text(json.dumps(manifest,indent=2))
        manifest['status']='completed';(out/'report.json').write_text(json.dumps(manifest,indent=2))
    else:
        collect(protocol, config_path, catalog, cases, out,resume=options.resume)


if __name__ == '__main__':
    main()
