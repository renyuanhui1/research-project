"""Offline cold-start ablations on lossless full frames and a saved endpoint."""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from perception.part_matching import CroppedPartMatcher, refine_local_image

COLD_SCALES = [.75, 1., 1.25, 1.5, 2., 3., 4., 6., 8., 10.]
COLD_ANGLES = [-5., 0., 5.]
METHODS = ['cold', 'center_prior_only', 'scale_prior_only', 'angle_prior_only',
           'saved_history', 'radius_3x', 'dense_scale', 'radius_3x_dense_scale',
           'cold_then_radius_3x']


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    w, x, y, z = np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def state(row):
    return {k: row['refinement'][k] for k in ('point_px', 'scale', 'angle_deg')}


def load_samples(run, part):
    rows = read_json(run/'control-signals.json')
    poses = read_json(run/'poses.json')
    samples = []
    with np.load(run/'validation-frames.npz') as data:
        samples.append(dict(name='start', rgb=data['start_rgb'].copy(), seed=None,
                            depth_m=rows[0]['depth_m'], pose=None,
                            bbox_normalized=data[part+'_bbox'].tolist(),
                            label_kind='stored_preflight_bbox', source='validation-frames.npz'))
    for index, row in enumerate(rows):
        path = run/'viz_dump'/('step_%04d.npz' % row['step'])
        with np.load(path) as data:
            if 'crop_rgb' not in data.files:
                continue
            rgb = data['crop_rgb']
            # The sensor resolution is checked by main before using these samples.
            if list(data['crop_xyxy']) != [0, 0, 1920, 1080] or rgb.shape != (1080, 1920, 3):
                continue
            prior = next((r for r in reversed(rows[:index]) if r.get('refinement', {}).get('accepted')), None)
            samples.append(dict(name='step_%04d' % row['step'], rgb=rgb.copy(),
                                seed=state(prior) if prior else None, depth_m=row['depth_m'],
                                pose=poses[index], label_kind='pose_derived_ray_approximate',
                                source=str(path.relative_to(run)), rgb_timestamp=row['rgb_timestamp']))
    inspection = read_json(run/'evaluation/inspection.json')
    bgr = cv2.imread(str(run/'evaluation/final-part.png'))
    if bgr is None:
        raise ValueError('Missing saved endpoint image')
    prior = next(row for row in reversed(rows) if row.get('refinement', {}).get('accepted'))
    samples.append(dict(name='endpoint', rgb=bgr[:, :, ::-1].copy(), seed=state(prior),
                        depth_m=rows[-1]['depth_m'], pose=None,
                        label_kind='camera_metadata_ray', source='evaluation/final-part.png',
                        camera_metadata=inspection['camera_metadata']))
    return samples, inspection


class Geometry:
    """Independent labels; never supplies matching candidates or search parameters."""
    def __init__(self, profile, setup, run, inspection):
        self.part = profile['parts'][setup['target_part']]
        self.ref = np.array(read_json(run/'template.json')['reference_ned_m'])
        pose = inspection['target_pose']
        self.origin = np.array([pose['translation'][k] for k in ('x', 'y', 'z')])
        q = pose['rotation']
        yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
        yaw += math.radians(profile['target']['forward_axis_angle_deg'])
        self.forward = np.array([math.cos(yaw), math.sin(yaw)])
        self.lateral = np.array([-self.forward[1], self.forward[0]])
        self.mesh_path = ROOT/profile['map']['root']/'meshes'/(profile['target']['actor']+'.npz')
        with np.load(self.mesh_path) as mesh:
            self.vertices = mesh['vertices_enu'][:, [1, 0, 2]].copy()
            self.vertices[:, 2] *= -1
            faces = mesh['triangles'].astype(np.uint32)
        self.selected = self.vertices[self.in_part(self.vertices)]
        if not len(self.selected):
            raise ValueError('No selected part vertices')
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.vertices.astype(np.float32)), o3d.core.Tensor(faces))
        cfg = read_json(run/'runtime-scene.json')
        sensor = next(s for a in cfg['actors'] for s in a['robot-config']['sensors']
                      if s['id'] == profile['simulator']['camera'])
        settings = next(s for s in sensor['capture-settings'] if s['image-type'] == 0)
        if (settings['width'], settings['height'], settings['fov-degrees']) != (1920, 1080, 90):
            raise ValueError('Study requires the recorded 1920x1080, 90 degree camera')
        if sensor.get('gimbal') or sensor['origin']['rpy-deg'] != '0 -35 0':
            raise ValueError('Approximate intermediate labels require the fixed -35 degree camera')
        self.offset = np.fromstring(sensor['origin']['xyz'], sep=' ')
        pitch = math.radians(-35)
        self.mount = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                               [-math.sin(pitch), 0, math.cos(pitch)]])

    def in_part(self, points):
        points = np.atleast_2d(points)
        p = self.part
        long = (points[:, :2]-self.origin[:2])@self.forward
        side = (points[:, :2]-self.origin[:2])@self.lateral
        valid = ((np.linalg.norm(points-self.ref, axis=1) <= p['radius_m']) &
                 (long > p['longitudinal_min_m']) & (long < p['longitudinal_max_m']) &
                 (side >= p['lateral_min_m']) & (-points[:, 2] > p['min_altitude_m']))
        if 'lateral_max_m' in p:
            valid &= side <= p['lateral_max_m']
        if 'max_altitude_m' in p:
            valid &= -points[:, 2] < p['max_altitude_m']
        return valid

    def label(self, sample):
        h, w = sample['rgb'].shape[:2]
        if sample['label_kind'] == 'stored_preflight_bbox':
            sample['bbox_px'] = ((np.array(sample['bbox_normalized'])+1)*[w/2, h/2, w/2, h/2]).tolist()
            return
        if sample['pose'] is not None:
            pose = sample['pose']
            body = rotation([pose['orientation'][k] for k in ('w', 'x', 'y', 'z')])
            sample['camera_position'] = np.array([pose['position'][k] for k in ('x', 'y', 'z')])+body@self.offset
            sample['camera_rotation'] = body@self.mount
        else:
            meta = sample['camera_metadata']
            sample['camera_position'] = np.array([meta['pos_'+k] for k in ('x', 'y', 'z')])
            sample['camera_rotation'] = rotation([meta['rot_'+k] for k in ('w', 'x', 'y', 'z')])
        points = (self.selected-sample['camera_position'])@sample['camera_rotation']
        if np.any(points[:, 0] <= 0):
            raise ValueError('Part crosses the camera plane')
        uv = np.c_[w/2+w/2*points[:, 1]/points[:, 0], h/2+w/2*points[:, 2]/points[:, 0]]
        sample['bbox_px'] = np.r_[uv.min(0), uv.max(0)].tolist()

    def hits(self, sample, point):
        if sample['label_kind'] == 'stored_preflight_bbox':
            x0, y0, x1, y1 = sample['bbox_px']
            return bool(x0 <= point[0] <= x1 and y0 <= point[1] <= y1)
        h, w = sample['rgb'].shape[:2]
        ray = sample['camera_rotation']@np.array([1., (point[0]-w/2)/(w/2), (point[1]-h/2)/(w/2)])
        ray /= np.linalg.norm(ray)
        origin = sample['camera_position']
        distance = float(self.scene.cast_rays(o3d.core.Tensor(np.r_[origin, ray][None].astype(np.float32)))['t_hit'].numpy()[0])
        return bool(np.isfinite(distance) and self.in_part(origin+distance*ray)[0])


def ablation_options(method, seed):
    dense = np.geomspace(.5, 12., 49).tolist()
    if method.endswith('prior_only') or method == 'saved_history':
        if seed is None:
            return None
    if method == 'center_prior_only':
        return dict(previous=seed, search_scales=COLD_SCALES, search_angles=COLD_ANGLES)
    if method == 'scale_prior_only':
        return dict(search_scales=[seed['scale']*s for s in (.9, 1., 1.1)])
    if method == 'angle_prior_only':
        return dict(search_angles=[seed['angle_deg']+a for a in (-3., 0., 3.)])
    if method == 'saved_history':
        return dict(previous=seed)
    return dict(radius_multiplier=3. if method in ('radius_3x', 'radius_3x_dense_scale') else 1.,
                search_scales=dense if 'dense_scale' in method else None)


def run_methods(sample, rgb, condition, fp, geometry, methods=None):
    fp.refinement_state.clear()
    begin = time.perf_counter()
    fp.query(rgb)
    if fp.dev == 'cuda':
        torch.cuda.synchronize()
    perception_s = time.perf_counter()-begin
    diagnostic = fp.last_diagnostics
    box = diagnostic['crop_xyxy']
    local = diagnostic.get(fp.selected_part, {})
    refinement = local.get('refinement', {})
    candidate = refinement.get('coarse_candidate_px')
    reference = fp.local_references[fp.selected_part]
    results = []
    for method in METHODS if methods is None else methods:
        options = ablation_options(method, sample['seed'])
        if options is None:
            continue
        begin = time.perf_counter()
        result = (refine_local_image(rgb, reference['rgb'], reference['anchor'], candidate, box, **options)
                  if candidate is not None else dict(accepted=False, reason='no_coarse_candidate'))
        result = dict(result)
        if method == 'cold_then_radius_3x':
            initial = dict(result)
            if candidate is not None and not initial['accepted']:
                result = refine_local_image(rgb, reference['rgb'], reference['anchor'], candidate, box,
                                            radius_multiplier=3.)
            result = dict(result, initial_attempt=initial,
                          search_stage='initial' if initial['accepted'] else 'expanded')
        result.update(sample=sample['name'], condition=condition, method=method,
                      label_kind=sample['label_kind'], depth_m=sample['depth_m'],
                      elapsed_s=time.perf_counter()-begin, candidate_pipeline_s=perception_s,
                      crop_xyxy=box, candidate_px=candidate,
                      whole_peak=diagnostic['whole_peak'],
                      part_dino_peak=float(np.max(local['local_similarity'])) if 'local_similarity' in local else None)
        if candidate is not None:
            result['base_radius_px'] = ((np.array(box[2:])-box[:2])/32*1.5).tolist()
        result['hits_label'] = geometry.hits(sample, result['point_px']) if 'point_px' in result else False
        results.append(result)
    return results


def summarize(records):
    summaries = []
    for method in METHODS:
        positive = [r for r in records if r['method'] == method and r['condition'] == 'original']
        negative = [r for r in records if r['method'] == method and r['condition'] == 'projected_box_erased']
        if not positive:
            continue
        summaries.append(dict(method=method, samples=len(positive),
                              accepted=sum(r['accepted'] for r in positive),
                              accepted_and_hits_label=sum(r['accepted'] and r['hits_label'] for r in positive),
                              negative_samples=len(negative), synthetic_false_accepts=sum(r['accepted'] for r in negative),
                              median_registration_ms=1000*float(np.median([r['elapsed_s'] for r in positive])),
                              max_registration_ms=1000*max(r['elapsed_s'] for r in positive)))
    return summaries


def render(samples, records, out, methods=None):
    selected = ([samples[0], samples[1], samples[-2], samples[-1]] if methods is None
                else [samples[i] for i in np.linspace(0, len(samples)-1, 4).round().astype(int)])
    methods = methods or ['cold', 'saved_history', 'radius_3x', 'cold_then_radius_3x']
    fig, axes = plt.subplots(4, len(methods), figsize=(4*len(methods), 14), squeeze=False)
    for i, sample in enumerate(selected):
        x0, y0, x1, y1 = sample['bbox_px']
        for j, method in enumerate(methods):
            ax = axes[i, j]
            rgb = sample['rgb']
            ax.imshow(rgb)
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='yellow', lw=1))
            record = next((r for r in records if r['sample'] == sample['name'] and r['condition'] == 'original' and r['method'] == method), None)
            title = method+' / no prior'
            if record and 'point_px' in record:
                ax.scatter(*record['candidate_px'], c='cyan', marker='+', s=90)
                ax.scatter(*record['point_px'], c='lime' if record['accepted'] else 'red', marker='+', s=100)
                title = '%s / %.3f / %s' % (method, record['score'], 'accept' if record['accepted'] else 'reject')
            ax.set(xlim=(max(0, x0-150), min(rgb.shape[1], x1+150)),
                   ylim=(min(rgb.shape[0], y1+120), max(0, y0-120)), title=title)
            ax.axis('off')
            if j == 0:
                ax.text(0, -.06, sample['name']+' / '+sample['label_kind'], transform=ax.transAxes, fontsize=9)
    fig.suptitle('Same fixed thresholds: cyan DINO / green accepted / red rejected / yellow evaluation region')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, sample in zip(axes.flat, selected):
        rgb = sample['rgb'].copy()
        x0, y0, x1, y1 = sample['erase_xyxy']
        rgb[y0:y1, x0:x1] = np.median(rgb, axis=(0, 1)).astype(np.uint8)
        ax.imshow(rgb)
        ax.set(xlim=(max(0, x0-100), min(rgb.shape[1], x1+100)),
               ylim=(min(rgb.shape[0], y1+100), max(0, y0-100)), title=sample['name'])
        ax.axis('off')
    fig.suptitle('Synthetic projected-box deletion; DINO candidates recomputed')
    fig.tight_layout()
    fig.savefig(out/'negative-examples.png', dpi=140)
    plt.close(fig)


def evaluate_capture(options, run, out, fp, setup, profile):
    """Evaluate a new collection under the protocol frozen before its first image."""
    capture = options.capture_dir.resolve()
    collected = read_json(capture/'report.json')
    frozen = read_json(capture/'protocol.json')
    protocol = frozen['protocol']
    if collected['status'] != 'completed' or len(collected['samples']) != frozen['planned_frames']:
        raise ValueError('Collection incomplete; preserve it as an interrupted experiment')
    if collected['protocol_sha256'] != sha256(capture/'protocol.json'):
        raise ValueError('Capture protocol changed')
    if Path(frozen['source_run']).resolve() != run or frozen['reference_sha256'] != setup['shared_asset_sha256']:
        raise ValueError('Capture and evaluation references disagree')
    for name, expected in frozen['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Frozen source changed after capture: '+name)
    methods = protocol['methods']
    if methods != ['cold', 'radius_3x', 'cold_then_radius_3x'] or (protocol['min_score'], protocol['min_margin'], protocol['expanded_radius_multiplier']) != (.85, .15, 3):
        raise ValueError('Unsupported frozen comparison')
    samples = []
    for item in collected['samples']:
        path = capture/item['rgb']
        if sha256(path) != item['rgb_sha256'] or sha256(capture/item['depth']) != item['depth_sha256']:
            raise ValueError('Captured RGB-D changed: '+item['name'])
        bgr = cv2.imread(str(path))
        if bgr is None or bgr.shape != (1080, 1920, 3):
            raise ValueError('Invalid native RGB')
        samples.append(dict(name=item['name'], rgb=bgr[:, :, ::-1].copy(), seed=None, pose=None,
                            depth_m=item['center_depth_m'], source=str(path),
                            label_kind='camera_metadata_ray', camera_metadata=item['camera_metadata'],
                            case=item['case']))
    geometry = Geometry(profile, setup, run, collected['inspection'])
    records = []
    for sample in samples:
        geometry.label(sample)
        records.extend(run_methods(sample, sample['rgb'], 'original', fp, geometry, methods))
        box = np.array(sample['bbox_px'])
        pad = max(12, .1*max(box[2:]-box[:2]))
        low = np.maximum(0, np.floor(box[:2]-pad).astype(int))
        high = np.minimum([1920, 1080], np.ceil(box[2:]+pad).astype(int))
        if np.any(high <= low):
            raise ValueError('Projected target outside the image')
        rgb = sample['rgb'].copy()
        rgb[low[1]:high[1], low[0]:high[0]] = np.median(rgb, axis=(0, 1)).astype(np.uint8)
        sample['erase_xyxy'] = np.r_[low, high].tolist()
        records.extend(run_methods(sample, rgb, 'projected_box_erased', fp, geometry, methods))
        (out/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
        positive = [r for r in records if r['sample'] == sample['name'] and r['condition'] == 'original']
        print('HELDOUT', sample['name'], [(r['method'], r['accepted'], r['hits_label']) for r in positive], flush=True)
    case_results = []
    for case in frozen['cases']:
        names = {s['name'] for s in samples if s['case']['id'] == case['id']}
        for method in methods:
            rows = [r for r in records if r['sample'] in names and r['method'] == method and r['condition'] == 'original']
            case_results.append(dict(case=case, method=method, frames=len(rows),
                                     accepted=sum(r['accepted'] for r in rows),
                                     verified=sum(r['accepted'] and r['hits_label'] for r in rows),
                                     all_frames_verified=all(r['accepted'] and r['hits_label'] for r in rows)))
    summary = summarize(records)
    for item in summary:
        item['accepted_but_wrong_part'] = item['accepted']-item['accepted_and_hits_label']
        item['conditions_all_frames_verified'] = sum(c['all_frames_verified'] for c in case_results if c['method'] == item['method'])
        item['conditions'] = len(frozen['cases'])
    manifests = [{k: v for k, v in s.items() if k not in ('rgb', 'camera_position', 'camera_rotation', 'pose')}
                 for s in samples]
    report = dict(status='completed', source_run=str(run), capture_dir=str(capture),
                  flight_commands_sent=False, evaluation='Frozen methods on newly collected simulation viewpoints',
                  capture_protocol_sha256=collected['protocol_sha256'], frozen_protocol=frozen,
                  samples=manifests, summary=summary, condition_results=case_results,
                  summary_by_planned_range={str(distance): summarize([r for r in records if any(
                      s['name'] == r['sample'] and s['case']['horizontal_range_m'] == distance for s in samples)])
                                           for distance in protocol['horizontal_ranges_m']},
                  source_sha256=frozen['source_sha256'], reference_sha256=frozen['reference_sha256'],
                  environment=dict(python=platform.python_version(), torch=torch.__version__, opencv=cv2.__version__,
                                   device=options.device, gpu=torch.cuda.get_device_name() if options.device == 'cuda' else None),
                  limitations=['New viewpoints in the same static scene and aircraft; not cross-aircraft generalization.',
                               'Three frames at each of eighteen conditions are correlated; report condition counts separately.',
                               'Matching starts without history on every image; no online loss-and-recovery flight was run.',
                               'Synthetic projected-box deletion is not real occlusion.',
                               'Depth values are center-pixel neighborhoods, not independent target-surface distance.',
                               'Registration timings omit DINO encoding and are not flight frame-rate measurements.'])
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    render(samples, records, out, methods)
    # Show every viewpoint once, including failures, using its first saved frame.
    fig, axes = plt.subplots(6, 3, figsize=(15, 22))
    for ax, case in zip(axes.flat, frozen['cases']):
        sample = next(s for s in samples if s['case']['id'] == case['id'])
        record = next(r for r in records if r['sample'] == sample['name'] and r['condition'] == 'original' and r['method'] == 'cold_then_radius_3x')
        ax.imshow(sample['rgb'])
        x0, y0, x1, y1 = sample['bbox_px']
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='yellow'))
        if 'point_px' in record:
            ax.scatter(*record['candidate_px'], color='cyan', marker='+', s=80)
            ax.scatter(*record['point_px'], color='lime' if record['accepted'] and record['hits_label'] else 'red', marker='+', s=80)
        ax.set(xlim=(max(0, x0-160), min(1920, x1+160)), ylim=(min(1080, y1+120), max(0, y0-120)),
               title=case['id']+' / '+('verified' if record['accepted'] and record['hits_label'] else 'not verified'))
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out/'view-overview.png', dpi=120)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--capture-dir', type=Path, help='New collection from collect_part_views.py under a frozen protocol')
    options = parser.parse_args(argv)
    run, out = options.run_dir.resolve(), options.out_dir.resolve()
    if run == out or run in out.parents:
        raise ValueError('Use a separate study directory outside the source run')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists() or (out/'records.json').exists():
        raise ValueError('Study already exists; use a fresh output directory')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    setup = read_json(run/'controller-setup.json')
    if setup['target_part'] != 'tail':
        raise ValueError('This study evaluates a configured vertical fin')
    args = SimpleNamespace(**read_json(run/'report.json')['controller_args'])
    if not getattr(args, 'thin_roi_refinement', False):
        raise ValueError('Source run must use thin ROI refinement')
    reference = Path(setup['shared_reference'])
    for name, expected in setup['shared_asset_sha256'].items():
        if sha256(reference/name) != expected:
            raise ValueError('Reference asset changed: '+name)
    profile = read_json(run/'scene-profile.json')
    meta = read_json(reference/'template.json')
    fp = CroppedPartMatcher(args, options.device, {k: v['template_roi'] for k, v in meta['parts'].items()}, setup['target_part'])
    if options.capture_dir is not None:
        return evaluate_capture(options, run, out, fp, setup, profile)
    samples, inspection = load_samples(run, setup['target_part'])
    if len(samples) <= 2:
        raise ValueError('No lossless full-frame intermediate data')
    geometry = Geometry(profile, setup, run, inspection)
    records, manifests = [], []
    for sample in samples:
        geometry.label(sample)
        results = run_methods(sample, sample['rgb'], 'original', fp, geometry)
        records.extend(results)
        # Synthetic deletion, with DINO candidates recomputed on the altered image.
        rgb = sample['rgb'].copy()
        box = np.array(sample['bbox_px'])
        pad = max(12, .1*max(box[2:]-box[:2]))
        low = np.maximum(0, np.floor(box[:2]-pad).astype(int))
        high = np.minimum([rgb.shape[1], rgb.shape[0]], np.ceil(box[2:]+pad).astype(int))
        if np.any(high <= low):
            raise ValueError('Projected target is outside the image')
        rgb[low[1]:high[1], low[0]:high[0]] = np.median(rgb, axis=(0, 1)).astype(np.uint8)
        records.extend(run_methods(sample, rgb, 'projected_box_erased', fp, geometry))
        manifest = {k: v for k, v in sample.items() if k not in ('rgb', 'camera_rotation', 'camera_position', 'pose')}
        manifest.update(source_sha256=sha256(run/sample['source']), erase_xyxy=np.r_[low, high].tolist())
        sample['erase_xyxy'] = manifest['erase_xyxy']
        manifests.append(manifest)
        (out/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
        cold = next(r for r in results if r['method'] == 'cold')
        print(sample['name'], 'depth', round(sample['depth_m'], 2), 'cold', cold.get('score'), cold['accepted'], flush=True)
    regression = None
    if (out/'baseline-before.json').exists():
        baseline = read_json(out/'baseline-before.json')['endpoint']
        checks = []
        for old, method in [('cold', 'cold'), ('tracking', 'saved_history')]:
            current = next(r for r in records if r['sample'] == 'endpoint' and r['condition'] == 'original' and r['method'] == method)
            checks.append(dict(method=method, old_fields_identical=all(current[k] == value for k, value in baseline[old].items())))
        regression = dict(checks=checks, passed=all(c['old_fields_identical'] for c in checks))
        if not regression['passed']:
            raise AssertionError('Default endpoint regression changed')
    groups = {'near_depth_lt12m': lambda r: r['depth_m'] < 12,
              'middle_depth_12_to35m': lambda r: 12 <= r['depth_m'] < 35,
              'far_depth_ge35m': lambda r: r['depth_m'] >= 35}
    failed = []
    widened_regressions = []
    for sample in samples:
        positive = {r['method']: r for r in records if r['sample'] == sample['name'] and r['condition'] == 'original'}
        cold = positive['cold']
        if not cold['accepted'] and 'saved_history' in positive:
            history = positive['saved_history']
            delta = np.array(history['point_px'])-cold['candidate_px']
            failed.append(dict(sample=sample['name'], delta_history_minus_coarse_px=delta.tolist(),
                               base_radius_px=cold['base_radius_px'],
                               history_point_outside_cold_radius=bool(np.any(np.abs(delta) > np.array(cold['base_radius_px'])+1)),
                               center_only_accepted=positive['center_prior_only']['accepted']))
        wide = positive['radius_3x']
        if cold['accepted'] and not wide['accepted']:
            widened_regressions.append(dict(sample=sample['name'], score=wide['score'], margin=wide['margin']))
    report = dict(status='completed', source_run=str(run), flight_commands_sent=False,
                  source_assets_modified=False, online_policy_modified=False,
                  samples=manifests, methods=METHODS, summary=summarize(records),
                  summary_by_depth={name: summarize([r for r in records if select(r)]) for name, select in groups.items()},
                  summary_on_history_available_samples=summarize([r for r in records if r['sample'] != 'start']),
                  analysis=dict(cold_failures=failed, cold_acceptances_lost_by_unconditional_expansion=widened_regressions),
                  posthoc_method=dict(name='cold_then_radius_3x',
                                      selection='Added after inspecting the first eight methods on these same samples; development validation only',
                                      screening_summary='screening.json' if (out/'screening.json').exists() else None),
                  parameters=dict(min_score=.85, min_margin=.15, cold_scales=COLD_SCALES,
                                  cold_angles=COLD_ANGLES, dense_scales=np.geomspace(.5, 12., 49).tolist(),
                                  expanded_radius_multiplier=3.),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   opencv=cv2.__version__, device=options.device,
                                   gpu=torch.cuda.get_device_name() if options.device == 'cuda' else None,
                                   torch_threads=2, opencv_threads=1),
                  default_endpoint_regression=regression,
                  reference_sha256=setup['shared_asset_sha256'],
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                 [Path(__file__).resolve(), ROOT/'src/perception/part_matching.py', geometry.mesh_path]},
                  limitations=['Same static development scene and original reference; no held-out data or new flights.',
                               'Selection includes every saved full native RGB crop, plus start and endpoint; uneven distance coverage.',
                               'Depth is saved filtered control depth, not independent distance ground truth.',
                               'Intermediate ray labels use asynchronous body poses plus fixed camera extrinsics; approximate only.',
                               'Start uses saved projected bbox; only endpoint uses exact saved camera metadata.',
                               'History ablations use the previous accepted recorded state; unavailable for the initial image.',
                               'Projected box deletion is synthetic; intermediate deletion boxes inherit pose uncertainty.',
                               'Timings are one observation per sample/method, registration only; not a flight frame-rate benchmark.'])
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['summary'][0]))
        writer.writeheader()
        writer.writerows(report['summary'])
    render(samples, records, out)
    print(json.dumps(report['summary'], indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
