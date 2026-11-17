"""Build nadir templates and target-level splits from saved inspection images; no simulator."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from capture.part_masks import load_mesh, part_membership


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def camera_rotation(message):
    return Rotation.from_quat([message['rot_' + k] for k in ('x', 'y', 'z', 'w')]).as_matrix()


def project(points, message, horizontal_fov_deg):
    position = np.array([message['pos_' + k] for k in ('x', 'y', 'z')])
    local = (np.asarray(points) - position) @ camera_rotation(message)
    if not np.isfinite(local).all() or np.any(local[:, 0] <= 0):
        raise ValueError('Target geometry must be finite and in front of the camera')
    focal = message['width'] / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    uv = local[:, 1:] / local[:, :1] * focal + [message['width'] / 2, message['height'] / 2]
    return uv, focal


def crop_bounds(pixels, width, height, margin):
    if not math.isfinite(margin) or margin < 1:
        raise ValueError('Crop margin must preserve the entire target')
    low, high = pixels.min(0), pixels.max(0)
    if (low < 0).any() or (high > [width - 1, height - 1]).any():
        raise ValueError('Assembly is not fully inside the recorded image')
    center = (low + high) / 2
    half = (high - low).max() * margin / 2
    bounds = np.r_[np.floor(center - half), np.ceil(center + half)].astype(int)
    bounds[:2] = np.maximum(bounds[:2], 0)
    bounds[2:] = np.minimum(bounds[2:], [width, height])
    return bounds.tolist()


def selected_region(points, name, region, origin, forward, reference):
    if name == 'nose':
        return np.linalg.norm(np.asarray(points) - reference, axis=-1) <= region['radius_m']
    return part_membership(points, region, origin, forward, reference)


def template_masks(meshes, primary, message, focal, bounds, name, region, origin, forward, reference):
    """Nearest assembly surface determines self-visibility; external occlusion is unverified."""
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    geometry_ids = {}
    for actor, (vertices, faces) in meshes.items():
        geometry_ids[actor] = scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    x0, y0, x1, y1 = bounds
    u, v = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    directions = np.stack([np.ones(u.shape), (u - message['width'] / 2) / focal,
                           (v - message['height'] / 2) / focal], -1).reshape(-1, 3)
    directions = directions @ camera_rotation(message).T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    position = np.array([message['pos_' + k] for k in ('x', 'y', 'z')])
    rays = np.c_[np.broadcast_to(position, directions.shape), directions].astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    distance = result['t_hit'].numpy()
    hit = np.isfinite(distance) & (distance > 0)
    on_primary = hit & (result['geometry_ids'].numpy() == geometry_ids[primary])
    part = np.zeros(len(hit), bool)
    points = position + distance[on_primary, None] * directions[on_primary]
    part[on_primary] = selected_region(points, name, region, origin, forward, reference)
    return hit.reshape(u.shape).astype(np.uint8) * 255, part.reshape(u.shape).astype(np.uint8) * 255


def resolve_splits(targets, protocols):
    by_index = {t['index']: t for t in targets}
    resolved = {}
    for name, protocol in protocols.items():
        parts = {key: protocol[key] for key in ('train', 'validation', 'test')}
        flat = [i for indices in parts.values() for i in indices]
        if len(flat) != len(set(flat)) or set(flat) != set(by_index):
            raise ValueError(name + ': every target must appear in exactly one partition')
        groups = {key: {by_index[i]['appearance_group'] for i in indices} for key, indices in parts.items()}
        if protocol['disjoint_primary_meshes']:
            if any(groups[a] & groups[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
                raise ValueError(name + ': primary mesh leaks across partitions')
        held_out = protocol.get('held_out_category')
        if held_out and any(by_index[i]['category'] == held_out for i in parts['train'] + parts['validation']):
            raise ValueError(name + ': held-out category is present in fitting data')
        category = protocol.get('evaluation_category')
        if category and any(by_index[i]['category'] != category for i in parts['validation'] + parts['test']):
            raise ValueError(name + ': evaluation contains a different category')
        if not protocol['disjoint_primary_meshes'] and not (groups['validation'] | groups['test']) <= groups['train']:
            raise ValueError(name + ': instance control requires training instances of every evaluation mesh')
        resolved[name] = dict(protocol)
        resolved[name].update({key: [by_index[i]['id'] for i in indices] for key, indices in parts.items()})
        if held_out:
            resolved[name]['test_by_scope'] = {
                'unseen_category': [by_index[i]['id'] for i in parts['test'] if by_index[i]['category'] == held_out],
                'seen_category_unseen_mesh': [by_index[i]['id'] for i in parts['test'] if by_index[i]['category'] != held_out]}
        resolved[name]['category_counts'] = {key: dict(Counter(by_index[i]['category'] for i in indices)) for key, indices in parts.items()}
    return resolved


def validate_catalog(directory):
    catalog = read_json(directory / 'catalog.json')
    for target in catalog['targets']:
        for key in ('template', 'assembly_mask', 'part_mask'):
            p = project_path(target[key]['file'])
            if sha256(p) != target[key]['sha256']:
                raise ValueError('Catalog asset changed: ' + str(p))
    protocols = read_json(project_path(catalog['configuration']))['protocols']
    if resolve_splits(catalog['targets'], protocols) != catalog['splits']:
        raise ValueError('Catalog splits disagree with configuration')
    for name, digest in catalog['input_sha256'].items():
        if sha256(project_path(name)) != digest:
            raise ValueError('Catalog input changed: ' + name)
    return catalog


def overview(targets, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
    fig, axes = plt.subplots(math.ceil(len(targets) / 5), 5, figsize=(16, 11))
    for ax in axes.flat:
        ax.axis('off')
    for ax, target in zip(axes.flat, targets):
        image = cv2.imread(str(project_path(target['template']['file'])))[:, :, ::-1].copy()
        mask = cv2.imread(str(project_path(target['part_mask']['file'])), 0) > 0
        image[mask] = (.6 * image[mask] + .4 * np.array([255, 150, 0])).astype(np.uint8)
        ax.imshow(image)
        ax.set_title(f"{target['index']:02d} {target['label']}\n{target['region']['name']}: {int(mask.sum())} 可见像素", fontsize=9)
    fig.suptitle('俯视模板库（橙色：所选部位的网格自可见区域）', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def build(config_path):
    config = read_json(config_path)
    inspection_dir = project_path(config['inspection_directory'])
    map_dir = project_path(config['map_directory'])
    out, report_dir = project_path(config['output_directory']), project_path(config['report_directory'])
    inspection = read_json(inspection_dir / 'report.json')
    capture = read_json(inspection_dir / 'capture.json')
    source = {t['name']: t for t in read_json(map_dir / 'source/targets.json')}
    results = read_json(project_path(config['approach_summary']))['results']
    frames = {f['target_index']: f for f in capture['frames'] if f['view'] == 'nadir'}
    targets, images, input_hashes = [], [], {}
    for p in [config_path, inspection_dir / 'report.json', inspection_dir / 'capture.json',
              map_dir / 'source/targets.json', project_path(config['approach_summary'])]:
        input_hashes[relative(p)] = sha256(p)
    for index, inspected in enumerate(inspection['targets'], 1):
        result = next(r for r in results if r['index'] == index)
        case = project_path(result['active_control_dir']).parent
        profile_path = case / 'profile.json'
        profile = read_json(profile_path)
        primary = inspected['selected_body']
        if profile['target']['actor'] != primary or frames[index]['body_actor'] != primary:
            raise ValueError('Primary actor disagrees across inputs')
        # Only the already exported, coincident-origin actors belong to this assembly.
        origin_ue = np.array(inspected['listed_ue_m'])
        actors = [primary] + sorted(a for a, record in source.items() if a != primary and
                                   np.linalg.norm(np.array(record['location_cm']) / 100 - origin_ue) < .01)
        meshes = {a: load_mesh(map_dir / 'meshes' / (a + '.npz')) for a in actors}
        vertices = np.concatenate([v for v, _ in meshes.values()])
        frame = frames[index]
        message = frame['image_metadata']
        if not np.allclose(camera_rotation(message)[:, 0], [0, 0, 1], atol=1e-5):
            raise ValueError('Recorded reference is not nadir')
        image_path = inspection_dir / frame['file']
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (message['height'], message['width']):
            raise ValueError('Missing or incorrectly sized reference image')
        pixels, focal = project(vertices, message, capture['camera']['fov_deg'])
        bounds = crop_bounds(pixels, message['width'], message['height'], config['crop_margin'])
        category = config['categories'][inspected['category']]
        part_name = 'roof' if category == 'hangar' else result['part']
        prepared_path = project_path(result.get('coarse_control_dir', result['active_control_dir'])) / 'template.json'
        reference = np.array(read_json(prepared_path)['reference_ned_m'])
        region = profile['parts'][part_name]
        pose = next(t for t in inspected['candidates'] if t['name'] == primary)
        origin = np.array(pose['position_ned_m'])
        q = pose['rotation_wxyz']
        yaw = math.atan2(2 * (q['w'] * q['z'] + q['x'] * q['y']), 1 - 2 * (q['y'] ** 2 + q['z'] ** 2))
        yaw += math.radians(profile['target']['forward_axis_angle_deg'])
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        selected = selected_region(meshes[primary][0], part_name, region, origin, forward, reference)
        if not selected.any():
            raise ValueError('Configured region contains no primary mesh vertices')
        whole, part = template_masks(meshes, primary, message, focal, bounds, part_name, region, origin, forward, reference)
        if not whole.any():
            raise ValueError('Assembly has no visible reference pixels')
        selected_uv, _ = project(meshes[primary][0][selected], message, capture['camera']['fov_deg'])
        x0, y0, x1, y1 = bounds
        target_id = case.name
        directory = out / target_id
        asset_paths = sorted(source[primary]['meshes'])
        group = hashlib.sha256(json.dumps(asset_paths, ensure_ascii=False).encode()).hexdigest()[:16]
        region_keys = ['description', 'radius_m', 'longitudinal_min_m', 'longitudinal_max_m',
                       'lateral_min_m', 'lateral_max_m', 'min_altitude_m', 'max_altitude_m']
        record = {
            'id': target_id, 'index': index, 'label': inspected['label'], 'category': category,
            'description': inspected['description'], 'damaged': inspected['description'] != '完好外观',
            'has_roof_opening': category == 'hangar', 'primary_actor': primary, 'appearance_group': group,
            'actors': [{'name': a, 'label': source[a]['label'], 'mesh_assets': source[a]['meshes'],
                        'mesh': relative(map_dir / 'meshes' / (a + '.npz')),
                        'location_ue_m': (np.array(source[a]['location_cm']) / 100).tolist(),
                        'rotation_wxyz': source[a]['rotation_wxyz'], 'scale': source[a]['scale']} for a in actors],
            'bounds_ned_m': [vertices.min(0).tolist(), vertices.max(0).tolist()],
            'reference_ned_m': reference.tolist(), 'source_image': relative(image_path),
            'camera': dict(message, horizontal_fov_deg=capture['camera']['fov_deg']),
            'crop_xyxy': bounds, 'projected_assembly_xyxy': np.r_[pixels.min(0), pixels.max(0)].tolist(),
            'crop_intrinsics': [[focal, 0, message['width'] / 2 - x0], [0, focal, message['height'] / 2 - y0], [0, 0, 1]],
            'camera_altitude_m': -message['pos_z'], 'gsd_at_world_z0_m_per_pixel': -message['pos_z'] / focal,
            'clearance_above_highest_mesh_m': -message['pos_z'] + float(vertices[:, 2].min()),
            'region': {'name': part_name, 'task': 'roof_entry' if category == 'hangar' else 'part_approach',
                       'geometry': {k: region[k] for k in region_keys if k in region},
                       'membership': 'reference_sphere' if part_name == 'nose' else 'local_bounds_and_reference_sphere',
                       'origin_ned_m': origin.tolist(), 'forward_ne': forward.tolist(),
                       'projected_crop_roi': (np.r_[selected_uv.min(0) - [x0, y0], selected_uv.max(0) - [x0, y0]] / [x1-x0, y1-y0, x1-x0, y1-y0]).tolist(),
                       'visible_pixels': int(np.count_nonzero(part)), 'visible_in_nadir': bool(part.any()),
                       'roof_region_is_free_opening': False},
            'image_quality': {'mean': float(image.mean()), 'std': float(image.std()),
                              'near_white_fraction': float((image.min(2) >= 250).mean())},
            'source_profile': relative(profile_path), 'source_prepared_template': relative(prepared_path)}
        for key, filename, value in [('template', 'template.png', image[y0:y1, x0:x1]),
                                     ('assembly_mask', 'assembly-mask.png', whole), ('part_mask', 'part-mask.png', part)]:
            record[key] = {'file': relative(directory / filename), 'shape_hw': list(value.shape[:2])}
            images.append((directory / filename, value))
        for p in [profile_path, prepared_path, image_path] + [map_dir / 'meshes' / (a + '.npz') for a in actors]:
            if relative(p) not in input_hashes:
                input_hashes[relative(p)] = sha256(p)
        targets.append(record)
        print('TARGET', index, primary, 'assembly', len(actors), 'visible_part_pixels', record['region']['visible_pixels'], flush=True)
    splits = resolve_splits(targets, config['protocols'])
    for path, value in images:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), value, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise OSError('Image write failed: ' + str(path))
    for target in targets:
        for key in ('template', 'assembly_mask', 'part_mask'):
            target[key]['sha256'] = sha256(project_path(target[key]['file']))
    catalog = {
        'configuration': relative(config_path), 'coordinate_frame': 'NED metres; image pixels are right/down',
        'input_sha256': input_hashes, 'targets': targets, 'splits': splits,
        'primary_protocol': config['primary_protocol'],
        'grouping_basis': 'primary source mesh asset paths; shared auxiliary meshes do not establish new primary appearances',
        'label_scope': 'nearest exported assembly surfaces; self-occlusion included, external scene occlusion not depth-verified; roof region mask is not a traversable opening',
        'data_policy': {'unit': 'whole target and all its future frames', 'template_at_test_time': 'provided nadir template is allowed; test search frames and labels cannot fit weights, statistics or thresholds',
                        'validation': 'thresholds and model selection use validation, never test',
                        'development_exposure': 'all targets were used in prior DINO/RGBD development; these splits reserve future model fitting, not untouched assets',
                        'protocol_models': 'fit separate models per protocol; partitions from different protocols must not be pooled'}}
    (out / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    validate_catalog(out)
    report_dir.mkdir(parents=True, exist_ok=True)
    overview(targets, report_dir / 'template-overview.png')
    groups = defaultdict(list)
    for target in targets:
        groups[target['appearance_group']].append(target['id'])
    report = {'status': 'passed', 'simulator_connected': False, 'flight_commands_sent': False,
              'catalog': relative(out / 'catalog.json'), 'target_count': len(targets),
              'assembly_actor_count': sum(len(t['actors']) for t in targets),
              'primary_mesh_group_count': len(groups), 'primary_mesh_groups': list(groups.values()),
              'invisible_nadir_parts': [t['id'] for t in targets if not t['region']['visible_in_nadir']],
              'split_counts': {name: {k: len(p[k]) for k in ('train', 'validation', 'test')} for name, p in splits.items()},
              'full_target_coverage_checked': True, 'input_and_output_hashes_checked': True,
              'external_occlusion_depth_verified': False, 'catalog_sha256': sha256(out / 'catalog.json'),
              'source_sha256': sha256(Path(__file__))}
    (report_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('CATALOG_READY', json.dumps(report, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/datasets/airport_targets.json')
    parser.add_argument('--validate-only', action='store_true', help='Check saved assets, input hashes and split separation; no writes')
    options = parser.parse_args(argv)
    config_path = project_path(options.config)
    if options.validate_only:
        catalog = validate_catalog(project_path(read_json(config_path)['output_directory']))
        print('CATALOG_VALID', len(catalog['targets']), 'targets')
    else:
        build(config_path)


if __name__ == '__main__':
    main()
