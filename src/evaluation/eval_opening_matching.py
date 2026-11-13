"""离线检查洞口DINO匹配，比较整帧与由整帧结果定位的高清裁剪；不连接仿真。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from common.profiles import load_profile
from perception.fingerprint import Fingerprint
from perception.opening_matching import depth_constrained_match




def read_rgb(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pixel_center(cx, cy, box):
    x0, y0, x1, y1 = box
    return [x0 + (cx + 1) * .5 * (x1 - x0 - 1),
            y0 + (cy + 1) * .5 * (y1 - y0 - 1)]


def candidate_box(center, shape, fraction):
    h, w = shape[:2]
    cw, ch = max(2, round(w * fraction)), max(2, round(h * fraction))
    x0 = int(np.clip(round(center[0] - cw / 2), 0, w - cw))
    y0 = int(np.clip(round(center[1] - ch / 2), 0, h - ch))
    return [x0, y0, x0 + cw, y0 + ch]


def panel(rgb, sim, center, title, temperature, crop_box=None, polygon=None):
    h, w = rgb.shape[:2]
    grid = int(round(np.sqrt(sim.size)))
    weights = np.exp((sim - sim.max()) / temperature)
    weights /= weights.sum()
    heat = cv2.resize(np.sqrt(weights / weights.max()).reshape(grid, grid), (w, h))
    colored = rgb.astype(np.float32)
    alpha = .45 * heat[..., None]
    colored = colored * (1 - alpha) + np.array([255, 0, 0]) * alpha
    view = cv2.cvtColor(colored.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if polygon is not None:
        cv2.polylines(view, [np.asarray(polygon, np.int32)], True, (0, 255, 255), 3)
    if crop_box is not None:
        cv2.rectangle(view, tuple(crop_box[:2]), tuple(crop_box[2:]), (255, 180, 0), 3)
    cv2.drawMarker(view, tuple(round(v) for v in center), (0, 255, 0), cv2.MARKER_CROSS, 28, 3)
    view = cv2.resize(view, (640, 360))
    view = cv2.copyMakeBorder(view, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
    cv2.putText(view, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/missions/hangar_opening_validation.json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, help='Append a completed development capture; frames are not held-out tests')
    parser.add_argument('--depth-geometry', action='store_true', help='Compare depth-proposed opening crops using spatial DINO features; diagnostics only')
    options = parser.parse_args(argv)
    config = json.loads(project_path(options.config).read_text())
    out = options.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Use an empty output directory: ' + str(out))
    fraction = config['refine_crop_fraction']
    if not 0 < fraction <= 1 or config['image_size'] % 14 or config['softmax_temperature']<=0:
        raise ValueError('Invalid crop fraction or DINO image size')
    source = read_rgb(project_path(config['reference_image']))
    x0, y0, x1, y1 = config['template_crop_xyxy']
    if not 0 <= x0 < x1 <= source.shape[1] or not 0 <= y0 < y1 <= source.shape[0]:
        raise ValueError('Template crop outside reference image')
    # 输入全部先读取，避免输出途中才发现素材缺失。
    frame_specs = list(config['frames'])
    if options.capture_dir:
        folder = options.capture_dir.expanduser().resolve()
        capture_report = json.loads((folder/'report.json').read_text())
        if capture_report['status']!='completed':
            raise RuntimeError('Development capture is incomplete')
        for frame in capture_report['frames']:
            frame_specs.append({'name':frame['name'], 'image':str(folder/frame['image']),
                                'depth':str(folder/frame['depth']),
                                'role':'development_capture_not_held_out_test'})
    frames = [(frame, read_rgb(project_path(frame['image']))) for frame in frame_specs]
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'opening-template.png'), source[y0:y1, x0:x1, ::-1])
    with h5py.File(out / 'reference-stats.h5', 'w') as h:
        h.create_dataset('rgb', data=source[None], compression='gzip')
        h.attrs['source'] = config['reference_image']
        h.attrs['role'] = 'Normalization from reference frame only; no query frames'
    profile = load_profile()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    args = SimpleNamespace(template=out / 'opening-template.png', stats_episode=out / 'reference-stats.h5',
        repo_dir=project_path(profile['assets']['dino_repo']), dino_weights=project_path(profile['assets']['dino_weights']),
        image_size=config['image_size'], template_roi=None, target_softmax_temp=config['softmax_temperature'],
        target_mass_thresh=.2, target_mass_sharpness=20.)
    fp = Fingerprint(args, device)
    template_rgb=read_rgb(out/'opening-template.png')
    with torch.no_grad():
        reference_tokens=fp._encode(template_rgb[None])[0]
        reference_tokens=torch.nn.functional.normalize((reference_tokens-fp.zm.cpu())/fp.zs.cpu(),dim=-1)
    rows, montage = [], []
    geometry_montage=[]
    for frame, rgb in frames:
        h, w = rgb.shape[:2]
        cx, cy, mass, peak, sim = fp.query(rgb)
        full_center = pixel_center(cx, cy, [0, 0, w, h])
        box = candidate_box(full_center, rgb.shape, fraction)
        bx0, by0, bx1, by1 = box
        crop = np.ascontiguousarray(rgb[by0:by1, bx0:bx1])
        rx, ry, rmass, rpeak, rsim = fp.query(crop)
        refined_center = pixel_center(rx, ry, box)
        polygon = config['opening_polygon_xy'] if frame['role'] == 'reference_self_check' else None
        row = dict(frame, full_center_px=full_center, full_peak=peak, full_mass=mass,
                   refinement_box_xyxy=box, refined_center_px=refined_center,
                   refined_peak=rpeak, refined_mass=rmass)
        if polygon is not None:
            outline = np.asarray(polygon, np.float32)
            row['reference_full_center_inside_outline'] = cv2.pointPolygonTest(outline, tuple(full_center), False) >= 0
            row['reference_refined_center_inside_outline'] = cv2.pointPolygonTest(outline, tuple(refined_center), False) >= 0
        else:
            row['ground_truth_annotation_available'] = False
        if options.depth_geometry:
            depth=np.load(project_path(frame['depth']))
            diagnostic,geometry_view=depth_constrained_match(fp,rgb,depth,reference_tokens,config.get('min_spatial_dino_score',.4))
            row['depth_constrained_dino']=diagnostic
            cv2.imwrite(str(out/(frame['name']+'-geometry.png')),geometry_view)
            geometry_montage.append(geometry_view)
            print('GEOMETRY',frame['name'],diagnostic['status'],
                  None if diagnostic['selected'] is None else diagnostic['selected']['candidate_label'],flush=True)
        left = panel(rgb, sim, full_center, frame['name'] + ' | full frame', config['softmax_temperature'], box, polygon)
        local_center = [refined_center[0] - bx0, refined_center[1] - by0]
        right = panel(crop, rsim, local_center, 'DINO-guided crop | peak %.3f' % rpeak, config['softmax_temperature'])
        combined = np.concatenate([left, right], axis=1)
        cv2.imwrite(str(out / (frame['name'] + '-matching.png')), combined)
        montage.append(combined)
        rows.append(row)
        print('MATCH', json.dumps(row), flush=True)
    cv2.imwrite(str(out / 'matching-overview.png'), np.concatenate(montage, axis=0))
    if geometry_montage:
        cv2.imwrite(str(out/'geometry-overview.png'),np.concatenate(geometry_montage,axis=0))
    report = {'status': 'completed', 'ready_for_live_control': False, 'geometry_free_space_checked': False,
        'device': device, 'simulation_connected': False,
        'flight_commands_sent': False, 'encoder_training': False,
        'depth_geometry_enabled': options.depth_geometry,
        'model': 'dinov2_vits14', 'weights': str(args.dino_weights.relative_to(ROOT)),
        'weights_sha256': hashlib.sha256(args.dino_weights.read_bytes()).hexdigest(),
        'config': config, 'normalization_source': config['reference_image'],
        'query_frames_used_for_normalization': False, 'refinement_crop_source': 'full-frame DINO center; no UE coordinates',
        'heatmap': 'square-root softmax control weights; not a segmentation mask',
        'validation_scope': 'offline development diagnostics; no live tracking, no passage verification, no success-rate claim',
        'limitations': ['Reference self-check is not independent validation',
            'Stored nadir development observations; no held-out flight test or rotation sequence',
            'Near and post-entry frames may not contain the complete opening',
            'No independent query-frame opening labels; peak does not establish correct matching',
            'DINO center is not a safe passage point'], 'results': rows}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / 'input-config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
