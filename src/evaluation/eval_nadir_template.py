"""Compare a nadir (top-down) template with an oblique template on saved frames.

The nadir template is cut from a recorded overhead image by projecting the
target mesh through the recorded camera pose; no simulator is needed. Each
template is encoded by the frozen DINOv2 encoder, and every frame is matched
against both template prototypes. Both prototypes share one set of
standardization statistics computed over frames of the evaluated scene, so the
only difference is the template viewpoint.

Reported: whether the response peak falls inside the whole-target mask, the
signed distance to that mask, and the same quantities for the softmax center.
Each template is also tried at four rotations and the per-frame highest peak is
reported as `best_of_4`, since the whole-template prototype is orientation
sensitive. This is a coarse whole-target baseline, not a part-level metric.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from perception.extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD


def rotation_from_quaternion(message):
    w, x, y, z = (float(message['rot_'+key]) for key in ('w', 'x', 'y', 'z'))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project(points, message):
    camera = np.array([float(message['pos_'+key]) for key in ('x', 'y', 'z')])
    local = (np.asarray(points, float)-camera)@rotation_from_quaternion(message)
    if np.any(local[:, 0] <= 0):
        raise RuntimeError('Target geometry behind the camera')
    focal = message['width']/2
    return np.column_stack((message['width']/2+focal*local[:, 1]/local[:, 0],
                            message['height']/2+focal*local[:, 2]/local[:, 0]))


def nadir_template(image, report, actor, mesh_path, image_name, margin=1.15):
    capture = next(item for item in report['captures'] if item['target'] == actor)
    message = capture['camera_frame']
    with np.load(mesh_path) as data:
        vertices = data['vertices_enu'][:, [1, 0, 2]].astype(np.float64)
    vertices[:, 2] *= -1
    pixels = project(vertices, message)
    low, high = pixels.min(0), pixels.max(0)
    center = (low+high)/2
    side = max(high-low)*margin
    x0 = int(round(center[0]-side/2)); y0 = int(round(center[1]-side/2))
    x1 = int(round(center[0]+side/2)); y1 = int(round(center[1]+side/2))
    height, width = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1-x0 < 32 or y1-y0 < 32:
        raise RuntimeError('Nadir template crop is too small')
    meta = dict(actor=actor, image=image_name, camera=message,
                projected_xyxy=[x0, y0, x1, y1],
                projected_target_pixels=[float(low[0]), float(low[1]),
                                         float(high[0]), float(high[1])],
                margin=margin)
    return image[y0:y1, x0:x1].copy(), meta


class Template:
    """One template encoded once; its prototype is a mean over all its patches."""

    def __init__(self, name, rgb, model, device, image_size=448):
        self.name = name
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = to_input_tensor(rgb[None], image_size, mean, std, device)
        with torch.no_grad():
            self.z = model.forward_features(x)['x_norm_patchtokens'][0].float()
        self.patches = int(self.z.shape[0])
        self.grid = int(round(self.patches**0.5))
        if self.grid*self.grid != self.patches:
            raise RuntimeError('DINO patch count is not a square grid')

    def prototype(self, mean, std):
        normalized = (self.z-mean)/std
        return F.normalize(F.normalize(normalized, dim=-1).mean(0), dim=0)

    def similarity(self, tokens, proto, mean, std):
        normalized = (tokens-mean)/std
        return (F.normalize(normalized, dim=-1)@proto).reshape(self.grid, self.grid)


def scene_stats(samples, capture, model, device, mean, std, image_size, count):
    """Standardization statistics from frames of the evaluated scene.

    Both templates share these statistics, so the comparison isolates the
    template viewpoint. In deployment they can be taken from the first frames
    of the approach.
    """
    index = np.unique(np.linspace(0, len(samples)-1, min(count, len(samples))).astype(int))
    rgbs = [np.ascontiguousarray(cv2.imread(str(capture/samples[i]['rgb']))[:, :, ::-1])
            for i in index]
    x = to_input_tensor(np.stack(rgbs), image_size, mean, std, device)
    with torch.no_grad():
        tokens = model.forward_features(x)['x_norm_patchtokens'].float()
    flat = tokens.reshape(-1, tokens.shape[-1])
    return (flat.mean(0), flat.std(0).clamp_min(1e-6),
            [samples[i]['name'] for i in index])


ROTATIONS = dict(r0=None, r90=cv2.ROTATE_90_CLOCKWISE, r180=cv2.ROTATE_180,
                 r270=cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotated(rgb, code):
    return rgb if code is None else cv2.rotate(rgb, code)


def distance_at(outside, mask, point):
    x, y = np.rint(point).astype(int)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return float('inf'), False
    if mask[y, x]:
        return 0., True
    return float(outside[y, x]), False


def peak_and_center(sim, width, height, temperature=.005):
    grid = sim.shape[0]
    index = int(sim.reshape(-1).argmax())
    row, column = divmod(index, grid)
    peak = ((column+.5)/grid*width, (row+.5)/grid*height)
    weights = torch.softmax((sim.reshape(-1)/temperature).float(), dim=0).cpu().numpy()
    axis = (np.arange(grid)+.5)/grid
    center = (float((weights.reshape(grid, grid).sum(0)*axis).sum()*width),
              float((weights.reshape(grid, grid).sum(1)*axis).sum()*height))
    return peak, center, float(sim.reshape(-1)[index])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--overhead-image', type=Path, required=True)
    parser.add_argument('--overhead-report', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--oblique-template', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--skip', type=int, default=0, help='Drop leading warm-up frames')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--stats-frames', type=int, default=32,
                        help='Frames from this capture used for standardization statistics')
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--dino-weights', type=Path,
                        default=ROOT/'weights/dinov2_vits14_pretrain.pth')
    parser.add_argument('--repo-dir', type=Path, default=ROOT/'third_party/dinov2')
    options = parser.parse_args(argv)
    image_path = options.overhead_image.resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit('Cannot read overhead image')
    report = json.loads(options.overhead_report.read_text())
    crop, meta = nadir_template(image, report, options.actor, options.mesh.resolve(),
                                str(image_path))
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out/'template-nadir.png'), crop)
    (out/'template-nadir.json').write_text(json.dumps(meta, indent=2))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model = load_model('dinov2_vits14', device,
                       weights=str(options.dino_weights), repo_dir=str(options.repo_dir))
    oblique = cv2.imread(str(options.oblique_template))
    if oblique is None:
        raise SystemExit('Cannot read oblique template')
    capture = options.capture_dir.resolve()
    samples = json.loads((capture/'report.json').read_text())['samples'][options.skip:]
    if options.limit:
        samples = samples[:options.limit]
    stat_mean, stat_std, stat_names = scene_stats(samples, capture, model, device, mean, std,
                                                  options.image_size, options.stats_frames)
    sources = {'nadir': np.ascontiguousarray(crop[:, :, ::-1]),
               'oblique': np.ascontiguousarray(oblique[:, :, ::-1])}
    templates = {(name, rotation): Template(name, rotated(rgb, code), model, device,
                                           options.image_size)
                 for name, rgb in sources.items() for rotation, code in ROTATIONS.items()}
    protos = {key: template.prototype(stat_mean, stat_std)
              for key, template in templates.items()}
    records, overlays = [], {}
    for start in range(0, len(samples), options.batch):
        chunk = samples[start:start+options.batch]
        rgbs = [np.ascontiguousarray(cv2.imread(str(capture/sample['rgb']))[:, :, ::-1])
                for sample in chunk]
        x = to_input_tensor(np.stack(rgbs), options.image_size, mean, std, device)
        with torch.no_grad():
            tokens = model.forward_features(x)['x_norm_patchtokens'].float()
        for sample, rgb, z in zip(chunk, rgbs, tokens):
            height, width = rgb.shape[:2]
            mask = cv2.imread(str(capture/sample['masks']['aircraft']['path']), 0) > 0
            outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
            distance = range_bin(sample['case']['horizontal_range_m'])
            for (name, rotation), template in templates.items():
                sim = template.similarity(z, protos[(name, rotation)], stat_mean, stat_std)
                peak, center, score = peak_and_center(sim, width, height)
                peak_distance, peak_hit = distance_at(outside, mask, peak)
                center_distance, center_hit = distance_at(outside, mask, center)
                records.append(dict(sample=sample['name'], template=name, rotation=rotation,
                                    range_m=sample['case']['horizontal_range_m'],
                                    range_bin=distance, peak=list(peak), center=list(center),
                                    peak_score=score, peak_hit=peak_hit,
                                    peak_distance_px=peak_distance,
                                    center_hit=center_hit,
                                    center_distance_px=center_distance))
            if distance not in overlays:
                best = max((row for row in records
                            if row['sample'] == sample['name'] and row['template'] == 'nadir'),
                           key=lambda row: row['peak_score'])
                overlays[distance] = (rgb.copy(), best['peak'], best['rotation'])
        print('FRAME', min(start+options.batch, len(samples)), 'of', len(samples), flush=True)
    (out/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    metrics = summarize(records)
    metrics['stats'] = dict(frames=len(stat_names), samples=stat_names)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    for name, (rgb, peak, rotation) in overlays.items():
        x, y = np.rint(peak).astype(int)
        cv2.circle(rgb, (int(x), int(y)), 12, (0, 0, 255), 3)
        cv2.imwrite(str(out/('peak-%s-%s.png' % (name, rotation))), rgb[:, :, ::-1])
    print(json.dumps(metrics, indent=2))
    return 0


def range_bin(distance):
    if distance < 30:
        return '12-30'
    if distance < 60:
        return '30-60'
    if distance < 100:
        return '60-100'
    return '100-150'


def summarize(records):
    """Report the original orientation and a 4-way rotation search.

    `best_of_4` picks, per frame, the rotation whose response peak is highest;
    the choice uses no ground truth and is what deployment can do.
    """
    result = {}
    for template in ('nadir', 'oblique'):
        rows = [row for row in records if row['template'] == template]
        best = {}
        for row in rows:
            current = best.get(row['sample'])
            if current is None or row['peak_score'] > current['peak_score']:
                best[row['sample']] = row
        best_rows = list(best.values())
        result[template] = dict(
            single=metrics([row for row in rows if row['rotation'] == 'r0']),
            best_of_4=metrics(best_rows),
            chosen_rotations={rotation: sum(row['rotation'] == rotation for row in best_rows)
                              for rotation in ROTATIONS})
    return result


def metrics(rows):
    entry = dict(frames=len(rows),
                 peak_hit_rate=float(np.mean([row['peak_hit'] for row in rows])),
                 center_hit_rate=float(np.mean([row['center_hit'] for row in rows])),
                 mean_peak_score=float(np.mean([row['peak_score'] for row in rows])),
                 median_peak_distance_px=float(np.median([row['peak_distance_px'] for row in rows])),
                 by_range={})
    for name in ('12-30', '30-60', '60-100', '100-150'):
        group = [row for row in rows if row['range_bin'] == name]
        if group:
            entry['by_range'][name] = dict(
                frames=len(group),
                peak_hit_rate=float(np.mean([row['peak_hit'] for row in group])),
                center_hit_rate=float(np.mean([row['center_hit'] for row in group])),
                mean_peak_score=float(np.mean([row['peak_score'] for row in group])))
    return entry


if __name__ == '__main__':
    raise SystemExit(main())
