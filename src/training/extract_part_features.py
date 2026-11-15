"""Extract frozen-matcher features and mask labels from training captures."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from perception.part_matching import CroppedPartMatcher


def read(path):
    return json.loads(path.read_text())


def rebase(path):
    """Map absolute paths recorded on another machine onto this checkout."""
    path = Path(path)
    if path.exists() or 'research-project' not in path.parts:
        return path
    index = len(path.parts)-1-path.parts[::-1].index('research-project')
    return ROOT.joinpath(*path.parts[index+1:])


def point_mask(mask, point):
    x, y = np.rint(point).astype(int)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return False
    return bool(mask[y, x])


def patch_region(box, shape, margin=.25):
    """Square region around the coarse box, clamped into the image."""
    x0, y0, x1, y1 = (float(v) for v in box)
    side = max(x1-x0, y1-y0)*(1.+2.*margin)
    side = min(side, shape[0], shape[1])
    cx, cy = (x0+x1)/2., (y0+y1)/2.
    x0 = int(round(min(max(cx-side/2., 0.), shape[1]-side)))
    y0 = int(round(min(max(cy-side/2., 0.), shape[0]-side)))
    x1, y1 = int(round(x0+side)), int(round(y0+side))
    return x0, y0, x1, y1


def outside_distance(mask, point):
    """Signed distance in pixels; negative inside the mask."""
    inverse = (~mask).astype(np.uint8)
    outside = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    x, y = np.rint(point).astype(int)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return float('nan')
    return float(-inside[y, x] if mask[y, x] else outside[y, x])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip', type=int, default=0, help='Drop this many leading frames (exposure warm-up)')
    parser.add_argument('--patches', type=Path, help='Also export 256x256 search patches and 64x64 part-mask targets')
    parser.add_argument('--all-parts', action='store_true',
                        help='Export one patch per frame with a target for every part in the masks')
    options = parser.parse_args(argv)
    capture, run, out = (p.resolve() for p in (options.capture_dir, options.run_dir, options.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    report = read(capture/'report.json')
    if report['status'] != 'completed':
        raise ValueError('Capture is not completed')
    setup = read(run/'controller-setup.json')
    args = argparse.Namespace(**read(run/'report.json')['controller_args'])
    for name in ('template', 'stats_episode', 'repo_dir', 'dino_weights'):
        setattr(args, name, str(rebase(getattr(args, name))))
    if not getattr(args, 'thin_roi_refinement', False):
        raise ValueError('Source run must use thin ROI refinement')
    reference = rebase(setup['shared_reference'])
    meta = read(reference/'template.json')
    rois = {name: part['template_roi'] for name, part in meta['parts'].items()}
    import torch
    device = options.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fp = CroppedPartMatcher(args, device, rois, setup['target_part'])
    part = setup['target_part']
    samples = report['samples'][options.skip:]
    if options.limit:
        samples = samples[:options.limit]
    patches = options.patches.resolve() if options.patches else None
    if patches is not None:
        patches.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for index, sample in enumerate(samples):
        rgb = np.ascontiguousarray(cv2.imread(str(capture/sample['rgb']))[:, :, ::-1])
        if rgb.mean() > 240 or rgb.std() < 5:
            skipped += 1
            continue
        masks = {name: cv2.imread(str(capture/record['path']), 0) > 0
                 for name, record in sample['masks'].items()}
        fp.refinement_state.clear()
        fp.query(rgb)
        diagnostics = fp.last_diagnostics
        local = diagnostics.get(part, {})
        refinement = local.get('refinement') or {}
        candidate = refinement.get('coarse_candidate_px')
        accepted = bool(refinement.get('accepted'))
        point = np.array(refinement['point_px']) if refinement.get('point_px') is not None else None
        mask = masks.get(part)
        patch_fields = dict(patch_xyxy=None, patch_side_px=None, part_pixels_in_patch=0)
        if patches is not None and diagnostics.get('crop_xyxy'):
            x0, y0, x1, y1 = patch_region(diagnostics['crop_xyxy'], rgb.shape[:2])
            patch = cv2.resize(rgb[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(patches/(sample['name']+'.png')), patch[:, :, ::-1]):
                raise RuntimeError('Cannot save patch')
            exported = [name for name in masks if name != 'aircraft'] if options.all_parts else [part]
            pixels = {}
            for name in exported:
                part_crop = masks[name][y0:y1, x0:x1]
                target = cv2.resize(part_crop.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
                target = cv2.GaussianBlur(target, (0, 0), 1.)
                if target.max() > 0:
                    target = target/target.max()
                suffix = ('-'+name) if options.all_parts else '-mask'
                np.save(patches/(sample['name']+suffix+'.npy'), target.astype(np.float16))
                pixels[name] = int(part_crop.sum())
            patch_fields = dict(patch_xyxy=[x0, y0, x1, y1], patch_side_px=x1-x0,
                                part_pixels_in_patch=pixels.get(part, 0),
                                part_pixels=pixels if options.all_parts else None)
        row = dict(
            sample=sample['name'], range_m=sample['case']['horizontal_range_m'],
            bearing_deg=sample['case']['bearing_deg'], heading_offset_deg=sample['case']['heading_offset_deg'],
            depth_m=sample.get('center_depth_m'), whole_peak=diagnostics.get('whole_peak'),
            box_width=(diagnostics['crop_xyxy'][2]-diagnostics['crop_xyxy'][0]) if diagnostics.get('crop_xyxy') else None,
            box_height=(diagnostics['crop_xyxy'][3]-diagnostics['crop_xyxy'][1]) if diagnostics.get('crop_xyxy') else None,
            part_dino_peak=float(np.max(local['local_similarity'])) if local.get('local_similarity') is not None else None,
            candidate_x=float(candidate[0]) if candidate is not None else None,
            candidate_y=float(candidate[1]) if candidate is not None else None,
            score=refinement.get('score'), margin=refinement.get('margin'),
            scale=refinement.get('scale'), angle_deg=refinement.get('angle_deg'),
            inside_radius=refinement.get('inside'),
            point_candidate_distance_px=(float(np.linalg.norm(point-np.array(candidate)))
                                          if point is not None and candidate is not None else None),
            accepted=accepted, reason=refinement.get('reason'),
            mask_pixels=int(mask.sum()) if mask is not None else 0,
            **patch_fields,
            candidate_inside=point_mask(mask, candidate) if (mask is not None and candidate is not None) else None,
            point_inside=point_mask(mask, point) if (mask is not None and point is not None) else None,
            point_distance_px=outside_distance(mask, point) if (mask is not None and point is not None) else None,
        )
        rows.append(row)
        if (index+1) % 100 == 0:
            print('FEATURE', index+1, 'of', len(samples), flush=True)
    with (out/'features.jsonl').open('w') as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False)+'\n')
    with (out/'features.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(samples=len(rows), skipped_blank=skipped, source=str(capture), run=str(run), device=device,
                   accepted=sum(bool(row['accepted']) for row in rows),
                   accepted_and_inside=sum(bool(row['accepted'] and row['point_inside']) for row in rows),
                   candidate_inside=sum(bool(row['candidate_inside']) for row in rows),
                   no_candidate=sum(row['candidate_x'] is None for row in rows))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
