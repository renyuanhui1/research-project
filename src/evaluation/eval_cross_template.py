"""Evaluate a learned local matcher on its own held-out folds or on another capture.

Two modes:
  self      each fold model predicts only the frames held out by that fold
  transfer  every fold model predicts every frame supplied by --features

The grouped split (15 degree bearing sector x range stratum) is rebuilt from the
feature rows, so `self` reproduces the training-time out-of-fold estimate.
`transfer` is used to test whether weights trained on one capture or aircraft
localize targets in another capture.

Both the original 3-pixel tolerance and an area-normalized distance are
reported. The normalized distance is divided by the part mask's equivalent
radius sqrt(area/pi), which makes parts with very different mask sizes
comparable.
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
sys.path.insert(0, str(ROOT/'src'/'training'))
from train_local_matcher import HeatmapNet, group_of, peak_stats, read_rows, full_image_point
from common.patches import open_patches, read_rgb


def distance_maps(mask):
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return outside, inside


def distance_at(maps, mask, point):
    outside, inside = maps
    x, y = np.rint(point).astype(int)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return float('inf')
    return float(-inside[y, x] if mask[y, x] else outside[y, x])


def range_bin(distance):
    if distance < 30:
        return '12-30'
    if distance < 60:
        return '30-60'
    if distance < 100:
        return '60-100'
    return '100-150'


def load_models(weights_dir, folds, channels, device):
    models = {}
    for fold in range(folds):
        path = weights_dir/('fold%d.pt' % fold)
        if not path.exists():
            raise FileNotFoundError('Missing fold weights: '+str(path))
        model = HeatmapNet(channels).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[fold] = model
    return models


def predict(rows, models, patch_dir, parts, device, batch):
    """Return {fold: {sample: {part: peak/statistics}}}."""
    predictions = {fold: {} for fold in models}
    with open_patches(patch_dir) as store:
        for start in range(0, len(rows), batch):
            chunk = rows[start:start+batch]
            images = torch.stack([torch.from_numpy(
                read_rgb(store, row['sample']).astype(np.float32)/255.).permute(2, 0, 1) for row in chunk]).to(device)
            for fold, model in models.items():
                with torch.no_grad():
                    logits = model(images)
                    prob = F.softmax(logits.flatten(2), dim=2).view_as(logits).cpu().numpy()
                    logits = logits.cpu().numpy()
                for row, heat, logit in zip(chunk, prob, logits):
                    entry = predictions[fold].setdefault(row['sample'], {})
                    for channel, part in enumerate(parts):
                        stats = peak_stats(logit[channel], heat[channel])
                        point = full_image_point(row, stats.pop('peak_x'), stats.pop('peak_y'))
                        entry[part] = dict(point=point, **stats)
            if (start//batch+1) % 20 == 0:
                print('PREDICT', min(start+batch, len(rows)), 'of', len(rows), flush=True)
    return predictions


def score(rows, predictions, parts, capture, masks, tolerance):
    out = []
    for index, row in enumerate(rows):
        sample = row['sample']
        x0, y0, x1, y1 = row['patch_xyxy']
        sample_masks = {part: cv2.imread(str(capture/masks[sample][part]['path']), 0) > 0
                        for part in parts}
        maps = {part: distance_maps(sample_masks[part]) for part in parts}
        for fold, per_sample in predictions.items():
            entry = per_sample.get(sample)
            if entry is None:
                continue
            for part in parts:
                mask = sample_masks[part]
                stats = entry[part]
                distance = distance_at(maps[part], mask, np.asarray(stats['point'], float))
                area = int(mask.sum())
                norm = distance/math.sqrt(area/math.pi) if area > 0 else float('inf')
                out.append(dict(fold=fold, part=part, sample=sample,
                                bearing_deg=row['bearing_deg'], range_m=row['range_m'],
                                range_bin=range_bin(row['range_m']),
                                visible=int(mask[y0:y1, x0:x1].sum()), area_px=area,
                                point=list(stats['point']), distance_px=distance,
                                norm_distance=norm, inside=bool(distance <= 0),
                                correct=bool(distance <= tolerance),
                                norm_le1=bool(abs(norm) <= 1.),
                                **{key: stats[key] for key in
                                   ('confidence', 'entropy', 'top_mass', 'max_logit',
                                    'logit_margin', 'peak_centroid_cells')}))
        if (index+1) % 500 == 0:
            print('SCORE', index+1, 'of', len(rows), flush=True)
    return out


def part_metrics(rows, parts, tolerance):
    visible = [row for row in rows if row['visible'] >= 20]
    result = {}
    for part in parts:
        selected = [row for row in visible if row['part'] == part]
        if not selected:
            result[part] = dict(samples=0)
            continue
        correct = np.array([row['correct'] for row in selected])
        inside = np.array([row['inside'] for row in selected])
        norm_le1 = np.array([row['norm_le1'] for row in selected])
        confidence = np.array([row['confidence'] for row in selected])
        order = np.argsort(-confidence)
        tp = np.cumsum(correct[order])
        precision = tp/np.arange(1, len(tp)+1)
        recall = tp/max(int(correct.sum()), 1)
        keep = precision >= .95
        by_range = {}
        for name in ('12-30', '30-60', '60-100', '100-150'):
            group = [row for row in selected if row['range_bin'] == name]
            if group:
                by_range[name] = dict(samples=len(group),
                                      correct_rate=float(np.mean([row['correct'] for row in group])),
                                      inside_rate=float(np.mean([row['inside'] for row in group])))
        result[part] = dict(
            samples=len(selected), correct=int(correct.sum()),
            correct_rate=float(correct.mean()),
            inside_rate=float(inside.mean()),
            norm_le1_rate=float(norm_le1.mean()),
            median_distance_px=float(np.median([row['distance_px'] for row in selected])),
            median_norm_distance=float(np.median([row['norm_distance'] for row in selected])),
            recall_at_precision_95=float(recall[keep].max()) if keep.any() else 0.,
            by_range=by_range)
    return result


def summarize(oof, parts, tolerance):
    folds = sorted({row['fold'] for row in oof})
    per_fold = {fold: part_metrics([row for row in oof if row['fold'] == fold], parts, tolerance)
                for fold in folds}
    summary = {}
    for part in parts:
        values = {key: [per_fold[fold][part].get(key) for fold in folds
                        if per_fold[fold][part].get(key) is not None]
                  for key in ('correct_rate', 'inside_rate', 'norm_le1_rate')}
        summary[part] = {key: (dict(mean=float(np.mean(value)), std=float(np.std(value)))
                               if value else None) for key, value in values.items()}
    return dict(per_fold={str(fold): fold_metrics for fold, fold_metrics in per_fold.items()},
                summary=summary,
                pooled=part_metrics(oof, parts, tolerance))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('self', 'transfer'), required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--patch-dir', type=Path, required=True,
                        help='HDF5 patch file or image directory')
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', default=['nose', 'wing', 'tail'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--tolerance-px', type=float, default=3.)
    parser.add_argument('--limit', type=int, help='Use only the first N feature rows for a smoke test')
    options = parser.parse_args(argv)
    from sklearn.model_selection import GroupKFold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_rows(options.features, options.patch_dir)
    if options.limit:
        rows = rows[:options.limit]
    groups = np.array([group_of(row) for row in rows])
    print('rows', len(rows), 'groups', len(set(groups)), 'mode', options.mode, flush=True)
    report = json.loads((options.capture_dir/'report.json').read_text())
    masks = {sample['name']: sample['masks'] for sample in report['samples']}
    parts = options.parts
    if options.mode == 'self':
        models = load_models(options.weights_dir, options.folds, len(parts), device)
        oof = []
        split = GroupKFold(n_splits=options.folds).split(rows, groups=groups)
        for fold, (_, test_index) in enumerate(split):
            test_rows = [rows[index] for index in test_index]
            print('fold', fold, 'test', len(test_rows), flush=True)
            predictions = predict(test_rows, {fold: models[fold]}, options.patch_dir,
                                  parts, device, options.batch)
            oof.extend(score(test_rows, predictions, parts, options.capture_dir, masks,
                             options.tolerance_px))
    else:
        models = load_models(options.weights_dir, options.folds, len(parts), device)
        predictions = predict(rows, models, options.patch_dir, parts, device, options.batch)
        oof = score(rows, predictions, parts, options.capture_dir, masks, options.tolerance_px)
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'oof.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in oof))
    result = dict(mode=options.mode, features=str(options.features.resolve()),
                  patch_dir=str(options.patch_dir.resolve()),
                  capture=str(options.capture_dir.resolve()),
                  weights=str(options.weights_dir.resolve()), parts=parts,
                  folds=options.folds, tolerance_px=options.tolerance_px,
                  samples=len(rows), device=device,
                  **summarize(oof, parts, options.tolerance_px))
    (out/'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
