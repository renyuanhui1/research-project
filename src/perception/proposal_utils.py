"""Shared candidate-peak extraction, rank fusion, and mask-coverage utilities."""
from itertools import zip_longest

import cv2
import numpy as np

SOURCES = ['original', 'all_balanced']


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def bin_name(value, edges):
    index = min(len(edges) - 2, max(0, int(np.searchsorted(edges, value, side='right')) - 1))
    return f'{edges[index]:g}..{edges[index + 1]:g}'


def point_hit(mask, point):
    h, w = mask.shape; x, y = np.floor(np.asarray(point) * [w, h]).astype(int)
    return bool(0 <= x < w and 0 <= y < h and mask[y, x])


def crop_coverage(mask, box, image_shape):
    h, w = mask.shape; ih, iw = image_shape
    low = np.floor(np.asarray(box[:2]) * [w / iw, h / ih]).astype(int)
    high = np.ceil(np.asarray(box[2:]) * [w / iw, h / ih]).astype(int)
    x0, y0 = np.maximum(low, 0); x1, y1 = np.minimum(high, [w, h])
    return float(mask[y0:y1, x0:x1].sum() / mask.sum()) if mask.any() else 0.


def grouped(rows, key, function):
    return {str(value): function([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})}


def box_iou(a, b):
    a = np.asarray(a); b = np.asarray(b)
    intersection = float(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod())
    union = float((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod()) - intersection
    return intersection / union if union else 0.


def peak_candidates(logits, image_shape, crop_cfg, count, nms_iou):
    maxima = (logits == cv2.dilate(logits, np.ones((3, 3), np.uint8))).astype(np.uint8)
    n, components = cv2.connectedComponents(maxima, connectivity=8)
    seeds = []
    for label in range(1, n):
        indices = np.flatnonzero(components == label)
        index = int(indices[np.argmax(logits.flat[indices])])
        seeds.append(index)
    seeds.sort(key=lambda index: (-float(logits.flat[index]), index))
    h, w = image_shape; grid = logits.shape[0]; candidates = []
    for index in seeds:
        y, x = divmod(index, grid)
        _, components = cv2.connectedComponents((logits >= logits[y, x] - crop_cfg['crop_logit_delta']).astype(np.uint8), connectivity=8)
        ys, xs = np.where(components == components[y, x])
        cx = (xs.min() + xs.max() + 1) / 2 / grid * w; cy = (ys.min() + ys.max() + 1) / 2 / grid * h
        side = max((xs.max() - xs.min() + 1) / grid * w, (ys.max() - ys.min() + 1) / grid * h,
                   max(h, w) * crop_cfg['crop_minimum_fraction']) * crop_cfg['crop_margin']
        low = np.maximum(np.floor([cx - side / 2, cy - side / 2]), [0, 0]).astype(int)
        high = np.minimum(np.ceil([cx + side / 2, cy + side / 2]), [w, h]).astype(int)
        box = [int(v) for v in np.r_[low, high]]
        if any(box_iou(box, prior['box_xyxy']) > nms_iou for prior in candidates):
            continue
        candidates.append({'point_xy': [(x + .5) / grid, (y + .5) / grid], 'box_xyxy': box, 'logit': float(logits[y, x])})
        if len(candidates) == count:
            break
    return candidates


def coverage_summary(rows, counts, minimum_crop):
    result = {}
    for k in counts:
        point = sum(any(r['point_hits'][:k]) for r in rows)
        crop = sum(max(r['crop_coverages'][:k], default=0) >= minimum_crop for r in rows)
        result[str(k)] = {'visible_objects': len(rows), 'point_covered': point, 'point_recall': ratio(point, len(rows)),
                          'crop_covered': crop, 'crop_recall': ratio(crop, len(rows))}
    return result


def fuse_candidates(pools, method, budget, nms_iou):
    """Use only source ranks and image boxes; labels and cross-model logits are unused."""
    if budget <= 0:
        return [], []
    tagged = {source: [dict(candidate, source=source, source_rank=i + 1)
                       for i, candidate in enumerate(pools[source])] for source in SOURCES}
    if method in SOURCES:
        return tagged[method][:budget], []
    if method == 'original_anchor':
        stream = tagged['original'][:1] + tagged['all_balanced'] + tagged['original'][1:]
    else:
        first = {'alternate_original': 'original', 'alternate_balanced': 'all_balanced'}[method]
        second = next(source for source in SOURCES if source != first)
        stream = [candidate for pair in zip_longest(tagged[first], tagged[second])
                  for candidate in pair if candidate is not None]
    selected = []; suppressed = []
    for candidate in stream:
        overlaps = [(prior, box_iou(candidate['box_xyxy'], prior['box_xyxy'])) for prior in selected]
        blocking = [(prior, overlap) for prior, overlap in overlaps if overlap > nms_iou]
        if blocking:
            prior, overlap = max(blocking, key=lambda item: item[1])
            suppressed.append({'source': candidate['source'], 'source_rank': candidate['source_rank'],
                               'blocked_by_source': prior['source'], 'blocked_by_rank': prior['source_rank'],
                               'iou': overlap})
            continue
        selected.append(candidate)
        if len(selected) == budget:
            break
    return selected, suppressed
