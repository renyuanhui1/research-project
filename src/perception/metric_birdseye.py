"""Template-resolution metric tiling of observed RGB-D surfaces."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from perception.birdseye_matching import BACKGROUND, backproject, sift_features, sift_candidates


def metric_cloud(rgb, depth, camera, fov, cfg, excluded):
    points = backproject(depth, camera, fov)
    keep = np.isfinite(points).all(-1) & (depth > 0) & (depth <= cfg['max_ray_depth_m'])
    keep &= (np.linalg.norm(points[..., :2], axis=-1) <= cfg['horizontal_radius_m']) & ~excluded
    indices = np.flatnonzero(keep); xyz = points.reshape(-1, 3)[indices]
    h, w = depth.shape
    colors = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).reshape(-1, 3)[indices]
    focal = w / (2 * np.tan(np.deg2rad(fov) / 2))
    footprint = depth.reshape(-1)[indices] ** 2 / np.maximum(np.abs(xyz[:, 2]), 1e-6) / focal
    return dict(xyz=xyz, es=np.c_[xyz[:, 1], -xyz[:, 0]], indices=indices, colors=colors,
                footprint=footprint, source_shape=(h, w), retained=keep)


def metric_reference(image, ground_resolution, cfg):
    ratio = cfg['reference_long_side'] / max(image.shape[:2])
    resized = cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
    return dict(rgb=resized, sift=sift_features(resized, cfg['sift']), gsd=ground_resolution / ratio)


def metric_search(cloud, reference, cfg):
    """Exhaust occupied tiles; resolution and tile placement use no target location."""
    gsd = reference['gsd']; side = cfg['tile_side']; stride = cfg['tile_stride']
    if not len(cloud['indices']):
        return [], dict(tiles=0, gsd_m=gsd)
    global_xy = np.floor(cloud['es'] / gsd).astype(int)
    starts = set()
    # Every point contributes to all overlapping windows on a camera-relative grid.
    for dx in range(int(np.ceil(side / stride))):
        for dy in range(int(np.ceil(side / stride))):
            starts.update(map(tuple, np.unique(global_xy // stride - [dx, dy], axis=0)))
    candidates = []; tiles = 0
    for grid_start in sorted(starts):
        low = np.array(grid_start) * stride; xy = global_xy - low
        selected = np.flatnonzero(((xy >= 0) & (xy < side)).all(1))
        if len(selected) < cfg['sift']['minimum_inliers']:
            continue
        positions = xy[selected]; flat = positions[:, 1] * side + positions[:, 0]
        order = np.lexsort((-cloud['indices'][selected], -cloud['xyz'][selected, 2], flat))
        end = np.r_[flat[order][1:] != flat[order][:-1], True]
        choose = order[end]; index = np.full(side * side, -1, np.int32)
        index[flat[choose]] = selected[choose]; index = index.reshape(side, side)
        known = index >= 0
        dist, nearest = distance_transform_edt(~known, return_indices=True)
        closest = index[tuple(nearest)]
        support = known | (dist <= cloud['footprint'][closest] / gsd * cfg['splat_radius_factor'])
        image = np.broadcast_to(BACKGROUND, (side, side, 3)).copy()
        image[support] = cloud['colors'][closest[support]]
        query = sift_features(image, cfg['sift'], support)
        predicted = sift_candidates(query, reference['sift'], image.shape[:2], reference['rgb'].shape[:2], cfg['sift'], cfg['candidates'])
        tiles += 1
        for row in predicted:
            x, y = np.floor(np.asarray(row['view_point_xy']) * side).astype(int)
            if not support[y, x]:
                continue
            source = int(cloud['indices'][closest[y, x]]); h, w = cloud['source_shape']
            candidates.append(dict(row, point_xy=[(source % w + .5) / w, (source // w + .5) / h],
                                   metric_es_m=cloud['es'][closest[y, x]].tolist(), tile_origin_pixel=low.tolist()))
    result = []; radius = max(reference['rgb'].shape[:2]) * gsd * cfg['nms_template_fraction']
    for row in sorted(candidates, key=lambda r: -r['score']):
        if any(np.linalg.norm(np.array(row['metric_es_m']) - r['metric_es_m']) < radius for r in result):
            continue
        result.append(row)
        if len(result) == cfg['candidates']:
            break
    return result, dict(tiles=tiles, gsd_m=gsd, nms_radius_m=radius)
