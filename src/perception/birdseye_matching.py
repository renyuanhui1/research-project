"""Training-free RGB-D orthographic views and template matching; no target poses."""
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation


BACKGROUND = np.array([124, 116, 104], np.uint8)


def backproject(depth, camera, horizontal_fov_deg):
    """Return camera-relative NED points from metric ray lengths at pixel centers."""
    h, w = depth.shape
    focal = w / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
    rays = np.stack([np.ones((h, w)), (u - w / 2) / focal, (v - h / 2) / focal], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rotation = Rotation.from_quat([camera['rot_' + k] for k in 'xyzw']).as_matrix()
    return (rays @ rotation.T) * depth[..., None]


def birdseye(rgb, depth, camera, horizontal_fov_deg, cfg, excluded=None):
    """Rasterize visible surfaces north-up (right=east, down=south).

    Only attitude is required: horizontal translation cancels and height is unused.
    Highest visible surface wins; interpolation retains source-pixel provenance.
    `excluded` is the dataset's predeclared redaction, never a target label.
    """
    depth = np.asarray(depth, np.float32)
    points = backproject(depth, camera, horizontal_fov_deg)
    good = np.isfinite(points).all(-1) & (depth > 0) & (depth <= cfg['max_ray_depth_m'])
    good &= np.linalg.norm(points[..., :2], axis=-1) <= cfg['horizontal_radius_m']
    if excluded is not None:
        good &= ~np.asarray(excluded, bool)
    source = np.flatnonzero(good)
    if not len(source):
        return dict(rgb=BACKGROUND[None, None].copy(), source=np.full((1, 1), -1, np.int32),
                    gsd_m=cfg['gsd_m'], source_shape=list(depth.shape), input_kept=good,
                    origin_es_m=[0., 0.])
    xyz = points.reshape(-1, 3)[source]
    es = np.c_[xyz[:, 1], -xyz[:, 0]]
    span = np.ptp(es, axis=0)
    gsd = max(cfg['gsd_m'], float(span.max()) / (cfg['maximum_side'] - 2))
    low = np.floor(es.min(0) / gsd) * gsd
    xy = np.floor((es - low) / gsd).astype(int)
    w, h = xy.max(0) + 1
    flat = xy[:, 1] * w + xy[:, 0]
    # Descending NED z: the last point in each cell is the highest surface.
    order = np.lexsort((-source, -xyz[:, 2], flat))
    end = np.r_[flat[order][1:] != flat[order][:-1], True]
    selected = order[end]
    index = np.full(h * w, -1, np.int32)
    index[flat[selected]] = source[selected]
    index = index.reshape(h, w)
    known = index >= 0
    distance, nearest = distance_transform_edt(~known, return_indices=True)
    nearest_source = index[tuple(nearest)]
    # Approximate one source pixel's footprint, including foreshortening.
    dh, dw = depth.shape
    focal = dw / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    vertical_cosine = np.abs(points[..., 2]) / np.maximum(depth, 1e-6)
    footprint = depth / focal / np.maximum(vertical_cosine, .05) / gsd
    radius = footprint.reshape(-1)[nearest_source] * cfg['splat_radius_factor']
    supported = known | (distance <= radius)
    index = np.where(supported, nearest_source, -1).astype(np.int32)
    small = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    image = np.broadcast_to(BACKGROUND, (h, w, 3)).copy()
    image[supported] = small.reshape(-1, 3)[index[supported]]
    return dict(rgb=image, source=index, gsd_m=gsd, source_shape=list(depth.shape),
                input_kept=good, origin_es_m=low.tolist())


def source_point(view, point):
    """Map a view point to an observed normalized source pixel, or return None."""
    h, w = view['source'].shape
    x, y = np.floor(np.asarray(point) * [w, h]).astype(int)
    if not (0 <= x < w and 0 <= y < h):
        return None
    index = int(view['source'][y, x])
    if index < 0:
        return None
    sh, sw = view['source_shape']
    return [(index % sw + .5) / sw, (index // sw + .5) / sh]


def sift_features(rgb, cfg, valid=None):
    mask = None if valid is None else np.asarray(valid, np.uint8) * 255
    keypoints, descriptors = cv2.SIFT_create(nfeatures=cfg['maximum_features']).detectAndCompute(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)
    if descriptors is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    # RootSIFT, with the same representation for raw and rectified images.
    descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(1, keepdims=True), 1e-8))
    return np.array([p.pt for p in keypoints], np.float32), descriptors


def sift_candidates(query, reference, image_shape, reference_shape, cfg, count):
    qp, qd = query; tp, td = reference
    if len(qd) < 2 or len(td) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(td, qd, k=2)
    reverse = matcher.knnMatch(qd, td, k=2)
    reverse = {a.queryIdx: a.trainIdx for a, b in reverse if a.distance < cfg['ratio'] * b.distance}
    matches = [a for a, b in forward if a.distance < cfg['ratio'] * b.distance
               and reverse.get(a.trainIdx) == a.queryIdx]
    remaining = np.arange(len(matches)); result = []
    template_points = tp[[m.queryIdx for m in matches]]
    query_points = qp[[m.trainIdx for m in matches]]
    th, tw = reference_shape; h, w = image_shape
    while len(remaining) >= cfg['minimum_inliers'] and len(result) < count:
        cv2.setRNGSeed(cfg['seed'])
        transform, inliers = cv2.estimateAffine2D(
            template_points[remaining], query_points[remaining], method=cv2.RANSAC,
            ransacReprojThreshold=cfg['reprojection_px'], maxIters=cfg['ransac_iterations'],
            confidence=.99, refineIters=10)
        if transform is None or inliers is None or not np.isfinite(transform).all():
            break
        inside = inliers[:, 0].astype(bool); selected = remaining[inside]
        if len(selected) < cfg['minimum_inliers']:
            break
        area = cv2.contourArea(cv2.convexHull(template_points[selected])) / (th * tw)
        score = len(selected) * math.sqrt(max(0., area)) * len(selected) / len(remaining)
        # An actual matched pixel near the template center, not a hallucinated center.
        near = selected[np.argmin(np.square(template_points[selected] / [tw, th] - .5).sum(1))]
        point = (query_points[near] / [w, h]).tolist()
        result.append(dict(view_point_xy=point, score=float(score), inliers=len(selected),
                           template_coverage=float(area)))
        remaining = remaining[~inside]
    return sorted(result, key=lambda item: -item['score'])


class DinoMatcher:
    """Frozen DINOv2 mean-template cosine peaks; four reference rotations, no fitting."""
    def __init__(self, cfg, device):
        import torch
        from common.paths import project_path
        from perception.extract_dino_features import load_model, IMAGENET_MEAN, IMAGENET_STD
        self.torch = torch; self.device = torch.device(device); self.cfg = cfg
        self.model = load_model('dinov2_vits14', self.device,
                                weights=project_path(cfg['weights']), repo_dir=project_path(cfg['repository']))
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def encode(self, images, size):
        from perception.extract_dino_features import to_input_tensor
        torch = self.torch
        images = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR) for im in images])
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            x = to_input_tensor(images, size, self.mean, self.std, self.device)
            tokens = self.model.forward_features(x)['x_norm_patchtokens'].float()
        return torch.nn.functional.normalize(tokens, dim=-1)

    def reference(self, image):
        z = self.encode([np.rot90(image, k).copy() for k in range(4)], self.cfg['template_size'])
        return self.torch.nn.functional.normalize(z.mean(1), dim=-1)

    def candidates(self, tokens, reference, support, count):
        grid = math.isqrt(len(tokens)); h, w = support.shape
        response = (tokens @ reference.T).max(-1).values.cpu().numpy().reshape(grid, grid)
        fraction = cv2.resize(support.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
        response[fraction < self.cfg['minimum_patch_support']] = -np.inf
        result = []
        for _ in range(count):
            y, x = np.unravel_index(response.argmax(), response.shape)
            score = float(response[y, x])
            if not np.isfinite(score):
                break
            x0, x1 = int(x * w / grid), max(int((x + 1) * w / grid), int(x * w / grid) + 1)
            y0, y1 = int(y * h / grid), max(int((y + 1) * h / grid), int(y * h / grid) + 1)
            sy, sx = np.where(support[y0:y1, x0:x1])
            if len(sx):
                sx = sx + x0; sy = sy + y0
                nearest = np.argmin((sx - (x + .5) * w / grid)**2 + (sy - (y + .5) * h / grid)**2)
                result.append(dict(view_point_xy=[(float(sx[nearest]) + .5) / w, (float(sy[nearest]) + .5) / h], score=score))
            radius = self.cfg['nms_radius_cells']
            response[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -np.inf
        return result
