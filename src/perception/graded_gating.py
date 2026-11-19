"""Graded motion gating with ego-motion spatial tracking, binned isotonic calibration, and SPRT."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))


def pava_isotonic(scores, labels, smooth=1.0):
    """Fit non-decreasing isotonic probabilities via Pool Adjacent Violators (pure NumPy)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return [0.0, 1.0], [0.5, 0.5]
    order = np.argsort(scores, kind='mergesort')
    xs = scores[order]
    ys = labels[order]
    # Merge identical score values first so output knots are strictly increasing.
    uniq_x, idx, counts = np.unique(xs, return_index=True, return_counts=True)
    sums = np.add.reduceat(ys, idx)
    # Blocks: [x_min, x_max, sum_y, count]
    blocks = []
    for x_val, s_val, c_val in zip(uniq_x, sums, counts):
        blocks.append([float(x_val), float(x_val), float(s_val), float(c_val)])
        while len(blocks) >= 2 and (blocks[-2][2] / blocks[-2][3]) > (blocks[-1][2] / blocks[-1][3]):
            b = blocks.pop()
            blocks[-1][1] = b[1]
            blocks[-1][2] += b[2]
            blocks[-1][3] += b[3]
    prior = (float(labels.sum()) + 0.5) / (len(labels) + 1.0)
    knot_x = [0.5 * (b[0] + b[1]) for b in blocks]
    knot_y = [(b[2] + smooth * prior) / (b[3] + smooth) for b in blocks]
    return knot_x, knot_y


class DistanceBinCalibrator:
    """Range-stratified isotonic calibrator converting verification score into clipped LLR."""

    DEFAULT_EDGES = [0.0, 45.0, 80.0, 120.0, 1e9]

    def __init__(self, edges=None, bins=None, global_bin=None, llr_clip=2.5, smooth=2.0):
        self.edges = list(self.DEFAULT_EDGES if edges is None else edges)
        self.bins = {} if bins is None else bins
        self.global_bin = global_bin
        self.llr_clip = float(llr_clip)
        self.smooth = float(smooth)

    def _bin_key(self, range_m):
        if range_m is None or not math.isfinite(range_m) or range_m <= 0:
            return None
        for i in range(len(self.edges) - 1):
            if self.edges[i] <= range_m < self.edges[i + 1]:
                hi = 'inf' if self.edges[i + 1] > 1e6 else f'{self.edges[i + 1]:g}'
                return f'{self.edges[i]:g}-{hi}'
        return None

    def fit(self, scores, labels, ranges_m, min_bin_samples=30):
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        ranges_m = [None if r is None else float(r) for r in ranges_m]
        prior = (float(labels.sum()) + 0.5) / (len(labels) + 1.0)
        gx, gy = pava_isotonic(scores, labels, smooth=self.smooth)
        self.global_bin = {'prior': prior, 'x': gx, 'y': gy, 'n': int(len(scores))}
        self.bins = {}
        grouped = {}
        for s, y, r in zip(scores, labels, ranges_m):
            k = self._bin_key(r)
            if k is not None:
                grouped.setdefault(k, ([], []))
                grouped[k][0].append(s)
                grouped[k][1].append(y)
        for k, (bs, by) in grouped.items():
            if len(bs) < min_bin_samples:
                continue
            bs_a = np.asarray(bs, dtype=np.float64)
            by_a = np.asarray(by, dtype=np.float64)
            bprior = (float(by_a.sum()) + 0.5) / (len(by_a) + 1.0)
            kx, ky = pava_isotonic(bs_a, by_a, smooth=self.smooth)
            self.bins[k] = {'prior': bprior, 'x': kx, 'y': ky, 'n': int(len(bs_a))}
        return self

    def probability_and_llr(self, score, range_m=None):
        if self.global_bin is None:
            raise RuntimeError('Calibrator is not fitted')
        key = self._bin_key(range_m)
        info = self.bins.get(key, self.global_bin)
        p = float(np.interp(float(score), info['x'], info['y']))
        prior = float(info['prior'])
        llr = _logit(p) - _logit(prior)
        llr = float(np.clip(llr, -self.llr_clip, self.llr_clip))
        return p, llr, (key if key in self.bins else 'global')

    def to_dict(self):
        return {
            'edges': self.edges,
            'llr_clip': self.llr_clip,
            'smooth': self.smooth,
            'global_bin': self.global_bin,
            'bins': self.bins,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(edges=data['edges'], bins=data['bins'], global_bin=data['global_bin'],
                   llr_clip=data['llr_clip'], smooth=data.get('smooth', 2.0))


def _camera_Rt(meta):
    R = Rotation.from_quat([meta['rot_x'], meta['rot_y'], meta['rot_z'], meta['rot_w']]).as_matrix()
    t = np.array([meta['pos_x'], meta['pos_y'], meta['pos_z']], dtype=np.float64)
    return R, t


class SpatialTracker:
    """Inter-frame candidate consistency using only consecutive relative camera pose + depth."""

    def __init__(self, image_size=(1920, 1080), horizontal_fov_deg=90.0,
                 max_ray_angle_deg=10.0, box_margin_frac=0.1, max_gap_s=2.5):
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.focal = self.width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
        self.max_ray_angle_deg = float(max_ray_angle_deg)
        self.box_margin_frac = float(box_margin_frac)
        self.max_gap_s = float(max_gap_s)
        self.prev = None  # (R_prev, t_prev, p_cam_prev, time_s)
        self.streak = 0

    def reset(self):
        self.prev = None
        self.streak = 0

    def _ray_cam(self, point_xy):
        u = float(point_xy[0]) * self.width
        v = float(point_xy[1]) * self.height
        ray = np.array([1.0, (u - self.width / 2.0) / self.focal, (v - self.height / 2.0) / self.focal], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def _project(self, p_cam):
        if p_cam[0] <= 0.1:
            return None
        return np.array([self.focal * (p_cam[1] / p_cam[0]) + self.width / 2.0,
                         self.focal * (p_cam[2] / p_cam[0]) + self.height / 2.0], dtype=np.float64)

    def update(self, point_xy, box_xyxy, depth_m, camera_meta, timestamp_s):
        if point_xy is None or box_xyxy is None or depth_m is None or not math.isfinite(depth_m) or depth_m <= 0.5:
            self.reset()
            return {'consistent': False, 'streak': 0, 'ray_ang_deg': None, 'reproj_in_box': False, 'reason': 'invalid_input'}
        R_cur, t_cur = _camera_Rt(camera_meta)
        ray_cur = self._ray_cam(point_xy)
        p_cam_cur = ray_cur * float(depth_m)
        if self.prev is None:
            self.prev = (R_cur, t_cur, p_cam_cur, timestamp_s)
            self.streak = 1
            return {'consistent': True, 'streak': 1, 'ray_ang_deg': 0.0, 'reproj_in_box': True, 'reason': 'init'}
        R_prev, t_prev, p_cam_prev, t_prev_s = self.prev
        gap = None if (timestamp_s is None or t_prev_s is None) else (timestamp_s - t_prev_s)
        if gap is not None and (gap < 0 or gap > self.max_gap_s):
            self.prev = (R_cur, t_cur, p_cam_cur, timestamp_s)
            self.streak = 1
            return {'consistent': True, 'streak': 1, 'ray_ang_deg': None, 'reproj_in_box': True, 'reason': 'gap_reset'}
        # Strictly relative inter-frame transform T_{prev -> cur}: no world target coords used.
        delta_R = R_cur.T @ R_prev
        delta_t = R_cur.T @ (t_prev - t_cur)
        p_pred = delta_R @ p_cam_prev + delta_t
        pred_norm = float(np.linalg.norm(p_pred))
        if pred_norm <= 0.1:
            self.prev = (R_cur, t_cur, p_cam_cur, timestamp_s)
            self.streak = 1
            return {'consistent': False, 'streak': 0, 'ray_ang_deg': None, 'reproj_in_box': False, 'reason': 'behind_camera'}
        ray_pred = p_pred / pred_norm
        ray_ang_deg = math.degrees(math.acos(float(np.clip(np.dot(ray_cur, ray_pred), -1.0, 1.0))))
        uv_pred = self._project(p_pred)
        x0, y0, x1, y1 = map(float, box_xyxy)
        mx = (x1 - x0) * self.box_margin_frac
        my = (y1 - y0) * self.box_margin_frac
        in_box = bool(uv_pred is not None and (x0 - mx) <= uv_pred[0] <= (x1 + mx) and (y0 - my) <= uv_pred[1] <= (y1 + my))
        consistent = bool(in_box and ray_ang_deg <= self.max_ray_angle_deg)
        self.streak = (self.streak + 1) if consistent else 0
        self.prev = (R_cur, t_cur, p_cam_cur, timestamp_s)
        return {'consistent': consistent, 'streak': self.streak, 'ray_ang_deg': round(ray_ang_deg, 3),
                'reproj_in_box': in_box, 'reason': 'ok' if consistent else 'inconsistent'}


class GradedGate:
    """Three-stage graded motion permission and SPRT identity confirmation gate."""

    DEFAULT_CONFIG = {
        'mid_range_m': 80.0,
        'near_handoff_range_m': 45.0,
        'far_consistent_frames': 2,
        'far_min_score': 0.50,
        'far_llr_threshold': -0.2,
        'far_speed_cap_mps': 1.5,
        'mid_speed_cap_mps': 1.0,
        'confirm_max_range_m': 80.0,
        'confirm_llr_threshold': 2.5,
        'confirm_min_mid_frames': 3,
        'reject_llr_threshold': -3.0,
        'llr_decay': 0.92,
        'min_evidence_move_m': 2.0,
        'stationary_llr_weight': 0.2,
        'max_inconsistent_frames': 8,
        'require_same_class_margin': False,
        'min_same_class_logit_margin': 0.0,
        'max_ray_angle_deg': 12.0,
        'box_margin_frac': 0.15,
        'max_gap_s': 2.5,
    }

    def __init__(self, calibrator: DistanceBinCalibrator, config=None):
        self.calibrator = calibrator
        self.cfg = dict(self.DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)
        self.tracker = SpatialTracker(
            max_ray_angle_deg=self.cfg['max_ray_angle_deg'],
            box_margin_frac=self.cfg['box_margin_frac'],
            max_gap_s=self.cfg['max_gap_s'],
        )
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.llr_sum = 0.0
        self.mid_llr_sum = 0.0
        self.mid_positive_frames = 0
        self.inconsistent_streak = 0
        self.identity_confirmed = False
        self.identity_rejected = False
        self.margin_history = []
        self.last_evidence_pos = None
        self.steps = 0

    def _evidence_weight(self, camera_meta):
        if camera_meta is None or 'pos_x' not in camera_meta:
            return 1.0
        pos = np.array([camera_meta['pos_x'], camera_meta['pos_y'], camera_meta['pos_z']], dtype=np.float64)
        if self.last_evidence_pos is None:
            self.last_evidence_pos = pos
            return 1.0
        if float(np.linalg.norm(pos - self.last_evidence_pos)) >= self.cfg['min_evidence_move_m']:
            self.last_evidence_pos = pos
            return 1.0
        return float(self.cfg['stationary_llr_weight'])

    def update(self, score, point_xy, box_xyxy, depth_m, camera_meta, timestamp_s,
               same_class_other_scores=None):
        self.steps += 1
        track = self.tracker.update(point_xy, box_xyxy, depth_m, camera_meta, timestamp_s)
        if score is None or not math.isfinite(score):
            self.inconsistent_streak += 1
            return self._output('HOLD', False, 0.0, track, None, None, None, 'missing_score')
        prob, raw_llr, cal_bin = self.calibrator.probability_and_llr(score, depth_m)
        weight = self._evidence_weight(camera_meta) if track['consistent'] else 0.0
        llr_step = raw_llr * weight
        in_mid_or_closer = depth_m is not None and depth_m <= self.cfg['confirm_max_range_m']

        if track['consistent']:
            self.inconsistent_streak = 0
            self.llr_sum = self.cfg['llr_decay'] * self.llr_sum + llr_step
            if in_mid_or_closer:
                self.mid_llr_sum = self.cfg['llr_decay'] * self.mid_llr_sum + llr_step
                if raw_llr > 0:
                    self.mid_positive_frames += 1
        else:
            self.inconsistent_streak += 1
            self.llr_sum = self.cfg['llr_decay'] * self.llr_sum
            self.mid_llr_sum = self.cfg['llr_decay'] * self.mid_llr_sum

        margin = None
        if same_class_other_scores:
            best_other = max(float(s) for s in same_class_other_scores)
            margin = _logit(score) - _logit(best_other)
            if track['consistent'] and in_mid_or_closer:
                self.margin_history.append(margin)
                if len(self.margin_history) > 10:
                    self.margin_history.pop(0)
        mean_margin = float(np.mean(self.margin_history)) if self.margin_history else None

        # SPRT identity confirmation is strictly gated to mid-range (<= confirm_max_range_m)
        margin_ok = True
        if self.cfg['require_same_class_margin']:
            margin_ok = (mean_margin is not None and mean_margin >= self.cfg['min_same_class_logit_margin'])
        if (not self.identity_rejected and in_mid_or_closer and
                self.mid_llr_sum >= self.cfg['confirm_llr_threshold'] and
                self.mid_positive_frames >= self.cfg['confirm_min_mid_frames'] and
                margin_ok):
            self.identity_confirmed = True
        if self.llr_sum <= self.cfg['reject_llr_threshold'] or self.mid_llr_sum <= self.cfg['reject_llr_threshold']:
            self.identity_rejected = True
            self.identity_confirmed = False

        if self.identity_rejected:
            return self._output('REJECTED', False, 0.0, track, prob, llr_step, margin, 'identity_rejected', cal_bin, mean_margin)
        if self.inconsistent_streak >= self.cfg['max_inconsistent_frames']:
            return self._output('HOLD', False, 0.0, track, prob, llr_step, margin, 'spatial_inconsistent', cal_bin, mean_margin)

        in_near = depth_m is not None and depth_m <= self.cfg['near_handoff_range_m']
        in_mid = depth_m is not None and self.cfg['near_handoff_range_m'] < depth_m <= self.cfg['mid_range_m']

        if in_near:
            if self.identity_confirmed and track['consistent']:
                return self._output('NEAR_ADMITTED', True, self.cfg['mid_speed_cap_mps'], track, prob, llr_step, margin, 'confirmed_near_handoff', cal_bin, mean_margin)
            return self._output('HOLD', False, 0.0, track, prob, llr_step, margin, 'unconfirmed_at_near_boundary', cal_bin, mean_margin)

        if in_mid:
            if track['consistent'] and track['streak'] >= self.cfg['far_consistent_frames']:
                if self.identity_confirmed:
                    return self._output('MID_CONFIRMED', True, self.cfg['mid_speed_cap_mps'], track, prob, llr_step, margin, 'mid_confirmed', cal_bin, mean_margin)
                if score >= self.cfg['far_min_score'] and self.llr_sum >= self.cfg['far_llr_threshold']:
                    return self._output('MID_OBSERVE', True, self.cfg['mid_speed_cap_mps'], track, prob, llr_step, margin, 'mid_cautious_approach', cal_bin, mean_margin)
            return self._output('HOLD', False, 0.0, track, prob, llr_step, margin, 'mid_hold', cal_bin, mean_margin)

        # Far stage (> mid_range_m or unknown depth)
        if (track['consistent'] and track['streak'] >= self.cfg['far_consistent_frames'] and
                score >= self.cfg['far_min_score'] and self.llr_sum >= self.cfg['far_llr_threshold']):
            return self._output('FAR_APPROACH', True, self.cfg['far_speed_cap_mps'], track, prob, llr_step, margin, 'far_consistent_approach', cal_bin, mean_margin)
        return self._output('HOLD', False, 0.0, track, prob, llr_step, margin, 'far_observing', cal_bin, mean_margin)

    def _output(self, phase, motion_permitted, speed_cap_mps, track, prob, llr_step, margin, reason, cal_bin=None, mean_margin=None):
        return {
            'phase': phase,
            'motion_permitted': bool(motion_permitted),
            'speed_cap_mps': float(speed_cap_mps),
            'identity_confirmed': bool(self.identity_confirmed),
            'identity_rejected': bool(self.identity_rejected),
            'near_handoff_permitted': bool(phase == 'NEAR_ADMITTED'),
            'consistent': bool(track['consistent']),
            'consistent_streak': int(track['streak']),
            'ray_ang_deg': track['ray_ang_deg'],
            'reproj_in_box': bool(track['reproj_in_box']),
            'calibrated_prob': None if prob is None else round(float(prob), 6),
            'llr_step': None if llr_step is None else round(float(llr_step), 4),
            'llr_sum': round(float(self.llr_sum), 4),
            'mid_llr_sum': round(float(self.mid_llr_sum), 4),
            'cal_bin': cal_bin,
            'same_class_margin': None if margin is None else round(float(margin), 4),
            'same_class_mean_margin': None if mean_margin is None else round(float(mean_margin), 4),
            'reason': reason,
        }
