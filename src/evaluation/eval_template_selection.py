"""Diagnose frozen verification scores and measure image-only top-K proposal coverage."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from common.patches import read_rgb
from common.view_geometry import redact_rgb
from common.artifacts import read, write, digest
from perception.proposal_utils import (ratio, bin_name, point_hit, crop_coverage,
                                       grouped, box_iou, peak_candidates, coverage_summary)
from perception.template_verification import FEATURE_NAMES, proposal_box, rgb_descriptors, square_rgb


def auc(scores, positive):
    scores = np.asarray(scores, float)
    positive = np.asarray(positive, bool)
    n = int(positive.sum()); m = len(positive) - n
    return float((rankdata(scores)[positive].sum() - n * (n + 1) / 2) / (n * m)) if n and m else None


def wilson(correct, count):
    if not count:
        return None
    z = 1.959963984540054; p = correct / count; denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def quantiles(values):
    return np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None


def summarize(rows, histogram_bins):
    visible = [r for r in rows if r['positive'] and r['visible_pixels'] > 0]
    negatives = [r for r in rows if not r['positive']]
    accepted = [r for r in rows if r['accepted']]
    correct = sum(r['verified_correct'] for r in accepted)
    p = [r['score'] for r in visible]; n = [r['score'] for r in negatives]
    overlap = None
    if p and n:
        hp, _ = np.histogram(p, bins=histogram_bins, range=(0, 1))
        hn, _ = np.histogram(n, bins=histogram_bins, range=(0, 1))
        overlap = float(np.minimum(hp / len(p), hn / len(n)).sum())
    return {'pairs': len(rows), 'visible_positive': len(visible), 'negative_pairs': len(negatives),
            'accepted': len(accepted), 'correct_accepted': correct,
            'accepted_precision': ratio(correct, len(accepted)), 'precision_wilson_95': wilson(correct, len(accepted)),
            'verified_positive_recall': ratio(correct, len(visible)),
            'negative_false_accept_rate': ratio(sum(r['accepted'] for r in negatives), len(negatives)),
            'identity_auc': auc(p + n, [True] * len(p) + [False] * len(n)),
            'verified_auc': auc([r['score'] for r in rows], [r['verified_correct'] for r in rows]),
            'positive_score_q10_q50_q90': quantiles(p), 'negative_score_q10_q50_q90': quantiles(n),
            'histogram_overlap': overlap,
            'positive_rgb_inliers_q10_q50_q90': quantiles([r['rgb_inliers'] for r in visible if 'rgb_inliers' in r]),
            'positive_rgb_inlier_fraction_q10_q50_q90': quantiles([r['rgb_inlier_fraction'] for r in visible])}


def diagnose(cfg, out):
    import torch
    base = project_path(cfg['baseline_directory']); catalog = read(project_path(cfg['catalog']))
    targets = {t['id']: t for t in catalog['targets']}; selection = read(base / 'selection.json')
    mode = selection['selected_method']; saved = read(base / 'validation-records.json')[mode]
    pairs = read(base / 'pair-plan.json'); features = np.load(base / 'verification-features.npz')
    bundle = torch.load(base / 'verifier.pt', map_location='cpu', weights_only=True)
    head = torch.nn.Linear(len(bundle['feature_center']), 1); head.load_state_dict(bundle['confidence'][mode]); head.eval()
    with torch.no_grad():
        values = (features['train'] - bundle['feature_center'].numpy()) / bundle['feature_scale'].numpy()
        train_scores = head(torch.from_numpy(values)).reshape(-1).double().sigmoid().numpy()
    rows = []; source_hashes = {}; rgb_reference_counts = {}; rgb_query_counts = {}; metadata = {}
    raw_root = project_path(cfg['capture_directory']); rgb_offset = 27
    names = read(base / 'manifest.json')['feature_names']
    assert names[rgb_offset:] == ['rgb_' + name for name in FEATURE_NAMES]
    allowed = set(catalog['splits'][cfg['protocol']]['train'] + catalog['splits'][cfg['protocol']]['validation'])
    query_names = [s['name'] for t in catalog['targets'] if t['id'] in allowed
                   for s in read(raw_root / t['id'] / 'report.json')['samples']]
    with ExitStack() as stack:
        stores = {}
        for tid in sorted({r['target_id'] for part in pairs.values() for r in part}):
            report = read(raw_root / tid / 'report.json')
            stores[tid] = stack.enter_context(h5py.File(raw_root / tid / 'samples.h5', 'r'))
            source_hashes[tid] = digest(raw_root / tid / 'samples.h5')
            if source_hashes[tid] != report['samples_h5_sha256']:
                raise ValueError('Capture changed: ' + tid)
            metadata.update({s['name']: s for s in report['samples']})
        proposals = np.load(base / 'proposals.npz')
        for partition in ['train', 'validation']:
            assert len(pairs[partition]) == len(features[partition])
            for i, pair in enumerate(pairs[partition]):
                if partition == 'validation':
                    row = dict(saved[i]); name = row['sample_name']; score = row['score']
                    if (row['query'], row['template'], row['positive']) != (pair['query'], pair['template'], pair['positive']):
                        raise ValueError('Saved validation pairing changed')
                else:
                    # Original feature order is catalogue order, train + validation only.
                    row = dict(pair); score = float(train_scores[i])
                    name = query_names[pair['query']]
                    row.update(sample_name=name, score=score, point_xy=proposals['points'][pair['query']].tolist(),
                               box_xyxy=proposals['boxes'][pair['query']].tolist(),
                               accepted=bool(score >= bundle['comparison'][mode]['threshold']))
                sample = metadata[name]; g = stores[pair['target_id']][name]
                counts = sample['mask_pixels']; area = counts['assembly_mask'] / np.prod(sample['depth_shape_hw'])
                correct = bool(features[partition + '_correct'][i])
                if partition == 'validation' and correct != row['verified_correct']:
                    raise ValueError('Saved labels changed')
                vector = features[partition][i]
                row.update(partition=partition, category=sample['category'], range_stratum=sample['range_stratum'],
                           horizontal_distance_m=sample['horizontal_distance_m'], visible_pixels=counts['assembly_mask'],
                           visible_area_fraction=float(area), area_bin=bin_name(area, cfg['area_fraction_bins']),
                           occluded_fraction=ratio(counts['occluded_mask'], counts['geometry_mask']),
                           occlusion_group='occluded' if counts['occluded_mask'] else 'no_foreground_occlusion',
                           verified_correct=correct, rgb_inlier_fraction=float(vector[rgb_offset + 8]),
                           rgb_mutual_fraction=float(vector[rgb_offset + 3]),
                           rgb_query_coverage=float(vector[rgb_offset + 11]),
                           rgb_template_coverage=float(vector[rgb_offset + 12]),
                           part_pixels=counts['part_mask'],
                           part_equivalent_input_pixels=counts['part_mask'] * bundle['configuration']['image_size'] ** 2 / np.prod(sample['depth_shape_hw']))
                if partition == 'validation':
                    mask = g['assembly_mask'][()].astype(bool)
                    if bool(row['positive'] and point_hit(mask, row['point_xy'])) != correct:
                        raise ValueError('Native-label replay changed')
                    row['crop_target_coverage'] = crop_coverage(mask, row['box_xyxy'], sample['rgb_shape_hw'])
                    tid = row['target_id'] if row['positive'] else row['negative_template_id']
                    if tid not in rgb_reference_counts:
                        reference = cv2.imread(str(project_path(targets[tid]['template']['file'])))[:, :, ::-1].copy()
                        rgb_reference_counts[tid] = len(rgb_descriptors(reference)[0])
                    if name not in rgb_query_counts:
                        image = redact_rgb(read_rgb(stores[row['target_id']], name), g['ignored_mask'][()])
                        x0, y0, x1, y1 = row['box_xyxy']
                        crop = square_rgb(image[y0:y1, x0:x1], bundle['configuration']['image_size'])
                        rgb_query_counts[name] = len(rgb_descriptors(crop)[0])
                    # The saved inlier/grid fraction uses min(query, template descriptor counts).
                    row['rgb_inliers'] = int(round(float(vector[rgb_offset + 9]) * min(rgb_query_counts[name], rgb_reference_counts[tid])))
                rows.append(row)
    result = {'selected_method': mode, 'threshold': bundle['comparison'][mode]['threshold'], 'partitions': {},
              'score_scope': {'validation': 'Repeatedly used development validation; no threshold changes.',
                              'train': 'Coarse proposals are query-group crossfit; acceptance head is fitted on these same features. Scores are NOT held-out head predictions.'},
              'part_area_scope': 'Task-part mask area only; NOT a damage-difference annotation or evidence that damage identity is visually resolvable.',
              'rgb_count_scope': 'Integer counts recovered from saved inlier/grid fractions and SIFT descriptor counts; no DINO inference.',
              'raw_sha256': source_hashes}
    for partition in ['train', 'validation']:
        subset = [r for r in rows if r['partition'] == partition]
        reduce = lambda r: summarize(r, cfg['score_histogram_bins'])
        result['partitions'][partition] = dict(summary=reduce(subset))
        for key in ['target_id', 'range_stratum', 'area_bin', 'occlusion_group']:
            result['partitions'][partition]['by_' + key] = grouped(subset, key, reduce)
    summary = result['partitions']['validation']['summary']
    for actual, expected in [('accepted', 'accepted'), ('correct_accepted', 'accepted_verified_correct'),
                             ('visible_positive', 'visible_positive_pairs')]:
        if summary[actual] != selection['validation'][expected]:
            raise ValueError('Frozen validation summary changed: ' + actual)
    write(out / 'diagnosis-records.json', rows); write(out / 'diagnosis.json', result)
    print('DIAGNOSIS', json.dumps(result['partitions']['validation'], ensure_ascii=False), flush=True)
    return rows


def coverage(cfg, out):
    import torch
    from perception.template_matcher import TemplateMatcher
    from training.train_template_matcher import load_features
    base = project_path(cfg['baseline_directory']); bundle = torch.load(base / 'verifier.pt', map_location='cpu', weights_only=True)
    parameters = bundle['coarse']['configuration']; parameters = dict(parameters, **{k: cfg[k] for k in ['catalog', 'capture_directory', 'feature_directory', 'protocol']}, context_labels=None)
    data = load_features(parameters, 'cuda', ('train', 'validation'))
    context_root = project_path(cfg['context_labels']); context = read(context_root / 'report.json')
    if digest(context_root / 'labels.h5') != context['labels_sha256'] or digest(project_path(cfg['catalog'])) != context['catalog_sha256']:
        raise ValueError('Context source changed')
    if any(data['raw_sha256'][tid] != sha for tid, sha in context['raw_sha256'].items()):
        raise ValueError('Context and feature captures differ')
    lookup = {r['name']: i for i, r in enumerate(data['records'])}; wanted = {lookup[s['sample_name']]: s for s in context['samples']}
    folder = project_path(cfg['proposal_directory']); saved = np.load(base / 'proposals.npz')
    candidates = {}; fold_hashes = {}; changed = Counter(); maximum = max(cfg['candidate_counts'])
    for path in sorted(folder.glob('proposal-fold-*/model.pt')):
        # Project-generated crossfit metadata contains NumPy scalar indices.
        state = torch.load(path, map_location='cuda', weights_only=False); fold = state['fold']; fold_hashes[str(path.relative_to(folder))] = digest(path)
        if set(fold['held_queries']) & set(fold['fit_queries']):
            raise ValueError('Crossfit query leakage')
        model = TemplateMatcher(conditioned=False, hidden_dim=parameters['hidden_dim'], output_grid=parameters['output_grid']).cuda()
        model.load_state_dict(state['model']); model.eval(); width = parameters['batch']
        with torch.no_grad():
            for begin in range(0, len(fold['held_queries']), width):
                indices = fold['held_queries'][begin:begin + width]
                q = (data['z'][indices].float() - state['mean']) / state['std']
                if len(q) < width:
                    q = torch.cat([q, q[-1:].expand(width - len(q), -1, -1)])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(q, torch.zeros_like(q))
                for index, heatmap in zip(indices, logits[:len(indices)].cpu().numpy()):
                    if index not in wanted:
                        continue
                    if index in candidates:
                        raise ValueError('Repeated crossfit query')
                    m = data['records'][index]['camera_metadata']; shape = (m['height'], m['width'])
                    crop_cfg = bundle['configuration']
                    peaks = peak_candidates(heatmap, shape, crop_cfg, maximum, cfg['nms_iou'])
                    changed['points'] += int(peaks[0]['point_xy'] != saved['points'][index].tolist())
                    changed['boxes'] += int(peaks[0]['box_xyxy'] != saved['boxes'][index].tolist())
                    if peaks[0]['box_xyxy'] != proposal_box(heatmap, shape, crop_cfg['crop_margin'], crop_cfg['crop_minimum_fraction'], crop_cfg['crop_logit_delta']):
                        raise ValueError('Top-1 crop differs from frozen image-only algorithm')
                    candidates[index] = peaks
        print('COVERAGE_FOLD', path.parent.name, 'context_frames', len(candidates), flush=True)
    if set(candidates) != set(wanted) or any(changed.values()):
        raise ValueError('Crossfit top-1 replay mismatch: ' + str(dict(changed)))
    rows = []; frames = []
    with h5py.File(context_root / 'labels.h5', 'r') as labels:
        for index, sample in wanted.items():
            record = data['records'][index]; main = data['groups'][record['target_id']]
            ids = labels[sample['sample_name']]['ids'][()]; peaks = candidates[index]
            requested = [{'target_id': main['id'], 'index': main['index']}] + sample['label_targets']
            current_rows = []
            for target in requested:
                mask = ids == target['index']
                if not mask.any():
                    continue
                area = float(mask.mean()); metadata = record['camera_metadata']; group = data['groups'][target['target_id']]
                distance = math.hypot(metadata['pos_x'] - group['reference_ned_m'][0], metadata['pos_y'] - group['reference_ned_m'][1])
                row = {'sample_name': sample['sample_name'], 'query_target_id': main['id'], 'requested_target_id': target['target_id'],
                       'role': 'original' if target['target_id'] == main['id'] else 'context', 'visible_pixels': int(mask.sum()),
                       'area_bin': bin_name(area, cfg['area_fraction_bins']), 'horizontal_distance_m': distance,
                       'point_hits': [point_hit(mask, peak['point_xy']) for peak in peaks],
                       'crop_coverages': [crop_coverage(mask, peak['box_xyxy'], record['rgb_shape_hw']) for peak in peaks]}
                rows.append(row); current_rows.append(row)
            frames.append({'sample_name': sample['sample_name'], 'candidates': peaks, 'visible_requests': len(current_rows),
                           'all_points_covered': {str(k): all(any(r['point_hits'][:k]) for r in current_rows) for k in cfg['candidate_counts']}})
    reduce = lambda r: coverage_summary(r, cfg['candidate_counts'], cfg['minimum_crop_coverage'])
    summary = {'summary': reduce(rows), 'by_role': grouped(rows, 'role', reduce), 'by_area_bin': grouped(rows, 'area_bin', reduce),
               'frames': len(frames), 'all_visible_points_covered': {str(k): sum(f['all_points_covered'][str(k)] for f in frames) for k in cfg['candidate_counts']},
               'top1_replay_changes': dict(changed), 'fold_sha256': fold_hashes,
               'scope': 'Training context queries with query-group-crossfit coarse models; shared cache may contain incidental held-out appearance backgrounds. No acceptance or independent generalization claim.'}
    write(out / 'coverage-records.json', rows); write(out / 'coverage-candidates.json', frames); write(out / 'coverage.json', summary)
    print('COVERAGE', json.dumps(summary, ensure_ascii=False), flush=True)


def plot_results(cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in read(out / 'diagnosis-records.json') if r['partition'] == 'validation']
    diagnosis = read(out / 'diagnosis.json'); targets = sorted({r['target_id'] for r in rows})
    fig, axes = plt.subplots(2, len(targets), figsize=(5 * len(targets), 7), squeeze=False)
    for column, target in enumerate(targets):
        subset = [r for r in rows if r['target_id'] == target]
        positive = [r for r in subset if r['positive'] and r['visible_pixels']]
        negative = [r for r in subset if not r['positive']]
        ax = axes[0, column]
        for values, label in [(positive, 'Present template'), (negative, 'Absent template')]:
            ax.hist([r['score'] for r in values], bins=cfg['score_histogram_bins'], range=(0, 1), density=True, alpha=.5, label=label)
        ax.axvline(diagnosis['threshold'], color='black', linestyle='--', label='Frozen threshold')
        ax.set(title=target, xlabel='Acceptance score', ylabel='Density'); ax.legend(fontsize=8)
        ax = axes[1, column]
        ax.boxplot([[r['rgb_inliers'] for r in values] for values in [positive, negative]])
        ax.set_xticks([1, 2], ['Present', 'Absent']); ax.set(ylabel='RGB affine inliers')
    fig.suptitle('Repeated development validation; score distributions and recovered RGB inlier counts')
    fig.tight_layout(); fig.savefig(out / 'score-diagnosis.png', dpi=160); plt.close(fig)
    result = read(out / 'coverage.json'); counts = cfg['candidate_counts']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), squeeze=False)
    groups = [('All visible objects', result['summary'])] + [(key, value) for key, value in result['by_role'].items()]
    for ax, (name, values) in zip(axes.flat, groups):
        x = np.arange(len(counts))
        ax.bar(x - .18, [values[str(k)]['point_recall'] for k in counts], width=.36, label='Peak in native mask')
        ax.bar(x + .18, [values[str(k)]['crop_recall'] for k in counts], width=.36,
               label=f"Crop covers >={cfg['minimum_crop_coverage']:.0%} target")
        ax.set_xticks(x, ['K=' + str(k) for k in counts]); ax.set(title=name, ylim=(0, 1), ylabel='Object coverage')
        ax.legend(fontsize=8)
    fig.suptitle('Training context labels; query-group crossfit proposals, not independent testing')
    fig.tight_layout(); fig.savefig(out / 'candidate-coverage.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evaluation/template_selection.json')
    parser.add_argument('--stage', choices=['diagnose', 'coverage', 'plot'], required=True)
    args = parser.parse_args(); cfg_path = project_path(args.config); cfg = read(cfg_path)
    out = project_path(cfg['output_directory']); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'plot':
        plot_results(cfg, out)
        return
    output = out / ('diagnosis.json' if args.stage == 'diagnose' else 'coverage.json')
    if output.exists():
        raise FileExistsError('Completed results already exist: ' + str(output))
    base = project_path(cfg['baseline_directory']); manifest = read(base / 'manifest.json')
    checkpoint_hash = digest(base / 'verifier.pt')
    if checkpoint_hash != manifest['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    registration = {'stage': args.stage, 'configuration': cfg, 'configuration_sha256': digest(cfg_path),
                    'source_sha256': digest(Path(__file__)), 'checkpoint_sha256': checkpoint_hash,
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(out / (args.stage + '-registration.json'), registration)
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(4)
    if args.stage == 'diagnose':
        diagnose(cfg, out)
    else:
        coverage(cfg, out)
    if digest(base / 'verifier.pt') != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation')


if __name__ == '__main__':
    main()
