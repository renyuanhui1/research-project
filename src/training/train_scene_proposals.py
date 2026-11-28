"""Paired crossfit comparison of original and instance-balanced scene supervision."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from perception.template_matcher import TemplateMatcher
from training.train_template_matcher import load_features, fitting_statistics, train_localizer
from common.artifacts import read, write, digest
from perception.proposal_utils import (peak_candidates, point_hit, crop_coverage,
                                       bin_name, grouped, coverage_summary)


def balanced_targets(data, queries, allowed):
    """Equal mass per visible instance; query masks and valid regions stay unchanged."""
    targets = data['mask'].clone(); counts = Counter()
    for qi in queries:
        parts = [data['mask'][qi]]
        for tid in data.get('context_targets', {}).get(qi, []):
            if tid in allowed:
                parts.append(data['context_mask'][qi, tid]); counts['extra_instances_used'] += 1
            else:
                counts['extra_instances_excluded'] += 1
        distribution = torch.zeros_like(parts[0]); visible = 0
        for mask in parts:
            mask = mask.masked_fill(data['valid'][qi] <= 0, 0)
            mass = mask.sum()
            if mass > 0:
                distribution += mask / mass; visible += 1
        targets[qi] = distribution / visible if visible else distribution
        counts['queries'] += 1; counts['visible_instances'] += visible
    return targets, dict(counts)


def portable_fold(plan):
    return {key: [int(v) if key in ['fit_queries', 'held_queries'] else str(v) for v in values]
            for key, values in plan.items()}


@torch.no_grad()
def candidate_predictions(model, data, indices, mean, std, cfg, crop_cfg):
    model.eval(); result = {}; width = cfg['batch']
    for start in range(0, len(indices), width):
        selected = indices[start:start + width]
        q = (data['z'][selected].float() - mean) / std
        if len(q) < width:
            q = torch.cat([q, q[-1:].expand(width - len(q), -1, -1)])
        with torch.autocast(q.device.type, dtype=torch.bfloat16, enabled=q.is_cuda):
            logits, _ = model(q, torch.zeros_like(q))
        for index, response in zip(selected, logits[:len(selected)].cpu().numpy()):
            m = data['records'][index]['camera_metadata']
            result[index] = peak_candidates(response, (m['height'], m['width']), crop_cfg,
                                             max(cfg['candidate_counts']), cfg['nms_iou'])
    return result


def score_predictions(data, context, labels, candidates, query_folds, cfg):
    lookup = {r['name']: i for i, r in enumerate(data['records'])}; rows = []; frames = []
    for sample in context['samples']:
        qi = lookup[sample['sample_name']]; record = data['records'][qi]
        main = data['groups'][record['target_id']]; ids = labels[sample['sample_name']]['ids'][()]
        requests = [{'target_id': main['id'], 'index': main['index']}] + sample['label_targets']
        frame_rows = []; peaks = candidates[qi]; held = set(query_folds[qi]['held_groups'])
        for target in requests:
            mask = ids == target['index']
            if not mask.any():
                continue
            tid = target['target_id']
            row = {'sample_name': sample['sample_name'], 'query_target_id': main['id'], 'requested_target_id': tid,
                   'role': 'original' if tid == main['id'] else 'context',
                   'request_scope': 'held' if data['groups'][tid]['appearance_group'] in held else 'fit',
                   'visible_pixels': int(mask.sum()), 'area_bin': bin_name(float(mask.mean()), cfg['area_fraction_bins']),
                   'point_hits': [point_hit(mask, peak['point_xy']) for peak in peaks],
                   'crop_coverages': [crop_coverage(mask, peak['box_xyxy'], record['rgb_shape_hw']) for peak in peaks]}
            rows.append(row); frame_rows.append(row)
        frames.append({'sample_name': sample['sample_name'], 'candidates': peaks,
                       'all_points_covered': {str(k): all(any(r['point_hits'][:k]) for r in frame_rows)
                                              for k in cfg['candidate_counts']}})
    reduce = lambda values: coverage_summary(values, cfg['candidate_counts'], cfg['minimum_crop_coverage'])
    metrics = {'summary': reduce(rows), 'by_role': grouped(rows, 'role', reduce),
               'by_request_scope': grouped(rows, 'request_scope', reduce),
               'by_role_and_area': {role: grouped([r for r in rows if r['role'] == role], 'area_bin', reduce)
                                    for role in sorted({r['role'] for r in rows})},
               'context_by_request_scope': grouped([r for r in rows if r['role'] == 'context'], 'request_scope', reduce),
               'frames': len(frames), 'all_points_covered': {str(k): sum(f['all_points_covered'][str(k)] for f in frames)
                                                          for k in cfg['candidate_counts']}}
    return metrics, rows, frames


def aggregate(seed_results, counts):
    result = {}
    for role in ['original', 'context']:
        result[role] = {}
        for k in counts:
            result[role][str(k)] = {}
            for metric in ['point_recall', 'crop_recall']:
                values = [r['by_role'][role][str(k)][metric] for r in seed_results]
                result[role][str(k)][metric] = {'values': values, 'mean': float(np.mean(values)),
                                               'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.}
    return result


def train(cfg, evaluation, out, config_path):
    started = time.monotonic(); paths = sorted(project_path(evaluation['proposal_directory']).glob('proposal-fold-*/model.pt'))
    if not paths:
        raise FileNotFoundError('Existing crossfit proposal weights not found')
    # These project-generated checkpoints contain NumPy fold-index scalars.
    states = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    plans = [portable_fold(state['fold']) for state in states]
    base_cfg = dict(states[0]['configuration'], **{key: evaluation[key] for key in
                    ['catalog', 'capture_directory', 'feature_directory', 'protocol', 'context_labels']})
    data = load_features(base_cfg, 'cuda', ('train', 'validation'))
    seen = [i for plan in plans for i in plan['held_queries']]
    if sorted(seen) != sorted(data['split']['train']) or len(seen) != len(set(seen)):
        raise ValueError('Saved folds do not partition these training queries')
    for plan in plans:
        fit = set(plan['fit_queries']); held = set(plan['held_queries'])
        if fit & held or fit | held != set(data['split']['train']):
            raise ValueError('Saved fold query exclusion is inconsistent')
        if any(data['groups'][tid]['appearance_group'] in plan['held_groups'] for tid in plan['fit_template_ids']):
            raise ValueError('A held appearance was included in fitting targets')
    context_path = project_path(evaluation['context_labels']); context = read(context_path / 'report.json')
    lookup = {r['name']: i for i, r in enumerate(data['records'])}; wanted = {lookup[s['sample_name']] for s in context['samples']}
    bundle_path = project_path(evaluation['baseline_directory']) / 'verifier.pt'
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    registration = {'configuration': cfg, 'evaluation_configuration': evaluation,
                    'configuration_sha256': digest(config_path), 'feature_index_sha256': data['feature_index_sha256'],
                    'context_report_sha256': digest(context_path / 'report.json'), 'context_labels_sha256': digest(context_path / 'labels.h5'),
                    'source_sha256': {str(path): digest(ROOT / path) for path in
                      ['src/training/train_scene_proposals.py', 'src/training/train_template_matcher.py',
                       'src/perception/template_matcher.py', 'src/common/artifacts.py',
                       'src/perception/proposal_utils.py']},
                    'fold_sha256': {path.parent.name: digest(path) for path in paths},
                    'checkpoint_sha256': digest(bundle_path), 'raw_sha256': data['raw_sha256'],
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'environment': {'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__,
                                    'gpu': torch.cuda.get_device_name()},
                    'training_budget': {'epochs': cfg['epochs'], 'batch': base_cfg['batch'], 'learning_rate': base_cfg['learning_rate'],
                                        'weight_decay': base_cfg['weight_decay'], 'folds': len(plans),
                                        'seeds': cfg['seeds'], 'variants': cfg['variants']},
                    'scope': cfg['scope'], 'selection': 'Fixed final epoch; no validation selection or threshold fitting.'}
    write(out / 'pre-registration.json', registration)
    balanced = []; statistics = []; supervision_audit = []
    for i, (plan, state) in enumerate(zip(plans, states)):
        subset = dict(data, split={'train': plan['fit_queries']})
        mean, std = fitting_statistics(subset); statistics.append((mean, std))
        torch.testing.assert_close(mean.cpu(), state['mean'], rtol=0, atol=0)
        torch.testing.assert_close(std.cpu(), state['std'], rtol=0, atol=0)
        targets, audit = balanced_targets(data, plan['fit_queries'], set(plan['fit_template_ids']))
        balanced.append(targets); supervision_audit.append(dict(fold=i, plan=plan, **audit))
    write(out / 'supervision-audit.json', supervision_audit)
    all_results = {}; reproduction = []; artifacts = {}
    query_folds = {qi: plan for plan in plans for qi in plan['held_queries']}
    with h5py.File(context_path / 'labels.h5', 'r') as labels:
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                candidates = {}; folder = out / ('seed-' + str(seed)) / variant; folder.mkdir(parents=True, exist_ok=True)
                for fold, (plan, state) in enumerate(zip(plans, states)):
                    fold_dir = folder / ('fold-' + str(fold)); fold_dir.mkdir(exist_ok=True)
                    parameters = dict(base_cfg, seed=seed, epochs=cfg['epochs'])
                    variant_masks = {'original': data['mask'], 'all_balanced': balanced[fold]}
                    subset = dict(data, mask=variant_masks[variant],
                                  split={'train': plan['fit_queries']}, training_template_ids=plan['fit_template_ids'])
                    pairs = {'train': [{'query': qi, 'template': data['targets'].index(data['records'][qi]['target_id']),
                                        'positive': True, 'target_id': data['records'][qi]['target_id']}
                                       for qi in plan['fit_queries']]}
                    mean, std = statistics[fold]
                    print('TRAIN_START', seed, variant, fold, flush=True)
                    model, selection = train_localizer(False, subset, pairs, mean, std, parameters, fold_dir, select=False)
                    if seed == state['configuration']['seed'] and variant == 'original':
                        delta = max(float((value.cpu() - state['model'][key]).abs().max()) for key, value in model.state_dict().items())
                        reproduction.append({'fold': fold, 'maximum_parameter_difference': delta})
                    checkpoint = {'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                  'mean': mean.cpu(), 'std': std.cpu(), 'configuration': parameters,
                                  'fold': plan, 'variant': variant, 'selection': selection}
                    torch.save(checkpoint, fold_dir / 'model.pt'); artifacts[str((fold_dir / 'model.pt').relative_to(out))] = digest(fold_dir / 'model.pt')
                    selected = [qi for qi in plan['held_queries'] if qi in wanted]
                    predict_cfg = dict(parameters, **{key: evaluation[key] for key in ['candidate_counts', 'nms_iou']})
                    candidates.update(candidate_predictions(model, data, selected, mean, std, predict_cfg, bundle['configuration']))
                    print('TRAIN_DONE', seed, variant, fold, flush=True)
                if set(candidates) != wanted:
                    raise ValueError('Crossfit context predictions are incomplete')
                metrics, rows, frames = score_predictions(data, context, labels, candidates, query_folds, evaluation)
                write(folder / 'metrics.json', metrics); write(folder / 'coverage-records.json', rows); write(folder / 'candidates.json', frames)
                all_results.setdefault(variant, {})[str(seed)] = metrics
                print('VARIANT_RESULT', seed, variant, json.dumps(metrics['by_role']), flush=True)
    aggregates = {variant: aggregate(list(values.values()), evaluation['candidate_counts']) for variant, values in all_results.items()}
    deltas = {}
    if 'original' in all_results and 'all_balanced' in all_results:
        for role in ['original', 'context']:
            deltas[role] = {}
            for k in evaluation['candidate_counts']:
                values = [100 * (all_results['all_balanced'][str(seed)]['by_role'][role][str(k)]['point_recall'] -
                                 all_results['original'][str(seed)]['by_role'][role][str(k)]['point_recall']) for seed in cfg['seeds']]
                deltas[role][str(k)] = {'paired_delta_pp': values, 'mean': float(np.mean(values)),
                                       'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.}
    write(out / 'comparison.json', {'per_seed': all_results, 'aggregate': aggregates, 'paired_point_deltas': deltas,
                                   'reference_reproduction': reproduction, 'seconds': time.monotonic() - started,
                                   'scope': cfg['scope'], 'uncertainty': 'Mean and sample SD across paired seeds, not pooled independent frames.'})
    write(out / 'manifest.json', {'model_sha256': artifacts, 'pre_registration_sha256': digest(out / 'pre-registration.json'),
                                'context_frames': len(wanted), 'code_commit': registration['git_commit']})
    print('COMPARISON_READY', json.dumps(deltas), flush=True)


def plot(cfg, evaluation, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = read(out / 'comparison.json'); counts = evaluation['candidate_counts']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, role in zip(axes, ['original', 'context']):
        x = np.arange(len(counts)); width = .35
        for j, variant in enumerate(cfg['variants']):
            values = [result['aggregate'][variant][role][str(k)]['point_recall'] for k in counts]
            ax.bar(x + (j - .5) * width, [v['mean'] for v in values], width=width,
                   yerr=[v['std'] for v in values], capsize=3, label=variant)
        ax.set_xticks(x, ['K=' + str(k) for k in counts]); ax.set(title=role, ylim=(0, 1), ylabel='Peak coverage')
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Paired crossfit training-context comparison; mean and SD across seeds')
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/airport_scene_proposals.json')
    parser.add_argument('--plot', action='store_true'); args = parser.parse_args()
    path = project_path(args.config); cfg = read(path); evaluation = read(project_path(cfg['evaluation_config']))
    out = project_path(cfg['output_directory']); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    if args.plot:
        plot(cfg, evaluation, out)
    else:
        if (out / 'comparison.json').exists():
            raise FileExistsError('Completed paired experiment already exists')
        train(cfg, evaluation, out, path)


if __name__ == '__main__':
    main()
