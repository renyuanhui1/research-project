"""Fuse cached proposal ranks at a fixed budget and time shared-encoder inference."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
import json
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from common.patches import read_rgb
from common.view_geometry import redact_rgb
from evaluation.eval_template_selection import read, write, digest, box_iou, peak_candidates
from training.train_scene_proposals import score_predictions, aggregate

SOURCES = ['original', 'all_balanced']


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


def inputs(cfg):
    source_cfg = read(project_path(cfg['source_config']))
    evaluation = read(project_path(source_cfg['evaluation_config']))
    source = project_path(source_cfg['output_directory']); catalog = read(project_path(evaluation['catalog']))
    context = read(project_path(evaluation['context_labels']) / 'report.json')
    parent = read(source / 'pre-registration.json')
    if digest(project_path(evaluation['catalog'])) != context['catalog_sha256']:
        raise ValueError('Target catalog changed')
    if digest(project_path(evaluation['context_labels']) / 'labels.h5') != parent['context_labels_sha256']:
        raise ValueError('Context labels changed')
    if digest(project_path(evaluation['context_labels']) / 'report.json') != parent['context_report_sha256']:
        raise ValueError('Context metadata changed')
    allowed = set(catalog['splits'][evaluation['protocol']]['train'] + catalog['splits'][evaluation['protocol']]['validation'])
    records = [sample for target in catalog['targets'] if target['id'] in allowed
               for sample in read(project_path(evaluation['capture_directory']) / target['id'] / 'report.json')['samples']]
    data = {'records': records, 'groups': {target['id']: target for target in catalog['targets']}}
    plans = [entry['plan'] for entry in read(source / 'supervision-audit.json')]
    query_folds = {qi: plan for plan in plans for qi in plan['held_queries']}
    return source_cfg, evaluation, source, context, data, query_folds


def run(cfg, out, config_path):
    source_cfg, evaluation, source, context, data, query_folds = inputs(cfg)
    lookup = {record['name']: i for i, record in enumerate(data['records'])}
    wanted = {sample['sample_name'] for sample in context['samples']}
    counts = evaluation['candidate_counts']; maximum = max(counts)
    parent_audit = read(source / 'result-audit.json'); source_hashes = {}; cached = {}
    for seed in source_cfg['seeds']:
        cached[seed] = {}
        for method in SOURCES:
            folder = source / ('seed-' + str(seed)) / method
            for name in ['candidates.json', 'metrics.json']:
                path = folder / name; source_hashes[str(path.relative_to(source))] = digest(path)
                if digest(path) != parent_audit['artifact_sha256'][str(path.relative_to(source))]:
                    raise ValueError('Audited source changed: ' + str(path))
            frames = read(folder / 'candidates.json')
            if {frame['sample_name'] for frame in frames} != wanted or len(frames) != len(wanted):
                raise ValueError('Candidate frame identities differ')
            cached[seed][method] = {frame['sample_name']: frame['candidates'] for frame in frames}
    registration = {'configuration': cfg, 'configuration_sha256': digest(config_path),
                    'source_sha256': {name: digest(ROOT / name) for name in
                      ['src/evaluation/eval_proposal_fusion.py', 'src/evaluation/eval_template_selection.py', 'src/training/train_scene_proposals.py']},
                    'source_candidates_and_metrics_sha256': source_hashes,
                    'source_registration_sha256': digest(source / 'pre-registration.json'),
                    'candidate_counts': counts, 'nms_iou': evaluation['nms_iou'],
                    'observed_cache_depth': {method: max(len(values) for seed in cached.values() for values in seed[method].values()) for method in SOURCES},
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'scope': cfg['scope'], 'budget': 'At most K retained candidates; no padding or additional candidates beyond the source caches.'}
    write(out / 'pre-registration.json', registration)
    results = {}; parity = []
    with h5py.File(project_path(evaluation['context_labels']) / 'labels.h5', 'r') as labels:
        for seed in source_cfg['seeds']:
            for method in cfg['methods']:
                folder = out / ('seed-' + str(seed)) / method; folder.mkdir(parents=True, exist_ok=True)
                candidates = {}; suppression = {}
                for name in wanted:
                    pools = {source_name: cached[seed][source_name][name] for source_name in SOURCES}
                    candidates[lookup[name]], suppression[name] = fuse_candidates(pools, method, maximum, evaluation['nms_iou'])
                metrics, rows, frames = score_predictions(data, context, labels, candidates, query_folds, evaluation)
                if method in SOURCES:
                    if metrics != read(source / ('seed-' + str(seed)) / method / 'metrics.json'):
                        raise ValueError('Single-source baseline did not reproduce')
                    parity.append({'seed': seed, 'method': method, 'equal': True})
                    write(folder / 'source-reference.json', {'directory': str(Path(source_cfg['output_directory']) / ('seed-' + str(seed)) / method),
                                                            'candidates_sha256': source_hashes[f'seed-{seed}/{method}/candidates.json']})
                else:
                    for frame in frames:
                        frame['suppressed'] = suppression[frame['sample_name']]
                    write(folder / 'candidates.json', frames); write(folder / 'coverage-records.json', rows)
                metrics['candidate_budget'] = {str(k): {'mean_returned': float(np.mean([min(k, len(frame['candidates'])) for frame in frames])),
                                                        'underfilled_frames': sum(len(frame['candidates']) < k for frame in frames)} for k in counts}
                write(folder / 'metrics.json', metrics); results.setdefault(method, {})[str(seed)] = metrics
                print('FUSION_RESULT', seed, method, json.dumps(metrics['by_role']), flush=True)
    aggregates = {method: aggregate(list(values.values()), counts) for method, values in results.items()}
    scopes = {}; deltas = {}
    for method, values in results.items():
        scopes[method] = {}; deltas[method] = {}
        for scope in ['fit', 'held']:
            scopes[method][scope] = {}
            for k in counts:
                rates = [metrics['context_by_request_scope'][scope][str(k)]['point_recall'] for metrics in values.values()]
                scopes[method][scope][str(k)] = {'values': rates, 'mean': float(np.mean(rates)),
                                                'std': float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.}
        for role in ['original', 'context']:
            deltas[method][role] = {}
            for k in counts:
                rates = [100 * (values[str(seed)]['by_role'][role][str(k)]['point_recall'] -
                                 results['original'][str(seed)]['by_role'][role][str(k)]['point_recall']) for seed in source_cfg['seeds']]
                deltas[method][role][str(k)] = {'values_pp': rates, 'mean': float(np.mean(rates)),
                                               'std': float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.}
    write(out / 'comparison.json', {'per_seed': results, 'aggregate': aggregates, 'context_by_scope': scopes,
                                   'paired_point_delta_vs_original': deltas, 'single_source_parity': parity,
                                   'scope': cfg['scope'], 'selection': 'All predeclared schedules reported; no fitted score calibration or label-aware candidate ordering.'})


def benchmark(cfg, out):
    import torch
    from perception.extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
    from perception.template_matcher import TemplateMatcher
    source_cfg, evaluation, source, context, data, query_folds = inputs(cfg)
    spec = cfg['benchmark']; device = torch.device('cuda'); heads = {}; states = {}
    for method in SOURCES:
        state = torch.load(source / f"seed-{spec['seed']}" / method / f"fold-{spec['fold']}" / 'model.pt', map_location='cpu', weights_only=True)
        p = state['configuration']; model = TemplateMatcher(conditioned=False, hidden_dim=p['hidden_dim'], output_grid=p['output_grid']).to(device)
        model.load_state_dict(state['model']); heads[method] = model.eval(); states[method] = state
    for key in ['mean', 'std']:
        torch.testing.assert_close(states['original'][key], states['all_balanced'][key], rtol=0, atol=0)
    parameters = states['original']['configuration']; mean = states['original']['mean'].to(device); std = states['original']['std'].to(device)
    crop_cfg = torch.load(project_path(evaluation['baseline_directory']) / 'verifier.pt', map_location='cpu', weights_only=True)['configuration']
    encoder = load_model('dinov2_vits14', device, weights=project_path(parameters['dino_weights']), repo_dir=project_path(parameters['dino_repository']))
    image_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    image_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    allowed = set(states['original']['fold']['held_queries']); lookup = {record['name']: i for i, record in enumerate(data['records'])}
    names = [sample['sample_name'] for sample in context['samples'] if lookup[sample['sample_name']] in allowed][:spec['frames']]
    if not names:
        raise ValueError('No context images available for the configured timing fold')
    images = []
    with ExitStack() as stack:
        stores = {}
        for name in names:
            tid = data['records'][lookup[name]]['target_id']
            if tid not in stores:
                stores[tid] = stack.enter_context(h5py.File(project_path(evaluation['capture_directory']) / tid / 'samples.h5', 'r'))
            raw = stores[tid]; images.append(redact_rgb(read_rgb(raw, name), raw[name]['ignored_mask'][()]))

    @torch.no_grad()
    def predict(image, method):
        size = parameters['image_size']; resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        x = to_input_tensor(np.stack([resized] * parameters['feature_batch']), size, image_mean, image_std, device)
        with torch.autocast('cuda', dtype=torch.float16):
            tokens = encoder.forward_features(x)['x_norm_patchtokens'][:1].to(torch.float16).float()
        q = ((tokens - mean) / std).expand(parameters['batch'], -1, -1).contiguous()
        pools = {}
        for name in [method] if method in SOURCES else SOURCES:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = heads[name](q, torch.zeros_like(q))
            pools[name] = peak_candidates(logits[0].cpu().numpy(), image.shape[:2], crop_cfg,
                                          max(evaluation['candidate_counts']), evaluation['nms_iou'])
        if method in SOURCES:
            return pools[method]
        return fuse_candidates(pools, method, max(evaluation['candidate_counts']), evaluation['nms_iou'])[0]

    for method in cfg['methods']:
        for i in range(spec['warmup']):
            predict(images[i % len(images)], method)
    times = {method: [] for method in cfg['methods']}
    for i, image in enumerate(images):
        shift = i % len(cfg['methods']); order = cfg['methods'][shift:] + cfg['methods'][:shift]
        for method in order:
            torch.cuda.synchronize(); started = time.perf_counter(); predict(image, method); torch.cuda.synchronize()
            times[method].append((time.perf_counter() - started) * 1000)
    result = {'environment': {'torch': torch.__version__, 'opencv': cv2.__version__, 'gpu': torch.cuda.get_device_name()},
              'seed': spec['seed'], 'fold': spec['fold'], 'frames': len(images), 'warmup_per_method': spec['warmup'],
              'sample_names': names, 'feature_batch': parameters['feature_batch'], 'head_batch': parameters['batch'],
              'image_size': parameters['image_size'], 'encoder_sha256': digest(project_path(parameters['dino_weights'])),
              'timing_ms': {method: {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'samples': values} for method, values in times.items()},
              'scope': 'One RGB image, one shared DINO encoding, one/two coarse heads, CPU peak extraction and fusion; includes resizing/transfers, excludes file reading, offline redaction, model loading, template verification, depth and control. Rotating method order.',
              'model_sha256': {method: digest(source / f"seed-{spec['seed']}" / method / f"fold-{spec['fold']}" / 'model.pt') for method in SOURCES},
              'source_sha256': digest(Path(__file__))}
    write(out / 'latency.json', result); print('LATENCY', json.dumps({k: {a: b for a, b in v.items() if a != 'samples'} for k, v in result['timing_ms'].items()}), flush=True)


def plot(cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = read(out / 'comparison.json'); source_cfg = read(project_path(cfg['source_config']))
    evaluation = read(project_path(source_cfg['evaluation_config'])); k = str(max(evaluation['candidate_counts']))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, role in zip(axes[:2], ['original', 'context']):
        values = [result['aggregate'][method][role][k]['point_recall'] for method in cfg['methods']]
        ax.bar(np.arange(len(values)), [v['mean'] for v in values], yerr=[v['std'] for v in values], capsize=3)
        ax.set_xticks(np.arange(len(values)), [name.replace('_', '\n') for name in cfg['methods']], fontsize=8)
        ax.set(title=role + ' K=' + k, ylim=(0, 1), ylabel='Peak coverage')
    values = [result['context_by_scope'][method]['held'][k] for method in cfg['methods']]
    axes[2].bar(np.arange(len(values)), [v['mean'] for v in values], yerr=[v['std'] for v in values], capsize=3)
    axes[2].set_xticks(np.arange(len(values)), [name.replace('_', '\n') for name in cfg['methods']], fontsize=8)
    axes[2].set(title='Other held-group objects K=' + k, ylim=(0, 1), ylabel='Peak coverage')
    fig.suptitle(f"Fixed retained-candidate budget; shared crossfit caches; mean and SD over {len(source_cfg['seeds'])} seeds")
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evaluation/proposal_fusion.json')
    parser.add_argument('--stage', choices=['evaluate', 'benchmark', 'plot'], required=True)
    args = parser.parse_args(); path = project_path(args.config); cfg = read(path)
    out = project_path(cfg['output_directory']); out.mkdir(parents=True, exist_ok=True); cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(4)
    if args.stage == 'evaluate':
        if (out / 'comparison.json').exists():
            raise FileExistsError('Completed fusion comparison already exists')
        run(cfg, out, path)
    elif args.stage == 'benchmark':
        benchmark(cfg, out)
    else:
        plot(cfg, out)


if __name__ == '__main__':
    main()
