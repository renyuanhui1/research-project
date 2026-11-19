"""Paired raw/orthographic template matching on development RGB-D only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
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
from evaluation.eval_template_selection import read, write, digest, point_hit, ratio, wilson
from perception.birdseye_matching import (birdseye, source_point, sift_features,
                                          sift_candidates, DinoMatcher)

METHODS = ['raw_sift', 'bev_sift', 'raw_dino', 'bev_dino']


def calibrate(rows, precision):
    """Largest complete score group attaining precision; None means reject all."""
    ranked = sorted((r for r in rows if r['available']), key=lambda r: -r['score'])
    correct = 0; best = None
    for i, row in enumerate(ranked):
        correct += row['correct']
        if i + 1 < len(ranked) and ranked[i + 1]['score'] == row['score']:
            continue
        if correct / (i + 1) >= precision:
            next_score = ranked[i + 1]['score'] if i + 1 < len(ranked) else row['score']
            best = dict(threshold=(row['score'] + next_score) / 2, accepted=i + 1,
                        correct=correct, precision=correct / (i + 1))
    return best or dict(threshold=None, accepted=0, correct=0, precision=None)


def summarize(rows, counts, threshold=None):
    positive = [r for r in rows if r['positive'] and r['visible_pixels'] > 0]
    negatives = [r for r in rows if not r['positive']]
    accepted = [r for r in rows if r['available'] and threshold is not None and r['score'] >= threshold]
    correct = sum(r['correct'] for r in accepted)
    return dict(pairs=len(rows), visible_positive=len(positive), absent_pairs=len(negatives),
                coverage={str(k): dict(hits=sum(any(r['point_hits'][:k]) for r in positive),
                                       rate=ratio(sum(any(r['point_hits'][:k]) for r in positive), len(positive))) for k in counts},
                missing_predictions=sum(not r['available'] for r in positive),
                accepted=len(accepted), correct_accepted=correct,
                accepted_precision=ratio(correct, len(accepted)), precision_wilson_95=wilson(correct, len(accepted)),
                verified_positive_recall=ratio(correct, len(positive)),
                absent_false_accept_rate=ratio(sum(not r['positive'] for r in accepted), len(negatives)),
                wrong_visible_target_at_top1=sum(r.get('wrong_visible_target_at_top1', False) for r in positive))


def metrics(rows, cfg, threshold):
    result = summarize(rows, cfg['candidate_counts'], threshold)
    for key in ['requested_target_id', 'category', 'range_stratum']:
        result['by_' + key] = {str(value): summarize([r for r in rows if r[key] == value], cfg['candidate_counts'], threshold)
                              for value in sorted({r[key] for r in rows})}
    return result


def prepare(cfg, out, config_path, resume=False):
    catalog = read(project_path(cfg['catalog'])); split = catalog['splits'][cfg['protocol']]
    allowed = set(split['train'] + split['validation'])
    targets = {t['id']: t for t in catalog['targets'] if t['id'] in allowed}
    records = []; capture = {}; hashes = {}
    for tid in targets:
        folder = project_path(cfg['capture_directory']) / tid
        report = read(folder / 'report.json'); actual = digest(folder / 'samples.h5')
        if actual != report['samples_h5_sha256']:
            raise ValueError('Capture hash changed: ' + tid)
        hashes[str((folder / 'samples.h5').relative_to(ROOT))] = actual
        hashes[str((folder / 'report.json').relative_to(ROOT))] = digest(folder / 'report.json')
        if any(s['target_id'] != tid or s['partition'] not in ['train', 'validation'] for s in report['samples']):
            raise ValueError('Unexpected capture partition: ' + tid)
        records.extend(report['samples']); capture[tid] = report['protocol']
        item = targets[tid]['template']
        if digest(project_path(item['file'])) != item['sha256']:
            raise ValueError('Template hash changed: ' + tid)
        hashes[item['file']] = item['sha256']
    pairs = read(project_path(cfg['reference_pair_plan']))
    requests = defaultdict(list)
    ordered = list(targets)
    for partition in ['train', 'validation']:
        for pair in pairs[partition]:
            sample = records[pair['query']]
            tid = ordered[pair['template']]
            if sample['target_id'] != pair['target_id'] or sample['partition'] != partition:
                raise ValueError('Pair-plan sample ordering differs')
            if (tid == sample['target_id']) != pair['positive']:
                raise ValueError('Pair-plan template identity differs')
            requests[sample['name']].append(dict(partition=partition, requested_target_id=tid,
                                                 positive=pair['positive'], role='primary'))
    context_root = project_path(cfg['context_labels'])
    context = read(context_root / 'report.json')
    if digest(project_path(cfg['catalog'])) != context['catalog_sha256'] or digest(context_root / 'labels.h5') != context['labels_sha256']:
        raise ValueError('Context labels changed')
    fusion = project_path(cfg['fusion_directory'])
    # All seeds share the same frames, request identities and fold scopes.
    coverage_path = fusion / 'seed-1731/alternate_original/coverage-records.json'
    historical = read(coverage_path)
    for row in historical:
        requests[row['sample_name']].append(dict(partition='context', requested_target_id=row['requested_target_id'],
                                                positive=True, role=row['role'], request_scope=row['request_scope']))
    sources = ['src/perception/birdseye_matching.py', 'src/evaluation/eval_birdseye_matching.py']
    for path in [project_path(cfg['catalog']), project_path(cfg['reference_pair_plan']), context_root / 'report.json',
                 context_root / 'labels.h5', coverage_path, fusion / 'comparison.json']:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    registration = dict(configuration=cfg, configuration_sha256=digest(config_path), input_sha256=hashes,
                        source_sha256={p: digest(ROOT / p) for p in sources},
                        encoder_sha256=digest(project_path(cfg['dino']['weights'])),
                        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        created_at_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS,
                        query_targets_read=ordered, query_partitions_read=['train', 'validation'],
                        test_queries_read=False, calibration='Train primary positive/absent pairs only; top-ranked candidate; whole score ties.',
                        prediction_inputs='Redacted RGB, measured ray depth, camera attitude, horizontal FOV, requested RGB template; no target coordinate, mask, bounds, query identity or mesh.',
                        budget='K per requested template, unlike the historical template-independent K per image; same labels/denominators, different candidate conditioning and budget allocation.',
                        projection='Camera-relative orthographic NED view. Template scale and yaw handled by SIFT; no target-height or target-position correction of perspective templates.',
                        scope=cfg['scope'])
    if resume:
        original = read(out / 'pre-registration.json')
        for key in ['configuration_sha256', 'input_sha256', 'encoder_sha256']:
            if original[key] != registration[key]:
                raise ValueError('Resume inputs changed: ' + key)
        source = 'src/perception/birdseye_matching.py'
        if original['source_sha256'][source] != registration['source_sha256'][source]:
            raise ValueError('Resume matching algorithm changed')
        history_path = out / 'resume-history.json'
        history = read(history_path) if history_path.exists() else []
        history.append(dict(created_at_utc=registration['created_at_utc'],
                            source_sha256=registration['source_sha256'],
                            candidates_before_resume_sha256=digest(out / 'candidates.jsonl'),
                            reason='Continue an interrupted experiment; orchestration supports frame checkpoints; matching configuration and module unchanged.'))
        write(history_path, history)
    else:
        write(out / 'pre-registration.json', registration)
    return targets, records, capture, requests, fusion


def score_request(request, sample, target, candidates, mask, ids, kept):
    positive = request['positive']
    hits = [bool(positive and point_hit(mask, c['point_xy'])) for c in candidates]
    distance = np.linalg.norm(np.array([sample['camera_metadata']['pos_' + k] for k in 'xy'])
                              - np.array(target['reference_ned_m'][:2]))
    row = dict(request, sample_name=sample['name'], query_target_id=sample['target_id'], category=target['category'],
               range_stratum=int(np.searchsorted([30, 60, 100], distance, side='right')),
               horizontal_distance_m=float(distance), visible_pixels=int(mask.sum()) if positive else 0,
               point_hits=hits, correct=bool(hits and hits[0]), available=bool(candidates),
               score=candidates[0]['score'] if candidates else None,
               kept_target_fraction=ratio(int((mask & kept).sum()), int(mask.sum())) if positive else None)
    if ids is not None and candidates:
        point = candidates[0]['point_xy']; h, w = ids.shape
        x, y = np.floor(np.asarray(point) * [w, h]).astype(int)
        other = int(ids[y, x]) if 0 <= x < w and 0 <= y < h else 0
        row['wrong_visible_target_at_top1'] = bool(positive and other and other != target['index'])
    return row


def evaluate(cfg, out, config_path, device, resume=False, max_frames=None):
    import torch
    targets, records, protocols, requests, fusion = prepare(cfg, out, config_path, resume)
    dino = DinoMatcher(cfg['dino'], device); references = {}; all_rows = {m: [] for m in METHODS}
    for tid, target in targets.items():
        rgb = cv2.imread(str(project_path(target['template']['file'])))[:, :, ::-1].copy()
        references[tid] = dict(sift=sift_features(rgb, cfg['sift']), dino=dino.reference(rgb), shape=rgb.shape[:2])
    checkpoints = {}; cached_candidates = defaultdict(dict)
    checkpoint_path = out / 'frame-checkpoints.jsonl'
    if resume:
        with (out / 'candidates.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                cached_candidates[row['sample_name']][row['method'], row['requested_target_id']] = row['candidates']
        if checkpoint_path.exists():
            with checkpoint_path.open() as stream:
                for line in stream:
                    row = json.loads(line); checkpoints[row['sample_name']] = row
    pending = [sample for sample in records if sample['name'] in requests and sample['name'] not in checkpoints]
    if max_frames is not None:
        pending = pending[:max_frames]
    rows_by_target = defaultdict(list)
    for sample in pending:
        rows_by_target[sample['target_id']].append(sample)
    timing = []; examples = []; example_keys = set()
    for checkpoint in checkpoints.values():
        for method in METHODS:
            all_rows[method].extend(checkpoint['rows'][method])
        timing.append(checkpoint['timing'])
        if checkpoint['example'] is not None:
            examples.append(checkpoint['example']); example_keys.add(tuple(checkpoint['example_key']))
    with h5py.File(project_path(cfg['context_labels']) / 'labels.h5', 'r') as labels, (out / 'candidates.jsonl').open('a' if resume else 'w') as saved, checkpoint_path.open('a' if resume else 'w') as checkpoint_stream:
        for tid, samples in rows_by_target.items():
            with h5py.File(project_path(cfg['capture_directory']) / tid / 'samples.h5', 'r') as raw:
                for si, sample in enumerate(samples):
                    name = sample['name']; group = raw[name]
                    ignored = group['ignored_mask'][()].astype(bool)
                    rgb = redact_rgb(read_rgb(raw, name), ignored)
                    depth = group['depth_m'][()].astype(np.float32)
                    requested = sorted({r['requested_target_id'] for r in requests[name]})
                    started = time.perf_counter()
                    view = birdseye(rgb, depth, sample['depth_metadata'], protocols[tid]['horizontal_fov_deg'], cfg['projection'], ignored)
                    projection_ms = (time.perf_counter() - started) * 1000
                    raw_support = cv2.resize((~ignored).astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    images = {'raw': rgb, 'bev': view['rgb']}; supports = {'raw': raw_support, 'bev': view['source'] >= 0}
                    prediction = {}; elapsed = {}; cached = cached_candidates.get(name, {})
                    complete_cache = set(cached) == {(m, t) for m in METHODS for t in requested}
                    for kind, image in ([] if complete_cache else images.items()):
                        start = time.perf_counter(); query = sift_features(image, cfg['sift'], supports[kind])
                        for requested_id in requested:
                            ref = references[requested_id]
                            prediction[kind + '_sift', requested_id] = sift_candidates(query, ref['sift'], image.shape[:2], ref['shape'], cfg['sift'], max(cfg['candidate_counts']))
                        elapsed[kind + '_sift_ms'] = (time.perf_counter() - start) * 1000
                        if device == 'cuda': torch.cuda.synchronize()
                        start = time.perf_counter(); tokens = dino.encode([image], cfg['dino']['query_size'])[0]
                        for requested_id in requested:
                            prediction[kind + '_dino', requested_id] = dino.candidates(tokens, references[requested_id]['dino'], supports[kind], max(cfg['candidate_counts']))
                        if device == 'cuda': torch.cuda.synchronize()
                        elapsed[kind + '_dino_ms'] = (time.perf_counter() - start) * 1000
                    for (method, requested_id), candidates in prediction.items():
                        mapped = []
                        for candidate in candidates:
                            point = source_point(view, candidate['view_point_xy']) if method.startswith('bev') else candidate['view_point_xy']
                            if point is not None:
                                mapped.append(dict(candidate, point_xy=point))
                        prediction[method, requested_id] = mapped
                        if (method, requested_id) not in cached:
                            saved.write(json.dumps(dict(sample_name=name, requested_target_id=requested_id, method=method, candidates=mapped), allow_nan=False) + '\n')
                    prediction.update(cached)
                    saved.flush()
                    # Query labels and target coordinates enter only after predictions finish.
                    ids = labels[name]['ids'][()] if name in labels else None
                    primary_mask = group['assembly_mask'][()].astype(bool)
                    frame_rows = {m: [] for m in METHODS}
                    for request in requests[name]:
                        target = targets[request['requested_target_id']]
                        mask = (ids == target['index']) if request['partition'] == 'context' else primary_mask
                        for method in METHODS:
                            row = score_request(request, sample, target, prediction[method, target['id']], mask, ids, view['input_kept'])
                            all_rows[method].append(row)
                            frame_rows[method].append(row)
                    frame_timing = dict(sample_name=name, requested_templates=len(requested), projection_ms=projection_ms,
                                        view_shape_hw=list(view['rgb'].shape[:2]), gsd_m=view['gsd_m'],
                                        candidates_reused=complete_cache, **elapsed)
                    timing.append(frame_timing)
                    key = (sample['partition'], sample['category'], sample['range_stratum'])
                    example = None
                    if key not in example_keys:
                        example_keys.add(key)
                        filename = 'example-' + name + '.jpg'
                        draw_example(rgb, view, primary_mask, prediction, tid, out / filename)
                        example = dict(sample_name=name, file=filename, selection='First capture-order sample per partition/category/range, independent of predictions.')
                        examples.append(example)
                    checkpoint = dict(sample_name=name, rows=frame_rows, timing=frame_timing, example=example, example_key=key)
                    checkpoint_stream.write(json.dumps(checkpoint, allow_nan=False) + '\n'); checkpoint_stream.flush()
                    checkpoints[name] = checkpoint
                    if (si + 1) % 32 == 0:
                        print('BIRDSEYE_PROGRESS', tid, si + 1, '/', len(samples), flush=True)
            print('BIRDSEYE_TARGET_DONE', tid, len(samples), flush=True)
    total = sum(sample['name'] in requests for sample in records)
    if len(checkpoints) < total:
        print('BIRDSEYE_CHECKPOINT', len(checkpoints), '/', total, 'use --resume to continue', flush=True)
        return
    write(out / 'records.json', all_rows); write(out / 'timing-records.json', timing); write(out / 'examples.json', examples)
    comparison = {}; calibration = {}
    for method, rows in all_rows.items():
        train = [r for r in rows if r['partition'] == 'train']
        calibration[method] = calibrate(train, cfg['calibration_precision'])
        threshold = calibration[method]['threshold']
        val = [r for r in rows if r['partition'] == 'validation']
        context = [r for r in rows if r['partition'] == 'context']
        comparison[method] = dict(train=metrics(train, cfg, threshold), validation=metrics(val, cfg, threshold),
                                  context=metrics(context, cfg, None), context_by_role={},
                                  context_other_held=metrics([r for r in context if r['role'] == 'context' and r['request_scope'] == 'held'], cfg, None))
        for role in ['original', 'context']:
            comparison[method]['context_by_role'][role] = metrics([r for r in context if r['role'] == role], cfg, None)
        positive = [r for r in rows if r['positive'] and r['visible_pixels'] > 0 and r['partition'] != 'context']
        comparison[method]['projection_target_retention'] = dict(
            fully_retained=sum(r['kept_target_fraction'] == 1 for r in positive),
            partial_or_outside=sum(r['kept_target_fraction'] < 1 for r in positive),
            minimum_fraction=min(r['kept_target_fraction'] for r in positive))
    historical = read(fusion / 'comparison.json')
    result = dict(methods=comparison, calibration=calibration,
                  historical_context_other_held=historical['context_by_scope'],
                  historical_aggregate=historical['aggregate'],
                  scope=cfg['scope'], comparison_limit='Historical K is template independent per image; new K is per requested template. Context denominators match, computational/candidate budgets differ.',
                  timing_scope='Per-frame feature extraction plus all requested templates, excludes I/O/redaction/reference setup; BEV additionally needs projection. First-frame warmup retained; not single-template online latency. Recovered candidate-only frames have no matching timings and are excluded from timing summaries.')
    write(out / 'comparison.json', result)
    write(out / 'environment.json', dict(torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__,
                                        device=device, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                                        cv_threads=cv2.getNumThreads(), torch_threads=torch.get_num_threads()))
    print('BIRDSEYE_COMPLETE', json.dumps({m: dict(validation=comparison[m]['validation']['coverage'], held_context=comparison[m]['context_other_held']['coverage'], calibration=calibration[m]) for m in METHODS}), flush=True)


def draw_example(rgb, view, mask, predictions, tid, path):
    panels = []
    colors = {'sift': (255, 120, 20), 'dino': (50, 210, 255)}
    for kind, original in [('raw', rgb), ('bev', view['rgb'])]:
        image = original.copy(); h, w = image.shape[:2]
        if kind == 'raw':
            contours, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (40, 255, 70), 2)
        for family, color in colors.items():
            for i, candidate in enumerate(predictions[kind + '_' + family, tid]):
                point = tuple(np.floor(np.array(candidate['view_point_xy']) * [w, h]).astype(int))
                cv2.circle(image, point, 5 if i == 0 else 3, color, 2)
        size = max(h, w); scale = 640 / size
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
        panel = np.full((680, 660, 3), 245, np.uint8); ih, iw = image.shape[:2]
        panel[35:35+ih, 10:10+iw] = image
        cv2.putText(panel, kind + ' | SIFT orange, DINO cyan', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1)
        panels.append(panel)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1)[:, :, ::-1])


def plot(cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison = read(out / 'comparison.json')['methods']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ['Raw SIFT', 'BEV SIFT', 'Raw DINO', 'BEV DINO']; x = np.arange(4)
    for i, k in enumerate(cfg['candidate_counts']):
        axes[0].bar(x + (i - 1) * .25, [comparison[m]['validation']['coverage'][str(k)]['rate'] or 0 for m in METHODS], .25, label='K=' + str(k))
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel='Native-mask point coverage', title='Appearance validation (190 visible)')
    axes[0].tick_params(axis='x', rotation=25); axes[0].legend()
    target_ids = [tid for tid, value in comparison[METHODS[0]]['validation']['by_requested_target_id'].items() if value['visible_positive']]
    tx = np.arange(len(target_ids))
    for i, (method, label) in enumerate(zip(METHODS, labels)):
        axes[1].bar(tx + (i - 1.5) * .2, [comparison[method]['validation']['by_requested_target_id'][tid]['coverage']['1']['rate'] for tid in target_ids], .2, label=label)
    axes[1].set(xticks=tx, xticklabels=[tid.split('-', 1)[0] for tid in target_ids], ylim=(0, 1), xlabel='Validation target ID', ylabel='Top-1 coverage', title='Held appearances in the primary task')
    axes[1].legend(fontsize=8)
    for method, label in zip(METHODS, labels):
        groups = comparison[method]['validation']['by_range_stratum']
        axes[2].plot(range(4), [groups.get(str(i), {}).get('coverage', {}).get('5', {}).get('rate') for i in range(4)], marker='o', label=label)
    axes[2].set(xticks=range(4), xticklabels=['12–30', '30–60', '60–100', '100–150'], ylim=(0, 1), xlabel='Horizontal range (m)', ylabel='K=5 coverage', title='Validation by range'); axes[2].legend()
    fig.suptitle('Training-free development; K per template; sealed test not read')
    fig.text(.5, .01, 'Historical held context: 432-1070 m, outside the 250 m BEV window; see protocol-audit.json', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95)); fig.savefig(out / 'comparison.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evaluation/birdseye_matching.json')
    parser.add_argument('--stage', choices=['evaluate', 'plot'], required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--resume', action='store_true', help='Reuse matching inputs and complete frame checkpoints')
    parser.add_argument('--max-frames', type=int, help='Process this many pending frames, checkpoint, and return; continue with --resume')
    args = parser.parse_args(); path = project_path(args.config); cfg = read(path)
    out = project_path(cfg['output_directory']); out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(4)
    if args.stage == 'evaluate':
        if (out / 'comparison.json').exists():
            print('BIRDSEYE_ALREADY_COMPLETE', out, 'use --stage plot to regenerate the figure')
            return
        if (out / 'pre-registration.json').exists() and not args.resume:
            raise FileExistsError('Registered experiment exists; use another configured output directory')
        evaluate(cfg, out, path, args.device, args.resume, args.max_frames)
    else:
        plot(cfg, out)


if __name__ == '__main__':
    main()
