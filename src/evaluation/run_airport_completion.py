"""Registered metric development, crossfit verification, freezing and test reevaluation."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.paths import project_path
from common.patches import read_rgb
from common.view_geometry import redact_rgb
from evaluation.eval_template_selection import read, write, digest, point_hit
from evaluation.eval_birdseye_matching import calibrate, summarize
from perception.metric_birdseye import metric_cloud, metric_reference, metric_search
from perception.multicandidate_verification import MultiCandidateVerifier
from training.train_template_matcher import pair_plan


def sources(cfg):
    paths = ['src/perception/metric_birdseye.py', 'src/perception/multicandidate_verification.py',
             'src/evaluation/run_airport_completion.py', 'src/perception/template_verification.py',
             'src/perception/birdseye_matching.py', 'src/evaluation/eval_proposal_fusion.py',
             'src/perception/template_matcher.py', 'src/evaluation/eval_template_selection.py',
             'src/perception/shadow_verification.py', 'src/control/run_approach.py']
    paths += [cfg['base_verifier'], cfg['catalog'], cfg['pair_plan']]
    paths += [str(Path(cfg['proposal_directory']) / s / f'fold-{f}/model.pt') for f in range(3) for s in ['original', 'all_balanced']]
    return {p: digest(project_path(p)) for p in paths}


def inference_inputs(cfg):
    bundle = torch.load(project_path(cfg['base_verifier']), map_location='cpu', weights_only=True)
    paths = [bundle['configuration']['dino_weights'], 'src/perception/extract_dino_features.py',
             'src/common/patches.py', 'src/evaluation/eval_birdseye_matching.py']
    for target in read(project_path(cfg['catalog']))['targets']:
        template = target['template']
        if digest(project_path(template['file'])) != template['sha256']:
            raise ValueError('Catalog template changed: '+target['id'])
        paths.append(template['file'])
    repository = project_path(bundle['configuration']['dino_repository'])
    paths += [str(p.relative_to(ROOT)) for p in sorted((repository/'dinov2').rglob('*.py'))]
    return {p: digest(project_path(p)) for p in paths}


def dataset(cfg, partition):
    catalog = read(project_path(cfg['catalog'])); split = catalog['splits'][cfg['protocol']]
    allowed = set(split['train'] + split['validation'])
    if partition == 'test': allowed.update(split['test'])
    targets = {t['id']: t for t in catalog['targets'] if t['id'] in allowed}
    records = []; protocols = {}; raw_hashes = {}
    for tid in targets:
        if partition == 'test' and tid not in split['test']: continue
        folder = project_path(cfg['capture_directory']) / tid; report = read(folder / 'report.json')
        if digest(folder / 'samples.h5') != report['samples_h5_sha256']: raise ValueError('Capture changed: ' + tid)
        raw_hashes[tid] = report['samples_h5_sha256']; protocols[tid] = report['protocol']
        records.extend(dict(s, horizontal_fov_deg=report['protocol']['horizontal_fov_deg']) for s in report['samples'])
    requests = defaultdict(list)
    if partition == 'test':
        data = dict(catalog=catalog, targets=list(targets), groups=targets, records=records, split={'test': list(range(len(records)))})
        pairs = pair_plan(data, dict(seed=cfg['confidence_seed'], protocol=cfg['protocol'], negative_visibility_margin_deg=2))['test']
    else:
        pairs = [r for part in ['train', 'validation'] for r in read(project_path(cfg['pair_plan']))[part]]
    for pair in pairs:
        sample = records[pair['query']]; tid = list(targets)[pair['template']]
        if sample['target_id'] != pair['target_id']: raise ValueError('Pair order mismatch')
        requests[sample['name']].append(dict(target_id=tid, positive=pair['positive'], role='primary'))
    if partition != 'test':
        context = read(project_path(cfg['context_labels']) / 'report.json')
        if digest(project_path(cfg['context_labels']) / 'labels.h5') != context['labels_sha256']: raise ValueError('Context changed')
        for frame in context['samples']:
            requests[frame['sample_name']].extend(dict(target_id=t['target_id'], positive=True, role='context') for t in frame['label_targets'])
    return catalog, targets, records, protocols, requests, raw_hashes


def register(cfg, out, config_path):
    catalog, targets, records, protocols, requests, hashes = dataset(cfg, 'train')
    registration = dict(configuration=cfg, configuration_sha256=digest(config_path), source_sha256=sources(cfg),
                        raw_sha256=hashes, created_at_utc=datetime.now(timezone.utc).isoformat(),
                        partitions=['train', 'validation'], test_history='Test queries evaluated by previous whole-target experiment; this run is a frozen legacy-test reevaluation.',
                        confidence='Fixed linear architecture/seed/epochs; fit on OOF training candidates; calibrate maximum score per complete candidate set on validation; no test selection.',
                        deployment='Seed 1731 fold 0 original/all_balanced, original-first alternating NMS K=5; same fold in validation/test/live. Training uses query-held fold.',
                        known_limit='Frozen crop normalization uses training queries, including inner held-fold queries; cached incidental held appearances remain. Not a strict end-to-end nested holdout.',
                        metric='Template GSD at world z=0, with equal metric downsampling of template and query; all occupied overlapping tiles; nominal GSD does not orthorectify raised template surfaces.',
                        decision='Retain registered fused generator unless metric validation meets 95% precision / 20% recall under its train-only calibrated threshold; metric is development diagnosis, never selected on test.')
    path = out / 'registration.json'
    if path.exists():
        old = read(path)
        for key in ['configuration_sha256', 'source_sha256', 'raw_sha256']:
            if old[key] != registration[key]: raise ValueError('Registration differs: ' + key)
    else: write(path, registration)
    print('REGISTERED', len(records), 'development frames', flush=True)


def metric_stage(cfg, out, max_frames):
    _, targets, records, protocols, requests, _ = dataset(cfg, 'train')
    selected = []; strata = defaultdict(int)
    for sample in records:
        key = sample['target_id'], sample['range_stratum']
        if sample['partition'] == 'train' and strata[key] >= cfg['metric_train_per_target_range']: continue
        selected.append(sample); strata[key] += 1
    path = out / 'metric-frames.jsonl'; done = {}
    if path.exists(): done = {r['sample_name']: r for r in map(json.loads, path.read_text().splitlines())}
    references = {}
    for tid, target in targets.items():
        image = cv2.imread(str(project_path(target['template']['file'])))[:, :, ::-1].copy()
        references[tid] = metric_reference(image, target['gsd_at_world_z0_m_per_pixel'], cfg['metric'])
    pending = [s for s in selected if s['name'] not in done]
    with ExitStack() as stack, path.open('a') as stream:
        stores = {}
        for sample in pending[:max_frames]:
            tid = sample['target_id']; name = sample['name']
            if tid not in stores: stores[tid] = stack.enter_context(h5py.File(project_path(cfg['capture_directory']) / tid / 'samples.h5', 'r'))
            group = stores[tid][name]; ignored = group['ignored_mask'][()].astype(bool)
            rgb = redact_rgb(read_rgb(stores[tid], name), ignored); depth = group['depth_m'][()].astype(np.float32)
            cloud = metric_cloud(rgb, depth, sample['depth_metadata'], protocols[tid]['horizontal_fov_deg'], cfg['metric'], ignored)
            entries = []
            for request in requests[name]:
                if request['role'] != 'primary': continue
                start = time.perf_counter(); predictions, diag = metric_search(cloud, references[request['target_id']], cfg['metric'])
                duration = time.perf_counter() - start
                mask = group['assembly_mask'][()].astype(bool)
                hits = [bool(request['positive'] and point_hit(mask, r['point_xy'])) for r in predictions]
                entries.append(dict(request, partition=sample['partition'], range_stratum=sample['range_stratum'], visible_pixels=int(mask.sum()) if request['positive'] else 0,
                                    point_hits=hits, available=bool(predictions), score=predictions[0]['score'] if predictions else None,
                                    correct=bool(hits and hits[0]), candidates=predictions, seconds=duration, **diag))
            row = dict(sample_name=name, query_target_id=tid, requests=entries)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); done[name] = row
            print('METRIC_FRAME', len(done), '/', len(selected), name, 'seconds', round(sum(r['seconds'] for r in entries), 2), flush=True)
    if len(done) != len(selected): return
    rows = [r for s in selected for r in done[s['name']]['requests']]
    calibration = calibrate([r for r in rows if r['partition'] == 'train'], cfg['calibration_precision'])
    validation = summarize([r for r in rows if r['partition'] == 'validation'], [1, 3, 5], calibration['threshold'])
    eligible = (validation['accepted_precision'] or 0) >= cfg['calibration_precision'] and (validation['verified_positive_recall'] or 0) >= cfg['minimum_verified_recall']
    result = dict(frames=len(done), calibration=calibration, validation=validation, eligible=eligible,
                  selected_generator='metric' if eligible else cfg['fusion'], frame_records_sha256=digest(path))
    write(out / 'metric-decision.json', result); print('METRIC_COMPLETE', json.dumps(result), flush=True)


def feature_stage(cfg, out, partition, max_frames):
    if partition == 'test': verify_freeze(cfg, out)
    _, targets, records, protocols, requests, hashes = dataset(cfg, partition)
    model = MultiCandidateVerifier(cfg)
    for tid, target in targets.items():
        image = cv2.imread(str(project_path(target['template']['file'])))[:, :, ::-1].copy()
        model.add_reference(tid, image)
    # Fold indices refer to the original catalog-order train+validation records.
    fold_for = {records[i]['name']: f for f in range(3) for i in model.states[f, 'original']['fold']['held_queries']} if partition != 'test' else {}
    selected = [s for s in records if s['partition'] == partition]
    with h5py.File(out / f'features-{partition}.h5', 'a') as saved, ExitStack() as stack:
        for name in list(saved):
            if name.startswith('_pending_'): del saved[name]
        stores = {}; labels = stack.enter_context(h5py.File(project_path(cfg['context_labels']) / 'labels.h5', 'r')) if partition == 'train' else None
        completed = set(saved)
        for sample in [s for s in selected if s['name'] not in completed][:max_frames]:
            tid = sample['target_id']; name = sample['name']
            if tid not in stores: stores[tid] = stack.enter_context(h5py.File(project_path(cfg['capture_directory']) / tid / 'samples.h5', 'r'))
            group = stores[tid][name]; ignored = group['ignored_mask'][()].astype(bool)
            rgb = redact_rgb(read_rgb(stores[tid], name), ignored)
            fold = fold_for[name] if partition == 'train' else cfg['deployment_fold']
            torch.cuda.synchronize(); start = time.perf_counter(); query = model.prepare_query(rgb, fold)
            entries = requests[name]; features = [model.candidate_features(query, r['target_id']) for r in entries]
            torch.cuda.synchronize(); seconds = time.perf_counter() - start
            ids = labels[name]['ids'][()] if labels is not None and name in labels else None
            primary = group['assembly_mask'][()].astype(bool); valid = group['valid_mask'][()].astype(bool)
            correct = []; known = []; visible = []; distances = []
            for request in entries:
                target = targets[request['target_id']]
                mask = (ids == target['index']) if request['role'] == 'context' else primary
                correct.append([bool(request['positive'] and point_hit(mask, c['point_xy'])) for c in query['candidates']])
                known.append([point_hit(valid, c['point_xy']) for c in query['candidates']])
                visible.append(int(mask.sum()) if request['positive'] else 0)
                distances.append(float(np.linalg.norm(np.array([sample['camera_metadata']['pos_'+k] for k in 'xy']) - target['reference_ned_m'][:2])))
            metadata = dict(sample_name=name, target_id=tid, partition=partition, fold=fold, requests=entries, candidates=query['candidates'],
                            visible_pixels=visible, distances_m=distances, seconds=seconds, range_stratum=sample['range_stratum'])
            pending = saved.create_group('_pending_' + name)
            pending.create_dataset('features', data=np.asarray(features), compression='gzip')
            pending.create_dataset('correct', data=np.asarray(correct, bool)); pending.create_dataset('known', data=np.asarray(known, bool))
            pending.attrs['metadata'] = json.dumps(metadata); saved.flush(); saved.move('_pending_'+name, name); saved.flush()
            completed.add(name); print('FEATURE_FRAME', partition, len(completed), '/', len(selected), name, round(seconds, 3), flush=True)
        complete = {s['name'] for s in selected} == set(saved)
        if complete:
            saved.attrs['complete'] = True; saved.attrs['raw_sha256'] = json.dumps(hashes)
            saved.attrs['configuration_sha256'] = digest(project_path('configs/evaluation/airport_completion.json'))
    if complete:
        write(out / f'features-{partition}-manifest.json', dict(frames=len(selected), sha256=digest(out/f'features-{partition}.h5'), raw_sha256=hashes))
        print('FEATURE_COMPLETE', partition, flush=True)


def training_array(out):
    x = []; y = []; weights = []
    with h5py.File(out/'features-train.h5', 'r') as store:
        if not store.attrs.get('complete'): raise ValueError('Training features incomplete')
        for name in store:
            group = store[name]; values = group['features'][()]; known = group['known'][()]; correct = group['correct'][()]
            for features, valid, labels in zip(values, known, correct):
                if not valid.any(): continue
                x.extend(features[valid]); y.extend(labels[valid]); weights.extend([1/int(valid.sum())]*int(valid.sum()))
    np.savez_compressed(out/'acceptance-training.npz', x=np.asarray(x, np.float32), y=np.asarray(y, np.float32), weights=np.asarray(weights, np.float32))
    print('TRAINING_ARRAY', len(y), 'candidate examples', flush=True)


def score_store(path, state):
    rows = []
    with h5py.File(path, 'r') as store:
        if not store.attrs.get('complete'): raise ValueError('Feature store incomplete')
        for name in store:
            group = store[name]; metadata = json.loads(group.attrs['metadata']); features = group['features'][()]; labels = group['correct'][()]
            for i, request in enumerate(metadata['requests']):
                values = features[i]
                logits = ((values-state['center'].numpy())/state['scale'].numpy()) @ state['weight'].numpy() + float(state['bias'])
                scores = 1/(1+np.exp(-np.clip(logits.astype(np.float64), -700, 700)))
                index = int(np.argmax(scores)) if len(scores) else None
                ordered = sorted(range(len(scores)), key=lambda k: -scores[k])
                rows.append(dict(request, sample_name=name, query_target_id=metadata['target_id'], partition=metadata['partition'],
                                 horizontal_distance_m=metadata['distances_m'][i], visible_pixels=metadata['visible_pixels'][i],
                                 available=index is not None, score=float(scores[index]) if index is not None else None,
                                 correct=bool(index is not None and labels[i,index]), point_hits=[bool(labels[i,j]) for j in ordered],
                                 proposal_point_hits=labels[i].tolist(), chosen_rank=index+1 if index is not None else None,
                                 point_xy=metadata['candidates'][index]['point_xy'] if index is not None else None,
                                 candidate_scores=scores.tolist(), candidates=metadata['candidates']))
    return rows


def calibration_stage(cfg, out):
    decision = read(out/'metric-decision.json')
    if decision['selected_generator'] != cfg['fusion']: raise ValueError('Metric passed its registered gate; metric candidate export is required before fusion calibration')
    state = torch.load(out/'acceptance-fit.pt', map_location='cpu', weights_only=True)
    train = score_store(out/'features-train.h5', state); validation = score_store(out/'features-validation.h5', state)
    selection = calibrate(validation, cfg['calibration_precision']); state['threshold'] = selection['threshold']
    state['configuration'] = cfg; torch.save(state, out/'verifier.pt')
    result = dict(calibration=selection, train=summarize(train,[1,3,5],selection['threshold']),
                  validation=summarize(validation,[1,3,5],selection['threshold']),
                  calibration_scope='Repeated development validation, independent of acceptance-head fitting; not independent evaluation of calibrated precision.')
    result['validation_by_target'] = {tid:summarize([r for r in validation if r['query_target_id']==tid],[1,3,5],selection['threshold']) for tid in sorted({r['query_target_id'] for r in validation})}
    for r in train+validation:r['accepted']=bool(r['available'] and selection['threshold'] is not None and r['score']>=selection['threshold'])
    write(out/'development-records.json', dict(train=train,validation=validation)); write(out/'development.json',result)
    frozen = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), configuration=cfg,
                  sources=sources(cfg), inputs=inference_inputs(cfg), verifier_sha256=digest(out/'verifier.pt'),
                  metric_decision_sha256=digest(out/'metric-decision.json'),
                  feature_sha256={part:digest(out/f'features-{part}.h5') for part in ['train','validation']},
                  threshold=selection['threshold'], scope=cfg['scope'], flight='Shadow only, using existing controller; no acceptance gating of motion.')
    write(out/'frozen.json', frozen); print('FROZEN', json.dumps(result), flush=True)


def verify_freeze(cfg, out):
    frozen = read(out/'frozen.json')
    if frozen['configuration'] != cfg or frozen['sources'] != sources(cfg) or frozen['verifier_sha256'] != digest(out/'verifier.pt'):
        raise ValueError('Frozen configuration/code/weights changed')
    for path, expected in frozen['inputs'].items():
        if digest(project_path(path)) != expected: raise ValueError('Frozen inference input changed: '+path)
    return frozen


def test_stage(cfg, out):
    frozen = verify_freeze(cfg, out); state = torch.load(out/'verifier.pt', map_location='cpu', weights_only=True)
    rows = score_store(out/'features-test.h5', state)
    for r in rows:r['accepted']=bool(r['available'] and state['threshold'] is not None and r['score']>=state['threshold'])
    summary = summarize(rows,[1,3,5],state['threshold'])
    summary['by_target'] = {tid:summarize([r for r in rows if r['query_target_id']==tid],[1,3,5],state['threshold']) for tid in sorted({r['query_target_id'] for r in rows})}
    summary['by_range'] = {str(i):summarize([r for r in rows if int(np.searchsorted([30,60,100],r['horizontal_distance_m'],side='right'))==i],[1,3,5],state['threshold']) for i in range(4)}
    summary.update(frozen_sha256=digest(out/'frozen.json'), scope='Frozen reevaluation of previously used test queries; no test tuning; not a new independent holdout.')
    write(out/'test-records.json', rows); write(out/'test.json',summary); print('TEST_COMPLETE',json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/evaluation/airport_completion.json')
    parser.add_argument('--stage',choices=['register','metric','features','training-array','calibrate','test'],required=True)
    parser.add_argument('--partition',choices=['train','validation','test'],default='train')
    parser.add_argument('--max-frames',type=int,default=None)
    args=parser.parse_args();path=project_path(args.config);cfg=read(path);out=project_path(cfg['output_directory']);out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);torch.set_num_threads(4)
    if args.stage=='register':register(cfg,out,path)
    elif args.stage=='metric':metric_stage(cfg,out,args.max_frames)
    elif args.stage=='features':feature_stage(cfg,out,args.partition,args.max_frames)
    elif args.stage=='training-array':training_array(out)
    elif args.stage=='calibrate':calibration_stage(cfg,out)
    elif args.stage=='test':test_stage(cfg,out)


if __name__=='__main__':main()
