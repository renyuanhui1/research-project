"""Compare frozen backbones with identical whole-image template/ROI matching."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from common.patches import read_rgb
from common.paths import project_path
from common.view_geometry import redact_rgb
from perception.extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
from training.train_template_matcher import pair_plan, calibrate, ranking_auc


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def repository_source(path):
    path=Path(path).resolve()
    top=subprocess.check_output(['git','-C',str(path),'rev-parse','--show-toplevel'],text=True).strip()
    commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip() if Path(top).resolve()==path else None
    files={str(p.relative_to(path)):digest(p) for p in sorted(path.rglob('*.py'))}
    return {'source_commit':commit,'python_source_sha256':files}


def reference_views(image, roi, size):
    """Use the same square padding/rotations for image and normalized template ROI."""
    h, w = image.shape[:2]
    side = max(h, w); ox = (side-w)//2; oy = (side-h)//2
    canvas = np.broadcast_to(np.array([124, 116, 104], np.uint8), (side, side, 3)).copy()
    canvas[oy:oy+h, ox:ox+w] = image
    corners = None
    if roi is not None:
        x0, y0, x1, y1 = np.asarray(roi)*[w, h, w, h]
        corners = (np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])+[ox, oy])/side
    images, boxes = [], []
    for k in range(4):
        images.append(cv2.resize(np.rot90(canvas, k).copy(), (size, size), interpolation=cv2.INTER_LINEAR))
        boxes.append(None if corners is None else np.r_[corners.min(0), corners.max(0)])
        if corners is not None:
            corners = np.c_[corners[:, 1], 1-corners[:, 0]]
    return images, boxes


def roi_patches(roi, grid):
    if roi is None:
        return np.zeros(grid*grid, bool)
    axis = (np.arange(grid)+.5)/grid; yy, xx = np.meshgrid(axis, axis, indexing='ij')
    x0, y0, x1, y1 = roi
    return ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).reshape(-1)


class Encoder:
    def __init__(self, condition, device):
        self.condition = condition; self.device = torch.device(device)
        repo = project_path(condition['repository']); weights = project_path(condition['weights'])
        if condition['model'].startswith('dinov3'):
            sys.path.insert(0, str(repo))
            from dinov3.hub import backbones
            self.model = getattr(backbones, condition['model'])(pretrained=False)
            self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            self.model.eval().to(self.device)
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = load_model(condition['model'], self.device, weights=weights, repo_dir=repo)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, images, width):
        count = len(images); batch = list(images)+[images[-1]]*(width-count)
        size = self.condition['image_size']
        batch = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR) for im in batch])
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            x = to_input_tensor(batch, size, self.mean, self.std, self.device)
            tokens = self.model.forward_features(x)['x_norm_patchtokens'][:count]
        return tokens.to(torch.float16).float()


def protocol(cfg):
    catalog = read(project_path(cfg['catalog'])); split = catalog['splits'][cfg['protocol']]
    allowed = set(split['train']) | set(split['validation'])
    groups = {t['id']: t for t in catalog['targets']}; targets = [t['id'] for t in catalog['targets'] if t['id'] in allowed]
    records, sources = [], {}
    for target in targets:
        folder = project_path(cfg['capture_directory'])/target; report = read(folder/'report.json')
        actual = digest(folder/'samples.h5')
        if actual != report['samples_h5_sha256']:
            raise ValueError('Raw capture hash changed: '+target)
        records.extend(dict(r, horizontal_fov_deg=report['protocol']['horizontal_fov_deg']) for r in report['samples'])
        sources[target] = actual
        for key in ('template', 'part_mask'):
            item = groups[target][key]
            if digest(project_path(item['file'])) != item['sha256']:
                raise ValueError('Reference changed: '+target+' '+key)
    data = dict(catalog=catalog, groups=groups, targets=targets, records=records)
    data['split'] = {key: [i for i, r in enumerate(records) if r['target_id'] in split[key]] for key in ('train', 'validation')}
    pairs = pair_plan(data, cfg)['validation']
    if pairs != read(project_path(cfg['reference_pair_plan']))['validation']:
        raise ValueError('Validation pairs differ from the frozen development comparison')
    return data, pairs, sources


def batches(data, cfg, partition, width):
    selected = set(data['split'][partition]); records = data['records']
    for target in data['targets']:
        indices = [i for i, r in enumerate(records) if i in selected and r['target_id'] == target]
        if not indices:
            continue
        path = project_path(cfg['capture_directory'])/target/'samples.h5'
        with h5py.File(path, 'r') as raw:
            for start in range(0, len(indices), width):
                chunk = indices[start:start+width]; images, masks = [], []
                for i in chunk:
                    name = records[i]['name']; g = raw[name]
                    images.append(redact_rgb(read_rgb(raw, name), g['ignored_mask'][()]))
                    masks.append({key: g[key][()] for key in ('valid_mask', 'assembly_mask', 'part_mask')})
                yield chunk, images, masks


def normalization(encoder, data, cfg):
    total = squares = None; count = 0.
    for indices, images, masks in batches(data, cfg, 'train', cfg['feature_batch']):
        z = encoder.encode(images, cfg['feature_batch']).double(); grid = math.isqrt(z.shape[1])
        valid = np.stack([cv2.resize(m['valid_mask'].astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA) for m in masks])
        weight = F.interpolate(torch.from_numpy(valid).to(z.device)[:, None], size=(grid, grid), mode='area').flatten(2).transpose(1, 2).double()
        a = (z*weight).sum((0, 1)); b = (z.square()*weight).sum((0, 1))
        total = a if total is None else total+a; squares = b if squares is None else squares+b
        count += float(weight.sum())
    if count <= 0:
        raise ValueError('No valid training pixels')
    mean = total/count; std = (squares/count-mean.square()).clamp_min(1e-6).sqrt()
    return mean.float(), std.float()


def prototypes(encoder, data, cfg, mean, std):
    full, part, counts = [], [], []
    for target in data['targets']:
        t = data['groups'][target]; image = cv2.imread(str(project_path(t['template']['file'])))[:, :, ::-1].copy()
        roi = t['region']['projected_crop_roi'] if t['region']['visible_in_nadir'] else None
        images, boxes = reference_views(image, roi, encoder.condition['image_size'])
        z = torch.cat([encoder.encode(images[i:i+cfg['feature_batch']], cfg['feature_batch']) for i in range(0, 4, cfg['feature_batch'])])
        z = F.normalize((z-mean)/std, dim=-1); grid = math.isqrt(z.shape[1])
        full.append(F.normalize(z.mean(1), dim=-1)); values = []; n = []
        for tokens, box in zip(z, boxes):
            mask = roi_patches(box, grid); n.append(int(mask.sum()))
            values.append(F.normalize(tokens[torch.from_numpy(mask).to(tokens.device)].mean(0), dim=0) if mask.any() else torch.zeros_like(tokens[0]))
        part.append(torch.stack(values)); counts.append(n)
    return torch.stack(full), torch.stack(part), np.asarray(counts)


def match(tokens, references, mean, std, rotations, temperature):
    """The online fingerprint rule: mean prototype, max rotation, softmax center."""
    query = F.normalize((tokens-mean)/std, dim=-1)
    similarity = torch.einsum('bnd,brd->brn', query, references[:, :rotations])
    scores, choices = similarity.max(-1).values.max(-1)
    response = similarity[torch.arange(len(tokens), device=tokens.device), choices]
    grid = math.isqrt(tokens.shape[1]); axis = torch.linspace(0, 1, grid, device=tokens.device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij'); xy = torch.stack([xx.flatten(), yy.flatten()], -1)
    centers = torch.softmax(response/temperature, -1)@xy
    index = response.argmax(-1); peaks = torch.stack([(index % grid+.5)/grid, (index//grid+.5)/grid], -1)
    return centers.cpu().numpy(), peaks.cpu().numpy(), scores.cpu().numpy(), choices.cpu().numpy()


def point_label(mask, point):
    if point is None or not mask.any():
        return False, None
    h, w = mask.shape; x, y = np.floor(np.asarray(point)*[w, h]).astype(int)
    if not (0 <= x < w and 0 <= y < h):
        return False, None
    inside = bool(mask[y, x]); field = cv2.distanceTransform((mask if inside else ~mask).astype(np.uint8), cv2.DIST_L2, 5)
    signed = float(field[y, x])*(-1 if inside else 1)
    return inside, signed/math.sqrt(float(mask.sum())/math.pi)


def summary(rows, required_precision):
    correct = np.array([r['positive'] and r['center_hit'] for r in rows]); scores = np.array([r['score'] for r in rows])
    threshold, calibration = calibrate(scores, correct, required_precision); accepted = scores >= threshold
    visible = np.array([r['positive'] and r['visible_pixels'] > 0 for r in rows]); negative = np.array([not r['positive'] for r in rows])
    hits = np.array([r['center_hit'] for r in rows]); peaks = np.array([r['peak_hit'] for r in rows])
    errors = [r['signed_error_equivalent_radii'] for r in rows if r['positive'] and r['signed_error_equivalent_radii'] is not None]
    result = dict(pairs=len(rows), visible_positive=int(visible.sum()), center_hits=int((hits & visible).sum()),
                  center_hit_rate=float(hits[visible].mean()) if visible.any() else None,
                  peak_hit_rate=float(peaks[visible].mean()) if visible.any() else None,
                  missing_template_roi=int(sum(r['positive'] and not r['template_available'] for r in rows)),
                  threshold=threshold, calibration=calibration, accepted=int(accepted.sum()), correct_accepted=int(correct[accepted].sum()),
                  accepted_precision=float(correct[accepted].mean()) if accepted.any() else None,
                  verified_positive_recall=float((accepted & correct).sum()/visible.sum()) if visible.any() else None,
                  wrong_template_false_accept_rate=float(accepted[negative].mean()) if negative.any() else None,
                  verified_auc=ranking_auc(scores, correct),
                  median_signed_error_equivalent_radii=float(np.median(errors)) if errors else None)
    return result


def evaluate_condition(condition, cfg, data, pairs, folder, device):
    started = time.monotonic(); encoder = Encoder(condition, device); mean, std = normalization(encoder, data, cfg)
    full, part, counts = prototypes(encoder, data, cfg, mean, std)
    np.savez_compressed(folder/'reference-state.npz', mean=mean.cpu().numpy(), std=std.cpu().numpy(), full=full.cpu().numpy(), part=part.cpu().numpy(), roi_patch_counts=counts)
    tokens, masks = {}, {}
    for indices, images, labels in batches(data, cfg, 'validation', cfg['feature_batch']):
        z = encoder.encode(images, cfg['feature_batch'])
        for i, value, label in zip(indices, z, labels):
            tokens[i] = value; masks[i] = label
    result = {}; saved = {}; n = len(pairs); width = 32
    for scope, references in [('whole', full), ('part', part)]:
        for rotations in (1, 4):
            rows = []
            for start in range(0, n, width):
                selected = pairs[start:start+width]; qi = [r['query'] for r in selected]; ti = [r['template'] for r in selected]
                centers, peaks, scores, choices = match(torch.stack([tokens[i] for i in qi]), references[ti], mean, std, rotations, cfg['temperature'])
                for r, center, peak, score, choice in zip(selected, centers, peaks, scores, choices):
                    available = scope == 'whole' or counts[r['template'], choice] > 0
                    mask = masks[r['query']]['assembly_mask' if scope == 'whole' else 'part_mask'].astype(bool)
                    center_hit, error = point_label(mask, center if available else None); peak_hit, _ = point_label(mask, peak if available else None)
                    record = data['records'][r['query']]
                    rows.append(dict(r, sample_name=record['name'], category=record['category'], range_stratum=record['range_stratum'],
                                     horizontal_distance_m=record['horizontal_distance_m'], center_xy=center.tolist() if available else None,
                                     peak_xy=peak.tolist() if available else None, score=float(score) if available else -1.000001,
                                     rotation_deg=int(choice)*90, template_available=bool(available), visible_pixels=int(mask.sum()),
                                     center_hit=center_hit, peak_hit=peak_hit, signed_error_equivalent_radii=error))
            name=f'{scope}_{rotations}rot'; metrics=summary(rows, cfg['calibration_precision'])
            for key in ('target_id', 'range_stratum', 'category'):
                metrics['by_'+key] = {}
                for value in sorted({r[key] for r in rows}):
                    subset=[r for r in rows if r[key] == value]
                    # Subgroup acceptance uses the global calibrated threshold.
                    sub=summary(subset, cfg['calibration_precision']); threshold=metrics['threshold']
                    accepted=[r for r in subset if r['score'] >= threshold]; correct=sum(r['positive'] and r['center_hit'] for r in accepted)
                    sub.update(threshold=threshold, accepted=len(accepted), correct_accepted=correct,
                               accepted_precision=correct/len(accepted) if accepted else None,
                               verified_positive_recall=correct/sub['visible_positive'] if sub['visible_positive'] else None,
                               wrong_template_false_accept_rate=sum(not r['positive'] for r in accepted)/sum(not r['positive'] for r in subset) if any(not r['positive'] for r in subset) else None)
                    sub.pop('calibration'); metrics['by_'+key][str(value)]=sub
            for row in rows:
                row['accepted']=row['score'] >= metrics['threshold']; row['verified_correct']=bool(row['positive'] and row['center_hit'])
            result[name]=metrics; saved[name]=rows
            print('RESULT', condition['name'], name, json.dumps({k:v for k,v in metrics.items() if not k.startswith('by_')}), flush=True)
    result['elapsed_seconds']=time.monotonic()-started
    write(folder/'comparison.json', result); write(folder/'records.json', saved)
    del encoder, tokens, full, part
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def benchmark(cfg, data, out, device):
    images=[]; requested=[]
    indices=data['split']['validation']
    chosen=set(indices[i] for i in np.unique(np.linspace(0,len(indices)-1,min(cfg['benchmark_frames'],len(indices))).astype(int)))
    for indices, rgbs, _ in batches(data, cfg, 'validation', 1):
        if indices[0] in chosen:
            images.extend(rgbs); requested.extend(data['targets'].index(data['records'][i]['target_id']) for i in indices)
    result={'device':str(device), 'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
            'scope':'single-frame RGB array to four-rotation whole/ROI matching; fixed template preencoded; includes resize and transfer; excludes file I/O'}
    for condition in cfg['conditions']:
        encoder=Encoder(condition, device); state=np.load(out/condition['name']/'reference-state.npz')
        mean=torch.from_numpy(state['mean']).to(device); std=torch.from_numpy(state['std']).to(device)
        references=[torch.from_numpy(state[key]).to(device) for key in ('full','part')]; timings=[]
        for j in range(cfg['benchmark_warmup']+len(images)):
            k=max(0,j-cfg['benchmark_warmup']) % len(images)
            if device=='cuda':torch.cuda.synchronize()
            started=time.perf_counter(); z=encoder.encode([images[k]], cfg['feature_batch'])
            for p in references:match(z,p[requested[k]:requested[k]+1],mean,std,4,cfg['temperature'])
            if device=='cuda':torch.cuda.synchronize()
            if j>=cfg['benchmark_warmup']:timings.append((time.perf_counter()-started)*1000)
        result[condition['name']]={'frames':len(timings),'median_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95)),
                                   'maximum_ms':float(max(timings)),'input_frames':1,'arithmetic_encoder_batch':cfg['feature_batch'],
                                   'precision':'autocast float16; patch tokens quantized float16'}
        del encoder
        if device=='cuda':torch.cuda.empty_cache()
    write(out/'latency.json',result);print('BENCHMARK',json.dumps(result),flush=True)


def plot(cfg, data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    comparison=read(out/'comparison.json');names=list(comparison);x=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for j,scope in enumerate(('whole','part')):
        for offset,key,label in [(-.18,'center_hit_rate','Softmax center'),(.18,'peak_hit_rate','Response peak')]:
            axes[j].bar(x+offset,[comparison[n][scope+'_4rot'][key] for n in names],width=.36,label=label)
        axes[j].set(xticks=x,xticklabels=names,ylim=(0,1),title=scope.title()+' visible-positive hit',ylabel='Fraction');axes[j].legend()
    for offset,scope,label in [(-.18,'whole','Whole'),(.18,'part','Part')]:
        axes[2].bar(x+offset,[comparison[n][scope+'_4rot']['verified_positive_recall'] for n in names],width=.36,label=label)
    axes[2].set(xticks=x,xticklabels=names,ylim=(0,1),title='Verified recall at calibrated 95% precision',ylabel='Fraction');axes[2].legend()
    fig.suptitle('Frozen-backbone development comparison; four template rotations; validation-calibrated thresholds')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    saved={n:read(out/n/'records.json') for n in names};base=saved[names[0]]['whole_4rot'];chosen=[]
    targets=data['catalog']['splits'][cfg['protocol']]['validation']
    for target in targets:
        rows=[r for r in base if r['target_id']==target]
        strata=range(4) if data['groups'][target]['category']=='aircraft' else [None]
        for s in strata:
            row=next((r for r in rows if r['positive'] and (s is None or r['range_stratum']==s)),None)
            if row is not None:chosen.append(row)
        row=next((r for r in rows if not r['positive']),None)
        if row is not None:chosen.append(row)
    lookup={n:{scope:{(r['query'],r['template'],r['positive']):r for r in rows} for scope,rows in saved[n].items()} for n in names}
    fig,axes=plt.subplots(len(chosen),len(names),figsize=(16,2.8*len(chosen)),squeeze=False)
    for i,row in enumerate(chosen):
        target=row['target_id'];sample=row['sample_name'];requested=target if row['positive'] else row['negative_template_id']
        with h5py.File(project_path(cfg['capture_directory'])/target/'samples.h5','r') as raw:
            g=raw[sample];image=redact_rgb(read_rgb(raw,sample),g['ignored_mask'][()]);mask=g['assembly_mask'][()]
        h,w=image.shape[:2];visible=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>0
        reference=cv2.imread(str(project_path(data['groups'][requested]['template']['file'])))[:,:,::-1]
        for j,name in enumerate(names):
            ax=axes[i,j];ax.set_axis_off();ax.imshow(image)
            if visible.any():ax.contour(visible.astype(float),levels=[.5],colors=['lime'],linewidths=.6)
            key=(row['query'],row['template'],row['positive']);labels=[]
            for scope,color,label in [('whole_4rot','cyan','whole'),('part_4rot','orange','part')]:
                r=lookup[name][scope][key];point=r['center_xy']
                if point is not None:
                    px,py=np.asarray(point)*[w,h];ax.scatter([px],[py],c=color,marker='o' if r['accepted'] else 'x',s=40)
                labels.append(label+f"={r['score']:.3f}")
            inset=ax.inset_axes([.01,.68,.22,.30]);inset.imshow(reference);inset.set_axis_off()
            roi=data['groups'][requested]['region']['projected_crop_roi']
            if data['groups'][requested]['region']['visible_in_nadir']:
                a,b,c,d=np.asarray(roi)*[reference.shape[1],reference.shape[0],reference.shape[1],reference.shape[0]]
                inset.add_patch(Rectangle((a,b),c-a,d-b,edgecolor='orange',fill=False,lw=1))
            ax.set_title(name+' '+sample+' '+('correct template' if row['positive'] else 'ABSENT template')+'\n'+requested+' '+', '.join(labels),fontsize=8)
    fig.suptitle('Fixed development examples; cyan: whole, orange: part; circle: accepted, cross: rejected; green: query target truth',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/'prediction-examples.png',dpi=130);plt.close(fig)
    write(out/'figure-selection.json',{'selection':'first aircraft positive per range; first other-target positive; first absent-template pair per target; independent of outcomes',
                                      'examples':[dict(sample_name=r['sample_name'],target_id=r['target_id'],positive=r['positive'],requested_template=r['target_id'] if r['positive'] else r['negative_template_id']) for r in chosen]})
    print('FIGURES_READY',len(chosen),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/evaluation/dino_backbone_comparison.json')
    parser.add_argument('--stage',choices=['evaluate','benchmark','plot'],default='evaluate');args=parser.parse_args()
    cfg=read(project_path(args.config));out=project_path(cfg['output_directory']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    data,pairs,sources=protocol(cfg);device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.stage=='plot':plot(cfg,data,out);return
    if args.stage=='benchmark':benchmark(cfg,data,out,device);return
    if (out/'comparison.json').exists():raise FileExistsError('Completed comparison exists')
    registration={'configuration':cfg,'configuration_sha256':digest(project_path(args.config)),'raw_sha256':sources,
                  'catalog_sha256':digest(project_path(cfg['catalog'])),'pair_plan_sha256':digest(project_path(cfg['reference_pair_plan'])),
                  'query_partitions_read':['train','validation'],'query_targets_read':data['targets'],'training_frames':len(data['split']['train']),
                  'validation_frames':len(data['split']['validation']),'validation_pairs':len(pairs),
                  'matching':'frozen normalized mean prototype; 1/4 reference rotations; online linspace softmax centroid is primary, grid-cell peak is secondary',
                  'roi':'fixed catalog projected_crop_roi; patch-center selection; invisible/empty template ROIs unavailable; query masks used for scoring only',
                  'normalization':'valid pixels in training frames only; ignored RGB regions redacted before encoding; 64-grid area weighting',
                  'precision':'CUDA autocast float16, patch tokens quantized float16, arithmetic batch fixed at feature_batch',
                  'scope':'development comparison; thresholds calibrated on this validation set; not independent test, not flight',
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'source_sha256':{p:digest(ROOT/p) for p in ['src/evaluation/eval_dino_backbones.py','src/perception/extract_dino_features.py','src/training/train_template_matcher.py']}}
    registration['encoders']={c['name']:{'weights_sha256':digest(project_path(c['weights'])),
        **repository_source(project_path(c['repository']))} for c in cfg['conditions']}
    write(out/'pre-registration.json',registration);write(out/'pair-plan.json',pairs)
    comparison={}
    for condition in cfg['conditions']:
        folder=out/condition['name'];folder.mkdir(exist_ok=True)
        comparison[condition['name']]=evaluate_condition(condition,cfg,data,pairs,folder,device)
    write(out/'comparison.json',comparison)
    write(out/'manifest.json',dict(registration,torch_version=torch.__version__,gpu=torch.cuda.get_device_name() if device=='cuda' else None))
    print('COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':main()
