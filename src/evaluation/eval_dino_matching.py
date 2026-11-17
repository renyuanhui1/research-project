"""Ablate feature normalization, point decoding and whole-image patch correspondence."""
import argparse
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from common.patches import read_rgb
from common.view_geometry import redact_rgb
from evaluation.eval_dino_backbones import (Encoder,read,write,digest,protocol,batches,reference_views,roi_patches,
                                           point_label,summary,repository_source)


def positions(grid,device=None,legacy=False):
    axis=torch.linspace(0,1,grid,device=device) if legacy else (torch.arange(grid,device=device)+.5)/grid
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    return torch.stack([x.flatten(),y.flatten()],-1)


def decode(response,mode,cfg):
    grid=math.isqrt(response.shape[-1]);peak=response.argmax(-1)
    xy=positions(grid,response.device,mode=='legacy')
    if mode=='peak':return xy[peak]
    temperature=cfg['soft_temperature'] if mode=='soft' else cfg['temperature']
    logits=response/temperature
    if mode=='local':
        index=torch.arange(grid*grid,device=response.device)
        keep=((index[None]//grid-peak[:,None]//grid).abs()<=cfg['local_radius_cells'])&((index[None]%grid-peak[:,None]%grid).abs()<=cfg['local_radius_cells'])
        logits=logits.masked_fill(~keep,-torch.inf)
    return logits.softmax(-1)@xy


def normalize(tokens,mode,mean,std):
    return F.normalize((tokens-mean)/std if mode=='standardized' else tokens,dim=-1)


def reference_bank(encoder,data,cfg):
    bank=[];masks=[]
    for target in data['targets']:
        t=data['groups'][target];image=cv2.imread(str(project_path(t['template']['file'])))[:,:,::-1].copy()
        roi=t['region']['projected_crop_roi'] if t['region']['visible_in_nadir'] else None
        images,boxes=reference_views(image,roi,encoder.condition['image_size'])
        tokens=torch.cat([encoder.encode(images[i:i+cfg['feature_batch']],cfg['feature_batch']) for i in range(0,4,cfg['feature_batch'])])
        grid=math.isqrt(tokens.shape[1]);bank.append(tokens);masks.append(np.stack([roi_patches(box,grid) for box in boxes]))
    return torch.stack(bank),np.stack(masks)


def prototype_bank(tokens,roi):
    full=F.normalize(tokens.mean(-2),dim=-1);parts=[]
    for views,masks in zip(tokens,roi):
        parts.append(torch.stack([F.normalize(z[torch.from_numpy(m).to(z.device)].mean(0),dim=0) if m.any() else torch.zeros_like(z[0]) for z,m in zip(views,masks)]))
    return {'whole':full,'part':torch.stack(parts)}


def patch_correspondence(sim,cfg,estimate_affine=True):
    """Mutual patch matches and affine inliers; no query geometry or labels."""
    nq,nt=sim.shape;best=sim.argmax(1);reverse=sim.argmax(0)
    q=np.flatnonzero(reverse[best]==np.arange(nq));t=best[q];score=sim[q,t]
    order=np.argsort(-score,kind='stable')[:cfg['maximum_correspondences']];q=q[order];t=t[order];score=score[order]
    qp=positions(math.isqrt(nq)).numpy()[q].astype(np.float32);tp=positions(math.isqrt(nt)).numpy()[t].astype(np.float32)
    inside=np.zeros(len(q),bool);coherence=0.
    if estimate_affine and len(q)>=3:
        transform,mask=cv2.estimateAffine2D(tp,qp,method=cv2.RANSAC,ransacReprojThreshold=cfg['reprojection_threshold'],
                                          maxIters=cfg['ransac_iterations'],confidence=cfg['ransac_confidence'],refineIters=10)
        if transform is not None and mask is not None and np.isfinite(transform).all():
            inside=mask[:,0].astype(bool)
            if inside.any():
                area_q=cv2.contourArea(cv2.convexHull(qp[inside])) if inside.sum()>=3 else 0.
                area_t=cv2.contourArea(cv2.convexHull(tp[inside])) if inside.sum()>=3 else 0.
                coherence=float(inside.sum()/min(nq,nt)*max(0.,float(score[inside].mean()))*math.sqrt(max(0.,area_q*area_t)))
    return {'q':q,'t':t,'scores':score,'q_xy':qp,'t_xy':tp,'inliers':inside,'coherence':coherence}


def correspondence_point(matches,roi_mask,method):
    # Select an observed match nearest the task anchor in the reference image.
    allowed=roi_mask[matches['t']]
    if method=='affine':allowed &= matches['inliers']
    candidates=np.flatnonzero(allowed)
    if not len(candidates):return None,-1.000001
    anchor=positions(math.isqrt(len(roi_mask))).numpy()[roi_mask].mean(0)
    distance=np.square(matches['t_xy'][candidates]-anchor).sum(1)
    choice=candidates[np.lexsort((-matches['scores'][candidates],distance))[0]]
    score=float(matches['scores'][choice])
    if method=='affine':
        score=matches['coherence']*len(candidates)/max(1,int(roi_mask.sum()))
    return matches['q_xy'][choice].tolist(),score


def row_record(row,point,score,rotation,available,scope,data,masks):
    label=masks[row['query']]['assembly_mask' if scope=='whole' else 'part_mask'].astype(bool)
    hit,error=point_label(label,point);record=data['records'][row['query']]
    return dict(row,sample_name=record['name'],category=record['category'],range_stratum=record['range_stratum'],
                horizontal_distance_m=record['horizontal_distance_m'],center_xy=point,score=float(score),rotation_deg=int(rotation)*90,
                template_available=bool(available),prediction_available=point is not None,visible_pixels=int(label.sum()),
                center_hit=hit,peak_hit=hit,signed_error_equivalent_radii=error)


def finalize(rows,cfg):
    result=summary(rows,cfg['calibration_precision']);threshold=result['threshold']
    result['missing_predictions']=sum(r['positive'] and not r['prediction_available'] for r in rows)
    result.pop('peak_hit_rate')
    for row in rows:
        row['accepted']=row['score']>=threshold;row['verified_correct']=bool(row['positive'] and row['center_hit'])
        row.pop('peak_hit')  # Internal compatibility field; each method has one actual prediction point.
    for field in ['target_id','range_stratum']:
        groups={}
        for value in sorted({r[field] for r in rows}):
            subset=[r for r in rows if r[field]==value];positive=[r for r in subset if r['positive'] and r['visible_pixels']>0]
            accepted=[r for r in subset if r['accepted']];correct=sum(r['verified_correct'] for r in accepted)
            groups[str(value)]={'visible_positive':len(positive),'hits':sum(r['center_hit'] for r in positive),
                                'center_hit_rate':sum(r['center_hit'] for r in positive)/len(positive) if positive else None,
                                'accepted':len(accepted),'correct_accepted':correct,'accepted_precision':correct/len(accepted) if accepted else None,
                                'verified_positive_recall':correct/len(positive) if positive else None}
        result['by_'+field]=groups
    return result


def evaluate(condition,cfg,data,pairs,out,device):
    started=time.monotonic();encoder=Encoder(condition,device)
    baseline=project_path(cfg['baseline_directory']);state=np.load(baseline/condition['name']/'reference-state.npz')
    mean=torch.from_numpy(state['mean']).to(device);std=torch.from_numpy(state['std']).to(device)
    references,roi=reference_bank(encoder,data,cfg);query={};masks={}
    for indices,images,labels in batches(data,cfg,'validation',cfg['feature_batch']):
        z=encoder.encode(images,cfg['feature_batch'])
        for i,tokens,label in zip(indices,z,labels):query[i]=tokens;masks[i]=label
    predictions={}
    for norm in cfg['normalizations']:
        ref=normalize(references,norm,mean,std);protos=prototype_bank(ref,roi)
        q={i:normalize(z,norm,mean,std) for i,z in query.items()}
        for scope,proto in protos.items():
            rows={mode:[] for mode in cfg['readouts'] if mode not in ['mutual','affine']}
            for start in range(0,len(pairs),32):
                selected=pairs[start:start+32];qi=[r['query'] for r in selected];ti=[r['template'] for r in selected]
                similarity=torch.einsum('bnd,brd->brn',torch.stack([q[i] for i in qi]),proto[ti])
                scores,rotation=similarity.max(-1).values.max(-1);response=similarity[torch.arange(len(selected),device=device),rotation]
                scores=scores.cpu().numpy();rotation=rotation.cpu().numpy()
                for mode in rows:
                    points=decode(response,mode,cfg).cpu().numpy()
                    for row,point,score,k in zip(selected,points,scores,rotation):
                        available=scope=='whole' or bool(roi[row['template'],k].any())
                        rows[mode].append(row_record(row,point.tolist() if available else None,score if available else -1.000001,k,available,scope,data,masks))
            for mode,entries in rows.items():predictions[f'{norm}/{mode}/{scope}']=entries
        dense={(method,scope):[] for method in cfg['readouts'] if method in ['mutual','affine'] for scope in ['whole','part']}
        for index,row in enumerate(pairs):
            similarities=torch.einsum('qd,rtd->rqt',q[row['query']],ref[row['template']]).cpu().numpy()
            matches=[patch_correspondence(s,cfg) for s in similarities]
            for (method,scope),entries in dense.items():
                trials=[]
                for k,m in enumerate(matches):
                    selected=roi[row['template'],k] if scope=='part' else np.ones(len(roi[row['template'],k]),bool)
                    point,score=correspondence_point(m,selected,method);trials.append((score,k,point,bool(selected.any())))
                score,k,point,available=max(trials,key=lambda x:x[0])
                entry=row_record(row,point,score,k,available,scope,data,masks);m=matches[k]
                selected=roi[row['template'],k] if scope=='part' else np.ones(len(roi[row['template'],k]),bool)
                entry.update(mutual_matches=len(m['q']),affine_inliers=int(m['inliers'].sum()),
                             roi_correspondences=int(selected[m['t']].sum()),
                             inlier_roi_correspondences=int((selected[m['t']]&m['inliers']).sum()),coherence=m['coherence'])
                entries.append(entry)
            if (index+1)%100==0:print('PATCH_MATCHED',condition['name'],norm,index+1,'/',len(pairs),flush=True)
        for (method,scope),entries in dense.items():predictions[f'{norm}/{method}/{scope}']=entries
    metrics={name:finalize(rows,cfg) for name,rows in predictions.items()}
    parity={}
    for scope in ['whole','part']:
        old=read(baseline/condition['name']/'comparison.json')[scope+'_4rot'];new=metrics[f'standardized/legacy/{scope}']
        parity[scope]={'previous_hits':old['center_hits'],'current_hits':new['center_hits'],'matched':old['center_hits']==new['center_hits']}
        if not parity[scope]['matched']:raise ValueError('Ablation baseline changed: '+condition['name']+' '+scope)
    folder=out/condition['name'];folder.mkdir(exist_ok=True)
    write(folder/'records.json',predictions);write(folder/'comparison.json',metrics);write(folder/'baseline-parity.json',parity)
    print('CONDITION_DONE',condition['name'],'seconds',time.monotonic()-started,flush=True)
    del encoder,references,ref,query,q
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return metrics


def selections(comparison):
    result={}
    for condition,methods in comparison.items():
        result[condition]={}
        for scope in ['whole','part']:
            pool={k:v for k,v in methods.items() if k.endswith('/'+scope)}
            raw=max(pool,key=lambda k:(pool[k]['center_hit_rate'] or 0,pool[k]['verified_positive_recall'] or 0))
            trusted=max(pool,key=lambda k:(pool[k]['verified_positive_recall'] or 0,pool[k]['verified_auc']))
            fields=['center_hits','visible_positive','center_hit_rate','accepted','correct_accepted','accepted_precision','verified_positive_recall','wrong_template_false_accept_rate']
            result[condition][scope]={kind:dict(method=name,**{k:pool[name][k] for k in fields}) for kind,name in [('raw_best',raw),('accepted_best',trusted)]}
    return result


def plot(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison=read(out/'comparison.json');fig,axes=plt.subplots(2,len(comparison),figsize=(19,9),squeeze=False)
    methods=cfg['readouts'];x=np.arange(len(methods));colors=['#2676aa','#df8430']
    for j,(condition,metrics) in enumerate(comparison.items()):
        for i,scope in enumerate(['whole','part']):
            ax=axes[i,j]
            for k,norm in enumerate(cfg['normalizations']):
                ax.bar(x+(k-.5)*.36,[metrics[f'{norm}/{method}/{scope}']['center_hit_rate'] for method in methods],width=.36,label=norm,color=colors[k])
            ax.set(xticks=x,xticklabels=methods,ylim=(0,1),title=condition+' '+scope,ylabel='Visible-positive hit rate');ax.tick_params(axis='x',rotation=40);ax.legend(fontsize=8)
    fig.suptitle('Development ablation: same 383 template pairs; all settings shown; no independent test')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print('FIGURE_READY',flush=True)


def benchmark(cfg,out):
    data,_,_=protocol(cfg);indices=data['split']['validation'];chosen=set(indices[i] for i in np.unique(np.linspace(0,len(indices)-1,min(cfg['benchmark_frames'],len(indices))).astype(int)))
    images=[];targets=[]
    for ids,rgbs,_ in batches(data,cfg,'validation',1):
        if ids[0] in chosen:images.extend(rgbs);targets.append(data['targets'].index(data['records'][ids[0]]['target_id']))
    selected=read(out/'selection.json');device='cuda' if torch.cuda.is_available() else 'cpu'
    results={'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'torch_version':torch.__version__,
             'scope':'RGB array to one requested whole/part location; reference encoded; includes resize, transfer and matching; excludes file I/O, depth and control; single input padded to configured encoder batch',
             'source_sha256':digest(Path(__file__)),'profiles':{}}
    for condition in cfg['conditions']:
        encoder=Encoder(condition,device);state=np.load(project_path(cfg['baseline_directory'])/condition['name']/'reference-state.npz')
        mean=torch.from_numpy(state['mean']).to(device);std=torch.from_numpy(state['std']).to(device);raw,roi=reference_bank(encoder,data,cfg)
        profiles=list(dict.fromkeys(['standardized/legacy/whole','standardized/legacy/part',selected[condition['name']]['whole']['raw_best']['method'],selected[condition['name']]['part']['raw_best']['method']]))
        references={norm:normalize(raw,norm,mean,std) for norm in cfg['normalizations']};protos={norm:prototype_bank(z,roi) for norm,z in references.items()}
        results['profiles'][condition['name']]={}
        for profile in profiles:
            norm,mode,scope=profile.split('/');times=[]
            for j in range(cfg['benchmark_warmup']+len(images)):
                k=max(0,j-cfg['benchmark_warmup'])%len(images);target=targets[k]
                if device=='cuda':torch.cuda.synchronize()
                started=time.perf_counter();q=normalize(encoder.encode([images[k]],cfg['feature_batch'])[0],norm,mean,std)
                if mode in ['mutual','affine']:
                    sims=torch.einsum('qd,rtd->rqt',q,references[norm][target]).cpu().numpy();trials=[]
                    for rotation,s in enumerate(sims):
                        matches=patch_correspondence(s,cfg,estimate_affine=mode=='affine');mask=roi[target,rotation] if scope=='part' else np.ones(len(roi[target,rotation]),bool)
                        trials.append(correspondence_point(matches,mask,mode))
                    prediction=max(trials,key=lambda p:p[1])
                else:
                    response=q@protos[norm][scope][target].T;rotation=int(response.max(0).values.argmax());prediction=decode(response[:,rotation][None],mode,cfg).cpu().numpy()
                if device=='cuda':torch.cuda.synchronize()
                if j>=cfg['benchmark_warmup']:times.append((time.perf_counter()-started)*1000)
            results['profiles'][condition['name']][profile]={'frames':len(times),'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
                                                           'maximum_ms':float(max(times)),'logical_batch':1,'encoder_batch':cfg['feature_batch']}
        del encoder,raw,references,protos
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write(out/'latency.json',results);print('BENCHMARK_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/evaluation/dino_matching_ablation.json');parser.add_argument('--plot',action='store_true');parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
    experiment=read(project_path(args.config));cfg=dict(read(project_path(experiment['base_config'])),**experiment);out=project_path(cfg['output_directory']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.plot:plot(cfg,out);return
    if args.benchmark:benchmark(cfg,out);return
    if (out/'comparison.json').exists():raise FileExistsError('Completed ablation exists')
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);cv2.setRNGSeed(cfg['seed'])
    data,pairs,sources=protocol(cfg);baseline=project_path(cfg['baseline_directory']);prior=read(baseline/'pre-registration.json')
    if sources!=prior['raw_sha256']:raise ValueError('Baseline raw inputs changed')
    registration={'configuration':cfg,'configuration_sha256':digest(project_path(args.config)),'base_configuration_sha256':digest(project_path(experiment['base_config'])),
                  'query_partitions_read':['train','validation'],'query_targets_read':data['targets'],'validation_frames':len(data['split']['validation']),'pairs':len(pairs),'raw_sha256':sources,
                  'normalization':'reuse baseline training-only means/std, or L2 only; no validation fitting of features',
                  'prototype_readouts':'legacy endpoint centroid; cell-center centroid; peak; cell-center centroid at soft_temperature; peak-local centroid',
                  'spatial_matching':'mutual whole-image patch matches; optional affine RANSAC; select observed match nearest reference anchor, restricted to template ROI for part; no query ROI or labels',
                  'scope':'development ablation; thresholds and best methods selected on same validation data; no independent test or flight',
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'source_sha256':{p:digest(ROOT/p) for p in ['src/evaluation/eval_dino_matching.py','src/evaluation/eval_dino_backbones.py','src/training/train_template_matcher.py']}}
    registration['encoders']={c['name']:{'weights_sha256':digest(project_path(c['weights'])),**repository_source(project_path(c['repository'])),
                                        'baseline_state_sha256':digest(baseline/c['name']/'reference-state.npz')} for c in cfg['conditions']}
    write(out/'pre-registration.json',registration);write(out/'pair-plan.json',pairs);device='cuda' if torch.cuda.is_available() else 'cpu';comparison={}
    for condition in cfg['conditions']:comparison[condition['name']]=evaluate(condition,cfg,data,pairs,out,device)
    write(out/'comparison.json',comparison);write(out/'selection.json',selections(comparison));write(out/'manifest.json',dict(registration,torch_version=torch.__version__,gpu=torch.cuda.get_device_name() if device=='cuda' else None))
    print('ABLATION_COMPLETE',flush=True)


if __name__=='__main__':main()
