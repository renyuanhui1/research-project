"""Train whole-target matching heads and calibrate confidence on held-out target assets."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from common.view_geometry import visible_groups
from perception.template_matcher import TemplateMatcher,ConfidenceHead,localization_loss,template_pair_loss
from training.extract_template_features import digest,read


def load_features(cfg,device,partitions=('train','validation','test')):
    directory=project_path(cfg['feature_directory']);index=read(directory/'index.json');catalog=read(project_path(cfg['catalog']))
    if index['signature']['catalog_sha256']!=digest(project_path(cfg['catalog'])):raise ValueError('Catalog changed')
    if digest(directory/index['template_file'])!=index['template_sha256']:raise ValueError('Template cache changed')
    groups={t['id']:t for t in catalog['targets']};records=[];z=[];mask=[];valid=[];native=[];templates=[];raw_hashes={}
    split=catalog['splits'][cfg['protocol']]
    allowed=set().union(*(set(split[k]) for k in partitions))
    targets=[t['id'] for t in catalog['targets'] if t['id'] in allowed];entry={x['target_id']:x for x in index['targets']}
    with h5py.File(directory/index['template_file'],'r') as f:
        for target in targets:templates.append(f[target]['z'][()])
        variants=json.loads(f[targets[0]].attrs['variants_json'])
    for target in targets:
        path=directory/entry[target]['file']
        if digest(path)!=entry[target]['sha256']:raise ValueError('Query cache changed: '+target)
        with h5py.File(path,'r') as f:
            raw_hashes[target]=f.attrs.get('raw_sha256')
            z.append(f['z'][()]);mask.append(f['mask64'][()]);valid.append(f['valid64'][()]);native.extend(f['native_mask'][()])
            records.extend(json.loads(v) for v in f['metadata'].asstr()[()])
    data={'z':torch.from_numpy(np.concatenate(z)).to(device),
          'mask':torch.from_numpy(np.concatenate(mask)).float().to(device),
          'valid':torch.from_numpy(np.concatenate(valid)).float().to(device),
          'template':torch.from_numpy(np.stack(templates)).to(device),
          'native':native,'records':records,'targets':targets,'catalog':catalog,'groups':groups,'variants':variants,
          'feature_index_sha256':digest(directory/'index.json'),'raw_sha256':raw_hashes}
    data['split']={key:[i for i,r in enumerate(records) if r['target_id'] in split[key]] for key in partitions}
    if any(not data['split'][key] for key in data['split']):raise ValueError('An experiment partition has no samples')
    if cfg.get('context_labels'):
        folder=project_path(cfg['context_labels']);report=read(folder/'report.json')
        if report['status']!='completed' or report['partition']!='train' or report['catalog_sha256']!=digest(project_path(cfg['catalog'])):raise ValueError('Context labels disagree with training protocol')
        if digest(folder/'labels.h5')!=report['labels_sha256']:raise ValueError('Context label cache changed')
        if any(raw_hashes.get(tid)!=sha for tid,sha in report['raw_sha256'].items()):raise ValueError('Context labels and features come from different captures')
        lookup={r['name']:i for i,r in enumerate(records)};train=set(split['train'])
        data.update(context_targets={},context_mask={},context_valid={},context_report_sha256=digest(folder/'report.json'))
        with h5py.File(folder/'labels.h5','r') as labels:
            for sample in report['samples']:
                qi=lookup[sample['sample_name']]
                if records[qi]['target_id'] not in train or records[qi]['target_id']!=sample['query_target_id']:raise ValueError('Context supervision outside its training query')
                ids=labels[sample['sample_name']]['ids'][()];valid=labels[sample['sample_name']]['valid'][()]
                shape=(cfg['output_grid'],cfg['output_grid'])
                data['context_valid'][qi]=torch.from_numpy(cv2.resize(valid.astype(np.float32),shape,interpolation=cv2.INTER_AREA)).to(device)
                data['context_targets'][qi]=[]
                for target in sample['label_targets']:
                    tid=target['target_id']
                    if tid not in train:raise ValueError('Context supervision outside training templates')
                    data['context_targets'][qi].append(tid)
                    mask=cv2.resize((ids==target['index']).astype(np.float32),shape,interpolation=cv2.INTER_AREA)
                    data['context_mask'][qi,tid]=torch.from_numpy(mask).to(device)
    return data


def fitting_statistics(data):
    count=0.;total=torch.zeros(384,dtype=torch.float64,device=data['z'].device);squares=total.clone()
    for begin in range(0,len(data['split']['train']),32):
        indices=data['split']['train'][begin:begin+32];z=data['z'][indices].double()
        grid=math.isqrt(z.shape[1]);weights=F.interpolate(data['valid'][indices,None],size=(grid,grid),mode='area').flatten(2).transpose(1,2).double()
        total+=(z*weights).sum((0,1));squares+=(z.square()*weights).sum((0,1));count+=float(weights.sum())
    if count<=0:raise ValueError('No known training pixels for normalization')
    mean=total/count;std=(squares/count-mean.square()).clamp_min(1e-6).sqrt()
    return mean.float(),std.float()


def pair_plan(data,cfg):
    rng=np.random.default_rng(cfg['seed']);split=data['catalog']['splits'][cfg['protocol']];result={}
    for partition,indices in data['split'].items():
        allowed=set(split['train'])|set(split[partition])
        if partition=='train' and 'training_template_ids' in data:allowed &= set(data['training_template_ids'])
        rows=[]
        for i in indices:
            record=data['records'][i];target=record['target_id'];target_record=data['groups'][target]
            rows.append({'query':i,'template':data['targets'].index(target),'positive':True,'target_id':target})
            m=record['camera_metadata'];position=[m['pos_'+k] for k in ('x','y','z')]
            rotation=Rotation.from_quat([m['rot_'+k] for k in ('x','y','z','w')]).as_matrix()
            fov=record['horizontal_fov_deg']+cfg['negative_visibility_margin_deg']
            visible=visible_groups(data['catalog']['targets'],position,rotation,fov,m['width']/m['height'])
            pool=[t for t in data['catalog']['targets'] if t['id'] in allowed and
                  t['appearance_group']!=target_record['appearance_group'] and t['appearance_group'] not in visible]
            same=[t for t in pool if t['category']==target_record['category']]
            pool=same or pool
            if pool:
                other=pool[int(rng.integers(len(pool)))]
                rows.append({'query':i,'template':data['targets'].index(other['id']),'positive':False,
                             'target_id':target,'negative_template_id':other['id'],
                             'negative_absence_evidence':'all catalog instances of its primary mesh group are outside an expanded view frustum'})
            if partition=='train':
                for tid in data.get('context_targets',{}).get(i,[]):
                    if tid in allowed:rows.append({'query':i,'template':data['targets'].index(tid),'positive':True,
                                                  'target_id':target,'context_positive':True})
        result[partition]=rows
    return result


def inputs(data,rows,mean,std,augment=False):
    qi=[r['query'] for r in rows];ti=[r['template'] for r in rows]
    variant=torch.randint(data['template'].shape[1],(len(rows),),device=data['z'].device) if augment else torch.zeros(len(rows),dtype=torch.long,device=data['z'].device)
    return ((data['z'][qi].float()-mean)/std,
            (data['template'][ti,variant].float()-mean)/std)


def point_from_map(logits):
    grid=logits.shape[-1];index=logits.flatten(1).argmax(-1)
    return torch.stack([(index%grid+.5)/grid,(index//grid+.5)/grid],-1).cpu().numpy()


def correct_point(data,row,point):
    mask=data['native'][row['query']];h,w=mask.shape;x,y=int(math.floor(point[0]*w)),int(math.floor(point[1]*h))
    return bool(row['positive'] and 0<=x<w and 0<=y<h and mask[y,x])


@torch.no_grad()
def predict_network(model,data,rows,mean,std,batch):
    model.eval();points=[];features=[]
    for start in range(0,len(rows),batch):
        q,t=inputs(data,rows[start:start+batch],mean,std)
        with torch.autocast(q.device.type,dtype=torch.bfloat16,enabled=q.is_cuda):logits,stats=model(q,t)
        points.extend(point_from_map(logits));features.append(stats)
    return np.array(points),torch.cat(features)


def response_scores(features,output_grid):
    # Recover logsumexp from the peak and its spatial-softmax probability.
    return (features[:,0].double()-features[:,5].double().clamp_min(1e-30).log()-math.log(output_grid**2)).sigmoid().cpu().numpy()


def ranking_auc(scores,correct):
    scores=np.asarray(scores);correct=np.asarray(correct,bool);p=scores[correct];n=scores[~correct]
    if not len(p) or not len(n):return .5
    return float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))


def selection_metrics(data,rows,points,scores,precision):
    correct=[correct_point(data,r,p) for r,p in zip(rows,points)]
    threshold,calibration=calibrate(scores,correct,precision)
    result=metrics(data,rows,points,scores,threshold)
    return result,calibration


def selection_key(result,template_conditioned=True):
    if not template_conditioned:return (result['macro_target_hit'],)
    return (result['verified_positive_recall'] or 0.,result['verified_auc'],result['macro_target_hit'])


def hit_rate(data,rows,points):
    buckets=defaultdict(list)
    for row,point in zip(rows,points):
        if row['positive'] and np.any(data['native'][row['query']]):buckets[row['target_id']].append(correct_point(data,row,point))
    return float(np.mean([np.mean(values) for values in buckets.values()]))


def supervision(data,rows):
    masks=[];valid=[]
    for row in rows:
        qi=row['query']
        if row.get('context_positive'):
            masks.append(data['context_mask'][qi,data['targets'][row['template']]]);valid.append(data['context_valid'][qi])
        else:masks.append(data['mask'][qi]);valid.append(data['valid'][qi])
    return torch.stack(masks),torch.stack(valid)


def train_localizer(conditioned,data,pairs,mean,std,cfg,output,pair_weight=0.,select=True):
    torch.manual_seed(cfg['seed']);model=TemplateMatcher(hidden_dim=cfg['hidden_dim'],temperature=cfg['attention_temperature'],
        conditioned=conditioned,output_grid=cfg['output_grid']).to(data['z'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    train=[r for r in pairs['train'] if r['positive'] and (conditioned or not r.get('context_positive'))]
    negatives={r['query']:r for r in pairs['train'] if not r['positive']}
    rng=np.random.default_rng(cfg['seed']);best=None;best_state=None;history=[]
    for epoch in range(cfg['epochs']):
        model.train();order=rng.permutation(len(train));losses=[];started=time.monotonic()
        for begin in range(0,len(order),cfg['batch']):
            rows=[train[int(i)] for i in order[begin:begin+cfg['batch']]];qi=[r['query'] for r in rows]
            q,t=inputs(data,rows,mean,std,augment=True)
            target,known=supervision(data,rows)
            with torch.autocast(q.device.type,dtype=torch.bfloat16,enabled=q.is_cuda):logits,_=model(q,t)
            matched=[j for j,r in enumerate(rows) if r['query'] in negatives] if conditioned else []
            if matched:
                wrong=[negatives[rows[j]['query']] for j in matched];nq,nt=inputs(data,wrong,mean,std,augment=True)
                with torch.autocast(q.device.type,dtype=torch.bfloat16,enabled=q.is_cuda):negative,_=model(nq,nt)
                loss=localization_loss(torch.cat([logits,negative]),torch.cat([target,torch.zeros_like(target[matched])]),torch.cat([known,known[matched]]))
                loss=loss+pair_weight*template_pair_loss(logits[matched],negative,target[matched],known[matched],cfg.get('pair_margin',1.))
            else:loss=localization_loss(logits,target,known)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.monotonic()-started}
        if select:
            points,features=predict_network(model,data,pairs['validation'],mean,std,cfg['batch'])
            result,calibration=selection_metrics(data,pairs['validation'],points,response_scores(features,cfg['output_grid']),cfg['calibration_precision'])
            key=selection_key(result,conditioned)
            row.update(validation_macro_hit=result['macro_target_hit'],validation_verified_recall=result['verified_positive_recall'],validation_verified_auc=result['verified_auc'],validation_accepted=calibration['accepted'])
        else:key=(epoch,)
        if best is None or key>best:
            best=key;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1;best_result=dict(row)
        history.append(row);(output/'history.json').write_text(json.dumps(history,indent=2))
        print('EPOCH','conditioned' if conditioned else 'search_only',json.dumps(row),flush=True)
    model.load_state_dict(best_state)
    return model,dict(best_epoch=best_epoch,**{k:v for k,v in best_result.items() if k.startswith('validation_')})


def confidence_fold_plan(data,count,seed):
    groups=sorted({data['groups'][data['records'][i]['target_id']]['appearance_group'] for i in data['split']['train']})
    rng=np.random.default_rng(seed);rng.shuffle(groups);result=[]
    for held_groups in np.array_split(groups,count):
        held=set(held_groups);fit=[];query=[]
        for i in data['split']['train']:
            group=data['groups'][data['records'][i]['target_id']]['appearance_group']
            (query if group in held else fit).append(i)
        result.append({'held_groups':sorted(held),'fit_queries':fit,'held_queries':query,
                       'fit_template_ids':sorted({data['records'][i]['target_id'] for i in fit})})
    return result


def confidence_fit(data,pairs,cfg,output,pair_weight,epochs):
    rows=[r for r in pairs['train'] if not r.get('context_positive')];features=torch.empty(len(rows),10,device=data['z'].device);points=np.zeros((len(rows),2));folds=confidence_fold_plan(data,cfg['confidence_folds'],cfg['seed'])
    for fold,plan in enumerate(folds):
        folder=output/f'confidence-fold-{fold}';folder.mkdir(exist_ok=True)
        subset=dict(data,split={'train':plan['fit_queries']},training_template_ids=plan['fit_template_ids'])
        mean,std=fitting_statistics(subset);fold_cfg=dict(cfg,epochs=epochs)
        fold_pairs=pair_plan(subset,fold_cfg)
        (folder/'pair-plan.json').write_text(json.dumps(fold_pairs,indent=2))
        model,selection=train_localizer(True,subset,fold_pairs,mean,std,fold_cfg,folder,pair_weight,select=False)
        held=set(plan['held_queries']);indices=[j for j,r in enumerate(rows) if r['query'] in held]
        xy,stats=predict_network(model,data,[rows[j] for j in indices],mean,std,cfg['batch'])
        features[indices]=stats;points[indices]=xy
        torch.save({'model':model.state_dict(),'mean':mean.cpu(),'std':std.cpu(),'fold':plan,'configuration':fold_cfg},folder/'model.pt')
        (folder/'fold.json').write_text(json.dumps(plan,indent=2))
    correct=[correct_point(data,r,p) for r,p in zip(rows,points)]
    np.savez_compressed(output/'confidence-oof.npz',stats=features.cpu().numpy(),points=points,correct=correct)
    target=torch.tensor(correct,device=features.device,dtype=torch.float32)
    center=features.mean(0);scale=features.std(0).clamp_min(1e-4);x=(features-center)/scale
    torch.manual_seed(cfg['seed']);head=ConfidenceHead().to(x.device);optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['confidence_learning_rate'])
    for _ in range(cfg['confidence_epochs']):
        logits=head(x);loss=F.binary_cross_entropy_with_logits(logits,target)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return head.eval(),center,scale


def calibrate(scores,correct,required_precision):
    order=np.argsort(-np.asarray(scores),kind='stable');s=np.asarray(scores)[order];c=np.asarray(correct,dtype=int)[order]
    last=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];precision=np.cumsum(c)[last]/(last+1)
    feasible=last[precision>=required_precision]
    if not len(feasible):return 1.000001,{'accepted':0,'precision':None,'status':'no_nonempty_threshold_meets_precision'}
    index=int(feasible[-1]);threshold=float(s[index])
    if index+1<len(s):
        midpoint=float(s[index]+(s[index+1]-s[index])/2)
        if midpoint>s[index+1]:threshold=midpoint
    return threshold,{'accepted':index+1,'precision':float(c[:index+1].mean()),'status':'calibrated_on_validation_targets'}


def metrics(data,rows,points,scores,threshold):
    accepted=np.asarray(scores)>=threshold;correct=np.array([correct_point(data,r,p) for r,p in zip(rows,points)])
    positive=np.array([r['positive'] for r in rows]);n=int(accepted.sum());negative=~positive
    visible_positive=positive&np.array([bool(np.any(data['native'][r['query']])) for r in rows])
    result={'pairs':len(rows),'positive_pairs':int(positive.sum()),'negative_pairs':int(negative.sum()),
            'visible_positive_pairs':int(visible_positive.sum()),
            'raw_positive_hit':float(correct[visible_positive].mean()) if visible_positive.any() else None,
            'macro_target_hit':hit_rate(data,rows,points),
            'accepted':n,'accepted_verified_correct':int((accepted&correct).sum()),
            'accepted_precision':float(correct[accepted].mean()) if n else None,
            'pair_coverage':float(accepted.mean()),
            'verified_positive_recall':float((accepted&correct).sum()/visible_positive.sum()) if visible_positive.any() else None,
            'negative_false_accept_rate':float(accepted[negative].mean()) if negative.any() else None,
            'verified_auc':ranking_auc(scores,correct)}
    return result


@torch.no_grad()
def predict_fingerprint(data,rows,mean,std,batch,rotations):
    points=[];scores=[]
    grid=math.isqrt(data['z'].shape[1]);axis=torch.linspace(0,1,grid,device=data['z'].device)
    y,x=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack([x.flatten(),y.flatten()],-1)
    for begin in range(0,len(rows),batch):
        selected=rows[begin:begin+batch];qi=[r['query'] for r in selected];ti=[r['template'] for r in selected]
        query=F.normalize((data['z'][qi].float()-mean)/std,dim=-1)
        template=(data['template'][ti,:rotations].float()-mean)/std
        prototype=F.normalize(F.normalize(template,dim=-1).mean(-2),dim=-1)
        similarity=torch.einsum('bnd,brd->brn',query,prototype)
        peak=similarity.max(-1).values;choice=peak.argmax(-1);response=similarity[torch.arange(len(selected),device=query.device),choice]
        center=torch.softmax(response/.005,-1)@xy
        points.extend(center.cpu().numpy());scores.extend(peak.max(-1).values.cpu().numpy())
    return np.array(points),np.array(scores)


def evaluate_method(data,pairs,predict,cfg,directory,frozen,partition='test'):
    threshold=frozen['threshold']
    # Test is first evaluated after checkpoint selection and validation calibration are frozen.
    rows=pairs[partition];tp,ts=predict(rows);summary=dict(frozen,**{partition:metrics(data,rows,tp,ts,threshold)})
    records=[]
    for row,point,score in zip(rows,tp,ts):
        records.append(dict(row,point_xy=list(map(float,point)),score=float(score),accepted=bool(score>=threshold),
                            verified_correct=correct_point(data,row,point),
                            sample_name=data['records'][row['query']]['name'],
                            horizontal_distance_m=data['records'][row['query']]['horizontal_distance_m'],
                            category=data['records'][row['query']]['category']))
    for scope,key in [('by_target','target_id'),('by_category','category')]:
        summary[partition][scope]={}
        for value in sorted({r[key] for r in records}):
            selection=[i for i,r in enumerate(records) if r[key]==value]
            summary[partition][scope][value]=metrics(data,[rows[i] for i in selection],tp[selection],ts[selection],threshold)
    summary[partition]['by_range']={}
    for low,high in [(12,30),(30,60),(60,100),(100,150.0001)]:
        indices=[i for i,r in enumerate(records) if low<=r['horizontal_distance_m']<high]
        if indices:summary[partition]['by_range'][f'{low}-{int(high)}']=metrics(data,[rows[i] for i in indices],tp[indices],ts[indices],threshold)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/(partition+'-records.json')).write_text(json.dumps(records,indent=2));(directory/'metrics.json').write_text(json.dumps(summary,indent=2))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/training/airport_template_matcher.json')
    options=parser.parse_args(argv);path=project_path(options.config);cfg=read(path);output=project_path(cfg['output_directory'])
    if (output/'comparison.json').exists():raise ValueError('Use an output directory without completed results')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    partition=cfg.get('evaluation_partition','validation')
    partitions=('train','validation') if partition=='validation' else ('train','validation',partition)
    device='cuda' if torch.cuda.is_available() else 'cpu';data=load_features(cfg,device,partitions);mean,std=fitting_statistics(data);pairs=pair_plan(data,cfg)
    (output/'pair-plan.json').write_text(json.dumps(pairs,indent=2));print('PAIRS',{k:len(v) for k,v in pairs.items()},flush=True)
    comparison={};methods=[];frozen={};checkpoints={}
    for rotations in [1,4]:
        name=f'fingerprint_{rotations}rot'
        def predict(rows,rotations=rotations):return predict_fingerprint(data,rows,mean,std,cfg['batch'],rotations)
        methods.append((name,predict,None))
    variants=[{'name':'search_only','pair_weight':0.,'conditioned':False}]+[dict(v,conditioned=True) for v in cfg['paired_variants']]
    for variant in variants:
        name=variant['name'];conditioned=variant['conditioned']
        directory=output/name;directory.mkdir(exist_ok=True)
        model,selection=train_localizer(conditioned,data,pairs,mean,std,cfg,directory,variant['pair_weight'])
        head,center,scale=confidence_fit(data,pairs,cfg,directory,variant['pair_weight'],selection['best_epoch']) if conditioned else (None,None,None)
        checkpoint={'model':model.state_dict(),'confidence':head.state_dict() if head is not None else None,'mean':mean.cpu(),'std':std.cpu(),
                    'confidence_center':center.cpu() if center is not None else None,'confidence_scale':scale.cpu() if scale is not None else None,'configuration':cfg,'conditioned':conditioned,
                    'selection':selection,'pair_weight':variant['pair_weight'],'thresholds':{}}
        checkpoints[name]=checkpoint
        for mode in (['presence','crossfit'] if conditioned else ['presence']):
            def predict(rows,model=model,head=head,center=center,scale=scale,mode=mode):
                points,features=predict_network(model,data,rows,mean,std,cfg['batch'])
                if mode=='presence':scores=response_scores(features,cfg['output_grid'])
                else:
                    with torch.no_grad():scores=head((features-center)/scale).double().sigmoid().cpu().numpy()
                return points,scores
            methods.append((name+'/'+mode,predict,name))
    for name,predict,group in methods:
        points,scores=predict(pairs['validation']);correct=[correct_point(data,r,p) for r,p in zip(pairs['validation'],points)]
        threshold,calibration=calibrate(scores,correct,cfg['calibration_precision'])
        frozen[name]={'threshold':threshold,'calibration':calibration,'validation':metrics(data,pairs['validation'],points,scores,threshold)}
        if group:
            checkpoints[group]['thresholds'][name.split('/')[-1]]=threshold
            frozen[name]['checkpoint_selection']=checkpoints[group]['selection']
    for group,checkpoint in checkpoints.items():torch.save(checkpoint,output/group/'model.pt')
    for name,_,group in methods:
        if group:
            frozen[name]['checkpoint_file']=group+'/model.pt'
            frozen[name]['checkpoint_sha256']=digest(output/group/'model.pt')
    (output/'frozen-validation-selection.json').write_text(json.dumps(frozen,indent=2))
    eligible=[name for name in frozen if '/' in name and not name.startswith('search_only/')]
    selected=max(eligible,key=lambda name:selection_key(frozen[name]['validation']))
    result=frozen[selected]['validation']
    decision={'selected_method':selected,'development_gate_passed':bool((result['accepted_precision'] or 0)>=cfg['calibration_precision'] and (result['verified_positive_recall'] or 0)>=cfg['minimum_verified_recall']),
              'required_precision':cfg['calibration_precision'],'minimum_verified_recall':cfg['minimum_verified_recall'],
              'validation':result,'scope':'validation development selection; independent test and continuous flight are unverified'}
    (output/'selection.json').write_text(json.dumps(decision,indent=2))
    for name,predict,_ in methods:
        comparison[name]=evaluate_method(data,pairs,predict,cfg,output/name,frozen[name],partition)
        print('RESULT',name,json.dumps({k:v for k,v in comparison[name][partition].items() if not k.startswith('by_')}),flush=True)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    sources=['src/perception/template_matcher.py','src/training/train_template_matcher.py','src/training/extract_template_features.py','src/common/view_geometry.py']
    manifest={'configuration':cfg,'configuration_sha256':digest(path),'feature_index_sha256':data['feature_index_sha256'],
              'source_sha256':{p:digest(ROOT/p) for p in sources},'torch_version':torch.__version__,'device':device,
              'normalization_scope':'only valid pixels in training search features; no validation or test statistics',
              'negative_pair_scope':'different primary mesh, absent from expanded catalog view; known catalog asset instances only',
              'precision_scope':'empirical verified-localization precision under one absent-template negative per eligible query; correlated fixed-scene views',
              'partitions_read':list(data['split']),'target_shards_read':data['targets'],
              'context_report_sha256':data.get('context_report_sha256'),
              'confidence_scope':'grouped out-of-query/template-fold predictions; fold normalization uses fitting queries only; inner held groups may remain in incidental backgrounds in the frozen cache',
              'scope':cfg['scope']}
    import subprocess
    manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest['training_arithmetic']='bfloat16 autocast, float32 loss'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('TRAINING_COMPLETE',str(output),flush=True)


if __name__=='__main__':main()
