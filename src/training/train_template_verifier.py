"""Fit template verification on out-of-fold coarse proposals and saved RGB crops."""
import argparse
import json
from pathlib import Path
import sys

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
from perception.template_matcher import TemplateMatcher,ConfidenceHead
from perception.template_verification import proposal_box,square_rgb,FEATURE_NAMES,rgb_descriptors,rgb_correspondence_features,patch_verification_features
from perception.extract_dino_features import load_model,to_input_tensor,IMAGENET_MEAN,IMAGENET_STD
from training.extract_template_features import read,digest
from training.train_template_matcher import load_features,fitting_statistics,pair_plan,confidence_fold_plan,train_localizer,point_from_map,correct_point,calibrate,metrics,selection_key


@torch.no_grad()
def proposals(model,data,indices,mean,std,cfg,boxes,points,stats):
    model.eval()
    for start in range(0,len(indices),cfg['batch']):
        batch=indices[start:start+cfg['batch']];q=(data['z'][batch].float()-mean)/std
        count=len(q)
        if count<cfg['batch']:q=torch.cat([q,q[-1:].expand(cfg['batch']-count,-1,-1)])
        with torch.autocast(q.device.type,dtype=torch.bfloat16,enabled=q.is_cuda):logits,features=model(q,torch.zeros_like(q))
        logits=logits[:count];features=features[:count]
        points[batch]=point_from_map(logits);stats[batch]=features.cpu().numpy()
        for i,map_ in zip(batch,logits.cpu().numpy()):
            m=data['records'][i]['camera_metadata']
            boxes[i]=proposal_box(map_,(m['height'],m['width']),cfg['crop_margin'],cfg['crop_minimum_fraction'],cfg['crop_logit_delta'])


def crop_tokens(data,boxes,cfg):
    device=data['z'].device;encoder=load_model('dinov2_vits14',device,weights=project_path(cfg['dino_weights']),repo_dir=project_path(cfg['dino_repository']))
    mean=torch.tensor(IMAGENET_MEAN,device=device).view(1,3,1,1);std=torch.tensor(IMAGENET_STD,device=device).view(1,3,1,1)
    tokens=torch.empty_like(data['z']);valid=torch.zeros(len(tokens),cfg['output_grid'],cfg['output_grid'],device=device)
    for target in data['targets']:
        indices=[i for i,r in enumerate(data['records']) if r['target_id']==target]
        with h5py.File(project_path(cfg['capture_directory'])/target/'samples.h5','r') as raw:
            for start in range(0,len(indices),cfg['feature_batch']):
                selected=indices[start:start+cfg['feature_batch']];images=[]
                for i in selected:
                    name=data['records'][i]['name'];group=raw[name];image=read_rgb(raw,name)
                    if 'ignored_mask' in group:image=redact_rgb(image,group['ignored_mask'][()])
                    known=cv2.resize(group['valid_mask'][()].astype(np.float32),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
                    x0,y0,x1,y1=boxes[i];images.append(square_rgb(image[y0:y1,x0:x1],cfg['image_size']))
                    mask=known[y0:y1,x0:x1];h,w=mask.shape;side=max(h,w);canvas=np.zeros((side,side),np.float32);oy=(side-h)//2;ox=(side-w)//2;canvas[oy:oy+h,ox:ox+w]=mask
                    valid[i]=torch.from_numpy(cv2.resize(canvas,(cfg['output_grid'],cfg['output_grid']),interpolation=cv2.INTER_AREA)).to(device)
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    images += [images[-1]]*(cfg['feature_batch']-len(images))
                    batch=to_input_tensor(np.stack(images),cfg['image_size'],mean,std,device)
                    tokens[selected]=encoder.forward_features(batch)['x_norm_patchtokens'][:len(selected)].to(tokens.dtype)
        print('CROP_ENCODED',target,len(indices),flush=True)
    return tokens,valid


def verify_rows(data,rows,query,template,proposal_stats,cfg):
    features=[];scores=[]
    for j,row in enumerate(rows):
        q=row['query'];t=row['template']
        feature,score=patch_verification_features(query[q],template[t,:4],proposal_stats[q],cfg['reprojection_threshold'])
        features.append(feature);scores.append(score)
        if (j+1)%200==0:print('VERIFIED_PAIRS',j+1,'/',len(rows),flush=True)
    return np.array(features,np.float32),np.array(scores)


def rgb_verification(data,boxes,pairs,cfg):
    references=[];queries={}
    for target in data['targets']:
        references.append(rgb_descriptors(cv2.imread(str(project_path(data['groups'][target]['template']['file'])))[:,:,::-1].copy()))
        with h5py.File(project_path(cfg['capture_directory'])/target/'samples.h5','r') as raw:
            for i,r in enumerate(data['records']):
                if r['target_id']!=target:continue
                g=raw[r['name']];image=read_rgb(raw,r['name'])
                if 'ignored_mask' in g:image=redact_rgb(image,g['ignored_mask'][()])
                x0,y0,x1,y1=boxes[i];queries[i]=rgb_descriptors(square_rgb(image[y0:y1,x0:x1],cfg['image_size']))
        print('RGB_DESCRIBED',target,flush=True)
    values={};scores={}
    for partition,rows in pairs.items():
        entries=[rgb_correspondence_features(queries[r['query']],references[r['template']],cfg['rgb_reprojection_threshold'],cfg['rgb_ratio']) for r in rows]
        values[partition]=np.array([e[0] for e in entries],np.float32);scores[partition]=np.array([e[1] for e in entries])
    return values,scores


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/training/airport_template_verifier.json');args=parser.parse_args();path=project_path(args.config);cfg=read(path)
    out=project_path(cfg['output_directory']);out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.json').exists():raise FileExistsError('Completed verification results exist')
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(cfg['seed']);cv2.setRNGSeed(cfg['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu';data=load_features(cfg,device,('train','validation'));pairs=pair_plan(data,cfg)
    checkpoint=torch.load(project_path(cfg['coarse_checkpoint']),map_location=device,weights_only=True)
    values={};labels={};raw_scores={}
    if cfg.get('reuse_verification'):
        previous=project_path(cfg['reuse_verification']);prior=read(previous/'manifest.json')
        if data['feature_index_sha256']!=prior['feature_index_sha256'] or pairs!=read(previous/'pair-plan.json'):raise ValueError('Reused proposal data disagree with this protocol')
        saved=np.load(previous/'proposals.npz');boxes=saved['boxes'];points=saved['points'];proposal_stats=saved['stats']
        saved=np.load(previous/'verification-features.npz')
        for partition in ['train','validation']:
            values[partition]=saved[partition];labels[partition]=saved[partition+'_correct']
        raw_scores['validation']=np.array([r['score'] for r in read(previous/'validation-records.json')['geometry']])
        frozen=torch.load(previous/'verifier.pt',map_location='cpu',weights_only=True);mean=frozen['crop_mean'];std=frozen['crop_std']
    else:
        n=len(data['records']);boxes=np.zeros((n,4),int);points=np.zeros((n,2));proposal_stats=np.zeros((n,10),np.float32)
        coarse_cfg=dict(checkpoint['configuration'],epochs=checkpoint['selection']['best_epoch'],context_labels=None)
        for fold,plan in enumerate(confidence_fold_plan(data,cfg['confidence_folds'],cfg['seed'])):
            folder=out/f'proposal-fold-{fold}';folder.mkdir(exist_ok=True)
            subset=dict(data,split={'train':plan['fit_queries']},training_template_ids=plan['fit_template_ids']);mean,std=fitting_statistics(subset)
            model,_=train_localizer(False,subset,pair_plan(subset,coarse_cfg),mean,std,coarse_cfg,folder,select=False)
            proposals(model,data,plan['held_queries'],mean,std,cfg,boxes,points,proposal_stats)
            torch.save({'model':model.state_dict(),'mean':mean.cpu(),'std':std.cpu(),'fold':plan,'configuration':coarse_cfg},folder/'model.pt')
            (folder/'fold.json').write_text(json.dumps(plan,indent=2))
        model=TemplateMatcher(conditioned=False,hidden_dim=coarse_cfg['hidden_dim'],output_grid=cfg['output_grid']).to(device);model.load_state_dict(checkpoint['model'])
        proposals(model,data,data['split']['validation'],checkpoint['mean'].to(device),checkpoint['std'].to(device),cfg,boxes,points,proposal_stats)
        crops,valid=crop_tokens(data,boxes,cfg);mean,std=fitting_statistics(dict(data,z=crops,valid=valid))
        query=F.normalize((crops.float()-mean)/std,dim=-1);template=F.normalize((data['template'].float()-mean)/std,dim=-1)
        for partition in ['train','validation']:
            values[partition],raw_scores[partition]=verify_rows(data,pairs[partition],query,template,proposal_stats,cfg)
            labels[partition]=np.array([correct_point(data,row,points[row['query']]) for row in pairs[partition]])
    np.savez_compressed(out/'proposals.npz',boxes=boxes,points=points,stats=proposal_stats)
    names=FEATURE_NAMES+['coherent_score','rotation_score_margin']+['coarse_'+str(i) for i in range(8)]
    rgb_scores=None
    if cfg.get('rgb_ratio'):
        rgb,rgb_scores=rgb_verification(data,boxes,pairs,cfg)
        for partition in values:values[partition]=np.c_[values[partition],rgb[partition]]
        names += ['rgb_'+name for name in FEATURE_NAMES]
    np.savez_compressed(out/'verification-features.npz',train=values['train'],validation=values['validation'],train_correct=labels['train'],validation_correct=labels['validation'])
    center=values['train'].mean(0);scale=values['train'].std(0).clip(1e-4);x=torch.from_numpy((values['train']-center)/scale).to(device);y=torch.tensor(labels['train'],device=device,dtype=torch.float32)
    scored={'geometry':raw_scores['validation']};heads={}
    if rgb_scores is not None:scored['rgb_geometry']=rgb_scores['validation']
    for seed in cfg['classifier_seeds']:
        for kind in ['linear','mlp']:
            name=f'{kind}-s{seed}';torch.manual_seed(seed);head=(torch.nn.Linear(x.shape[1],1) if kind=='linear' else ConfidenceHead(x.shape[1])).to(device)
            optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['confidence_learning_rate'],weight_decay=cfg['confidence_weight_decay'])
            for _ in range(cfg['confidence_epochs']):
                logits=head(x).reshape(-1);loss=F.binary_cross_entropy_with_logits(logits,y);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():scored[name]=head(torch.from_numpy((values['validation']-center)/scale).to(device)).reshape(-1).double().sigmoid().cpu().numpy()
            heads[name]=head.cpu().state_dict()
    comparison={};records={}
    for name,scores in scored.items():
        threshold,calibration=calibrate(scores,labels['validation'],cfg['calibration_precision']);rows=pairs['validation'];xy=np.array([points[row['query']] for row in rows]);m=metrics(data,rows,xy,scores,threshold)
        m['by_target']={}
        for target in sorted({r['target_id'] for r in rows}):
            selection=[i for i,r in enumerate(rows) if r['target_id']==target]
            m['by_target'][target]=metrics(data,[rows[i] for i in selection],xy[selection],scores[selection],threshold)
        comparison[name]={'threshold':threshold,'calibration':calibration,'validation':m}
        records[name]=[dict(row,sample_name=data['records'][row['query']]['name'],point_xy=xy[i].tolist(),box_xyxy=boxes[row['query']].tolist(),score=float(scores[i]),accepted=bool(scores[i]>=threshold),verified_correct=bool(labels['validation'][i])) for i,row in enumerate(rows)]
        print('VERIFIER_RESULT',name,json.dumps({k:v for k,v in m.items() if k!='by_target'}),flush=True)
    selected=max(comparison,key=lambda name:selection_key(comparison[name]['validation']));m=comparison[selected]['validation'];decision={'selected_method':selected,'development_gate_passed':bool((m['accepted_precision'] or 0)>=cfg['calibration_precision'] and (m['verified_positive_recall'] or 0)>=cfg['minimum_verified_recall']),'validation':m,'scope':cfg['scope']}
    torch.save({'coarse':checkpoint,'confidence':heads,'feature_center':torch.from_numpy(center),'feature_scale':torch.from_numpy(scale),'crop_mean':mean.cpu(),'crop_std':std.cpu(),'configuration':cfg,'comparison':comparison,'selected_method':selected},out/'verifier.pt')
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2));(out/'selection.json').write_text(json.dumps(decision,indent=2));(out/'validation-records.json').write_text(json.dumps(records,indent=2));(out/'pair-plan.json').write_text(json.dumps(pairs,indent=2))
    import subprocess
    sources=['src/training/train_template_verifier.py','src/perception/template_verification.py','src/training/train_template_matcher.py','src/perception/template_matcher.py']
    manifest={'configuration':cfg,'configuration_sha256':digest(path),'sources_sha256':{p:digest(ROOT/p) for p in sources},'coarse_checkpoint_sha256':digest(project_path(cfg['coarse_checkpoint'])),'checkpoint_sha256':digest(out/'verifier.pt'),'feature_index_sha256':data['feature_index_sha256'],'partitions_read':list(data['split']),'target_shards_read':data['targets'],'feature_names':names,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':cfg['scope']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('VERIFICATION_COMPLETE',json.dumps(decision),flush=True)


if __name__=='__main__':main()
