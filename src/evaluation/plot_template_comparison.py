"""Plot saved whole-target results using examples fixed by target and distance order."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from common.patches import read_rgb
from common.view_geometry import redact_rgb


def read(path):return json.loads(Path(path).read_text())


def plot_verifier(cfg,out,comparison):
    selected=read(out/'selection.json')['selected_method'];records=read(out/'validation-records.json');names=list(comparison)
    fig,axes=plt.subplots(1,2,figsize=(16,5));x=np.arange(len(names))
    for offset,key,label in [(-.25,'accepted_precision','Accepted precision'),(0,'verified_positive_recall','Verified recall'),(.25,'negative_false_accept_rate','Wrong-template false accept')]:
        axes[0].bar(x+offset,[np.nan if comparison[n]['validation'][key] is None else comparison[n]['validation'][key] for n in names],width=.25,label=label)
    axes[0].set_xticks(x,names,rotation=45,fontsize=8);axes[0].set(ylim=(0,1.05),ylabel='Validation fraction');axes[0].legend(fontsize=8)
    for name,rows in records.items():
        order=np.argsort(-np.array([r['score'] for r in rows]),kind='stable');scores=np.array([r['score'] for r in rows])[order];correct=np.array([r['verified_correct'] for r in rows])[order]
        last=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1];hits=np.cumsum(correct)[last]
        axes[1].plot(hits/comparison[name]['validation']['visible_positive_pairs'],hits/(last+1),label=name)
    axes[1].axhline(cfg['calibration_precision'],color='black',linestyle='--');axes[1].axvline(cfg['minimum_verified_recall'],color='black',linestyle=':')
    axes[1].set(xlim=(0,1),ylim=(0,1.05),xlabel='Verified positive recall',ylabel='Accepted precision');axes[1].legend(fontsize=7)
    fig.suptitle('Validation development: thresholds and method selected on these targets');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    catalog=read(project_path(cfg['catalog']));targets=catalog['splits'][cfg['protocol']]['validation'];by_id={t['id']:t for t in catalog['targets']};chosen=[];metadata={}
    for target in targets:
        capture=read(project_path(cfg['capture_directory'])/target/'report.json');metadata.update({s['name']:s for s in capture['samples']})
        rows=[r for r in records[selected] if r['target_id']==target]
        for stratum in range(4):chosen.append(next((r for r in rows if r['positive'] and metadata[r['sample_name']]['range_stratum']==stratum),None))
        chosen.append(next((r for r in rows if not r['positive']),None))
    fig,axes=plt.subplots(len(targets),5,figsize=(20,3*len(targets)),squeeze=False);manifest=[]
    for ax,row in zip(axes.flat,chosen):
        ax.axis('off')
        if row is None:continue
        with h5py.File(project_path(cfg['capture_directory'])/row['target_id']/'samples.h5','r') as raw:
            g=raw[row['sample_name']];image=read_rgb(raw,row['sample_name']);image=redact_rgb(image,g['ignored_mask'][()]);mask=g['assembly_mask'][()]
        h,w=image.shape[:2];visible=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>0;image[visible]=(.7*image[visible]+.3*np.array([0,255,0])).astype(np.uint8)
        x0,y0,x1,y1=row['box_xyxy'];cv2.rectangle(image,(x0,y0),(x1-1,y1-1),(255,220,0),2);ax.imshow(image)
        px,py=np.array(row['point_xy'])*[w,h];ax.scatter([px],[py],marker='o' if row['accepted'] else 'x',c='red',s=50)
        requested=row['target_id'] if row['positive'] else row['negative_template_id'];inset=ax.inset_axes([.01,.7,.22,.28]);inset.imshow(cv2.imread(str(project_path(by_id[requested]['template']['file'])))[:,:,::-1]);inset.axis('off')
        ax.set_title(f"{row['sample_name']}  {metadata[row['sample_name']]['horizontal_distance_m']:.0f} m\n{requested} score={row['score']:.3f} "+('accept' if row['accepted'] else 'reject'),fontsize=8)
        manifest.append({'sample_name':row['sample_name'],'requested_template':requested,'positive':row['positive']})
    fig.suptitle('Validation development; fixed first frame per range; last column: absent template. Yellow: image-only crop; circle: accepted; cross: rejected.',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'prediction-examples.png',dpi=150);plt.close(fig)
    (out/'figure-selection.json').write_text(json.dumps({'selected_method':selected,'selection':'first positive per target/range and first absent template; independent of scores','examples':manifest},indent=2));print('VERIFIER_FIGURES_READY',len(manifest))
    accepted=[row for row in records[selected] if row['accepted']]
    if accepted:
        fig,axes=plt.subplots((len(accepted)+4)//5,5,figsize=(20,2.6*((len(accepted)+4)//5)),squeeze=False)
        for ax in axes.flat:ax.axis('off')
        for ax,row in zip(axes.flat,accepted):
            with h5py.File(project_path(cfg['capture_directory'])/row['target_id']/'samples.h5','r') as raw:
                g=raw[row['sample_name']];image=redact_rgb(read_rgb(raw,row['sample_name']),g['ignored_mask'][()])
            h,w=image.shape[:2];ax.imshow(image);px,py=np.array(row['point_xy'])*[w,h];color='lime' if row['verified_correct'] else 'red';ax.scatter([px],[py],c=color,s=55)
            requested=row['target_id'] if row['positive'] else row['negative_template_id'];inset=ax.inset_axes([.01,.7,.22,.28]);inset.imshow(cv2.imread(str(project_path(by_id[requested]['template']['file'])))[:,:,::-1]);inset.axis('off')
            ax.set_title(row['sample_name']+'  '+requested+f"\nscore={row['score']:.3f} "+('correct' if row['verified_correct'] else 'FALSE ACCEPT'),color=color if color=='red' else 'black',fontsize=8)
        fig.suptitle(f'All {len(accepted)} accepted validation pairs; green: verified correct; red: false accept. Development data.',fontsize=12);fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/'accepted-overview.png',dpi=140);plt.close(fig)


def plot(config_path):
    cfg=read(config_path);out=project_path(cfg['output_directory']);comparison=read(out/'comparison.json')
    if (out/'validation-records.json').exists():plot_verifier(cfg,out,comparison);return
    partition=cfg.get('evaluation_partition','test');names=list(comparison);fig,axes=plt.subplots(1,3,figsize=(18,5))
    for name in dict.fromkeys(n.split('/')[0] for n in names):
        if not (out/name/'history.json').exists():continue
        history=read(out/name/'history.json')
        axes[0].plot([h['epoch'] for h in history],[h['loss'] for h in history],label=name)
        axes[1].plot([h['epoch'] for h in history],[h['validation_macro_hit'] for h in history],label=name)
    axes[0].set(xlabel='Epoch',ylabel='Training loss');axes[1].set(xlabel='Epoch',ylabel='Validation macro target hit',ylim=(0,1))
    x=np.arange(len(names))
    for offset,key,label in [(-.25,'raw_positive_hit','Positive hit'),(0,'accepted_precision','Accepted precision'),(.25,'pair_coverage','Pair coverage')]:
        values=[comparison[n][partition][key] for n in names]
        axes[2].bar(x+offset,[np.nan if v is None else v for v in values],width=.25,label=label)
        if key=='accepted_precision':
            for j,value in enumerate(values):
                if value is None:axes[2].text(j,.03,'P: n/a\n0 accepted',ha='center',fontsize=7)
    axes[2].set_xticks(x,[n.replace('/','\n') for n in names],rotation=45,fontsize=7);axes[2].set(ylim=(0,1.05),ylabel=partition.title()+' fraction')
    for ax in axes:ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation development: model selection and thresholds use these targets' if partition=='validation' else 'Frozen test evaluation')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)

    catalog=read(project_path(cfg['catalog']));targets=catalog['splits'][cfg['protocol']][partition]
    by_id={t['id']:t for t in catalog['targets']}
    selected=read(out/'selection.json')['selected_method'] if (out/'selection.json').exists() else 'conditioned'
    methods={n:read(out/n/(partition+'-records.json')) for n in ['fingerprint_4rot',selected]}
    lookup={n:{(r['query'],r['template'],r['positive']):r for r in rows} for n,rows in methods.items()}
    chosen=[]
    for target in targets:
        rows=[r for r in methods[selected] if r['target_id']==target]
        for low,high in [(12,30),(30,60),(60,100),(100,151)]:
            chosen.append(next((r for r in rows if r['positive'] and low<=r['horizontal_distance_m']<high),None))
        chosen.append(next((r for r in rows if not r['positive']),None))
    fig,axes=plt.subplots(len(targets),5,figsize=(20,3*len(targets)),squeeze=False)
    manifest=[]
    for ax,row in zip(axes.flat,chosen):
        ax.axis('off')
        if row is None:continue
        raw=project_path(cfg['capture_directory'])/row['target_id']/'samples.h5'
        with h5py.File(raw,'r') as store:
            image=read_rgb(store,row['sample_name']);mask=store[row['sample_name']]['assembly_mask'][()]
            if 'ignored_mask' in store[row['sample_name']]:image=redact_rgb(image,store[row['sample_name']]['ignored_mask'][()])
        h,w=image.shape[:2];visible=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>0
        image[visible]=(.7*image[visible]+.3*np.array([0,255,0])).astype(np.uint8)
        ax.imshow(image);key=(row['query'],row['template'],row['positive']);scores=[]
        for name,color in [('fingerprint_4rot','cyan'),(selected,'red')]:
            record=lookup[name][key];px,py=np.array(record['point_xy'])*[w,h]
            ax.scatter([px],[py],marker='o' if record['accepted'] else 'x',s=55,c=color,linewidths=1.7)
            scores.append(('F' if name=='fingerprint_4rot' else 'C')+f"={record['score']:.2f}")
        requested=row['target_id'] if row['positive'] else row['negative_template_id']
        inset=ax.inset_axes([.01,.68,.22,.3]);inset.imshow(cv2.imread(str(project_path(by_id[requested]['template']['file'])))[:,:,::-1]);inset.axis('off')
        ax.set_title(f"{row['sample_name']}  {row['horizontal_distance_m']:.0f} m\nrequest {requested}  "+' '.join(scores),fontsize=8)
        manifest.append({'sample_name':row['sample_name'],'target_id':row['target_id'],'requested_template_id':requested,'positive':row['positive']})
    legend=[Line2D([],[],marker='o',color='cyan',linestyle='',label='Four-rotation fingerprint'),
            Line2D([],[],marker='o',color='red',linestyle='',label=selected),
            Line2D([],[],marker='o',color='black',linestyle='',label='Accepted'),
            Line2D([],[],marker='x',color='black',linestyle='',label='Rejected')]
    fig.legend(handles=legend,loc='upper center',ncol=4)
    fig.suptitle(partition.title()+': first saved frame per target/range; last column: first absent-template pair. Green: visible query target.',y=.97,fontsize=10)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'prediction-examples.png',dpi=150);plt.close(fig)
    (out/'figure-selection.json').write_text(json.dumps({'partition':partition,'selected_method':selected,'selection':'first saved positive per target/range and first saved negative per target; independent of scores','examples':manifest},indent=2))
    print('FIGURES_READY',len(manifest),'examples')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/training/airport_template_matcher.json')
    args=parser.parse_args();plot(project_path(args.config))
