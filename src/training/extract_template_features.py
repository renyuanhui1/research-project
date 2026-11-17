"""Cache identical frozen-DINO inputs for learned and fingerprint whole-target comparisons."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from common.patches import read_rgb
from common.view_geometry import redact_rgb
from perception.extract_dino_features import load_model,to_input_tensor,IMAGENET_MEAN,IMAGENET_STD


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def template_variants(image,cfg):
    variants=[];metadata=[]
    for padding in cfg['template_padding']:
        for brightness in cfg['template_brightness']:
            for angle in cfg['template_rotations_deg']:
                h,w=image.shape[:2];side=int(np.ceil(max(h,w)*padding))
                canvas=np.broadcast_to(np.array([124,116,104],np.uint8),(side,side,3)).copy()
                x=(side-w)//2;y=(side-h)//2;canvas[y:y+h,x:x+w]=np.clip(image.astype(float)*brightness,0,255).astype(np.uint8)
                canvas=np.rot90(canvas,int(angle)//90).copy()
                variants.append(cv2.resize(canvas,(cfg['image_size'],cfg['image_size']),interpolation=cv2.INTER_LINEAR))
                metadata.append({'padding':padding,'brightness':brightness,'rotation_deg':angle})
    return variants,metadata


def extract(config_path,target_id=None):
    cfg=read(config_path);catalog_path=project_path(cfg['catalog']);catalog=read(catalog_path)
    output=project_path(cfg['feature_directory']);output.mkdir(parents=True,exist_ok=True)
    source=project_path(cfg['capture_directory']);device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=load_model('dinov2_vits14',device,weights=project_path(cfg['dino_weights']),repo_dir=project_path(cfg['dino_repository']))
    mean=torch.tensor(IMAGENET_MEAN,device=device).view(1,3,1,1);std=torch.tensor(IMAGENET_STD,device=device).view(1,3,1,1)
    signature={'catalog_sha256':digest(catalog_path),'encoder_sha256':digest(project_path(cfg['dino_weights'])),
               'encoder_precision':'float16' if device=='cuda' else 'float32',
               'feature_config':{k:cfg[k] for k in ['image_size','feature_batch','output_grid','template_rotations_deg','template_brightness','template_padding']}}
    def encode(images):
        count=len(images);images=list(images)+[images[-1]]*(cfg['feature_batch']-count)
        batch=np.stack([cv2.resize(x,(cfg['image_size'],cfg['image_size']),interpolation=cv2.INTER_LINEAR) for x in images])
        with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            x=to_input_tensor(batch,cfg['image_size'],mean,std,device)
            return model.forward_features(x)['x_norm_patchtokens'][:count].float().cpu().numpy().astype(np.float16)
    template_path=output/'templates.h5'
    if template_path.exists():
        with h5py.File(template_path,'r') as store:
            if json.loads(store.attrs['signature'])!=signature:raise ValueError('Template feature inputs changed')
            if set(store)!={t['id'] for t in catalog['targets']}:raise ValueError('Incomplete template feature bank')
    else:
        with h5py.File(template_path,'w') as store:
            store.attrs['signature']=json.dumps(signature)
            for target in catalog['targets']:
                path=project_path(target['template']['file'])
                if digest(path)!=target['template']['sha256']:raise ValueError('Reference template changed')
                image=cv2.imread(str(path))[:,:,::-1].copy();images,metadata=template_variants(image,cfg)
                z=np.concatenate([encode(images[i:i+cfg['feature_batch']]) for i in range(0,len(images),cfg['feature_batch'])])
                group=store.create_group(target['id']);group.create_dataset('z',data=z,compression='gzip',compression_opts=1)
                group.attrs['variants_json']=json.dumps(metadata)
                print('TEMPLATE_FEATURE',target['id'],z.shape,flush=True)
    selected=[t for t in catalog['targets'] if target_id is None or t['id']==target_id]
    for target in selected:
        report_path=source/target['id']/'report.json'
        if not report_path.exists():continue
        report=read(report_path)
        if report['status']!='completed':continue
        raw=report_path.parent/'samples.h5';destination=output/(target['id']+'.h5')
        raw_hash=digest(raw)
        if raw_hash!=report['samples_h5_sha256']:raise ValueError('Raw capture hash changed')
        if destination.exists():
            with h5py.File(destination,'r') as saved:
                if saved.attrs['raw_sha256']!=raw_hash or json.loads(saved.attrs['signature'])!=signature:
                    raise ValueError('Feature inputs changed')
                if len(saved['z'])!=len(report['samples']):raise ValueError('Incomplete query feature cache')
            continue
        n=len(report['samples']);grid=cfg['image_size']//14;shape=(n,grid*grid,384)
        with h5py.File(raw,'r') as data,h5py.File(destination,'w') as saved:
            saved.attrs['raw_sha256']=raw_hash;saved.attrs['signature']=json.dumps(signature)
            z=saved.create_dataset('z',shape,dtype=np.float16,chunks=(1,grid*grid,384),compression='gzip',compression_opts=1)
            labels=saved.create_dataset('mask64',(n,cfg['output_grid'],cfg['output_grid']),dtype=np.float16,compression='gzip',compression_opts=1)
            valid=saved.create_dataset('valid64',labels.shape,dtype=np.float16,compression='gzip',compression_opts=1)
            h,w=report['samples'][0]['depth_shape_hw']
            native=saved.create_dataset('native_mask',(n,h,w),dtype=np.uint8,chunks=(1,h,w),compression='gzip',compression_opts=1)
            metadata=saved.create_dataset('metadata',(n,),dtype=h5py.string_dtype('utf-8'))
            for start in range(0,n,cfg['feature_batch']):
                images=[]
                for j,sample in enumerate(report['samples'][start:start+cfg['feature_batch']],start):
                    group=data[sample['name']];image=read_rgb(data,sample['name'])
                    if 'ignored_mask' in group:image=redact_rgb(image,group['ignored_mask'][()])
                    images.append(image);mask=group['assembly_mask'][()];weight=group['valid_mask'][()]
                    labels[j]=cv2.resize(mask.astype(np.float32),(cfg['output_grid'],cfg['output_grid']),interpolation=cv2.INTER_AREA)
                    valid[j]=cv2.resize(weight.astype(np.float32),(cfg['output_grid'],cfg['output_grid']),interpolation=cv2.INTER_AREA)
                    native[j]=mask;metadata[j]=json.dumps(dict(sample,horizontal_fov_deg=report['protocol']['horizontal_fov_deg']))
                z[start:start+len(images)]=encode(images)
                print('QUERY_FEATURE',target['id'],min(n,start+len(images)),'/',n,flush=True)
    entries=[]
    for target in catalog['targets']:
        p=output/(target['id']+'.h5')
        if p.exists():
            with h5py.File(p,'r') as f:entries.append({'target_id':target['id'],'file':p.name,'samples':len(f['z']),'sha256':digest(p)})
    index={'status':'completed' if len(entries)==len(catalog['targets']) else 'partial','signature':signature,
           'configuration_sha256':digest(config_path),
           'targets':entries,'template_file':'templates.h5','template_sha256':digest(template_path),
           'source_sha256':digest(Path(__file__)),'redaction':'ignored regions replaced before DINO encoding; test frames unchanged'}
    (output/'index.json').write_text(json.dumps(index,indent=2))
    print('FEATURES_READY',len(entries),'targets',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/airport_template_matcher.json');parser.add_argument('--target')
    options=parser.parse_args();extract(project_path(options.config),options.target)


if __name__=='__main__':main()
