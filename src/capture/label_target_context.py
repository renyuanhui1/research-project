"""Label additional training targets in saved RGB-D frames without connecting to UE."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common.paths import project_path
from capture.part_masks import load_mesh
from capture.build_target_catalog import camera_rotation,sha256
from capture.collect_target_pilot import depth_filter


class ContextRenderer:
    def __init__(self,targets):
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=2);mapping={}
        for target in targets:
            for actor in target['actors']:
                vertices,faces=load_mesh(project_path(actor['mesh']))
                gid=self.scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
                mapping[gid]=target['index']
        self.lookup=np.zeros(max(mapping)+1,np.uint16)
        for gid,index in mapping.items():self.lookup[gid]=index

    def render(self,depth,message,fov,tolerance,valid,ignored):
        h,w=depth.shape;focal=w/(2*math.tan(math.radians(fov)/2))
        u,v=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
        directions=np.stack([np.ones((h,w)),(u-w/2)/focal,(v-h/2)/focal],-1).reshape(-1,3)
        directions=directions@camera_rotation(message).T
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        position=np.array([message['pos_'+k] for k in ('x','y','z')])
        rays=np.c_[np.broadcast_to(position,directions.shape),directions].astype(np.float32)
        hit=self.scene.cast_rays(o3d.core.Tensor(rays),nthreads=2)
        distance=hit['t_hit'].numpy().reshape(h,w)
        masks=depth_filter(distance,np.zeros((h,w),bool),depth,tolerance)
        known=masks['valid_mask']&valid&~ignored
        visible=masks['assembly_mask']&known;labels=np.zeros((h,w),np.uint16)
        labels[visible]=self.lookup[hit['geometry_ids'].numpy().reshape(h,w)[visible]]
        return labels,known


def unambiguous_context(labels,current,targets):
    present=set(map(int,np.unique(labels)))-{0};groups={}
    for index in present:
        target=targets[index];groups.setdefault(target['appearance_group'],[]).append(index)
    result=[]
    for group,indices in groups.items():
        if group!=current['appearance_group'] and len(indices)==1:result.extend(indices)
    return sorted(result)


def annotate(config_path):
    cfg=json.loads(Path(config_path).read_text());catalog_path=project_path(cfg['catalog']);catalog=json.loads(catalog_path.read_text())
    allowed=set(catalog['splits'][cfg['protocol']]['train']);targets=[t for t in catalog['targets'] if t['id'] in allowed]
    by_index={t['index']:t for t in targets};renderer=ContextRenderer(targets)
    source=project_path(cfg['capture_directory']);out=project_path(cfg['context_labels']);out.mkdir(parents=True,exist_ok=True)
    if (out/'labels.h5').exists():raise FileExistsError('Context labels already exist')
    report={'status':'started','catalog_sha256':sha256(catalog_path),'partition':'train','samples':[],'raw_sha256':{},
            'source_sha256':{p:sha256(ROOT/p) for p in ['src/capture/label_target_context.py','src/capture/collect_target_pilot.py','src/capture/build_target_catalog.py','src/capture/part_masks.py']}}
    stats=Counter()
    with h5py.File(out/'labels.h5','w') as saved:
        for target in targets:
            raw=source/target['id'];capture=json.loads((raw/'report.json').read_text());digest=sha256(raw/'samples.h5')
            if digest!=capture['samples_h5_sha256']:raise ValueError('Raw source changed')
            report['raw_sha256'][target['id']]=digest
            with h5py.File(raw/'samples.h5','r') as data:
                for sample in capture['samples']:
                    group=data[sample['name']];ids,known=renderer.render(group['depth_m'][()].astype(np.float32),sample['depth_metadata'],capture['protocol']['horizontal_fov_deg'],capture['protocol']['depth_tolerance_m'],group['valid_mask'][()].astype(bool),group['ignored_mask'][()].astype(bool))
                    extra=unambiguous_context(ids,target,by_index);stats['frames_checked']+=1
                    stats['current_mask_disagreement_pixels']+=int(np.count_nonzero((ids==target['index'])!=group['assembly_mask'][()].astype(bool)))
                    if not extra:continue
                    store=saved.create_group(sample['name']);store.create_dataset('ids',data=ids,compression='gzip',compression_opts=1);store.create_dataset('valid',data=known.astype(np.uint8),compression='gzip',compression_opts=1)
                    entry={'sample_name':sample['name'],'query_target_id':target['id'],'label_targets':[{'target_id':by_index[i]['id'],'index':i,'pixels':int(np.count_nonzero(ids==i))} for i in extra]}
                    report['samples'].append(entry);stats['context_frames']+=1;stats['additional_positive_pairs']+=len(extra)
            print('CONTEXT_LABELS',target['id'],dict(stats),flush=True)
    report.update(status='completed',stats=dict(stats),labels_sha256=sha256(out/'labels.h5'),mesh_sha256={a['mesh']:sha256(project_path(a['mesh'])) for t in targets for a in t['actors']})
    (out/'report.json').write_text(json.dumps(report,indent=2));print('CONTEXT_COMPLETE',dict(stats),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/training/airport_template_matcher.json')
    options=parser.parse_args();annotate(project_path(options.config))
