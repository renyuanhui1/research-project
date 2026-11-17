"""Image-only proposal crops and spatial verification of frozen patch correspondences."""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


FEATURE_NAMES=['query_similarity','template_similarity','similarity_margin','mutual_fraction','mutual_similarity','mutual_low_similarity',
               'query_coverage','template_coverage','inlier_fraction','inlier_fraction_of_grid','inlier_similarity',
               'inlier_query_coverage','inlier_template_coverage','inlier_residual','affine_area','affine_anisotropy','distinctiveness']


def proposal_box(logits,image_shape,margin,minimum_fraction,logit_delta):
    h,w=image_shape;grid=logits.shape[0];y,x=np.unravel_index(np.argmax(logits),logits.shape)
    _,labels=cv2.connectedComponents((logits>=logits.max()-logit_delta).astype(np.uint8),connectivity=8)
    ys,xs=np.where(labels==labels[y,x]);cx=(xs.min()+xs.max()+1)/2/grid*w;cy=(ys.min()+ys.max()+1)/2/grid*h
    side=max((xs.max()-xs.min()+1)/grid*w,(ys.max()-ys.min()+1)/grid*h,max(h,w)*minimum_fraction)*margin
    low=np.maximum(np.floor([cx-side/2,cy-side/2]),[0,0]).astype(int);high=np.minimum(np.ceil([cx+side/2,cy+side/2]),[w,h]).astype(int)
    return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]


def square_rgb(image,size):
    h,w=image.shape[:2];side=max(h,w);canvas=np.broadcast_to(np.array([124,116,104],np.uint8),(side,side,3)).copy()
    x=(side-w)//2;y=(side-h)//2;canvas[y:y+h,x:x+w]=image
    return cv2.resize(canvas,(size,size),interpolation=cv2.INTER_LINEAR)


def hull_area(points):
    return float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) if len(points)>=3 else 0.


def correspondence_features(similarity,reprojection_threshold,query_positions=None,template_positions=None,minimum_distinctiveness=0.):
    similarity=np.clip(similarity,-1.,1.)
    nq,nt=similarity.shape;gq=math.isqrt(nq);gt=math.isqrt(nt)
    feature=np.zeros(len(FEATURE_NAMES),float);feature[13]=1.
    if nq==0 or nt<2:return feature,0.
    query_to_template=similarity.argmax(1);template_to_query=similarity.argmax(0)
    q=np.flatnonzero(template_to_query[query_to_template]==np.arange(nq));t=query_to_template[q]
    top=np.partition(similarity,-2,axis=1)[:,-2:];best=top.max(1);second=top.min(1)
    distinct=((best-second).clip(0)/(1-second).clip(1e-8)).clip(0,1)
    q=q[distinct[q]>=minimum_distinctiveness];t=query_to_template[q]
    feature[:4]=[similarity.max(1).mean(),similarity.max(0).mean(),(best-second).mean(),len(q)/min(nq,nt)]
    feature[13]=1.;feature[16]=distinct.mean()
    if not len(q):return feature,0.
    qp=(np.c_[(q%gq+.5)/gq,(q//gq+.5)/gq] if query_positions is None else query_positions[q]).astype(np.float32)
    tp=(np.c_[(t%gt+.5)/gt,(t//gt+.5)/gt] if template_positions is None else template_positions[t]).astype(np.float32)
    scores=similarity[q,t];feature[4:8]=[scores.mean(),np.quantile(scores,.1),hull_area(qp),hull_area(tp)]
    if len(q)<3:return feature,0.
    transform,inliers=cv2.estimateAffine2D(tp,qp,method=cv2.RANSAC,ransacReprojThreshold=reprojection_threshold,maxIters=2000,confidence=.99,refineIters=10)
    if transform is None or inliers is None or not np.isfinite(transform).all():return feature,0.
    keep=inliers[:,0].astype(bool)
    if not keep.any():return feature,0.
    residual=np.linalg.norm(np.c_[tp,np.ones(len(tp))]@transform.T-qp,axis=1);singular=np.linalg.svd(transform[:,:2],compute_uv=False)
    feature[8:16]=[keep.mean(),keep.sum()/min(nq,nt),scores[keep].mean(),hull_area(qp[keep]),hull_area(tp[keep]),
                   np.median(residual[keep]),np.log1p(abs(np.linalg.det(transform[:,:2]))),np.log1p(singular.max()/max(singular.min(),1e-8))]
    coherent=feature[9]*max(0.,feature[10])*math.sqrt(max(0.,feature[11]*feature[12]))
    return feature,float(coherent)


def rgb_descriptors(image):
    points,descriptors=cv2.SIFT_create().detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
    if descriptors is None:return np.empty((0,2),np.float32),np.empty((0,128),np.float32)
    xy=np.array([p.pt for p in points],np.float32)/[image.shape[1],image.shape[0]]
    descriptors=descriptors/(np.linalg.norm(descriptors,axis=1,keepdims=True)+1e-8)
    return xy,descriptors


def rgb_correspondence_features(query,template,reprojection_threshold,ratio):
    qp,qd=query;tp,td=template
    return correspondence_features(qd@td.T,reprojection_threshold,qp,tp,1-ratio**2)


def patch_verification_features(query,template,proposal_stats,reprojection_threshold):
    similarity=(query[None]@template[:4].transpose(-1,-2)).detach().cpu().numpy()
    rotations=[correspondence_features(s,reprojection_threshold) for s in similarity]
    order=np.argsort([r[1] for r in rotations]);best=rotations[int(order[-1])]
    feature=np.r_[best[0],best[1],best[1]-rotations[int(order[-2])][1],proposal_stats[:8]]
    return feature.astype(np.float32),best[1]


class TemplateVerifier:
    """Frozen RGB inference; set one reference, then verify each search image."""
    def __init__(self,checkpoint,device='cuda'):
        from common.paths import project_path
        from perception.template_matcher import TemplateMatcher,ConfidenceHead
        from perception.extract_dino_features import load_model,IMAGENET_MEAN,IMAGENET_STD
        self.device=torch.device(device);self.bundle=torch.load(project_path(checkpoint),map_location='cpu',weights_only=True);self.cfg=self.bundle['configuration']
        coarse=self.bundle['coarse'];params=coarse['configuration'];self.coarse=TemplateMatcher(hidden_dim=params['hidden_dim'],conditioned=False,output_grid=params['output_grid']).to(self.device);self.coarse.load_state_dict(coarse['model']);self.coarse.eval()
        self.mean=coarse['mean'].to(self.device);self.std=coarse['std'].to(self.device);self.crop_mean=self.bundle['crop_mean'].to(self.device);self.crop_std=self.bundle['crop_std'].to(self.device)
        self.encoder=load_model('dinov2_vits14',self.device,weights=project_path(self.cfg['dino_weights']),repo_dir=project_path(self.cfg['dino_repository']))
        self.image_mean=torch.tensor(IMAGENET_MEAN,device=self.device).view(1,3,1,1);self.image_std=torch.tensor(IMAGENET_STD,device=self.device).view(1,3,1,1)
        self.mode=self.bundle['selected_method'];self.threshold=self.bundle['comparison'][self.mode]['threshold'];self.head=None
        if self.mode in self.bundle['confidence']:
            dim=len(self.bundle['feature_center']);self.head=(torch.nn.Linear(dim,1) if self.mode.startswith('linear') else ConfidenceHead(dim)).to(self.device);self.head.load_state_dict(self.bundle['confidence'][self.mode]);self.head.eval()
        self.reference=None;cv2.setNumThreads(1);cv2.setRNGSeed(self.cfg['seed'])

    @torch.no_grad()
    def encode(self,images):
        from perception.extract_dino_features import to_input_tensor
        features=[];width=self.cfg['feature_batch']
        for start in range(0,len(images),width):
            batch=list(images[start:start+width]);count=len(batch);batch += [batch[-1]]*(width-count)
            with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                x=to_input_tensor(np.stack(batch),self.cfg['image_size'],self.image_mean,self.image_std,self.device)
                features.append(self.encoder.forward_features(x)['x_norm_patchtokens'][:count].to(torch.float16).float())
        return torch.cat(features)

    def set_reference(self,rgb):
        size=self.cfg['image_size'];native=square_rgb(rgb,max(rgb.shape[:2]));views=[cv2.resize(np.rot90(native,k).copy(),(size,size),interpolation=cv2.INTER_LINEAR) for k in range(4)]
        self.reference=F.normalize((self.encode(views)-self.crop_mean)/self.crop_std,dim=-1);self.rgb_reference=rgb_descriptors(rgb)

    @torch.no_grad()
    def predict(self,rgb):
        if self.reference is None:raise RuntimeError('Call set_reference with an RGB template first')
        size=self.cfg['image_size'];tokens=self.encode([cv2.resize(rgb,(size,size),interpolation=cv2.INTER_LINEAR)]);q=(tokens-self.mean)/self.std
        batch=q.expand(self.cfg['batch'],-1,-1).contiguous()
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):logits,stats=self.coarse(batch,torch.zeros_like(batch))
        logits=logits[:1];stats=stats[:1]
        map_=logits[0].cpu().numpy();grid=map_.shape[0];y,x=np.unravel_index(map_.argmax(),map_.shape);point=[(int(x)+.5)/grid,(int(y)+.5)/grid]
        box=proposal_box(map_,rgb.shape[:2],self.cfg['crop_margin'],self.cfg['crop_minimum_fraction'],self.cfg['crop_logit_delta']);x0,y0,x1,y1=box;crop=square_rgb(rgb[y0:y1,x0:x1],size)
        crop_tokens=F.normalize((self.encode([crop])[0]-self.crop_mean)/self.crop_std,dim=-1)
        vector,geometry=patch_verification_features(crop_tokens,self.reference,stats[0].cpu().numpy(),self.cfg['reprojection_threshold']);rgb_score=0.
        if self.cfg.get('rgb_ratio'):
            rgb_vector,rgb_score=rgb_correspondence_features(rgb_descriptors(crop),self.rgb_reference,self.cfg['rgb_reprojection_threshold'],self.cfg['rgb_ratio']);vector=np.r_[vector,rgb_vector.astype(np.float32)]
        self.last_features=vector
        if self.head is None:score=geometry if self.mode=='geometry' else rgb_score
        else:
            value=(vector-self.bundle['feature_center'].numpy())/self.bundle['feature_scale'].numpy()
            score=float(self.head(torch.from_numpy(value).to(self.device)).reshape(()).double().sigmoid())
        return {'point_xy':point,'box_xyxy':box,'score':score,'accepted':bool(score>=self.threshold),'threshold':self.threshold,'method':self.mode}
