"""Shared frozen-token projection, template cross-attention and whole-target localization."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class TemplateMatcher(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=64, temperature=.1, conditioned=True, output_grid=64):
        super().__init__()
        self.conditioned,self.temperature,self.output_grid=conditioned,temperature,output_grid
        self.project=nn.Sequential(nn.Linear(feature_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(4*hidden_dim+2,hidden_dim),nn.GELU())
        self.decode=nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.GELU(),
                                  nn.Conv2d(hidden_dim,hidden_dim//2,3,padding=1),nn.GELU(),
                                  nn.Conv2d(hidden_dim//2,1,1))

    def forward(self, search, template):
        query=self.project(search)
        grid=math.isqrt(query.shape[1])
        if grid*grid!=query.shape[1]:raise ValueError('Search tokens must form a square grid')
        if self.conditioned:
            keys=self.project(template)
            tg=math.isqrt(keys.shape[1])
            keys=F.adaptive_avg_pool2d(keys.transpose(1,2).reshape(len(keys),-1,tg,tg),(8,8)).flatten(2).transpose(1,2)
            similarity=F.normalize(query,dim=-1)@F.normalize(keys,dim=-1).transpose(1,2)
            context=torch.softmax(similarity/self.temperature,dim=-1)@keys
            maximum=similarity.max(-1).values
            average=similarity.mean(-1)
        else:
            context=torch.zeros_like(query)
            maximum=average=torch.zeros(query.shape[:2],device=query.device,dtype=query.dtype)
        joint=torch.cat([query,context,query*context,torch.abs(query-context),maximum[...,None],average[...,None]],-1)
        spatial=self.fuse(joint).transpose(1,2).reshape(len(query),-1,grid,grid)
        spatial=F.interpolate(spatial,size=(self.output_grid,self.output_grid),mode='bilinear',align_corners=False)
        logits=self.decode(spatial)[:,0].float()
        flat=logits.flatten(1);top=torch.topk(flat,8,dim=1).values;prob=flat.softmax(-1)
        features=torch.stack([top[:,0],top.mean(-1),flat.mean(-1),flat.std(-1),top[:,0]-top[:,1],
                              prob.max(-1).values,-(prob*prob.clamp_min(1e-12).log()).sum(-1)/math.log(flat.shape[1]),
                              flat.sigmoid().mean(-1),maximum.float().max(-1).values,maximum.float().mean(-1)],-1)
        return logits,features


class ConfidenceHead(nn.Module):
    def __init__(self,input_dim=10):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,32),nn.GELU(),nn.Linear(32,1))

    def forward(self, features):return self.net(features).squeeze(-1)


def localization_loss(logits, target, valid):
    known=valid.flatten(1).sum(-1)>0
    if not known.any():return logits.sum()*0
    weighted=(logits+valid.clamp_min(1e-6).log()).masked_fill(valid<=0,-10000).flatten(1)
    mass=target.flatten(1).sum(-1)
    present=(mass>0)&known
    distribution=target.flatten(1)/mass[:,None].clamp_min(1e-8)
    loss=-(distribution*weighted.log_softmax(-1)).sum(-1)
    spatial=loss[present].mean() if present.any() else logits.sum()*0
    presence=torch.logsumexp(weighted,-1)-valid.flatten(1).sum(-1).clamp_min(1).log()
    return spatial+F.binary_cross_entropy_with_logits(presence[known],present[known].float())


def presence_score(logits):
    """Image-only existence score; no evaluation mask enters inference."""
    return (torch.logsumexp(logits.flatten(1).double(),-1)-math.log(logits[0].numel())).sigmoid()


def template_pair_loss(positive,negative,target,valid,margin=1.):
    weight=target*valid
    mass=weight.flatten(1).sum(-1);known=mass>0
    if not known.any():return (positive.sum()+negative.sum())*0
    contrast=F.softplus(negative-positive+margin)*weight
    return (contrast.flatten(1).sum(-1)/mass.clamp_min(1e-8))[known].mean()
