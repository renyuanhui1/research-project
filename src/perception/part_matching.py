"""Locate the aircraft in RGB, then match a chosen part in an enlarged RGB crop."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from perception.fingerprint import Fingerprint
from perception.extract_dino_features import to_input_tensor


def response_crop(similarity, shape, min_peak=.2):
    """RGB-only crop from the connected coarse DINO response, with context padding."""
    sim = np.asarray(similarity, np.float32)
    grid = int(round(np.sqrt(sim.size)))
    sim = sim.reshape(grid, grid)
    if not np.isfinite(sim).all() or float(sim.max()) < min_peak:
        return None
    mask = (sim >= max(min_peak, float(sim.max())-.15)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(mask)
    component = labels == labels[np.unravel_index(sim.argmax(), sim.shape)]
    yy, xx = np.nonzero(component)
    if len(xx) < 2:
        return None
    h, w = shape[:2]
    padding = max(1.5, .2*max(xx.max()-xx.min()+1, yy.max()-yy.min()+1))
    return [max(0, int(np.floor((xx.min()-padding)*w/grid))),
            max(0, int(np.floor((yy.min()-padding)*h/grid))),
            min(w, int(np.ceil((xx.max()+1+padding)*w/grid))),
            min(h, int(np.ceil((yy.max()+1+padding)*h/grid)))]


def crop_to_full(center, box, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return (2*(x0+(center[0]+1)*.5*(x1-x0))/w-1,
            2*(y0+(center[1]+1)*.5*(y1-y0))/h-1)


def roi_overlap_weights(roi, grid, padding=1.):
    """Pool intersecting cells by area; padding is measured in feature cells."""
    roi=np.asarray(roi,float)
    if roi.shape!=(4,) or not np.isfinite(roi).all() or np.any(roi[:2]>=roi[2:]) or np.any(roi<0) or np.any(roi>1):
        raise ValueError('Invalid reference ROI')
    expanded=np.clip(roi+np.array([-1,-1,1,1])*padding/grid,0,1)
    low=np.arange(grid)/grid;high=(np.arange(grid)+1)/grid
    wx=np.maximum(0,np.minimum(high,expanded[2])-np.maximum(low,expanded[0]))
    wy=np.maximum(0,np.minimum(high,expanded[3])-np.maximum(low,expanded[1]))
    weights=np.outer(wy,wx)
    return weights/weights.sum(),expanded


def refine_local_image(rgb, template, anchor, candidate, coarse_box, previous=None,
                       min_score=.85, min_margin=.15):
    """Bounded, multiscale registration of a fixed context image and part anchor."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);raw=cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
    invalid={'accepted':False,'score':-1.,'margin':-1.,'reason':'no_local_match'}
    if raw.size<16 or float(raw.std())<1.:return dict(invalid,reason='uninformative_reference')
    h,w=gray.shape;radius=(np.array(coarse_box[2:])-coarse_box[:2])/32*1.5
    coarse_candidate=np.asarray(candidate,float);anchor=np.asarray(anchor,float)
    candidate=np.asarray(previous.get('point_px',coarse_candidate) if previous else coarse_candidate,float)
    if previous is None:
        scales=[.75,1.,1.25,1.5,2.,3.,4.,6.,8.,10.];angles=[-5.,0.,5.]
    else:
        scales=[previous['scale']*x for x in (.9,1.,1.1)]
        angles=[previous['angle_deg']+x for x in (-3.,0.,3.)]
    matches=[]
    for scale in scales:
        if not .5<=scale<=12.:continue
        tw,th=np.maximum(4,np.rint(np.array([raw.shape[1],raw.shape[0]])*scale).astype(int))
        if tw>w or th>h:continue
        scaled=cv2.resize(raw,(int(tw),int(th)));scaled_anchor=anchor*[tw/raw.shape[1],th/raw.shape[0]]
        for angle in angles:
            if abs(angle)>15.:continue
            transform=cv2.getRotationMatrix2D((tw/2,th/2),angle,1.)
            rotated=cv2.warpAffine(scaled,transform,(int(tw),int(th)),borderMode=cv2.BORDER_REPLICATE)
            ar=transform@np.r_[scaled_anchor,1.]
            low=np.maximum(0,np.floor(candidate-radius-ar).astype(int))
            high=np.minimum([w,h],np.ceil(candidate+radius+[tw,th]-ar).astype(int))
            search=gray[low[1]:high[1],low[0]:high[0]]
            if search.shape[0]<th or search.shape[1]<tw:continue
            corr=cv2.matchTemplate(search,rotated,cv2.TM_CCOEFF_NORMED)
            corr=np.nan_to_num(corr,nan=-1.,posinf=-1.,neginf=-1.)
            _,score,_,location=cv2.minMaxLoc(corr);x,y=location
            sidelobes=corr.copy();sidelobes[max(0,y-int(th/4)):y+int(th/4)+1,max(0,x-int(tw/2)):x+int(tw/2)+1]=-1.
            _,second,_,second_location=cv2.minMaxLoc(sidelobes)
            matches.append({'score':float(score),'second':float(second),'point_px':(low+location+ar).tolist(),
                            'second_point_px':(low+second_location+ar).tolist(),'scale':float(scale),
                            'angle_deg':float(angle),'template_wh':[int(tw),int(th)],'search_xyxy':np.r_[low,high].tolist()})
    if not matches:return invalid
    best=max(matches,key=lambda v:v['score']);point=np.array(best['point_px'])
    competitors=[best['second']]
    for match in matches:
        delta=np.abs(np.array(match['point_px'])-point)
        if delta[0]>best['template_wh'][0]/2 or delta[1]>best['template_wh'][1]/4:
            competitors.append(match['score'])
    margin=best['score']-max(competitors);inside=bool(np.all(np.abs(point-candidate)<=radius+1.))
    accepted=best['score']>=min_score and margin>=min_margin and inside
    return dict(best,accepted=accepted,margin=float(margin),reason=None if accepted else 'registration_confidence',
                min_score=min_score,min_margin=min_margin,coarse_candidate_px=coarse_candidate.tolist(),
                search_center_px=candidate.tolist(),search_source='previous_registered_point' if previous and 'point_px' in previous else 'DINO_candidate')


class CroppedPartMatcher(Fingerprint):
    """One frozen encoder, one reference and shared statistics for every part."""
    def _encode(self, rgb, bs=2):
        return super()._encode(rgb, bs=2)

    def __init__(self, args, device, rois, selected_part):
        super().__init__(args, device)
        self.selected_part = selected_part
        template = cv2.imread(str(args.template))[:, :, ::-1].copy()
        tokens = (self._encode(template[None], bs=1)[0].to(device)-self.zm)/self.zs
        axis = (torch.arange(self.grid, device=device)+.5)/self.grid
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.part_protos = {}
        self.reference_patch_counts = {}
        self.reference_center_patch_counts = {}
        self.reference_sampling = {}
        self.local_references = {}
        self.refinement_state = {}
        self.thin_roi_refinement = bool(getattr(args,'thin_roi_refinement',False))
        for name, (x0, y0, x1, y1) in rois.items():
            mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
            self.reference_patch_counts[name] = int(mask.sum())
            self.reference_center_patch_counts[name] = int(mask.sum())
            self.reference_sampling[name] = 'center'
            if not mask.any():
                if self.thin_roi_refinement:
                    weights,expanded=roi_overlap_weights([x0,y0,x1,y1],self.grid)
                    weight=torch.tensor(weights.reshape(-1),dtype=tokens.dtype,device=device)
                    self.part_protos[name]=F.normalize((F.normalize(tokens,dim=-1)*weight[:,None]).sum(0),dim=0)
                    self.reference_patch_counts[name]=int((weights>0).sum())
                    self.reference_sampling[name]='overlap_context_local_registration'
                    shape=np.array([template.shape[1],template.shape[0]])
                    low=np.floor(expanded[:2]*shape).astype(int);high=np.ceil(expanded[2:]*shape).astype(int)
                    self.local_references[name]={'rgb':np.ascontiguousarray(template[low[1]:high[1],low[0]:high[0]]),
                                                'anchor':np.array([(x0+x1)/2,(y0+y1)/2])*shape-low,
                                                'part_wh':np.array([x1-x0,y1-y0])*shape,
                                                'crop_xyxy':np.r_[low,high].tolist()}
                    continue
                self.part_protos[name] = None
                continue
            self.part_protos[name] = F.normalize(F.normalize(tokens[mask.reshape(-1)], dim=-1).mean(0), dim=0)
        self.last_diagnostics = {}

    def _response(self, tokens, proto):
        sim = F.normalize(tokens, dim=-1) @ proto
        weights = torch.softmax(sim/self.args.target_softmax_temp, dim=-1)
        center = weights @ self.patch_xy
        mass = torch.sigmoid((sim-self.args.target_mass_thresh)*self.args.target_mass_sharpness).mean()
        return float(center[0]), float(center[1]), float(mass), float(sim.max()), sim.cpu().numpy()

    @torch.no_grad()
    def query_all(self, rgb):
        coarse = super().query(rgb, use_roi=False)
        box = response_crop(coarse[-1], rgb.shape)
        self.last_diagnostics = {'whole_peak': coarse[3], 'crop_xyxy': box}
        if box is None:
            return {name: (0., 0., 0., -1., np.full(self.grid**2, -1., np.float32)) for name in self.part_protos}
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        x = to_input_tensor(crop[None], self.args.image_size, self.mean, self.std, self.dev)
        tokens = (self.dino.forward_features(x)['x_norm_patchtokens'][0].float()-self.zm)/self.zs
        result = {}
        for name, proto in self.part_protos.items():
            if proto is None:
                invalid = np.full(self.grid**2, -1., np.float32)
                result[name] = 0., 0., 0., -1., invalid
                self.last_diagnostics[name] = {'local_center': [0., 0.], 'local_similarity': invalid,
                                               'failure_reason': 'no_reference_patches'}
                continue
            cx, cy, mass, peak, local_sim = self._response(tokens, proto)
            fx, fy = crop_to_full((cx, cy), box, rgb.shape)
            canvas = np.full(rgb.shape[:2], -1., np.float32)
            canvas[y0:y1, x0:x1] = cv2.resize(local_sim.reshape(self.grid, self.grid), (x1-x0, y1-y0))
            result[name] = fx, fy, mass, peak, cv2.resize(canvas, (self.grid, self.grid)).reshape(-1)
            self.last_diagnostics[name] = {'local_center': [cx, cy], 'local_similarity': local_sim}
            if name in self.local_references:
                reference=self.local_references[name]
                candidate=(np.array([fx,fy])+1)*[rgb.shape[1]/2,rgb.shape[0]/2]
                refined=refine_local_image(rgb,reference['rgb'],reference['anchor'],candidate,box,
                                           self.refinement_state.get(name),
                                           getattr(self.args,'local_refinement_min_score',.85),
                                           getattr(self.args,'local_refinement_min_margin',.15))
                self.last_diagnostics[name].update(coarse_center=[fx,fy],refinement=refined,
                                                   point_source='local_image_registration')
                if refined['accepted']:
                    self.refinement_state[name]={'scale':refined['scale'],'angle_deg':refined['angle_deg'],'point_px':refined['point_px']}
                    point=np.array(refined['point_px']);fx,fy=point/[rgb.shape[1]/2,rgb.shape[0]/2]-1
                    self.last_diagnostics[name]['refined_part_wh_px']=(reference['part_wh']*refined['scale']).tolist()
                    result[name]=float(fx),float(fy),mass,peak,result[name][-1]
                else:
                    result[name]=fx,fy,mass,-1.,result[name][-1]
                    self.last_diagnostics[name]['failure_reason']=refined['reason']
        return result

    def query(self, rgb, use_roi=True):
        return self.query_all(rgb)[self.selected_part]
