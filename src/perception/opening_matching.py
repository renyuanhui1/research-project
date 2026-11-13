"""固定洞口模板的深度候选与DINO空间对应，供离线评价及水平对准复用。"""
import cv2
import numpy as np
import torch
from perception.opening_geometry import opening_candidates


@torch.no_grad()
def depth_constrained_match(fp, rgb, depth, reference_tokens, min_score, down=None):
    geometry = opening_candidates(depth, down=down)
    h,w=rgb.shape[:2]
    dh,dw=depth.shape
    choices=[]
    for candidate in geometry['candidates']:
        x0,y0,x1,y1=candidate['bbox_depth_xyxy']
        # 由传感器候选决定裁剪，保留洞缘上下文；不使用目标世界坐标。
        px,py=(x1-x0)*.15,(y1-y0)*.15
        box=[max(0,int((x0-px)*w/dw)),max(0,int((y0-py)*h/dh)),
             min(w,int(np.ceil((x1+px)*w/dw))),min(h,int(np.ceil((y1+py)*h/dh)))]
        bx0,by0,bx1,by1=box
        tokens=fp._encode(np.ascontiguousarray(rgb[by0:by1,bx0:bx1])[None])[0]
        tokens=torch.nn.functional.normalize((tokens-fp.zm.cpu())/fp.zs.cpu(),dim=-1)
        spatial_score=float((tokens*reference_tokens).sum(-1).mean())
        global_score=float(torch.nn.functional.cosine_similarity(tokens.mean(0),reference_tokens.mean(0),dim=0))
        point=candidate['aperture_point_depth_xy']
        choices.append({'candidate_label':candidate['label'],'crop_rgb_xyxy':box,
                        'bbox_depth_xyxy':candidate['bbox_depth_xyxy'],
                        'spatial_dino_score':spatial_score,'global_dino_score':global_score,
                        'group_area_px':candidate['group_area_px'],
                        'projected_aperture_radius_m':candidate['projected_clearance_m'],
                        'aperture_point_rgb_xy':[(point[0]+.5)*w/dw-.5,(point[1]+.5)*h/dh-.5]})
    choices.sort(key=lambda c:c['spatial_dino_score'],reverse=True)
    selected=choices[0] if choices else None
    strong=selected is not None and selected['spatial_dino_score']>=min_score
    view=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for choice in choices:
        box=choice['crop_rgb_xyxy'];color=(0,255,255) if choice is selected else (180,180,180)
        cv2.rectangle(view,tuple(box[:2]),tuple(box[2:]),color,3)
        cv2.putText(view,'%d %.3f'%(choice['candidate_label'],choice['spatial_dino_score']),
                    (box[0]+4,box[1]+23),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2,cv2.LINE_AA)
    if strong:
        obj=next(c for c in geometry['candidates'] if c['label']==selected['candidate_label'])
        visible=cv2.resize(obj['visible_gap_mask'].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        view[visible]=(view[visible]*.7+np.array([0,150,0])*.3).astype(np.uint8)
        point=tuple(round(v) for v in selected['aperture_point_rgb_xy'])
        cv2.drawMarker(view,point,(255,255,0),cv2.MARKER_CROSS,30,3)
    view=cv2.resize(view,(960,540))
    view=cv2.copyMakeBorder(view,45,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
    identity='no_candidate' if selected is None else ('development_threshold_pass' if strong else 'weak_reference_match')
    title=identity+' | visible rays only; NOT 3D passage clearance'
    cv2.putText(view,title,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    result={k:geometry[k] for k in ('status','near_layer_depth_m','far_layer_depth_m','threshold_m','grouping_kernel_px') if k in geometry}
    result.update(candidates=choices,selected=selected,ue_target_coordinates_used=False,
                  depth_layer_axis='gravity' if down is not None else 'camera_forward',
                  collision_truth_used=False,selection_method='spatial DINO similarity across geometry candidate crops',
                  projected_clearance_is_3d_passage_proof=False,reference_match_status=identity,
                  development_score_threshold=min_score,threshold_calibrated_on_held_out_data=False,
                  proposed_aperture_point_rgb_xy=selected['aperture_point_rgb_xy'] if strong else None)
    if selected:
        largest=max(choices,key=lambda c:c['group_area_px'])
        result['largest_area_candidate_label']=largest['candidate_label']
        result['dino_top_equals_largest_area']=selected['candidate_label']==largest['candidate_label']
    if len(choices)>1:result['top_two_score_margin']=choices[0]['spatial_dino_score']-choices[1]['spatial_dino_score']
    return result,view

