"""整目标屋面关联或固定洞口模板的RGB-D候选匹配，供评价及水平对准复用。"""
import cv2
import numpy as np
import torch
from perception.opening_geometry import opening_candidates, plane_aperture, plane_region_center


def body_constrained_match(fp, rgb, depth, min_peak, down=None, previous_box=None, allow_tracked_gap=False, previous_group_box=None):
    """确认整目标后，在定位点所属屋面中选择任一闭合破口；不辨认指定洞口。"""
    geometry = opening_candidates(depth, down=down,tracked_box=previous_box if allow_tracked_gap else None,
        tracked_group_box=previous_group_box if allow_tracked_gap else None)
    dh,dw=depth.shape
    query_box=[0,0,rgb.shape[1],rgb.shape[0]]
    query_previous=previous_group_box if previous_group_box is not None else previous_box
    if query_previous is not None and allow_tracked_gap:
        x0,y0,x1,y1=np.asarray(query_previous,float)
        query_box=[max(0,int((2*x0-x1)*rgb.shape[1]/dw)),max(0,int((2*y0-y1)*rgb.shape[0]/dh)),
                   min(rgb.shape[1],int((2*x1-x0)*rgb.shape[1]/dw)),min(rgb.shape[0],int((2*y1-y0)*rgb.shape[0]/dh))]
    x0,y0,x1,y1=query_box
    cx,cy,mass,peak,_ = fp.query(np.ascontiguousarray(rgb[y0:y1,x0:x1]))
    cx=2*(x0+(cx+1)*(x1-x0)/2)/rgb.shape[1]-1
    cy=2*(y0+(cy+1)*(y1-y0)/2)/rgb.shape[0]-1
    px=int(np.clip((cx+1)*dw/2,0,dw-1))
    py=int(np.clip((cy+1)*dh/2,0,dh-1))
    near=geometry['valid'] & ~geometry['far_mask']
    _,labels=cv2.connectedComponents(near.astype(np.uint8),8)
    roof_label=int(labels[py,px])
    choices=[]
    for candidate in geometry['candidates']:
        ring=(cv2.dilate(candidate['group_mask'].astype(np.uint8),np.ones((5,5),np.uint8))>0)&~candidate['group_mask']
        surrounding=labels[ring & near]
        ids,counts=np.unique(surrounding[surrounding>0],return_counts=True)
        if not len(ids):continue
        enclosing=int(ids[np.argmax(counts)])
        same_roof=(roof_label>0 and roof_label==enclosing) or bool(candidate['group_mask'][py,px])
        if not same_roof and not (allow_tracked_gap and previous_box is not None):continue
        if down is not None:
            group_point=plane_region_center(candidate['visible_gap_mask'],candidate['bbox_depth_xyxy'],
                geometry['near_layer_depth_m'],dw/2,np.asarray(down)/np.linalg.norm(down))
        else:
            yy,xx=np.where(candidate['visible_gap_mask']);group_point=[xx.mean(),yy.mean()]
        group_choices=[]
        count,gaps,stats,_=cv2.connectedComponentsWithStats(candidate['visible_gap_mask'].astype(np.uint8),8)
        for gap_label in range(1,count):
            x,y,width,height,area=map(int,stats[gap_label])
            if area<30:continue
            visible=gaps==gap_label;box=[x,y,x+width,y+height]
            if down is not None:
                gx,gy,radius=plane_aperture(visible,box,geometry['near_layer_depth_m'],dw/2,np.asarray(down)/np.linalg.norm(down))
            else:
                distance=cv2.distanceTransform(np.pad(visible.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
                gy,gx=np.unravel_index(np.argmax(distance),distance.shape)
                radius=float(distance[gy,gx]*geometry['near_layer_depth_m']/(dw/2))
            overlap=0.
            if previous_box is not None:
                old=np.asarray(previous_box,float);new=np.asarray(box,float)
                intersection=np.maximum(0,np.minimum(old[2:],new[2:])-np.maximum(old[:2],new[:2])).prod()
                overlap=float(intersection/((old[2:]-old[:2]).prod()+(new[2:]-new[:2]).prod()-intersection))
            group_choices.append({'candidate_label':candidate['label'],'visible_gap_component_label':gap_label,
                'bbox_depth_xyxy':box,'group_bbox_depth_xyxy':candidate['bbox_depth_xyxy'],
                'group_alignment_point_rgb_xy':[(float(group_point[0])+.5)*rgb.shape[1]/dw-.5,
                                                (float(group_point[1])+.5)*rgb.shape[0]/dh-.5],
                'crop_rgb_xyxy':[int(box[0]*rgb.shape[1]/dw),int(box[1]*rgb.shape[0]/dh),
                                 int(box[2]*rgb.shape[1]/dw),int(box[3]*rgb.shape[0]/dh)],
                'group_area_px':area,'projected_aperture_radius_m':radius,
                'aperture_point_rgb_xy':[(gx+.5)*rgb.shape[1]/dw-.5,(gy+.5)*rgb.shape[0]/dh-.5],
                'enclosing_roof_component':enclosing,'previous_box_iou':overlap,
                'tracked_boundary_gap':candidate['tracked_boundary_gap']})
        tracked=allow_tracked_gap and any(c['previous_box_iou']>=.5 for c in group_choices)
        if same_roof or tracked:
            for choice in group_choices:choice['identity_association']='current_body_point' if same_roof else 'verified_free_gap_continuity'
            choices.extend(group_choices)
    choices.sort(key=lambda c:(c['previous_box_iou'],c['projected_aperture_radius_m']),reverse=True)
    selected=choices[0] if choices else None
    strong=bool(np.isfinite(peak) and peak>=min_peak and selected is not None)
    result={k:geometry[k] for k in ('status','near_layer_depth_m','far_layer_depth_m','threshold_m','grouping_kernel_px') if k in geometry}
    result.update(candidates=choices,selected=selected,identity_score=peak,
        group_alignment_point_rgb_xy=selected['group_alignment_point_rgb_xy'] if strong else None,
        group_alignment_point_is_free_passage_proof=False,
        body_identity={'peak':peak,'center_normalized_xy':[cx,cy],'roof_component':roof_label,'query_crop_rgb_xyxy':query_box},
        proposed_aperture_point_rgb_xy=selected['aperture_point_rgb_xy'] if strong else None,
        reference_match_status='whole_target_roof_association' if strong else 'body_or_roof_association_missing',
        selection_method='any closed opening on the DINO identified connected roof',
        ue_target_coordinates_used=False,collision_truth_used=False,
        projected_clearance_is_3d_passage_proof=False,
        development_score_threshold=min_peak,threshold_calibrated_on_held_out_data=False)
    view=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.drawMarker(view,(round((cx+1)*rgb.shape[1]/2),round((cy+1)*rgb.shape[0]/2)),(255,0,255),cv2.MARKER_CROSS,25,2)
    if selected:
        box=selected['crop_rgb_xyxy'];cv2.rectangle(view,tuple(box[:2]),tuple(box[2:]),(0,255,255),3)
        cv2.drawMarker(view,tuple(round(v) for v in selected['aperture_point_rgb_xy']),(255,255,0),cv2.MARKER_CROSS,25,2)
    return result,cv2.resize(view,(960,540))


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

