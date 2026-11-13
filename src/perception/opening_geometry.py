"""俯视透视深度的洞口候选与投影净空诊断；不读取目标坐标，不证明三维通道安全。"""
import cv2
import numpy as np


def opening_candidates(depth, fov_degrees=90., min_layer_gap_m=6., group_bridge_m=2., down=None):
    depth = np.asarray(depth, np.float32)
    if depth.ndim != 2:
        raise ValueError('Expected a 2D perspective range image')
    h, w = depth.shape
    focal = w / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    yy, xx = np.mgrid[:h, :w]
    ray_norm = np.sqrt(1 + ((xx + .5 - w/2)/focal)**2 + ((yy + .5 - h/2)/focal)**2)
    planar = depth / ray_norm
    if down is not None:
        down = np.asarray(down, float)
        if down.shape != (3,) or not np.isfinite(down).all() or np.linalg.norm(down) < .9:
            raise ValueError('Invalid optical-frame gravity direction')
        down = down / np.linalg.norm(down)
        # 水平屋面的各像素具有相同重力投影距离，斜视下相机前向距离并不相同。
        planar = planar * (down[2] + (xx+.5-w/2)/focal*down[0] + (yy+.5-h/2)/focal*down[1])
    valid = np.isfinite(planar) & (planar > .1) & (planar < 500)
    result = {'status':'insufficient_depth', 'candidates':[], 'planar_depth':planar,
              'valid':valid, 'far_mask':np.zeros_like(valid), 'focal_px':float(focal)}
    if valid.sum() < 100:
        return result
    lo, hi = np.percentile(planar[valid], [1,99])
    if hi-lo < min_layer_gap_m:
        result['status']='no_separated_depth_layers'
        return result
    scaled = np.zeros_like(depth, np.uint8)
    scaled[valid] = np.clip((planar[valid]-lo)/(hi-lo)*255,0,255).astype(np.uint8)
    threshold,_ = cv2.threshold(scaled[valid].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # threshold是整数箱索引；使用箱上界，避免浮点噪声将同一平面误分到远层。
    threshold_m = lo + (threshold+1)/255*(hi-lo)
    near = valid & (planar <= threshold_m)
    far = valid & (planar > threshold_m)
    if min(near.sum(),far.sum()) < 100:
        result['status']='insufficient_layer_support'
        return result
    near_depth = float(np.median(planar[near]))
    far_depth = float(np.median(planar[far]))
    result.update(near_layer_depth_m=near_depth,far_layer_depth_m=far_depth,
                  threshold_m=float(threshold_m),far_mask=far)
    if far_depth-near_depth < min_layer_gap_m:
        result['status']='no_separated_depth_layers'
        return result
    # 仅为身份匹配归组。真正的可见空隙仍使用原far，绝不填掉钢梁或无效深度。
    kernel = min(41,max(3,2*int(np.ceil(group_bridge_m*focal/near_depth/2))+1))
    # 先移除与画外相连的远层，避免归组时把屋外地面与洞内地面连成一个候选。
    _,raw_labels,_,_=cv2.connectedComponentsWithStats(far.astype(np.uint8),8)
    boundary_labels=np.unique(np.r_[raw_labels[0],raw_labels[-1],raw_labels[:,0],raw_labels[:,-1]])
    bounded_far=far & ~np.isin(raw_labels,boundary_labels)
    # 显式补零，避免OpenCV闭运算默认边界将临近画边的洞口扩展到画外。
    padded=np.pad(bounded_far.astype(np.uint8),kernel)
    grouped = cv2.morphologyEx(padded,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))[kernel:-kernel,kernel:-kernel]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped,8)
    result['grouping_kernel_px']=kernel
    for label in range(1,count):
        x,y,bw,bh,area = map(int,stats[label])
        if area < 30 or x==0 or y==0 or x+bw>=w or y+bh>=h:
            continue
        group = labels==label
        visible = group & bounded_far & valid
        if visible.sum() < 30:
            continue
        distances = cv2.distanceTransform(np.pad(visible.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
        py,px=np.unravel_index(np.argmax(distances),distances.shape)
        clearance = float(distances[py,px]*near_depth/focal)
        result['candidates'].append({'label':label,'bbox_depth_xyxy':[x,y,x+bw,y+bh],
            'group_area_px':area,'visible_gap_area_px':int(visible.sum()),'group_mask':group,
            'visible_gap_mask':visible,'aperture_point_depth_xy':[int(px),int(py)],
            'projected_clearance_m':clearance})
    result['status']='candidates_found' if result['candidates'] else 'no_closed_opening'
    return result
