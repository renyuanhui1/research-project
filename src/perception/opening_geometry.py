"""俯视透视深度的洞口候选与投影净空诊断；不读取目标坐标，不证明三维通道安全。"""
import cv2
import numpy as np


def _plane_mask(visible, box, depth_m, focal, down):
    """按真实可见区域生成米制屋面栅格。"""
    h,w=visible.shape
    right=np.array([1.,0.,0.])-down[0]*down
    right/=np.linalg.norm(right)
    basis=np.array([right,np.cross(down,right),down])
    k=np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1.]])
    homography=basis@np.linalg.inv(k)
    homography[:2]*=depth_m
    x,y,a,b=box
    corners=cv2.perspectiveTransform(np.float32([[[x,y],[a,y],[a,b],[x,b]]]),homography)[0]
    low,high=corners.min(0),corners.max(0)
    resolution=max(depth_m/focal,float(max(high-low))/(2*w))
    transform=np.array([[1/resolution,0,-low[0]/resolution],
                        [0,1/resolution,-low[1]/resolution],[0,0,1.]])@homography
    size=tuple(np.ceil((high-low)/resolution).astype(int)+1)
    mask=cv2.warpPerspective(visible.astype(np.uint8),transform,size,flags=cv2.INTER_NEAREST)
    return mask,transform,resolution


def plane_region_center(visible, box, depth_m, focal, down):
    """屋面区域的米制几何中心，仅作上方水平航点，不宣称该点可穿过。"""
    mask,transform,_=_plane_mask(visible,box,depth_m,focal,down)
    ys,xs=np.where(mask)
    return cv2.perspectiveTransform(np.float32([[[xs.mean(),ys.mean()]]]),np.linalg.inv(transform))[0,0]


def plane_aperture(visible, box, depth_m, focal, down):
    """把真实可见空隙投影到水平屋面，按米制距离选择圆心；不填补遮挡。"""
    mask,transform,resolution=_plane_mask(visible,box,depth_m,focal,down)
    distances=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    ys,xs=np.where(mask)
    index=np.argmin((xs-xs.mean())**2+(ys-ys.mean())**2)
    py,px=int(ys[index]),int(xs[index])
    point=cv2.perspectiveTransform(np.float32([[[px,py]]]),np.linalg.inv(transform))[0,0]
    ys,xs=np.where(visible)
    index=np.argmin((xs-point[0])**2+(ys-point[1])**2)
    return int(xs[index]),int(ys[index]),float(distances[py,px]*resolution)


def opening_candidates(depth, fov_degrees=90., min_layer_gap_m=6., group_bridge_m=2., down=None, tracked_box=None, tracked_group_box=None):
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
    # 用于身份和上方航点归组。真实通道仍使用原far，不填掉钢梁或无效深度。
    kernel = min(41,max(3,2*int(np.ceil(group_bridge_m*focal/near_depth/2))+1))
    # 先移除与画外相连的远层，避免归组时把屋外地面与洞内地面连成一个候选。
    _,raw_labels,raw_stats,_=cv2.connectedComponentsWithStats(far.astype(np.uint8),8)
    boundary_labels=np.unique(np.r_[raw_labels[0],raw_labels[-1],raw_labels[:,0],raw_labels[:,-1]])
    retained_far=far & ~np.isin(raw_labels,boundary_labels)
    tracked_boundary_mask=np.zeros_like(far)
    for previous_box in (tracked_box,tracked_group_box):
        if previous_box is None:continue
        old=np.asarray(previous_box,float)
        for label in boundary_labels[boundary_labels>0]:
            x,y,bw,bh,area=map(int,raw_stats[label]);box=np.array([x,y,x+bw,y+bh],float)
            overlap=np.maximum(0,np.minimum(box[2:],old[2:])-np.maximum(box[:2],old[:2])).prod()
            union=bw*bh+(old[2:]-old[:2]).prod()-overlap
            if union>0 and overlap/union>=.5:
                tracked_boundary_mask |= raw_labels==label
    retained_far |= tracked_boundary_mask
    # 显式补零，避免OpenCV闭运算默认边界将临近画边的洞口扩展到画外。
    padded=np.pad(retained_far.astype(np.uint8),kernel)
    grouped = cv2.morphologyEx(padded,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))[kernel:-kernel,kernel:-kernel]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped,8)
    result['grouping_kernel_px']=kernel
    for label in range(1,count):
        group=labels==label
        tracked_boundary=bool(np.any(group & tracked_boundary_mask))
        x,y,bw,bh,area = map(int,stats[label])
        if area < 30 or (not tracked_boundary and (x==0 or y==0 or x+bw>=w or y+bh>=h)):
            continue
        visible = group & retained_far & valid
        if visible.sum() < 30:
            continue
        if down is None:
            distances = cv2.distanceTransform(np.pad(visible.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
            py,px=np.unravel_index(np.argmax(distances),distances.shape)
            clearance = float(distances[py,px]*near_depth/focal)
        else:
            px,py,clearance=plane_aperture(visible,[x,y,x+bw,y+bh],near_depth,focal,down)
        result['candidates'].append({'label':label,'bbox_depth_xyxy':[x,y,x+bw,y+bh],
            'group_area_px':area,'visible_gap_area_px':int(visible.sum()),'group_mask':group,
            'visible_gap_mask':visible,'aperture_point_depth_xy':[int(px),int(py)],
            'projected_clearance_m':clearance,'tracked_boundary_gap':tracked_boundary})
    result['status']='candidates_found' if result['candidates'] else 'no_closed_opening'
    return result
