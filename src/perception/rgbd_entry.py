"""RGB-D视觉锚点与静态场景的采样下降通道；不使用UE目标或相机世界位置。"""
import cv2
import numpy as np

# OpenCV光学坐标(右、下、前)转ProjectAirSim相机坐标(前、右、下)。
CV_TO_SIM = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def intrinsics(width=960, height=540):
    return np.array([[width/2, 0., (width-1)/2],
                     [0., width/2, (height-1)/2], [0., 0., 1.]])


def points_from_range(pixels, ranges, matrix):
    rays=np.column_stack(((pixels[:,0]-matrix[0,2])/matrix[0,0],
                          (pixels[:,1]-matrix[1,2])/matrix[1,1], np.ones(len(pixels))))
    return rays/np.linalg.norm(rays,axis=1,keepdims=True)*ranges[:,None]


def rigid_transform(source,target):
    a=source.mean(0);b=target.mean(0)
    u,_,vt=np.linalg.svd((source-a).T@(target-b))
    sign=np.eye(3);sign[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@sign@u.T
    return rotation,b-rotation@a


def robust_rgbd_transform(source,target):
    rng=np.random.default_rng(0);best=np.zeros(len(source),bool)
    for _ in range(100):
        ids=rng.choice(len(source),3,replace=False)
        if np.linalg.norm(np.cross(source[ids[1]]-source[ids[0]],source[ids[2]]-source[ids[0]]))<.01:continue
        rotation,translation=rigid_transform(source[ids],target[ids])
        valid=np.linalg.norm(source@rotation.T+translation-target,axis=1)<.2
        if valid.sum()>best.sum():best=valid
    if best.sum()<25 or best.sum()<.6*len(source):raise RuntimeError('RGBD_pose_not_supported')
    rotation,translation=rigid_transform(source[best],target[best])
    return rotation,translation,np.flatnonzero(best)


class VisualAnchor:
    """首帧RGB-D固定三维特征，LK跟踪及三维刚体配准求当前相机相对位姿。"""
    def __init__(self, rgb, depth, mask=None):
        self.matrix=intrinsics()
        self.gray=self.gray_image(rgb)
        self.pixels,self.points=self.seed_points(self.gray,depth,mask)
        self.rotation=np.eye(3);self.translation=np.zeros(3)
        self.previous_center=np.zeros(3);self.keyframe_center=np.zeros(3);self.keyframes=1
        self.quality={'tracked':len(self.points),'inliers':len(self.points),'median_reprojection_px':0.}

    def seed_points(self,gray,depth,mask=None):
        ranges=cv2.resize(np.asarray(depth,np.float32),(960,540),interpolation=cv2.INTER_NEAREST)
        valid=np.isfinite(ranges)&(ranges>.5)&(ranges<150)
        if mask is not None:
            valid &= cv2.resize(mask.astype(np.uint8),(960,540),interpolation=cv2.INTER_NEAREST)>0
        else:
            central=np.zeros_like(valid);central[81:459,144:816]=True;valid &= central
        features=cv2.goodFeaturesToTrack(gray,2000,.002,4,mask=valid.astype(np.uint8)*255,blockSize=5)
        if features is None or len(features)<40:raise RuntimeError('insufficient_visual_anchor_features')
        pixels=features.reshape(-1,2);x,y=np.rint(pixels).astype(int).T
        return pixels,points_from_range(pixels,ranges[y,x],self.matrix).astype(np.float64)

    @staticmethod
    def gray_image(rgb):
        return cv2.cvtColor(cv2.resize(rgb,(960,540)),cv2.COLOR_RGB2GRAY)

    def update(self,rgb,depth):
        gray=self.gray_image(rgb)
        old=self.pixels.reshape(-1,1,2).astype(np.float32)
        new,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,old,None,
            winSize=(25,25),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
        if new is None:raise RuntimeError('visual_tracking_lost')
        back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,new,None,
            winSize=(25,25),maxLevel=4)
        if back is None:raise RuntimeError('visual_tracking_lost')
        pixels=new.reshape(-1,2)
        valid=(status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back.reshape(-1,2)-self.pixels,axis=1)<.8)
        valid &= np.isfinite(pixels).all(1)&(pixels[:,0]>=2)&(pixels[:,0]<958)&(pixels[:,1]>=2)&(pixels[:,1]<538)
        points=self.points[valid];pixels=pixels[valid]
        if len(points)<25:raise RuntimeError('too_few_visual_tracks')
        ranges=cv2.resize(np.asarray(depth,np.float32),(960,540),interpolation=cv2.INTER_NEAREST)
        x,y=np.rint(pixels).astype(int).T
        values=ranges[y,x]
        valid_depth=np.isfinite(values)&(values>.5)&(values<150)
        points,pixels,values=points[valid_depth],pixels[valid_depth],values[valid_depth]
        if len(points)<25:raise RuntimeError('insufficient_RGBD_tracks')
        current=points_from_range(pixels,values,self.matrix)
        rotation,translation,ids=robust_rgbd_transform(points,current)
        local=points[ids]@rotation.T+translation
        projected=np.column_stack((local[:,0]/local[:,2]*self.matrix[0,0]+self.matrix[0,2],
                                   local[:,1]/local[:,2]*self.matrix[1,1]+self.matrix[1,2]))
        residual=float(np.median(np.linalg.norm(projected-pixels[ids],axis=1)))
        metric_residual=float(np.median(np.linalg.norm(local-current[ids],axis=1)))
        pixel_limit=max(1.2,.1*self.matrix[0,0]/float(np.median(local[:,2])))
        center=-rotation.T@translation
        angle=np.arccos(np.clip((np.trace(rotation@self.rotation.T)-1)/2,-1,1))
        if (not np.isfinite(center).all() or residual>pixel_limit or metric_residual>.1 or
                np.linalg.norm(center-self.previous_center)>2. or angle>np.deg2rad(5)):
            raise RuntimeError('visual_pose_jump_or_residual: %.3f px, %.3f m residual, %.3f m step, %.3f deg' %
                               (residual,metric_residual,np.linalg.norm(center-self.previous_center),np.rad2deg(angle)))
        # 固定锚点，不积分无人机真值位姿；剔除失配后继续跟踪原三维点。
        self.points=points[ids];self.pixels=pixels[ids];self.gray=gray
        self.rotation,self.translation=rotation,translation
        self.previous_center=center
        self.quality={'tracked':len(points),'inliers':len(ids),'median_reprojection_px':residual,
                      'median_registration_m':metric_residual,'reprojection_limit_px':pixel_limit}
        if np.linalg.norm(center-self.keyframe_center)>4. or metric_residual>.06 or len(ids)<150:
            pixels,current_points=self.seed_points(gray,depth)
            self.pixels=pixels;self.points=(current_points-translation)@rotation
            self.keyframe_center=center.copy();self.keyframes+=1
        self.quality['keyframes']=self.keyframes
        return center


class FreeTube:
    """用真实深度射线核验下降棱柱，未知不计自由；缓存仅适用于静态场景。"""
    def __init__(self,aperture,down,world_from_anchor,inside_m=8.,half_width_m=1.25,step_m=.25,oblique=False):
        self.down=np.asarray(down,float)
        self.down/=np.linalg.norm(self.down)
        self.roof_distance=float(np.dot(aperture,self.down))
        if not np.isfinite(aperture).all() or self.roof_distance <= 3.5:
            raise ValueError('Invalid roof intersection')
        self.goal_progress=self.roof_distance+inside_m
        self.oblique=oblique
        self.aperture=np.asarray(aperture,float)
        self.axis_slope=self.aperture/self.roof_distance if oblique else self.down.copy()
        self.axis_origin=np.zeros(3) if oblique else self.aperture-self.down*self.roof_distance
        axes=np.asarray(world_from_anchor).T
        # 斜向中心线在机体上下表面之间也会水平偏移；扩大截面以包含整机扫掠体。
        slope_world=np.asarray(world_from_anchor)@self.axis_slope
        self.horizontal_half_widths=half_width_m*(1+np.abs(slope_world[:2]))
        def offsets(width):
            count=int(np.ceil(width/step_m-1e-10))
            return np.arange(-count,count+1)*step_m
        # 相机近盲区之外，从初始下方3.5米起覆盖整条下降路径。
        levels=np.arange(3.5,self.goal_progress+half_width_m+step_m,step_m)
        n,e,z=np.meshgrid(offsets(self.horizontal_half_widths[0]),offsets(self.horizontal_half_widths[1]),levels,indexing='ij')
        self.points=(self.axis_origin+n.ravel()[:,None]*axes[:,0]+
                     e.ravel()[:,None]*axes[:,1]+z.ravel()[:,None]*self.axis_slope)
        self.last_seen=np.full(len(self.points),-np.inf)
        cell_diagonal=np.linalg.norm(np.r_[1+np.abs(slope_world[:2]),1.])
        self.margin_m=.3+cell_diagonal*step_m/2
        self.max_cached_age_s=45.
        self.half_width_m=half_width_m
        self.step_m=step_m

    def update(self,depth,rotation,translation,now,center):
        depth=np.asarray(depth,np.float32)
        h,w=depth.shape
        safe=np.where(np.isfinite(depth)&(depth>.1)&(depth<150),depth,0).astype(np.float32)
        # 取邻域最小值，横梁与无效像素不能被平均掉。
        safe=cv2.erode(safe,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        local=self.points@rotation.T+translation
        future=(self.points-center)@self.down>=-self.half_width_m
        front=local[:,2]>.5
        u=np.full(len(local),-1.);v=u.copy()
        u[front]=local[front,0]/local[front,2]*(w/2)+(w-1)/2
        v[front]=local[front,1]/local[front,2]*(w/2)+(h-1)/2
        visible=front&(u>=2)&(u<w-3)&(v>=2)&(v<h-3)&future
        ids=np.flatnonzero(visible)
        x=np.rint(u[ids]).astype(int);y=np.rint(v[ids]).astype(int)
        measured=safe[y,x];distance=np.linalg.norm(local[ids],axis=1)
        if np.any(measured<=.1):raise RuntimeError('invalid_depth_in_entry_tube')
        blocked=(measured>.1)&(measured<=distance+self.margin_m)
        if np.any(blocked):raise RuntimeError('observed_obstacle_in_entry_tube')
        clear=measured>distance+self.margin_m
        self.last_seen[ids[clear]]=now
        fresh=(now-self.last_seen)<=self.max_cached_age_s
        if not np.all(fresh[future]):raise RuntimeError('unknown_or_stale_entry_tube')
        return {'samples':len(self.points),'future_samples':int(future.sum()),
                'refreshed_samples':int(clear.sum()),'half_width_m':self.half_width_m,
                'sampling_step_m':self.step_m,'depth_margin_m':self.margin_m,
                'oblique':self.oblique,'axis_slope_anchor':self.axis_slope.tolist(),
                'horizontal_half_widths_m':self.horizontal_half_widths.tolist(),
                'max_future_observation_age_s':float(np.max(now-self.last_seen[future])),
                'static_scene_assumption':True,'collision_truth_used':False}


def entry_velocity(center,tube,world_from_anchor):
    """视觉相对位置生成NED速度；不使用UE位置，下降阶段仍保持水平闭环。"""
    progress=float(np.dot(center,tube.down))
    delta=tube.axis_origin+tube.axis_slope*progress-center
    error=(world_from_anchor@delta)[:2]
    horizontal=.5*error
    horizontal*=min(1.,.25/max(np.linalg.norm(horizontal),1e-9))
    remaining=tube.goal_progress-progress
    down=min(.8 if tube.roof_distance-progress>10 else .35,max(0.,.5*remaining))
    if np.linalg.norm(error)>.35:down=0.
    if tube.oblique and down>0:
        # 同比例限制前进和下降，保持斜向路径；横向纠偏优先保留速度余量。
        slope=(world_from_anchor@tube.axis_slope)[:2]
        correction=horizontal.copy()
        def admissible(v):return np.linalg.norm(correction+slope*v)<=.25+1e-10
        if not admissible(down):
            lo,hi=0.,down
            for _ in range(24):
                mid=(lo+hi)/2
                if admissible(mid):lo=mid
                else:hi=mid
            down=lo
        horizontal=correction+slope*down
    if remaining<=.3 and np.linalg.norm(error)<=.35:
        return 'INTERIOR_CONFIRM',np.zeros(3),error,progress
    state='DESCEND' if tube.roof_distance-progress>5 else ('CROSS_OPENING' if progress<tube.roof_distance+2 else 'ENTER_INTERIOR')
    return state,np.r_[horizontal,down],error,progress
