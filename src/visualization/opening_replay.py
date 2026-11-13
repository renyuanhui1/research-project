"""将保存的洞口匹配、坐标入室或实时对准记录整理为RViz回放；不连接仿真。"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path as FilePath

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

ROOT=FilePath(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from perception.depth_map import _quat_to_R


def enu(position):
    return np.array([position['y'],position['x'],-position['z']],float)


def projected_aperture(geometry, metadata):
    aperture=geometry.get('proposed_aperture_point_rgb_xy')
    if aperture is None:return None
    m=metadata
    u=(aperture[0]+.5-m['width']/2)/(m['width']/2)
    v=(aperture[1]+.5-m['height']/2)/(m['width']/2)
    ray=np.array([1,u,v]);rotation=_quat_to_R([m['rot_'+k] for k in ('w','x','y','z')])
    denominator=float(rotation[2]@ray) if geometry.get('depth_layer_axis')=='gravity' else 1.
    if denominator<=.1:return None
    local=ray*geometry['near_layer_depth_m']/denominator
    ned=rotation@local+np.array([m['pos_'+k] for k in ('x','y','z')])
    return [float(ned[1]),float(ned[0]),float(-ned[2])]


def rgbd_points(folder,frame,stride=4):
    depth=np.load(folder/frame['depth'])
    rgb=cv2.imread(str(folder/frame['image']))[:,:,::-1]
    h,w=depth.shape
    rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    yy,xx=np.mgrid[0:h:stride,0:w:stride]
    values=depth[::stride,::stride]
    valid=np.isfinite(values)&(values>.1)&(values<150)
    u=(xx[valid]+.5-w/2)/(w/2)
    v=(yy[valid]+.5-h/2)/(w/2)
    directions=np.column_stack([np.ones(u.size),u,v])
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    local=directions*values[valid,None]
    meta=frame['camera_metadata']
    q=[meta['rot_'+key] for key in ('w','x','y','z')]
    position=np.array([meta['pos_'+key] for key in ('x','y','z')])
    ned=local@_quat_to_R(q).T+position
    points=ned[:,[1,0,2]].copy();points[:,2]*=-1
    colors=rgb[yy[valid],xx[valid]]
    return points,colors


def prepare(options):
    capture=(options.flight_dir if options.mode=='alignment' else options.capture_dir).resolve()
    capture_report=json.loads((capture/'report.json').read_text())
    if capture_report['status']!='completed':raise RuntimeError('Incomplete RGB-D source')
    rows=[]
    if options.mode=='alignment':
        if not capture_report.get('execute') or not capture_report.get('flight_commands_sent'):
            raise RuntimeError('Alignment replay requires an executed control recording')
        rows=[json.loads(line) for line in (capture/'signals.jsonl').read_text().splitlines() if line.strip()]
        if not rows or len(rows)!=capture_report['rows']:raise RuntimeError('Incomplete alignment signals')
        for key in ('rgb_metadata','depth_metadata'):
            stamps=[row[key]['time_stamp'] for row in rows]
            if any(b<=a for a,b in zip(stamps,stamps[1:])):raise RuntimeError('Non-independent source frames')
        times=[row['time_s'] for row in rows]
        if not np.isfinite(times).all() or any(b<=a for a,b in zip(times,times[1:])):
            raise RuntimeError('Invalid control playback times')
        # 地图每10帧取一组并包含末帧；回放保留全部控制观测。
        map_indices=sorted(set(range(0,len(rows),10))|{len(rows)-1})
        source_frames=[{'image':rows[i]['image'],'depth':rows[i]['depth'],
                        'camera_metadata':rows[i]['depth_metadata']} for i in map_indices]
    else:
        source_frames=capture_report['frames']
    all_points=[];all_colors=[]
    for frame in source_frames:
        points,colors=rgbd_points(capture,frame)
        all_points.append(points);all_colors.append(colors)
    points=np.concatenate(all_points);colors=np.concatenate(all_colors)
    _,idx=np.unique(np.floor(points/.35).astype(np.int32),axis=0,return_index=True)
    points=points[idx];colors=colors[idx]
    items=[];trajectory=[]
    if options.mode=='matching':
        matches=json.loads((options.match_dir/'report.json').read_text())
        by_name={row['name']:row for row in matches['results']}
        for frame in capture_report['frames']:
            row=by_name[frame['name']]
            geometry=row['depth_constrained_dino']
            target=projected_aperture(geometry,frame['camera_metadata'])
            items.append({'name':frame['name'],'pose':frame['pose'],
                'image':str((options.match_dir/(frame['name']+'-geometry.png')).resolve()),
                'caption':'OFFLINE MATCHING | discrete capture pose | not a DINO flight',
                'projected_aperture_enu_m':target,'timestamp_ns':frame['camera_metadata']['time_stamp']})
        out=options.out_dir or options.match_dir/'rviz'
    elif options.mode=='alignment':
        trajectory=[row['pose'] for row in rows]
        feature_items=None
        if options.feature_view_dir:
            import hashlib
            feature_report=json.loads((options.feature_view_dir/'report.json').read_text())
            if (feature_report['status']!='completed' or
                feature_report['source_signals_sha256']!=hashlib.sha256((capture/'signals.jsonl').read_bytes()).hexdigest()):
                raise RuntimeError('Feature view does not match this control recording')
            feature_items={item['step']:item for item in feature_report['items']}
        for row in rows:
            error=float(np.linalg.norm(row['error_ne_m']))
            entry=row.get('entry')
            label='RECORDED VISUAL ENTRY' if capture_report.get('scope')=='visual_approach_and_entry' else 'RECORDED VISUAL ALIGNMENT'
            name='%03d %s'%(row['step'],row['state'])
            if entry:name+=' | roof remaining %.1fm'%entry['roof_remaining_m']
            items.append({'name':'%03d %s'%(row['step'],row['state']),'pose':row['pose'],
                'image':str((capture/(row['image'] if entry else row['diagnostic'])).resolve()),
                'caption':label+' | t %.1fs | horizontal error %.2fm'%(row['time_s'],error),
                'timestamp_ns':row['rgb_metadata']['time_stamp'],
                'depth_timestamp_ns':row['depth_metadata']['time_stamp'],
                'playback_time_s':row['time_s']-rows[0]['time_s'],
                'projected_aperture_enu_m':projected_aperture(row['match'],row['rgb_metadata'])})
            items[-1]['name']=name
            if feature_items is not None:
                item=feature_items[row['step']]
                if item['timestamp_ns']!=row['rgb_metadata']['time_stamp']:raise RuntimeError('Feature frame timestamp mismatch')
                items[-1]['image']=str((options.feature_view_dir/item['image']).resolve())
                items[-1]['caption']=item['caption']
        out=options.out_dir or capture/'rviz'
    else:
        flight_report=json.loads((options.flight_dir/'report.json').read_text())
        flight=json.loads((options.flight_dir/'flight.json').read_text())
        snapshots=sorted(flight_report['captures'],key=lambda c:c['cameras']['forward']['metadata']['time_stamp'])
        trajectory=[row['state']['pose'] for row in flight]
        for row in flight:
            stamp=row['state']['time_stamp']
            photos=[c for c in snapshots if c['cameras']['forward']['metadata']['time_stamp']<=stamp]
            if not photos:continue
            photo=photos[-1];m=photo['cameras']['forward']['metadata']
            items.append({'name':row['phase'],'pose':row['state']['pose'],
                'image':str((options.flight_dir/photo['cameras']['forward']['image']).resolve()),
                'caption':'COORDINATE ENTRY | recorded samples | held sparse photo; age %.1fs'%((stamp-m['time_stamp'])/1e9),
                'timestamp_ns':stamp,'projected_aperture_enu_m':None})
        # 最终照片晚于末条运动采样，附加真实最终悬停位姿，不插值轨迹或照片。
        photo=snapshots[-1]
        items.append({'name':'final recorded hover','pose':flight_report['final_kinematics']['pose'],
            'image':str((options.flight_dir/photo['cameras']['forward']['image']).resolve()),
            'caption':'COORDINATE ENTRY | final recorded interior snapshot',
            'timestamp_ns':photo['cameras']['forward']['metadata']['time_stamp'],'projected_aperture_enu_m':None})
        out=options.out_dir or options.flight_dir/'rviz'
    if not items:raise RuntimeError('No replay items')
    for item in items:
        if not FilePath(item['image']).is_file():raise FileNotFoundError(item['image'])
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'rgbd-map.npz',points_enu_m=points.astype(np.float32),rgb=colors)
    manifest={'mode':options.mode,'frame':'world','coordinates':'ENU=(E,N,-D)',
        'simulation_connection':False,'timing':('recorded control wall-time offsets; speed scales playback' if options.mode=='alignment' else 'indexed playback; does not reproduce recorded time gaps'),
        'background_source':str(capture),'background_method':'recorded perspective RGB-D + per-image camera extrinsics; 90deg HFOV; 0.35m deduplication',
        'background_points':len(points),'items':items,'recorded_trajectory':trajectory,
        'background_frame_count':len(source_frames),
        'control_scope':capture_report.get('scope'),
        'feature_view_recomputed_after_flight':bool(options.feature_view_dir),
        'image_semantics':('recorded live-control diagnostics, one image per actual observation' if options.mode=='alignment' else ('offline diagnostics' if options.mode=='matching' else 'sparse recorded photos held between captures')),
        'projected_aperture_is_3d_passage_proof':False}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('REPLAY_PACKAGE',str(out),'items',len(items),'points',len(points),flush=True)
    return manifest,points,colors


class OpeningReplay(Node):
    def __init__(self,manifest,points,colors,rate,speed=1.,image_window=False):
        super().__init__('opening_replay')
        self.manifest=manifest;self.items=manifest['items'];self.index=0
        self.recorded_timing=rate is None;self.speed=speed;self.epoch=time.monotonic()
        self.image_window=image_window
        if image_window:
            cv2.namedWindow('DINO response and control',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('DINO response and control',960,592)
            cv2.imshow('DINO response and control',np.zeros((592,960,3),np.uint8));cv2.waitKey(1)
        sticky=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub=self.create_publisher(PointCloud2,'/opening/map',sticky)
        self.path_pub=self.create_publisher(Path,'/opening/path',sticky)
        self.marker_pub=self.create_publisher(MarkerArray,'/opening/markers',10)
        self.image_pub=self.create_publisher(Image,'/opening/view',10)
        msg=PointCloud2();msg.header.frame_id='world';msg.header.stamp=self.get_clock().now().to_msg()
        msg.height=1;msg.width=len(points);msg.is_dense=True;msg.is_bigendian=False
        msg.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32 if i<3 else PointField.UINT32,count=1) for i,name in enumerate(['x','y','z','rgb'])]
        cloud=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
        for i,name in enumerate(['x','y','z']):cloud[name]=points[:,i]
        packed=colors.astype(np.uint32);cloud['rgb']=(packed[:,0]<<16)|(packed[:,1]<<8)|packed[:,2]
        msg.point_step=16;msg.row_step=16*len(points);msg.data=cloud.tobytes();self.map_pub.publish(msg)
        path=Path();path.header=msg.header
        for pose in manifest['recorded_trajectory']:
            p=PoseStamped();p.header=path.header;p.pose.position=Point(x=float(pose['position']['y']),y=float(pose['position']['x']),z=float(-pose['position']['z']));p.pose.orientation.w=1.;path.poses.append(p)
        self.path_pub.publish(path)
        self.timer=self.create_timer(.02 if self.recorded_timing else 1/rate,self.tick)
        self.get_logger().info('Mode=%s; %d items; %d actual trajectory samples'%(manifest['mode'],len(self.items),len(path.poses)))
        self.tick()

    def marker(self,idx,kind,stamp):
        m=Marker();m.header.frame_id='world';m.header.stamp=stamp;m.ns='opening';m.id=idx;m.type=kind;m.action=Marker.ADD;m.pose.orientation.w=1.
        return m

    def tick(self):
        item=self.items[self.index]
        if self.recorded_timing and time.monotonic()-self.epoch<item['playback_time_s']/self.speed:return
        stamp=self.get_clock().now().to_msg()
        pos=enu(item['pose']['position']);arr=MarkerArray()
        clear=self.marker(0,Marker.CUBE,stamp);clear.action=Marker.DELETEALL;clear.ns='';arr.markers.append(clear)
        if self.manifest['mode']=='matching':
            sites=self.marker(1,Marker.SPHERE_LIST,stamp);sites.scale.x=sites.scale.y=sites.scale.z=.8
            sites.color.r=.6;sites.color.g=.6;sites.color.b=.6;sites.color.a=1.
            sites.points=[Point(x=float(v[0]),y=float(v[1]),z=float(v[2])) for v in [enu(x['pose']['position']) for x in self.items]]
            arr.markers.append(sites)
        current=self.marker(2,Marker.SPHERE,stamp);current.pose.position=Point(x=float(pos[0]),y=float(pos[1]),z=float(pos[2]));current.scale.x=current.scale.y=current.scale.z=1.5
        current.color.r=.1;current.color.g=.5;current.color.b=1.;current.color.a=1.;arr.markers.append(current)
        target=item['projected_aperture_enu_m']
        if target is not None:
            point=self.marker(3,Marker.SPHERE,stamp);point.pose.position=Point(x=target[0],y=target[1],z=target[2]);point.scale.x=point.scale.y=point.scale.z=.8
            point.color.g=point.color.b=point.color.a=1.;arr.markers.append(point)
            line=self.marker(4,Marker.LINE_STRIP,stamp);line.scale.x=.12;line.color=point.color;line.points=[current.pose.position,point.pose.position];arr.markers.append(line)
        self.marker_pub.publish(arr)
        bgr=cv2.imread(item['image']);bgr=cv2.resize(bgr,(960,540))
        bgr=cv2.copyMakeBorder(bgr,52,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(bgr,item['caption'],(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(bgr,'%s | sample %d/%d'%(item['name'],self.index+1,len(self.items)),(10,42),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        msg=Image();msg.header.frame_id='world';msg.header.stamp=stamp;msg.height,msg.width=bgr.shape[:2];msg.encoding='bgr8';msg.step=msg.width*3;msg.data=bgr.tobytes();self.image_pub.publish(msg)
        if self.image_window:
            if cv2.getWindowProperty('DINO response and control',cv2.WND_PROP_VISIBLE)<1:
                self.image_window=False
            else:
                cv2.imshow('DINO response and control',bgr);cv2.waitKey(1)
        if self.index==0:self.get_logger().info('Replay cycle started: '+self.manifest['mode'])
        self.index=(self.index+1)%len(self.items)
        if self.recorded_timing and self.index==0:self.epoch=time.monotonic()+2.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['matching','entry','alignment'],default='matching')
    parser.add_argument('--capture-dir',type=FilePath,default=ROOT/'data/captures/hangar_2019/20261112-development-views')
    parser.add_argument('--match-dir',type=FilePath,default=ROOT/'runs/opening-validation/20261112-hangar2019-geometry')
    parser.add_argument('--flight-dir',type=FilePath,default=ROOT/'runs/indoor/20261111-160335-hangar2019')
    parser.add_argument('--out-dir',type=FilePath)
    parser.add_argument('--rate',type=float,help='Indexed samples per second; not sensor FPS')
    parser.add_argument('--speed',type=float,default=1.,help='Time multiplier for recorded alignment playback')
    parser.add_argument('--feature-view-dir',type=FilePath,help='Verified offline feature reconstruction for this flight')
    parser.add_argument('--image-window',action='store_true',help='Show the same control image in a separate large window')
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--rviz',action='store_true')
    options=parser.parse_args()
    rate=options.rate if options.rate is not None else ({'matching':.5,'entry':8.}.get(options.mode))
    if rate is not None and (not np.isfinite(rate) or rate<=0):parser.error('--rate must be positive and finite')
    if not np.isfinite(options.speed) or options.speed<=0:parser.error('--speed must be positive and finite')
    manifest,points,colors=prepare(options)
    if options.export_only:return
    if options.image_window:
        os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None);os.environ.pop('QT_QPA_FONTDIR',None)
        os.environ['QT_QPA_PLATFORM']='xcb'
    rclpy.init();node=OpeningReplay(manifest,points,colors,rate,options.speed,options.image_window);rviz=None
    if options.rviz:
        env=os.environ.copy();env.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None);env.pop('QT_QPA_FONTDIR',None);env['QT_QPA_PLATFORM']='xcb'
        config=ROOT/'configs/rviz/opening.rviz'
        if options.mode=='alignment':
            import yaml
            data=yaml.safe_load(config.read_text())
            view=data['Visualization Manager']['Views']['Current']
            view.update(Distance=75,Pitch=.9)
            center=np.mean([enu(item['pose']['position']) for item in manifest['items']],axis=0)
            targets=[item['projected_aperture_enu_m'] for item in manifest['items'] if item['projected_aperture_enu_m'] is not None]
            if targets:center[2]=(center[2]+np.median(np.array(targets)[:,2]))/2
            view['Focal Point']=dict(zip(('X','Y','Z'),map(float,center)))
            config=(options.out_dir or options.flight_dir/'rviz')/'opening.rviz'
            config.write_text(yaml.safe_dump(data,sort_keys=False))
        rviz=subprocess.Popen(['rviz2','-d',str(config)],env=env)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    except RCLError:
        if rclpy.ok():raise
    finally:
        if rviz is not None and rviz.poll() is None:rviz.terminate()
        node.destroy_node()
        if options.image_window:cv2.destroyAllWindows()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
