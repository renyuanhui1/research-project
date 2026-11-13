import argparse,asyncio,copy,hashlib,json,math,threading,time,sys,traceback
from pathlib import Path
import cv2,commentjson,h5py,numpy as np
from projectairsim import ProjectAirSimClient,World,Drone
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import load_profile, selected_profile, load_mission, scene_bundle
from common.runtime import configure_client_log
from common.sensors import decode_image
from common.start_state import start_state_check
def initialize(argv=None):
 global options, P, BASE, cfg, client, drone, latest, counts, arrivals, PROFILE, mission
 PROFILE=selected_profile(argv);mission=load_mission(PROFILE)
 parser=argparse.ArgumentParser(description="Prepare templates and the controlled experiment start using known geometry")
 parser.add_argument('--profile',default=None)
 parser.add_argument('--mission',type=Path,default=None)
 parser.add_argument('--address',default=None,help='Transport address override, saved in the run profile')
 preliminary,_=parser.parse_known_args(argv)
 if preliminary.mission:mission=load_mission(PROFILE,preliminary.mission)
 parser.add_argument('--out-dir',type=Path,default=PROJECT_ROOT/'runs/approach'/str(time.time_ns()))
 mode=parser.add_mutually_exclusive_group()
 mode.add_argument('--oblique',dest='oblique',action='store_true')
 mode.add_argument('--nadir',dest='oblique',action='store_false')
 parser.set_defaults(oblique=mission['mode']=='oblique')
 for flag,key in [('start-distance','start_distance'),('speed','speed'),('camera-pitch','camera_pitch'),('stop-distance','stop_distance')]:
  parser.add_argument('--'+flag,type=float,default=mission[key])
 parser.add_argument('--target-part',choices=tuple(PROFILE['parts']),default=mission['target_part'])
 parser.add_argument('--common-start',action='store_true',help='Use the shared mission pose and RGB reference for every part')
 parser.add_argument('--reference-dir',type=Path,help='Shared reference directory; created once, then reused without modification')
 parser.add_argument('--validate-only',action='store_true',help="Validate configuration without connecting or writing a run")
 options=parser.parse_args(argv);P=options.out_dir.expanduser().resolve()
 if options.common_start:
  if not options.oblique or 'common_start' not in mission or options.reference_dir is None:parser.error('common-start requires oblique mode, mission common_start and reference-dir')
  options.reference_dir=options.reference_dir.expanduser().resolve()
 if options.address:PROFILE['simulator']['address']=options.address
 if options.target_part in ('wing','tail') and not options.oblique:parser.error('Wing/tail preparation requires --oblique')
 if options.oblique and not (30<=options.start_distance<=300 and 0<options.speed<=5 and 15<=options.camera_pitch<=50):parser.error('Invalid oblique distance/speed/pitch')
 if not 3<=options.stop_distance<=12:parser.error('Stop distance must be between 3 and 12 m')
 BASE=project_path(PROFILE['assets']['preparation_assets'])
 cfg=scene_bundle(PROFILE)
 if options.common_start:
  actor=next(a for a in cfg['actors'] if a['name']==PROFILE['simulator']['drone'])
  start=mission['common_start'];actor['origin']={'xyz':'%s %s -1.0'%tuple(start['position_ned_m'][:2]),'rpy-deg':'0 0 '+str(start['yaw_deg'])}
 robot=next(a for a in cfg['actors'] if a['name']==PROFILE['simulator']['drone'])['robot-config']
 front=next(sensor for sensor in robot['sensors'] if sensor['id']==PROFILE['simulator']['camera'])
 front['origin']['rpy-deg']=f"0 {-options.camera_pitch if options.oblique else -90} 0"
 front['capture-settings']=[cap for cap in front['capture-settings'] if cap['image-type'] in (0,2)]
 for cap in front['capture-settings']:
  if cap['image-type']==2:cap['width']=640;cap['height']=360
 for required in [BASE/'template.json',project_path(PROFILE['map']['root'])/'source/targets.json',project_path(PROFILE['map']['root'])/'meshes'/(PROFILE['target']['actor']+'.npz')]:
  if not required.is_file():raise FileNotFoundError(required)
 if options.validate_only:
  print(json.dumps({'profile':PROFILE['name'],'output_dir':str(P),'target':PROFILE['target']['actor'],'mode':'oblique' if options.oblique else 'nadir','front_pitch_deg':front['origin']['rpy-deg'],'start_distance_m':options.start_distance,'speed_mps':options.speed,'stop_distance_m':options.stop_distance,'coordinate_preparation':True,'common_start':mission.get('common_start') if options.common_start else None},indent=2));return False
 if (P/'runtime-scene.json').exists() or (P/'run.h5').exists():raise RuntimeError('Use an empty output directory: '+str(P))
 P.mkdir(parents=True,exist_ok=True)
 (P/'runtime-scene.json').write_text(json.dumps(cfg,indent=2))
 (P/'scene-profile.json').write_text(json.dumps(PROFILE,ensure_ascii=False,indent=2))
 configure_client_log(P)
 client=ProjectAirSimClient(address=PROFILE['simulator']['address']);drone=None;latest={};counts={};arrivals={}
 return True
def cb(tag):
 def recv(t,m):
  latest[tag]=m;counts[tag]=counts.get(tag,0)+1;arrivals.setdefault(tag,[]).append((time.monotonic(),int(m['time_stamp'])))
 return recv
async def call(label,fn):
 result=[];await asyncio.wait_for(await fn(lambda v:result.append(v)),65)
 pos=drone.get_ground_truth_kinematics()['pose']['position'];print(label,result,pos,flush=True)
async def move(x,y,z,v):
 start=np.array([drone.get_ground_truth_kinematics()['pose']['position'][k] for k in ('x','y','z')]);goal=np.array([x,y,z],float)
 count=max(1,int(np.ceil(np.linalg.norm(goal-start)/(v*25))))
 for i in range(1,count+1):
  point=start+(goal-start)*i/count
  await call('move',lambda callback:drone.move_to_position_async(*map(float,point),v,timeout_sec=55,callback=callback))
  actual=np.array([drone.get_ground_truth_kinematics()['pose']['position'][k] for k in ('x','y','z')])
  if np.linalg.norm(actual-point)>2:raise RuntimeError('Preparation waypoint not reached')
async def hover():await call('hover',lambda callback:drone.hover_async(callback=callback))
def project(points,msg):
 w,x,y,z=[float(msg['rot_'+k]) for k in ('w','x','y','z')]
 rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
 camera=np.array([msg['pos_'+k] for k in ('x','y','z')]);local=(np.asarray(points)-camera)@rotation
 if np.any(local[:,0]<=0):raise RuntimeError('Calibration geometry behind camera')
 focal=msg['width']/2
 return np.column_stack((msg['width']/2+focal*local[:,1]/local[:,0],msg['height']/2+focal*local[:,2]/local[:,0]))
async def prepare_oblique(world):
 pose=world.get_object_pose(PROFILE['target']['actor']);q=pose['rotation']
 yaw=math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))+math.radians(PROFILE['target']['forward_axis_angle_deg'])
 direction=np.array([math.cos(yaw),math.sin(yaw)]);heading=math.atan2(-direction[1],-direction[0])
 def waypoint(distance):return np.r_[reference[:2]+direction*distance,reference[2]-distance*math.tan(math.radians(options.camera_pitch))]
 map_root=project_path(PROFILE['map']['root'])
 expected=next(t for t in json.loads((map_root/'source/targets.json').read_text()) if t['name']==PROFILE['target']['actor'])
 if np.linalg.norm(np.array([pose['translation'][k] for k in ('x','y','z')])-np.array(expected['location_cm'])*[.01,.01,-.01])>.25:raise RuntimeError('Target moved relative to saved map')
 if not np.allclose(world.get_object_scale(PROFILE['target']['actor']),expected['scale'],atol=.01):raise RuntimeError('Target scale changed')
 with np.load(map_root/'meshes'/(PROFILE['target']['actor']+'.npz')) as data:vertices=data['vertices_enu'][:,[1,0,2]].copy();vertices[:,2]*=-1
 reference=np.array(json.loads((BASE/'template.json').read_text())['reference_ned_m'])
 if options.target_part in ('wing','tail'):
  origin=np.array([pose['translation'][k] for k in ('x','y','z')]);lateral=np.array([-direction[1],direction[0]])
  longitudinal=(vertices[:,:2]-origin[:2])@direction;side=(vertices[:,:2]-origin[:2])@lateral
  part=PROFILE['parts'][options.target_part]
  wing=(side>=part['lateral_min_m'])&(longitudinal>part['longitudinal_min_m'])&(longitudinal<part['longitudinal_max_m'])&(-vertices[:,2]>part['min_altitude_m'])
  if 'lateral_max_m' in part:wing &= side<=part['lateral_max_m']
  if 'max_altitude_m' in part:wing &= -vertices[:,2]<part['max_altitude_m']
  if wing.sum()<10:raise RuntimeError('Configured wing/tail region missing from mesh')
  seed=np.r_[origin[:2]+part['seed_longitudinal_m']*direction+part['seed_lateral_m']*lateral,-part['seed_altitude_m']]
  candidates=vertices[wing];reference=candidates[np.linalg.norm(candidates-seed,axis=1).argmin()]
  near=wing&(np.linalg.norm(vertices-reference,axis=1)<=part['radius_m'])
  direction=-direction if part.get('approach_axis')=='rear' else lateral
  heading=math.atan2(-direction[1],-direction[0])
 else:near=np.linalg.norm(vertices-reference,axis=1)<=PROFILE['parts']['nose']['radius_m']
 if near.sum()<10:raise RuntimeError('Target-part reference missing from mesh')
 with np.load(map_root/'grids/context_voxels.npz') as data:obstacles=data['centers_enu_m']
 distances=tuple(d for d in PROFILE['preparation']['calibration_distances_m'] if d!=8 or options.stop_distance<8)+(options.start_distance,)
 route=[np.array(PROFILE['preparation']['cruise_ned_m'])[[1,0,2]]*np.array([1,1,-1])]+[waypoint(d)[[1,0,2]]*np.array([1,1,-1]) for d in distances]
 clearances=[]
 for a,b in zip(route[:-1],route[1:]):
  v=b-a;t=np.clip(((obstacles-a)@v)/(v@v),0,1);clearances.append(float(np.linalg.norm(obstacles-(a+t[:,None]*v),axis=1).min()-5*math.sqrt(3)/2-.6))
 if min(clearances)<5:raise RuntimeError('Calibration route obstructed: '+str(clearances))
 await move(*PROFILE['preparation']['cruise_ned_m'],8);await hover()
 for tag,cam,key in (('front','FrontCamera','scene_camera'),('depth','FrontCamera','depth_camera'),('chase','Chase','scene_camera')):client.subscribe(drone.sensors[cam][key],cb(tag))
 stats=[];checks={};meta={'target_part':options.target_part,'approach_direction_ne':direction.tolist(),'reference_ned_m':reference.tolist(),'reference_role':'offline template construction and evaluation only','route_clearance_m':clearances,'part_mesh_vertex_count':int(near.sum())}
 for index,distance in enumerate(distances):
  point=waypoint(distance);await move(*point,2 if distance==8 else 8);await hover()
  await call('face',lambda callback:drone.rotate_to_yaw_async(heading,timeout_sec=20,callback=callback))
  await hover();await asyncio.sleep(5)
  last=-1;frames=[];deadline=time.monotonic()+10
  while len(frames)<6 and time.monotonic()<deadline:
   msg=latest.get('front')
   if msg is not None and int(msg['time_stamp'])!=last:
    last=int(msg['time_stamp']);rgb=decode_image(msg)[0];frames.append(cv2.resize(rgb,(448,252)));stats.append(frames[-1])
   await asyncio.sleep(.12)
  if len(frames)<6:raise RuntimeError('Missing calibration RGB frames')
  pixels=project(vertices if index in (0,len(distances)-1) else vertices[near],msg);low=np.floor(pixels.min(0)).astype(int);high=np.ceil(pixels.max(0)).astype(int)
  padding=np.maximum(np.rint((high-low)*.05).astype(int),4);low=np.maximum(low-padding,0);high=np.minimum(high+padding,[msg['width'],msg['height']])
  if index==0:
   rgb=decode_image(msg)[0];crop=rgb[low[1]:high[1],low[0]:high[0]];cv2.imwrite(str(P/'aircraft-template.png'),crop[:,:,::-1])
   nose_pixels=project(vertices[near],msg);roi_low=np.clip((nose_pixels.min(0)-low)/(high-low),0,1);roi_high=np.clip((nose_pixels.max(0)-low)/(high-low),0,1)
   roi=[float(roi_low[0]),float(roi_low[1]),float(roi_high[0]),float(roi_high[1])]
   checks['near_rgb']=frames[-1];checks['near_bbox']=np.r_[nose_pixels.min(0)/[msg['width'],msg['height']]*2-1,nose_pixels.max(0)/[msg['width'],msg['height']]*2-1]
   annotated=rgb.copy();cv2.rectangle(annotated,tuple(low),tuple(high),(255,0,0),2)
   nose_low=np.rint(low+roi_low*(high-low)).astype(int);nose_high=np.rint(low+roi_high*(high-low)).astype(int);cv2.rectangle(annotated,tuple(nose_low),tuple(nose_high),(0,255,0),2);cv2.imwrite(str(P/'template-location.png'),annotated[:,:,::-1])
   meta.update(crop_xyxy=[*low.tolist(),*high.tolist()],template_roi=roi)
  if distance==8:
   nose_pixels=project(vertices[near],msg);checks['close_rgb']=frames[-1];checks['close_bbox']=np.r_[nose_pixels.min(0)/[msg['width'],msg['height']]*2-1,nose_pixels.max(0)/[msg['width'],msg['height']]*2-1]
   cv2.imwrite(str(P/'close-validation.png'),decode_image(msg)[0][:,:,::-1])
  if index==len(distances)-1:
   checks['far_rgb']=frames[-1];checks['far_bbox']=np.r_[low/[msg['width'],msg['height']]*2-1,high/[msg['width'],msg['height']]*2-1]
   cv2.imwrite(str(P/'start-front.png'),decode_image(msg)[0][:,:,::-1])
  print('OBLIQUE_CALIBRATION',distance,'m',len(frames),'frames',flush=True)
 with h5py.File(P/'stats-episode.h5','w') as f:f.create_dataset('rgb',data=np.stack(stats),compression='gzip')
 np.savez_compressed(P/'validation-frames.npz',**checks)
 (P/'template.json').write_text(json.dumps(meta,indent=2))
 setup={'profile':PROFILE['name'],'target_actor':PROFILE['target']['actor'],'mode':'oblique','target_part':options.target_part,'start_horizontal_distance_m':options.start_distance,'camera_pitch_deg':options.camera_pitch,'v_forward':options.speed,'v_down':options.speed*math.tan(math.radians(options.camera_pitch)),'slow_dist':35.,'stop_dist':options.stop_distance,'altitude_guard_m':PROFILE['controller']['close_altitude_guard_m'] if options.stop_distance<8 else PROFILE['controller']['altitude_guard_m'],'max_altitude_m':PROFILE['controller']['max_altitude_m'],'template_roi':roi,'roi_switch_dist':PROFILE['controller']['roi_switch_dist_m']}
 if options.stop_distance<8:setup.update(close_range_m=PROFILE['controller']['close_range_m'],close_speed_mps=PROFILE['controller']['close_speed_mps'],depth_guard_m=PROFILE['controller']['depth_guard_m'])
 setup['start_ned_m']=waypoint(options.start_distance).tolist();setup['reference_ned_m']=reference.tolist()
 (P/'controller-setup.json').write_text(json.dumps(setup,indent=2));print('OBLIQUE_READY',setup,flush=True)
def common_part_regions(vertices, pose):
 origin=np.array([pose['translation'][k] for k in ('x','y','z')]);q=pose['rotation']
 yaw=math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))+math.radians(PROFILE['target']['forward_axis_angle_deg'])
 forward=np.array([math.cos(yaw),math.sin(yaw)]);lateral=np.array([-forward[1],forward[0]])
 longitudinal=(vertices[:,:2]-origin[:2])@forward;side=(vertices[:,:2]-origin[:2])@lateral
 result={}
 for name,part in PROFILE['parts'].items():
  if name=='nose':
   reference=np.array(json.loads((BASE/'template.json').read_text())['reference_ned_m']);mask=np.ones(len(vertices),bool)
  else:
   mask=(side>=part['lateral_min_m'])&(longitudinal>part['longitudinal_min_m'])&(longitudinal<part['longitudinal_max_m'])&(-vertices[:,2]>part['min_altitude_m'])
   if 'lateral_max_m' in part:mask &= side<=part['lateral_max_m']
   if 'max_altitude_m' in part:mask &= -vertices[:,2]<part['max_altitude_m']
   seed=np.r_[origin[:2]+part['seed_longitudinal_m']*forward+part['seed_lateral_m']*lateral,-part['seed_altitude_m']]
   candidates=vertices[mask]
   if not len(candidates):raise RuntimeError('Part region missing: '+name)
   reference=candidates[np.linalg.norm(candidates-seed,axis=1).argmin()]
  mask &= np.linalg.norm(vertices-reference,axis=1)<=part['radius_m']
  if mask.sum()<10:raise RuntimeError('Part region too small: '+name)
  result[name]=(reference,mask)
 return result

async def settle_common_start():
 expected=mission['common_start'];goal=np.array(expected['position_ned_m']);yaw=math.radians(expected['yaw_deg'])
 await move(*goal,2);await hover()
 await call('common_yaw',lambda callback:drone.rotate_to_yaw_async(yaw,timeout_sec=20,margin=math.radians(.2),callback=callback))
 deadline=time.monotonic()+30;stable=0
 while time.monotonic()<deadline:
  kin=drone.get_ground_truth_kinematics();check=start_state_check(kin,expected)
  stable=stable+1 if check['passed'] and check['position_error_m']<.15 else 0
  if stable>=5:break
  pos=np.array([kin['pose']['position'][k] for k in ('x','y','z')]);velocity=np.clip((goal-pos)*.7,-.5,.5)
  await asyncio.wait_for(await drone.move_by_velocity_async(*map(float,velocity),duration=.2,yaw_is_rate=False,yaw=yaw),3)
 else:raise RuntimeError('Common start did not settle: '+json.dumps(check))
 await hover();await asyncio.sleep(2)
 check=start_state_check(drone.get_ground_truth_kinematics(),expected)
 (P/'start-state.json').write_text(json.dumps(check,indent=2))
 if not check['passed']:raise RuntimeError('Common start outside tolerance: '+json.dumps(check))
 print('COMMON_START',json.dumps(check),flush=True)

async def prepare_common(world):
 shared=options.reference_dir;manifest=shared/'template.json'
 pose=world.get_object_pose(PROFILE['target']['actor']);map_root=project_path(PROFILE['map']['root'])
 expected=next(t for t in json.loads((map_root/'source/targets.json').read_text()) if t['name']==PROFILE['target']['actor'])
 if np.linalg.norm(np.array([pose['translation'][k] for k in ('x','y','z')])-np.array(expected['location_cm'])*[.01,.01,-.01])>.25:raise RuntimeError('Target moved relative to saved map')
 if not np.allclose(world.get_object_scale(PROFILE['target']['actor']),expected['scale'],atol=.01):raise RuntimeError('Target scale changed')
 await settle_common_start()
 for tag,cam,key in (('front','FrontCamera','scene_camera'),('depth','FrontCamera','depth_camera'),('chase','Chase','scene_camera')):client.subscribe(drone.sensors[cam][key],cb(tag))
 await asyncio.sleep(3)
 frames=[];last=-1;deadline=time.monotonic()+10
 while len(frames)<6 and time.monotonic()<deadline:
  msg=latest.get('front')
  if msg is not None and int(msg['time_stamp'])!=last:
   last=int(msg['time_stamp']);frames.append(decode_image(msg)[0])
  await asyncio.sleep(.15)
 if len(frames)<6:raise RuntimeError('Missing common-start RGB frames')
 with np.load(map_root/'meshes'/(PROFILE['target']['actor']+'.npz')) as data:vertices=data['vertices_enu'][:,[1,0,2]].copy();vertices[:,2]*=-1
 regions=common_part_regions(vertices,pose);pixels=project(vertices,msg);lo=pixels.min(0);hi=pixels.max(0)
 checks={'start_rgb':frames[-1]}
 for name,(_,mask) in regions.items():
  uv=project(vertices[mask],msg);box=np.r_[uv.min(0),uv.max(0)]
  if np.any(box[:2]<0) or np.any(box[2:]>[msg['width'],msg['height']]):raise RuntimeError('Part outside common initial image: '+name)
  checks[name+'_bbox']=box/[msg['width'],msg['height'],msg['width'],msg['height']]*2-1
 if manifest.exists():
  meta=json.loads(manifest.read_text())
  if meta['target_actor']!=PROFILE['target']['actor'] or meta['common_start']!=mission['common_start'] or meta['camera_pitch_deg']!=options.camera_pitch:raise RuntimeError('Shared reference configuration mismatch')
  if meta['runtime_scene_sha256']!=hashlib.sha256((P/'runtime-scene.json').read_bytes()).hexdigest():raise RuntimeError('Shared reference runtime scene mismatch')
 else:
  if shared.exists() and any(shared.iterdir()):raise RuntimeError('Incomplete shared reference; preserve it and choose a new directory')
  shared.mkdir(parents=True,exist_ok=True)
  pad=np.maximum((hi-lo)*.05,4);low=np.floor(np.maximum(lo-pad,0)).astype(int);high=np.ceil(np.minimum(hi+pad,[msg['width'],msg['height']])).astype(int)
  crop=frames[-1][low[1]:high[1],low[0]:high[0]];cv2.imwrite(str(shared/'aircraft-template.png'),crop[:,:,::-1]);cv2.imwrite(str(shared/'reference-front.png'),frames[-1][:,:,::-1])
  with h5py.File(shared/'stats-episode.h5','w') as f:f.create_dataset('rgb',data=np.stack([cv2.resize(im,(448,252)) for im in frames]),compression='gzip')
  meta={'target_actor':PROFILE['target']['actor'],'common_start':mission['common_start'],'camera_pitch_deg':options.camera_pitch,'reference_role':'fixed development reference; geometry for preparation and evaluation only','crop_xyxy':np.r_[low,high].tolist(),'runtime_scene_sha256':hashlib.sha256((P/'runtime-scene.json').read_bytes()).hexdigest(),'parts':{},'reference_camera_metadata':{k:v for k,v in msg.items() if k!='data'}}
  annotated=frames[-1][:,:,::-1].copy()
  for name,(reference,mask) in regions.items():
   uv=project(vertices[mask],msg);a,b=uv.min(0),uv.max(0);roi=np.r_[np.clip((a-low)/(high-low),0,1),np.clip((b-low)/(high-low),0,1)]
   meta['parts'][name]={'reference_ned_m':reference.tolist(),'template_roi':roi.tolist(),'part_mesh_vertex_count':int(mask.sum())}
   cv2.rectangle(annotated,tuple(a.astype(int)),tuple(b.astype(int)),(0,255,255),2);cv2.putText(annotated,name,tuple(a.astype(int)-[0,5]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
  cv2.imwrite(str(shared/'template-location.png'),annotated)
  np.savez_compressed(shared/'validation-frames.npz',**checks)
  manifest.write_text(json.dumps(meta,indent=2))
 hashes={name:hashlib.sha256((shared/name).read_bytes()).hexdigest() for name in ('aircraft-template.png','stats-episode.h5','template.json')}
 selected=meta['parts'][options.target_part]
 (P/'template.json').write_text(json.dumps(dict(selected,target_part=options.target_part,shared_reference=str(shared)),indent=2))
 np.savez_compressed(P/'validation-frames.npz',**checks);cv2.imwrite(str(P/'start-front.png'),frames[-1][:,:,::-1])
 setup={'profile':PROFILE['name'],'target_actor':PROFILE['target']['actor'],'mode':'oblique','target_part':options.target_part,'common_start':mission['common_start'],'shared_reference':str(shared),'shared_asset_sha256':hashes,'reference_ned_m':selected['reference_ned_m'],'start_ned_m':mission['common_start']['position_ned_m'],'camera_pitch_deg':options.camera_pitch,'v_forward':options.speed,'v_down':options.speed*math.tan(math.radians(options.camera_pitch)),'slow_dist':35.,'stop_dist':options.stop_distance,'altitude_guard_m':PROFILE['controller']['close_altitude_guard_m'],'max_altitude_m':PROFILE['controller']['max_altitude_m'],'template_roi':selected['template_roi'],'roi_switch_dist':0,'close_range_m':PROFILE['controller']['close_range_m'],'close_speed_mps':PROFILE['controller']['close_speed_mps'],'depth_guard_m':PROFILE['controller']['depth_guard_m']}
 (P/'controller-setup.json').write_text(json.dumps(setup,indent=2));print('COMMON_READY',json.dumps(setup),flush=True)

async def run_preparation():
 global drone
 client.connect();client.socket_services.recv_timeout=65000;client.socket_topics.recv_buffer_size=32
 w=World.__new__(World);w.client=client;w.sim_config_path=str(project_path(PROFILE['simulator']['scene']).parent);w.sim_instance_idx=-1
 w.load_scene(cfg,delay_after_load_sec=1)
 print('TARGET',w.get_object_pose(PROFILE['target']['actor']),flush=True)
 drone=Drone(client,w,PROFILE['simulator']['drone'])
 if not drone.is_api_control_enabled():assert drone.enable_api_control()
 assert drone.arm()
 await call('takeoff',lambda callback:drone.takeoff_async(callback=callback))
 if options.common_start:
  await prepare_common(w)
  return
 if options.oblique:
  await prepare_oblique(w)
  return
 await move(*PROFILE['preparation']['nadir_cruise_ned_m'],8);await hover()
 await move(*PROFILE['preparation']['nadir_start_ned_m'],10);await hover();await asyncio.sleep(2)
 await move(*PROFILE['preparation']['nadir_start_ned_m'],1);await hover();await asyncio.sleep(2)
 for tag,cam,key in (('front','FrontCamera','scene_camera'),('depth','FrontCamera','depth_camera'),('chase','Chase','scene_camera')):
  client.subscribe(drone.sensors[cam][key],cb(tag))
 await asyncio.sleep(3);start=time.monotonic();before=counts.copy();await asyncio.sleep(8);dt=time.monotonic()-start
 r={'input_fps':{k:(v-before.get(k,0))/dt for k,v in counts.items()},'state':drone.get_ground_truth_kinematics(),'width_height':{}}
 for tag in ('front','chase'):
  if tag not in latest:raise RuntimeError('missing '+tag)
  im=decode_image(latest[tag])[0];cv2.imwrite(str(P/(tag+'-preview.png')),im[:,:,::-1]);r['width_height'][tag]=[im.shape[1],im.shape[0]]
 setup={'profile':PROFILE['name'],'target_actor':PROFILE['target']['actor'],'mode':'nadir','target_part':options.target_part,'stop_dist':options.stop_distance,'altitude_guard_m':PROFILE['controller']['nadir_altitude_guard_m'],'max_altitude_m':PROFILE['controller']['max_altitude_m']}
 (P/'controller-setup.json').write_text(json.dumps(setup,indent=2))
 (P/'preflight.json').write_text(json.dumps(r,indent=2));print('PROBE',r['input_fps'],r['width_height'],flush=True)
def main(argv=None):
 if not initialize(argv):return 0
 try:asyncio.run(run_preparation())
 except BaseException:
  traceback.print_exc();return 1
 finally:
  if drone:
   try:asyncio.run(hover())
   except Exception:pass
  client.disconnect()
 return 0

if __name__=='__main__':raise SystemExit(main())
