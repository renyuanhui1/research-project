"""按本次部位配置执行俯视或斜俯视抵近，实时导出RViz帧并录制双路1080p。"""
import argparse,asyncio,collections,csv,hashlib,json,math,sys,threading,time,traceback
from pathlib import Path
import cv2,h5py,numpy as np,torch
from projectairsim import ProjectAirSimClient,World,Drone
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, SIMULATION_DIR, project_path
from common.profiles import load_profile, selected_profile, load_mission, scene_bundle
from common.runtime import configure_client_log
from control import servo_closed_loop as sc
from common.start_state import start_state_check
from perception.part_matching import CroppedPartMatcher
def initialize(argv=None):
 global OUT, VDIR, BASE, setup, target_part, oblique, assets, cfg, meta, ref, args, PROFILE, options, common_start
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--profile',default=None)
 parser.add_argument('--out-dir',type=Path,required=True,help="Prepared run directory")
 parser.add_argument('--validate-only',action='store_true',help="Run recorded matching preflight without simulator connection")
 parser.add_argument('--preflight-selected-part-only',action='store_true',help='Common-start experiment: require reference and start matching for the selected part; retain all part diagnostics')
 parser.add_argument('--thin-roi-refinement',action='store_true',help='Use overlap-weighted context and bounded image registration for empty center-selected reference ROIs')
 parser.add_argument('--max-duration',type=float,default=600.,help='Control time limit in seconds; stop and hover on expiry')
 options=parser.parse_args(argv);OUT=options.out_dir.expanduser().resolve();VDIR=OUT/'viz_dump'
 if not math.isfinite(options.max_duration) or not 1<=options.max_duration<=1800:parser.error('max-duration must be finite and within [1,1800]')
 snapshot=OUT/'scene-profile.json'
 PROFILE=load_profile(options.profile) if options.profile or not snapshot.exists() else json.loads(snapshot.read_text())
 if any((OUT/name).exists() for name in ('run.h5','report.json','front.mp4','chase.mp4')) and not options.validate_only:
  raise RuntimeError('Output directory already contains a run: '+str(OUT))
 if not options.validate_only and any(VDIR.glob('step_*.npz')):raise RuntimeError('Visualization directory must be empty')
 BASE=project_path(PROFILE['assets']['preparation_assets'])
 setup=json.loads((OUT/'controller-setup.json').read_text())
 if setup.get('target_actor',PROFILE['target']['actor'])!=PROFILE['target']['actor']:raise RuntimeError('Run target disagrees with scene profile')
 target_part=setup.get('target_part','nose');oblique=setup.get('mode')=='oblique';common_start='common_start' in setup;assets=Path(setup['shared_reference']) if common_start else (OUT if oblique else BASE)
 if options.preflight_selected_part_only and not common_start:parser.error('selected-part preflight requires a common-start experiment')
 if options.thin_roi_refinement and not common_start:parser.error('thin ROI refinement requires a common-start experiment')
 cfg=json.loads((OUT/'runtime-scene.json').read_text());meta=json.loads((assets/'template.json').read_text());ref=np.array(meta['reference_ned_m']) if meta.get('reference_ned_m') is not None else None
 if common_start:
  for name,expected in setup['shared_asset_sha256'].items():
   if hashlib.sha256((assets/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Shared reference changed: '+name)
  ref=np.array(meta['parts'][target_part]['reference_ned_m'])
 camera=next(s for a in cfg['actors'] if a['name']==PROFILE['simulator']['drone'] for s in a['robot-config']['sensors'] if s['id']==PROFILE['simulator']['camera'])
 if abs(float(camera['origin']['rpy-deg'].split()[1])-(-setup['camera_pitch_deg'] if oblique else -90.))>.1:
  raise RuntimeError('Camera pitch and controller mode disagree; use the same --out-dir for preparation and control')
 controller_argv=['--address',PROFILE['simulator']['address'],'--template',str(assets/('aircraft-template.png' if oblique else 'nose-template.png')),'--stats-episode',str(assets/'stats-episode.h5'),'--repo-dir',str(project_path(PROFILE['assets']['dino_repo'])),'--dino-weights',str(project_path(PROFILE['assets']['dino_weights'])),'--no-seg-depth','--image-size','448','--target-softmax-temp','0.005','--stop-dist',str(setup.get('stop_dist',12)),'--slow-dist',str(setup.get('slow_dist',25)),'--v-forward',str(setup.get('v_forward',.5)),'--v-down',str(setup.get('v_down',.5)),'--k-xy','1','--center-gate','0.25','--lost-forward','0','--control-min-peak','0.2','--dist-min-peak','0.2','--target',PROFILE['target']['actor'],'--max-steps','0']
 if oblique and not common_start:controller_argv+=['--template-roi',*map(str,setup['template_roi']),'--roi-switch-dist',str(setup['roi_switch_dist'])]
 if not oblique:controller_argv+=['--nadir']
 args=sc.parse_args(controller_argv)
 if options.thin_roi_refinement:
  args.thin_roi_refinement=True;args.local_refinement_min_score=.85;args.local_refinement_min_margin=.15
 if not options.validate_only:VDIR.mkdir(parents=True,exist_ok=True);configure_client_log(OUT)
class SmallBatchFingerprint(sc.Fingerprint):
 def _encode(self,rgb,bs=2):return super()._encode(rgb,bs=2)
def common_preflight_passed(validation,selected_part=None,min_peak=.2):
 checks=[v for v in validation if selected_part is None or v['part']==selected_part]
 return bool(checks) and {v['frame'] for v in checks}=={'reference','start'} and all(np.isfinite(v['peak']) and v['peak']>=min_peak and v['inside'] for v in checks)
def refined_depth_half_width(part_wh,rgb_wh,depth_wh):
 footprint=np.asarray(part_wh,float)*np.asarray(depth_wh)/np.asarray(rgb_wh)
 return max(0,min(4,int(np.floor(float(footprint.min())/4))))
def matching_preflight():
 global fp
 torch.set_num_threads(2);cv2.setNumThreads(1)
 fp=CroppedPartMatcher(args,'cuda',{k:v['template_roi'] for k,v in meta['parts'].items()},target_part) if common_start else SmallBatchFingerprint(args,'cuda')
 torch.cuda.empty_cache()
 if common_start:
  validation=[]
  for source,path in [('reference',assets/'validation-frames.npz'),('start',OUT/'validation-frames.npz')]:
   with np.load(path) as checks:
    rgb=checks['start_rgb'];results=fp.query_all(rgb);annotated=rgb[:,:,::-1].copy()
    for name,(cx,cy,mass,peak,sim) in results.items():
     box=checks[name+'_bbox'];tolerance=0. if fp.reference_sampling[name]=='overlap_context_local_registration' else .02
     inside=bool(box[0]-tolerance<=cx<=box[2]+tolerance and box[1]-tolerance<=cy<=box[3]+tolerance)
     diagnostic=fp.last_diagnostics.get(name,{})
     validation.append({'frame':source,'part':name,'cx':cx,'cy':cy,'peak':peak,'expected_bbox':box.tolist(),'bbox_tolerance':tolerance,'inside':inside,'crop_xyxy':fp.last_diagnostics['crop_xyxy'],'reference_patch_count':fp.reference_patch_counts[name],'reference_center_patch_count':fp.reference_center_patch_counts[name],'reference_sampling':fp.reference_sampling[name],'refinement':diagnostic.get('refinement'),'failure_reason':'no_reference_patches' if fp.reference_patch_counts[name]==0 else diagnostic.get('failure_reason')})
     if peak>=args.control_min_peak:
      point=(round((cx+1)*rgb.shape[1]/2),round((cy+1)*rgb.shape[0]/2));color=(0,255,0) if inside else (0,0,255);cv2.drawMarker(annotated,point,color,cv2.MARKER_CROSS,20,2);cv2.putText(annotated,name,(point[0]+8,point[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
     else:cv2.putText(annotated,name+': unavailable',(20,35+25*list(results).index(name)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    if not options.validate_only:cv2.imwrite(str(OUT/('matching-'+source+'.png')),annotated)
  if not options.validate_only:(OUT/'matching-preflight.json').write_text(json.dumps(validation,indent=2))
  print('MATCHING_PREFLIGHT',json.dumps(validation),flush=True)
  selected=target_part if options.preflight_selected_part_only else None
  print('PREFLIGHT_SCOPE',selected or 'all_parts',flush=True)
  if not common_preflight_passed(validation,selected,args.control_min_peak):raise RuntimeError('Common-start part matching failed; no flight commands sent')
  return
 if oblique:
  validation=[]
  with np.load(OUT/'validation-frames.npz') as checks:
   for name,roi in [('far',False),('near',True)]+([('close',True)] if 'close_rgb' in checks.files else []):
    cx,cy,mass,peak,sim=fp.query(checks[name+'_rgb'],use_roi=roi);box=checks[name+'_bbox'];tolerance=.02 if roi and target_part in ('wing','tail') else .1;inside=bool(box[0]-tolerance<=cx<=box[2]+tolerance and box[1]-tolerance<=cy<=box[3]+tolerance)
    validation.append({'frame':name,'use_roi':roi,'cx':cx,'cy':cy,'peak':peak,'expected_bbox':box.tolist(),'bbox_tolerance':tolerance,'inside':inside})
  if not options.validate_only:(OUT/'matching-preflight.json').write_text(json.dumps(validation,indent=2))
  print('MATCHING_PREFLIGHT',json.dumps(validation),flush=True)
  if not all(v['peak']>=args.control_min_peak and v['inside'] for v in validation):raise RuntimeError('Oblique matching validation failed; drone remains hovering')
def write_viz_step(directory,row,pose,rgb,sim,diagnostics=None):
 q=pose['orientation'];yaw=math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))
 image=cv2.resize(rgb,(512,288)).copy()
 u=round((row['cx']+1)*.5*511);v=round((row['cy']+1)*.5*287)
 cv2.drawMarker(image,(u,v),(0,255,0),cv2.MARKER_CROSS,11,1)
 extra={}
 if diagnostics and diagnostics.get('crop_xyxy') is not None:
  box=diagnostics['crop_xyxy'];x0,y0,x1,y1=box
  extra={'crop_xyxy':box,'crop_rgb':rgb[y0:y1,x0:x1],'crop_similarity':diagnostics[target_part]['local_similarity'],'target_part':target_part,'whole_peak':diagnostics['whole_peak']}
  if 'refinement' in diagnostics.get(target_part,{}):
   refined=diagnostics[target_part]['refinement']
   extra.update(point_source='local_image_registration',coarse_dino_center=diagnostics[target_part]['coarse_center'],refinement_score=refined['score'],refinement_margin=refined['margin'],refinement_accepted=refined['accepted'],raw_match_center=[row['raw_cx'],row['raw_cy']])
  scale=np.array([512/rgb.shape[1],288/rgb.shape[0]]);a=np.rint(np.array(box[:2])*scale).astype(int);b=np.rint(np.array(box[2:])*scale).astype(int)
  cv2.rectangle(image,tuple(a),tuple(b),(255,255,0),1);cv2.putText(image,target_part,(12,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
 destination=directory/('step_%04d.npz'%row['step']);temporary=destination.with_suffix('.tmp')
 with temporary.open('wb') as f:
  np.savez_compressed(f,step=row['step'],pose=np.array([row['north'],row['east'],row['down'],yaw]),rgb=image,sim=np.asarray(sim,np.float32),grid=int(round(math.sqrt(np.asarray(sim).size))),dt=.2,phase=row['state'],peak=row['peak'],cx=row['cx'],cy=row['cy'],depth_m=row['depth_m'],t=row['time_s'],**extra)
 temporary.replace(destination)
def bgr(m):
 a=np.frombuffer(m['data'],np.uint8).reshape(m['height'],m['width'],-1)[:,:,:3]
 if not str(m['encoding']).lower().startswith('bgr'):a=a[:,:,::-1]
 return np.ascontiguousarray(a)
class Recorder:
 def __init__(self,name):
  self.name=name;self.lock=threading.Lock();self.pending=collections.deque(maxlen=32);self.arrivals=[];self.output=[];self.start=None;self.end=None;self.error=None;self.last_frame=None
 def push(self,m):
  now=time.monotonic()
  with self.lock:
   self.pending.append((now,m))
   if self.start is not None and now>=self.start and (self.end is None or now<=self.end):
    self.arrivals.append((now,int(m['time_stamp'])))
 def launch(self,start):
  self.start=start;self.thread=threading.Thread(target=self.run,name='record-'+self.name);self.thread.start()
 def run(self):
  writer=None
  try:
   writer=cv2.VideoWriter(str(OUT/(self.name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,1080))
   if not writer.isOpened():raise RuntimeError('VideoWriter failed')
   i=0;lastts=None;frame=None
   while True:
    target=self.start+i/30
    if self.end is not None and target>=self.end:break
    delay=target+.15-time.monotonic()
    if delay>0:time.sleep(min(delay,.02));continue
    with self.lock:
     while len(self.pending)>2 and self.pending[1][0]<target:self.pending.popleft()
     item=min(self.pending,key=lambda a:abs(a[0]-target)) if self.pending else None
    if item is None:time.sleep(.005);continue
    arrival,m=item;ts=int(m['time_stamp'])
    if ts!=lastts:
     if (m['width'],m['height'])!=(1920,1080):raise RuntimeError('native resolution mismatch')
     frame=bgr(m);lastts=ts
    writer.write(frame);self.last_frame=frame
    self.output.append((i,target-self.start,ts,arrival-self.start));i+=1
  except BaseException as e:self.error=repr(e);traceback.print_exc()
  finally:
   if writer:writer.release()
 def finish(self,end):
  self.end=end;self.thread.join()
  if self.last_frame is not None:cv2.imwrite(str(OUT/('final-'+self.name+'.png')),self.last_frame)
  with (OUT/(self.name+'-frames.csv')).open('w') as f:
   wr=csv.writer(f);wr.writerow(['frame','wall_time_s','source_timestamp_ns','source_arrival_s']);wr.writerows(self.output)
  dt=end-self.start;unique=len({a[2] for a in self.output});self.arrivals=[a for a in self.arrivals if self.start<=a[0]<=end]
  return {'file':self.name+'.mp4','width':1920,'height':1080,'output_fps':30,'output_frames':len(self.output),'duration_s':len(self.output)/30,'received_source_frames':len(self.arrivals),'input_fps':len(self.arrivals)/dt,'unique_frames_in_video':unique,'effective_video_fps':unique/(len(self.output)/30) if self.output else 0,'repeat_fraction':1-unique/len(self.output) if self.output else 1,'error':self.error}
def cb(tag):
 def receive(t,m):
  if tag in recorders:recorders[tag].push(m)
  with lock:
   latest[tag]=m
   if tag in ('front','depth'):
    cache=buffers.setdefault(tag,{});cache[int(m['time_stamp'])]=m
    while len(cache)>20:cache.pop(min(cache))
 return receive
async def hover():
 if drone is not None:await asyncio.wait_for(await drone.hover_async(),8)
async def velocity(f,r,d,duration=.4,yaw_rate=0.):
 return await drone.move_by_velocity_body_frame_async(f,r,d,duration=duration,yaw=yaw_rate,yaw_is_rate=True)
async def run_control():
 global drone
 c.connect();c.socket_services.recv_timeout=12000
 c.socket_topics.recv_buffer_size=32
 ev=threading.Event()
 def info(t,m):c.on_topic_info(t,m);ev.set()
 c.subscribe('/$topics',info)
 if not ev.wait(6):raise RuntimeError('no topic info')
 w=World.__new__(World);w.client=c;w.parent_topic=next(t.rsplit('/clock',1)[0] for t in c.topics if t.endswith('/clock'));w.sim_config=cfg;w.home_geo_point=cfg['home-geo-point']
 drone=Drone(c,w,PROFILE['simulator']['drone'])
 if not drone.is_api_control_enabled():assert drone.enable_api_control()
 if drone.get_landed_state()!=1:raise RuntimeError('drone must already be airborne')
 await hover()
 for tag,cam,key in (('front','FrontCamera','scene_camera'),('depth','FrontCamera','depth_camera'),('chase','Chase','scene_camera')):
  c.subscribe(drone.sensors[cam][key],cb(tag))
 c.subscribe(drone.robot_info['collision_info'],cb('collision'))
 deadline=time.monotonic()+15
 while not all(k in latest for k in ('front','depth','chase')) and time.monotonic()<deadline:await asyncio.sleep(.05)
 if not all(k in latest for k in ('front','depth','chase')):raise RuntimeError('missing sensor images')
 await asyncio.sleep(2)
 if common_start:
  report['start_state']=start_state_check(drone.get_ground_truth_kinematics(),setup['common_start'])
  report.update(common_start=setup['common_start'],shared_asset_sha256=setup['shared_asset_sha256'],matching='whole_aircraft_then_RGB_crop_part',selected_part_active_from_first_command=True)
  if not report['start_state']['passed']:raise RuntimeError('Actual control start outside shared tolerance')
 collision_baseline=int(latest.get('collision',{}).get('time_stamp',0))
 start=time.monotonic()+.2
 for rec in recorders.values():rec.launch(start)
 report['recording_start_monotonic']=start
 report.update(mode='oblique' if oblique else 'nadir',setup=setup,altitude_guard_m=setup.get('altitude_guard_m',14))
 filtered={k:None for k in ('cx','cy','dist')};last=-1;lastdepth=-1;contact=0;aligned=0;lost=0;step=0;pending=[];roi_active=common_start;roi_hits=0;previous_cmd=np.zeros(3);previous_tick=time.monotonic()
 while True:
  tick=time.monotonic();deadline=tick+3;matched=None
  if tick-start>=options.max_duration:reason='duration_limit';break
  while time.monotonic()<deadline:
   with lock:
    rgb_cache=buffers.get('front',{});depth_cache=buffers.get('depth',{})
    for rt in sorted((ts for ts in rgb_cache if ts>last),reverse=True):
     if not depth_cache:break
     dt=min(depth_cache,key=lambda ts:abs(ts-rt))
     if abs(dt-rt)<=250000000:
      matched=rt;m=rgb_cache[rt];dm=depth_cache[dt];break
   if matched is not None:break
   await asyncio.sleep(.005)
  if matched is None:reason='synchronized_image_timeout';break
  perception_start=time.monotonic()
  last=matched;rgb=sc.decode_image(m)[0];cx,cy,mass,peak,sim=fp.query(rgb,use_roi=roi_active)
  perception_duration=time.monotonic()-perception_start
  if options.thin_roi_refinement and perception_duration>.75:
   report['perception_overrun_s']=perception_duration;reason='perception_overrun';break
  for k,val in (('cx',cx),('cy',cy)):filtered[k]=sc.ema_update(filtered[k],val,.3)
  xf,yf=filtered['cx'],filtered['cy'];locked=peak>=.2;lost=0 if locked else lost+1
  depth_cx,depth_cy,depth_half=xf,yf,4
  diagnostic=fp.last_diagnostics.get(target_part,{}) if common_start else {}
  if diagnostic.get('refinement',{}).get('accepted'):
   depth_cx,depth_cy=cx,cy;depth_half=refined_depth_half_width(diagnostic['refined_part_wh_px'],[m['width'],m['height']],[dm['width'],dm['height']])
  dist=sc.sample_depth(dm,depth_cx,depth_cy,half=depth_half) if locked else None
  if dist is not None:filtered['dist']=sc.ema_update(filtered['dist'],dist,.5)
  df=filtered['dist'] if dist is not None else None
  kin=drone.get_ground_truth_kinematics();pos=kin['pose']['position'];alt=-pos['z'];distance=None if ref is None else float(np.linalg.norm(ref-np.array([pos[k] for k in ('x','y','z')])))
  aligned=aligned+1 if locked and math.hypot(xf,yf)<.15 else 0
  f,r,d,yaw_rate,_,_=sc.compute_body_cmd(args,xf,yf,0)
  if args.nadir and aligned<3:d=0
  if oblique:
   f*=max(0.,1-abs(xf)/.5);d=float(np.clip(d,0,2.2))
  if not locked or df is None:f=r=d=yaw_rate=0
  if df is not None:
   scale=sc.depth_speed_scale(df,args.slow_dist,args.stop_dist,.15);f*=scale;r*=scale;d*=scale
  if oblique and locked and df is not None:
   delta=max(.02,min(time.monotonic()-previous_tick,.3));command=np.clip(np.array([f,r,d]),previous_cmd-delta,previous_cmd+delta);f,r,d=map(float,command)
  if df is not None and df<=setup.get('close_range_m',0):
   speed=math.sqrt(f*f+r*r+d*d);cap=setup['close_speed_mps']
   if speed>cap:f*=cap/speed;r*=cap/speed;d*=cap/speed
  previous_cmd=np.array([f,r,d]);previous_tick=time.monotonic()
  depthts=int(dm['time_stamp'])
  if depthts!=lastdepth:
   contact=contact+1 if df is not None and df<=args.stop_dist and locked else 0;lastdepth=depthts
  if contact:f=r=d=yaw_rate=0;previous_cmd[:]=0
  state='CONFIRM' if contact else (('APPROACH_'+target_part.upper() if roi_active else 'APPROACH_FULL') if oblique and locked else ('DESCEND' if d>0 else ('ALIGN' if locked and df is not None else 'HOLD')))
  stop=None
  guard_depth=None
  if 'depth_guard_m' in setup:
   depth=np.asarray(sc.ce.decode(dm),np.float32);h,w=depth.shape[:2];part=depth[h//4:3*h//4,w//4:3*w//4];valid=part[np.isfinite(part)&(part>.1)]
   if valid.size:guard_depth=float(np.percentile(valid,1))
  collision=latest.get('collision',{})
  if int(collision.get('time_stamp',0))>collision_baseline:
   stop='collision';report['collision_event']=collision
  elif contact>=3:stop='range_threshold'
  elif guard_depth is not None and guard_depth<setup['depth_guard_m']:stop='depth_clearance_guard'
  elif alt<=setup.get('altitude_guard_m',14):stop='altitude_guard'
  elif lost>=15:stop='tracking_lost'
  elif alt>setup.get('max_altitude_m',40):stop='unexpected_altitude'
  row={'step':step,'time_s':time.monotonic()-start,'north':pos['x'],'east':pos['y'],'down':pos['z'],'cx':xf,'cy':yf,'peak':peak,'depth_m':-1 if df is None else df,'forward':f,'right':r,'down_speed':d,'reference_distance_m':distance,'rgb_timestamp':last,'depth_timestamp':depthts,'sensor_skew_ms':abs(last-depthts)/1e6,'contact_count':contact,'state':state}
  row.update(yaw_rate=yaw_rate,roi_active=roi_active,guard_depth_m=guard_depth)
  if common_start:
   row.update(crop_xyxy=fp.last_diagnostics['crop_xyxy'],whole_peak=fp.last_diagnostics['whole_peak'])
  if options.thin_roi_refinement:
   row.update(raw_cx=cx,raw_cy=cy,point_source=diagnostic.get('point_source','dino_center'),refinement=diagnostic.get('refinement'),coarse_dino_center=diagnostic.get('coarse_center'),depth_target_cx=depth_cx,depth_target_cy=depth_cy,depth_sample_half=depth_half,perception_duration_s=perception_duration)
  rows.append(row);rgbs.append(cv2.resize(rgb,(448,252)));poses.append(kin['pose']);sims.append(sim)
  write_viz_step(VDIR,row,kin['pose'],rgb,sim,fp.last_diagnostics if common_start else None)
  if step%20==0 or stop:print('STEP',step,state,'alt',round(alt,2),'depth',None if df is None else round(df,2),'center',[round(xf,3),round(yf,3)],'reference',None if distance is None else round(distance,2),'confirm',contact,flush=True)
  if any(rec.error for rec in recorders.values()):raise RuntimeError('recording thread failed')
  if stop:reason=stop;break
  pending.append(await velocity(f,r,d,yaw_rate=yaw_rate));step+=1
  if oblique and not roi_active:
   roi_hits=roi_hits+1 if locked and df is not None and df<=args.roi_switch_dist else 0
   if roi_hits>=3:
    roi_active=True;contact=0;aligned=0
    for key in filtered:filtered[key]=None
    print('PROTOTYPE_SWITCH FULL -> '+target_part.upper(),flush=True)
  await asyncio.sleep(max(0,.2-(time.monotonic()-tick)))
 await velocity(0,0,0,.2)
 if pending:await asyncio.gather(*pending,return_exceptions=True)
 await hover();await asyncio.sleep(2)
 report['stop_reason']=reason;report['arrival_confirmed']=reason=='range_threshold';report['final_state']=drone.get_ground_truth_kinematics();report['status']='completed'
 print('CONTROL_DONE',reason,'steps',len(rows),flush=True)
def main(argv=None):
 global recorders,c,drone,latest,buffers,lock,rows,rgbs,poses,sims,report
 initialize(argv)
 matching_preflight()
 if options.validate_only:
  print('VALIDATION_ONLY: no simulator connection or flight commands');return 0
 recorders={k:Recorder(k) for k in ('front','chase')}
 c=ProjectAirSimClient(address=args.address);drone=None;latest={};buffers={};lock=threading.Lock();rows=[];rgbs=[];poses=[];sims=[]
 report={'status':'started','output_dir':str(OUT),'live_viz_dump':str(VDIR),'map_used_for_control':False,'max_steps':None,'max_duration_s':None,'camera_capture_interval_s':.0167,'distance_stop_m':args.stop_dist,'stop_confirm_frames':3,'altitude_guard_m':14,'reference_used_for_control':False,'online_segmentation':False,'depth_resolution':[640,360],'source_configuration_modified':False,'transport_address':args.address,'max_sensor_skew_s':.25,'topic_receive_buffer_messages':32}
 report['max_duration_s']=options.max_duration
 report['matching_preflight_scope']=target_part if options.preflight_selected_part_only else 'all_parts'
 report['thin_roi_refinement']=options.thin_roi_refinement
 if options.thin_roi_refinement:
  report.update(refinement_min_score=.85,refinement_min_margin=.15,max_perception_duration_s=.75,refined_point_depth_sampling='raw registered anchor; neighborhood limited by registered part size')
 try:asyncio.run(run_control())
 except BaseException as e:report['status']='failed';report['error']=repr(e);traceback.print_exc()
 finally:
  try:asyncio.run(hover())
  except Exception:pass
  end=time.monotonic()
  for rec in recorders.values():rec.end=end
  for name,rec in recorders.items():
   if rec.start is not None:
    report[name+'_recording']=rec.finish(end)
  try:c.disconnect()
  except Exception:pass
  if rows:
   with (OUT/'signals.csv').open('w') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
   (OUT/'control-signals.json').write_text(json.dumps(rows,indent=2))
   (OUT/'poses.json').write_text(json.dumps(poses))
   with h5py.File(OUT/'run.h5','w') as f:
    f.create_dataset('rgb',data=np.stack(rgbs),compression='gzip');f.create_dataset('similarity',data=np.stack(sims))
    for name,data in {'pose_ned':[[r['north'],r['east'],r['down']] for r in rows],'action_body':[[r['forward'],r['right'],r['down_speed']] for r in rows],'t':[r['time_s'] for r in rows],'depth_m':[r['depth_m'] for r in rows]}.items():f.create_dataset(name,data=np.asarray(data))
    f.attrs['template']=str(args.template);f.attrs['stats_episode']=str(args.stats_episode);f.attrs['max_steps']='unlimited'
   report['steps']=len(rows);report['initial_reference_distance_m']=rows[0]['reference_distance_m'];report['final_reference_distance_m']=rows[-1]['reference_distance_m'];report['final_depth_m']=rows[-1]['depth_m']
  report['controller_args']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
  (OUT/'report.json').write_text(json.dumps(report,indent=2))
  print('REPORT',json.dumps({k:v for k,v in report.items() if k not in ('controller_args','final_state')}),flush=True)
 return 0 if report['status']=='completed' and report.get('arrival_confirmed') else 1

if __name__=='__main__':raise SystemExit(main())
