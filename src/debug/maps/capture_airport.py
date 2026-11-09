"""Export georeferenced imagery using a temporary UE capture actor."""
import unreal, json, traceback
from pathlib import Path
import sys
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()
import tempfile
ROOT=Path(tempfile.gettempdir())/PROFILE['editor']['export_name']
class AirportCaptureJob:
 def __init__(self):
  self.actor=None;self.handle=None;self.texture=None;self.index=0;self.frames=0
  self.report={"status":"started","captures":{},"errors":[]}
 def save(self):
  (ROOT/"capture-report.json").write_text(json.dumps(self.report,ensure_ascii=False,indent=2),encoding="utf-8")
 def finish(self,status):
  self.report["status"]=status
  if status=="completed":
   check=self.report["captures"].pop("capture_check",None)
   if check:
    check.pop("file",None);self.report["preflight"]=check
    (ROOT/"capture_check.png").unlink(missing_ok=True)
  if self.handle is not None:
   unreal.unregister_slate_post_tick_callback(self.handle);self.handle=None
  if self.actor is not None:
   self.actor.destroy_actor();self.actor=None
  self.texture=None;self.save()
  unreal.log("AIRPORT_CAPTURE_END "+json.dumps({"status":status,"errors":self.report["errors"]}))
 def start(self):
  unreal.log("AIRPORT_CAPTURE_BEGIN")
  try:
   self.world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
   if self.world is None:raise RuntimeError("Play world missing")
   self.target_actors=[a for a in unreal.GameplayStatics.get_all_actors_of_class(self.world,unreal.StaticMeshActor) if a.get_name() in PROFILE['map']['actors']]
   if len(self.target_actors)!=len(PROFILE['map']['actors']):raise RuntimeError("Expected both aircraft actors")
   self.report["ground_texture_hidden_actor_names"]=[a.get_name() for a in self.target_actors]
   before={a.get_name() for a in unreal.GameplayStatics.get_all_actors_of_class(self.world,unreal.SceneCapture2D)}
   unreal.SystemLibrary.execute_console_command(self.world,"summon /Script/Engine.SceneCapture2D")
   spawned=[a for a in unreal.GameplayStatics.get_all_actors_of_class(self.world,unreal.SceneCapture2D) if a.get_name() not in before]
   if len(spawned)!=1:raise RuntimeError("SceneCapture2D spawn count: "+str(len(spawned)))
   self.actor=spawned[0];self.capture=self.actor.get_component_by_class(unreal.SceneCaptureComponent2D)
   for key,value in (("capture_every_frame",False),("capture_on_movement",False),("always_persist_rendering_state",True),("projection_type",unreal.CameraProjectionMode.ORTHOGRAPHIC),("capture_source",unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)):
    self.capture.set_editor_property(key,value)
   settings=self.capture.get_editor_property("post_process_settings")
   extended=int(unreal.SystemLibrary.get_console_variable_int_value("r.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange"))
   brightness=11. if extended else 1.
   for key,value in (("override_auto_exposure_min_brightness",True),("override_auto_exposure_max_brightness",True),("auto_exposure_min_brightness",brightness),("auto_exposure_max_brightness",brightness)):
    settings.set_editor_property(key,value)
   self.capture.set_editor_property("post_process_settings",settings);self.capture.set_editor_property("post_process_blend_weight",1.)
   self.report.update(exposure_extended=extended,exposure_value=brightness,warmup_frames_per_image=20)
   catalog=json.loads((ROOT/"actors.json").read_text(encoding="utf-8"))
   terrain=[a for a in catalog if a["class"]=="StaticMeshActor" and a["bounds_extent_cm"][2]<15 and a["bounds_extent_cm"][0]>5000 and a["bounds_extent_cm"][1]>5000]
   lows=[min(a["bounds_center_cm"][k]-a["bounds_extent_cm"][k] for a in terrain) for k in range(3)]
   highs=[max(a["bounds_center_cm"][k]+a["bounds_extent_cm"][k] for a in terrain) for k in range(3)]
   cx=(lows[0]+highs[0])/2;cy=(lows[1]+highs[1])/2;pad=max(highs[0]-lows[0],highs[1]-lows[1])*.02
   span_n=highs[0]-lows[0]+2*pad;span_e=highs[1]-lows[1]+2*pad
   if span_e>=span_n:width=4096;height=round(4096*span_n/span_e)
   else:height=4096;width=round(4096*span_e/span_n)
   tc=next(a for a in catalog if a['name']==PROFILE['target']['actor'])["bounds_center_cm"]
   self.tasks=[("capture_check",tc[0],tc[1],10000.,512,512),("airport_ortho",cx,cy,span_e,width,height),("near_target_ortho",tc[0],tc[1],10000.,1024,1024)]
   self.begin_image();self.handle=unreal.register_slate_post_tick_callback(self.tick);self.save()
  except Exception as error:
   self.report["errors"].append(repr(error));unreal.log_error(traceback.format_exc());self.finish("failed")
 def begin_image(self):
  name,cx,cy,span_e,width,height=self.tasks[self.index]
  self.capture.set_editor_property("capture_every_frame",False)
  self.capture.set_editor_property("hidden_actors",[] if name=="capture_check" else self.target_actors)
  self.actor.set_actor_location_and_rotation(unreal.Vector(cx,cy,30000),unreal.Rotator(pitch=-90,yaw=0,roll=0),False,True)
  self.capture.set_editor_property("ortho_width",span_e)
  self.texture=unreal.RenderingLibrary.create_render_target2d(self.world,width,height,unreal.TextureRenderTargetFormat.RTF_RGBA8)
  self.capture.set_editor_property("texture_target",self.texture);self.capture.set_editor_property("capture_every_frame",True)
  self.frames=0;unreal.log("AIRPORT_CAPTURE_WARMUP "+name)
 def tick(self,delta):
  try:
   self.frames+=1
   if self.frames<20:return
   name,cx,cy,span_e,width,height=self.tasks[self.index]
   self.capture.set_editor_property("capture_every_frame",False)
   samples=[]
   for row in range(1,8):
    for col in range(1,8):
     pixel=unreal.RenderingLibrary.read_render_target_pixel(self.world,self.texture,round(width*col/8),round(height*row/8))
     samples.extend([pixel.r,pixel.g,pixel.b])
   pixel_min,pixel_max=min(samples),max(samples)
   unreal.RenderingLibrary.export_render_target(self.world,self.texture,str(ROOT),name+".png")
   span_n=span_e*height/width
   self.report["captures"][name]={"file":name+".png","width":width,"height":height,"east_min_m":(cy-span_e/2)/100,"east_max_m":(cy+span_e/2)/100,"north_min_m":(cx-span_n/2)/100,"north_max_m":(cx+span_n/2)/100,"camera_ue_cm":[cx,cy,30000],"rotation_pitch_yaw_roll":[-90,0,0],"projection":"orthographic","sample_rgb_range":[pixel_min,pixel_max],"actual_rotation":str(self.actor.get_actor_rotation()),"actual_location":str(self.actor.get_actor_location())}
   self.save();unreal.log("AIRPORT_CAPTURE_IMAGE "+name+" range="+str([pixel_min,pixel_max]))
   if pixel_min>=250 or pixel_max<=5 or pixel_max-pixel_min<5:raise RuntimeError("Capture has no usable contrast: "+name)
   self.index+=1
   if self.index==len(self.tasks):self.finish("completed")
   else:self.begin_image()
  except Exception as error:
   self.report["errors"].append(repr(error));unreal.log_error(traceback.format_exc());self.finish("failed")
if "_airport_capture_job" in globals() and _airport_capture_job.handle is not None:raise RuntimeError("Capture is already running")
_airport_capture_job=AirportCaptureJob()
_airport_capture_job.start()
