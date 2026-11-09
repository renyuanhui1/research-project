import unreal,json,traceback
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
ROOT.mkdir(parents=True,exist_ok=True)
def vec(v):return [float(v.x),float(v.y),float(v.z)]
def quat(q):return [float(q.w),float(q.x),float(q.y),float(q.z)]
def save(name,data):(ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
unreal.log("AIRPORT_EXPORT_BEGIN")
report={"status":"started","errors":[]}
try:
 world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
 if world is None:raise RuntimeError('No active Play world')
 actors=unreal.GameplayStatics.get_all_actors_of_class(world,unreal.Actor)
 catalog=[];lookup={}
 for actor in actors:
  comps=actor.get_components_by_class(unreal.StaticMeshComponent)
  if not comps:continue
  name=actor.get_name();lookup[name]=(actor,comps)
  center,extent=actor.get_actor_bounds(False)
  transform=actor.get_actor_transform()
  meshes=[]
  for comp in comps:
   mesh=comp.get_editor_property("static_mesh")
   if mesh:meshes.append(mesh.get_path_name())
  catalog.append({"name":name,"label":actor.get_actor_label(),"class":actor.get_class().get_name(),"location_cm":vec(actor.get_actor_location()),"rotation_wxyz":quat(transform.rotation),"scale":vec(actor.get_actor_scale3d()),"bounds_center_cm":vec(center),"bounds_extent_cm":vec(extent),"meshes":meshes})
 save("actors.json",catalog)
 report["actor_count"]=len(actors);report["static_mesh_actor_count"]=len(catalog)
 report["runtime_world"]=str(world.get_name())
 targets=[]
 for name in PROFILE['map']['actors']:
  if name not in lookup:
   report["errors"].append("Target not found in Play world: "+name);continue
  actor,comps=lookup[name];comp=comps[0];mesh=comp.get_editor_property("static_mesh")
  target=next(item.copy() for item in catalog if item["name"]==name)
  try:
   body=mesh.get_editor_property("body_setup")
   target["collision_trace_flag"]=str(body.get_editor_property("collision_trace_flag"))
   agg=body.get_editor_property("agg_geom")
   target["simple_collision"]={k:len(agg.get_editor_property(k)) for k in ("box_elems","sphere_elems","sphyl_elems","convex_elems")}
  except Exception as exc:target["collision_info_error"]=str(exc)
  task=unreal.AssetExportTask()
  task.object=mesh;task.filename=str(ROOT/(name+".obj"));task.automated=True;task.prompt=False;task.replace_identical=True;task.exporter=unreal.StaticMeshExporterOBJ()
  target["obj_export_success"]=bool(unreal.Exporter.run_asset_export_task(task))
  targets.append(target)
 save("targets.json",targets)
 report["status"]="completed" if not report["errors"] and all(t["obj_export_success"] for t in targets) else "failed"
except Exception as exc:
 report["status"]="failed";report["errors"].append(repr(exc));unreal.log_error(traceback.format_exc())
finally:
 for name in PROFILE['map']['actors']:
  for suffix in ("_UV1.obj","_Internal.obj"):
   path=ROOT/(name+suffix)
   if path.exists():path.unlink()
 save("export-report.json",report)
 unreal.log("AIRPORT_EXPORT_END "+json.dumps({"status":report["status"],"errors":report["errors"]}))
