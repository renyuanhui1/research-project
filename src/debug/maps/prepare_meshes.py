"""处理UE导出的实际飞机网格，验证世界坐标后保存RViz使用的简化三角面。"""
import json,shutil,time
from pathlib import Path
import sys
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()
import numpy as np
import open3d as o3d
ROOT=project_path(PROFILE['map']['root'])
SOURCE=Path(PROFILE['editor']['wsl_export_dir'])
def main():
 (ROOT/'source').mkdir(exist_ok=True);(ROOT/'meshes').mkdir(exist_ok=True)
 for name in ('actors.json','targets.json'):
  shutil.copy2(SOURCE/name,ROOT/'source'/name)
 targets=json.loads((SOURCE/'targets.json').read_text())
 trial=project_path(PROFILE['assets']['preparation_assets'])/'template.json'
 reference=np.array(json.loads(trial.read_text())['reference_ned_m']) if trial.exists() else None
 report=[]
 for t in targets:
  name=t['name'];start=time.monotonic();file=SOURCE/(name+'.obj')
  print('READ_MESH',name,round(file.stat().st_size/1e6,1),'MB',flush=True)
  mesh=o3d.io.read_triangle_mesh(str(file),enable_post_processing=False)
  if len(mesh.vertices)==0 or len(mesh.triangles)==0:raise RuntimeError('empty OBJ mesh')
  w,x,y,z=t['rotation_wxyz'];rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
  scale=np.array(t['scale']);location=np.array(t['location_cm'])
  def to_world(vertices):
   ue=vertices[:,[0,2,1]]*scale
   ue=ue@rotation.T+location
   return ue[:,[1,0,2]]/100
  vertices_obj=np.asarray(mesh.vertices)
  original_world=to_world(vertices_obj);lo=original_world.min(0);hi=original_world.max(0)
  local_min=vertices_obj.min(0);local_max=vertices_obj.max(0)
  corners=np.array([[x,y,z] for x in (local_min[0],local_max[0]) for y in (local_min[1],local_max[1]) for z in (local_min[2],local_max[2])])
  bounds_world=to_world(corners)
  bc=np.array(t['bounds_center_cm'])[[1,0,2]]/100;be=np.array(t['bounds_extent_cm'])[[1,0,2]]/100
  bbox_error=float(np.max(np.abs(np.r_[bounds_world.min(0)-(bc-be),bounds_world.max(0)-(bc+be)])))
  if bbox_error>.3:raise RuntimeError('OBJ/world bounds disagree by '+str(bbox_error))
  original_triangles=len(mesh.triangles)
  mesh.remove_duplicated_vertices();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles()
  if len(mesh.triangles)>30000:mesh=mesh.simplify_quadric_decimation(target_number_of_triangles=30000)
  mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
  vertices=to_world(np.asarray(mesh.vertices));faces=np.asarray(mesh.triangles).astype(np.int32)
  np.savez_compressed(ROOT/'meshes'/(name+'.npz'),vertices_enu=vertices.astype(np.float64),triangles=faces,actor_name=np.array(name),actor_label=np.array(t['label']))
  info={'name':name,'label':t['label'],'source_mesh':t['meshes'],'source_triangles':original_triangles,'display_triangles':len(faces),'display_vertices':len(vertices),'bounds_error_m':bbox_error,'world_bounds_enu':[lo.tolist(),hi.tolist()],'seconds':time.monotonic()-start}
  if name==PROFILE['target']['actor'] and reference is not None:
   mesh.vertices=o3d.utility.Vector3dVector(vertices)
   scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
   ref_enu=reference[[1,0,2]].copy();ref_enu[2]*=-1
   info['nose_reference_mesh_distance_m']=float(scene.compute_distance(o3d.core.Tensor(ref_enu[None].astype(np.float32))).numpy()[0])
  print('MESH_READY',json.dumps(info,ensure_ascii=False),flush=True);report.append(info)
 (ROOT/'mesh-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',default=None)
 parser.parse_args();main()
