"""按当前UE机场主体范围生成全场碰撞占用背景，不重新加载场景。"""
import json,math,sys,time,threading,socket
from pathlib import Path
import sys
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()
import numpy as np
from projectairsim import ProjectAirSimClient,World
ROOT=project_path(PROFILE['map']['root'])
from debug.probe_voxel_map import make_pose,cell_centers,save_obstacle_projection
def main():
 from common.runtime import configure_client_log
 configure_client_log()
 ROOT.mkdir(parents=True,exist_ok=True)
 objects=json.loads((ROOT/'object-bounds.json').read_text())
 terrain=[b for b in objects if b['size']['z']<.3 and b['size']['x']>100 and b['size']['y']>100]
 centers=np.array([[b['center'][k] for k in ('x','y','z')] for b in terrain]);sizes=np.array([[b['size'][k] for k in ('x','y','z')] for b in terrain])
 lo=(centers-sizes/2).min(0);hi=(centers+sizes/2).max(0)
 cx,cy=(lo[:2]+hi[:2])/2;size_n=math.ceil((hi[0]-lo[0]+100)/500)*500;size_e=math.ceil((hi[1]-lo[1]+100)/500)*500
 res=5.;nx=int(size_n/res);ny=int(size_e/res);nz=20;grid=np.zeros((ny,nz,nx),bool)
 folder=ROOT/'global';folder.mkdir(exist_ok=True)
 with socket.create_connection((PROFILE['simulator']['address'],8990),timeout=3):pass
 c=ProjectAirSimClient(address=PROFILE['simulator']['address']);c.connect();c.socket_services.recv_timeout=60000
 start=time.monotonic()
 try:
  ev=threading.Event()
  def cb(t,m):c.on_topic_info(t,m);ev.set()
  c.subscribe('/$topics',cb)
  if not ev.wait(6):raise RuntimeError('no topic info')
  prefix=next(t.rsplit('/clock',1)[0] for t in c.topics if t.endswith('/clock'));w=World.__new__(World);w.client=c;w.parent_topic=prefix
  done=0;total=(nx//100)*(ny//100)
  for j in range(0,ny,100):
   for i in range(0,nx,100):
    n=float(cx+(i+50-nx//2)*res);e=float(cy+(j+50-ny//2)*res)
    empty=w.create_voxel_grid(make_pose((n,e,-1000)),1,1,1,1.,actors_to_ignore=[])
    if any(empty):raise RuntimeError('empty air query occupied')
    flat=w.create_voxel_grid(make_pose((n,e,-40)),500,500,100,res,actors_to_ignore=[])
    if len(flat)!=100*100*nz:raise RuntimeError('invalid grid count')
    grid[j:j+100,:,i:i+100]=np.asarray(flat,bool).reshape(100,nz,100);done+=1
    if done%8==0 or done==total:print('AIRPORT_GRID',done,'/',total,'elapsed_s',round(time.monotonic()-start,1),flush=True)
  center=(float(cx),float(cy),-40.)
  np.savez_compressed(folder/'occupancy_packed.npz',packed=np.packbits(grid.reshape(-1),bitorder='little'),shape=np.array(grid.shape,np.int32),center_ned_m=np.array(center),size_m=np.array([size_n,size_e,100]),resolution_m=np.float64(res),bitorder=np.array('little'))
  north,east,altitude=cell_centers(center,grid.shape,res)
  save_obstacle_projection(grid,north,east,altitude,2.5,folder/'obstacle_projection.png')
  metadata={'scene_topic':prefix,'center_ned_m':list(center),'size_m':[size_n,size_e,100],'resolution_m':res,'shape_east_altitude_north':list(grid.shape),'terrain_bounds_ned_m':{'min':lo.tolist(),'max':hi.tolist()},'ground_mesh_count':len(terrain),'queried_static_meshes':len(objects),'occupied_voxels':int(grid.sum()),'above_ground_occupied_voxels':int(grid[:,altitude>2.5,:].sum()),'generation_seconds':time.monotonic()-start,'source':'live UE ECC_Pawn collision sampling; buildings/background only','scene_reloaded':False}
  (folder/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2));print('AIRPORT_GRID_DONE',json.dumps(metadata),flush=True)
 finally:c.disconnect()

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',default=None)
 parser.parse_args();main()
