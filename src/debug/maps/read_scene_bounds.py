import json,time,threading
from pathlib import Path
import sys
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()
from projectairsim import ProjectAirSimClient,World
from projectairsim.types import BoxAlignment
ROOT=project_path(PROFILE['map']['root'])
def main():
 from common.runtime import configure_client_log
 configure_client_log()
 ROOT.mkdir(parents=True,exist_ok=True)
 c=ProjectAirSimClient(address=PROFILE['simulator']['address']);c.connect();c.socket_services.recv_timeout=15000
 try:
  ev=threading.Event()
  def cb(t,m):c.on_topic_info(t,m);ev.set()
  c.subscribe('/$topics',cb)
  if not ev.wait(6):raise RuntimeError('no topic info')
  prefix=next(t.rsplit('/clock',1)[0] for t in c.topics if t.endswith('/clock'))
  w=World.__new__(World);w.client=c;w.parent_topic=prefix
  names=w.list_objects('StaticMeshActor.*')
  data=[];start=time.monotonic()
  for i,name in enumerate(names):
   b=w.get_3d_bounding_box(name,BoxAlignment.WORLD_AXIS);data.append({'name':name,**b})
   if (i+1)%200==0 or i+1==len(names):
    (ROOT/'object-bounds.json').write_text(json.dumps(data,indent=2))
    print('BOUNDS',i+1,'/',len(names),'seconds',round(time.monotonic()-start,1),flush=True)
  (ROOT/'bounds-source.json').write_text(json.dumps({'scene_topic':prefix,'count':len(data),'queried_live':True,'elapsed_s':time.monotonic()-start},indent=2))
 finally:c.disconnect()

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',default=None)
 parser.parse_args();main()
