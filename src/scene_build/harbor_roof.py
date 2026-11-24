import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))
import bpy, harbor_meshes

bpy.ops.wm.read_factory_settings(use_empty=True)
harbor_meshes.building("Shed", 3, True).save("ShedTornRoof")
