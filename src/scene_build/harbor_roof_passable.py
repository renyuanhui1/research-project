"""Export ShedTornRoof_Passable.fbx with purlins cleared from missing bay j=4 (for Building_098).

Run with Blender 3.6:
  blender --background --python src/scene_build/harbor_roof_passable.py -- <OUTPUT_DIR>
"""

import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))
import bpy
import harbor_meshes

bpy.ops.wm.read_factory_settings(use_empty=True)
harbor_meshes.building("Shed", 3, True, clear_passable_bay=True).save("ShedTornRoof_Passable")
