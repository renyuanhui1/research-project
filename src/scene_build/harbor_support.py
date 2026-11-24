import bpy, bmesh, json
from pathlib import Path
from mathutils import Vector

root = Path(__file__).parent / "library"
clouds = {}
for name in ("tugboat", "speedboat", "skiff", "fishing-trawler", "dump-truck"):
    for f in (root / name).glob("*.fbx"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(f))
        obj = next(o for o in bpy.context.scene.objects if o.type == "MESH")
        m = bmesh.new()
        m.from_mesh(obj.data)
        res = bmesh.ops.convex_hull(m, input=list(m.verts))
        corners = [
            obj.matrix_world @ v.co
            for v in res["geom"]
            if isinstance(v, bmesh.types.BMVert)
        ]
        clouds[f.stem] = [
            [round(p.x * 100, 3), round(-p.y * 100, 3), round(p.z * 100, 3)]
            for p in corners
        ]
        assert len(corners) > 3
        m.free()
(root.parent / "contact_points.json").write_text(json.dumps(clouds))
