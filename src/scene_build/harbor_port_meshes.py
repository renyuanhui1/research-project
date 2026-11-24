"""Blender FBX export of terrain strips, road junctions and house plinths in UE world coordinates."""

import json
import sys
from pathlib import Path
import bpy

source, destination = map(Path, sys.argv[sys.argv.index("--") + 1 :])
sys.path.insert(0, str(Path(__file__).parent))
from harbor_coast_edges import write_mesh

for name, groups in json.loads(source.read_text()).items():
    if name in ("CoastalWater", "TerrainCutFaces"):
        (material, faces), = groups.items()
        blender_faces = [[(x,-y,z) for x,y,z in reversed(face)] for face in faces]
        write_mesh(destination / (name + ".fbx"), material, blender_faces)
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    points = []
    faces = []
    slots = []
    index = {}
    for mi, (material, triangles) in enumerate(groups.items()):
        for tri in triangles:
            face = []
            for x, y, z in tri:
                p = (round(x, 4), round(-y, 4), round(z, 5))
                if p not in index:
                    index[p] = len(points)
                    points.append(p)
                face.append(index[p])
            if len(set(face)) != 3:
                continue
            a, b, c = [points[i] for i in face]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                face.reverse()
            faces.append(face)
            slots.append(mi)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    for mat in groups:
        mesh.materials.append(bpy.data.materials.new(mat))
    mesh.polygons.foreach_set("material_index", slots)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=str(destination / (name + ".fbx")),
        use_selection=True,
        axis_forward="-Y",
        axis_up="Z",
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
    )
