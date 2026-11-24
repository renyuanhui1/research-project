"""Export the perimeter meshes with planar UVs and their authored face normals."""

import bpy

def write_mesh(path, material, faces):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    indices = {}
    polygons = []
    for face in faces:
        polygons.append([indices.setdefault(p, len(indices)) for p in face])
    mesh = bpy.data.meshes.new(path.stem)
    mesh.from_pydata(list(indices), [], polygons)
    mesh.update()
    uv = mesh.uv_layers.new(name="SurfaceUV")
    for face in mesh.polygons:
        normal = face.normal
        axes = (0, 1) if abs(normal.z) > 0.7 else ((1, 2) if abs(normal.x) > abs(normal.y) else (0, 2))
        for i in face.loop_indices:
            point = mesh.vertices[mesh.loops[i].vertex_index].co
            uv.data[i].uv = (point[axes[0]] / 3, point[axes[1]] / 3)
    mesh.materials.append(bpy.data.materials.new(material))
    mesh.update()
    obj = bpy.data.objects.new(path.stem, mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, axis_forward="-Y",
                             axis_up="Z", mesh_smooth_type="FACE", bake_anim=False)
