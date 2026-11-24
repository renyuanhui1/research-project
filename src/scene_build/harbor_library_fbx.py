"""Flatten licensed glTF assemblies into independently placeable UE meshes; preserve source UVs."""

from pathlib import Path
import math
import sys
import bpy
from mathutils import Vector, Matrix

root = Path(sys.argv[sys.argv.index("--") + 1])
variants = {
    "red-cottage",
    "village-house",
    "warehouse",
    "harbour-office",
    "delivery-van",
    "pickup-truck",
    "family-sedan",
    "dump-truck",
    "fishing-trawler",
    "skiff",
    "tugboat",
    "speedboat",
}
for path in sorted(root.glob("*/model.gltf")):
    name = path.parent.name
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    # Freeze hierarchy transforms, including original nonuniform quantization scales, before joining.
    bpy.ops.object.select_all(action="DESELECT")
    transforms = [o.matrix_world.copy() for o in objs]
    for o, mat in zip(objs, transforms):
        o.parent = None
        o.data = o.data.copy()
        o.data.transform(mat)
        o.matrix_world = Matrix.Identity(4)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    vs = o.data.vertices
    low = Vector([min(v.co[i] for v in vs) for i in range(3)])
    high = Vector([max(v.co[i] for v in vs) for i in range(3)])
    ctr = (low + high) * 0.5
    for v in vs:
        x, y, z = v.co
        x -= ctr.x
        y -= ctr.y
        if high.y - low.y > high.x - low.x:
            x, y = y, -x
        v.co = (x, y, z - low.z)

    def export(suffix):
        o.name = name.replace("-", "_") + suffix
        bpy.ops.export_scene.fbx(
            filepath=str(path.parent / (name.replace("-", "_") + suffix + ".fbx")),
            use_selection=True,
            axis_forward="-Y",
            axis_up="Z",
            add_leaf_bones=False,
            object_types={"MESH"},
            bake_anim=False,
            mesh_smooth_type="FACE",
        )

    if name == "harbour-office":
        sys.path.insert(0, str(Path(__file__).parent))
        from harbor_office_cladding import clip_office_cladding

        clip_office_cladding(o)
        vs = o.data.vertices
    if name == "dump-truck":
        # The five stiffeners on each side of the tipper bed terminate on a
        # sheet face. Seat their hidden ends halfway into the 40 mm sheet.
        rib_centres = (-0.535, 0.565, 1.665, 2.765, 3.715)
        for vertex in vs:
            x, y, z = vertex.co
            if not 1.6 < z < 3.25 or not any(
                abs(abs(x - centre) - 0.05) < 0.001 for centre in rib_centres
            ):
                continue
            if abs(y + 1.30986) < 0.001 or abs(y - 1.53015) < 0.001:
                vertex.co.y -= 0.02
    export("")
    if name not in variants:
        continue
    length = max(v.co.x for v in vs) - min(v.co.x for v in vs)
    width = max(v.co.y for v in vs) - min(v.co.y for v in vs)
    height = max(v.co.z for v in vs)
    for v in vs:
        x, y, z = v.co
        nx, ny, nz = x / length, y / width, z / height
        if name in ("red-cottage", "village-house", "warehouse", "harbour-office"):
            # One eave peels upward where the end of the roof has lost its fasteners.
            w = max(0, (nx + 0.12) / 0.62) ** 1.3 * max(0, (ny + 0.05) / 0.55) ** 1.5
            w *= min(1, max(0, (nz - 0.53) / 0.25))
            v.co.z += height * 0.27 * w
            v.co.y -= width * 0.14 * w
        elif name in ("tugboat", "speedboat", "skiff", "fishing-trawler"):
            # Local impact to the port quarter; deform the shell and attached rail together.
            w = math.exp(-(((nx + 0.20) / 0.185) ** 2)) * min(
                1, max(0, (ny - 0.08) / 0.31)
            )
            h = math.exp(-(((nz - 0.32) / 0.24) ** 2))
            v.co.y -= width * 0.22 * w * h
            v.co.z -= height * 0.11 * w * min(1, max(0, nz / 0.21))
        else:
            # Localized roof/bonnet crushing with a small lateral fold.
            w = math.exp(
                -(((nx - 0.10) / 0.24) ** 2) - ((ny - 0.16) / 0.30) ** 2
            ) * min(1, max(0, (nz - 0.38) / 0.43))
            v.co.z -= height * 0.38 * w
            v.co.y -= width * 0.07 * w
    o.data.update()
    export("_Storm")
