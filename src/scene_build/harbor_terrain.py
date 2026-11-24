"""Additional shoreline, rock and hinterland geometry, in Blender meters."""

import bpy
import math
import random
import sys
from pathlib import Path
from mathutils import Vector

dest = Path(sys.argv[sys.argv.index("--") + 1])
dest.mkdir(exist_ok=True)


def write(name, vs, fs):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vs, [], fs)
    mesh.update()
    actor = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(actor)
    bpy.context.view_layer.objects.active = actor
    actor.select_set(True)
    bpy.ops.export_scene.fbx(
        filepath=str(dest / (name + ".fbx")),
        use_selection=True,
        axis_forward="-Y",
        axis_up="Z",
        add_leaf_bones=False,
    )


rng = random.Random(266)
for j in range(4):
    vs = []
    fs = []
    for z, r in [(-0.4, 0.63), (-0.12, 1), (0.32, 0.80), (0.58, 0.40)]:
        for i in range(9):
            a = (i + 0.12 * rng.random()) * math.tau / 9
            d = r * rng.uniform(0.77, 1.18)
            vs.append(
                (d * math.cos(a), d * math.sin(a) * 0.85, z + rng.uniform(-0.11, 0.11))
            )
    fs.extend([tuple(reversed(range(9))), tuple(range(27, 36))])
    for l in range(3):
        for i in range(9):
            a = l * 9 + i
            b = l * 9 + (i + 1) % 9
            fs.extend([(a, b, b + 9), (a, b + 9, a + 9)])
    write("Armour" + str(j), vs, fs)
# Outlying relief surrounds the measured 2 km square and slopes down to its boundary.
# Export using UE Y convention, for placement at identity.
vs = []
fs = []
n = 120
for j in range(n + 1):
    y = -5000 + j * 10000 / n
    for i in range(n + 1):
        x = -5000 + i * 10000 / n
        coast = -240 + 95 * math.sin(x / 710) + 130 * math.sin(x / 1290)
        dist = max(abs(x) - 990, y - 990)
        relief = max(0, min(1, dist / 900))
        z = relief * (
            50
            + 43 * math.sin(x / 420 + y / 520)
            + 35 * math.sin(x / 370 - y / 630)
            + 20 * math.sin(y / 170)
        )
        if y < coast:
            z -= min(45, (coast - y) * 0.12)
        vs.append((x, -y, z - 0.9))
for j in range(n):
    for i in range(n):
        x = -5000 + (i + 0.5) * 10000 / n
        y = -5000 + (j + 0.5) * 10000 / n
        if abs(x) < 990 and y < 990:
            continue
        a = j * (n + 1) + i
        fs.append((a, a + n + 1, a + n + 2, a + 1))
write("FarCoast", vs, fs)
