"""Build the coastal town mesh library with Blender 3.6: --background --python FILE -- OUTPUT."""

import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

OUT = Path(sys.argv[sys.argv.index("--") + 1])
OUT.mkdir(parents=True, exist_ok=True)
WHITE = (0.80, 0.81, 0.75)
GLASS = (0.09, 0.23, 0.27)
STEEL = (0.23, 0.27, 0.27)
WOOD = (0.44, 0.32, 0.20)
DARK = (0.075, 0.085, 0.095)
PAINT = [
    (0.65, 0.28, 0.19),
    (0.22, 0.42, 0.47),
    (0.60, 0.53, 0.32),
    (0.31, 0.37, 0.42),
    (0.51, 0.60, 0.53),
]
CUBE_FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


class Model:
    def __init__(self):
        self.points, self.faces, self.colors = [], [], []

    def part(self, points, faces, color):
        start = len(self.points)
        self.points.extend(points)
        for face in faces:
            self.faces.append(tuple(start + i for i in face))
            self.colors.append(color)

    def box(self, center, size, color, rotation=(0, 0, 0)):
        rot = Euler(tuple(math.radians(v) for v in rotation)).to_matrix()
        signs = [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ]
        self.part(
            [
                Vector(center)
                + rot @ Vector(tuple(s * d / 2 for s, d in zip(sign, size)))
                for sign in signs
            ],
            CUBE_FACES,
            color,
        )

    def tube(self, center, radius, height, color, sides=12, axis="z"):
        points = []
        for end in (-height / 2, height / 2):
            for i in range(sides):
                a = i * math.tau / sides
                p = (radius * math.cos(a), radius * math.sin(a), end)
                if axis == "y":
                    p = (p[0], p[2], -p[1])
                points.append(tuple(v + c for v, c in zip(p, center)))
        faces = [tuple(reversed(range(sides))), tuple(range(sides, sides * 2))]
        faces.extend(
            (i, (i + 1) % sides, sides + (i + 1) % sides, sides + i)
            for i in range(sides)
        )
        self.part(points, faces, color)

    def save(self, name):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(self.points, [], self.faces)
        mesh.update()
        uv = mesh.uv_layers.new(name="MetricUV")
        slots = {}
        for poly, color in zip(mesh.polygons, self.colors):
            material = "Paint"
            if color == WOOD:
                material = "Timber"
            elif color == GLASS:
                material = "Glass"
            elif color == DARK:
                material = "Rubber"
            elif name.startswith(("Shop", "House", "Shed", "Civic")):
                if color in PAINT:
                    material = "Roof"
                elif color not in (WHITE, STEEL):
                    material = "Plaster"
            if material not in slots:
                slots[material] = len(mesh.materials)
                mesh.materials.append(
                    bpy.data.materials.get(material) or bpy.data.materials.new(material)
                )
            poly.material_index = slots[material]
            axis = max(range(3), key=lambda i: abs(poly.normal[i]))
            axes = [(1, 2), (0, 2), (0, 1)][axis]
            for li in poly.loop_indices:
                pos = mesh.vertices[mesh.loops[li].vertex_index].co
                uv.data[li].uv = (pos[axes[0]] / 2, pos[axes[1]] / 2)
        attr = mesh.color_attributes.new(
            name="Color", type="BYTE_COLOR", domain="CORNER"
        )
        for poly, color in zip(mesh.polygons, self.colors):
            for j in poly.loop_indices:
                attr.data[j].color_srgb = (*color, 1)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.export_scene.fbx(
            filepath=str(OUT / (name + ".fbx")),
            use_selection=True,
            axis_forward="-Y",
            axis_up="Z",
            colors_type="SRGB",
            use_mesh_modifiers=False,
            add_leaf_bones=False,
        )
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def building(kind, variant, damaged):
    m = Model()
    w, d, h = {
        "House": (12, 9, 5),
        "Shop": (16, 10, 6),
        "Shed": (31, 19, 9),
        "Civic": (23, 15, 10),
    }[kind]
    w += (variant % 3) * 1.7
    h += (variant % 2) * 2.4
    wall = [
        (0.78, 0.72, 0.61),
        (0.68, 0.74, 0.72),
        (0.79, 0.67, 0.56),
        (0.62, 0.66, 0.67),
        (0.77, 0.77, 0.70),
    ][variant]
    roof = PAINT[variant]
    m.box((0, 0, 0.17), (w + 0.6, d + 0.6, 0.34), STEEL)
    for sy in (-1, 1):
        m.box((0, sy * d / 2, h / 2), (w, 0.23, h), wall)
        for j in range(1, int(w / 3)):
            x = -w / 2 + j * 3.1
            if abs(x) < 1.7 and sy == -1:
                continue
            for z in [2.5, h - 2] if h > 7 else [2.6]:
                m.box((x, sy * (d / 2 + 0.16), z), (1.75, 0.14, 1.55), WHITE)
                m.box((x, sy * (d / 2 + 0.25), z + 0.04), (1.47, 0.07, 1.26), GLASS)
                m.box((x, sy * (d / 2 + 0.30), z + 0.04), (0.07, 0.06, 1.26), WHITE)
    for sx in (-1, 1):
        m.box((sx * w / 2, 0, h / 2), (0.23, d, h), wall)
    m.box((0, -d / 2 - 0.18, 1.45), (2.25, 0.16, 2.9), WOOD)
    if kind == "Shed":
        m.box((0, -d / 2 - 0.32, 2.8), (w * 0.39, 0.16, 5.6), STEEL)
    angle = 17 if kind in ("House", "Shed") else 4
    rad = math.radians(angle)
    for sx in (-1, 1):
        x = sx * w / 2
        points = [(x, -d / 2, h), (x, d / 2, h), (x, 0, h + d / 2 * math.tan(rad))]
        m.part(points, [(0, 1, 2), (2, 1, 0)], wall)
    panels = 6
    width = (w + 1.2) / panels
    for side in (-1, 1):
        for j in range(panels):
            x = -width * (panels - 1) / 2 + j * width
            y = side * (d + 0.95) / 4
            z = h + (d - 0.95) / 4 * math.tan(rad)
            if damaged and side == -1 and j in (3, 4) and kind != "Shop":
                if j == 3:
                    # Folds grow toward the unsecured eave while the upper edge stays fastened.
                    for ix in range(24):
                        for iy in range(9):
                            corners = []
                            for k, l in (
                                (ix, iy),
                                (ix + 1, iy),
                                (ix + 1, iy + 1),
                                (ix, iy + 1),
                            ):
                                u, v = k / 24, l / 9
                                ridge = 0.035 * math.cos(k * math.pi)
                                lift = v**2 * (1.5 + 1.2 * u) + 0.14 * v * math.sin(
                                    5 * u
                                )
                                torn_edge = 0.16 * math.sin(8 * k) * v**6
                                yy = -v * (d / 2 + 0.35) + v**3 * (0.9 + u * 0.65)
                                zz = (
                                    h
                                    + (d / 2 + yy) * math.tan(rad)
                                    + lift
                                    + ridge
                                    + torn_edge
                                )
                                corners.append(
                                    (x + (u - 0.5) * width + 0.18 * v * u, yy, zz)
                                )
                            m.part(corners, [(3, 2, 1, 0)], roof)
                    for yy in (0.32, 0.69):
                        m.box(
                            (
                                x + width * 0.55,
                                -d * 0.5 * yy,
                                h + (1 - yy) * d * 0.5 * math.tan(rad) - 0.22,
                            ),
                            (2.1 * width, 0.12, 0.15),
                            STEEL,
                        )
                continue
            m.box(
                (x, y, z),
                (width - 0.025, (d + 0.95) / 2 / math.cos(rad), 0.20),
                roof,
                (-side * angle, 0, 0),
            )
            m.box(
                (x - width * 0.48, y, z + 0.11),
                (0.07, (d + 0.95) / 2 / math.cos(rad), 0.06),
                STEEL,
                (-side * angle, 0, 0),
            )
    if kind == "Shop":
        for j in range(5):
            x = (j - 2) * w / 5
            pane_width = w / 5 - 0.3
            if j == 2:
                # Two sidelights flank the 2.25 m door. Glazing over the door
                # would put two outward-facing surfaces at the same depth.
                opening = 2.31
                sidelight = (pane_width - opening) / 2
                for side in (-1, 1):
                    cx = side * (opening + sidelight) / 2
                    m.box((cx, -d / 2 - 0.2, 2.0), (sidelight, 0.12, 2.8), GLASS)
            else:
                m.box((x, -d / 2 - 0.2, 2.0), (pane_width, 0.12, 2.8), GLASS)
            if j != 1 or not damaged:
                m.box((x, -d / 2 - 1.35, 3.9), (w / 5, 2.65, 0.18), roof, (9, 0, 0))
        if damaged:
            m.box((-w / 5, -d / 2 - 2.5, 0.8), (w / 5, 2.65, 0.16), roof, (33, 8, 5))
    if damaged and kind != "Shop":
        for j in (2, 3, 4, 5):
            x = -w / 2 + j * w / 6
            m.box(
                (x, -d * 0.24, h + d * 0.065),
                (0.15, d * 0.50, 0.23),
                WOOD,
                (angle, 0, 0),
            )
        for j in range(5):
            m.box(
                (w * 0.13 + j * 0.14, -d * 0.27, h + 0.35 + j * 0.09),
                (0.075, d * 0.35, 0.07),
                STEEL,
                (26, 0, -4),
            )
    if kind == "House":
        m.box((-w * 0.28, d * 0.17, h + d * 0.08), (0.78, 0.88, 2.4), WOOD)
    return m


def vessel(kind, variant, damaged):
    m = Model()
    length, beam = {
        "Fishing": (13, 4.1),
        "Launch": (7, 2.3),
        "Workboat": (16, 5.1),
        "Coaster": (31, 7.4),
    }[kind]
    length *= 1 + 0.08 * variant
    height = beam * 0.49
    paint = PAINT[variant]
    rings = []
    for x, width in [(-0.49, 0.78), (-0.35, 1), (0.12, 1), (0.32, 0.78), (0.5, 0.05)]:
        rings.append(
            [
                (x * length, -width * beam * 0.50, height),
                (x * length, -width * beam * 0.31, -0.1),
                (x * length, 0, -height * 0.22),
                (x * length, width * beam * 0.31, -0.1),
                (x * length, width * beam * 0.50, height),
            ]
        )
    for j, (a, b) in enumerate(zip(rings, rings[1:])):
        for k in range(4):
            if damaged and j == 1 and k == 0:
                mid = tuple((p + q) / 2 for p, q in zip(a[k], b[k]))
                m.part(
                    [
                        a[k],
                        a[k + 1],
                        b[k + 1],
                        b[k],
                        (mid[0], mid[1] + 0.35, mid[2] - 0.7),
                    ],
                    [(0, 1, 4), (1, 2, 4), (2, 3, 4)],
                    paint,
                )
            else:
                m.part([a[k], a[k + 1], b[k + 1], b[k]], [(0, 1, 2, 3)], paint)
        m.part([a[0], b[0], b[4], a[4]], [(0, 1, 2, 3)], WOOD)
    for idx in (0, -1):
        m.part(
            rings[idx],
            [tuple(range(5)) if idx == 0 else tuple(reversed(range(5)))],
            paint,
        )
    cab = beam * 0.56
    m.box(
        (length * 0.04, 0, height + cab * 0.42),
        (length * 0.26, beam * 0.63, cab * 0.84),
        WHITE,
    )
    m.box(
        (length * 0.04, 0, height + cab * 0.78),
        (length * 0.24, beam * 0.65, cab * 0.33),
        GLASS,
    )
    m.box((length * 0.04, 0, height + cab), (length * 0.29, beam * 0.70, 0.17), paint)
    m.tube((length * 0.05, 0, height + cab + beam * 0.34), 0.07, beam * 0.68, STEEL)
    if kind != "Launch":
        m.tube((-length * 0.30, 0, height + beam * 0.46), 0.13, beam * 0.92, STEEL)
        m.box(
            (-length * 0.30, 0, height + beam * 0.85), (0.18, beam * 0.92, 0.19), WHITE
        )
        for sy in (-1, 1):
            m.box(
                (-length * 0.18, sy * beam * 0.44, height + 0.34),
                (length * 0.48, 0.09, 0.11),
                WHITE,
            )
            for j in range(4):
                m.tube(
                    (length * (-0.39 + j * 0.13), sy * beam * 0.44, height + 0.17),
                    0.055,
                    0.34,
                    WHITE,
                )
    # Rub rail follows the sheer. Fenders, bollards and cabin mullions remain visible close up.
    for sy in (-1, 1):
        for ii in range(5):
            x = length * (-0.36 + ii * 0.116)
            m.tube((x, sy * beam * 0.49, height * 0.79), 0.14, 0.61, DARK)
        for j in range(4):
            m.box(
                (length * (-0.065 + j * 0.068), sy * beam * 0.333, height + cab * 0.78),
                (0.07, 0.07, cab * 0.35),
                WHITE,
            )
        for xx in (-0.38, 0.29):
            m.tube((xx * length, sy * 0.32 * beam, height + 0.15), 0.10, 0.28, STEEL)
    if kind in ("Fishing", "Workboat"):
        for j in range(6):
            m.box(
                (
                    (-0.33 + j % 3 * 0.066) * length,
                    (-0.18 + j // 3 * 0.23) * beam,
                    height + 0.21,
                ),
                (0.72, 0.55, 0.4),
                (0.35, 0.39, 0.27),
            )
        m.tube((-0.20 * length, 0, height + 0.45), 0.37, 0.87, STEEL, 18, "y")
        for j in range(9):
            m.tube(
                (-0.20 * length, -0.38 + j * 0.095, height + 0.45),
                0.39,
                0.044,
                WOOD,
                18,
                "y",
            )
        for sy in (-1, 1):
            m.box(
                (-0.23 * length, sy * beam * 0.31, height + beam * 0.47),
                (0.105, 0.11, beam * 1.04),
                STEEL,
                (sy * 15, -18, 0),
            )
    return m


def vehicle(kind, variant, damaged):
    m = Model()
    length, width = {
        "Sedan": (4.6, 1.85),
        "Van": (5.4, 2),
        "Truck": (8.5, 2.55),
        "Utility": (4.1, 2.05),
        "Rescue": (5.7, 2.15),
    }[kind]
    tire = width * 0.196
    paint = (0.82, 0.25, 0.14) if kind == "Rescue" else PAINT[variant]
    m.box((0, 0, tire + 0.30), (length, width, 0.62), paint)
    x = length * (0.20 if kind == "Truck" else 0.025)
    roof_z = 1.48 if kind == "Sedan" else 2.07
    if damaged:
        roof_z -= 0.45
    cab_len = length * (0.25 if kind == "Truck" else 0.53)
    # Tapered sides and sloping windshield.
    pts = []
    for z, l, w, off in [
        (tire + 0.55, cab_len, width * 0.95, 0),
        (roof_z, cab_len * 0.76, width * 0.78, -0.10),
    ]:
        pts.extend(
            (x + off + sx * l / 2, sy * w / 2, z)
            for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        )
    m.part(pts, CUBE_FACES, GLASS)
    m.box(
        (x - 0.1, 0, roof_z + 0.045),
        (cab_len * 0.77, width * 0.80, 0.11),
        paint,
        (0, 5 if damaged else 0, 0),
    )
    for sy in (-1, 1):
        m.box((x, sy * width * 0.43, roof_z - 0.22), (0.13, 0.11, 0.48), paint)
        for sx in (-1, 1):
            xx = sx * length * 0.32
            m.tube((xx, sy * width * 0.46, tire), tire, 0.24, DARK, 16, "y")
            m.tube(
                (xx, sy * (width * 0.46 + 0.13), tire),
                tire * 0.51,
                0.04,
                STEEL,
                12,
                "y",
            )
        m.box((length / 2 + 0.025, sy * width * 0.34, 0.73), (0.07, 0.33, 0.18), WHITE)
        m.box(
            (-length / 2 - 0.025, sy * width * 0.34, 0.76),
            (0.07, 0.26, 0.17),
            (0.68, 0.13, 0.10),
        )
    if damaged:
        for j in range(4):
            m.box(
                (length * 0.32 + j * 0.055, 0, tire + 0.65 + j * 0.026),
                (0.115, width * 0.90, 0.075),
                paint,
                (0, -9 + j * 6, 0),
            )
        m.box(
            (-length * 0.18, width * 0.51, tire + 0.26),
            (length * 0.32, 0.09, 0.16),
            STEEL,
            (7, 6, -2),
        )
    if kind == "Truck":
        m.box((-length * 0.19, 0, 1.75), (length * 0.53, width, 1.8), WHITE)
    if kind == "Rescue":
        m.box((x, 0, roof_z + 0.16), (0.22, 0.83, 0.15), (0.2, 0.4, 0.8))
    return m


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for maker, kinds, n in [
        (building, ("House", "Shop", "Shed", "Civic"), 5),
        (vessel, ("Fishing", "Launch", "Workboat", "Coaster"), 2),
        (vehicle, ("Sedan", "Van", "Truck", "Utility", "Rescue"), 3),
    ]:
        for kind in kinds:
            for i in range(n):
                maker(kind, i, False).save(f"{kind}{i}")
            maker(kind, 0, True).save(kind + "Damaged")

    # A tidal beach, with a sheltered quay in the west and a broad exposed mudflat eastwards.
    def shore(x):
        return -265 - 60 * math.sin(x / 190) - 26 * math.sin(x / 71)

    for name, start, finish in (("Foreshore", 0, 1), ("Backshore", 1, 2)):
        m = Model()
        for j in range(200):
            pts = []
            for x in (-1000 + 10 * j, -990 + 10 * j):
                ys = [shore(x) - 32, -150, 1000]
                zs = [-4.5, 0, 0]
                pts.extend([(x, ys[k], zs[k]) for k in (start, finish)])
            m.part(pts, [(0, 2, 3, 1)], WHITE)
        m.save(name)
    # Windrows of seaweed and driftwood at the high-water line.
    m = Model()
    rng = random.Random(181)
    for j in range(38):
        x, y = rng.uniform(-8, 8), rng.uniform(-1.1, 1.1)
        m.box(
            (x, y, 0.06),
            (rng.uniform(0.23, 1.9), rng.uniform(0.06, 0.19), rng.uniform(0.045, 0.14)),
            WOOD,
            (0, rng.uniform(-4, 4), rng.uniform(-32, 32)),
        )
    m.save("Wrack")
