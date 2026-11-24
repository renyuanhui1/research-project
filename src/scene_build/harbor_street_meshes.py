"""Blender 3.6 street architecture. Pass the persistent SourceArt streets directory after --."""

import math
import sys
import random
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from harbor_meshes import Model, WHITE, GLASS, STEEL, WOOD, PAINT

TRIM = (0.87, 0.85, 0.76)
STONE = (0.57, 0.54, 0.49)
BRICK = (0.57, 0.31, 0.23)
SASH = (0.29, 0.38, 0.35)
SLATE = (0.32, 0.35, 0.37)
MOSS = (0.51, 0.57, 0.50)
CREAM = (0.85, 0.76, 0.60)


def beam(m, a, b, w, h, color):
    a, b = Vector(a), Vector(b)
    d = b - a
    angle = d.to_track_quat("Z", "Y").to_euler()
    m.box((a + b) / 2, (w, h, d.length), color, tuple(map(math.degrees, angle)))


def sash(m, x, y, z, w, h, shutters=False):
    side = 1 if y > 0 else -1
    m.box((x, y, z), (w + 0.22, 0.17, h + 0.20), TRIM)
    m.box((x, y + side * 0.10, z), (w, 0.08, h), GLASS)
    for dx in (-w * 0.25, w * 0.25):
        m.box((x + dx, y + side * 0.16, z), (0.045, 0.055, h), TRIM)
    m.box((x, y + side * 0.16, z + 0.08), (w, 0.055, 0.065), TRIM)
    m.box((x, y + side * 0.12, z - h / 2 - 0.11), (w + 0.37, 0.36, 0.12), STONE)
    if shutters:
        for sign in (-1, 1):
            xx = x + sign * (w / 2 + 0.31)
            m.box((xx, y, z), (0.40, 0.19, h + 0.08), SASH)
            for k in range(7):
                m.box(
                    (xx, y + side * 0.13, z - h * 0.41 + k * h * 0.134),
                    (0.33, 0.075, 0.05),
                    MOSS,
                )


def door(m, x, y, z=0.24):
    m.box((x, y, z + 1.11), (1.18, 0.17, 2.22), TRIM)
    m.box((x, y - 0.12, z + 1.07), (0.95, 0.12, 2.14), SASH)
    m.box((x, y - 0.20, z + 1.52), (0.65, 0.06, 0.78), GLASS)
    m.box((x + 0.31, y - 0.26, z + 1.01), (0.10, 0.10, 0.055), WHITE)


def roof(m, cx, cy, w, d, eave, rise, color, hip=False):
    # An actual pitched envelope, verge flashing, gutters and standing seams.
    w, d = w / 2, d / 2
    inset = min(d * 0.8, w * 0.65) if hip else 0
    p = [
        (cx - w, cy - d, eave),
        (cx + w, cy - d, eave),
        (cx + w, cy + d, eave),
        (cx - w, cy + d, eave),
        (cx - w + inset, cy, eave + rise),
        (cx + w - inset, cy, eave + rise),
    ]
    faces = [(0, 1, 5, 4), (3, 4, 5, 2), (0, 4, 3), (1, 2, 5)]
    for i, face in enumerate(faces):
        m.part(p, [face, tuple(reversed(face))], color if hip or i < 2 else CREAM)
    for a, b in [(0, 1), (2, 3), (4, 5), (0, 4), (3, 4), (1, 5), (2, 5)]:
        beam(m, p[a], p[b], 0.12, 0.13, STEEL)
    for side in (-1, 1):
        for k in range(1, int(w * 2 / 0.65)):
            x = cx - w + k * 0.65
            t = min(1, (w - abs(x - cx)) / inset) if hip else 1
            beam(
                m,
                (x, cy + side * d, eave + 0.025),
                (x, cy + side * d * (1 - t), eave + rise * t + 0.025),
                0.035,
                0.045,
                color,
            )


def chimney(m, x, y, z):
    m.box((x, y, z), (0.86, 0.73, 2.4), BRICK)
    for i in range(7):
        m.box((x, y, z - 0.95 + i * 0.29), (0.88, 0.75, 0.028), STONE)
    m.box((x, y, z + 1.15), (1.02, 0.92, 0.16), STONE)
    for dx in (-0.21, 0.21):
        m.tube((x + dx, y, z + 1.43), 0.15, 0.42, BRICK)


def house(name, units, h, hip, wall, roof_color, main_roof=roof):
    m = Model()
    w, d = units * 6.1 + (2 if units == 1 else 0), 8.2 if hip else 7.6
    m.box((0, 0, 0.26), (w + 0.26, d + 0.20, 0.52), STONE)
    m.box((0, 0, (h + 0.45) / 2), (w, d, h - 0.45), wall)
    for y in (-d / 2, d / 2):
        for z in ([0.53, 3.12, h - 0.16] if h > 5 else [0.53, h - 0.16]):
            m.box((0, y, z), (w + 0.16, 0.25, 0.14), TRIM)
        for x in (-w / 2 + 0.14, w / 2 - 0.14):
            m.box((x, y, h / 2), (0.26, 0.23, h), TRIM)
    main_roof(m, 0, 0, w + 0.85, d + 0.85, h, 2.35, roof_color, hip)
    for i in range(units):
        x = (i - (units - 1) / 2) * 6.1
        front, back = -d / 2 - 0.10, d / 2 + 0.10
        door(m, x + 1.55, front)
        sash(m, x - 1.05, front, 1.64, 1.72, 1.43, units != 3)
        sash(m, x, back, 1.7, 1.6, 1.4)
        if h > 5:
            for offset in (-1.1, 1.52):
                sash(m, x + offset, front, 4.53, 1.29, 1.54)
            sash(m, x, back, 4.55, 1.52, 1.47)
        for j in range(3):
            m.box(
                (x + 1.55, front - 1.45 - j * 0.28, 0.24 - j * 0.07),
                (2.20, 0.65, 0.46 - j * 0.14),
                STONE,
            )
        m.box((x + 1.55, front - 0.77, 0.40), (2.35, 1.5, 0.16), STONE)
        for dx in (-0.97, 0.97):
            m.box((x + 1.55 + dx, front - 1.38, 1.55), (0.12, 0.13, 2.28), TRIM)
        roof(m, x + 1.55, front - 0.74, 2.65, 1.95, 2.67, 0.48, roof_color, True)
        if i % 2 == 0:
            chimney(m, x - 1.85, 0.85, h + 1.95)
        if units == 2:
            m.box((x - 0.92, -d * 0.24, h + 0.99), (2.1, 1.72, 1.8), wall)
            sash(m, x - 0.92, -d * 0.24 - 0.9, h + 1.38, 1.18, 0.84)
            roof(m, x - 0.92, -d * 0.24, 2.38, 2.06, h + 1.94, 0.66, roof_color)
    # End-wall openings run perpendicular to the main frontage.
    end = Model()
    sash(end, 0, -0.11, 1.68, 1.5, 1.45)
    if h > 5:
        sash(end, 0, -0.11, 4.56, 1.28, 1.4)
    for side in (-1, 1):
        pts = [(side * (w / 2 - y), side * x, z) for x, y, z in end.points]
        m.part(pts, [], WHITE)
        start = len(m.points) - len(pts)
        for face, c in zip(end.faces, end.colors):
            m.faces.append(tuple(start + j for j in face))
            m.colors.append(c)
    m.save("House" + name)


def letters(m, label, x, y, z, width):
    curve = bpy.data.curves.new("ShopLettering", "FONT")
    curve.body = label
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.extrude = 0.008
    ob = bpy.data.objects.new("Lettering", curve)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.update()
    size = width / max(ob.dimensions.x, 0.01)
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    pts = [
        (x + v.co.x * size, y - v.co.z * size, z + v.co.y * size) for v in mesh.vertices
    ]
    m.part(pts, [tuple(t.vertices) for t in mesh.loop_triangles], WHITE)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(curve)


def shop(name, sign, accent, damage=False):
    m = Model()
    w, d, h = 14, 8, 6.8
    m.box((0, 0, h / 2), (w, d, h), CREAM)
    for sy in (-1, 1):
        for x in (-5.3, -1.8, 1.8, 5.3):
            sash(m, x, sy * 4.09, 4.95, 1.55, 1.65, True)
    m.box((0, 0, h + 0.12), (w + 0.45, d + 0.45, 0.24), STONE)
    for x in (-w / 2, w / 2):
        m.box((x, 0, h + 0.38), (0.24, d + 0.3, 0.65), CREAM)
    for y in (-4, 4):
        m.box((0, y, h + 0.4), (w + 0.25, 0.27, 0.68), CREAM)
        m.box((0, y, h + 0.76), (w + 0.5, 0.45, 0.12), STONE)
    m.box((2.5, 1, h + 0.52), (1.8, 1.1, 0.7), STEEL)
    m.box((0, -4.12, 2.94), (13.75, 0.32, 0.82), accent)
    letters(m, sign, 0, -4.31, 2.95, min(8.2, len(sign) * 0.41))
    for x in (-4.37, 4.37):
        sash(m, x, -4.12, 1.43, 4.62, 1.92)
    door(m, 0, -4.16, 0.12)
    for x in (-6.77, -1.46, 1.46, 6.77):
        m.box((x, -4.18, 1.47), (0.25, 0.34, 2.88), accent)
    for j in range(20):
        x = -6.46 + j * 0.68
        # Failed bolts leave one awning bay hanging by its inside edge.
        drop = max(0, (j - 13) / 6) if damage else 0
        angle = 9 + 48 * drop
        proj = 1.55 * math.cos(math.radians(angle))
        z = 2.56 - 0.775 * math.sin(math.radians(angle)) - 0.20 * drop
        c = accent if j % 2 else TRIM
        m.box((x, -4.24 - proj / 2, z), (0.683, 1.55, 0.07), c, (angle, 0, 0))
        m.box(
            (x, -4.24 - proj, z - 0.775 * math.sin(math.radians(angle)) - 0.09),
            (0.68, 0.06, 0.18),
            c,
        )
    for x in (-6.5, -1.6, 1.6, 6.5):
        if damage and x > 5:
            continue
        beam(m, (x, -4.20, 1.98), (x, -5.63, 2.31), 0.048, 0.05, STEEL)
    m.save("Shop" + name)


def boundary(name, width):
    m = Model()
    for x0, x1 in [(-width / 2, -1.5), (1.5, width / 2 - 3.2)]:
        m.box(((x0 + x1) / 2, 0, 0.17), (x1 - x0, 0.27, 0.34), STONE)
        for z in (0.48, 0.87):
            m.box(((x0 + x1) / 2, 0, z), (x1 - x0, 0.09, 0.08), SASH)
        n = math.ceil((x1 - x0) / 0.30)
        for j in range(n):
            x = x0 + j * (x1 - x0) / n
            m.box((x, 0, 0.69), (0.068, 0.10, 0.74), TRIM)
        for x in (x0, x1):
            m.box((x, 0, 0.55), (0.36, 0.40, 1.1), STONE)
            m.box((x, 0, 1.11), (0.44, 0.46, 0.12), TRIM)
    m.save(name)


def garden_shed():
    m = Model()
    m.box((0, 0, 0.10), (4.25, 3.1, 0.20), STONE)
    m.box((0, 0, 1.27), (4.1, 3, 2.28), WOOD)
    roof(m, 0, 0, 4.55, 3.45, 2.42, 0.82, SLATE)
    sash(m, -1, -1.57, 1.40, 1.1, 0.86)
    door(m, 0.83, -1.54, 0.12)
    for x in (-1.94, 1.94):
        m.box((x, -1.52, 1.25), (0.14, 0.15, 2.35), TRIM)
    m.save("GardenWorkshop")


def market_stall():
    m = Model()
    for x in (-1.6, 1.6):
        for y in (-0.47, 0.47):
            m.box((x, y, 0.43), (0.09, 0.09, 0.86), STEEL)
    m.box((0, 0, 0.84), (3.6, 1.17, 0.14), WOOD)
    colors = [(0.27, 0.39, 0.43), (0.63, 0.66, 0.58), (0.32, 0.39, 0.31)]
    rng = random.Random(812)
    for j in range(3):
        x = (j - 1) * 1.10
        m.box((x, 0, 0.94), (1.04, 0.86, 0.13), STONE)
        for side in (-1, 1):
            m.box((x, side * 0.43, 1.05), (1.07, 0.06, 0.22), WOOD)
            m.box((x + side * 0.51, 0, 1.05), (0.06, 0.86, 0.22), WOOD)
        for k in range(5):
            xx = x - 0.35 + k * 0.17
            yy = rng.uniform(-0.05, 0.05)
            p = [
                (xx, yy - 0.29, 1.04),
                (xx - 0.068, yy, 1.05),
                (xx, yy + 0.26, 1.04),
                (xx + 0.068, yy, 1.05),
                (xx, yy, 1.12),
                (xx - 0.055, yy - 0.35, 1.04),
                (xx + 0.055, yy - 0.35, 1.04),
            ]
            m.part(
                p,
                [(0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0), (5, 6, 0)],
                colors[(j + k) % 3],
            )
    m.save("QuayFishStall")


def terrace_cafe():
    m = Model()
    for cx in (-1.35, 1.35):
        m.tube((cx, 0, 0.77), 0.49, 0.075, WOOD, 24)
        m.tube((cx, 0, 0.40), 0.055, 0.75, STEEL)
        m.tube((cx, 0, 0.06), 0.25, 0.07, STEEL)
        for side in (-1, 1):
            y = side * 0.77
            m.box((cx, y, 0.45), (0.40, 0.4, 0.075), WOOD)
            m.box((cx, y + side * 0.19, 0.71), (0.40, 0.06, 0.38), WOOD)
            for dx in (-0.16, 0.16):
                for dy in (-0.16, 0.16):
                    m.box((cx + dx, y + dy, 0.23), (0.036, 0.036, 0.44), STEEL)
    m.save("BakeryTables")


if __name__ == "__main__":
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    house("HarborTerrace", 3, 5.95, False, CREAM, SLATE)
    house("DormerPair", 2, 5.55, False, MOSS, PAINT[0])
    house("HippedCottage", 1, 3.24, True, (0.79, 0.70, 0.60), PAINT[3])
    shop("FishMarket", "TIDELINE FISH CO.", (0.18, 0.36, 0.36), True)
    shop("Chandlery", "HARBOUR SUPPLY", (0.32, 0.36, 0.48))
    shop("Bakery", "QUAY BAKERY", (0.52, 0.30, 0.21))
    boundary("GardenRail", 18)
    garden_shed()
    market_stall()
    terrace_cafe()
