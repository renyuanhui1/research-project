"""Cut hidden cladding behind coplanar painted trim, before the office's storm deformation."""

import bmesh
from collections import defaultdict
from mathutils import Vector


def area(poly):
    if len(poly) < 3:
        return 0.0
    return sum(
        a[0] * b[1] - a[1] * b[0] for a, b in zip(poly, poly[1:] + poly[:1])
    ) * 0.5


def half_plane(poly, a, b, sign):
    """Clip a convex polygon to one side of the directed line a -> b."""
    result = []
    dx, dy = b[0] - a[0], b[1] - a[1]
    prev = poly[-1]
    left = sign * (dx * (prev[1] - a[1]) - dy * (prev[0] - a[0]))
    for point in poly:
        right = sign * (dx * (point[1] - a[1]) - dy * (point[0] - a[0]))
        if (left < 0) != (right < 0):
            result.append(prev + (point - prev) * (left / (left - right)))
        if right >= 0:
            result.append(point)
        prev, left = point, right
    return result


def subtract(poly, mask):
    orientation = 1 if area(mask) > 0 else -1
    interior = poly
    outside = []
    for a, b in zip(mask, mask[1:] + mask[:1]):
        piece = half_plane(interior, a, b, -orientation)
        if abs(area(piece)) > 1e-8:
            outside.append(piece)
        interior = half_plane(interior, a, b, orientation)
        if abs(area(interior)) < 1e-8:
            return [poly]
    return outside


def clip_office_cladding(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()
    layers = list(bm.loops.layers.uv.values())
    names = [slot.name for slot in obj.material_slots]
    surfaces = defaultdict(list)
    candidates = []
    for f in bm.faces:
        name = names[f.material_index]
        if name not in ('Slot04', 'Slot05', 'Slot06'):
            continue
        axis = 0 if abs(f.normal.x) > abs(f.normal.y) else 1
        if abs(f.normal[axis]) < 0.99999:
            continue
        plane = sum(v.co[axis] for v in f.verts) / len(f.verts)
        key = (axis, round(plane, 4), f.normal[axis] > 0)
        points = [Vector((v.co[1 - axis], v.co.z)) for v in f.verts]
        is_wall = name == 'Slot04'
        surfaces[key, is_wall].append(points)
        candidates.append((f, key, points, is_wall))
    removed_area = 0.0
    for face, key, points, is_wall in candidates:
        axis, plane, positive = key
        # Keep the front of a trim piece. Where its back touches the facade,
        # remove the contacting back face instead of cutting the wall away.
        mask_key = (axis, plane, positive if is_wall else not positive)
        masks = surfaces[mask_key, not is_wall]
        if not masks:
            continue
        assert len(points) == 3
        pieces = [points]
        for mask in masks:
            if any(max(p[j] for p in mask) <= min(p[j] for p in points) or
                   min(p[j] for p in mask) >= max(p[j] for p in points) for j in (0, 1)):
                continue
            pieces = [q for part in pieces for q in subtract(part, mask)]
        diff = abs(area(points)) - sum(abs(area(q)) for q in pieces)
        if diff < 1e-7:
            continue
        removed_area += diff
        loops = list(face.loops)
        origin = points[0]
        ab, ac = points[1] - origin, points[2] - origin
        det = ab.cross(ac)
        cache = {}
        for piece in pieces:
            for i in range(1, len(piece)-1):
                triangle = [piece[0], piece[i], piece[i+1]]
                if abs(area(triangle)) < 1e-8:
                    continue
                verts, weights = [], []
                for p in triangle:
                    off = p - origin
                    u, v = off.cross(ac) / det, ab.cross(off) / det
                    w = [1-u-v, u, v]
                    weights.append(w)
                    point_key = tuple(round(k, 7) for k in p)
                    if point_key not in cache:
                        original = next((l.vert for l, q in zip(loops,points) if (q-p).length < 1e-6), None)
                        cache[point_key] = original or bm.verts.new(sum(
                            (l.vert.co * k for l,k in zip(loops,w)), Vector((0,0,0))))
                    verts.append(cache[point_key])
                if len(set(verts)) != 3:
                    continue
                patch = bm.faces.new(verts)
                patch.material_index, patch.smooth = face.material_index, face.smooth
                for l, w in zip(patch.loops,weights):
                    for layer in layers:
                        l[layer].uv = sum((src[layer].uv*k for src,k in zip(loops,w)),Vector((0,0)))
        bm.faces.remove(face)
    assert removed_area > 8, removed_area
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    print('OFFICE_CLADDING_CUT_M2', removed_area)
