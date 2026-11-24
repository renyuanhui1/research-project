"""Clip the outer land and build the matching tidal water and rock faces in survey metres."""

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient


def survey_perimeter(meshes, boundary):
    level = -2.65
    shore_cut = level + 0.15
    contour = {}
    walls = []
    exterior = boundary.exterior
    for asset in ("TidalShelf", "CoastBeyondTown"):
        for material, triangles in meshes[asset].items():
            kept = []
            for tri in triangles:
                if material == "Mud":
                    for a, b in zip(tri, tri[1:] + tri[:1]):
                        if (a[2] < shore_cut) != (b[2] < shore_cut):
                            t = (shore_cut - a[2]) / (b[2] - a[2])
                            contour[round(a[0] + t * (b[0]-a[0]), 6)] = a[1] + t * (b[1]-a[1])
                if asset != "CoastBeyondTown":
                    continue
                footprint = Polygon([p[:2] for p in tri])
                if boundary.covers(footprint):
                    kept.append(tri)
                    continue
                cut = boundary.intersection(footprint)
                if cut.area < 1e-8:
                    continue
                a, b, c = np.array(tri)
                normal = np.cross(b-a, c-a)
                def point(x, y):
                    return (x, y, a[2] - (normal[0]*(x-a[0]) + normal[1]*(y-a[1])) / normal[2])
                for piece in shapely.get_parts(cut):
                    if piece.geom_type != "Polygon" or piece.area < 1e-8:
                        continue
                    for face in shapely.get_parts(shapely.constrained_delaunay_triangles(piece)):
                        kept.append([point(x,y) for x,y in list(face.exterior.coords)[:3]])
                    ring = list(orient(piece, sign=1).exterior.coords)
                    for start, end in zip(ring, ring[1:]):
                        mid = Point((start[0]+end[0])/2, (start[1]+end[1])/2)
                        if exterior.distance(mid) < 1e-6:
                            walls.append([(*start,-6), (*end,-6), point(*end), point(*start)])
            if asset == "CoastBeyondTown":
                meshes[asset][material] = kept

    coast = sorted(contour.items())
    south = boundary.bounds[1] - 10
    sea = orient(boundary.intersection(Polygon([*coast, (coast[-1][0],south),
                                                (coast[0][0],south)])), sign=1)
    water = []
    for face in shapely.get_parts(shapely.constrained_delaunay_triangles(sea)):
        coords = list(orient(face,sign=1).exterior.coords)[:3]
        water.append([(*p,level) for p in coords])
        water.append([(*p,level-1.2) for p in reversed(coords)])
    for a,b in zip(sea.exterior.coords, list(sea.exterior.coords)[1:]):
        water.append([(*a,level-1.2),(*b,level-1.2),(*b,level),(*a,level)])
    meshes["CoastalWater"]["Estuary"] = water
    meshes["TerrainCutFaces"]["CoastalCutRock"] = walls
    return sea.bounds
