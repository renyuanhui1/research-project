"""Build occupancy grids from saved UE collision samples and aircraft geometry."""
import json
from pathlib import Path
import sys
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()

import cv2
import numpy as np
import open3d as o3d

ROOT = project_path(PROFILE['map']['root'])
OUT = ROOT / 'grids'
FINE_RES = 0.25


def save_map(name, occupancy, origin, resolution):
    """Rows increase north; columns increase east; origin is the southwest corner."""
    path = OUT / name
    np.savez_compressed(path.with_suffix('.npz'), occupancy=occupancy.astype(np.int8),
                        origin_enu_m=np.array([*origin, 0.0]),
                        resolution_m=np.float64(resolution))
    pixels = np.full(occupancy.shape, 205, np.uint8)
    pixels[occupancy == 0] = 254
    pixels[occupancy == 100] = 0
    if not cv2.imwrite(str(path.with_suffix('.pgm')), pixels[::-1]):
        raise RuntimeError('Failed to write map image: ' + name)
    path.with_suffix('.yaml').write_text(
        f'image: {name}.pgm\nmode: trinary\nresolution: {resolution}\n'
        f'origin: [{origin[0]}, {origin[1]}, 0.0]\nnegate: 0\n'
        'occupied_thresh: 0.65\nfree_thresh: 0.196\n')
    return {'name': name, 'resolution_m': resolution,
            'origin_enu_m': [*map(float, origin), 0.0],
            'height': occupancy.shape[0], 'width': occupancy.shape[1],
            'occupied_cells': int((occupancy == 100).sum()),
            'free_cells': int((occupancy == 0).sum()),
            'unknown_cells': int((occupancy == -1).sum())}


def main():
    OUT.mkdir(exist_ok=True)
    targets = json.loads((ROOT / 'source/targets.json').read_text())
    bounds = json.loads((ROOT / 'object-bounds.json').read_text())
    with np.load(ROOT / 'global/occupancy_packed.npz') as data:
        shape = tuple(data['shape'])
        grid = np.unpackbits(data['packed'], bitorder=str(data['bitorder']))[
            :np.prod(shape)].reshape(shape).astype(bool)
        center = data['center_ned_m']
        res = float(data['resolution_m'])
    ne, nz, nn = shape
    east = center[1] + (np.arange(ne) - ne // 2) * res
    north = center[0] + (np.arange(nn) - nn // 2) * res
    altitude = -center[2] + (np.arange(nz) - nz // 2) * res
    origin = np.array([east[0] - res / 2, north[0] - res / 2])
    idx = np.argwhere(grid & (altitude[None, :, None] > 2.5))
    context = np.column_stack((east[idx[:, 0]], north[idx[:, 2]], altitude[idx[:, 1]]))
    for target in targets:
        bc = np.array(target['bounds_center_cm'])[[1, 0, 2]] / 100
        be = np.array(target['bounds_extent_cm'])[[1, 0, 2]] / 100
        context = context[~(np.abs(context - bc) <= be + res / 2).all(axis=1)]
    np.savez_compressed(OUT / 'context_voxels.npz', centers_enu_m=context,
                        resolution_m=np.float64(res))

    # Ground collision establishes mapped support; outside the airport stays unknown.
    occupancy = np.full((nn, ne), -1, np.int8)
    ground = grid[:, np.argmin(np.abs(altitude)), :].T
    occupancy[ground] = 0
    ij = np.floor((context[:, :2] - origin) / res).astype(int)
    occupancy[ij[:, 1], ij[:, 0]] = 100
    report = {'frame': 'world', 'axes': 'ENU meters; array rows north, columns east',
              'source': 'saved UE scene snapshot; not a new live survey',
              'global_above_ground_min_center_altitude_m': 2.5,
              'aircraft_occupancy': 'triangle-intersecting surface voxels; not interior filling',
              'context_voxels': len(context), 'targets': [], 'maps': []}

    for target in targets:
        name = target['name']
        with np.load(ROOT / 'meshes' / (name + '.npz')) as data:
            vertices = data['vertices_enu']
            faces = data['triangles']
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                        o3d.utility.Vector3iVector(faces))
        low = np.floor(vertices.min(0) / FINE_RES) * FINE_RES
        high = np.ceil(vertices.max(0) / FINE_RES) * FINE_RES + FINE_RES
        voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh, FINE_RES, low, high)
        indices = np.array([v.grid_index for v in voxels.get_voxels()], np.int32)
        points = np.asarray(voxels.origin) + (indices + 0.5) * FINE_RES
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        distances = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
        limit = np.sqrt(3) * FINE_RES / 2
        if not np.isfinite(distances).all() or distances.max() > limit + 1e-3:
            raise RuntimeError('Aircraft voxel geometry validation failed: ' + name)
        np.savez_compressed(OUT / (name + '_voxels.npz'), centers_enu_m=points,
                            indices=indices, origin_enu_m=np.asarray(voxels.origin),
                            resolution_m=np.float64(FINE_RES))

        # One coarse-cell margin covers the entire coarse aircraft footprint.
        # Verify no other
        # elevated scene object's bounds intersect before marking cells free.
        local_origin = low[:2] - res
        local_high = high[:2] + res
        for item in bounds:
            if item['name'] == name or item['size']['z'] <= 0.5:
                continue
            bcenter = np.array([item['center']['y'], item['center']['x']])
            bsize = np.array([item['size']['y'], item['size']['x']])
            if ((bcenter - bsize / 2 < local_high) &
                    (bcenter + bsize / 2 > local_origin)).all():
                raise RuntimeError('Other object intersects fine object map: ' + item['name'])
        local_shape = np.rint((local_high - local_origin) / FINE_RES).astype(int)
        local = np.zeros((local_shape[1], local_shape[0]), np.int8)
        uv = np.floor((points[:, :2] - local_origin) / FINE_RES).astype(int)
        local[uv[:, 1], uv[:, 0]] = 100
        report['maps'].append(save_map(name, local, local_origin, FINE_RES))

        # Replace the coarse aircraft hull footprint with the actual surface projection.
        uv = np.floor((points[:, :2] - origin) / res).astype(int)
        occupancy[uv[:, 1], uv[:, 0]] = 100
        info = {'name': name, 'label': target['label'], 'resolution_m': FINE_RES,
                'voxels': len(points), 'max_center_to_mesh_distance_m': float(distances.max()),
                'distance_upper_bound_m': limit, 'bounds_enu_m': [low.tolist(), high.tolist()],
                'local_scope': f'aircraft bounds plus {res:g} m; not a 100 m fine scene survey'}
        report['targets'].append(info)
        print('AIRCRAFT_GRID', json.dumps(info, ensure_ascii=False), flush=True)
    report['maps'].insert(0, save_map('global', occupancy, origin, res))
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('GRID_MAPS_READY', str(OUT), flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',default=None)
    parser.parse_args();main()
