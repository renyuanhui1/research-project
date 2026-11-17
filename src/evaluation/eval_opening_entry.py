"""Independently inspect the endpoint of a roof-entry experiment; no flight commands."""
import argparse
import json
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from projectairsim import Drone, ProjectAirSimClient, World

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.profiles import load_profile
from common.runtime import configure_client_log
from common.sensors import decode_image
from capture.collect_part_views import decode_rpc_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-dir', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    options = parser.parse_args()
    entry, out = options.entry_dir.resolve(), options.out_dir.resolve()
    report = json.loads((entry / 'report.json').read_text())
    profile = load_profile(options.profile)
    cfg = json.loads((entry / 'runtime-scene.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use an empty output directory: ' + str(out))
    configure_client_log(out)
    client = ProjectAirSimClient(profile['simulator']['address'])
    inspection = {'flight_commands_sent': False}
    try:
        client.connect()
        client.socket_services.recv_timeout = 10000
        event = threading.Event()
        def info(topic, message):
            client.on_topic_info(topic, message)
            event.set()
        client.subscribe('/$topics', info)
        if not event.wait(6):
            raise RuntimeError('Topic discovery timeout')
        world = World.__new__(World)
        world.client, world.parent_topic, world.sim_config = client, '/Sim/' + cfg['id'], cfg
        drone = Drone(client, world, profile['simulator']['drone'])
        inspection['vehicle_kinematics'] = drone.get_ground_truth_kinematics()
        inspection['target_pose'] = world.get_object_pose(profile['target']['actor'])
        images = drone.get_images(profile['simulator']['camera'], [0, 2])
        rgb = decode_image(images[0])[0]
        depth = decode_rpc_depth(images[2]).astype(np.float32)
        inspection['rgb_metadata'] = {k: v for k, v in images[0].items() if k != 'data'}
        inspection['depth_metadata'] = {k: v for k, v in images[2].items() if k != 'data'}
        np.savez_compressed(out / 'endpoint-rgbd.npz', rgb=rgb, depth_m=depth)
        cv2.imwrite(str(out / 'endpoint.png'), rgb[:, :, ::-1])
    finally:
        client.disconnect()
        (out / 'inspection.json').write_text(json.dumps(inspection, indent=2))
    body = np.array([inspection['vehicle_kinematics']['pose']['position'][k] for k in ('x', 'y', 'z')])
    scene, vertices = o3d.t.geometry.RaycastingScene(), []
    for actor in profile['map']['actors']:
        with np.load(ROOT / profile['map']['root'] / 'meshes' / (actor + '.npz')) as mesh:
            points = mesh['vertices_enu'][:, [1, 0, 2]].copy()
            points[:, 2] *= -1
            scene.add_triangles(o3d.core.Tensor(points.astype(np.float32)),
                                o3d.core.Tensor(mesh['triangles'].astype(np.uint32)))
            vertices.append(points)
    vertices = np.concatenate(vertices)
    hull = ConvexHull(vertices[:, :2])
    footprint = vertices[hull.vertices, :2].astype(np.float32)
    footprint_margin = float(cv2.pointPolygonTest(footprint, tuple(map(float, body[:2])), True))
    altitude, floor, roof = -float(body[2]), -float(vertices[:, 2].max()), -float(vertices[:, 2].min())
    roof_world_z = None
    handoff = report.get('handoff')
    if handoff is not None:
        with (entry / 'signals.jsonl').open() as stream:
            anchor = next(row for row in map(json.loads, stream) if row['step'] == handoff['at_step'])
        roof_world_z = float(anchor['rgb_metadata']['pos_z']) + handoff['roof_distance_from_anchor_m']
    photo_name = report.get('interior_observation', {}).get('image')
    photo = cv2.imread(str(entry / photo_name)) if photo_name else None
    photo_ok = photo is not None and float(photo.std()) > 4 and float((photo.min(2) >= 250).mean()) <= .5
    below_roof = roof_world_z is not None and float(body[2]) > roof_world_z
    inside = footprint_margin > 0 and floor < altitude < roof
    passed = bool(report.get('entered') and report.get('final_hover_confirmed') and
                  report['status'] == 'completed' and inside and below_roof and photo_ok)
    distance = float(scene.compute_distance(o3d.core.Tensor(body[None].astype(np.float32))).numpy()[0])
    result = {'status': 'passed' if passed else 'entry_not_verified',
              'target_actor': profile['target']['actor'], 'assembly_actors': profile['map']['actors'],
              'entry_reported': bool(report.get('entered')), 'entry_stop_reason': report.get('stop_reason'),
              'inside_assembly_convex_footprint': footprint_margin > 0,
              'inside_assembly_height_range': floor < altitude < roof, 'footprint_margin_m': footprint_margin,
              'body_altitude_m': altitude, 'assembly_floor_altitude_m': floor, 'assembly_roof_altitude_m': roof,
              'below_visual_roof_plane': below_roof, 'visual_roof_world_z_ned_m': roof_world_z,
              'body_to_assembly_surface_m': distance, 'interior_photo': photo_name,
              'interior_photo_quality_passed': bool(photo_ok),
              'endpoint_speed_mps': float(np.linalg.norm([inspection['vehicle_kinematics']['twist']['linear'][k] for k in ('x', 'y', 'z')])),
              'geometry_used_for_evaluation_only': True, 'flight_commands_sent': False,
              'footprint_model': 'convex hull of the exported hangar assembly'}
    (out / 'entry-validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
