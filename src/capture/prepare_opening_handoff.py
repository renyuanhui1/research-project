"""Prepare the opening camera on an existing drone after a confirmed visual handoff."""
import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
from projectairsim import Drone, ProjectAirSimClient, World
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.runtime import configure_client_log
from common.sensors import decode_image
from capture.collect_part_views import decode_rpc_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse-run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--camera-down-angle', type=float, default=45.)
    options = parser.parse_args()
    if not math.isfinite(options.camera_down_angle) or not 40 <= options.camera_down_angle <= 90:
        parser.error('Opening control supports camera-down-angle in [40,90]')
    coarse = options.coarse_run_dir.resolve()
    previous = json.loads((coarse / 'report.json').read_text())
    if not previous.get('handoff_ready'):
        raise RuntimeError('Coarse control has not confirmed the part handoff')
    cfg = json.loads((coarse / 'runtime-scene.json').read_text())
    profile = json.loads((coarse / 'scene-profile.json').read_text())
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use an empty output directory: ' + str(out))
    configure_client_log(out)
    client = ProjectAirSimClient(profile['simulator']['address'])
    report = {'status': 'started', 'flight_commands_sent': False,
              'coarse_run_dir': str(coarse), 'camera_down_angle_deg': options.camera_down_angle}
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
        report['initial_kinematics'] = drone.get_ground_truth_kinematics()
        def yaw(kin):
            q=kin['pose']['orientation']
            return math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))
        expected_yaw=yaw(previous['final_state'])
        for _ in range(5):
            kin=drone.get_ground_truth_kinematics()
            delta=yaw(kin)-expected_yaw
            if (sum(kin['twist']['linear'][k]**2 for k in ('x','y','z'))**.5>=.1
                    or abs(kin['twist']['angular']['z'])>=.02
                    or (options.camera_down_angle<90 and abs(math.atan2(math.sin(delta),math.cos(delta)))>math.radians(5))):
                raise RuntimeError('Coarse handoff must be stationary with its recorded heading')
            time.sleep(.2)
        sensor = next(s for a in cfg['actors'] if a['name'] == profile['simulator']['drone']
                      for s in a['robot-config']['sensors'] if s['id'] == profile['simulator']['camera'])
        offset = list(map(float, sensor['origin']['xyz'].split()))
        angle = math.radians(options.camera_down_angle)
        pose = {'frame_id': 'DEFAULT_FRAME', 'translation': dict(zip(('x', 'y', 'z'), offset)),
                'rotation': {'w': math.cos(angle/2), 'x': 0., 'y': -math.sin(angle/2), 'z': 0.}}
        if not drone.set_camera_pose(profile['simulator']['camera'], pose):
            raise RuntimeError('Camera pose change not confirmed')
        sensor['origin']['rpy-deg'] = f'0 {-options.camera_down_angle} 0'
        time.sleep(2)
        images = drone.get_images(profile['simulator']['camera'], [0, 2])
        rgb = decode_image(images[0])[0]
        depth = decode_rpc_depth(images[2]).astype(np.float32)
        cv2.imwrite(str(out / 'handoff-front.png'), rgb[:, :, ::-1])
        np.save(out / 'handoff-depth.npy', depth)
        (out / 'runtime-scene.json').write_text(json.dumps(cfg, indent=2))
        report.update(status='completed', camera_pose=pose,
                      rgb_metadata={k: v for k, v in images[0].items() if k != 'data'},
                      depth_metadata={k: v for k, v in images[2].items() if k != 'data'},
                      final_kinematics=drone.get_ground_truth_kinematics())
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        client.disconnect()
        (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('OPENING_CAMERA_READY', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
