"""Re-render part masks for a saved capture from its stored camera poses.

No simulator is needed: camera poses, the scene profile and the target mesh are
enough. Existing mask files are replaced and the report's pixel counts and
hashes are updated. With --patches-dir the 64x64 patch targets are rewritten
from the patch boxes recorded by the feature extractor. --targets-only does only
that second step from the masks already on disk, so a machine without Open3D can
refresh targets after masks were re-rendered elsewhere.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from common.patches import open_patches, write_targets


def targets_from_masks(masks, store, name, box):
    x0, y0, x1, y1 = (int(v) for v in box)
    targets = {}
    for part, mask in masks.items():
        if part == 'aircraft':
            continue
        crop = mask[y0:y1, x0:x1]
        target = cv2.resize(crop.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
        target = cv2.GaussianBlur(target, (0, 0), 1.)
        if target.max() > 0:
            target = target/target.max()
        targets[part] = target.astype(np.float16)
    write_targets(store, name, targets)


def load_boxes(features):
    boxes = {}
    for line in Path(features).read_text().splitlines():
        row = json.loads(line)
        if row.get('patch_xyxy') is not None:
            boxes[row['sample']] = row['patch_xyxy']
    return boxes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--patches-dir', type=Path, help='HDF5 patch file or existing image/target directory')
    parser.add_argument('--features', type=Path, help='features.jsonl with patch_xyxy (required with --patches-dir)')
    parser.add_argument('--targets-only', action='store_true',
                        help='Skip rendering; rebuild patch targets from the masks on disk')
    parser.add_argument('--limit', type=int)
    options = parser.parse_args(argv)
    capture, run = options.capture_dir.resolve(), options.run_dir.resolve()
    if (options.patches_dir is None) != (options.features is None):
        raise ValueError('--patches-dir and --features go together')
    boxes = load_boxes(options.features) if options.features else {}
    with ExitStack() as files:
        store = files.enter_context(open_patches(options.patches_dir, 'r+')) if options.patches_dir else None
        report = json.loads((capture/'report.json').read_text())
        samples = report['samples'][:options.limit] if options.limit else report['samples']
        if options.targets_only:
            parts = [name for name in samples[0]['masks'] if name != 'aircraft']
            written = 0
            for sample in samples:
                if sample['name'] not in boxes:
                    continue
                masks = {name: cv2.imread(str(capture/sample['masks'][name]['path']), 0) > 0
                         for name in parts}
                targets_from_masks(masks, store, sample['name'], boxes[sample['name']])
                written += 1
            print(json.dumps(dict(samples=len(samples), targets=written, parts=parts), indent=2))
            return 0
        from scipy.spatial.transform import Rotation
        from capture.collect_part_views import digest
        from capture.collect_training_views import forward_axis
        from capture.part_masks import PartMaskRenderer
        profile = json.loads((run/'scene-profile.json').read_text())
        setup = json.loads((run/'controller-setup.json').read_text())
        target_pose = json.loads((run/'evaluation/inspection.json').read_text())['target_pose']
        forward, origin = forward_axis(target_pose, profile)
        reference = np.asarray(setup['reference_ned_m'], float)
        cfg = json.loads((capture/'runtime-scene.json').read_text())
        actor = next(a for a in cfg['actors'] if a['name'] == profile['simulator']['drone'])
        sensor = next(s for s in actor['robot-config']['sensors'] if s['id'] == profile['simulator']['camera'])
        if sensor.get('gimbal'):
            raise ValueError('Gimballed camera is not supported')
        offset = np.fromstring(sensor['origin']['xyz'], sep=' ')
        pitch = np.radians(float(sensor['origin']['rpy-deg'].split()[1]))
        mount = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
        mesh = ROOT/profile['map']['root']/'meshes'/(profile['target']['actor']+'.npz')
        protocol = json.loads((capture/'protocol.json').read_text()).get('protocol', {})
        renderer = PartMaskRenderer(mesh, profile['parts'], origin, forward, reference,
                                    offset, mount, shape=(1080, 1920), horizontal_fov_deg=90.,
                                    stride=protocol.get('mask_stride', 2))
        pixels = {name: 0 for name in profile['parts']}
        for index, sample in enumerate(samples):
            q = sample['camera_quaternion']
            rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            masks = renderer.render_camera(np.array(sample['camera_position_ned']), rotation)
            for part in profile['parts']:
                mask = masks[part]
                path = capture/sample['masks'][part]['path']
                cv2.imwrite(str(path), mask.astype(np.uint8)*255)
                sample['masks'][part] = dict(path=sample['masks'][part]['path'],
                                             pixels=int(mask.sum()), sha256=digest(path))
                pixels[part] += int(mask.sum())
            if sample['name'] in boxes:
                targets_from_masks(masks, store, sample['name'], boxes[sample['name']])
            if (index+1) % 200 == 0:
                print('RENDER', index+1, 'of', len(samples), flush=True)
        (capture/'report.json').write_text(json.dumps(report, allow_nan=False))
        print(json.dumps(dict(samples=len(samples), pixels=pixels,
                              frames_with_part={name: sum(s['masks'][name]['pixels'] > 0 for s in samples)
                                                for name in profile['parts']}), indent=2))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
