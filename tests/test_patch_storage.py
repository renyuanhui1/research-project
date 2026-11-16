"""Check storage changes through the training and target-rebuild entry points."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common.patches import open_patches, read_rgb, read_target, write_sample
from capture.rerender_part_masks import main as rebuild_targets
from training.train_local_matcher import PatchSet, read_rows


class PatchStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root/'images'
        self.images.mkdir()
        self.packed = self.root/'patches.h5'
        self.rgb = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.target = np.zeros((64, 64), np.float16)
        self.target[8:24, 16:40] = 1
        cv2.imwrite(str(self.images/'frame.png'), self.rgb[:, :, ::-1])
        np.save(self.images/'frame-mask.npy', self.target)
        with open_patches(self.packed, 'w') as store:
            write_sample(store, 'frame', self.rgb, {'tail': self.target})
        self.features = self.root/'features.jsonl'
        self.features.write_text('\n'.join(json.dumps(row) for row in [
            dict(sample='frame', patch_xyxy=[0, 0, 256, 256]),
            dict(sample='missing', patch_xyxy=[0, 0, 256, 256]),
            dict(sample='no_box', patch_xyxy=None)]))

    def test_training_inputs_match_with_augmentation_and_single_part_targets(self):
        datasets = []
        for path in (self.images, self.packed):
            rows = read_rows(self.features, path)
            self.assertEqual([row['sample'] for row in rows], ['frame'])
            with open_patches(path) as store:
                cache = {'frame': (read_rgb(store, 'frame'),
                                   {'tail': read_target(store, 'frame', 'tail')})}
            np.random.seed(7)
            datasets.append(PatchSet(rows, path, True, cache, ['tail'])[0])
        for index in (0, 1):
            self.assertTrue(torch.equal(datasets[0][index], datasets[1][index]))
        self.assertEqual(datasets[0][2], datasets[1][2])

    def test_target_rebuild_updates_both_formats_and_preserves_rgb(self):
        mask = np.zeros((256, 256), np.uint8)
        mask[80:120, 120:200] = 255
        cv2.imwrite(str(self.root/'mask.png'), mask)
        (self.root/'report.json').write_text(json.dumps(dict(samples=[dict(
            name='frame', masks={'tail': {'path': 'mask.png'}})])))
        targets = []
        for path in (self.images, self.packed):
            self.assertEqual(rebuild_targets([
                '--capture-dir', str(self.root), '--run-dir', str(self.root),
                '--features', str(self.features), '--patches-dir', str(path),
                '--targets-only']), 0)
            with open_patches(path) as store:
                np.testing.assert_array_equal(read_rgb(store, 'frame'), self.rgb)
                targets.append(read_target(store, 'frame', 'tail'))
        np.testing.assert_array_equal(*targets)
        self.assertEqual(targets[0].dtype, np.float16)
        self.assertFalse(np.array_equal(targets[0], self.target))


if __name__ == '__main__':
    unittest.main()
