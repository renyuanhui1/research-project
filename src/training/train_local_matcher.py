"""Train a learned local matcher on exported search patches.

Input is the DINO coarse crop expanded by 25% and resized to 256x256; the target
is the rendered part mask downsampled to 64x64. The network predicts a spatial
distribution over the patch and is evaluated by the distance from its peak to
the full-resolution part mask, using the same (bearing, range) grouped split as
the acceptance layer.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def read_rows(path, patch_dir):
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('patch_xyxy') is None:
            continue
        if not (Path(patch_dir)/(row['sample']+'.png')).exists():
            continue
        rows.append(row)
    return rows


def group_of(row, width=15.):
    bearing = int(np.floor((row['bearing_deg']+90.)/width))
    return bearing*2+(1 if row['range_m'] >= 60. else 0)


def subsample(rows, count, seed):
    """Keep the view-cell mix while reducing the training set to count frames."""
    if not count or count >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    groups = np.array([group_of(row) for row in rows])
    keys, counts = np.unique(groups, return_counts=True)
    quota = np.maximum(1, np.floor(count*counts/counts.sum()).astype(int))
    chosen = []
    for key, size in zip(keys, quota):
        index = np.where(groups == key)[0]
        chosen.extend(rng.choice(index, size=min(int(size), len(index)), replace=False).tolist())
    chosen = np.array(chosen)
    if len(chosen) > count:
        chosen = rng.choice(chosen, size=count, replace=False)
    elif len(chosen) < count:
        rest = np.setdiff1d(np.arange(len(rows)), chosen)
        chosen = np.r_[chosen, rng.choice(rest, size=count-len(chosen), replace=False)]
    return [rows[i] for i in chosen]


def block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class HeatmapNet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.enc1 = block(3, 32)
        self.enc2 = block(32, 64, 2)
        self.enc3 = block(64, 128, 2)
        self.enc4 = block(128, 128, 2)
        self.up3 = block(128+128, 128)
        self.head = nn.Conv2d(128, channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        u3 = self.up3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode='bilinear', align_corners=False), e3], 1))
        return self.head(u3)


class PatchSet(torch.utils.data.Dataset):
    def __init__(self, rows, patch_dir, train, cache, parts):
        self.rows, self.dir, self.train, self.cache, self.parts = rows, Path(patch_dir), train, cache, parts

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image, targets = self.cache[row['sample']]
        image = image.astype(np.float32)/255.
        target = np.stack([targets[name] for name in self.parts]).astype(np.float32)
        if self.train:
            if np.random.rand() < .5:
                image, target = image[:, ::-1], target[:, :, ::-1]
            image = np.clip(image*float(np.random.uniform(.8, 1.2))+float(np.random.uniform(-.1, .1)), 0., 1.)
        image, target = np.ascontiguousarray(image), np.ascontiguousarray(target)
        totals = target.sum(axis=(1, 2), keepdims=True)
        target = np.divide(target, totals, out=np.zeros_like(target), where=totals > 0)
        return (torch.from_numpy(image.transpose(2, 0, 1)),
                torch.from_numpy(target), row['sample'])


def peak_stats(logits, prob):
    """Localization peak plus shape statistics that can serve as confidence."""
    flat_logits = logits.reshape(-1)
    flat_prob = prob.reshape(-1)
    index = int(flat_prob.argmax())
    y, x = divmod(index, prob.shape[-1])
    grid_y, grid_x = np.mgrid[0:prob.shape[-2], 0:prob.shape[-1]]
    centroid = np.array([(flat_prob*grid_x.reshape(-1)).sum(), (flat_prob*grid_y.reshape(-1)).sum()])
    window = prob[max(0, y-1):y+2, max(0, x-1):x+2]
    entropy = float(-(flat_prob*np.log(np.clip(flat_prob, 1e-12, None))).sum())
    return dict(peak_x=(x+.5)/prob.shape[-1], peak_y=(y+.5)/prob.shape[-2],
                confidence=float(flat_prob[index]), entropy=entropy,
                top_mass=float(window.sum()), max_logit=float(flat_logits[index]),
                logit_margin=float(flat_logits[index]-flat_logits.mean()),
                peak_centroid_cells=float(np.linalg.norm(centroid-[x, y])))


def full_image_point(row, px, py):
    x0, y0, x1, _ = row['patch_xyxy']
    side = x1-x0
    return x0+px*side, y0+py*side


def mask_distance(mask, point):
    inverse = (~mask).astype(np.uint8)
    outside = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    x, y = np.rint(point).astype(int)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return float('inf')
    return float(-inside[y, x] if mask[y, x] else outside[y, x])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--patch-dir', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', default=['tail'],
                        help='Parts to predict; each part needs a <sample>-<part>.npy target')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--train-count', type=int, default=0,
                        help='Subsample each fold training set to this many frames; 0 keeps all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--tolerance-px', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=7)
    options = parser.parse_args(argv)
    from sklearn.model_selection import GroupKFold
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_rows(options.features, options.patch_dir)
    capture = options.capture_dir.resolve()
    report = json.loads((capture/'report.json').read_text())
    masks = {sample['name']: sample['masks'] for sample in report['samples']}
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    parts = options.parts
    groups = np.array([group_of(row) for row in rows])
    print('samples', len(rows), 'groups', len(set(groups)), 'parts', parts, flush=True)
    cache = {}
    for row in rows:
        targets = {}
        for name in parts:
            path = options.patch_dir/(row['sample']+'-'+name+'.npy')
            if not path.exists():
                path = options.patch_dir/(row['sample']+'-mask.npy')
            targets[name] = np.load(path)
        cache[row['sample']] = (
            cv2.imread(str(options.patch_dir/(row['sample']+'.png')))[:, :, ::-1], targets)
    print('cache loaded', len(cache), flush=True)
    oof = []
    for fold, (train_index, test_index) in enumerate(GroupKFold(n_splits=options.folds).split(rows, groups=groups)):
        train_rows = subsample([rows[i] for i in train_index], options.train_count,
                               options.seed+17*fold)
        test_rows = [rows[i] for i in test_index]
        print('fold %d train %d test %d' % (fold, len(train_rows), len(test_rows)), flush=True)
        loader = torch.utils.data.DataLoader(PatchSet(train_rows, options.patch_dir, True, cache, parts),
                                             batch_size=options.batch, shuffle=True, num_workers=6,
                                             persistent_workers=True)
        model = HeatmapNet(len(parts)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(options.epochs):
            model.train()
            total = 0.
            for image, target, _ in loader:
                image, target = image.to(device), target.to(device)
                logits = model(image)
                log_prob = F.log_softmax(logits.flatten(2), dim=2).view_as(logits)
                loss = -(target*log_prob).sum(dim=(2, 3)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss)
            print('fold %d epoch %d loss %.4f' % (fold, epoch, total/len(loader)), flush=True)
        model.eval()
        torch.save(model.state_dict(), out/('fold%d.pt' % fold))
        with torch.no_grad():
            for start in range(0, len(test_rows), options.batch):
                batch = test_rows[start:start+options.batch]
                image = torch.stack([torch.from_numpy(
                    np.ascontiguousarray(cache[row['sample']][0].astype(np.float32)/255.).transpose(2, 0, 1))
                    for row in batch]).to(device)
                logits = model(image)
                prob = F.softmax(logits.flatten(2), dim=2).view_as(logits).cpu().numpy()
                logits = logits.cpu().numpy()
                for row, heat, logit in zip(batch, prob, logits):
                    x0, y0, x1, y1 = row['patch_xyxy']
                    parts_masks = {name: cv2.imread(str(capture/masks[row['sample']][name]['path']), 0) > 0
                                   for name in parts}
                    for channel, name in enumerate(parts):
                        stats = peak_stats(logit[channel], heat[channel])
                        point = full_image_point(row, stats.pop('peak_x'), stats.pop('peak_y'))
                        mask = parts_masks[name]
                        visible = int(mask[y0:y1, x0:x1].sum())
                        distance = mask_distance(mask, point)
                        oof.append(dict(part=name, sample=row['sample'], fold=fold,
                                        bearing_deg=row['bearing_deg'], range_m=row['range_m'],
                                        visible=visible, point=list(point), **stats,
                                        distance_px=distance,
                                        correct=bool(distance <= options.tolerance_px)))
        del model, optimizer, loader
    torch.cuda.empty_cache()
    visible = [row for row in oof if row['visible'] >= 20]
    summary = dict(samples=len(oof), visible=len(visible), parts={},
                   folds=options.folds, epochs=options.epochs, tolerance_px=options.tolerance_px,
                   device=device)
    for name in parts:
        rows = [row for row in visible if row['part'] == name]
        correct = np.array([row['correct'] for row in rows])
        confidence = np.array([row['confidence'] for row in rows])
        order = np.argsort(-confidence)
        tp = np.cumsum(correct[order])
        precision = tp/np.arange(1, len(tp)+1)
        recall = tp/max(int(correct.sum()), 1)
        keep = precision >= .95
        summary['parts'][name] = dict(
            samples=len(rows), correct=int(correct.sum()),
            correct_rate=float(correct.mean()) if rows else None,
            median_distance_px=float(np.median([row['distance_px'] for row in rows])) if rows else None,
            recall_at_precision_95=float(recall[keep].max()) if keep.any() else 0.)
    (out/'oof.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in oof))
    (out/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
