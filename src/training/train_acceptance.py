"""Train a learned acceptance layer on frozen-matcher features.

The frozen rule accepts a local registration when score>=0.85, margin>=0.15 and
the point stays inside the search radius. That rule rejects many registrations
whose point is still correct. This script fits a classifier on features that are
available at test time and evaluates it against the frozen rule with a
viewpoint-grouped split.
"""
import argparse
import json
from pathlib import Path

import numpy as np


FEATURES = ('whole_peak', 'part_dino_peak', 'score', 'margin', 'scale', 'angle_deg',
            'inside_radius', 'point_candidate_distance_px', 'box_width', 'box_height')
EXTRA = ('entropy', 'top_mass', 'max_logit', 'logit_margin', 'peak_centroid_cells')


def load_rows(path, tolerance_frac, tolerance_min, extra=None, labels_from_extra=False):
    extra_rows = {}
    if extra is not None:
        extra_rows = {row['sample']: row for row in
                      (json.loads(line) for line in Path(extra).read_text().splitlines())}
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('candidate_x') is None or row.get('point_distance_px') is None:
            continue
        pixels = row.get('mask_pixels') or 0
        tolerance = max(tolerance_min, tolerance_frac*np.sqrt(pixels))
        row['tolerance_px'] = tolerance
        row['label'] = bool(row['point_distance_px'] <= tolerance)
        if row['sample'] in extra_rows:
            row.update({name: extra_rows[row['sample']].get(name) for name in EXTRA})
            if labels_from_extra:
                row['point_distance_px'] = extra_rows[row['sample']]['distance_px']
                row['label'] = bool(extra_rows[row['sample']]['correct'])
        elif labels_from_extra:
            continue
        rows.append(row)
    return rows


def design_matrix(rows, features=FEATURES):
    x = np.array([[float(row[name]) if row.get(name) is not None else np.nan
                   for name in features] for row in rows], float)
    x[:, features.index('inside_radius')] = np.array(
        [1. if row.get('inside_radius') else 0. for row in rows])
    x[:, features.index('angle_deg')] = np.abs(x[:, features.index('angle_deg')])
    return x


def group_of(row, width=15.):
    """Held-out groups are (bearing sector, range stratum) cells, not single frames."""
    bearing = int(np.floor((row['bearing_deg']+90.)/width))
    return bearing*2+(1 if row['range_m'] >= 60. else 0)


def precision_recall(scores, labels):
    order = np.argsort(-scores)
    labels = np.asarray(labels, bool)[order]
    tp = np.cumsum(labels)
    accepted = np.arange(1, len(labels)+1)
    precision = tp/accepted
    recall = tp/max(int(labels.sum()), 1)
    return precision, recall


def recall_at_precision(scores, labels, target):
    precision, recall = precision_recall(scores, labels)
    keep = precision >= target
    return float(recall[keep].max()) if keep.any() else 0.


def threshold_at_precision(scores, labels, target):
    order = np.argsort(-scores)
    labels = np.asarray(labels, bool)[order]
    tp = np.cumsum(labels)
    precision = tp/np.arange(1, len(labels)+1)
    keep = np.where(precision >= target)[0]
    if not len(keep):
        return None
    index = keep[-1]
    return dict(threshold=float(scores[order][index]), precision=float(precision[index]),
                recall=float(tp[index]/max(int(labels.sum()), 1)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--tolerance-frac', type=float, default=.1,
                        help='Label tolerance as a fraction of sqrt(mask pixels)')
    parser.add_argument('--tolerance-min', type=float, default=3.)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--extra', type=Path, help='Local-matcher OOF JSONL joined by sample name')
    parser.add_argument('--labels-from-extra', action='store_true',
                        help='Use the joined local-matcher distance as the label')
    parser.add_argument('--seed', type=int, default=7)
    options = parser.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    extra = options.extra.resolve() if options.extra else None
    rows = load_rows(options.features, options.tolerance_frac, options.tolerance_min,
                     extra, options.labels_from_extra)
    features = list(FEATURES)+([name for name in EXTRA] if extra else [])
    x = design_matrix(rows, features)
    y = np.array([row['label'] for row in rows])
    groups = np.array([group_of(row) for row in rows])
    frozen = np.array([bool(row['accepted']) for row in rows])
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = dict(samples=len(rows), positives=int(y.sum()),
                   positive_rate=float(y.mean()), groups=int(len(set(groups))),
                   frozen_accepts=int(frozen.sum()),
                   frozen_precision=(float(y[frozen].mean()) if frozen.any() else None),
                   frozen_recall=(float(y[frozen].sum()/max(int(y.sum()), 1)) if frozen.any() else None),
                   features=features, tolerance_frac=options.tolerance_frac,
                   tolerance_min=options.tolerance_min)
    models = {
        'logistic': make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight='balanced')),
        'hist_gradient': HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                        learning_rate=.05, random_state=options.seed),
    }
    fold_rows = []
    oof = {}
    for name, model in models.items():
        scores = np.full(len(rows), np.nan)
        for train, test in GroupKFold(n_splits=options.folds).split(x, y, groups):
            fitted = model.fit(x[train], y[train])
            scores[test] = fitted.predict_proba(x[test])[:, 1]
        oof[name] = scores
        fold_rows.append(dict(
            model=name,
            recall_at_precision_95=recall_at_precision(scores, y, .95),
            recall_at_precision_90=recall_at_precision(scores, y, .90),
            recall_at_precision_85=recall_at_precision(scores, y, .85),
            auroc=float(roc_auc_score(y, scores))))
    best = max(fold_rows, key=lambda row: (row['recall_at_precision_95'], row['auroc']))
    model = models[best['model']].fit(x, y)
    summary['models'] = fold_rows
    summary['selected'] = best['model']
    summary['operating_point'] = threshold_at_precision(oof[best['model']], y, .95)
    summary['baseline_recall_at_precision_95'] = (
        summary['frozen_recall'] if (summary['frozen_precision'] or 0) >= .95 else 0.)
    (out/'metrics.json').write_text(json.dumps(summary, indent=2))
    np.save(out/'oof_scores.npy', np.stack([oof[name] for name in oof], 1))
    with (out/'oof.jsonl').open('w') as file:
        for index, row in enumerate(rows):
            file.write(json.dumps(dict(sample=row['sample'], label=bool(y[index]),
                                       bearing_deg=row['bearing_deg'], range_m=row['range_m'],
                                       point_distance_px=row['point_distance_px'],
                                       frozen=bool(frozen[index]),
                                       scores={name: float(oof[name][index]) for name in oof}))+'\n')
    try:
        import joblib
        joblib.dump(model, out/'acceptance.joblib')
    except ImportError:
        pass
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
