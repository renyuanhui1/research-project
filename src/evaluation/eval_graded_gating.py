"""Offline replay for GradedGate (P2.4) and same-class template library confirmation (P4.2)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from perception.graded_gating import DistanceBinCalibrator, GradedGate, _logit


def evaluate_p42(rows, by_cat):
    dist_bins = [(0, 45), (45, 80), (80, 120), (120, 300)]
    aircraft_b_ids = {
        '05-aircraft-b-intact',
        '06-aircraft-b-folded-wing',
        '07-aircraft-b-collapsed-cockpit',
        '08-aircraft-b-tail-back',
    }
    out = {}
    for a, b in dist_bins:
        sub = [r for r in rows if r['horiz_dist_m'] is not None and a <= r['horiz_dist_m'] < b and r['candidates']]
        if not sub:
            continue
        top1_crop = top1_max = aircraft_b_ok = aircraft_b_n = 0
        m_true, m_wrong_same, s_cross = [], [], []
        for r in sub:
            tid, cat, sc_map = r['target_id'], r['category'], r['scores_by_template']
            same_tids = [x for x in by_cat[cat] if x in sc_map and sc_map[x]]
            cross_tids = [x for x in sc_map if x not in by_cat[cat] and sc_map[x]]
            if tid not in same_tids:
                continue
            best_i = int(np.argmax(sc_map[tid]))
            others = [sc_map[x][best_i] for x in same_tids if x != tid]
            if others:
                mt = _logit(sc_map[tid][best_i]) - _logit(max(others))
                m_true.append(mt)
                if mt > 0:
                    top1_crop += 1
            max_sc = {x: max(sc_map[x]) for x in same_tids}
            others_max = [max_sc[x] for x in same_tids if x != tid]
            if others_max and max_sc[tid] > max(others_max):
                top1_max += 1
            for wx in same_tids:
                if wx == tid:
                    continue
                wi = int(np.argmax(sc_map[wx]))
                w_oth = [sc_map[y][wi] for y in same_tids if y != wx]
                if w_oth:
                    m_wrong_same.append(_logit(sc_map[wx][wi]) - _logit(max(w_oth)))
            for cx in cross_tids:
                s_cross.append(max(sc_map[cx]))
            if tid in aircraft_b_ids:
                fo = [sc_map[x][best_i] for x in aircraft_b_ids if x != tid and x in sc_map and sc_map[x]]
                if fo:
                    aircraft_b_n += 1
                    aircraft_b_ok += int(sc_map[tid][best_i] > max(fo))
        out[f'{a}-{b}m'] = {
            'n': len(sub),
            'same_crop_top1_rate': round(top1_crop / len(sub), 4),
            'max_cand_top1_rate': round(top1_max / len(sub), 4),
            'true_logit_margin_p25_p50_p75': [round(float(q), 3) for q in np.quantile(m_true, [0.25, 0.5, 0.75])] if m_true else None,
            'wrong_same_logit_margin_p50_p90': [round(float(q), 3) for q in np.quantile(m_wrong_same, [0.5, 0.9])] if m_wrong_same else None,
            'cross_cat_max_score_p50_p90': [round(float(q), 3) for q in np.quantile(s_cross, [0.5, 0.9])] if s_cross else None,
            'aircraft_b_variant_top1': f'{aircraft_b_ok}/{aircraft_b_n} ({aircraft_b_ok/aircraft_b_n:.3f})' if aircraft_b_n else None,
        }
    return out


def eval_sequence(seq_rows, query_tid, cal, tby_id, by_cat, policy_name, gate_cfg=None, use_margin=False):
    cat_q = tby_id[query_tid]['category']
    same_peers = [x for x in by_cat[cat_q] if x != query_tid]
    if policy_name == 'legacy_0913':
        streak = 0; last_t = None; perm = conf = near_adm = 0; first_p = None
        for i, r in enumerate(seq_rows):
            sc_list = r['scores_by_template'].get(query_tid, [])
            if not sc_list:
                streak = 0; continue
            best_i = int(np.argmax(sc_list))
            sc = sc_list[best_i]; dm = r['candidates'][best_i]['depth_m']
            ts = r['time_s'] if r['time_s'] is not None else i * 0.8
            streak = (streak + 1 if (last_t is not None and ts - last_t <= 2.0) else 1) if sc >= 0.9130731477441585 else 0
            last_t = ts
            if streak >= 3:
                perm += 1; conf += 1
                if first_p is None: first_p = i
                if dm is not None and dm <= 45.0: near_adm += 1
        return {'permitted': perm, 'confirmed': conf > 0, 'near_admitted': near_adm > 0, 'rejected': False, 'first_perm': first_p}

    gate = GradedGate(cal, gate_cfg)
    perm = 0; first_p = None; confirmed = False; near_adm = False; rejected = False
    for i, r in enumerate(seq_rows):
        sc_list = r['scores_by_template'].get(query_tid, [])
        if not sc_list:
            continue
        best_i = int(np.argmax(sc_list))
        cand = r['candidates'][best_i]
        sc = sc_list[best_i]
        ts = r['time_s'] if r['time_s'] is not None else i * 0.8
        others = None
        if use_margin:
            others = [r['scores_by_template'][p][best_i] for p in same_peers
                      if p in r['scores_by_template'] and len(r['scores_by_template'][p]) > best_i]
        out = gate.update(sc, cand['point_xy'], cand['box_xyxy'], cand['depth_m'], r['rgb_metadata'], ts,
                          same_class_other_scores=others)
        if out['motion_permitted']:
            perm += 1
            if first_p is None: first_p = i
        if out['identity_confirmed']: confirmed = True
        if out['near_handoff_permitted']: near_adm = True
        if out['identity_rejected']: rejected = True
    return {'permitted': perm, 'confirmed': confirmed, 'near_admitted': near_adm, 'rejected': rejected,
            'first_perm': first_p, 'final_llr': round(gate.llr_sum, 3), 'mid_llr': round(gate.mid_llr_sum, 3),
            'mean_margin': round(float(np.mean(gate.margin_history)), 3) if gate.margin_history else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/evaluation/20261119-p2-replay-cache/multitemplate-sequences.jsonl')
    parser.add_argument('--calibrator', type=Path, default=ROOT / 'runs/evaluation/20261119-p2-gating-offline/calibrator-train.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/evaluation/20261119-p2-gating-offline/multitemplate-replay-summary.json')
    args = parser.parse_args()

    cal = DistanceBinCalibrator.from_dict(json.loads(args.calibrator.read_text()))
    catalog = json.loads((ROOT / 'data/templates/airport_targets_20261116/catalog.json').read_text())
    tby_id = {t['id']: t for t in catalog['targets']}
    by_cat = defaultdict(list)
    for t in catalog['targets']:
        by_cat[t['category']].append(t['id'])

    rows = [json.loads(l) for l in args.cache.read_text().splitlines() if l.strip()]
    seqs = defaultdict(list)
    for r in rows:
        seqs[(r['tag'], r['target_id'])].append(r)

    p42 = evaluate_p42(rows, by_cat)
    print('=== PART 1: P4.2 SAME-CLASS RELATIVE DISCRIMINATION BY DISTANCE ===')
    for k, v in p42.items():
        print(f"  {k:9s}: n={v['n']:3d} top1_crop={v['same_crop_top1_rate']:.3f} true_m={v['true_logit_margin_p25_p50_p75']} wrong_m={v['wrong_same_logit_margin_p50_p90']} cross_s={v['cross_cat_max_score_p50_p90']} aircraft_b={v['aircraft_b_variant_top1']}")

    cfg_p1 = {
        'mid_range_m': 80.0, 'near_handoff_range_m': 45.0,
        'far_consistent_frames': 2, 'far_min_score': 0.50, 'far_llr_threshold': -0.2,
        'confirm_max_range_m': 80.0, 'confirm_llr_threshold': 2.0, 'confirm_min_mid_frames': 3,
        'reject_llr_threshold': -3.0, 'llr_decay': 0.92,
        'min_evidence_move_m': 2.0, 'stationary_llr_weight': 0.2,
        'require_same_class_margin': False,
    }
    cfg_p2 = dict(cfg_p1, require_same_class_margin=True, min_same_class_logit_margin=0.0)
    cfg_p2_strict = dict(cfg_p1, require_same_class_margin=True, min_same_class_logit_margin=0.3)

    long_tags = {'1118-e2e-2', '1118-repeat', '1118-gw', '1118-shadow-nom', '1118-shadow-p12', '1118-shadow-id'}
    policies = [
        ('legacy_0913', None, False),
        ('graded_single_sprt', cfg_p1, False),
        ('graded_sprt_p42_m0', cfg_p2, True),
        ('graded_sprt_p42_m03', cfg_p2_strict, True),
    ]

    policy_summary = {}
    print('\n=== PART 2: FULL POLICY REPLAY COMPARISON ===')
    for pname, pcfg, use_m in policies:
        far19_move = far19_tot = 0
        lp_conf = lp_near = lp_tot = 0
        lw_conf = lw_near = lw_tot = 0
        cx_move = cx_conf = cx_near = cx_tot = 0
        long_details = []
        for (tag, true_tid), srows in sorted(seqs.items()):
            cat = tby_id[true_tid]['category']
            same_peers = [x for x in by_cat[cat] if x != true_tid and x in srows[0]['scores_by_template']]
            cross_peers = [x for x in srows[0]['scores_by_template'] if x not in by_cat[cat]]
            res_pos = eval_sequence(srows, true_tid, cal, tby_id, by_cat, pname, pcfg, use_m)
            if tag.startswith('1119'):
                far19_tot += 1; far19_move += int(res_pos['permitted'] > 0)
            if tag in long_tags:
                lp_tot += 1; lp_conf += int(res_pos['confirmed']); lp_near += int(res_pos['near_admitted'])
                w_confs, w_nears = [], []
                for wt in same_peers:
                    rw = eval_sequence(srows, wt, cal, tby_id, by_cat, pname, pcfg, use_m)
                    lw_tot += 1; lw_conf += int(rw['confirmed']); lw_near += int(rw['near_admitted'])
                    if rw['confirmed']: w_confs.append(wt)
                    if rw['near_admitted']: w_nears.append(wt)
                long_details.append({'tag': tag, 'target': true_tid, 'pos': res_pos,
                                     'wrong_same_confirmed': w_confs, 'wrong_same_near': w_nears})
            for ct in cross_peers:
                rc = eval_sequence(srows, ct, cal, tby_id, by_cat, pname, pcfg, use_m)
                cx_tot += 1; cx_move += int(rc['permitted'] > 0); cx_conf += int(rc['confirmed']); cx_near += int(rc['near_admitted'])
        policy_summary[pname] = {
            'far19_motion': f'{far19_move}/{far19_tot}',
            'long_pos_confirmed': f'{lp_conf}/{lp_tot}',
            'long_pos_near_admitted': f'{lp_near}/{lp_tot}',
            'long_wrong_same_confirmed': f'{lw_conf}/{lw_tot}',
            'long_wrong_same_near_admitted': f'{lw_near}/{lw_tot}',
            'cross_cat_far_motion': f'{cx_move}/{cx_tot}',
            'cross_cat_confirmed': f'{cx_conf}/{cx_tot}',
            'cross_cat_near_admitted': f'{cx_near}/{cx_tot}',
            'long_details': long_details,
        }
        print(f"\n[{pname}] far19={far19_move}/{far19_tot} | long_pos_conf={lp_conf}/{lp_tot} near={lp_near}/{lp_tot} | wrong_same_conf={lw_conf}/{lw_tot} near={lw_near}/{lw_tot} | cross_move={cx_move}/{cx_tot} conf={cx_conf}/{cx_tot}")
        for ld in long_details:
            print(f"  {ld['tag']:16s} ({ld['target']:20s}): pos_conf={str(ld['pos']['confirmed']):5s} near={str(ld['pos']['near_admitted']):5s} margin={ld['pos'].get('mean_margin')} | wrong_conf={ld['wrong_same_confirmed']}")

    obs = [json.loads(l) for l in (ROOT / 'runs/evaluation/20261119-tower-live-independent/observations.jsonl').read_text().splitlines() if l.strip()]
    by_view = defaultdict(list)
    for o in obs:
        by_view[o['view']].append(o)
    tower_abs = {}
    for vw, items in sorted(by_view.items()):
        gate = GradedGate(cal, cfg_p2)
        perm = conf = rej = 0
        for idx_i, o in enumerate(items):
            pt = o['point']
            box = [int(pt[0]*1920)-100, int(pt[1]*1080)-100, int(pt[0]*1920)+100, int(pt[1]*1080)+100]
            out = gate.update(o['score'], pt, box, 70.0,
                              {'pos_x': 0, 'pos_y': 0, 'pos_z': 0, 'rot_w': 1, 'rot_x': 0, 'rot_y': 0, 'rot_z': 0},
                              idx_i * 0.8, same_class_other_scores=[o['rival_score']])
            perm += int(out['motion_permitted']); conf += int(out['identity_confirmed']); rej += int(out['identity_rejected'])
        tower_abs[vw] = {'n': len(items), 'visible_px_max': max(o['visible_px'] for o in items),
                         'permitted': perm, 'confirmed': conf, 'rejected': rej}
    print('\n=== FIXED-CAMERA TOWER INDEPENDENT ===')
    print(json.dumps(tower_abs, ensure_ascii=False, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'frames': len(rows), 'p42_by_distance': p42,
                                    'policy_comparison': policy_summary,
                                    'fixed_camera_tower_independent': tower_abs},
                                   ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
