"""P3 flight-distribution shift analysis, train-only flight calibration, and val replay."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from perception.graded_gating import DistanceBinCalibrator, GradedGate, _logit


DIST_BINS = [(0.0, 45.0), (45.0, 80.0), (80.0, 120.0), (120.0, 1e9)]
DIST_NAMES = ["0-45m", "45-80m", "80-120m", "120+m"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bin_label(d):
    if d is None or not np.isfinite(d):
        return "unknown"
    for (a, b), name in zip(DIST_BINS, DIST_NAMES):
        if a <= d < b:
            return name
    return DIST_NAMES[-1]


def summarize_scores(vals):
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "p10": round(float(np.quantile(arr, 0.10)), 4),
        "p25": round(float(np.quantile(arr, 0.25)), 4),
        "p50": round(float(np.quantile(arr, 0.50)), 4),
        "p75": round(float(np.quantile(arr, 0.75)), 4),
        "p90": round(float(np.quantile(arr, 0.90)), 4),
        "acc_0913": round(float((arr >= 0.9130731477441585).mean()), 4),
    }


def write_sealed_test_manifest(p1_dir: Path, test_ids: list, out_path: Path):
    entries = []
    for tid in sorted(test_ids):
        tdir = p1_dir / tid / "repeat-01"
        h5 = tdir / "shadow.h5"
        jl = tdir / "shadow-records.jsonl"
        entries.append({
            "target_id": tid,
            "present": h5.is_file() and jl.is_file(),
            "shadow_h5_sha256": sha256_file(h5) if h5.is_file() else None,
            "shadow_h5_bytes": h5.stat().st_size if h5.is_file() else 0,
            "shadow_jsonl_sha256": sha256_file(jl) if jl.is_file() else None,
            "frames": sum(1 for l in jl.read_text().splitlines() if l.strip()) if jl.is_file() else 0,
        })
    manifest = {
        "policy": "appearance.test flight sequences sealed strictly for P5; never synced, scored, or inspected during P3 calibration/selection.",
        "targets": entries,
    }
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_teleport_oof_train(dev_records_path: Path, train_ids: set, tby_id: dict):
    """Load teleport OOF records strictly for appearance.train targets."""
    raw = json.loads(dev_records_path.read_text())
    recs = raw["train"] if isinstance(raw, dict) and "train" in raw else raw
    # Map sample_name -> primary visible scene distance from positive records (leak-free distance bin)
    img_dist = {}
    for r in recs:
        if r.get("positive") and r.get("horizontal_distance_m") is not None:
            img_dist[r["sample_name"]] = float(r["horizontal_distance_m"])

    pos_by_cat_bin = defaultdict(list)
    pos_pairs, neg_pairs = [], []
    for r in recs:
        tid = r.get("target_id")
        if tid not in train_ids:
            continue
        sc = r.get("score")
        if sc is None:
            continue
        is_pos = bool(r.get("positive"))
        d = r.get("horizontal_distance_m") if is_pos else img_dist.get(r.get("sample_name"))
        if d is None:
            continue
        d = float(d)
        cat = tby_id[tid]["category"]
        if is_pos:
            pos_by_cat_bin[(cat, bin_label(d))].append(float(sc))
            pos_by_cat_bin[("all", bin_label(d))].append(float(sc))
            if r.get("correct"):
                pos_pairs.append((float(sc), 1, d))
            else:
                neg_pairs.append((float(sc), 0, d))
        else:
            neg_pairs.append((float(sc), 0, d))
    return pos_by_cat_bin, pos_pairs, neg_pairs


def build_flight_train_samples(train_rows: list, tby_id: dict, by_cat: dict):
    """Build train-only flight positive/negative samples binned by observed RGB-D candidate depth."""
    pos_by_cat_bin = defaultdict(list)
    hit_by_cat_bin = defaultdict(list)
    pos_samples, neg_samples = [], []
    for r in train_rows:
        if not r["candidates"]:
            continue
        tid = r["target_id"]
        cat = r["category"]
        sc_map = r["scores_by_template"]
        if tid not in sc_map or not sc_map[tid]:
            continue
        best_i = int(np.argmax(sc_map[tid]))
        best_sc = float(sc_map[tid][best_i])
        cand = r["candidates"][best_i]
        obs_d = cand.get("depth_m")
        gt_d = r.get("horiz_dist_m")
        hit = bool(cand.get("point_in_gt_box"))

        pos_by_cat_bin[(cat, bin_label(gt_d))].append(best_sc)
        pos_by_cat_bin[("all", bin_label(gt_d))].append(best_sc)
        hit_by_cat_bin[(cat, bin_label(gt_d))].append(int(hit))
        hit_by_cat_bin[("all", bin_label(gt_d))].append(int(hit))

        d_for_cal = obs_d if (obs_d is not None and np.isfinite(obs_d) and obs_d > 0) else gt_d
        if d_for_cal is None:
            continue
        if hit:
            pos_samples.append((best_sc, 1, float(d_for_cal)))
        else:
            neg_samples.append((best_sc, 0, float(d_for_cal)))

        # Absent-template negative queries on the same train flight frame:
        # Use cross-class templates (and wrong same-class templates with lower weight via subsampling)
        # binned strictly by the candidate's observed depth_m.
        cross_tids = [x for x in sc_map if x not in by_cat[cat] and sc_map[x]]
        if cross_tids:
            # Take max cross-class template per frame + median cross-class template (hard + typical negatives)
            cross_best = [max(sc_map[x]) for x in cross_tids]
            order = np.argsort(cross_best)
            for idx_pick in {int(order[-1]), int(order[len(order) // 2])}:
                cx = cross_tids[idx_pick]
                ci = int(np.argmax(sc_map[cx]))
                cd = r["candidates"][ci].get("depth_m")
                cd = cd if (cd is not None and np.isfinite(cd) and cd > 0) else d_for_cal
                neg_samples.append((float(sc_map[cx][ci]), 0, float(cd)))
    return pos_by_cat_bin, hit_by_cat_bin, pos_samples, neg_samples


def eval_sequence(seq_rows, query_tid, cal, tby_id, by_cat, policy_name, gate_cfg=None, use_margin=False,
                  exclude_same_mesh=False, hangar_confirm_max_m=None):
    cat_q = tby_id[query_tid]["category"]
    ag_q = tby_id[query_tid].get("appearance_group")
    same_peers = [
        x for x in by_cat[cat_q]
        if x != query_tid and (not exclude_same_mesh or not ag_q or tby_id[x].get("appearance_group") != ag_q)
    ]
    if policy_name == "legacy_0913":
        streak = 0
        last_t = None
        perm = conf = near_adm = hit_perm = 0
        first_p = None
        for i, r in enumerate(seq_rows):
            sc_list = r["scores_by_template"].get(query_tid, [])
            if not sc_list:
                streak = 0
                continue
            best_i = int(np.argmax(sc_list))
            cand = r["candidates"][best_i]
            sc = sc_list[best_i]
            dm = cand["depth_m"]
            ts = r["time_s"] if r["time_s"] is not None else i * 1.0
            streak = (streak + 1 if (last_t is not None and ts - last_t <= 2.0) else 1) if sc >= 0.9130731477441585 else 0
            last_t = ts
            if streak >= 3:
                perm += 1
                conf += 1
                if cand.get("point_in_gt_box"):
                    hit_perm += 1
                if first_p is None:
                    first_p = i
                if dm is not None and dm <= 45.0:
                    near_adm += 1
        return {
            "frames": len(seq_rows),
            "permitted": perm,
            "permitted_hit": hit_perm,
            "confirmed": conf > 0,
            "near_admitted": near_adm > 0,
            "rejected": False,
            "first_perm": first_p,
        }

    eff_cfg = dict(gate_cfg or {})
    if hangar_confirm_max_m is not None and cat_q == "hangar":
        eff_cfg["confirm_max_range_m"] = float(hangar_confirm_max_m)
    gate = GradedGate(cal, eff_cfg)
    perm = hit_perm = 0
    first_p = None
    confirmed = near_adm = rejected = False
    for i, r in enumerate(seq_rows):
        sc_list = r["scores_by_template"].get(query_tid, [])
        if not sc_list:
            continue
        best_i = int(np.argmax(sc_list))
        cand = r["candidates"][best_i]
        sc = sc_list[best_i]
        ts = r["time_s"] if r["time_s"] is not None else i * 1.0
        others = None
        if use_margin:
            others = [
                r["scores_by_template"][p][best_i]
                for p in same_peers
                if p in r["scores_by_template"] and len(r["scores_by_template"][p]) > best_i
            ]
        out = gate.update(
            sc, cand["point_xy"], cand["box_xyxy"], cand["depth_m"],
            r["rgb_metadata"], ts, same_class_other_scores=others,
        )
        if out["motion_permitted"]:
            perm += 1
            if cand.get("point_in_gt_box"):
                hit_perm += 1
            if first_p is None:
                first_p = i
        if out["identity_confirmed"]:
            confirmed = True
        if out["near_handoff_permitted"]:
            near_adm = True
        if out["identity_rejected"]:
            rejected = True
    return {
        "frames": len(seq_rows),
        "permitted": perm,
        "permitted_hit": hit_perm,
        "confirmed": confirmed,
        "near_admitted": near_adm,
        "rejected": rejected,
        "first_perm": first_p,
        "final_llr": round(gate.llr_sum, 3),
        "mid_llr": round(gate.mid_llr_sum, 3),
        "mean_margin": round(float(np.mean(gate.margin_history)), 3) if gate.margin_history else None,
    }


def evaluate_split_replay(seqs_by_tid: dict, cal, tby_id: dict, by_cat: dict, policy_name: str, gate_cfg=None,
                          use_margin=False, exclude_same_mesh=False, hangar_confirm_max_m=None):
    pos_res = {}
    wrong_same_conf = wrong_same_near = wrong_same_tot = 0
    diff_mesh_ws_conf = diff_mesh_ws_tot = 0
    same_mesh_ws_conf = same_mesh_ws_tot = 0
    cross_perm = cross_conf = cross_tot = 0
    perm_total = perm_hit_total = 0

    for tid, seq_rows in sorted(seqs_by_tid.items()):
        cat = tby_id[tid]["category"]
        ag_t = tby_id[tid].get("appearance_group")
        r_pos = eval_sequence(seq_rows, tid, cal, tby_id, by_cat, policy_name, gate_cfg, use_margin,
                              exclude_same_mesh, hangar_confirm_max_m)
        pos_res[tid] = r_pos
        perm_total += r_pos["permitted"]
        perm_hit_total += r_pos["permitted_hit"]

        for peer in by_cat[cat]:
            if peer == tid:
                continue
            r_ws = eval_sequence(seq_rows, peer, cal, tby_id, by_cat, policy_name, gate_cfg, use_margin,
                                 exclude_same_mesh, hangar_confirm_max_m)
            wrong_same_tot += 1
            wrong_same_conf += int(r_ws["confirmed"])
            wrong_same_near += int(r_ws["near_admitted"])
            if ag_t and tby_id[peer].get("appearance_group") == ag_t:
                same_mesh_ws_tot += 1
                same_mesh_ws_conf += int(r_ws["confirmed"])
            else:
                diff_mesh_ws_tot += 1
                diff_mesh_ws_conf += int(r_ws["confirmed"])

        for other_cat, other_tids in by_cat.items():
            if other_cat == cat:
                continue
            for cx in other_tids:
                r_cx = eval_sequence(seq_rows, cx, cal, tby_id, by_cat, policy_name, gate_cfg, use_margin,
                                     exclude_same_mesh, hangar_confirm_max_m)
                cross_tot += 1
                cross_perm += int(r_cx["permitted"] > 0)
                cross_conf += int(r_cx["confirmed"])

    pos_perm_seqs = sum(1 for v in pos_res.values() if v["permitted"] > 0)
    pos_conf_seqs = sum(1 for v in pos_res.values() if v["confirmed"])
    pos_near_seqs = sum(1 for v in pos_res.values() if v["near_admitted"])
    return {
        "pos_seqs": len(pos_res),
        "pos_permitted_seqs": pos_perm_seqs,
        "pos_confirmed_seqs": pos_conf_seqs,
        "pos_near_admitted_seqs": pos_near_seqs,
        "permitted_frame_hit_rate": round(perm_hit_total / perm_total, 4) if perm_total else None,
        "permitted_frames": perm_total,
        "permitted_hit_frames": perm_hit_total,
        "wrong_same_confirmed": f"{wrong_same_conf}/{wrong_same_tot}",
        "diff_mesh_ws_confirmed": f"{diff_mesh_ws_conf}/{diff_mesh_ws_tot}",
        "same_mesh_ws_confirmed": f"{same_mesh_ws_conf}/{same_mesh_ws_tot}",
        "wrong_same_near": f"{wrong_same_near}/{wrong_same_tot}",
        "cross_cat_moved": f"{cross_perm}/{cross_tot}",
        "cross_cat_confirmed": f"{cross_conf}/{cross_tot}",
        "per_target": pos_res,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--p1-dir", type=Path, default=ROOT / "runs/evaluation/20261119-p1-revalidation")
    parser.add_argument("--dev-records", type=Path, default=ROOT / "runs/evaluation/20261117-opus-completion/development-records.json")
    parser.add_argument("--static-calibrator", type=Path, default=ROOT / "runs/evaluation/20261119-p2-gating-offline/calibrator-train.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/evaluation/20261120-p3-flight-calibration")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registry = json.loads((ROOT / "runs/evaluation/20261119-dev-registry/registry.json").read_text())
    catalog = json.loads((ROOT / "data/templates/airport_targets_20261116/catalog.json").read_text())
    tby_id = {t["id"]: t for t in catalog["targets"]}
    by_idx = {t["index"]: t["id"] for t in catalog["targets"]}
    by_cat = defaultdict(list)
    for t in catalog["targets"]:
        by_cat[t["category"]].append(t["id"])

    train_ids = {by_idx[i] for i in registry["appearance_split"]["train"]}
    val_ids = {by_idx[i] for i in registry["appearance_split"]["validation"]}
    test_ids = {by_idx[i] for i in registry["appearance_split"]["test"]}

    # 1. Seal test targets strictly
    sealed = write_sealed_test_manifest(args.p1_dir, list(test_ids), args.out_dir / "sealed-test-manifest.json")

    rows = [json.loads(l) for l in args.cache.read_text().splitlines() if l.strip()]
    leaked = [r["target_id"] for r in rows if r["target_id"] in test_ids]
    if leaked:
        raise RuntimeError(f"Test target leakage detected in P3 cache: {set(leaked)}")

    train_rows = [r for r in rows if r["target_id"] in train_ids]
    val_rows = [r for r in rows if r["target_id"] in val_ids]

    # 2. Teleport vs continuous flight distribution shift (strictly on appearance.train)
    tp_pos_bin, tp_pos_pairs, tp_neg_pairs = load_teleport_oof_train(args.dev_records, train_ids, tby_id)
    fl_pos_bin, fl_hit_bin, fl_pos_pairs, fl_neg_pairs = build_flight_train_samples(train_rows, tby_id, by_cat)

    shift = {}
    for cat in ["all", "aircraft", "hangar", "tower"]:
        shift[cat] = {}
        for dname in DIST_NAMES:
            tp_s = summarize_scores(tp_pos_bin.get((cat, dname), []))
            fl_s = summarize_scores(fl_pos_bin.get((cat, dname), []))
            hits = fl_hit_bin.get((cat, dname), [])
            fl_s["hit_rate"] = round(float(np.mean(hits)), 4) if hits else None
            shift[cat][dname] = {"teleport_oof": tp_s, "flight_shadow": fl_s}

    shift_out = {
        "scope": "Strictly appearance.train targets (10/11 present; 11-hangar-2019 missing due to baseline far preflight failure). Selection bias and single-trajectory autocorrelation apply.",
        "train_targets_present": sorted({r["target_id"] for r in train_rows}),
        "train_targets_missing": sorted(train_ids - {r["target_id"] for r in train_rows}),
        "train_flight_frames": len(train_rows),
        "by_category_distance": shift,
    }
    (args.out_dir / "distribution-shift.json").write_text(json.dumps(shift_out, ensure_ascii=False, indent=2) + "\n")

    # 3. Fit flight-only & blended (static OOF + flight) train-only PAVA calibrators
    static_cal = DistanceBinCalibrator.from_dict(json.loads(args.static_calibrator.read_text()))

    fl_all = fl_pos_pairs + fl_neg_pairs
    flight_cal = DistanceBinCalibrator()
    flight_cal.fit([x[0] for x in fl_all], [x[1] for x in fl_all], [x[2] for x in fl_all])
    (args.out_dir / "calibrator-flight-train.json").write_text(
        json.dumps(flight_cal.to_dict(), ensure_ascii=False, indent=2) + "\n"
    )

    blend_all = tp_pos_pairs + tp_neg_pairs + fl_pos_pairs + fl_neg_pairs
    blend_cal = DistanceBinCalibrator()
    blend_cal.fit([x[0] for x in blend_all], [x[1] for x in blend_all], [x[2] for x in blend_all])
    (args.out_dir / "calibrator-blended-train.json").write_text(
        json.dumps(blend_cal.to_dict(), ensure_ascii=False, indent=2) + "\n"
    )

    # 4. Evaluate policies on appearance.validation (3 continuous flight targets: 06, 12, 16)
    #    and record train-only diagnostics separately.
    val_seqs = defaultdict(list)
    for r in val_rows:
        val_seqs[r["target_id"]].append(r)
    train_seqs = defaultdict(list)
    for r in train_rows:
        train_seqs[r["target_id"]].append(r)

    base_cfg = {
        "mid_range_m": 80.0,
        "near_handoff_range_m": 45.0,
        "far_consistent_frames": 2,
        "far_min_score": 0.50,
        "far_llr_threshold": -0.2,
        "confirm_max_range_m": 80.0,
        "confirm_llr_threshold": 2.0,
        "confirm_min_mid_frames": 3,
        "reject_llr_threshold": -3.0,
        "llr_decay": 0.92,
        "min_evidence_move_m": 2.0,
        "stationary_llr_weight": 0.2,
        "require_same_class_margin": True,
        "min_same_class_logit_margin": 0.3,
    }
    relaxed_cfg = dict(base_cfg, far_min_score=0.35, min_same_class_logit_margin=0.0)

    eval_configs = [
        ("legacy_0913", static_cal, None, False, False, None),
        ("static_cal_p42_m03", static_cal, base_cfg, True, False, None),
        ("flight_cal_p42_m03", flight_cal, base_cfg, True, False, None),
        ("blended_cal_p42_m03", blend_cal, base_cfg, True, False, None),
        ("p26_blended_h160", blend_cal, base_cfg, True, True, 160.0),
    ]

    val_summary = {}
    train_summary = {}
    for name, cal_obj, cfg, use_m, ex_same, h160 in eval_configs:
        val_summary[name] = evaluate_split_replay(
            val_seqs, cal_obj, tby_id, by_cat, name if name == "legacy_0913" else "graded",
            cfg, use_m, ex_same, h160,
        )
        train_summary[name] = evaluate_split_replay(
            train_seqs, cal_obj, tby_id, by_cat, name if name == "legacy_0913" else "graded",
            cfg, use_m, ex_same, h160,
        )

    replay_out = {
        "sealed_test_targets": [e["target_id"] for e in sealed["targets"]],
        "validation_targets": sorted(val_seqs.keys()),
        "train_targets": sorted(train_seqs.keys()),
        "validation_replay": val_summary,
        "train_diagnostic_replay": train_summary,
    }
    (args.out_dir / "val-replay-summary.json").write_text(
        json.dumps(replay_out, ensure_ascii=False, indent=2) + "\n"
    )

    print("=== P3 DISTRIBUTION SHIFT (appearance.train: teleport OOF vs continuous flight) ===")
    for cat in ["aircraft", "hangar", "tower"]:
        print(f"  [{cat}]")
        for dname in DIST_NAMES:
            tp = shift[cat][dname]["teleport_oof"]
            fl = shift[cat][dname]["flight_shadow"]
            if tp["n"] or fl["n"]:
                print(
                    f"    {dname:7s}: teleport(n={tp['n']:4d}, p50={tp.get('p50')}, acc913={tp.get('acc_0913')}) | "
                    f"flight(n={fl['n']:3d}, p50={fl.get('p50')}, acc913={fl.get('acc_0913')}, hit={fl.get('hit_rate')})"
                )

    print("\n=== P3 VALIDATION REPLAY (appearance.validation: 06, 12, 16 continuous flight) ===")
    for name, res in val_summary.items():
        print(
            f"  {name:24s}: pos_perm={res['pos_permitted_seqs']}/{res['pos_seqs']} "
            f"pos_conf={res['pos_confirmed_seqs']}/{res['pos_seqs']} "
            f"pos_near={res['pos_near_admitted_seqs']}/{res['pos_seqs']} "
            f"hit_on_perm={res['permitted_frame_hit_rate']} "
            f"diff_ws={res['diff_mesh_ws_confirmed']} same_mesh_ws={res['same_mesh_ws_confirmed']} "
            f"cx_move={res['cross_cat_moved']} cx_conf={res['cross_cat_confirmed']}"
        )

    print("\n=== P3 TRAIN DIAGNOSTIC REPLAY (appearance.train: 10 targets incl. hangars + tower 18) ===")
    for name, res in train_summary.items():
        print(
            f"  {name:24s}: pos_perm={res['pos_permitted_seqs']}/{res['pos_seqs']} "
            f"pos_conf={res['pos_confirmed_seqs']}/{res['pos_seqs']} "
            f"pos_near={res['pos_near_admitted_seqs']}/{res['pos_seqs']} "
            f"hit_on_perm={res['permitted_frame_hit_rate']} "
            f"diff_ws={res['diff_mesh_ws_confirmed']} same_mesh_ws={res['same_mesh_ws_confirmed']} "
            f"cx_move={res['cross_cat_moved']} cx_conf={res['cross_cat_confirmed']}"
        )


if __name__ == "__main__":
    main()
