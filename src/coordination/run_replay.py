"""Offline two-stage multi-UAV cooperative survey replay entry point."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.paths import CONFIGS, RUNS, project_path
from coordination.center import SurveyCenter
from coordination.vlm import BaseVLMClient, MockVLMClient, OllamaVLMClient


DEFAULT_MANIFEST = CONFIGS / "missions" / "coordination_replay_3targets.json"
DEFAULT_SSH_HOST = "ryh2025@192.168.31.237"
DEFAULT_SSH_KEY = "/home/hui/.ssh/id_ed25519_501"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline two-stage multi-UAV cooperative survey replay via SurveyCenter."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to replay manifest JSON (default: configs/missions/coordination_replay_3targets.json).",
    )
    parser.add_argument(
        "--mission-id",
        default=None,
        help="Mission identifier (default: timestamped id).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: runs/coordination/<mission-id>).",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of neutral target IDs to run (e.g. OBJ-A).",
    )
    parser.add_argument(
        "--vlm",
        choices=["mock", "ollama"],
        default="mock",
        help="VLM backend: mock (offline deterministic) or ollama (default: mock).",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:9b",
        help="Ollama model name when --vlm ollama (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP base URL (local or remote when paired with --ssh-host).",
    )
    parser.add_argument(
        "--ssh-host",
        default=DEFAULT_SSH_HOST,
        help="SSH host running Ollama; pass empty string '' for direct local HTTP.",
    )
    parser.add_argument(
        "--ssh-key",
        default=DEFAULT_SSH_KEY,
        help="SSH private key path for --ssh-host.",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable thinking mode on Ollama Qwen3.5 models (default: disabled).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for Ollama VLM (default: 0.1).",
    )
    parser.add_argument(
        "--dispatch-policy",
        choices=["always_confirm", "only_flagged"],
        default="always_confirm",
        help="Stage-2 dispatch policy: always_confirm (dispatch even if Stage 1 sees no damage) or only_flagged.",
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Scramble cross-target arrival order and inject duplicate uploads & delayed returns.",
    )
    return parser.parse_args(argv)


def _build_vlm(args: argparse.Namespace) -> BaseVLMClient:
    if args.vlm == "mock":
        return MockVLMClient(model_name="mock-vlm-v1")
    ssh_host = args.ssh_host.strip() if args.ssh_host else None
    ssh_key = args.ssh_key.strip() if (ssh_host and args.ssh_key) else None
    return OllamaVLMClient(
        model_name=args.model,
        base_url=args.ollama_url,
        ssh_host=ssh_host,
        ssh_key=ssh_key,
        think=args.think,
        temperature=args.temperature,
    )


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = list(manifest["targets"])
    if args.targets:
        wanted = set(args.targets)
        targets = [t for t in targets if t["target_id"] in wanted]
        if not targets:
            raise ValueError(f"No matching targets found in {manifest_path} for {args.targets}")

    mission_id = args.mission_id or datetime.now().strftime("replay-%Y%m%d-%H%M%S")
    output_dir = project_path(args.output_dir) if args.output_dir else (RUNS / "coordination" / mission_id)
    vlm = _build_vlm(args)

    center = SurveyCenter(
        mission_id=mission_id,
        output_dir=output_dir,
        vlm_client=vlm,
        dispatch_policy=args.dispatch_policy,
        config={
            "manifest": str(manifest_path),
            "stress_test": bool(args.stress_test),
            "dispatch_policy": args.dispatch_policy,
            "vlm_backend": args.vlm,
            "vlm_model": vlm.model_name,
            "think": bool(args.think) if args.vlm == "ollama" else None,
        },
    )

    try:
        for item in targets:
            center.register_target(
                target_id=item["target_id"],
                round1_drone_id=item["round1_drone_id"],
                ground_truth_label=item.get("ground_truth_label"),
            )

        # In stress-test mode, reverse target arrival order and interleave stages
        ordered = list(reversed(targets)) if args.stress_test else list(targets)

        if args.stress_test and len(ordered) >= 2:
            # First target in scrambled order completes Stage 1 AND Stage 2 before others even start
            first = ordered[0]
            _deliver_round1(center, first, duplicate=True)
            res1 = center.run_initial_assessment(first["target_id"])
            # Duplicate upload AFTER initial assessment must not create a second recheck task
            _deliver_round1(center, first, duplicate=False)
            center.run_initial_assessment(first["target_id"])
            if res1["recheck_task"] is not None:
                center.dispatch_recheck(first["target_id"], first["round2_drone_id"])
                _deliver_round2(center, first, duplicate=True)
                center.run_final_assessment(first["target_id"])
                center.report_return(first["target_id"], stage=2, drone_id=first["round2_drone_id"], status="returned")
            center.report_return(first["target_id"], stage=1, drone_id=first["round1_drone_id"], status="returned")

            # Remaining targets arrive with interleaved upload/return and duplicate transmissions
            for item in ordered[1:]:
                _deliver_round1(center, item, duplicate=True)
                res1 = center.run_initial_assessment(item["target_id"])
                center.report_return(item["target_id"], stage=1, drone_id=item["round1_drone_id"], status="returned")
                if res1["recheck_task"] is not None:
                    center.dispatch_recheck(item["target_id"], item["round2_drone_id"])
                    _deliver_round2(center, item, duplicate=True)
                    center.run_final_assessment(item["target_id"])
                    center.report_return(item["target_id"], stage=2, drone_id=item["round2_drone_id"], status="returned")
        else:
            for item in ordered:
                _deliver_round1(center, item, duplicate=args.stress_test)
                res1 = center.run_initial_assessment(item["target_id"])
                center.report_return(item["target_id"], stage=1, drone_id=item["round1_drone_id"], status="returned")
                if res1["recheck_task"] is not None:
                    center.dispatch_recheck(item["target_id"], item["round2_drone_id"])
                    _deliver_round2(center, item, duplicate=args.stress_test)
                    center.run_final_assessment(item["target_id"])
                    center.report_return(item["target_id"], stage=2, drone_id=item["round2_drone_id"], status="returned")

        json_path, md_path = center.export_report()
        report = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"Mission completed: {mission_id}")
        print(f"  DB:     {output_dir / 'survey.db'}")
        print(f"  JSON:   {json_path}")
        print(f"  Report: {md_path}")
        print(f"  Summary: {json.dumps(report['summary'], ensure_ascii=False)}")
        return report
    finally:
        center.close()


def _deliver_round1(center: SurveyCenter, item: dict[str, Any], duplicate: bool) -> None:
    for u in item["round1_uploads"]:
        img_path = project_path(u["path"])
        center.submit_upload(
            target_id=item["target_id"],
            stage=1,
            drone_id=item["round1_drone_id"],
            image_path=img_path,
            image_id=u["image_id"],
            captured_at=u.get("captured_at"),
            view_type=u.get("view_type", "nadir"),
        )
        if duplicate:
            center.submit_upload(
                target_id=item["target_id"],
                stage=1,
                drone_id=item["round1_drone_id"],
                image_path=img_path,
                image_id=u["image_id"],
                captured_at=u.get("captured_at"),
                view_type=u.get("view_type", "nadir"),
            )


def _deliver_round2(center: SurveyCenter, item: dict[str, Any], duplicate: bool) -> None:
    for u in item["round2_uploads"]:
        img_path = project_path(u["path"])
        center.submit_upload(
            target_id=item["target_id"],
            stage=2,
            drone_id=item["round2_drone_id"],
            image_path=img_path,
            image_id=u["image_id"],
            captured_at=u.get("captured_at"),
            view_type=u.get("view_type", "closeup"),
        )
        if duplicate:
            center.submit_upload(
                target_id=item["target_id"],
                stage=2,
                drone_id=item["round2_drone_id"],
                image_path=img_path,
                image_id=u["image_id"],
                captured_at=u.get("captured_at"),
                view_type=u.get("view_type", "closeup"),
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_replay(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
