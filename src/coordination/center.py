"""Central dispatcher for two-stage multi-UAV cooperative survey."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from coordination.store import SurveyStore, utc_now_iso
from coordination.vlm import BaseVLMClient, VLMImage


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SurveyCenter:
    """Centralized dispatcher coordinating two batches of UAVs and a VLM assessor."""

    def __init__(
        self,
        mission_id: str,
        output_dir: Path | str,
        vlm_client: BaseVLMClient,
        config: dict[str, Any] | None = None,
        allow_existing: bool = False,
        dispatch_policy: str = "always_confirm",
    ) -> None:
        if dispatch_policy not in {"always_confirm", "only_flagged"}:
            raise ValueError(f"Unsupported dispatch_policy: {dispatch_policy}")
        self.mission_id = mission_id
        self.dispatch_policy = dispatch_policy
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not allow_existing:
            raise FileExistsError(f"Refusing to overwrite non-empty mission directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)

        self.vlm_client = vlm_client
        self.store = SurveyStore(self.output_dir / "survey.db")
        merged_config = dict(config or {})
        merged_config.setdefault("dispatch_policy", dispatch_policy)
        try:
            self.store.get_mission(mission_id)
        except KeyError:
            self.store.create_mission(
                mission_id=mission_id,
                vlm_backend=vlm_client.backend_name,
                vlm_model=vlm_client.model_name,
                config=merged_config,
            )

    def close(self) -> None:
        self.store.close()

    def register_target(
        self,
        target_id: str,
        round1_drone_id: str,
        round1_task_id: str | None = None,
        ground_truth_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_id = round1_task_id or f"{self.mission_id}-{target_id}-R1"
        return self.store.register_target(
            mission_id=self.mission_id,
            target_id=target_id,
            round1_task_id=task_id,
            round1_drone_id=round1_drone_id,
            ground_truth_label=ground_truth_label,
            metadata=metadata,
        )

    def submit_upload(
        self,
        target_id: str,
        stage: int,
        drone_id: str,
        image_path: Path | str,
        image_id: str,
        captured_at: str | None = None,
        task_id: str | None = None,
        view_type: str = "nadir",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Archive a UAV image under a neutral filename and record it idempotently."""
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        target = self.store.get_target(self.mission_id, target_id)
        src = Path(image_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Upload image not found: {src}")

        suffix = src.suffix.lower() or ".png"
        neutral_rel = Path("images") / target_id / f"{image_id}{suffix}"
        neutral_abs = self.output_dir / neutral_rel
        neutral_abs.parent.mkdir(parents=True, exist_ok=True)

        sha256 = _sha256_file(src)
        if not neutral_abs.exists():
            shutil.copy2(src, neutral_abs)

        effective_task_id = task_id or (
            target["round1_task_id"] if stage == 1 else (target["round2_task_id"] or f"{self.mission_id}-{target_id}-R2")
        )
        record, is_duplicate = self.store.record_upload(
            mission_id=self.mission_id,
            target_id=target_id,
            stage=stage,
            task_id=effective_task_id,
            drone_id=drone_id,
            image_id=image_id,
            captured_at=captured_at or utc_now_iso(),
            view_type=view_type,
            stored_path=neutral_rel.as_posix(),
            sha256=sha256,
            metadata=metadata,
        )

        if not is_duplicate:
            current_status = target["status"]
            if stage == 1 and current_status == "pending_round1":
                self.store.update_target_status(self.mission_id, target_id, "round1_received")
            elif stage == 2:
                recheck = self.store.get_recheck_task(self.mission_id, target_id)
                if recheck is not None and recheck["status"] != "completed":
                    self.store.update_recheck_task_status(self.mission_id, target_id, "image_received")
                if current_status in {"recheck_pending_dispatch", "recheck_dispatched", "round2_vlm_failed"}:
                    self.store.update_target_status(self.mission_id, target_id, "round2_received")

        return {"upload": record, "duplicate": is_duplicate}

    def report_return(
        self,
        target_id: str,
        stage: int,
        drone_id: str,
        status: str = "returned",
        returned_at: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record UAV return-to-home status independently from photo upload."""
        return self.store.record_return_status(
            mission_id=self.mission_id,
            target_id=target_id,
            stage=stage,
            drone_id=drone_id,
            status=status,
            returned_at=returned_at,
            note=note,
        )

    def _build_vlm_images(self, target_id: str, stage: int) -> list[VLMImage]:
        uploads = self.store.list_uploads(self.mission_id, target_id, stage=stage)
        return [
            VLMImage(
                image_id=item["image_id"],
                view_type=item["view_type"],
                image_path=self.output_dir / item["stored_path"],
            )
            for item in uploads
        ]

    def run_initial_assessment(self, target_id: str, force_reassess: bool = False) -> dict[str, Any]:
        """Run Stage-1 VLM assessment for one target and create at most one recheck task."""
        existing = self.store.get_assessment(self.mission_id, target_id, stage=1)
        if existing is not None and existing["status"] == "ok" and not force_reassess:
            recheck = self.store.get_recheck_task(self.mission_id, target_id)
            return {
                "target_id": target_id,
                "status": "ok",
                "cached": True,
                "assessment": existing,
                "recheck_task": recheck,
                "recheck_created": False,
            }

        r1_images = self._build_vlm_images(target_id, stage=1)
        if not r1_images:
            raise ValueError(f"No stage-1 uploads available for target {target_id}")

        try:
            resp = self.vlm_client.assess_round1(target_id, r1_images)
        except Exception as exc:
            assessment = self.store.save_assessment(
                mission_id=self.mission_id,
                target_id=target_id,
                stage=1,
                status="failed",
                model_name=self.vlm_client.model_name,
                latency_ms=None,
                error_message=str(exc),
            )
            self.store.update_target_status(self.mission_id, target_id, "round1_vlm_failed")
            return {
                "target_id": target_id,
                "status": "failed",
                "cached": False,
                "assessment": assessment,
                "recheck_task": None,
                "recheck_created": False,
                "error": str(exc),
            }

        assessment = self.store.save_assessment(
            mission_id=self.mission_id,
            target_id=target_id,
            stage=1,
            status="ok",
            model_name=resp.model_name,
            latency_ms=resp.latency_ms,
            result=resp.result,
            raw_response=resp.raw_response,
        )

        recheck_task = None
        recheck_created = False
        needs_recheck = bool(resp.result.get("needs_recheck"))
        should_dispatch = needs_recheck or (self.dispatch_policy == "always_confirm")
        if should_dispatch:
            uploads = self.store.list_uploads(self.mission_id, target_id, stage=1)
            question = str(resp.result.get("recheck_question", "")).strip()
            if not question:
                uncertainties = [str(x).strip() for x in resp.result.get("uncertainties", []) if str(x).strip()]
                if uncertainties:
                    question = f"首轮初评未见明显表面损伤，请低空近距复核以下盲区并确认是否存在隐蔽损伤：{'；'.join(uncertainties[:2])}"
                else:
                    question = f"首轮初评未见明显损伤，请低空近距拍摄目标 {target_id} 关键部位，二次确认目标是否确实完好无隐蔽形变或破损。"
            focus_region = str(resp.result.get("focus_region", "")).strip() or "目标主体及关键部位"
            action = str(resp.result.get("recommended_action", "closeup_approach")).strip()
            if action == "none":
                action = "closeup_approach"
            recheck_task, recheck_created = self.store.create_recheck_task_once(
                task_id=f"{self.mission_id}-{target_id}-R2",
                mission_id=self.mission_id,
                target_id=target_id,
                round1_image_ids=[u["image_id"] for u in uploads],
                round1_image_paths=[u["stored_path"] for u in uploads],
                question=question,
                focus_region=focus_region,
                recommended_action=action,
            )
            # If stage-2 image already arrived out-of-order, advance directly to round2_received
            r2_uploads = self.store.list_uploads(self.mission_id, target_id, stage=2)
            if r2_uploads:
                self.store.update_recheck_task_status(self.mission_id, target_id, "image_received")
                self.store.update_target_status(self.mission_id, target_id, "round2_received")
            else:
                self.store.update_target_status(self.mission_id, target_id, "recheck_pending_dispatch")
        else:
            self.store.record_return_status(
                mission_id=self.mission_id,
                target_id=target_id,
                stage=2,
                drone_id="",
                status="not_needed",
                returned_at=None,
                note="Stage-1 assessment determined no recheck needed",
            )
            self.store.update_target_status(self.mission_id, target_id, "completed_no_recheck")

        return {
            "target_id": target_id,
            "status": "ok",
            "cached": False,
            "assessment": assessment,
            "recheck_task": recheck_task,
            "recheck_created": recheck_created,
        }

    def dispatch_recheck(self, target_id: str, drone_id: str) -> dict[str, Any]:
        """Assign a Batch-2 UAV and return the complete semantic handover packet."""
        task = self.store.dispatch_recheck_task(self.mission_id, target_id, drone_id)
        target = self.store.get_target(self.mission_id, target_id)
        s1 = self.store.get_assessment(self.mission_id, target_id, stage=1)
        s1_res = (s1.get("result") or {}) if s1 else {}
        if target["status"] == "recheck_pending_dispatch":
            self.store.update_target_status(self.mission_id, target_id, "recheck_dispatched")
        return {
            "task_id": task["task_id"],
            "mission_id": self.mission_id,
            "target_id": target_id,
            "assigned_drone_id": drone_id,
            "target_type": s1_res.get("target_type", "未知目标"),
            "suspected_damaged_parts": s1_res.get("suspected_damaged_parts", []),
            "visible_damage_description": s1_res.get("visible_damage_description", ""),
            "initial_damage_level": s1_res.get("initial_damage_level", "suspected"),
            "secondary_task": s1_res.get("secondary_task")
            or {
                "task_type": task["recommended_action"],
                "focus_part": task["focus_region"],
                "desired_viewpoint": "",
                "task_instruction": task["question"],
                "priority": "medium",
            },
            "round1_image_ids": task["round1_image_ids"],
            "round1_image_paths": [str(self.output_dir / p) for p in task["round1_image_paths"]],
            "question": task["question"],
            "focus_region": task["focus_region"],
            "recommended_action": task["recommended_action"],
            "status": task["status"],
        }

    def run_final_assessment(self, target_id: str, force_reassess: bool = False) -> dict[str, Any]:
        """Run Stage-2 VLM reassessment combining Stage-1 images, question, and Stage-2 images."""
        existing = self.store.get_assessment(self.mission_id, target_id, stage=2)
        if existing is not None and existing["status"] == "ok" and not force_reassess:
            return {
                "target_id": target_id,
                "status": "ok",
                "cached": True,
                "assessment": existing,
            }

        recheck = self.store.get_recheck_task(self.mission_id, target_id)
        if recheck is None:
            raise ValueError(f"No recheck task exists for target {target_id}; cannot run final assessment")

        r1_images = self._build_vlm_images(target_id, stage=1)
        r2_images = self._build_vlm_images(target_id, stage=2)
        if not r1_images or not r2_images:
            raise ValueError(
                f"Both stage-1 ({len(r1_images)}) and stage-2 ({len(r2_images)}) uploads are required for {target_id}"
            )

        try:
            resp = self.vlm_client.assess_round2(
                target_id=target_id,
                question=recheck["question"],
                round1_images=r1_images,
                round2_images=r2_images,
            )
        except Exception as exc:
            assessment = self.store.save_assessment(
                mission_id=self.mission_id,
                target_id=target_id,
                stage=2,
                status="failed",
                model_name=self.vlm_client.model_name,
                latency_ms=None,
                error_message=str(exc),
            )
            self.store.update_target_status(self.mission_id, target_id, "round2_vlm_failed")
            return {
                "target_id": target_id,
                "status": "failed",
                "cached": False,
                "assessment": assessment,
                "error": str(exc),
            }

        assessment = self.store.save_assessment(
            mission_id=self.mission_id,
            target_id=target_id,
            stage=2,
            status="ok",
            model_name=resp.model_name,
            latency_ms=resp.latency_ms,
            result=resp.result,
            raw_response=resp.raw_response,
        )
        self.store.update_recheck_task_status(self.mission_id, target_id, "completed")
        self.store.update_target_status(self.mission_id, target_id, "completed")
        return {
            "target_id": target_id,
            "status": "ok",
            "cached": False,
            "assessment": assessment,
        }

    def get_target_dossier(self, target_id: str) -> dict[str, Any]:
        target = self.store.get_target(self.mission_id, target_id)
        r1_uploads = self.store.list_uploads(self.mission_id, target_id, stage=1)
        r2_uploads = self.store.list_uploads(self.mission_id, target_id, stage=2)
        stage1_assessment = self.store.get_assessment(self.mission_id, target_id, stage=1)
        stage2_assessment = self.store.get_assessment(self.mission_id, target_id, stage=2)
        recheck_task = self.store.get_recheck_task(self.mission_id, target_id)
        events = self.store.list_events(self.mission_id, target_id=target_id)
        return {
            "target": target,
            "round1_uploads": r1_uploads,
            "round2_uploads": r2_uploads,
            "stage1_assessment": stage1_assessment,
            "recheck_task": recheck_task,
            "stage2_assessment": stage2_assessment,
            "events": events,
        }

    def build_report(self) -> dict[str, Any]:
        mission = self.store.get_mission(self.mission_id)
        targets = self.store.list_targets(self.mission_id)
        dossiers = [self.get_target_dossier(t["target_id"]) for t in targets]
        all_events = self.store.list_events(self.mission_id)

        terminal_states = {"completed", "completed_no_recheck"}
        completed_count = sum(1 for d in dossiers if d["target"]["status"] in terminal_states)
        failed_count = sum(1 for d in dossiers if "failed" in d["target"]["status"])
        mission_status = (
            "completed"
            if targets and completed_count == len(targets)
            else ("completed_with_pending" if completed_count > 0 else "active")
        )
        self.store.update_mission_status(self.mission_id, mission_status)
        mission["status"] = mission_status

        return {
            "mission": mission,
            "output_dir": str(self.output_dir),
            "generated_at": utc_now_iso(),
            "summary": {
                "total_targets": len(targets),
                "completed_targets": completed_count,
                "failed_targets": failed_count,
                "pending_targets": len(targets) - completed_count - failed_count,
                "duplicate_uploads_ignored": sum(
                    1 for e in all_events if e["event_type"] == "duplicate_upload_ignored"
                ),
                "recheck_tasks_created": sum(1 for d in dossiers if d["recheck_task"] is not None),
            },
            "targets": dossiers,
        }

    def export_report(self) -> tuple[Path, Path]:
        report = self.build_report()
        json_path = self.output_dir / "report.json"
        md_path = self.output_dir / "report.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._render_markdown_report(report), encoding="utf-8")
        return json_path, md_path

    def _render_markdown_report(self, report: dict[str, Any]) -> str:
        mission = report["mission"]
        summary = report["summary"]
        lines = [
            f"# 多机两阶段协同勘测报告：`{mission['mission_id']}`",
            "",
            f"- 生成时间：{report['generated_at']}",
            f"- 任务状态：`{mission['status']}`",
            f"- VLM 后端：`{mission['vlm_backend']}` / 模型：`{mission['vlm_model']}`",
            f"- 目标进度：完成 {summary['completed_targets']}/{summary['total_targets']}，失败 {summary['failed_targets']}，待处理 {summary['pending_targets']}",
            f"- 复查任务创建数：{summary['recheck_tasks_created']}；拦截重复上传数：{summary['duplicate_uploads_ignored']}",
            "",
            "## 语义评估总览",
            "",
            "| 目标编号 | 识别目标种类 | 受损部位（终评/初评） | 损伤分级 | 复评结论 | 二次任务动作 | 事后核对标签（不送模型） |",
            "|---|---|---|---|---|---|---|",
        ]
        verdict_cn = {
            "confirmed": "已确认损伤 (confirmed)",
            "ruled_out": "已排除损伤/完好 (ruled_out)",
            "still_uncertain": "仍不确定 (still_uncertain)",
        }
        severity_cn = {
            "intact": "完好 (intact)",
            "none": "无明显异常 (none)",
            "suspected": "疑似受损 (suspected)",
            "minor": "轻度损伤 (minor)",
            "moderate": "中度损伤 (moderate)",
            "severe": "重度损伤 (severe)",
            "uncertain": "待定 (uncertain)",
        }
        task_type_cn = {
            "closeup_approach": "低空抵近精查",
            "roof_entry": "屋顶破口入室",
            "side_inspection": "侧视近距精查",
            "intact_verification": "近距完好复核",
            "none": "无需复查",
        }
        priority_cn = {"high": "高", "medium": "中", "low": "低"}

        for d in report["targets"]:
            t = d["target"]
            s1 = d["stage1_assessment"]
            s2 = d["stage2_assessment"]
            res1 = (s1["result"] or {}) if (s1 and s1["status"] == "ok") else {}
            res2 = (s2["result"] or {}) if (s2 and s2["status"] == "ok") else {}

            target_type = res2.get("target_type") or res1.get("target_type") or "-"
            if res2:
                parts_list = res2.get("confirmed_damaged_parts") or []
                parts_str = "、".join(parts_list) if parts_list else "无受损部位"
                sev_str = severity_cn.get(res2.get("damage_severity", ""), res2.get("damage_severity", "-"))
            elif res1:
                parts_list = res1.get("suspected_damaged_parts") or []
                parts_str = ("疑似：" + "、".join(parts_list)) if parts_list else "暂未见明显受损"
                sev_str = severity_cn.get(res1.get("initial_damage_level", ""), res1.get("initial_damage_level", "-"))
            else:
                parts_str = "-"
                sev_str = "-"

            verdict = "-"
            if s2 and s2["status"] == "ok" and res2:
                v = res2.get("verdict", "")
                verdict = verdict_cn.get(v, v)
            elif s2 and s2["status"] == "failed":
                verdict = "复评失败"
            elif t["status"] == "completed_no_recheck":
                verdict = "无需复查"

            sec_task = res1.get("secondary_task") or {}
            act_code = sec_task.get("task_type") or (d["recheck_task"]["recommended_action"] if d["recheck_task"] else "none")
            if res1 and not res1.get("needs_recheck") and d["recheck_task"] is not None:
                act_str = "无损近距复核"
            else:
                act_str = task_type_cn.get(act_code, act_code)
            gt = t["ground_truth_label"] or "-"
            lines.append(
                f"| `{t['target_id']}` | {target_type} | {parts_str} | {sev_str} | {verdict} | {act_str} | {gt} |"
            )

        for d in report["targets"]:
            t = d["target"]
            lines.extend(["", f"## 对象 `{t['target_id']}`", ""])
            lines.append(f"- 状态：`{t['status']}`")
            lines.append(
                f"- 第一批：`{t['round1_drone_id']}`（任务 `{t['round1_task_id']}`），返航状态：`{t['round1_return_status']}`"
                + (f"（时间：{t['round1_returned_at']}）" if t["round1_returned_at"] else "")
            )
            lines.append(
                f"- 第二批：`{t['round2_drone_id'] or '-'}`（任务 `{t['round2_task_id'] or '-'}`），返航状态：`{t['round2_return_status']}`"
                + (f"（时间：{t['round2_returned_at']}）" if t["round2_returned_at"] else "")
            )
            if t["ground_truth_label"]:
                lines.append(f"- 事后对照真值（仅离线审计，未传给模型）：`{t['ground_truth_label']}`")

            lines.extend(["", "### 1. 第一轮原图与语义初评"])
            for u in d["round1_uploads"]:
                abs_p = (self.output_dir / u["stored_path"]).resolve()
                lines.append(
                    f"- 图像 `{u['image_id']}`（视角 `{u['view_type']}`，由 `{u['drone_id']}` 拍摄于 {u['captured_at']}）："
                    f"[{u['stored_path']}](file://{abs_p})"
                )
            s1 = d["stage1_assessment"]
            if s1 is None:
                lines.append("- 尚未执行第一轮评估。")
            elif s1["status"] == "failed":
                lines.append(f"- **第一轮评估失败**：`{s1['error_message']}`")
            else:
                res1 = s1["result"]
                sus_parts = res1.get("suspected_damaged_parts") or []
                lvl1 = severity_cn.get(res1.get("initial_damage_level", ""), res1.get("initial_damage_level", "-"))
                lines.append(f"- 耗时：{s1['latency_ms']:.1f} ms；引用证据：`{', '.join(res1['evidence_image_ids'])}`")
                lines.append(f"- **目标种类**：{res1.get('target_type', '未知目标')}；**初评损伤等级**：{lvl1}")
                lines.append(f"- **疑似受损部位**：{'、'.join(sus_parts) if sus_parts else '未见明显受损部位'}")
                lines.append(f"- **可见损伤描述**：{res1.get('visible_damage_description', res1['initial_summary'])}")
                lines.append(f"- 初评摘要：{res1['initial_summary']}")
                lines.append("- 观察事实：")
                for obs in res1["observations"]:
                    lines.append(f"  - {obs}")
                lines.append("- 待核实疑点：")
                for unc in res1["uncertainties"]:
                    lines.append(f"  - {unc}")

            recheck = d["recheck_task"]
            lines.extend(["", "### 2. 二次复查任务单与交接记录"])
            if recheck is None:
                lines.append("- 未生成复查任务。")
            else:
                res1 = (s1["result"] or {}) if (s1 and s1["status"] == "ok") else {}
                sec_task = res1.get("secondary_task") or {}
                t_type = sec_task.get("task_type") or recheck["recommended_action"]
                prio = priority_cn.get(sec_task.get("priority", "medium"), sec_task.get("priority", "中"))
                lines.append(f"- 复查任务编号：`{recheck['task_id']}`（状态：`{recheck['status']}`，优先级：**{prio}**）")
                lines.append(f"- 任务类型：{task_type_cn.get(t_type, t_type)}（`{t_type}`）")
                lines.append(f"- 瞄准部位：{sec_task.get('focus_part') or recheck['focus_region']}")
                if sec_task.get("desired_viewpoint"):
                    lines.append(f"- 期望机位与视角：{sec_task['desired_viewpoint']}")
                lines.append(f"- **二次任务指令**：{sec_task.get('task_instruction') or recheck['question']}")
                lines.append(f"- 交接首轮图像：`{', '.join(recheck['round1_image_ids'])}`")

            lines.extend(["", "### 3. 第二轮补拍与语义定损终评"])
            if not d["round2_uploads"]:
                lines.append("- 暂无第二轮图像。")
            else:
                for u in d["round2_uploads"]:
                    abs_p = (self.output_dir / u["stored_path"]).resolve()
                    lines.append(
                        f"- 图像 `{u['image_id']}`（视角 `{u['view_type']}`，由 `{u['drone_id']}` 拍摄于 {u['captured_at']}）："
                        f"[{u['stored_path']}](file://{abs_p})"
                    )
            s2 = d["stage2_assessment"]
            if s2 is None:
                lines.append("- 尚未执行第二轮复评。")
            elif s2["status"] == "failed":
                lines.append(f"- **第二轮复评失败**：`{s2['error_message']}`")
            else:
                res2 = s2["result"]
                v_label = verdict_cn.get(res2["verdict"], res2["verdict"])
                sev_label = severity_cn.get(res2.get("damage_severity", ""), res2.get("damage_severity", "-"))
                conf_parts = res2.get("confirmed_damaged_parts") or []
                lines.append(f"- 耗时：{s2['latency_ms']:.1f} ms；引用证据：`{', '.join(res2['evidence_image_ids'])}`")
                lines.append(f"- **目标种类**：{res2.get('target_type', '未知目标')}；**复评判定**：{v_label}；**损伤分级**：{sev_label}")
                lines.append(f"- **确认受损部位**：{'、'.join(conf_parts) if conf_parts else '无（确认完好）'}")
                lines.append(f"- **综合损伤描述**：{res2.get('damage_description', res2['final_summary'])}")
                lines.append(f"- 原复查问题：{res2['original_question']}")
                lines.append("- 第二轮新增证据：")
                for ev in res2["new_evidence"]:
                    lines.append(f"  - {ev}")
                lines.append(f"- 推理与变化说明：{res2['reasoning']}")
                lines.append(f"- **最终结论**：{res2['final_summary']}")
                if res2.get("recommended_followup"):
                    lines.append(f"- **后续处置建议**：{res2['recommended_followup']}")

        lines.append("")
        return "\n".join(lines)
