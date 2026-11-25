"""VLM assessment interfaces and Ollama/Mock implementations for two-stage survey."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request


ROUND1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_type": {"type": "string"},
        "suspected_damaged_parts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "visible_damage_description": {"type": "string"},
        "initial_damage_level": {
            "type": "string",
            "enum": ["none", "suspected", "minor", "moderate", "severe"],
        },
        "observations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_image_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
        },
        "needs_recheck": {"type": "boolean"},
        "recheck_question": {"type": "string"},
        "focus_region": {"type": "string"},
        "recommended_action": {
            "type": "string",
            "enum": ["closeup_approach", "roof_entry", "side_inspection", "none"],
        },
        "secondary_task": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["closeup_approach", "roof_entry", "side_inspection", "intact_verification"],
                },
                "focus_part": {"type": "string"},
                "desired_viewpoint": {"type": "string"},
                "task_instruction": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["task_type", "focus_part", "desired_viewpoint", "task_instruction", "priority"],
        },
        "initial_summary": {"type": "string"},
    },
    "required": [
        "target_type",
        "suspected_damaged_parts",
        "visible_damage_description",
        "initial_damage_level",
        "observations",
        "evidence_image_ids",
        "uncertainties",
        "needs_recheck",
        "recheck_question",
        "focus_region",
        "recommended_action",
        "secondary_task",
        "initial_summary",
    ],
}

ROUND2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_type": {"type": "string"},
        "original_question": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["confirmed", "ruled_out", "still_uncertain"],
        },
        "damage_severity": {
            "type": "string",
            "enum": ["intact", "minor", "moderate", "severe", "uncertain"],
        },
        "confirmed_damaged_parts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "damage_description": {"type": "string"},
        "evidence_image_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "new_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
        "final_summary": {"type": "string"},
        "recommended_followup": {"type": "string"},
    },
    "required": [
        "target_type",
        "original_question",
        "verdict",
        "damage_severity",
        "confirmed_damaged_parts",
        "damage_description",
        "evidence_image_ids",
        "new_evidence",
        "reasoning",
        "final_summary",
        "recommended_followup",
    ],
}

ROUND1_SYSTEM_PROMPT = (
    "你是灾后现场无人机视觉巡检评估员，正在执行第一轮高空俯视/斜俯视初评并生成结构化二次任务单。\n"
    "必须遵守以下规则：\n"
    "1. 只依据给定图像作答，严禁臆测图像里不存在的事实；每条观察尽量引用给定的图像编号。\n"
    "2. 先做语义识别：\n"
    "   - target_type：准确概括目标种类（例如“双尾翼固定翼飞机”、“单发固定翼飞机”、“大型机库建筑”、“机场控制塔台”等）；\n"
    "   - suspected_damaged_parts：列出疑似受损的具体部位名称（无明显受损则输出空列表 []）；\n"
    "   - visible_damage_description：用清晰专业的中文描述可见损伤的具体形态、位置与范围（如翼面上卷变形、表面放射裂纹、屋顶多边形穿孔、立柱混凝土剥落、钢筋外露等；若当前视角未见明显异常，明确写明“当前视角未见明显外观与结构损伤”）；\n"
    "   - initial_damage_level：取值为 none / suspected / minor / moderate / severe。\n"
    "3. 重点核查结构与外观异常：几何不对称、翼面/构件弯曲卷曲、撕裂或破口、屋顶破洞、边缘缺损、阴影与破损难以区分处，以及因高空正俯视视角遮挡而无法确认的侧面/内部结构。\n"
    "4. 明确区分“已看到的现象(observations)”与“单凭本图仍看不清的具体疑点(uncertainties)”。\n"
    "5. 无论初评是否看到损伤，都必须生成结构化的二次任务单(secondary_task)：\n"
    "   - task_type：屋顶有明显破洞需穿洞入室核查填 roof_entry；需抵近精查机翼/机身/顶面填 closeup_approach；需侧视精查立柱/墙体填 side_inspection；初评看似完好仅做近距完好复核填 intact_verification；\n"
    "   - focus_part：二次无人机重点瞄准的具体部位；\n"
    "   - desired_viewpoint：期望拍摄角度与距离（例如“低空斜俯视 30° 抵近至 6 m 内观察右侧主翼”或“正俯视对准屋顶破口并垂直下降入室拍摄内部桁架”）；\n"
    "   - task_instruction：发给第二批无人机的一句话清晰可执行复查指令；\n"
    "   - priority：已见明确破洞/结构破坏填 high，存在疑似异常填 medium，常规完好复核填 low。\n"
    "6. 若发现疑似或明确异常将 needs_recheck 设为 true，若初评看似完好设为 false；同时保持 recommended_action、focus_region、recheck_question 与 secondary_task 一致。\n"
    "7. 严格按指定 JSON Schema 输出简体中文结果。"
)

ROUND2_SYSTEM_PROMPT = (
    "你是灾后现场无人机视觉巡检复评员，正在对比第一轮初查图像与第二轮抵近/入室补拍图像，输出结构化语义定损报告。\n"
    "必须遵守以下规则：\n"
    "1. 必须围绕第一轮提出的复查/复核问题(original_question)逐项作答，同时检查第二轮近距图像是否暴露了第一轮高空视角漏检的新损伤，严禁编造。\n"
    "2. 语义定损字段要求：\n"
    "   - target_type：结合两轮视角确认的目标种类；\n"
    "   - confirmed_damaged_parts：最终确认存在损伤的具体部位列表（若确认完好无损，必须为空列表 []）；\n"
    "   - damage_description：综合两轮视角的完整语义化损伤描述，清晰区分“已证实的受损部位与具体破坏形态”及“已核实完好的结构区域”；\n"
    "   - damage_severity：综合严重度分级，只能从 intact（完好）/ minor（轻度）/ moderate（中度）/ severe（重度）/ uncertain（待定）中选择；\n"
    "   - recommended_followup：面向现场指挥员的一句话后续处置建议。\n"
    "3. new_evidence 中必须明确指出第二轮图像(引用具体图像编号)提供了哪些第一轮看不到的新视觉证据。\n"
    "4. verdict 只能从以下三项中选择：\n"
    "   - confirmed：第二轮图像明确证实目标存在结构损伤或异常（包括证实第一轮怀疑的损伤，或发现了第一轮漏检的新损伤；注意：只要第一轮或第二轮任意图像中存在明确的屋顶破洞、墙体缺口或结构变形，即使次轮聚焦的内部立柱或局部构件完好，整体目标仍属存在结构损伤，应判为 confirmed 并在 damage_description 中分别写清受损处与完好处）；\n"
    "   - ruled_out：结合两轮全部视角清晰排除了所有损伤疑点，确认目标整体完好无损（此时 damage_severity 应为 intact，confirmed_damaged_parts 应为 []）；\n"
    "   - still_uncertain：即使结合两轮图像仍因遮挡、距离、视角或画质无法明确判断（必须诚实保留不确定，不得强行下结论）。\n"
    "5. reasoning 要说明结论相比第一轮为何改变（例如从初评无异常转为确认损伤，或从存疑转为排除）、维持或仍存疑。\n"
    "6. 严格按指定 JSON Schema 输出简体中文结果。"
)


@dataclass(frozen=True)
class VLMImage:
    """Neutral image handle passed to VLM clients (no ground-truth labels)."""

    image_id: str
    view_type: str
    image_path: Path


@dataclass(frozen=True)
class VLMResponse:
    result: dict[str, Any]
    raw_response: str
    model_name: str
    latency_ms: float


class BaseVLMClient:
    """Abstract VLM interface consumed by SurveyCenter."""

    backend_name: str = "base"
    model_name: str = "unknown"

    def assess_round1(self, target_id: str, images: list[VLMImage]) -> VLMResponse:
        raise NotImplementedError

    def assess_round2(
        self,
        target_id: str,
        question: str,
        round1_images: list[VLMImage],
        round2_images: list[VLMImage],
    ) -> VLMResponse:
        raise NotImplementedError


def build_round1_user_prompt(target_id: str, images: list[VLMImage]) -> str:
    lines = [
        f"目标编号：{target_id}",
        "以下为第一轮采集图像（按图像传入顺序编号）：",
    ]
    for idx, img in enumerate(images, start=1):
        lines.append(f"- 图{idx}：图像编号={img.image_id}，拍摄视角={img.view_type}")
    lines.append("请基于以上图像完成第一轮损伤初评，并说明是否需要第二轮无人机抵近复查。")
    return "\n".join(lines)


def build_round2_user_prompt(
    target_id: str,
    question: str,
    round1_images: list[VLMImage],
    round2_images: list[VLMImage],
) -> str:
    lines = [
        f"目标编号：{target_id}",
        f"第一轮遗留的复查问题：{question}",
        "以下按顺序传入第一轮与第二轮图像：",
    ]
    idx = 1
    for img in round1_images:
        lines.append(f"- 图{idx}（第一轮）：图像编号={img.image_id}，拍摄视角={img.view_type}")
        idx += 1
    for img in round2_images:
        lines.append(f"- 图{idx}（第二轮复查）：图像编号={img.image_id}，拍摄视角={img.view_type}")
        idx += 1
    lines.append("请对照复查问题综合两轮图像给出最终复评结论，并在证据中写清引用的图像编号。")
    return "\n".join(lines)


def _validate_round1_result(data: dict[str, Any], image_ids: list[str]) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned["target_type"] = str(cleaned.get("target_type", "")).strip() or "未知目标"
    cleaned["suspected_damaged_parts"] = [
        str(x).strip() for x in cleaned.get("suspected_damaged_parts", []) if str(x).strip()
    ]
    cleaned["visible_damage_description"] = str(cleaned.get("visible_damage_description", "")).strip()
    dmg_lvl = str(cleaned.get("initial_damage_level", "")).strip()
    if dmg_lvl not in {"none", "suspected", "minor", "moderate", "severe"}:
        dmg_lvl = "suspected" if cleaned.get("needs_recheck") else "none"
    cleaned["initial_damage_level"] = dmg_lvl

    cleaned["observations"] = [str(x) for x in cleaned.get("observations", [])]
    cleaned["uncertainties"] = [str(x) for x in cleaned.get("uncertainties", [])]
    ev_ids = [str(x) for x in cleaned.get("evidence_image_ids", []) if str(x)]
    cleaned["evidence_image_ids"] = ev_ids or list(image_ids)
    cleaned["needs_recheck"] = bool(cleaned.get("needs_recheck", False))
    cleaned["recheck_question"] = str(cleaned.get("recheck_question", "")).strip()
    cleaned["focus_region"] = str(cleaned.get("focus_region", "")).strip() or (
        "、".join(cleaned["suspected_damaged_parts"]) if cleaned["suspected_damaged_parts"] else "目标主体"
    )
    action = str(cleaned.get("recommended_action", "none")).strip()
    if action not in {"closeup_approach", "roof_entry", "side_inspection", "none"}:
        action = "closeup_approach" if cleaned["needs_recheck"] else "none"
    cleaned["recommended_action"] = action
    cleaned["initial_summary"] = str(cleaned.get("initial_summary", "")).strip()
    if not cleaned["visible_damage_description"]:
        cleaned["visible_damage_description"] = (
            cleaned["initial_summary"] or "当前视角未见明显外观与结构损伤。"
        )
    if cleaned["needs_recheck"] and not cleaned["recheck_question"]:
        cleaned["recheck_question"] = "请抵近拍摄目标疑似异常部位，核实是否存在结构损伤。"

    raw_task = cleaned.get("secondary_task")
    task = dict(raw_task) if isinstance(raw_task, dict) else {}
    task_type = str(task.get("task_type", "")).strip()
    if task_type not in {"closeup_approach", "roof_entry", "side_inspection", "intact_verification"}:
        task_type = action if action != "none" else ("closeup_approach" if cleaned["needs_recheck"] else "intact_verification")
    focus_part = str(task.get("focus_part", "")).strip() or cleaned["focus_region"]
    desired_vp = str(task.get("desired_viewpoint", "")).strip() or (
        "正俯视对准屋顶破口并垂直下降入室观察内部结构"
        if task_type == "roof_entry"
        else ("侧视低空抵近至 6 m 内观察立柱与墙面" if task_type == "side_inspection" else "低空斜俯视抵近至 6 m 内精查重点部位")
    )
    task_instr = str(task.get("task_instruction", "")).strip() or cleaned["recheck_question"] or f"抵近复核{focus_part}状态。"
    priority = str(task.get("priority", "")).strip()
    if priority not in {"high", "medium", "low"}:
        priority = "high" if dmg_lvl in {"minor", "moderate", "severe"} else ("medium" if cleaned["needs_recheck"] else "low")
    cleaned["secondary_task"] = {
        "task_type": task_type,
        "focus_part": focus_part,
        "desired_viewpoint": desired_vp,
        "task_instruction": task_instr,
        "priority": priority,
    }
    return cleaned


def _validate_round2_result(data: dict[str, Any], question: str, all_image_ids: list[str]) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned["target_type"] = str(cleaned.get("target_type", "")).strip() or "未知目标"
    cleaned["original_question"] = str(cleaned.get("original_question", "")).strip() or question
    verdict = str(cleaned.get("verdict", "still_uncertain")).strip()
    if verdict not in {"confirmed", "ruled_out", "still_uncertain"}:
        verdict = "still_uncertain"
    cleaned["verdict"] = verdict
    severity = str(cleaned.get("damage_severity", "")).strip()
    if severity not in {"intact", "minor", "moderate", "severe", "uncertain"}:
        severity = "intact" if verdict == "ruled_out" else ("minor" if verdict == "confirmed" else "uncertain")
    if verdict == "ruled_out":
        severity = "intact"
    cleaned["damage_severity"] = severity
    parts = [str(x).strip() for x in cleaned.get("confirmed_damaged_parts", []) if str(x).strip()]
    cleaned["confirmed_damaged_parts"] = [] if verdict == "ruled_out" else parts
    ev_ids = [str(x) for x in cleaned.get("evidence_image_ids", []) if str(x)]
    cleaned["evidence_image_ids"] = ev_ids or list(all_image_ids)
    cleaned["new_evidence"] = [str(x) for x in cleaned.get("new_evidence", [])]
    cleaned["reasoning"] = str(cleaned.get("reasoning", "")).strip()
    cleaned["final_summary"] = str(cleaned.get("final_summary", "")).strip()
    cleaned["damage_description"] = str(cleaned.get("damage_description", "")).strip() or cleaned["final_summary"]
    cleaned["recommended_followup"] = str(cleaned.get("recommended_followup", "")).strip() or (
        "确认完好，无需后续处置。" if verdict == "ruled_out" else "列入受损目标清单，建议结合近景图开展工程评估。"
    )
    return cleaned


def _parse_json_content(raw_content: str) -> dict[str, Any]:
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class MockVLMClient(BaseVLMClient):
    """Deterministic offline VLM stub for skeleton testing and unit tests."""

    backend_name = "mock"

    def __init__(
        self,
        model_name: str = "mock-vlm-v1",
        round1_overrides: dict[str, dict[str, Any]] | None = None,
        round2_overrides: dict[str, dict[str, Any]] | None = None,
        fail_targets_stage1: set[str] | None = None,
        fail_targets_stage2: set[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.round1_overrides = dict(round1_overrides or {})
        self.round2_overrides = dict(round2_overrides or {})
        self.fail_targets_stage1 = set(fail_targets_stage1 or set())
        self.fail_targets_stage2 = set(fail_targets_stage2 or set())
        self.recorded_calls: list[dict[str, Any]] = []

    def assess_round1(self, target_id: str, images: list[VLMImage]) -> VLMResponse:
        t0 = time.perf_counter()
        prompt = build_round1_user_prompt(target_id, images)
        self.recorded_calls.append(
            {
                "stage": 1,
                "target_id": target_id,
                "prompt": prompt,
                "image_ids": [img.image_id for img in images],
                "image_paths": [str(img.image_path) for img in images],
            }
        )
        if target_id in self.fail_targets_stage1:
            raise RuntimeError(f"Simulated stage-1 VLM failure for {target_id}")

        img_ids = [img.image_id for img in images]
        default_res = {
            "target_type": "机场设施/载具目标",
            "suspected_damaged_parts": ["疑似异常局部区域"],
            "visible_damage_description": f"依据 {img_ids[0]} 可见目标局部存在阴影与边缘不规则异常，暂无法排除结构破损。",
            "initial_damage_level": "suspected",
            "observations": [f"依据首轮图像 {img_ids[0]} 可见目标 {target_id} 整体轮廓及局部阴影/边缘异常。"],
            "evidence_image_ids": img_ids,
            "uncertainties": [f"仅凭 {img_ids[0]} 难以区分局部异常是视角畸变/阴影还是实质结构破损。"],
            "needs_recheck": True,
            "recheck_question": f"请抵近拍摄目标 {target_id}，核实 {img_ids[0]} 中疑似异常区域是否存在实质结构破损。",
            "focus_region": "疑似异常区域",
            "recommended_action": "closeup_approach",
            "secondary_task": {
                "task_type": "closeup_approach",
                "focus_part": "疑似异常局部区域",
                "desired_viewpoint": "低空斜俯视抵近至 6 m 内拍摄特写",
                "task_instruction": f"抵近目标 {target_id} 疑似异常区域，拍摄近景图像核实结构是否受损。",
                "priority": "medium",
            },
            "initial_summary": f"目标 {target_id} 首轮存在待核实疑点，建议派发第二轮近距复查。",
        }
        override = self.round1_overrides.get(target_id)
        if override:
            default_res.update(override)
        result = _validate_round1_result(default_res, img_ids)
        raw = json.dumps(result, ensure_ascii=False)
        return VLMResponse(
            result=result,
            raw_response=raw,
            model_name=self.model_name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def assess_round2(
        self,
        target_id: str,
        question: str,
        round1_images: list[VLMImage],
        round2_images: list[VLMImage],
    ) -> VLMResponse:
        t0 = time.perf_counter()
        prompt = build_round2_user_prompt(target_id, question, round1_images, round2_images)
        self.recorded_calls.append(
            {
                "stage": 2,
                "target_id": target_id,
                "question": question,
                "prompt": prompt,
                "round1_image_ids": [img.image_id for img in round1_images],
                "round2_image_ids": [img.image_id for img in round2_images],
                "image_paths": [str(img.image_path) for img in [*round1_images, *round2_images]],
            }
        )
        if target_id in self.fail_targets_stage2:
            raise RuntimeError(f"Simulated stage-2 VLM failure for {target_id}")

        r1_ids = [img.image_id for img in round1_images]
        r2_ids = [img.image_id for img in round2_images]
        all_ids = r1_ids + r2_ids
        default_res = {
            "target_type": "机场设施/载具目标",
            "original_question": question,
            "verdict": "confirmed",
            "damage_severity": "minor",
            "confirmed_damaged_parts": ["主承力局部构件"],
            "damage_description": f"综合 {r1_ids[0]} 与 {r2_ids[0]}，确认目标局部存在明显结构异常，其余可见区域基本完整。",
            "evidence_image_ids": all_ids,
            "new_evidence": [f"第二轮近距图像 {r2_ids[0]} 清晰展示了目标 {target_id} 的近景结构细节，补足了 {r1_ids[0]} 的视角盲区。"],
            "reasoning": f"结合 {r1_ids[0]} 与 {r2_ids[0]}，原复查问题已获得直接视觉证据支持。",
            "final_summary": f"目标 {target_id} 复评完成：已依据 {r2_ids[0]} 确认原复查问题。",
            "recommended_followup": "列入受损目标台账，安排工程人员进一步核查。",
        }
        override = self.round2_overrides.get(target_id)
        if override:
            default_res.update(override)
        result = _validate_round2_result(default_res, question, all_ids)
        raw = json.dumps(result, ensure_ascii=False)
        return VLMResponse(
            result=result,
            raw_response=raw,
            model_name=self.model_name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class OllamaVLMClient(BaseVLMClient):
    """Ollama /api/chat vision client (direct HTTP or stateless SSH proxy)."""

    backend_name = "ollama"

    def __init__(
        self,
        model_name: str = "qwen3.5:9b",
        base_url: str = "http://127.0.0.1:11434",
        ssh_host: str | None = None,
        ssh_key: str | Path | None = None,
        think: bool = False,
        temperature: float = 0.1,
        timeout_s: float = 90.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.ssh_host = ssh_host
        self.ssh_key = str(ssh_key) if ssh_key is not None else None
        self.think = bool(think)
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)

    def _encode_images(self, images: list[VLMImage]) -> list[str]:
        encoded: list[str] = []
        for img in images:
            encoded.append(base64.b64encode(Path(img.image_path).read_bytes()).decode("ascii"))
        return encoded

    def _call_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[VLMImage],
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], str, float]:
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": self.think,
            "format": schema,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": self._encode_images(images),
                },
            ],
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        t0 = time.perf_counter()

        if self.ssh_host:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
            if self.ssh_key:
                cmd.extend(["-i", self.ssh_key, "-o", "IdentitiesOnly=yes"])
            remote_cmd = (
                f"curl -sS --max-time {int(self.timeout_s)} "
                f"{self.base_url}/api/chat -H 'Content-Type: application/json' -d @-"
            )
            cmd.extend([self.ssh_host, remote_cmd])
            proc = None
            for attempt in range(3):
                proc = subprocess.run(
                    cmd,
                    input=payload_bytes,
                    capture_output=True,
                    timeout=self.timeout_s + 15.0,
                )
                if proc.returncode == 0:
                    break
                if proc.returncode == 255 and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            assert proc is not None
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Ollama SSH call failed (rc={proc.returncode}): {stderr}")
            raw_http = proc.stdout.decode("utf-8")
        else:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw_http = resp.read().decode("utf-8")
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Ollama HTTP call failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        envelope = json.loads(raw_http)
        if "error" in envelope:
            raise RuntimeError(f"Ollama returned error: {envelope['error']}")
        content = envelope.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"Ollama returned empty message content: {raw_http[:500]}")
        parsed = _parse_json_content(content)
        return parsed, raw_http, latency_ms

    def assess_round1(self, target_id: str, images: list[VLMImage]) -> VLMResponse:
        user_prompt = build_round1_user_prompt(target_id, images)
        parsed, raw_http, latency_ms = self._call_chat(
            ROUND1_SYSTEM_PROMPT,
            user_prompt,
            images,
            ROUND1_SCHEMA,
        )
        result = _validate_round1_result(parsed, [img.image_id for img in images])
        return VLMResponse(
            result=result,
            raw_response=raw_http,
            model_name=self.model_name,
            latency_ms=latency_ms,
        )

    def assess_round2(
        self,
        target_id: str,
        question: str,
        round1_images: list[VLMImage],
        round2_images: list[VLMImage],
    ) -> VLMResponse:
        user_prompt = build_round2_user_prompt(target_id, question, round1_images, round2_images)
        all_images = [*round1_images, *round2_images]
        parsed, raw_http, latency_ms = self._call_chat(
            ROUND2_SYSTEM_PROMPT,
            user_prompt,
            all_images,
            ROUND2_SCHEMA,
        )
        result = _validate_round2_result(parsed, question, [img.image_id for img in all_images])
        return VLMResponse(
            result=result,
            raw_response=raw_http,
            model_name=self.model_name,
            latency_ms=latency_ms,
        )
