"""Coastal Harbor Task 1: Targeted Part Re-Inspection (3-condition benchmark).

Compares three conditions on the exact same frozen Round-1 far imagery, with identical
start poses, time budget, and capture budget for the two flight conditions:
1. `far_only`: Round-1 far image assessment only (0 s flight, 1 capture).
2. `preset_recheck`: Direct visual-servo approach along the initial viewing ray without
   active viewpoint selection (replicates the airport baseline behavior).
3. `active_viewpoint`: VLM inspects the Round-1 image against the requested part, decides
   whether to approach directly or actively orbit/reposition (using ONLY depth-estimated
   standoff distance, never reading target ground-truth coordinates), confirms visibility
   on relay imagery, then runs closed-loop DINOv2 RGB-D visual servoing to the stop range.

Ground-truth target/part coordinates (`evaluation_gt`) are strictly isolated for:
- Start-viewpoint staging (`verify-starts` and episode reset)
- Post-hoc independent geometric scoring (`evaluate_capture_geometry`)
and are NEVER passed to VLM prompts, orbit planning, or closed-loop visual servoing.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import commentjson
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.paths import CONFIGS, RUNS, SIMULATION_DIR, project_path
from common.runtime import configure_client_log
from common.sensors import decode_image
from coordination.visual_servo import SharedDinoMatcher, run_dino_visual_servo
from coordination.vlm import OllamaVLMClient, VLMImage, VLMResponse
from harbor.recorder import HarborVideoRecorder


DEFAULT_CONFIG = CONFIGS / "missions" / "coastal_harbor_part_recheck.json"
DEFAULT_SSH_HOST = "ryh2025@192.168.31.237"
DEFAULT_SSH_KEY = "/home/hui/.ssh/id_ed25519_501"


# ---------------------------------------------------------------------------
# VLM schemas & prompts (strictly neutral — no ground-truth damage/coord hints)
# ---------------------------------------------------------------------------

VIEWPOINT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_type": {"type": "string"},
        "key_landmarks": {"type": "string"},
        "observed_side_description": {"type": "string"},
        "part_visibility": {
            "type": "string",
            "enum": ["visible_clear", "visible_far", "occluded_or_backside"],
        },
        "recommended_maneuver": {
            "type": "string",
            "enum": ["direct_approach", "reposition_opposite", "reposition_left_90", "reposition_right_90"],
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "target_type",
        "key_landmarks",
        "observed_side_description",
        "part_visibility",
        "recommended_maneuver",
        "reasoning",
    ],
}

FINAL_ASSESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_type": {"type": "string"},
        "part_clearly_visible": {"type": "boolean"},
        "verdict": {
            "type": "string",
            "enum": ["confirmed", "ruled_out", "still_uncertain"],
        },
        "damage_severity": {
            "type": "string",
            "enum": ["intact", "minor", "moderate", "severe", "uncertain"],
        },
        "observed_evidence": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "target_type",
        "part_clearly_visible",
        "verdict",
        "damage_severity",
        "observed_evidence",
        "reasoning",
    ],
}

VIEWPOINT_SYSTEM_PROMPT = (
    "你是沿海渔港灾后无人机巡检的视角规划员。你会看到目标的远景整图和画幅中心自动放大裁剪图，\n"
    "任务要求复核目标的【指定部位】是否存在外观或结构损伤。\n"
    "请严格按步骤依据当前画面判断，严禁猜测被挡住的一面：\n"
    "1. key_landmarks（先找客观视觉地标）：\n"
    "   - 对于商铺/建筑：明确指出画面中是否看到了门面大门、大面积橱窗、招牌或遮阳雨棚。\n"
    "   - 对于渔船：先在中心放大图里找到挂在平船尾外侧的深蓝色/黑色舷外马达是在船体【左端】还是【右端】，尖形船头指向画面【左侧】还是【右侧】。\n"
    "2. observed_side_description（根据地标推导出当前面对哪一面）：\n"
    "   - 商铺：带门面大门、橱窗、招牌或雨棚的是正立面；只有普通墙面和普通窗户的是背面。\n"
    "   - 渔船几何铁律：船头朝左 + 尾马达在右 => 镜头正对着【左舷】；船头朝右 + 尾马达在左 => 镜头正对着【右舷】（此时左舷外船壳被船体和驾驶室遮挡）。\n"
    "3. part_visibility：\n"
    "   - visible_clear：指定部位正对镜头、无遮挡且已经非常近、细节清晰；\n"
    "   - visible_far：指定部位位于当前面对的这一侧且没有被主体挡住，但距离还较远，需要直接抵近拍清；\n"
    "   - occluded_or_backside：当前位于目标背面或另一侧（例如商铺背面，或任务要求左舷但当前镜头面对右舷），必须绕飞换位才能看到。\n"
    "4. recommended_maneuver：\n"
    "   - 若 part_visibility 为 visible_clear 或 visible_far，选 direct_approach；\n"
    "   - 若当前在建筑背面或船只对侧（180° 相反一侧），选 reposition_opposite；\n"
    "   - 若需向左/右绕飞 90° 才能转到指定侧面，分别选 reposition_left_90 或 reposition_right_90。\n"
    "5. 严格按指定 JSON Schema 输出简体中文。"
)

FINAL_ASSESS_SYSTEM_PROMPT = (
    "你是沿海渔港灾后无人机视觉定损员。你将收到围绕同一目标拍摄的 1 至 3 张图像，\n"
    "以及本次任务需要重点核查的【指定部位】说明。\n"
    "必须遵守以下规则：\n"
    "1. 只依据给定图像作答，严禁臆测画面里看不清或被遮挡的部位。\n"
    "2. 首先判断 part_clearly_visible：全部给定图像中，是否至少有一张从正面/斜正面近距离清晰拍到了指定部位的细节。\n"
    "   - 如果所有图像都距离太远看不清细节，或者所有图像都在目标背面/对侧、被主体挡住了指定部位，part_clearly_visible 必须为 false，且 verdict 必须为 still_uncertain，不得凭背面或远景强行下结论！\n"
    "3. 当 part_clearly_visible 为 true 时，仔细核查指定部位本身是否存在结构/外观损伤：\n"
    "   - 商铺正立面雨棚：检查整排雨棚是否平直连续、支架完整；若一端脱落、倾斜下垂或撕裂，判 confirmed；若整排雨棚与门窗结构平直完好无变形脱落，判 ruled_out。\n"
    "   - 渔船船舷与栏杆：检查指定船舷外壳与栏杆是否平滑顺直；若船舷壳体出现明显局部凹陷、变形或栏杆弯折，判 confirmed；若船舷壳体与栏杆平滑连续无凹陷弯折，判 ruled_out（注意：船体因搁浅在泥滩上的整体倾斜属于搁浅姿态，不等同于船壳凹陷变形）。\n"
    "4. verdict 只能取 confirmed（确认指定部位存在外观/结构损伤）、ruled_out（近景清晰核实指定部位完好无损）、still_uncertain（未拍清或被遮挡，无法判断）。\n"
    "5. 严格按指定 JSON Schema 输出简体中文。"
)


def _make_center_crop(image_path: Path) -> Path:
    """Create a ground-truth-free center crop (middle 44%x53%) so VLMs can inspect fine details."""
    crop_path = image_path.with_name(image_path.stem + "_centercrop.jpg")
    if crop_path.exists():
        return crop_path
    img = cv2.imread(str(image_path))
    if img is None:
        return image_path
    h, w = img.shape[:2]
    crop = img[int(h * 0.25) : int(h * 0.78), int(w * 0.28) : int(w * 0.72)]
    cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return crop_path


class TemplateViewpointPlanner:
    """Onboard standalone viewpoint planner using frozen DINOv2 spatial correlation (zero VLM/server).

    Compares a given part reference template kernel against the live center-cropped image in both
    original and horizontally-flipped orientations:
    - Low absolute spatial correlation (`orig < min_spatial_score`) indicates a different face
      (e.g. building backside without awning/signage).
    - Vanishing orientation advantage (`orig - flip < min_flip_margin`) indicates a mirrored
      opposite flank (e.g. starboard vs port side of a symmetric vessel).
    Note: Default thresholds (0.58 / 0.012) were calibrated on the 4-target harbor pilot set and
    should be verified before deploying to unseen target categories.
    """

    def __init__(
        self,
        matcher: SharedDinoMatcher,
        *,
        min_spatial_score: float = 0.57,
        min_flip_margin: float = 0.0,
        scales: tuple[float, ...] = (0.35, 0.45, 0.55, 0.70, 0.85),
    ) -> None:
        self.matcher = matcher
        self.min_spatial_score = float(min_spatial_score)
        self.min_flip_margin = float(min_flip_margin)
        self.scales = scales
        self._kernel_cache: dict[tuple[str, tuple[float, ...]], tuple[Any, Any]] = {}

    def _encode_grid(self, rgb: np.ndarray) -> Any:
        import torch.nn.functional as F

        z = self.matcher._encode(np.ascontiguousarray(rgb)[None], bs=1)[0].to(self.matcher.device)
        z = F.normalize((z - self.matcher.zm) / self.matcher.zs, dim=-1)
        g = self.matcher.grid
        return z.reshape(g, g, -1).permute(2, 0, 1).unsqueeze(0)

    def _get_kernels(self, template_path: Path, roi: list[float]) -> tuple[Any, Any]:
        import torch

        key = (str(template_path), tuple(float(v) for v in roi))
        if key not in self._kernel_cache:
            bgr = cv2.imread(str(template_path))
            if bgr is None:
                raise FileNotFoundError(f"Standalone reference template not found: {template_path}")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            zt = self._encode_grid(rgb)
            g = self.matcher.grid
            x0, y0, x1, y1 = [int(round(float(v) * g)) for v in roi]
            x0, x1 = max(0, min(g - 2, x0)), max(x0 + 2, min(g, x1))
            y0, y1 = max(0, min(g - 2, y0)), max(y0 + 2, min(g, y1))
            k_orig = zt[:, :, y0:y1, x0:x1]
            k_flip = torch.flip(k_orig, dims=[3])
            self._kernel_cache[key] = (k_orig, k_flip)
        return self._kernel_cache[key]

    def _max_corr(self, kernel: Any, zs: Any) -> float:
        import torch.nn.functional as F

        best = -1.0
        for scale in self.scales:
            kh = max(2, int(round(kernel.shape[2] * scale)))
            kw = max(2, int(round(kernel.shape[3] * scale)))
            if kh > zs.shape[2] or kw > zs.shape[3]:
                continue
            k_s = F.normalize(F.interpolate(kernel, size=(kh, kw), mode="bilinear", align_corners=False), dim=1)
            sc = float((F.conv2d(zs, k_s) / (kh * kw)).max().item())
            if sc > best:
                best = sc
        return best

    def plan_viewpoint(
        self,
        *,
        template_path: Path,
        template_roi: list[float],
        image_path: Path,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        k_orig, k_flip = self._get_kernels(template_path, template_roi)
        crop_path = _make_center_crop(image_path)
        bgr = cv2.imread(str(crop_path))
        if bgr is None:
            raise FileNotFoundError(f"Query image not found: {crop_path}")
        zs = self._encode_grid(np.ascontiguousarray(bgr[:, :, ::-1]))
        sc_orig = self._max_corr(k_orig, zs)
        sc_flip = self._max_corr(k_flip, zs)
        margin = sc_orig - sc_flip
        occluded = bool((sc_orig < self.min_spatial_score) or (margin < self.min_flip_margin))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "latency_ms": round(latency_ms, 1),
            "result": {
                "planner": "template_spatial_dino",
                "spatial_score_orig": round(sc_orig, 4),
                "spatial_score_flip": round(sc_flip, 4),
                "orientation_margin": round(margin, 4),
                "part_visibility": "occluded_or_backside" if occluded else "visible_far",
                "recommended_maneuver": "reposition_opposite" if occluded else "direct_approach",
                "reasoning": (
                    f"orig={sc_orig:.4f}, flip={sc_flip:.4f}, margin={margin:+.4f} "
                    f"(thresholds: orig>={self.min_spatial_score}, margin>={self.min_flip_margin})"
                ),
            },
        }


class HarborPartVLM:
    """Thin VLM wrapper for Task 1 viewpoint planning and final damage assessment."""

    def __init__(
        self,
        *,
        backend: str = "ollama",
        model_name: str = "qwen3.5:9b",
        ollama_url: str = "http://127.0.0.1:11434",
        ssh_host: str | None = DEFAULT_SSH_HOST,
        ssh_key: str | None = DEFAULT_SSH_KEY,
        think: bool = False,
        temperature: float = 0.1,
    ) -> None:
        self.backend = backend
        self.model_name = "none" if backend == "none" else ("mock" if backend == "mock" else model_name)
        self._client: OllamaVLMClient | None = None
        if backend == "ollama":
            self._client = OllamaVLMClient(
                model_name=model_name,
                base_url=ollama_url,
                ssh_host=ssh_host or None,
                ssh_key=ssh_key if ssh_host else None,
                think=think,
                temperature=temperature,
                timeout_s=110.0,
            )

    def plan_viewpoint(
        self,
        *,
        neutral_id: str,
        target_class_hint: str,
        focus_part_request: str,
        image: VLMImage,
        attempt_index: int = 1,
    ) -> VLMResponse:
        user_prompt = (
            f"目标编号：{neutral_id}\n"
            f"目标类别提示：{target_class_hint}\n"
            f"指定核查部位：{focus_part_request}\n"
            f"当前为第 {attempt_index} 次视角观察（图像编号={image.image_id}，视角={image.view_type}）。\n"
            "请判断指定核查部位在当前画面中是否可见、是否被目标自身遮挡，并决定是直接抵近还是绕飞换位。"
        )
        if self.backend == "none":
            t0 = time.perf_counter()
            res = {
                "target_type": target_class_hint,
                "key_landmarks": "none",
                "observed_side_description": "Standalone mode without VLM planner.",
                "part_visibility": "visible_far",
                "recommended_maneuver": "direct_approach",
                "reasoning": "Skipped remote VLM planning (--vlm none).",
            }
            return VLMResponse(res, json.dumps(res, ensure_ascii=False), "none", (time.perf_counter() - t0) * 1000.0)
        if self._client is None:
            t0 = time.perf_counter()
            res = {
                "target_type": target_class_hint,
                "key_landmarks": "Mock landmarks",
                "observed_side_description": f"Mock 观测 {image.image_id}",
                "part_visibility": "occluded_or_backside" if "occluded" in image.view_type else "visible_far",
                "recommended_maneuver": "reposition_opposite" if "occluded" in image.view_type else "direct_approach",
                "reasoning": "Mock VLM deterministic decision.",
            }
            return VLMResponse(res, json.dumps(res, ensure_ascii=False), "mock", (time.perf_counter() - t0) * 1000.0)
        crop_img = VLMImage(
            image_id=f"{image.image_id}-CROP",
            view_type="画幅中心自动放大图",
            image_path=_make_center_crop(image.image_path),
        )
        parsed, raw_http, latency_ms = self._client._call_chat(
            VIEWPOINT_SYSTEM_PROMPT,
            user_prompt + "\n（已附上整图与画幅中心自动放大图两张图，请重点看中心放大图分辨地标。）",
            [image, crop_img],
            VIEWPOINT_PLAN_SCHEMA,
        )
        vis = str(parsed.get("part_visibility", "visible_far")).strip()
        if vis not in {"visible_clear", "visible_far", "occluded_or_backside"}:
            vis = "visible_far"
        maneuver = str(parsed.get("recommended_maneuver", "direct_approach")).strip()
        if maneuver not in {"direct_approach", "reposition_opposite", "reposition_left_90", "reposition_right_90"}:
            maneuver = "reposition_opposite" if vis == "occluded_or_backside" else "direct_approach"
        if vis in {"visible_clear", "visible_far"} and maneuver != "direct_approach":
            maneuver = "direct_approach"
        cleaned = {
            "target_type": str(parsed.get("target_type", target_class_hint)).strip() or target_class_hint,
            "key_landmarks": str(parsed.get("key_landmarks", "")).strip(),
            "observed_side_description": str(parsed.get("observed_side_description", "")).strip(),
            "part_visibility": vis,
            "recommended_maneuver": maneuver,
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }
        return VLMResponse(cleaned, raw_http, self.model_name, latency_ms)

    def assess_damage(
        self,
        *,
        neutral_id: str,
        target_class_hint: str,
        focus_part_request: str,
        images: list[VLMImage],
        condition_label: str,
    ) -> VLMResponse:
        lines = [
            f"目标编号：{neutral_id}",
            f"目标类别提示：{target_class_hint}",
            f"指定核查部位：{focus_part_request}",
            f"共提供 {len(images)} 张图像（按采集先后顺序）：",
        ]
        for idx, img in enumerate(images, start=1):
            lines.append(f"- 图{idx}：图像编号={img.image_id}，阶段={img.view_type}")
        lines.append("请先判断是否已清晰拍到指定核查部位，再给出最终定损结论。")
        user_prompt = "\n".join(lines)
        if self.backend == "none":
            t0 = time.perf_counter()
            res = {
                "target_type": target_class_hint,
                "part_clearly_visible": len(images) > 1,
                "verdict": "skipped_standalone",
                "damage_severity": "uncertain",
                "observed_evidence": "Standalone onboard execution mode (no remote VLM assessment).",
                "reasoning": "Skipped remote VLM assessment (--vlm none).",
            }
            return VLMResponse(res, json.dumps(res, ensure_ascii=False), "none", (time.perf_counter() - t0) * 1000.0)
        if self._client is None:
            t0 = time.perf_counter()
            res = {
                "target_type": target_class_hint,
                "part_clearly_visible": len(images) > 1,
                "verdict": "still_uncertain" if len(images) == 1 else "confirmed",
                "damage_severity": "uncertain" if len(images) == 1 else "moderate",
                "observed_evidence": f"Mock assessment for {condition_label}",
                "reasoning": "Mock VLM deterministic assessment.",
            }
            return VLMResponse(res, json.dumps(res, ensure_ascii=False), "mock", (time.perf_counter() - t0) * 1000.0)
        # Attach a ground-truth-free center crop of the final (closest) image fairly for all conditions
        last_img = images[-1]
        crop_img = VLMImage(
            image_id=f"{last_img.image_id}-CROP",
            view_type="末帧画幅中心自动放大图",
            image_path=_make_center_crop(last_img.image_path),
        )
        send_imgs = [*images, crop_img]
        user_prompt += f"\n- 图{len(send_imgs)}：图像编号={crop_img.image_id}，阶段={crop_img.view_type}"
        parsed, raw_http, latency_ms = self._client._call_chat(
            FINAL_ASSESS_SYSTEM_PROMPT, user_prompt, send_imgs, FINAL_ASSESS_SCHEMA
        )
        clearly_visible = bool(parsed.get("part_clearly_visible", False))
        verdict = str(parsed.get("verdict", "still_uncertain")).strip()
        if verdict not in {"confirmed", "ruled_out", "still_uncertain"}:
            verdict = "still_uncertain"
        if not clearly_visible:
            verdict = "still_uncertain"
        sev = str(parsed.get("damage_severity", "uncertain")).strip()
        if sev not in {"intact", "minor", "moderate", "severe", "uncertain"}:
            sev = "uncertain"
        if verdict == "ruled_out":
            sev = "intact"
        elif verdict == "still_uncertain":
            sev = "uncertain"
        cleaned = {
            "target_type": str(parsed.get("target_type", target_class_hint)).strip() or target_class_hint,
            "part_clearly_visible": clearly_visible,
            "verdict": verdict,
            "damage_severity": sev,
            "observed_evidence": str(parsed.get("observed_evidence", "")).strip(),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }
        return VLMResponse(cleaned, raw_http, self.model_name, latency_ms)


# ---------------------------------------------------------------------------
# Pure geometry helpers (independent scoring + ground-truth-free orbit planner)
# ---------------------------------------------------------------------------

def quat_wxyz_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=float)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def camera_rot_from_yaw_pitch(yaw_rad: float, pitch_deg: float) -> np.ndarray:
    """Build world-NED-from-camera-SIM rotation matrix for level drone at yaw + camera pitch."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    p = math.radians(pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    return rz @ ry


def ue_to_ned(pt_ue: list[float] | np.ndarray) -> np.ndarray:
    return np.array([float(pt_ue[0]), float(pt_ue[1]), -float(pt_ue[2])], dtype=float)


def evaluate_capture_geometry(
    *,
    cam_pos_ned: list[float] | np.ndarray,
    cam_rot: np.ndarray,
    depth_arr: np.ndarray | None,
    evaluation_gt: dict[str, Any],
    scoring_cfg: dict[str, Any],
    rgb_wh: tuple[int, int] = (1920, 1080),
    fov_deg: float = 90.0,
) -> dict[str, Any]:
    """Independent post-hoc geometric evaluation of whether a capture clearly sees the designated part.

    Checks four objective criteria:
    1. In-frame projection (with border margin)
    2. Front-facing view angle against part outward normal (< max_view_angle_deg)
    3. Depth consistency at the projected pixel (no foreground occlusion > tol_m)
    4. Resolution / standoff clarity (dist <= clear_max_dist_m and projected span >= min_span_px)
    """
    w, h = int(rgb_wh[0]), int(rgb_wh[1])
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx0, cy0 = w / 2.0, h / 2.0

    cam_p = np.asarray(cam_pos_ned, dtype=float)
    part_p = ue_to_ned(evaluation_gt["part_center_ue_m"])
    n_ue = np.asarray(evaluation_gt["part_normal_ue"], dtype=float)
    part_n_ned = np.array([n_ue[0], n_ue[1], -n_ue[2]], dtype=float)
    part_n_ned /= max(float(np.linalg.norm(part_n_ned)), 1e-9)
    part_radius_m = float(evaluation_gt["part_radius_m"])

    v_ned = part_p - cam_p
    dist_3d = float(np.linalg.norm(v_ned))
    v_cam = cam_rot.T @ v_ned  # SIM camera frame: +X forward, +Y right, +Z down
    planar_depth = float(v_cam[0])

    # Angle between vector from part toward camera and outward part normal
    part_to_cam = (-v_ned) / max(dist_3d, 1e-9)
    cos_ang = float(np.clip(np.dot(part_to_cam, part_n_ned), -1.0, 1.0))
    view_angle_deg = math.degrees(math.acos(cos_ang))

    margin = float(scoring_cfg.get("part_in_frame_margin", 0.04))
    max_ang = float(scoring_cfg.get("part_max_view_angle_deg", 75.0))
    occ_tol = float(scoring_cfg.get("part_depth_occlusion_tol_m", 2.5))
    clear_max_d = float(scoring_cfg.get("part_clear_max_dist_m", 15.0))
    clear_min_px = float(scoring_cfg.get("part_clear_min_span_px", 100.0))

    in_front = planar_depth > 0.25
    if in_front:
        u = fx * (float(v_cam[1]) / planar_depth) + cx0
        v = fy * (float(v_cam[2]) / planar_depth) + cy0
        u_norm = u / w
        v_norm = v / h
        in_frame = (margin <= u_norm <= 1.0 - margin) and (margin <= v_norm <= 1.0 - margin)
        span_px = 2.0 * part_radius_m * fx / planar_depth
    else:
        u = v = u_norm = v_norm = None
        in_frame = False
        span_px = 0.0

    measured_part_depth = None
    depth_occlusion_gap = None
    depth_occluded = False
    center_depth_m = None
    if depth_arr is not None:
        d = np.asarray(depth_arr, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        dh, dw = d.shape[:2]
        c_patch = d[dh // 2 - 12 : dh // 2 + 12, dw // 2 - 12 : dw // 2 + 12]
        c_valid = c_patch[np.isfinite(c_patch) & (c_patch > 0.2) & (c_patch < 500.0)]
        if c_valid.size:
            center_depth_m = float(np.median(c_valid))
        if in_frame and u_norm is not None and v_norm is not None:
            du = int(np.clip(round(u_norm * dw), 2, dw - 3))
            dv = int(np.clip(round(v_norm * dh), 2, dh - 3))
            patch = d[dv - 2 : dv + 3, du - 2 : du + 3]
            valid = patch[np.isfinite(patch) & (patch > 0.2) & (patch < 500.0)]
            if valid.size:
                measured_part_depth = float(np.median(valid))
                depth_occlusion_gap = dist_3d - measured_part_depth
                depth_occluded = bool(depth_occlusion_gap > occ_tol)

    front_facing = view_angle_deg <= max_ang
    part_visible = bool(in_frame and front_facing and (not depth_occluded))
    part_clear = bool(part_visible and (dist_3d <= clear_max_d) and (span_px >= clear_min_px))

    return {
        "in_front": in_front,
        "in_frame": in_frame,
        "front_facing": front_facing,
        "depth_occluded": depth_occluded,
        "part_visible": part_visible,
        "part_clear": part_clear,
        "dist_to_part_m": round(dist_3d, 3),
        "planar_depth_to_part_m": round(planar_depth, 3),
        "measured_part_depth_m": round(measured_part_depth, 3) if measured_part_depth is not None else None,
        "depth_occlusion_gap_m": round(depth_occlusion_gap, 3) if depth_occlusion_gap is not None else None,
        "center_depth_m": round(center_depth_m, 3) if center_depth_m is not None else None,
        "view_angle_deg": round(view_angle_deg, 2),
        "part_pixel_uv": [round(float(u), 1), round(float(v), 1)] if u is not None and v is not None else None,
        "part_span_px": round(float(span_px), 1),
    }


def estimate_target_center_from_depth(
    *,
    drone_pos_ned: list[float] | np.ndarray,
    drone_yaw_rad: float,
    camera_pitch_deg: float,
    depth_arr: np.ndarray,
    min_radius_m: float = 15.0,
    max_radius_m: float = 28.0,
    depth_center_offset_m: float = 0.0,
    roi: list[float] | tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate target horizontal center purely from drone pose + foreground depth patch (no GT!)."""
    d = np.asarray(depth_arr, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    dh, dw = d.shape[:2]
    if roi is not None and len(roi) == 4:
        x0 = max(0, min(dw - 2, int(round(float(roi[0]) * dw))))
        y0 = max(0, min(dh - 2, int(round(float(roi[1]) * dh))))
        x1 = max(x0 + 2, min(dw, int(round(float(roi[2]) * dw))))
        y1 = max(y0 + 2, min(dh, int(round(float(roi[3]) * dh))))
        patch = d[y0:y1, x0:x1]
    else:
        # Sample central 24% patch where the target sits in Round-1 framing
        patch = d[int(dh * 0.38) : int(dh * 0.62), int(dw * 0.38) : int(dw * 0.62)]
    valid = patch[np.isfinite(patch) & (patch > 2.0) & (patch < 120.0)]
    if valid.size >= 10:
        # Use 25th percentile to lock onto foreground structure rather than distant ground behind it
        slant_d = float(np.percentile(valid, 25))
    else:
        slant_d = 22.0
    horiz_r = slant_d * math.cos(math.radians(abs(camera_pitch_deg))) + float(depth_center_offset_m)
    horiz_r = float(np.clip(horiz_r, min_radius_m, max_radius_m))
    pos = np.asarray(drone_pos_ned, dtype=float)
    center_ne = np.array(
        [pos[0] + horiz_r * math.cos(drone_yaw_rad), pos[1] + horiz_r * math.sin(drone_yaw_rad)],
        dtype=float,
    )
    return center_ne, horiz_r


def plan_orbit_waypoints(
    *,
    drone_pos_ned: list[float] | np.ndarray,
    center_ne: np.ndarray,
    radius_m: float,
    delta_angle_deg: float,
    orbit_alt_m: float,
    final_alt_m: float,
    num_arc_steps: int = 3,
) -> list[dict[str, Any]]:
    """Generate smooth collision-safe arc waypoints around an estimated target center."""
    pos = np.asarray(drone_pos_ned, dtype=float)
    start_angle = math.atan2(pos[1] - float(center_ne[1]), pos[0] - float(center_ne[0]))
    orbit_z = -float(orbit_alt_m)
    final_z = -float(final_alt_m)
    waypoints: list[dict[str, Any]] = []
    # Step 1: climb to safe orbit altitude if currently lower
    if pos[2] > orbit_z + 0.5:
        yaw_in = math.atan2(float(center_ne[1]) - pos[1], float(center_ne[0]) - pos[0])
        waypoints.append({"position_ned": [float(pos[0]), float(pos[1]), orbit_z], "yaw_rad": yaw_in})
    # Step 2: arc around center_ne
    for i in range(1, num_arc_steps + 1):
        ang = start_angle + math.radians(delta_angle_deg) * (i / num_arc_steps)
        wn = float(center_ne[0]) + radius_m * math.cos(ang)
        we = float(center_ne[1]) + radius_m * math.sin(ang)
        wz = final_z if i == num_arc_steps else orbit_z
        yaw_in = math.atan2(float(center_ne[1]) - we, float(center_ne[0]) - wn)
        waypoints.append({"position_ned": [wn, we, wz], "yaw_rad": yaw_in})
    return waypoints


def score_verdict_correct(verdict: str, damage_gt: str) -> bool:
    if damage_gt == "damaged":
        return verdict == "confirmed"
    if damage_gt == "intact":
        return verdict == "ruled_out"
    return False


# ---------------------------------------------------------------------------
# ProjectAirSim execution helpers
# ---------------------------------------------------------------------------

def _load_harbor_runtime_scene(scene_path: Path, enable_depth_stream: bool = True) -> dict[str, Any]:
    scene = commentjson.loads(scene_path.read_text(encoding="utf-8"))
    for actor in scene.get("actors", []):
        rc = actor.get("robot-config")
        if isinstance(rc, str):
            rc_path = scene_path.parent / rc
            if not rc_path.is_file():
                rc_path = SIMULATION_DIR / rc
            rc_dict = commentjson.loads(rc_path.read_text(encoding="utf-8"))
            if enable_depth_stream:
                for sensor in rc_dict.get("sensors", []):
                    if sensor.get("id") == "FrontCamera":
                        for cap in sensor.get("capture-settings", []):
                            if cap.get("image-type") == 2:
                                cap["streaming-enabled"] = True
            actor["robot-config"] = rc_dict
    return scene


def _pitch_to_quat(pitch_deg: float) -> dict[str, float]:
    half = math.radians(pitch_deg) / 2.0
    return {"w": math.cos(half), "x": 0.0, "y": math.sin(half), "z": 0.0}


def _camera_pose(pitch_deg: float, x: float = 0.0, y: float = 0.0, z: float = 0.3) -> dict[str, Any]:
    return {
        "translation": {"x": x, "y": y, "z": z},
        "rotation": _pitch_to_quat(pitch_deg),
        "frame_id": "DEFAULT_FRAME",
    }


async def _await_task(task_coro: Any, timeout_s: float = 45.0) -> Any:
    res: list[Any] = []
    task = await task_coro(lambda v: res.append(v))
    await asyncio.wait_for(task, timeout=timeout_s)
    return res[0] if res else None


async def _stage_drone_at_start(
    drone: Any,
    position_ned: list[float],
    yaw_deg: float,
    camera_pitch_deg: float,
    transit_alt_m: float = 65.0,
) -> dict[str, Any]:
    """Safely move drone to an episode start pose via elevated transit, then stabilize hover."""
    from projectairsim.types import Pose, Quaternion, Vector3

    yaw_rad = math.radians(yaw_deg)
    rot = Quaternion({"w": math.cos(yaw_rad / 2.0), "x": 0.0, "y": 0.0, "z": math.sin(yaw_rad / 2.0)})
    cur = drone.get_ground_truth_kinematics()["pose"]["position"]

    for xy in ((float(cur["x"]), float(cur["y"])), (float(position_ned[0]), float(position_ned[1]))):
        elev = Pose(
            {
                "translation": Vector3({"x": xy[0], "y": xy[1], "z": -float(transit_alt_m)}),
                "rotation": rot,
                "frame_id": "DEFAULT_ID",
            }
        )
        if not drone.set_pose(elev, reset_kinematics=True):
            raise RuntimeError(f"Elevated transit pose rejected at {xy}")
        await asyncio.sleep(0.35)

    start_pose = Pose(
        {
            "translation": Vector3(
                {"x": float(position_ned[0]), "y": float(position_ned[1]), "z": float(position_ned[2])}
            ),
            "rotation": rot,
            "frame_id": "DEFAULT_ID",
        }
    )
    if not drone.set_pose(start_pose, reset_kinematics=True):
        raise RuntimeError(f"Start pose rejected at {position_ned}")
    await asyncio.sleep(0.45)

    # Stabilize SimpleFlight controller at the exact start position & yaw
    drone.enable_api_control()
    drone.arm()
    try:
        await _await_task(
            lambda cb: drone.move_to_position_async(
                float(position_ned[0]), float(position_ned[1]), float(position_ned[2]), 2.5, timeout_sec=12.0, callback=cb
            ),
            timeout_s=18.0,
        )
    except Exception:
        pass
    try:
        await _await_task(lambda cb: drone.rotate_to_yaw_async(yaw_rad, timeout_sec=10.0, callback=cb), timeout_s=15.0)
    except Exception:
        pass
    try:
        await _await_task(lambda cb: drone.hover_async(callback=cb), timeout_s=10.0)
    except Exception:
        pass
    drone.set_camera_pose("FrontCamera", _camera_pose(camera_pitch_deg))
    await asyncio.sleep(0.9)
    kin = drone.get_ground_truth_kinematics()
    return {
        "position_ned": [
            round(float(kin["pose"]["position"]["x"]), 3),
            round(float(kin["pose"]["position"]["y"]), 3),
            round(float(kin["pose"]["position"]["z"]), 3),
        ],
        "yaw_deg": round(yaw_deg, 2),
    }


def _capture_rgbd(
    drone: Any,
    camera_id: str,
    save_stem: Path,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    """Capture one synchronized RGB + Depth frame via RPC, saving JPG + NPZ and exact camera extrinsics."""
    from capture.collect_part_views import decode_rpc_depth

    save_stem.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            frames = drone.get_images(camera_id, [0, 2])
        except RuntimeError as exc:
            last_err = exc
            time.sleep(0.3)
            continue
        rgb, info = decode_image(frames[0])
        depth = decode_rpc_depth(frames[2]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        valid = np.isfinite(depth) & (depth > 0.2) & (depth < 500.0)
        if rgb.std() < 2.0 or valid.mean() < 0.2:
            time.sleep(0.3)
            continue

        jpg_path = save_stem.with_suffix(".jpg")
        npz_path = save_stem.with_suffix(".npz")
        cv2.imwrite(str(jpg_path), rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        np.savez_compressed(npz_path, depth_m=depth.astype(np.float16))

        cam_pos = [float(frames[0]["pos_" + k]) for k in ("x", "y", "z")]
        cam_q = [float(frames[0]["rot_" + k]) for k in ("w", "x", "y", "z")]
        cam_rot = quat_wxyz_to_rot(*cam_q)
        kin = drone.get_ground_truth_kinematics()
        dpos = kin["pose"]["position"]
        dori = kin["pose"]["orientation"]
        drone_yaw = math.atan2(
            2.0 * (float(dori["w"]) * float(dori["z"]) + float(dori["x"]) * float(dori["y"])),
            1.0 - 2.0 * (float(dori["y"]) ** 2 + float(dori["z"]) ** 2),
        )
        return {
            "jpg_path": str(jpg_path),
            "npz_path": str(npz_path),
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "time_stamp_rgb": int(frames[0].get("time_stamp", 0)),
            "time_stamp_depth": int(frames[2].get("time_stamp", 0)),
            "cam_pos_ned": [round(x, 3) for x in cam_pos],
            "cam_rot_wxyz": [round(x, 5) for x in cam_q],
            "cam_rot_matrix": cam_rot,
            "drone_pos_ned": [round(float(dpos["x"]), 3), round(float(dpos["y"]), 3), round(float(dpos["z"]), 3)],
            "drone_yaw_rad": drone_yaw,
            "depth_arr": depth,
            "encoding": info.get("encoding"),
        }
    raise RuntimeError(f"Failed to capture RGB-D within {timeout_s}s: {last_err}")


def _strip_runtime_arrays(cap: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cap.items() if k not in ("depth_arr", "cam_rot_matrix")}


async def _fly_orbit_waypoints(
    drone: Any,
    waypoints: list[dict[str, Any]],
    speed_mps: float,
    camera_pitch_deg: float,
    deadline_mono: float,
) -> bool:
    for wp in waypoints:
        rem = deadline_mono - time.monotonic()
        if rem <= 5.0:
            return False
        wn, we, wz = wp["position_ned"]
        yaw_rad = float(wp["yaw_rad"])
        try:
            await _await_task(
                lambda cb, n=wn, e=we, z=wz, t=min(25.0, rem): drone.move_to_position_async(
                    n, e, z, float(speed_mps), timeout_sec=t, callback=cb
                ),
                timeout_s=min(30.0, rem + 3.0),
            )
        except Exception:
            pass
        rem = deadline_mono - time.monotonic()
        if rem <= 3.0:
            return False
        try:
            await _await_task(
                lambda cb, y=yaw_rad, t=min(12.0, rem): drone.rotate_to_yaw_async(y, timeout_sec=t, callback=cb),
                timeout_s=min(15.0, rem + 2.0),
            )
        except Exception:
            pass
    try:
        await _await_task(lambda cb: drone.hover_async(callback=cb), timeout_s=10.0)
    except Exception:
        pass
    drone.set_camera_pose("FrontCamera", _camera_pose(camera_pitch_deg))
    await asyncio.sleep(0.8)
    return True


def _build_starts_collage(items: list[tuple[str, Path]], out_path: Path) -> None:
    tiles = []
    for label, p in items:
        img = cv2.imread(str(p))
        if img is None:
            continue
        tile = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (640, 36), (20, 20, 20), -1)
        cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    cols = 2
    rows = int(math.ceil(len(tiles) / cols))
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((360, 640, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ---------------------------------------------------------------------------
# Summary aggregation & Markdown report generation
# ---------------------------------------------------------------------------

def summarize_trials(trials: list[dict[str, Any]], target_stop_dist_m: float = 6.0) -> dict[str, Any]:
    """Aggregate metrics split by viewpoint regime (`direct`, `occluded`, `all`) and condition."""
    summary: dict[str, Any] = {}
    for regime in ("direct", "occluded", "all"):
        summary[regime] = {}
        for cond in ("far_only", "preset_recheck", "active_viewpoint"):
            sub = [
                t
                for t in trials
                if t["condition"] == cond and (regime == "all" or t["view_regime"] == regime)
            ]
            n = len(sub)
            if n == 0:
                continue
            clear_cnt = sum(1 for t in sub if t["best_part_clear"])
            vis_cnt = sum(1 for t in sub if t["best_part_visible"])
            corr_cnt = sum(1 for t in sub if t["verdict_correct"])
            fp_cnt = sum(1 for t in sub if t["damage_gt"] == "intact" and t["verdict"] == "confirmed")
            fn_cnt = sum(1 for t in sub if t["damage_gt"] == "damaged" and t["verdict"] == "ruled_out")
            unc_cnt = sum(1 for t in sub if t["verdict"] == "still_uncertain")
            col_cnt = sum(int(t.get("collisions", 0)) for t in sub)
            stop_errs = [
                float(t["stop_dist_err_m"])
                if t.get("stop_dist_err_m") is not None
                else abs(float(t["final_depth_m"]) - target_stop_dist_m)
                for t in sub
                if t.get("stop_dist_err_m") is not None or t.get("final_depth_m") is not None
            ]
            part_dists = [float(t["best_dist_to_part_m"]) for t in sub if t.get("best_dist_to_part_m") is not None]
            summary[regime][cond] = {
                "n_trials": n,
                "n_unique_targets": len({t["target_id"] for t in sub}),
                "part_clear_count": clear_cnt,
                "part_clear_rate": round(clear_cnt / n, 4),
                "part_visible_count": vis_cnt,
                "part_visible_rate": round(vis_cnt / n, 4),
                "verdict_correct_count": corr_cnt,
                "verdict_accuracy": round(corr_cnt / n, 4),
                "false_positive_count": fp_cnt,
                "false_negative_count": fn_cnt,
                "uncertain_count": unc_cnt,
                "mean_stop_dist_err_m": round(float(np.mean(stop_errs)), 3) if stop_errs else None,
                "mean_best_dist_to_part_m": round(float(np.mean(part_dists)), 2) if part_dists else None,
                "mean_flight_time_s": round(float(np.mean([t["flight_time_s"] for t in sub])), 2),
                "mean_captures": round(float(np.mean([t["num_captures"] for t in sub])), 2),
                "total_collisions": col_cnt,
            }
    return summary


def render_report_md(
    *,
    meta: dict[str, Any],
    summary: dict[str, Any],
    trials: list[dict[str, Any]],
) -> str:
    lines = [
        "# 海港任务 1：指定部位二次抵近观测三条件对照报告",
        "",
        f"- 生成时间：{meta['finished_at']}",
        f"- 仿真地址：`{meta['address']}` · VLM 模型：`{meta['vlm_model']}`（think={meta['think']}）",
        f"- 统一预算：时间上限 `{meta['budget']['max_duration_s']} s`，取图上限 `{meta['budget']['max_captures']} 张`，目标停距 `{meta['budget']['target_stop_dist_m']} m`",
        f"- 独立拍清判据：投影入画（边距 ≥ {meta['scoring']['part_in_frame_margin']*100:.0f}%）+ 正对法向 ≤ {meta['scoring']['part_max_view_angle_deg']}° + 无前景深度遮挡（差 ≤ {meta['scoring']['part_depth_occlusion_tol_m']} m）+ 距离 ≤ {meta['scoring']['part_clear_max_dist_m']} m 且等效跨度 ≥ {meta['scoring']['part_clear_min_span_px']} px",
        "",
        "## 一、分工况汇总对比（无遮挡 vs 有遮挡）",
        "",
        "| 工况 | 条件 | 试验数(独立目标) | 指定部位拍清率 | 复评正确率 | 误报/漏报/存疑 | 平均停距误差(m) | 平均部位距(m) | 平均飞行耗时(s) | 平均取图(张) | 碰撞 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    cond_cn = {
        "far_only": "A. 只用首轮远景",
        "preset_recheck": "B. 预设机位直飞复查",
        "active_viewpoint": "C. 主动选视角复查",
    }
    regime_cn = {"direct": "无遮挡(direct)", "occluded": "有遮挡(occluded)", "all": "全部合计(all)"}
    for regime in ("direct", "occluded", "all"):
        for cond in ("far_only", "preset_recheck", "active_viewpoint"):
            s = summary.get(regime, {}).get(cond)
            if not s:
                continue
            se = f"{s['mean_stop_dist_err_m']:.2f}" if s["mean_stop_dist_err_m"] is not None else "—"
            pd = f"{s['mean_best_dist_to_part_m']:.2f}" if s["mean_best_dist_to_part_m"] is not None else "—"
            lines.append(
                f"| {regime_cn[regime]} | {cond_cn[cond]} | {s['n_trials']} ({s['n_unique_targets']}) "
                f"| {s['part_clear_count']}/{s['n_trials']} ({s['part_clear_rate']*100:.1f}%) "
                f"| {s['verdict_correct_count']}/{s['n_trials']} ({s['verdict_accuracy']*100:.1f}%) "
                f"| {s['false_positive_count']}/{s['false_negative_count']}/{s['uncertain_count']} "
                f"| {se} | {pd} | {s['mean_flight_time_s']:.1f} | {s['mean_captures']:.2f} | {s['total_collisions']} |"
            )

    lines.extend(
        [
            "",
            "## 二、逐次试验明细",
            "",
            "| 目标 | 真值 | 工况 | 条件 | 换位动作 | 伺服停机 | 拍清 | 最佳部位距(m) | 末帧深度(m) | VLM 终评 | 正确 | 耗时(s) | 张数 | 碰撞 |",
            "|---|---|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for t in trials:
        fd = f"{t['final_depth_m']:.2f}" if t.get("final_depth_m") is not None else "—"
        bd = f"{t['best_dist_to_part_m']:.2f}" if t.get("best_dist_to_part_m") is not None else "—"
        lines.append(
            f"| {t['target_id']} | {t['damage_gt']} | {t['view_regime']} | {t['condition']} "
            f"| {t.get('maneuver_taken', '—')} | {t.get('servo_stop_reason', '—')} "
            f"| {'是' if t['best_part_clear'] else '否'} | {bd} | {fd} "
            f"| `{t['verdict']}` | {'✓' if t['verdict_correct'] else '✗'} "
            f"| {t['flight_time_s']:.1f} | {t['num_captures']} | {t['collisions']} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main async workflow
# ---------------------------------------------------------------------------

async def run_verify_starts(
    *,
    cfg: dict[str, Any],
    address: str,
    out_dir: Path,
    selected_targets: set[str] | None = None,
    selected_views: set[str] | None = None,
) -> dict[str, Any]:
    from projectairsim import Drone, ProjectAirSimClient, World

    out_dir.mkdir(parents=True, exist_ok=True)
    configure_client_log(out_dir)
    scene_path = project_path(cfg["scene_config"])
    runtime_scene = _load_harbor_runtime_scene(scene_path, enable_depth_stream=False)
    (out_dir / "runtime-scene.json").write_text(json.dumps(runtime_scene, indent=2), encoding="utf-8")

    client = ProjectAirSimClient(address)
    results: list[dict[str, Any]] = []
    collage_items: list[tuple[str, Path]] = []
    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        world = World.__new__(World)
        world.client = client
        world.sim_instance_idx = -1
        world.load_scene(runtime_scene, delay_after_load_sec=2)
        drone = Drone(client, world, cfg["drone_id"])
        drone.enable_api_control()
        drone.arm()
        try:
            await _await_task(lambda cb: drone.takeoff_async(timeout_sec=20, callback=cb), timeout_s=28.0)
        except Exception:
            pass

        for tgt in cfg["targets"]:
            tid = tgt["target_id"]
            if selected_targets and tid not in selected_targets:
                continue
            for vname, scfg in tgt["starts"].items():
                if selected_views and vname not in selected_views:
                    continue
                print(f"[verify-starts] Staging {tid} / {vname} ...", flush=True)
                staged = await _stage_drone_at_start(
                    drone, scfg["position_ned"], scfg["yaw_deg"], scfg["camera_pitch_deg"]
                )
                cap = _capture_rgbd(drone, cfg["camera_id"], out_dir / "starts" / f"{tid}_{vname}")
                geom = evaluate_capture_geometry(
                    cam_pos_ned=cap["cam_pos_ned"],
                    cam_rot=cap["cam_rot_matrix"],
                    depth_arr=cap["depth_arr"],
                    evaluation_gt=tgt["evaluation_gt"],
                    scoring_cfg=cfg["scoring"],
                    rgb_wh=tuple(cfg["rgb_resolution"]),
                    fov_deg=float(cfg["camera_fov_deg"]),
                )
                row = {
                    "target_id": tid,
                    "view_regime": vname,
                    "damage_gt": tgt["evaluation_gt"]["damage_gt"],
                    "staged": staged,
                    "capture": _strip_runtime_arrays(cap),
                    "geometry": geom,
                }
                results.append(row)
                tag = f"{tid} [{vname}] vis={geom['part_visible']} ang={geom['view_angle_deg']}d d={geom['dist_to_part_m']}m"
                collage_items.append((tag, Path(cap["jpg_path"])))
                print(f"  -> {tag} occ={geom['depth_occluded']} uv={geom['part_pixel_uv']}", flush=True)
    finally:
        client.disconnect()

    _build_starts_collage(collage_items, out_dir / "starts_grid.jpg")
    report = {"checked_at": datetime.now().isoformat(), "address": address, "starts": results}
    (out_dir / "starts_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


async def run_benchmark(
    *,
    cfg: dict[str, Any],
    address: str,
    out_dir: Path,
    vlm: HarborPartVLM,
    think: bool,
    planner_mode: str = "vlm",
    template_dir: Path | None = None,
    selected_targets: set[str] | None = None,
    selected_views: set[str] | None = None,
    selected_conditions: set[str] | None = None,
    reuse_starts_dir: Path | None = None,
    record_video: bool = False,
    controller: str = "baseline_p",
    cbf_overrides: dict[str, Any] | None = None,
    wind_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    from projectairsim import Drone, ProjectAirSimClient, World

    out_dir.mkdir(parents=True, exist_ok=True)
    configure_client_log(out_dir)
    scene_path = project_path(cfg["scene_config"])
    runtime_scene = _load_harbor_runtime_scene(scene_path, enable_depth_stream=True)
    (out_dir / "runtime-scene.json").write_text(json.dumps(runtime_scene, indent=2), encoding="utf-8")

    conditions = [c for c in ("far_only", "preset_recheck", "active_viewpoint") if not selected_conditions or c in selected_conditions]
    budget = cfg["budget"]
    scoring = cfg["scoring"]
    servo_defaults = cfg["servo_defaults"]
    max_duration_s = float(budget["max_duration_s"])
    max_captures = int(budget["max_captures"])
    stop_dist_m = float(budget["target_stop_dist_m"])

    print("Loading shared DINOv2 ViT-S/14 matcher on local GPU...", flush=True)
    matcher = SharedDinoMatcher()
    print(f"DINOv2 ready on device={matcher.device}", flush=True)
    tmpl_planner = TemplateViewpointPlanner(matcher) if planner_mode == "template" else None

    client = ProjectAirSimClient(address)
    collision_flags: dict[str, int] = {cfg["drone_id"]: 0}
    trials: list[dict[str, Any]] = []
    collage_items: list[tuple[str, Path]] = []

    try:
        client.connect()
        client.socket_services.recv_timeout = 60000
        client.socket_topics.recv_buffer_size = 24
        world = World.__new__(World)
        world.client = client
        world.sim_instance_idx = -1
        world.load_scene(runtime_scene, delay_after_load_sec=2)
        if any(abs(v) > 1e-6 for v in wind_ned_mps):
            world.set_wind_velocity(float(wind_ned_mps[0]), float(wind_ned_mps[1]), float(wind_ned_mps[2]))
            print(f"[Native Wind] world.set_wind_velocity={world.get_wind_velocity()} m/s", flush=True)

        drone_id = cfg["drone_id"]
        drone = Drone(client, world, drone_id)
        if "collision_info" in drone.robot_info:
            def _on_col(_t: str, _m: dict[str, Any]) -> None:
                collision_flags[drone_id] = collision_flags.get(drone_id, 0) + 1
            client.subscribe(drone.robot_info["collision_info"], _on_col)

        drone.enable_api_control()
        drone.arm()
        try:
            await _await_task(lambda cb: drone.takeoff_async(timeout_sec=20, callback=cb), timeout_s=28.0)
        except Exception:
            pass

        for tgt in cfg["targets"]:
            tid = tgt["target_id"]
            if selected_targets and tid not in selected_targets:
                continue
            eval_gt = tgt["evaluation_gt"]
            tgt_stop_dist = float(tgt.get("stop_dist_m", stop_dist_m))
            for vname, scfg in tgt["starts"].items():
                if selected_views and vname not in selected_views:
                    continue

                # Step 0: Capture (or load) ONE frozen Round-1 far RGB-D shared by all 3 conditions
                r1_stem = out_dir / "captures" / tid / vname / "round1_far"
                print(f"\n================ {tid} / {vname} ================", flush=True)
                await _stage_drone_at_start(
                    drone, scfg["position_ned"], scfg["yaw_deg"], scfg["camera_pitch_deg"]
                )
                r1_cap = _capture_rgbd(drone, cfg["camera_id"], r1_stem)
                r1_geom = evaluate_capture_geometry(
                    cam_pos_ned=r1_cap["cam_pos_ned"],
                    cam_rot=r1_cap["cam_rot_matrix"],
                    depth_arr=r1_cap["depth_arr"],
                    evaluation_gt=eval_gt,
                    scoring_cfg=scoring,
                    rgb_wh=tuple(cfg["rgb_resolution"]),
                    fov_deg=float(cfg["camera_fov_deg"]),
                )
                r1_img = VLMImage(image_id=f"{tgt['neutral_id']}-R1", view_type="首轮远景斜俯视", image_path=Path(r1_cap["jpg_path"]))
                print(
                    f"[Round-1 frozen] vis={r1_geom['part_visible']} clear={r1_geom['part_clear']} "
                    f"dist={r1_geom['dist_to_part_m']}m ang={r1_geom['view_angle_deg']}deg occ={r1_geom['depth_occluded']}",
                    flush=True,
                )

                for cond in conditions:
                    t_cond_start = time.monotonic()
                    ep_dir = out_dir / "captures" / tid / vname / cond
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    ep_captures: list[dict[str, Any]] = [
                        {"role": "round1_far", "capture": _strip_runtime_arrays(r1_cap), "geometry": r1_geom}
                    ]
                    vlm_imgs: list[VLMImage] = [r1_img]
                    flight_time_s = 0.0
                    col_delta = 0
                    maneuver_taken = "none"
                    servo_summary: dict[str, Any] | None = None
                    vlm_plan_log: list[dict[str, Any]] = []
                    video_info: dict[str, Any] | None = None

                    if cond == "far_only":
                        # Condition A: No flight, assess Round-1 image only
                        pass

                    elif cond == "preset_recheck":
                        # Condition B: Same start pose -> direct DINO visual servo along initial ray (no active viewpoint switch)
                        await _stage_drone_at_start(
                            drone, scfg["position_ned"], scfg["yaw_deg"], scfg["camera_pitch_deg"]
                        )
                        col_before = collision_flags.get(drone_id, 0)
                        t_fly = time.monotonic()
                        rec: HarborVideoRecorder | None = None
                        if record_video:
                            rec = HarborVideoRecorder(
                                ep_dir / "flight.mp4",
                                title=f"任务1｜{tid} ({vname}) 预设直飞",
                            )
                            rec.set_badge("B:预设直飞")
                            rec.set_status(tgt["neutral_id"], "准备视觉抵近")
                            rec.attach_streams(client, drone)
                            rec.start()
                        try:
                            servo_cfg = {
                                **servo_defaults,
                                "camera_pitch_deg": scfg["camera_pitch_deg"],
                                "stop_dist_m": tgt_stop_dist,
                                "min_alt_m": float(tgt["min_alt_m"]),
                                "template_roi": tgt.get("template_roi", servo_defaults["template_roi"]),
                                "max_duration_s": min(float(servo_defaults["max_duration_s"]), max_duration_s - 6.0),
                                "controller": controller,
                                "cbf_config": dict(cbf_overrides or {}),
                            }
                            servo_res = await run_dino_visual_servo(
                                client=client,
                                drone=drone,
                                drone_id=drone_id,
                                target_id=tgt["neutral_id"],
                                matcher=matcher,
                                template_image_path=r1_cap["jpg_path"],
                                servo_cfg=servo_cfg,
                                collision_flags=collision_flags,
                                recorder=rec,
                            )
                            if rec is not None:
                                rec.set_status(tgt["neutral_id"], "悬停近景取证")
                            await asyncio.sleep(0.5)
                            close_cap = _capture_rgbd(drone, cfg["camera_id"], ep_dir / "round2_preset_close")
                        finally:
                            if rec is not None:
                                video_info = rec.stop()
                        flight_time_s = round(time.monotonic() - t_fly, 2)
                        col_delta = collision_flags.get(drone_id, 0) - col_before
                        close_geom = evaluate_capture_geometry(
                            cam_pos_ned=close_cap["cam_pos_ned"],
                            cam_rot=close_cap["cam_rot_matrix"],
                            depth_arr=close_cap["depth_arr"],
                            evaluation_gt=eval_gt,
                            scoring_cfg=scoring,
                            rgb_wh=tuple(cfg["rgb_resolution"]),
                            fov_deg=float(cfg["camera_fov_deg"]),
                        )
                        ep_captures.append(
                            {"role": "round2_close", "capture": _strip_runtime_arrays(close_cap), "geometry": close_geom}
                        )
                        vlm_imgs.append(
                            VLMImage(
                                image_id=f"{tgt['neutral_id']}-R2-CLOSE",
                                view_type="预设直飞近景",
                                image_path=Path(close_cap["jpg_path"]),
                            )
                        )
                        maneuver_taken = "direct_approach"
                        servo_summary = {k: v for k, v in servo_res.items() if k != "trace"}
                        (ep_dir / "servo_trace.json").write_text(
                            json.dumps(servo_res, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        collage_items.append((f"{tid}[{vname}] B:preset clear={close_geom['part_clear']}", Path(close_cap["jpg_path"])))

                    elif cond == "active_viewpoint":
                        # Condition C: Active planner inspects Round-1 -> repositions if occluded (depth-only center) -> DINO servo
                        await _stage_drone_at_start(
                            drone, scfg["position_ned"], scfg["yaw_deg"], scfg["camera_pitch_deg"]
                        )
                        col_before = collision_flags.get(drone_id, 0)
                        t_fly = time.monotonic()
                        deadline = t_fly + max_duration_s
                        rec = None
                        if record_video:
                            rec = HarborVideoRecorder(
                                ep_dir / "flight.mp4",
                                title=f"任务1｜{tid} ({vname}) 主动换位({planner_mode})",
                            )
                            rec.set_badge(f"C:主动视角({planner_mode})")
                            rec.set_status(tgt["neutral_id"], "分析首轮远景视角")
                            rec.attach_streams(client, drone)
                            rec.start()

                        ref_path: Path | None = None
                        ref_roi: list[float] | None = None
                        try:
                            if tmpl_planner is not None:
                                if template_dir is not None:
                                    for cand in (template_dir / f"{tid}.jpg", template_dir / f"{tid}.png"):
                                        if cand.exists():
                                            ref_path = cand
                                            break
                                if ref_path is None and tgt.get("reference_template"):
                                    ref_path = project_path(tgt["reference_template"])
                                if ref_path is None:
                                    raise FileNotFoundError(
                                        f"--planner template requires a reference template for {tid} via --template-dir or target.reference_template"
                                    )
                                ref_roi = tgt.get(
                                    "reference_template_roi",
                                    [0.22, 0.35, 0.78, 0.75] if tgt.get("category") == "Vessel" else [0.15, 0.35, 0.85, 0.78],
                                )
                                plan_out = tmpl_planner.plan_viewpoint(
                                    template_path=ref_path,
                                    template_roi=ref_roi,
                                    image_path=Path(r1_cap["jpg_path"]),
                                )
                                vlm_plan_log.append({"attempt": 1, **plan_out})
                                plan_res = plan_out["result"]
                                plan_lat_ms = float(plan_out["latency_ms"])
                            else:
                                plan1 = vlm.plan_viewpoint(
                                    neutral_id=tgt["neutral_id"],
                                    target_class_hint=tgt["target_class_hint"],
                                    focus_part_request=tgt["focus_part_request"],
                                    image=r1_img,
                                    attempt_index=1,
                                )
                                vlm_plan_log.append(
                                    {"attempt": 1, "latency_ms": round(plan1.latency_ms, 1), "result": plan1.result}
                                )
                                plan_res = plan1.result
                                plan_lat_ms = float(plan1.latency_ms)

                            maneuver = plan_res["recommended_maneuver"]
                            maneuver_taken = maneuver
                            template_path = Path(r1_cap["jpg_path"])
                            cur_cap = r1_cap

                            print(
                                f"  [Active {planner_mode} plan #1] vis={plan_res['part_visibility']} "
                                f"maneuver={maneuver} ({plan_lat_ms:.0f}ms)",
                                flush=True,
                            )

                            # Active repositioning if planner judges target part is occluded / on opposite side
                            angle_map = {
                                "reposition_opposite": 180.0,
                                "reposition_left_90": -90.0,
                                "reposition_right_90": 90.0,
                            }
                            if maneuver in angle_map and len(vlm_imgs) < max_captures:
                                if rec is not None:
                                    rec.set_status(tgt["neutral_id"], f"绕飞换位 {int(angle_map[maneuver])}°")
                                depth_off = float(
                                    tgt.get("depth_center_offset_m", budget.get("depth_center_offset_m", 4.0))
                                )
                                est_center_ne, est_r = estimate_target_center_from_depth(
                                    drone_pos_ned=cur_cap["drone_pos_ned"],
                                    drone_yaw_rad=cur_cap["drone_yaw_rad"],
                                    camera_pitch_deg=scfg["camera_pitch_deg"],
                                    depth_arr=cur_cap["depth_arr"],
                                    min_radius_m=float(budget["reposition_min_radius_m"]),
                                    max_radius_m=float(budget["reposition_max_radius_m"]),
                                    depth_center_offset_m=depth_off,
                                    roi=tgt.get("template_roi"),
                                )
                                start_alt = -float(cur_cap["drone_pos_ned"][2])
                                orbit_alt = max(start_alt, float(budget["orbit_min_alt_m"]))
                                max_standoff = float(tgt.get("max_orbit_standoff_m", budget["reposition_max_radius_m"]))
                                orbit_r = min(max_standoff, est_r)
                                pitch_aligned_alt = orbit_r * math.tan(math.radians(abs(float(scfg["camera_pitch_deg"]))))
                                final_alt = max(
                                    float(tgt["min_alt_m"]) + 6.0,
                                    start_alt - 1.5,
                                    min(start_alt + 1.5, pitch_aligned_alt),
                                )
                                wps = plan_orbit_waypoints(
                                    drone_pos_ned=cur_cap["drone_pos_ned"],
                                    center_ne=est_center_ne,
                                    radius_m=orbit_r,
                                    delta_angle_deg=angle_map[maneuver],
                                    orbit_alt_m=orbit_alt,
                                    final_alt_m=final_alt,
                                    num_arc_steps=3,
                                )
                                print(
                                    f"  [Active orbit] est_center_ne={np.round(est_center_ne,2).tolist()} "
                                    f"est_r={est_r:.1f}m orbit_r={orbit_r:.1f}m final_alt={final_alt:.1f}m delta={angle_map[maneuver]}deg",
                                    flush=True,
                                )
                                await _fly_orbit_waypoints(
                                    drone,
                                    wps,
                                    speed_mps=float(budget["orbit_speed_mps"]),
                                    camera_pitch_deg=scfg["camera_pitch_deg"],
                                    deadline_mono=deadline,
                                )
                                if rec is not None:
                                    rec.set_status(tgt["neutral_id"], "中继视角确认")
                                relay_cap = _capture_rgbd(drone, cfg["camera_id"], ep_dir / "round2_active_relay")
                                relay_geom = evaluate_capture_geometry(
                                    cam_pos_ned=relay_cap["cam_pos_ned"],
                                    cam_rot=relay_cap["cam_rot_matrix"],
                                    depth_arr=relay_cap["depth_arr"],
                                    evaluation_gt=eval_gt,
                                    scoring_cfg=scoring,
                                    rgb_wh=tuple(cfg["rgb_resolution"]),
                                    fov_deg=float(cfg["camera_fov_deg"]),
                                )
                                ep_captures.append(
                                    {"role": "round2_relay", "capture": _strip_runtime_arrays(relay_cap), "geometry": relay_geom}
                                )
                                if tmpl_planner is not None and ref_path is not None and ref_roi is not None:
                                    relay_check = tmpl_planner.plan_viewpoint(
                                        template_path=ref_path,
                                        template_roi=ref_roi,
                                        image_path=Path(relay_cap["jpg_path"]),
                                    )
                                    vlm_plan_log.append({"attempt": 2, "stage": "relay_verify", **relay_check})
                                relay_img = VLMImage(
                                    image_id=f"{tgt['neutral_id']}-R2-RELAY",
                                    view_type="主动换位中继视角",
                                    image_path=Path(relay_cap["jpg_path"]),
                                )
                                vlm_imgs.append(relay_img)
                                template_path = Path(relay_cap["jpg_path"])
                                cur_cap = relay_cap
                                print(
                                    f"  [Relay captured] vis={relay_geom['part_visible']} ang={relay_geom['view_angle_deg']}deg "
                                    f"dist={relay_geom['dist_to_part_m']}m uv={relay_geom['part_pixel_uv']}",
                                    flush=True,
                                )

                            # Closed-loop DINO visual servo approach from current (or repositioned) viewpoint
                            rem_s = deadline - time.monotonic()
                            if len(vlm_imgs) < max_captures and rem_s > 8.0:
                                servo_cfg = {
                                    **servo_defaults,
                                    "camera_pitch_deg": scfg["camera_pitch_deg"],
                                    "stop_dist_m": tgt_stop_dist,
                                    "min_alt_m": float(tgt["min_alt_m"]),
                                    "template_roi": tgt.get("template_roi", servo_defaults["template_roi"]),
                                    "max_duration_s": min(float(servo_defaults["max_duration_s"]), max(8.0, rem_s - 4.0)),
                                    "controller": controller,
                                    "cbf_config": dict(cbf_overrides or {}),
                                    "wind_disturbance_ned": list(wind_ned_mps),
                                }
                                servo_res = await run_dino_visual_servo(
                                    client=client,
                                    drone=drone,
                                    drone_id=drone_id,
                                    target_id=tgt["neutral_id"],
                                    matcher=matcher,
                                    template_image_path=template_path,
                                    servo_cfg=servo_cfg,
                                    collision_flags=collision_flags,
                                    recorder=rec,
                                )
                                if rec is not None:
                                    rec.set_status(tgt["neutral_id"], "悬停近景取证")
                                await asyncio.sleep(0.5)
                                close_cap = _capture_rgbd(drone, cfg["camera_id"], ep_dir / "round2_active_close")
                                close_geom = evaluate_capture_geometry(
                                    cam_pos_ned=close_cap["cam_pos_ned"],
                                    cam_rot=close_cap["cam_rot_matrix"],
                                    depth_arr=close_cap["depth_arr"],
                                    evaluation_gt=eval_gt,
                                    scoring_cfg=scoring,
                                    rgb_wh=tuple(cfg["rgb_resolution"]),
                                    fov_deg=float(cfg["camera_fov_deg"]),
                                )
                                ep_captures.append(
                                    {"role": "round2_close", "capture": _strip_runtime_arrays(close_cap), "geometry": close_geom}
                                )
                                vlm_imgs.append(
                                    VLMImage(
                                        image_id=f"{tgt['neutral_id']}-R2-CLOSE",
                                        view_type="主动抵近近景",
                                        image_path=Path(close_cap["jpg_path"]),
                                    )
                                )
                                servo_summary = {k: v for k, v in servo_res.items() if k != "trace"}
                                (ep_dir / "servo_trace.json").write_text(
                                    json.dumps(servo_res, indent=2, ensure_ascii=False), encoding="utf-8"
                                )
                                collage_items.append((f"{tid}[{vname}] C:active clear={close_geom['part_clear']}", Path(close_cap["jpg_path"])))
                        finally:
                            if rec is not None:
                                video_info = rec.stop()
                        flight_time_s = round(time.monotonic() - t_fly, 2)
                        col_delta = collision_flags.get(drone_id, 0) - col_before

                    # Final VLM assessment on collected images for this condition
                    assess = vlm.assess_damage(
                        neutral_id=tgt["neutral_id"],
                        target_class_hint=tgt["target_class_hint"],
                        focus_part_request=tgt["focus_part_request"],
                        images=vlm_imgs,
                        condition_label=cond,
                    )
                    verdict = assess.result["verdict"]
                    verdict_correct = score_verdict_correct(verdict, eval_gt["damage_gt"])

                    best_clear = any(c["geometry"]["part_clear"] for c in ep_captures)
                    best_vis = any(c["geometry"]["part_visible"] for c in ep_captures)
                    vis_dists = [c["geometry"]["dist_to_part_m"] for c in ep_captures if c["geometry"]["part_visible"]]
                    all_dists = [c["geometry"]["dist_to_part_m"] for c in ep_captures]
                    best_dist = min(vis_dists) if vis_dists else min(all_dists)
                    final_depth_m = servo_summary.get("final_depth_m") if servo_summary else None
                    if final_depth_m is None and len(ep_captures) > 1:
                        final_depth_m = ep_captures[-1]["geometry"].get("center_depth_m")

                    trial = {
                        "target_id": tid,
                        "neutral_id": tgt["neutral_id"],
                        "damage_gt": eval_gt["damage_gt"],
                        "view_regime": vname,
                        "condition": cond,
                        "maneuver_taken": maneuver_taken,
                        "servo_stop_reason": servo_summary.get("stop_reason") if servo_summary else None,
                        "final_depth_m": final_depth_m,
                        "target_stop_dist_m": tgt_stop_dist,
                        "stop_dist_err_m": round(abs(float(final_depth_m) - tgt_stop_dist), 3) if final_depth_m is not None else None,
                        "best_part_visible": best_vis,
                        "best_part_clear": best_clear,
                        "best_dist_to_part_m": round(float(best_dist), 3),
                        "verdict": verdict,
                        "damage_severity": assess.result["damage_severity"],
                        "vlm_part_clearly_visible": assess.result["part_clearly_visible"],
                        "verdict_correct": verdict_correct,
                        "flight_time_s": flight_time_s,
                        "total_wall_time_s": round(time.monotonic() - t_cond_start, 2),
                        "num_captures": len(vlm_imgs),
                        "collisions": int(col_delta),
                        "vlm_plan_log": vlm_plan_log,
                        "vlm_assessment": {
                            "latency_ms": round(assess.latency_ms, 1),
                            "result": assess.result,
                        },
                        "servo_summary": servo_summary,
                        "video": video_info,
                        "captures": ep_captures,
                    }
                    trials.append(trial)
                    print(
                        f"  [{cond}] clear={best_clear} dist={best_dist:.2f}m stop_d={final_depth_m} "
                        f"verdict={verdict} (gt={eval_gt['damage_gt']}, ok={verdict_correct}) "
                        f"t={flight_time_s}s caps={len(vlm_imgs)} col={col_delta}",
                        flush=True,
                    )
                    # Checkpoint incrementally so partial progress is never lost
                    (out_dir / "trials_partial.json").write_text(
                        json.dumps(trials, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
    finally:
        try:
            if any(abs(v) > 1e-6 for v in wind_ned_mps):
                world.set_wind_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
        client.disconnect()

    summary = summarize_trials(trials, target_stop_dist_m=stop_dist_m)
    meta = {
        "finished_at": datetime.now().isoformat(),
        "address": address,
        "controller": controller,
        "cbf_overrides": dict(cbf_overrides or {}),
        "wind_ned_mps": list(wind_ned_mps),
        "planner_mode": planner_mode,
        "vlm_backend": vlm.backend,
        "vlm_model": vlm.model_name,
        "think": think,
        "budget": budget,
        "scoring": scoring,
    }
    full = {"meta": meta, "summary": summary, "trials": trials}
    (out_dir / "summary.json").write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report_md(meta=meta, summary=summary, trials=trials), encoding="utf-8")
    _build_starts_collage(collage_items, out_dir / "final_closeups_grid.jpg")
    partial = out_dir / "trials_partial.json"
    if partial.exists():
        partial.unlink()
    return full


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coastal Harbor Task 1: Targeted Part Re-Inspection benchmark")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--address", default="172.21.192.1")
    p.add_argument("--mode", choices=("verify-starts", "run"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--targets", nargs="*", default=None, help="Filter target IDs (default: all 4)")
    p.add_argument("--views", nargs="*", choices=("direct", "occluded"), default=None)
    p.add_argument("--conditions", nargs="*", choices=("far_only", "preset_recheck", "active_viewpoint"), default=None)
    p.add_argument("--vlm", choices=("ollama", "mock", "none"), default="ollama")
    p.add_argument(
        "--planner",
        choices=("vlm", "template"),
        default="vlm",
        help="Active viewpoint planner: 'vlm' (remote server) or 'template' (standalone onboard DINOv2, zero server).",
    )
    p.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Directory containing <target_id>.jpg reference templates for standalone --planner template mode.",
    )
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--controller", choices=("baseline_p", "cbf_qp"), default="baseline_p")
    p.add_argument("--cbf-fov", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cbf-depth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cbf-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wind-north-mps", type=float, default=0.0)
    p.add_argument("--wind-east-mps", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    sel_t = set(args.targets) if args.targets else None
    sel_v = set(args.views) if args.views else None
    sel_c = set(args.conditions) if args.conditions else None

    if args.mode == "verify-starts":
        asyncio.run(
            run_verify_starts(
                cfg=cfg,
                address=args.address,
                out_dir=args.out,
                selected_targets=sel_t,
                selected_views=sel_v,
            )
        )
    else:
        vlm = HarborPartVLM(
            backend=args.vlm,
            model_name=args.model,
            ollama_url=args.ollama_url,
            ssh_host=args.ssh_host.strip() if args.ssh_host else None,
            ssh_key=args.ssh_key.strip() if args.ssh_key else None,
            think=args.think,
        )
        cbf_overrides = {
            "enable_fov_cbf": bool(args.cbf_fov),
            "enable_depth_cbf": bool(args.cbf_depth),
            "enable_obstacle_memory": bool(args.cbf_memory),
        }
        asyncio.run(
            run_benchmark(
                cfg=cfg,
                address=args.address,
                out_dir=args.out,
                vlm=vlm,
                think=args.think,
                planner_mode=args.planner,
                template_dir=args.template_dir,
                selected_targets=sel_t,
                selected_views=sel_v,
                selected_conditions=sel_c,
                record_video=args.record_video,
                controller=args.controller,
                cbf_overrides=cbf_overrides,
                wind_ned_mps=(float(args.wind_north_mps), float(args.wind_east_mps), 0.0),
            )
        )


if __name__ == "__main__":
    main()
