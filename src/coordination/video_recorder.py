"""Real-time 1920x720 multi-UAV visualization recorder with 2D airport trajectory map & altitude profile."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.paths import DATA
from common.sensors import decode_image


TILE_W = 480
TILE_H = 270
HEADER_H = 52
ALT_H = 128
MAP_PANEL_W = 480
MAP_BOX_H = 456
TELEM_BOX_H = 212
CANVAS_W = TILE_W * 3 + MAP_PANEL_W  # 1920
CANVAS_H = HEADER_H + TILE_H * 2 + ALT_H  # 720

DEFAULT_GLOBAL_GRID = DATA / "maps" / "airport_scene" / "grids" / "global.npz"

_FONT_CANDIDATES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_PILL_CACHE: dict[tuple[str, int, tuple[int, int, int], tuple[int, int, int]], np.ndarray] = {}

# Distinct BGR colors for all 6 UAV trajectories
DRONE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Scout-1": (245, 200, 75),    # Sky Cyan
    "Scout-2": (130, 225, 115),   # Emerald Mint
    "Scout-3": (235, 145, 195),   # Soft Violet
    "Inspect-1": (55, 155, 255),  # Bright Orange
    "Inspect-2": (60, 225, 255),  # Golden Yellow
    "Inspect-3": (95, 105, 255),  # Coral Red
}

ALL_DRONE_ORDER = ("Scout-1", "Scout-2", "Scout-3", "Inspect-1", "Inspect-2", "Inspect-3")


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size=size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _render_pill(
    text: str,
    font_size: int = 16,
    bg_bgr: tuple[int, int, int] = (24, 28, 36),
    fg_bgr: tuple[int, int, int] = (245, 248, 252),
    pad_x: int = 8,
    pad_y: int = 4,
) -> np.ndarray:
    cache_key = (text, font_size, bg_bgr, fg_bgr)
    cached = _PILL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    font = _load_font(font_size)
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    w = tw + pad_x * 2
    h = th + pad_y * 2

    bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    fg_rgb = (fg_bgr[2], fg_bgr[1], fg_bgr[0])
    pil_img = Image.new("RGB", (w, h), bg_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=fg_rgb)
    bgr = np.array(pil_img, dtype=np.uint8)[:, :, ::-1].copy()

    if len(_PILL_CACHE) < 1024:
        _PILL_CACHE[cache_key] = bgr
    return bgr


def _put_pill(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_bgr: tuple[int, int, int] = (24, 28, 36),
    fg_bgr: tuple[int, int, int] = (245, 248, 252),
    font_size: int = 16,
    align_right: bool = False,
) -> None:
    if not text:
        return
    pill = _render_pill(text, font_size=font_size, bg_bgr=bg_bgr, fg_bgr=fg_bgr)
    ph, pw = pill.shape[:2]
    ih, iw = img.shape[:2]
    if align_right:
        x = iw - pw - x
    x0 = max(0, min(iw - 1, x))
    y0 = max(0, min(ih - 1, y))
    x1 = min(iw, x0 + pw)
    y1 = min(ih, y0 + ph)
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = pill[: y1 - y0, : x1 - x0]


def _blank_tile(label: str) -> np.ndarray:
    tile = np.full((TILE_H, TILE_W, 3), (26, 30, 36), dtype=np.uint8)
    _put_pill(tile, label, 16, TILE_H // 2 - 14, bg_bgr=(42, 48, 58), fg_bgr=(195, 202, 214), font_size=16)
    return tile


def format_drone_cn(drone_id: str) -> str:
    mapping = {
        "Scout-1": "侦察1号",
        "Scout-2": "侦察2号",
        "Scout-3": "侦察3号",
        "Inspect-1": "复查1号",
        "Inspect-2": "复查2号",
        "Inspect-3": "复查3号",
    }
    return mapping.get(drone_id, drone_id)


def format_target_cn(target_id: str) -> str:
    mapping = {
        "OBJ-A": "目标A（飞机）",
        "OBJ-B": "目标B（机库）",
        "OBJ-C": "目标C（塔台）",
    }
    return mapping.get(target_id, target_id)


def format_target_short_cn(target_id: str) -> str:
    mapping = {
        "OBJ-A": "目标A",
        "OBJ-B": "目标B",
        "OBJ-C": "目标C",
    }
    return mapping.get(target_id, target_id)


class _AirportRadarMap:
    """Renders real airport occupancy grid + live 6-UAV 2D trajectories."""

    def __init__(
        self,
        width: int,
        height: int,
        e_bounds: tuple[float, float] = (-75.0, 615.0),
        n_bounds: tuple[float, float] = (-150.0, 480.0),
        grid_npz: Path | str | None = DEFAULT_GLOBAL_GRID,
        target_positions_ned: dict[str, tuple[float, float]] | None = None,
        home_ned: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.w = width
        self.h = height
        self.pad_top = 34
        self.pad_side = 14
        self.pad_bottom = 14
        self.plot_w = self.w - 2 * self.pad_side
        self.plot_h = self.h - self.pad_top - self.pad_bottom
        self.e_min, self.e_max = e_bounds
        self.n_min, self.n_max = n_bounds

        # Keep isotropic meters-per-pixel scaling
        scale_e = self.plot_w / max(1.0, self.e_max - self.e_min)
        scale_n = self.plot_h / max(1.0, self.n_max - self.n_min)
        self.scale = min(scale_e, scale_n)
        used_w = (self.e_max - self.e_min) * self.scale
        used_h = (self.n_max - self.n_min) * self.scale
        self.off_x = self.pad_side + (self.plot_w - used_w) * 0.5
        self.off_y = self.pad_top + (self.plot_h - used_h) * 0.5

        self.target_positions_ned = dict(target_positions_ned or {})
        self.home_ned = home_ned
        self.base_img = self._build_base_map(Path(grid_npz) if grid_npz else None)

    def ned_to_px(self, north: float, east: float) -> tuple[int, int]:
        px = int(round(self.off_x + (east - self.e_min) * self.scale))
        py = int(round(self.off_y + (self.n_max - north) * self.scale))
        return (
            max(self.pad_side, min(self.w - self.pad_side - 1, px)),
            max(self.pad_top, min(self.h - self.pad_bottom - 1, py)),
        )

    def _build_base_map(self, grid_npz: Path | None) -> np.ndarray:
        img = np.full((self.h, self.w, 3), (30, 24, 20), dtype=np.uint8)
        # Plot area background (cool tactical dark slate in BGR)
        x0, y0 = int(self.off_x), int(self.off_y)
        x1 = int(self.off_x + (self.e_max - self.e_min) * self.scale)
        y1 = int(self.off_y + (self.n_max - self.n_min) * self.scale)
        cv2.rectangle(img, (x0, y0), (x1, y1), (42, 33, 26), -1)

        # Overlay real airport occupancy grid if available
        if grid_npz is not None and grid_npz.is_file():
            try:
                with np.load(grid_npz) as data:
                    occ = data["occupancy"]
                    origin = data["origin_enu_m"]
                    res = float(data["resolution_m"])
                c0 = max(0, int(np.floor((self.e_min - origin[0]) / res)))
                c1 = min(occ.shape[1], int(np.ceil((self.e_max - origin[0]) / res)))
                r0 = max(0, int(np.floor((self.n_min - origin[1]) / res)))
                r1 = min(occ.shape[0], int(np.ceil((self.n_max - origin[1]) / res)))
                if c1 > c0 and r1 > r0:
                    crop = occ[r0:r1, c0:c1]
                    crop = np.flipud(crop)  # North up
                    bgr = np.full((crop.shape[0], crop.shape[1], 3), (46, 36, 28), dtype=np.uint8)
                    bgr[crop == 0] = (54, 43, 34)       # Free apron/runway (cool slate)
                    bgr[crop == 100] = (116, 94, 76)    # Buildings / hangars / towers
                    resized = cv2.resize(bgr, (max(1, x1 - x0), max(1, y1 - y0)), interpolation=cv2.INTER_NEAREST)
                    img[y0:y1, x0:x1] = resized
            except Exception:
                pass

        # 100m coordinate grid lines
        for e_val in range(0, int(self.e_max), 100):
            gx, _ = self.ned_to_px(0.0, float(e_val))
            cv2.line(img, (gx, y0), (gx, y1), (68, 56, 44), 1)
        for n_val in range(-100, int(self.n_max), 100):
            _, gy = self.ned_to_px(float(n_val), 0.0)
            cv2.line(img, (x0, gy), (x1, gy), (68, 56, 44), 1)

        cv2.rectangle(img, (x0, y0), (x1, y1), (96, 78, 64), 1)
        _put_pill(img, "机场实时二维轨迹态势（北↑ 100m格）", 10, 6, bg_bgr=(52, 38, 30), font_size=15)

        # Home pad beacon
        hx, hy = self.ned_to_px(self.home_ned[0], self.home_ned[1])
        cv2.drawMarker(img, (hx, hy), (255, 225, 160), cv2.MARKER_SQUARE, 10, 2)
        _put_pill(img, "起降区", hx + 8, hy + 8, bg_bgr=(44, 34, 26), fg_bgr=(255, 230, 175), font_size=13)

        # Target anchors (offset above marker so drone tags on right don't collide)
        for tid, (tn, te) in self.target_positions_ned.items():
            tx, ty = self.ned_to_px(tn, te)
            cv2.circle(img, (tx, ty), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), 3, (255, 210, 80), -1, cv2.LINE_AA)
            label_x = max(self.pad_side + 4, tx - 48) if te > 440 else tx + 8
            label_y = max(self.pad_top + 4, ty - 28)
            _put_pill(img, format_target_short_cn(tid), label_x, label_y, bg_bgr=(42, 34, 26), font_size=13)

        return img

    def render(
        self,
        trajectories: dict[str, list[tuple[float, float, float, float]]],
        active_drone_ids: set[str],
    ) -> np.ndarray:
        canvas = self.base_img.copy()
        # Draw inactive/completed batch trails first, then active batch on top
        draw_order = [d for d in ALL_DRONE_ORDER if d not in active_drone_ids] + [
            d for d in ALL_DRONE_ORDER if d in active_drone_ids
        ]
        for drone_id in draw_order:
            pts_raw = trajectories.get(drone_id)
            if not pts_raw:
                continue
            color = DRONE_COLORS_BGR.get(drone_id, (220, 220, 220))
            is_active = drone_id in active_drone_ids
            if not is_active:
                color_line = tuple(int(c * 0.55) for c in color)
                thickness = 1
            else:
                color_line = color
                thickness = 2
            pts = np.array([self.ned_to_px(n, e) for _, n, e, _ in pts_raw], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, color_line, thickness, cv2.LINE_AA)
            cur_t, cur_n, cur_e, cur_alt = pts_raw[-1]
            cx, cy = self.ned_to_px(cur_n, cur_e)
            if is_active:
                cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
                _put_pill(
                    canvas,
                    f"{format_drone_cn(drone_id)} {int(round(cur_alt))}m",
                    min(cx + 8, self.w - 115),
                    max(self.pad_top + 2, cy - 12),
                    bg_bgr=(20, 26, 34),
                    fg_bgr=color,
                    font_size=12,
                )
            else:
                cv2.circle(canvas, (cx, cy), 3, color_line, -1, cv2.LINE_AA)
        return canvas


def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for candidate in (
        "/home/hui/miniconda3/envs/airsim/bin/ffmpeg",
        "/home/hui/.local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _transcode_h264_inplace(mp4_path: Path) -> bool:
    """Re-encode OpenCV mp4v output in-place to browser-playable H.264 yuv420p +faststart."""
    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin or not mp4_path.is_file() or mp4_path.stat().st_size == 0:
        return False
    tmp_path = mp4_path.with_name(mp4_path.stem + ".h264.tmp.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode == 0 and tmp_path.is_file() and tmp_path.stat().st_size > 0:
        tmp_path.replace(mp4_path)
        return True
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return False


def _draw_servo_overlay(tile: np.ndarray, overlay: dict[str, Any]) -> np.ndarray:
    """Render live DINOv2 heatmap / RGB-D roof-opening mask and Chinese servo HUD onto a FrontCamera tile."""
    h, w = tile.shape[:2]
    mode = overlay.get("mode", "dino")

    if mode == "dino":
        sim_2d = overlay.get("sim_2d")
        if sim_2d is not None:
            sim = np.asarray(sim_2d, dtype=np.float32)
            lo = float(np.percentile(sim, 55))
            hi = max(lo + 1e-4, float(sim.max()))
            norm = np.clip((sim - lo) / (hi - lo), 0.0, 1.0)
            norm_up = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)
            heat = cv2.applyColorMap(np.clip(norm_up * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            mask = norm_up > 0.32
            if np.any(mask):
                alpha = np.clip((norm_up[mask] - 0.32) / 0.68, 0.0, 1.0)[:, None] * 0.45
                tile[mask] = np.clip(
                    tile[mask].astype(np.float32) * (1.0 - alpha) + heat[mask].astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
    elif mode == "opening":
        gap_mask = overlay.get("gap_mask")
        if gap_mask is not None:
            gm = cv2.resize(np.asarray(gap_mask, dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if np.any(gm):
                tint = np.array([210, 235, 70], dtype=np.float32)  # Cyan-emerald BGR
                tile[gm] = np.clip(tile[gm].astype(np.float32) * 0.65 + tint * 0.35, 0, 255).astype(np.uint8)
                contours, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, contours, -1, (120, 255, 110), 2, cv2.LINE_AA)

    # Center reference reticle
    cx0, cy0 = w // 2, h // 2
    cv2.line(tile, (cx0 - 10, cy0), (cx0 + 10, cy0), (220, 225, 230), 1, cv2.LINE_AA)
    cv2.line(tile, (cx0, cy0 - 10), (cx0, cy0 + 10), (220, 225, 230), 1, cv2.LINE_AA)

    # Target lock crosshair & error vector
    locked = bool(overlay.get("locked", False))
    if locked and overlay.get("cx") is not None and overlay.get("cy") is not None:
        u = int(round((float(overlay["cx"]) + 1.0) * 0.5 * (w - 1)))
        v = int(round((float(overlay["cy"]) + 1.0) * 0.5 * (h - 1)))
        u = max(4, min(w - 5, u))
        v = max(4, min(h - 5, v))
        cv2.line(tile, (cx0, cy0), (u, v), (80, 240, 255), 1, cv2.LINE_AA)
        cv2.circle(tile, (u, v), 11, (70, 255, 110), 2, cv2.LINE_AA)
        cv2.drawMarker(tile, (u, v), (70, 255, 110), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    # Top-right active badge
    badge = overlay.get("badge", "实时视觉伺服")
    _put_pill(tile, badge, 8, 8, bg_bgr=(35, 118, 185), fg_bgr=(255, 255, 255), font_size=13, align_right=True)

    # Bottom telemetry bar
    f_cmd = float(overlay.get("cmd_fwd", 0.0))
    r_cmd = float(overlay.get("cmd_right", 0.0))
    d_cmd = float(overlay.get("cmd_down", 0.0))
    if mode == "dino":
        dist_m = overlay.get("depth_m")
        peak = float(overlay.get("peak", 0.0))
        dist_txt = f"{dist_m:.1f}m" if dist_m is not None else "--"
        telem = f"测距 {dist_txt}｜峰值 {peak:.2f}｜前{f_cmd:+.1f} 右{r_cmd:+.1f} 下{d_cmd:+.1f}"
    else:
        err_m = overlay.get("err_horiz_m")
        clr_m = float(overlay.get("clearance_m", 0.0))
        err_txt = f"{err_m:.1f}m" if err_m is not None else "--"
        telem = f"偏移 {err_txt}｜净空 {clr_m:.1f}m｜前{f_cmd:+.1f} 右{r_cmd:+.1f} 下{d_cmd:+.1f}"
    _put_pill(tile, telem, 8, h - 30, bg_bgr=(24, 20, 18), fg_bgr=(238, 244, 250), font_size=12)

    # Confirmation progress bar (bottom-right)
    conf = int(overlay.get("confirm", 0))
    conf_tgt = max(1, int(overlay.get("confirm_target", 3)))
    phase_cn = str(overlay.get("phase_cn", "伺服中"))
    bar_w, bar_h = 84, 10
    bx1, by1 = w - 10, h - 38
    bx0, by0 = bx1 - bar_w, by1 - bar_h
    cv2.rectangle(tile, (bx0, by0), (bx1, by1), (36, 40, 48), -1)
    fill_w = int(round(bar_w * min(1.0, conf / conf_tgt)))
    if fill_w > 0:
        cv2.rectangle(tile, (bx0, by0), (bx0 + fill_w, by1), (80, 235, 130), -1)
    cv2.rectangle(tile, (bx0, by0), (bx1, by1), (180, 195, 210), 1)
    _put_pill(tile, f"{phase_cn}", 10, by0 - 24, bg_bgr=(28, 36, 46), fg_bgr=(160, 245, 180), font_size=12, align_right=True)

    # Active visual-servo border glow
    cv2.rectangle(tile, (1, 1), (w - 2, h - 2), (75, 215, 255), 2)
    return tile


class MultiUAVVideoRecorder:
    """Composites 3 active UAVs (Chase + Front) + 2D airport trajectory map + altitude profile."""

    def __init__(
        self,
        output_dir: Path | str,
        target_order: list[str],
        fps: int = 30,
        highlight_stride: int = 4,
        target_labels: dict[str, str] | None = None,
        target_positions_ned: dict[str, tuple[float, float]] | None = None,
        home_ned: tuple[float, float] = (0.0, 0.0),
        grid_npz: Path | str | None = DEFAULT_GLOBAL_GRID,
        encode_h264: bool = True,
    ) -> None:
        if len(target_order) != 3:
            raise ValueError(f"MultiUAVVideoRecorder expects 3 targets, got {len(target_order)}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_order = list(target_order)
        self.target_labels = {
            t: (target_labels[t] if target_labels and t in target_labels else format_target_cn(t))
            for t in self.target_order
        }
        self.fps = int(fps)
        self.highlight_stride = max(1, int(highlight_stride))
        self.encode_h264 = bool(encode_h264)

        self.realtime_path = self.output_dir / "mission_6uav_overview.mp4"
        self.highlight_path = self.output_dir / "mission_6uav_highlight_4x.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer_rt = cv2.VideoWriter(str(self.realtime_path), fourcc, self.fps, (CANVAS_W, CANVAS_H))
        self._writer_hl = cv2.VideoWriter(str(self.highlight_path), fourcc, self.fps, (CANVAS_W, CANVAS_H))
        if not self._writer_rt.isOpened() or not self._writer_hl.isOpened():
            raise RuntimeError("Failed to open OpenCV mp4v VideoWriter")

        e_min, e_max = -75.0, 615.0
        n_min, n_max = -150.0, 480.0
        if target_positions_ned:
            all_n = [home_ned[0]] + [p[0] for p in target_positions_ned.values()]
            all_e = [home_ned[1]] + [p[1] for p in target_positions_ned.values()]
            n_min = min(n_min, min(all_n) - 65.0)
            n_max = max(n_max, max(all_n) + 65.0)
            e_min = min(e_min, min(all_e) - 65.0)
            e_max = max(e_max, max(all_e) + 65.0)

        self._radar = _AirportRadarMap(
            width=MAP_PANEL_W,
            height=MAP_BOX_H,
            e_bounds=(e_min, e_max),
            n_bounds=(n_min, n_max),
            grid_npz=grid_npz,
            target_positions_ned=target_positions_ned,
            home_ned=home_ned,
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_mono = time.monotonic()

        self._phase_title = "多机两阶段协同勘测准备中"
        self._active_drones: dict[str, str] = {t: "-" for t in self.target_order}
        self._drone_status: dict[str, str] = {t: "待命" for t in self.target_order}
        self._drone_id_status: dict[str, str] = {d: "待命" for d in ALL_DRONE_ORDER}
        self._vlm_badge: dict[str, str] = {t: "" for t in self.target_order}
        self._servo_overlays: dict[str, dict[str, Any]] = {}
        self._front_listeners: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._recording_live = False

        # Live trajectories: drone_id -> list of (t_s, north, east, alt_m)
        self._trajectories: dict[str, list[tuple[float, float, float, float]]] = {}
        self._latest_telem: dict[str, tuple[float, float]] = {}  # drone_id -> (alt_m, speed_mps)

        self._latest_msg: dict[tuple[str, str], dict[str, Any]] = {}
        self._latest_ts: dict[tuple[str, str], int] = {}
        self._cached_tile: dict[tuple[str, str], np.ndarray] = {}
        self._cached_ts: dict[tuple[str, str], int] = {}
        self._rx_counts: dict[str, int] = {}
        self._unique_ts: dict[str, set[int]] = {}

        self._rt_frames = 0
        self._hl_frames = 0
        self._live_tick_idx = 0
        self._subscribed_topics: list[tuple[Any, str]] = []

    def set_phase(self, title: str, active_drones: dict[str, str] | None = None) -> None:
        with self._lock:
            self._phase_title = title
            if active_drones is not None:
                self._active_drones.update(active_drones)
                self._latest_msg.clear()
                self._latest_ts.clear()
                self._cached_tile.clear()
                self._cached_ts.clear()

    def set_status(self, target_id: str, status: str) -> None:
        with self._lock:
            self._drone_status[target_id] = status
            drone_id = self._active_drones.get(target_id)
            if drone_id and drone_id in self._drone_id_status:
                self._drone_id_status[drone_id] = status

    def set_vlm_badge(self, target_id: str, badge: str) -> None:
        with self._lock:
            self._vlm_badge[target_id] = badge

    def set_servo_overlay(self, target_id: str, overlay: dict[str, Any] | None) -> None:
        with self._lock:
            if overlay is None:
                self._servo_overlays.pop(target_id, None)
            else:
                self._servo_overlays[target_id] = dict(overlay)

    def set_front_listener(self, target_id: str, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """Attach/detach a live FrontCamera RGB frame listener without double-subscribing."""
        with self._lock:
            if callback is None:
                self._front_listeners.pop(target_id, None)
            else:
                self._front_listeners[target_id] = callback

    def update_pose(
        self,
        drone_id: str,
        north: float,
        east: float,
        down: float,
        speed_mps: float = 0.0,
        t_s: float | None = None,
    ) -> None:
        """Record a live drone pose sample for 2D trajectory map and altitude profile."""
        ts = float(t_s) if t_s is not None else (time.monotonic() - self._started_mono)
        alt_m = max(0.0, -float(down))
        with self._lock:
            self._latest_telem[drone_id] = (alt_m, float(speed_mps))
            pts = self._trajectories.setdefault(drone_id, [])
            if not pts or abs(north - pts[-1][1]) + abs(east - pts[-1][2]) + abs(alt_m - pts[-1][3]) >= 0.4 or (ts - pts[-1][0]) >= 1.0:
                pts.append((ts, float(north), float(east), alt_m))

    def ingest_frame_bgr(self, target_id: str, cam_role: str, bgr: np.ndarray, timestamp: int = 0) -> None:
        resized = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        key = (target_id, cam_role)
        stat_key = f"{target_id}/{cam_role}"
        with self._lock:
            self._cached_tile[key] = resized
            self._cached_ts[key] = int(timestamp)
            self._latest_ts[key] = int(timestamp)
            self._rx_counts[stat_key] = self._rx_counts.get(stat_key, 0) + 1
            self._unique_ts.setdefault(stat_key, set()).add(int(timestamp))

    def attach_batch_streams(
        self,
        client: Any,
        drones: dict[str, Any],
        target_to_drone: dict[str, str],
    ) -> None:
        self.detach_streams()
        self.set_phase(self._phase_title, active_drones=target_to_drone)

        def make_cb(tid: str, cam_role: str, stat_key: str):
            def _cb(_topic: str, msg: dict[str, Any]) -> None:
                ts = int(msg.get("time_stamp", 0))
                listener = None
                with self._lock:
                    self._latest_msg[(tid, cam_role)] = msg
                    self._latest_ts[(tid, cam_role)] = ts
                    self._rx_counts[stat_key] = self._rx_counts.get(stat_key, 0) + 1
                    self._unique_ts.setdefault(stat_key, set()).add(ts)
                    if cam_role == "front":
                        listener = self._front_listeners.get(tid)
                if listener is not None:
                    try:
                        listener(msg)
                    except Exception:
                        pass
            return _cb

        for tid in self.target_order:
            drone_id = target_to_drone.get(tid)
            if not drone_id or drone_id not in drones:
                continue
            drone = drones[drone_id]
            for cam_role, sensor_name in (("chase", "Chase"), ("front", "FrontCamera")):
                if sensor_name in drone.sensors and "scene_camera" in drone.sensors[sensor_name]:
                    topic = drone.sensors[sensor_name]["scene_camera"]
                    stat_key = f"{drone_id}/{sensor_name}/rgb"
                    client.subscribe(topic, make_cb(tid, cam_role, stat_key))
                    self._subscribed_topics.append((client, topic))

        with self._lock:
            self._recording_live = True

    def detach_streams(self) -> None:
        with self._lock:
            self._recording_live = False
        for client, topic in self._subscribed_topics:
            try:
                client.unsubscribe(topic)
            except Exception:
                pass
        self._subscribed_topics.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_mono = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="multi-uav-video-recorder", daemon=True)
        self._thread.start()

    def write_summary_card(
        self,
        title: str,
        lines: list[str],
        duration_s: float = 3.0,
        thumbnails: dict[str, Path] | None = None,
    ) -> None:
        with self._lock:
            prev_live = self._recording_live
            self._recording_live = False

        canvas = np.full((CANVAS_H, CANVAS_W, 3), (20, 24, 31), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), (30, 36, 48), -1)
        _put_pill(canvas, title, 18, 9, bg_bgr=(45, 78, 138), fg_bgr=(255, 255, 255), font_size=21)

        y_cursor = HEADER_H + 18
        for line in lines[:5]:
            _put_pill(canvas, line, 24, y_cursor, bg_bgr=(32, 38, 48), fg_bgr=(235, 240, 246), font_size=19)
            y_cursor += 42

        if thumbnails:
            thumb_top = max(y_cursor + 12, HEADER_H + 190)
            thumb_h = CANVAS_H - thumb_top - 20
            if thumb_h >= 120:
                col_w = CANVAS_W // 3
                for col, tid in enumerate(self.target_order):
                    x0 = col * col_w + 18
                    w_box = col_w - 36
                    p = thumbnails.get(tid)
                    if p and Path(p).is_file():
                        img = cv2.imread(str(p))
                        if img is not None:
                            resized = cv2.resize(img, (w_box, thumb_h), interpolation=cv2.INTER_AREA)
                            canvas[thumb_top : thumb_top + thumb_h, x0 : x0 + w_box] = resized
                    label = self.target_labels.get(tid, tid)
                    badge = self._vlm_badge.get(tid, "")
                    cap_txt = f"{label}｜{badge}" if badge else label
                    _put_pill(canvas, cap_txt, x0 + 8, thumb_top + 8, bg_bgr=(20, 24, 30), font_size=17)

        n_frames = max(1, int(round(duration_s * self.fps)))
        with self._lock:
            for _ in range(n_frames):
                self._writer_rt.write(canvas)
                self._writer_hl.write(canvas)
                self._rt_frames += 1
                self._hl_frames += 1
            self._recording_live = prev_live

    def _render_altitude_strip(
        self,
        trajectories: dict[str, list[tuple[float, float, float, float]]],
        active_drones: dict[str, str],
        elapsed_s: float,
    ) -> np.ndarray:
        strip_w = TILE_W * 3
        strip = np.full((ALT_H, strip_w, 3), (20, 24, 31), dtype=np.uint8)
        _put_pill(strip, "实时飞行高度剖面（75m / 90m / 105m 分层巡航与低空抵近）", 12, 6, bg_bgr=(30, 38, 52), font_size=14)

        px0, py0, px1, py1 = 54, 34, strip_w - 16, ALT_H - 12
        cv2.rectangle(strip, (px0, py0), (px1, py1), (26, 32, 42), -1)
        cv2.rectangle(strip, (px0, py0), (px1, py1), (52, 62, 78), 1)
        max_alt = 120.0
        t_max = max(60.0, elapsed_s)

        for alt_tick in (0, 30, 60, 90, 120):
            gy = int(round(py1 - (alt_tick / max_alt) * (py1 - py0)))
            cv2.line(strip, (px0, gy), (px1, gy), (38, 46, 58), 1)
            cv2.putText(strip, f"{alt_tick:3d}m", (6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

        active_set = {d for d in active_drones.values() if d and d != "-"}
        for drone_id in ALL_DRONE_ORDER:
            pts_raw = trajectories.get(drone_id)
            if not pts_raw or len(pts_raw) < 2:
                continue
            color = DRONE_COLORS_BGR.get(drone_id, (200, 200, 200))
            is_act = drone_id in active_set
            draw_col = color if is_act else tuple(int(c * 0.45) for c in color)
            pts = np.array(
                [
                    (
                        int(round(px0 + min(1.0, max(0.0, t / t_max)) * (px1 - px0))),
                        int(round(py1 - min(1.0, max(0.0, alt / max_alt)) * (py1 - py0))),
                    )
                    for t, _, _, alt in pts_raw
                ],
                dtype=np.int32,
            )
            cv2.polylines(strip, [pts], False, draw_col, 2 if is_act else 1, cv2.LINE_AA)
            if is_act:
                cv2.circle(strip, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)
        return strip

    def _render_telemetry_panel(
        self,
        latest_telem: dict[str, tuple[float, float]],
        drone_id_status: dict[str, str],
        active_set: set[str],
    ) -> np.ndarray:
        panel = np.full((TELEM_BOX_H, MAP_PANEL_W, 3), (20, 24, 31), dtype=np.uint8)
        cv2.line(panel, (0, 0), (MAP_PANEL_W, 0), (54, 64, 80), 1)
        _put_pill(panel, "六机轨迹图例与实时遥测", 10, 6, bg_bgr=(30, 38, 52), font_size=14)

        y = 36
        for drone_id in ALL_DRONE_ORDER:
            color = DRONE_COLORS_BGR.get(drone_id, (200, 200, 200))
            is_act = drone_id in active_set
            alt, spd = latest_telem.get(drone_id, (0.0, 0.0))
            st = drone_id_status.get(drone_id, "待命")
            if len(st) > 12:
                st = st[:12] + "…"
            row_bg = (30, 38, 50) if is_act else (22, 26, 33)
            cv2.rectangle(panel, (10, y), (MAP_PANEL_W - 10, y + 25), row_bg, -1)
            cv2.rectangle(panel, (14, y + 6), (26, y + 19), color, -1)
            txt = f"{format_drone_cn(drone_id)}  高{alt:5.1f}m  速{spd:4.1f}m/s  {st}"
            fg = (242, 246, 252) if is_act else (155, 165, 180)
            _put_pill(panel, txt, 30, y + 1, bg_bgr=row_bg, fg_bgr=fg, font_size=14)
            y += 28
        return panel

    def _refresh_decoded_tiles(self) -> None:
        to_decode: list[tuple[tuple[str, str], dict[str, Any], int]] = []
        with self._lock:
            for key, msg in list(self._latest_msg.items()):
                ts = self._latest_ts.get(key, -1)
                if ts != self._cached_ts.get(key, -2):
                    to_decode.append((key, msg, ts))
        for key, msg, ts in to_decode:
            try:
                rgb, _ = decode_image(msg)
                bgr = rgb[:, :, ::-1]
                resized = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._cached_tile[key] = resized
                    self._cached_ts[key] = ts
            except Exception:
                pass

    def _compose_frame(self) -> np.ndarray:
        self._refresh_decoded_tiles()
        canvas = np.full((CANVAS_H, CANVAS_W, 3), (18, 22, 28), dtype=np.uint8)
        elapsed = time.monotonic() - self._started_mono

        with self._lock:
            phase = self._phase_title
            active = dict(self._active_drones)
            statuses = dict(self._drone_status)
            drone_id_statuses = dict(self._drone_id_status)
            badges = dict(self._vlm_badge)
            servo_overlays = dict(self._servo_overlays)
            tiles = dict(self._cached_tile)
            trajs = {k: list(v) for k, v in self._trajectories.items()}
            telem = dict(self._latest_telem)

        # Clean cool-slate header/chrome BGR colors (B > R)
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), (44, 34, 28), -1)
        _put_pill(canvas, phase, 14, 9, bg_bgr=(148, 88, 42), fg_bgr=(255, 255, 255), font_size=20)
        _put_pill(
            canvas,
            f"计时 {elapsed:5.1f} 秒｜上排：追踪视角　下排：机载前视｜右侧：六机轨迹",
            14,
            11,
            bg_bgr=(56, 44, 36),
            fg_bgr=(240, 232, 224),
            font_size=16,
            align_right=True,
        )

        # Left 3x2 live camera tiles
        for col, tid in enumerate(self.target_order):
            x0 = col * TILE_W
            drone_id = active.get(tid, "-")
            drone_cn = format_drone_cn(drone_id)
            target_cn = self.target_labels.get(tid, tid)
            st = statuses.get(tid, "")
            badge = badges.get(tid, "")

            for row_idx, (cam_role, cam_label) in enumerate((("chase", "追踪视角"), ("front", "机载前视"))):
                y0 = HEADER_H + row_idx * TILE_H
                tile = tiles.get((tid, cam_role))
                if tile is None:
                    tile = _blank_tile(f"{drone_cn}｜{target_cn}｜{cam_label}等待画面...")
                else:
                    tile = tile.copy()

                if row_idx == 1 and tid in servo_overlays:
                    tile = _draw_servo_overlay(tile, servo_overlays[tid])

                _put_pill(tile, f"{drone_cn}｜{target_cn}｜{cam_label}", 8, 8, bg_bgr=(32, 24, 20), font_size=14)
                if row_idx == 0 and st:
                    _put_pill(tile, st, 8, TILE_H - 34, bg_bgr=(96, 62, 34), font_size=14)
                if row_idx == 0 and badge:
                    color = (64, 128, 42) if ("确认受损" in badge or "疑似受损" in badge) else (142, 86, 44)
                    _put_pill(tile, badge, 8, 8, bg_bgr=color, font_size=14, align_right=True)

                canvas[y0 : y0 + TILE_H, x0 : x0 + TILE_W] = tile
                cv2.rectangle(canvas, (x0, y0), (x0 + TILE_W - 1, y0 + TILE_H - 1), (48, 56, 68), 1)

        # Left bottom: live altitude profile
        alt_strip = self._render_altitude_strip(trajs, active, elapsed)
        canvas[HEADER_H + TILE_H * 2 : CANVAS_H, 0 : TILE_W * 3] = alt_strip

        # Right panel: 2D airport trajectory map + 6-UAV telemetry table
        active_set = {d for d in active.values() if d and d != "-"}
        radar_img = self._radar.render(trajs, active_set)
        telem_img = self._render_telemetry_panel(telem, drone_id_statuses, active_set)
        rx0 = TILE_W * 3
        canvas[HEADER_H : HEADER_H + MAP_BOX_H, rx0:CANVAS_W] = radar_img
        canvas[HEADER_H + MAP_BOX_H : CANVAS_H, rx0:CANVAS_W] = telem_img
        cv2.line(canvas, (rx0, HEADER_H), (rx0, CANVAS_H), (62, 74, 92), 1)

        return canvas

    def _loop(self) -> None:
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t = max(next_t + period, now)
            with self._lock:
                active_live = self._recording_live
            if not active_live:
                continue
            frame = self._compose_frame()
            with self._lock:
                self._writer_rt.write(frame)
                self._rt_frames += 1
                if self._live_tick_idx % self.highlight_stride == 0:
                    self._writer_hl.write(frame)
                    self._hl_frames += 1
                self._live_tick_idx += 1

    def stop(self) -> dict[str, Any]:
        self.detach_streams()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self._writer_rt.release()
            self._writer_hl.release()
            stream_summary = {
                k: {
                    "received_frames": self._rx_counts.get(k, 0),
                    "unique_timestamps": len(self._unique_ts.get(k, set())),
                }
                for k in sorted(self._rx_counts.keys())
            }
            rt_frames = self._rt_frames
            hl_frames = self._hl_frames

        h264_rt = _transcode_h264_inplace(self.realtime_path) if self.encode_h264 else False
        h264_hl = _transcode_h264_inplace(self.highlight_path) if self.encode_h264 else False
        return {
            "realtime_video": str(self.realtime_path),
            "highlight_video_4x": str(self.highlight_path),
            "h264_encoded": {"realtime": h264_rt, "highlight_4x": h264_hl},
            "resolution": [CANVAS_W, CANVAS_H],
            "output_fps": self.fps,
            "realtime_frames": rt_frames,
            "realtime_duration_s": round(rt_frames / self.fps, 2),
            "highlight_frames": hl_frames,
            "highlight_duration_s": round(hl_frames / self.fps, 2),
            "stream_counts": stream_summary,
        }
