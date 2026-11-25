"""Lightweight single-UAV Chase + FrontCamera MP4 recorder for Coastal Harbor missions.

Compatible with `coordination.visual_servo.run_dino_visual_servo` and
`run_opening_visual_servo` via `set_front_listener`, `set_status`, and
`set_servo_overlay` without double-subscribing `FrontCamera/scene_camera`.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from common.sensors import decode_image
from coordination.video_recorder import (
    _draw_servo_overlay,
    _put_pill,
    _transcode_h264_inplace,
)

TILE_W = 640
TILE_H = 360
HEADER_H = 48
FOOTER_H = 8
CANVAS_W = TILE_W * 2  # 1280
CANVAS_H = HEADER_H + TILE_H + FOOTER_H  # 416 (divisible by 16 for H.264)


def _blank_panel(label: str) -> np.ndarray:
    tile = np.full((TILE_H, TILE_W, 3), (24, 28, 34), dtype=np.uint8)
    _put_pill(tile, label, 20, TILE_H // 2 - 14, bg_bgr=(40, 46, 56), fg_bgr=(195, 202, 214), font_size=16)
    return tile


class HarborVideoRecorder:
    """Real-time 1280x416 dual-view (Chase + FrontCamera) MP4 recorder for one UAV."""

    def __init__(
        self,
        output_path: Path | str,
        *,
        title: str = "沿海渔港巡检录像",
        fps: int = 20,
        encode_h264: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.fps = int(fps)
        self.encode_h264 = bool(encode_h264)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (CANVAS_W, CANVAS_H))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.output_path}")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_mono = time.monotonic()

        self._status = "准备中"
        self._badge = ""
        self._servo_overlay: dict[str, Any] | None = None
        self._line_overlay: dict[str, Any] | None = None
        self._front_listeners: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._latest_pos: tuple[float, float, float] | None = None

        self._latest_msg: dict[str, dict[str, Any]] = {}
        self._latest_ts: dict[str, int] = {}
        self._cached_tile: dict[str, np.ndarray] = {}
        self._cached_ts: dict[str, int] = {}
        self._rx_counts: dict[str, int] = {"chase": 0, "front": 0}
        self._frames_written = 0
        self._recording_live = False
        self._subscribed_topics: list[tuple[Any, str]] = []

    def set_title(self, title: str) -> None:
        with self._lock:
            self.title = title

    def set_status(self, _target_id: str, status: str) -> None:
        with self._lock:
            self._status = status

    def set_badge(self, badge: str) -> None:
        with self._lock:
            self._badge = badge

    def set_servo_overlay(self, _target_id: str, overlay: dict[str, Any] | None) -> None:
        with self._lock:
            self._servo_overlay = dict(overlay) if overlay is not None else None

    def set_line_overlay(self, overlay: dict[str, Any] | None) -> None:
        with self._lock:
            self._line_overlay = dict(overlay) if overlay is not None else None

    def set_front_listener(self, target_id: str, callback: Callable[[dict[str, Any]], None] | None) -> None:
        with self._lock:
            if callback is None:
                self._front_listeners.pop(target_id, None)
            else:
                self._front_listeners[target_id] = callback

    def ingest_frame_bgr(self, cam_role: str, bgr: np.ndarray, timestamp: int = 0) -> None:
        resized = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._cached_tile[cam_role] = resized
            self._cached_ts[cam_role] = int(timestamp)
            self._latest_ts[cam_role] = int(timestamp)
            self._rx_counts[cam_role] = self._rx_counts.get(cam_role, 0) + 1

    def attach_streams(self, client: Any, drone: Any) -> None:
        self.detach_streams()

        def make_cb(cam_role: str):
            def _cb(_topic: str, msg: dict[str, Any]) -> None:
                ts = int(msg.get("time_stamp", 0))
                listeners: list[Callable[[dict[str, Any]], None]] = []
                with self._lock:
                    self._latest_msg[cam_role] = msg
                    self._latest_ts[cam_role] = ts
                    self._rx_counts[cam_role] = self._rx_counts.get(cam_role, 0) + 1
                    if cam_role == "front":
                        if "pos_x" in msg and "pos_y" in msg and "pos_z" in msg:
                            self._latest_pos = (
                                float(msg["pos_x"]),
                                float(msg["pos_y"]),
                                float(msg["pos_z"]),
                            )
                        listeners = list(self._front_listeners.values())
                for fn in listeners:
                    try:
                        fn(msg)
                    except Exception:
                        pass

            return _cb

        for cam_role, sensor_name in (("chase", "Chase"), ("front", "FrontCamera")):
            if sensor_name in drone.sensors and "scene_camera" in drone.sensors[sensor_name]:
                topic = drone.sensors[sensor_name]["scene_camera"]
                client.subscribe(topic, make_cb(cam_role))
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
        self._thread = threading.Thread(target=self._loop, name="harbor-video-recorder", daemon=True)
        self._thread.start()

    def _refresh_tiles(self) -> None:
        to_decode: list[tuple[str, dict[str, Any], int]] = []
        with self._lock:
            for role, msg in list(self._latest_msg.items()):
                ts = self._latest_ts.get(role, -1)
                if ts != self._cached_ts.get(role, -2):
                    to_decode.append((role, msg, ts))
        for role, msg, ts in to_decode:
            try:
                rgb, _ = decode_image(msg)
                bgr = rgb[:, :, ::-1]
                resized = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._cached_tile[role] = resized
                    self._cached_ts[role] = ts
            except Exception:
                pass

    def _compose_frame(self) -> np.ndarray:
        self._refresh_tiles()
        with self._lock:
            title = self.title
            status = self._status
            badge = self._badge
            servo_ov = dict(self._servo_overlay) if self._servo_overlay is not None else None
            line_ov = dict(self._line_overlay) if self._line_overlay is not None else None
            pos = self._latest_pos
            chase_tile = self._cached_tile.get("chase")
            front_tile = self._cached_tile.get("front")
            chase_tile = chase_tile.copy() if chase_tile is not None else _blank_panel("等待 Chase 第三人称流...")
            front_tile = front_tile.copy() if front_tile is not None else _blank_panel("等待 FrontCamera 机载流...")

        if servo_ov is not None:
            front_tile = _draw_servo_overlay(front_tile, servo_ov)
        elif line_ov is not None:
            if line_ov.get("has_roof_damage"):
                u0 = int(round(float(line_ov.get("damage_u_start", 0.0)) * TILE_W))
                u1 = int(round(float(line_ov.get("damage_u_end", 0.0)) * TILE_W))
                cv2.rectangle(front_tile, (max(2, u0), 36), (min(TILE_W - 2, u1), TILE_H - 12), (40, 80, 255), 3)
                _put_pill(
                    front_tile,
                    f"检测到屋面破损 [{line_ov.get('damage_u_start', 0):.2f}, {line_ov.get('damage_u_end', 0):.2f}]",
                    8,
                    TILE_H - 34,
                    bg_bgr=(35, 35, 165),
                    fg_bgr=(255, 255, 255),
                    font_size=13,
                )
            else:
                _put_pill(
                    front_tile,
                    "屋面深度平稳（完好）",
                    8,
                    TILE_H - 34,
                    bg_bgr=(26, 95, 52),
                    fg_bgr=(240, 255, 245),
                    font_size=13,
                )

        _put_pill(chase_tile, "第三人称跟踪视角 (Chase)", 8, 8, bg_bgr=(20, 24, 32), font_size=14)
        _put_pill(front_tile, "机载主视角 (FrontCamera)", 8, 8, bg_bgr=(20, 24, 32), font_size=14)

        canvas = np.full((CANVAS_H, CANVAS_W, 3), (18, 22, 28), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), (26, 32, 42), -1)

        elapsed = time.monotonic() - self._started_mono
        _put_pill(canvas, title, 12, 9, bg_bgr=(38, 72, 128), fg_bgr=(255, 255, 255), font_size=17)
        if status:
            _put_pill(canvas, status, 430, 10, bg_bgr=(34, 42, 54), fg_bgr=(230, 242, 255), font_size=15)
        pos_txt = f"高度 {-pos[2]:.1f}m" if pos is not None else "高度 --"
        right_txt = f"{badge} ｜ {pos_txt} ｜ {elapsed:05.1f}s" if badge else f"{pos_txt} ｜ {elapsed:05.1f}s"
        _put_pill(canvas, right_txt, 12, 10, bg_bgr=(24, 30, 40), fg_bgr=(180, 225, 255), font_size=15, align_right=True)

        canvas[HEADER_H : HEADER_H + TILE_H, 0:TILE_W] = chase_tile
        canvas[HEADER_H : HEADER_H + TILE_H, TILE_W:CANVAS_W] = front_tile
        cv2.line(canvas, (TILE_W, HEADER_H), (TILE_W, HEADER_H + TILE_H), (60, 72, 88), 2)
        return canvas

    def _loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                live = self._recording_live
            if live:
                frame = self._compose_frame()
                with self._lock:
                    if self._writer.isOpened():
                        self._writer.write(frame)
                        self._frames_written += 1
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
            else:
                next_tick = time.monotonic()

    def stop(self) -> dict[str, Any]:
        self.detach_streams()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
            self._thread = None
        with self._lock:
            self._writer.release()
            frames = self._frames_written
            rx = dict(self._rx_counts)
        transcoded = _transcode_h264_inplace(self.output_path) if (self.encode_h264 and frames > 0) else False
        return {
            "path": str(self.output_path),
            "frames": frames,
            "duration_s": round(frames / max(1, self.fps), 2),
            "fps": self.fps,
            "rx_counts": rx,
            "h264_transcoded": transcoded,
        }
