"""SQLite persistence for centralized two-stage multi-UAV cooperative survey."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS missions (
    mission_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    vlm_backend TEXT NOT NULL,
    vlm_model TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS targets (
    mission_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL,
    round1_task_id TEXT NOT NULL,
    round1_drone_id TEXT NOT NULL,
    round1_return_status TEXT NOT NULL DEFAULT 'assigned',
    round1_returned_at TEXT,
    round1_return_note TEXT NOT NULL DEFAULT '',
    round2_task_id TEXT,
    round2_drone_id TEXT,
    round2_return_status TEXT NOT NULL DEFAULT 'not_assigned',
    round2_returned_at TEXT,
    round2_return_note TEXT NOT NULL DEFAULT '',
    ground_truth_label TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (mission_id, target_id),
    FOREIGN KEY (mission_id) REFERENCES missions (mission_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK (stage IN (1, 2)),
    task_id TEXT NOT NULL,
    drone_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    view_type TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    received_at TEXT NOT NULL,
    UNIQUE (mission_id, target_id, stage, image_id),
    FOREIGN KEY (mission_id, target_id) REFERENCES targets (mission_id, target_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recheck_tasks (
    task_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    round1_image_ids_json TEXT NOT NULL,
    round1_image_paths_json TEXT NOT NULL,
    question TEXT NOT NULL,
    focus_region TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    assigned_drone_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dispatched_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (mission_id, target_id),
    FOREIGN KEY (mission_id, target_id) REFERENCES targets (mission_id, target_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK (stage IN (1, 2)),
    status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    model_name TEXT NOT NULL,
    latency_ms REAL,
    result_json TEXT,
    raw_response TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (mission_id, target_id, stage),
    FOREIGN KEY (mission_id, target_id) REFERENCES targets (mission_id, target_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    target_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (mission_id) REFERENCES missions (mission_id) ON DELETE CASCADE
);
"""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


class SurveyStore:
    """Thread-safe-enough single-process SQLite store for survey missions."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def create_mission(
        self,
        mission_id: str,
        vlm_backend: str,
        vlm_model: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO missions (mission_id, created_at, status, vlm_backend, vlm_model, config_json)
            VALUES (?, ?, 'active', ?, ?, ?)
            """,
            (mission_id, now, vlm_backend, vlm_model, json.dumps(config or {}, ensure_ascii=False)),
        )
        self.log_event(mission_id, None, "mission_created", {"vlm_backend": vlm_backend, "vlm_model": vlm_model})
        self.conn.commit()
        return self.get_mission(mission_id)

    def get_mission(self, mission_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM missions WHERE mission_id = ?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown mission_id: {mission_id}")
        data = dict(row)
        data["config"] = json.loads(data.pop("config_json"))
        return data

    def update_mission_status(self, mission_id: str, status: str) -> None:
        self.conn.execute("UPDATE missions SET status = ? WHERE mission_id = ?", (status, mission_id))
        self.conn.commit()

    def register_target(
        self,
        mission_id: str,
        target_id: str,
        round1_task_id: str,
        round1_drone_id: str,
        ground_truth_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO targets (
                mission_id, target_id, status, round1_task_id, round1_drone_id,
                round1_return_status, round2_return_status, ground_truth_label,
                metadata_json, updated_at
            ) VALUES (?, ?, 'pending_round1', ?, ?, 'assigned', 'not_assigned', ?, ?, ?)
            """,
            (
                mission_id,
                target_id,
                round1_task_id,
                round1_drone_id,
                ground_truth_label,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
            ),
        )
        self.log_event(
            mission_id,
            target_id,
            "target_registered",
            {"round1_task_id": round1_task_id, "round1_drone_id": round1_drone_id},
        )
        self.conn.commit()
        return self.get_target(mission_id, target_id)

    def get_target(self, mission_id: str, target_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM targets WHERE mission_id = ? AND target_id = ?",
            (mission_id, target_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown target ({mission_id}, {target_id})")
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json"))
        return data

    def list_targets(self, mission_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM targets WHERE mission_id = ? ORDER BY target_id ASC",
            (mission_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

    def update_target_status(self, mission_id: str, target_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE targets SET status = ?, updated_at = ? WHERE mission_id = ? AND target_id = ?",
            (status, utc_now_iso(), mission_id, target_id),
        )
        self.conn.commit()

    def record_upload(
        self,
        mission_id: str,
        target_id: str,
        stage: int,
        task_id: str,
        drone_id: str,
        image_id: str,
        captured_at: str,
        view_type: str,
        stored_path: str,
        sha256: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Insert an upload idempotently. Returns (record, is_duplicate)."""
        existing = self.conn.execute(
            """
            SELECT * FROM uploads
            WHERE mission_id = ? AND target_id = ? AND stage = ? AND image_id = ?
            """,
            (mission_id, target_id, stage, image_id),
        ).fetchone()
        if existing is not None:
            self.log_event(
                mission_id,
                target_id,
                "duplicate_upload_ignored",
                {"stage": stage, "image_id": image_id, "drone_id": drone_id, "task_id": task_id},
            )
            self.conn.commit()
            data = dict(existing)
            data["metadata"] = json.loads(data.pop("metadata_json"))
            return data, True

        now = utc_now_iso()
        cur = self.conn.execute(
            """
            INSERT INTO uploads (
                mission_id, target_id, stage, task_id, drone_id, image_id,
                captured_at, view_type, stored_path, sha256, metadata_json, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mission_id,
                target_id,
                stage,
                task_id,
                drone_id,
                image_id,
                captured_at,
                view_type,
                stored_path,
                sha256,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
            ),
        )
        self.log_event(
            mission_id,
            target_id,
            "upload_recorded",
            {
                "stage": stage,
                "task_id": task_id,
                "drone_id": drone_id,
                "image_id": image_id,
                "view_type": view_type,
                "stored_path": stored_path,
                "sha256": sha256,
            },
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM uploads WHERE id = ?", (cur.lastrowid,)).fetchone()
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json"))
        return data, False

    def list_uploads(self, mission_id: str, target_id: str, stage: int | None = None) -> list[dict[str, Any]]:
        if stage is None:
            rows = self.conn.execute(
                "SELECT * FROM uploads WHERE mission_id = ? AND target_id = ? ORDER BY stage ASC, captured_at ASC, id ASC",
                (mission_id, target_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM uploads WHERE mission_id = ? AND target_id = ? AND stage = ? ORDER BY captured_at ASC, id ASC",
                (mission_id, target_id, stage),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

    def record_return_status(
        self,
        mission_id: str,
        target_id: str,
        stage: int,
        drone_id: str,
        status: str,
        returned_at: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        if stage not in (1, 2):
            raise ValueError(f"Invalid stage: {stage}")
        ts = returned_at or utc_now_iso()
        now = utc_now_iso()
        if stage == 1:
            self.conn.execute(
                """
                UPDATE targets
                SET round1_drone_id = ?, round1_return_status = ?, round1_returned_at = ?,
                    round1_return_note = ?, updated_at = ?
                WHERE mission_id = ? AND target_id = ?
                """,
                (drone_id, status, ts, note, now, mission_id, target_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE targets
                SET round2_drone_id = COALESCE(round2_drone_id, ?),
                    round2_return_status = ?, round2_returned_at = ?,
                    round2_return_note = ?, updated_at = ?
                WHERE mission_id = ? AND target_id = ?
                """,
                (drone_id, status, ts, note, now, mission_id, target_id),
            )
        self.log_event(
            mission_id,
            target_id,
            "drone_return_reported",
            {"stage": stage, "drone_id": drone_id, "status": status, "returned_at": ts, "note": note},
        )
        self.conn.commit()
        return self.get_target(mission_id, target_id)

    def create_recheck_task_once(
        self,
        task_id: str,
        mission_id: str,
        target_id: str,
        round1_image_ids: list[str],
        round1_image_paths: list[str],
        question: str,
        focus_region: str,
        recommended_action: str,
    ) -> tuple[dict[str, Any], bool]:
        """Create a target's recheck task at most once. Returns (task, created_new)."""
        existing = self.get_recheck_task(mission_id, target_id)
        if existing is not None:
            self.log_event(
                mission_id,
                target_id,
                "recheck_task_already_exists",
                {"task_id": existing["task_id"]},
            )
            self.conn.commit()
            return existing, False

        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO recheck_tasks (
                task_id, mission_id, target_id, round1_image_ids_json, round1_image_paths_json,
                question, focus_region, recommended_action, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'created', ?, ?)
            """,
            (
                task_id,
                mission_id,
                target_id,
                json.dumps(round1_image_ids, ensure_ascii=False),
                json.dumps(round1_image_paths, ensure_ascii=False),
                question,
                focus_region,
                recommended_action,
                now,
                now,
            ),
        )
        self.conn.execute(
            """
            UPDATE targets
            SET round2_task_id = ?, updated_at = ?
            WHERE mission_id = ? AND target_id = ?
            """,
            (task_id, now, mission_id, target_id),
        )
        self.log_event(
            mission_id,
            target_id,
            "recheck_task_created",
            {
                "task_id": task_id,
                "round1_image_ids": round1_image_ids,
                "question": question,
                "focus_region": focus_region,
                "recommended_action": recommended_action,
            },
        )
        self.conn.commit()
        task = self.get_recheck_task(mission_id, target_id)
        assert task is not None
        return task, True

    def get_recheck_task(self, mission_id: str, target_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM recheck_tasks WHERE mission_id = ? AND target_id = ?",
            (mission_id, target_id),
        ).fetchone()
        data = _row_to_dict(row)
        if data is None:
            return None
        data["round1_image_ids"] = json.loads(data.pop("round1_image_ids_json"))
        data["round1_image_paths"] = json.loads(data.pop("round1_image_paths_json"))
        return data

    def dispatch_recheck_task(self, mission_id: str, target_id: str, drone_id: str) -> dict[str, Any]:
        task = self.get_recheck_task(mission_id, target_id)
        if task is None:
            raise KeyError(f"No recheck task exists for ({mission_id}, {target_id})")
        now = utc_now_iso()
        new_status = task["status"] if task["status"] in ("image_received", "completed") else "dispatched"
        self.conn.execute(
            """
            UPDATE recheck_tasks
            SET assigned_drone_id = ?, status = ?, dispatched_at = COALESCE(dispatched_at, ?), updated_at = ?
            WHERE mission_id = ? AND target_id = ?
            """,
            (drone_id, new_status, now, now, mission_id, target_id),
        )
        self.conn.execute(
            """
            UPDATE targets
            SET round2_drone_id = ?,
                round2_return_status = CASE
                    WHEN round2_return_status IN ('not_assigned', 'not_needed') THEN 'assigned'
                    ELSE round2_return_status
                END,
                updated_at = ?
            WHERE mission_id = ? AND target_id = ?
            """,
            (drone_id, now, mission_id, target_id),
        )
        self.log_event(
            mission_id,
            target_id,
            "recheck_task_dispatched",
            {"task_id": task["task_id"], "drone_id": drone_id},
        )
        self.conn.commit()
        updated = self.get_recheck_task(mission_id, target_id)
        assert updated is not None
        return updated

    def update_recheck_task_status(self, mission_id: str, target_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE recheck_tasks SET status = ?, updated_at = ? WHERE mission_id = ? AND target_id = ?",
            (status, utc_now_iso(), mission_id, target_id),
        )
        self.conn.commit()

    def save_assessment(
        self,
        mission_id: str,
        target_id: str,
        stage: int,
        status: str,
        model_name: str,
        latency_ms: float | None,
        result: dict[str, Any] | None = None,
        raw_response: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO assessments (
                mission_id, target_id, stage, status, model_name, latency_ms,
                result_json, raw_response, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mission_id, target_id, stage) DO UPDATE SET
                status = excluded.status,
                model_name = excluded.model_name,
                latency_ms = excluded.latency_ms,
                result_json = excluded.result_json,
                raw_response = excluded.raw_response,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                mission_id,
                target_id,
                stage,
                status,
                model_name,
                latency_ms,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                raw_response,
                error_message,
                now,
                now,
            ),
        )
        self.log_event(
            mission_id,
            target_id,
            f"stage{stage}_assessment_{status}",
            {"model_name": model_name, "latency_ms": latency_ms, "error_message": error_message},
        )
        self.conn.commit()
        record = self.get_assessment(mission_id, target_id, stage)
        assert record is not None
        return record

    def get_assessment(self, mission_id: str, target_id: str, stage: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM assessments WHERE mission_id = ? AND target_id = ? AND stage = ?",
            (mission_id, target_id, stage),
        ).fetchone()
        data = _row_to_dict(row)
        if data is None:
            return None
        data["result"] = json.loads(data.pop("result_json")) if data["result_json"] is not None else None
        return data

    def log_event(
        self,
        mission_id: str,
        target_id: str | None,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO events (mission_id, target_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (mission_id, target_id, event_type, json.dumps(payload or {}, ensure_ascii=False), utc_now_iso()),
        )

    def list_events(self, mission_id: str, target_id: str | None = None) -> list[dict[str, Any]]:
        if target_id is None:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE mission_id = ? ORDER BY id ASC",
                (mission_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE mission_id = ? AND target_id = ? ORDER BY id ASC",
                (mission_id, target_id),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            results.append(item)
        return results
