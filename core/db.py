"""
SQLite persistence layer for VRAM Forecaster.

Design notes:
- models_json stores [{name, quant_type, size_mb}] — normalising into a
  separate table would add JOIN complexity for zero benefit at this scale.
- node_pack_versions_json stores {pack_name: version} for future data
  versioning when discarding stale community rows.
- Path/prompt data is intentionally absent — it would block future
  anonymous export.
- schema_version lets us migrate rows without dropping the table.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "forecaster.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Timing
    started_at              TEXT    NOT NULL,
    ended_at                TEXT,
    duration_seconds        REAL,

    -- Hardware / software environment
    gpu_model               TEXT    NOT NULL,
    gpu_vram_total_mb       INTEGER NOT NULL,
    gpu_driver_version      TEXT,
    cuda_version            TEXT,
    pytorch_version         TEXT,
    comfyui_version         TEXT,

    -- Runtime state that affects memory layout
    attention_backend       TEXT    NOT NULL,   -- vanilla / flash / sage / xformers

    -- Workflow geometry
    resolution_w            INTEGER,
    resolution_h            INTEGER,
    frame_count             INTEGER,
    loop_iteration_count    INTEGER,            -- easy forLoop block count
    block_swap_count        INTEGER,            -- WAN block-swap setting

    -- Loaded models (JSON array: [{name, quant_type, size_mb}])
    models_json             TEXT,

    -- LoRA
    lora_count              INTEGER DEFAULT 0,
    lora_total_weight_mb    REAL,

    -- Measured peaks
    vram_peak_mb            REAL,
    ram_peak_mb             REAL,

    -- Run outcome
    outcome                 TEXT    NOT NULL DEFAULT 'running',
    error_message           TEXT,

    -- Versioning for future community export / staleness filtering
    schema_version          INTEGER NOT NULL DEFAULT 1,
    node_pack_versions_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_gpu   ON runs (gpu_model);
CREATE INDEX IF NOT EXISTS idx_runs_start ON runs (started_at);
CREATE INDEX IF NOT EXISTS idx_runs_outcome ON runs (outcome);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    name: str
    quant_type: str          # fp16 / fp8 / gguf / bnb4 / …
    size_mb: float = 0.0


@dataclass
class RunRecord:
    """Represents one complete run entry.  id=None until persisted."""

    # Timing
    started_at: str = field(default_factory=lambda: _now_iso())
    ended_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Hardware / software
    gpu_model: str = ""
    gpu_vram_total_mb: int = 0
    gpu_driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    comfyui_version: Optional[str] = None

    # Runtime state
    attention_backend: str = "unknown"

    # Workflow geometry
    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    frame_count: Optional[int] = None
    loop_iteration_count: Optional[int] = None
    block_swap_count: Optional[int] = None

    # Models
    models: list[ModelInfo] = field(default_factory=list)

    # LoRA
    lora_count: int = 0
    lora_total_weight_mb: Optional[float] = None

    # Measured peaks (filled in by logger_end)
    vram_peak_mb: Optional[float] = None
    ram_peak_mb: Optional[float] = None

    # Outcome
    outcome: str = "running"   # running / ok / oom / error
    error_message: Optional[str] = None

    # Versioning
    schema_version: int = SCHEMA_VERSION
    node_pack_versions: dict[str, str] = field(default_factory=dict)

    # DB primary key — not set until after insert
    id: Optional[int] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_row(r: RunRecord) -> dict[str, Any]:
    return {
        "started_at":              r.started_at,
        "ended_at":                r.ended_at,
        "duration_seconds":        r.duration_seconds,
        "gpu_model":               r.gpu_model,
        "gpu_vram_total_mb":       r.gpu_vram_total_mb,
        "gpu_driver_version":      r.gpu_driver_version,
        "cuda_version":            r.cuda_version,
        "pytorch_version":         r.pytorch_version,
        "comfyui_version":         r.comfyui_version,
        "attention_backend":       r.attention_backend,
        "resolution_w":            r.resolution_w,
        "resolution_h":            r.resolution_h,
        "frame_count":             r.frame_count,
        "loop_iteration_count":    r.loop_iteration_count,
        "block_swap_count":        r.block_swap_count,
        "models_json":             json.dumps([asdict(m) for m in r.models]),
        "lora_count":              r.lora_count,
        "lora_total_weight_mb":    r.lora_total_weight_mb,
        "vram_peak_mb":            r.vram_peak_mb,
        "ram_peak_mb":             r.ram_peak_mb,
        "outcome":                 r.outcome,
        "error_message":           r.error_message,
        "schema_version":          r.schema_version,
        "node_pack_versions_json": json.dumps(r.node_pack_versions),
    }


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    models_raw = row["models_json"]
    models = [ModelInfo(**m) for m in json.loads(models_raw)] if models_raw else []

    packs_raw = row["node_pack_versions_json"]
    packs = json.loads(packs_raw) if packs_raw else {}

    return RunRecord(
        id=row["id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        duration_seconds=row["duration_seconds"],
        gpu_model=row["gpu_model"],
        gpu_vram_total_mb=row["gpu_vram_total_mb"],
        gpu_driver_version=row["gpu_driver_version"],
        cuda_version=row["cuda_version"],
        pytorch_version=row["pytorch_version"],
        comfyui_version=row["comfyui_version"],
        attention_backend=row["attention_backend"],
        resolution_w=row["resolution_w"],
        resolution_h=row["resolution_h"],
        frame_count=row["frame_count"],
        loop_iteration_count=row["loop_iteration_count"],
        block_swap_count=row["block_swap_count"],
        models=models,
        lora_count=row["lora_count"],
        lora_total_weight_mb=row["lora_total_weight_mb"],
        vram_peak_mb=row["vram_peak_mb"],
        ram_peak_mb=row["ram_peak_mb"],
        outcome=row["outcome"],
        error_message=row["error_message"],
        schema_version=row["schema_version"],
        node_pack_versions=packs,
    )


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_DDL)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def insert_run(record: RunRecord, db_path: Path = _DEFAULT_DB_PATH) -> int:
    """Persist a new run and return its assigned id.

    Typically called at the start of a run with outcome='running'.
    """
    row = _record_to_row(record)
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    sql = f"INSERT INTO runs ({cols}) VALUES ({placeholders})"

    with _connect(db_path) as conn:
        cur = conn.execute(sql, row)
        run_id = cur.lastrowid

    record.id = run_id
    return run_id


def update_run(record: RunRecord, db_path: Path = _DEFAULT_DB_PATH) -> None:
    """Update all mutable fields for a run that has already been inserted.

    Raises ValueError if record.id is None.
    """
    if record.id is None:
        raise ValueError("Cannot update a RunRecord that has no id (not yet inserted).")

    row = _record_to_row(record)
    sets = ", ".join(f"{k} = :{k}" for k in row.keys())
    sql = f"UPDATE runs SET {sets} WHERE id = :_id"
    row["_id"] = record.id

    with _connect(db_path) as conn:
        conn.execute(sql, row)


def finalize_run(
    record: RunRecord,
    outcome: str,
    vram_peak_mb: Optional[float],
    ram_peak_mb: Optional[float],
    error_message: Optional[str] = None,
    db_path: Path = _DEFAULT_DB_PATH,
) -> None:
    """Mark a run as finished and record measured peaks.

    Computes duration automatically from started_at.
    """
    record.ended_at = _now_iso()
    try:
        start = datetime.fromisoformat(record.started_at)
        end = datetime.fromisoformat(record.ended_at)
        record.duration_seconds = (end - start).total_seconds()
    except Exception:
        record.duration_seconds = None

    record.outcome = outcome
    record.vram_peak_mb = vram_peak_mb
    record.ram_peak_mb = ram_peak_mb
    record.error_message = error_message
    update_run(record, db_path)


def reap_stale_runs(
    gpu_model: Optional[str] = None,
    stale_after_seconds: float = 6 * 3600,
    error_message: str = "did not complete (crash or OOM — never reached logger_end)",
    db_path: Path = _DEFAULT_DB_PATH,
) -> int:
    """Mark abandoned 'running' rows as failed and return how many were reaped.

    A run that OOMs or crashes never reaches logger_end, so its row is left
    stuck at outcome='running'.  These are the most valuable negative data
    points (they tell the model where the ceiling is), so instead of deleting
    them we reclassify them as 'error'.

    Only rows older than `stale_after_seconds` are touched, so a genuinely
    in-progress run on another GPU/process is never clobbered.  When
    `gpu_model` is given, only that device's stale rows are reaped — important
    for the dual-GPU work machine where two ComfyUI instances may run at once.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - stale_after_seconds

    clauses = ["outcome = 'running'"]
    params: list[Any] = []
    if gpu_model:
        clauses.append("gpu_model = ?")
        params.append(gpu_model)
    where = " AND ".join(clauses)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, started_at FROM runs WHERE {where}", params
        ).fetchall()

        stale_ids = []
        for row in rows:
            try:
                started = datetime.fromisoformat(row["started_at"]).timestamp()
            except Exception:
                # Unparseable timestamp — treat as stale to avoid orphans
                started = 0
            if started <= cutoff:
                stale_ids.append(row["id"])

        for rid in stale_ids:
            conn.execute(
                "UPDATE runs SET outcome = 'error', error_message = ?, ended_at = ? "
                "WHERE id = ?",
                (error_message, _now_iso(), rid),
            )

    return len(stale_ids)


def get_run(run_id: int, db_path: Path = _DEFAULT_DB_PATH) -> Optional[RunRecord]:
    """Fetch a single run by id, or None if not found."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_record(row) if row else None


def get_all_runs(
    gpu_model: Optional[str] = None,
    outcome: Optional[str] = None,
    min_runs_for_ml: int = 0,
    db_path: Path = _DEFAULT_DB_PATH,
) -> list[RunRecord]:
    """Return completed runs, optionally filtered by GPU or outcome.

    Pass gpu_model to isolate one device; pass outcome='ok' to exclude
    crashed runs from ML training data.
    """
    clauses = ["outcome != 'running'"]
    params: list[Any] = []

    if gpu_model:
        clauses.append("gpu_model = ?")
        params.append(gpu_model)
    if outcome:
        clauses.append("outcome = ?")
        params.append(outcome)

    where = " AND ".join(clauses)
    sql = f"SELECT * FROM runs WHERE {where} ORDER BY started_at ASC"

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [_row_to_record(r) for r in rows]


def count_runs(
    gpu_model: Optional[str] = None,
    outcome: Optional[str] = None,
    db_path: Path = _DEFAULT_DB_PATH,
) -> int:
    """Return the count of completed runs matching the given filters."""
    return len(get_all_runs(gpu_model=gpu_model, outcome=outcome, db_path=db_path))


def export_anonymized(
    db_path: Path = _DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Export only numeric/categorical fields — no paths, names, or prompts.

    Intended for future community data sharing.  Never call unless the user
    explicitly triggers the export action.
    """
    runs = get_all_runs(outcome="ok", db_path=db_path)
    safe_fields = [
        "gpu_model", "gpu_vram_total_mb", "cuda_version", "pytorch_version",
        "comfyui_version", "attention_backend",
        "resolution_w", "resolution_h", "frame_count",
        "loop_iteration_count", "block_swap_count",
        "lora_count", "lora_total_weight_mb",
        "vram_peak_mb", "ram_peak_mb",
        "duration_seconds", "schema_version",
    ]
    result = []
    for r in runs:
        row: dict[str, Any] = {f: getattr(r, f) for f in safe_fields}
        # Quantization types only, no model names
        row["model_quant_types"] = sorted({m.quant_type for m in r.models})
        row["model_count"] = len(r.models)
        result.append(row)
    return result


def get_db_stats(db_path: Path = _DEFAULT_DB_PATH) -> dict[str, Any]:
    """Summary stats for display in the ComfyUI node UI."""
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM runs WHERE outcome != 'running'").fetchone()[0]
        by_outcome = dict(
            conn.execute(
                "SELECT outcome, COUNT(*) FROM runs WHERE outcome != 'running' GROUP BY outcome"
            ).fetchall()
        )
        gpus = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT gpu_model FROM runs WHERE gpu_model != ''"
            ).fetchall()
        ]
    return {"total_runs": total, "by_outcome": by_outcome, "gpus": gpus}
