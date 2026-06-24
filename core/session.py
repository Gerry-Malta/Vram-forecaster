"""
Run-session lifecycle for VRAM Forecaster.

A session bridges the two ComfyUI nodes: logger_start opens one, logger_end
closes it.  The session object is what flows along the wire between the two
nodes, which is also how ComfyUI's data-dependency graph guarantees that
logger_end runs after the work it measures.

Crash / OOM handling has three layers, from most to least precise:

1. logger_end closes the session normally with the real measured peaks.
2. If the process dies mid-run (e.g. ComfyUI killed, hard OOM), an atexit
   handler flushes every still-open session as 'error' so the row is not
   left dangling at 'running'.
3. If neither fires (e.g. SIGKILL with no atexit), the next logger_start
   calls db.reap_stale_runs(), which reclassifies old 'running' rows.

Layers 2 and 3 lose the exact peak but still preserve the negative data
point — knowing a config failed is what teaches the model where the ceiling
is.
"""

import atexit
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import db
from .db import RunRecord, ModelInfo  # noqa: F401  (re-exported for nodes)
from .env_detect import EnvironmentSnapshot, capture_snapshot
from .monitor import PeakMemoryMonitor, read_vram_peak_mb, reset_vram_peak


# ---------------------------------------------------------------------------
# Open-session registry + crash safety net
# ---------------------------------------------------------------------------

_OPEN_SESSIONS: "dict[int, ForecastSession]" = {}
_REGISTRY_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _ensure_atexit_registered() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_flush_open_sessions_on_exit)
        _ATEXIT_REGISTERED = True


def _flush_open_sessions_on_exit() -> None:
    """Mark every still-open session as 'error' when the process exits."""
    with _REGISTRY_LOCK:
        sessions = list(_OPEN_SESSIONS.values())
    for session in sessions:
        try:
            session.close(
                outcome="error",
                error_message="process exited before logger_end ran (crash or OOM)",
            )
        except Exception:
            # Never let cleanup raise during interpreter shutdown.
            pass


def open_sessions_count() -> int:
    """Number of sessions currently open in this process (for tests/UI)."""
    with _REGISTRY_LOCK:
        return len(_OPEN_SESSIONS)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class ForecastSession:
    """Live handle for one run, carried on the wire from start to end node."""

    record: RunRecord
    db_path: Path
    monitor: PeakMemoryMonitor
    snapshot: EnvironmentSnapshot
    closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def close(
        self,
        outcome: str,
        error_message: Optional[str] = None,
        vram_peak_mb: Optional[float] = None,
    ) -> None:
        """Finalize the run: stop sampling, read peaks, persist. Idempotent."""
        with self._lock:
            if self.closed:
                return
            self.closed = True

        peaks = self.monitor.stop()

        # Prefer torch's exact high-water mark; fall back to the NVML-sampled
        # system VRAM peak when torch is unavailable.
        if vram_peak_mb is None:
            vram_peak_mb = read_vram_peak_mb()
        if vram_peak_mb is None:
            vram_peak_mb = peaks.get("nvml_vram_peak_mb")

        db.finalize_run(
            self.record,
            outcome=outcome,
            vram_peak_mb=vram_peak_mb,
            ram_peak_mb=peaks.get("ram_peak_mb"),
            error_message=error_message,
            db_path=self.db_path,
        )

        if self.record.id is not None:
            with _REGISTRY_LOCK:
                _OPEN_SESSIONS.pop(self.record.id, None)


def open_session(
    record: RunRecord,
    snapshot: EnvironmentSnapshot,
    db_path: Path = db._DEFAULT_DB_PATH,
    reap_stale: bool = True,
    monitor_interval_seconds: float = 0.25,
) -> ForecastSession:
    """Begin measuring a run: reap stale rows, insert, reset + start monitors.

    Returns a ForecastSession to be carried to logger_end.
    """
    _ensure_atexit_registered()

    if reap_stale:
        try:
            db.reap_stale_runs(gpu_model=record.gpu_model, db_path=db_path)
        except Exception:
            # Reaping is best-effort housekeeping; never block a real run.
            pass

    run_id = db.insert_run(record, db_path)

    # Reset torch's high-water mark and begin RAM sampling as late as possible
    # so the measured peak reflects the work that follows this node.
    reset_vram_peak()
    monitor = PeakMemoryMonitor(interval_seconds=monitor_interval_seconds)
    monitor.start()

    session = ForecastSession(
        record=record, db_path=db_path, monitor=monitor, snapshot=snapshot
    )
    with _REGISTRY_LOCK:
        _OPEN_SESSIONS[run_id] = session
    return session


def build_record_from_snapshot(
    snapshot: EnvironmentSnapshot,
    *,
    resolution_w: Optional[int] = None,
    resolution_h: Optional[int] = None,
    frame_count: Optional[int] = None,
    loop_iteration_count: Optional[int] = None,
    block_swap_count: Optional[int] = None,
    lora_count: int = 0,
    lora_total_weight_mb: Optional[float] = None,
    models: Optional[list[ModelInfo]] = None,
) -> RunRecord:
    """Assemble a RunRecord, merging detected environment with declared params."""
    return RunRecord(
        gpu_model=snapshot.gpu_model,
        gpu_vram_total_mb=snapshot.gpu_vram_total_mb,
        gpu_driver_version=snapshot.gpu_driver_version,
        cuda_version=snapshot.cuda_version,
        pytorch_version=snapshot.pytorch_version,
        comfyui_version=snapshot.comfyui_version,
        attention_backend=snapshot.attention_backend,
        resolution_w=resolution_w,
        resolution_h=resolution_h,
        frame_count=frame_count,
        loop_iteration_count=loop_iteration_count,
        block_swap_count=block_swap_count,
        lora_count=lora_count,
        lora_total_weight_mb=lora_total_weight_mb,
        models=models or [],
        node_pack_versions=snapshot.node_pack_versions,
    )
