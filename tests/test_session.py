"""
Tests for core/session.py and the crash-recovery path in core/db.reap_stale_runs.

Runs without torch/psutil — the monitor degrades to None peaks, which is the
correct behaviour to assert here.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from core import db, session
from core.db import RunRecord, get_run, insert_run, reap_stale_runs
from core.env_detect import EnvironmentSnapshot
from core.session import (
    build_record_from_snapshot,
    open_session,
    open_sessions_count,
)


@pytest.fixture()
def dbp(tmp_path: Path) -> Path:
    return tmp_path / "session_test.db"


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure the global open-session registry starts empty for each test."""
    with session._REGISTRY_LOCK:
        session._OPEN_SESSIONS.clear()
    yield
    with session._REGISTRY_LOCK:
        session._OPEN_SESSIONS.clear()


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        gpu_model="RTX 4090",
        gpu_vram_total_mb=24576,
        gpu_driver_version="555.85",
        cuda_version="12.4",
        pytorch_version="2.3.1",
        comfyui_version="0.3.10",
        attention_backend="flash",
        node_pack_versions={"KJNodes": "1.0"},
        ram_total_mb=65536,
    )


# ---------------------------------------------------------------------------
# build_record_from_snapshot
# ---------------------------------------------------------------------------

def test_build_record_merges_snapshot_and_params():
    rec = build_record_from_snapshot(
        _snapshot(),
        resolution_w=896,
        resolution_h=512,
        frame_count=81,
        block_swap_count=20,
        lora_count=2,
    )
    assert rec.gpu_model == "RTX 4090"
    assert rec.attention_backend == "flash"
    assert rec.resolution_w == 896
    assert rec.block_swap_count == 20
    assert rec.lora_count == 2
    assert rec.node_pack_versions == {"KJNodes": "1.0"}


# ---------------------------------------------------------------------------
# open_session / close happy path
# ---------------------------------------------------------------------------

def test_open_session_inserts_running_row(dbp):
    rec = build_record_from_snapshot(_snapshot(), resolution_w=896)
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        sess = open_session(rec, _snapshot(), db_path=dbp)

    assert rec.id is not None
    stored = get_run(rec.id, dbp)
    assert stored.outcome == "running"
    assert open_sessions_count() == 1
    # cleanup
    sess.close(outcome="ok")


def test_close_finalizes_row_and_clears_registry(dbp):
    rec = build_record_from_snapshot(_snapshot(), resolution_w=896)
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        sess = open_session(rec, _snapshot(), db_path=dbp)
        sess.close(outcome="ok")

    stored = get_run(rec.id, dbp)
    assert stored.outcome == "ok"
    assert stored.ended_at is not None
    assert stored.duration_seconds is not None
    assert open_sessions_count() == 0


def test_close_is_idempotent(dbp):
    rec = build_record_from_snapshot(_snapshot(), resolution_w=896)
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        sess = open_session(rec, _snapshot(), db_path=dbp)
        sess.close(outcome="ok")
        sess.close(outcome="error")  # second call must be a no-op

    stored = get_run(rec.id, dbp)
    assert stored.outcome == "ok"  # not overwritten


def test_close_with_explicit_vram_peak(dbp):
    rec = build_record_from_snapshot(_snapshot(), resolution_w=896)
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        sess = open_session(rec, _snapshot(), db_path=dbp)
        sess.close(outcome="ok", vram_peak_mb=19800.0)

    stored = get_run(rec.id, dbp)
    assert stored.vram_peak_mb == pytest.approx(19800.0)


# ---------------------------------------------------------------------------
# atexit flush
# ---------------------------------------------------------------------------

def test_atexit_flush_marks_open_sessions_as_error(dbp):
    rec = build_record_from_snapshot(_snapshot(), resolution_w=896)
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        open_session(rec, _snapshot(), db_path=dbp)
        assert open_sessions_count() == 1
        session._flush_open_sessions_on_exit()

    stored = get_run(rec.id, dbp)
    assert stored.outcome == "error"
    assert "process exited" in stored.error_message
    assert open_sessions_count() == 0


# ---------------------------------------------------------------------------
# reap_stale_runs
# ---------------------------------------------------------------------------

def test_reap_marks_old_running_rows(dbp):
    old = RunRecord(gpu_model="RTX 4090", gpu_vram_total_mb=24576,
                    attention_backend="flash")
    old.started_at = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
    insert_run(old, dbp)

    reaped = reap_stale_runs(gpu_model="RTX 4090", db_path=dbp)
    assert reaped == 1

    stored = get_run(old.id, dbp)
    assert stored.outcome == "error"
    assert "OOM" in stored.error_message


def test_reap_leaves_recent_running_rows(dbp):
    fresh = RunRecord(gpu_model="RTX 4090", gpu_vram_total_mb=24576,
                      attention_backend="flash")
    # started just now → within the staleness window
    insert_run(fresh, dbp)

    reaped = reap_stale_runs(gpu_model="RTX 4090", db_path=dbp)
    assert reaped == 0
    assert get_run(fresh.id, dbp).outcome == "running"


def test_reap_respects_gpu_filter(dbp):
    a = RunRecord(gpu_model="RTX 4090", gpu_vram_total_mb=24576, attention_backend="flash")
    b = RunRecord(gpu_model="RTX PRO 6000", gpu_vram_total_mb=98304, attention_backend="flash")
    for r in (a, b):
        r.started_at = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        insert_run(r, dbp)

    reaped = reap_stale_runs(gpu_model="RTX 4090", db_path=dbp)
    assert reaped == 1
    assert get_run(a.id, dbp).outcome == "error"
    assert get_run(b.id, dbp).outcome == "running"  # other GPU untouched


def test_open_session_reaps_previous_crash(dbp):
    """A new run's logger_start should reclaim the previous crashed run."""
    crashed = RunRecord(gpu_model="RTX 4090", gpu_vram_total_mb=24576,
                        attention_backend="flash")
    crashed.started_at = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
    insert_run(crashed, dbp)

    rec = build_record_from_snapshot(_snapshot())
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        sess = open_session(rec, _snapshot(), db_path=dbp)

    assert get_run(crashed.id, dbp).outcome == "error"
    sess.close(outcome="ok")
