"""Tests for core/db.py — runnable without ComfyUI or a GPU."""

import tempfile
from pathlib import Path

import pytest

from core.db import (
    ModelInfo,
    RunRecord,
    count_runs,
    export_anonymized,
    finalize_run,
    get_all_runs,
    get_db_stats,
    get_run,
    insert_run,
    update_run,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "test_forecaster.db"


def _base_record() -> RunRecord:
    return RunRecord(
        gpu_model="RTX 4090",
        gpu_vram_total_mb=24576,
        gpu_driver_version="555.85",
        cuda_version="12.4",
        pytorch_version="2.3.1",
        comfyui_version="0.3.10",
        attention_backend="flash",
        resolution_w=896,
        resolution_h=512,
        frame_count=81,
        loop_iteration_count=4,
        block_swap_count=20,
        models=[
            ModelInfo(name="wan2.2-fp8.safetensors", quant_type="fp8", size_mb=8192),
            ModelInfo(name="vae.safetensors", quant_type="fp16", size_mb=335),
        ],
        lora_count=2,
        lora_total_weight_mb=256.0,
    )


# ---------------------------------------------------------------------------
# insert / get round-trip
# ---------------------------------------------------------------------------

def test_insert_assigns_id(db: Path):
    rec = _base_record()
    assert rec.id is None
    run_id = insert_run(rec, db)
    assert run_id == 1
    assert rec.id == 1


def test_get_round_trip(db: Path):
    rec = _base_record()
    insert_run(rec, db)
    fetched = get_run(1, db)

    assert fetched is not None
    assert fetched.gpu_model == "RTX 4090"
    assert fetched.attention_backend == "flash"
    assert fetched.resolution_w == 896
    assert fetched.frame_count == 81
    assert fetched.lora_count == 2
    assert fetched.outcome == "running"
    assert len(fetched.models) == 2
    assert fetched.models[0].quant_type == "fp8"
    assert fetched.models[1].name == "vae.safetensors"


def test_get_nonexistent_returns_none(db: Path):
    assert get_run(999, db) is None


# ---------------------------------------------------------------------------
# update / finalize
# ---------------------------------------------------------------------------

def test_update_run(db: Path):
    rec = _base_record()
    insert_run(rec, db)

    rec.attention_backend = "sage"
    update_run(rec, db)

    fetched = get_run(rec.id, db)
    assert fetched.attention_backend == "sage"


def test_update_without_id_raises(db: Path):
    rec = _base_record()
    with pytest.raises(ValueError, match="no id"):
        update_run(rec, db)


def test_finalize_ok(db: Path):
    rec = _base_record()
    insert_run(rec, db)
    finalize_run(rec, outcome="ok", vram_peak_mb=19800.5, ram_peak_mb=32000.0, db_path=db)

    fetched = get_run(rec.id, db)
    assert fetched.outcome == "ok"
    assert fetched.vram_peak_mb == pytest.approx(19800.5)
    assert fetched.ram_peak_mb == pytest.approx(32000.0)
    assert fetched.ended_at is not None
    assert fetched.duration_seconds is not None
    assert fetched.duration_seconds >= 0


def test_finalize_oom(db: Path):
    rec = _base_record()
    insert_run(rec, db)
    finalize_run(rec, outcome="oom", vram_peak_mb=None, ram_peak_mb=None,
                 error_message="CUDA out of memory", db_path=db)

    fetched = get_run(rec.id, db)
    assert fetched.outcome == "oom"
    assert fetched.error_message == "CUDA out of memory"
    assert fetched.vram_peak_mb is None


# ---------------------------------------------------------------------------
# get_all_runs / count_runs
# ---------------------------------------------------------------------------

def test_get_all_runs_excludes_running(db: Path):
    r1 = _base_record()
    r2 = _base_record()
    insert_run(r1, db)
    insert_run(r2, db)
    finalize_run(r1, "ok", 19000.0, 30000.0, db_path=db)
    # r2 stays 'running'

    rows = get_all_runs(db_path=db)
    assert len(rows) == 1
    assert rows[0].outcome == "ok"


def test_get_all_runs_filter_gpu(db: Path):
    r1 = _base_record()
    r2 = _base_record()
    r2.gpu_model = "RTX PRO 6000"
    r2.gpu_vram_total_mb = 96 * 1024
    insert_run(r1, db)
    insert_run(r2, db)
    finalize_run(r1, "ok", 19000.0, 30000.0, db_path=db)
    finalize_run(r2, "ok", 40000.0, 50000.0, db_path=db)

    rows_4090 = get_all_runs(gpu_model="RTX 4090", db_path=db)
    rows_6000 = get_all_runs(gpu_model="RTX PRO 6000", db_path=db)

    assert len(rows_4090) == 1
    assert len(rows_6000) == 1
    assert rows_4090[0].vram_peak_mb == pytest.approx(19000.0)


def test_count_runs(db: Path):
    r1 = _base_record()
    r2 = _base_record()
    insert_run(r1, db)
    insert_run(r2, db)
    finalize_run(r1, "ok", 19000.0, 30000.0, db_path=db)
    finalize_run(r2, "oom", None, None, db_path=db)

    assert count_runs(db_path=db) == 2
    assert count_runs(outcome="ok", db_path=db) == 1
    assert count_runs(outcome="oom", db_path=db) == 1


# ---------------------------------------------------------------------------
# export_anonymized
# ---------------------------------------------------------------------------

def test_export_contains_no_model_names(db: Path):
    rec = _base_record()
    insert_run(rec, db)
    finalize_run(rec, "ok", 19000.0, 30000.0, db_path=db)

    rows = export_anonymized(db_path=db)
    assert len(rows) == 1
    row = rows[0]

    # Numeric fields present
    assert row["gpu_vram_total_mb"] == 24576
    assert row["vram_peak_mb"] == pytest.approx(19000.0)
    assert row["model_count"] == 2
    assert set(row["model_quant_types"]) == {"fp8", "fp16"}

    # No names or paths
    assert "models_json" not in row
    assert "error_message" not in row
    assert "started_at" not in row


def test_export_excludes_oom_runs(db: Path):
    r1 = _base_record()
    r2 = _base_record()
    insert_run(r1, db)
    insert_run(r2, db)
    finalize_run(r1, "ok", 19000.0, 30000.0, db_path=db)
    finalize_run(r2, "oom", None, None, db_path=db)

    rows = export_anonymized(db_path=db)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# get_db_stats
# ---------------------------------------------------------------------------

def test_db_stats_empty(db: Path):
    stats = get_db_stats(db_path=db)
    assert stats["total_runs"] == 0
    assert stats["by_outcome"] == {}
    assert stats["gpus"] == []


def test_db_stats_populated(db: Path):
    r1 = _base_record()
    r2 = _base_record()
    r2.gpu_model = "RTX PRO 6000"
    r2.gpu_vram_total_mb = 96 * 1024
    insert_run(r1, db)
    insert_run(r2, db)
    finalize_run(r1, "ok", 19000.0, 30000.0, db_path=db)
    finalize_run(r2, "oom", None, None, db_path=db)

    stats = get_db_stats(db_path=db)
    assert stats["total_runs"] == 2
    assert stats["by_outcome"]["ok"] == 1
    assert stats["by_outcome"]["oom"] == 1
    assert set(stats["gpus"]) == {"RTX 4090", "RTX PRO 6000"}


# ---------------------------------------------------------------------------
# Node-pack versioning
# ---------------------------------------------------------------------------

def test_node_pack_versions_round_trip(db: Path):
    rec = _base_record()
    rec.node_pack_versions = {
        "WanVideoWrapper": "1.2.3",
        "KJNodes": "0.9.1",
    }
    insert_run(rec, db)
    finalize_run(rec, "ok", 19000.0, 30000.0, db_path=db)

    fetched = get_run(rec.id, db)
    assert fetched.node_pack_versions["WanVideoWrapper"] == "1.2.3"
    assert fetched.node_pack_versions["KJNodes"] == "0.9.1"


# ---------------------------------------------------------------------------
# Multiple runs ordering
# ---------------------------------------------------------------------------

def test_get_all_runs_sorted_ascending(db: Path):
    for _ in range(3):
        r = _base_record()
        insert_run(r, db)
        finalize_run(r, "ok", 18000.0, 28000.0, db_path=db)

    rows = get_all_runs(db_path=db)
    ids = [r.id for r in rows]
    assert ids == sorted(ids)
