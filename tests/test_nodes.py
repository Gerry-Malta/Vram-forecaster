"""
Tests for the two ComfyUI logger nodes.

We exercise the node classes directly (no ComfyUI runtime), driving the same
start -> end flow a real graph would, with torch/psutil/pynvml absent so the
peaks degrade to None/0 cleanly.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from core import session
from core.db import get_run, get_all_runs
from nodes.common import ANY, AnyType, SESSION_TYPE
from nodes.logger_start import VramForecasterLoggerStart
from nodes.logger_end import VramForecasterLoggerEnd


@pytest.fixture(autouse=True)
def _clear_registry():
    with session._REGISTRY_LOCK:
        session._OPEN_SESSIONS.clear()
    yield
    with session._REGISTRY_LOCK:
        session._OPEN_SESSIONS.clear()


@pytest.fixture()
def patched_db(tmp_path, monkeypatch):
    """Point both nodes at a temp DB and a temp custom_nodes dir."""
    dbp = tmp_path / "nodes_test.db"
    import nodes.logger_start as ls
    monkeypatch.setattr(ls, "DEFAULT_DB_PATH", dbp)
    monkeypatch.setattr(ls, "DEFAULT_CUSTOM_NODES_DIR", tmp_path)
    return dbp


# ---------------------------------------------------------------------------
# AnyType wildcard
# ---------------------------------------------------------------------------

def test_anytype_equals_everything():
    assert ANY == "IMAGE"
    assert ANY == "LATENT"
    assert not (ANY != "MODEL")


def test_anytype_is_hashable():
    {ANY: 1}  # must not raise


# ---------------------------------------------------------------------------
# Node metadata / ComfyUI contract
# ---------------------------------------------------------------------------

def test_start_node_contract():
    it = VramForecasterLoggerStart.INPUT_TYPES()
    assert "resolution_w" in it["required"]
    assert "passthrough" in it["optional"]
    assert VramForecasterLoggerStart.RETURN_NAMES == ("passthrough", "session")
    assert VramForecasterLoggerStart.RETURN_TYPES[1] == SESSION_TYPE
    # always re-executes
    assert VramForecasterLoggerStart.IS_CHANGED() != VramForecasterLoggerStart.IS_CHANGED()


def test_end_node_contract():
    it = VramForecasterLoggerEnd.INPUT_TYPES()
    assert it["required"]["session"][0] == SESSION_TYPE
    assert VramForecasterLoggerEnd.OUTPUT_NODE is True
    assert VramForecasterLoggerEnd.IS_CHANGED() != VramForecasterLoggerEnd.IS_CHANGED()


# ---------------------------------------------------------------------------
# Full start -> end flow
# ---------------------------------------------------------------------------

def test_start_then_end_writes_ok_row(patched_db):
    dbp = patched_db
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()

    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        passthrough, sess = start.start(
            resolution_w=896, resolution_h=512, frame_count=81,
            loop_iteration_count=4, block_swap_count=20,
            lora_count=2, lora_total_weight_mb=256.0,
            passthrough="LATENT_DATA",
            model_name="wan2.2-fp8.safetensors", quant_type="fp8",
            model_size_mb=8192.0,
        )
        assert passthrough == "LATENT_DATA"
        assert sess in session._OPEN_SESSIONS.values()

        out_passthrough, vram, ram = end.end(session=sess, passthrough=passthrough)

    assert out_passthrough == "LATENT_DATA"
    assert isinstance(vram, float)
    assert isinstance(ram, float)

    runs = get_all_runs(db_path=dbp)
    assert len(runs) == 1
    row = runs[0]
    assert row.outcome == "ok"
    assert row.resolution_w == 896
    assert row.block_swap_count == 20
    assert row.lora_count == 2
    assert len(row.models) == 1
    assert row.models[0].quant_type == "fp8"
    assert row.models[0].size_mb == pytest.approx(8192.0)


def test_start_without_model_name_records_no_models(patched_db):
    dbp = patched_db
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        _, sess = start.start(
            resolution_w=512, resolution_h=512, frame_count=1,
            loop_iteration_count=1, block_swap_count=0,
            lora_count=0, lora_total_weight_mb=0.0,
        )
        end.end(session=sess)

    row = get_all_runs(db_path=dbp)[0]
    assert row.models == []


def test_end_with_invalid_session_does_not_crash(patched_db):
    end = VramForecasterLoggerEnd()
    out, vram, ram = end.end(session="not a session", passthrough="X")
    assert out == "X"
    assert vram == 0.0
    assert ram == 0.0


def test_end_passthrough_none(patched_db):
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        _, sess = start.start(
            resolution_w=512, resolution_h=512, frame_count=1,
            loop_iteration_count=1, block_swap_count=0,
            lora_count=0, lora_total_weight_mb=0.0,
        )
        out, _, _ = end.end(session=sess)
    assert out is None
