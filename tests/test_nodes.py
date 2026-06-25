"""
Tests for the two ComfyUI logger nodes.

We exercise the node classes directly (no ComfyUI runtime), driving the same
start -> end flow a real graph would, with torch/psutil/pynvml absent so the
peaks degrade to None/0 cleanly.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core import session
from core.db import get_all_runs
from nodes.common import ANY, AnyType, SESSION_TYPE
from nodes.logger_start import VramForecasterLoggerStart, _resolution_from_latent
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


def _fake_torch_tensor(shape):
    """Return a fake tensor-like object with a .shape attribute."""
    t = MagicMock(name="torch.Tensor")
    t.shape = shape
    return t


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
# _resolution_from_latent
# ---------------------------------------------------------------------------

class TestResolutionFromLatent:
    def test_none_returns_nones(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert _resolution_from_latent(None) == (None, None, None)

    def test_image_latent_dict(self):
        # Image latent: B=1, C=4, H=64, W=112  → 896×512 pixels
        tensor = _fake_torch_tensor((1, 4, 64, 112))
        torch_mod = MagicMock(name="torch")
        torch_mod.Tensor = type(tensor)
        with patch.dict(sys.modules, {"torch": torch_mod}):
            w, h, f = _resolution_from_latent({"samples": tensor})
        assert w == 896
        assert h == 512
        assert f is None

    def test_video_latent_dict(self):
        # Video latent: B=1, C=16, F=21, H=64, W=112  → 896×512, 21 frames
        tensor = _fake_torch_tensor((1, 16, 21, 64, 112))
        torch_mod = MagicMock(name="torch")
        torch_mod.Tensor = type(tensor)
        with patch.dict(sys.modules, {"torch": torch_mod}):
            w, h, f = _resolution_from_latent({"samples": tensor})
        assert w == 896
        assert h == 512
        assert f == 21

    def test_no_torch_returns_nones(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert _resolution_from_latent({"samples": "anything"}) == (None, None, None)

    def test_wrong_shape_returns_nones(self):
        tensor = _fake_torch_tensor((1, 4))  # 2D — unexpected
        torch_mod = MagicMock(name="torch")
        torch_mod.Tensor = type(tensor)
        with patch.dict(sys.modules, {"torch": torch_mod}):
            assert _resolution_from_latent({"samples": tensor}) == (None, None, None)


# ---------------------------------------------------------------------------
# Node metadata / ComfyUI contract
# ---------------------------------------------------------------------------

def test_start_node_contract():
    it = VramForecasterLoggerStart.INPUT_TYPES()
    # All inputs are now optional — no required fields except the node exists
    assert "required" in it
    assert it["required"] == {}
    assert "passthrough" in it["optional"]
    assert "latent" in it["optional"]
    assert "resolution_w" in it["optional"]
    assert "block_swap_count" in it["optional"]
    assert VramForecasterLoggerStart.RETURN_NAMES == ("passthrough", "session")
    assert VramForecasterLoggerStart.RETURN_TYPES[1] == SESSION_TYPE
    # IS_CHANGED always returns NaN so successive calls differ
    import math
    assert math.isnan(VramForecasterLoggerStart.IS_CHANGED())


def test_end_node_contract():
    it = VramForecasterLoggerEnd.INPUT_TYPES()
    assert it["required"]["session"][0] == SESSION_TYPE
    assert VramForecasterLoggerEnd.OUTPUT_NODE is True
    import math
    assert math.isnan(VramForecasterLoggerEnd.IS_CHANGED())


# ---------------------------------------------------------------------------
# Full start -> end: minimal setup (no manual params)
# ---------------------------------------------------------------------------

def test_minimal_start_end_no_params(patched_db):
    """Node works with zero manual input — the minimum viable wiring."""
    dbp = patched_db
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()

    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        _, sess = start.start()
        end.end(session=sess)

    row = get_all_runs(db_path=dbp)[0]
    assert row.outcome == "ok"
    assert row.resolution_w is None
    assert row.frame_count is None


# ---------------------------------------------------------------------------
# Full start -> end: explicit manual params
# ---------------------------------------------------------------------------

def test_manual_params_stored(patched_db):
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
        out_passthrough, vram, ram = end.end(session=sess, passthrough=passthrough)

    assert out_passthrough == "LATENT_DATA"
    assert isinstance(vram, float)
    assert isinstance(ram, float)

    row = get_all_runs(db_path=dbp)[0]
    assert row.outcome == "ok"
    assert row.resolution_w == 896
    assert row.block_swap_count == 20
    assert row.lora_count == 2
    assert len(row.models) == 1
    assert row.models[0].quant_type == "fp8"
    assert row.models[0].size_mb == pytest.approx(8192.0)


# ---------------------------------------------------------------------------
# Latent auto-detection
# ---------------------------------------------------------------------------

def _latent_torch_mod(tensor):
    """Fake torch configured for latent-reading tests.

    All attributes that env_detect.py reads must return real Python scalars
    (not MagicMocks) so the RunRecord can be persisted to SQLite.
    """
    from types import SimpleNamespace
    m = MagicMock(name="torch")
    m.__version__ = "2.3.1+cu124"
    m.version = SimpleNamespace(cuda="12.4")
    m.cuda.is_available.return_value = False   # short-circuits GPU branch
    m.Tensor = type(tensor)
    return m


def test_resolution_auto_read_from_latent_socket(patched_db):
    """Wire the latent socket → resolution fills in without manual entry."""
    dbp = patched_db
    tensor = _fake_torch_tensor((1, 16, 21, 64, 112))  # 896×512, 21 frames
    fake_latent = {"samples": tensor}

    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()

    with patch.dict(sys.modules, {"torch": _latent_torch_mod(tensor),
                                  "psutil": None, "pynvml": None}):
        _, sess = start.start(latent=fake_latent)
        end.end(session=sess)

    row = get_all_runs(db_path=dbp)[0]
    assert row.resolution_w == 896
    assert row.resolution_h == 512
    assert row.frame_count == 21


def test_manual_resolution_overrides_latent(patched_db):
    """Explicit widget values win over auto-detected latent shape."""
    dbp = patched_db
    tensor = _fake_torch_tensor((1, 4, 64, 112))  # would auto-give 896×512
    fake_latent = {"samples": tensor}

    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()

    with patch.dict(sys.modules, {"torch": _latent_torch_mod(tensor),
                                  "psutil": None, "pynvml": None}):
        _, sess = start.start(latent=fake_latent, resolution_w=512, resolution_h=512)
        end.end(session=sess)

    row = get_all_runs(db_path=dbp)[0]
    assert row.resolution_w == 512   # manual override wins
    assert row.resolution_h == 512


def test_resolution_auto_from_passthrough_latent(patched_db):
    """When latent socket is absent, try to read resolution from passthrough."""
    dbp = patched_db
    tensor = _fake_torch_tensor((1, 4, 64, 112))
    fake_latent = {"samples": tensor}

    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()

    with patch.dict(sys.modules, {"torch": _latent_torch_mod(tensor),
                                  "psutil": None, "pynvml": None}):
        _, sess = start.start(passthrough=fake_latent)  # latent not wired separately
        end.end(session=sess)

    row = get_all_runs(db_path=dbp)[0]
    assert row.resolution_w == 896
    assert row.resolution_h == 512


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_end_with_invalid_session_does_not_crash(patched_db):
    end = VramForecasterLoggerEnd()
    out, vram, ram = end.end(session="not a session", passthrough="X")
    assert out == "X"
    assert vram == 0.0
    assert ram == 0.0


def test_start_without_model_name_records_no_models(patched_db):
    dbp = patched_db
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        _, sess = start.start()
        end.end(session=sess)
    row = get_all_runs(db_path=dbp)[0]
    assert row.models == []


def test_end_passthrough_none(patched_db):
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        _, sess = start.start()
        out, _, _ = end.end(session=sess)
    assert out is None


def test_zero_block_swap_stored_as_none(patched_db):
    """0 for block_swap_count means 'not set', stored as NULL not 0."""
    dbp = patched_db
    start = VramForecasterLoggerStart()
    end = VramForecasterLoggerEnd()
    with patch.dict(sys.modules, {"torch": None, "psutil": None, "pynvml": None}):
        _, sess = start.start(block_swap_count=0)
        end.end(session=sess)
    row = get_all_runs(db_path=dbp)[0]
    assert row.block_swap_count is None
