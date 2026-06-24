"""Tests for core/monitor.py — run without torch, psutil, or pynvml."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.monitor import (
    PeakMemoryMonitor,
    read_vram_peak_mb,
    reset_vram_peak,
)

_MB = 1024 * 1024


# ---------------------------------------------------------------------------
# VRAM high-water mark helpers
# ---------------------------------------------------------------------------

def test_reset_vram_peak_no_torch_is_noop():
    with patch.dict(sys.modules, {"torch": None}):
        reset_vram_peak()  # must not raise


def test_read_vram_peak_no_torch_returns_none():
    with patch.dict(sys.modules, {"torch": None}):
        assert read_vram_peak_mb() is None


def test_read_vram_peak_with_torch():
    torch = MagicMock(name="torch")
    torch.cuda.is_available.return_value = True
    torch.cuda.max_memory_allocated.return_value = 12 * 1024 * _MB  # 12 GB
    with patch.dict(sys.modules, {"torch": torch}):
        assert read_vram_peak_mb() == 12 * 1024


def test_read_vram_peak_cuda_unavailable_returns_none():
    torch = MagicMock(name="torch")
    torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": torch}):
        assert read_vram_peak_mb() is None


def test_reset_vram_peak_calls_torch():
    torch = MagicMock(name="torch")
    torch.cuda.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": torch}):
        reset_vram_peak()
    torch.cuda.reset_peak_memory_stats.assert_called_once()


# ---------------------------------------------------------------------------
# PeakMemoryMonitor
# ---------------------------------------------------------------------------

def test_monitor_no_psutil_returns_none_peaks():
    with patch.dict(sys.modules, {"psutil": None, "pynvml": None}):
        mon = PeakMemoryMonitor(interval_seconds=0.01)
        mon.start()
        peaks = mon.stop()
    assert peaks["ram_peak_mb"] is None
    assert peaks["nvml_vram_peak_mb"] is None


def test_monitor_tracks_ram_peak():
    psutil = MagicMock(name="psutil")
    # Return rising then falling values; peak should be the max.
    values = [10 * _MB, 30 * _MB, 20 * _MB]
    psutil.virtual_memory.side_effect = [SimpleNamespace(used=v) for v in values] + [
        SimpleNamespace(used=5 * _MB)
    ] * 50
    with patch.dict(sys.modules, {"psutil": psutil, "pynvml": None}):
        mon = PeakMemoryMonitor(interval_seconds=0.005, sample_nvml=False)
        mon.start()           # samples once (10)
        import time
        time.sleep(0.05)      # background thread samples more
        peaks = mon.stop()
    assert peaks["ram_peak_mb"] == 30  # MB


def test_monitor_immediate_sample_on_start():
    """A workflow shorter than one interval must still record a sample."""
    psutil = MagicMock(name="psutil")
    psutil.virtual_memory.return_value = SimpleNamespace(used=42 * _MB)
    with patch.dict(sys.modules, {"psutil": psutil, "pynvml": None}):
        mon = PeakMemoryMonitor(interval_seconds=100.0, sample_nvml=False)
        mon.start()
        peaks = mon.stop()
    assert peaks["ram_peak_mb"] == 42


def test_monitor_nvml_fallback_peak():
    psutil = MagicMock(name="psutil")
    psutil.virtual_memory.return_value = SimpleNamespace(used=1 * _MB)
    pynvml = MagicMock(name="pynvml")
    pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=8 * 1024 * _MB)
    with patch.dict(sys.modules, {"psutil": psutil, "pynvml": pynvml}):
        mon = PeakMemoryMonitor(interval_seconds=100.0, sample_nvml=True)
        mon.start()
        peaks = mon.stop()
    assert peaks["nvml_vram_peak_mb"] == 8 * 1024


def test_monitor_double_start_is_safe():
    with patch.dict(sys.modules, {"psutil": None, "pynvml": None}):
        mon = PeakMemoryMonitor(interval_seconds=0.01)
        mon.start()
        mon.start()  # must not spawn a second thread or raise
        mon.stop()


def test_monitor_stop_without_start_is_safe():
    with patch.dict(sys.modules, {"psutil": None, "pynvml": None}):
        mon = PeakMemoryMonitor()
        peaks = mon.stop()  # must not raise
    assert "ram_peak_mb" in peaks
