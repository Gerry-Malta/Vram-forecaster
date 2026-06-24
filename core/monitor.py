"""
Peak memory measurement for VRAM Forecaster.

Two kinds of peak are measured differently:

- VRAM: torch keeps an internal high-water mark
  (`torch.cuda.max_memory_allocated`) that is exact for this process's
  allocations and costs nothing to read — so we reset it at the start of a
  run and read it at the end.  No sampling thread needed.

- System RAM: there is no built-in high-water mark, so we poll
  `psutil.virtual_memory().used` on a background daemon thread and keep the
  maximum.  System-wide (not just our RSS) because the schema asks for
  "picco RAM di sistema" and because ComfyUI spreads work across helper
  processes.

Both paths degrade gracefully when torch / psutil are missing, so the module
imports and runs under test without a GPU.
"""

import threading
import time
from typing import Optional


_MB = 1024 * 1024


# ---------------------------------------------------------------------------
# VRAM — torch high-water mark
# ---------------------------------------------------------------------------

def reset_vram_peak() -> None:
    """Reset torch's per-process VRAM high-water mark. No-op without CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def read_vram_peak_mb() -> Optional[float]:
    """Read torch's peak VRAM allocation in MB since the last reset.

    Returns None if torch/CUDA are unavailable (so the caller stores NULL
    rather than a misleading 0).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / _MB
    except Exception:
        pass
    return None


def _read_system_ram_used_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().used / _MB
    except Exception:
        return None


def _read_nvml_used_mb() -> Optional[float]:
    """System-wide VRAM in use on the current device, via NVML.

    Used only as a fallback when torch's high-water mark is unavailable.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / _MB
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RAM (+ fallback VRAM) — sampling thread
# ---------------------------------------------------------------------------

class PeakMemoryMonitor:
    """Background sampler tracking peak system RAM (and fallback VRAM) usage.

    Usage:
        mon = PeakMemoryMonitor()
        mon.start()
        ... run workflow ...
        peaks = mon.stop()   # {"ram_peak_mb": ..., "nvml_vram_peak_mb": ...}
    """

    def __init__(self, interval_seconds: float = 0.25, sample_nvml: bool = True):
        self.interval_seconds = interval_seconds
        self.sample_nvml = sample_nvml
        self._peak_ram_mb: Optional[float] = None
        self._peak_nvml_vram_mb: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return  # already running
        # Take an immediate sample so a workflow that finishes faster than one
        # interval still records something.
        self._sample_once()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="vram-forecaster-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, Optional[float]]:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        # One last sample to catch a peak right at the end.
        self._sample_once()
        return {
            "ram_peak_mb": self._peak_ram_mb,
            "nvml_vram_peak_mb": self._peak_nvml_vram_mb,
        }

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        ram = _read_system_ram_used_mb()
        if ram is not None:
            if self._peak_ram_mb is None or ram > self._peak_ram_mb:
                self._peak_ram_mb = ram

        if self.sample_nvml:
            vram = _read_nvml_used_mb()
            if vram is not None:
                if self._peak_nvml_vram_mb is None or vram > self._peak_nvml_vram_mb:
                    self._peak_nvml_vram_mb = vram

    @property
    def peak_ram_mb(self) -> Optional[float]:
        return self._peak_ram_mb
