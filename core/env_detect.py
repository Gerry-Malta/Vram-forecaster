"""
Runtime environment detection for VRAM Forecaster.

Every function is designed to be graceful when a required package is absent:
it returns None or a safe fallback string rather than raising.  This makes
the module testable without torch, CUDA, or a GPU.

Detection priority for attention_backend mirrors ComfyUI's own selection:
    sage > flash > xformers > vanilla > unknown
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentSnapshot:
    gpu_model: str                              # "RTX 4090" or "CPU" if no GPU
    gpu_vram_total_mb: int                      # 0 if no GPU
    gpu_driver_version: Optional[str]
    cuda_version: Optional[str]
    pytorch_version: Optional[str]
    comfyui_version: Optional[str]
    attention_backend: str                      # vanilla / flash / sage / xformers / unknown
    node_pack_versions: dict[str, str] = field(default_factory=dict)
    ram_total_mb: int = 0                       # system RAM total (informational)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_gpu() -> tuple[str, int]:
    """Return (model_name, vram_total_mb). Falls back to ('CPU', 0) if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU", 0
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram_mb = props.total_memory // (1024 * 1024)
        return name, vram_mb
    except Exception:
        return "CPU", 0


def detect_driver_version() -> Optional[str]:
    """Return NVIDIA driver version string, or None if not detectable."""
    # Strategy 1: pynvml — no subprocess, preferred
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        pass

    # Strategy 2: nvidia-smi subprocess fallback
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip().splitlines()[0].strip()
            if version:
                return version
    except Exception:
        pass

    return None


def detect_cuda_version() -> Optional[str]:
    """Return CUDA version as a string (e.g. '12.4'), or None."""
    try:
        import torch
        return torch.version.cuda  # may be None for CPU-only torch builds
    except Exception:
        return None


def detect_pytorch_version() -> Optional[str]:
    """Return PyTorch version string, or None if not installed."""
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def detect_attention_backend() -> str:
    """
    Return the attention backend that ComfyUI will actually use.

    When running inside ComfyUI, comfy.model_management gives the
    authoritative answer because it reflects CLI flags (e.g.
    --use-pytorch-cross-attention) as well as installed packages.
    Outside ComfyUI, we probe importable packages as a proxy — correct
    in the common case, but can't account for override flags.
    """
    # --- Path 1: inside ComfyUI, use the already-loaded model_management ---
    # We read from sys.modules directly rather than using `import` to avoid
    # the import machinery caching a stale module reference after the first call.
    mm = sys.modules.get("comfy.model_management")
    if mm is not None:
        if getattr(mm, "sage_attention_enabled", lambda: False)():
            return "sage"
        if getattr(mm, "flash_attention_enabled", lambda: False)():
            return "flash"
        # Older ComfyUI API name
        if getattr(mm, "pytorch_attention_flash_attention", lambda: False)():
            return "flash"
        if getattr(mm, "xformers_enabled", lambda: False)():
            return "xformers"
        return "vanilla"

    # --- Path 2: outside ComfyUI, probe packages ---
    for pkg, backend in [
        ("sageattention", "sage"),
        ("flash_attn", "flash"),
        ("xformers", "xformers"),
    ]:
        try:
            __import__(pkg)
            return backend
        except ImportError:
            continue

    # torch present but no acceleration package → vanilla
    try:
        import torch  # noqa: F401, PLC0415
        return "vanilla"
    except ImportError:
        return "unknown"


def detect_comfyui_version() -> Optional[str]:
    """Return ComfyUI version string, or None if not detectable."""
    # importlib.metadata works when ComfyUI is installed as a package
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("comfyui")
        except PackageNotFoundError:
            pass
    except Exception:
        pass

    # comfy module __version__ attribute
    try:
        import comfy  # noqa: PLC0415
        v = getattr(comfy, "__version__", None)
        if v:
            return str(v)
    except ImportError:
        pass

    return None


def detect_ram_total() -> int:
    """Return total system RAM in MB, or 0 if psutil is unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 * 1024)
    except Exception:
        return 0


def detect_node_pack_versions(custom_nodes_dir: Optional[Path] = None) -> dict[str, str]:
    """
    Best-effort detection of version strings for installed custom node packs.

    For each subdirectory of custom_nodes_dir, tries (in order):
      1. pyproject.toml [project.version] or [tool.poetry.version]
      2. git describe --tags --abbrev=0

    Returns an empty dict if custom_nodes_dir is None, missing, or nothing
    is detectable.
    """
    if custom_nodes_dir is None:
        return {}

    custom_nodes_dir = Path(custom_nodes_dir)
    if not custom_nodes_dir.is_dir():
        return {}

    versions: dict[str, str] = {}
    for pack_dir in sorted(custom_nodes_dir.iterdir()):
        if not pack_dir.is_dir() or pack_dir.name.startswith("."):
            continue
        v = _version_from_pyproject(pack_dir) or _version_from_git_tag(pack_dir)
        if v:
            versions[pack_dir.name] = v

    return versions


def _version_from_pyproject(pack_dir: Path) -> Optional[str]:
    pyproject = pack_dir / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(pyproject, "rb") as fh:
            data = tomllib.load(fh)
        return (
            data.get("project", {}).get("version")
            or data.get("tool", {}).get("poetry", {}).get("version")
        )
    except Exception:
        return None


def _version_from_git_tag(pack_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=pack_dir,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            tag = result.stdout.strip()
            return tag or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------

def capture_snapshot(custom_nodes_dir: Optional[Path] = None) -> EnvironmentSnapshot:
    """
    Collect the full environment snapshot in one call.

    Called by logger_start.py at the beginning of every run.
    Each sub-detection is independent — a failure in one does not abort others.
    """
    gpu_model, vram_total_mb = detect_gpu()
    return EnvironmentSnapshot(
        gpu_model=gpu_model,
        gpu_vram_total_mb=vram_total_mb,
        gpu_driver_version=detect_driver_version(),
        cuda_version=detect_cuda_version(),
        pytorch_version=detect_pytorch_version(),
        comfyui_version=detect_comfyui_version(),
        attention_backend=detect_attention_backend(),
        node_pack_versions=detect_node_pack_versions(custom_nodes_dir),
        ram_total_mb=detect_ram_total(),
    )
