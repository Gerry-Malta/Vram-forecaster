"""
Tests for core/env_detect.py.

All tests run without torch, psutil, pynvml, xformers, flash_attn,
sageattention, or comfy.  The happy-path behaviour of each detector is
covered by injecting fake modules via sys.modules patching.
"""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.env_detect import (
    EnvironmentSnapshot,
    capture_snapshot,
    detect_attention_backend,
    detect_comfyui_version,
    detect_cuda_version,
    detect_driver_version,
    detect_gpu,
    detect_node_pack_versions,
    detect_pytorch_version,
    detect_ram_total,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_torch(*, cuda_available: bool = True, device_name: str = "RTX 4090",
                vram_bytes: int = 24 * 1024**3, cuda_version: str = "12.4",
                torch_version: str = "2.3.1+cu124") -> MagicMock:
    """Build a minimal fake torch module."""
    torch = MagicMock(name="torch")
    torch.__version__ = torch_version
    torch.version = SimpleNamespace(cuda=cuda_version)
    torch.cuda.is_available.return_value = cuda_available
    torch.cuda.current_device.return_value = 0
    torch.cuda.get_device_name.return_value = device_name
    props = SimpleNamespace(total_memory=vram_bytes)
    torch.cuda.get_device_properties.return_value = props
    return torch


def _fake_mm(*, sage: bool = False, flash: bool = False,
             xformers: bool = False, vanilla: bool = True) -> MagicMock:
    """Build a fake comfy.model_management with configurable enabled flags."""
    mm = MagicMock(name="comfy.model_management")
    mm.sage_attention_enabled.return_value = sage
    mm.flash_attention_enabled.return_value = flash
    mm.pytorch_attention_flash_attention.return_value = False
    mm.xformers_enabled.return_value = xformers
    return mm


# ---------------------------------------------------------------------------
# detect_gpu
# ---------------------------------------------------------------------------

class TestDetectGpu:
    def test_no_torch_returns_cpu(self):
        with patch.dict(sys.modules, {"torch": None}):
            model, vram = detect_gpu()
        assert model == "CPU"
        assert vram == 0

    def test_torch_no_cuda_returns_cpu(self):
        fake = _fake_torch(cuda_available=False)
        with patch.dict(sys.modules, {"torch": fake}):
            model, vram = detect_gpu()
        assert model == "CPU"
        assert vram == 0

    def test_torch_with_cuda(self):
        fake = _fake_torch(device_name="NVIDIA RTX 4090", vram_bytes=24 * 1024**3)
        with patch.dict(sys.modules, {"torch": fake}):
            model, vram = detect_gpu()
        assert model == "NVIDIA RTX 4090"
        assert vram == 24 * 1024  # 24 GB in MB

    def test_vram_calculation_6000(self):
        # RTX PRO 6000: 96 GB
        fake = _fake_torch(device_name="RTX PRO 6000", vram_bytes=96 * 1024**3)
        with patch.dict(sys.modules, {"torch": fake}):
            _, vram = detect_gpu()
        assert vram == 96 * 1024


# ---------------------------------------------------------------------------
# detect_driver_version
# ---------------------------------------------------------------------------

class TestDetectDriverVersion:
    def test_no_pynvml_no_smi_returns_none(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = detect_driver_version()
        assert result is None

    def test_pynvml_success(self):
        mock_pynvml = MagicMock(name="pynvml")
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "555.85"
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            result = detect_driver_version()
        assert result == "555.85"

    def test_pynvml_fails_smi_succeeds(self):
        mock_pynvml = MagicMock(name="pynvml")
        mock_pynvml.nvmlInit.side_effect = RuntimeError("nvml error")
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="555.85\n", stderr=""
        )
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            with patch("subprocess.run", return_value=completed):
                result = detect_driver_version()
        assert result == "555.85"

    def test_smi_nonzero_returncode_returns_none(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            completed = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="error"
            )
            with patch("subprocess.run", return_value=completed):
                result = detect_driver_version()
        assert result is None


# ---------------------------------------------------------------------------
# detect_cuda_version / detect_pytorch_version
# ---------------------------------------------------------------------------

class TestTorchVersions:
    def test_cuda_version_no_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert detect_cuda_version() is None

    def test_cuda_version_with_torch(self):
        fake = _fake_torch(cuda_version="12.4")
        with patch.dict(sys.modules, {"torch": fake}):
            assert detect_cuda_version() == "12.4"

    def test_cuda_version_cpu_torch(self):
        fake = _fake_torch()
        fake.version = SimpleNamespace(cuda=None)
        with patch.dict(sys.modules, {"torch": fake}):
            assert detect_cuda_version() is None

    def test_pytorch_version_no_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert detect_pytorch_version() is None

    def test_pytorch_version_with_torch(self):
        fake = _fake_torch(torch_version="2.3.1+cu124")
        with patch.dict(sys.modules, {"torch": fake}):
            assert detect_pytorch_version() == "2.3.1+cu124"


# ---------------------------------------------------------------------------
# detect_attention_backend  — the most critical detector
# ---------------------------------------------------------------------------

class TestDetectAttentionBackend:

    # --- Inside ComfyUI path (comfy.model_management already in sys.modules) ---

    def test_comfy_mm_sage(self):
        mm = _fake_mm(sage=True)
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "sage"

    def test_comfy_mm_flash(self):
        mm = _fake_mm(flash=True)
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "flash"

    def test_comfy_mm_xformers(self):
        mm = _fake_mm(xformers=True)
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "xformers"

    def test_comfy_mm_vanilla(self):
        mm = _fake_mm()  # all False
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "vanilla"

    def test_comfy_mm_sage_beats_flash(self):
        mm = _fake_mm(sage=True, flash=True, xformers=True)
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "sage"

    def test_comfy_mm_flash_beats_xformers(self):
        mm = _fake_mm(flash=True, xformers=True)
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "flash"

    def test_comfy_mm_older_flash_api(self):
        """Older ComfyUI uses pytorch_attention_flash_attention instead of flash_attention_enabled."""
        mm = _fake_mm()
        mm.flash_attention_enabled.return_value = False
        mm.pytorch_attention_flash_attention.return_value = True
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "flash"

    def test_comfy_mm_missing_new_attributes(self):
        """mm with only the classic xformers_enabled attribute (no sage/flash attrs at all)."""
        mm = MagicMock(name="comfy.model_management", spec=["xformers_enabled"])
        mm.xformers_enabled.return_value = True
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            assert detect_attention_backend() == "xformers"

    # --- Outside ComfyUI: package probe fallback ---

    def _no_comfy(self):
        """Context: comfy.model_management not in sys.modules (or explicitly None)."""
        return patch.dict(sys.modules, {"comfy.model_management": None})

    def test_no_comfy_sageattention_installed(self):
        sage_mod = MagicMock()
        with self._no_comfy():
            with patch.dict(sys.modules, {"sageattention": sage_mod,
                                          "flash_attn": None, "xformers": None, "torch": None}):
                assert detect_attention_backend() == "sage"

    def test_no_comfy_flash_installed(self):
        flash_mod = MagicMock()
        with self._no_comfy():
            with patch.dict(sys.modules, {"sageattention": None,
                                          "flash_attn": flash_mod, "xformers": None, "torch": None}):
                assert detect_attention_backend() == "flash"

    def test_no_comfy_xformers_installed(self):
        xformers_mod = MagicMock()
        with self._no_comfy():
            with patch.dict(sys.modules, {"sageattention": None, "flash_attn": None,
                                          "xformers": xformers_mod, "torch": None}):
                assert detect_attention_backend() == "xformers"

    def test_no_comfy_torch_only_returns_vanilla(self):
        torch_mod = _fake_torch()
        with self._no_comfy():
            with patch.dict(sys.modules, {"sageattention": None, "flash_attn": None,
                                          "xformers": None, "torch": torch_mod}):
                assert detect_attention_backend() == "vanilla"

    def test_no_comfy_no_torch_returns_unknown(self):
        with self._no_comfy():
            with patch.dict(sys.modules, {"sageattention": None, "flash_attn": None,
                                          "xformers": None, "torch": None}):
                assert detect_attention_backend() == "unknown"

    def test_no_comfy_sage_beats_flash_in_fallback(self):
        sage_mod = MagicMock()
        flash_mod = MagicMock()
        with self._no_comfy():
            with patch.dict(sys.modules, {"sageattention": sage_mod,
                                          "flash_attn": flash_mod, "xformers": None, "torch": None}):
                assert detect_attention_backend() == "sage"


# ---------------------------------------------------------------------------
# detect_comfyui_version
# ---------------------------------------------------------------------------

class TestDetectComfyuiVersion:
    def test_importlib_metadata_path(self):
        with patch("core.env_detect.detect_comfyui_version.__wrapped__", None, create=True):
            pass  # just ensure importable
        # Patch importlib.metadata.version to return a string
        with patch("importlib.metadata.version", return_value="0.3.10"):
            result = detect_comfyui_version()
        assert result == "0.3.10"

    def test_comfy_dunder_version(self):
        fake_comfy = MagicMock()
        fake_comfy.__version__ = "0.3.10"
        # Make importlib.metadata raise PackageNotFoundError
        from importlib.metadata import PackageNotFoundError
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            with patch.dict(sys.modules, {"comfy": fake_comfy}):
                result = detect_comfyui_version()
        assert result == "0.3.10"

    def test_neither_available_returns_none(self):
        from importlib.metadata import PackageNotFoundError
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            with patch.dict(sys.modules, {"comfy": None}):
                result = detect_comfyui_version()
        assert result is None


# ---------------------------------------------------------------------------
# detect_ram_total
# ---------------------------------------------------------------------------

class TestDetectRamTotal:
    def test_no_psutil_returns_zero(self):
        with patch.dict(sys.modules, {"psutil": None}):
            assert detect_ram_total() == 0

    def test_with_psutil(self):
        mock_psutil = MagicMock(name="psutil")
        mock_psutil.virtual_memory.return_value = SimpleNamespace(total=64 * 1024**3)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = detect_ram_total()
        assert result == 64 * 1024  # 64 GB in MB


# ---------------------------------------------------------------------------
# detect_node_pack_versions
# ---------------------------------------------------------------------------

class TestDetectNodePackVersions:
    def test_none_returns_empty(self):
        assert detect_node_pack_versions(None) == {}

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert detect_node_pack_versions(tmp_path / "does_not_exist") == {}

    def test_pyproject_toml_pep621(self, tmp_path):
        pack = tmp_path / "WanVideoWrapper"
        pack.mkdir()
        (pack / "pyproject.toml").write_text(
            '[project]\nname = "WanVideoWrapper"\nversion = "1.2.3"\n'
        )
        result = detect_node_pack_versions(tmp_path)
        assert result["WanVideoWrapper"] == "1.2.3"

    def test_pyproject_toml_poetry(self, tmp_path):
        pack = tmp_path / "KJNodes"
        pack.mkdir()
        (pack / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "KJNodes"\nversion = "0.9.1"\n'
        )
        result = detect_node_pack_versions(tmp_path)
        assert result["KJNodes"] == "0.9.1"

    def test_git_tag_fallback(self, tmp_path):
        pack = tmp_path / "ComfyUI-Custom-Scripts"
        pack.mkdir()
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="v2.1.0\n", stderr=""
        )
        with patch("subprocess.run", return_value=completed):
            result = detect_node_pack_versions(tmp_path)
        assert result["ComfyUI-Custom-Scripts"] == "v2.1.0"

    def test_skips_dotfiles(self, tmp_path):
        (tmp_path / ".hidden_pack").mkdir()
        result = detect_node_pack_versions(tmp_path)
        assert ".hidden_pack" not in result

    def test_skips_files(self, tmp_path):
        (tmp_path / "not_a_pack.txt").write_text("hello")
        pack = tmp_path / "RealPack"
        pack.mkdir()
        (pack / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n')
        result = detect_node_pack_versions(tmp_path)
        assert "not_a_pack.txt" not in result
        assert result["RealPack"] == "1.0.0"

    def test_no_version_not_in_result(self, tmp_path):
        pack = tmp_path / "NoVersionPack"
        pack.mkdir()
        (pack / "README.md").write_text("nothing useful")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = detect_node_pack_versions(tmp_path)
        assert "NoVersionPack" not in result

    def test_multiple_packs(self, tmp_path):
        for name, version in [("PackA", "1.0"), ("PackB", "2.0"), ("PackC", "3.0")]:
            d = tmp_path / name
            d.mkdir()
            (d / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
        result = detect_node_pack_versions(tmp_path)
        assert len(result) == 3
        assert result["PackB"] == "2.0"


# ---------------------------------------------------------------------------
# capture_snapshot  — integration
# ---------------------------------------------------------------------------

class TestCaptureSnapshot:
    def test_all_absent_returns_safe_defaults(self):
        """With no packages at all, capture_snapshot must not raise."""
        absent = {k: None for k in [
            "torch", "psutil", "pynvml", "xformers", "flash_attn",
            "sageattention", "comfy", "comfy.model_management",
        ]}
        with patch.dict(sys.modules, absent):
            snap = capture_snapshot()

        assert isinstance(snap, EnvironmentSnapshot)
        assert snap.gpu_model == "CPU"
        assert snap.gpu_vram_total_mb == 0
        assert snap.gpu_driver_version is None
        assert snap.cuda_version is None
        assert snap.pytorch_version is None
        assert snap.attention_backend == "unknown"
        assert snap.ram_total_mb == 0
        assert snap.node_pack_versions == {}

    def test_full_happy_path(self, tmp_path):
        torch_mod = _fake_torch(device_name="RTX 4090", vram_bytes=24 * 1024**3)
        pynvml_mod = MagicMock(name="pynvml")
        pynvml_mod.nvmlSystemGetDriverVersion.return_value = "555.85"
        psutil_mod = MagicMock(name="psutil")
        psutil_mod.virtual_memory.return_value = SimpleNamespace(total=64 * 1024**3)
        mm = _fake_mm(flash=True)

        pack = tmp_path / "WanVideoWrapper"
        pack.mkdir()
        (pack / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')

        from importlib.metadata import PackageNotFoundError
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            fake_comfy = MagicMock()
            fake_comfy.__version__ = "0.3.10"
            with patch.dict(sys.modules, {
                "torch": torch_mod,
                "pynvml": pynvml_mod,
                "psutil": psutil_mod,
                "comfy": fake_comfy,            # needed for comfyui_version detection
                "comfy.model_management": mm,   # needed for attention_backend detection
            }):
                snap = capture_snapshot(custom_nodes_dir=tmp_path)

        assert snap.gpu_model == "RTX 4090"
        assert snap.gpu_vram_total_mb == 24 * 1024
        assert snap.gpu_driver_version == "555.85"
        assert snap.cuda_version == "12.4"
        assert snap.pytorch_version == "2.3.1+cu124"
        assert snap.comfyui_version == "0.3.10"
        assert snap.attention_backend == "flash"
        assert snap.ram_total_mb == 64 * 1024
        assert snap.node_pack_versions == {"WanVideoWrapper": "1.2.3"}
