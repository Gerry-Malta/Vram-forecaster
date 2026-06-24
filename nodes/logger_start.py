"""
VRAM Forecaster — Logger Start node.

Place this near the beginning of the measured region of a workflow (typically
right before the loaders or the sampler).  It:

  * detects the runtime environment (GPU, attention backend, versions),
  * records the workflow settings you declare on the node,
  * resets the VRAM high-water mark and starts RAM sampling,
  * opens a session that must be wired into the Logger End node.

The session output is what forces Logger End to run after the work, and the
`passthrough` socket lets you splice the node into an existing wire without
changing the graph's data flow.
"""

# Normal path: loaded by ComfyUI as a package.
try:
    from ..core.env_detect import capture_snapshot
    from ..core.session import build_record_from_snapshot, open_session
    from ..core.db import ModelInfo
except ImportError:
    # Standalone / test path: core is importable as a top-level package.
    from core.env_detect import capture_snapshot
    from core.session import build_record_from_snapshot, open_session
    from core.db import ModelInfo

from .common import ANY, SESSION_TYPE, DEFAULT_DB_PATH, DEFAULT_CUSTOM_NODES_DIR

QUANT_TYPES = ["fp16", "bf16", "fp8", "gguf", "bnb4", "other"]


class VramForecasterLoggerStart:
    """Opens a measurement session and emits it to the Logger End node."""

    CATEGORY = "VRAM Forecaster"
    FUNCTION = "start"
    RETURN_TYPES = (ANY, SESSION_TYPE)
    RETURN_NAMES = ("passthrough", "session")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_w": ("INT", {"default": 896, "min": 0, "max": 16384}),
                "resolution_h": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "frame_count": ("INT", {"default": 81, "min": 0, "max": 100000}),
                "loop_iteration_count": ("INT", {"default": 1, "min": 0, "max": 100000}),
                "block_swap_count": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "lora_count": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "lora_total_weight_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6}),
            },
            "optional": {
                "passthrough": (ANY, {}),
                "model_name": ("STRING", {"default": ""}),
                "quant_type": (QUANT_TYPES, {"default": "fp16"}),
                "model_size_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute: every run must take a fresh measurement, even when
        # the declared settings are identical to the previous run.
        return float("nan")

    def start(
        self,
        resolution_w,
        resolution_h,
        frame_count,
        loop_iteration_count,
        block_swap_count,
        lora_count,
        lora_total_weight_mb,
        passthrough=None,
        model_name="",
        quant_type="fp16",
        model_size_mb=0.0,
    ):
        snapshot = capture_snapshot(custom_nodes_dir=DEFAULT_CUSTOM_NODES_DIR)

        models = []
        if model_name.strip():
            models.append(
                ModelInfo(name=model_name.strip(), quant_type=quant_type, size_mb=model_size_mb)
            )

        record = build_record_from_snapshot(
            snapshot,
            resolution_w=resolution_w,
            resolution_h=resolution_h,
            frame_count=frame_count,
            loop_iteration_count=loop_iteration_count,
            block_swap_count=block_swap_count,
            lora_count=lora_count,
            lora_total_weight_mb=lora_total_weight_mb,
            models=models,
        )

        session = open_session(record, snapshot, db_path=DEFAULT_DB_PATH)

        print(
            f"[VRAM Forecaster] Run #{record.id} started on {snapshot.gpu_model} "
            f"(attention={snapshot.attention_backend}, "
            f"{resolution_w}x{resolution_h}, {frame_count} frames, "
            f"block_swap={block_swap_count}, loras={lora_count})"
        )

        return (passthrough, session)
