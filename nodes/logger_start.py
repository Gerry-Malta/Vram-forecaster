"""
VRAM Forecaster — Logger Start node.

Place this near the beginning of the measured region of a workflow (typically
right before the loaders or the sampler).  It:

  * detects the runtime environment (GPU, attention backend, versions) fully
    automatically — you do nothing for that part,
  * reads resolution / frame_count from a wired latent tensor when provided,
  * lets you optionally fill in or wire the remaining workflow settings
    (block_swap, LoRA count, model info) for richer predictions,
  * resets the VRAM high-water mark and starts RAM sampling,
  * opens a session that must be wired into the Logger End node.

Minimal setup (zero manual input):
  Wire any latent through `passthrough` and connect `session` to Logger End.
  Resolution auto-reads from the latent shape; everything else is optional.

Full setup:
  Also wire `latent` directly (for reliable resolution/frame detection even
  when passthrough carries a different type), and fill in block_swap_count,
  lora_count etc. as widgets or wire them from the nodes that own those values.
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


def _resolution_from_latent(latent) -> tuple[int | None, int | None, int | None]:
    """Extract (w, h, frames) from a ComfyUI latent dict or raw tensor.

    ComfyUI latents are dicts: {"samples": tensor[B, C, H, W]} for images or
    {"samples": tensor[B, C, F, H, W]} for video.  Pixel dimensions are 8×
    the latent spatial dims (VAE factor).

    Returns (None, None, None) when the latent is absent or has an unexpected
    shape — the caller treats None values as "not recorded".
    """
    if latent is None:
        return None, None, None

    try:
        import torch

        # Support both raw tensors and the {"samples": tensor} dict.
        if isinstance(latent, dict):
            tensor = latent.get("samples")
        elif isinstance(latent, torch.Tensor):
            tensor = latent
        else:
            return None, None, None

        if tensor is None:
            return None, None, None

        shape = tensor.shape  # (B, C, [F,] H, W)
        if len(shape) == 4:          # image latent: B C H W
            _, _, lh, lw = shape
            return lw * 8, lh * 8, None
        elif len(shape) == 5:        # video latent: B C F H W
            _, _, lf, lh, lw = shape
            return lw * 8, lh * 8, int(lf)
    except Exception:
        pass

    return None, None, None


class VramForecasterLoggerStart:
    """Opens a measurement session and emits it to Logger End."""

    CATEGORY = "VRAM Forecaster"
    FUNCTION = "start"
    RETURN_TYPES = (ANY, SESSION_TYPE)
    RETURN_NAMES = ("passthrough", "session")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # ---- Wire-through / auto-detection ----
                "passthrough": (ANY, {}),
                "latent": ("LATENT", {}),           # resolution auto-read from shape

                # ---- Override / supplement auto-detected values ----
                # These show as widgets (manual entry) but can all be
                # right-clicked → "Convert to Input" to wire from the node
                # that actually owns them (e.g. wire frame_count from your
                # forLoop or video node).
                "resolution_w": ("INT", {"default": 0, "min": 0, "max": 16384,
                                         "tooltip": "0 = auto-read from latent"}),
                "resolution_h": ("INT", {"default": 0, "min": 0, "max": 16384,
                                         "tooltip": "0 = auto-read from latent"}),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 100000,
                                        "tooltip": "0 = auto-read from video latent"}),
                "loop_iteration_count": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "block_swap_count": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "lora_count": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "lora_total_weight_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6}),

                # ---- Main model info (best-effort; fill what you know) ----
                "model_name": ("STRING", {"default": ""}),
                "quant_type": (QUANT_TYPES, {"default": "fp16"}),
                "model_size_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute: every run must take a fresh measurement even when
        # the declared settings are identical to the previous run.
        return float("nan")

    def start(
        self,
        passthrough=None,
        latent=None,
        resolution_w=0,
        resolution_h=0,
        frame_count=0,
        loop_iteration_count=0,
        block_swap_count=0,
        lora_count=0,
        lora_total_weight_mb=0.0,
        model_name="",
        quant_type="fp16",
        model_size_mb=0.0,
    ):
        snapshot = capture_snapshot(custom_nodes_dir=DEFAULT_CUSTOM_NODES_DIR)

        # Auto-read resolution / frame count from the latent when the widgets
        # are left at 0 (their "not set" sentinel).
        auto_w, auto_h, auto_f = _resolution_from_latent(latent)
        if latent is None and passthrough is not None:
            # Passthrough might carry a latent dict when the user doesn't wire
            # the dedicated latent socket — try it too.
            auto_w, auto_h, auto_f = _resolution_from_latent(passthrough)

        final_w = resolution_w if resolution_w > 0 else auto_w
        final_h = resolution_h if resolution_h > 0 else auto_h
        final_f = frame_count if frame_count > 0 else auto_f

        models = []
        if model_name.strip():
            models.append(
                ModelInfo(name=model_name.strip(), quant_type=quant_type, size_mb=model_size_mb)
            )

        record = build_record_from_snapshot(
            snapshot,
            resolution_w=final_w,
            resolution_h=final_h,
            frame_count=final_f,
            loop_iteration_count=loop_iteration_count or None,
            block_swap_count=block_swap_count or None,
            lora_count=lora_count,
            lora_total_weight_mb=lora_total_weight_mb or None,
            models=models,
        )

        session = open_session(record, snapshot, db_path=DEFAULT_DB_PATH)

        res_str = f"{final_w}x{final_h}" if final_w and final_h else "res=?"
        f_str = f", {final_f}f" if final_f else ""
        print(
            f"[VRAM Forecaster] Run #{record.id} started on {snapshot.gpu_model} "
            f"(attention={snapshot.attention_backend}, {res_str}{f_str}, "
            f"block_swap={block_swap_count}, loras={lora_count})"
        )

        return (passthrough, session)
