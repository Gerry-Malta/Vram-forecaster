"""ComfyUI node registration for VRAM Forecaster."""

from .logger_start import VramForecasterLoggerStart
from .logger_end import VramForecasterLoggerEnd

NODE_CLASS_MAPPINGS = {
    "VramForecasterLoggerStart": VramForecasterLoggerStart,
    "VramForecasterLoggerEnd": VramForecasterLoggerEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VramForecasterLoggerStart": "VRAM Forecaster — Logger Start",
    "VramForecasterLoggerEnd": "VRAM Forecaster — Logger End",
}
