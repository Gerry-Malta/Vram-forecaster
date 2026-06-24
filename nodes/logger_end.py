"""
VRAM Forecaster — Logger End node.

Place this at the end of the measured region (typically right before the final
Save node).  Wire the Logger Start node's `session` output into this node's
`session` input; that wire is what guarantees this node runs after the work
being measured.

On execution it stops the RAM sampler, reads the peak VRAM, and writes the
finished row to the database with outcome='ok'.  Failures that crash the
workflow never reach this node — those are caught by the session's atexit
flush and by reap_stale_runs() on the next run (see core/session.py).
"""

try:
    from ..core.session import ForecastSession
except ImportError:
    from core.session import ForecastSession

from .common import ANY, SESSION_TYPE


class VramForecasterLoggerEnd:
    """Closes a measurement session and records the observed peaks."""

    CATEGORY = "VRAM Forecaster"
    FUNCTION = "end"
    RETURN_TYPES = (ANY, "FLOAT", "FLOAT")
    RETURN_NAMES = ("passthrough", "vram_peak_mb", "ram_peak_mb")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE, {}),
            },
            "optional": {
                "passthrough": (ANY, {}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Must re-run every execution to take a fresh end-of-run measurement.
        return float("nan")

    def end(self, session, passthrough=None):
        if not isinstance(session, ForecastSession):
            print("[VRAM Forecaster] Logger End received no valid session; skipping.")
            return (passthrough, 0.0, 0.0)

        session.close(outcome="ok")

        rec = session.record
        vram = rec.vram_peak_mb if rec.vram_peak_mb is not None else 0.0
        ram = rec.ram_peak_mb if rec.ram_peak_mb is not None else 0.0

        total = rec.gpu_vram_total_mb or 0
        pct = (vram / total * 100.0) if total else 0.0
        print(
            f"[VRAM Forecaster] Run #{rec.id} OK in "
            f"{rec.duration_seconds:.1f}s — VRAM peak {vram:.0f} MB"
            f"{f' ({pct:.0f}% of {total} MB)' if total else ''}, "
            f"RAM peak {ram:.0f} MB"
        )

        return (passthrough, float(vram), float(ram))
