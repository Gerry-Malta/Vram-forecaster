"""Shared helpers for the VRAM Forecaster ComfyUI nodes."""

from pathlib import Path


class AnyType(str):
    """A type that compares equal to every other type.

    This is the well-established ComfyUI pattern (used by pysssss, KJNodes,
    etc.) for a wildcard socket: it lets the logger nodes sit anywhere in a
    graph and pass any value straight through, so they can wrap an arbitrary
    pipeline without caring whether the wire carries a MODEL, LATENT, IMAGE…
    """

    def __ne__(self, _other: object) -> bool:
        return False

    def __eq__(self, _other: object) -> bool:
        return True

    def __hash__(self) -> int:  # keep it hashable since we subclass str
        return hash(str(self))


ANY = AnyType("*")

# Custom socket type carrying the live ForecastSession from start -> end.
SESSION_TYPE = "VRAMFORECAST_SESSION"

# Default location of the SQLite database, alongside the package's data dir.
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "forecaster.db"

# The custom_nodes directory is the parent of this package's own folder.
DEFAULT_CUSTOM_NODES_DIR = Path(__file__).parent.parent.parent
