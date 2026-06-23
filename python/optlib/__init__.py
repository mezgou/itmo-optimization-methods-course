"""Python entry point for the optlib C++ extension."""

from __future__ import annotations

from ._optlib import Add, Version

__version__ = Version()
__all__ = ["Add", "Version", "__version__"]
