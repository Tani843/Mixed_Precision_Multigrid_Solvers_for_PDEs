"""Core mathematical abstractions for multigrid methods."""

from .grid import Grid
from .precision import PrecisionManager

__all__ = ["Grid", "PrecisionManager"]