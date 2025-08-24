"""Multigrid solvers and iterative methods."""

from .base import BaseSolver
from .multigrid import MultigridSolver
from .smoothers import GaussSeidelSmoother, JacobiSmoother
from .iterative import (
    EnhancedJacobiSolver, EnhancedGaussSeidelSolver, SORSolver,
    WeightedJacobiSolver, AdaptivePrecisionSolver
)

__all__ = [
    "BaseSolver", 
    "MultigridSolver", 
    "GaussSeidelSmoother", 
    "JacobiSmoother",
    "EnhancedJacobiSolver",
    "EnhancedGaussSeidelSolver", 
    "SORSolver",
    "WeightedJacobiSolver",
    "AdaptivePrecisionSolver"
]