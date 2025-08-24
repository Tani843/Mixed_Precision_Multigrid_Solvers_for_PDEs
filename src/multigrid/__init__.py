"""
Mixed-Precision Multigrid Solvers for PDEs

A high-performance computational mathematics library for solving partial 
differential equations using multigrid methods with adaptive precision and GPU acceleration.
"""

# Version information
from ._version import __version__
__author__ = "Tanisha Gupta"

from .core import Grid, PrecisionManager
from .operators import LaplacianOperator, RestrictionOperator, ProlongationOperator
from .solvers import (
    MultigridSolver, EnhancedJacobiSolver, EnhancedGaussSeidelSolver, 
    SORSolver, WeightedJacobiSolver, AdaptivePrecisionSolver
)
from .preconditioning import (
    DiagonalPreconditioner, ILUPreconditioner, MultigridPreconditioner
)

# Try to import GPU modules
try:
    from .gpu import (
        GPUMemoryManager, GPUMultigridSolver, GPUCommunicationAvoidingMultigrid,
        GPUPrecisionManager, GPUPerformanceProfiler, GPUBenchmarkSuite
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    "Grid",
    "PrecisionManager", 
    "LaplacianOperator",
    "RestrictionOperator", 
    "ProlongationOperator",
    "MultigridSolver",
    "EnhancedJacobiSolver",
    "EnhancedGaussSeidelSolver",
    "SORSolver", 
    "WeightedJacobiSolver",
    "AdaptivePrecisionSolver",
    "DiagonalPreconditioner",
    "ILUPreconditioner",
    "MultigridPreconditioner",
    "GPU_AVAILABLE"
]

# Add GPU modules to __all__ if available
if GPU_AVAILABLE:
    __all__.extend([
        "GPUMemoryManager",
        "GPUMultigridSolver", 
        "GPUCommunicationAvoidingMultigrid",
        "GPUPrecisionManager",
        "GPUPerformanceProfiler",
        "GPUBenchmarkSuite"
    ])