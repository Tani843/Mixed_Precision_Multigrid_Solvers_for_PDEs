"""Applications module for specific PDE problems."""

from .poisson_solver import PoissonSolver2D, PoissonSolver3D
from .heat_solver import HeatSolver2D, HeatSolver3D, TimeSteppingConfig, TimeSteppingMethod
from .test_problems import PoissonTestProblems, HeatTestProblems, BenchmarkProblems
from .validation import ValidationSuite, MethodOfManufacturedSolutions
from .convergence_analysis import GridConvergenceAnalyzer, ConvergenceData
from .performance_analysis import PerformanceAnalyzer, ScalabilityAnalyzer, run_comprehensive_performance_analysis
from .mixed_precision_analysis import MixedPrecisionAnalyzer, run_comprehensive_mixed_precision_analysis
from .comprehensive_validation import ComprehensiveValidationSuite, run_comprehensive_validation, run_quick_validation
from .integration_tests import SystemIntegrationTests, run_complete_system_demonstration, run_quick_integration_tests

__all__ = [
    # Core solvers
    "PoissonSolver2D",
    "PoissonSolver3D", 
    "HeatSolver2D",
    "HeatSolver3D",
    "TimeSteppingConfig",
    "TimeSteppingMethod",
    
    # Test problems and benchmarks
    "PoissonTestProblems",
    "HeatTestProblems",
    "BenchmarkProblems",
    
    # Validation and analysis tools
    "ValidationSuite",
    "MethodOfManufacturedSolutions",
    "GridConvergenceAnalyzer",
    "ConvergenceData",
    "PerformanceAnalyzer",
    "ScalabilityAnalyzer",
    "MixedPrecisionAnalyzer",
    "ComprehensiveValidationSuite",
    "SystemIntegrationTests",
    
    # High-level analysis functions
    "run_comprehensive_performance_analysis",
    "run_comprehensive_mixed_precision_analysis",
    "run_comprehensive_validation",
    "run_quick_validation",
    "run_complete_system_demonstration",
    "run_quick_integration_tests"
]