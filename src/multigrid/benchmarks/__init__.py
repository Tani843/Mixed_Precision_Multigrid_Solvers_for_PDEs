"""
Benchmarking Module for Mixed-Precision Multigrid Solvers

This module provides comprehensive benchmarking tools for validating and analyzing
the performance of mixed-precision multigrid methods for solving PDEs.

Classes:
    PerformanceBenchmark: Main benchmarking framework
    ValidationSuite: Comprehensive validation testing
    ScalingAnalysis: Parallel scaling studies
    
Functions:
    run_quick_benchmark: Quick performance validation
    generate_validation_report: Comprehensive validation report
"""

from .performance_benchmark import PerformanceBenchmark, run_quick_benchmark
from .validation_suite import ValidationSuite, run_comprehensive_validation
from .scaling_analysis import ScalingAnalysis, run_scaling_study

__all__ = [
    'PerformanceBenchmark',
    'ValidationSuite', 
    'ScalingAnalysis',
    'run_quick_benchmark',
    'run_comprehensive_validation',
    'run_scaling_study'
]