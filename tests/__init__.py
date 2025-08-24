"""
Test Suite for Mixed-Precision Multigrid Solvers

This package contains comprehensive tests for the mixed-precision multigrid
solver framework, including unit tests, integration tests, and performance
regression tests.

Test Categories:
    - Unit tests: Individual component testing
    - Integration tests: End-to-end pipeline testing
    - Performance tests: Benchmark and regression testing
    - Platform tests: Multi-platform compatibility
    - Validation tests: Numerical accuracy verification
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path for imports
test_dir = Path(__file__).parent
src_dir = test_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

# Test configuration
TEST_CONFIG = {
    'tolerance': {
        'single_precision': 1e-6,
        'double_precision': 1e-12,
        'mixed_precision': 1e-9
    },
    'grid_sizes': [17, 33, 65, 129],
    'max_iterations': 100,
    'convergence_threshold': 1e-10
}

# Test data generators
def generate_poisson_problem(nx=65, ny=65):
    """Generate standard Poisson test problem."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Manufactured solution: u = sin(πx)sin(πy)
    exact_solution = np.sin(np.pi * X) * np.sin(np.pi * Y)
    source_term = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    return X, Y, exact_solution, source_term

def generate_heat_problem(nx=65, ny=65, nt=10):
    """Generate time-dependent heat equation test problem."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 0.1, nt)
    
    X, Y = np.meshgrid(x, y)
    
    # Initial condition: Gaussian
    initial = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))
    
    return X, Y, t, initial

__all__ = ['TEST_CONFIG', 'generate_poisson_problem', 'generate_heat_problem']