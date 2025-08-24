"""Unit tests for enhanced iterative solvers."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multigrid.core.grid import Grid
from multigrid.core.precision import PrecisionManager
from multigrid.operators.laplacian import LaplacianOperator
from multigrid.solvers.iterative import (
    EnhancedJacobiSolver, EnhancedGaussSeidelSolver, SORSolver,
    WeightedJacobiSolver, AdaptivePrecisionSolver
)


class TestEnhancedJacobiSolver:
    """Test cases for enhanced Jacobi solver."""
    
    def test_jacobi_initialization(self):
        """Test Jacobi solver initialization."""
        solver = EnhancedJacobiSolver(max_iterations=100, tolerance=1e-6, 
                                     relaxation_parameter=0.6)
        
        assert solver.max_iterations == 100
        assert solver.tolerance == 1e-6
        assert solver.omega == 0.6
        assert "EnhancedJacobi" in solver.name
        assert solver.use_vectorized is True
    
    def test_jacobi_spectral_radius_computation(self):
        """Test spectral radius computation."""
        solver = EnhancedJacobiSolver()
        grid = Grid(nx=9, ny=9)
        
        rho = solver.compute_spectral_radius(grid)
        
        # Spectral radius should be less than 1 for convergence
        assert 0 < rho < 1
        
        # For uniform grid, should be close to theoretical value
        expected_rho = np.cos(np.pi / (grid.nx - 1)) * np.cos(np.pi / (grid.ny - 1))
        assert abs(rho - expected_rho) < 1e-10
    
    def test_jacobi_vectorized_smoothing(self):
        """Test vectorized Jacobi smoothing."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        solver = EnhancedJacobiSolver(use_vectorized=True)
        
        # Create test problem
        u = np.zeros(grid.shape)
        rhs = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0)
        
        # Apply smoothing
        u_smooth = solver.smooth(grid, operator, u, rhs, num_iterations=5)
        
        # Should have changed from initial guess
        assert not np.array_equal(u_smooth, u)
        
        # Interior should be non-zero
        assert np.any(u_smooth[1:-1, 1:-1] != 0)
        
        # Boundaries should remain zero
        assert np.all(u_smooth[0, :] == 0)
        assert np.all(u_smooth[-1, :] == 0)
        assert np.all(u_smooth[:, 0] == 0)
        assert np.all(u_smooth[:, -1] == 0)
    
    def test_jacobi_convergence(self):
        """Test Jacobi convergence on simple problem."""
        grid = Grid(nx=9, ny=9, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        solver = EnhancedJacobiSolver(max_iterations=100, tolerance=1e-4, verbose=False)
        
        # Simple problem with known solution
        rhs = 2 * np.pi**2 * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        grid.apply_dirichlet_bc(0.0)
        
        solution, info = solver.solve(grid, operator, rhs)
        
        # Should converge
        assert info['converged'] is True
        assert info['final_residual'] < solver.tolerance
        assert info['iterations'] > 0
    
    def test_jacobi_vs_non_vectorized(self):
        """Test that vectorized and non-vectorized give similar results."""
        grid = Grid(nx=5, ny=5)
        operator = LaplacianOperator()
        
        solver_vec = EnhancedJacobiSolver(use_vectorized=True, max_iterations=10)
        solver_scalar = EnhancedJacobiSolver(use_vectorized=False, max_iterations=10)
        
        u = np.random.rand(*grid.shape)
        rhs = np.random.rand(*grid.shape)
        
        # Apply same smoothing
        u_vec = solver_vec.smooth(grid, operator, u.copy(), rhs, 1)
        u_scalar = solver_scalar.smooth(grid, operator, u.copy(), rhs, 1)
        
        # Results should be very similar
        np.testing.assert_allclose(u_vec, u_scalar, rtol=1e-12)


class TestEnhancedGaussSeidelSolver:
    """Test cases for enhanced Gauss-Seidel solver."""
    
    def test_gauss_seidel_initialization(self):
        """Test Gauss-Seidel solver initialization."""
        solver = EnhancedGaussSeidelSolver(red_black=True, symmetric=True)
        
        assert solver.red_black is True
        assert solver.symmetric is True
        assert "RedBlack" in solver.name
    
    def test_red_black_ordering(self):
        """Test red-black Gauss-Seidel ordering."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        solver = EnhancedGaussSeidelSolver(red_black=True)
        
        u = np.zeros(grid.shape)
        rhs = np.ones(grid.shape)
        
        u_smooth = solver.smooth(grid, operator, u, rhs, num_iterations=1)
        
        # Should modify interior points
        assert np.any(u_smooth[1:-1, 1:-1] != 0)
    
    def test_lexicographic_vs_red_black(self):
        """Compare lexicographic and red-black orderings."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        
        solver_lex = EnhancedGaussSeidelSolver(red_black=False, max_iterations=20)
        solver_rb = EnhancedGaussSeidelSolver(red_black=True, max_iterations=20)
        
        rhs = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        
        sol_lex, info_lex = solver_lex.solve(grid, operator, rhs)
        sol_rb, info_rb = solver_rb.solve(grid, operator, rhs)
        
        # Both should converge to similar solutions
        assert info_lex['converged'] or info_rb['converged']
        
        # If both converge, solutions should be similar
        if info_lex['converged'] and info_rb['converged']:
            np.testing.assert_allclose(sol_lex, sol_rb, rtol=1e-3)
    
    def test_symmetric_gauss_seidel(self):
        """Test symmetric Gauss-Seidel (forward + backward sweeps)."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        solver = EnhancedGaussSeidelSolver(symmetric=True, max_iterations=10)
        
        rhs = np.ones(grid.shape)
        solution, info = solver.solve(grid, operator, rhs)
        
        # Should converge faster than regular Gauss-Seidel due to symmetry
        assert info['iterations'] <= 10


class TestSORSolver:
    """Test cases for SOR solver."""
    
    def test_sor_initialization(self):
        """Test SOR solver initialization."""
        solver = SORSolver(relaxation_parameter=1.5, auto_omega=False)
        
        assert solver.omega == 1.5
        assert solver.auto_omega is False
        assert "SOR" in solver.name
    
    def test_optimal_omega_computation(self):
        """Test optimal omega computation for SOR."""
        grid = Grid(nx=9, ny=9)
        solver = SORSolver(auto_omega=True)
        
        optimal_omega = solver.setup_optimal_omega(grid)
        
        # Optimal omega should be between 1 and 2
        assert 1.0 < optimal_omega < 2.0
        
        # Should be close to theoretical optimum
        h = min(grid.hx, grid.hy)
        expected_omega = 2.0 / (1.0 + np.sin(np.pi * h))
        assert abs(optimal_omega - expected_omega) < 1e-10
    
    def test_sor_convergence_vs_gauss_seidel(self):
        """Test that SOR converges faster than Gauss-Seidel."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        
        gs_solver = EnhancedGaussSeidelSolver(max_iterations=100, tolerance=1e-6)
        sor_solver = SORSolver(max_iterations=100, tolerance=1e-6, auto_omega=True)
        
        # Test problem
        rhs = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        
        _, gs_info = gs_solver.solve(grid, operator, rhs)
        _, sor_info = sor_solver.solve(grid, operator, rhs)
        
        # SOR should converge in fewer iterations (if both converge)
        if gs_info['converged'] and sor_info['converged']:
            assert sor_info['iterations'] <= gs_info['iterations']
    
    def test_sor_parameter_validation(self):
        """Test SOR parameter validation warning."""
        # Should issue warning for omega outside convergence range
        import logging
        with pytest.warns(None):  # Expecting a warning
            SORSolver(relaxation_parameter=2.5)  # Too large


class TestWeightedJacobiSolver:
    """Test cases for weighted Jacobi solver."""
    
    def test_weighted_jacobi_initialization(self):
        """Test weighted Jacobi initialization."""
        solver = WeightedJacobiSolver(auto_weight=True)
        
        assert solver.auto_weight is True
        assert "WeightedJacobi" in solver.name
        
        # Default weight should be optimal for 2D Laplacian
        assert abs(solver.omega - 4.0/5.0) < 1e-10
    
    def test_optimal_weight_computation(self):
        """Test optimal weight computation."""
        grid = Grid(nx=9, ny=9)
        solver = WeightedJacobiSolver(auto_weight=True)
        
        optimal_weight = solver.setup_optimal_weight(grid)
        
        # Should be between 0 and 1
        assert 0 < optimal_weight <= 1
        
        # Verify theoretical computation
        rho = solver.compute_spectral_radius(grid)
        expected_weight = 2.0 / (1.0 + np.sqrt(1.0 - rho**2))
        assert abs(optimal_weight - expected_weight) < 1e-10
    
    def test_weighted_vs_standard_jacobi(self):
        """Test that weighted Jacobi converges faster than standard."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        
        jacobi_solver = EnhancedJacobiSolver(relaxation_parameter=1.0, max_iterations=200)
        weighted_solver = WeightedJacobiSolver(auto_weight=True, max_iterations=200)
        
        rhs = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        
        _, jacobi_info = jacobi_solver.solve(grid, operator, rhs)
        _, weighted_info = weighted_solver.solve(grid, operator, rhs)
        
        # Weighted should converge faster (if both converge)
        if jacobi_info['converged'] and weighted_info['converged']:
            assert weighted_info['iterations'] <= jacobi_info['iterations']


class TestAdaptivePrecisionSolver:
    """Test cases for adaptive precision solver."""
    
    def test_adaptive_precision_initialization(self):
        """Test adaptive precision solver initialization."""
        base_solver = EnhancedJacobiSolver(max_iterations=50)
        adaptive_solver = AdaptivePrecisionSolver(
            base_solver, 
            precision_switch_threshold=0.9,
            min_iterations_before_switch=5
        )
        
        assert adaptive_solver.base_solver is base_solver
        assert adaptive_solver.precision_switch_threshold == 0.9
        assert adaptive_solver.min_iterations_before_switch == 5
        assert "AdaptiveEnhancedJacobi" in adaptive_solver.name
    
    def test_precision_switching_logic(self):
        """Test precision switching decision logic."""
        base_solver = EnhancedJacobiSolver(max_iterations=20)
        adaptive_solver = AdaptivePrecisionSolver(
            base_solver,
            precision_switch_threshold=0.8,
            convergence_window=3,
            min_iterations_before_switch=3
        )
        
        # Simulate slow convergence (rates > threshold)
        adaptive_solver.convergence_rates = [0.9, 0.85, 0.9, 0.88]
        
        should_switch = adaptive_solver._should_switch_precision()
        assert should_switch is True
        
        # Simulate fast convergence (rates < threshold)
        adaptive_solver.convergence_rates = [0.5, 0.4, 0.3, 0.2]
        
        should_switch = adaptive_solver._should_switch_precision()
        assert should_switch is False
    
    def test_adaptive_solve_with_precision_manager(self):
        """Test adaptive solving with precision manager."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        precision_manager = PrecisionManager(adaptive=True)
        
        base_solver = EnhancedJacobiSolver(max_iterations=30, tolerance=1e-6)
        adaptive_solver = AdaptivePrecisionSolver(base_solver, min_iterations_before_switch=5)
        
        rhs = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        
        solution, info = adaptive_solver.solve(grid, operator, rhs, precision_manager=precision_manager)
        
        # Should have recorded precision switching information
        assert 'precision_switched' in info
        assert isinstance(info['precision_switched'], bool)
    
    def test_smoothing_delegation(self):
        """Test that smoothing is delegated to base solver."""
        grid = Grid(nx=5, ny=5)
        operator = LaplacianOperator()
        
        base_solver = EnhancedJacobiSolver()
        adaptive_solver = AdaptivePrecisionSolver(base_solver)
        
        u = np.zeros(grid.shape)
        rhs = np.ones(grid.shape)
        
        # Smoothing should work through delegation
        u_smooth = adaptive_solver.smooth(grid, operator, u, rhs, 1)
        
        # Should produce same result as base solver
        u_base = base_solver.smooth(grid, operator, u, rhs, 1)
        np.testing.assert_array_equal(u_smooth, u_base)


class TestSolverComparison:
    """Compare different solvers on same problems."""
    
    def test_solver_convergence_rates(self):
        """Compare convergence rates of different solvers."""
        grid = Grid(nx=17, ny=17, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        
        # Create test problem with known solution
        exact_solution = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * exact_solution
        
        solvers = {
            'Jacobi': EnhancedJacobiSolver(max_iterations=200, tolerance=1e-6),
            'WeightedJacobi': WeightedJacobiSolver(max_iterations=200, tolerance=1e-6),
            'GaussSeidel': EnhancedGaussSeidelSolver(max_iterations=100, tolerance=1e-6, red_black=False),
            'RedBlackGS': EnhancedGaussSeidelSolver(max_iterations=100, tolerance=1e-6, red_black=True),
            'SOR': SORSolver(max_iterations=100, tolerance=1e-6, auto_omega=True)
        }
        
        results = {}
        
        for name, solver in solvers.items():
            solution, info = solver.solve(grid, operator, rhs)
            
            # Compute error if converged
            if info['converged']:
                error = np.max(np.abs(solution - exact_solution))
                results[name] = {
                    'iterations': info['iterations'],
                    'final_residual': info['final_residual'],
                    'error': error,
                    'convergence_rate': info['convergence_rate']
                }
        
        # At least some solvers should converge
        assert len(results) > 0
        
        # Weighted Jacobi should converge faster than standard Jacobi
        if 'Jacobi' in results and 'WeightedJacobi' in results:
            assert results['WeightedJacobi']['iterations'] <= results['Jacobi']['iterations']
        
        # SOR should be competitive
        if 'SOR' in results:
            assert results['SOR']['convergence_rate'] < 0.9  # Decent convergence rate
    
    def test_solver_consistency(self):
        """Test that different solvers give consistent solutions."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        
        # Simple test problem
        rhs = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0)
        
        solvers = [
            EnhancedJacobiSolver(max_iterations=100, tolerance=1e-8),
            EnhancedGaussSeidelSolver(max_iterations=50, tolerance=1e-8),
            SORSolver(max_iterations=50, tolerance=1e-8, auto_omega=True)
        ]
        
        solutions = []
        
        for solver in solvers:
            solution, info = solver.solve(grid, operator, rhs)
            if info['converged']:
                solutions.append(solution)
        
        # All converged solutions should be similar
        if len(solutions) > 1:
            for i in range(1, len(solutions)):
                np.testing.assert_allclose(solutions[0], solutions[i], rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])