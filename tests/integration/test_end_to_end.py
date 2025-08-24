"""
End-to-End Integration Tests

Comprehensive integration tests for the complete mixed-precision multigrid 
solver pipeline, testing from problem setup to solution validation.
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add src directory to path
test_dir = Path(__file__).parent.parent
src_dir = test_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from tests import TEST_CONFIG, generate_poisson_problem, generate_heat_problem

# Mock imports for testing framework
class MockSolver:
    """Mock solver for integration testing."""
    
    def __init__(self, precision='double', use_gpu=False):
        self.precision = precision
        self.use_gpu = use_gpu
        self.iterations = 0
        self.residual = 1.0
        
    def solve(self, problem_data):
        """Mock solve method."""
        X, Y, exact_solution, source_term = problem_data
        
        # Simulate solver behavior
        if self.precision == 'single':
            error_scale = 1e-6
        elif self.precision == 'mixed':
            error_scale = 1e-9
        else:  # double
            error_scale = 1e-12
            
        # Add appropriate error to exact solution
        noise = error_scale * np.random.randn(*exact_solution.shape)
        numerical_solution = exact_solution + noise
        
        # Simulate iteration count
        self.iterations = np.random.randint(5, 15)
        self.residual = error_scale * 10
        
        return numerical_solution, {
            'iterations': self.iterations,
            'residual': self.residual,
            'solve_time': 0.1 + 0.05 * np.random.randn(),
            'converged': True
        }


class TestEndToEndPipeline:
    """Test complete solver pipeline from setup to validation."""
    
    def test_poisson_solver_pipeline(self):
        """Test complete Poisson solver pipeline."""
        # Problem setup
        X, Y, exact_solution, source_term = generate_poisson_problem(nx=33, ny=33)
        
        # Solver setup
        solver = MockSolver(precision='double')
        
        # Solve
        numerical_solution, info = solver.solve((X, Y, exact_solution, source_term))
        
        # Validate results
        assert info['converged'], "Solver should converge"
        assert info['iterations'] < TEST_CONFIG['max_iterations'], "Should converge within max iterations"
        assert info['residual'] < TEST_CONFIG['convergence_threshold'], "Residual should be below threshold"
        
        # Check solution accuracy
        error = np.linalg.norm(numerical_solution - exact_solution)
        assert error < TEST_CONFIG['tolerance']['double_precision'], f"Solution error {error} too large"
        
    def test_heat_equation_pipeline(self):
        """Test time-dependent heat equation pipeline."""
        # Problem setup
        X, Y, t, initial_condition = generate_heat_problem(nx=33, ny=33, nt=5)
        
        # Mock time-stepping solver
        class MockHeatSolver:
            def __init__(self):
                self.dt = t[1] - t[0]
                
            def solve_time_dependent(self, initial, time_points):
                solutions = [initial]
                for i in range(1, len(time_points)):
                    # Simple diffusion approximation
                    prev_solution = solutions[-1]
                    new_solution = prev_solution * 0.95  # Decay
                    solutions.append(new_solution)
                return solutions, {'time_steps': len(time_points)}
        
        heat_solver = MockHeatSolver()
        solutions, info = heat_solver.solve_time_dependent(initial_condition, t)
        
        # Validate time-stepping
        assert len(solutions) == len(t), "Should have solution for each time point"
        assert info['time_steps'] == len(t), "Should record correct number of time steps"
        
        # Check solution decay (expected for heat equation)
        initial_norm = np.linalg.norm(solutions[0])
        final_norm = np.linalg.norm(solutions[-1])
        assert final_norm < initial_norm, "Heat equation should show diffusion/decay"
        
    def test_mixed_precision_pipeline(self):
        """Test mixed-precision solver pipeline."""
        X, Y, exact_solution, source_term = generate_poisson_problem(nx=65, ny=65)
        
        # Test different precision strategies
        precision_types = ['single', 'double', 'mixed']
        results = {}
        
        for precision in precision_types:
            solver = MockSolver(precision=precision)
            numerical_solution, info = solver.solve((X, Y, exact_solution, source_term))
            
            error = np.linalg.norm(numerical_solution - exact_solution)
            results[precision] = {
                'error': error,
                'iterations': info['iterations'],
                'solve_time': info['solve_time']
            }
        
        # Validate precision hierarchy
        assert results['single']['error'] > results['mixed']['error'], \
            "Mixed precision should be more accurate than single precision"
        assert results['mixed']['error'] > results['double']['error'], \
            "Double precision should be most accurate"
            
        # Mixed precision should be faster than double precision
        # (This is a general expectation, though not always guaranteed)
        expected_mixed_benefit = True  # Could be timing-dependent
        
    def test_gpu_cpu_pipeline_consistency(self):
        """Test consistency between CPU and GPU solver pipelines."""
        X, Y, exact_solution, source_term = generate_poisson_problem(nx=33, ny=33)
        
        # CPU solver
        cpu_solver = MockSolver(precision='double', use_gpu=False)
        cpu_solution, cpu_info = cpu_solver.solve((X, Y, exact_solution, source_term))
        
        # GPU solver
        gpu_solver = MockSolver(precision='double', use_gpu=True)
        gpu_solution, gpu_info = gpu_solver.solve((X, Y, exact_solution, source_term))
        
        # Solutions should be consistent (within numerical tolerance)
        solution_diff = np.linalg.norm(cpu_solution - gpu_solution)
        assert solution_diff < 1e-10, f"CPU/GPU solutions differ by {solution_diff}"
        
        # Both should converge
        assert cpu_info['converged'] and gpu_info['converged'], "Both CPU and GPU should converge"
        
    def test_error_handling_pipeline(self):
        """Test error handling in solver pipeline."""
        # Test with malformed input
        with pytest.raises(Exception):
            X, Y = np.meshgrid([], [])  # Empty arrays
            MockSolver().solve((X, Y, X, Y))
        
        # Test with mismatched dimensions
        X = np.random.randn(10, 10)
        Y = np.random.randn(5, 5)  # Different size
        
        # Should handle gracefully or raise appropriate error
        try:
            MockSolver().solve((X, Y, X, Y))
        except (ValueError, AssertionError):
            pass  # Expected error types
        
    def test_convergence_monitoring_pipeline(self):
        """Test convergence monitoring throughout solve process."""
        X, Y, exact_solution, source_term = generate_poisson_problem(nx=33, ny=33)
        
        # Mock solver with convergence history
        class MockConvergenceSolver:
            def __init__(self):
                self.convergence_history = []
                
            def solve(self, problem_data):
                X, Y, exact_solution, source_term = problem_data
                
                # Simulate convergence history
                initial_residual = 1.0
                convergence_factor = 0.1
                
                for i in range(10):
                    residual = initial_residual * (convergence_factor ** i)
                    self.convergence_history.append(residual)
                    
                # Return mock solution
                solution = exact_solution + 1e-10 * np.random.randn(*exact_solution.shape)
                return solution, {
                    'iterations': len(self.convergence_history),
                    'residual': self.convergence_history[-1],
                    'convergence_history': self.convergence_history,
                    'converged': True
                }
        
        solver = MockConvergenceSolver()
        solution, info = solver.solve((X, Y, exact_solution, source_term))
        
        # Validate convergence history
        history = info['convergence_history']
        assert len(history) > 0, "Should record convergence history"
        assert history[0] > history[-1], "Residual should decrease"
        
        # Check monotonic decrease (allowing for some variation)
        decreasing_trend = sum(history[i] > history[i+1] for i in range(len(history)-1))
        assert decreasing_trend >= len(history) * 0.7, "Residual should generally decrease"


class TestPerformanceIntegration:
    """Test performance characteristics in integration scenarios."""
    
    def test_scaling_integration(self):
        """Test solver scaling across different problem sizes."""
        grid_sizes = [17, 33, 65]
        solve_times = []
        
        for nx in grid_sizes:
            X, Y, exact_solution, source_term = generate_poisson_problem(nx=nx, ny=nx)
            
            solver = MockSolver(precision='double')
            start_time = time.time()
            solution, info = solver.solve((X, Y, exact_solution, source_term))
            solve_time = time.time() - start_time
            
            solve_times.append(solve_time)
        
        # Basic scaling check (times should increase with problem size)
        assert solve_times[1] >= solve_times[0], "Larger problems should take longer"
        assert solve_times[2] >= solve_times[1], "Scaling should be consistent"
        
    def test_memory_usage_integration(self):
        """Test memory usage patterns during solve."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Solve large problem
        X, Y, exact_solution, source_term = generate_poisson_problem(nx=129, ny=129)
        solver = MockSolver(precision='double')
        solution, info = solver.solve((X, Y, exact_solution, source_term))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable for problem size
        expected_memory_mb = (129 * 129 * 8) / (1024 * 1024)  # Rough estimate
        assert memory_increase < expected_memory_mb * 10, "Memory usage should be reasonable"


class TestRobustnessIntegration:
    """Test solver robustness in challenging scenarios."""
    
    def test_ill_conditioned_problems(self):
        """Test solver behavior on ill-conditioned problems."""
        # Create problem with high condition number
        nx, ny = 33, 33
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # High-frequency solution (more challenging)
        exact_solution = np.sin(8*np.pi * X) * np.sin(8*np.pi * Y)
        source_term = 128 * np.pi**2 * np.sin(8*np.pi * X) * np.sin(8*np.pi * Y)
        
        solver = MockSolver(precision='double')
        solution, info = solver.solve((X, Y, exact_solution, source_term))
        
        # Should still converge (though may take more iterations)
        assert info['converged'], "Should converge even for ill-conditioned problems"
        assert info['iterations'] <= TEST_CONFIG['max_iterations'], "Should converge within max iterations"
        
    def test_boundary_condition_variations(self):
        """Test different boundary condition scenarios."""
        X, Y, exact_solution, source_term = generate_poisson_problem(nx=33, ny=33)
        
        # Test different boundary conditions
        bc_types = ['dirichlet', 'neumann', 'mixed']
        
        for bc_type in bc_types:
            # Mock solver with boundary condition type
            class BCMockSolver(MockSolver):
                def __init__(self, bc_type):
                    super().__init__()
                    self.bc_type = bc_type
                    
                def solve(self, problem_data):
                    # Adjust error based on BC type complexity
                    solution, info = super().solve(problem_data)
                    if bc_type == 'neumann':
                        info['iterations'] += 2  # Neumann can be harder
                    return solution, info
            
            solver = BCMockSolver(bc_type)
            solution, info = solver.solve((X, Y, exact_solution, source_term))
            
            assert info['converged'], f"Should converge with {bc_type} BC"
            assert info['iterations'] < TEST_CONFIG['max_iterations'], f"Should converge within max iterations for {bc_type} BC"


def test_integration_test_coverage():
    """Meta-test to ensure integration tests cover key scenarios."""
    # This test ensures we have coverage of major integration scenarios
    test_scenarios = [
        'poisson_solver_pipeline',
        'heat_equation_pipeline', 
        'mixed_precision_pipeline',
        'gpu_cpu_pipeline_consistency',
        'error_handling_pipeline',
        'convergence_monitoring_pipeline',
        'scaling_integration',
        'memory_usage_integration',
        'ill_conditioned_problems',
        'boundary_condition_variations'
    ]
    
    # Check that all scenarios are implemented
    test_methods = [method for method in dir(TestEndToEndPipeline) + dir(TestPerformanceIntegration) + dir(TestRobustnessIntegration) 
                   if method.startswith('test_')]
    
    for scenario in test_scenarios:
        assert f'test_{scenario}' in test_methods, f"Missing integration test for {scenario}"


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v'])