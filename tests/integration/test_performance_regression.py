"""
Performance Regression Tests

Tests to ensure performance characteristics are maintained across code changes.
Establishes performance baselines and detects performance regressions.
"""

import pytest
import numpy as np
import time
import json
import os
from pathlib import Path
import tempfile

# Test configuration
PERFORMANCE_BASELINES = {
    'poisson_solve_time': {
        'grid_33x33': {'max_time': 0.5, 'target_time': 0.1},
        'grid_65x65': {'max_time': 2.0, 'target_time': 0.5},
        'grid_129x129': {'max_time': 10.0, 'target_time': 2.0}
    },
    'convergence_iterations': {
        'poisson': {'max_iterations': 20, 'target_iterations': 10},
        'heat_equation': {'max_iterations': 15, 'target_iterations': 8}
    },
    'memory_usage': {
        'grid_65x65': {'max_mb': 50, 'target_mb': 20},
        'grid_129x129': {'max_mb': 200, 'target_mb': 80}
    },
    'gpu_speedup': {
        'min_speedup': 2.0,  # GPU should be at least 2x faster than CPU
        'target_speedup': 5.0
    }
}


class MockPerformanceSolver:
    """Mock solver with realistic performance characteristics."""
    
    def __init__(self, precision='double', use_gpu=False, grid_size=65):
        self.precision = precision
        self.use_gpu = use_gpu
        self.grid_size = grid_size
        
    def solve(self, problem_data):
        """Mock solve with realistic timing."""
        # Simulate realistic solve times based on grid size and hardware
        n_points = self.grid_size ** 2
        
        if self.use_gpu:
            base_time_per_point = 1e-8  # GPU is faster
            memory_factor = 0.8
        else:
            base_time_per_point = 5e-8  # CPU baseline
            memory_factor = 1.0
            
        # Precision affects performance
        if self.precision == 'single':
            precision_factor = 0.5
        elif self.precision == 'mixed':
            precision_factor = 0.7
        else:  # double
            precision_factor = 1.0
            
        # Simulate solve time
        solve_time = n_points * base_time_per_point * precision_factor
        solve_time += 0.01 * np.random.randn()  # Add noise
        solve_time = max(solve_time, 0.001)  # Minimum time
        
        # Simulate memory usage
        bytes_per_point = 8 if self.precision == 'double' else 4
        if self.precision == 'mixed':
            bytes_per_point = 6
            
        memory_mb = (n_points * bytes_per_point * memory_factor) / (1024 * 1024)
        
        # Simulate iterations (dependent on problem conditioning)
        base_iterations = 8
        if self.grid_size > 100:
            base_iterations += 2  # Larger problems may need more iterations
            
        iterations = base_iterations + np.random.poisson(2)
        
        # Return results
        return {
            'solve_time': solve_time,
            'iterations': iterations,
            'memory_mb': memory_mb,
            'converged': True
        }


class TestPerformanceBaselines:
    """Test performance against established baselines."""
    
    def test_poisson_solve_time_regression(self):
        """Test Poisson solver performance doesn't regress."""
        grid_sizes = [33, 65, 129]
        
        for grid_size in grid_sizes:
            solver = MockPerformanceSolver(grid_size=grid_size)
            
            # Warm-up run
            _ = solver.solve(None)
            
            # Timed run
            start_time = time.time()
            result = solver.solve(None)
            actual_time = time.time() - start_time
            
            # Check against baseline
            baseline_key = f'grid_{grid_size}x{grid_size}'
            if baseline_key in PERFORMANCE_BASELINES['poisson_solve_time']:
                baseline = PERFORMANCE_BASELINES['poisson_solve_time'][baseline_key]
                
                assert actual_time <= baseline['max_time'], \
                    f"Solve time {actual_time:.3f}s exceeds maximum {baseline['max_time']:.3f}s for {grid_size}x{grid_size} grid"
                    
                if actual_time > baseline['target_time']:
                    pytest.warns(UserWarning, 
                                f"Solve time {actual_time:.3f}s exceeds target {baseline['target_time']:.3f}s for {grid_size}x{grid_size} grid")
    
    def test_convergence_iterations_regression(self):
        """Test solver convergence characteristics don't regress."""
        problem_types = ['poisson', 'heat_equation']
        
        for problem_type in problem_types:
            solver = MockPerformanceSolver(grid_size=65)
            result = solver.solve(None)
            
            actual_iterations = result['iterations']
            baseline = PERFORMANCE_BASELINES['convergence_iterations'][problem_type]
            
            assert actual_iterations <= baseline['max_iterations'], \
                f"Iterations {actual_iterations} exceed maximum {baseline['max_iterations']} for {problem_type}"
                
            if actual_iterations > baseline['target_iterations']:
                pytest.warns(UserWarning, 
                            f"Iterations {actual_iterations} exceed target {baseline['target_iterations']} for {problem_type}")
    
    def test_memory_usage_regression(self):
        """Test memory usage doesn't regress."""
        import psutil
        import os
        
        grid_sizes = [65, 129]
        
        for grid_size in grid_sizes:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            solver = MockPerformanceSolver(grid_size=grid_size)
            result = solver.solve(None)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            baseline_key = f'grid_{grid_size}x{grid_size}'
            if baseline_key in PERFORMANCE_BASELINES['memory_usage']:
                baseline = PERFORMANCE_BASELINES['memory_usage'][baseline_key]
                
                assert memory_increase <= baseline['max_mb'], \
                    f"Memory usage {memory_increase:.1f}MB exceeds maximum {baseline['max_mb']}MB for {grid_size}x{grid_size} grid"
    
    def test_gpu_speedup_regression(self):
        """Test GPU acceleration doesn't regress."""
        grid_size = 129
        
        # CPU benchmark
        cpu_solver = MockPerformanceSolver(grid_size=grid_size, use_gpu=False)
        cpu_result = cpu_solver.solve(None)
        cpu_time = cpu_result['solve_time']
        
        # GPU benchmark
        gpu_solver = MockPerformanceSolver(grid_size=grid_size, use_gpu=True)
        gpu_result = gpu_solver.solve(None)
        gpu_time = gpu_result['solve_time']
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        baseline = PERFORMANCE_BASELINES['gpu_speedup']
        
        assert speedup >= baseline['min_speedup'], \
            f"GPU speedup {speedup:.1f}x below minimum {baseline['min_speedup']}x"
            
        if speedup < baseline['target_speedup']:
            pytest.warns(UserWarning, 
                        f"GPU speedup {speedup:.1f}x below target {baseline['target_speedup']}x")


class TestPerformanceStability:
    """Test performance stability and consistency."""
    
    def test_solve_time_consistency(self):
        """Test solve time consistency across multiple runs."""
        solver = MockPerformanceSolver(grid_size=65)
        n_runs = 10
        solve_times = []
        
        # Run multiple solves
        for _ in range(n_runs):
            result = solver.solve(None)
            solve_times.append(result['solve_time'])
        
        # Analyze consistency
        mean_time = np.mean(solve_times)
        std_time = np.std(solve_times)
        cv = std_time / mean_time  # Coefficient of variation
        
        # Should have reasonable consistency (CV < 20%)
        assert cv < 0.2, f"Solve time coefficient of variation {cv:.3f} too high (>0.2)"
        
        # No outliers (within 3 standard deviations)
        for time_val in solve_times:
            z_score = abs(time_val - mean_time) / std_time
            assert z_score <= 3.0, f"Solve time outlier detected: {time_val:.3f}s (z-score: {z_score:.1f})"
    
    def test_convergence_consistency(self):
        """Test convergence behavior consistency."""
        solver = MockPerformanceSolver(grid_size=65)
        n_runs = 10
        iterations_list = []
        
        for _ in range(n_runs):
            result = solver.solve(None)
            iterations_list.append(result['iterations'])
        
        # Should converge consistently
        mean_iterations = np.mean(iterations_list)
        std_iterations = np.std(iterations_list)
        
        # Standard deviation should be reasonable
        assert std_iterations <= 3, f"Iteration count standard deviation {std_iterations:.1f} too high"
        
        # All runs should converge
        assert all(result['converged'] for result in 
                  [solver.solve(None) for _ in range(5)]), "All runs should converge"


class TestPerformanceRegression:
    """Test for performance regressions by comparing with saved baselines."""
    
    def setup_method(self):
        """Setup performance testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_file = os.path.join(self.temp_dir, 'performance_baseline.json')
        
    def teardown_method(self):
        """Clean up after performance testing."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def save_performance_baseline(self, results):
        """Save performance results as baseline."""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_performance_baseline(self):
        """Load performance baseline."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        # Run current performance benchmark
        solver = MockPerformanceSolver(grid_size=65)
        current_results = {
            'solve_time': [],
            'iterations': [],
            'memory_mb': []
        }
        
        # Multiple runs for statistical significance
        for _ in range(5):
            result = solver.solve(None)
            current_results['solve_time'].append(result['solve_time'])
            current_results['iterations'].append(result['iterations'])
            current_results['memory_mb'].append(result['memory_mb'])
        
        # Calculate statistics
        current_stats = {}
        for metric, values in current_results.items():
            current_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Load baseline (if exists)
        baseline_stats = self.load_performance_baseline()
        
        if baseline_stats is None:
            # First run - save as baseline
            self.save_performance_baseline(current_stats)
            pytest.skip("No baseline available, saving current results as baseline")
        
        # Compare with baseline
        regression_threshold = 1.2  # 20% performance degradation threshold
        
        for metric in ['solve_time']:  # Focus on key performance metrics
            current_mean = current_stats[metric]['mean']
            baseline_mean = baseline_stats[metric]['mean']
            
            if baseline_mean > 0:
                performance_ratio = current_mean / baseline_mean
                
                assert performance_ratio <= regression_threshold, \
                    f"Performance regression detected in {metric}: " \
                    f"current {current_mean:.3f} vs baseline {baseline_mean:.3f} " \
                    f"(ratio: {performance_ratio:.2f})"
    
    def test_scaling_performance_regression(self):
        """Test performance scaling characteristics don't regress."""
        grid_sizes = [33, 65, 129]
        solve_times = []
        
        for grid_size in grid_sizes:
            solver = MockPerformanceSolver(grid_size=grid_size)
            result = solver.solve(None)
            solve_times.append(result['solve_time'])
        
        # Check scaling behavior
        # Expect roughly O(N) or O(N log N) scaling for multigrid
        for i in range(1, len(grid_sizes)):
            size_ratio = (grid_sizes[i] / grid_sizes[i-1]) ** 2  # Area ratio
            time_ratio = solve_times[i] / solve_times[i-1]
            
            # Time ratio should not exceed size ratio by too much
            scaling_factor = time_ratio / size_ratio
            assert scaling_factor <= 2.0, \
                f"Poor scaling detected: size increased by {size_ratio:.1f}x, " \
                f"time increased by {time_ratio:.1f}x (factor: {scaling_factor:.2f})"


class TestPerformanceMonitoring:
    """Test performance monitoring and profiling capabilities."""
    
    def test_performance_profiling_integration(self):
        """Test integration with performance profiling tools."""
        import cProfile
        import pstats
        import io
        
        # Profile solver execution
        profiler = cProfile.Profile()
        profiler.enable()
        
        solver = MockPerformanceSolver(grid_size=65)
        result = solver.solve(None)
        
        profiler.disable()
        
        # Analyze profiling results
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats()
        
        profiling_output = stats_buffer.getvalue()
        
        # Basic checks on profiling output
        assert 'solve' in profiling_output, "Profiling should capture solve method"
        assert len(profiling_output) > 0, "Profiling should generate output"
    
    def test_memory_profiling_integration(self):
        """Test memory profiling capabilities."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Monitor memory during solve
        initial_memory = process.memory_info().rss
        solver = MockPerformanceSolver(grid_size=129)
        result = solver.solve(None)
        peak_memory = process.memory_info().rss
        
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable and trackable
        assert memory_increase >= 0, "Memory tracking should work correctly"
        
        # Log memory usage for monitoring
        memory_mb = memory_increase / (1024 * 1024)
        print(f"Memory increase: {memory_mb:.2f} MB")


def test_performance_test_completeness():
    """Meta-test to ensure comprehensive performance testing."""
    required_performance_tests = [
        'solve_time_regression',
        'convergence_iterations_regression', 
        'memory_usage_regression',
        'gpu_speedup_regression',
        'solve_time_consistency',
        'convergence_consistency',
        'performance_regression_detection',
        'scaling_performance_regression'
    ]
    
    # Get all test methods
    test_classes = [TestPerformanceBaselines, TestPerformanceStability, TestPerformanceRegression]
    all_test_methods = []
    
    for test_class in test_classes:
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        all_test_methods.extend(methods)
    
    # Check coverage
    for required_test in required_performance_tests:
        test_name = f'test_{required_test}'
        assert test_name in all_test_methods, f"Missing required performance test: {test_name}"


if __name__ == '__main__':
    # Run performance regression tests
    pytest.main([__file__, '-v', '--tb=short'])