"""Unit tests for GPU acceleration components."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import base modules
from multigrid.core.grid import Grid
from multigrid.operators.laplacian import LaplacianOperator
from multigrid.operators.transfer import RestrictionOperator, ProlongationOperator

# Try to import GPU modules
GPU_AVAILABLE = True
try:
    import cupy as cp
    from multigrid.gpu.memory_manager import GPUMemoryManager, GPUMemoryPool, check_gpu_availability
    from multigrid.gpu.cuda_kernels import SmoothingKernels, TransferKernels
    from multigrid.gpu.gpu_precision import GPUPrecisionManager, GPUPrecisionLevel
    from multigrid.gpu.gpu_solver import GPUMultigridSolver, GPUCommunicationAvoidingMultigrid
    from multigrid.gpu.gpu_profiler import GPUPerformanceProfiler
    from multigrid.gpu.gpu_benchmark import GPUBenchmarkSuite
    from multigrid.gpu.multi_gpu import MultiGPUManager, DistributedMultigridSolver
except ImportError:
    GPU_AVAILABLE = False
    pytest.skip("GPU modules not available", allow_module_level=True)


@pytest.fixture(scope="module")
def gpu_info():
    """Check GPU availability for tests."""
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    info = check_gpu_availability()
    if not info['cupy_available'] or info['gpu_count'] == 0:
        pytest.skip("No CUDA-capable GPU available")
    
    return info


class TestGPUMemoryManager:
    """Test GPU memory management."""
    
    def test_gpu_memory_pool_initialization(self, gpu_info):
        """Test GPU memory pool initialization."""
        pool = GPUMemoryPool(max_pool_size_mb=100.0, device_id=0)
        
        assert pool.max_pool_size_bytes == 100 * 1024 * 1024
        assert pool.device_id == 0
        assert pool.enable_statistics is True
        
        # Test statistics
        stats = pool.get_statistics()
        assert 'total_allocated_bytes' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
    
    def test_gpu_array_allocation_deallocation(self, gpu_info):
        """Test GPU array allocation and deallocation."""
        pool = GPUMemoryPool(max_pool_size_mb=50.0)
        
        # Allocate array
        array1 = pool.allocate((10, 10), np.float32)
        assert array1.shape == (10, 10)
        assert array1.dtype == np.float32
        
        # Check statistics
        stats = pool.get_statistics()
        assert stats['total_allocations'] >= 1
        
        # Deallocate
        pool.deallocate(array1)
        
        # Allocate same size - should reuse
        stats_before = pool.get_statistics()
        array2 = pool.allocate((10, 10), np.float32)
        stats_after = pool.get_statistics()
        
        assert stats_after['cache_hits'] > stats_before['cache_hits']
        
        pool.deallocate(array2)
        pool.clear()
    
    def test_gpu_memory_manager_initialization(self, gpu_info):
        """Test GPU memory manager initialization."""
        manager = GPUMemoryManager(device_id=0, max_pool_size_mb=100.0)
        
        assert manager.device_id == 0
        assert manager.memory_pool is not None
        assert len(manager.streams) > 0
        
        # Test array allocation
        array = manager.allocate_gpu_array((5, 5), np.float64)
        assert array.shape == (5, 5)
        assert array.dtype == np.float64
        
        # Test CPU-GPU transfers
        cpu_array = np.random.rand(5, 5)
        gpu_array = manager.to_gpu(cpu_array)
        cpu_result = manager.to_cpu(gpu_array)
        
        assert np.allclose(cpu_array, cpu_result)
        
        manager.cleanup()


class TestCUDAKernels:
    """Test CUDA kernel implementations."""
    
    def test_smoothing_kernels_initialization(self, gpu_info):
        """Test smoothing kernels initialization."""
        kernels = SmoothingKernels(device_id=0)
        
        assert kernels.device_id == 0
        assert len(kernels.compiled_kernels) > 0
        assert 'jacobi_smoothing_kernel' in kernels.compiled_kernels
    
    def test_jacobi_smoothing_kernel(self, gpu_info):
        """Test Jacobi smoothing kernel."""
        kernels = SmoothingKernels(device_id=0)
        
        # Create test data
        nx, ny = 17, 17
        hx = hy = 1.0 / (nx - 1)
        
        u_old = cp.random.rand(nx, ny, dtype=cp.float32)
        u_new = cp.zeros_like(u_old)
        rhs = cp.random.rand(nx, ny, dtype=cp.float32)
        
        # Apply boundary conditions
        u_old[0, :] = u_old[-1, :] = u_old[:, 0] = u_old[:, -1] = 0.0
        rhs[0, :] = rhs[-1, :] = rhs[:, 0] = rhs[:, -1] = 0.0
        
        # Run smoothing
        kernels.jacobi_smoothing(u_old, u_new, rhs, hx, hy, num_iterations=5)
        
        # Check result
        assert not cp.array_equal(u_new, u_old)
        assert cp.allclose(u_new[0, :], 0.0)  # Boundary conditions maintained
        assert cp.allclose(u_new[-1, :], 0.0)
        assert cp.allclose(u_new[:, 0], 0.0)
        assert cp.allclose(u_new[:, -1], 0.0)
    
    def test_transfer_kernels_initialization(self, gpu_info):
        """Test transfer kernels initialization."""
        kernels = TransferKernels(device_id=0)
        
        assert kernels.device_id == 0
        assert len(kernels.compiled_kernels) > 0
        assert 'restriction_kernel' in kernels.compiled_kernels
        assert 'prolongation_kernel' in kernels.compiled_kernels
    
    def test_restriction_prolongation_kernels(self, gpu_info):
        """Test restriction and prolongation kernels."""
        kernels = TransferKernels(device_id=0)
        
        # Create fine and coarse grids
        fine_nx, fine_ny = 17, 17
        coarse_nx, coarse_ny = 9, 9
        
        fine_grid = cp.random.rand(fine_nx, fine_ny, dtype=cp.float32)
        coarse_grid = cp.zeros((coarse_nx, coarse_ny), dtype=cp.float32)
        
        # Test restriction
        kernels.restriction(fine_grid, coarse_grid)
        assert not cp.allclose(coarse_grid, 0.0)
        
        # Test prolongation
        fine_result = cp.zeros_like(fine_grid)
        kernels.prolongation(coarse_grid, fine_result)
        assert not cp.allclose(fine_result, 0.0)
    
    def test_residual_computation_kernel(self, gpu_info):
        """Test residual computation kernel."""
        kernels = TransferKernels(device_id=0)
        
        # Create test problem
        nx, ny = 17, 17
        hx = hy = 1.0 / (nx - 1)
        
        u = cp.random.rand(nx, ny, dtype=cp.float32)
        rhs = cp.random.rand(nx, ny, dtype=cp.float32)
        residual = cp.zeros_like(u)
        
        # Apply boundary conditions
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
        
        # Compute residual
        kernels.compute_residual(u, rhs, residual, hx, hy)
        
        # Check that residual was computed
        assert not cp.allclose(residual, 0.0)
        assert cp.allclose(residual[0, :], 0.0)  # Boundary
        assert cp.allclose(residual[-1, :], 0.0)


class TestGPUPrecisionManager:
    """Test GPU precision management."""
    
    def test_gpu_precision_manager_initialization(self, gpu_info):
        """Test GPU precision manager initialization."""
        manager = GPUPrecisionManager(device_id=0, enable_tensor_cores=True)
        
        assert manager.device_id == 0
        assert manager.enable_tensor_cores is True
        assert manager.current_precision in GPUPrecisionLevel
    
    def test_optimal_dtype_selection(self, gpu_info):
        """Test optimal data type selection."""
        manager = GPUPrecisionManager(device_id=0, default_precision="mixed_tc")
        
        # Test different operation types and levels
        smooth_dtype = manager.get_optimal_dtype('smoothing', 0)
        transfer_dtype = manager.get_optimal_dtype('transfer', 1)
        residual_dtype = manager.get_optimal_dtype('residual', 0)
        
        assert smooth_dtype in [np.float16, np.float32, np.float64]
        assert transfer_dtype in [np.float16, np.float32, np.float64]
        assert residual_dtype in [np.float16, np.float32, np.float64]
    
    def test_precision_conversion(self, gpu_info):
        """Test precision conversion."""
        manager = GPUPrecisionManager(device_id=0)
        
        # Create test array
        test_array = cp.random.rand(10, 10).astype(cp.float64)
        
        # Convert to optimal precision
        converted = manager.convert_to_optimal_precision(test_array, 'smoothing', 0)
        
        assert converted.shape == test_array.shape
        # Dtype may or may not change depending on precision settings
    
    def test_tensor_core_optimization(self, gpu_info):
        """Test Tensor Core optimization."""
        manager = GPUPrecisionManager(device_id=0, enable_tensor_cores=True)
        
        # Create test arrays
        array1 = cp.random.rand(32, 32).astype(cp.float32)
        array2 = cp.random.rand(32, 32).astype(cp.float32)
        
        # Apply Tensor Core optimization
        result = manager.apply_tensor_core_optimization(array1, array2, "multiply")
        
        assert result.shape == array1.shape
        # Check that operation was performed
        expected = array1 * array2
        # Results should be close (allowing for precision differences)
        assert cp.allclose(result, expected, rtol=1e-3)


class TestGPUSolvers:
    """Test GPU multigrid solvers."""
    
    def test_gpu_multigrid_solver_initialization(self, gpu_info):
        """Test GPU multigrid solver initialization."""
        solver = GPUMultigridSolver(
            device_id=0,
            max_levels=4,
            max_iterations=50,
            tolerance=1e-6
        )
        
        assert solver.device_id == 0
        assert solver.max_levels == 4
        assert solver.max_iterations == 50
        assert solver.tolerance == 1e-6
        assert solver.memory_manager is not None
        assert solver.smoothing_kernels is not None
        assert solver.transfer_kernels is not None
    
    def test_gpu_multigrid_solver_setup_and_solve(self, gpu_info):
        """Test GPU multigrid solver setup and solve."""
        # Create test problem
        grid = Grid(nx=17, ny=17, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        restriction = RestrictionOperator("full_weighting")
        prolongation = ProlongationOperator("bilinear")
        
        # Create analytical solution
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u_exact
        
        # Apply boundary conditions
        grid.apply_dirichlet_bc(0.0)
        
        # Create and setup solver
        solver = GPUMultigridSolver(
            device_id=0,
            max_levels=3,
            max_iterations=20,
            tolerance=1e-6,
            enable_mixed_precision=False  # Use single precision for test
        )
        
        solver.setup(grid, operator, restriction, prolongation)
        
        # Solve
        solution, info = solver.solve(grid, operator, rhs)
        
        # Check results
        assert info['iterations'] > 0
        assert solution.shape == grid.shape
        assert info['final_residual'] < 1e-3  # Reasonable convergence
        
        # Check solution accuracy
        error = np.max(np.abs(solution - u_exact))
        assert error < 0.5  # Reasonable accuracy for discretization
        
        solver.cleanup()
    
    def test_gpu_ca_multigrid_solver(self, gpu_info):
        """Test GPU Communication-Avoiding multigrid solver."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = GPUCommunicationAvoidingMultigrid(
            device_id=0,
            max_levels=3,
            max_iterations=20,
            tolerance=1e-6,
            block_size=8,
            enable_memory_pool=True
        )
        
        solver.setup(grid, operator, restriction, prolongation)
        
        # Simple test problem
        rhs = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0)
        
        solution, info = solver.solve(grid, operator, rhs)
        
        # Check results
        assert info['iterations'] > 0
        assert 'ca_optimizations' in info
        assert info['block_size'] == 8
        assert info['ca_stats'] is not None
        
        solver.cleanup()


class TestGPUPerformanceProfiler:
    """Test GPU performance profiling."""
    
    def test_gpu_profiler_initialization(self, gpu_info):
        """Test GPU profiler initialization."""
        profiler = GPUPerformanceProfiler(device_id=0, enable_detailed_profiling=True)
        
        assert profiler.device_id == 0
        assert profiler.enable_detailed_profiling is True
        assert len(profiler.event_pool) > 0
    
    def test_gpu_operation_profiling(self, gpu_info):
        """Test GPU operation profiling."""
        profiler = GPUPerformanceProfiler(device_id=0)
        
        # Profile a simple operation
        with profiler.profile_gpu_operation("test_operation", memory_bytes=1000):
            # Simulate GPU work
            test_array = cp.random.rand(100, 100)
            result = cp.sum(test_array)
            cp.cuda.Device().synchronize()
        
        # Check profiling results
        summary = profiler.get_profiling_summary()
        assert 'operation_profiles' in summary
        assert len(summary['operation_profiles']) > 0
        
        # Find our test operation
        test_profile = None
        for profile in summary['operation_profiles']:
            if profile['name'] == 'test_operation':
                test_profile = profile
                break
        
        assert test_profile is not None
        assert test_profile['call_count'] == 1
        assert test_profile['total_gpu_time'] > 0
    
    def test_memory_snapshot(self, gpu_info):
        """Test memory snapshot functionality."""
        profiler = GPUPerformanceProfiler(device_id=0, track_memory_usage=True)
        
        # Take initial snapshot
        snapshot1 = profiler.take_memory_snapshot("initial")
        assert 'used_bytes' in snapshot1
        assert 'total_bytes' in snapshot1
        
        # Allocate memory
        test_arrays = [cp.random.rand(100, 100) for _ in range(10)]
        
        # Take second snapshot
        snapshot2 = profiler.take_memory_snapshot("after_allocation")
        
        # Memory usage should have increased
        assert snapshot2['used_bytes'] >= snapshot1['used_bytes']
    
    def test_gpu_report_generation(self, gpu_info):
        """Test GPU performance report generation."""
        profiler = GPUPerformanceProfiler(device_id=0)
        
        # Profile some operations
        with profiler.profile_gpu_operation("operation_1"):
            cp.random.rand(50, 50)
        
        with profiler.profile_gpu_operation("operation_2"):
            cp.random.rand(100, 100)
        
        # Generate report
        report = profiler.generate_gpu_report()
        
        assert isinstance(report, str)
        assert "GPU Performance Profiling Report" in report
        assert "operation_1" in report
        assert "operation_2" in report


class TestGPUBenchmarking:
    """Test GPU benchmarking suite."""
    
    def test_gpu_benchmark_suite_initialization(self, gpu_info):
        """Test GPU benchmark suite initialization."""
        suite = GPUBenchmarkSuite(device_id=0, enable_profiling=True)
        
        assert suite.device_id == 0
        assert suite.enable_profiling is True
        assert suite.gpu_info['cupy_available'] is True
        assert suite.gpu_info['gpu_count'] > 0
    
    def test_quick_gpu_benchmark(self, gpu_info):
        """Test quick GPU benchmark."""
        # Run quick benchmark with small problems
        suite = GPUBenchmarkSuite(device_id=0, enable_profiling=False)
        
        results = suite.run_comprehensive_benchmark(
            problem_sizes=[(33, 33)],  # Small problem for quick test
            solver_types=['cpu_multigrid', 'gpu_multigrid'],
            precision_levels=['single'],
            num_runs=2
        )
        
        assert 'summary' in results
        assert 'speedup_analysis' in results
        assert results['summary']['total_benchmarks'] > 0
    
    def test_benchmark_report_generation(self, gpu_info):
        """Test benchmark report generation."""
        suite = GPUBenchmarkSuite(device_id=0)
        
        # Run minimal benchmark
        suite.run_comprehensive_benchmark(
            problem_sizes=[(17, 17)],
            solver_types=['gpu_multigrid'],
            precision_levels=['single'],
            num_runs=1
        )
        
        # Generate report
        report = suite.generate_benchmark_report()
        
        assert isinstance(report, str)
        assert "GPU Multigrid Solver Benchmark Report" in report


@pytest.mark.skipif(
    not GPU_AVAILABLE or check_gpu_availability()['gpu_count'] < 2,
    reason="Multi-GPU testing requires at least 2 GPUs"
)
class TestMultiGPU:
    """Test multi-GPU capabilities."""
    
    def test_multi_gpu_manager_initialization(self):
        """Test multi-GPU manager initialization."""
        manager = MultiGPUManager(device_ids=[0, 1])
        
        assert manager.num_devices == 2
        assert 0 in manager.device_info
        assert 1 in manager.device_info
        assert len(manager.memory_managers) == 2
        
        manager.cleanup()
    
    def test_device_allocation(self):
        """Test device allocation and release."""
        manager = MultiGPUManager()
        
        # Allocate device for task
        device_id = manager.allocate_device_for_task("test_task", memory_requirement_mb=100.0)
        assert device_id is not None
        assert device_id in manager.device_ids
        
        # Release device
        manager.release_device(device_id, "test_task")
        
        manager.cleanup()
    
    def test_distributed_multigrid_solver_initialization(self):
        """Test distributed multigrid solver initialization."""
        solver = DistributedMultigridSolver(
            device_ids=[0, 1],
            decomposition_strategy="stripe",
            max_levels=3
        )
        
        assert solver.num_devices == 2
        assert solver.decomposition_strategy == "stripe"
        assert len(solver.device_ids) == 2
        
        solver.cleanup()


class TestIntegration:
    """Integration tests for GPU components."""
    
    def test_full_gpu_solver_pipeline(self, gpu_info):
        """Test complete GPU solver pipeline."""
        # Create test problem
        grid = Grid(nx=33, ny=33, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        restriction = RestrictionOperator("full_weighting")
        prolongation = ProlongationOperator("bilinear")
        
        # Analytical solution
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u_exact
        grid.apply_dirichlet_bc(0.0)
        
        # Create solver with profiling
        profiler = GPUPerformanceProfiler(device_id=0)
        solver = GPUCommunicationAvoidingMultigrid(
            device_id=0,
            max_levels=4,
            max_iterations=50,
            tolerance=1e-6,
            enable_mixed_precision=True,
            use_fmg=True
        )
        
        solver.setup(grid, operator, restriction, prolongation)
        
        # Solve with profiling
        with profiler.profile_gpu_operation("full_solve", kernel_count=10):
            solution, info = solver.solve(grid, operator, rhs)
        
        # Verify results
        assert info['converged'] or info['final_residual'] < 1e-4
        error = np.max(np.abs(solution - u_exact))
        assert error < 0.5
        
        # Check profiling data
        profile_summary = profiler.get_profiling_summary()
        assert len(profile_summary['operation_profiles']) > 0
        
        # Check solver statistics
        solver_stats = solver.get_performance_statistics()
        assert 'ca_optimizations' in solver_stats
        assert 'gpu_info' in solver_stats
        
        solver.cleanup()
    
    def test_precision_and_performance_integration(self, gpu_info):
        """Test integration of precision management and performance profiling."""
        # Create precision manager
        precision_manager = GPUPrecisionManager(
            device_id=0,
            enable_tensor_cores=True,
            adaptive=True
        )
        
        profiler = GPUPerformanceProfiler(device_id=0)
        
        # Test different precision operations
        test_array = cp.random.rand(64, 64).astype(cp.float32)
        
        # Profile operations with different precisions
        for precision in ['single', 'mixed_tc']:
            precision_manager.current_precision = GPUPrecisionLevel(precision)
            
            with profiler.profile_gpu_operation(f"operation_{precision}"):
                converted = precision_manager.convert_to_optimal_precision(
                    test_array, 'smoothing', 0
                )
                result = converted * 2.0
        
        # Check precision statistics
        precision_stats = precision_manager.get_precision_statistics()
        assert 'performance_stats' in precision_stats
        
        # Check profiling results
        profile_summary = profiler.get_profiling_summary()
        assert len(profile_summary['operation_profiles']) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])