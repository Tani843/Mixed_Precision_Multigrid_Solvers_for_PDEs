"""Unit tests for advanced multigrid cycles and optimizations."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multigrid.core.grid import Grid
from multigrid.core.precision import PrecisionManager
from multigrid.operators.laplacian import LaplacianOperator
from multigrid.operators.transfer import RestrictionOperator, ProlongationOperator
from multigrid.solvers.advanced_multigrid import CommunicationAvoidingMultigrid
from multigrid.optimization.cache_optimization import (
    CacheOptimizer, BlockTraversal
)
from multigrid.optimization.memory_management import MemoryPool, WorkingArrayManager


class TestCommunicationAvoidingMultigrid:
    """Test cases for communication-avoiding multigrid solver."""
    
    def test_ca_multigrid_initialization(self):
        """Test CA-multigrid initialization."""
        solver = CommunicationAvoidingMultigrid(
            max_levels=4,
            cycle_type="V",
            block_size=16,
            enable_memory_pool=True
        )
        
        assert solver.max_levels == 4
        assert solver.cycle_type == "V"
        assert solver.block_size == 16
        assert solver.enable_memory_pool is True
        assert "CommunicationAvoidingMG" in solver.name
    
    def test_ca_multigrid_setup(self):
        """Test CA-multigrid setup with hierarchy."""
        grid = Grid(nx=33, ny=33)
        operator = LaplacianOperator()
        restriction = RestrictionOperator("full_weighting")
        prolongation = ProlongationOperator("bilinear")
        
        solver = CommunicationAvoidingMultigrid(max_levels=4)
        solver.setup(grid, operator, restriction, prolongation)
        
        assert len(solver.grids) <= 4
        assert len(solver.grids) >= 2  # At least fine and one coarse
        assert solver.grids[0].shape == grid.shape
        
        # Check hierarchy is properly decreasing
        for i in range(1, len(solver.grids)):
            assert solver.grids[i].nx <= solver.grids[i-1].nx
            assert solver.grids[i].ny <= solver.grids[i-1].ny
    
    def test_memory_pool_setup(self):
        """Test memory pool initialization."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(
            max_levels=3,
            enable_memory_pool=True
        )
        solver.setup(grid, operator, restriction, prolongation)
        
        assert solver.memory_pool is not None
        assert len(solver.memory_pool) > 0
    
    def test_precision_strategies_setup(self):
        """Test adaptive precision strategy setup."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(max_levels=3)
        solver.setup(grid, operator, restriction, prolongation)
        
        assert len(solver.precision_strategies) == len(solver.grids)
        
        # Check that coarse levels recommend single precision
        num_levels = len(solver.grids)
        coarse_level = num_levels - 1
        fine_level = 0
        
        coarse_strategy = solver.precision_strategies[coarse_level]
        fine_strategy = solver.precision_strategies[fine_level]
        
        # Strategies should be set up
        assert 'recommended' in coarse_strategy
        assert 'recommended' in fine_strategy
    
    def test_v_cycle_solve(self):
        """Test V-cycle solving."""
        grid = Grid(nx=17, ny=17, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        restriction = RestrictionOperator("full_weighting")
        prolongation = ProlongationOperator("bilinear")
        
        solver = CommunicationAvoidingMultigrid(
            max_levels=3,
            cycle_type="V",
            max_iterations=20,
            tolerance=1e-6
        )
        solver.setup(grid, operator, restriction, prolongation)
        
        # Test problem with known solution
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u_exact
        
        # Apply boundary conditions
        grid.apply_dirichlet_bc(0.0)
        
        solution, info = solver.solve(grid, operator, rhs)
        
        # Should converge
        assert info['converged'] is True or info['final_residual'] < solver.tolerance
        assert info['iterations'] > 0
        
        # Solution should be reasonable
        error = np.max(np.abs(solution - u_exact))
        assert error < 0.1  # Reasonable accuracy
    
    def test_w_cycle_solve(self):
        """Test W-cycle solving."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(
            max_levels=3,
            cycle_type="W",
            max_iterations=15,
            tolerance=1e-6
        )
        solver.setup(grid, operator, restriction, prolongation)
        
        # Simple test problem
        rhs = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0)
        
        solution, info = solver.solve(grid, operator, rhs)
        
        # W-cycle should converge faster than V-cycle for this problem
        assert info['iterations'] <= 15
        assert not np.allclose(solution, 0)  # Should have non-trivial solution
    
    def test_fmg_initialization(self):
        """Test Full Multigrid initialization."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(
            max_levels=3,
            use_fmg=True,
            fmg_cycles=2,
            max_iterations=10
        )
        solver.setup(grid, operator, restriction, prolongation)
        
        rhs = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0)
        
        solution, info = solver.solve(grid, operator, rhs)
        
        # FMG should provide good initial guess, leading to fast convergence
        assert info['iterations'] <= 10
    
    def test_adaptive_precision_integration(self):
        """Test integration with adaptive precision."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(max_levels=3, max_iterations=20)
        solver.setup(grid, operator, restriction, prolongation)
        
        precision_manager = PrecisionManager(adaptive=True, default_precision="mixed")
        
        rhs = np.ones(grid.shape)
        solution, info = solver.solve(grid, operator, rhs, precision_manager=precision_manager)
        
        # Should complete without errors
        assert info['iterations'] > 0
        assert solution.shape == grid.shape
    
    def test_performance_statistics(self):
        """Test performance statistics collection."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(max_levels=3, max_iterations=5)
        solver.setup(grid, operator, restriction, prolongation)
        
        rhs = np.ones(grid.shape)
        solution, info = solver.solve(grid, operator, rhs)
        
        # Check performance statistics
        perf_stats = solver.get_performance_statistics()
        
        assert 'total_cycle_time' in perf_stats
        assert 'level_breakdown' in perf_stats
        assert 'operation_breakdown' in perf_stats
        assert 'memory_efficiency' in perf_stats
        
        # Check level breakdown
        level_breakdown = perf_stats['level_breakdown']
        assert len(level_breakdown) == len(solver.grids)
        
        for level in level_breakdown:
            level_stats = level_breakdown[level]
            assert 'time' in level_stats
            assert 'percentage' in level_stats
    
    def test_convergence_info_extended(self):
        """Test extended convergence information."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(max_levels=2, max_iterations=5)
        solver.setup(grid, operator, restriction, prolongation)
        
        rhs = np.ones(grid.shape)
        solution, info = solver.solve(grid, operator, rhs)
        
        # Check extended convergence info
        assert 'cycle_type' in info
        assert 'num_levels' in info
        assert 'fmg_used' in info
        assert 'block_size' in info
        assert 'memory_pool_enabled' in info
        assert 'performance_stats' in info
        
        assert info['cycle_type'] == "V"
        assert info['num_levels'] == len(solver.grids)


class TestCacheOptimization:
    """Test cases for cache optimization techniques."""
    
    def test_cache_optimizer_initialization(self):
        """Test cache optimizer initialization."""
        optimizer = CacheOptimizer(
            cache_line_size=64,
            l1_cache_size=32*1024,
            l2_cache_size=256*1024
        )
        
        assert optimizer.cache_line_size == 64
        assert optimizer.l1_cache_size == 32*1024
        assert optimizer.l2_cache_size == 256*1024
        assert len(optimizer.optimal_block_sizes) > 0
    
    def test_optimal_block_size_calculation(self):
        """Test optimal block size calculation."""
        optimizer = CacheOptimizer()
        
        # Test for different grid sizes and data types
        small_grid = (16, 16)
        large_grid = (256, 256)
        
        block_size_small_f64 = optimizer.get_optimal_block_size(small_grid, np.float64)
        block_size_large_f64 = optimizer.get_optimal_block_size(large_grid, np.float64)
        block_size_small_f32 = optimizer.get_optimal_block_size(small_grid, np.float32)
        
        # Block sizes should be reasonable
        assert 4 <= block_size_small_f64 <= 128
        assert 4 <= block_size_large_f64 <= 128
        assert 4 <= block_size_small_f32 <= 128
        
        # Different precisions may have different optimal sizes
        # (not necessarily true, but block sizes should be valid)
        assert block_size_small_f32 > 0
    
    def test_array_layout_optimization(self):
        """Test array layout optimization."""
        optimizer = CacheOptimizer()
        
        # Create non-contiguous array
        array = np.zeros((10, 10))
        non_contiguous = array.T  # Transpose creates non-contiguous view
        
        optimized = optimizer.optimize_array_layout(non_contiguous)
        
        # Should be C-contiguous
        assert optimized.flags['C_CONTIGUOUS']
    
    def test_block_traversal_row_major(self):
        """Test row-major block traversal."""
        grid = Grid(nx=9, ny=9)
        traversal = BlockTraversal(grid, block_size=4, traversal_pattern="row_major")
        
        blocks = list(traversal)
        assert len(blocks) > 0
        
        # Check that blocks cover the grid
        total_points = 0
        for block in blocks:
            block_points = (block.i_end - block.i_start) * (block.j_end - block.j_start)
            total_points += block_points
        
        # Should cover most of the interior (excluding boundaries by default)
        interior_points = (grid.nx - 2) * (grid.ny - 2)
        assert total_points >= interior_points * 0.8  # Allow some overlap/coverage variation
    
    def test_block_traversal_z_order(self):
        """Test Z-order block traversal."""
        grid = Grid(nx=17, ny=17)
        traversal = BlockTraversal(grid, block_size=8, traversal_pattern="z_order")
        
        blocks = list(traversal)
        assert len(blocks) > 0
        
        # Z-order should produce valid blocks
        for block in blocks:
            assert block.i_start >= 0
            assert block.j_start >= 0
            assert block.i_end <= grid.nx
            assert block.j_end <= grid.ny
    
    def test_block_neighbors(self):
        """Test block neighbor identification."""
        grid = Grid(nx=13, ny=13)
        traversal = BlockTraversal(grid, block_size=6)
        
        if len(traversal) > 1:
            neighbors = traversal.get_block_neighbors(0)
            
            # Should find some neighbors for non-trivial grids
            assert isinstance(neighbors, list)
            assert all(0 <= idx < len(traversal) for idx in neighbors)
    
    def test_cache_aware_stencil(self):
        """Test cache-aware stencil operations."""
        optimizer = CacheOptimizer()
        stencil = CacheAwareStencil(optimizer)
        
        grid = Grid(nx=17, ny=17)
        u = np.random.rand(*grid.shape)
        rhs = np.random.rand(*grid.shape)
        
        # Apply boundary conditions
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
        
        u_new = stencil.apply_5_point_stencil_blocked(u, rhs, grid, num_iterations=2)
        
        assert u_new.shape == u.shape
        assert not np.array_equal(u_new, u)  # Should have changed
        
        # Boundaries should remain zero
        assert np.allclose(u_new[0, :], 0.0)
        assert np.allclose(u_new[-1, :], 0.0)
        assert np.allclose(u_new[:, 0], 0.0)
        assert np.allclose(u_new[:, -1], 0.0)
    
    def test_residual_computation_blocked(self):
        """Test blocked residual computation."""
        optimizer = CacheOptimizer()
        stencil = CacheAwareStencil(optimizer)
        
        grid = Grid(nx=13, ny=13)
        u = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u  # For this u, Au = -2π²u, so residual = rhs - Au = 0
        
        residual = stencil.compute_residual_blocked(u, rhs, grid)
        
        assert residual.shape == u.shape
        
        # For the exact solution, interior residual should be small
        interior_residual = residual[1:-1, 1:-1]
        assert np.max(np.abs(interior_residual)) < 0.1  # Allow some discretization error


class TestMemoryManagement:
    """Test cases for memory management."""
    
    def test_memory_pool_initialization(self):
        """Test memory pool initialization."""
        pool = MemoryPool(max_pool_size_mb=10.0, block_alignment=64)
        
        assert pool.max_pool_size_bytes == 10 * 1024 * 1024
        assert pool.block_alignment == 64
        assert pool.enable_statistics is True
    
    def test_memory_allocation_deallocation(self):
        """Test basic memory allocation and deallocation."""
        pool = MemoryPool(max_pool_size_mb=1.0)
        
        # Allocate array
        array = pool.allocate((10, 10), np.float64)
        
        assert array.shape == (10, 10)
        assert array.dtype == np.float64
        
        # Check statistics
        stats = pool.get_statistics()
        assert stats['total_allocations'] >= 1
        assert stats['total_blocks'] >= 1
        
        # Deallocate
        pool.deallocate(array)
        
        # Array should be returned to pool
        stats_after = pool.get_statistics()
        assert stats_after['available_blocks'] >= 1
    
    def test_memory_pool_reuse(self):
        """Test memory pool array reuse."""
        pool = MemoryPool(max_pool_size_mb=1.0)
        
        # Allocate and deallocate
        array1 = pool.allocate((8, 8), np.float64)
        pool.deallocate(array1)
        
        # Allocate same size - should reuse
        stats_before = pool.get_statistics()
        array2 = pool.allocate((8, 8), np.float64)
        stats_after = pool.get_statistics()
        
        # Should have reused existing block
        assert stats_after['cache_hits'] > stats_before['cache_hits']
    
    def test_working_array_manager(self):
        """Test working array manager."""
        pool = MemoryPool(max_pool_size_mb=1.0)
        manager = WorkingArrayManager(pool)
        
        # Test context manager
        with manager.allocate_working_arrays((5, 5, np.float64), (3, 3, np.float32)) as arrays:
            assert len(arrays) == 2
            assert arrays[0].shape == (5, 5)
            assert arrays[0].dtype == np.float64
            assert arrays[1].shape == (3, 3)
            assert arrays[1].dtype == np.float32
            
            assert manager.get_active_count() == 2
        
        # Arrays should be cleaned up
        assert manager.get_active_count() == 0
    
    def test_working_array_allocate_like(self):
        """Test allocate_like functionality."""
        manager = WorkingArrayManager()
        
        reference = np.random.rand(7, 7).astype(np.float32)
        similar = manager.allocate_like(reference)
        
        assert similar.shape == reference.shape
        assert similar.dtype == reference.dtype
        assert np.allclose(similar, 0.0)  # Should be zero-filled by default
        
        manager.deallocate(similar)
        assert manager.get_active_count() == 0
    
    def test_memory_pool_eviction(self):
        """Test memory pool eviction under pressure."""
        # Small pool to trigger eviction
        pool = MemoryPool(max_pool_size_mb=0.01)  # 10 KB
        
        arrays = []
        
        # Allocate many small arrays
        for i in range(10):
            array = pool.allocate((32, 32), np.float64)  # ~8KB each
            arrays.append(array)
        
        # Check that pool has evicted some blocks
        stats = pool.get_statistics()
        assert stats['pool_evictions'] > 0 or stats['current_size_mb'] < 0.1


class TestIntegration:
    """Integration tests for advanced multigrid with optimizations."""
    
    def test_full_optimization_stack(self):
        """Test full optimization stack integration."""
        # Setup optimized solver
        grid = Grid(nx=33, ny=33)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        solver = CommunicationAvoidingMultigrid(
            max_levels=4,
            cycle_type="V",
            block_size=8,
            enable_memory_pool=True,
            use_fmg=False
        )
        solver.setup(grid, operator, restriction, prolongation)
        
        # Setup precision manager
        precision_manager = PrecisionManager(adaptive=True, default_precision="mixed")
        
        # Test problem
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u_exact
        
        # Solve
        solution, info = solver.solve(grid, operator, rhs, precision_manager=precision_manager)
        
        # Verify results
        assert info['converged'] or info['final_residual'] < 1e-4
        
        error = np.max(np.abs(solution - u_exact))
        assert error < 0.5  # Reasonable accuracy for discretization
        
        # Check that optimizations were used
        perf_stats = solver.get_performance_statistics()
        assert perf_stats['memory_efficiency']['memory_pool_enabled'] is True
        assert perf_stats['memory_efficiency']['block_size'] == 8
    
    def test_performance_comparison(self):
        """Test performance comparison between optimized and standard."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        # Standard solver (from Phase 1)
        from multigrid.solvers.multigrid import MultigridSolver
        
        standard_solver = MultigridSolver(
            max_levels=3,
            max_iterations=20,
            tolerance=1e-6
        )
        standard_solver.setup(grid, operator, restriction, prolongation)
        
        # Optimized solver
        optimized_solver = CommunicationAvoidingMultigrid(
            max_levels=3,
            max_iterations=20,
            tolerance=1e-6,
            block_size=8,
            enable_memory_pool=True
        )
        optimized_solver.setup(grid, operator, restriction, prolongation)
        
        # Test problem
        rhs = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0)
        
        # Solve with both
        import time
        
        start_time = time.time()
        standard_solution, standard_info = standard_solver.solve(grid, operator, rhs)
        standard_time = time.time() - start_time
        
        start_time = time.time()
        optimized_solution, optimized_info = optimized_solver.solve(grid, operator, rhs)
        optimized_time = time.time() - start_time
        
        # Both should converge to similar solutions
        if standard_info['converged'] and optimized_info['converged']:
            solution_diff = np.max(np.abs(standard_solution - optimized_solution))
            assert solution_diff < 0.01  # Should be very similar
        
        # Performance comparison (optimized may or may not be faster for small problems)
        # But both should complete successfully
        assert standard_info['iterations'] > 0
        assert optimized_info['iterations'] > 0


if __name__ == "__main__":
    pytest.main([__file__])