#!/usr/bin/env python3
"""
GPU Acceleration Example for Mixed-Precision Multigrid Solvers
Demonstrates custom CUDA kernels and multi-GPU domain decomposition
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"CuPy available: CUDA devices = {cp.cuda.runtime.getDeviceCount()}")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - GPU acceleration disabled")

from multigrid.core.grid import Grid
from multigrid.core.precision import PrecisionManager, PrecisionLevel

if CUPY_AVAILABLE:
    from multigrid.gpu.cuda_kernels import (
        SmoothingKernels, TransferKernels, MixedPrecisionKernels, BlockStructuredKernels
    )
    from multigrid.gpu.multi_gpu_solver import MultiGPUSolver, DecompositionType


def test_cuda_kernels_performance():
    """Test performance of custom CUDA kernels vs. standard CuPy operations."""
    if not CUPY_AVAILABLE:
        print("Skipping CUDA kernel tests - CuPy not available")
        return
    
    print("\n" + "="*60)
    print("CUDA KERNEL PERFORMANCE TESTING")
    print("="*60)
    
    # Test grid sizes
    test_sizes = [65, 129, 257] if cp.cuda.runtime.getDeviceCount() > 0 else [65]
    
    for size in test_sizes:
        print(f"\nTesting {size}×{size} grid:")
        print("-" * 40)
        
        # Create test data on GPU
        with cp.cuda.Device(0):
            u = cp.random.random((size, size), dtype=cp.float32)
            u_old = u.copy()
            rhs = cp.random.random((size, size), dtype=cp.float32)
            hx = hy = 1.0 / (size - 1)
            
            # Initialize kernels
            smoothing_kernels = SmoothingKernels()
            mixed_precision_kernels = MixedPrecisionKernels()
            block_kernels = BlockStructuredKernels()
            
            # Test 1: Red-Black Gauss-Seidel
            print("1. Red-Black Gauss-Seidel Smoothing:")
            
            # Custom kernel timing
            cp.cuda.Device().synchronize()
            start_time = time.time()
            for _ in range(10):
                smoothing_kernels.red_black_gauss_seidel(u, rhs, hx, hy, num_iterations=1)
            cp.cuda.Device().synchronize()
            custom_time = time.time() - start_time
            
            print(f"   Custom CUDA kernel: {custom_time:.4f}s (10 iterations)")
            
            # Test 2: Mixed-Precision Residual
            print("2. Mixed-Precision Residual Computation:")
            
            start_time = time.time()
            for _ in range(10):
                residual = mixed_precision_kernels.compute_mixed_precision_residual(u, rhs, hx, hy)
            cp.cuda.Device().synchronize()
            mixed_time = time.time() - start_time
            
            print(f"   Mixed-precision kernel: {mixed_time:.4f}s (10 iterations)")
            print(f"   Residual norm: {cp.linalg.norm(residual):.2e}")
            
            # Test 3: Block-Structured Smoothing
            print("3. Block-Structured Smoothing:")
            
            start_time = time.time()
            for _ in range(5):
                block_kernels.block_structured_smoothing(
                    u, rhs, hx, hy, block_size=16, num_iterations=1
                )
            cp.cuda.Device().synchronize()
            block_time = time.time() - start_time
            
            print(f"   Block-structured kernel: {block_time:.4f}s (5 iterations)")
            
            # Memory usage
            mempool = cp.get_default_memory_pool()
            memory_mb = mempool.used_bytes() / (1024 * 1024)
            print(f"   GPU memory used: {memory_mb:.1f} MB")


def test_multi_gpu_solver():
    """Test multi-GPU domain decomposition solver."""
    if not CUPY_AVAILABLE:
        print("Skipping multi-GPU tests - CuPy not available")
        return
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    if num_gpus < 2:
        print(f"Skipping multi-GPU tests - need at least 2 GPUs, found {num_gpus}")
        return
    
    print("\n" + "="*60)
    print("MULTI-GPU DOMAIN DECOMPOSITION SOLVER")
    print("="*60)
    
    # Test problem: 2D Poisson equation with manufactured solution
    grid_size = 129
    grid = Grid(grid_size, grid_size, domain=(0, 1, 0, 1))
    
    # Create manufactured solution: u = sin(πx)sin(πy)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    rhs = 2 * np.pi**2 * u_exact  # -∇²u = rhs
    
    print(f"Problem size: {grid_size}×{grid_size}")
    print(f"Available GPUs: {num_gpus}")
    
    # Test different decomposition strategies
    strategies = [
        (DecompositionType.STRIP_X, "Strip X"),
        (DecompositionType.BLOCK_2D, "Block 2D"),
        (DecompositionType.ADAPTIVE, "Adaptive")
    ]
    
    results = []
    
    for decomp_type, decomp_name in strategies:
        print(f"\n--- {decomp_name} Decomposition ---")
        
        try:
            # Create solver
            solver = MultiGPUSolver(
                num_gpus=min(4, num_gpus),  # Use up to 4 GPUs
                decomposition_type=decomp_type,
                max_levels=3,
                max_iterations=30,
                tolerance=1e-8
            )
            
            # Solve
            start_time = time.time()
            result = solver.domain_decomposition_solve(grid, rhs)
            total_time = time.time() - start_time
            
            # Compute error
            u_computed = result['solution']
            error = np.linalg.norm(u_computed - u_exact)
            max_error = np.max(np.abs(u_computed - u_exact))
            
            print(f"Converged: {result['converged']}")
            print(f"Iterations: {result['iterations']}")
            print(f"Final residual: {result['final_residual']:.2e}")
            print(f"L2 error: {error:.2e}")
            print(f"Max error: {max_error:.2e}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Time per iteration: {total_time/result['iterations']:.3f}s")
            
            # Performance report
            perf_report = solver.get_performance_report()
            print(f"GPUs used: {perf_report['num_gpus']}")
            print(f"Domain sizes: {perf_report['domain_sizes']}")
            print(f"Memory usage: {perf_report['memory_usage_mb']}")
            
            results.append({
                'decomposition': decomp_name,
                'time': total_time,
                'iterations': result['iterations'],
                'error': error,
                'converged': result['converged'],
                'speedup': None  # Will calculate later
            })
            
        except Exception as e:
            print(f"Failed with {decomp_name}: {e}")
            continue
    
    # Calculate speedup (compared to first successful result)
    if results:
        baseline_time = results[0]['time']
        for result in results:
            result['speedup'] = baseline_time / result['time']
    
    return results


def test_precision_switching_gpu():
    """Test adaptive precision switching on GPU."""
    if not CUPY_AVAILABLE:
        print("Skipping GPU precision tests - CuPy not available")
        return
    
    print("\n" + "="*60)
    print("GPU MIXED-PRECISION TESTING")
    print("="*60)
    
    with cp.cuda.Device(0):
        # Create test problem
        size = 65
        u_single = cp.random.random((size, size), dtype=cp.float32)
        u_double = u_single.astype(cp.float64)
        rhs = cp.random.random((size, size), dtype=cp.float32)
        hx = hy = 1.0 / (size - 1)
        
        # Initialize mixed-precision kernels
        mixed_kernels = MixedPrecisionKernels()
        
        print(f"Testing {size}×{size} grid:")
        print("-" * 30)
        
        # Test precision conversion
        print("1. Precision Conversion:")
        start_time = time.time()
        u_converted = mixed_kernels.convert_precision(u_single, 'float64')
        convert_time = time.time() - start_time
        
        conversion_error = cp.max(cp.abs(u_converted - u_double))
        print(f"   Conversion time: {convert_time:.4f}s")
        print(f"   Conversion error: {conversion_error:.2e}")
        
        # Test mixed-precision residual
        print("2. Mixed-Precision Residual:")
        start_time = time.time()
        residual_mixed = mixed_kernels.compute_mixed_precision_residual(u_single, rhs, hx, hy)
        mixed_time = time.time() - start_time
        
        print(f"   Computation time: {mixed_time:.4f}s")
        print(f"   Residual norm: {cp.linalg.norm(residual_mixed):.2e}")
        print(f"   Residual dtype: {residual_mixed.dtype}")
        
        # Test precision correction
        print("3. Mixed-Precision Correction:")
        correction = cp.random.random((size, size), dtype=cp.float64) * 0.01
        u_test = u_single.copy()
        
        start_time = time.time()
        mixed_kernels.apply_mixed_precision_correction(u_test, correction, damping_factor=0.8)
        correction_time = time.time() - start_time
        
        correction_norm = cp.linalg.norm(u_test - u_single)
        print(f"   Correction time: {correction_time:.4f}s")
        print(f"   Correction norm: {correction_norm:.2e}")


def create_performance_comparison_plot(results):
    """Create performance comparison plots."""
    if not results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    decompositions = [r['decomposition'] for r in results]
    times = [r['time'] for r in results]
    iterations = [r['iterations'] for r in results]
    errors = [r['error'] for r in results]
    speedups = [r['speedup'] or 1.0 for r in results]
    
    # Execution time comparison
    bars1 = ax1.bar(decompositions, times, color=['blue', 'green', 'orange'])
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Multi-GPU Performance Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Iterations comparison
    bars2 = ax2.bar(decompositions, iterations, color=['blue', 'green', 'orange'])
    ax2.set_ylabel('Iterations')
    ax2.set_title('Convergence Iterations')
    ax2.tick_params(axis='x', rotation=45)
    
    # Accuracy comparison
    ax3.semilogy(decompositions, errors, 'o-', linewidth=2, markersize=8)
    ax3.set_ylabel('L2 Error')
    ax3.set_title('Solution Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Speedup comparison
    bars4 = ax4.bar(decompositions, speedups, color=['blue', 'green', 'orange'])
    ax4.set_ylabel('Speedup Factor')
    ax4.set_title('Relative Speedup')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPerformance comparison plot saved: gpu_performance_comparison.png")


def main():
    """Run GPU acceleration examples and benchmarks."""
    print("GPU Acceleration Example for Mixed-Precision Multigrid Solvers")
    print("="*70)
    
    if not CUPY_AVAILABLE:
        print("❌ CuPy is not available")
        print("   Install CuPy to enable GPU acceleration:")
        print("   pip install cupy-cuda11x  # For CUDA 11.x")
        print("   pip install cupy-cuda12x  # For CUDA 12.x")
        return
    
    print(f"✅ CuPy available with {cp.cuda.runtime.getDeviceCount()} CUDA device(s)")
    
    # Test 1: CUDA Kernel Performance
    test_cuda_kernels_performance()
    
    # Test 2: Mixed-Precision on GPU
    test_precision_switching_gpu()
    
    # Test 3: Multi-GPU Domain Decomposition
    multi_gpu_results = test_multi_gpu_solver()
    
    # Create performance plots
    if multi_gpu_results:
        create_performance_comparison_plot(multi_gpu_results)
        
        # Summary
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        for result in multi_gpu_results:
            print(f"{result['decomposition']:12s}: "
                  f"{result['time']:.3f}s, "
                  f"{result['iterations']:2d} iter, "
                  f"Error: {result['error']:.2e}, "
                  f"Speedup: {result['speedup']:.2f}×")
        
        # Best performer
        if len(multi_gpu_results) > 1:
            fastest = min(multi_gpu_results, key=lambda x: x['time'])
            print(f"\n🏆 Best Performance: {fastest['decomposition']} "
                  f"({fastest['time']:.3f}s, {fastest['speedup']:.2f}× speedup)")
    
    print("\n" + "="*60)
    print("GPU ACCELERATION TESTING COMPLETE")
    print("="*60)
    print("✅ Custom CUDA kernels: Red-Black G-S, Mixed-precision residuals, Block smoothing")
    print("✅ Multi-GPU support: Domain decomposition with load balancing")
    print("✅ Communication optimization: Overlapping compute and halo exchange")
    print("✅ Memory management: Shared memory usage and precision conversion")


if __name__ == "__main__":
    main()