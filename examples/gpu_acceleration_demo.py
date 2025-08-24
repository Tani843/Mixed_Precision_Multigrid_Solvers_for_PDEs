"""
GPU Acceleration Demo for Mixed-Precision Multigrid Solvers

This example demonstrates the performance benefits of GPU acceleration
and compares different solver configurations.
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from multigrid.core.grid import Grid
    from multigrid.operators.laplacian import LaplacianOperator
    from multigrid.operators.transfer import RestrictionOperator, ProlongationOperator
    from multigrid.solvers.multigrid import MultigridSolver
    
    # Try to import GPU modules
    try:
        from multigrid.gpu.gpu_solver import GPUMultigridSolver, GPUCommunicationAvoidingMultigrid
        from multigrid.gpu.gpu_benchmark import GPUBenchmarkSuite, run_quick_gpu_benchmark
        from multigrid.gpu.memory_manager import check_gpu_availability
        from multigrid.gpu.multi_gpu import DistributedMultigridSolver
        GPU_AVAILABLE = True
    except ImportError as e:
        print(f"GPU modules not available: {e}")
        GPU_AVAILABLE = False
        
except ImportError as e:
    print(f"Error importing multigrid modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def create_test_problem(nx: int, ny: int):
    """Create a test Poisson problem with known analytical solution."""
    grid = Grid(nx=nx, ny=ny, domain=(0, 1, 0, 1))
    
    # Analytical solution: u(x,y) = sin(πx)sin(πy)
    u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
    
    # Right-hand side: f = -∇²u = 2π²sin(πx)sin(πy)
    rhs = 2 * np.pi**2 * u_exact
    
    # Apply homogeneous Dirichlet boundary conditions
    grid.apply_dirichlet_bc(0.0)
    
    return grid, rhs, u_exact


def benchmark_cpu_solver(grid, operator, restriction, prolongation, rhs):
    """Benchmark CPU multigrid solver."""
    print(f"  CPU Multigrid Solver ({grid.shape})...")
    
    solver = MultigridSolver(
        max_levels=6,
        max_iterations=100,
        tolerance=1e-6,
        cycle_type="V"
    )
    
    solver.setup(grid, operator, restriction, prolongation)
    
    start_time = time.time()
    solution, info = solver.solve(grid, operator, rhs.copy())
    end_time = time.time()
    
    return {
        'solver_type': 'CPU Multigrid',
        'solve_time': end_time - start_time,
        'iterations': info['iterations'],
        'final_residual': info['final_residual'],
        'converged': info['converged'],
        'solution': solution
    }


def benchmark_gpu_solver(grid, operator, restriction, prolongation, rhs, device_id=0):
    """Benchmark GPU multigrid solver."""
    print(f"  GPU Multigrid Solver ({grid.shape})...")
    
    solver = GPUMultigridSolver(
        device_id=device_id,
        max_levels=6,
        max_iterations=100,
        tolerance=1e-6,
        cycle_type="V",
        enable_mixed_precision=True,
        use_tensor_cores=True
    )
    
    solver.setup(grid, operator, restriction, prolongation)
    
    start_time = time.time()
    solution, info = solver.solve(grid, operator, rhs.copy())
    end_time = time.time()
    
    # Get GPU statistics
    gpu_stats = solver.get_performance_statistics()
    
    return {
        'solver_type': 'GPU Multigrid',
        'solve_time': end_time - start_time,
        'iterations': info['iterations'],
        'final_residual': info['final_residual'],
        'converged': info['converged'],
        'solution': solution,
        'gpu_stats': gpu_stats
    }


def benchmark_gpu_ca_solver(grid, operator, restriction, prolongation, rhs, device_id=0):
    """Benchmark GPU Communication-Avoiding multigrid solver."""
    print(f"  GPU CA-Multigrid Solver ({grid.shape})...")
    
    solver = GPUCommunicationAvoidingMultigrid(
        device_id=device_id,
        max_levels=6,
        max_iterations=100,
        tolerance=1e-6,
        cycle_type="V",
        block_size=32,
        enable_memory_pool=True,
        use_fmg=True,
        enable_mixed_precision=True
    )
    
    solver.setup(grid, operator, restriction, prolongation)
    
    start_time = time.time()
    solution, info = solver.solve(grid, operator, rhs.copy())
    end_time = time.time()
    
    # Get performance statistics
    ca_stats = solver.get_performance_statistics()
    
    return {
        'solver_type': 'GPU CA-Multigrid',
        'solve_time': end_time - start_time,
        'iterations': info['iterations'],
        'final_residual': info['final_residual'],
        'converged': info['converged'],
        'solution': solution,
        'ca_stats': ca_stats
    }


def run_performance_comparison():
    """Run comprehensive performance comparison."""
    print("=" * 60)
    print("GPU Multigrid Solver Performance Comparison")
    print("=" * 60)
    
    # Check GPU availability
    if not GPU_AVAILABLE:
        print("GPU acceleration not available. Running CPU-only comparison.")
        return run_cpu_only_comparison()
    
    gpu_info = check_gpu_availability()
    if not gpu_info['cupy_available']:
        print("CuPy not available. Please install CuPy for GPU acceleration.")
        return run_cpu_only_comparison()
    
    print(f"Found {gpu_info['gpu_count']} GPU(s)")
    for i, device in enumerate(gpu_info['devices']):
        print(f"  GPU {i}: {device['name']} ({device['total_memory_mb']:.0f} MB)")
    
    # Test problem sizes
    problem_sizes = [(129, 129), (257, 257), (513, 513)]
    
    # Operators
    operator = LaplacianOperator()
    restriction = RestrictionOperator("full_weighting")
    prolongation = ProlongationOperator("bilinear")
    
    results = []
    
    for nx, ny in problem_sizes:
        print(f"\nTesting problem size: {nx}x{ny} = {nx*ny:,} unknowns")
        
        # Create test problem
        grid, rhs, u_exact = create_test_problem(nx, ny)
        
        problem_results = {
            'problem_size': (nx, ny),
            'total_unknowns': nx * ny,
            'solvers': []
        }
        
        # Benchmark CPU solver
        try:
            cpu_result = benchmark_cpu_solver(grid, operator, restriction, prolongation, rhs)
            cpu_result['solution_error'] = np.max(np.abs(cpu_result['solution'] - u_exact))
            problem_results['solvers'].append(cpu_result)
        except Exception as e:
            print(f"    CPU solver failed: {e}")
        
        # Benchmark GPU solver
        try:
            gpu_result = benchmark_gpu_solver(grid, operator, restriction, prolongation, rhs)
            gpu_result['solution_error'] = np.max(np.abs(gpu_result['solution'] - u_exact))
            problem_results['solvers'].append(gpu_result)
        except Exception as e:
            print(f"    GPU solver failed: {e}")
        
        # Benchmark GPU CA solver
        try:
            gpu_ca_result = benchmark_gpu_ca_solver(grid, operator, restriction, prolongation, rhs)
            gpu_ca_result['solution_error'] = np.max(np.abs(gpu_ca_result['solution'] - u_exact))
            problem_results['solvers'].append(gpu_ca_result)
        except Exception as e:
            print(f"    GPU CA solver failed: {e}")
        
        results.append(problem_results)
        
        # Print results for this problem size
        print_problem_results(problem_results)
    
    # Generate summary report
    print_summary_report(results)
    
    # Generate plots if possible
    try:
        create_performance_plots(results)
    except Exception as e:
        print(f"Failed to create plots: {e}")
    
    return results


def run_cpu_only_comparison():
    """Run CPU-only performance comparison."""
    print("\nRunning CPU-only performance comparison...")
    
    problem_sizes = [(65, 65), (129, 129), (257, 257)]
    operator = LaplacianOperator()
    restriction = RestrictionOperator("full_weighting")
    prolongation = ProlongationOperator("bilinear")
    
    results = []
    
    for nx, ny in problem_sizes:
        print(f"\nTesting problem size: {nx}x{ny} = {nx*ny:,} unknowns")
        
        grid, rhs, u_exact = create_test_problem(nx, ny)
        
        cpu_result = benchmark_cpu_solver(grid, operator, restriction, prolongation, rhs)
        cpu_result['solution_error'] = np.max(np.abs(cpu_result['solution'] - u_exact))
        
        results.append({
            'problem_size': (nx, ny),
            'total_unknowns': nx * ny,
            'solvers': [cpu_result]
        })
        
        print_problem_results(results[-1])
    
    return results


def print_problem_results(problem_results):
    """Print results for a single problem size."""
    print(f"\n  Results for {problem_results['problem_size']} grid:")
    print("  " + "-" * 50)
    
    # Find CPU baseline
    cpu_time = None
    for solver_result in problem_results['solvers']:
        if solver_result['solver_type'] == 'CPU Multigrid':
            cpu_time = solver_result['solve_time']
            break
    
    for solver_result in problem_results['solvers']:
        solve_time = solver_result['solve_time']
        speedup = cpu_time / solve_time if cpu_time else 1.0
        
        print(f"  {solver_result['solver_type']:20s}: "
              f"{solve_time:6.3f}s ({speedup:4.1f}x) "
              f"[{solver_result['iterations']:2d} iters, "
              f"residual={solver_result['final_residual']:.2e}, "
              f"error={solver_result['solution_error']:.2e}]")


def print_summary_report(results):
    """Print comprehensive summary report."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY REPORT")
    print("=" * 60)
    
    # Collect all solver types
    solver_types = set()
    for problem_result in results:
        for solver_result in problem_result['solvers']:
            solver_types.add(solver_result['solver_type'])
    
    solver_types = sorted(solver_types)
    
    # Performance summary table
    print("\nPerformance Summary (Time in seconds):")
    print("-" * 80)
    print(f"{'Problem Size':>15s} {'Total Points':>12s}", end='')
    for solver_type in solver_types:
        print(f"{solver_type:>15s}", end='')
    print()
    print("-" * 80)
    
    for problem_result in results:
        nx, ny = problem_result['problem_size']
        total_points = problem_result['total_unknowns']
        
        print(f"{nx:d}x{ny:d}":>15s, f"{total_points:,}":>12s, end='')
        
        # Find results for each solver type
        solver_results = {sr['solver_type']: sr for sr in problem_result['solvers']}
        
        for solver_type in solver_types:
            if solver_type in solver_results:
                time_str = f"{solver_results[solver_type]['solve_time']:.3f}s"
            else:
                time_str = "N/A"
            print(f"{time_str:>15s}", end='')
        print()
    
    # Speedup summary
    if len(solver_types) > 1 and 'CPU Multigrid' in solver_types:
        print("\nSpeedup vs CPU (x times faster):")
        print("-" * 60)
        print(f"{'Problem Size':>15s} {'Total Points':>12s}", end='')
        gpu_solvers = [st for st in solver_types if 'GPU' in st]
        for solver_type in gpu_solvers:
            print(f"{solver_type:>15s}", end='')
        print()
        print("-" * 60)
        
        for problem_result in results:
            nx, ny = problem_result['problem_size']
            total_points = problem_result['total_unknowns']
            
            solver_results = {sr['solver_type']: sr for sr in problem_result['solvers']}
            cpu_time = solver_results.get('CPU Multigrid', {}).get('solve_time', 0)
            
            print(f"{nx:d}x{ny:d}":>15s, f"{total_points:,}":>12s, end='')
            
            for solver_type in gpu_solvers:
                if solver_type in solver_results and cpu_time > 0:
                    gpu_time = solver_results[solver_type]['solve_time']
                    speedup = cpu_time / gpu_time
                    speedup_str = f"{speedup:.1f}x"
                else:
                    speedup_str = "N/A"
                print(f"{speedup_str:>15s}", end='')
            print()
    
    # Accuracy summary
    print("\nSolution Accuracy (Maximum Error):")
    print("-" * 60)
    print(f"{'Problem Size':>15s} {'Total Points':>12s}", end='')
    for solver_type in solver_types:
        print(f"{solver_type:>15s}", end='')
    print()
    print("-" * 60)
    
    for problem_result in results:
        nx, ny = problem_result['problem_size']
        total_points = problem_result['total_unknowns']
        
        print(f"{nx:d}x{ny:d}":>15s, f"{total_points:,}":>12s, end='')
        
        solver_results = {sr['solver_type']: sr for sr in problem_result['solvers']}
        
        for solver_type in solver_types:
            if solver_type in solver_results:
                error = solver_results[solver_type]['solution_error']
                error_str = f"{error:.2e}"
            else:
                error_str = "N/A"
            print(f"{error_str:>15s}", end='')
        print()


def create_performance_plots(results):
    """Create performance visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    # Extract data for plotting
    problem_sizes = [r['total_unknowns'] for r in results]
    solver_data = {}
    
    for problem_result in results:
        for solver_result in problem_result['solvers']:
            solver_type = solver_result['solver_type']
            if solver_type not in solver_data:
                solver_data[solver_type] = {'times': [], 'speedups': [], 'errors': []}
            
            solver_data[solver_type]['times'].append(solver_result['solve_time'])
            solver_data[solver_type]['errors'].append(solver_result['solution_error'])
    
    # Calculate speedups
    if 'CPU Multigrid' in solver_data:
        cpu_times = solver_data['CPU Multigrid']['times']
        for solver_type, data in solver_data.items():
            if solver_type != 'CPU Multigrid' and len(data['times']) == len(cpu_times):
                data['speedups'] = [cpu_times[i] / data['times'][i] for i in range(len(cpu_times))]
            else:
                data['speedups'] = [1.0] * len(data['times'])
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Solve times vs problem size
    ax1.set_title('Solve Time vs Problem Size')
    ax1.set_xlabel('Number of Unknowns')
    ax1.set_ylabel('Solve Time (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    for solver_type, data in solver_data.items():
        if data['times']:
            ax1.plot(problem_sizes[:len(data['times'])], data['times'], 
                    'o-', label=solver_type, linewidth=2, markersize=6)
    
    ax1.legend()
    
    # Plot 2: Speedup vs problem size
    ax2.set_title('Speedup vs Problem Size')
    ax2.set_xlabel('Number of Unknowns')
    ax2.set_ylabel('Speedup (x times faster)')
    ax2.set_xscale('log')
    ax2.grid(True)
    
    for solver_type, data in solver_data.items():
        if solver_type != 'CPU Multigrid' and data['speedups']:
            ax2.plot(problem_sizes[:len(data['speedups'])], data['speedups'], 
                    'o-', label=solver_type, linewidth=2, markersize=6)
    
    ax2.legend()
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='CPU Baseline')
    
    # Plot 3: Solution accuracy
    ax3.set_title('Solution Accuracy vs Problem Size')
    ax3.set_xlabel('Number of Unknowns')
    ax3.set_ylabel('Maximum Error')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    for solver_type, data in solver_data.items():
        if data['errors']:
            ax3.plot(problem_sizes[:len(data['errors'])], data['errors'], 
                    'o-', label=solver_type, linewidth=2, markersize=6)
    
    ax3.legend()
    
    # Plot 4: Throughput (points per second)
    ax4.set_title('Throughput vs Problem Size')
    ax4.set_xlabel('Number of Unknowns')
    ax4.set_ylabel('Throughput (unknowns/sec)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True)
    
    for solver_type, data in solver_data.items():
        if data['times']:
            throughputs = [problem_sizes[i] / data['times'][i] 
                          for i in range(min(len(problem_sizes), len(data['times'])))]
            ax4.plot(problem_sizes[:len(throughputs)], throughputs, 
                    'o-', label=solver_type, linewidth=2, markersize=6)
    
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPerformance plots saved to 'gpu_performance_comparison.png'")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass


def run_comprehensive_gpu_benchmark():
    """Run comprehensive GPU benchmark suite."""
    if not GPU_AVAILABLE:
        print("GPU benchmarking not available")
        return
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE GPU BENCHMARK SUITE")
    print("=" * 60)
    
    try:
        benchmark_suite = GPUBenchmarkSuite(device_id=0, enable_profiling=True)
        
        # Run comprehensive benchmark
        results = benchmark_suite.run_comprehensive_benchmark(
            problem_sizes=[(129, 129), (257, 257), (513, 513)],
            solver_types=['cpu_multigrid', 'gpu_multigrid', 'gpu_ca_multigrid'],
            precision_levels=['single', 'mixed_tc'],
            num_runs=3
        )
        
        # Print benchmark report
        report = benchmark_suite.generate_benchmark_report()
        print("\n" + report)
        
        # Export results
        benchmark_suite.export_results('gpu_benchmark_results.json')
        print("\nBenchmark results exported to 'gpu_benchmark_results.json'")
        
        return results
        
    except Exception as e:
        print(f"Comprehensive benchmark failed: {e}")
        return None


def demonstrate_multi_gpu():
    """Demonstrate multi-GPU capabilities."""
    if not GPU_AVAILABLE:
        print("Multi-GPU demonstration not available")
        return
    
    gpu_info = check_gpu_availability()
    if gpu_info['gpu_count'] < 2:
        print("Multi-GPU demonstration requires at least 2 GPUs")
        return
    
    print("\n" + "=" * 60)
    print("MULTI-GPU DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Create larger problem for multi-GPU
        nx, ny = 513, 513
        grid, rhs, u_exact = create_test_problem(nx, ny)
        
        operator = LaplacianOperator()
        restriction = RestrictionOperator("full_weighting")
        prolongation = ProlongationOperator("bilinear")
        
        print(f"Solving {nx}x{ny} problem using {gpu_info['gpu_count']} GPUs...")
        
        # Create distributed solver
        distributed_solver = DistributedMultigridSolver(
            device_ids=list(range(gpu_info['gpu_count'])),
            decomposition_strategy="stripe",
            max_levels=6,
            max_iterations=100,
            tolerance=1e-6
        )
        
        distributed_solver.setup(grid, operator, restriction, prolongation)
        
        start_time = time.time()
        solution, info = distributed_solver.solve(grid, operator, rhs)
        end_time = time.time()
        
        solution_error = np.max(np.abs(solution - u_exact))
        
        print(f"\nMulti-GPU Results:")
        print(f"  Solve Time: {end_time - start_time:.3f}s")
        print(f"  Iterations: {info['total_iterations']}")
        print(f"  Final Residual: {info['final_residual']:.2e}")
        print(f"  Solution Error: {solution_error:.2e}")
        print(f"  Converged: {info['converged']}")
        
        # Get performance statistics
        perf_stats = distributed_solver.get_performance_statistics()
        print(f"\nPerformance Statistics:")
        print(f"  Number of Devices: {perf_stats['multi_gpu_stats']['num_devices']}")
        print(f"  Decomposition Strategy: {perf_stats['multi_gpu_stats']['decomposition_strategy']}")
        
        distributed_solver.cleanup()
        
    except Exception as e:
        print(f"Multi-GPU demonstration failed: {e}")


def main():
    """Main demonstration function."""
    print("Mixed-Precision Multigrid GPU Acceleration Demo")
    print("=" * 60)
    
    # Run performance comparison
    performance_results = run_performance_comparison()
    
    # Run comprehensive GPU benchmark if available
    if GPU_AVAILABLE:
        comprehensive_results = run_comprehensive_gpu_benchmark()
        
        # Demonstrate multi-GPU if multiple GPUs available
        demonstrate_multi_gpu()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)
    
    if GPU_AVAILABLE:
        print("\nKey Findings:")
        print("• GPU acceleration provides significant speedup for large problems")
        print("• Mixed-precision with Tensor Cores improves performance further")
        print("• Communication-avoiding optimizations reduce memory overhead")
        print("• Multi-GPU scaling enables solving larger problems")
        print("\nFiles generated:")
        print("• gpu_performance_comparison.png - Performance visualization")
        print("• gpu_benchmark_results.json - Detailed benchmark data")
    else:
        print("\nTo see GPU acceleration benefits:")
        print("• Install CuPy: pip install cupy")
        print("• Ensure CUDA-capable GPU is available")
        print("• Re-run this demonstration")


if __name__ == "__main__":
    main()