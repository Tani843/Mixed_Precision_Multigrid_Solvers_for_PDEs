"""GPU benchmarking suite for multigrid solvers."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass, field

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..core.grid import Grid
from ..operators.laplacian import LaplacianOperator
from ..operators.transfer import RestrictionOperator, ProlongationOperator
from ..solvers.multigrid import MultigridSolver
from .gpu_solver import GPUMultigridSolver, GPUCommunicationAvoidingMultigrid
from .gpu_profiler import GPUPerformanceProfiler
from .memory_manager import check_gpu_availability

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    problem_size: Tuple[int, int]
    solver_type: str
    solve_time: float
    iterations: int
    final_residual: float
    memory_usage_mb: float
    throughput_points_per_sec: float
    speedup_vs_cpu: float = 1.0
    gpu_utilization: float = 0.0
    precision_used: str = "unknown"
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class GPUBenchmarkSuite:
    """
    Comprehensive GPU benchmarking suite for multigrid solvers.
    
    Compares GPU vs CPU performance across different problem sizes,
    precision levels, and solver configurations.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        enable_profiling: bool = True,
        warmup_iterations: int = 3
    ):
        """
        Initialize GPU benchmark suite.
        
        Args:
            device_id: GPU device ID
            enable_profiling: Enable detailed profiling
            warmup_iterations: Number of warmup iterations
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU benchmarking")
        
        self.device_id = device_id
        self.enable_profiling = enable_profiling
        self.warmup_iterations = warmup_iterations
        
        # Check GPU availability
        self.gpu_info = check_gpu_availability()
        if not self.gpu_info['cupy_available'] or self.gpu_info['gpu_count'] == 0:
            raise RuntimeError("No GPU available for benchmarking")
        
        # Initialize profiler
        if enable_profiling:
            self.profiler = GPUPerformanceProfiler(device_id)
        else:
            self.profiler = None
        
        # Benchmark results storage
        self.benchmark_results: List[BenchmarkResult] = []
        
        logger.info(f"GPU benchmark suite initialized: device={device_id}, "
                   f"GPU={self.gpu_info['devices'][device_id]['name']}")
    
    def run_comprehensive_benchmark(
        self,
        problem_sizes: Optional[List[Tuple[int, int]]] = None,
        solver_types: Optional[List[str]] = None,
        precision_levels: Optional[List[str]] = None,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparing GPU vs CPU performance.
        
        Args:
            problem_sizes: List of (nx, ny) problem sizes
            solver_types: List of solver types to benchmark
            precision_levels: List of precision levels to test
            num_runs: Number of runs per configuration
            
        Returns:
            Comprehensive benchmark results
        """
        if problem_sizes is None:
            problem_sizes = [
                (65, 65), (129, 129), (257, 257), (513, 513), (1025, 1025)
            ]
        
        if solver_types is None:
            solver_types = ['cpu_multigrid', 'gpu_multigrid', 'gpu_ca_multigrid']
        
        if precision_levels is None:
            precision_levels = ['single', 'mixed_tc'] if self._has_tensor_cores() else ['single']
        
        logger.info(f"Starting comprehensive benchmark: {len(problem_sizes)} sizes, "
                   f"{len(solver_types)} solvers, {len(precision_levels)} precisions, "
                   f"{num_runs} runs each")
        
        # Clear previous results
        self.benchmark_results.clear()
        
        total_configs = len(problem_sizes) * len(solver_types) * len(precision_levels)
        current_config = 0
        
        for problem_size in problem_sizes:
            for solver_type in solver_types:
                for precision_level in precision_levels:
                    current_config += 1
                    
                    logger.info(f"Benchmarking {current_config}/{total_configs}: "
                               f"{solver_type} on {problem_size} with {precision_level} precision")
                    
                    # Run multiple times for statistical significance
                    run_results = []
                    for run in range(num_runs):
                        try:
                            result = self._run_single_benchmark(
                                problem_size, solver_type, precision_level
                            )
                            run_results.append(result)
                        except Exception as e:
                            logger.warning(f"Benchmark failed: {e}")
                            continue
                    
                    if run_results:
                        # Compute average result
                        avg_result = self._compute_average_result(run_results)
                        self.benchmark_results.append(avg_result)
        
        # Compute speedups
        self._compute_speedups()
        
        # Generate comprehensive analysis
        analysis = self._analyze_benchmark_results()
        
        logger.info(f"Comprehensive benchmark completed: {len(self.benchmark_results)} results")
        
        return analysis
    
    def _run_single_benchmark(
        self,
        problem_size: Tuple[int, int],
        solver_type: str,
        precision_level: str
    ) -> BenchmarkResult:
        """Run single benchmark configuration."""
        nx, ny = problem_size
        
        # Create test problem
        grid = Grid(nx=nx, ny=ny, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        restriction = RestrictionOperator("full_weighting")
        prolongation = ProlongationOperator("bilinear")
        
        # Generate test problem (Poisson equation with analytical solution)
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u_exact
        
        # Apply boundary conditions
        grid.apply_dirichlet_bc(0.0)
        
        # Create solver based on type
        if solver_type == 'cpu_multigrid':
            solver = MultigridSolver(
                max_levels=6,
                max_iterations=50,
                tolerance=1e-6
            )
            use_gpu = False
        elif solver_type == 'gpu_multigrid':
            solver = GPUMultigridSolver(
                device_id=self.device_id,
                max_levels=6,
                max_iterations=50,
                tolerance=1e-6,
                enable_mixed_precision=(precision_level != 'single')
            )
            use_gpu = True
        elif solver_type == 'gpu_ca_multigrid':
            solver = GPUCommunicationAvoidingMultigrid(
                device_id=self.device_id,
                max_levels=6,
                max_iterations=50,
                tolerance=1e-6,
                enable_mixed_precision=(precision_level != 'single'),
                use_fmg=True
            )
            use_gpu = True
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        # Setup solver
        solver.setup(grid, operator, restriction, prolongation)
        
        # Warmup runs
        if use_gpu:
            for _ in range(self.warmup_iterations):
                try:
                    solver.solve(grid, operator, rhs.copy())
                except:
                    pass
        
        # Measure memory usage before
        memory_before = self._get_memory_usage(use_gpu)
        
        # Benchmark run
        start_time = time.time()
        
        if self.profiler and use_gpu:
            with self.profiler.profile_gpu_operation(f"{solver_type}_solve"):
                solution, info = solver.solve(grid, operator, rhs.copy())
        else:
            solution, info = solver.solve(grid, operator, rhs.copy())
        
        end_time = time.time()
        
        # Measure memory usage after
        memory_after = self._get_memory_usage(use_gpu)
        memory_usage_mb = (memory_after - memory_before) / (1024**2)
        
        # Calculate metrics
        solve_time = end_time - start_time
        total_points = nx * ny
        throughput = total_points * info['iterations'] / solve_time
        
        # Get GPU utilization if available
        gpu_utilization = 0.0
        if use_gpu and hasattr(solver, 'get_performance_statistics'):
            stats = solver.get_performance_statistics()
            gpu_utilization = stats.get('performance_metrics', {}).get('gpu_utilization', 0.0)
        
        # Create benchmark result
        result = BenchmarkResult(
            problem_size=problem_size,
            solver_type=solver_type,
            solve_time=solve_time,
            iterations=info['iterations'],
            final_residual=info['final_residual'],
            memory_usage_mb=memory_usage_mb,
            throughput_points_per_sec=throughput,
            gpu_utilization=gpu_utilization,
            precision_used=precision_level,
            additional_metrics={
                'converged': info['converged'],
                'solution_error': np.max(np.abs(solution - u_exact)),
                'solver_info': info
            }
        )
        
        return result
    
    def _compute_average_result(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Compute average benchmark result from multiple runs."""
        if not results:
            raise ValueError("No results to average")
        
        # Use first result as template
        avg_result = BenchmarkResult(
            problem_size=results[0].problem_size,
            solver_type=results[0].solver_type,
            solve_time=np.mean([r.solve_time for r in results]),
            iterations=int(np.mean([r.iterations for r in results])),
            final_residual=np.mean([r.final_residual for r in results]),
            memory_usage_mb=np.mean([r.memory_usage_mb for r in results]),
            throughput_points_per_sec=np.mean([r.throughput_points_per_sec for r in results]),
            gpu_utilization=np.mean([r.gpu_utilization for r in results]),
            precision_used=results[0].precision_used
        )
        
        # Add statistics as additional metrics
        avg_result.additional_metrics = {
            'num_runs': len(results),
            'solve_time_std': np.std([r.solve_time for r in results]),
            'throughput_std': np.std([r.throughput_points_per_sec for r in results]),
            'all_converged': all(r.additional_metrics.get('converged', False) for r in results)
        }
        
        return avg_result
    
    def _compute_speedups(self) -> None:
        """Compute speedup factors relative to CPU baseline."""
        # Group results by problem size and precision
        cpu_baselines = {}
        
        for result in self.benchmark_results:
            if result.solver_type == 'cpu_multigrid':
                key = (result.problem_size, result.precision_used)
                cpu_baselines[key] = result.solve_time
        
        # Compute speedups for GPU results
        for result in self.benchmark_results:
            key = (result.problem_size, result.precision_used)
            cpu_baseline_key = (result.problem_size, 'single')  # CPU always uses single precision
            
            if cpu_baseline_key in cpu_baselines:
                cpu_time = cpu_baselines[cpu_baseline_key]
                result.speedup_vs_cpu = cpu_time / result.solve_time
            else:
                result.speedup_vs_cpu = 1.0
    
    def _analyze_benchmark_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights."""
        if not self.benchmark_results:
            return {'error': 'No benchmark results available'}
        
        analysis = {
            'summary': self._generate_summary_statistics(),
            'speedup_analysis': self._analyze_speedups(),
            'scaling_analysis': self._analyze_scaling(),
            'precision_analysis': self._analyze_precision_impact(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        gpu_results = [r for r in self.benchmark_results if 'gpu' in r.solver_type]
        cpu_results = [r for r in self.benchmark_results if r.solver_type == 'cpu_multigrid']
        
        summary = {
            'total_benchmarks': len(self.benchmark_results),
            'gpu_benchmarks': len(gpu_results),
            'cpu_benchmarks': len(cpu_results),
            'problem_sizes_tested': list(set(r.problem_size for r in self.benchmark_results)),
            'solver_types_tested': list(set(r.solver_type for r in self.benchmark_results)),
            'precision_levels_tested': list(set(r.precision_used for r in self.benchmark_results))
        }
        
        if gpu_results:
            speedups = [r.speedup_vs_cpu for r in gpu_results if r.speedup_vs_cpu > 0]
            if speedups:
                summary.update({
                    'best_speedup': max(speedups),
                    'average_speedup': np.mean(speedups),
                    'worst_speedup': min(speedups),
                    'speedup_std': np.std(speedups)
                })
        
        return summary
    
    def _analyze_speedups(self) -> Dict[str, Any]:
        """Analyze speedup characteristics."""
        speedup_analysis = {}
        
        # Group by solver type
        for solver_type in set(r.solver_type for r in self.benchmark_results):
            if 'gpu' not in solver_type:
                continue
            
            solver_results = [r for r in self.benchmark_results if r.solver_type == solver_type]
            speedups = [r.speedup_vs_cpu for r in solver_results if r.speedup_vs_cpu > 0]
            
            if speedups:
                speedup_analysis[solver_type] = {
                    'mean_speedup': np.mean(speedups),
                    'max_speedup': max(speedups),
                    'min_speedup': min(speedups),
                    'std_speedup': np.std(speedups),
                    'speedup_above_2x': sum(1 for s in speedups if s >= 2.0),
                    'speedup_above_5x': sum(1 for s in speedups if s >= 5.0),
                    'speedup_above_10x': sum(1 for s in speedups if s >= 10.0)
                }
        
        return speedup_analysis
    
    def _analyze_scaling(self) -> Dict[str, Any]:
        """Analyze scaling behavior with problem size."""
        scaling_analysis = {}
        
        # Group by solver type
        for solver_type in set(r.solver_type for r in self.benchmark_results):
            solver_results = [r for r in self.benchmark_results if r.solver_type == solver_type]
            
            # Sort by problem size
            solver_results.sort(key=lambda r: np.prod(r.problem_size))
            
            if len(solver_results) >= 2:
                sizes = [np.prod(r.problem_size) for r in solver_results]
                times = [r.solve_time for r in solver_results]
                throughputs = [r.throughput_points_per_sec for r in solver_results]
                
                # Estimate scaling exponent (log-log fit)
                try:
                    log_sizes = np.log(sizes)
                    log_times = np.log(times)
                    scaling_coeff = np.polyfit(log_sizes, log_times, 1)[0]
                    
                    scaling_analysis[solver_type] = {
                        'scaling_exponent': scaling_coeff,
                        'scaling_interpretation': self._interpret_scaling(scaling_coeff),
                        'throughput_range': [min(throughputs), max(throughputs)],
                        'size_range': [min(sizes), max(sizes)]
                    }
                except Exception:
                    scaling_analysis[solver_type] = {'error': 'scaling_analysis_failed'}
        
        return scaling_analysis
    
    def _interpret_scaling(self, exponent: float) -> str:
        """Interpret scaling exponent."""
        if exponent < 1.1:
            return "nearly_linear"
        elif exponent < 1.5:
            return "better_than_n_log_n"
        elif exponent < 2.1:
            return "approximately_quadratic"
        else:
            return "worse_than_quadratic"
    
    def _analyze_precision_impact(self) -> Dict[str, Any]:
        """Analyze impact of different precision levels."""
        precision_analysis = {}
        
        precisions = set(r.precision_used for r in self.benchmark_results)
        
        for precision in precisions:
            precision_results = [r for r in self.benchmark_results if r.precision_used == precision]
            
            if precision_results:
                speedups = [r.speedup_vs_cpu for r in precision_results if r.speedup_vs_cpu > 0]
                accuracies = [r.additional_metrics.get('solution_error', 0) for r in precision_results]
                
                precision_analysis[precision] = {
                    'num_results': len(precision_results),
                    'mean_speedup': np.mean(speedups) if speedups else 0,
                    'mean_accuracy': np.mean(accuracies) if accuracies else 0,
                    'convergence_rate': sum(1 for r in precision_results 
                                         if r.additional_metrics.get('all_converged', False)) / len(precision_results)
                }
        
        return precision_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.benchmark_results:
            return recommendations
        
        # Find best performing configurations
        gpu_results = [r for r in self.benchmark_results if 'gpu' in r.solver_type]
        
        if gpu_results:
            best_result = max(gpu_results, key=lambda r: r.speedup_vs_cpu)
            
            recommendations.append(
                f"Best GPU performance: {best_result.solver_type} with {best_result.precision_used} "
                f"precision achieved {best_result.speedup_vs_cpu:.1f}x speedup on {best_result.problem_size}"
            )
        
        # Analyze scaling trends
        large_problem_results = [r for r in gpu_results if np.prod(r.problem_size) > 100000]
        small_problem_results = [r for r in gpu_results if np.prod(r.problem_size) <= 10000]
        
        if large_problem_results and small_problem_results:
            large_avg_speedup = np.mean([r.speedup_vs_cpu for r in large_problem_results])
            small_avg_speedup = np.mean([r.speedup_vs_cpu for r in small_problem_results])
            
            if large_avg_speedup > small_avg_speedup * 1.5:
                recommendations.append(
                    f"GPU shows better performance on larger problems "
                    f"({large_avg_speedup:.1f}x vs {small_avg_speedup:.1f}x speedup)"
                )
        
        # Memory utilization recommendations
        high_memory_results = [r for r in gpu_results if r.memory_usage_mb > 1000]
        if high_memory_results:
            avg_speedup = np.mean([r.speedup_vs_cpu for r in high_memory_results])
            recommendations.append(
                f"Large problems (>1GB GPU memory) achieve average {avg_speedup:.1f}x speedup"
            )
        
        # Precision recommendations
        if any(r.precision_used == 'mixed_tc' for r in gpu_results):
            mixed_results = [r for r in gpu_results if r.precision_used == 'mixed_tc']
            single_results = [r for r in gpu_results if r.precision_used == 'single']
            
            if mixed_results and single_results:
                mixed_speedup = np.mean([r.speedup_vs_cpu for r in mixed_results])
                single_speedup = np.mean([r.speedup_vs_cpu for r in single_results])
                
                if mixed_speedup > single_speedup * 1.2:
                    recommendations.append(
                        f"Mixed precision with Tensor Cores provides {mixed_speedup:.1f}x vs "
                        f"{single_speedup:.1f}x speedup for single precision"
                    )
        
        return recommendations
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.benchmark_results:
            return "No benchmark results available"
        
        analysis = self._analyze_benchmark_results()
        
        report = [
            "GPU Multigrid Solver Benchmark Report",
            "=" * 60,
            "",
            "Hardware Configuration:",
            f"  GPU: {self.gpu_info['devices'][self.device_id]['name']}",
            f"  Total Memory: {self.gpu_info['devices'][self.device_id]['total_memory_mb']:.0f} MB",
            f"  Compute Capability: {self.gpu_info['devices'][self.device_id]['compute_capability']}",
            ""
        ]
        
        # Summary statistics
        summary = analysis['summary']
        report.extend([
            "Benchmark Summary:",
            f"  Total Benchmarks: {summary['total_benchmarks']}",
            f"  Problem Sizes: {summary['problem_sizes_tested']}",
            f"  Solver Types: {summary['solver_types_tested']}",
            f"  Precision Levels: {summary['precision_levels_tested']}",
            ""
        ])
        
        if 'best_speedup' in summary:
            report.extend([
                "GPU Performance Summary:",
                f"  Best Speedup: {summary['best_speedup']:.1f}x",
                f"  Average Speedup: {summary['average_speedup']:.1f}x",
                f"  Worst Speedup: {summary['worst_speedup']:.1f}x",
                ""
            ])
        
        # Speedup analysis
        if 'speedup_analysis' in analysis:
            report.extend(["Speedup Analysis by Solver Type:", "-" * 35])
            for solver, stats in analysis['speedup_analysis'].items():
                report.extend([
                    f"{solver}:",
                    f"  Mean Speedup: {stats['mean_speedup']:.2f}x",
                    f"  Max Speedup: {stats['max_speedup']:.2f}x",
                    f"  Results ≥2x: {stats['speedup_above_2x']}",
                    f"  Results ≥5x: {stats['speedup_above_5x']}",
                    f"  Results ≥10x: {stats['speedup_above_10x']}",
                    ""
                ])
        
        # Scaling analysis
        if 'scaling_analysis' in analysis:
            report.extend(["Scaling Analysis:", "-" * 20])
            for solver, stats in analysis['scaling_analysis'].items():
                if 'error' not in stats:
                    report.extend([
                        f"{solver}:",
                        f"  Scaling: {stats['scaling_interpretation']} ({stats['scaling_exponent']:.2f})",
                        f"  Throughput Range: {stats['throughput_range'][0]:.0f} - {stats['throughput_range'][1]:.0f} points/sec",
                        ""
                    ])
        
        # Recommendations
        if analysis['recommendations']:
            report.extend(["Performance Recommendations:", "-" * 30])
            for rec in analysis['recommendations']:
                report.append(f"• {rec}")
        
        return "\n".join(report)
    
    def _get_memory_usage(self, use_gpu: bool) -> int:
        """Get current memory usage in bytes."""
        if use_gpu:
            try:
                with cp.cuda.Device(self.device_id):
                    mempool = cp.get_default_memory_pool()
                    return mempool.used_bytes()
            except Exception:
                return 0
        else:
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss
            except ImportError:
                return 0
    
    def _has_tensor_cores(self) -> bool:
        """Check if GPU has Tensor Core support."""
        if self.device_id < len(self.gpu_info['devices']):
            compute_cap = self.gpu_info['devices'][self.device_id]['compute_capability']
            major, minor = map(float, compute_cap.split('.'))
            return major >= 7.0  # Volta and newer
        return False
    
    def export_results(self, filename: str) -> None:
        """Export benchmark results to file."""
        import json
        
        export_data = {
            'gpu_info': self.gpu_info,
            'benchmark_results': [
                {
                    'problem_size': r.problem_size,
                    'solver_type': r.solver_type,
                    'solve_time': r.solve_time,
                    'iterations': r.iterations,
                    'final_residual': r.final_residual,
                    'memory_usage_mb': r.memory_usage_mb,
                    'throughput_points_per_sec': r.throughput_points_per_sec,
                    'speedup_vs_cpu': r.speedup_vs_cpu,
                    'gpu_utilization': r.gpu_utilization,
                    'precision_used': r.precision_used,
                    'additional_metrics': r.additional_metrics
                }
                for r in self.benchmark_results
            ],
            'analysis': self._analyze_benchmark_results()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Benchmark results exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export benchmark results: {e}")


def run_quick_gpu_benchmark(device_id: int = 0) -> Dict[str, Any]:
    """
    Run quick GPU benchmark for basic performance assessment.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Quick benchmark results
    """
    benchmark_suite = GPUBenchmarkSuite(device_id=device_id)
    
    # Quick benchmark with smaller problem sizes
    results = benchmark_suite.run_comprehensive_benchmark(
        problem_sizes=[(65, 65), (129, 129), (257, 257)],
        solver_types=['cpu_multigrid', 'gpu_multigrid'],
        precision_levels=['single'],
        num_runs=3
    )
    
    return results