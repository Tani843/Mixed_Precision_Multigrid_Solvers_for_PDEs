"""Performance analysis and benchmarking framework for CPU vs GPU comparison."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import platform
import json
import matplotlib.pyplot as plt

from .poisson_solver import PoissonSolver2D, PoissonProblem  
from .heat_solver import HeatSolver2D, HeatProblem, TimeSteppingConfig
from .test_problems import PoissonTestProblems, HeatTestProblems

# Try to import GPU monitoring
try:
    import pynvml
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Results from a single performance benchmark."""
    problem_name: str
    solver_type: str
    grid_size: Tuple[int, int]
    total_unknowns: int
    solve_time: float
    throughput: float  # unknowns per second
    iterations: int
    memory_usage: Dict[str, float]
    device_info: Dict[str, Any]
    additional_metrics: Dict[str, Any] = None


@dataclass 
class ComparisonResult:
    """Results from CPU vs GPU comparison."""
    problem_name: str
    grid_size: Tuple[int, int] 
    cpu_result: PerformanceResult
    gpu_result: Optional[PerformanceResult]
    speedup: float
    efficiency_ratio: float
    memory_comparison: Dict[str, Any]


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and benchmarking tool.
    
    Provides CPU vs GPU comparisons, throughput analysis, 
    and scalability studies.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.benchmark_results: List[PerformanceResult] = []
        self.comparison_results: List[ComparisonResult] = []
        self.system_info = self._get_system_info()
        
        logger.info("Performance analyzer initialized")
        logger.info(f"System: {self.system_info['cpu']['model']}")
        
        if GPU_MONITORING:
            self._init_gpu_monitoring()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        cpu_info = {
            'model': platform.processor(),
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'cache_size': 'Unknown'  # Would need platform-specific code
        }
        
        memory_info = {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
        }
        
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu': cpu_info,
            'memory': memory_info,
            'gpu': []
        }
        
        return system_info
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available."""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info = {
                    'id': i,
                    'name': name,
                    'memory_total': memory_info.total,
                    'memory_free': memory_info.free,
                    'memory_used': memory_info.used
                }
                
                self.system_info['gpu'].append(gpu_info)
                logger.info(f"Found GPU {i}: {name}")
                
        except Exception as e:
            logger.warning(f"GPU monitoring initialization failed: {e}")
    
    def benchmark_poisson_solver(
        self,
        solver_configs: List[Dict[str, Any]],
        problem_names: List[str],
        grid_sizes: List[Tuple[int, int]],
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> List[PerformanceResult]:
        """
        Comprehensive Poisson solver benchmarking.
        
        Args:
            solver_configs: List of solver configurations to test
            problem_names: List of test problem names
            grid_sizes: List of grid sizes to benchmark
            num_runs: Number of runs for averaging
            warmup_runs: Number of warmup runs to discard
            
        Returns:
            List of performance results
        """
        logger.info(f"Benchmarking Poisson solvers: {len(solver_configs)} configs, "
                   f"{len(problem_names)} problems, {len(grid_sizes)} grid sizes")
        
        test_problems = PoissonTestProblems()
        results = []
        
        for config in solver_configs:
            for problem_name in problem_names:
                for grid_size in grid_sizes:
                    logger.debug(f"Benchmarking {config['solver_type']} on {problem_name} "
                               f"with {grid_size[0]}x{grid_size[1]} grid")
                    
                    # Get test problem
                    problem = test_problems.get_problem(problem_name)
                    
                    # Create solver
                    solver = PoissonSolver2D(**config)
                    
                    # Run benchmark
                    result = self._run_poisson_benchmark(
                        solver, problem, grid_size, num_runs, warmup_runs
                    )
                    
                    results.append(result)
                    self.benchmark_results.append(result)
        
        logger.info(f"Poisson benchmarking completed: {len(results)} results")
        return results
    
    def benchmark_heat_solver(
        self,
        solver_configs: List[Dict[str, Any]],
        time_configs: List[TimeSteppingConfig],
        problem_names: List[str],
        grid_sizes: List[Tuple[int, int]],
        num_runs: int = 3,
        warmup_runs: int = 1
    ) -> List[PerformanceResult]:
        """
        Comprehensive Heat solver benchmarking.
        
        Args:
            solver_configs: List of solver configurations to test
            time_configs: List of time stepping configurations
            problem_names: List of test problem names
            grid_sizes: List of grid sizes to benchmark
            num_runs: Number of runs for averaging
            warmup_runs: Number of warmup runs to discard
            
        Returns:
            List of performance results
        """
        logger.info(f"Benchmarking Heat solvers: {len(solver_configs)} configs, "
                   f"{len(time_configs)} time configs, {len(problem_names)} problems")
        
        test_problems = HeatTestProblems()
        results = []
        
        for config in solver_configs:
            for time_config in time_configs:
                for problem_name in problem_names:
                    for grid_size in grid_sizes:
                        logger.debug(f"Benchmarking heat {config['solver_type']} on {problem_name}")
                        
                        # Get test problem
                        problem = test_problems.get_problem(problem_name)
                        
                        # Create solver
                        solver = HeatSolver2D(**config)
                        
                        # Run benchmark
                        result = self._run_heat_benchmark(
                            solver, problem, time_config, grid_size, num_runs, warmup_runs
                        )
                        
                        results.append(result)
                        self.benchmark_results.append(result)
        
        logger.info(f"Heat benchmarking completed: {len(results)} results")
        return results
    
    def _run_poisson_benchmark(
        self,
        solver: PoissonSolver2D,
        problem: PoissonProblem,
        grid_size: Tuple[int, int],
        num_runs: int,
        warmup_runs: int
    ) -> PerformanceResult:
        """Run single Poisson benchmark with multiple runs."""
        nx, ny = grid_size
        total_unknowns = nx * ny
        
        # Get device info
        device_info = {
            'device_type': 'GPU' if solver.use_gpu else 'CPU',
            'solver_type': solver.solver_type,
            'mixed_precision': solver.enable_mixed_precision
        }
        
        # Monitor memory before benchmark
        memory_before = self._get_memory_usage()
        
        # Warmup runs
        for _ in range(warmup_runs):
            solver.solve_poisson_problem(problem, nx, ny)
        
        # Benchmark runs
        solve_times = []
        iterations_list = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            result = solver.solve_poisson_problem(problem, nx, ny)
            end_time = time.perf_counter()
            
            solve_times.append(end_time - start_time)
            iterations_list.append(result['solver_info']['iterations'])
        
        # Monitor memory after benchmark
        memory_after = self._get_memory_usage()
        memory_usage = {
            'before': memory_before,
            'after': memory_after,
            'peak_increase': memory_after['system']['used'] - memory_before['system']['used']
        }
        
        # Calculate statistics
        avg_time = np.mean(solve_times)
        std_time = np.std(solve_times)
        throughput = total_unknowns / avg_time
        avg_iterations = np.mean(iterations_list)
        
        # Additional metrics
        additional_metrics = {
            'min_time': np.min(solve_times),
            'max_time': np.max(solve_times),
            'std_time': std_time,
            'coefficient_of_variation': std_time / avg_time if avg_time > 0 else 0,
            'std_iterations': np.std(iterations_list),
            'num_runs': num_runs,
            'warmup_runs': warmup_runs
        }
        
        return PerformanceResult(
            problem_name=problem.name,
            solver_type=solver.solver_type,
            grid_size=grid_size,
            total_unknowns=total_unknowns,
            solve_time=avg_time,
            throughput=throughput,
            iterations=int(avg_iterations),
            memory_usage=memory_usage,
            device_info=device_info,
            additional_metrics=additional_metrics
        )
    
    def _run_heat_benchmark(
        self,
        solver: HeatSolver2D,
        problem: HeatProblem,
        time_config: TimeSteppingConfig,
        grid_size: Tuple[int, int],
        num_runs: int,
        warmup_runs: int
    ) -> PerformanceResult:
        """Run single Heat benchmark with multiple runs."""
        nx, ny = grid_size
        total_unknowns = nx * ny
        
        # Get device info  
        device_info = {
            'device_type': 'GPU' if solver.use_gpu else 'CPU',
            'solver_type': solver.solver_type,
            'mixed_precision': solver.enable_mixed_precision,
            'time_stepping': time_config.method.value
        }
        
        # Monitor memory before benchmark
        memory_before = self._get_memory_usage()
        
        # Warmup runs
        for _ in range(warmup_runs):
            solver.solve_heat_problem(problem, nx, ny, time_config)
        
        # Benchmark runs
        solve_times = []
        total_steps_list = []
        mg_iterations_list = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            result = solver.solve_heat_problem(problem, nx, ny, time_config)
            end_time = time.perf_counter()
            
            solve_times.append(end_time - start_time)
            total_steps_list.append(result['total_steps'])
            mg_iterations_list.append(result['total_mg_iterations'])
        
        # Monitor memory after benchmark
        memory_after = self._get_memory_usage()
        memory_usage = {
            'before': memory_before,
            'after': memory_after,
            'peak_increase': memory_after['system']['used'] - memory_before['system']['used']
        }
        
        # Calculate statistics
        avg_time = np.mean(solve_times)
        std_time = np.std(solve_times)
        throughput = total_unknowns / avg_time  # Approximate throughput
        avg_steps = np.mean(total_steps_list)
        avg_mg_iterations = np.mean(mg_iterations_list)
        
        # Additional metrics for heat equation
        additional_metrics = {
            'min_time': np.min(solve_times),
            'max_time': np.max(solve_times),
            'std_time': std_time,
            'coefficient_of_variation': std_time / avg_time if avg_time > 0 else 0,
            'avg_time_steps': avg_steps,
            'avg_mg_iterations_total': avg_mg_iterations,
            'avg_mg_iterations_per_step': avg_mg_iterations / avg_steps if avg_steps > 0 else 0,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
            'time_stepping_method': time_config.method.value,
            'dt': time_config.dt,
            't_final': time_config.t_final
        }
        
        return PerformanceResult(
            problem_name=problem.name,
            solver_type=solver.solver_type,
            grid_size=grid_size,
            total_unknowns=total_unknowns,
            solve_time=avg_time,
            throughput=throughput,
            iterations=int(avg_mg_iterations),  # Total MG iterations
            memory_usage=memory_usage,
            device_info=device_info,
            additional_metrics=additional_metrics
        )
    
    def compare_cpu_gpu_performance(
        self,
        problem_names: List[str],
        grid_sizes: List[Tuple[int, int]],
        solver_type: str = "multigrid",
        num_runs: int = 5
    ) -> List[ComparisonResult]:
        """
        Direct CPU vs GPU performance comparison.
        
        Args:
            problem_names: List of problems to compare
            grid_sizes: List of grid sizes to test
            solver_type: Base solver type
            num_runs: Number of runs for averaging
            
        Returns:
            List of comparison results
        """
        logger.info("Running CPU vs GPU performance comparison")
        
        # CPU configuration
        cpu_config = {
            'solver_type': solver_type,
            'max_levels': 6,
            'tolerance': 1e-8,
            'use_gpu': False,
            'enable_mixed_precision': False
        }
        
        # GPU configuration  
        gpu_config = {
            'solver_type': 'gpu_multigrid',
            'max_levels': 6,
            'tolerance': 1e-8,
            'use_gpu': True,
            'enable_mixed_precision': True,
            'device_id': 0
        }
        
        test_problems = PoissonTestProblems()
        comparison_results = []
        
        for problem_name in problem_names:
            for grid_size in grid_sizes:
                logger.debug(f"Comparing CPU vs GPU for {problem_name} on {grid_size[0]}x{grid_size[1]} grid")
                
                problem = test_problems.get_problem(problem_name)
                
                # CPU benchmark
                cpu_solver = PoissonSolver2D(**cpu_config)
                cpu_result = self._run_poisson_benchmark(
                    cpu_solver, problem, grid_size, num_runs, warmup_runs=1
                )
                
                # GPU benchmark
                gpu_result = None
                try:
                    gpu_solver = PoissonSolver2D(**gpu_config)
                    gpu_result = self._run_poisson_benchmark(
                        gpu_solver, problem, grid_size, num_runs, warmup_runs=2
                    )
                except Exception as e:
                    logger.warning(f"GPU benchmark failed for {problem_name}: {e}")
                
                # Calculate comparison metrics
                speedup = 0.0
                efficiency_ratio = 0.0
                memory_comparison = {}
                
                if gpu_result:
                    speedup = cpu_result.solve_time / gpu_result.solve_time
                    efficiency_ratio = gpu_result.throughput / cpu_result.throughput
                    
                    memory_comparison = {
                        'cpu_peak': cpu_result.memory_usage['peak_increase'],
                        'gpu_peak': gpu_result.memory_usage['peak_increase'],
                        'memory_overhead': gpu_result.memory_usage['peak_increase'] / cpu_result.memory_usage['peak_increase'] if cpu_result.memory_usage['peak_increase'] > 0 else 1.0
                    }
                
                comparison_result = ComparisonResult(
                    problem_name=problem_name,
                    grid_size=grid_size,
                    cpu_result=cpu_result,
                    gpu_result=gpu_result,
                    speedup=speedup,
                    efficiency_ratio=efficiency_ratio,
                    memory_comparison=memory_comparison
                )
                
                comparison_results.append(comparison_result)
                self.comparison_results.append(comparison_result)
                
                logger.info(f"CPU vs GPU comparison completed for {problem_name}: "
                           f"Speedup = {speedup:.2f}x")
        
        return comparison_results
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        system_memory = psutil.virtual_memory()
        
        usage_info = {
            'system': {
                'total': system_memory.total,
                'available': system_memory.available, 
                'used': system_memory.used,
                'percentage': system_memory.percent
            },
            'gpu': []
        }
        
        # GPU memory if available
        if GPU_MONITORING:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    usage_info['gpu'].append({
                        'device_id': i,
                        'total': memory_info.total,
                        'free': memory_info.free,
                        'used': memory_info.used,
                        'percentage': 100 * memory_info.used / memory_info.total
                    })
            except Exception:
                pass
        
        return usage_info
    
    def analyze_scaling_performance(
        self,
        problem_name: str,
        grid_sizes: List[Tuple[int, int]],
        solver_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze performance scaling across grid sizes and configurations.
        
        Args:
            problem_name: Problem to analyze
            grid_sizes: List of grid sizes in ascending order
            solver_configs: List of solver configurations
            
        Returns:
            Scaling analysis results
        """
        logger.info(f"Analyzing scaling performance for {problem_name}")
        
        test_problems = PoissonTestProblems()
        problem = test_problems.get_problem(problem_name)
        
        scaling_results = {}
        
        for config in solver_configs:
            config_name = f"{config['solver_type']}_GPU" if config.get('use_gpu', False) else f"{config['solver_type']}_CPU"
            
            solver = PoissonSolver2D(**config)
            results = []
            
            for grid_size in grid_sizes:
                result = self._run_poisson_benchmark(
                    solver, problem, grid_size, num_runs=3, warmup_runs=1
                )
                results.append(result)
            
            # Analyze scaling
            scaling_analysis = self._analyze_scaling_results(results)
            scaling_results[config_name] = scaling_analysis
        
        return {
            'problem_name': problem_name,
            'grid_sizes': grid_sizes,
            'scaling_results': scaling_results,
            'analysis_summary': self._summarize_scaling_analysis(scaling_results)
        }
    
    def _analyze_scaling_results(self, results: List[PerformanceResult]) -> Dict[str, Any]:
        """Analyze scaling behavior from performance results."""
        if len(results) < 2:
            return {}
        
        grid_points = [r.total_unknowns for r in results]
        solve_times = [r.solve_time for r in results]
        throughputs = [r.throughput for r in results]
        
        # Fit scaling models
        log_points = np.log(grid_points)
        log_times = np.log(solve_times)
        
        # Time complexity: T = C * N^α
        time_coeffs = np.polyfit(log_points, log_times, 1)
        time_scaling_exponent = time_coeffs[0]
        
        # Throughput scaling
        throughput_ratios = []
        scaling_efficiencies = []
        
        for i in range(1, len(results)):
            size_ratio = grid_points[i] / grid_points[0]
            throughput_ratio = throughputs[i] / throughputs[0]
            
            # Ideal throughput should remain constant
            # Efficiency = actual_throughput / ideal_throughput
            efficiency = throughput_ratio  
            
            throughput_ratios.append(throughput_ratio)
            scaling_efficiencies.append(efficiency)
        
        return {
            'time_scaling_exponent': time_scaling_exponent,
            'throughput_ratios': throughput_ratios,
            'scaling_efficiencies': scaling_efficiencies,
            'parallel_efficiency': np.mean(scaling_efficiencies) if scaling_efficiencies else 0.0,
            'grid_points': grid_points,
            'solve_times': solve_times,
            'throughputs': throughputs
        }
    
    def _summarize_scaling_analysis(self, scaling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize scaling analysis across configurations."""
        if not scaling_results:
            return {}
        
        # Find best and worst scaling configurations
        config_efficiencies = {}
        config_exponents = {}
        
        for config_name, analysis in scaling_results.items():
            if 'parallel_efficiency' in analysis:
                config_efficiencies[config_name] = analysis['parallel_efficiency']
            if 'time_scaling_exponent' in analysis:
                config_exponents[config_name] = analysis['time_scaling_exponent']
        
        summary = {}
        
        if config_efficiencies:
            best_config = max(config_efficiencies.items(), key=lambda x: x[1])
            worst_config = min(config_efficiencies.items(), key=lambda x: x[1])
            
            summary['efficiency_ranking'] = {
                'best': best_config,
                'worst': worst_config,
                'all_efficiencies': config_efficiencies
            }
        
        if config_exponents:
            # Lower exponent is better (closer to linear O(N))
            best_scaling = min(config_exponents.items(), key=lambda x: x[1])
            worst_scaling = max(config_exponents.items(), key=lambda x: x[1])
            
            summary['scaling_ranking'] = {
                'best': best_scaling,
                'worst': worst_scaling,
                'all_exponents': config_exponents
            }
        
        return summary
    
    def generate_performance_report(
        self,
        include_comparisons: bool = True,
        save_to_file: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive performance analysis report.
        
        Args:
            include_comparisons: Include CPU vs GPU comparisons
            save_to_file: Optional filename to save report
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "PERFORMANCE ANALYSIS REPORT",
            "=" * 50,
            "",
            f"System Information:",
            f"Platform: {self.system_info['platform']} {self.system_info['platform_version']}",
            f"CPU: {self.system_info['cpu']['model']}",
            f"CPU Cores: {self.system_info['cpu']['cores']} physical, {self.system_info['cpu']['logical_cores']} logical",
            f"Memory: {self.system_info['memory']['total'] / (1024**3):.1f} GB total",
            ""
        ]
        
        # GPU information
        if self.system_info['gpu']:
            report_lines.append("GPU Information:")
            for gpu in self.system_info['gpu']:
                report_lines.append(f"  GPU {gpu['id']}: {gpu['name']}")
                report_lines.append(f"    Memory: {gpu['memory_total'] / (1024**3):.1f} GB")
            report_lines.append("")
        
        # Benchmark results summary
        if self.benchmark_results:
            report_lines.extend([
                "BENCHMARK RESULTS SUMMARY:",
                "-" * 30,
                f"Total benchmarks: {len(self.benchmark_results)}"
            ])
            
            # Group by solver type
            solver_groups = {}
            for result in self.benchmark_results:
                key = f"{result.solver_type} ({'GPU' if result.device_info.get('device_type') == 'GPU' else 'CPU'})"
                if key not in solver_groups:
                    solver_groups[key] = []
                solver_groups[key].append(result)
            
            for solver_name, results in solver_groups.items():
                avg_throughput = np.mean([r.throughput for r in results])
                avg_time = np.mean([r.solve_time for r in results])
                
                report_lines.extend([
                    f"\n{solver_name}:",
                    f"  Tests: {len(results)}",
                    f"  Avg throughput: {avg_throughput:.2e} unknowns/sec",
                    f"  Avg solve time: {avg_time:.4f} seconds"
                ])
            
            report_lines.append("")
        
        # CPU vs GPU comparisons
        if include_comparisons and self.comparison_results:
            report_lines.extend([
                "CPU vs GPU COMPARISON:",
                "-" * 25
            ])
            
            speedups = [c.speedup for c in self.comparison_results if c.gpu_result is not None]
            if speedups:
                report_lines.extend([
                    f"Successful GPU comparisons: {len(speedups)}",
                    f"Average speedup: {np.mean(speedups):.2f}x",
                    f"Max speedup: {np.max(speedups):.2f}x",
                    f"Min speedup: {np.min(speedups):.2f}x",
                    ""
                ])
                
                # Best and worst cases
                best_comparison = max(self.comparison_results, key=lambda x: x.speedup if x.gpu_result else 0)
                worst_comparison = min(self.comparison_results, key=lambda x: x.speedup if x.gpu_result else float('inf'))
                
                if best_comparison.gpu_result:
                    report_lines.extend([
                        f"Best case: {best_comparison.problem_name} ({best_comparison.grid_size[0]}x{best_comparison.grid_size[1]})",
                        f"  Speedup: {best_comparison.speedup:.2f}x",
                        f"  CPU time: {best_comparison.cpu_result.solve_time:.4f}s",
                        f"  GPU time: {best_comparison.gpu_result.solve_time:.4f}s",
                        ""
                    ])
                
                if worst_comparison.gpu_result and worst_comparison.speedup < float('inf'):
                    report_lines.extend([
                        f"Worst case: {worst_comparison.problem_name} ({worst_comparison.grid_size[0]}x{worst_comparison.grid_size[1]})",
                        f"  Speedup: {worst_comparison.speedup:.2f}x",
                        f"  CPU time: {worst_comparison.cpu_result.solve_time:.4f}s", 
                        f"  GPU time: {worst_comparison.gpu_result.solve_time:.4f}s",
                        ""
                    ])
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write(report)
                logger.info(f"Performance report saved to {save_to_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def export_benchmark_data(
        self,
        filename: str,
        format: str = 'json',
        include_system_info: bool = True
    ) -> None:
        """
        Export benchmark data to file.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'csv')
            include_system_info: Include system information
        """
        try:
            export_data = {
                'benchmark_results': [],
                'comparison_results': []
            }
            
            if include_system_info:
                export_data['system_info'] = self.system_info
            
            # Convert results to serializable format
            for result in self.benchmark_results:
                export_data['benchmark_results'].append({
                    'problem_name': result.problem_name,
                    'solver_type': result.solver_type,
                    'grid_size': result.grid_size,
                    'total_unknowns': result.total_unknowns,
                    'solve_time': result.solve_time,
                    'throughput': result.throughput,
                    'iterations': result.iterations,
                    'device_info': result.device_info,
                    'additional_metrics': result.additional_metrics or {}
                })
            
            for comp in self.comparison_results:
                export_data['comparison_results'].append({
                    'problem_name': comp.problem_name,
                    'grid_size': comp.grid_size,
                    'speedup': comp.speedup,
                    'efficiency_ratio': comp.efficiency_ratio,
                    'cpu_time': comp.cpu_result.solve_time,
                    'gpu_time': comp.gpu_result.solve_time if comp.gpu_result else None,
                    'memory_comparison': comp.memory_comparison
                })
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import pandas as pd
                
                # Export benchmark results to CSV
                if export_data['benchmark_results']:
                    df_benchmarks = pd.DataFrame(export_data['benchmark_results'])
                    benchmark_file = filename.replace('.csv', '_benchmarks.csv')
                    df_benchmarks.to_csv(benchmark_file, index=False)
                
                # Export comparison results to CSV
                if export_data['comparison_results']:
                    df_comparisons = pd.DataFrame(export_data['comparison_results'])
                    comparison_file = filename.replace('.csv', '_comparisons.csv')
                    df_comparisons.to_csv(comparison_file, index=False)
            
            logger.info(f"Benchmark data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export benchmark data: {e}")


class ScalabilityAnalyzer:
    """
    Specialized analyzer for weak and strong scaling studies.
    """
    
    def __init__(self, performance_analyzer: PerformanceAnalyzer):
        """Initialize scalability analyzer."""
        self.perf_analyzer = performance_analyzer
        self.scaling_studies: List[Dict[str, Any]] = []
        
        logger.info("Scalability analyzer initialized")
    
    def run_strong_scaling_study(
        self,
        problem_name: str,
        grid_sizes: List[Tuple[int, int]],
        solver_configs: List[Dict[str, Any]],
        baseline_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run strong scaling study (fixed problem size, varying resources).
        
        Args:
            problem_name: Problem to study
            grid_sizes: Fixed problem sizes to test
            solver_configs: Different solver configurations (representing resource variations)
            baseline_config: Baseline configuration for comparison
            
        Returns:
            Strong scaling results
        """
        logger.info(f"Running strong scaling study for {problem_name}")
        
        if baseline_config is None:
            baseline_config = solver_configs[0]
        
        scaling_results = {}
        
        for grid_size in grid_sizes:
            size_results = []
            
            for config in solver_configs:
                # Run benchmark for this configuration
                results = self.perf_analyzer.benchmark_poisson_solver(
                    [config], [problem_name], [grid_size], num_runs=3, warmup_runs=1
                )
                size_results.extend(results)
            
            scaling_results[f"{grid_size[0]}x{grid_size[1]}"] = size_results
        
        # Analyze strong scaling efficiency
        analysis = self._analyze_strong_scaling(scaling_results, solver_configs)
        
        study_result = {
            'study_type': 'strong_scaling',
            'problem_name': problem_name,
            'grid_sizes': grid_sizes,
            'solver_configs': solver_configs,
            'results': scaling_results,
            'analysis': analysis
        }
        
        self.scaling_studies.append(study_result)
        return study_result
    
    def run_weak_scaling_study(
        self,
        base_problem_size: Tuple[int, int],
        scaling_factors: List[int],
        solver_config: Dict[str, Any],
        problem_name: str = 'trigonometric'
    ) -> Dict[str, Any]:
        """
        Run weak scaling study (problem size scales with resources).
        
        Args:
            base_problem_size: Base problem size (nx, ny)
            scaling_factors: List of scaling factors to apply
            solver_config: Solver configuration
            problem_name: Problem to study
            
        Returns:
            Weak scaling results
        """
        logger.info(f"Running weak scaling study for {problem_name}")
        
        base_nx, base_ny = base_problem_size
        grid_sizes = []
        
        # Generate scaled problem sizes
        for factor in scaling_factors:
            # Scale in both dimensions approximately by sqrt(factor)
            scale_factor = int(np.sqrt(factor))
            scaled_nx = base_nx * scale_factor
            scaled_ny = base_ny * scale_factor
            grid_sizes.append((scaled_nx, scaled_ny))
        
        # Run benchmarks for all problem sizes
        results = self.perf_analyzer.benchmark_poisson_solver(
            [solver_config], [problem_name], grid_sizes, num_runs=3, warmup_runs=1
        )
        
        # Analyze weak scaling efficiency
        analysis = self._analyze_weak_scaling(results, scaling_factors)
        
        study_result = {
            'study_type': 'weak_scaling',
            'problem_name': problem_name,
            'base_problem_size': base_problem_size,
            'scaling_factors': scaling_factors,
            'grid_sizes': grid_sizes,
            'solver_config': solver_config,
            'results': results,
            'analysis': analysis
        }
        
        self.scaling_studies.append(study_result)
        return study_result
    
    def _analyze_strong_scaling(
        self,
        scaling_results: Dict[str, List[PerformanceResult]], 
        solver_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze strong scaling efficiency."""
        analysis = {
            'scaling_efficiency': {},
            'speedup_factors': {},
            'parallel_efficiency': {}
        }
        
        for grid_size_str, results in scaling_results.items():
            if len(results) < 2:
                continue
            
            # Assume first result is baseline (single core/basic config)
            baseline_time = results[0].solve_time
            
            efficiencies = []
            speedups = []
            
            for i, result in enumerate(results):
                speedup = baseline_time / result.solve_time
                # Theoretical speedup for i+1 processors is i+1
                theoretical_speedup = i + 1
                efficiency = speedup / theoretical_speedup if theoretical_speedup > 0 else 0
                
                speedups.append(speedup)
                efficiencies.append(efficiency)
            
            analysis['scaling_efficiency'][grid_size_str] = efficiencies
            analysis['speedup_factors'][grid_size_str] = speedups
            analysis['parallel_efficiency'][grid_size_str] = np.mean(efficiencies[1:]) if len(efficiencies) > 1 else 0
        
        return analysis
    
    def _analyze_weak_scaling(
        self,
        results: List[PerformanceResult],
        scaling_factors: List[int]
    ) -> Dict[str, Any]:
        """Analyze weak scaling efficiency."""
        if len(results) != len(scaling_factors):
            logger.warning("Mismatch between results and scaling factors")
            return {}
        
        # Ideal weak scaling: solve time should remain constant
        baseline_time = results[0].solve_time
        
        efficiency_factors = []
        time_ratios = []
        throughput_scaling = []
        
        for i, (result, scale_factor) in enumerate(zip(results, scaling_factors)):
            # Time efficiency (should be close to 1.0 for perfect weak scaling)
            time_efficiency = baseline_time / result.solve_time
            
            # Throughput scaling (should scale linearly with problem size)
            expected_throughput = results[0].throughput * scale_factor
            throughput_efficiency = result.throughput / expected_throughput if expected_throughput > 0 else 0
            
            efficiency_factors.append(time_efficiency)
            time_ratios.append(result.solve_time / baseline_time)
            throughput_scaling.append(throughput_efficiency)
        
        return {
            'time_efficiency': efficiency_factors,
            'time_ratios': time_ratios,
            'throughput_scaling': throughput_scaling,
            'overall_efficiency': np.mean(efficiency_factors),
            'throughput_efficiency': np.mean(throughput_scaling)
        }


def run_comprehensive_performance_analysis() -> Dict[str, Any]:
    """
    Run comprehensive performance analysis including CPU vs GPU comparisons.
    
    Returns:
        Complete performance analysis results
    """
    logger.info("Starting comprehensive performance analysis")
    
    # Initialize analyzers
    perf_analyzer = PerformanceAnalyzer()
    scalability_analyzer = ScalabilityAnalyzer(perf_analyzer)
    
    # Test configurations
    cpu_config = {
        'solver_type': 'multigrid',
        'max_levels': 6,
        'tolerance': 1e-8,
        'use_gpu': False,
        'enable_mixed_precision': False
    }
    
    gpu_config = {
        'solver_type': 'gpu_multigrid',
        'max_levels': 6,
        'tolerance': 1e-8,
        'use_gpu': True,
        'enable_mixed_precision': True
    }
    
    configs = [cpu_config]
    
    # Add GPU config if available
    try:
        from ..gpu.gpu_solver import GPUMultigridSolver
        configs.append(gpu_config)
        logger.info("GPU configuration added to analysis")
    except ImportError:
        logger.info("GPU not available, running CPU-only analysis")
    
    # Test problems and grid sizes
    test_problems = ['trigonometric', 'polynomial', 'high_frequency']
    grid_sizes = [(33, 33), (65, 65), (129, 129)]
    
    results = {
        'system_info': perf_analyzer.system_info,
        'benchmark_results': [],
        'comparison_results': [],
        'scaling_studies': [],
        'analysis_summary': {}
    }
    
    # Run benchmarks
    try:
        benchmark_results = perf_analyzer.benchmark_poisson_solver(
            configs, test_problems, grid_sizes, num_runs=3, warmup_runs=1
        )
        results['benchmark_results'] = benchmark_results
        logger.info(f"Completed {len(benchmark_results)} benchmarks")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    
    # CPU vs GPU comparison if GPU available
    if len(configs) > 1:
        try:
            comparison_results = perf_analyzer.compare_cpu_gpu_performance(
                test_problems, grid_sizes[:2], num_runs=3  # Smaller subset for comparison
            )
            results['comparison_results'] = comparison_results
            logger.info(f"Completed {len(comparison_results)} CPU vs GPU comparisons")
        except Exception as e:
            logger.error(f"CPU vs GPU comparison failed: {e}")
    
    # Scaling studies
    try:
        # Strong scaling study (if multiple configs available)
        if len(configs) > 1:
            strong_scaling = scalability_analyzer.run_strong_scaling_study(
                'trigonometric', [(65, 65)], configs
            )
            results['scaling_studies'].append(strong_scaling)
        
        # Weak scaling study
        weak_scaling = scalability_analyzer.run_weak_scaling_study(
            (17, 17), [1, 4, 9], cpu_config  # 1x, 2x2, 3x3 scaling
        )
        results['scaling_studies'].append(weak_scaling)
        
        logger.info("Completed scaling studies")
    except Exception as e:
        logger.error(f"Scaling studies failed: {e}")
    
    # Generate summary
    results['analysis_summary'] = _generate_analysis_summary(results)
    results['performance_analyzer'] = perf_analyzer
    results['scalability_analyzer'] = scalability_analyzer
    
    logger.info("Comprehensive performance analysis completed")
    
    return results


def _generate_analysis_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of performance analysis."""
    summary = {
        'total_benchmarks': len(results.get('benchmark_results', [])),
        'total_comparisons': len(results.get('comparison_results', [])),
        'total_scaling_studies': len(results.get('scaling_studies', []))
    }
    
    # Benchmark summary
    if results['benchmark_results']:
        cpu_results = [r for r in results['benchmark_results'] if r.device_info.get('device_type') != 'GPU']
        gpu_results = [r for r in results['benchmark_results'] if r.device_info.get('device_type') == 'GPU']
        
        summary['cpu_benchmarks'] = len(cpu_results)
        summary['gpu_benchmarks'] = len(gpu_results)
        
        if cpu_results:
            summary['cpu_avg_throughput'] = np.mean([r.throughput for r in cpu_results])
        if gpu_results:
            summary['gpu_avg_throughput'] = np.mean([r.throughput for r in gpu_results])
    
    # Comparison summary
    if results['comparison_results']:
        speedups = [c.speedup for c in results['comparison_results'] if c.gpu_result]
        if speedups:
            summary['avg_gpu_speedup'] = np.mean(speedups)
            summary['max_gpu_speedup'] = np.max(speedups)
    
    return summary