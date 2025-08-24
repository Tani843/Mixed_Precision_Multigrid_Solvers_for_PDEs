"""Performance profiling tools for multigrid solvers."""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperationProfile:
    """Profile data for a single operation."""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    total_flops: int = 0
    memory_allocated: int = 0
    cache_misses: int = 0
    
    @property
    def average_time(self) -> float:
        """Average execution time."""
        return self.total_time / max(1, self.call_count)
    
    @property
    def flops_per_second(self) -> float:
        """Floating point operations per second."""
        return self.total_flops / max(1e-9, self.total_time)
    
    @property
    def memory_bandwidth_gb_s(self) -> float:
        """Memory bandwidth in GB/s."""
        return (self.memory_allocated / (1024**3)) / max(1e-9, self.total_time)


class PerformanceProfiler:
    """
    Advanced performance profiler for multigrid operations.
    
    Tracks timing, memory usage, FLOP counts, and cache performance.
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        """
        Initialize performance profiler.
        
        Args:
            enable_detailed_profiling: Enable detailed profiling metrics
        """
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profiles: Dict[str, OperationProfile] = {}
        self.call_stack: List[str] = []
        self.start_times: Dict[str, float] = {}
        
        # Global statistics
        self.global_stats = {
            'total_operations': 0,
            'total_time': 0.0,
            'peak_memory': 0,
            'total_flops': 0
        }
        
        # Hardware performance counters (if available)
        self.hardware_counters = self._init_hardware_counters()
    
    def _init_hardware_counters(self) -> Dict[str, Any]:
        """Initialize hardware performance counters."""
        counters = {'available': False}
        
        try:
            # Try to initialize perf counters (Linux)
            import subprocess
            result = subprocess.run(['perf', '--version'], 
                                  capture_output=True, timeout=1)
            if result.returncode == 0:
                counters['perf_available'] = True
                logger.debug("Hardware performance counters available via perf")
        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return counters
    
    @contextmanager
    def profile_operation(
        self,
        operation_name: str,
        flop_count: Optional[int] = None,
        memory_bytes: Optional[int] = None
    ):
        """
        Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation
            flop_count: Number of floating point operations
            memory_bytes: Memory bytes accessed
        """
        # Initialize profile if needed
        if operation_name not in self.profiles:
            self.profiles[operation_name] = OperationProfile(operation_name)
        
        profile = self.profiles[operation_name]
        
        # Start timing
        start_time = self._get_high_precision_time()
        start_memory = self._get_memory_usage()
        
        self.call_stack.append(operation_name)
        
        try:
            yield profile
        finally:
            # End timing
            end_time = self._get_high_precision_time()
            end_memory = self._get_memory_usage()
            
            elapsed_time = end_time - start_time
            memory_used = max(0, end_memory - start_memory)
            
            # Update profile
            profile.call_count += 1
            profile.total_time += elapsed_time
            profile.min_time = min(profile.min_time, elapsed_time)
            profile.max_time = max(profile.max_time, elapsed_time)
            
            if flop_count:
                profile.total_flops += flop_count
            
            if memory_bytes:
                profile.memory_allocated += memory_bytes
            elif memory_used > 0:
                profile.memory_allocated += memory_used
            
            # Update global stats
            self.global_stats['total_operations'] += 1
            self.global_stats['total_time'] += elapsed_time
            self.global_stats['peak_memory'] = max(
                self.global_stats['peak_memory'], end_memory
            )
            
            if flop_count:
                self.global_stats['total_flops'] += flop_count
            
            self.call_stack.pop()
    
    def profile_function(self, operation_name: str, flop_count: Optional[int] = None):
        """Decorator for profiling functions."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                with self.profile_operation(operation_name, flop_count):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def record_cache_miss(self, operation_name: str, miss_count: int = 1) -> None:
        """
        Record cache miss events.
        
        Args:
            operation_name: Operation name
            miss_count: Number of cache misses
        """
        if operation_name in self.profiles:
            self.profiles[operation_name].cache_misses += miss_count
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get comprehensive profiling summary."""
        if not self.profiles:
            return {'error': 'No profiling data available'}
        
        # Sort profiles by total time
        sorted_profiles = sorted(
            self.profiles.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        # Calculate percentages
        total_time = self.global_stats['total_time']
        
        summary = {
            'global_stats': self.global_stats.copy(),
            'operation_profiles': [],
            'top_operations_by_time': [],
            'performance_metrics': {}
        }
        
        for name, profile in sorted_profiles:
            time_percentage = (profile.total_time / max(total_time, 1e-9)) * 100
            
            profile_data = {
                'name': name,
                'call_count': profile.call_count,
                'total_time': profile.total_time,
                'average_time': profile.average_time,
                'time_percentage': time_percentage,
                'flops_per_second': profile.flops_per_second,
                'memory_bandwidth_gb_s': profile.memory_bandwidth_gb_s
            }
            
            summary['operation_profiles'].append(profile_data)
            
            if len(summary['top_operations_by_time']) < 10:
                summary['top_operations_by_time'].append({
                    'name': name,
                    'time': profile.total_time,
                    'percentage': time_percentage
                })
        
        # Calculate performance metrics
        if total_time > 0:
            summary['performance_metrics'] = {
                'operations_per_second': len(self.profiles) / total_time,
                'average_flops_per_second': self.global_stats['total_flops'] / total_time,
                'total_memory_gb': self.global_stats['peak_memory'] / (1024**3)
            }
        
        return summary
    
    def generate_report(self, top_n: int = 15) -> str:
        """
        Generate human-readable performance report.
        
        Args:
            top_n: Number of top operations to include
            
        Returns:
            Formatted performance report
        """
        summary = self.get_profile_summary()
        
        if 'error' in summary:
            return summary['error']
        
        report = [
            "Performance Profiling Report",
            "=" * 50,
            ""
        ]
        
        # Global statistics
        global_stats = summary['global_stats']
        report.extend([
            "Global Statistics:",
            f"  Total Operations: {global_stats['total_operations']:,}",
            f"  Total Time: {global_stats['total_time']:.3f} s",
            f"  Peak Memory: {global_stats['peak_memory'] / (1024**2):.2f} MB",
            f"  Total FLOPs: {global_stats['total_flops']:,}",
            ""
        ])
        
        # Performance metrics
        if 'performance_metrics' in summary:
            metrics = summary['performance_metrics']
            report.extend([
                "Performance Metrics:",
                f"  Operations/sec: {metrics['operations_per_second']:.2f}",
                f"  FLOPS: {metrics['average_flops_per_second']:.2e}",
                f"  Memory: {metrics['total_memory_gb']:.3f} GB",
                ""
            ])
        
        # Top operations
        report.extend([
            f"Top {top_n} Operations by Time:",
            "-" * 40
        ])
        
        for i, op in enumerate(summary['operation_profiles'][:top_n]):
            report.append(
                f"{i+1:2d}. {op['name']:25s} "
                f"{op['total_time']:8.3f}s ({op['time_percentage']:5.1f}%) "
                f"[{op['call_count']:6d} calls, {op['average_time']*1000:6.2f}ms avg]"
            )
        
        # Performance bottlenecks
        bottlenecks = self._identify_bottlenecks(summary)
        if bottlenecks:
            report.extend(["", "Performance Bottlenecks:", "-" * 25])
            report.extend(bottlenecks)
        
        return "\n".join(report)
    
    def _identify_bottlenecks(self, summary: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if not summary['operation_profiles']:
            return bottlenecks
        
        # Check for operations taking >20% of time
        for op in summary['operation_profiles']:
            if op['time_percentage'] > 20:
                bottlenecks.append(
                    f"• {op['name']} consumes {op['time_percentage']:.1f}% of execution time"
                )
        
        # Check for low FLOPS operations
        for op in summary['operation_profiles'][:5]:
            if op['flops_per_second'] > 0 and op['flops_per_second'] < 1e6:  # < 1 MFLOPS
                bottlenecks.append(
                    f"• {op['name']} has low computational intensity: "
                    f"{op['flops_per_second']:.2e} FLOPS"
                )
        
        # Check for high-frequency, low-impact operations
        for op in summary['operation_profiles']:
            if op['call_count'] > 1000 and op['average_time'] < 1e-6:
                bottlenecks.append(
                    f"• {op['name']} called very frequently ({op['call_count']} times) "
                    f"- consider batching"
                )
        
        return bottlenecks
    
    def save_profile_data(self, filename: str) -> None:
        """Save profile data to file."""
        import json
        
        summary = self.get_profile_summary()
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Profile data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save profile data: {e}")
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.profiles.clear()
        self.call_stack.clear()
        self.start_times.clear()
        
        self.global_stats = {
            'total_operations': 0,
            'total_time': 0.0,
            'peak_memory': 0,
            'total_flops': 0
        }
        
        logger.debug("Performance profiler reset")
    
    def _get_high_precision_time(self) -> float:
        """Get high-precision timestamp."""
        return time.perf_counter()
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0


class MultigridProfiler(PerformanceProfiler):
    """
    Specialized profiler for multigrid operations.
    
    Provides multigrid-specific profiling and analysis.
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        """Initialize multigrid profiler."""
        super().__init__(enable_detailed_profiling)
        
        # Multigrid-specific tracking
        self.level_stats: Dict[int, Dict[str, Any]] = {}
        self.cycle_stats: Dict[str, List[float]] = {
            'V': [], 'W': [], 'F': []
        }
        self.convergence_data: List[Tuple[int, float, float]] = []  # (iter, residual, time)
    
    @contextmanager
    def profile_mg_level(self, level: int, operation: str):
        """Profile multigrid operation at specific level."""
        level_op_name = f"level_{level}_{operation}"
        
        with self.profile_operation(level_op_name) as profile:
            # Initialize level stats if needed
            if level not in self.level_stats:
                self.level_stats[level] = {
                    'smoothing_time': 0.0,
                    'restriction_time': 0.0,
                    'prolongation_time': 0.0,
                    'residual_time': 0.0,
                    'total_time': 0.0
                }
            
            start_time = time.perf_counter()
            yield profile
            elapsed_time = time.perf_counter() - start_time
            
            # Update level stats
            self.level_stats[level][f'{operation}_time'] += elapsed_time
            self.level_stats[level]['total_time'] += elapsed_time
    
    def record_mg_cycle(self, cycle_type: str, cycle_time: float) -> None:
        """
        Record multigrid cycle timing.
        
        Args:
            cycle_type: Type of cycle ('V', 'W', 'F')
            cycle_time: Time taken for the cycle
        """
        if cycle_type in self.cycle_stats:
            self.cycle_stats[cycle_type].append(cycle_time)
    
    def record_convergence_data(
        self,
        iteration: int,
        residual_norm: float,
        iteration_time: float
    ) -> None:
        """
        Record convergence iteration data.
        
        Args:
            iteration: Iteration number
            residual_norm: Residual norm
            iteration_time: Time for this iteration
        """
        self.convergence_data.append((iteration, residual_norm, iteration_time))
    
    def get_multigrid_analysis(self) -> Dict[str, Any]:
        """Get multigrid-specific analysis."""
        analysis = {
            'level_breakdown': {},
            'cycle_performance': {},
            'convergence_analysis': {}
        }
        
        # Level breakdown
        total_time_all_levels = sum(
            stats['total_time'] for stats in self.level_stats.values()
        )
        
        for level, stats in self.level_stats.items():
            level_percentage = (stats['total_time'] / max(total_time_all_levels, 1e-9)) * 100
            
            analysis['level_breakdown'][level] = {
                **stats,
                'level_percentage': level_percentage,
                'operations': {
                    op: (time_val / max(stats['total_time'], 1e-9)) * 100
                    for op, time_val in stats.items()
                    if op.endswith('_time') and op != 'total_time'
                }
            }
        
        # Cycle performance
        for cycle_type, times in self.cycle_stats.items():
            if times:
                analysis['cycle_performance'][cycle_type] = {
                    'count': len(times),
                    'average_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'std_time': np.std(times)
                }
        
        # Convergence analysis
        if self.convergence_data:
            iterations = [data[0] for data in self.convergence_data]
            residuals = [data[1] for data in self.convergence_data]
            times = [data[2] for data in self.convergence_data]
            
            # Calculate convergence rate
            convergence_rates = []
            for i in range(1, len(residuals)):
                if residuals[i-1] > 0:
                    rate = residuals[i] / residuals[i-1]
                    if 0 < rate < 1:
                        convergence_rates.append(rate)
            
            analysis['convergence_analysis'] = {
                'total_iterations': len(iterations),
                'final_residual': residuals[-1] if residuals else 0,
                'average_convergence_rate': np.mean(convergence_rates) if convergence_rates else 1.0,
                'average_iteration_time': np.mean(times) if times else 0,
                'total_solve_time': np.sum(times) if times else 0,
                'work_units': len(iterations) * sum(2**i for i in range(len(self.level_stats)))  # Rough estimate
            }
        
        return analysis
    
    def generate_multigrid_report(self) -> str:
        """Generate multigrid-specific performance report."""
        base_report = self.generate_report()
        mg_analysis = self.get_multigrid_analysis()
        
        mg_report = [
            "",
            "Multigrid-Specific Analysis",
            "=" * 50,
            ""
        ]
        
        # Level breakdown
        if mg_analysis['level_breakdown']:
            mg_report.extend([
                "Performance by Grid Level:",
                "-" * 30
            ])
            
            for level in sorted(mg_analysis['level_breakdown'].keys()):
                stats = mg_analysis['level_breakdown'][level]
                mg_report.append(
                    f"Level {level:2d}: {stats['total_time']:8.3f}s "
                    f"({stats['level_percentage']:5.1f}%)"
                )
                
                # Operation breakdown for this level
                for op, percentage in stats['operations'].items():
                    if percentage > 5:  # Only show significant operations
                        op_name = op.replace('_time', '')
                        mg_report.append(f"  {op_name:12s}: {percentage:5.1f}%")
            
            mg_report.append("")
        
        # Cycle performance
        if any(mg_analysis['cycle_performance'].values()):
            mg_report.extend([
                "Cycle Performance:",
                "-" * 20
            ])
            
            for cycle_type, stats in mg_analysis['cycle_performance'].items():
                if stats['count'] > 0:
                    mg_report.append(
                        f"{cycle_type}-cycle: {stats['count']:3d} cycles, "
                        f"avg {stats['average_time']:.3f}s, "
                        f"std {stats['std_time']:.3f}s"
                    )
            
            mg_report.append("")
        
        # Convergence analysis
        if mg_analysis['convergence_analysis']:
            conv = mg_analysis['convergence_analysis']
            mg_report.extend([
                "Convergence Analysis:",
                "-" * 22,
                f"Total iterations: {conv['total_iterations']}",
                f"Final residual: {conv['final_residual']:.2e}",
                f"Avg convergence rate: {conv['average_convergence_rate']:.4f}",
                f"Avg iteration time: {conv['average_iteration_time']:.3f}s",
                f"Total solve time: {conv['total_solve_time']:.3f}s",
                f"Work units: {conv['work_units']}"
            ])
        
        return base_report + "\n".join(mg_report)