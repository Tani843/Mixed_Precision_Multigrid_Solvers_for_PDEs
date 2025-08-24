#!/usr/bin/env python3
"""
Advanced Performance Profiler for Mixed-Precision Multigrid Solvers
Provides comprehensive profiling capabilities including CPU, memory, and GPU analysis
"""

import time
import psutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import json
import cProfile
import pstats
import io
from pathlib import Path
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

try:
    from memory_profiler import profile as memory_profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import line_profiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    execution_time: float = 0.0
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    peak_memory: float = 0.0
    gpu_memory_usage: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    function_calls: Dict[str, int] = field(default_factory=dict)
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """Background system resource monitoring"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.gpu_utilization = []
        self._thread = None
        
    def start(self):
        """Start monitoring system resources"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_memory_usage.clear()
        self.gpu_utilization.clear()
        
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, List[float]]:
        """Stop monitoring and return collected data"""
        self.monitoring = False
        if self._thread:
            self._thread.join()
            
        return {
            'cpu_usage': self.cpu_usage.copy(),
            'memory_usage': self.memory_usage.copy(),
            'gpu_memory_usage': self.gpu_memory_usage.copy(),
            'gpu_utilization': self.gpu_utilization.copy()
        }
    
    def _monitor(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and memory monitoring
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_mb)
                
                # GPU monitoring if available
                if HAS_GPU:
                    try:
                        mempool = cp.get_default_memory_pool()
                        gpu_memory_mb = mempool.used_bytes() / 1024 / 1024
                        self.gpu_memory_usage.append(gpu_memory_mb)
                        
                        # GPU utilization (simplified metric)
                        device = cp.cuda.Device()
                        with device:
                            # This is a simplified metric - real GPU utilization
                            # would require nvidia-ml-py or similar
                            self.gpu_utilization.append(50.0)  # Placeholder
                    except:
                        pass
                        
            except psutil.NoSuchProcess:
                break
            except Exception:
                pass  # Continue monitoring despite errors
                
            time.sleep(self.interval)


class AdvancedProfiler:
    """Advanced profiler with multiple profiling modes"""
    
    def __init__(self, 
                 enable_cpu_profiling: bool = True,
                 enable_memory_profiling: bool = True,
                 enable_line_profiling: bool = False,
                 enable_gpu_profiling: bool = HAS_GPU,
                 monitor_interval: float = 0.1):
        
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_line_profiling = enable_line_profiling and HAS_LINE_PROFILER
        self.enable_gpu_profiling = enable_gpu_profiling and HAS_GPU
        
        self.system_monitor = SystemMonitor(monitor_interval)
        self.cpu_profiler = None
        self.results = PerformanceMetrics()
        
    @contextmanager
    def profile(self, name: str = "execution"):
        """Context manager for comprehensive profiling"""
        print(f"Starting profiling: {name}")
        
        # Initialize profilers
        if self.enable_cpu_profiling:
            self.cpu_profiler = cProfile.Profile()
            self.cpu_profiler.enable()
            
        # Start system monitoring
        self.system_monitor.start()
        
        # Record start time and memory
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # GPU profiling start
        if self.enable_gpu_profiling:
            try:
                cp.cuda.profiler.start()
            except:
                pass
        
        try:
            yield self.results
        finally:
            # Record end time
            end_time = time.perf_counter()
            self.results.execution_time = end_time - start_time
            
            # Stop system monitoring
            monitor_data = self.system_monitor.stop()
            self.results.cpu_usage = monitor_data['cpu_usage']
            self.results.memory_usage = monitor_data['memory_usage']
            self.results.peak_memory = max(monitor_data['memory_usage']) if monitor_data['memory_usage'] else 0.0
            self.results.gpu_memory_usage = monitor_data['gpu_memory_usage']
            self.results.gpu_utilization = monitor_data['gpu_utilization']
            
            # Stop CPU profiling
            if self.enable_cpu_profiling and self.cpu_profiler:
                self.cpu_profiler.disable()
                self._analyze_cpu_profile()
                
            # Stop GPU profiling
            if self.enable_gpu_profiling:
                try:
                    cp.cuda.profiler.stop()
                except:
                    pass
                    
            print(f"Profiling completed: {name}")
            print(f"Execution time: {self.results.execution_time:.4f}s")
            print(f"Peak memory: {self.results.peak_memory:.1f} MB")
    
    def _analyze_cpu_profile(self):
        """Analyze CPU profiling results"""
        if not self.cpu_profiler:
            return
            
        # Capture profiling stats
        s = io.StringIO()
        ps = pstats.Stats(self.cpu_profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Parse stats for hotspots
        stats = ps.get_stats_profile()
        self.results.function_calls = {}
        self.results.hotspots = []
        
        for func_key, (cc, nc, tt, ct, callers) in stats.func_profiles.items():
            filename, line, func_name = func_key
            self.results.function_calls[f"{filename}:{func_name}"] = nc
            
            if tt > 0.01:  # Functions taking more than 10ms
                self.results.hotspots.append({
                    'function': f"{filename}:{line}({func_name})",
                    'calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'per_call_time': tt / nc if nc > 0 else 0
                })
        
        # Sort hotspots by total time
        self.results.hotspots.sort(key=lambda x: x['total_time'], reverse=True)
    
    def save_results(self, output_path: Path):
        """Save profiling results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        results_dict = {
            'execution_time': self.results.execution_time,
            'peak_memory_mb': self.results.peak_memory,
            'average_cpu_usage': np.mean(self.results.cpu_usage) if self.results.cpu_usage else 0.0,
            'average_memory_mb': np.mean(self.results.memory_usage) if self.results.memory_usage else 0.0,
            'gpu_peak_memory_mb': max(self.results.gpu_memory_usage) if self.results.gpu_memory_usage else 0.0,
            'function_calls_count': len(self.results.function_calls),
            'hotspots': self.results.hotspots[:10],  # Top 10 hotspots
            'custom_metrics': self.results.custom_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Profiling results saved to: {output_path}")
    
    def generate_report(self) -> str:
        """Generate human-readable profiling report"""
        report = []
        report.append("=" * 50)
        report.append("PERFORMANCE PROFILING REPORT")
        report.append("=" * 50)
        
        # Execution summary
        report.append(f"\nExecution Time: {self.results.execution_time:.4f} seconds")
        report.append(f"Peak Memory Usage: {self.results.peak_memory:.1f} MB")
        
        if self.results.cpu_usage:
            avg_cpu = np.mean(self.results.cpu_usage)
            max_cpu = np.max(self.results.cpu_usage)
            report.append(f"CPU Usage - Average: {avg_cpu:.1f}%, Peak: {max_cpu:.1f}%")
        
        if self.results.gpu_memory_usage:
            avg_gpu_mem = np.mean(self.results.gpu_memory_usage)
            max_gpu_mem = np.max(self.results.gpu_memory_usage)
            report.append(f"GPU Memory - Average: {avg_gpu_mem:.1f} MB, Peak: {max_gpu_mem:.1f} MB")
        
        # Function hotspots
        if self.results.hotspots:
            report.append(f"\nTop {min(5, len(self.results.hotspots))} Performance Hotspots:")
            for i, hotspot in enumerate(self.results.hotspots[:5], 1):
                report.append(f"{i}. {hotspot['function']}")
                report.append(f"   Total Time: {hotspot['total_time']:.4f}s")
                report.append(f"   Calls: {hotspot['calls']}")
                report.append(f"   Time/Call: {hotspot['per_call_time']:.6f}s")
        
        # Custom metrics
        if self.results.custom_metrics:
            report.append("\nCustom Metrics:")
            for key, value in self.results.custom_metrics.items():
                report.append(f"  {key}: {value}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)


class ProfileGuidedOptimizer:
    """Profile-guided optimization for multigrid solvers"""
    
    def __init__(self, profiler: AdvancedProfiler):
        self.profiler = profiler
        self.optimization_suggestions = []
    
    def analyze_and_optimize(self, solver_instance) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations"""
        suggestions = {
            'grid_optimization': self._analyze_grid_performance(),
            'precision_optimization': self._analyze_precision_performance(),
            'memory_optimization': self._analyze_memory_performance(),
            'parallelization_optimization': self._analyze_parallelization_performance()
        }
        
        return suggestions
    
    def _analyze_grid_performance(self) -> Dict[str, str]:
        """Analyze grid-related performance"""
        suggestions = {}
        
        if self.profiler.results.peak_memory > 1000:  # > 1GB
            suggestions['grid_size'] = "Consider reducing grid size to decrease memory usage"
        
        if self.profiler.results.execution_time > 60:  # > 1 minute
            suggestions['grid_coarsening'] = "Consider more aggressive grid coarsening"
            
        return suggestions
    
    def _analyze_precision_performance(self) -> Dict[str, str]:
        """Analyze precision-related performance"""
        suggestions = {}
        
        # Look for functions related to precision conversion in hotspots
        precision_hotspots = [h for h in self.profiler.results.hotspots 
                            if 'precision' in h['function'].lower() or 
                               'convert' in h['function'].lower()]
        
        if precision_hotspots:
            total_precision_time = sum(h['total_time'] for h in precision_hotspots)
            if total_precision_time > 0.1 * self.profiler.results.execution_time:
                suggestions['precision_strategy'] = "Precision conversions are costly - consider adaptive strategy"
        
        return suggestions
    
    def _analyze_memory_performance(self) -> Dict[str, str]:
        """Analyze memory-related performance"""
        suggestions = {}
        
        memory_usage = self.profiler.results.memory_usage
        if memory_usage:
            memory_variance = np.var(memory_usage)
            if memory_variance > 100:  # High variance in memory usage
                suggestions['memory_management'] = "High memory variance detected - consider memory pooling"
        
        return suggestions
    
    def _analyze_parallelization_performance(self) -> Dict[str, str]:
        """Analyze parallelization opportunities"""
        suggestions = {}
        
        avg_cpu = np.mean(self.profiler.results.cpu_usage) if self.profiler.results.cpu_usage else 0
        if avg_cpu < 50:  # Low CPU utilization
            suggestions['threading'] = "Low CPU utilization - consider increasing thread count"
        
        return suggestions


def profile_solver_execution(solver_func: Callable, 
                           *args, 
                           output_dir: Path = Path("profiling_results"),
                           **kwargs) -> PerformanceMetrics:
    """
    Comprehensive profiling of solver execution
    
    Args:
        solver_func: Function to profile
        *args: Arguments to pass to solver_func
        output_dir: Directory to save profiling results
        **kwargs: Keyword arguments to pass to solver_func
    
    Returns:
        PerformanceMetrics object with profiling results
    """
    
    profiler = AdvancedProfiler(
        enable_cpu_profiling=True,
        enable_memory_profiling=True,
        enable_gpu_profiling=HAS_GPU
    )
    
    with profiler.profile("solver_execution") as metrics:
        result = solver_func(*args, **kwargs)
        metrics.custom_metrics['solver_result'] = str(type(result).__name__)
    
    # Save results
    timestamp = int(time.time())
    results_file = output_dir / f"profiling_results_{timestamp}.json"
    profiler.save_results(results_file)
    
    # Generate and save report
    report = profiler.generate_report()
    report_file = output_dir / f"profiling_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    
    # Generate optimization suggestions
    optimizer = ProfileGuidedOptimizer(profiler)
    suggestions = optimizer.analyze_and_optimize(None)
    
    suggestions_file = output_dir / f"optimization_suggestions_{timestamp}.json"
    with open(suggestions_file, 'w') as f:
        json.dump(suggestions, f, indent=2)
    
    print(f"\nOptimization suggestions saved to: {suggestions_file}")
    
    return profiler.results


if __name__ == "__main__":
    # Example usage and self-test
    def dummy_solver(grid_size: int = 100, iterations: int = 100):
        """Dummy solver for testing profiler"""
        import numpy as np
        import time
        
        # Simulate solver work
        data = np.random.random((grid_size, grid_size))
        
        for i in range(iterations):
            data = np.fft.fft2(data)
            data = np.fft.ifft2(data)
            if i % 10 == 0:
                time.sleep(0.01)  # Simulate I/O or synchronization
        
        return np.sum(data.real)
    
    # Profile the dummy solver
    result = profile_solver_execution(
        dummy_solver,
        grid_size=200,
        iterations=50,
        output_dir=Path("test_profiling")
    )
    
    print(f"\nProfiling completed. Peak memory: {result.peak_memory:.1f} MB")