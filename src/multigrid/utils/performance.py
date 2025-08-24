"""Performance profiling utilities."""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Container for timing results."""
    name: str
    total_time: float
    call_count: int
    average_time: float = field(init=False)
    min_time: float = field(default=float('inf'))
    max_time: float = field(default=0.0)
    times: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate average time."""
        self.average_time = self.total_time / max(1, self.call_count)
    
    def add_time(self, elapsed_time: float) -> None:
        """Add a new timing measurement."""
        self.total_time += elapsed_time
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
        self.times.append(elapsed_time)
        self.average_time = self.total_time / self.call_count
    
    def get_statistics(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.times:
            return {'count': 0}
        
        times_array = np.array(self.times)
        return {
            'count': self.call_count,
            'total': self.total_time,
            'average': self.average_time,
            'min': self.min_time,
            'max': self.max_time,
            'median': np.median(times_array),
            'std': np.std(times_array),
            'percentile_95': np.percentile(times_array, 95)
        }


class Timer:
    """Simple timer for measuring elapsed time."""
    
    def __init__(self, name: str = "Timer"):
        """Initialize timer."""
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def start(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def __enter__(self) -> 'Timer':
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.stop()
        logger.debug(f"{self.name}: {self.elapsed_time:.3f}s")


class PerformanceProfiler:
    """Performance profiler for tracking operation timings."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timings: Dict[str, TimingResult] = {}
        self.memory_usage: List[float] = []
        self.active_timers: Dict[str, float] = {}
    
    def start_timing(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.active_timers[operation_name] = time.time()
    
    def end_timing(self, operation_name: str) -> float:
        """End timing an operation and record result."""
        if operation_name not in self.active_timers:
            raise ValueError(f"No active timer for operation: {operation_name}")
        
        elapsed_time = time.time() - self.active_timers.pop(operation_name)
        
        if operation_name not in self.timings:
            self.timings[operation_name] = TimingResult(
                name=operation_name,
                total_time=0.0,
                call_count=0
            )
        
        self.timings[operation_name].add_time(elapsed_time)
        return elapsed_time
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        self.start_timing(operation_name)
        try:
            yield
        finally:
            self.end_timing(operation_name)
    
    def time_function(self, operation_name: Optional[str] = None):
        """Decorator for timing function calls."""
        def decorator(func: Callable):
            name = operation_name or func.__name__
            
            def wrapper(*args, **kwargs):
                with self.time_operation(name):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def record_memory_usage(self) -> None:
        """Record current memory usage (requires psutil)."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
    
    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all timing results."""
        return {name: result.get_statistics() 
                for name, result in self.timings.items()}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_usage:
            return {}
        
        memory_array = np.array(self.memory_usage)
        return {
            'count': len(self.memory_usage),
            'current': self.memory_usage[-1],
            'average': np.mean(memory_array),
            'min': np.min(memory_array),
            'max': np.max(memory_array),
            'std': np.std(memory_array)
        }
    
    def log_summary(self, top_n: int = 10) -> None:
        """Log performance summary."""
        logger.info("Performance Summary:")
        logger.info("=" * 50)
        
        # Sort operations by total time
        sorted_timings = sorted(
            self.timings.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        for i, (name, result) in enumerate(sorted_timings[:top_n]):
            stats = result.get_statistics()
            logger.info(
                f"{i+1:2d}. {name:25s}: "
                f"{stats['total']:8.3f}s total, "
                f"{stats['average']:8.3f}s avg, "
                f"{stats['count']:6d} calls"
            )
        
        # Memory summary
        memory_stats = self.get_memory_stats()
        if memory_stats:
            logger.info(f"\nMemory Usage:")
            logger.info(f"  Current: {memory_stats['current']:.1f} MB")
            logger.info(f"  Average: {memory_stats['average']:.1f} MB")
            logger.info(f"  Peak:    {memory_stats['max']:.1f} MB")
    
    def reset(self) -> None:
        """Reset all performance data."""
        self.timings.clear()
        self.memory_usage.clear()
        self.active_timers.clear()
        logger.info("Performance profiler reset")
    
    def export_data(self) -> Dict[str, Any]:
        """Export all performance data."""
        return {
            'timings': {name: {
                'name': result.name,
                'total_time': result.total_time,
                'call_count': result.call_count,
                'statistics': result.get_statistics()
            } for name, result in self.timings.items()},
            'memory_usage': self.memory_usage,
            'memory_statistics': self.get_memory_stats()
        }


class MultigridProfiler(PerformanceProfiler):
    """Specialized profiler for multigrid operations."""
    
    def __init__(self):
        """Initialize multigrid profiler."""
        super().__init__()
        self.level_timings: Dict[int, Dict[str, TimingResult]] = {}
        self.iteration_timings: List[float] = []
    
    def time_level_operation(self, level: int, operation: str):
        """Context manager for timing level-specific operations."""
        operation_name = f"level_{level}_{operation}"
        return self.time_operation(operation_name)
    
    def record_iteration(self, iteration_time: float) -> None:
        """Record timing for a complete iteration."""
        self.iteration_timings.append(iteration_time)
        
        if len(self.iteration_timings) % 10 == 0:
            avg_time = np.mean(self.iteration_timings[-10:])
            logger.debug(f"Last 10 iterations average: {avg_time:.3f}s")
    
    def get_level_summary(self) -> Dict[int, Dict[str, float]]:
        """Get summary of level-specific timings."""
        level_summary = {}
        
        for name, result in self.timings.items():
            if name.startswith("level_"):
                parts = name.split("_", 2)
                if len(parts) >= 3:
                    level = int(parts[1])
                    operation = parts[2]
                    
                    if level not in level_summary:
                        level_summary[level] = {}
                    
                    level_summary[level][operation] = result.total_time
        
        return level_summary
    
    def log_multigrid_summary(self) -> None:
        """Log multigrid-specific performance summary."""
        logger.info("Multigrid Performance Summary:")
        logger.info("=" * 60)
        
        # Overall iteration timing
        if self.iteration_timings:
            iter_array = np.array(self.iteration_timings)
            logger.info(f"Iterations: {len(self.iteration_timings)}")
            logger.info(f"  Total time:   {np.sum(iter_array):.3f}s")
            logger.info(f"  Average time: {np.mean(iter_array):.3f}s")
            logger.info(f"  Min time:     {np.min(iter_array):.3f}s")
            logger.info(f"  Max time:     {np.max(iter_array):.3f}s")
        
        # Level-specific timings
        level_summary = self.get_level_summary()
        if level_summary:
            logger.info(f"\nLevel-wise Timings:")
            for level in sorted(level_summary.keys()):
                operations = level_summary[level]
                total_level_time = sum(operations.values())
                logger.info(f"  Level {level}: {total_level_time:.3f}s total")
                for op, time_val in sorted(operations.items()):
                    pct = (time_val / total_level_time * 100) if total_level_time > 0 else 0
                    logger.info(f"    {op:15s}: {time_val:8.3f}s ({pct:5.1f}%)")
        
        # Call parent summary for general operations
        self.log_summary(top_n=5)


# Global profiler instance
global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return global_profiler


def time_operation(operation_name: str):
    """Decorator to time operations using global profiler."""
    return global_profiler.time_function(operation_name)


@contextmanager
def profile_operation(operation_name: str):
    """Context manager to profile operations using global profiler."""
    with global_profiler.time_operation(operation_name):
        yield


def benchmark_function(func: Callable, *args, num_runs: int = 1, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function with multiple runs.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of benchmark runs
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark statistics
    """
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
    
    times_array = np.array(times)
    
    return {
        'runs': num_runs,
        'total_time': np.sum(times_array),
        'average_time': np.mean(times_array),
        'min_time': np.min(times_array),
        'max_time': np.max(times_array),
        'std_time': np.std(times_array),
        'median_time': np.median(times_array)
    }