"""GPU performance profiling tools for multigrid solvers."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class GPUOperationProfile:
    """Profile data for GPU operations."""
    name: str
    call_count: int = 0
    total_gpu_time: float = 0.0
    total_cpu_time: float = 0.0
    min_gpu_time: float = float('inf')
    max_gpu_time: float = 0.0
    total_memory_bytes: int = 0
    kernel_launches: int = 0
    
    @property
    def average_gpu_time(self) -> float:
        """Average GPU execution time."""
        return self.total_gpu_time / max(1, self.call_count)
    
    @property
    def gpu_efficiency(self) -> float:
        """GPU time efficiency vs CPU time."""
        if self.total_cpu_time > 0:
            return self.total_gpu_time / self.total_cpu_time
        return 0.0
    
    @property
    def memory_bandwidth_gb_s(self) -> float:
        """Memory bandwidth in GB/s."""
        return (self.total_memory_bytes / (1024**3)) / max(1e-9, self.total_gpu_time)


class GPUPerformanceProfiler:
    """
    Advanced GPU performance profiler for multigrid operations.
    
    Tracks GPU timing, memory usage, kernel launches, and efficiency metrics.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        enable_detailed_profiling: bool = True,
        track_memory_usage: bool = True
    ):
        """
        Initialize GPU performance profiler.
        
        Args:
            device_id: GPU device ID
            enable_detailed_profiling: Enable detailed profiling metrics
            track_memory_usage: Track memory usage patterns
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU performance profiler")
        
        self.device_id = device_id
        self.enable_detailed_profiling = enable_detailed_profiling
        self.track_memory_usage = track_memory_usage
        
        # Profiling data
        self.profiles: Dict[str, GPUOperationProfile] = {}
        self.gpu_events: List[Dict[str, Any]] = []
        self.memory_snapshots: List[Dict[str, Any]] = []
        
        # CUDA events for accurate timing
        self.event_pool: List['cp.cuda.Event'] = []
        self.active_events: Dict[str, Tuple['cp.cuda.Event', 'cp.cuda.Event']] = {}
        
        # Global GPU statistics
        self.global_gpu_stats = {
            'total_gpu_operations': 0,
            'total_gpu_time': 0.0,
            'total_kernel_launches': 0,
            'total_memory_transfers': 0,
            'peak_memory_usage': 0
        }
        
        # Initialize CUDA events pool
        with cp.cuda.Device(device_id):
            for _ in range(100):  # Pre-allocate event objects
                self.event_pool.append(cp.cuda.Event())
        
        logger.info(f"GPU performance profiler initialized: device={device_id}")
    
    def _get_cuda_events(self) -> Tuple['cp.cuda.Event', 'cp.cuda.Event']:
        """Get pair of CUDA events for timing."""
        if len(self.event_pool) >= 2:
            start_event = self.event_pool.pop()
            end_event = self.event_pool.pop()
        else:
            # Create new events if pool is empty
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
        
        return start_event, end_event
    
    def _return_cuda_events(self, start_event: 'cp.cuda.Event', end_event: 'cp.cuda.Event') -> None:
        """Return CUDA events to pool."""
        self.event_pool.append(start_event)
        self.event_pool.append(end_event)
    
    @contextmanager
    def profile_gpu_operation(
        self,
        operation_name: str,
        memory_bytes: Optional[int] = None,
        kernel_count: int = 1
    ):
        """
        Context manager for profiling GPU operations.
        
        Args:
            operation_name: Name of the operation
            memory_bytes: Memory bytes accessed
            kernel_count: Number of kernel launches
        """
        # Initialize profile if needed
        if operation_name not in self.profiles:
            self.profiles[operation_name] = GPUOperationProfile(operation_name)
        
        profile = self.profiles[operation_name]
        
        # Get CUDA events for GPU timing
        start_event, end_event = self._get_cuda_events()
        
        # Record memory usage before operation
        if self.track_memory_usage:
            memory_before = self._get_gpu_memory_usage()
        
        # Start timing
        cpu_start = time.perf_counter()
        
        with cp.cuda.Device(self.device_id):
            start_event.record()
        
        try:
            yield profile
        finally:
            # End timing
            with cp.cuda.Device(self.device_id):
                end_event.record()
                end_event.synchronize()  # Wait for GPU to complete
            
            cpu_end = time.perf_counter()
            
            # Calculate timings
            gpu_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0  # Convert to seconds
            cpu_time = cpu_end - cpu_start
            
            # Update profile
            profile.call_count += 1
            profile.total_gpu_time += gpu_time
            profile.total_cpu_time += cpu_time
            profile.min_gpu_time = min(profile.min_gpu_time, gpu_time)
            profile.max_gpu_time = max(profile.max_gpu_time, gpu_time)
            profile.kernel_launches += kernel_count
            
            if memory_bytes:
                profile.total_memory_bytes += memory_bytes
            
            # Record memory usage after operation
            if self.track_memory_usage:
                memory_after = self._get_gpu_memory_usage()
                memory_delta = memory_after - memory_before
                
                if memory_delta > 0:
                    profile.total_memory_bytes += memory_delta
                
                self.global_gpu_stats['peak_memory_usage'] = max(
                    self.global_gpu_stats['peak_memory_usage'],
                    memory_after
                )
            
            # Update global stats
            self.global_gpu_stats['total_gpu_operations'] += 1
            self.global_gpu_stats['total_gpu_time'] += gpu_time
            self.global_gpu_stats['total_kernel_launches'] += kernel_count
            
            # Store detailed event data
            if self.enable_detailed_profiling:
                self.gpu_events.append({
                    'operation': operation_name,
                    'gpu_time': gpu_time,
                    'cpu_time': cpu_time,
                    'memory_bytes': memory_bytes or 0,
                    'kernel_count': kernel_count,
                    'timestamp': time.time()
                })
            
            # Return events to pool
            self._return_cuda_events(start_event, end_event)
    
    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage."""
        try:
            with cp.cuda.Device(self.device_id):
                mempool = cp.get_default_memory_pool()
                return mempool.used_bytes()
        except Exception:
            return 0
    
    def take_memory_snapshot(self, label: str) -> Dict[str, Any]:
        """
        Take GPU memory usage snapshot.
        
        Args:
            label: Label for the snapshot
            
        Returns:
            Memory usage snapshot
        """
        try:
            with cp.cuda.Device(self.device_id):
                mempool = cp.get_default_memory_pool()
                meminfo = cp.cuda.runtime.memGetInfo()
                
                snapshot = {
                    'label': label,
                    'timestamp': time.time(),
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes(),
                    'free_gpu_memory': meminfo[0],
                    'total_gpu_memory': meminfo[1],
                    'utilization': (mempool.used_bytes() / max(meminfo[1], 1)) * 100
                }
                
                if self.track_memory_usage:
                    self.memory_snapshots.append(snapshot)
                
                return snapshot
        
        except Exception as e:
            logger.warning(f"Failed to take memory snapshot: {e}")
            return {'label': label, 'error': str(e)}
    
    def profile_kernel_launch(self, kernel_name: str, grid_size: Tuple, block_size: Tuple) -> None:
        """
        Profile kernel launch parameters.
        
        Args:
            kernel_name: Name of the kernel
            grid_size: Grid dimensions
            block_size: Block dimensions
        """
        if self.enable_detailed_profiling:
            self.gpu_events.append({
                'type': 'kernel_launch',
                'kernel': kernel_name,
                'grid_size': grid_size,
                'block_size': block_size,
                'threads_per_block': np.prod(block_size),
                'total_threads': np.prod(grid_size) * np.prod(block_size),
                'timestamp': time.time()
            })
        
        self.global_gpu_stats['total_kernel_launches'] += 1
    
    def get_gpu_utilization(self, time_window: float = 10.0) -> Dict[str, float]:
        """
        Estimate GPU utilization over time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Utilization metrics
        """
        current_time = time.time()
        recent_events = [
            event for event in self.gpu_events
            if current_time - event.get('timestamp', 0) <= time_window
        ]
        
        if not recent_events:
            return {'compute_utilization': 0.0, 'memory_utilization': 0.0}
        
        total_gpu_time = sum(event.get('gpu_time', 0) for event in recent_events)
        compute_utilization = (total_gpu_time / time_window) * 100
        
        # Estimate memory utilization from bandwidth usage
        total_memory_ops = sum(event.get('memory_bytes', 0) for event in recent_events)
        peak_bandwidth_gb_s = 900  # Rough estimate for modern GPUs
        actual_bandwidth = (total_memory_ops / (1024**3)) / time_window
        memory_utilization = (actual_bandwidth / peak_bandwidth_gb_s) * 100
        
        return {
            'compute_utilization': min(compute_utilization, 100.0),
            'memory_utilization': min(memory_utilization, 100.0),
            'recent_operations': len(recent_events)
        }
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get comprehensive GPU profiling summary."""
        if not self.profiles:
            return {'error': 'No GPU profiling data available'}
        
        # Sort profiles by GPU time
        sorted_profiles = sorted(
            self.profiles.items(),
            key=lambda x: x[1].total_gpu_time,
            reverse=True
        )
        
        total_gpu_time = self.global_gpu_stats['total_gpu_time']
        
        summary = {
            'global_stats': self.global_gpu_stats.copy(),
            'operation_profiles': [],
            'top_gpu_operations': [],
            'gpu_efficiency_metrics': {},
            'memory_analysis': {}
        }
        
        # Operation profiles
        for name, profile in sorted_profiles:
            gpu_time_percentage = (profile.total_gpu_time / max(total_gpu_time, 1e-9)) * 100
            
            profile_data = {
                'name': name,
                'call_count': profile.call_count,
                'total_gpu_time': profile.total_gpu_time,
                'average_gpu_time': profile.average_gpu_time,
                'gpu_time_percentage': gpu_time_percentage,
                'gpu_efficiency': profile.gpu_efficiency,
                'memory_bandwidth_gb_s': profile.memory_bandwidth_gb_s,
                'kernel_launches': profile.kernel_launches
            }
            
            summary['operation_profiles'].append(profile_data)
            
            # Top operations
            if len(summary['top_gpu_operations']) < 10:
                summary['top_gpu_operations'].append({
                    'name': name,
                    'gpu_time': profile.total_gpu_time,
                    'percentage': gpu_time_percentage
                })
        
        # GPU efficiency metrics
        if total_gpu_time > 0:
            total_cpu_time = sum(p.total_cpu_time for p in self.profiles.values())
            
            summary['gpu_efficiency_metrics'] = {
                'average_gpu_utilization': self.get_gpu_utilization()['compute_utilization'],
                'gpu_vs_cpu_ratio': total_gpu_time / max(total_cpu_time, 1e-9),
                'kernels_per_second': self.global_gpu_stats['total_kernel_launches'] / total_gpu_time,
                'operations_per_second': len(self.profiles) / total_gpu_time
            }
        
        # Memory analysis
        if self.memory_snapshots:
            peak_memory = max(snap.get('used_bytes', 0) for snap in self.memory_snapshots)
            avg_utilization = np.mean([snap.get('utilization', 0) for snap in self.memory_snapshots])
            
            summary['memory_analysis'] = {
                'peak_memory_mb': peak_memory / (1024**2),
                'average_utilization': avg_utilization,
                'memory_snapshots': len(self.memory_snapshots)
            }
        
        return summary
    
    def generate_gpu_report(self, top_n: int = 15) -> str:
        """
        Generate comprehensive GPU performance report.
        
        Args:
            top_n: Number of top operations to include
            
        Returns:
            Formatted GPU performance report
        """
        summary = self.get_profiling_summary()
        
        if 'error' in summary:
            return summary['error']
        
        report = [
            "GPU Performance Profiling Report",
            "=" * 50,
            ""
        ]
        
        # Global GPU statistics
        global_stats = summary['global_stats']
        report.extend([
            "Global GPU Statistics:",
            f"  Total GPU Operations: {global_stats['total_gpu_operations']:,}",
            f"  Total GPU Time: {global_stats['total_gpu_time']:.3f} s",
            f"  Total Kernel Launches: {global_stats['total_kernel_launches']:,}",
            f"  Peak Memory Usage: {global_stats['peak_memory_usage'] / (1024**2):.2f} MB",
            ""
        ])
        
        # GPU efficiency metrics
        if 'gpu_efficiency_metrics' in summary:
            metrics = summary['gpu_efficiency_metrics']
            report.extend([
                "GPU Efficiency Metrics:",
                f"  GPU Utilization: {metrics['average_gpu_utilization']:.1f}%",
                f"  GPU vs CPU Ratio: {metrics['gpu_vs_cpu_ratio']:.2f}",
                f"  Kernels/sec: {metrics['kernels_per_second']:.2f}",
                f"  Operations/sec: {metrics['operations_per_second']:.2f}",
                ""
            ])
        
        # Top GPU operations
        report.extend([
            f"Top {top_n} GPU Operations by Time:",
            "-" * 45
        ])
        
        for i, op in enumerate(summary['operation_profiles'][:top_n]):
            report.append(
                f"{i+1:2d}. {op['name']:25s} "
                f"{op['total_gpu_time']:8.3f}s ({op['gpu_time_percentage']:5.1f}%) "
                f"[{op['call_count']:6d} calls, {op['kernel_launches']:4d} kernels]"
            )
        
        # Memory analysis
        if 'memory_analysis' in summary and summary['memory_analysis']:
            mem_analysis = summary['memory_analysis']
            report.extend([
                "",
                "GPU Memory Analysis:",
                "-" * 25,
                f"Peak Memory: {mem_analysis['peak_memory_mb']:.2f} MB",
                f"Average Utilization: {mem_analysis['average_utilization']:.1f}%",
                f"Memory Snapshots: {mem_analysis['memory_snapshots']}"
            ])
        
        # Performance recommendations
        recommendations = self._generate_gpu_recommendations(summary)
        if recommendations:
            report.extend(["", "GPU Performance Recommendations:", "-" * 35])
            report.extend(recommendations)
        
        return "\n".join(report)
    
    def _generate_gpu_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate GPU performance optimization recommendations."""
        recommendations = []
        
        if not summary['operation_profiles']:
            return recommendations
        
        # Check for low GPU utilization
        if 'gpu_efficiency_metrics' in summary:
            gpu_util = summary['gpu_efficiency_metrics']['average_gpu_utilization']
            if gpu_util < 50:
                recommendations.append(
                    f"• Low GPU utilization ({gpu_util:.1f}%) - consider larger problem sizes or block sizes"
                )
        
        # Check for inefficient operations
        for op in summary['operation_profiles'][:5]:
            if op['gpu_efficiency'] > 0.8:  # GPU time close to CPU time
                recommendations.append(
                    f"• {op['name']} may not benefit from GPU acceleration (efficiency: {op['gpu_efficiency']:.2f})"
                )
            
            if op['memory_bandwidth_gb_s'] < 100:  # Low memory bandwidth
                recommendations.append(
                    f"• {op['name']} has low memory bandwidth ({op['memory_bandwidth_gb_s']:.1f} GB/s) - check memory access patterns"
                )
        
        # Check kernel launch efficiency
        total_kernels = sum(op['kernel_launches'] for op in summary['operation_profiles'])
        total_ops = len(summary['operation_profiles'])
        
        if total_kernels / max(total_ops, 1) > 10:
            recommendations.append(
                "• High kernel launch overhead detected - consider kernel fusion or batching"
            )
        
        # Memory utilization recommendations
        if 'memory_analysis' in summary and summary['memory_analysis']:
            mem_util = summary['memory_analysis']['average_utilization']
            if mem_util > 90:
                recommendations.append(
                    f"• High memory utilization ({mem_util:.1f}%) - consider memory optimization"
                )
            elif mem_util < 30:
                recommendations.append(
                    f"• Low memory utilization ({mem_util:.1f}%) - GPU may be underutilized"
                )
        
        return recommendations
    
    def export_profiling_data(self, filename: str) -> None:
        """Export profiling data to file."""
        import json
        
        export_data = {
            'summary': self.get_profiling_summary(),
            'gpu_events': self.gpu_events,
            'memory_snapshots': self.memory_snapshots,
            'device_id': self.device_id
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"GPU profiling data exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export profiling data: {e}")
    
    def reset(self) -> None:
        """Reset all GPU profiling data."""
        self.profiles.clear()
        self.gpu_events.clear()
        self.memory_snapshots.clear()
        self.active_events.clear()
        
        self.global_gpu_stats = {
            'total_gpu_operations': 0,
            'total_gpu_time': 0.0,
            'total_kernel_launches': 0,
            'total_memory_transfers': 0,
            'peak_memory_usage': 0
        }
        
        logger.debug("GPU performance profiler reset")
    
    def __del__(self):
        """Cleanup GPU profiler resources."""
        try:
            # Return all events to pool
            for events in self.active_events.values():
                self._return_cuda_events(events[0], events[1])
            
            self.active_events.clear()
        except Exception:
            pass