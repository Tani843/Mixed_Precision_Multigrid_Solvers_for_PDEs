"""Memory management and optimization for multigrid solvers."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import gc
import threading
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    array: np.ndarray
    in_use: bool = False
    last_used: float = 0.0
    allocation_id: int = 0
    
    @property
    def size_bytes(self) -> int:
        """Size of the memory block in bytes."""
        return self.array.nbytes
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the memory block."""
        return self.array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Data type of the memory block."""
        return self.array.dtype


class MemoryPool:
    """
    Memory pool for efficient array allocation and reuse.
    
    Manages memory blocks to reduce allocation overhead and
    improve cache locality in multigrid operations.
    """
    
    def __init__(
        self,
        max_pool_size_mb: float = 512.0,
        block_alignment: int = 64,
        enable_statistics: bool = True
    ):
        """
        Initialize memory pool.
        
        Args:
            max_pool_size_mb: Maximum pool size in megabytes
            block_alignment: Memory alignment for blocks (cache line size)
            enable_statistics: Enable memory usage statistics
        """
        self.max_pool_size_bytes = int(max_pool_size_mb * 1024 * 1024)
        self.block_alignment = block_alignment
        self.enable_statistics = enable_statistics
        
        # Pool storage: {(shape, dtype): [MemoryBlock, ...]}
        self.pools: Dict[Tuple[Tuple[int, ...], np.dtype], List[MemoryBlock]] = {}
        
        # Statistics
        self.stats = {
            'total_allocated_bytes': 0,
            'total_allocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'pool_evictions': 0,
            'allocation_id_counter': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Memory pool initialized: max_size={max_pool_size_mb}MB, "
                   f"alignment={block_alignment}")
    
    def allocate(
        self,
        shape: Union[Tuple[int, ...], int],
        dtype: np.dtype = np.float64,
        zero_fill: bool = True
    ) -> np.ndarray:
        """
        Allocate array from memory pool.
        
        Args:
            shape: Array shape
            dtype: Array data type
            zero_fill: Whether to zero-fill the array
            
        Returns:
            Allocated numpy array
        """
        if isinstance(shape, int):
            shape = (shape,)
        
        pool_key = (shape, dtype)
        
        with self._lock:
            # Check if we have available blocks
            if pool_key in self.pools:
                pool = self.pools[pool_key]
                
                # Find available block
                for block in pool:
                    if not block.in_use:
                        block.in_use = True
                        block.last_used = self._get_time()
                        
                        if zero_fill:
                            block.array.fill(0)
                        
                        if self.enable_statistics:
                            self.stats['cache_hits'] += 1
                        
                        logger.debug(f"Allocated from pool: {shape}, {dtype}")
                        return block.array
            
            # No available block, allocate new
            array = self._allocate_new_block(shape, dtype)
            
            if zero_fill:
                array.fill(0)
            
            if self.enable_statistics:
                self.stats['cache_misses'] += 1
                self.stats['total_allocations'] += 1
                self.stats['allocation_id_counter'] += 1
            
            return array
    
    def deallocate(self, array: np.ndarray) -> None:
        """
        Deallocate array back to memory pool.
        
        Args:
            array: Array to deallocate
        """
        if array is None:
            return
        
        shape = array.shape
        dtype = array.dtype
        pool_key = (shape, dtype)
        
        with self._lock:
            if pool_key in self.pools:
                pool = self.pools[pool_key]
                
                # Find the corresponding block
                for block in pool:
                    if np.shares_memory(array, block.array):
                        block.in_use = False
                        block.last_used = self._get_time()
                        logger.debug(f"Deallocated to pool: {shape}, {dtype}")
                        return
            
            # If not found in pool, ignore (was allocated outside pool)
            logger.debug(f"Array not found in pool, ignoring: {shape}, {dtype}")
    
    def _allocate_new_block(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Allocate new memory block."""
        # Check if we need to evict blocks
        required_bytes = np.prod(shape) * dtype.itemsize
        self._ensure_space_available(required_bytes)
        
        # Create aligned array
        array = self._create_aligned_array(shape, dtype)
        
        # Create memory block
        block = MemoryBlock(
            array=array,
            in_use=True,
            last_used=self._get_time(),
            allocation_id=self.stats['allocation_id_counter']
        )
        
        # Add to pool
        pool_key = (shape, dtype)
        if pool_key not in self.pools:
            self.pools[pool_key] = []
        
        self.pools[pool_key].append(block)
        
        if self.enable_statistics:
            self.stats['total_allocated_bytes'] += block.size_bytes
        
        logger.debug(f"Allocated new block: {shape}, {dtype}, {block.size_bytes} bytes")
        return array
    
    def _create_aligned_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Create memory-aligned numpy array."""
        # Calculate total size
        total_elements = np.prod(shape)
        element_size = dtype.itemsize
        total_bytes = total_elements * element_size
        
        # Add padding for alignment
        aligned_bytes = ((total_bytes + self.block_alignment - 1) // 
                        self.block_alignment) * self.block_alignment
        
        # Allocate aligned memory
        try:
            # Try to use numpy's aligned allocation if available
            array = np.empty(shape, dtype=dtype)
            
            # Check alignment
            if array.ctypes.data % self.block_alignment == 0:
                return array
            
            # Fall back to manual alignment
            buffer_size = aligned_bytes + self.block_alignment
            buffer = np.empty(buffer_size, dtype=np.uint8)
            
            # Find aligned offset
            offset = (-buffer.ctypes.data) % self.block_alignment
            aligned_buffer = buffer[offset:offset + total_bytes]
            
            # Create view with correct dtype and shape
            array = aligned_buffer.view(dtype).reshape(shape)
            
            return array
        
        except Exception as e:
            logger.warning(f"Failed to create aligned array, using standard allocation: {e}")
            return np.empty(shape, dtype=dtype)
    
    def _ensure_space_available(self, required_bytes: int) -> None:
        """Ensure sufficient space is available in the pool."""
        current_size = sum(
            sum(block.size_bytes for block in pool)
            for pool in self.pools.values()
        )
        
        if current_size + required_bytes <= self.max_pool_size_bytes:
            return
        
        # Need to evict some blocks
        bytes_to_evict = current_size + required_bytes - self.max_pool_size_bytes
        bytes_evicted = 0
        
        # Sort blocks by last used time (LRU eviction)
        all_blocks = []
        for pool_key, pool in self.pools.items():
            for block in pool:
                if not block.in_use:
                    all_blocks.append((pool_key, block))
        
        all_blocks.sort(key=lambda x: x[1].last_used)
        
        # Evict oldest unused blocks
        for pool_key, block in all_blocks:
            if bytes_evicted >= bytes_to_evict:
                break
            
            pool = self.pools[pool_key]
            if block in pool:
                pool.remove(block)
                bytes_evicted += block.size_bytes
                
                if self.enable_statistics:
                    self.stats['pool_evictions'] += 1
                    self.stats['total_allocated_bytes'] -= block.size_bytes
                
                logger.debug(f"Evicted block: {block.shape}, {block.dtype}")
        
        # Clean up empty pools
        empty_pools = [k for k, v in self.pools.items() if not v]
        for k in empty_pools:
            del self.pools[k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            current_size = sum(
                sum(block.size_bytes for block in pool)
                for pool in self.pools.values()
            )
            
            total_blocks = sum(len(pool) for pool in self.pools.values())
            in_use_blocks = sum(
                sum(1 for block in pool if block.in_use)
                for pool in self.pools.values()
            )
            
            hit_rate = (self.stats['cache_hits'] / 
                       max(1, self.stats['cache_hits'] + self.stats['cache_misses']))
            
            return {
                **self.stats,
                'current_size_bytes': current_size,
                'current_size_mb': current_size / (1024 * 1024),
                'total_blocks': total_blocks,
                'in_use_blocks': in_use_blocks,
                'available_blocks': total_blocks - in_use_blocks,
                'hit_rate': hit_rate,
                'pool_utilization': current_size / self.max_pool_size_bytes,
                'num_pool_types': len(self.pools)
            }
    
    def clear(self) -> None:
        """Clear all memory pools."""
        with self._lock:
            self.pools.clear()
            
            if self.enable_statistics:
                self.stats['total_allocated_bytes'] = 0
                self.stats['pool_evictions'] += 1  # Count as eviction
            
            logger.info("Memory pool cleared")
    
    def _get_time(self) -> float:
        """Get current time for LRU tracking."""
        import time
        return time.time()


class WorkingArrayManager:
    """
    Manager for working arrays used in multigrid operations.
    
    Provides scoped allocation and automatic cleanup of temporary arrays.
    """
    
    def __init__(self, memory_pool: Optional[MemoryPool] = None):
        """
        Initialize working array manager.
        
        Args:
            memory_pool: Optional memory pool for array allocation
        """
        self.memory_pool = memory_pool
        self.active_arrays: List[np.ndarray] = []
        self._lock = threading.RLock()
    
    @contextmanager
    def allocate_working_arrays(self, *array_specs):
        """
        Context manager for allocating multiple working arrays.
        
        Args:
            *array_specs: Tuples of (shape, dtype) for each array
            
        Yields:
            List of allocated arrays
        """
        arrays = []
        
        try:
            with self._lock:
                for spec in array_specs:
                    if len(spec) == 2:
                        shape, dtype = spec
                        zero_fill = True
                    else:
                        shape, dtype, zero_fill = spec
                    
                    if self.memory_pool:
                        array = self.memory_pool.allocate(shape, dtype, zero_fill)
                    else:
                        array = np.zeros(shape, dtype=dtype) if zero_fill else np.empty(shape, dtype=dtype)
                    
                    arrays.append(array)
                    self.active_arrays.append(array)
            
            yield arrays
        
        finally:
            # Clean up arrays
            with self._lock:
                for array in arrays:
                    if array in self.active_arrays:
                        self.active_arrays.remove(array)
                    
                    if self.memory_pool:
                        self.memory_pool.deallocate(array)
    
    def allocate_like(self, reference_array: np.ndarray, zero_fill: bool = True) -> np.ndarray:
        """
        Allocate array with same shape and dtype as reference.
        
        Args:
            reference_array: Reference array
            zero_fill: Whether to zero-fill the array
            
        Returns:
            Allocated array
        """
        with self._lock:
            if self.memory_pool:
                array = self.memory_pool.allocate(
                    reference_array.shape, reference_array.dtype, zero_fill
                )
            else:
                if zero_fill:
                    array = np.zeros_like(reference_array)
                else:
                    array = np.empty_like(reference_array)
            
            self.active_arrays.append(array)
            return array
    
    def deallocate(self, array: np.ndarray) -> None:
        """
        Deallocate working array.
        
        Args:
            array: Array to deallocate
        """
        with self._lock:
            if array in self.active_arrays:
                self.active_arrays.remove(array)
            
            if self.memory_pool:
                self.memory_pool.deallocate(array)
    
    def cleanup_all(self) -> None:
        """Clean up all active arrays."""
        with self._lock:
            for array in self.active_arrays:
                if self.memory_pool:
                    self.memory_pool.deallocate(array)
            
            self.active_arrays.clear()
            logger.debug("Cleaned up all working arrays")
    
    def get_active_count(self) -> int:
        """Get number of active arrays."""
        with self._lock:
            return len(self.active_arrays)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            total_bytes = sum(array.nbytes for array in self.active_arrays)
            
            stats = {
                'active_arrays': len(self.active_arrays),
                'total_memory_bytes': total_bytes,
                'total_memory_mb': total_bytes / (1024 * 1024)
            }
            
            if self.memory_pool:
                stats['pool_stats'] = self.memory_pool.get_statistics()
            
            return stats


class MemoryProfiler:
    """
    Memory profiler for tracking memory usage patterns.
    
    Helps identify memory bottlenecks and optimization opportunities.
    """
    
    def __init__(self, enable_detailed_tracking: bool = False):
        """
        Initialize memory profiler.
        
        Args:
            enable_detailed_tracking: Enable detailed memory tracking
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.snapshots: List[Dict[str, Any]] = []
        self.peak_memory = 0.0
        self.allocation_history: List[Dict[str, Any]] = []
        
    def take_snapshot(self, label: str) -> Dict[str, Any]:
        """
        Take memory usage snapshot.
        
        Args:
            label: Label for the snapshot
            
        Returns:
            Memory usage snapshot
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'label': label,
                'timestamp': self._get_time(),
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
            }
            
            # Track peak memory
            self.peak_memory = max(self.peak_memory, snapshot['rss_mb'])
            snapshot['peak_mb'] = self.peak_memory
            
            if self.enable_detailed_tracking:
                # Add garbage collection stats
                gc_stats = {
                    'collections': [gc.get_count()],
                    'total_objects': len(gc.get_objects())
                }
                snapshot['gc_stats'] = gc_stats
            
            self.snapshots.append(snapshot)
            return snapshot
            
        except ImportError:
            logger.warning("psutil not available for memory profiling")
            return {'label': label, 'error': 'psutil_unavailable'}
    
    def record_allocation(
        self,
        operation: str,
        size_bytes: int,
        allocation_type: str = "unknown"
    ) -> None:
        """
        Record memory allocation.
        
        Args:
            operation: Operation name
            size_bytes: Size of allocation in bytes
            allocation_type: Type of allocation
        """
        if self.enable_detailed_tracking:
            record = {
                'timestamp': self._get_time(),
                'operation': operation,
                'size_bytes': size_bytes,
                'size_mb': size_bytes / (1024 * 1024),
                'type': allocation_type
            }
            
            self.allocation_history.append(record)
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Get memory usage trend analysis."""
        if len(self.snapshots) < 2:
            return {'error': 'insufficient_snapshots'}
        
        # Calculate growth rate
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        time_diff = last_snapshot['timestamp'] - first_snapshot['timestamp']
        memory_diff = last_snapshot['rss_mb'] - first_snapshot['rss_mb']
        
        growth_rate = memory_diff / max(time_diff, 1e-6)  # MB per second
        
        # Find peak memory point
        peak_snapshot = max(self.snapshots, key=lambda s: s.get('rss_mb', 0))
        
        return {
            'total_snapshots': len(self.snapshots),
            'memory_growth_mb': memory_diff,
            'growth_rate_mb_per_sec': growth_rate,
            'peak_memory_mb': peak_snapshot.get('rss_mb', 0),
            'peak_label': peak_snapshot.get('label', 'unknown'),
            'current_memory_mb': last_snapshot.get('rss_mb', 0)
        }
    
    def generate_report(self) -> str:
        """Generate memory profiling report."""
        if not self.snapshots:
            return "No memory snapshots available"
        
        report = ["Memory Profiling Report", "=" * 50]
        
        # Summary statistics
        trend = self.get_memory_trend()
        if 'error' not in trend:
            report.append(f"Peak Memory: {trend['peak_memory_mb']:.2f} MB")
            report.append(f"Memory Growth: {trend['memory_growth_mb']:.2f} MB")
            report.append(f"Growth Rate: {trend['growth_rate_mb_per_sec']:.4f} MB/s")
            report.append("")
        
        # Snapshot details
        report.append("Memory Snapshots:")
        report.append("-" * 30)
        
        for snapshot in self.snapshots:
            if 'rss_mb' in snapshot:
                report.append(f"{snapshot['label']:20s}: {snapshot['rss_mb']:8.2f} MB")
            else:
                report.append(f"{snapshot['label']:20s}: {snapshot.get('error', 'unknown')}")
        
        # Allocation history summary
        if self.allocation_history:
            report.append("")
            report.append("Top Allocations:")
            report.append("-" * 20)
            
            # Group by operation
            op_totals = {}
            for alloc in self.allocation_history:
                op = alloc['operation']
                op_totals[op] = op_totals.get(op, 0) + alloc['size_mb']
            
            # Sort by total size
            for op, total_mb in sorted(op_totals.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"{op:20s}: {total_mb:8.2f} MB")
        
        return "\n".join(report)
    
    def _get_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()