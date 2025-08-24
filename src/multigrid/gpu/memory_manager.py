"""GPU memory management for multigrid solvers."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import threading
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryBlock:
    """GPU memory block in the pool."""
    array: 'cp.ndarray'
    in_use: bool = False
    last_used: float = 0.0
    allocation_id: int = 0
    device_id: int = 0
    
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


class GPUMemoryPool:
    """
    GPU memory pool for efficient allocation and reuse.
    
    Manages GPU memory blocks to reduce allocation overhead
    and improve performance in multigrid operations.
    """
    
    def __init__(
        self,
        max_pool_size_mb: float = 2048.0,
        device_id: int = 0,
        enable_statistics: bool = True
    ):
        """
        Initialize GPU memory pool.
        
        Args:
            max_pool_size_mb: Maximum pool size in megabytes
            device_id: GPU device ID
            enable_statistics: Enable memory usage statistics
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU memory pool")
        
        self.max_pool_size_bytes = int(max_pool_size_mb * 1024 * 1024)
        self.device_id = device_id
        self.enable_statistics = enable_statistics
        
        # Set device context
        with cp.cuda.Device(device_id):
            # Pool storage: {(shape, dtype): [GPUMemoryBlock, ...]}
            self.pools: Dict[Tuple[Tuple[int, ...], np.dtype], List[GPUMemoryBlock]] = {}
        
        # Statistics
        self.stats = {
            'total_allocated_bytes': 0,
            'total_allocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'pool_evictions': 0,
            'allocation_id_counter': 0,
            'device_transfers': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"GPU memory pool initialized: device={device_id}, "
                   f"max_size={max_pool_size_mb}MB")
    
    def allocate(
        self,
        shape: Union[Tuple[int, ...], int],
        dtype: np.dtype = np.float64,
        zero_fill: bool = True
    ) -> 'cp.ndarray':
        """
        Allocate GPU array from memory pool.
        
        Args:
            shape: Array shape
            dtype: Array data type
            zero_fill: Whether to zero-fill the array
            
        Returns:
            Allocated CuPy array
        """
        if isinstance(shape, int):
            shape = (shape,)
        
        pool_key = (shape, dtype)
        
        with cp.cuda.Device(self.device_id):
            with self._lock:
                # Check if we have available blocks
                if pool_key in self.pools:
                    pool = self.pools[pool_key]
                    
                    # Find available block
                    for block in pool:
                        if not block.in_use:
                            block.in_use = True
                            block.last_used = time.time()
                            
                            if zero_fill:
                                block.array.fill(0)
                            
                            if self.enable_statistics:
                                self.stats['cache_hits'] += 1
                            
                            logger.debug(f"Allocated from GPU pool: {shape}, {dtype}")
                            return block.array
                
                # No available block, allocate new
                array = self._allocate_new_gpu_block(shape, dtype)
                
                if zero_fill:
                    array.fill(0)
                
                if self.enable_statistics:
                    self.stats['cache_misses'] += 1
                    self.stats['total_allocations'] += 1
                    self.stats['allocation_id_counter'] += 1
                
                return array
    
    def deallocate(self, array: 'cp.ndarray') -> None:
        """
        Deallocate GPU array back to memory pool.
        
        Args:
            array: CuPy array to deallocate
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
                    if cp.shares_memory(array, block.array):
                        block.in_use = False
                        block.last_used = time.time()
                        logger.debug(f"Deallocated to GPU pool: {shape}, {dtype}")
                        return
            
            # If not found in pool, ignore (was allocated outside pool)
            logger.debug(f"GPU array not found in pool, ignoring: {shape}, {dtype}")
    
    def _allocate_new_gpu_block(self, shape: Tuple[int, ...], dtype: np.dtype) -> 'cp.ndarray':
        """Allocate new GPU memory block."""
        # Check if we need to evict blocks
        required_bytes = np.prod(shape) * dtype.itemsize
        self._ensure_gpu_space_available(required_bytes)
        
        # Create GPU array
        array = cp.zeros(shape, dtype=dtype)
        
        # Create memory block
        block = GPUMemoryBlock(
            array=array,
            in_use=True,
            last_used=time.time(),
            allocation_id=self.stats['allocation_id_counter'],
            device_id=self.device_id
        )
        
        # Add to pool
        pool_key = (shape, dtype)
        if pool_key not in self.pools:
            self.pools[pool_key] = []
        
        self.pools[pool_key].append(block)
        
        if self.enable_statistics:
            self.stats['total_allocated_bytes'] += block.size_bytes
        
        logger.debug(f"Allocated new GPU block: {shape}, {dtype}, {block.size_bytes} bytes")
        return array
    
    def _ensure_gpu_space_available(self, required_bytes: int) -> None:
        """Ensure sufficient GPU space is available."""
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
                
                logger.debug(f"Evicted GPU block: {block.shape}, {block.dtype}")
        
        # Clean up empty pools
        empty_pools = [k for k, v in self.pools.items() if not v]
        for k in empty_pools:
            del self.pools[k]
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        with cp.cuda.Device(self.device_id):
            mempool = cp.get_default_memory_pool()
            total_bytes = mempool.total_bytes()
            used_bytes = mempool.used_bytes()
            
            return {
                'device_id': self.device_id,
                'total_gpu_memory': total_bytes,
                'used_gpu_memory': used_bytes,
                'free_gpu_memory': total_bytes - used_bytes,
                'pool_stats': self.get_statistics()
            }
    
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
        """Clear all GPU memory pools."""
        with self._lock:
            self.pools.clear()
            
            if self.enable_statistics:
                self.stats['total_allocated_bytes'] = 0
                self.stats['pool_evictions'] += 1
            
            # Clear CuPy memory pool
            with cp.cuda.Device(self.device_id):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            
            logger.info("GPU memory pool cleared")


class GPUMemoryManager:
    """
    High-level GPU memory manager for multigrid operations.
    
    Provides efficient host-device transfers, pinned memory,
    and automatic memory management.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        max_pool_size_mb: float = 2048.0,
        enable_pinned_memory: bool = True,
        num_streams: int = 4
    ):
        """
        Initialize GPU memory manager.
        
        Args:
            device_id: GPU device ID
            max_pool_size_mb: Maximum memory pool size in MB
            enable_pinned_memory: Enable pinned memory for transfers
            num_streams: Number of CUDA streams for async operations
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU memory manager")
        
        self.device_id = device_id
        self.enable_pinned_memory = enable_pinned_memory
        
        # Set device context
        with cp.cuda.Device(device_id):
            # Initialize memory pool
            self.memory_pool = GPUMemoryPool(max_pool_size_mb, device_id)
            
            # Create CUDA streams for asynchronous operations
            self.streams = [cp.cuda.Stream() for _ in range(num_streams)]
            self.current_stream_idx = 0
            
            # Pinned memory allocation tracking
            self.pinned_arrays: List[cp.ndarray] = []
        
        logger.info(f"GPU memory manager initialized: device={device_id}, "
                   f"streams={num_streams}, pinned_memory={enable_pinned_memory}")
    
    def allocate_gpu_array(
        self,
        shape: Union[Tuple[int, ...], int],
        dtype: np.dtype = np.float64,
        zero_fill: bool = True
    ) -> 'cp.ndarray':
        """Allocate GPU array using memory pool."""
        return self.memory_pool.allocate(shape, dtype, zero_fill)
    
    def allocate_like_gpu(self, reference: np.ndarray, zero_fill: bool = True) -> 'cp.ndarray':
        """Allocate GPU array with same shape and dtype as reference."""
        return self.allocate_gpu_array(reference.shape, reference.dtype, zero_fill)
    
    def to_gpu(self, cpu_array: np.ndarray, stream_idx: Optional[int] = None) -> 'cp.ndarray':
        """
        Transfer array from CPU to GPU.
        
        Args:
            cpu_array: CPU numpy array
            stream_idx: Optional stream index for async transfer
            
        Returns:
            GPU array
        """
        with cp.cuda.Device(self.device_id):
            if stream_idx is not None and 0 <= stream_idx < len(self.streams):
                stream = self.streams[stream_idx]
                with stream:
                    gpu_array = cp.asarray(cpu_array)
                    self.memory_pool.stats['device_transfers'] += 1
                    return gpu_array
            else:
                gpu_array = cp.asarray(cpu_array)
                self.memory_pool.stats['device_transfers'] += 1
                return gpu_array
    
    def to_cpu(self, gpu_array: 'cp.ndarray', stream_idx: Optional[int] = None) -> np.ndarray:
        """
        Transfer array from GPU to CPU.
        
        Args:
            gpu_array: GPU CuPy array
            stream_idx: Optional stream index for async transfer
            
        Returns:
            CPU array
        """
        with cp.cuda.Device(self.device_id):
            if stream_idx is not None and 0 <= stream_idx < len(self.streams):
                stream = self.streams[stream_idx]
                with stream:
                    cpu_array = cp.asnumpy(gpu_array)
                    self.memory_pool.stats['device_transfers'] += 1
                    return cpu_array
            else:
                cpu_array = cp.asnumpy(gpu_array)
                self.memory_pool.stats['device_transfers'] += 1
                return cpu_array
    
    def allocate_pinned_memory(
        self,
        shape: Union[Tuple[int, ...], int],
        dtype: np.dtype = np.float64
    ) -> np.ndarray:
        """
        Allocate pinned (page-locked) host memory for faster transfers.
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            Pinned memory array
        """
        if not self.enable_pinned_memory:
            return np.zeros(shape, dtype=dtype)
        
        with cp.cuda.Device(self.device_id):
            # Allocate pinned memory using CuPy
            pinned_array = cp.cuda.alloc_pinned_memory(
                np.prod(shape) * dtype.itemsize
            )
            
            # Create numpy array view
            numpy_array = np.frombuffer(
                pinned_array, dtype=dtype
            ).reshape(shape)
            
            self.pinned_arrays.append(pinned_array)
            
            logger.debug(f"Allocated pinned memory: {shape}, {dtype}")
            return numpy_array
    
    def get_next_stream(self) -> 'cp.cuda.Stream':
        """Get next available CUDA stream for async operations."""
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)
        return stream
    
    def synchronize_all_streams(self) -> None:
        """Synchronize all CUDA streams."""
        with cp.cuda.Device(self.device_id):
            for stream in self.streams:
                stream.synchronize()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        gpu_info = self.memory_pool.get_gpu_memory_info()
        
        return {
            'gpu_memory_info': gpu_info,
            'pinned_memory_arrays': len(self.pinned_arrays),
            'cuda_streams': len(self.streams),
            'device_id': self.device_id,
            'enable_pinned_memory': self.enable_pinned_memory
        }
    
    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        # Clear memory pool
        self.memory_pool.clear()
        
        # Free pinned memory
        for pinned_array in self.pinned_arrays:
            try:
                cp.cuda.MemoryPointer(pinned_array, 0).free()
            except Exception as e:
                logger.warning(f"Failed to free pinned memory: {e}")
        
        self.pinned_arrays.clear()
        
        # Synchronize and cleanup streams
        self.synchronize_all_streams()
        
        logger.info("GPU memory manager cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error in GPU memory manager cleanup: {e}")


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and capabilities."""
    info = {
        'cupy_available': CUPY_AVAILABLE,
        'gpu_count': 0,
        'devices': [],
        'error': None
    }
    
    if not CUPY_AVAILABLE:
        info['error'] = "CuPy not installed"
        return info
    
    try:
        info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
        
        for device_id in range(info['gpu_count']):
            with cp.cuda.Device(device_id):
                device_props = cp.cuda.runtime.getDeviceProperties(device_id)
                meminfo = cp.cuda.runtime.memGetInfo()
                
                device_info = {
                    'device_id': device_id,
                    'name': device_props['name'].decode(),
                    'compute_capability': f"{device_props['major']}.{device_props['minor']}",
                    'total_memory_mb': meminfo[1] / (1024 * 1024),
                    'free_memory_mb': meminfo[0] / (1024 * 1024),
                    'multiprocessors': device_props['multiProcessorCount'],
                    'max_threads_per_block': device_props['maxThreadsPerBlock'],
                    'max_shared_memory_per_block': device_props['sharedMemPerBlock']
                }
                
                info['devices'].append(device_info)
    
    except Exception as e:
        info['error'] = str(e)
    
    return info