"""GPU mixed-precision management for multigrid solvers."""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from enum import Enum

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..core.precision import PrecisionLevel

logger = logging.getLogger(__name__)


class GPUPrecisionLevel(Enum):
    """GPU-specific precision levels including Tensor Core support."""
    HALF = "half"           # float16 - fastest, lowest precision
    SINGLE = "single"       # float32 - balanced
    DOUBLE = "double"       # float64 - highest precision
    MIXED_TC = "mixed_tc"   # Mixed precision with Tensor Core optimization


class GPUPrecisionManager:
    """
    GPU mixed-precision manager with Tensor Core optimization.
    
    Manages precision switching on GPU for optimal performance
    while maintaining numerical stability.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        enable_tensor_cores: bool = True,
        adaptive: bool = True,
        default_precision: str = "mixed_tc"
    ):
        """
        Initialize GPU precision manager.
        
        Args:
            device_id: GPU device ID
            enable_tensor_cores: Enable Tensor Core optimizations
            adaptive: Enable adaptive precision switching
            default_precision: Default precision level
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU precision manager")
        
        self.device_id = device_id
        self.enable_tensor_cores = enable_tensor_cores
        self.adaptive = adaptive
        
        # Set current precision
        if default_precision in [level.value for level in GPUPrecisionLevel]:
            self.current_precision = GPUPrecisionLevel(default_precision)
        else:
            self.current_precision = GPUPrecisionLevel.MIXED_TC
        
        # Precision switching thresholds
        self.thresholds = {
            'downgrade_residual': 1e-2,
            'upgrade_residual': 1e-8,
            'memory_pressure_mb': 4096  # 4GB threshold
        }
        
        # Track precision history and performance
        self.precision_history: List[Dict[str, Any]] = []
        self.performance_stats = {
            'fp16_operations': 0,
            'fp32_operations': 0,
            'fp64_operations': 0,
            'tensor_core_operations': 0,
            'precision_switches': 0
        }
        
        # Check Tensor Core availability
        self.tensor_core_available = self._check_tensor_core_support()
        
        logger.info(f"GPU precision manager initialized: device={device_id}, "
                   f"tensor_cores={self.tensor_core_available}, "
                   f"default_precision={default_precision}")
    
    def _check_tensor_core_support(self) -> bool:
        """Check if GPU supports Tensor Cores."""
        if not self.enable_tensor_cores:
            return False
        
        try:
            with cp.cuda.Device(self.device_id):
                # Get device properties
                props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                major = props['major']
                minor = props['minor']
                
                # Tensor Cores available on Volta (7.0+), Turing (7.5+), Ampere (8.0+)
                compute_capability = major + minor * 0.1
                tensor_core_support = compute_capability >= 7.0
                
                if tensor_core_support:
                    logger.debug(f"Tensor Core support detected: compute capability {major}.{minor}")
                else:
                    logger.debug(f"No Tensor Core support: compute capability {major}.{minor}")
                
                return tensor_core_support
        
        except Exception as e:
            logger.warning(f"Failed to check Tensor Core support: {e}")
            return False
    
    def get_optimal_dtype(self, operation_type: str, grid_level: int = 0) -> np.dtype:
        """
        Get optimal data type for GPU operation.
        
        Args:
            operation_type: Type of operation ('smoothing', 'transfer', 'residual')
            grid_level: Grid level (0 = finest)
            
        Returns:
            Optimal numpy data type
        """
        if self.current_precision == GPUPrecisionLevel.HALF:
            return np.float16
        elif self.current_precision == GPUPrecisionLevel.SINGLE:
            return np.float32
        elif self.current_precision == GPUPrecisionLevel.DOUBLE:
            return np.float64
        elif self.current_precision == GPUPrecisionLevel.MIXED_TC:
            # Mixed precision strategy with Tensor Core optimization
            if operation_type in ['smoothing', 'transfer'] and grid_level > 0:
                # Use lower precision for coarser levels and iterative operations
                return np.float16 if self.tensor_core_available else np.float32
            else:
                # Use higher precision for fine level and residual computation
                return np.float32
        else:
            return np.float32
    
    def convert_to_optimal_precision(
        self,
        array: 'cp.ndarray',
        operation_type: str,
        grid_level: int = 0
    ) -> 'cp.ndarray':
        """
        Convert array to optimal precision for GPU operation.
        
        Args:
            array: Input GPU array
            operation_type: Type of operation
            grid_level: Grid level
            
        Returns:
            Array in optimal precision
        """
        optimal_dtype = self.get_optimal_dtype(operation_type, grid_level)
        
        if array.dtype != optimal_dtype:
            # Track precision conversion
            if optimal_dtype == np.float16:
                self.performance_stats['fp16_operations'] += 1
            elif optimal_dtype == np.float32:
                self.performance_stats['fp32_operations'] += 1
            else:
                self.performance_stats['fp64_operations'] += 1
            
            return array.astype(optimal_dtype)
        
        return array
    
    def apply_tensor_core_optimization(
        self,
        array1: 'cp.ndarray',
        array2: 'cp.ndarray',
        operation: str = "multiply"
    ) -> 'cp.ndarray':
        """
        Apply Tensor Core optimizations for matrix operations.
        
        Args:
            array1: First input array
            array2: Second input array
            operation: Operation type ('multiply', 'add')
            
        Returns:
            Result with Tensor Core optimization
        """
        if not self.tensor_core_available:
            # Fall back to standard operations
            if operation == "multiply":
                return array1 * array2
            elif operation == "add":
                return array1 + array2
            else:
                return array1
        
        # Convert to half precision for Tensor Core operations if beneficial
        if array1.size > 1024 and array1.dtype != np.float16:
            array1_tc = array1.astype(np.float16)
            array2_tc = array2.astype(np.float16)
            
            # Perform operation in half precision
            if operation == "multiply":
                result_tc = array1_tc * array2_tc
            elif operation == "add":
                result_tc = array1_tc + array2_tc
            else:
                result_tc = array1_tc
            
            # Convert back to original precision if needed
            result = result_tc.astype(array1.dtype)
            
            self.performance_stats['tensor_core_operations'] += 1
            return result
        else:
            # Use standard operations for small arrays
            if operation == "multiply":
                return array1 * array2
            elif operation == "add":
                return array1 + array2
            else:
                return array1
    
    def update_precision_adaptive(
        self,
        residual_norm: float,
        grid_shapes: Optional[List[tuple]] = None,
        memory_usage_mb: Optional[float] = None
    ) -> bool:
        """
        Update precision based on adaptive strategy.
        
        Args:
            residual_norm: Current residual norm
            grid_shapes: Grid shapes for memory estimation
            memory_usage_mb: Current GPU memory usage
            
        Returns:
            True if precision was changed
        """
        if not self.adaptive:
            return False
        
        old_precision = self.current_precision
        precision_changed = False
        
        # Memory-based precision adjustment
        if memory_usage_mb and memory_usage_mb > self.thresholds['memory_pressure_mb']:
            if self.current_precision in [GPUPrecisionLevel.DOUBLE, GPUPrecisionLevel.SINGLE]:
                if self.tensor_core_available:
                    self.current_precision = GPUPrecisionLevel.MIXED_TC
                else:
                    self.current_precision = GPUPrecisionLevel.SINGLE
                precision_changed = True
                logger.info(f"Reduced precision due to memory pressure: {memory_usage_mb:.1f} MB")
        
        # Convergence-based precision adjustment
        if residual_norm > self.thresholds['downgrade_residual']:
            # Can use lower precision when residual is large
            if self.current_precision == GPUPrecisionLevel.DOUBLE:
                self.current_precision = GPUPrecisionLevel.MIXED_TC if self.tensor_core_available else GPUPrecisionLevel.SINGLE
                precision_changed = True
        elif residual_norm < self.thresholds['upgrade_residual']:
            # Need higher precision for final convergence
            if self.current_precision in [GPUPrecisionLevel.HALF, GPUPrecisionLevel.MIXED_TC]:
                self.current_precision = GPUPrecisionLevel.SINGLE
                precision_changed = True
        
        # Record precision change
        if precision_changed:
            self.performance_stats['precision_switches'] += 1
            
            self.precision_history.append({
                'timestamp': cp.cuda.Event().record(),
                'old_precision': old_precision.value,
                'new_precision': self.current_precision.value,
                'residual_norm': residual_norm,
                'memory_usage_mb': memory_usage_mb,
                'reason': 'adaptive_switch'
            })
            
            logger.debug(f"Precision switched: {old_precision.value} -> {self.current_precision.value}")
        
        return precision_changed
    
    def get_precision_recommendations(self, grid_hierarchy_info: Dict[str, Any]) -> Dict[int, str]:
        """
        Get precision recommendations for each grid level.
        
        Args:
            grid_hierarchy_info: Information about grid hierarchy
            
        Returns:
            Dict mapping grid level to recommended precision
        """
        recommendations = {}
        num_levels = grid_hierarchy_info.get('num_levels', 1)
        total_points = grid_hierarchy_info.get('total_points', 0)
        
        for level in range(num_levels):
            # Coarser levels can use lower precision
            if level == 0:
                # Finest level - use current precision
                recommendations[level] = self.current_precision.value
            elif level < num_levels // 2:
                # Medium levels - balanced precision
                if self.tensor_core_available:
                    recommendations[level] = GPUPrecisionLevel.MIXED_TC.value
                else:
                    recommendations[level] = GPUPrecisionLevel.SINGLE.value
            else:
                # Coarsest levels - can use lowest precision
                if total_points < 1000000:  # < 1M points
                    recommendations[level] = GPUPrecisionLevel.HALF.value
                else:
                    recommendations[level] = GPUPrecisionLevel.SINGLE.value
        
        return recommendations
    
    def create_mixed_precision_arrays(
        self,
        base_array: 'cp.ndarray',
        operation_types: List[str],
        grid_levels: List[int]
    ) -> Dict[str, 'cp.ndarray']:
        """
        Create multiple precision versions of array for different operations.
        
        Args:
            base_array: Base array to convert
            operation_types: List of operation types
            grid_levels: List of grid levels
            
        Returns:
            Dict of arrays in different precisions
        """
        precision_arrays = {}
        
        for op_type, level in zip(operation_types, grid_levels):
            optimal_dtype = self.get_optimal_dtype(op_type, level)
            key = f"{op_type}_level_{level}"
            
            if optimal_dtype != base_array.dtype:
                precision_arrays[key] = base_array.astype(optimal_dtype)
            else:
                precision_arrays[key] = base_array
        
        return precision_arrays
    
    def estimate_speedup(self, operation_type: str, array_size: int) -> float:
        """
        Estimate speedup from mixed precision for given operation.
        
        Args:
            operation_type: Type of operation
            array_size: Size of arrays involved
            
        Returns:
            Estimated speedup factor
        """
        if not self.tensor_core_available:
            # Without Tensor Cores, speedup is mainly from memory bandwidth
            if self.current_precision == GPUPrecisionLevel.HALF:
                return 1.8  # Approximate 2x memory bandwidth improvement
            elif self.current_precision == GPUPrecisionLevel.MIXED_TC:
                return 1.4  # Mixed precision benefit
            else:
                return 1.0
        else:
            # With Tensor Cores, can get significant compute speedup
            if operation_type in ['smoothing', 'transfer'] and array_size > 10000:
                if self.current_precision == GPUPrecisionLevel.HALF:
                    return 3.5  # Tensor Core + memory bandwidth
                elif self.current_precision == GPUPrecisionLevel.MIXED_TC:
                    return 2.2  # Balanced mixed precision
                else:
                    return 1.0
            else:
                return 1.2  # Small benefit for other operations
    
    def get_precision_statistics(self) -> Dict[str, Any]:
        """Get precision usage statistics."""
        total_ops = sum([
            self.performance_stats['fp16_operations'],
            self.performance_stats['fp32_operations'], 
            self.performance_stats['fp64_operations']
        ])
        
        return {
            'current_precision': self.current_precision.value,
            'tensor_core_available': self.tensor_core_available,
            'performance_stats': self.performance_stats.copy(),
            'precision_distribution': {
                'fp16_percent': (self.performance_stats['fp16_operations'] / max(total_ops, 1)) * 100,
                'fp32_percent': (self.performance_stats['fp32_operations'] / max(total_ops, 1)) * 100,
                'fp64_percent': (self.performance_stats['fp64_operations'] / max(total_ops, 1)) * 100
            },
            'precision_switches': self.performance_stats['precision_switches'],
            'tensor_core_utilization': self.performance_stats['tensor_core_operations'],
            'adaptive_enabled': self.adaptive
        }
    
    def reset_statistics(self) -> None:
        """Reset precision statistics."""
        self.performance_stats = {
            'fp16_operations': 0,
            'fp32_operations': 0,
            'fp64_operations': 0,
            'tensor_core_operations': 0,
            'precision_switches': 0
        }
        self.precision_history.clear()
        logger.debug("GPU precision statistics reset")


class GPUPrecisionOptimizer:
    """
    Advanced GPU precision optimizer with automatic tuning.
    
    Automatically determines optimal precision strategies
    based on problem characteristics and hardware capabilities.
    """
    
    def __init__(self, precision_manager: GPUPrecisionManager):
        """Initialize precision optimizer."""
        self.precision_manager = precision_manager
        self.optimization_cache: Dict[str, Dict[str, Any]] = {}
        
    def optimize_for_problem(
        self,
        grid_hierarchy: List[tuple],
        operator_type: str,
        target_accuracy: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Optimize precision strategy for specific problem.
        
        Args:
            grid_hierarchy: List of grid shapes for each level
            operator_type: Type of PDE operator
            target_accuracy: Target solution accuracy
            
        Returns:
            Optimization strategy
        """
        problem_key = f"{len(grid_hierarchy)}_{operator_type}_{target_accuracy}"
        
        if problem_key in self.optimization_cache:
            return self.optimization_cache[problem_key]
        
        # Analyze problem characteristics
        total_points = sum(np.prod(shape) for shape in grid_hierarchy)
        max_grid_size = max(max(shape) for shape in grid_hierarchy)
        
        strategy = {
            'precision_per_level': {},
            'use_tensor_cores': self.precision_manager.tensor_core_available,
            'memory_optimization': total_points > 1000000,
            'compute_optimization': max_grid_size > 512
        }
        
        # Determine precision for each level
        for level, shape in enumerate(grid_hierarchy):
            grid_points = np.prod(shape)
            
            if level == 0:
                # Finest level - balance accuracy and performance
                if target_accuracy < 1e-10:
                    strategy['precision_per_level'][level] = 'double'
                elif target_accuracy < 1e-6:
                    strategy['precision_per_level'][level] = 'single'
                else:
                    strategy['precision_per_level'][level] = 'mixed_tc'
            else:
                # Coarser levels - can use lower precision
                if grid_points < 1000:
                    strategy['precision_per_level'][level] = 'half'
                elif grid_points < 100000:
                    strategy['precision_per_level'][level] = 'mixed_tc'
                else:
                    strategy['precision_per_level'][level] = 'single'
        
        # Cache the strategy
        self.optimization_cache[problem_key] = strategy
        
        logger.debug(f"Optimized precision strategy for problem: {problem_key}")
        return strategy
    
    def benchmark_precision_performance(
        self,
        test_array_shape: Tuple[int, int],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark performance of different precision levels.
        
        Args:
            test_array_shape: Shape of test arrays
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance results for each precision
        """
        results = {}
        
        # Create test data
        test_data = cp.random.rand(*test_array_shape, dtype=cp.float32)
        
        for precision in GPUPrecisionLevel:
            if precision == GPUPrecisionLevel.MIXED_TC and not self.precision_manager.tensor_core_available:
                continue
            
            # Convert to target precision
            if precision == GPUPrecisionLevel.HALF:
                test_array = test_data.astype(cp.float16)
            elif precision == GPUPrecisionLevel.SINGLE:
                test_array = test_data.astype(cp.float32)
            elif precision == GPUPrecisionLevel.DOUBLE:
                test_array = test_data.astype(cp.float64)
            else:  # MIXED_TC
                test_array = test_data  # Use mixed operations
            
            # Benchmark simple operations
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            
            start_event.record()
            
            for _ in range(num_iterations):
                # Simulate typical multigrid operations
                temp = test_array * 2.0
                temp = temp + test_array
                temp = cp.sum(temp, axis=1, keepdims=True)
            
            end_event.record()
            end_event.synchronize()
            
            elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
            results[precision.value] = elapsed_time
        
        logger.debug(f"Precision benchmark completed: {results}")
        return results