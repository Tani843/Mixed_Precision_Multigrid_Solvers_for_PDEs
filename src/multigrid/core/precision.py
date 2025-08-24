"""Precision management for mixed-precision computations."""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Enumeration of supported precision levels."""
    SINGLE = "float32"
    DOUBLE = "float64"
    MIXED = "mixed"


class PrecisionManager:
    """
    Manages precision levels and conversions for mixed-precision multigrid.
    
    This class handles automatic precision switching based on convergence criteria,
    memory constraints, and computational efficiency requirements.
    """
    
    def __init__(
        self,
        default_precision: Union[PrecisionLevel, str] = PrecisionLevel.DOUBLE,
        adaptive: bool = True,
        convergence_threshold: float = 1e-6,
        memory_threshold_gb: float = 4.0
    ):
        """
        Initialize precision manager.
        
        Args:
            default_precision: Default precision level
            adaptive: Enable adaptive precision switching
            convergence_threshold: Convergence threshold for precision switching
            memory_threshold_gb: Memory threshold for automatic downgrade (GB)
        """
        self.default_precision = self._parse_precision(default_precision)
        self.adaptive = adaptive
        self.convergence_threshold = convergence_threshold
        self.memory_threshold_bytes = memory_threshold_gb * 1024**3
        
        # Precision hierarchy (lower precision first for early iterations)
        self.precision_hierarchy = [
            PrecisionLevel.SINGLE,
            PrecisionLevel.DOUBLE
        ]
        
        # Current precision state
        self.current_precision = self.default_precision
        self.precision_history = [self.current_precision]
        
        # Performance tracking
        self.precision_stats = {
            PrecisionLevel.SINGLE: {"operations": 0, "time": 0.0},
            PrecisionLevel.DOUBLE: {"operations": 0, "time": 0.0}
        }
        
        logger.info(f"Initialized PrecisionManager: default={self.default_precision.value}, "
                   f"adaptive={adaptive}")
    
    def _parse_precision(self, precision: Union[PrecisionLevel, str]) -> PrecisionLevel:
        """Parse precision level from string or enum."""
        if isinstance(precision, str):
            precision_map = {
                "single": PrecisionLevel.SINGLE,
                "double": PrecisionLevel.DOUBLE,
                "mixed": PrecisionLevel.MIXED,
                "float32": PrecisionLevel.SINGLE,
                "float64": PrecisionLevel.DOUBLE
            }
            if precision.lower() in precision_map:
                return precision_map[precision.lower()]
            else:
                raise ValueError(f"Unknown precision level: {precision}")
        elif isinstance(precision, PrecisionLevel):
            return precision
        else:
            raise TypeError(f"Precision must be PrecisionLevel or str, got {type(precision)}")
    
    def get_dtype(self, precision: Optional[PrecisionLevel] = None) -> np.dtype:
        """
        Get numpy dtype for given precision level.
        
        Args:
            precision: Precision level (default: current precision)
            
        Returns:
            Corresponding numpy dtype
        """
        if precision is None:
            precision = self.current_precision
        
        dtype_map = {
            PrecisionLevel.SINGLE: np.float32,
            PrecisionLevel.DOUBLE: np.float64,
            PrecisionLevel.MIXED: np.float64  # Use double as default for mixed
        }
        
        return dtype_map[precision]
    
    def convert_array(
        self, 
        array: np.ndarray, 
        target_precision: Optional[PrecisionLevel] = None
    ) -> np.ndarray:
        """
        Convert array to target precision.
        
        Args:
            array: Input array
            target_precision: Target precision (default: current precision)
            
        Returns:
            Array converted to target precision
        """
        if target_precision is None:
            target_precision = self.current_precision
        
        target_dtype = self.get_dtype(target_precision)
        
        if array.dtype == target_dtype:
            return array
        
        converted = array.astype(target_dtype)
        
        logger.debug(f"Converted array from {array.dtype} to {target_dtype}, "
                    f"shape={array.shape}")
        
        return converted
    
    def estimate_memory_usage(self, grid_shapes: list) -> float:
        """
        Estimate memory usage for given grid shapes.
        
        Args:
            grid_shapes: List of (nx, ny) tuples for each grid level
            
        Returns:
            Estimated memory usage in bytes
        """
        total_points = sum(nx * ny for nx, ny in grid_shapes)
        dtype_size = np.dtype(self.get_dtype()).itemsize
        
        # Factor in multiple arrays per grid (solution, residual, RHS, etc.)
        arrays_per_grid = 4
        memory_bytes = total_points * dtype_size * arrays_per_grid
        
        return memory_bytes
    
    def should_downgrade_precision(
        self, 
        grid_shapes: list, 
        residual_norm: float
    ) -> bool:
        """
        Determine if precision should be downgraded.
        
        Args:
            grid_shapes: Grid shapes for memory estimation
            residual_norm: Current residual norm
            
        Returns:
            True if precision should be downgraded
        """
        if not self.adaptive:
            return False
        
        # Check memory constraints
        memory_usage = self.estimate_memory_usage(grid_shapes)
        if memory_usage > self.memory_threshold_bytes:
            logger.info(f"Memory usage {memory_usage/1024**3:.2f} GB exceeds threshold, "
                       f"suggesting precision downgrade")
            return True
        
        # Check convergence progress
        if (self.current_precision == PrecisionLevel.DOUBLE and 
            residual_norm > self.convergence_threshold * 100):
            logger.info(f"Residual norm {residual_norm:.2e} suggests early iterations, "
                       f"downgrading to single precision")
            return True
        
        return False
    
    def should_promote_precision(self, convergence_history: list, current_precision: PrecisionLevel) -> bool:
        """
        Logic for when to switch from float32 to float64
        - Monitor convergence stagnation
        - Check residual plateauing 
        - Detect numerical instability
        
        Args:
            convergence_history: List of recent residual norms
            current_precision: Current precision level
            
        Returns:
            True if precision should be promoted to higher accuracy
        """
        if not self.adaptive or current_precision == PrecisionLevel.DOUBLE:
            return False
        
        if len(convergence_history) < 5:
            return False
        
        recent_residuals = convergence_history[-5:]
        
        # Check for convergence stagnation
        improvement_ratios = []
        for i in range(1, len(recent_residuals)):
            if recent_residuals[i-1] > 0:
                ratio = recent_residuals[i] / recent_residuals[i-1]
                improvement_ratios.append(ratio)
        
        if improvement_ratios:
            avg_improvement = np.mean(improvement_ratios)
            
            # Stagnation detected if improvement ratio > 0.9 (little progress)
            if avg_improvement > 0.9:
                logger.info(f"Convergence stagnation detected (avg improvement: {avg_improvement:.3f}), "
                           f"promoting precision")
                return True
            
            # Check for plateauing - consecutive residuals very similar
            relative_changes = []
            for i in range(1, len(recent_residuals)):
                if recent_residuals[i-1] > 0:
                    rel_change = abs(recent_residuals[i] - recent_residuals[i-1]) / recent_residuals[i-1]
                    relative_changes.append(rel_change)
            
            if relative_changes and np.mean(relative_changes) < 1e-3:
                logger.info("Residual plateauing detected, promoting precision")
                return True
        
        # Check for numerical instability (increasing residuals)
        if len(recent_residuals) >= 3:
            increasing_trend = all(recent_residuals[i] >= recent_residuals[i-1] * 0.99 
                                 for i in range(1, len(recent_residuals)))
            if increasing_trend:
                logger.info("Numerical instability detected (increasing residuals), promoting precision")
                return True
        
        return False

    def should_upgrade_precision(self, residual_norm: float) -> bool:
        """
        Determine if precision should be upgraded.
        
        Args:
            residual_norm: Current residual norm
            
        Returns:
            True if precision should be upgraded
        """
        if not self.adaptive:
            return False
        
        # Upgrade when approaching convergence
        if (self.current_precision == PrecisionLevel.SINGLE and 
            residual_norm < self.convergence_threshold * 10):
            logger.info(f"Residual norm {residual_norm:.2e} approaching convergence, "
                       f"upgrading to double precision")
            return True
        
        return False
    
    def update_precision(
        self, 
        residual_norm: float, 
        grid_shapes: Optional[list] = None
    ) -> bool:
        """
        Update precision level based on current state.
        
        Args:
            residual_norm: Current residual norm
            grid_shapes: Grid shapes for memory estimation
            
        Returns:
            True if precision was changed
        """
        if not self.adaptive:
            return False
        
        old_precision = self.current_precision
        
        if grid_shapes and self.should_downgrade_precision(grid_shapes, residual_norm):
            if self.current_precision == PrecisionLevel.DOUBLE:
                self.current_precision = PrecisionLevel.SINGLE
        elif self.should_upgrade_precision(residual_norm):
            if self.current_precision == PrecisionLevel.SINGLE:
                self.current_precision = PrecisionLevel.DOUBLE
        
        if self.current_precision != old_precision:
            self.precision_history.append(self.current_precision)
            logger.info(f"Precision changed: {old_precision.value} -> {self.current_precision.value}")
            return True
        
        return False
    
    def optimal_precision_per_level(self, grid_level: int, problem_size: int) -> PrecisionLevel:
        """
        Different precisions for different grid levels
        - Fine grids: float64 for accuracy
        - Coarse grids: float32 for speed
        
        Args:
            grid_level: Current grid level (0 = finest)
            problem_size: Total number of grid points
            
        Returns:
            Optimal precision for this level
        """
        if not self.adaptive:
            return self.current_precision
        
        # Decision matrix based on grid level and problem size
        if grid_level == 0:  # Finest grid
            # Always use double precision for finest grid for accuracy
            return PrecisionLevel.DOUBLE
        
        elif grid_level <= 2:  # Fine to medium grids
            # Use double precision for accuracy on important levels
            # But consider single precision for very large problems
            if problem_size > 500000:  # Very large problem
                return PrecisionLevel.SINGLE
            else:
                return PrecisionLevel.DOUBLE
        
        else:  # Coarse grids (level >= 3)
            # Always use single precision for coarse grids - speed matters more
            return PrecisionLevel.SINGLE

    def get_precision_for_level(self, level: int, max_levels: int) -> PrecisionLevel:
        """
        Get recommended precision for multigrid level.
        
        Args:
            level: Current multigrid level (0 = finest)
            max_levels: Total number of levels
            
        Returns:
            Recommended precision for this level
        """
        if not self.adaptive or self.current_precision != PrecisionLevel.MIXED:
            return self.current_precision
        
        # Use single precision for coarse grids, double for fine grids
        coarse_threshold = max_levels // 2
        
        if level >= coarse_threshold:
            return PrecisionLevel.SINGLE
        else:
            return PrecisionLevel.DOUBLE
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get precision usage statistics.
        
        Returns:
            Dictionary with precision statistics
        """
        total_ops = sum(stats["operations"] for stats in self.precision_stats.values())
        total_time = sum(stats["time"] for stats in self.precision_stats.values())
        
        stats = {
            "current_precision": self.current_precision.value,
            "precision_history": [p.value for p in self.precision_history],
            "total_operations": total_ops,
            "total_time": total_time,
            "precision_breakdown": {
                level.value: {
                    "operations": self.precision_stats[level]["operations"],
                    "time": self.precision_stats[level]["time"],
                    "percentage": (self.precision_stats[level]["operations"] / total_ops * 100 
                                 if total_ops > 0 else 0)
                }
                for level in PrecisionLevel if level != PrecisionLevel.MIXED
            }
        }
        
        return stats
    
    def record_operation(self, precision: PrecisionLevel, time_taken: float) -> None:
        """
        Record an operation for statistics.
        
        Args:
            precision: Precision level used
            time_taken: Time taken for operation
        """
        if precision in self.precision_stats:
            self.precision_stats[precision]["operations"] += 1
            self.precision_stats[precision]["time"] += time_taken
    
    def reset_statistics(self) -> None:
        """Reset all precision statistics."""
        for level in self.precision_stats:
            self.precision_stats[level]["operations"] = 0
            self.precision_stats[level]["time"] = 0.0
        
        self.precision_history = [self.current_precision]
        
        logger.info("Reset precision statistics")
    
    def __str__(self) -> str:
        """String representation of precision manager."""
        return (f"PrecisionManager(current={self.current_precision.value}, "
                f"adaptive={self.adaptive})")
    
    def __repr__(self) -> str:
        """Detailed representation of precision manager."""
        return (f"PrecisionManager(default={self.default_precision.value}, "
                f"current={self.current_precision.value}, adaptive={self.adaptive}, "
                f"threshold={self.convergence_threshold})")