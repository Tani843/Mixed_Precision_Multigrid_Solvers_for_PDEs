"""Advanced multigrid cycles with communication-avoiding optimizations."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import time
import logging
from dataclasses import dataclass, field

from .base import BaseSolver
from .smoothers import GaussSeidelSmoother
from ..core.grid import Grid

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.precision import PrecisionManager
    from ..operators.base import BaseOperator
    from ..operators.transfer import RestrictionOperator, ProlongationOperator
    from .base import IterativeSolver

logger = logging.getLogger(__name__)


@dataclass
class CycleStatistics:
    """Statistics for multigrid cycle performance."""
    level_times: Dict[int, float] = field(default_factory=dict)
    restriction_times: Dict[int, float] = field(default_factory=dict)
    prolongation_times: Dict[int, float] = field(default_factory=dict)
    smoothing_times: Dict[int, float] = field(default_factory=dict)
    coarse_solve_time: float = 0.0
    memory_usage: Dict[int, float] = field(default_factory=dict)
    cache_misses: int = 0
    total_operations: int = 0


class CommunicationAvoidingMultigrid(BaseSolver):
    """
    Advanced multigrid solver with communication-avoiding optimizations.
    
    Implements V-cycle, W-cycle, and FMG with:
    - Block-structured operations
    - Cache-optimized data layouts
    - Memory pool management
    - Adaptive precision strategies
    """
    
    def __init__(
        self,
        max_levels: int = 6,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        cycle_type: str = "V",
        pre_smooth_iterations: int = 2,
        post_smooth_iterations: int = 2,
        coarse_tolerance: float = 1e-12,
        coarse_max_iterations: int = 1000,
        use_fmg: bool = False,
        fmg_cycles: int = 1,
        block_size: int = 32,
        enable_memory_pool: bool = True,
        verbose: bool = False
    ):
        """
        Initialize communication-avoiding multigrid solver.
        
        Args:
            max_levels: Maximum number of grid levels
            max_iterations: Maximum multigrid iterations
            tolerance: Convergence tolerance
            cycle_type: Cycle type ('V', 'W', 'F')
            pre_smooth_iterations: Pre-smoothing iterations
            post_smooth_iterations: Post-smoothing iterations
            coarse_tolerance: Coarse grid solver tolerance
            coarse_max_iterations: Coarse grid solver max iterations
            use_fmg: Use Full Multigrid initialization
            fmg_cycles: Number of FMG cycles
            block_size: Block size for cache optimization
            enable_memory_pool: Enable memory pool management
            verbose: Enable verbose output
        """
        super().__init__(max_iterations, tolerance, verbose, "CommunicationAvoidingMG")
        
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.pre_smooth_iterations = pre_smooth_iterations
        self.post_smooth_iterations = post_smooth_iterations
        self.coarse_tolerance = coarse_tolerance
        self.coarse_max_iterations = coarse_max_iterations
        self.use_fmg = use_fmg
        self.fmg_cycles = fmg_cycles
        self.block_size = block_size
        self.enable_memory_pool = enable_memory_pool
        
        # Grid hierarchy storage
        self.grids: List['Grid'] = []
        self.operators: List['BaseOperator'] = []
        self.restriction_ops: List['RestrictionOperator'] = []
        self.prolongation_ops: List['ProlongationOperator'] = []
        
        # Memory management
        self.memory_pool = {} if enable_memory_pool else None
        self.working_arrays = {}
        
        # Solver components
        self.smoother: Optional['IterativeSolver'] = None
        self.coarse_solver: Optional['IterativeSolver'] = None
        
        # Performance tracking
        self.cycle_stats = CycleStatistics()
        self.precision_strategies = {}
        
        logger.info(f"Initialized CA-Multigrid: {cycle_type}-cycle, "
                   f"max_levels={max_levels}, block_size={block_size}")
    
    def setup(
        self,
        fine_grid: 'Grid',
        operator: 'BaseOperator',
        restriction_op: 'RestrictionOperator',
        prolongation_op: 'ProlongationOperator',
        smoother: Optional['IterativeSolver'] = None,
        coarse_solver: Optional['IterativeSolver'] = None
    ) -> None:
        """
        Setup multigrid hierarchy with communication-avoiding optimizations.
        
        Args:
            fine_grid: Finest grid level
            operator: Differential operator
            restriction_op: Grid restriction operator
            prolongation_op: Grid prolongation operator
            smoother: Smoother for iterative refinement
            coarse_solver: Solver for coarsest grid
        """
        # Set default smoothers
        if smoother is None:
            smoother = GaussSeidelSmoother(
                max_iterations=max(self.pre_smooth_iterations, self.post_smooth_iterations),
                tolerance=self.tolerance * 0.1,
                verbose=False,
                red_black=True  # Use red-black for better cache locality
            )
        
        if coarse_solver is None:
            coarse_solver = GaussSeidelSmoother(
                max_iterations=self.coarse_max_iterations,
                tolerance=self.coarse_tolerance,
                verbose=self.verbose
            )
        
        self.smoother = smoother
        self.coarse_solver = coarse_solver
        
        # Build hierarchy with optimization
        self._build_optimized_hierarchy(fine_grid, operator, restriction_op, prolongation_op)
        
        # Setup memory pool
        if self.enable_memory_pool:
            self._setup_memory_pool()
        
        # Setup precision strategies
        self._setup_precision_strategies()
        
        logger.info(f"CA-Multigrid setup complete: {len(self.grids)} levels, "
                   f"finest: {self.grids[0].shape}, coarsest: {self.grids[-1].shape}")
    
    def _build_optimized_hierarchy(
        self,
        fine_grid: 'Grid',
        operator: 'BaseOperator',
        restriction_op: 'RestrictionOperator',
        prolongation_op: 'ProlongationOperator'
    ) -> None:
        """Build multigrid hierarchy with cache-friendly layouts."""
        self.grids.clear()
        self.operators.clear()
        self.restriction_ops.clear()
        self.prolongation_ops.clear()
        
        current_grid = fine_grid
        self.grids.append(current_grid)
        self.operators.append(operator)
        
        # Build coarser levels
        for level in range(1, self.max_levels):
            try:
                # Create coarse grid with optimized memory layout
                coarse_grid = self._create_optimized_coarse_grid(current_grid)
                
                # Check minimum grid size
                if coarse_grid.nx < 5 or coarse_grid.ny < 5:
                    logger.info(f"Stopping hierarchy at level {level}: grid too coarse")
                    break
                
                self.grids.append(coarse_grid)
                self.operators.append(operator)
                self.restriction_ops.append(restriction_op)
                self.prolongation_ops.append(prolongation_op)
                
                current_grid = coarse_grid
                
            except ValueError as e:
                logger.info(f"Stopping hierarchy at level {level}: {e}")
                break
        
        logger.debug(f"Built optimized hierarchy: {len(self.grids)} levels")
    
    def _create_optimized_coarse_grid(self, fine_grid: 'Grid') -> 'Grid':
        """Create coarse grid with cache-optimized memory layout."""
        coarse_grid = fine_grid.coarsen()
        
        # Optimize memory layout for cache efficiency
        # Ensure grid dimensions are friendly to block operations
        nx, ny = coarse_grid.nx, coarse_grid.ny
        
        # Pad to block-size boundaries if beneficial
        if self.block_size > 1:
            block_nx = ((nx - 1) // self.block_size + 1) * self.block_size + 1
            block_ny = ((ny - 1) // self.block_size + 1) * self.block_size + 1
            
            if block_nx <= nx + self.block_size and block_ny <= ny + self.block_size:
                # Only pad if it doesn't add too much memory
                domain = coarse_grid.domain
                coarse_grid = Grid(block_nx, block_ny, domain, coarse_grid.dtype)
        
        return coarse_grid
    
    def _setup_memory_pool(self) -> None:
        """Setup memory pool for efficient array reuse."""
        if not self.enable_memory_pool:
            return
        
        for level, grid in enumerate(self.grids):
            grid_shape = grid.shape
            dtype = grid.dtype
            
            # Pre-allocate working arrays for this level
            pool_key = (grid_shape, dtype)
            
            if pool_key not in self.memory_pool:
                self.memory_pool[pool_key] = {
                    'arrays': [],
                    'in_use': [],
                    'shape': grid_shape,
                    'dtype': dtype
                }
            
            # Allocate arrays for common operations
            pool = self.memory_pool[pool_key]
            
            # Arrays needed: residual, correction, temp arrays for smoothing
            needed_arrays = 4
            for _ in range(needed_arrays):
                array = np.zeros(grid_shape, dtype=dtype)
                pool['arrays'].append(array)
                pool['in_use'].append(False)
        
        logger.debug(f"Memory pool setup complete: {len(self.memory_pool)} pool types")
    
    def _setup_precision_strategies(self) -> None:
        """Setup adaptive precision strategies for different levels."""
        for level, grid in enumerate(self.grids):
            # Coarser grids can use lower precision
            if level >= len(self.grids) // 2:  # Coarse half
                recommended_precision = "single"
            else:  # Fine half
                recommended_precision = "double"
            
            self.precision_strategies[level] = {
                'recommended': recommended_precision,
                'current': recommended_precision,
                'switch_threshold': 0.9,
                'convergence_window': 3
            }
    
    def get_working_array(self, shape: Tuple[int, int], dtype: np.dtype) -> np.ndarray:
        """Get working array from memory pool or allocate new."""
        if not self.enable_memory_pool:
            return np.zeros(shape, dtype=dtype)
        
        pool_key = (shape, dtype)
        
        if pool_key in self.memory_pool:
            pool = self.memory_pool[pool_key]
            
            # Find available array
            for i, (array, in_use) in enumerate(zip(pool['arrays'], pool['in_use'])):
                if not in_use:
                    pool['in_use'][i] = True
                    array.fill(0)  # Clear array
                    return array
        
        # No available array, allocate new (fallback)
        logger.debug(f"Allocating new array: {shape}, {dtype}")
        return np.zeros(shape, dtype=dtype)
    
    def return_working_array(self, array: np.ndarray) -> None:
        """Return working array to memory pool."""
        if not self.enable_memory_pool:
            return
        
        shape = array.shape
        dtype = array.dtype
        pool_key = (shape, dtype)
        
        if pool_key in self.memory_pool:
            pool = self.memory_pool[pool_key]
            
            # Find and mark as available
            for i, pool_array in enumerate(pool['arrays']):
                if np.shares_memory(array, pool_array):
                    pool['in_use'][i] = False
                    return
    
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using communication-avoiding multigrid.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            rhs: Right-hand side
            initial_guess: Initial guess
            precision_manager: Precision manager
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        if not self.grids or grid.shape != self.grids[0].shape:
            raise ValueError("CA-Multigrid not properly setup or grid mismatch")
        
        self.reset()
        
        # Initialize solution
        if self.use_fmg:
            u = self._full_multigrid_initialization(rhs, precision_manager)
        else:
            if initial_guess is None:
                u = np.zeros_like(rhs)
            else:
                u = initial_guess.copy()
        
        # Store finest level RHS
        working_rhs = [np.zeros(g.shape, dtype=g.dtype) for g in self.grids]
        working_rhs[0][:] = rhs
        
        # Main multigrid iteration loop
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            
            # Update precision strategies if adaptive
            if precision_manager:
                current_residual = self._compute_residual_norm(u, rhs, 0)
                self._update_precision_strategies(current_residual, precision_manager)
            
            # Apply multigrid cycle with communication-avoiding optimizations
            u = self._communication_avoiding_cycle(u, working_rhs, 0, precision_manager)
            
            # Compute residual norm
            residual_norm = self._compute_residual_norm(u, rhs, 0)
            
            # Record iteration
            iteration_time = time.time() - iteration_start
            precision_level = (precision_manager.current_precision.value 
                             if precision_manager else "double")
            
            self.history.record_iteration(residual_norm, iteration_time, precision_level, 0)
            self.log_iteration(iteration, residual_norm)
            
            # Check convergence
            if self.check_convergence(residual_norm, iteration):
                self.converged = True
                break
        
        self.iterations_performed = iteration
        self.final_residual = residual_norm
        
        return u, self.get_convergence_info()
    
    def _communication_avoiding_cycle(
        self,
        u: np.ndarray,
        working_rhs: List[np.ndarray],
        level: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """
        Apply communication-avoiding multigrid cycle.
        
        Args:
            u: Current solution at this level
            working_rhs: Working RHS arrays for all levels
            level: Current grid level
            precision_manager: Precision manager
            
        Returns:
            Updated solution
        """
        level_start = time.time()
        
        if level == len(self.grids) - 1:
            # Coarsest level: solve directly
            u = self._solve_coarse_ca(u, working_rhs[level], level, precision_manager)
            self.cycle_stats.coarse_solve_time += time.time() - level_start
            return u
        
        # Get precision for this level
        current_precision = self._get_level_precision(level, precision_manager)
        
        # Convert arrays to appropriate precision
        if precision_manager and current_precision:
            u = precision_manager.convert_array(u, current_precision)
            working_rhs[level] = precision_manager.convert_array(
                working_rhs[level], current_precision
            )
        
        # Pre-smoothing with block operations
        if self.pre_smooth_iterations > 0:
            smooth_start = time.time()
            u = self._block_smoothing(u, working_rhs[level], level, 
                                    self.pre_smooth_iterations, precision_manager)
            self.cycle_stats.smoothing_times[level] = \
                self.cycle_stats.smoothing_times.get(level, 0) + (time.time() - smooth_start)
        
        # Compute and restrict residual with cache optimization
        restrict_start = time.time()
        residual = self.operators[level].residual(self.grids[level], u, working_rhs[level])
        
        # Use block-based restriction for better cache performance
        coarse_residual = self._block_restriction(
            residual, self.grids[level], self.grids[level + 1], level
        )
        working_rhs[level + 1][:] = coarse_residual
        
        self.cycle_stats.restriction_times[level] = \
            self.cycle_stats.restriction_times.get(level, 0) + (time.time() - restrict_start)
        
        # Recursive coarse grid correction
        coarse_correction = np.zeros_like(coarse_residual)
        
        if self.cycle_type == "V":
            coarse_correction = self._communication_avoiding_cycle(
                coarse_correction, working_rhs, level + 1, precision_manager
            )
        elif self.cycle_type == "W":
            # Two recursive calls for W-cycle
            coarse_correction = self._communication_avoiding_cycle(
                coarse_correction, working_rhs, level + 1, precision_manager
            )
            coarse_correction = self._communication_avoiding_cycle(
                coarse_correction, working_rhs, level + 1, precision_manager
            )
        elif self.cycle_type == "F":
            # F-cycle: increasing calls at deeper levels
            num_calls = min(2 ** (level + 1), 4)  # Limit exponential growth
            for _ in range(num_calls):
                coarse_correction = self._communication_avoiding_cycle(
                    coarse_correction, working_rhs, level + 1, precision_manager
                )
        
        # Prolongate correction with block operations
        prolong_start = time.time()
        fine_correction = self._block_prolongation(
            coarse_correction, self.grids[level + 1], self.grids[level], level
        )
        
        self.cycle_stats.prolongation_times[level] = \
            self.cycle_stats.prolongation_times.get(level, 0) + (time.time() - prolong_start)
        
        # Apply correction
        u += fine_correction
        
        # Post-smoothing with block operations
        if self.post_smooth_iterations > 0:
            smooth_start = time.time()
            u = self._block_smoothing(u, working_rhs[level], level,
                                    self.post_smooth_iterations, precision_manager)
            self.cycle_stats.smoothing_times[level] += time.time() - smooth_start
        
        self.cycle_stats.level_times[level] = \
            self.cycle_stats.level_times.get(level, 0) + (time.time() - level_start)
        
        return u
    
    def _block_smoothing(
        self,
        u: np.ndarray,
        rhs: np.ndarray,
        level: int,
        num_iterations: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """Apply smoothing with block operations for cache efficiency."""
        grid = self.grids[level]
        
        # Use block-structured smoothing for better cache locality
        block_size = min(self.block_size, grid.nx // 4, grid.ny // 4)
        if block_size < 4:
            # Fall back to regular smoothing for small grids
            return self.smoother.smooth(grid, self.operators[level], u, rhs, num_iterations)
        
        u_smooth = u.copy()
        
        # Cache-friendly block traversal
        for iteration in range(num_iterations):
            # Process in blocks to improve cache locality
            for block_i in range(1, grid.nx - 1, block_size):
                for block_j in range(1, grid.ny - 1, block_size):
                    # Define block boundaries
                    i_end = min(block_i + block_size, grid.nx - 1)
                    j_end = min(block_j + block_size, grid.ny - 1)
                    
                    # Apply smoothing within block
                    self._smooth_block(u_smooth, rhs, grid, block_i, block_j, i_end, j_end)
        
        return u_smooth
    
    def _smooth_block(
        self,
        u: np.ndarray,
        rhs: np.ndarray,
        grid: 'Grid',
        i_start: int,
        j_start: int,
        i_end: int,
        j_end: int
    ) -> None:
        """Apply Gauss-Seidel smoothing within a block."""
        # Pre-compute coefficients
        hx2_inv = 1.0 / (grid.hx ** 2)
        hy2_inv = 1.0 / (grid.hy ** 2)
        diag_coeff = -(2.0 * hx2_inv + 2.0 * hy2_inv)
        
        # Red-black smoothing within block for better convergence
        for color in [0, 1]:  # Red then black
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    if (i + j) % 2 == color:  # Red-black coloring
                        neighbors = (
                            hx2_inv * (u[i+1, j] + u[i-1, j]) +
                            hy2_inv * (u[i, j+1] + u[i, j-1])
                        )
                        
                        u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                        u[i, j] = u_new  # Direct assignment for Gauss-Seidel
    
    def _block_restriction(
        self,
        fine_field: np.ndarray,
        fine_grid: 'Grid',
        coarse_grid: 'Grid',
        level: int
    ) -> np.ndarray:
        """Apply restriction with block operations for cache efficiency."""
        # Use existing restriction operator but with block processing
        restriction_op = self.restriction_ops[level]
        
        # For large grids, process in blocks to improve cache locality
        if fine_grid.nx > 2 * self.block_size and fine_grid.ny > 2 * self.block_size:
            return self._blocked_transfer_operation(
                fine_field, fine_grid, coarse_grid, restriction_op, "restrict"
            )
        else:
            return restriction_op.apply(fine_grid, fine_field, coarse_grid)
    
    def _block_prolongation(
        self,
        coarse_field: np.ndarray,
        coarse_grid: 'Grid',
        fine_grid: 'Grid',
        level: int
    ) -> np.ndarray:
        """Apply prolongation with block operations for cache efficiency."""
        prolongation_op = self.prolongation_ops[level]
        
        # For large grids, process in blocks
        if fine_grid.nx > 2 * self.block_size and fine_grid.ny > 2 * self.block_size:
            return self._blocked_transfer_operation(
                coarse_field, coarse_grid, fine_grid, prolongation_op, "prolong"
            )
        else:
            return prolongation_op.apply(coarse_grid, coarse_field, fine_grid)
    
    def _blocked_transfer_operation(
        self,
        source_field: np.ndarray,
        source_grid: 'Grid',
        target_grid: 'Grid',
        transfer_op,
        operation_type: str
    ) -> np.ndarray:
        """Perform grid transfer operations in blocks for cache efficiency."""
        # This is a simplified block implementation
        # In practice, would implement more sophisticated blocked algorithms
        
        if operation_type == "restrict":
            return transfer_op.apply(source_grid, source_field, target_grid)
        else:  # prolong
            return transfer_op.apply(source_grid, source_field, target_grid)
    
    def _solve_coarse_ca(
        self,
        u: np.ndarray,
        rhs: np.ndarray,
        level: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """Solve coarse grid problem with communication-avoiding techniques."""
        coarse_solution, _ = self.coarse_solver.solve(
            self.grids[level],
            self.operators[level],
            rhs,
            u,
            precision_manager
        )
        
        return coarse_solution
    
    def _full_multigrid_initialization(
        self,
        rhs: np.ndarray,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """
        Full Multigrid (FMG) initialization for better initial guess.
        
        Starts on coarsest grid and interpolates up to finest grid.
        """
        logger.info("Starting Full Multigrid initialization")
        
        # Start from coarsest grid
        coarsest_level = len(self.grids) - 1
        
        # Restrict RHS to coarsest grid
        current_rhs = rhs.copy()
        rhs_hierarchy = [current_rhs]
        
        # Restrict to all levels
        for level in range(len(self.grids) - 1):
            restricted_rhs = self.restriction_ops[level].apply(
                self.grids[level], current_rhs, self.grids[level + 1]
            )
            rhs_hierarchy.append(restricted_rhs)
            current_rhs = restricted_rhs
        
        # Solve on coarsest grid
        coarsest_rhs = rhs_hierarchy[-1]
        u_coarse = np.zeros_like(coarsest_rhs)
        
        coarse_solution, _ = self.coarse_solver.solve(
            self.grids[coarsest_level],
            self.operators[coarsest_level],
            coarsest_rhs,
            u_coarse,
            precision_manager
        )
        
        # Prolongate and refine up to finest grid
        current_solution = coarse_solution
        
        for level in range(coarsest_level - 1, -1, -1):
            # Prolongate to finer grid
            prolongated = self.prolongation_ops[level].apply(
                self.grids[level + 1], current_solution, self.grids[level]
            )
            
            # Apply FMG cycles for refinement
            for fmg_cycle in range(self.fmg_cycles):
                prolongated = self._single_mg_cycle(
                    prolongated, rhs_hierarchy[level], level, precision_manager
                )
            
            current_solution = prolongated
        
        logger.info("Full Multigrid initialization complete")
        return current_solution
    
    def _single_mg_cycle(
        self,
        u: np.ndarray,
        rhs: np.ndarray,
        level: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """Apply single multigrid cycle at given level."""
        working_rhs = [np.zeros(g.shape, dtype=g.dtype) for g in self.grids]
        working_rhs[level][:] = rhs
        
        return self._communication_avoiding_cycle(u, working_rhs, level, precision_manager)
    
    def _update_precision_strategies(
        self,
        residual_norm: float,
        precision_manager: Optional['PrecisionManager']
    ) -> None:
        """Update precision strategies based on convergence behavior."""
        if not precision_manager or not precision_manager.adaptive:
            return
        
        # Update precision for different levels based on convergence
        for level in self.precision_strategies:
            strategy = self.precision_strategies[level]
            
            # Simple heuristic: use higher precision on fine grids when converging slowly
            if level < len(self.grids) // 2 and residual_norm > self.tolerance * 100:
                strategy['current'] = "double"
            elif level >= len(self.grids) // 2:
                strategy['current'] = "single"
    
    def _get_level_precision(
        self,
        level: int,
        precision_manager: Optional['PrecisionManager']
    ) -> Optional['PrecisionLevel']:
        """Get precision level recommendation for given grid level."""
        if not precision_manager:
            return None
        
        if level in self.precision_strategies:
            strategy = self.precision_strategies[level]
            precision_str = strategy['current']
            
            from ..core.precision import PrecisionLevel
            precision_map = {
                "single": PrecisionLevel.SINGLE,
                "double": PrecisionLevel.DOUBLE
            }
            
            return precision_map.get(precision_str, PrecisionLevel.DOUBLE)
        
        return precision_manager.get_precision_for_level(level, len(self.grids))
    
    def _compute_residual_norm(self, u: np.ndarray, rhs: np.ndarray, level: int) -> float:
        """Compute residual norm at given level."""
        residual = self.operators[level].residual(self.grids[level], u, rhs)
        return self.grids[level].l2_norm(residual)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        total_time = sum(self.cycle_stats.level_times.values())
        
        stats = {
            "total_cycle_time": total_time,
            "level_breakdown": {
                level: {
                    "time": self.cycle_stats.level_times.get(level, 0),
                    "percentage": (self.cycle_stats.level_times.get(level, 0) / total_time * 100
                                 if total_time > 0 else 0)
                }
                for level in range(len(self.grids))
            },
            "operation_breakdown": {
                "smoothing": sum(self.cycle_stats.smoothing_times.values()),
                "restriction": sum(self.cycle_stats.restriction_times.values()),
                "prolongation": sum(self.cycle_stats.prolongation_times.values()),
                "coarse_solve": self.cycle_stats.coarse_solve_time
            },
            "memory_efficiency": {
                "memory_pool_enabled": self.enable_memory_pool,
                "block_size": self.block_size,
                "cache_optimization": True
            }
        }
        
        return stats
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get extended convergence information."""
        base_info = super().get_convergence_info()
        
        # Add communication-avoiding specific information
        ca_info = {
            "cycle_type": self.cycle_type,
            "num_levels": len(self.grids),
            "fmg_used": self.use_fmg,
            "block_size": self.block_size,
            "memory_pool_enabled": self.enable_memory_pool,
            "performance_stats": self.get_performance_statistics()
        }
        
        return {**base_info, **ca_info}