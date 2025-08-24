"""Multigrid solver implementation."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import time
import logging

from .base import BaseSolver
from .smoothers import GaussSeidelSmoother

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.precision import PrecisionManager
    from ..operators.base import BaseOperator
    from ..operators.transfer import RestrictionOperator, ProlongationOperator
    from .base import IterativeSolver

logger = logging.getLogger(__name__)


class MultigridCycle:
    """Enumeration of multigrid cycle types."""
    V_CYCLE = "V"
    W_CYCLE = "W"
    F_CYCLE = "F"


class MultigridSolver(BaseSolver):
    """
    Geometric multigrid solver with mixed precision support.
    
    Implements V-cycle, W-cycle, and F-cycle multigrid methods with
    adaptive precision management.
    """
    
    def __init__(
        self,
        max_levels: int = 4,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        cycle_type: str = MultigridCycle.V_CYCLE,
        pre_smooth_iterations: int = 2,
        post_smooth_iterations: int = 2,
        coarse_tolerance: float = 1e-12,
        coarse_max_iterations: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize multigrid solver.
        
        Args:
            max_levels: Maximum number of grid levels
            max_iterations: Maximum multigrid iterations
            tolerance: Convergence tolerance
            cycle_type: Multigrid cycle type ('V', 'W', 'F')
            pre_smooth_iterations: Pre-smoothing iterations
            post_smooth_iterations: Post-smoothing iterations
            coarse_tolerance: Coarse grid solver tolerance
            coarse_max_iterations: Coarse grid solver max iterations
            verbose: Enable verbose output
        """
        super().__init__(max_iterations, tolerance, verbose, "Multigrid")
        
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.pre_smooth_iterations = pre_smooth_iterations
        self.post_smooth_iterations = post_smooth_iterations
        self.coarse_tolerance = coarse_tolerance
        self.coarse_max_iterations = coarse_max_iterations
        
        # Grid hierarchy storage
        self.grids: List['Grid'] = []
        self.solutions: List[np.ndarray] = []
        self.rhs_vectors: List[np.ndarray] = []
        self.residuals: List[np.ndarray] = []
        
        # Operator storage
        self.operators: List['BaseOperator'] = []
        self.restriction_ops: List['RestrictionOperator'] = []
        self.prolongation_ops: List['ProlongationOperator'] = []
        
        # Default smoother
        self.smoother: Optional['IterativeSolver'] = None
        self.coarse_solver: Optional['IterativeSolver'] = None
        
        # Performance tracking
        self.level_stats = {}
        
        logger.info(f"Initialized MultigridSolver: {cycle_type}-cycle, max_levels={max_levels}")
    
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
        Setup multigrid hierarchy.
        
        Args:
            fine_grid: Finest grid level
            operator: Differential operator
            restriction_op: Grid restriction operator
            prolongation_op: Grid prolongation operator
            smoother: Smoother for iterative refinement
            coarse_solver: Solver for coarsest grid
        """
        # Set default smoothers if not provided
        if smoother is None:
            smoother = GaussSeidelSmoother(
                max_iterations=max(self.pre_smooth_iterations, self.post_smooth_iterations),
                tolerance=self.tolerance * 0.1,
                verbose=False
            )
        
        if coarse_solver is None:
            coarse_solver = GaussSeidelSmoother(
                max_iterations=self.coarse_max_iterations,
                tolerance=self.coarse_tolerance,
                verbose=self.verbose
            )
        
        self.smoother = smoother
        self.coarse_solver = coarse_solver
        
        # Build grid hierarchy
        self._build_hierarchy(fine_grid, operator, restriction_op, prolongation_op)
        
        logger.info(f"Setup complete: {len(self.grids)} levels, "
                   f"finest: {self.grids[0].shape}, coarsest: {self.grids[-1].shape}")
    
    def _build_hierarchy(
        self,
        fine_grid: 'Grid',
        operator: 'BaseOperator', 
        restriction_op: 'RestrictionOperator',
        prolongation_op: 'ProlongationOperator'
    ) -> None:
        """Build multigrid hierarchy."""
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
                coarse_grid = current_grid.coarsen()
                
                # Check if grid is too coarse
                if coarse_grid.nx < 5 or coarse_grid.ny < 5:
                    logger.info(f"Stopping hierarchy at level {level}: grid too coarse")
                    break
                
                self.grids.append(coarse_grid)
                self.operators.append(operator)  # Same operator type for all levels
                self.restriction_ops.append(restriction_op)
                self.prolongation_ops.append(prolongation_op)
                
                current_grid = coarse_grid
                
            except ValueError as e:
                logger.info(f"Stopping hierarchy at level {level}: {e}")
                break
        
        # Initialize storage arrays
        self.solutions = [np.zeros(grid.shape, dtype=grid.dtype) for grid in self.grids]
        self.rhs_vectors = [np.zeros(grid.shape, dtype=grid.dtype) for grid in self.grids]
        self.residuals = [np.zeros(grid.shape, dtype=grid.dtype) for grid in self.grids]
        
        # Initialize level statistics
        self.level_stats = {
            level: {"smooth_time": 0.0, "restrict_time": 0.0, "prolong_time": 0.0}
            for level in range(len(self.grids))
        }
    
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using multigrid method.
        
        Args:
            grid: Computational grid (should match setup)
            operator: Linear operator
            rhs: Right-hand side
            initial_guess: Initial guess
            precision_manager: Precision manager
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        if not self.grids or grid.shape != self.grids[0].shape:
            raise ValueError("Multigrid not properly setup or grid mismatch")
        
        self.reset()
        
        # Initialize solution
        if initial_guess is None:
            u = np.zeros_like(rhs)
        else:
            u = initial_guess.copy()
        
        # Store finest level RHS
        self.rhs_vectors[0] = rhs.copy()
        
        # Main multigrid iteration loop
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            
            # Update precision if adaptive
            if precision_manager:
                grid_shapes = [(g.nx, g.ny) for g in self.grids]
                current_residual = self._compute_residual_norm(u, rhs, 0)
                precision_manager.update_precision(current_residual, grid_shapes)
            
            # Apply multigrid cycle
            u = self._multigrid_cycle(u, 0, precision_manager)
            
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
    
    def _multigrid_cycle(
        self, 
        u: np.ndarray, 
        level: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """
        Apply one multigrid cycle.
        
        Args:
            u: Current solution at this level
            level: Current grid level (0 = finest)
            precision_manager: Precision manager
            
        Returns:
            Updated solution
        """
        if level == len(self.grids) - 1:
            # Coarsest level: solve directly
            return self._solve_coarse(u, level, precision_manager)
        
        # Get precision for this level
        current_precision = (
            precision_manager.get_precision_for_level(level, len(self.grids))
            if precision_manager else None
        )
        
        # Convert to appropriate precision
        if precision_manager and current_precision:
            u = precision_manager.convert_array(u, current_precision)
            self.rhs_vectors[level] = precision_manager.convert_array(
                self.rhs_vectors[level], current_precision
            )
        
        # Pre-smoothing
        if self.pre_smooth_iterations > 0:
            start_time = time.time()
            u = self._smooth(u, level, self.pre_smooth_iterations, precision_manager)
            self.level_stats[level]["smooth_time"] += time.time() - start_time
        
        # Compute residual
        residual = self.operators[level].residual(self.grids[level], u, self.rhs_vectors[level])
        
        # Restrict residual to coarse grid
        start_time = time.time()
        coarse_residual = self.restriction_ops[level].apply(
            self.grids[level], residual, self.grids[level + 1]
        )
        self.level_stats[level]["restrict_time"] += time.time() - start_time
        
        # Store coarse RHS
        self.rhs_vectors[level + 1] = coarse_residual.copy()
        
        # Solve coarse problem (initialize with zeros)
        coarse_correction = np.zeros_like(coarse_residual)
        
        if self.cycle_type == MultigridCycle.V_CYCLE:
            coarse_correction = self._multigrid_cycle(coarse_correction, level + 1, precision_manager)
        elif self.cycle_type == MultigridCycle.W_CYCLE:
            # Two recursive calls for W-cycle
            coarse_correction = self._multigrid_cycle(coarse_correction, level + 1, precision_manager)
            coarse_correction = self._multigrid_cycle(coarse_correction, level + 1, precision_manager)
        elif self.cycle_type == MultigridCycle.F_CYCLE:
            # F-cycle: increasing number of calls at each level
            num_calls = 2 ** (len(self.grids) - level - 2)
            for _ in range(max(1, num_calls)):
                coarse_correction = self._multigrid_cycle(coarse_correction, level + 1, precision_manager)
        
        # Prolongate correction to fine grid
        start_time = time.time()
        fine_correction = self.prolongation_ops[level].apply(
            self.grids[level + 1], coarse_correction, self.grids[level]
        )
        self.level_stats[level]["prolong_time"] += time.time() - start_time
        
        # Apply correction
        u += fine_correction
        
        # Post-smoothing
        if self.post_smooth_iterations > 0:
            start_time = time.time()
            u = self._smooth(u, level, self.post_smooth_iterations, precision_manager)
            self.level_stats[level]["smooth_time"] += time.time() - start_time
        
        return u
    
    def _smooth(
        self, 
        u: np.ndarray, 
        level: int, 
        num_iterations: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """Apply smoothing at given level."""
        return self.smoother.smooth(
            self.grids[level], 
            self.operators[level],
            u,
            self.rhs_vectors[level],
            num_iterations
        )
    
    def _solve_coarse(
        self, 
        u: np.ndarray, 
        level: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """Solve coarse grid problem directly."""
        coarse_solution, _ = self.coarse_solver.solve(
            self.grids[level],
            self.operators[level],
            self.rhs_vectors[level],
            u,
            precision_manager
        )
        
        return coarse_solution
    
    def _compute_residual_norm(self, u: np.ndarray, rhs: np.ndarray, level: int) -> float:
        """Compute residual norm at given level."""
        residual = self.operators[level].residual(self.grids[level], u, rhs)
        return self.grids[level].l2_norm(residual)
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get extended convergence information."""
        base_info = super().get_convergence_info()
        
        # Add multigrid-specific information
        mg_info = {
            "cycle_type": self.cycle_type,
            "num_levels": len(self.grids),
            "grid_hierarchy": [(g.nx, g.ny) for g in self.grids],
            "level_timings": self.level_stats.copy(),
            "pre_smooth_iterations": self.pre_smooth_iterations,
            "post_smooth_iterations": self.post_smooth_iterations
        }
        
        return {**base_info, **mg_info}