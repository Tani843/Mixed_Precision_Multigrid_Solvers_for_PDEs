"""Multigrid as preconditioner implementation."""

import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Any
import logging

from .base import BasePreconditioner

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.precision import PrecisionManager
    from ..operators.base import BaseOperator
    from ..operators.transfer import RestrictionOperator, ProlongationOperator
    from ..solvers.base import IterativeSolver
    from ..solvers.multigrid import MultigridSolver

logger = logging.getLogger(__name__)


class MultigridPreconditioner(BasePreconditioner):
    """
    Multigrid method used as a preconditioner.
    
    This applies one or few multigrid cycles as a preconditioner
    for outer iterative methods like CG or GMRES.
    """
    
    def __init__(
        self,
        max_levels: int = 3,
        cycle_type: str = "V",
        pre_smooth_iterations: int = 1,
        post_smooth_iterations: int = 1,
        num_cycles: int = 1,
        coarse_tolerance: float = 1e-6,
        coarse_max_iterations: int = 100
    ):
        """
        Initialize multigrid preconditioner.
        
        Args:
            max_levels: Maximum number of grid levels
            cycle_type: Multigrid cycle type ('V', 'W', 'F')
            pre_smooth_iterations: Pre-smoothing iterations
            post_smooth_iterations: Post-smoothing iterations
            num_cycles: Number of multigrid cycles per application
            coarse_tolerance: Tolerance for coarse grid solver
            coarse_max_iterations: Max iterations for coarse grid solver
        """
        super().__init__("MultigridPreconditioner")
        
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.pre_smooth_iterations = pre_smooth_iterations
        self.post_smooth_iterations = post_smooth_iterations
        self.num_cycles = num_cycles
        self.coarse_tolerance = coarse_tolerance
        self.coarse_max_iterations = coarse_max_iterations
        
        # Multigrid components (will be set during setup)
        self.mg_solver = None
        self.grid = None
        self.operator = None
        self.precision_manager = None
    
    def setup(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        restriction_op: Optional['RestrictionOperator'] = None,
        prolongation_op: Optional['ProlongationOperator'] = None,
        smoother: Optional['IterativeSolver'] = None,
        coarse_solver: Optional['IterativeSolver'] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> None:
        """
        Setup multigrid preconditioner.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            restriction_op: Grid restriction operator
            prolongation_op: Grid prolongation operator
            smoother: Smoothing method
            coarse_solver: Coarse grid solver
            precision_manager: Precision management
        """
        from ..operators.transfer import RestrictionOperator, ProlongationOperator
        from ..solvers.smoothers import GaussSeidelSmoother
        from ..solvers.multigrid import MultigridSolver
        
        # Store references
        self.grid = grid
        self.operator = operator
        self.precision_manager = precision_manager
        
        # Set default operators if not provided
        if restriction_op is None:
            restriction_op = RestrictionOperator("full_weighting")
        
        if prolongation_op is None:
            prolongation_op = ProlongationOperator("bilinear")
        
        if smoother is None:
            smoother = GaussSeidelSmoother(
                max_iterations=max(self.pre_smooth_iterations, self.post_smooth_iterations),
                tolerance=self.coarse_tolerance * 0.1,
                verbose=False
            )
        
        if coarse_solver is None:
            coarse_solver = GaussSeidelSmoother(
                max_iterations=self.coarse_max_iterations,
                tolerance=self.coarse_tolerance,
                verbose=False
            )
        
        # Create multigrid solver for preconditioning
        self.mg_solver = MultigridSolver(
            max_levels=self.max_levels,
            max_iterations=self.num_cycles,  # Use cycles as iterations
            tolerance=1e-16,  # Don't stop early when used as preconditioner
            cycle_type=self.cycle_type,
            pre_smooth_iterations=self.pre_smooth_iterations,
            post_smooth_iterations=self.post_smooth_iterations,
            coarse_tolerance=self.coarse_tolerance,
            coarse_max_iterations=self.coarse_max_iterations,
            verbose=False
        )
        
        # Setup multigrid hierarchy
        self.mg_solver.setup(grid, operator, restriction_op, prolongation_op, smoother, coarse_solver)
        
        self.setup_completed = True
        
        logger.debug(f"Setup multigrid preconditioner: {len(self.mg_solver.grids)} levels, "
                    f"{self.cycle_type}-cycle, {self.num_cycles} cycles per application")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply multigrid preconditioner: solve M*z ≈ x.
        
        Args:
            x: Input residual vector
            
        Returns:
            Preconditioned vector z
        """
        if not self.setup_completed:
            raise RuntimeError("Multigrid preconditioner not setup")
        
        # Use zero initial guess for preconditioning
        initial_guess = np.zeros_like(x)
        
        # Apply multigrid cycles
        solution, _ = self.mg_solver.solve(
            self.grid,
            self.operator,
            x,  # RHS is the residual
            initial_guess,
            self.precision_manager
        )
        
        return solution
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """
        Apply transpose of multigrid preconditioner.
        
        For symmetric problems, the multigrid preconditioner is approximately symmetric,
        so we can use the same application.
        """
        # For symmetric operators, MG preconditioner is approximately symmetric
        return self.apply(x)


class FlexibleMultigridPreconditioner(MultigridPreconditioner):
    """
    Flexible multigrid preconditioner that adapts parameters during iterations.
    
    Can adjust smoothing iterations, cycle type, or precision based on
    convergence behavior.
    """
    
    def __init__(
        self,
        max_levels: int = 3,
        initial_cycle_type: str = "V",
        adaptive_cycles: bool = True,
        adaptive_smoothing: bool = True,
        min_smooth_iterations: int = 1,
        max_smooth_iterations: int = 3,
        **kwargs
    ):
        """
        Initialize flexible multigrid preconditioner.
        
        Args:
            max_levels: Maximum grid levels
            initial_cycle_type: Initial cycle type
            adaptive_cycles: Enable adaptive cycle type switching
            adaptive_smoothing: Enable adaptive smoothing iteration adjustment
            min_smooth_iterations: Minimum smoothing iterations
            max_smooth_iterations: Maximum smoothing iterations
            **kwargs: Additional arguments for parent class
        """
        super().__init__(max_levels=max_levels, cycle_type=initial_cycle_type, **kwargs)
        self.name = "FlexibleMultigridPreconditioner"
        
        self.adaptive_cycles = adaptive_cycles
        self.adaptive_smoothing = adaptive_smoothing
        self.min_smooth_iterations = min_smooth_iterations
        self.max_smooth_iterations = max_smooth_iterations
        self.initial_cycle_type = initial_cycle_type
        
        # Adaptation tracking
        self.application_count = 0
        self.performance_history = []
        self.current_smooth_iterations = self.pre_smooth_iterations
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply flexible multigrid preconditioner with adaptation."""
        if not self.setup_completed:
            raise RuntimeError("Flexible multigrid preconditioner not setup")
        
        # Adapt parameters based on performance history
        if self.application_count > 0:
            self._adapt_parameters()
        
        # Apply multigrid with current parameters
        start_time = self._get_time()
        result = super().apply(x)
        end_time = self._get_time()
        
        # Record performance
        application_time = end_time - start_time
        residual_reduction = np.linalg.norm(result) / max(np.linalg.norm(x), 1e-16)
        
        self.performance_history.append({
            'time': application_time,
            'residual_reduction': residual_reduction,
            'smooth_iterations': self.current_smooth_iterations,
            'cycle_type': self.cycle_type
        })
        
        self.application_count += 1
        
        return result
    
    def _adapt_parameters(self) -> None:
        """Adapt preconditioner parameters based on performance history."""
        if len(self.performance_history) < 3:
            return
        
        # Analyze recent performance
        recent_performance = self.performance_history[-3:]
        avg_reduction = np.mean([p['residual_reduction'] for p in recent_performance])
        avg_time = np.mean([p['time'] for p in recent_performance])
        
        # Adapt smoothing iterations
        if self.adaptive_smoothing:
            if avg_reduction > 0.8:  # Poor reduction
                if self.current_smooth_iterations < self.max_smooth_iterations:
                    self.current_smooth_iterations += 1
                    self._update_smoother_iterations()
                    logger.debug(f"Increased smoothing iterations to {self.current_smooth_iterations}")
            
            elif avg_reduction < 0.3 and avg_time > 0.1:  # Good reduction but slow
                if self.current_smooth_iterations > self.min_smooth_iterations:
                    self.current_smooth_iterations -= 1
                    self._update_smoother_iterations()
                    logger.debug(f"Decreased smoothing iterations to {self.current_smooth_iterations}")
        
        # Adapt cycle type
        if self.adaptive_cycles and len(self.performance_history) >= 6:
            self._adapt_cycle_type(avg_reduction)
    
    def _update_smoother_iterations(self) -> None:
        """Update smoothing iterations in the multigrid solver."""
        if self.mg_solver:
            self.mg_solver.pre_smooth_iterations = self.current_smooth_iterations
            self.mg_solver.post_smooth_iterations = self.current_smooth_iterations
    
    def _adapt_cycle_type(self, avg_reduction: float) -> None:
        """Adapt cycle type based on performance."""
        if avg_reduction > 0.7:  # Poor performance
            if self.cycle_type == "V":
                self.cycle_type = "W"
                self.mg_solver.cycle_type = "W"
                logger.debug("Switched to W-cycle for better convergence")
            elif self.cycle_type == "W":
                self.cycle_type = "F"
                self.mg_solver.cycle_type = "F"
                logger.debug("Switched to F-cycle for better convergence")
        
        elif avg_reduction < 0.2:  # Very good performance
            if self.cycle_type == "F":
                self.cycle_type = "W"
                self.mg_solver.cycle_type = "W"
                logger.debug("Switched to W-cycle for efficiency")
            elif self.cycle_type == "W":
                self.cycle_type = "V"
                self.mg_solver.cycle_type = "V"
                logger.debug("Switched to V-cycle for efficiency")
    
    def _get_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about parameter adaptation."""
        if not self.performance_history:
            return {"applications": 0}
        
        # Compute statistics
        reductions = [p['residual_reduction'] for p in self.performance_history]
        times = [p['time'] for p in self.performance_history]
        smooth_iterations = [p['smooth_iterations'] for p in self.performance_history]
        
        return {
            "applications": len(self.performance_history),
            "avg_residual_reduction": np.mean(reductions),
            "avg_application_time": np.mean(times),
            "current_smooth_iterations": self.current_smooth_iterations,
            "current_cycle_type": self.cycle_type,
            "smooth_iteration_range": [min(smooth_iterations), max(smooth_iterations)],
            "total_adaptations": len(set(smooth_iterations)) + 
                               len(set([p['cycle_type'] for p in self.performance_history])) - 2
        }


class TwoLevelPreconditioner(BasePreconditioner):
    """
    Two-level preconditioner (coarse grid correction).
    
    Applies: z = smoother(x) + prolongation(coarse_solve(restriction(residual)))
    """
    
    def __init__(
        self,
        pre_smooth_iterations: int = 1,
        post_smooth_iterations: int = 1,
        coarse_tolerance: float = 1e-6,
        coarse_max_iterations: int = 100
    ):
        """
        Initialize two-level preconditioner.
        
        Args:
            pre_smooth_iterations: Pre-smoothing iterations
            post_smooth_iterations: Post-smoothing iterations
            coarse_tolerance: Coarse grid solver tolerance
            coarse_max_iterations: Coarse grid solver max iterations
        """
        super().__init__("TwoLevel")
        
        self.pre_smooth_iterations = pre_smooth_iterations
        self.post_smooth_iterations = post_smooth_iterations
        self.coarse_tolerance = coarse_tolerance
        self.coarse_max_iterations = coarse_max_iterations
        
        # Components
        self.fine_grid = None
        self.coarse_grid = None
        self.operator = None
        self.smoother = None
        self.coarse_solver = None
        self.restriction_op = None
        self.prolongation_op = None
    
    def setup(
        self,
        grid: 'Grid',
        operator: 'BaseOperator',
        restriction_op: Optional['RestrictionOperator'] = None,
        prolongation_op: Optional['ProlongationOperator'] = None,
        smoother: Optional['IterativeSolver'] = None,
        coarse_solver: Optional['IterativeSolver'] = None
    ) -> None:
        """Setup two-level preconditioner components."""
        from ..operators.transfer import RestrictionOperator, ProlongationOperator
        from ..solvers.smoothers import GaussSeidelSmoother
        
        # Store fine grid and operator
        self.fine_grid = grid
        self.operator = operator
        
        # Create coarse grid
        self.coarse_grid = grid.coarsen()
        
        # Set default components
        if restriction_op is None:
            restriction_op = RestrictionOperator("full_weighting")
        
        if prolongation_op is None:
            prolongation_op = ProlongationOperator("bilinear")
        
        if smoother is None:
            smoother = GaussSeidelSmoother(
                max_iterations=max(self.pre_smooth_iterations, self.post_smooth_iterations),
                tolerance=1e-16,  # Don't stop early
                verbose=False
            )
        
        if coarse_solver is None:
            coarse_solver = GaussSeidelSmoother(
                max_iterations=self.coarse_max_iterations,
                tolerance=self.coarse_tolerance,
                verbose=False
            )
        
        self.restriction_op = restriction_op
        self.prolongation_op = prolongation_op
        self.smoother = smoother
        self.coarse_solver = coarse_solver
        
        self.setup_completed = True
        
        logger.debug(f"Setup two-level preconditioner: {self.fine_grid.shape} -> {self.coarse_grid.shape}")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply two-level preconditioner.
        
        Args:
            x: Input residual
            
        Returns:
            Preconditioned result
        """
        if not self.setup_completed:
            raise RuntimeError("Two-level preconditioner not setup")
        
        # Start with zero guess
        result = np.zeros_like(x)
        
        # Pre-smoothing
        if self.pre_smooth_iterations > 0:
            result = self.smoother.smooth(
                self.fine_grid, self.operator, result, x, self.pre_smooth_iterations
            )
        
        # Compute fine grid residual
        fine_residual = self.operator.residual(self.fine_grid, result, x)
        
        # Restrict residual to coarse grid
        coarse_residual = self.restriction_op.apply(
            self.fine_grid, fine_residual, self.coarse_grid
        )
        
        # Solve coarse grid problem
        coarse_correction, _ = self.coarse_solver.solve(
            self.coarse_grid, self.operator, coarse_residual
        )
        
        # Prolongate correction to fine grid
        fine_correction = self.prolongation_op.apply(
            self.coarse_grid, coarse_correction, self.fine_grid
        )
        
        # Apply correction
        result += fine_correction
        
        # Post-smoothing
        if self.post_smooth_iterations > 0:
            result = self.smoother.smooth(
                self.fine_grid, self.operator, result, x, self.post_smooth_iterations
            )
        
        return result
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """Apply transpose (approximately the same for symmetric problems)."""
        return self.apply(x)