"""Base classes for iterative solvers."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import time
import logging

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.precision import PrecisionManager
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class ConvergenceHistory:
    """Track convergence history for solvers."""
    
    def __init__(self):
        """Initialize convergence history."""
        self.residual_norms = []
        self.iteration_times = []
        self.precision_levels = []
        self.grid_levels = []
    
    def record_iteration(
        self, 
        residual_norm: float, 
        iteration_time: float,
        precision_level: str,
        grid_level: Optional[int] = None
    ) -> None:
        """Record an iteration."""
        self.residual_norms.append(residual_norm)
        self.iteration_times.append(iteration_time)
        self.precision_levels.append(precision_level)
        self.grid_levels.append(grid_level)
    
    def get_convergence_rate(self) -> float:
        """Estimate asymptotic convergence rate."""
        if len(self.residual_norms) < 3:
            return 0.0
        
        # Use last few iterations to estimate rate
        recent_residuals = self.residual_norms[-5:]
        if len(recent_residuals) < 2:
            return 0.0
        
        ratios = []
        for i in range(1, len(recent_residuals)):
            if recent_residuals[i-1] > 0:
                ratio = recent_residuals[i] / recent_residuals[i-1]
                if 0 < ratio < 1:  # Converging
                    ratios.append(ratio)
        
        return np.mean(ratios) if ratios else 0.0
    
    def clear(self) -> None:
        """Clear convergence history."""
        self.residual_norms.clear()
        self.iteration_times.clear()
        self.precision_levels.clear()
        self.grid_levels.clear()


class BaseSolver(ABC):
    """Abstract base class for iterative solvers."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
        name: str = "BaseSolver"
    ):
        """
        Initialize base solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Enable verbose output
            name: Solver name for logging
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.name = name
        
        # Convergence tracking
        self.history = ConvergenceHistory()
        self.converged = False
        self.final_residual = float('inf')
        self.iterations_performed = 0
        
        logger.info(f"Initialized {name}: max_iter={max_iterations}, tol={tolerance}")
    
    @abstractmethod
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the linear system.
        
        Args:
            grid: Computational grid
            operator: Linear operator (e.g., Laplacian)
            rhs: Right-hand side vector
            initial_guess: Initial guess for solution
            precision_manager: Precision management system
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        pass
    
    def check_convergence(self, residual_norm: float, iteration: int) -> bool:
        """
        Check convergence criteria.
        
        Args:
            residual_norm: Current residual norm
            iteration: Current iteration number
            
        Returns:
            True if converged
        """
        converged = residual_norm < self.tolerance
        
        if converged:
            logger.info(f"{self.name} converged in {iteration} iterations: "
                       f"residual = {residual_norm:.2e}")
        elif iteration >= self.max_iterations:
            logger.warning(f"{self.name} reached max iterations ({self.max_iterations}): "
                          f"residual = {residual_norm:.2e}")
        
        return converged
    
    def log_iteration(self, iteration: int, residual_norm: float, grid_level: Optional[int] = None) -> None:
        """Log iteration information."""
        if self.verbose and iteration % max(1, self.max_iterations // 10) == 0:
            level_str = f" (level {grid_level})" if grid_level is not None else ""
            logger.info(f"{self.name} iteration {iteration}{level_str}: "
                       f"residual = {residual_norm:.2e}")
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get convergence information.
        
        Returns:
            Dictionary with convergence statistics
        """
        return {
            "converged": self.converged,
            "iterations": self.iterations_performed,
            "final_residual": self.final_residual,
            "convergence_rate": self.history.get_convergence_rate(),
            "residual_history": self.history.residual_norms.copy(),
            "total_time": sum(self.history.iteration_times),
            "average_time_per_iteration": (
                np.mean(self.history.iteration_times) 
                if self.history.iteration_times else 0.0
            ),
            "precision_levels_used": list(set(self.history.precision_levels))
        }
    
    def reset(self) -> None:
        """Reset solver state."""
        self.history.clear()
        self.converged = False
        self.final_residual = float('inf')
        self.iterations_performed = 0
        
        logger.debug(f"Reset {self.name} solver state")


class IterativeSolver(BaseSolver):
    """Base class for simple iterative solvers like Jacobi and Gauss-Seidel."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 1.0,
        verbose: bool = False,
        name: str = "IterativeSolver"
    ):
        """
        Initialize iterative solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            relaxation_parameter: Relaxation parameter (ω)
            verbose: Enable verbose output
            name: Solver name
        """
        super().__init__(max_iterations, tolerance, verbose, name)
        self.omega = relaxation_parameter
        
        if not 0 < relaxation_parameter <= 2:
            logger.warning(f"Relaxation parameter {relaxation_parameter} may cause instability")
    
    @abstractmethod
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """
        Apply smoothing iterations.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            u: Current solution estimate
            rhs: Right-hand side
            num_iterations: Number of smoothing iterations
            
        Returns:
            Smoothed solution
        """
        pass
    
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using iterative method.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            rhs: Right-hand side
            initial_guess: Initial guess
            precision_manager: Precision manager
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        self.reset()
        
        # Initialize solution
        if initial_guess is None:
            u = np.zeros_like(rhs)
        else:
            u = initial_guess.copy()
        
        # Main iteration loop
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            
            # Apply one smoothing iteration
            u = self.smooth(grid, operator, u, rhs, 1)
            
            # Compute residual
            residual = operator.residual(grid, u, rhs)
            residual_norm = grid.l2_norm(residual)
            
            # Record iteration
            iteration_time = time.time() - iteration_start
            precision_level = (precision_manager.current_precision.value 
                             if precision_manager else "unknown")
            
            self.history.record_iteration(residual_norm, iteration_time, precision_level)
            self.log_iteration(iteration, residual_norm)
            
            # Check convergence
            if self.check_convergence(residual_norm, iteration):
                self.converged = True
                break
        
        self.iterations_performed = iteration
        self.final_residual = residual_norm
        
        return u, self.get_convergence_info()