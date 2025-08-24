"""Enhanced iterative solvers with mixed-precision support."""

import numpy as np
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import time
import logging

from .base import IterativeSolver

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.precision import PrecisionManager
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class EnhancedJacobiSolver(IterativeSolver):
    """
    Enhanced Jacobi solver with vectorized operations and mixed precision.
    
    Update formula: u^{k+1}_i = (f_i - Σ_{j≠i} a_{ij}u^k_j) / a_{ii}
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 2.0/3.0,
        verbose: bool = False,
        use_vectorized: bool = True
    ):
        """
        Initialize enhanced Jacobi solver.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            relaxation_parameter: Damping parameter (optimal ≈ 2/3 for 2D Laplacian)
            verbose: Enable verbose output
            use_vectorized: Use vectorized operations for performance
        """
        super().__init__(max_iterations, tolerance, relaxation_parameter, verbose, "EnhancedJacobi")
        self.use_vectorized = use_vectorized
    
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """
        Apply Jacobi smoothing with vectorized operations.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            u: Current solution estimate
            rhs: Right-hand side
            num_iterations: Number of smoothing iterations
            
        Returns:
            Smoothed solution
        """
        if self.use_vectorized:
            return self._vectorized_smooth(grid, u, rhs, num_iterations)
        else:
            return super().smooth(grid, operator, u, rhs, num_iterations)
    
    def _vectorized_smooth(
        self, 
        grid: 'Grid',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int
    ) -> np.ndarray:
        """Vectorized Jacobi smoothing for better performance."""
        u_smooth = u.copy()
        
        # Pre-compute coefficients for 5-point stencil
        hx2_inv = 1.0 / (grid.hx ** 2)
        hy2_inv = 1.0 / (grid.hy ** 2)
        diag_coeff = -(2.0 * hx2_inv + 2.0 * hy2_inv)
        
        for _ in range(num_iterations):
            u_old = u_smooth.copy()
            
            # Vectorized computation for interior points
            interior_i = slice(1, -1)
            interior_j = slice(1, -1)
            
            # Compute neighbor contributions using array slicing
            neighbors = (
                hx2_inv * (u_old[2:, 1:-1] + u_old[:-2, 1:-1]) +
                hy2_inv * (u_old[1:-1, 2:] + u_old[1:-1, :-2])
            )
            
            # Jacobi update with relaxation
            u_new = (rhs[interior_i, interior_j] + neighbors) / (-diag_coeff)
            u_smooth[interior_i, interior_j] = (
                (1.0 - self.omega) * u_old[interior_i, interior_j] + 
                self.omega * u_new
            )
        
        logger.debug(f"Applied {num_iterations} vectorized Jacobi iterations")
        return u_smooth
    
    def compute_spectral_radius(self, grid: 'Grid') -> float:
        """
        Compute spectral radius for optimal relaxation parameter.
        
        For 2D Laplacian: ρ(J) = cos(π/(nx-1)) * cos(π/(ny-1))
        """
        rho_x = np.cos(np.pi / (grid.nx - 1))
        rho_y = np.cos(np.pi / (grid.ny - 1))
        spectral_radius = rho_x * rho_y
        
        logger.debug(f"Jacobi spectral radius: {spectral_radius:.4f}")
        return spectral_radius


class EnhancedGaussSeidelSolver(IterativeSolver):
    """
    Enhanced Gauss-Seidel solver with red-black ordering option.
    
    Update formula: u^{k+1}_i = (f_i - Σ_{j<i} a_{ij}u^{k+1}_j - Σ_{j>i} a_{ij}u^k_j) / a_{ii}
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 1.0,
        verbose: bool = False,
        red_black: bool = True,
        symmetric: bool = False
    ):
        """
        Initialize enhanced Gauss-Seidel solver.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            relaxation_parameter: SOR parameter (1.0 = pure Gauss-Seidel)
            verbose: Enable verbose output
            red_black: Use red-black ordering for parallelization
            symmetric: Use symmetric Gauss-Seidel (forward + backward)
        """
        name = "SymmetricGaussSeidel" if symmetric else "EnhancedGaussSeidel"
        if red_black:
            name += "RedBlack"
            
        super().__init__(max_iterations, tolerance, relaxation_parameter, verbose, name)
        self.red_black = red_black
        self.symmetric = symmetric
    
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """Apply Gauss-Seidel smoothing."""
        u_smooth = u.copy()
        
        # Pre-compute coefficients
        hx2_inv = 1.0 / (grid.hx ** 2)
        hy2_inv = 1.0 / (grid.hy ** 2)
        diag_coeff = -(2.0 * hx2_inv + 2.0 * hy2_inv)
        
        for _ in range(num_iterations):
            if self.red_black:
                u_smooth = self._red_black_sweep(grid, u_smooth, rhs, hx2_inv, hy2_inv, diag_coeff)
                if self.symmetric:
                    u_smooth = self._red_black_sweep(grid, u_smooth, rhs, hx2_inv, hy2_inv, diag_coeff, reverse=True)
            else:
                u_smooth = self._lexicographic_sweep(grid, u_smooth, rhs, hx2_inv, hy2_inv, diag_coeff)
                if self.symmetric:
                    u_smooth = self._lexicographic_sweep(grid, u_smooth, rhs, hx2_inv, hy2_inv, diag_coeff, reverse=True)
        
        logger.debug(f"Applied {num_iterations} Gauss-Seidel iterations "
                    f"({'red-black' if self.red_black else 'lexicographic'}"
                    f"{', symmetric' if self.symmetric else ''})")
        return u_smooth
    
    def _red_black_sweep(
        self, 
        grid: 'Grid',
        u: np.ndarray,
        rhs: np.ndarray,
        hx2_inv: float,
        hy2_inv: float,
        diag_coeff: float,
        reverse: bool = False
    ) -> np.ndarray:
        """Red-black Gauss-Seidel sweep for parallelization."""
        colors = [(0, 1), (1, 0)] if not reverse else [(1, 0), (0, 1)]
        
        for color_i, color_j in colors:
            # Vectorized red-black update
            for i in range(1 + color_i, grid.nx - 1, 2):
                for j in range(1 + color_j, grid.ny - 1, 2):
                    neighbors = (
                        hx2_inv * (u[i+1, j] + u[i-1, j]) +
                        hy2_inv * (u[i, j+1] + u[i, j-1])
                    )
                    
                    u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                    u[i, j] = (1.0 - self.omega) * u[i, j] + self.omega * u_new
        
        return u
    
    def _lexicographic_sweep(
        self, 
        grid: 'Grid',
        u: np.ndarray,
        rhs: np.ndarray,
        hx2_inv: float,
        hy2_inv: float,
        diag_coeff: float,
        reverse: bool = False
    ) -> np.ndarray:
        """Lexicographic Gauss-Seidel sweep."""
        if reverse:
            i_range = range(grid.nx - 2, 0, -1)
            j_range = range(grid.ny - 2, 0, -1)
        else:
            i_range = range(1, grid.nx - 1)
            j_range = range(1, grid.ny - 1)
        
        for i in i_range:
            for j in j_range:
                neighbors = (
                    hx2_inv * (u[i+1, j] + u[i-1, j]) +
                    hy2_inv * (u[i, j+1] + u[i, j-1])
                )
                
                u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                u[i, j] = (1.0 - self.omega) * u[i, j] + self.omega * u_new
        
        return u


class SORSolver(EnhancedGaussSeidelSolver):
    """
    Successive Over-Relaxation (SOR) solver.
    
    Update formula: u^{k+1}_i = u^k_i + ω(u^{k+1}_{GS,i} - u^k_i)
    where u^{k+1}_{GS,i} is the Gauss-Seidel update.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 1.8,
        verbose: bool = False,
        red_black: bool = True,
        auto_omega: bool = True
    ):
        """
        Initialize SOR solver.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            relaxation_parameter: SOR parameter (1 < ω < 2 for convergence)
            verbose: Enable verbose output
            red_black: Use red-black ordering
            auto_omega: Automatically compute optimal ω
        """
        super().__init__(max_iterations, tolerance, relaxation_parameter, verbose, red_black)
        self.name = "SOR"
        self.auto_omega = auto_omega
        
        # Validate SOR parameter
        if not (0 < relaxation_parameter < 2):
            logger.warning(f"SOR parameter ω={relaxation_parameter} may not converge. "
                          f"Recommend 1 < ω < 2")
    
    def setup_optimal_omega(self, grid: 'Grid') -> float:
        """
        Compute optimal SOR parameter for 2D Laplacian.
        
        Optimal ω = 2 / (1 + sin(π*h)) where h = min(hx, hy)
        """
        if not self.auto_omega:
            return self.omega
        
        h = min(grid.hx, grid.hy)
        optimal_omega = 2.0 / (1.0 + np.sin(np.pi * h))
        
        logger.info(f"Computed optimal SOR parameter: ω = {optimal_omega:.4f}")
        self.omega = optimal_omega
        
        return optimal_omega
    
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve with optimal SOR parameter."""
        if self.auto_omega:
            self.setup_optimal_omega(grid)
        
        return super().solve(grid, operator, rhs, initial_guess, precision_manager)


class WeightedJacobiSolver(EnhancedJacobiSolver):
    """
    Weighted Jacobi solver with optimal damping parameter.
    
    For 2D Laplacian, optimal weight is ω = 4/5 = 0.8
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
        auto_weight: bool = True
    ):
        """
        Initialize weighted Jacobi solver.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Enable verbose output
            auto_weight: Automatically compute optimal weight
        """
        # Default to optimal weight for 2D Laplacian
        optimal_weight = 4.0 / 5.0
        super().__init__(max_iterations, tolerance, optimal_weight, verbose, True)
        self.name = "WeightedJacobi"
        self.auto_weight = auto_weight
    
    def setup_optimal_weight(self, grid: 'Grid') -> float:
        """
        Compute optimal weight based on spectral radius.
        
        Optimal weight: ω = 2 / (1 + √(1 - ρ²))
        where ρ is the spectral radius of the Jacobi iteration matrix.
        """
        if not self.auto_weight:
            return self.omega
        
        spectral_radius = self.compute_spectral_radius(grid)
        optimal_weight = 2.0 / (1.0 + np.sqrt(1.0 - spectral_radius**2))
        
        logger.info(f"Computed optimal Jacobi weight: ω = {optimal_weight:.4f}")
        self.omega = optimal_weight
        
        return optimal_weight
    
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve with optimal weight."""
        if self.auto_weight:
            self.setup_optimal_weight(grid)
        
        return super().solve(grid, operator, rhs, initial_guess, precision_manager)


class AdaptivePrecisionSolver(IterativeSolver):
    """
    Base solver with adaptive precision switching capabilities.
    
    Implements dynamic precision switching strategies:
    - Start with float32 for speed
    - Switch to float64 when convergence slows
    - Monitor convergence rate for precision decisions
    """
    
    def __init__(
        self,
        base_solver: IterativeSolver,
        precision_switch_threshold: float = 0.95,
        convergence_window: int = 5,
        min_iterations_before_switch: int = 10
    ):
        """
        Initialize adaptive precision solver wrapper.
        
        Args:
            base_solver: Underlying iterative solver
            precision_switch_threshold: Convergence rate threshold for switching
            convergence_window: Window size for convergence rate calculation
            min_iterations_before_switch: Minimum iterations before allowing switch
        """
        super().__init__(
            base_solver.max_iterations,
            base_solver.tolerance,
            base_solver.omega,
            base_solver.verbose,
            f"Adaptive{base_solver.name}"
        )
        
        self.base_solver = base_solver
        self.precision_switch_threshold = precision_switch_threshold
        self.convergence_window = convergence_window
        self.min_iterations_before_switch = min_iterations_before_switch
        
        # Tracking for precision decisions
        self.convergence_rates = []
        self.precision_switched = False
    
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """Apply smoothing with base solver."""
        return self.base_solver.smooth(grid, operator, u, rhs, num_iterations)
    
    def solve(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve with adaptive precision switching.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            rhs: Right-hand side
            initial_guess: Initial guess
            precision_manager: Precision manager for switching
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        self.reset()
        
        # Start with single precision if precision manager supports it
        if precision_manager and precision_manager.adaptive:
            original_precision = precision_manager.current_precision
            from ..core.precision import PrecisionLevel
            precision_manager.current_precision = PrecisionLevel.SINGLE
            logger.info("Starting with single precision for speed")
        
        # Initialize solution
        if initial_guess is None:
            u = np.zeros_like(rhs)
        else:
            u = initial_guess.copy()
        
        previous_residual = float('inf')
        
        # Main iteration loop with precision adaptation
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            
            # Apply smoothing
            u = self.smooth(grid, operator, u, rhs, 1)
            
            # Compute residual
            residual = operator.residual(grid, u, rhs)
            residual_norm = grid.l2_norm(residual)
            
            # Track convergence rate
            if previous_residual != float('inf'):
                convergence_rate = residual_norm / previous_residual
                self.convergence_rates.append(convergence_rate)
                
                # Check for precision switching
                if (precision_manager and 
                    not self.precision_switched and 
                    iteration >= self.min_iterations_before_switch):
                    
                    if self._should_switch_precision():
                        from ..core.precision import PrecisionLevel
                        precision_manager.current_precision = PrecisionLevel.DOUBLE
                        
                        # Convert current solution to higher precision
                        u = precision_manager.convert_array(u, PrecisionLevel.DOUBLE)
                        
                        self.precision_switched = True
                        logger.info(f"Switched to double precision at iteration {iteration}")
            
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
                
            previous_residual = residual_norm
        
        self.iterations_performed = iteration
        self.final_residual = residual_norm
        
        # Include precision switching information
        info = self.get_convergence_info()
        info['precision_switched'] = self.precision_switched
        info['switch_iteration'] = (
            self.min_iterations_before_switch + 
            len([r for r in self.convergence_rates[:self.min_iterations_before_switch] 
                 if r >= self.precision_switch_threshold])
            if self.precision_switched else None
        )
        
        return u, info
    
    def _should_switch_precision(self) -> bool:
        """
        Determine if precision should be switched based on convergence rate.
        
        Switch when convergence rate is consistently high over a window.
        """
        if len(self.convergence_rates) < self.convergence_window:
            return False
        
        # Check recent convergence rates
        recent_rates = self.convergence_rates[-self.convergence_window:]
        average_rate = np.mean(recent_rates)
        
        # Switch if convergence is slow
        should_switch = average_rate >= self.precision_switch_threshold
        
        if should_switch:
            logger.debug(f"Convergence rate {average_rate:.3f} >= threshold "
                        f"{self.precision_switch_threshold:.3f}, switching precision")
        
        return should_switch