"""
Corrected Multigrid Solver Implementation
Fixes numerical issues and implements proper mathematical formulation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import time
import logging

from .base import BaseSolver
from .smoothers import GaussSeidelSmoother
from ..core.grid import Grid
from ..operators.laplacian import LaplacianOperator
from ..operators.transfer import RestrictionOperator, ProlongationOperator

if TYPE_CHECKING:
    from ..core.precision import PrecisionManager
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class CorrectedMultigridSolver(BaseSolver):
    """
    Numerically stable multigrid solver with proper mathematical formulation.
    
    Key corrections:
    1. Proper boundary condition handling
    2. Correct operator application and residual computation
    3. Stable grid transfer operations
    4. Proper coarse grid solver with adequate iterations
    5. Correct smoother parameter ordering
    """
    
    def __init__(
        self,
        max_levels: int = 4,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        cycle_type: str = "V",
        pre_smooth_iterations: int = 2,
        post_smooth_iterations: int = 2,
        coarse_tolerance: float = 1e-12,
        coarse_max_iterations: int = 100,
        verbose: bool = False
    ):
        """Initialize corrected multigrid solver."""
        super().__init__(max_iterations, tolerance, verbose, "CorrectedMultigrid")
        
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.pre_smooth_iterations = pre_smooth_iterations
        self.post_smooth_iterations = post_smooth_iterations
        self.coarse_tolerance = coarse_tolerance
        self.coarse_max_iterations = coarse_max_iterations
        
        # Grid hierarchy
        self.grids: List[Grid] = []
        self.operators: List[LaplacianOperator] = []
        self.restriction_op = RestrictionOperator()
        self.prolongation_op = ProlongationOperator()
        
        # Smoothers
        self.smoother = GaussSeidelSmoother(max_iterations=1)
        self.coarse_solver = GaussSeidelSmoother(max_iterations=self.coarse_max_iterations)
        
        logger.info(f"Initialized CorrectedMultigridSolver: {cycle_type}-cycle")
    
    def setup_grid_hierarchy(self, fine_grid: Grid) -> None:
        """Set up multigrid hierarchy with proper grid generation."""
        self.grids.clear()
        self.operators.clear()
        
        # Start with fine grid
        current_grid = fine_grid
        self.grids.append(current_grid)
        
        # Create grid hierarchy
        for level in range(1, self.max_levels):
            # Coarsen grid (ensure odd dimensions for proper nesting)
            coarse_nx = max(5, (current_grid.nx - 1) // 2 + 1)
            coarse_ny = max(5, (current_grid.ny - 1) // 2 + 1)
            
            # Ensure proper domain coverage
            coarse_grid = Grid(
                coarse_nx, coarse_ny, 
                domain=current_grid.domain,
                dtype=current_grid.dtype
            )
            
            self.grids.append(coarse_grid)
            current_grid = coarse_grid
            
            # Stop if grid is too coarse
            if coarse_nx <= 5 or coarse_ny <= 5:
                break
        
        # Set up operators for each level
        for grid in self.grids:
            operator = LaplacianOperator()
            self.operators.append(operator)
        
        logger.info(f"Grid hierarchy created: {len(self.grids)} levels")
        for i, grid in enumerate(self.grids):
            h = 1.0 / (grid.nx - 1)
            logger.info(f"  Level {i}: {grid.nx}×{grid.ny} (h={h:.4f})")
    
    def solve(
        self,
        initial_guess: np.ndarray,
        rhs: np.ndarray,
        grid: Grid,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> Dict[str, Any]:
        """
        Solve using corrected multigrid method.
        
        Args:
            initial_guess: Initial guess for solution
            rhs: Right-hand side vector
            grid: Fine grid
            precision_manager: Optional precision manager
            
        Returns:
            Dictionary with solution and convergence information
        """
        # Set up hierarchy if needed
        if not self.grids or self.grids[0].shape != grid.shape:
            self.setup_grid_hierarchy(grid)
        
        self.reset()
        
        # Initialize solution with proper boundary conditions
        u = initial_guess.copy()
        self._apply_boundary_conditions(u, grid)
        
        # Store initial residual
        initial_residual = self._compute_residual_norm(u, rhs, grid)
        logger.info(f"Initial residual: {initial_residual:.2e}")
        
        residual_history = [initial_residual]
        
        # Main iteration loop
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            
            # Update precision if using adaptive management
            if precision_manager:
                should_promote = precision_manager.should_promote_precision(
                    residual_history, precision_manager.current_precision
                )
                if should_promote:
                    precision_manager.current_precision = precision_manager.PrecisionLevel.DOUBLE
            
            # Apply multigrid V-cycle
            u = self._v_cycle(u, rhs, 0, precision_manager)
            
            # Compute residual norm
            residual_norm = self._compute_residual_norm(u, rhs, grid)
            residual_history.append(residual_norm)
            
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
        
        return {
            'solution': u,
            'converged': self.converged,
            'iterations': self.iterations_performed,
            'final_residual': self.final_residual,
            'residual_history': residual_history,
            'convergence_info': self.get_convergence_info()
        }
    
    def _v_cycle(
        self, 
        u: np.ndarray, 
        rhs: np.ndarray, 
        level: int,
        precision_manager: Optional['PrecisionManager'] = None
    ) -> np.ndarray:
        """
        Apply one V-cycle with proper mathematical formulation.
        
        Args:
            u: Current solution approximation
            rhs: Right-hand side (only used at finest level)
            level: Current grid level (0 = finest)
            precision_manager: Optional precision manager
            
        Returns:
            Improved solution approximation
        """
        grid = self.grids[level]
        operator = self.operators[level]
        
        # Base case: coarsest grid
        if level == len(self.grids) - 1:
            return self._solve_coarsest(u, rhs, level)
        
        # Pre-smoothing
        if self.pre_smooth_iterations > 0:
            for _ in range(self.pre_smooth_iterations):
                u = self._gauss_seidel_iteration(u, rhs, grid, operator)
                self._apply_boundary_conditions(u, grid)
        
        # Compute residual: r = f - Au
        residual = self._compute_residual(u, rhs, grid, operator)
        
        # Restrict residual to coarse grid
        coarse_grid = self.grids[level + 1]
        coarse_residual = self._restrict(residual, grid, coarse_grid)
        
        # Solve coarse grid correction equation: A_2h * e_2h = r_2h
        coarse_correction = np.zeros_like(coarse_residual)
        coarse_correction = self._v_cycle(
            coarse_correction, coarse_residual, level + 1, precision_manager
        )
        
        # Prolongate correction to fine grid
        fine_correction = self._prolongate(coarse_correction, coarse_grid, grid)
        
        # Apply correction
        u = u + fine_correction
        self._apply_boundary_conditions(u, grid)
        
        # Post-smoothing
        if self.post_smooth_iterations > 0:
            for _ in range(self.post_smooth_iterations):
                u = self._gauss_seidel_iteration(u, rhs, grid, operator)
                self._apply_boundary_conditions(u, grid)
        
        return u
    
    def _gauss_seidel_iteration(
        self, 
        u: np.ndarray, 
        rhs: np.ndarray, 
        grid: Grid, 
        operator: LaplacianOperator
    ) -> np.ndarray:
        """
        Single Gauss-Seidel iteration with proper formulation.
        
        For 2D Poisson equation: -∇²u = f
        Discrete form: -(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])/h² = f[i,j]
        
        Solving for u[i,j]:
        u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h²*f[i,j]) / 4
        """
        h = grid.hx  # Assume square grid: hx = hy
        u_new = u.copy()
        
        # Update interior points using Gauss-Seidel
        for i in range(1, grid.nx - 1):
            for j in range(1, grid.ny - 1):
                # Gauss-Seidel: use most recent values
                u_new[i, j] = 0.25 * (
                    u_new[i-1, j] + u[i+1, j] + 
                    u_new[i, j-1] + u[i, j+1] + 
                    h**2 * rhs[i, j]
                )
        
        return u_new
    
    def _compute_residual(
        self, 
        u: np.ndarray, 
        rhs: np.ndarray, 
        grid: Grid, 
        operator: LaplacianOperator
    ) -> np.ndarray:
        """Compute residual: r = f - Au"""
        Au = self._apply_laplacian(u, grid)
        residual = rhs - Au
        
        # Set residual to zero at boundaries (homogeneous Dirichlet)
        residual[0, :] = residual[-1, :] = 0.0
        residual[:, 0] = residual[:, -1] = 0.0
        
        return residual
    
    def _apply_laplacian(self, u: np.ndarray, grid: Grid) -> np.ndarray:
        """Apply discrete Laplacian operator."""
        h = grid.hx
        Au = np.zeros_like(u)
        
        # Interior points: -∇²u ≈ -(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])/h²
        for i in range(1, grid.nx - 1):
            for j in range(1, grid.ny - 1):
                Au[i, j] = -(
                    u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j]
                ) / h**2
        
        return Au
    
    def _compute_residual_norm(self, u: np.ndarray, rhs: np.ndarray, grid: Grid) -> float:
        """Compute L2 norm of residual."""
        residual = self._compute_residual(u, rhs, grid, self.operators[0])
        
        # Compute L2 norm over interior points only
        interior_residual = residual[1:-1, 1:-1]
        return np.linalg.norm(interior_residual)
    
    def _restrict(self, fine_array: np.ndarray, fine_grid: Grid, coarse_grid: Grid) -> np.ndarray:
        """Full-weighting restriction operator."""
        coarse_array = np.zeros((coarse_grid.nx, coarse_grid.ny))
        
        # Full-weighting stencil: [1 2 1; 2 4 2; 1 2 1] / 16
        for i in range(1, coarse_grid.nx - 1):
            for j in range(1, coarse_grid.ny - 1):
                # Map to fine grid indices
                fi, fj = 2*i, 2*j
                
                if fi < fine_grid.nx - 1 and fj < fine_grid.ny - 1:
                    coarse_array[i, j] = (
                        fine_array[fi-1, fj-1] + 2*fine_array[fi-1, fj] + fine_array[fi-1, fj+1] +
                        2*fine_array[fi, fj-1] + 4*fine_array[fi, fj] + 2*fine_array[fi, fj+1] +
                        fine_array[fi+1, fj-1] + 2*fine_array[fi+1, fj] + fine_array[fi+1, fj+1]
                    ) / 16.0
        
        return coarse_array
    
    def _prolongate(self, coarse_array: np.ndarray, coarse_grid: Grid, fine_grid: Grid) -> np.ndarray:
        """Bilinear prolongation operator."""
        fine_array = np.zeros((fine_grid.nx, fine_grid.ny))
        
        # Bilinear interpolation
        for i in range(coarse_grid.nx):
            for j in range(coarse_grid.ny):
                # Direct injection to corresponding fine grid points
                fi, fj = 2*i, 2*j
                if fi < fine_grid.nx and fj < fine_grid.ny:
                    fine_array[fi, fj] = coarse_array[i, j]
                
                # Interpolate to intermediate points
                if fi + 1 < fine_grid.nx and fj < fine_grid.ny and i + 1 < coarse_grid.nx:
                    fine_array[fi + 1, fj] = 0.5 * (coarse_array[i, j] + coarse_array[i + 1, j])
                
                if fi < fine_grid.nx and fj + 1 < fine_grid.ny and j + 1 < coarse_grid.ny:
                    fine_array[fi, fj + 1] = 0.5 * (coarse_array[i, j] + coarse_array[i, j + 1])
                
                # Bilinear for corner points
                if (fi + 1 < fine_grid.nx and fj + 1 < fine_grid.ny and 
                    i + 1 < coarse_grid.nx and j + 1 < coarse_grid.ny):
                    fine_array[fi + 1, fj + 1] = 0.25 * (
                        coarse_array[i, j] + coarse_array[i + 1, j] +
                        coarse_array[i, j + 1] + coarse_array[i + 1, j + 1]
                    )
        
        return fine_array
    
    def _solve_coarsest(self, u: np.ndarray, rhs: np.ndarray, level: int) -> np.ndarray:
        """Solve coarsest grid problem accurately."""
        grid = self.grids[level]
        operator = self.operators[level]
        
        # Use many iterations for accurate coarse grid solution
        for iteration in range(self.coarse_max_iterations):
            u_old = u.copy()
            u = self._gauss_seidel_iteration(u, rhs, grid, operator)
            self._apply_boundary_conditions(u, grid)
            
            # Check convergence
            residual_norm = self._compute_residual_norm(u, rhs, grid)
            if residual_norm < self.coarse_tolerance:
                logger.debug(f"Coarse solver converged in {iteration + 1} iterations")
                break
            
            # Check for convergence based on solution change
            solution_change = np.linalg.norm(u - u_old)
            if solution_change < self.coarse_tolerance:
                break
        else:
            logger.warning(f"Coarse solver reached max iterations ({self.coarse_max_iterations})")
        
        return u
    
    def _apply_boundary_conditions(self, u: np.ndarray, grid: Grid) -> None:
        """Apply homogeneous Dirichlet boundary conditions."""
        u[0, :] = 0.0    # Bottom boundary
        u[-1, :] = 0.0   # Top boundary
        u[:, 0] = 0.0    # Left boundary
        u[:, -1] = 0.0   # Right boundary
    
    def create_test_problem(self, grid: Grid, problem_type: str = "manufactured") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create test problems with known analytical solutions.
        
        Args:
            grid: Computational grid
            problem_type: Type of test problem
            
        Returns:
            Tuple of (rhs, exact_solution)
        """
        x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
        y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if problem_type == "manufactured":
            # u(x,y) = sin(π*x)*sin(π*y)
            # -∇²u = 2π²*sin(π*x)*sin(π*y)
            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
            rhs = 2 * np.pi**2 * u_exact
        
        elif problem_type == "polynomial":
            # u(x,y) = x*(1-x)*y*(1-y) (satisfies homogeneous BCs)
            u_exact = X * (1 - X) * Y * (1 - Y)
            # -∇²u = 2*x*(1-x) + 2*y*(1-y)
            rhs = 2 * X * (1 - X) + 2 * Y * (1 - Y)
        
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        return rhs, u_exact


def test_corrected_solver():
    """Test the corrected multigrid solver."""
    import matplotlib.pyplot as plt
    
    # Create test problem
    grid = Grid(33, 33, domain=(0, 1, 0, 1))
    solver = CorrectedMultigridSolver(
        max_levels=4,
        max_iterations=20,
        tolerance=1e-10,
        verbose=True
    )
    
    # Create manufactured solution problem
    rhs, u_exact = solver.create_test_problem(grid, "manufactured")
    
    # Solve
    initial_guess = np.zeros_like(rhs)
    result = solver.solve(initial_guess, rhs, grid)
    
    u_computed = result['solution']
    
    # Compute error
    error = u_computed - u_exact
    l2_error = np.linalg.norm(error[1:-1, 1:-1])
    max_error = np.max(np.abs(error[1:-1, 1:-1]))
    
    print(f"\n=== CORRECTED MULTIGRID SOLVER TEST ===")
    print(f"Grid size: {grid.nx}×{grid.ny}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final residual: {result['final_residual']:.2e}")
    print(f"L2 error: {l2_error:.2e}")
    print(f"Max error: {max_error:.2e}")
    
    # Compute convergence rate
    residuals = result['residual_history']
    if len(residuals) >= 3:
        factors = []
        for i in range(1, min(len(residuals), 6)):
            if residuals[i-1] > 0:
                factor = residuals[i] / residuals[i-1]
                factors.append(factor)
        
        if factors:
            avg_factor = np.exp(np.mean(np.log(factors)))
            print(f"Average convergence factor: {avg_factor:.4f}")
    
    return result


if __name__ == "__main__":
    # Run test if called directly
    test_corrected_solver()