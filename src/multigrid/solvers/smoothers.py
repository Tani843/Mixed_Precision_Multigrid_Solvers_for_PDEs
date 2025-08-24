"""Smoothing operators for multigrid methods."""

import numpy as np
from typing import TYPE_CHECKING
import logging

from .base import IterativeSolver

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class JacobiSmoother(IterativeSolver):
    """
    Jacobi smoother for multigrid methods.
    
    Updates: u_new[i,j] = (rhs[i,j] - A_off*u_old[i,j]) / A_diag[i,j]
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 2.0/3.0,
        verbose: bool = False
    ):
        """
        Initialize Jacobi smoother.
        
        Args:
            max_iterations: Maximum iterations for solve
            tolerance: Convergence tolerance
            relaxation_parameter: Relaxation parameter (optimal ≈ 2/3 for Laplacian)
            verbose: Enable verbose output
        """
        super().__init__(max_iterations, tolerance, relaxation_parameter, verbose, "Jacobi")
    
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """
        Apply Jacobi smoothing iterations.
        
        Args:
            grid: Computational grid
            operator: Linear operator (assumed to be Laplacian-like)
            u: Current solution estimate
            rhs: Right-hand side
            num_iterations: Number of smoothing iterations
            
        Returns:
            Smoothed solution
        """
        u_smooth = u.copy()
        
        # Diagonal coefficient for discrete Laplacian
        diag_coeff = -2.0 / grid.hx**2 - 2.0 / grid.hy**2
        
        for _ in range(num_iterations):
            u_old = u_smooth.copy()
            
            # Apply Jacobi update to interior points
            for i in range(1, grid.nx - 1):
                for j in range(1, grid.ny - 1):
                    # Off-diagonal terms
                    neighbors = (
                        (u_old[i+1, j] + u_old[i-1, j]) / grid.hx**2 +
                        (u_old[i, j+1] + u_old[i, j-1]) / grid.hy**2
                    )
                    
                    # Jacobi update
                    u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                    
                    # Apply relaxation
                    u_smooth[i, j] = (1 - self.omega) * u_old[i, j] + self.omega * u_new
        
        logger.debug(f"Applied {num_iterations} Jacobi smoothing iterations")
        return u_smooth


class GaussSeidelSmoother(IterativeSolver):
    """
    Gauss-Seidel smoother for multigrid methods.
    
    Updates points in lexicographic order using most recent values.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 1.0,
        verbose: bool = False,
        red_black: bool = False
    ):
        """
        Initialize Gauss-Seidel smoother.
        
        Args:
            max_iterations: Maximum iterations for solve
            tolerance: Convergence tolerance
            relaxation_parameter: SOR parameter (1.0 = pure Gauss-Seidel)
            verbose: Enable verbose output
            red_black: Use red-black ordering for parallelization
        """
        super().__init__(max_iterations, tolerance, relaxation_parameter, verbose, "Gauss-Seidel")
        self.red_black = red_black
    
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """
        Apply Gauss-Seidel smoothing iterations.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            u: Current solution estimate
            rhs: Right-hand side
            num_iterations: Number of smoothing iterations
            
        Returns:
            Smoothed solution
        """
        u_smooth = u.copy()
        
        # Diagonal coefficient for discrete Laplacian
        diag_coeff = -2.0 / grid.hx**2 - 2.0 / grid.hy**2
        
        for _ in range(num_iterations):
            if self.red_black:
                u_smooth = self._red_black_sweep(grid, u_smooth, rhs, diag_coeff)
            else:
                u_smooth = self._lexicographic_sweep(grid, u_smooth, rhs, diag_coeff)
        
        logger.debug(f"Applied {num_iterations} Gauss-Seidel smoothing iterations "
                    f"({'red-black' if self.red_black else 'lexicographic'})")
        return u_smooth
    
    def _lexicographic_sweep(
        self, 
        grid: 'Grid', 
        u: np.ndarray, 
        rhs: np.ndarray, 
        diag_coeff: float
    ) -> np.ndarray:
        """Apply lexicographic Gauss-Seidel sweep."""
        for i in range(1, grid.nx - 1):
            for j in range(1, grid.ny - 1):
                # Use most recent values (already updated in current sweep)
                neighbors = (
                    (u[i+1, j] + u[i-1, j]) / grid.hx**2 +
                    (u[i, j+1] + u[i, j-1]) / grid.hy**2
                )
                
                # Gauss-Seidel update with SOR
                u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                u[i, j] = (1 - self.omega) * u[i, j] + self.omega * u_new
        
        return u
    
    def _red_black_sweep(
        self, 
        grid: 'Grid', 
        u: np.ndarray, 
        rhs: np.ndarray, 
        diag_coeff: float
    ) -> np.ndarray:
        """Apply red-black Gauss-Seidel sweep."""
        # Red points: (i + j) even
        for i in range(1, grid.nx - 1):
            for j in range(1, grid.ny - 1):
                if (i + j) % 2 == 0:  # Red point
                    neighbors = (
                        (u[i+1, j] + u[i-1, j]) / grid.hx**2 +
                        (u[i, j+1] + u[i, j-1]) / grid.hy**2
                    )
                    
                    u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                    u[i, j] = (1 - self.omega) * u[i, j] + self.omega * u_new
        
        # Black points: (i + j) odd
        for i in range(1, grid.nx - 1):
            for j in range(1, grid.ny - 1):
                if (i + j) % 2 == 1:  # Black point
                    neighbors = (
                        (u[i+1, j] + u[i-1, j]) / grid.hx**2 +
                        (u[i, j+1] + u[i, j-1]) / grid.hy**2
                    )
                    
                    u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                    u[i, j] = (1 - self.omega) * u[i, j] + self.omega * u_new
        
        return u


class WeightedJacobiSmoother(JacobiSmoother):
    """Weighted Jacobi smoother with optimal damping parameter."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False
    ):
        """
        Initialize weighted Jacobi smoother with optimal parameter.
        
        For 2D Laplacian, optimal weight is 4/5 = 0.8
        """
        super().__init__(max_iterations, tolerance, 4.0/5.0, verbose)
        self.name = "WeightedJacobi"


class SymmetricGaussSeidelSmoother(GaussSeidelSmoother):
    """
    Symmetric Gauss-Seidel smoother.
    
    Applies forward sweep followed by backward sweep for symmetry.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        relaxation_parameter: float = 1.0,
        verbose: bool = False
    ):
        """Initialize symmetric Gauss-Seidel smoother."""
        super().__init__(max_iterations, tolerance, relaxation_parameter, verbose)
        self.name = "SymmetricGauss-Seidel"
    
    def smooth(
        self, 
        grid: 'Grid', 
        operator: 'BaseOperator',
        u: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 1
    ) -> np.ndarray:
        """Apply symmetric Gauss-Seidel smoothing."""
        u_smooth = u.copy()
        diag_coeff = -2.0 / grid.hx**2 - 2.0 / grid.hy**2
        
        for _ in range(num_iterations):
            # Forward sweep
            u_smooth = self._lexicographic_sweep(grid, u_smooth, rhs, diag_coeff)
            # Backward sweep
            u_smooth = self._backward_sweep(grid, u_smooth, rhs, diag_coeff)
        
        logger.debug(f"Applied {num_iterations} symmetric Gauss-Seidel iterations")
        return u_smooth
    
    def _backward_sweep(
        self, 
        grid: 'Grid', 
        u: np.ndarray, 
        rhs: np.ndarray, 
        diag_coeff: float
    ) -> np.ndarray:
        """Apply backward lexicographic sweep."""
        for i in range(grid.nx - 2, 0, -1):
            for j in range(grid.ny - 2, 0, -1):
                neighbors = (
                    (u[i+1, j] + u[i-1, j]) / grid.hx**2 +
                    (u[i, j+1] + u[i, j-1]) / grid.hy**2
                )
                
                u_new = (rhs[i, j] + neighbors) / (-diag_coeff)
                u[i, j] = (1 - self.omega) * u[i, j] + self.omega * u_new
        
        return u