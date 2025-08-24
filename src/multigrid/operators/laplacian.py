"""Discrete Laplacian operator implementation."""

import numpy as np
from typing import TYPE_CHECKING, Optional
import logging

from .base import BaseOperator

if TYPE_CHECKING:
    from ..core.grid import Grid

logger = logging.getLogger(__name__)


class LaplacianOperator(BaseOperator):
    """
    Discrete Laplacian operator using five-point stencil.
    
    Implements: ∇²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j})/h²
    """
    
    def __init__(self, coefficient: float = 1.0):
        """
        Initialize Laplacian operator.
        
        Args:
            coefficient: Coefficient for the Laplacian (default: 1.0)
        """
        super().__init__(f"Laplacian(coeff={coefficient})")
        self.coefficient = coefficient
    
    def can_apply(self, grid: 'Grid') -> bool:
        """
        Check if Laplacian can be applied to the grid.
        
        Args:
            grid: Grid to check
            
        Returns:
            True if grid has at least 3 points in each direction
        """
        return grid.nx >= 3 and grid.ny >= 3
    
    def apply(self, grid: 'Grid', field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply discrete Laplacian operator.
        
        Args:
            grid: Computational grid
            field: Input field (default: grid.values)
            
        Returns:
            Laplacian of the field
        """
        if not self.can_apply(grid):
            raise ValueError(f"Cannot apply Laplacian to grid {grid.shape}")
        
        if field is None:
            field = grid.values
            
        if field.shape != grid.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {grid.shape}")
        
        result = np.zeros_like(field)
        
        # Apply five-point stencil to interior points
        i_slice = slice(1, -1)
        j_slice = slice(1, -1)
        
        # For uniform grid spacing, use average of hx² and hy²
        h_sq = (grid.hx**2 + grid.hy**2) / 2.0
        
        result[i_slice, j_slice] = self.coefficient * (
            (field[2:, 1:-1] + field[:-2, 1:-1]) / grid.hx**2 +
            (field[1:-1, 2:] + field[1:-1, :-2]) / grid.hy**2 -
            field[i_slice, j_slice] * (2.0/grid.hx**2 + 2.0/grid.hy**2)
        )
        
        logger.debug(f"Applied Laplacian operator with coefficient {self.coefficient}")
        return result
    
    def apply_stencil(self, grid: 'Grid', field: np.ndarray, i: int, j: int) -> float:
        """
        Apply Laplacian stencil at a single point.
        
        Args:
            grid: Computational grid
            field: Input field
            i, j: Grid indices
            
        Returns:
            Laplacian value at point (i, j)
        """
        if i < 1 or i >= grid.nx - 1 or j < 1 or j >= grid.ny - 1:
            raise ValueError(f"Point ({i}, {j}) is not an interior point")
        
        laplacian = self.coefficient * (
            (field[i+1, j] + field[i-1, j]) / grid.hx**2 +
            (field[i, j+1] + field[i, j-1]) / grid.hy**2 -
            field[i, j] * (2.0/grid.hx**2 + 2.0/grid.hy**2)
        )
        
        return laplacian
    
    def residual(self, grid: 'Grid', u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Compute residual r = f - Au.
        
        Args:
            grid: Computational grid
            u: Approximate solution
            f: Right-hand side
            
        Returns:
            Residual field
        """
        Au = self.apply(grid, u)
        residual = f - Au
        
        # Store residual in grid for debugging
        grid.residual = residual.copy()
        
        logger.debug(f"Computed residual: L2 norm = {grid.l2_norm(residual):.6e}")
        return residual
    
    def eigenvalues_1d(self, n: int, h: float) -> np.ndarray:
        """
        Compute eigenvalues of 1D Laplacian operator.
        
        Args:
            n: Number of interior grid points
            h: Grid spacing
            
        Returns:
            Array of eigenvalues
        """
        k = np.arange(1, n + 1)
        eigenvals = self.coefficient * (-4.0 / h**2) * np.sin(k * np.pi / (2 * (n + 1)))**2
        return eigenvals
    
    def condition_number(self, grid: 'Grid') -> float:
        """
        Estimate condition number of the discrete Laplacian.
        
        Args:
            grid: Computational grid
            
        Returns:
            Condition number estimate
        """
        # Use minimum dimension for conservative estimate
        n = min(grid.nx - 2, grid.ny - 2)  # Interior points
        h = max(grid.hx, grid.hy)  # Maximum spacing
        
        eigenvals = self.eigenvalues_1d(n, h)
        cond_num = np.abs(eigenvals[-1] / eigenvals[0])
        
        logger.debug(f"Condition number estimate: {cond_num:.2e}")
        return cond_num