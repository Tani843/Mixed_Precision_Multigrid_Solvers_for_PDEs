"""Grid transfer operators for multigrid methods."""

import numpy as np
from typing import TYPE_CHECKING, Tuple
import logging

from .base import BaseOperator

if TYPE_CHECKING:
    from ..core.grid import Grid

logger = logging.getLogger(__name__)


class RestrictionOperator(BaseOperator):
    """
    Restriction operator: transfers data from fine grid to coarse grid.
    
    Implements I_{2h}^h using weighted averaging (full weighting).
    """
    
    def __init__(self, method: str = "full_weighting"):
        """
        Initialize restriction operator.
        
        Args:
            method: Restriction method ('injection', 'full_weighting', 'half_weighting')
        """
        super().__init__(f"Restriction({method})")
        self.method = method
        
        if method not in ['injection', 'full_weighting', 'half_weighting']:
            raise ValueError(f"Unknown restriction method: {method}")
    
    def can_apply(self, fine_grid: 'Grid', coarse_grid: 'Grid') -> bool:
        """
        Check if restriction can be applied between grids.
        
        Args:
            fine_grid: Source (fine) grid
            coarse_grid: Target (coarse) grid
            
        Returns:
            True if grids are compatible for restriction
        """
        # Check if coarse grid has half the resolution
        expected_coarse_nx = (fine_grid.nx - 1) // 2 + 1
        expected_coarse_ny = (fine_grid.ny - 1) // 2 + 1
        
        return (coarse_grid.nx == expected_coarse_nx and 
                coarse_grid.ny == expected_coarse_ny)
    
    def apply(self, fine_grid: 'Grid', field: np.ndarray, coarse_grid: 'Grid') -> np.ndarray:
        """
        Apply restriction operator.
        
        Args:
            fine_grid: Source grid
            field: Field on fine grid
            coarse_grid: Target grid
            
        Returns:
            Restricted field on coarse grid
        """
        if not self.can_apply(fine_grid, coarse_grid):
            raise ValueError(f"Cannot restrict from {fine_grid.shape} to {coarse_grid.shape}")
        
        if field.shape != fine_grid.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match fine grid {fine_grid.shape}")
        
        coarse_field = np.zeros(coarse_grid.shape, dtype=coarse_grid.dtype)
        
        if self.method == "injection":
            coarse_field = self._injection_restriction(field, fine_grid, coarse_grid)
        elif self.method == "full_weighting":
            coarse_field = self._full_weighting_restriction(field, fine_grid, coarse_grid)
        elif self.method == "half_weighting":
            coarse_field = self._half_weighting_restriction(field, fine_grid, coarse_grid)
        
        logger.debug(f"Applied {self.method} restriction: {fine_grid.shape} -> {coarse_grid.shape}")
        return coarse_field
    
    def _injection_restriction(self, field: np.ndarray, fine_grid: 'Grid', coarse_grid: 'Grid') -> np.ndarray:
        """Simple injection: take every other point."""
        coarse_field = np.zeros(coarse_grid.shape, dtype=coarse_grid.dtype)
        
        for i in range(coarse_grid.nx):
            for j in range(coarse_grid.ny):
                fine_i = 2 * i
                fine_j = 2 * j
                
                # Ensure we don't go out of bounds
                fine_i = min(fine_i, fine_grid.nx - 1)
                fine_j = min(fine_j, fine_grid.ny - 1)
                
                coarse_field[i, j] = field[fine_i, fine_j]
        
        return coarse_field
    
    def _full_weighting_restriction(self, field: np.ndarray, fine_grid: 'Grid', coarse_grid: 'Grid') -> np.ndarray:
        """Full weighting: 9-point stencil with weights [1/16, 1/8, 1/4, 1/2, 1]."""
        coarse_field = np.zeros(coarse_grid.shape, dtype=coarse_grid.dtype)
        
        for i in range(coarse_grid.nx):
            for j in range(coarse_grid.ny):
                fine_i = 2 * i
                fine_j = 2 * j
                
                if i == 0 or i == coarse_grid.nx - 1 or j == 0 or j == coarse_grid.ny - 1:
                    # Boundary points: use injection
                    fine_i = min(fine_i, fine_grid.nx - 1)
                    fine_j = min(fine_j, fine_grid.ny - 1)
                    coarse_field[i, j] = field[fine_i, fine_j]
                else:
                    # Interior points: use 9-point stencil
                    coarse_field[i, j] = (
                        1.0/16.0 * (field[fine_i-1, fine_j-1] + field[fine_i-1, fine_j+1] + 
                                   field[fine_i+1, fine_j-1] + field[fine_i+1, fine_j+1]) +
                        1.0/8.0 * (field[fine_i-1, fine_j] + field[fine_i+1, fine_j] +
                                  field[fine_i, fine_j-1] + field[fine_i, fine_j+1]) +
                        1.0/4.0 * field[fine_i, fine_j]
                    )
        
        return coarse_field
    
    def _half_weighting_restriction(self, field: np.ndarray, fine_grid: 'Grid', coarse_grid: 'Grid') -> np.ndarray:
        """Half weighting: 5-point stencil."""
        coarse_field = np.zeros(coarse_grid.shape, dtype=coarse_grid.dtype)
        
        for i in range(coarse_grid.nx):
            for j in range(coarse_grid.ny):
                fine_i = 2 * i
                fine_j = 2 * j
                
                if i == 0 or i == coarse_grid.nx - 1 or j == 0 or j == coarse_grid.ny - 1:
                    # Boundary points: use injection
                    fine_i = min(fine_i, fine_grid.nx - 1)
                    fine_j = min(fine_j, fine_grid.ny - 1)
                    coarse_field[i, j] = field[fine_i, fine_j]
                else:
                    # Interior points: use 5-point stencil
                    coarse_field[i, j] = (
                        1.0/8.0 * (field[fine_i-1, fine_j] + field[fine_i+1, fine_j] +
                                  field[fine_i, fine_j-1] + field[fine_i, fine_j+1]) +
                        1.0/2.0 * field[fine_i, fine_j]
                    )
        
        return coarse_field


class ProlongationOperator(BaseOperator):
    """
    Prolongation operator: transfers data from coarse grid to fine grid.
    
    Implements I_h^{2h} using bilinear interpolation.
    """
    
    def __init__(self, method: str = "bilinear"):
        """
        Initialize prolongation operator.
        
        Args:
            method: Prolongation method ('injection', 'bilinear')
        """
        super().__init__(f"Prolongation({method})")
        self.method = method
        
        if method not in ['injection', 'bilinear']:
            raise ValueError(f"Unknown prolongation method: {method}")
    
    def can_apply(self, coarse_grid: 'Grid', fine_grid: 'Grid') -> bool:
        """
        Check if prolongation can be applied between grids.
        
        Args:
            coarse_grid: Source (coarse) grid
            fine_grid: Target (fine) grid
            
        Returns:
            True if grids are compatible for prolongation
        """
        # Check if fine grid has double the resolution
        expected_fine_nx = 2 * (coarse_grid.nx - 1) + 1
        expected_fine_ny = 2 * (coarse_grid.ny - 1) + 1
        
        return (fine_grid.nx == expected_fine_nx and 
                fine_grid.ny == expected_fine_ny)
    
    def apply(self, coarse_grid: 'Grid', field: np.ndarray, fine_grid: 'Grid') -> np.ndarray:
        """
        Apply prolongation operator.
        
        Args:
            coarse_grid: Source grid
            field: Field on coarse grid
            fine_grid: Target grid
            
        Returns:
            Prolongated field on fine grid
        """
        if not self.can_apply(coarse_grid, fine_grid):
            raise ValueError(f"Cannot prolongate from {coarse_grid.shape} to {fine_grid.shape}")
        
        if field.shape != coarse_grid.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match coarse grid {coarse_grid.shape}")
        
        fine_field = np.zeros(fine_grid.shape, dtype=fine_grid.dtype)
        
        if self.method == "injection":
            fine_field = self._injection_prolongation(field, coarse_grid, fine_grid)
        elif self.method == "bilinear":
            fine_field = self._bilinear_prolongation(field, coarse_grid, fine_grid)
        
        logger.debug(f"Applied {self.method} prolongation: {coarse_grid.shape} -> {fine_grid.shape}")
        return fine_field
    
    def _injection_prolongation(self, field: np.ndarray, coarse_grid: 'Grid', fine_grid: 'Grid') -> np.ndarray:
        """Simple injection: copy values to corresponding points."""
        fine_field = np.zeros(fine_grid.shape, dtype=fine_grid.dtype)
        
        for i in range(coarse_grid.nx):
            for j in range(coarse_grid.ny):
                fine_i = 2 * i
                fine_j = 2 * j
                
                # Ensure we don't go out of bounds
                fine_i = min(fine_i, fine_grid.nx - 1)
                fine_j = min(fine_j, fine_grid.ny - 1)
                
                fine_field[fine_i, fine_j] = field[i, j]
        
        return fine_field
    
    def _bilinear_prolongation(self, field: np.ndarray, coarse_grid: 'Grid', fine_grid: 'Grid') -> np.ndarray:
        """Bilinear interpolation prolongation."""
        fine_field = np.zeros(fine_grid.shape, dtype=fine_grid.dtype)
        
        # Copy coarse grid points directly
        for i in range(coarse_grid.nx):
            for j in range(coarse_grid.ny):
                fine_i = 2 * i
                fine_j = 2 * j
                
                fine_i = min(fine_i, fine_grid.nx - 1)
                fine_j = min(fine_j, fine_grid.ny - 1)
                
                fine_field[fine_i, fine_j] = field[i, j]
        
        # Interpolate horizontally (odd i, even j)
        for i in range(1, fine_grid.nx - 1, 2):
            for j in range(0, fine_grid.ny, 2):
                if j < fine_grid.ny - 1:
                    fine_field[i, j] = 0.5 * (fine_field[i-1, j] + fine_field[i+1, j])
        
        # Interpolate vertically (even i, odd j)
        for i in range(0, fine_grid.nx, 2):
            for j in range(1, fine_grid.ny - 1, 2):
                if i < fine_grid.nx - 1:
                    fine_field[i, j] = 0.5 * (fine_field[i, j-1] + fine_field[i, j+1])
        
        # Interpolate diagonally (odd i, odd j)
        for i in range(1, fine_grid.nx - 1, 2):
            for j in range(1, fine_grid.ny - 1, 2):
                fine_field[i, j] = 0.25 * (fine_field[i-1, j-1] + fine_field[i-1, j+1] + 
                                          fine_field[i+1, j-1] + fine_field[i+1, j+1])
        
        return fine_field