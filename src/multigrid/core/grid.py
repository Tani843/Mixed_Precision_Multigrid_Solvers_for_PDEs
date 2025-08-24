"""Grid hierarchy implementation for multigrid methods."""

import numpy as np
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class Grid:
    """
    Represents a computational grid for finite difference methods.
    
    This class manages grid hierarchies, boundary conditions, and provides
    fundamental grid operations required for multigrid methods.
    """
    
    def __init__(
        self, 
        nx: int, 
        ny: int, 
        domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        dtype: np.dtype = np.float64
    ):
        """
        Initialize a computational grid.
        
        Args:
            nx: Number of grid points in x-direction (interior points)
            ny: Number of grid points in y-direction (interior points) 
            domain: Domain boundaries (x_min, x_max, y_min, y_max)
            dtype: Data type for grid values (float32 or float64)
        """
        if nx < 3 or ny < 3:
            raise ValueError("Grid must have at least 3 points in each direction")
        
        self.nx = nx
        self.ny = ny
        self.domain = domain
        self.dtype = dtype
        
        # Calculate grid spacing
        self.hx = (domain[1] - domain[0]) / (nx - 1)
        self.hy = (domain[3] - domain[2]) / (ny - 1)
        self.h = min(self.hx, self.hy)  # Minimum spacing for stability
        
        # Grid coordinates
        self.x = np.linspace(domain[0], domain[1], nx, dtype=dtype)
        self.y = np.linspace(domain[2], domain[3], ny, dtype=dtype)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Grid values (including ghost cells for boundary conditions)
        self.values = np.zeros((nx, ny), dtype=dtype)
        
        # Residual storage
        self.residual = np.zeros((nx, ny), dtype=dtype)
        
        logger.info(f"Created grid: {nx}x{ny}, h=({self.hx:.6f}, {self.hy:.6f}), dtype={dtype}")
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return grid shape (nx, ny)."""
        return (self.nx, self.ny)
    
    @property
    def size(self) -> int:
        """Return total number of grid points."""
        return self.nx * self.ny
    
    def interior_slice(self) -> Tuple[slice, slice]:
        """Return slice for interior grid points (excluding boundaries)."""
        return (slice(1, -1), slice(1, -1))
    
    def boundary_slice(self, side: str) -> Tuple[slice, slice]:
        """
        Return slice for boundary grid points.
        
        Args:
            side: Boundary side ('left', 'right', 'bottom', 'top')
        """
        if side == 'left':
            return (slice(0, 1), slice(None))
        elif side == 'right':
            return (slice(-1, None), slice(None))
        elif side == 'bottom':
            return (slice(None), slice(0, 1))
        elif side == 'top':
            return (slice(None), slice(-1, None))
        else:
            raise ValueError(f"Unknown boundary side: {side}")
    
    def apply_dirichlet_bc(
        self, 
        value: Union[float, np.ndarray], 
        side: Optional[str] = None
    ) -> None:
        """
        Apply Dirichlet boundary conditions.
        
        Args:
            value: Boundary value(s)
            side: Boundary side ('left', 'right', 'bottom', 'top', 'all')
        """
        if side is None or side == 'all':
            # Apply to all boundaries
            for boundary in ['left', 'right', 'bottom', 'top']:
                boundary_slice = self.boundary_slice(boundary)
                self.values[boundary_slice] = value
        else:
            boundary_slice = self.boundary_slice(side)
            self.values[boundary_slice] = value
        
        logger.debug(f"Applied Dirichlet BC: value={value}, side={side}")
    
    def apply_neumann_bc(
        self, 
        derivative: Union[float, np.ndarray], 
        side: str
    ) -> None:
        """
        Apply Neumann boundary conditions using finite differences.
        
        Args:
            derivative: Normal derivative value at boundary
            side: Boundary side ('left', 'right', 'bottom', 'top')
        """
        if side == 'left':
            self.values[0, :] = self.values[1, :] - self.hx * derivative
        elif side == 'right':
            self.values[-1, :] = self.values[-2, :] + self.hx * derivative
        elif side == 'bottom':
            self.values[:, 0] = self.values[:, 1] - self.hy * derivative
        elif side == 'top':
            self.values[:, -1] = self.values[:, -2] + self.hy * derivative
        else:
            raise ValueError(f"Unknown boundary side: {side}")
        
        logger.debug(f"Applied Neumann BC: derivative={derivative}, side={side}")
    
    def coarsen(self) -> 'Grid':
        """
        Create a coarser grid with half the resolution.
        
        Returns:
            Coarsened grid with (nx-1)//2 + 1 points in each direction
        """
        # Ensure odd number of interior points for proper coarsening
        if (self.nx - 1) % 2 != 0 or (self.ny - 1) % 2 != 0:
            raise ValueError("Cannot coarsen grid: need even number of interior points")
        
        coarse_nx = (self.nx - 1) // 2 + 1
        coarse_ny = (self.ny - 1) // 2 + 1
        
        coarse_grid = Grid(coarse_nx, coarse_ny, self.domain, self.dtype)
        
        logger.debug(f"Coarsened grid: {self.nx}x{self.ny} -> {coarse_nx}x{coarse_ny}")
        return coarse_grid
    
    def refine(self) -> 'Grid':
        """
        Create a finer grid with double the resolution.
        
        Returns:
            Refined grid with 2*(nx-1)+1 points in each direction
        """
        fine_nx = 2 * (self.nx - 1) + 1
        fine_ny = 2 * (self.ny - 1) + 1
        
        fine_grid = Grid(fine_nx, fine_ny, self.domain, self.dtype)
        
        logger.debug(f"Refined grid: {self.nx}x{self.ny} -> {fine_nx}x{fine_ny}")
        return fine_grid
    
    def l2_norm(self, field: Optional[np.ndarray] = None) -> float:
        """
        Compute L2 norm of field values.
        
        Args:
            field: Field to compute norm for (default: self.values)
            
        Returns:
            L2 norm scaled by grid spacing
        """
        if field is None:
            field = self.values
            
        return np.sqrt(self.hx * self.hy * np.sum(field**2))
    
    def max_norm(self, field: Optional[np.ndarray] = None) -> float:
        """
        Compute maximum norm of field values.
        
        Args:
            field: Field to compute norm for (default: self.values)
            
        Returns:
            Maximum absolute value
        """
        if field is None:
            field = self.values
            
        return np.max(np.abs(field))
    
    def copy(self) -> 'Grid':
        """Create a deep copy of the grid."""
        new_grid = Grid(self.nx, self.ny, self.domain, self.dtype)
        new_grid.values = self.values.copy()
        new_grid.residual = self.residual.copy()
        return new_grid
    
    def __str__(self) -> str:
        """String representation of the grid."""
        return f"Grid({self.nx}x{self.ny}, h=({self.hx:.6f}, {self.hy:.6f}), dtype={self.dtype})"
    
    def __repr__(self) -> str:
        """Detailed representation of the grid."""
        return (f"Grid(nx={self.nx}, ny={self.ny}, domain={self.domain}, "
                f"dtype={self.dtype})")