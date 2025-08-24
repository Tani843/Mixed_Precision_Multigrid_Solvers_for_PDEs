"""Diagonal preconditioning implementation."""

import numpy as np
from typing import TYPE_CHECKING
import logging

from .base import BasePreconditioner

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class DiagonalPreconditioner(BasePreconditioner):
    """
    Diagonal preconditioning (Jacobi preconditioning).
    
    Uses the diagonal elements of the matrix A as the preconditioner:
    M = diag(A)
    
    For 2D Laplacian with 5-point stencil:
    M[i,j] = -2/h_x² - 2/h_y²
    """
    
    def __init__(self, regularization: float = 1e-12):
        """
        Initialize diagonal preconditioner.
        
        Args:
            regularization: Small value added to diagonal for numerical stability
        """
        super().__init__("Diagonal")
        self.regularization = regularization
        self.diagonal_inverse = None
        self.grid_shape = None
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """
        Setup diagonal preconditioner by extracting diagonal elements.
        
        Args:
            grid: Computational grid
            operator: Linear operator
        """
        self.grid_shape = grid.shape
        self.diagonal_inverse = np.zeros(grid.shape, dtype=grid.dtype)
        
        # For 2D Laplacian with 5-point stencil
        if hasattr(operator, 'coefficient'):
            coeff = operator.coefficient
        else:
            coeff = 1.0
        
        # Diagonal elements for interior points
        diag_value = -coeff * (2.0 / grid.hx**2 + 2.0 / grid.hy**2)
        
        # Set diagonal values
        self.diagonal_inverse[1:-1, 1:-1] = 1.0 / (diag_value - self.regularization)
        
        # Handle boundary points (set to identity for simplicity)
        # Left and right boundaries
        self.diagonal_inverse[0, :] = 1.0
        self.diagonal_inverse[-1, :] = 1.0
        
        # Top and bottom boundaries
        self.diagonal_inverse[:, 0] = 1.0
        self.diagonal_inverse[:, -1] = 1.0
        
        self.setup_completed = True
        
        logger.debug(f"Setup diagonal preconditioner for {grid.shape} grid")
        logger.debug(f"Interior diagonal value: {diag_value:.4e}")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply diagonal preconditioner: z = M^{-1} * x.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector z
        """
        if not self.setup_completed:
            raise RuntimeError("Diagonal preconditioner not setup")
        
        if x.shape != self.grid_shape:
            raise ValueError(f"Input shape {x.shape} doesn't match setup shape {self.grid_shape}")
        
        # Element-wise multiplication with diagonal inverse
        return self.diagonal_inverse * x
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """
        Apply transpose of diagonal preconditioner.
        
        Since diagonal matrix is symmetric: M^T = M
        """
        return self.apply(x)
    
    def get_diagonal_elements(self) -> np.ndarray:
        """Get the diagonal elements of the preconditioner."""
        if not self.setup_completed:
            raise RuntimeError("Diagonal preconditioner not setup")
        
        return 1.0 / self.diagonal_inverse
    
    def get_condition_number_estimate(self) -> float:
        """
        Estimate condition number of the preconditioned system.
        
        Returns:
            Estimated condition number
        """
        if not self.setup_completed:
            raise RuntimeError("Diagonal preconditioner not setup")
        
        # Get diagonal elements (excluding boundaries)
        diag_elements = 1.0 / self.diagonal_inverse[1:-1, 1:-1]
        
        # Condition number estimate: max/min of diagonal elements
        cond_estimate = np.abs(np.max(diag_elements) / np.min(diag_elements))
        
        logger.debug(f"Diagonal preconditioner condition number estimate: {cond_estimate:.2e}")
        return cond_estimate


class ScaledDiagonalPreconditioner(DiagonalPreconditioner):
    """
    Scaled diagonal preconditioner with adaptive scaling.
    
    Applies additional scaling based on grid spacing to improve conditioning.
    """
    
    def __init__(self, scaling_factor: float = 1.0, regularization: float = 1e-12):
        """
        Initialize scaled diagonal preconditioner.
        
        Args:
            scaling_factor: Additional scaling factor
            regularization: Regularization parameter
        """
        super().__init__(regularization)
        self.name = "ScaledDiagonal"
        self.scaling_factor = scaling_factor
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """Setup scaled diagonal preconditioner."""
        # First setup base diagonal preconditioner
        super().setup(grid, operator)
        
        # Apply additional scaling based on grid spacing
        # Scaling helps with anisotropic grids
        hx_hy_ratio = grid.hx / grid.hy
        
        if abs(hx_hy_ratio - 1.0) > 0.1:  # Anisotropic grid
            # Scale based on aspect ratio
            x_scale = np.ones(grid.shape)
            y_scale = np.ones(grid.shape)
            
            if hx_hy_ratio > 1.0:  # hx > hy
                x_scale *= np.sqrt(hx_hy_ratio)
            else:  # hy > hx
                y_scale *= np.sqrt(1.0 / hx_hy_ratio)
            
            # Apply scaling to diagonal inverse
            self.diagonal_inverse *= self.scaling_factor * x_scale * y_scale
            
            logger.debug(f"Applied anisotropic scaling: hx/hy = {hx_hy_ratio:.3f}")
        
        logger.debug(f"Setup scaled diagonal preconditioner with factor {self.scaling_factor}")


class BlockDiagonalPreconditioner(BasePreconditioner):
    """
    Block diagonal preconditioner for structured grids.
    
    Treats each row or column of the grid as a block and inverts
    the diagonal blocks independently.
    """
    
    def __init__(self, block_direction: str = "row"):
        """
        Initialize block diagonal preconditioner.
        
        Args:
            block_direction: Direction for blocks ("row" or "column")
        """
        super().__init__("BlockDiagonal")
        self.block_direction = block_direction
        self.block_inverses = None
        self.grid_shape = None
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """
        Setup block diagonal preconditioner.
        
        Args:
            grid: Computational grid  
            operator: Linear operator
        """
        self.grid_shape = grid.shape
        nx, ny = grid.shape
        
        if hasattr(operator, 'coefficient'):
            coeff = operator.coefficient
        else:
            coeff = 1.0
        
        # Coefficients for 5-point stencil
        center_coeff = -coeff * (2.0 / grid.hx**2 + 2.0 / grid.hy**2)
        hx_coeff = coeff / grid.hx**2
        hy_coeff = coeff / grid.hy**2
        
        if self.block_direction == "row":
            # Each row is a block (tridiagonal systems)
            self.block_inverses = []
            
            for i in range(1, nx - 1):  # Interior rows
                # Build tridiagonal matrix for this row
                n = ny - 2  # Interior columns
                diag = np.full(n, center_coeff)
                off_diag = np.full(n - 1, hy_coeff)
                
                # Create tridiagonal matrix
                block_matrix = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
                
                # Store inverse (or factorization for efficiency)
                try:
                    block_inv = np.linalg.inv(block_matrix)
                    self.block_inverses.append(block_inv)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse
                    block_inv = np.linalg.pinv(block_matrix)
                    self.block_inverses.append(block_inv)
                    logger.warning(f"Used pseudo-inverse for row {i}")
        
        else:  # column direction
            # Each column is a block
            self.block_inverses = []
            
            for j in range(1, ny - 1):  # Interior columns
                # Build tridiagonal matrix for this column
                n = nx - 2  # Interior rows
                diag = np.full(n, center_coeff)
                off_diag = np.full(n - 1, hx_coeff)
                
                # Create tridiagonal matrix
                block_matrix = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
                
                # Store inverse
                try:
                    block_inv = np.linalg.inv(block_matrix)
                    self.block_inverses.append(block_inv)
                except np.linalg.LinAlgError:
                    block_inv = np.linalg.pinv(block_matrix)
                    self.block_inverses.append(block_inv)
                    logger.warning(f"Used pseudo-inverse for column {j}")
        
        self.setup_completed = True
        
        logger.debug(f"Setup block diagonal preconditioner: {len(self.block_inverses)} blocks")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply block diagonal preconditioner.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        if not self.setup_completed:
            raise RuntimeError("Block diagonal preconditioner not setup")
        
        result = np.zeros_like(x)
        nx, ny = self.grid_shape
        
        if self.block_direction == "row":
            # Apply row-wise blocks
            for i, block_inv in enumerate(self.block_inverses):
                row_idx = i + 1  # Adjust for boundary
                row_interior = x[row_idx, 1:-1]  # Interior columns
                result[row_idx, 1:-1] = block_inv @ row_interior
            
            # Handle boundaries (identity)
            result[0, :] = x[0, :]
            result[-1, :] = x[-1, :]
            result[:, 0] = x[:, 0]
            result[:, -1] = x[:, -1]
        
        else:  # column direction
            # Apply column-wise blocks
            for j, block_inv in enumerate(self.block_inverses):
                col_idx = j + 1  # Adjust for boundary
                col_interior = x[1:-1, col_idx]  # Interior rows
                result[1:-1, col_idx] = block_inv @ col_interior
            
            # Handle boundaries (identity)
            result[0, :] = x[0, :]
            result[-1, :] = x[-1, :]
            result[:, 0] = x[:, 0]
            result[:, -1] = x[:, -1]
        
        return result
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """Apply transpose of block diagonal preconditioner."""
        # For symmetric blocks, transpose is the same as forward application
        return self.apply(x)