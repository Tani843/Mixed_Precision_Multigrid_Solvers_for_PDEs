"""Incomplete LU (ILU) preconditioning implementation."""

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve_triangular
import logging

from .base import BasePreconditioner

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class ILUPreconditioner(BasePreconditioner):
    """
    Incomplete LU(0) preconditioner.
    
    Computes an incomplete LU factorization of the coefficient matrix A,
    where the sparsity pattern of L and U is constrained to that of A.
    
    The preconditioner solves: M*z = x where M ≈ A and M = L*U
    """
    
    def __init__(self, fill_level: int = 0, drop_tolerance: float = 0.0):
        """
        Initialize ILU preconditioner.
        
        Args:
            fill_level: Level of fill-in allowed (0 = ILU(0))
            drop_tolerance: Drop small elements below this threshold
        """
        super().__init__(f"ILU({fill_level})")
        self.fill_level = fill_level
        self.drop_tolerance = drop_tolerance
        
        # Factorization storage
        self.L_matrix = None
        self.U_matrix = None
        self.grid_shape = None
        self.sparse_matrix = None
        
        # Mapping between 2D grid and 1D vector indices
        self.grid_to_vector = None
        self.vector_to_grid = None
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """
        Setup ILU preconditioner by computing the factorization.
        
        Args:
            grid: Computational grid
            operator: Linear operator to precondition
        """
        self.grid_shape = grid.shape
        nx, ny = grid.shape
        
        # Create mapping between 2D grid and 1D vector indices
        self._create_index_mapping(grid)
        
        # Build sparse matrix representation of the operator
        self.sparse_matrix = self._build_sparse_matrix(grid, operator)
        
        # Compute ILU factorization
        self.L_matrix, self.U_matrix = self._compute_ilu_factorization(self.sparse_matrix)
        
        self.setup_completed = True
        
        logger.debug(f"Setup ILU({self.fill_level}) preconditioner for {grid.shape} grid")
        logger.debug(f"Sparse matrix: {self.sparse_matrix.nnz} non-zeros")
        logger.debug(f"L matrix: {self.L_matrix.nnz} non-zeros")
        logger.debug(f"U matrix: {self.U_matrix.nnz} non-zeros")
    
    def _create_index_mapping(self, grid: 'Grid') -> None:
        """Create mapping between 2D grid indices and 1D vector indices."""
        nx, ny = grid.shape
        
        # Only interior points are included in the linear system
        n_interior = (nx - 2) * (ny - 2)
        
        self.grid_to_vector = {}
        self.vector_to_grid = {}
        
        vector_idx = 0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                self.grid_to_vector[(i, j)] = vector_idx
                self.vector_to_grid[vector_idx] = (i, j)
                vector_idx += 1
    
    def _build_sparse_matrix(self, grid: 'Grid', operator: 'BaseOperator') -> csr_matrix:
        """
        Build sparse matrix representation of the discrete operator.
        
        Args:
            grid: Computational grid
            operator: Discrete operator
            
        Returns:
            Sparse matrix in CSR format
        """
        nx, ny = grid.shape
        n_interior = (nx - 2) * (ny - 2)
        
        # Get operator coefficient
        if hasattr(operator, 'coefficient'):
            coeff = operator.coefficient
        else:
            coeff = 1.0
        
        # 5-point stencil coefficients
        center_coeff = -coeff * (2.0 / grid.hx**2 + 2.0 / grid.hy**2)
        hx_coeff = coeff / grid.hx**2
        hy_coeff = coeff / grid.hy**2
        
        # Build sparse matrix using COO format first
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                center_idx = self.grid_to_vector[(i, j)]
                
                # Center point
                row_indices.append(center_idx)
                col_indices.append(center_idx)
                data.append(center_coeff)
                
                # Left neighbor
                if j > 1:
                    left_idx = self.grid_to_vector[(i, j - 1)]
                    row_indices.append(center_idx)
                    col_indices.append(left_idx)
                    data.append(hy_coeff)
                
                # Right neighbor
                if j < ny - 2:
                    right_idx = self.grid_to_vector[(i, j + 1)]
                    row_indices.append(center_idx)
                    col_indices.append(right_idx)
                    data.append(hy_coeff)
                
                # Bottom neighbor
                if i > 1:
                    bottom_idx = self.grid_to_vector[(i - 1, j)]
                    row_indices.append(center_idx)
                    col_indices.append(bottom_idx)
                    data.append(hx_coeff)
                
                # Top neighbor
                if i < nx - 2:
                    top_idx = self.grid_to_vector[(i + 1, j)]
                    row_indices.append(center_idx)
                    col_indices.append(top_idx)
                    data.append(hx_coeff)
        
        # Create sparse matrix in CSR format
        sparse_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_interior, n_interior)
        )
        
        return sparse_matrix
    
    def _compute_ilu_factorization(self, A: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """
        Compute ILU(0) factorization of sparse matrix A.
        
        Args:
            A: Sparse matrix to factorize
            
        Returns:
            Tuple of (L, U) matrices
        """
        n = A.shape[0]
        
        # Convert to LIL format for efficient modification
        A_work = A.tolil()
        
        # ILU(0) factorization with fill-level constraint
        for k in range(n):
            # Get non-zero pattern for row k and column k
            k_row_indices = A_work.rows[k]
            k_col_indices = []
            
            for i in range(n):
                if k in A_work.rows[i]:
                    k_col_indices.append(i)
            
            # Update elements
            for i in k_col_indices:
                if i > k and k in A_work.rows[i]:
                    # Update L[i,k]
                    if A_work[k, k] != 0:
                        A_work[i, k] = A_work[i, k] / A_work[k, k]
                    
                    # Update row i
                    for j in k_row_indices:
                        if j > k and j in A_work.rows[i]:
                            # Only update if within fill pattern for ILU(0)
                            if self._allow_fill(i, j, k):
                                A_work[i, j] = A_work[i, j] - A_work[i, k] * A_work[k, j]
            
            # Apply drop tolerance
            if self.drop_tolerance > 0:
                self._apply_drop_tolerance(A_work, k)
        
        # Extract L and U matrices
        L_data = []
        L_row_indices = []
        L_col_indices = []
        
        U_data = []
        U_row_indices = []
        U_col_indices = []
        
        for i in range(n):
            for j_idx, j in enumerate(A_work.rows[i]):
                value = A_work.data[i][j_idx]
                
                if i == j:
                    # Diagonal: L[i,i] = 1, U[i,i] = A[i,i]
                    L_data.append(1.0)
                    L_row_indices.append(i)
                    L_col_indices.append(i)
                    
                    U_data.append(value)
                    U_row_indices.append(i)
                    U_col_indices.append(j)
                    
                elif i > j:
                    # Lower triangular
                    L_data.append(value)
                    L_row_indices.append(i)
                    L_col_indices.append(j)
                    
                else:  # i < j
                    # Upper triangular
                    U_data.append(value)
                    U_row_indices.append(i)
                    U_col_indices.append(j)
        
        # Create sparse L and U matrices
        L_matrix = csr_matrix((L_data, (L_row_indices, L_col_indices)), shape=(n, n))
        U_matrix = csr_matrix((U_data, (U_row_indices, U_col_indices)), shape=(n, n))
        
        return L_matrix, U_matrix
    
    def _allow_fill(self, i: int, j: int, k: int) -> bool:
        """
        Check if fill-in is allowed at position (i,j) during elimination of k.
        
        For ILU(0), only allow fill where original matrix had non-zero.
        """
        if self.fill_level == 0:
            # Only allow fill where original matrix is non-zero
            return self.sparse_matrix[i, j] != 0
        else:
            # For higher fill levels, implement more sophisticated logic
            return True
    
    def _apply_drop_tolerance(self, A_work, row: int) -> None:
        """Apply drop tolerance to remove small elements."""
        if self.drop_tolerance <= 0:
            return
        
        # Remove elements smaller than drop tolerance
        new_data = []
        new_cols = []
        
        for j_idx, col in enumerate(A_work.rows[row]):
            value = A_work.data[row][j_idx]
            
            if abs(value) >= self.drop_tolerance or row == col:  # Keep diagonal
                new_data.append(value)
                new_cols.append(col)
        
        A_work.rows[row] = new_cols
        A_work.data[row] = new_data
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ILU preconditioner: solve L*U*z = x for z.
        
        Args:
            x: Input vector (2D grid format)
            
        Returns:
            Preconditioned vector z (2D grid format)
        """
        if not self.setup_completed:
            raise RuntimeError("ILU preconditioner not setup")
        
        # Convert 2D grid to 1D vector (interior points only)
        x_vector = self._grid_to_vector_format(x)
        
        # Solve L*y = x
        y = spsolve_triangular(self.L_matrix, x_vector, lower=True)
        
        # Solve U*z = y
        z_vector = spsolve_triangular(self.U_matrix, y, lower=False)
        
        # Convert back to 2D grid format
        z = self._vector_to_grid_format(z_vector, x.shape)
        
        return z
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """
        Apply transpose of ILU preconditioner: solve U^T*L^T*z = x for z.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector z
        """
        if not self.setup_completed:
            raise RuntimeError("ILU preconditioner not setup")
        
        # Convert to vector format
        x_vector = self._grid_to_vector_format(x)
        
        # Solve U^T*y = x
        U_T = self.U_matrix.T.tocsr()
        y = spsolve_triangular(U_T, x_vector, lower=True)
        
        # Solve L^T*z = y
        L_T = self.L_matrix.T.tocsr()
        z_vector = spsolve_triangular(L_T, y, lower=False)
        
        # Convert back to grid format
        z = self._vector_to_grid_format(z_vector, x.shape)
        
        return z
    
    def _grid_to_vector_format(self, grid_array: np.ndarray) -> np.ndarray:
        """Convert 2D grid array to 1D vector (interior points only)."""
        nx, ny = self.grid_shape
        n_interior = (nx - 2) * (ny - 2)
        vector = np.zeros(n_interior)
        
        for vector_idx, (i, j) in self.vector_to_grid.items():
            vector[vector_idx] = grid_array[i, j]
        
        return vector
    
    def _vector_to_grid_format(self, vector: np.ndarray, grid_shape: tuple) -> np.ndarray:
        """Convert 1D vector to 2D grid format."""
        grid_array = np.zeros(grid_shape)
        
        # Copy boundary values (unchanged by preconditioning)
        for vector_idx, (i, j) in self.vector_to_grid.items():
            grid_array[i, j] = vector[vector_idx]
        
        return grid_array
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics for the preconditioner."""
        if not self.setup_completed:
            return {"status": "not_setup"}
        
        # Estimate memory usage
        original_nnz = self.sparse_matrix.nnz
        l_nnz = self.L_matrix.nnz if self.L_matrix else 0
        u_nnz = self.U_matrix.nnz if self.U_matrix else 0
        
        # Bytes for double precision
        bytes_per_element = 8
        
        memory_stats = {
            "original_nnz": original_nnz,
            "l_nnz": l_nnz,
            "u_nnz": u_nnz,
            "fill_ratio": (l_nnz + u_nnz) / original_nnz if original_nnz > 0 else 0,
            "estimated_memory_mb": (l_nnz + u_nnz) * bytes_per_element / (1024**2)
        }
        
        return memory_stats


class ModifiedILUPreconditioner(ILUPreconditioner):
    """
    Modified ILU preconditioner with diagonal modification.
    
    Adds diagonal modification to improve stability:
    A_modified = A + α*I where α is chosen to improve conditioning.
    """
    
    def __init__(self, fill_level: int = 0, diagonal_shift: float = 0.01):
        """
        Initialize modified ILU preconditioner.
        
        Args:
            fill_level: Fill level for ILU factorization
            diagonal_shift: Diagonal modification parameter
        """
        super().__init__(fill_level)
        self.name = f"MILU({fill_level})"
        self.diagonal_shift = diagonal_shift
    
    def _build_sparse_matrix(self, grid: 'Grid', operator: 'BaseOperator') -> csr_matrix:
        """Build sparse matrix with diagonal modification."""
        # Get base sparse matrix
        A = super()._build_sparse_matrix(grid, operator)
        
        # Add diagonal shift
        n = A.shape[0]
        diagonal_modification = diags([self.diagonal_shift], shape=(n, n))
        
        A_modified = A + diagonal_modification
        
        logger.debug(f"Applied diagonal shift: {self.diagonal_shift}")
        
        return A_modified.tocsr()