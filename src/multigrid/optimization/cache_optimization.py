"""Cache optimization techniques for multigrid operations."""

import numpy as np
from typing import Tuple, List, Iterator, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from ..core.grid import Grid

logger = logging.getLogger(__name__)


@dataclass
class BlockInfo:
    """Information about a cache-optimized block."""
    i_start: int
    i_end: int
    j_start: int
    j_end: int
    size: int
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get block shape."""
        return (self.i_end - self.i_start, self.j_end - self.j_start)


class CacheOptimizer:
    """
    Cache optimization strategies for multigrid operations.
    
    Implements block-structured operations to improve cache locality
    and reduce memory bandwidth requirements.
    """
    
    def __init__(
        self,
        cache_line_size: int = 64,
        l1_cache_size: int = 32 * 1024,  # 32 KB L1 cache
        l2_cache_size: int = 256 * 1024,  # 256 KB L2 cache
        prefetch_distance: int = 2
    ):
        """
        Initialize cache optimizer.
        
        Args:
            cache_line_size: Size of cache line in bytes
            l1_cache_size: L1 cache size in bytes
            l2_cache_size: L2 cache size in bytes
            prefetch_distance: Prefetch distance for memory access
        """
        self.cache_line_size = cache_line_size
        self.l1_cache_size = l1_cache_size
        self.l2_cache_size = l2_cache_size
        self.prefetch_distance = prefetch_distance
        
        # Calculate optimal block sizes
        self.optimal_block_sizes = self._calculate_optimal_block_sizes()
        
        logger.debug(f"CacheOptimizer initialized: L1={l1_cache_size//1024}KB, "
                    f"L2={l2_cache_size//1024}KB, optimal_blocks={self.optimal_block_sizes}")
    
    def _calculate_optimal_block_sizes(self) -> dict:
        """Calculate optimal block sizes for different operations."""
        # Rough estimates based on cache sizes and typical access patterns
        
        # For double precision (8 bytes)
        double_elements_l1 = self.l1_cache_size // 8
        double_elements_l2 = self.l2_cache_size // 8
        
        # For single precision (4 bytes)
        single_elements_l1 = self.l1_cache_size // 4
        single_elements_l2 = self.l2_cache_size // 4
        
        # Block sizes should fit multiple blocks in cache with some overhead
        # Typical stencil operations access 5 points, so factor that in
        
        optimal_sizes = {
            'double_l1': int(np.sqrt(double_elements_l1 // 10)),  # Conservative
            'double_l2': int(np.sqrt(double_elements_l2 // 20)),
            'single_l1': int(np.sqrt(single_elements_l1 // 10)),
            'single_l2': int(np.sqrt(single_elements_l2 // 20)),
        }
        
        # Ensure minimum and maximum reasonable sizes
        for key in optimal_sizes:
            optimal_sizes[key] = max(8, min(optimal_sizes[key], 128))
        
        return optimal_sizes
    
    def get_optimal_block_size(
        self, 
        grid_shape: Tuple[int, int], 
        dtype: np.dtype,
        cache_level: str = "l1"
    ) -> int:
        """
        Get optimal block size for given grid and data type.
        
        Args:
            grid_shape: Shape of the grid
            dtype: Data type (float32 or float64)
            cache_level: Cache level to optimize for ("l1" or "l2")
            
        Returns:
            Optimal block size
        """
        precision = "double" if dtype == np.float64 else "single"
        key = f"{precision}_{cache_level}"
        
        base_size = self.optimal_block_sizes.get(key, 32)
        
        # Adjust based on grid size
        max_dim = max(grid_shape)
        if max_dim < base_size * 2:
            # For small grids, use smaller blocks
            return max(4, max_dim // 4)
        elif max_dim > base_size * 8:
            # For very large grids, may benefit from larger blocks
            return min(base_size * 2, 64)
        else:
            return base_size
    
    def optimize_array_layout(self, array: np.ndarray) -> np.ndarray:
        """
        Optimize array memory layout for cache efficiency.
        
        Args:
            array: Input array to optimize
            
        Returns:
            Array with optimized layout
        """
        # Ensure C-contiguous layout for better cache performance
        if not array.flags['C_CONTIGUOUS']:
            logger.debug("Converting array to C-contiguous layout")
            array = np.ascontiguousarray(array)
        
        # For large arrays, consider using memory mapping
        if array.nbytes > self.l2_cache_size * 4:
            logger.debug(f"Large array detected: {array.nbytes // (1024*1024)} MB")
        
        return array
    
    def prefetch_block(self, array: np.ndarray, block: BlockInfo) -> None:
        """
        Software prefetch for upcoming block operations.
        
        Args:
            array: Array to prefetch from
            block: Block information
        """
        # This is a placeholder for software prefetching
        # In practice, would use platform-specific intrinsics
        # or rely on hardware prefetchers
        
        # Touch the memory to encourage prefetching
        if block.i_start < array.shape[0] and block.j_start < array.shape[1]:
            _ = array[block.i_start, block.j_start]


class BlockTraversal:
    """
    Iterator for cache-optimized block traversal of grids.
    
    Provides different traversal patterns optimized for cache locality.
    """
    
    def __init__(
        self,
        grid: 'Grid',
        block_size: int,
        traversal_pattern: str = "row_major",
        include_boundaries: bool = False
    ):
        """
        Initialize block traversal.
        
        Args:
            grid: Grid to traverse
            block_size: Size of blocks for traversal
            traversal_pattern: Traversal pattern ("row_major", "column_major", "z_order")
            include_boundaries: Include boundary points in traversal
        """
        self.grid = grid
        self.block_size = block_size
        self.traversal_pattern = traversal_pattern
        self.include_boundaries = include_boundaries
        
        # Determine traversal bounds
        if include_boundaries:
            self.i_start, self.i_end = 0, grid.nx
            self.j_start, self.j_end = 0, grid.ny
        else:
            self.i_start, self.i_end = 1, grid.nx - 1
            self.j_start, self.j_end = 1, grid.ny - 1
        
        self.blocks = self._generate_blocks()
    
    def _generate_blocks(self) -> List[BlockInfo]:
        """Generate list of blocks for traversal."""
        blocks = []
        
        if self.traversal_pattern == "row_major":
            blocks = self._generate_row_major_blocks()
        elif self.traversal_pattern == "column_major":
            blocks = self._generate_column_major_blocks()
        elif self.traversal_pattern == "z_order":
            blocks = self._generate_z_order_blocks()
        else:
            raise ValueError(f"Unknown traversal pattern: {self.traversal_pattern}")
        
        logger.debug(f"Generated {len(blocks)} blocks for {self.traversal_pattern} traversal")
        return blocks
    
    def _generate_row_major_blocks(self) -> List[BlockInfo]:
        """Generate blocks in row-major order."""
        blocks = []
        
        for i in range(self.i_start, self.i_end, self.block_size):
            for j in range(self.j_start, self.j_end, self.block_size):
                block = BlockInfo(
                    i_start=i,
                    i_end=min(i + self.block_size, self.i_end),
                    j_start=j,
                    j_end=min(j + self.block_size, self.j_end),
                    size=self.block_size
                )
                blocks.append(block)
        
        return blocks
    
    def _generate_column_major_blocks(self) -> List[BlockInfo]:
        """Generate blocks in column-major order."""
        blocks = []
        
        for j in range(self.j_start, self.j_end, self.block_size):
            for i in range(self.i_start, self.i_end, self.block_size):
                block = BlockInfo(
                    i_start=i,
                    i_end=min(i + self.block_size, self.i_end),
                    j_start=j,
                    j_end=min(j + self.block_size, self.j_end),
                    size=self.block_size
                )
                blocks.append(block)
        
        return blocks
    
    def _generate_z_order_blocks(self) -> List[BlockInfo]:
        """Generate blocks in Z-order (Morton order) for better locality."""
        # Simplified Z-order implementation
        # For full implementation, would use bit interleaving
        
        blocks = []
        
        # Use recursive quadtree-like subdivision
        def add_z_blocks(i_min, i_max, j_min, j_max):
            if i_max - i_min <= self.block_size and j_max - j_min <= self.block_size:
                # Base case: create block
                block = BlockInfo(
                    i_start=i_min,
                    i_end=i_max,
                    j_start=j_min,
                    j_end=j_max,
                    size=self.block_size
                )
                blocks.append(block)
            else:
                # Recursive case: subdivide
                i_mid = (i_min + i_max) // 2
                j_mid = (j_min + j_max) // 2
                
                # Visit quadrants in Z-order
                add_z_blocks(i_min, i_mid, j_min, j_mid)      # Bottom-left
                add_z_blocks(i_min, i_mid, j_mid, j_max)      # Top-left
                add_z_blocks(i_mid, i_max, j_min, j_mid)      # Bottom-right
                add_z_blocks(i_mid, i_max, j_mid, j_max)      # Top-right
        
        add_z_blocks(self.i_start, self.i_end, self.j_start, self.j_end)
        
        return blocks
    
    def __iter__(self) -> Iterator[BlockInfo]:
        """Iterate over blocks."""
        return iter(self.blocks)
    
    def __len__(self) -> int:
        """Number of blocks."""
        return len(self.blocks)
    
    def get_block_neighbors(self, block_idx: int) -> List[int]:
        """
        Get indices of neighboring blocks.
        
        Args:
            block_idx: Index of current block
            
        Returns:
            List of neighbor block indices
        """
        if block_idx >= len(self.blocks):
            return []
        
        current_block = self.blocks[block_idx]
        neighbors = []
        
        for idx, block in enumerate(self.blocks):
            if idx == block_idx:
                continue
            
            # Check if blocks are adjacent
            if self._are_adjacent(current_block, block):
                neighbors.append(idx)
        
        return neighbors
    
    def _are_adjacent(self, block1: BlockInfo, block2: BlockInfo) -> bool:
        """Check if two blocks are adjacent."""
        # Check horizontal adjacency
        h_adjacent = (
            (block1.i_end == block2.i_start or block2.i_end == block1.i_start) and
            not (block1.j_end <= block2.j_start or block2.j_end <= block1.j_start)
        )
        
        # Check vertical adjacency
        v_adjacent = (
            (block1.j_end == block2.j_start or block2.j_end == block1.j_start) and
            not (block1.i_end <= block2.i_start or block2.i_end <= block1.i_start)
        )
        
        return h_adjacent or v_adjacent


class CacheAwareStencil:
    """
    Cache-aware stencil operations for multigrid methods.
    
    Implements stencil computations with optimal cache utilization.
    """
    
    def __init__(self, cache_optimizer: CacheOptimizer):
        """Initialize cache-aware stencil operations."""
        self.cache_optimizer = cache_optimizer
    
    def apply_5_point_stencil_blocked(
        self,
        u: np.ndarray,
        rhs: np.ndarray,
        grid: 'Grid',
        coefficient: float = 1.0,
        num_iterations: int = 1
    ) -> np.ndarray:
        """
        Apply 5-point stencil with cache-optimized blocking.
        
        Args:
            u: Solution array
            rhs: Right-hand side array
            grid: Computational grid
            coefficient: Stencil coefficient
            num_iterations: Number of iterations
            
        Returns:
            Updated solution array
        """
        # Optimize array layout
        u = self.cache_optimizer.optimize_array_layout(u)
        rhs = self.cache_optimizer.optimize_array_layout(rhs)
        
        # Get optimal block size
        block_size = self.cache_optimizer.get_optimal_block_size(u.shape, u.dtype)
        
        # Create block traversal
        traversal = BlockTraversal(grid, block_size, "row_major", False)
        
        # Pre-compute stencil coefficients
        hx2_inv = coefficient / (grid.hx ** 2)
        hy2_inv = coefficient / (grid.hy ** 2)
        center_coeff = -2 * (hx2_inv + hy2_inv)
        
        u_new = u.copy()
        
        for iteration in range(num_iterations):
            # Process blocks for cache efficiency
            for block in traversal:
                self._process_stencil_block(
                    u_new, u, rhs, block, hx2_inv, hy2_inv, center_coeff
                )
            
            # Swap arrays for next iteration
            if iteration < num_iterations - 1:
                u, u_new = u_new, u
        
        return u_new
    
    def _process_stencil_block(
        self,
        u_new: np.ndarray,
        u_old: np.ndarray,
        rhs: np.ndarray,
        block: BlockInfo,
        hx2_inv: float,
        hy2_inv: float,
        center_coeff: float
    ) -> None:
        """Process stencil computation within a block."""
        # Apply stencil within block bounds
        for i in range(block.i_start, block.i_end):
            for j in range(block.j_start, block.j_end):
                if i > 0 and i < u_old.shape[0] - 1 and j > 0 and j < u_old.shape[1] - 1:
                    # 5-point stencil computation
                    stencil_result = (
                        hx2_inv * (u_old[i+1, j] + u_old[i-1, j]) +
                        hy2_inv * (u_old[i, j+1] + u_old[i, j-1]) +
                        center_coeff * u_old[i, j]
                    )
                    
                    # Update solution
                    u_new[i, j] = (rhs[i, j] - stencil_result) / (-center_coeff)
    
    def compute_residual_blocked(
        self,
        u: np.ndarray,
        rhs: np.ndarray,
        grid: 'Grid',
        coefficient: float = 1.0
    ) -> np.ndarray:
        """
        Compute residual with cache-optimized blocking.
        
        Args:
            u: Solution array
            rhs: Right-hand side array
            grid: Computational grid
            coefficient: Operator coefficient
            
        Returns:
            Residual array
        """
        # Get optimal block size
        block_size = self.cache_optimizer.get_optimal_block_size(u.shape, u.dtype)
        
        # Create block traversal
        traversal = BlockTraversal(grid, block_size, "row_major", True)
        
        # Initialize residual array
        residual = np.zeros_like(u)
        
        # Pre-compute stencil coefficients
        hx2_inv = coefficient / (grid.hx ** 2)
        hy2_inv = coefficient / (grid.hy ** 2)
        center_coeff = -2 * (hx2_inv + hy2_inv)
        
        # Process blocks
        for block in traversal:
            self._compute_residual_block(
                residual, u, rhs, block, hx2_inv, hy2_inv, center_coeff
            )
        
        return residual
    
    def _compute_residual_block(
        self,
        residual: np.ndarray,
        u: np.ndarray,
        rhs: np.ndarray,
        block: BlockInfo,
        hx2_inv: float,
        hy2_inv: float,
        center_coeff: float
    ) -> None:
        """Compute residual within a block."""
        for i in range(block.i_start, block.i_end):
            for j in range(block.j_start, block.j_end):
                if i > 0 and i < u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
                    # Apply operator
                    Au = (
                        center_coeff * u[i, j] +
                        hx2_inv * (u[i+1, j] + u[i-1, j]) +
                        hy2_inv * (u[i, j+1] + u[i, j-1])
                    )
                    
                    # Compute residual: r = f - Au
                    residual[i, j] = rhs[i, j] - Au
                else:
                    # Boundary points
                    residual[i, j] = 0.0