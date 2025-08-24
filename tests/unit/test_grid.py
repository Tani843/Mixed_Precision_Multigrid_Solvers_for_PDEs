"""Unit tests for Grid class."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multigrid.core.grid import Grid


class TestGrid:
    """Test cases for Grid class."""
    
    def test_grid_initialization(self):
        """Test basic grid initialization."""
        grid = Grid(nx=5, ny=7, domain=(0, 1, 0, 2))
        
        assert grid.nx == 5
        assert grid.ny == 7
        assert grid.domain == (0, 1, 0, 2)
        assert grid.shape == (5, 7)
        assert grid.size == 35
        
        # Check grid spacing
        expected_hx = 1.0 / 4  # (1-0)/(5-1)
        expected_hy = 2.0 / 6  # (2-0)/(7-1)
        
        assert abs(grid.hx - expected_hx) < 1e-12
        assert abs(grid.hy - expected_hy) < 1e-12
    
    def test_grid_coordinates(self):
        """Test grid coordinate generation."""
        grid = Grid(nx=3, ny=3, domain=(0, 2, 1, 3))
        
        expected_x = np.array([0, 1, 2])
        expected_y = np.array([1, 2, 3])
        
        np.testing.assert_allclose(grid.x, expected_x)
        np.testing.assert_allclose(grid.y, expected_y)
        
        # Check meshgrid
        assert grid.X.shape == (3, 3)
        assert grid.Y.shape == (3, 3)
    
    def test_grid_validation(self):
        """Test grid parameter validation."""
        with pytest.raises(ValueError, match="at least 3 points"):
            Grid(nx=2, ny=3)
        
        with pytest.raises(ValueError, match="at least 3 points"):
            Grid(nx=3, ny=1)
    
    def test_boundary_conditions_dirichlet(self):
        """Test Dirichlet boundary conditions."""
        grid = Grid(nx=5, ny=5)
        
        # Apply Dirichlet BC to all boundaries
        grid.apply_dirichlet_bc(1.0)
        
        # Check boundaries
        assert np.all(grid.values[0, :] == 1.0)  # Left
        assert np.all(grid.values[-1, :] == 1.0)  # Right  
        assert np.all(grid.values[:, 0] == 1.0)   # Bottom
        assert np.all(grid.values[:, -1] == 1.0)  # Top
        
        # Interior should remain zero
        assert np.all(grid.values[1:-1, 1:-1] == 0.0)
    
    def test_boundary_conditions_single_side(self):
        """Test applying BC to single boundary."""
        grid = Grid(nx=4, ny=4)
        
        grid.apply_dirichlet_bc(2.0, 'left')
        
        # Only left boundary should be set
        assert np.all(grid.values[0, :] == 2.0)
        assert np.all(grid.values[1:, :] == 0.0)
    
    def test_neumann_boundary_conditions(self):
        """Test Neumann boundary conditions."""
        grid = Grid(nx=5, ny=5)
        
        # Set interior values
        grid.values[1:-1, 1:-1] = 1.0
        
        # Apply zero Neumann BC on left boundary
        grid.apply_neumann_bc(0.0, 'left')
        
        # Left boundary should equal adjacent interior values
        np.testing.assert_allclose(grid.values[0, :], grid.values[1, :])
    
    def test_grid_coarsening(self):
        """Test grid coarsening."""
        grid = Grid(nx=9, ny=9)  # 8 interior points, divisible by 2
        
        coarse_grid = grid.coarsen()
        
        assert coarse_grid.nx == 5  # (9-1)//2 + 1
        assert coarse_grid.ny == 5
        assert coarse_grid.domain == grid.domain
        
        # Check that coarse grid spacing is roughly double
        assert abs(coarse_grid.hx - 2 * grid.hx) < 1e-12
        assert abs(coarse_grid.hy - 2 * grid.hy) < 1e-12
    
    def test_grid_coarsening_invalid(self):
        """Test coarsening with invalid grid size."""
        grid = Grid(nx=6, ny=6)  # 5 interior points, not divisible by 2
        
        with pytest.raises(ValueError, match="Cannot coarsen grid"):
            grid.coarsen()
    
    def test_grid_refinement(self):
        """Test grid refinement."""
        grid = Grid(nx=5, ny=5)
        
        fine_grid = grid.refine()
        
        assert fine_grid.nx == 9   # 2*(5-1)+1
        assert fine_grid.ny == 9
        assert fine_grid.domain == grid.domain
        
        # Check that fine grid spacing is roughly half
        assert abs(fine_grid.hx - grid.hx / 2) < 1e-12
        assert abs(fine_grid.hy - grid.hy / 2) < 1e-12
    
    def test_grid_norms(self):
        """Test grid norm calculations."""
        grid = Grid(nx=5, ny=5)
        
        # Set a simple field
        grid.values[1:-1, 1:-1] = 1.0
        
        # Test L2 norm
        l2_norm = grid.l2_norm()
        expected_l2 = np.sqrt(grid.hx * grid.hy * 9)  # 3x3 interior with value 1
        assert abs(l2_norm - expected_l2) < 1e-12
        
        # Test max norm
        max_norm = grid.max_norm()
        assert max_norm == 1.0
        
        # Test with different field
        grid.values = np.ones(grid.shape) * 2.0
        assert grid.max_norm() == 2.0
    
    def test_grid_copy(self):
        """Test grid copying."""
        grid = Grid(nx=4, ny=4)
        grid.values[:] = np.random.rand(4, 4)
        
        grid_copy = grid.copy()
        
        # Check that copy has same properties
        assert grid_copy.nx == grid.nx
        assert grid_copy.ny == grid.ny
        assert grid_copy.domain == grid.domain
        np.testing.assert_array_equal(grid_copy.values, grid.values)
        
        # Check that it's a deep copy
        grid_copy.values[0, 0] = -999
        assert grid.values[0, 0] != -999
    
    def test_grid_slices(self):
        """Test grid slicing methods."""
        grid = Grid(nx=5, ny=5)
        
        # Interior slice
        i_slice, j_slice = grid.interior_slice()
        assert i_slice == slice(1, -1)
        assert j_slice == slice(1, -1)
        
        # Boundary slices
        left_slice = grid.boundary_slice('left')
        assert left_slice == (slice(0, 1), slice(None))
        
        right_slice = grid.boundary_slice('right')
        assert right_slice == (slice(-1, None), slice(None))
        
        bottom_slice = grid.boundary_slice('bottom')
        assert bottom_slice == (slice(None), slice(0, 1))
        
        top_slice = grid.boundary_slice('top')
        assert top_slice == (slice(None), slice(-1, None))
        
        # Invalid boundary
        with pytest.raises(ValueError, match="Unknown boundary side"):
            grid.boundary_slice('invalid')
    
    def test_grid_dtype_handling(self):
        """Test different data types."""
        # Float32 grid
        grid32 = Grid(nx=4, ny=4, dtype=np.float32)
        assert grid32.values.dtype == np.float32
        assert grid32.residual.dtype == np.float32
        
        # Float64 grid
        grid64 = Grid(nx=4, ny=4, dtype=np.float64)
        assert grid64.values.dtype == np.float64
        assert grid64.residual.dtype == np.float64
    
    def test_grid_string_representation(self):
        """Test string representations."""
        grid = Grid(nx=5, ny=7, domain=(0, 1, 0, 2))
        
        str_repr = str(grid)
        assert "5x7" in str_repr
        assert "float64" in str_repr
        
        repr_str = repr(grid)
        assert "Grid(nx=5, ny=7" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])