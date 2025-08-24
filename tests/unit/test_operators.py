"""Unit tests for discrete operators."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multigrid.core.grid import Grid
from multigrid.operators.laplacian import LaplacianOperator
from multigrid.operators.transfer import RestrictionOperator, ProlongationOperator


class TestLaplacianOperator:
    """Test cases for Laplacian operator."""
    
    def test_laplacian_initialization(self):
        """Test Laplacian operator initialization."""
        laplacian = LaplacianOperator()
        assert laplacian.coefficient == 1.0
        assert "Laplacian" in laplacian.name
        
        laplacian_scaled = LaplacianOperator(coefficient=2.0)
        assert laplacian_scaled.coefficient == 2.0
    
    def test_laplacian_can_apply(self):
        """Test can_apply method."""
        laplacian = LaplacianOperator()
        
        # Valid grid
        grid = Grid(nx=5, ny=5)
        assert laplacian.can_apply(grid)
        
        # Invalid grids
        grid_small = Grid(nx=3, ny=3)
        assert laplacian.can_apply(grid_small)  # Minimum size
        
        # Grid too small would fail at Grid creation
        with pytest.raises(ValueError):
            Grid(nx=2, ny=2)
    
    def test_laplacian_constant_function(self):
        """Test Laplacian of constant function should be zero."""
        grid = Grid(nx=7, ny=7)
        laplacian = LaplacianOperator()
        
        # Constant function in interior
        field = np.ones(grid.shape)
        
        result = laplacian.apply(grid, field)
        
        # Laplacian of constant should be zero in interior
        interior_result = result[1:-1, 1:-1]
        np.testing.assert_allclose(interior_result, 0.0, atol=1e-12)
    
    def test_laplacian_linear_function(self):
        """Test Laplacian of linear function should be zero."""
        grid = Grid(nx=7, ny=7)
        laplacian = LaplacianOperator()
        
        # Linear function u(x,y) = x + 2*y
        field = grid.X + 2 * grid.Y
        
        result = laplacian.apply(grid, field)
        
        # Laplacian of linear function should be zero
        interior_result = result[1:-1, 1:-1]
        np.testing.assert_allclose(interior_result, 0.0, atol=1e-10)
    
    def test_laplacian_quadratic_function(self):
        """Test Laplacian of quadratic function."""
        grid = Grid(nx=9, ny=9, domain=(0, 1, 0, 1))
        laplacian = LaplacianOperator()
        
        # Quadratic function u(x,y) = x² + y²
        # Laplacian should be ∇²u = 2 + 2 = 4
        field = grid.X**2 + grid.Y**2
        
        result = laplacian.apply(grid, field)
        
        # Check interior points (should be approximately 4)
        interior_result = result[1:-1, 1:-1]
        np.testing.assert_allclose(interior_result, 4.0, rtol=1e-2)
    
    def test_laplacian_with_different_coefficients(self):
        """Test Laplacian with different coefficients."""
        grid = Grid(nx=7, ny=7)
        
        laplacian1 = LaplacianOperator(coefficient=1.0)
        laplacian2 = LaplacianOperator(coefficient=2.0)
        
        field = grid.X**2 + grid.Y**2
        
        result1 = laplacian1.apply(grid, field)
        result2 = laplacian2.apply(grid, field)
        
        # Result should be scaled by coefficient
        np.testing.assert_allclose(result2, 2.0 * result1, rtol=1e-12)
    
    def test_laplacian_single_point(self):
        """Test Laplacian at a single point."""
        grid = Grid(nx=7, ny=7)
        laplacian = LaplacianOperator()
        
        field = np.zeros(grid.shape)
        field[3, 3] = 1.0  # Point source at center
        
        # Apply stencil at neighboring point
        result_point = laplacian.apply_stencil(grid, field, 3, 2)
        
        # Should get contribution from the point source
        expected = 1.0 / grid.hy**2
        assert abs(result_point - expected) < 1e-10
    
    def test_laplacian_residual_computation(self):
        """Test residual computation."""
        grid = Grid(nx=7, ny=7)
        laplacian = LaplacianOperator()
        
        # Exact solution: u(x,y) = sin(πx)sin(πy)
        # RHS: f = 2π²sin(πx)sin(πy)
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        f = 2 * np.pi**2 * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        
        # Residual should be close to zero for exact solution
        residual = laplacian.residual(grid, u_exact, f)
        
        # Interior residual should be small
        interior_residual = residual[1:-1, 1:-1]
        assert np.max(np.abs(interior_residual)) < 1e-2
    
    def test_laplacian_eigenvalues(self):
        """Test eigenvalue computation for 1D case."""
        laplacian = LaplacianOperator()
        
        n = 10
        h = 1.0 / (n + 1)
        
        eigenvals = laplacian.eigenvalues_1d(n, h)
        
        # Check that eigenvalues are negative (for stability)
        assert np.all(eigenvals < 0)
        
        # Check that eigenvalues are ordered
        assert np.all(eigenvals[:-1] >= eigenvals[1:])
    
    def test_laplacian_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        grid = Grid(nx=5, ny=5)
        laplacian = LaplacianOperator()
        
        # Wrong field shape
        field_wrong = np.zeros((3, 3))
        with pytest.raises(ValueError, match="doesn't match grid shape"):
            laplacian.apply(grid, field_wrong)
        
        # Invalid point for stencil
        field = np.zeros(grid.shape)
        with pytest.raises(ValueError, match="not an interior point"):
            laplacian.apply_stencil(grid, field, 0, 0)  # Boundary point


class TestRestrictionOperator:
    """Test cases for restriction operator."""
    
    def test_restriction_initialization(self):
        """Test restriction operator initialization."""
        restriction = RestrictionOperator()
        assert restriction.method == "full_weighting"
        
        restriction_injection = RestrictionOperator("injection")
        assert restriction_injection.method == "injection"
    
    def test_restriction_can_apply(self):
        """Test can_apply method."""
        fine_grid = Grid(nx=9, ny=9)
        coarse_grid = Grid(nx=5, ny=5)
        
        restriction = RestrictionOperator()
        assert restriction.can_apply(fine_grid, coarse_grid)
        
        # Invalid coarse grid
        wrong_coarse = Grid(nx=4, ny=4)
        assert not restriction.can_apply(fine_grid, wrong_coarse)
    
    def test_restriction_injection(self):
        """Test injection restriction."""
        fine_grid = Grid(nx=9, ny=9)
        coarse_grid = Grid(nx=5, ny=5)
        
        restriction = RestrictionOperator("injection")
        
        # Create test field on fine grid
        fine_field = np.zeros(fine_grid.shape)
        fine_field[4, 4] = 1.0  # Center point
        
        coarse_field = restriction.apply(fine_grid, fine_field, coarse_grid)
        
        # Center point should be transferred
        assert coarse_field[2, 2] == 1.0
        assert np.sum(coarse_field) == 1.0
    
    def test_restriction_full_weighting(self):
        """Test full weighting restriction."""
        fine_grid = Grid(nx=9, ny=9)
        coarse_grid = Grid(nx=5, ny=5)
        
        restriction = RestrictionOperator("full_weighting")
        
        # Constant field should remain constant
        fine_field = np.ones(fine_grid.shape)
        coarse_field = restriction.apply(fine_grid, fine_field, coarse_grid)
        
        # Interior points should be 1.0 (constant preserved)
        interior_coarse = coarse_field[1:-1, 1:-1]
        np.testing.assert_allclose(interior_coarse, 1.0, atol=1e-12)
    
    def test_restriction_conservation(self):
        """Test that restriction preserves certain properties."""
        fine_grid = Grid(nx=9, ny=9, domain=(0, 1, 0, 1))
        coarse_grid = Grid(nx=5, ny=5, domain=(0, 1, 0, 1))
        
        restriction = RestrictionOperator("full_weighting")
        
        # Create a smooth test function
        fine_field = np.sin(np.pi * fine_grid.X) * np.sin(np.pi * fine_grid.Y)
        coarse_field = restriction.apply(fine_grid, fine_field, coarse_grid)
        
        # Check that coarse field has reasonable values
        assert np.all(np.isfinite(coarse_field))
        assert np.max(np.abs(coarse_field)) <= 1.1  # Slightly larger than max of sin
    
    def test_restriction_invalid_method(self):
        """Test invalid restriction method."""
        with pytest.raises(ValueError, match="Unknown restriction method"):
            RestrictionOperator("invalid_method")


class TestProlongationOperator:
    """Test cases for prolongation operator."""
    
    def test_prolongation_initialization(self):
        """Test prolongation operator initialization."""
        prolongation = ProlongationOperator()
        assert prolongation.method == "bilinear"
        
        prolongation_injection = ProlongationOperator("injection")  
        assert prolongation_injection.method == "injection"
    
    def test_prolongation_can_apply(self):
        """Test can_apply method."""
        coarse_grid = Grid(nx=5, ny=5)
        fine_grid = Grid(nx=9, ny=9)
        
        prolongation = ProlongationOperator()
        assert prolongation.can_apply(coarse_grid, fine_grid)
        
        # Invalid fine grid
        wrong_fine = Grid(nx=7, ny=7)
        assert not prolongation.can_apply(coarse_grid, wrong_fine)
    
    def test_prolongation_injection(self):
        """Test injection prolongation."""
        coarse_grid = Grid(nx=5, ny=5)
        fine_grid = Grid(nx=9, ny=9)
        
        prolongation = ProlongationOperator("injection")
        
        # Create test field on coarse grid
        coarse_field = np.zeros(coarse_grid.shape)
        coarse_field[2, 2] = 1.0  # Center point
        
        fine_field = prolongation.apply(coarse_grid, coarse_field, fine_grid)
        
        # Should have nonzero value at corresponding fine grid point
        assert fine_field[4, 4] == 1.0
    
    def test_prolongation_bilinear(self):
        """Test bilinear prolongation."""
        coarse_grid = Grid(nx=3, ny=3, domain=(0, 1, 0, 1))
        fine_grid = Grid(nx=5, ny=5, domain=(0, 1, 0, 1))
        
        prolongation = ProlongationOperator("bilinear")
        
        # Linear function should be preserved exactly
        coarse_field = coarse_grid.X + coarse_grid.Y
        fine_field = prolongation.apply(coarse_grid, coarse_field, fine_grid)
        
        expected_fine = fine_grid.X + fine_grid.Y
        
        # Should match for a linear function (up to numerical precision)
        np.testing.assert_allclose(fine_field, expected_fine, atol=1e-10)
    
    def test_prolongation_constant_preservation(self):
        """Test that prolongation preserves constants."""
        coarse_grid = Grid(nx=5, ny=5)
        fine_grid = Grid(nx=9, ny=9)
        
        prolongation = ProlongationOperator("bilinear")
        
        # Constant field
        coarse_field = np.ones(coarse_grid.shape) * 3.14
        fine_field = prolongation.apply(coarse_grid, coarse_field, fine_grid)
        
        # Should preserve constant value
        np.testing.assert_allclose(fine_field, 3.14, atol=1e-12)
    
    def test_prolongation_restriction_consistency(self):
        """Test that prolongation followed by restriction gives reasonable results."""
        coarse_grid = Grid(nx=5, ny=5, domain=(0, 1, 0, 1))
        fine_grid = Grid(nx=9, ny=9, domain=(0, 1, 0, 1))
        
        prolongation = ProlongationOperator("bilinear")
        restriction = RestrictionOperator("full_weighting")
        
        # Start with coarse field
        original_coarse = np.sin(np.pi * coarse_grid.X) * np.sin(np.pi * coarse_grid.Y)
        
        # Prolongate to fine, then restrict back
        fine_field = prolongation.apply(coarse_grid, original_coarse, fine_grid)
        recovered_coarse = restriction.apply(fine_grid, fine_field, coarse_grid)
        
        # Should be reasonably close to original
        np.testing.assert_allclose(recovered_coarse, original_coarse, rtol=0.1)
    
    def test_prolongation_invalid_method(self):
        """Test invalid prolongation method."""
        with pytest.raises(ValueError, match="Unknown prolongation method"):
            ProlongationOperator("invalid_method")


if __name__ == "__main__":
    pytest.main([__file__])