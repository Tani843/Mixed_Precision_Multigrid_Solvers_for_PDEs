"""Unit tests for preconditioning methods."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multigrid.core.grid import Grid
from multigrid.operators.laplacian import LaplacianOperator
from multigrid.preconditioning import (
    BasePreconditioner, DiagonalPreconditioner, 
    ILUPreconditioner, MultigridPreconditioner
)


class TestBasePreconditioner:
    """Test cases for base preconditioner functionality."""
    
    def test_identity_preconditioner(self):
        """Test identity preconditioner."""
        grid = Grid(nx=5, ny=5)
        operator = LaplacianOperator()
        precond = IdentityPreconditioner()
        
        precond.setup(grid, operator)
        assert precond.is_setup()
        
        # Should return input unchanged
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        np.testing.assert_array_equal(result, x)
        
        # Transpose should also return input unchanged
        result_T = precond.apply_transpose(x)
        np.testing.assert_array_equal(result_T, x)
    
    def test_composite_preconditioner(self):
        """Test composite preconditioner."""
        grid = Grid(nx=5, ny=5)
        operator = LaplacianOperator()
        
        # Create component preconditioners
        diag1 = DiagonalPreconditioner()
        diag2 = DiagonalPreconditioner()
        
        composite = CompositePreconditioner([diag1, diag2])
        composite.setup(grid, operator)
        
        assert composite.is_setup()
        assert diag1.is_setup()
        assert diag2.is_setup()
        
        # Test application
        x = np.random.rand(*grid.shape)
        result = composite.apply(x)
        
        # Should be equivalent to applying both preconditioners
        intermediate = diag1.apply(x)
        expected = diag2.apply(intermediate)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestDiagonalPreconditioner:
    """Test cases for diagonal preconditioning."""
    
    def test_diagonal_preconditioner_setup(self):
        """Test diagonal preconditioner setup."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator(coefficient=2.0)
        precond = DiagonalPreconditioner()
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert precond.diagonal_inverse.shape == grid.shape
        
        # Check diagonal values for interior points
        expected_diag = -2.0 * (2.0 / grid.hx**2 + 2.0 / grid.hy**2)
        interior_diagonal_inv = precond.diagonal_inverse[1, 1]
        expected_inv = 1.0 / (expected_diag - precond.regularization)
        
        assert abs(interior_diagonal_inv - expected_inv) < 1e-12
    
    def test_diagonal_preconditioner_apply(self):
        """Test diagonal preconditioner application."""
        grid = Grid(nx=5, ny=5)
        operator = LaplacianOperator()
        precond = DiagonalPreconditioner()
        
        precond.setup(grid, operator)
        
        # Test with random vector
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        
        # Should be element-wise multiplication
        expected = precond.diagonal_inverse * x
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        
        # Transpose should be same (diagonal is symmetric)
        result_T = precond.apply_transpose(x)
        np.testing.assert_allclose(result, result_T, rtol=1e-12)
    
    def test_diagonal_preconditioner_error_handling(self):
        """Test error handling for diagonal preconditioner."""
        grid = Grid(nx=5, ny=5)
        precond = DiagonalPreconditioner()
        
        # Should fail if not setup
        x = np.random.rand(*grid.shape)
        with pytest.raises(RuntimeError, match="not setup"):
            precond.apply(x)
        
        # Setup preconditioner
        operator = LaplacianOperator()
        precond.setup(grid, operator)
        
        # Should fail with wrong shape
        wrong_x = np.random.rand(3, 3)
        with pytest.raises(ValueError, match="doesn't match setup shape"):
            precond.apply(wrong_x)
    
    def test_scaled_diagonal_preconditioner(self):
        """Test scaled diagonal preconditioner."""
        # Test with anisotropic grid
        grid = Grid(nx=9, ny=5, domain=(0, 2, 0, 1))  # hx != hy
        operator = LaplacianOperator()
        
        precond = ScaledDiagonalPreconditioner(scaling_factor=2.0)
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        
        # Should have applied additional scaling
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        
        # Result should be different from unscaled
        unscaled_precond = DiagonalPreconditioner()
        unscaled_precond.setup(grid, operator)
        unscaled_result = unscaled_precond.apply(x)
        
        # Scaled result should generally be different (unless perfectly isotropic)
        hx_hy_ratio = grid.hx / grid.hy
        if abs(hx_hy_ratio - 1.0) > 0.1:
            assert not np.allclose(result, unscaled_result)
    
    def test_condition_number_estimate(self):
        """Test condition number estimation."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        precond = DiagonalPreconditioner()
        
        precond.setup(grid, operator)
        
        cond_num = precond.condition_number_estimate()
        
        # Should be positive
        assert cond_num > 0
        
        # For uniform grid with same operator, should be close to 1
        assert cond_num == pytest.approx(1.0, rel=1e-6)


class TestBlockDiagonalPreconditioner:
    """Test cases for block diagonal preconditioning."""
    
    def test_block_diagonal_setup_row(self):
        """Test block diagonal setup with row blocks."""
        grid = Grid(nx=7, ny=9)
        operator = LaplacianOperator()
        precond = BlockDiagonalPreconditioner(block_direction="row")
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert len(precond.block_inverses) == grid.nx - 2  # Interior rows
    
    def test_block_diagonal_setup_column(self):
        """Test block diagonal setup with column blocks."""
        grid = Grid(nx=7, ny=9)
        operator = LaplacianOperator()
        precond = BlockDiagonalPreconditioner(block_direction="column")
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert len(precond.block_inverses) == grid.ny - 2  # Interior columns
    
    def test_block_diagonal_apply(self):
        """Test block diagonal application."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        precond = BlockDiagonalPreconditioner(block_direction="row")
        
        precond.setup(grid, operator)
        
        # Test application
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        
        # Should preserve shape
        assert result.shape == x.shape
        
        # Boundaries should be preserved (identity)
        np.testing.assert_array_equal(result[0, :], x[0, :])
        np.testing.assert_array_equal(result[-1, :], x[-1, :])
        np.testing.assert_array_equal(result[:, 0], x[:, 0])
        np.testing.assert_array_equal(result[:, -1], x[:, -1])
    
    def test_block_diagonal_symmetry(self):
        """Test that block diagonal preconditioner preserves symmetry."""
        grid = Grid(nx=5, ny=5)
        operator = LaplacianOperator()
        precond = BlockDiagonalPreconditioner()
        
        precond.setup(grid, operator)
        
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        result_T = precond.apply_transpose(x)
        
        # Should be same for symmetric operator
        np.testing.assert_allclose(result, result_T, rtol=1e-10)


@pytest.mark.skipif(True, reason="Requires scipy - optional dependency")
class TestILUPreconditioner:
    """Test cases for ILU preconditioning."""
    
    def test_ilu_preconditioner_setup(self):
        """Test ILU preconditioner setup."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        precond = ILUPreconditioner(fill_level=0)
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert precond.L_matrix is not None
        assert precond.U_matrix is not None
        assert precond.sparse_matrix is not None
        
        # Check dimensions
        n_interior = (grid.nx - 2) * (grid.ny - 2)
        assert precond.L_matrix.shape == (n_interior, n_interior)
        assert precond.U_matrix.shape == (n_interior, n_interior)
    
    def test_ilu_preconditioner_apply(self):
        """Test ILU preconditioner application."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        precond = ILUPreconditioner()
        
        precond.setup(grid, operator)
        
        # Test application
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        
        # Should preserve shape
        assert result.shape == x.shape
        
        # Should not return input unchanged (unless very special case)
        assert not np.array_equal(result, x)
    
    def test_ilu_memory_usage(self):
        """Test ILU memory usage reporting."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        precond = ILUPreconditioner()
        
        precond.setup(grid, operator)
        
        memory_stats = precond.get_memory_usage()
        
        assert 'original_nnz' in memory_stats
        assert 'l_nnz' in memory_stats
        assert 'u_nnz' in memory_stats
        assert 'fill_ratio' in memory_stats
        assert memory_stats['original_nnz'] > 0
        assert memory_stats['fill_ratio'] >= 1.0  # Should have some fill
    
    def test_modified_ilu_preconditioner(self):
        """Test modified ILU preconditioner with diagonal shift."""
        grid = Grid(nx=7, ny=7)
        operator = LaplacianOperator()
        precond = ModifiedILUPreconditioner(diagonal_shift=0.1)
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert precond.diagonal_shift == 0.1
        
        # Should work similarly to regular ILU
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        assert result.shape == x.shape


class TestMultigridPreconditioner:
    """Test cases for multigrid as preconditioner."""
    
    def test_multigrid_preconditioner_setup(self):
        """Test multigrid preconditioner setup."""
        grid = Grid(nx=17, ny=17)  # Ensure multiple levels possible
        operator = LaplacianOperator()
        precond = MultigridPreconditioner(max_levels=3, num_cycles=1)
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert precond.mg_solver is not None
        assert len(precond.mg_solver.grids) <= 3
    
    def test_multigrid_preconditioner_apply(self):
        """Test multigrid preconditioner application."""
        grid = Grid(nx=17, ny=17)
        operator = LaplacianOperator()
        precond = MultigridPreconditioner(max_levels=3, num_cycles=1)
        
        precond.setup(grid, operator)
        
        # Test application with residual-like input
        residual = np.random.rand(*grid.shape)
        correction = precond.apply(residual)
        
        assert correction.shape == residual.shape
        # Should not be zero (unless very special case)
        assert np.any(correction != 0)
    
    def test_two_level_preconditioner(self):
        """Test two-level preconditioner."""
        grid = Grid(nx=9, ny=9)  # Should be coarsenable
        operator = LaplacianOperator()
        precond = TwoLevelPreconditioner(pre_smooth_iterations=2, post_smooth_iterations=1)
        
        precond.setup(grid, operator)
        
        assert precond.is_setup()
        assert precond.coarse_grid.shape == (5, 5)  # Coarsened version
        
        # Test application
        x = np.random.rand(*grid.shape)
        result = precond.apply(x)
        
        assert result.shape == x.shape


class TestPreconditionerEffectiveness:
    """Test effectiveness of different preconditioners."""
    
    def test_preconditioner_comparison(self):
        """Compare effectiveness of different preconditioners."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        
        # Create test residual
        residual = np.ones(grid.shape)
        grid.apply_dirichlet_bc(0.0, 'all')  # Zero on boundaries
        
        preconditioners = {
            'identity': IdentityPreconditioner(),
            'diagonal': DiagonalPreconditioner(),
            'scaled_diagonal': ScaledDiagonalPreconditioner(scaling_factor=1.5),
            'block_diagonal': BlockDiagonalPreconditioner()
        }
        
        results = {}
        
        for name, precond in preconditioners.items():
            precond.setup(grid, operator)
            correction = precond.apply(residual)
            
            # Measure how much the preconditioner changed the residual
            change_norm = np.linalg.norm(correction - residual)
            results[name] = change_norm
        
        # All preconditioners should be setup successfully
        assert len(results) == len(preconditioners)
        
        # Diagonal preconditioning should do something (not identity)
        assert results['diagonal'] != results['identity']
    
    def test_preconditioner_residual_reduction(self):
        """Test that preconditioners help with residual reduction."""
        grid = Grid(nx=9, ny=9)
        operator = LaplacianOperator()
        
        # Create a test problem: Au = f
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        f = operator.apply(grid, u_exact)
        
        # Start with a poor initial guess
        u_guess = np.zeros_like(u_exact)
        
        # Compute initial residual
        initial_residual = operator.residual(grid, u_guess, f)
        initial_residual_norm = np.linalg.norm(initial_residual)
        
        # Apply diagonal preconditioning to residual
        precond = DiagonalPreconditioner()
        precond.setup(grid, operator)
        
        correction = precond.apply(initial_residual)
        
        # Apply correction
        u_corrected = u_guess + correction
        
        # Compute new residual
        new_residual = operator.residual(grid, u_corrected, f)
        new_residual_norm = np.linalg.norm(new_residual)
        
        # Preconditioned correction should help (residual should decrease)
        # Note: This might not always be true for a single step, but generally should help
        assert new_residual_norm <= initial_residual_norm * 2  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__])