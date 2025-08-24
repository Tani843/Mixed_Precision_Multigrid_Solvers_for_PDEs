"""Unit tests for precision management."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multigrid.core.precision import PrecisionManager, PrecisionLevel


class TestPrecisionManager:
    """Test cases for PrecisionManager."""
    
    def test_precision_manager_initialization(self):
        """Test basic initialization."""
        pm = PrecisionManager()
        
        assert pm.default_precision == PrecisionLevel.DOUBLE
        assert pm.current_precision == PrecisionLevel.DOUBLE
        assert pm.adaptive is True
        assert pm.convergence_threshold == 1e-6
    
    def test_precision_parsing(self):
        """Test precision level parsing."""
        pm = PrecisionManager(default_precision="single")
        assert pm.default_precision == PrecisionLevel.SINGLE
        
        pm = PrecisionManager(default_precision="float32")
        assert pm.default_precision == PrecisionLevel.SINGLE
        
        pm = PrecisionManager(default_precision=PrecisionLevel.MIXED)
        assert pm.default_precision == PrecisionLevel.MIXED
        
        with pytest.raises(ValueError, match="Unknown precision level"):
            PrecisionManager(default_precision="invalid")
    
    def test_dtype_conversion(self):
        """Test dtype conversion."""
        pm = PrecisionManager()
        
        assert pm.get_dtype(PrecisionLevel.SINGLE) == np.float32
        assert pm.get_dtype(PrecisionLevel.DOUBLE) == np.float64
        assert pm.get_dtype() == np.float64  # Current precision
    
    def test_array_conversion(self):
        """Test array precision conversion."""
        pm = PrecisionManager()
        
        # Create test array
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        # Convert to single precision
        converted = pm.convert_array(original, PrecisionLevel.SINGLE)
        assert converted.dtype == np.float32
        np.testing.assert_allclose(converted, original, rtol=1e-6)
        
        # Convert back to double
        converted_back = pm.convert_array(converted, PrecisionLevel.DOUBLE)
        assert converted_back.dtype == np.float64
    
    def test_array_conversion_no_change(self):
        """Test array conversion when no change needed."""
        pm = PrecisionManager()
        
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = pm.convert_array(original, PrecisionLevel.DOUBLE)
        
        # Should return same array when no conversion needed
        assert result is original
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        pm = PrecisionManager()
        
        grid_shapes = [(64, 64), (32, 32), (16, 16)]
        memory_bytes = pm.estimate_memory_usage(grid_shapes)
        
        # Should be positive and reasonable
        assert memory_bytes > 0
        
        # Roughly: (64*64 + 32*32 + 16*16) * 8 bytes * 4 arrays
        total_points = 64*64 + 32*32 + 16*16
        expected_approx = total_points * 8 * 4
        assert abs(memory_bytes - expected_approx) < expected_approx * 0.1
    
    def test_precision_downgrade_decision(self):
        """Test precision downgrade logic."""
        pm = PrecisionManager(adaptive=True)
        
        grid_shapes = [(1000, 1000)]  # Large grid
        large_residual = 1.0  # Large residual
        
        # Should suggest downgrade for large memory usage
        should_downgrade = pm.should_downgrade_precision(grid_shapes, large_residual)
        # Result depends on memory threshold, but test that it runs
        assert isinstance(should_downgrade, bool)
        
        # Should suggest downgrade for large residual with double precision
        pm.current_precision = PrecisionLevel.DOUBLE
        should_downgrade = pm.should_downgrade_precision([], large_residual)
        assert should_downgrade is True
    
    def test_precision_upgrade_decision(self):
        """Test precision upgrade logic."""
        pm = PrecisionManager(adaptive=True)
        pm.current_precision = PrecisionLevel.SINGLE
        
        small_residual = 1e-8  # Small residual
        should_upgrade = pm.should_upgrade_precision(small_residual)
        assert should_upgrade is True
        
        # Should not upgrade from double precision
        pm.current_precision = PrecisionLevel.DOUBLE
        should_upgrade = pm.should_upgrade_precision(small_residual)
        assert should_upgrade is False
    
    def test_adaptive_precision_update(self):
        """Test adaptive precision updating."""
        pm = PrecisionManager(adaptive=True)
        pm.current_precision = PrecisionLevel.DOUBLE
        
        # Large residual should trigger downgrade
        large_residual = 1.0
        changed = pm.update_precision(large_residual)
        assert changed is True
        assert pm.current_precision == PrecisionLevel.SINGLE
        
        # Small residual should trigger upgrade
        small_residual = 1e-8
        changed = pm.update_precision(small_residual)
        assert changed is True
        assert pm.current_precision == PrecisionLevel.DOUBLE
    
    def test_non_adaptive_precision(self):
        """Test non-adaptive precision management."""
        pm = PrecisionManager(adaptive=False)
        
        # Should not change precision
        original_precision = pm.current_precision
        changed = pm.update_precision(1e-8)
        assert changed is False
        assert pm.current_precision == original_precision
    
    def test_precision_for_level(self):
        """Test level-specific precision recommendations."""
        pm = PrecisionManager()
        pm.current_precision = PrecisionLevel.MIXED
        
        max_levels = 4
        
        # Fine levels should use double precision
        fine_precision = pm.get_precision_for_level(0, max_levels)
        assert fine_precision == PrecisionLevel.DOUBLE
        
        fine_precision = pm.get_precision_for_level(1, max_levels)
        assert fine_precision == PrecisionLevel.DOUBLE
        
        # Coarse levels should use single precision
        coarse_precision = pm.get_precision_for_level(2, max_levels)
        assert coarse_precision == PrecisionLevel.SINGLE
        
        coarse_precision = pm.get_precision_for_level(3, max_levels)
        assert coarse_precision == PrecisionLevel.SINGLE
    
    def test_precision_for_level_non_mixed(self):
        """Test level precision when not using mixed precision."""
        pm = PrecisionManager()
        pm.current_precision = PrecisionLevel.DOUBLE
        
        # Should return current precision for all levels
        precision = pm.get_precision_for_level(0, 4)
        assert precision == PrecisionLevel.DOUBLE
        
        precision = pm.get_precision_for_level(3, 4)
        assert precision == PrecisionLevel.DOUBLE
    
    def test_statistics_tracking(self):
        """Test precision statistics tracking."""
        pm = PrecisionManager()
        
        # Record some operations
        pm.record_operation(PrecisionLevel.SINGLE, 0.1)
        pm.record_operation(PrecisionLevel.SINGLE, 0.2)
        pm.record_operation(PrecisionLevel.DOUBLE, 0.3)
        
        stats = pm.get_statistics()
        
        assert stats["total_operations"] == 3
        assert stats["total_time"] == 0.6
        assert stats["precision_breakdown"]["float32"]["operations"] == 2
        assert stats["precision_breakdown"]["float64"]["operations"] == 1
    
    def test_statistics_reset(self):
        """Test statistics reset."""
        pm = PrecisionManager()
        
        pm.record_operation(PrecisionLevel.SINGLE, 0.1)
        pm.current_precision = PrecisionLevel.SINGLE
        
        pm.reset_statistics()
        
        stats = pm.get_statistics()
        assert stats["total_operations"] == 0
        assert len(stats["precision_history"]) == 1  # Only current precision
    
    def test_precision_history(self):
        """Test precision change history."""
        pm = PrecisionManager(adaptive=True)
        
        # Initial state
        assert len(pm.precision_history) == 1
        assert pm.precision_history[0] == PrecisionLevel.DOUBLE
        
        # Change precision
        pm.current_precision = PrecisionLevel.SINGLE
        pm.precision_history.append(pm.current_precision)
        
        stats = pm.get_statistics()
        history = stats["precision_history"]
        assert len(history) == 2
        assert history[0] == "double"
        assert history[1] == "single"
    
    def test_string_representations(self):
        """Test string representations."""
        pm = PrecisionManager(default_precision="single", adaptive=True)
        
        str_repr = str(pm)
        assert "single" in str_repr
        assert "adaptive=True" in str_repr
        
        repr_str = repr(pm)
        assert "PrecisionManager" in repr_str
        assert "adaptive=True" in repr_str


class TestPrecisionLevel:
    """Test cases for PrecisionLevel enum."""
    
    def test_precision_level_values(self):
        """Test precision level enum values."""
        assert PrecisionLevel.SINGLE.value == "float32"
        assert PrecisionLevel.DOUBLE.value == "float64"
        assert PrecisionLevel.MIXED.value == "mixed"
    
    def test_precision_level_comparison(self):
        """Test precision level comparisons."""
        assert PrecisionLevel.SINGLE == PrecisionLevel.SINGLE
        assert PrecisionLevel.SINGLE != PrecisionLevel.DOUBLE
        
        # Test membership in collections
        levels = [PrecisionLevel.SINGLE, PrecisionLevel.DOUBLE]
        assert PrecisionLevel.SINGLE in levels
        assert PrecisionLevel.MIXED not in levels


if __name__ == "__main__":
    pytest.main([__file__])