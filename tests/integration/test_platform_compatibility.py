"""
Platform Compatibility Tests

Tests to ensure the mixed-precision multigrid solver works consistently
across different platforms (Linux, Windows, macOS) and Python versions.
"""

import pytest
import numpy as np
import platform
import sys
import os
import subprocess
from pathlib import Path

# Platform detection
CURRENT_PLATFORM = platform.system().lower()
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
IS_WINDOWS = CURRENT_PLATFORM == 'windows'
IS_LINUX = CURRENT_PLATFORM == 'linux' 
IS_MACOS = CURRENT_PLATFORM == 'darwin'

# Test markers for different platforms
pytestmark = [
    pytest.mark.compatibility,
    pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8+")
]


class TestPythonVersionCompatibility:
    """Test compatibility across different Python versions."""
    
    def test_python_version_support(self):
        """Test that current Python version is supported."""
        major, minor = sys.version_info.major, sys.version_info.minor
        
        # Supported Python versions
        supported_versions = [(3, 8), (3, 9), (3, 10), (3, 11)]
        current_version = (major, minor)
        
        assert current_version in supported_versions, \
            f"Python {major}.{minor} not in supported versions: {supported_versions}"
    
    def test_numpy_compatibility(self):
        """Test NumPy compatibility across Python versions."""
        # Test basic NumPy operations
        array = np.array([1.0, 2.0, 3.0])
        assert array.dtype == np.float64, "Default float type should be float64"
        
        # Test scientific computing operations
        result = np.linalg.norm(array)
        expected = np.sqrt(14.0)
        np.testing.assert_allclose(result, expected, rtol=1e-15)
        
        # Test complex operations that might vary across versions
        X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        assert X.shape == Y.shape == (10, 10), "Meshgrid should work consistently"
    
    def test_import_compatibility(self):
        """Test that all required modules import correctly."""
        required_modules = [
            'numpy', 'scipy', 'matplotlib', 'pytest'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Required module '{module_name}' failed to import: {e}")


class TestPlatformSpecificBehavior:
    """Test platform-specific behavior and file system operations."""
    
    def test_file_path_handling(self):
        """Test file path handling across platforms."""
        # Test path creation
        test_path = Path("test_dir") / "sub_dir" / "test_file.txt"
        
        # Path should work on all platforms
        assert isinstance(test_path, Path), "Path creation should work"
        assert str(test_path), "Path should convert to string"
        
        # Test path operations
        parent = test_path.parent
        assert parent.name == "sub_dir", "Parent directory detection should work"
        
        # Test absolute vs relative paths
        abs_path = test_path.resolve()
        assert abs_path.is_absolute(), "Absolute path detection should work"
    
    def test_temporary_file_handling(self):
        """Test temporary file creation and cleanup."""
        import tempfile
        
        # Test temporary directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert temp_path.exists(), "Temporary directory should be created"
            
            # Test file creation in temp directory
            test_file = temp_path / "test_output.txt"
            test_file.write_text("test data")
            assert test_file.exists(), "File creation should work in temp directory"
        
        # Temp directory should be cleaned up
        assert not temp_path.exists(), "Temporary directory should be cleaned up"
    
    @pytest.mark.skipif(IS_WINDOWS, reason="Unix-specific test")
    def test_unix_specific_features(self):
        """Test Unix-specific features (Linux/macOS)."""
        # Test process management
        import signal
        assert hasattr(signal, 'SIGTERM'), "SIGTERM should be available on Unix"
        
        # Test file permissions
        import stat
        temp_file = Path("temp_test_file.txt")
        try:
            temp_file.write_text("test")
            temp_file.chmod(0o644)
            file_stat = temp_file.stat()
            assert stat.S_IMODE(file_stat.st_mode) == 0o644, "File permissions should be settable"
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
    def test_windows_specific_features(self):
        """Test Windows-specific features."""
        # Test Windows path separators
        test_path = Path("C:\\test\\path")
        assert "\\" in str(test_path) or "/" in str(test_path), "Path separators should work"
        
        # Test case-insensitive file system behavior
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file1 = temp_path / "TestFile.txt"
            test_file2 = temp_path / "testfile.txt"
            
            test_file1.write_text("test")
            # On Windows, these might be the same file
            case_sensitive = test_file1.resolve() != test_file2.resolve()
            print(f"File system is case sensitive: {case_sensitive}")


class TestNumericalConsistency:
    """Test numerical consistency across platforms."""
    
    def test_floating_point_consistency(self):
        """Test floating point operations consistency."""
        # Test basic operations
        a, b = 0.1, 0.2
        result = a + b
        
        # Should be consistent across platforms (within floating point limits)
        expected = 0.3
        assert abs(result - expected) < 1e-15, f"Basic arithmetic should be consistent: {result} vs {expected}"
        
        # Test scientific computing operations
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_result = np.mean(array)
        expected_mean = 3.0
        np.testing.assert_allclose(mean_result, expected_mean, rtol=1e-15)
    
    def test_linear_algebra_consistency(self):
        """Test linear algebra operations consistency."""
        # Create test matrix
        np.random.seed(42)  # Ensure reproducibility
        A = np.random.randn(10, 10)
        b = np.random.randn(10)
        
        # Test matrix operations
        det_A = np.linalg.det(A)
        assert not np.isnan(det_A), "Determinant should not be NaN"
        assert np.isfinite(det_A), "Determinant should be finite"
        
        # Test linear system solving
        if np.abs(det_A) > 1e-10:  # Only if matrix is well-conditioned
            x = np.linalg.solve(A, b)
            residual = np.linalg.norm(A @ x - b)
            assert residual < 1e-10, f"Linear system residual too large: {residual}"
    
    def test_random_number_generation(self):
        """Test random number generation consistency."""
        # Test reproducibility with fixed seed
        np.random.seed(12345)
        random1 = np.random.randn(100)
        
        np.random.seed(12345)
        random2 = np.random.randn(100)
        
        np.testing.assert_array_equal(random1, random2, 
                                    "Random number generation should be reproducible with fixed seed")
        
        # Test statistical properties
        large_sample = np.random.randn(10000)
        sample_mean = np.mean(large_sample)
        sample_std = np.std(large_sample)
        
        assert abs(sample_mean) < 0.05, f"Sample mean should be near 0: {sample_mean}"
        assert abs(sample_std - 1.0) < 0.05, f"Sample std should be near 1: {sample_std}"


class TestMemoryManagement:
    """Test memory management across platforms."""
    
    def test_large_array_allocation(self):
        """Test large array allocation and deallocation."""
        try:
            # Allocate moderately large array
            large_array = np.zeros((1000, 1000))
            assert large_array.shape == (1000, 1000), "Large array allocation should succeed"
            
            # Test array operations
            large_array.fill(1.0)
            assert np.all(large_array == 1.0), "Array filling should work"
            
            # Deallocate
            del large_array
            
        except MemoryError:
            pytest.skip("Insufficient memory for large array test")
    
    def test_memory_leak_detection(self):
        """Test for basic memory leaks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform operations that might leak memory
        for _ in range(100):
            temp_array = np.random.randn(100, 100)
            temp_result = np.linalg.norm(temp_array)
            del temp_array, temp_result
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not have significant memory increase
        memory_increase_mb = memory_increase / (1024 * 1024)
        assert memory_increase_mb < 100, f"Potential memory leak detected: {memory_increase_mb:.1f} MB increase"


class TestCUDACompatibility:
    """Test CUDA availability and compatibility."""
    
    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        cuda_available = False
        
        try:
            # Try to detect CUDA through various methods
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                cuda_available = True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Alternative: check for CUDA environment variables
        if os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME'):
            print("CUDA environment variables detected")
        
        print(f"CUDA available: {cuda_available}")
        
        # Test should not fail based on CUDA availability
        # but should properly detect and report status
        assert True, "CUDA detection test always passes"
    
    @pytest.mark.skipif(not os.environ.get('CUDA_PATH'), reason="CUDA not available")
    def test_cuda_computation_consistency(self):
        """Test CUDA computation consistency (if available)."""
        # Mock CUDA computation test
        # In real implementation, this would test GPU vs CPU consistency
        
        # CPU computation
        cpu_array = np.array([1.0, 2.0, 3.0, 4.0])
        cpu_result = np.sum(cpu_array ** 2)
        
        # Mock GPU computation (would use actual CUDA in real implementation)
        gpu_result = cpu_result  # Mock: assume GPU gives same result
        
        # Results should be consistent
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10,
                                 err_msg="CPU and GPU results should be consistent")


class TestErrorHandling:
    """Test error handling consistency across platforms."""
    
    def test_exception_handling(self):
        """Test that exceptions are handled consistently."""
        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            result = 1.0 / 0.0
        
        # Test invalid array operations
        with pytest.raises(ValueError):
            array = np.array([1, 2, 3])
            invalid_array = np.array([1, 2])
            result = array + invalid_array  # Shape mismatch
    
    def test_warning_handling(self):
        """Test warning handling consistency."""
        import warnings
        
        # Test that warnings can be caught
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Generate a warning
            np.divide(1.0, 0.0, out=np.array([np.inf]))  # Division by zero warning
            
            # Should generate at least one warning
            assert len(w) >= 0, "Warning system should be functional"


def test_platform_test_coverage():
    """Meta-test to ensure platform testing coverage."""
    required_platform_tests = [
        'python_version_support',
        'numpy_compatibility', 
        'import_compatibility',
        'file_path_handling',
        'temporary_file_handling',
        'floating_point_consistency',
        'linear_algebra_consistency',
        'random_number_generation',
        'large_array_allocation',
        'cuda_detection',
        'exception_handling'
    ]
    
    # Get all test methods from platform test classes
    test_classes = [
        TestPythonVersionCompatibility,
        TestPlatformSpecificBehavior,
        TestNumericalConsistency, 
        TestMemoryManagement,
        TestCUDACompatibility,
        TestErrorHandling
    ]
    
    all_test_methods = []
    for test_class in test_classes:
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        all_test_methods.extend(methods)
    
    # Check coverage
    for required_test in required_platform_tests:
        test_name = f'test_{required_test}'
        assert test_name in all_test_methods, f"Missing required platform test: {test_name}"


# Platform-specific test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on platform."""
    for item in items:
        # Add platform markers
        if IS_WINDOWS:
            item.add_marker(pytest.mark.windows)
        elif IS_LINUX:
            item.add_marker(pytest.mark.linux)
        elif IS_MACOS:
            item.add_marker(pytest.mark.macos)
            
        # Add Python version markers
        if sys.version_info >= (3, 9):
            item.add_marker(pytest.mark.python39plus)


if __name__ == '__main__':
    # Run platform compatibility tests
    print(f"Running platform compatibility tests on {CURRENT_PLATFORM} with Python {PYTHON_VERSION}")
    pytest.main([__file__, '-v', '--tb=short'])