#!/usr/bin/env python3
"""
Test Script for Advanced Visualizations

This script tests all the newly implemented advanced visualization capabilities
to ensure they work correctly with the existing codebase.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def test_imports():
    """Test that all visualization modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from multigrid.visualization import create_missing_visualizations
        print("✓ Advanced visualizations import successful")
        
        from multigrid.visualization.advanced_visualizations import AdvancedVisualizationTools
        print("✓ AdvancedVisualizationTools import successful")
        
        # Test creating instance
        viz_tools = create_missing_visualizations()
        print("✓ Visualization tools instance created successfully")
        
        return viz_tools
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return None

def test_3d_visualization(viz_tools):
    """Test 3D solution visualization with minimal data."""
    print("\nTesting 3D visualization...")
    
    try:
        # Create minimal 3D data
        nx, ny, nz = 8, 8, 4
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 0.5, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        solution = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-Z)
        
        solution_data = {'Test_Solution': solution}
        grid_coords = {'x': x, 'y': y, 'z': z}
        
        fig, axes, widgets = viz_tools.create_interactive_3d_solution_visualization(
            solution_data, grid_coords, title="Test 3D Visualization"
        )
        
        print("✓ 3D visualization created successfully")
        plt.close(fig)  # Close to prevent display during testing
        return True
        
    except Exception as e:
        print(f"✗ 3D visualization failed: {e}")
        return False

def test_convergence_comparison(viz_tools):
    """Test convergence history comparison."""
    print("\nTesting convergence comparison...")
    
    try:
        # Create sample convergence data
        methods = ['Method_A', 'Method_B']
        convergence_data = {}
        
        for method in methods:
            n_iter = 20
            residuals = [1.0 * (0.5 ** i) for i in range(n_iter)]
            errors = [r * 1.1 for r in residuals]
            times = [0.01] * n_iter
            
            convergence_data[method] = {
                'residual': residuals,
                'error': errors,
                'solve_time': times
            }
        
        fig, axes, widgets = viz_tools.create_convergence_history_comparison(
            convergence_data, statistical_analysis=False,
            title="Test Convergence Comparison"
        )
        
        print("✓ Convergence comparison created successfully")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Convergence comparison failed: {e}")
        return False

def test_gpu_memory_visualization(viz_tools):
    """Test GPU memory visualization."""
    print("\nTesting GPU memory visualization...")
    
    try:
        # Create sample memory data
        memory_data = {
            '0': {
                'allocated': [1000, 1200, 1100, 1300],
                'cached': [200, 240, 220, 260],
                'free': [6792, 6552, 6672, 6432],
                'total': [8192, 8192, 8192, 8192],
                'max_memory': [8192, 8192, 8192, 8192]
            }
        }
        
        fig, axes, widgets = viz_tools.create_gpu_memory_visualization(
            memory_data, real_time=False, title="Test GPU Memory"
        )
        
        print("✓ GPU memory visualization created successfully")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ GPU memory visualization failed: {e}")
        return False

def test_error_propagation_analysis(viz_tools):
    """Test precision error propagation analysis."""
    print("\nTesting error propagation analysis...")
    
    try:
        # Create sample error data
        n_ops, n_vars = 50, 20
        error_data = {
            'fp32': {
                'error_matrix': np.random.exponential(1e-6, (n_ops, n_vars)),
                'error_evolution': np.random.exponential(1e-6, n_ops),
                'max_errors': np.random.exponential(1e-5, n_ops),
                'std_errors': np.random.exponential(1e-7, n_ops)
            }
        }
        
        fig, axes, widgets = viz_tools.create_precision_error_propagation_analysis(
            error_data, precision_levels=['fp32'], title="Test Error Analysis"
        )
        
        print("✓ Error propagation analysis created successfully")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error propagation analysis failed: {e}")
        return False

def test_scaling_analysis(viz_tools):
    """Test performance scaling with error bars."""
    print("\nTesting scaling analysis...")
    
    try:
        # Create sample scaling data
        scaling_data = {
            'Test_Method': {
                'solve_time': [100, 52, 28, 16],
                'efficiency': [1.0, 0.96, 0.89, 0.78],
                'speedup': [1.0, 1.92, 3.57, 6.25],
                'memory_usage': [1024, 1024, 1024, 1024]
            }
        }
        
        confidence_intervals = {
            'Test_Method': {
                'solve_time': [5, 3, 2, 1.5],
                'efficiency': [0.02, 0.03, 0.04, 0.05],
                'speedup': [0.05, 0.1, 0.2, 0.3],
                'memory_usage': [50, 50, 50, 50]
            }
        }
        
        fig, axes, widgets = viz_tools.create_performance_scaling_with_error_bars(
            scaling_data, confidence_intervals, title="Test Scaling Analysis"
        )
        
        print("✓ Scaling analysis created successfully")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Scaling analysis failed: {e}")
        return False

def test_integration_with_existing_modules():
    """Test integration with existing visualization modules."""
    print("\nTesting integration with existing modules...")
    
    try:
        # Test importing from main visualization module
        from multigrid.visualization import AdvancedVisualizationTools
        from multigrid.visualization import create_missing_visualizations
        
        # Test that it's included in __all__
        import multigrid.visualization as viz_module
        assert 'AdvancedVisualizationTools' in viz_module.__all__
        assert 'create_missing_visualizations' in viz_module.__all__
        
        print("✓ Integration with existing modules successful")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("Advanced Visualizations Test Suite")
    print("=" * 40)
    
    # Test results
    results = {}
    
    # Test imports
    viz_tools = test_imports()
    results['imports'] = viz_tools is not None
    
    if viz_tools:
        # Test individual components
        results['3d_viz'] = test_3d_visualization(viz_tools)
        results['convergence'] = test_convergence_comparison(viz_tools)
        results['gpu_memory'] = test_gpu_memory_visualization(viz_tools)
        results['error_analysis'] = test_error_propagation_analysis(viz_tools)
        results['scaling'] = test_scaling_analysis(viz_tools)
    else:
        # Skip other tests if imports failed
        for key in ['3d_viz', 'convergence', 'gpu_memory', 'error_analysis', 'scaling']:
            results[key] = False
    
    # Test integration
    results['integration'] = test_integration_with_existing_modules()
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST RESULTS SUMMARY")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{test_name:20} {status:>4}")
    
    print("-" * 40)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Advanced visualizations are ready to use.")
        print("\nNext steps:")
        print("1. Run the demo: python examples/advanced_visualizations_demo.py")
        print("2. Check documentation: docs/ADVANCED_VISUALIZATIONS.md")
        print("3. Integrate with your analysis workflows")
        
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)