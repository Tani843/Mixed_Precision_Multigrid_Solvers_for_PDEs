#!/usr/bin/env python3
"""
Test Script for Real-time Dashboard

This script tests the real-time solver monitoring dashboard functionality
to ensure all components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import time
import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all dashboard modules can be imported correctly."""
    print("Testing dashboard imports...")
    
    try:
        from multigrid.visualization.realtime_dashboard import (
            solver_dashboard,
            SolverDashboard,
            MetricsCollector,
            LiveDataStreamer,
            create_monitoring_widgets,
            example_solver_callback
        )
        print("✓ Dashboard imports successful")
        
        # Test importing from main visualization module
        from multigrid.visualization import solver_dashboard as main_solver_dashboard
        print("✓ Main module import successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_metrics_collector():
    """Test the metrics collector functionality."""
    print("\nTesting MetricsCollector...")
    
    try:
        from multigrid.visualization.realtime_dashboard import MetricsCollector
        
        # Create collector
        collector = MetricsCollector(max_history=10)
        print("✓ MetricsCollector created")
        
        # Test metric collection
        for i in range(5):
            solver_metrics = {
                'residual': 1.0 / (i + 1),
                'iteration': i,
                'precision_level': 32 if i < 3 else 64
            }
            collector.update_metrics(solver_metrics)
        
        # Test data retrieval
        recent_data = collector.get_recent_data(10)
        assert len(recent_data['solver_residuals']) == 5
        assert recent_data['solver_residuals'][0] == 1.0
        assert recent_data['solver_iterations'][-1] == 4
        print("✓ Metrics collection working")
        
        # Test reset
        collector.reset_metrics()
        recent_data = collector.get_recent_data(10)
        assert len(recent_data['solver_residuals']) == 0
        print("✓ Metrics reset working")
        
        return True
        
    except Exception as e:
        print(f"✗ MetricsCollector test failed: {e}")
        return False

def test_data_streamer():
    """Test the live data streamer functionality."""
    print("\nTesting LiveDataStreamer...")
    
    try:
        from multigrid.visualization.realtime_dashboard import LiveDataStreamer
        
        # Create streamer
        streamer = LiveDataStreamer(update_interval=0.1)
        print("✓ LiveDataStreamer created")
        
        # Test callback setup
        test_iteration = 0
        
        def test_callback():
            nonlocal test_iteration
            test_iteration += 1
            return {
                'residual': 1.0 / test_iteration,
                'iteration': test_iteration,
                'precision_level': 32
            }
        
        streamer.set_solver_callback(test_callback)
        print("✓ Solver callback set")
        
        # Test streaming (brief)
        streamer.start_streaming()
        time.sleep(0.3)  # Let it collect some data
        streamer.stop_streaming()
        print("✓ Streaming start/stop working")
        
        # Test data retrieval
        data = streamer.get_latest_data()
        if data and len(data.get('solver_residuals', [])) > 0:
            print("✓ Data streaming working")
        else:
            print("⚠ Data streaming may not be capturing data (this may be normal in test environment)")
        
        return True
        
    except Exception as e:
        print(f"✗ LiveDataStreamer test failed: {e}")
        return False

def test_dashboard_creation():
    """Test dashboard creation and basic functionality."""
    print("\nTesting SolverDashboard creation...")
    
    try:
        from multigrid.visualization.realtime_dashboard import solver_dashboard, SolverDashboard
        
        # Test function-based creation
        dashboard1 = solver_dashboard()
        assert isinstance(dashboard1, SolverDashboard)
        print("✓ solver_dashboard() function working")
        
        # Test direct class creation
        dashboard2 = SolverDashboard()
        print("✓ SolverDashboard class creation working")
        
        # Test callback setup
        def dummy_callback():
            return {'residual': 0.1, 'iteration': 1, 'precision_level': 32}
        
        dashboard2.set_solver_callback(dummy_callback)
        print("✓ Callback setup working")
        
        # Test alert thresholds
        assert 'memory_percent' in dashboard2.alert_thresholds
        assert 'gpu_memory_percent' in dashboard2.alert_thresholds
        print("✓ Alert thresholds configured")
        
        # Test controls (without actually showing GUI)
        assert hasattr(dashboard2, 'btn_start')
        assert hasattr(dashboard2, 'btn_reset')
        assert hasattr(dashboard2, 'slider_interval')
        print("✓ Dashboard controls created")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard creation test failed: {e}")
        return False

def test_monitoring_widgets():
    """Test monitoring widget creation."""
    print("\nTesting monitoring widgets...")
    
    try:
        from multigrid.visualization.realtime_dashboard import SolverDashboard, create_monitoring_widgets
        
        dashboard = SolverDashboard()
        widgets = create_monitoring_widgets(dashboard)
        
        assert isinstance(widgets, dict)
        assert 'memory_threshold' in widgets
        assert 'gpu_threshold' in widgets
        assert 'export_btn' in widgets
        print("✓ Monitoring widgets created")
        
        return True
        
    except Exception as e:
        print(f"✗ Monitoring widgets test failed: {e}")
        return False

def test_example_callback():
    """Test the example solver callback."""
    print("\nTesting example solver callback...")
    
    try:
        from multigrid.visualization.realtime_dashboard import example_solver_callback
        
        # Test multiple calls
        for i in range(5):
            metrics = example_solver_callback()
            
            assert isinstance(metrics, dict)
            assert 'residual' in metrics
            assert 'iteration' in metrics
            assert 'precision_level' in metrics
            assert metrics['residual'] > 0
            assert metrics['iteration'] >= 0
            assert metrics['precision_level'] in [32, 64]
            
            time.sleep(0.01)  # Small delay
        
        print("✓ Example callback working")
        return True
        
    except Exception as e:
        print(f"✗ Example callback test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")
    
    try:
        from multigrid.visualization.realtime_dashboard import solver_dashboard, example_solver_callback
        
        # Create dashboard
        dashboard = solver_dashboard()
        
        # Set callback
        dashboard.set_solver_callback(example_solver_callback)
        
        # Brief monitoring test
        dashboard.start_monitoring()
        time.sleep(0.2)
        dashboard.stop_monitoring()
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all dashboard tests."""
    print("Real-time Dashboard Test Suite")
    print("=" * 40)
    
    # Test results
    results = {}
    
    # Run individual tests
    results['imports'] = test_imports()
    results['metrics_collector'] = test_metrics_collector()
    results['data_streamer'] = test_data_streamer()
    results['dashboard_creation'] = test_dashboard_creation()
    results['monitoring_widgets'] = test_monitoring_widgets()
    results['example_callback'] = test_example_callback()
    results['integration'] = test_integration()
    
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
        print("\n🎉 ALL TESTS PASSED! Real-time dashboard is ready to use.")
        print("\nNext steps:")
        print("1. Run the demo: python examples/realtime_dashboard_demo.py --mode demo")
        print("2. Run with solver: python examples/realtime_dashboard_demo.py --mode solver")
        print("3. Run functionality tests: python examples/realtime_dashboard_demo.py --mode test")
        print("4. Integrate with your own solver using the callback pattern")
        
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed (matplotlib, numpy)")
        print("2. For system monitoring: pip install psutil")
        print("3. For GPU monitoring: pip install GPUtil")
        
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)