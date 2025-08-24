#!/usr/bin/env python3
"""
Real-time Dashboard Demonstration

This script demonstrates the real-time solver monitoring dashboard with:
- Live convergence plotting
- Memory usage tracking
- GPU utilization monitoring  
- Precision switching visualization

Usage:
    python examples/realtime_dashboard_demo.py [--mode MODE] [--duration SECONDS]
    
Modes:
    demo    - Run with simulated solver data (default)
    solver  - Run with actual solver integration (requires solver callback)
    test    - Run basic functionality tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from multigrid.visualization.realtime_dashboard import (
    solver_dashboard, 
    example_solver_callback,
    create_monitoring_widgets
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimulatedSolver:
    """
    Simulated multigrid solver for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize simulated solver."""
        self.iteration = 0
        self.residual = 1.0
        self.precision_level = 32
        self.start_time = time.time()
        self.convergence_rate = 0.85
        self.precision_switches = [100, 200, 350, 500]  # Switch at these iterations
        
    def step(self):
        """Simulate one solver iteration."""
        self.iteration += 1
        
        # Simulate convergence with some noise
        base_reduction = self.convergence_rate ** self.iteration
        noise = 1.0 + 0.1 * np.random.randn()
        self.residual = base_reduction * noise
        
        # Simulate precision switching
        if self.iteration in self.precision_switches:
            self.precision_level = 64 if self.precision_level == 32 else 32
            logger.info(f"Precision switched to FP{self.precision_level} at iteration {self.iteration}")
        
        # Add occasional stalls to test alert system
        if self.iteration % 150 == 0:
            logger.info(f"Simulating convergence stall at iteration {self.iteration}")
            for _ in range(10):
                time.sleep(0.1)  # Simulate computational work
        
        return {
            'residual': max(self.residual, 1e-15),
            'iteration': self.iteration,
            'precision_level': self.precision_level,
            'solve_time': time.time() - self.start_time
        }
    
    def get_current_metrics(self):
        """Get current solver metrics."""
        return {
            'residual': max(self.residual, 1e-15),
            'iteration': self.iteration,
            'precision_level': self.precision_level
        }
    
    def is_converged(self, tolerance=1e-10):
        """Check if solver has converged."""
        return self.residual < tolerance

class IntegratedSolverDemo:
    """
    Demonstration of dashboard integration with actual solver.
    """
    
    def __init__(self):
        """Initialize integrated demo."""
        self.solver = SimulatedSolver()
        self.dashboard = None
        
    def setup_dashboard(self):
        """Setup the monitoring dashboard."""
        logger.info("Setting up real-time monitoring dashboard...")
        
        # Create dashboard
        self.dashboard = solver_dashboard()
        
        # Set solver callback
        self.dashboard.set_solver_callback(self.solver.get_current_metrics)
        
        # Create additional monitoring widgets
        widgets = create_monitoring_widgets(self.dashboard)
        
        logger.info("Dashboard setup complete")
        return self.dashboard, widgets
    
    def run_solver_with_monitoring(self, max_iterations=1000):
        """
        Run solver with real-time monitoring.
        
        Args:
            max_iterations: Maximum number of solver iterations
        """
        logger.info(f"Starting solver with real-time monitoring (max {max_iterations} iterations)")
        
        # Setup dashboard
        dashboard, widgets = self.setup_dashboard()
        
        # Start monitoring
        dashboard.start_monitoring()
        
        try:
            # Run solver iterations
            for i in range(max_iterations):
                # Perform solver step
                metrics = self.solver.step()
                
                # Log progress occasionally
                if i % 50 == 0:
                    logger.info(f"Iteration {metrics['iteration']}: residual = {metrics['residual']:.2e}")
                
                # Check convergence
                if self.solver.is_converged():
                    logger.info(f"Converged after {metrics['iteration']} iterations!")
                    break
                
                # Small delay to make monitoring visible
                time.sleep(0.01)
            
            # Keep dashboard open for viewing
            logger.info("Solver completed. Dashboard will remain open for monitoring...")
            logger.info("Close the dashboard window to exit.")
            
            # Show dashboard
            dashboard.show()
            
        except KeyboardInterrupt:
            logger.info("Solver interrupted by user")
        finally:
            dashboard.stop_monitoring()

def demo_mode():
    """Run dashboard in demo mode with simulated data."""
    logger.info("Starting real-time dashboard in demo mode")
    
    # Create dashboard
    dashboard = solver_dashboard()
    
    # Set up simulated solver callback
    dashboard.set_solver_callback(example_solver_callback)
    
    # Create additional widgets
    widgets = create_monitoring_widgets(dashboard)
    
    # Start monitoring
    dashboard.start_monitoring()
    
    logger.info("Demo dashboard is running with simulated data")
    logger.info("Features demonstrated:")
    logger.info("- Live convergence plotting")
    logger.info("- Memory usage tracking")
    logger.info("- GPU utilization monitoring")
    logger.info("- Precision switching visualization")
    logger.info("- Real-time alerts and system metrics")
    logger.info("")
    logger.info("Use dashboard controls:")
    logger.info("- Start/Stop: Toggle monitoring")
    logger.info("- Reset: Clear all data")
    logger.info("- Update slider: Adjust refresh rate")
    logger.info("- Window slider: Adjust data window size")
    logger.info("- Export: Save current data to JSON")
    logger.info("")
    logger.info("Close the dashboard window to exit.")
    
    # Show dashboard
    dashboard.show()

def solver_mode():
    """Run dashboard with integrated solver."""
    logger.info("Starting real-time dashboard with integrated solver")
    
    demo = IntegratedSolverDemo()
    demo.run_solver_with_monitoring(max_iterations=500)

def test_mode():
    """Run basic functionality tests."""
    logger.info("Running dashboard functionality tests...")
    
    tests_passed = 0
    total_tests = 0
    
    def test_result(name, passed):
        nonlocal tests_passed, total_tests
        total_tests += 1
        if passed:
            tests_passed += 1
            logger.info(f"✓ {name}")
        else:
            logger.error(f"✗ {name}")
    
    try:
        # Test 1: Dashboard creation
        total_tests += 1
        try:
            dashboard = solver_dashboard()
            test_result("Dashboard creation", True)
        except Exception as e:
            test_result(f"Dashboard creation - {e}", False)
            return
        
        # Test 2: Widget creation
        total_tests += 1
        try:
            widgets = create_monitoring_widgets(dashboard)
            test_result("Widget creation", len(widgets) > 0)
        except Exception as e:
            test_result(f"Widget creation - {e}", False)
        
        # Test 3: Callback setup
        total_tests += 1
        try:
            dashboard.set_solver_callback(example_solver_callback)
            test_result("Callback setup", True)
        except Exception as e:
            test_result(f"Callback setup - {e}", False)
        
        # Test 4: Start/stop monitoring
        total_tests += 1
        try:
            dashboard.start_monitoring()
            time.sleep(0.5)  # Let it run briefly
            dashboard.stop_monitoring()
            test_result("Start/stop monitoring", True)
        except Exception as e:
            test_result(f"Start/stop monitoring - {e}", False)
        
        # Test 5: Data collection
        total_tests += 1
        try:
            dashboard.start_monitoring()
            time.sleep(1.0)  # Collect some data
            data = dashboard.data_streamer.metrics_collector.get_recent_data(10)
            dashboard.stop_monitoring()
            test_result("Data collection", len(data.get('timestamps', [])) > 0)
        except Exception as e:
            test_result(f"Data collection - {e}", False)
        
        logger.info(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            logger.info("🎉 All tests passed! Dashboard is ready for use.")
        else:
            logger.warning(f"⚠ {total_tests - tests_passed} tests failed.")
        
    except Exception as e:
        logger.error(f"Test suite error: {e}")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Real-time Dashboard Demonstration")
    parser.add_argument('--mode', choices=['demo', 'solver', 'test'], default='demo',
                      help='Demonstration mode (default: demo)')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration for demo mode in seconds (default: 60)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("Real-time Solver Monitoring Dashboard")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print("=" * 50)
    
    try:
        if args.mode == 'demo':
            demo_mode()
        elif args.mode == 'solver':
            solver_mode()
        elif args.mode == 'test':
            test_mode()
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Dashboard demo interrupted by user")
    except Exception as e:
        logger.error(f"Dashboard demo error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()