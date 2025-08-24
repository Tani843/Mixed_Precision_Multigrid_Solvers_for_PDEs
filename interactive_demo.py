#!/usr/bin/env python3
"""
Interactive Mixed-Precision Multigrid Demo

This script provides interactive demonstrations of the visualization framework
and mixed-precision analysis capabilities. Run this to explore different
solver configurations and see real-time results.

Usage:
    python interactive_demo.py [--demo-type TYPE]
    
Demo types:
    - parameter_explorer: Interactive parameter exploration
    - convergence_monitor: Real-time convergence monitoring  
    - precision_comparison: Mixed-precision trade-off analysis
    - method_dashboard: Comprehensive method comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from multigrid.visualization.interactive_plots import InteractivePlotter
    from multigrid.benchmarks.performance_benchmark import PerformanceBenchmark
    INTERACTIVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Interactive modules not available: {e}")
    INTERACTIVE_AVAILABLE = False


class InteractiveDemo:
    """Interactive demonstration of mixed-precision multigrid capabilities."""
    
    def __init__(self):
        """Initialize demo with sample data and configurations."""
        self.plotter = InteractivePlotter() if INTERACTIVE_AVAILABLE else None
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self):
        """Generate sample data for demonstrations."""
        return {
            'problem_sizes': [65, 129, 257, 513, 1025],
            'grid_sizes': [17, 33, 65, 129, 257],
            'precision_types': ['FP32', 'FP64', 'Mixed_Conservative', 'Mixed_Aggressive'],
            'solver_methods': ['CPU_double', 'GPU_double', 'GPU_mixed']
        }
    
    def run_parameter_explorer_demo(self):
        """Run interactive parameter exploration demo."""
        if not INTERACTIVE_AVAILABLE:
            print("Interactive features not available. Running static demo instead.")
            self._run_static_parameter_demo()
            return
        
        print("Starting Parameter Explorer Demo...")
        print("Use sliders to explore how different parameters affect solver performance.")
        
        # Define parameter ranges
        parameter_ranges = {
            'Grid Size': (32, 512, 128),
            'Precision Threshold': (1e-8, 1e-3, 1e-6),
            'Convergence Factor': (0.05, 0.5, 0.1),
            'Smoother Iterations': (1, 10, 3)
        }
        
        def update_function(ax, params):
            """Update plot based on parameter values."""
            ax.clear()
            
            # Simulate solver performance based on parameters
            grid_size = int(params['Grid Size'])
            threshold = params['Precision Threshold']
            conv_factor = params['Convergence Factor']
            smooth_iters = int(params['Smoother Iterations'])
            
            # Generate synthetic performance data
            iterations = range(20)
            residuals = [conv_factor**i for i in iterations]
            
            # Plot residual convergence
            ax.semilogy(iterations, residuals, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Switch Threshold: {threshold:.1e}')
            
            ax.set_xlabel('Multigrid Iteration')
            ax.set_ylabel('Relative Residual')
            ax.set_title(f'Convergence for {grid_size}×{grid_size} Grid')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add performance annotation
            n_points = grid_size ** 2
            est_time = n_points * 1e-6 * (1 + 0.1 * np.random.randn())
            ax.text(0.02, 0.98, f'Est. Solve Time: {est_time:.3f}s\nSmoother Iters: {smooth_iters}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        # Create interactive explorer
        fig, ax, sliders = self.plotter.create_parameter_explorer(
            parameter_ranges, update_function,
            title="Mixed-Precision Multigrid Parameter Explorer"
        )
        
        plt.show()
    
    def run_convergence_monitor_demo(self):
        """Run real-time convergence monitoring demo."""
        if not INTERACTIVE_AVAILABLE:
            print("Interactive features not available. Running static demo instead.")
            self._run_static_convergence_demo()
            return
        
        print("Starting Convergence Monitor Demo...")
        print("This simulates real-time monitoring of multigrid convergence.")
        
        def solver_simulation(params):
            """Simulate solver execution that yields residual values."""
            max_iterations = 15
            convergence_factor = 0.1
            
            for i in range(max_iterations):
                residual = convergence_factor ** i
                conv_rate = convergence_factor if i > 0 else 1.0
                iter_time = 0.05 + 0.01 * np.random.randn()  # Simulate timing variation
                
                yield {
                    'iteration': i,
                    'residual': residual,
                    'convergence_rate': conv_rate,
                    'iteration_time': max(iter_time, 0.01)
                }
        
        # Create convergence monitor
        fig, axes, anim = self.plotter.create_convergence_monitor(
            solver_simulation, {},
            title="Real-Time Convergence Monitor"
        )
        
        # Simulate solver progress
        import threading
        import time
        
        def simulate_solver():
            for result in solver_simulation({}):
                self.plotter.add_residual_data(
                    result['residual'],
                    result['convergence_rate'],
                    result['iteration_time']
                )
                time.sleep(0.5)  # Simulate computation time
        
        # Start simulation in background
        solver_thread = threading.Thread(target=simulate_solver)
        solver_thread.daemon = True
        solver_thread.start()
        
        plt.show()
    
    def run_precision_comparison_demo(self):
        """Run mixed-precision comparison demo."""
        if not INTERACTIVE_AVAILABLE:
            print("Interactive features not available. Running static demo instead.")
            self._run_static_precision_demo()
            return
        
        print("Starting Precision Comparison Demo...")
        print("Explore trade-offs between different precision strategies.")
        
        def precision_analysis_function(params):
            """Simulate precision analysis with given parameters."""
            threshold = params['threshold']
            factor = params['factor']
            strategy = params['strategy']
            problem_sizes = params['problem_sizes']
            
            # Simulate results for different precision types
            results = {}
            
            # Strategy-dependent parameters
            strategy_configs = {
                'Conservative': {'speed_factor': 0.7, 'error_factor': 1e-9},
                'Balanced': {'speed_factor': 0.6, 'error_factor': 1e-8},
                'Aggressive': {'speed_factor': 0.5, 'error_factor': 1e-7}
            }
            
            config = strategy_configs.get(strategy, strategy_configs['Balanced'])
            
            for precision in ['FP32', 'FP64', 'Mixed']:
                if precision == 'FP32':
                    solve_times = [n * 1e-6 * 0.5 for n in problem_sizes]
                    errors = [3e-6 for _ in problem_sizes]
                    memory = [n * 4e-6 for n in problem_sizes]
                elif precision == 'FP64':
                    solve_times = [n * 1e-6 for n in problem_sizes]
                    errors = [1e-12 for _ in problem_sizes]
                    memory = [n * 8e-6 for n in problem_sizes]
                else:  # Mixed
                    solve_times = [n * 1e-6 * config['speed_factor'] for n in problem_sizes]
                    errors = [config['error_factor'] for _ in problem_sizes]
                    memory = [n * 6e-6 for n in problem_sizes]
                
                results[precision] = {
                    'solve_times': solve_times,
                    'errors': errors,
                    'memory_usage': memory
                }
            
            return results
        
        # Create precision explorer
        fig, axes, widgets_dict = self.plotter.create_precision_explorer(
            precision_analysis_function,
            self.sample_data['problem_sizes'],
            title="Mixed-Precision Strategy Explorer"
        )
        
        plt.show()
    
    def run_method_dashboard_demo(self):
        """Run comprehensive method comparison dashboard."""
        if not INTERACTIVE_AVAILABLE:
            print("Interactive features not available. Running static demo instead.")
            self._run_static_dashboard_demo()
            return
        
        print("Starting Method Comparison Dashboard...")
        print("Compare different solver methods across multiple metrics.")
        
        # Generate comprehensive method data
        methods_data = {}
        
        for method in self.sample_data['solver_methods']:
            if method == 'CPU_double':
                solve_times = [0.012, 0.089, 0.721, 5.892, 47.234]
                iterations = [8, 8, 9, 9, 10]
                errors = [1.2e-10, 3.1e-11, 7.8e-12, 1.9e-12, 4.8e-13]
            elif method == 'GPU_double':
                solve_times = [0.008, 0.025, 0.156, 1.023, 7.156]
                iterations = [9, 9, 10, 10, 11]
                errors = [1.3e-10, 3.2e-11, 8.1e-12, 2.0e-12, 5.1e-13]
            else:  # GPU_mixed
                solve_times = [0.005, 0.015, 0.092, 0.603, 4.234]
                iterations = [10, 11, 11, 12, 13]
                errors = [2.1e-9, 5.3e-10, 1.3e-10, 3.2e-11, 8.1e-12]
            
            methods_data[method] = {
                'problem_sizes': self.sample_data['problem_sizes'],
                'solve_time': solve_times,
                'iterations': iterations,
                'error': errors
            }
        
        # Create comparison dashboard
        fig, axes, widgets_dict = self.plotter.create_comparison_dashboard(
            methods_data,
            metrics=['solve_time', 'iterations', 'error'],
            title="Solver Method Comparison Dashboard"
        )
        
        plt.show()
    
    def _run_static_parameter_demo(self):
        """Run static version of parameter exploration demo."""
        print("Running static parameter exploration demo...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Parameter variations
        grid_sizes = [64, 128, 256, 512]
        convergence_factors = [0.05, 0.1, 0.2, 0.3]
        
        # Plot 1: Grid size effect on convergence
        for i, size in enumerate(grid_sizes):
            iterations = range(15)
            residuals = [0.1**j for j in iterations]
            ax1.semilogy(iterations, residuals, 'o-', label=f'{size}×{size}', linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual')
        ax1.set_title('Effect of Grid Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence factor effect
        iterations = range(20)
        for i, factor in enumerate(convergence_factors):
            residuals = [factor**j for j in iterations]
            ax2.semilogy(iterations, residuals, 's-', label=f'ρ={factor}', linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Residual')
        ax2.set_title('Effect of Convergence Factor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance vs grid size
        solve_times = [s**2 * 1e-8 for s in grid_sizes]
        ax3.loglog(grid_sizes, solve_times, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Grid Size')
        ax3.set_ylabel('Solve Time (s)')
        ax3.set_title('Performance Scaling')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory usage
        memory_usage = [s**2 * 8e-6 for s in grid_sizes]  # MB
        ax4.loglog(grid_sizes, memory_usage, '^-', linewidth=2, markersize=8, color='green')
        ax4.set_xlabel('Grid Size')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Scaling')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _run_static_convergence_demo(self):
        """Run static version of convergence monitoring demo."""
        print("Running static convergence monitoring demo...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Simulate convergence histories for different methods
        iterations = range(15)
        
        methods = {
            'CPU Multigrid': [0.1**i for i in iterations],
            'GPU Multigrid': [0.095**i for i in iterations], 
            'Mixed Precision': [0.103**i for i in iterations]
        }
        
        colors = ['#2563eb', '#dc2626', '#059669']
        
        # Plot residual histories
        for i, (method, residuals) in enumerate(methods.items()):
            ax1.semilogy(iterations, residuals, 'o-', color=colors[i], 
                        linewidth=2, markersize=6, label=method)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Relative Residual')
        ax1.set_title('Convergence Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot convergence rates
        rates = [0.1, 0.095, 0.103]
        method_names = list(methods.keys())
        bars = ax2.bar(method_names, rates, color=colors, alpha=0.7)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target Rate')
        ax2.set_ylabel('Convergence Rate')
        ax2.set_title('Convergence Rate Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot iteration times
        iter_times = [0.05 + 0.01*np.random.randn() for _ in iterations]
        ax3.plot(iterations, iter_times, 'o-', linewidth=2, markersize=6, color='orange')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Time per Iteration (s)')
        ax3.set_title('Iteration Timing')
        ax3.grid(True, alpha=0.3)
        
        # Plot cumulative solve time
        cumulative_time = np.cumsum(iter_times)
        ax4.plot(iterations, cumulative_time, 's-', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cumulative Time (s)')
        ax4.set_title('Total Solve Time Progress')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _run_static_precision_demo(self):
        """Run static version of precision comparison demo."""
        print("Running static precision comparison demo...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        problem_sizes = self.sample_data['problem_sizes']
        colors = ['#2563eb', '#dc2626', '#059669', '#d97706']
        
        # Performance comparison
        precision_data = {
            'FP64': [n * 1e-6 for n in problem_sizes],
            'FP32': [n * 1e-6 * 0.48 for n in problem_sizes],
            'Mixed Conservative': [n * 1e-6 * 0.59 for n in problem_sizes],
            'Mixed Aggressive': [n * 1e-6 * 0.53 for n in problem_sizes]
        }
        
        for i, (precision, times) in enumerate(precision_data.items()):
            ax1.loglog(problem_sizes, times, 'o-', color=colors[i], 
                      linewidth=2, markersize=8, label=precision)
        
        ax1.set_xlabel('Problem Size (N)')
        ax1.set_ylabel('Solve Time (s)')
        ax1.set_title('Performance vs Precision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy comparison
        error_data = {
            'FP64': [1e-10 for _ in problem_sizes],
            'FP32': [3e-6 for _ in problem_sizes],
            'Mixed Conservative': [1e-9 for _ in problem_sizes], 
            'Mixed Aggressive': [1e-8 for _ in problem_sizes]
        }
        
        for i, (precision, errors) in enumerate(error_data.items()):
            ax2.semilogy(problem_sizes, errors, 'o-', color=colors[i],
                        linewidth=2, markersize=8, label=precision)
        
        ax2.set_xlabel('Problem Size (N)')
        ax2.set_ylabel('L² Error')
        ax2.set_title('Accuracy vs Precision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Memory usage comparison
        memory_data = {
            'FP64': [n * 8e-6 for n in problem_sizes],
            'FP32': [n * 4e-6 for n in problem_sizes], 
            'Mixed Conservative': [n * 5.2e-6 for n in problem_sizes],
            'Mixed Aggressive': [n * 4.4e-6 for n in problem_sizes]
        }
        
        for i, (precision, memory) in enumerate(memory_data.items()):
            ax3.loglog(problem_sizes, memory, 'o-', color=colors[i],
                      linewidth=2, markersize=8, label=precision)
        
        ax3.set_xlabel('Problem Size (N)')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory vs Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance-accuracy trade-off
        for i, precision in enumerate(precision_data.keys()):
            times = precision_data[precision]
            errors = error_data[precision]
            ax4.loglog(times, errors, 'o', color=colors[i], markersize=10, 
                      alpha=0.7, label=precision)
        
        ax4.set_xlabel('Solve Time (s)')
        ax4.set_ylabel('L² Error')
        ax4.set_title('Performance-Accuracy Trade-off')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add arrow indicating "better" direction
        ax4.annotate('Better', xy=(0.1, 0.9), xytext=(0.3, 0.7),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _run_static_dashboard_demo(self):
        """Run static version of method comparison dashboard."""
        print("Running static method comparison dashboard...")
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots with custom layout
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        
        problem_sizes = self.sample_data['problem_sizes']
        colors = ['#2563eb', '#dc2626', '#059669']
        
        # Method data
        methods = {
            'CPU Double': {
                'solve_times': [0.012, 0.089, 0.721, 5.892, 47.234],
                'iterations': [8, 8, 9, 9, 10],
                'errors': [1.2e-10, 3.1e-11, 7.8e-12, 1.9e-12, 4.8e-13]
            },
            'GPU Double': {
                'solve_times': [0.008, 0.025, 0.156, 1.023, 7.156],
                'iterations': [9, 9, 10, 10, 11],
                'errors': [1.3e-10, 3.2e-11, 8.1e-12, 2.0e-12, 5.1e-13]
            },
            'GPU Mixed': {
                'solve_times': [0.005, 0.015, 0.092, 0.603, 4.234],
                'iterations': [10, 11, 11, 12, 13],
                'errors': [2.1e-9, 5.3e-10, 1.3e-10, 3.2e-11, 8.1e-12]
            }
        }
        
        # Plot 1: Solve time comparison
        for i, (method, data) in enumerate(methods.items()):
            ax1.loglog(problem_sizes, data['solve_times'], 'o-', 
                      color=colors[i], linewidth=2, markersize=8, label=method)
        ax1.set_xlabel('Problem Size')
        ax1.set_ylabel('Solve Time (s)')
        ax1.set_title('Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup analysis
        baseline_times = methods['CPU Double']['solve_times']
        for i, (method, data) in enumerate(list(methods.items())[1:], 1):
            speedups = [bt/st for bt, st in zip(baseline_times, data['solve_times'])]
            ax2.semilogx(problem_sizes, speedups, 'o-', 
                        color=colors[i], linewidth=2, markersize=8, 
                        label=f'{method} vs CPU Double')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Iteration comparison
        for i, (method, data) in enumerate(methods.items()):
            ax3.semilogx(problem_sizes, data['iterations'], 'o-',
                        color=colors[i], linewidth=2, markersize=8, label=method)
        ax3.set_xlabel('Problem Size')
        ax3.set_ylabel('Iterations to Convergence')
        ax3.set_title('Iteration Count Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error comparison
        for i, (method, data) in enumerate(methods.items()):
            ax4.loglog(problem_sizes, data['errors'], 'o-',
                      color=colors[i], linewidth=2, markersize=8, label=method)
        ax4.set_xlabel('Problem Size')
        ax4.set_ylabel('L² Error')
        ax4.set_title('Accuracy Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Efficiency analysis
        for i, (method, data) in enumerate(methods.items()):
            efficiency = [n/t for n, t in zip(problem_sizes, data['solve_times'])]
            ax5.loglog(problem_sizes, efficiency, 'o-',
                      color=colors[i], linewidth=2, markersize=8, label=method)
        ax5.set_xlabel('Problem Size')
        ax5.set_ylabel('Efficiency (N/time)')
        ax5.set_title('Computational Efficiency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance-error trade-off
        for i, (method, data) in enumerate(methods.items()):
            ax6.loglog(data['solve_times'], data['errors'], 'o',
                      color=colors[i], markersize=10, alpha=0.7, label=method)
        ax6.set_xlabel('Solve Time (s)')
        ax6.set_ylabel('L² Error')
        ax6.set_title('Performance-Accuracy Trade-off')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_all_demos(self):
        """Run all available demos in sequence."""
        print("Running all interactive demos...")
        print("Close each window to proceed to the next demo.")
        
        demos = [
            ("Parameter Explorer", self.run_parameter_explorer_demo),
            ("Convergence Monitor", self.run_convergence_monitor_demo),
            ("Precision Comparison", self.run_precision_comparison_demo),
            ("Method Dashboard", self.run_method_dashboard_demo)
        ]
        
        for demo_name, demo_func in demos:
            print(f"\n{'='*20}")
            print(f"Starting: {demo_name}")
            print(f"{'='*20}")
            
            try:
                demo_func()
            except Exception as e:
                print(f"Error in {demo_name}: {e}")
                print("Continuing to next demo...")
        
        print("\nAll demos completed!")


def main():
    """Main function to run interactive demonstrations."""
    parser = argparse.ArgumentParser(description='Interactive Mixed-Precision Multigrid Demo')
    parser.add_argument('--demo-type', choices=[
        'parameter_explorer', 'convergence_monitor', 'precision_comparison',
        'method_dashboard', 'all'
    ], default='all', help='Type of demo to run')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = InteractiveDemo()
    
    print("Mixed-Precision Multigrid Interactive Demo")
    print("="*45)
    print("This demo showcases the visualization and analysis capabilities")
    print("of the mixed-precision multigrid solver framework.")
    print()
    
    # Run requested demo
    if args.demo_type == 'parameter_explorer':
        demo.run_parameter_explorer_demo()
    elif args.demo_type == 'convergence_monitor':
        demo.run_convergence_monitor_demo()
    elif args.demo_type == 'precision_comparison':
        demo.run_precision_comparison_demo()
    elif args.demo_type == 'method_dashboard':
        demo.run_method_dashboard_demo()
    else:  # all
        demo.run_all_demos()
    
    print("\nDemo completed! Thank you for exploring mixed-precision multigrid methods.")


if __name__ == '__main__':
    main()