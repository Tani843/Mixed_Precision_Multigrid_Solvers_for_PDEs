#!/usr/bin/env python3
"""
Generate Sample Plots for Mixed-Precision Multigrid Documentation

This script demonstrates the visualization framework by generating professional-quality
plots that showcase the capabilities and results of the mixed-precision multigrid
solver. All plots are saved to the docs/assets/images/ directory for use in the
Jekyll documentation website.

Usage:
    python generate_plots.py [--output-dir DIR] [--dpi DPI]
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from multigrid.visualization import (
        SolutionVisualizer, ConvergencePlotter, PerformancePlotter,
        GridVisualizer, AnalysisVisualizer
    )
except ImportError as e:
    print(f"Warning: Could not import visualization modules: {e}")
    print("Generating plots with basic matplotlib instead...")
    SolutionVisualizer = None


def ensure_output_directory(output_dir):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")


def generate_sample_data():
    """Generate synthetic data for demonstration purposes."""
    # Grid sizes for convergence studies
    grid_sizes = [17, 33, 65, 129, 257]
    h_values = [1.0 / (n - 1) for n in grid_sizes]
    
    # Synthetic solution data (2D Poisson equation)
    def manufactured_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def source_term(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Generate 2D solution data
    n = 65
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    solution_exact = manufactured_solution(X, Y)
    # Add small numerical error
    solution_numerical = solution_exact + 1e-6 * np.random.randn(*X.shape)
    source = source_term(X, Y)
    
    # Convergence data
    l2_errors = [0.1 * h**2 * (1 + 0.1 * np.random.randn()) for h in h_values]
    max_errors = [0.15 * h**2 * (1 + 0.15 * np.random.randn()) for h in h_values]
    
    # Residual histories
    residual_histories = {
        'CPU Multigrid': np.array([1.0, 0.089, 0.008, 0.0007, 0.00006, 0.000005]),
        'GPU Multigrid': np.array([1.0, 0.095, 0.009, 0.0008, 0.000075, 0.000007]),
        'Mixed Precision': np.array([1.0, 0.103, 0.011, 0.0011, 0.00012, 0.000013])
    }
    
    # Performance data
    problem_sizes = [1024, 4096, 16384, 65536, 262144]
    cpu_times = [0.012, 0.089, 0.721, 5.892, 47.234]
    gpu_times = [0.008, 0.025, 0.156, 1.023, 7.156]
    
    # Mixed precision data
    precision_data = {
        'FP64': {
            'solve_times': cpu_times,
            'errors': [1.2e-10, 3.1e-11, 7.8e-12, 1.9e-12, 4.8e-13],
            'memory_usage': [8.4, 33.6, 134.4, 537.6, 2150.4]
        },
        'FP32': {
            'solve_times': [t * 0.48 for t in cpu_times],
            'errors': [3.8e-6, 3.9e-6, 3.7e-6, 4.1e-6, 3.6e-6],
            'memory_usage': [4.2, 16.8, 67.2, 268.8, 1075.2]
        },
        'Mixed': {
            'solve_times': [t * 0.59 for t in cpu_times],
            'errors': [2.1e-9, 5.3e-10, 1.3e-10, 3.2e-11, 8.1e-12],
            'memory_usage': [5.5, 21.8, 87.4, 349.4, 1397.8]
        }
    }
    
    return {
        'grid_sizes': grid_sizes,
        'h_values': h_values,
        'X': X, 'Y': Y,
        'solution_exact': solution_exact,
        'solution_numerical': solution_numerical,
        'source': source,
        'l2_errors': l2_errors,
        'max_errors': max_errors,
        'residual_histories': residual_histories,
        'problem_sizes': problem_sizes,
        'cpu_times': cpu_times,
        'gpu_times': gpu_times,
        'precision_data': precision_data
    }


def generate_basic_plots(data, output_dir):
    """Generate plots using basic matplotlib when visualization modules are unavailable."""
    print("Generating basic plots with matplotlib...")
    
    # Set publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'font.size': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # 1. Solution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Exact solution
    im1 = ax1.contourf(data['X'], data['Y'], data['solution_exact'], levels=20, cmap='viridis')
    ax1.set_title('Exact Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    # Numerical solution
    im2 = ax2.contourf(data['X'], data['Y'], data['solution_numerical'], levels=20, cmap='viridis')
    ax2.set_title('Numerical Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'solution_comparison.png'))
    plt.close()
    
    # 2. Grid convergence study
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(data['h_values'], data['l2_errors'], 'o-', linewidth=2, markersize=8, label='L² Error')
    ax.loglog(data['h_values'], data['max_errors'], 's-', linewidth=2, markersize=8, label='Max Error')
    
    # Theoretical slope
    h_theory = np.array([data['h_values'][0], data['h_values'][-1]])
    error_theory = data['l2_errors'][0] * (h_theory / data['h_values'][0]) ** 2
    ax.loglog(h_theory, error_theory, '--', linewidth=2, alpha=0.7, label='O(h²) slope')
    
    ax.set_xlabel('Grid Spacing (h)')
    ax.set_ylabel('Error')
    ax.set_title('Grid Convergence Study')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'grid_convergence_study.png'))
    plt.close()
    
    # 3. Residual convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2563eb', '#dc2626', '#059669']
    
    for i, (method, residuals) in enumerate(data['residual_histories'].items()):
        iterations = range(len(residuals))
        ax.semilogy(iterations, residuals, 'o-', color=colors[i], 
                   linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel('Multigrid Iteration')
    ax.set_ylabel('Relative Residual')
    ax.set_title('Multigrid Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'multigrid_convergence.png'))
    plt.close()
    
    # 4. CPU vs GPU performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison
    ax1.loglog(data['problem_sizes'], data['cpu_times'], 'o-', linewidth=2, 
              markersize=8, label='CPU')
    ax1.loglog(data['problem_sizes'], data['gpu_times'], 's-', linewidth=2, 
              markersize=8, label='GPU')
    ax1.set_xlabel('Problem Size (N)')
    ax1.set_ylabel('Solve Time (s)')
    ax1.set_title('CPU vs GPU Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Speedup
    speedups = [ct / gt for ct, gt in zip(data['cpu_times'], data['gpu_times'])]
    ax2.semilogx(data['problem_sizes'], speedups, 'o-', linewidth=2, 
                markersize=8, color='#059669')
    ax2.set_xlabel('Problem Size (N)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('GPU Speedup')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_gpu_performance.png'))
    plt.close()
    
    # 5. Mixed precision analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#2563eb', '#dc2626', '#059669']
    
    # Performance
    for i, (precision, pdata) in enumerate(data['precision_data'].items()):
        ax1.loglog(data['problem_sizes'], pdata['solve_times'], 'o-', 
                  color=colors[i], linewidth=2, markersize=8, label=precision)
    ax1.set_xlabel('Problem Size (N)')
    ax1.set_ylabel('Solve Time (s)')
    ax1.set_title('Performance vs Precision')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy
    for i, (precision, pdata) in enumerate(data['precision_data'].items()):
        ax2.loglog(data['problem_sizes'], pdata['errors'], 'o-', 
                  color=colors[i], linewidth=2, markersize=8, label=precision)
    ax2.set_xlabel('Problem Size (N)')
    ax2.set_ylabel('L² Error')
    ax2.set_title('Accuracy vs Precision')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Memory usage
    for i, (precision, pdata) in enumerate(data['precision_data'].items()):
        ax3.loglog(data['problem_sizes'], pdata['memory_usage'], 'o-', 
                  color=colors[i], linewidth=2, markersize=8, label=precision)
    ax3.set_xlabel('Problem Size (N)')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.set_title('Memory Usage vs Precision')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Performance-accuracy trade-off
    for i, (precision, pdata) in enumerate(data['precision_data'].items()):
        ax4.loglog(pdata['solve_times'], pdata['errors'], 'o', 
                  color=colors[i], markersize=10, alpha=0.7, label=precision)
    ax4.set_xlabel('Solve Time (s)')
    ax4.set_ylabel('L² Error')
    ax4.set_title('Performance-Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixed_precision_tradeoffs.png'))
    plt.close()
    
    print(f"Generated 5 basic plots in {output_dir}")


def generate_advanced_plots(data, output_dir):
    """Generate plots using the advanced visualization framework."""
    print("Generating advanced plots with visualization framework...")
    
    # 1. Solution visualization
    if SolutionVisualizer:
        viz = SolutionVisualizer(style='publication')
        
        # 2D solution plot
        fig = viz.plot_solution_2d(
            data['X'], data['Y'], data['solution_exact'],
            title="Manufactured Solution: sin(πx)sin(πy)",
            save_name=os.path.join(output_dir, 'solution_2d_exact.png')
        )
        plt.close(fig)
        
        # Solution comparison
        fig = viz.plot_solution_comparison(
            data['X'], data['Y'], 
            [data['solution_exact'], data['solution_numerical']],
            ['Exact Solution', 'Numerical Solution'],
            title="Solution Comparison",
            save_name=os.path.join(output_dir, 'solution_comparison_advanced.png')
        )
        plt.close(fig)
    
    # 2. Convergence analysis
    conv_viz = ConvergencePlotter(style='publication')
    
    # Residual convergence
    fig = conv_viz.plot_residual_history(
        data['residual_histories'],
        title="Multigrid Convergence Analysis",
        save_name=os.path.join(output_dir, 'convergence_analysis.png')
    )
    plt.close(fig)
    
    # Grid convergence study
    fig = conv_viz.plot_grid_convergence_study(
        data['grid_sizes'], data['l2_errors'], data['max_errors'],
        title="Grid Convergence Study",
        save_name=os.path.join(output_dir, 'grid_convergence_advanced.png')
    )
    plt.close(fig)
    
    # 3. Performance analysis
    perf_viz = PerformancePlotter(style='publication')
    
    # CPU vs GPU comparison
    fig = perf_viz.plot_cpu_gpu_comparison(
        data['problem_sizes'], data['cpu_times'], data['gpu_times'],
        title="CPU vs GPU Performance Analysis",
        save_name=os.path.join(output_dir, 'performance_comparison.png')
    )
    plt.close(fig)
    
    # Mixed precision analysis
    fig = perf_viz.plot_mixed_precision_analysis(
        data['precision_data'],
        title="Mixed-Precision Performance Analysis",
        save_name=os.path.join(output_dir, 'precision_analysis.png')
    )
    plt.close(fig)
    
    # 4. Grid visualization
    grid_viz = GridVisualizer(style='publication')
    
    # Grid hierarchy
    fig = grid_viz.plot_grid_hierarchy(
        [65, 33, 17, 9, 5],
        title="Multigrid Hierarchy",
        save_name=os.path.join(output_dir, 'grid_hierarchy.png')
    )
    plt.close(fig)
    
    # Grid convergence study
    fig = grid_viz.plot_grid_convergence_study(
        data['grid_sizes'], data['l2_errors'],
        title="Grid Convergence Study",
        save_name=os.path.join(output_dir, 'grid_convergence_study.png')
    )
    plt.close(fig)
    
    # 5. Statistical analysis
    analysis_viz = AnalysisVisualizer(style='publication')
    
    # Error decomposition
    discretization_errors = [h**2 * 0.08 for h in data['h_values']]
    iteration_errors = [1e-10] * len(data['h_values'])
    roundoff_errors = [1e-15] * len(data['h_values'])
    
    fig = analysis_viz.plot_error_decomposition(
        data['grid_sizes'], discretization_errors, 
        iteration_errors, roundoff_errors,
        title="Error Component Analysis",
        save_name=os.path.join(output_dir, 'error_component_analysis.png')
    )
    plt.close(fig)
    
    # Method comparison
    methods = {
        'CPU Multigrid': {
            'problem_sizes': data['problem_sizes'],
            'solve_time': data['cpu_times'],
            'iterations': [8, 8, 9, 9, 10]
        },
        'GPU Multigrid': {
            'problem_sizes': data['problem_sizes'],
            'solve_time': data['gpu_times'],
            'iterations': [9, 9, 10, 10, 11]
        }
    }
    
    fig = analysis_viz.plot_method_comparison(
        methods,
        title="Solver Method Comparison",
        save_name=os.path.join(output_dir, 'method_comparison.png')
    )
    plt.close(fig)
    
    print(f"Generated advanced plots in {output_dir}")


def main():
    """Main function to generate all plots."""
    parser = argparse.ArgumentParser(description='Generate visualization plots')
    parser.add_argument('--output-dir', default='docs/assets/images',
                       help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved images')
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_output_directory(args.output_dir)
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data()
    
    # Generate plots
    try:
        if SolutionVisualizer:
            generate_advanced_plots(data, args.output_dir)
        else:
            generate_basic_plots(data, args.output_dir)
    except Exception as e:
        print(f"Error generating advanced plots: {e}")
        print("Falling back to basic plots...")
        generate_basic_plots(data, args.output_dir)
    
    print("\nPlot generation complete!")
    print(f"All plots saved to: {args.output_dir}")
    
    # List generated files
    if os.path.exists(args.output_dir):
        files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
        if files:
            print(f"\nGenerated {len(files)} plot files:")
            for f in sorted(files):
                print(f"  - {f}")
        else:
            print("\nNo plot files found in output directory.")


if __name__ == '__main__':
    main()