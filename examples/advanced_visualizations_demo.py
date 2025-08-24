#!/usr/bin/env python3
"""
Advanced Visualizations Demonstration

This script demonstrates the newly implemented advanced visualization capabilities:
1. Interactive 3D solution visualization
2. Multigrid cycle animation (showing grid transfers)
3. Convergence history comparison plots
4. GPU memory usage visualization
5. Precision error propagation analysis
6. Performance scaling plots with error bars

Usage:
    python examples/advanced_visualizations_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from multigrid.visualization.advanced_visualizations import create_missing_visualizations

def generate_sample_3d_solution_data():
    """Generate sample 3D solution data for demonstration."""
    nx, ny, nz = 64, 64, 32
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)  
    z = np.linspace(0, 0.5, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Generate sample solutions for different methods
    solution_data = {}
    
    # Method 1: Analytical solution
    solution_data['Analytical'] = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-Z)
    
    # Method 2: FP32 Multigrid solution (with small errors)
    noise = 0.01 * np.random.randn(*X.shape)
    solution_data['FP32_Multigrid'] = solution_data['Analytical'] + noise
    
    # Method 3: FP64 Multigrid solution (with smaller errors)  
    noise_fp64 = 0.001 * np.random.randn(*X.shape)
    solution_data['FP64_Multigrid'] = solution_data['Analytical'] + noise_fp64
    
    # Method 4: Mixed Precision solution
    noise_mixed = 0.005 * np.random.randn(*X.shape)
    solution_data['Mixed_Precision'] = solution_data['Analytical'] + noise_mixed
    
    grid_coords = {'x': x, 'y': y, 'z': z}
    
    return solution_data, grid_coords

def generate_sample_convergence_data():
    """Generate sample convergence data for demonstration."""
    methods = ['Jacobi', 'Gauss-Seidel', 'V-Cycle', 'W-Cycle', 'F-Cycle', 'Mixed-Precision']
    convergence_data = {}
    
    for i, method in enumerate(methods):
        n_iterations = np.random.randint(20, 100)
        
        # Generate realistic convergence behavior
        if 'Cycle' in method:
            # Multigrid methods converge faster
            base_rate = 0.1 + 0.05 * i
            residuals = [1.0]
            for j in range(n_iterations):
                factor = base_rate * (1 + 0.1 * np.random.randn())
                residuals.append(residuals[-1] * factor)
        else:
            # Classical methods converge slower
            base_rate = 0.8 + 0.1 * i
            residuals = [1.0]
            for j in range(n_iterations):
                factor = base_rate * (1 + 0.05 * np.random.randn())
                residuals.append(residuals[-1] * factor)
        
        # Generate corresponding errors and solve times
        errors = [r * (1 + 0.1 * np.random.randn()) for r in residuals]
        solve_times = [0.01 + 0.005 * np.random.randn() for _ in residuals]
        
        # Add confidence intervals
        residual_std = [r * 0.1 for r in residuals]
        error_std = [e * 0.1 for e in errors]
        
        convergence_data[method] = {
            'residual': residuals,
            'error': errors,
            'solve_time': solve_times,
            'residual_std': residual_std,
            'error_std': error_std
        }
    
    return convergence_data

def generate_sample_memory_data():
    """Generate sample GPU memory usage data."""
    gpu_ids = ['0', '1', '2', '3']
    memory_data = {}
    
    n_timesteps = 200
    
    for gpu_id in gpu_ids:
        # Simulate different memory usage patterns
        base_usage = 1000 + 500 * int(gpu_id)  # MB
        max_memory = 8192  # 8GB per GPU
        
        allocated = []
        cached = []
        free = []
        total = []
        
        for t in range(n_timesteps):
            # Simulate memory allocation patterns
            usage_factor = 0.5 + 0.3 * np.sin(t * 0.1) + 0.1 * np.random.randn()
            usage_factor = np.clip(usage_factor, 0.2, 0.9)
            
            alloc = base_usage * usage_factor
            cache = alloc * 0.2
            free_mem = max_memory - alloc - cache
            total_mem = max_memory
            
            allocated.append(alloc)
            cached.append(cache)
            free.append(free_mem)
            total.append(total_mem)
        
        memory_data[gpu_id] = {
            'allocated': allocated,
            'cached': cached,
            'free': free,
            'total': total,
            'max_memory': [max_memory] * n_timesteps,
            'performance_impact': [max(0, (alloc - base_usage) / base_usage * 100) 
                                 for alloc in allocated]
        }
    
    return memory_data

def generate_sample_error_propagation_data():
    """Generate sample precision error propagation data."""
    precision_levels = ['fp16', 'fp32', 'fp64']
    n_operations = 1000
    n_variables = 100
    
    error_data = {}
    
    for precision in precision_levels:
        # Simulate different error characteristics for each precision
        if precision == 'fp16':
            base_error = 1e-3
            growth_rate = 1.02
        elif precision == 'fp32':
            base_error = 1e-6
            growth_rate = 1.001
        else:  # fp64
            base_error = 1e-14
            growth_rate = 1.0001
        
        # Generate error matrices
        error_matrix = np.zeros((n_operations, n_variables))
        error_evolution = []
        
        for op in range(n_operations):
            for var in range(n_variables):
                if op == 0:
                    error_matrix[op, var] = base_error * (1 + 0.1 * np.random.randn())
                else:
                    error_matrix[op, var] = error_matrix[op-1, var] * growth_rate * (1 + 0.01 * np.random.randn())
            
            error_evolution.append(np.mean(error_matrix[op, :]))
        
        error_data[precision] = {
            'error_matrix': error_matrix,
            'error_evolution': np.array(error_evolution),
            'max_errors': np.max(error_matrix, axis=1),
            'min_errors': np.min(error_matrix, axis=1),
            'std_errors': np.std(error_matrix, axis=1)
        }
    
    return error_data

def generate_sample_scaling_data():
    """Generate sample performance scaling data."""
    methods = ['Serial', 'OpenMP', 'MPI', 'CUDA', 'Mixed-Precision-GPU']
    n_cores = [1, 2, 4, 8, 16, 32, 64]
    problem_sizes = [128, 256, 512, 1024, 2048]
    
    scaling_data = {}
    confidence_intervals = {}
    
    for method in methods:
        scaling_data[method] = {}
        confidence_intervals[method] = {}
        
        for metric in ['solve_time', 'efficiency', 'speedup', 'memory_usage']:
            scaling_data[method][metric] = []
            confidence_intervals[method][metric] = []
            
            for n in n_cores:
                if metric == 'solve_time':
                    if method == 'Serial':
                        base_time = 100.0
                        time = base_time
                    elif method == 'OpenMP':
                        base_time = 100.0
                        efficiency = min(0.9, 0.95 - 0.02 * np.log2(n))
                        time = base_time / (n * efficiency)
                    elif method == 'MPI':
                        base_time = 100.0
                        efficiency = min(0.85, 0.9 - 0.03 * np.log2(n))
                        time = base_time / (n * efficiency)
                    elif method == 'CUDA':
                        base_time = 100.0
                        time = base_time / (n * 0.7)  # GPU scaling
                    else:  # Mixed-Precision-GPU
                        base_time = 80.0  # Faster base time
                        time = base_time / (n * 0.8)
                    
                    scaling_data[method][metric].append(time)
                    confidence_intervals[method][metric].append(time * 0.1)  # 10% CI
                
                elif metric == 'efficiency':
                    if method == 'Serial':
                        eff = 1.0
                    else:
                        base_time = scaling_data[method]['solve_time'][0] if scaling_data[method]['solve_time'] else 100.0
                        current_time = scaling_data[method]['solve_time'][-1]
                        eff = base_time / (current_time * n)
                    
                    scaling_data[method][metric].append(eff)
                    confidence_intervals[method][metric].append(eff * 0.05)
                
                elif metric == 'speedup':
                    if method == 'Serial':
                        speedup = 1.0
                    else:
                        base_time = 100.0  # Serial baseline
                        current_time = scaling_data[method]['solve_time'][-1]
                        speedup = base_time / current_time
                    
                    scaling_data[method][metric].append(speedup)
                    confidence_intervals[method][metric].append(speedup * 0.1)
                
                elif metric == 'memory_usage':
                    base_memory = 1024  # MB
                    if 'GPU' in method:
                        memory = base_memory * 0.5  # GPU memory efficiency
                    else:
                        memory = base_memory * n / 8  # Memory per core
                    
                    scaling_data[method][metric].append(memory)
                    confidence_intervals[method][metric].append(memory * 0.05)
    
    return scaling_data, confidence_intervals

def main():
    """Main demonstration function."""
    print("Advanced Visualizations Demonstration")
    print("=====================================")
    
    # Create advanced visualization tools
    viz_tools = create_missing_visualizations()
    print("✓ Advanced visualization tools initialized")
    
    # 1. Interactive 3D Solution Visualization
    print("\n1. Generating 3D solution visualization...")
    solution_data, grid_coords = generate_sample_3d_solution_data()
    
    fig_3d, axes_3d, widgets_3d = viz_tools.create_interactive_3d_solution_visualization(
        solution_data, grid_coords, 
        title="Interactive 3D Solution Comparison"
    )
    print("✓ Interactive 3D visualization created")
    
    # 2. Convergence History Comparison
    print("\n2. Generating convergence history comparison...")
    convergence_data = generate_sample_convergence_data()
    
    fig_conv, axes_conv, widgets_conv = viz_tools.create_convergence_history_comparison(
        convergence_data, statistical_analysis=True,
        title="Convergence Analysis with Statistical Comparison"
    )
    print("✓ Convergence history comparison created")
    
    # 3. GPU Memory Visualization
    print("\n3. Generating GPU memory usage visualization...")
    memory_data = generate_sample_memory_data()
    
    fig_mem, axes_mem, widgets_mem = viz_tools.create_gpu_memory_visualization(
        memory_data, real_time=False,
        title="Multi-GPU Memory Usage Analysis"
    )
    print("✓ GPU memory visualization created")
    
    # 4. Precision Error Propagation Analysis
    print("\n4. Generating precision error propagation analysis...")
    error_data = generate_sample_error_propagation_data()
    
    fig_error, axes_error, widgets_error = viz_tools.create_precision_error_propagation_analysis(
        error_data, precision_levels=['fp16', 'fp32', 'fp64'],
        title="Mixed-Precision Error Propagation Analysis"
    )
    print("✓ Error propagation analysis created")
    
    # 5. Performance Scaling with Error Bars
    print("\n5. Generating performance scaling analysis...")
    scaling_data, confidence_intervals = generate_sample_scaling_data()
    
    fig_scale, axes_scale, widgets_scale = viz_tools.create_performance_scaling_with_error_bars(
        scaling_data, confidence_intervals,
        title="Performance Scaling Analysis with Confidence Intervals"
    )
    print("✓ Performance scaling analysis created")
    
    # Display all visualizations
    print("\n" + "="*50)
    print("All advanced visualizations have been created!")
    print("="*50)
    
    print("\nInteractive Features Available:")
    print("• 3D Solution: Use sliders to adjust slice positions and visualization parameters")
    print("• Convergence: Select different metrics and methods for comparison")
    print("• GPU Memory: Monitor memory usage across multiple GPUs with threshold alerts")
    print("• Error Analysis: Analyze precision error propagation across different data types")
    print("• Scaling: Compare performance scaling with statistical confidence intervals")
    
    print("\nNote: Close any figure window to proceed to the next visualization.")
    print("All figures are interactive - experiment with the controls!")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()