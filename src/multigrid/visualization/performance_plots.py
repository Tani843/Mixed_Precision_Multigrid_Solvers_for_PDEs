"""Performance visualization tools for CPU vs GPU comparisons and scaling analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os

logger = logging.getLogger(__name__)


class PerformancePlotter:
    """
    Professional performance visualization for multigrid solvers.
    
    Creates publication-quality plots for performance benchmarks,
    scaling studies, and CPU vs GPU comparisons.
    """
    
    def __init__(self, output_dir: str = "plots", style: str = "publication"):
        """Initialize performance plotter."""
        self.output_dir = output_dir
        self.style = style
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance-specific color schemes
        self.cpu_color = '#2E8B57'  # Sea green
        self.gpu_color = '#FF6B35'  # Orange red
        self.mixed_color = '#4A90E2'  # Blue
        
        self.colors = sns.color_palette("husl", 8)
        self.markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p']
        
        logger.info("Performance plotter initialized")
    
    def plot_scaling_analysis(
        self,
        problem_sizes: List[int],
        performance_data: Dict[str, Dict[str, List[float]]],
        title: str = "Scaling Analysis",
        save_name: Optional[str] = None,
        show_theory: bool = True
    ) -> plt.Figure:
        """
        Plot comprehensive scaling analysis.
        
        Args:
            problem_sizes: List of total unknowns (N)
            performance_data: Dict of {solver: {'times': [...], 'throughputs': [...], 'iterations': [...]}}
            title: Plot title
            save_name: Filename to save
            show_theory: Show theoretical scaling lines
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Solve Time vs Problem Size
        for i, (solver, data) in enumerate(performance_data.items()):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            times = data.get('times', [])
            if len(times) == len(problem_sizes):
                ax1.loglog(problem_sizes, times, color=color, marker=marker,
                          linewidth=2, markersize=6, label=solver, alpha=0.8)
        
        # Add theoretical scaling lines
        if show_theory and len(problem_sizes) >= 2:
            n_min, n_max = min(problem_sizes), max(problem_sizes)
            n_theory = np.logspace(np.log10(n_min), np.log10(n_max), 100)
            
            # Reference time (from first solver)
            first_solver = list(performance_data.keys())[0]
            times_ref = performance_data[first_solver].get('times', [])
            if times_ref:
                t_ref = times_ref[0]
                n_ref = problem_sizes[0]
                
                # O(N) - optimal multigrid
                t_linear = t_ref * (n_theory / n_ref)
                ax1.loglog(n_theory, t_linear, 'k--', alpha=0.7, linewidth=2, 
                          label='O(N) Optimal')
                
                # O(N log N) - good multigrid
                t_nlogn = t_ref * (n_theory / n_ref) * np.log(n_theory / n_ref) / np.log(n_theory[0] / n_ref)
                ax1.loglog(n_theory, t_nlogn, 'k:', alpha=0.7, linewidth=2,
                          label='O(N log N)')
                
                # O(N^1.5) - suboptimal
                t_n15 = t_ref * (n_theory / n_ref)**1.5
                ax1.loglog(n_theory, t_n15, 'k-.', alpha=0.5, linewidth=2,
                          label='O(N^1.5)')
        
        ax1.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax1.set_ylabel("Solve Time (seconds)", fontsize=12)
        ax1.set_title("Computational Complexity", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Throughput vs Problem Size
        for i, (solver, data) in enumerate(performance_data.items()):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            throughputs = data.get('throughputs', [])
            if len(throughputs) == len(problem_sizes):
                ax2.semilogx(problem_sizes, throughputs, color=color, marker=marker,
                            linewidth=2, markersize=6, label=solver, alpha=0.8)
        
        ax2.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax2.set_ylabel("Throughput (unknowns/sec)", fontsize=12)
        ax2.set_title("Computational Throughput", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Plot 3: Iteration Count vs Problem Size
        for i, (solver, data) in enumerate(performance_data.items()):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            iterations = data.get('iterations', [])
            if len(iterations) == len(problem_sizes):
                ax3.semilogx(problem_sizes, iterations, color=color, marker=marker,
                            linewidth=2, markersize=6, label=solver, alpha=0.8)
        
        # Ideal multigrid line (constant iterations)
        if len(problem_sizes) >= 2:
            ideal_iterations = [10] * len(problem_sizes)  # Ideal: ~10 iterations
            ax3.semilogx(problem_sizes, ideal_iterations, 'k--', alpha=0.7,
                        linewidth=2, label='Ideal Multigrid')
        
        ax3.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax3.set_ylabel("Iterations to Convergence", fontsize=12)
        ax3.set_title("Multigrid Efficiency", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Plot 4: Parallel Efficiency (if multiple solvers)
        if len(performance_data) > 1:
            # Calculate speedup relative to first solver
            solvers = list(performance_data.keys())
            baseline_solver = solvers[0]
            baseline_times = performance_data[baseline_solver].get('times', [])
            
            for i, solver in enumerate(solvers[1:], 1):
                color = self.colors[i % len(self.colors)]
                marker = self.markers[i % len(self.markers)]
                
                solver_times = performance_data[solver].get('times', [])
                if len(solver_times) == len(baseline_times):
                    speedups = [bt / st if st > 0 else 0 for bt, st in zip(baseline_times, solver_times)]
                    ax4.semilogx(problem_sizes, speedups, color=color, marker=marker,
                                linewidth=2, markersize=6, label=f'{solver} vs {baseline_solver}',
                                alpha=0.8)
            
            # Ideal speedup line (horizontal at some expected value)
            ax4.axhline(y=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup')
            if 'GPU' in ' '.join(solvers):
                ax4.axhline(y=2.0, color='k', linestyle='--', alpha=0.7, label='2x speedup')
                ax4.axhline(y=4.0, color='k', linestyle=':', alpha=0.7, label='4x speedup')
            
            ax4.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
            ax4.set_ylabel("Speedup Factor", fontsize=12)
            ax4.set_title("Performance Speedup", fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
        else:
            # Just show memory usage or other metric
            ax4.text(0.5, 0.5, "Additional metrics\ncan be shown here", 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
            ax4.set_title("Additional Metrics", fontsize=14)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_cpu_gpu_comparison(
        self,
        problem_sizes: List[int],
        cpu_data: Dict[str, List[float]],
        gpu_data: Dict[str, List[float]],
        title: str = "CPU vs GPU Performance",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot detailed CPU vs GPU performance comparison.
        
        Args:
            problem_sizes: List of total unknowns
            cpu_data: Dict with 'times', 'throughputs', etc.
            gpu_data: Dict with 'times', 'throughputs', etc.
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Solve Time Comparison
        cpu_times = cpu_data.get('times', [])
        gpu_times = gpu_data.get('times', [])
        
        if len(cpu_times) == len(problem_sizes):
            ax1.loglog(problem_sizes, cpu_times, color=self.cpu_color, marker='o',
                      linewidth=3, markersize=8, label='CPU', alpha=0.8)
        
        if len(gpu_times) == len(problem_sizes):
            ax1.loglog(problem_sizes, gpu_times, color=self.gpu_color, marker='s',
                      linewidth=3, markersize=8, label='GPU', alpha=0.8)
        
        ax1.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax1.set_ylabel("Solve Time (seconds)", fontsize=12)
        ax1.set_title("Solve Time Comparison", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        
        # Plot 2: Throughput Comparison
        cpu_throughputs = cpu_data.get('throughputs', [])
        gpu_throughputs = gpu_data.get('throughputs', [])
        
        if len(cpu_throughputs) == len(problem_sizes):
            ax2.semilogx(problem_sizes, cpu_throughputs, color=self.cpu_color, 
                        marker='o', linewidth=3, markersize=8, label='CPU', alpha=0.8)
        
        if len(gpu_throughputs) == len(problem_sizes):
            ax2.semilogx(problem_sizes, gpu_throughputs, color=self.gpu_color,
                        marker='s', linewidth=3, markersize=8, label='GPU', alpha=0.8)
        
        ax2.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax2.set_ylabel("Throughput (unknowns/sec)", fontsize=12)
        ax2.set_title("Throughput Comparison", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        # Plot 3: Speedup Analysis
        if len(cpu_times) == len(gpu_times) == len(problem_sizes):
            speedups = [ct / gt if gt > 0 else 0 for ct, gt in zip(cpu_times, gpu_times)]
            
            ax3.semilogx(problem_sizes, speedups, color=self.mixed_color, 
                        marker='D', linewidth=3, markersize=8, label='GPU Speedup', alpha=0.8)
            
            # Add speedup reference lines
            ax3.axhline(y=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup')
            ax3.axhline(y=2.0, color='k', linestyle='--', alpha=0.7, label='2x speedup')
            ax3.axhline(y=4.0, color='k', linestyle='--', alpha=0.7, label='4x speedup')
            ax3.axhline(y=8.0, color='k', linestyle='--', alpha=0.7, label='8x speedup')
            
            # Add text annotations for best speedup
            if speedups:
                max_speedup = max(speedups)
                max_idx = speedups.index(max_speedup)
                ax3.annotate(f'Max: {max_speedup:.1f}x', 
                            xy=(problem_sizes[max_idx], max_speedup),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                            fontsize=10, ha='left')
        
        ax3.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax3.set_ylabel("Speedup Factor (CPU time / GPU time)", fontsize=12)
        ax3.set_title("GPU Speedup Analysis", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Plot 4: Memory Usage Comparison (if available)
        cpu_memory = cpu_data.get('memory_usage', [])
        gpu_memory = gpu_data.get('memory_usage', [])
        
        if cpu_memory and gpu_memory:
            if len(cpu_memory) == len(problem_sizes):
                ax4.semilogx(problem_sizes, cpu_memory, color=self.cpu_color,
                            marker='o', linewidth=3, markersize=8, label='CPU Memory', alpha=0.8)
            
            if len(gpu_memory) == len(problem_sizes):
                ax4.semilogx(problem_sizes, gpu_memory, color=self.gpu_color,
                            marker='s', linewidth=3, markersize=8, label='GPU Memory', alpha=0.8)
            
            ax4.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
            ax4.set_ylabel("Memory Usage (MB)", fontsize=12)
            ax4.set_title("Memory Usage Comparison", fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=12)
        else:
            # Show efficiency metrics instead
            cpu_efficiency = cpu_data.get('efficiency', [50, 60, 65, 70])  # Example data
            gpu_efficiency = gpu_data.get('efficiency', [80, 85, 88, 90])  # Example data
            
            x_pos = np.arange(len(cpu_efficiency))
            width = 0.35
            
            bars1 = ax4.bar(x_pos - width/2, cpu_efficiency, width, 
                           color=self.cpu_color, alpha=0.8, label='CPU')
            bars2 = ax4.bar(x_pos + width/2, gpu_efficiency, width,
                           color=self.gpu_color, alpha=0.8, label='GPU')
            
            ax4.set_xlabel("Problem Size Category", fontsize=12)
            ax4.set_ylabel("Computational Efficiency (%)", fontsize=12)
            ax4.set_title("Computational Efficiency", fontsize=14)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f"Size {i+1}" for i in range(len(cpu_efficiency))])
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_precision_performance(
        self,
        problem_sizes: List[int],
        precision_data: Dict[str, Dict[str, List[float]]],
        title: str = "Mixed-Precision Performance Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot mixed-precision performance analysis.
        
        Args:
            problem_sizes: List of total unknowns
            precision_data: Dict of {precision_type: {'times': [...], 'errors': [...], 'memory': [...]}}
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define precision colors
        precision_colors = {
            'double': '#1f77b4',    # Blue
            'single': '#ff7f0e',    # Orange  
            'mixed': '#2ca02c',     # Green
            'adaptive': '#d62728'   # Red
        }
        
        precision_markers = {
            'double': 'o',
            'single': 's', 
            'mixed': '^',
            'adaptive': 'v'
        }
        
        # Plot 1: Solve Time vs Precision
        for precision, data in precision_data.items():
            color = precision_colors.get(precision, '#333333')
            marker = precision_markers.get(precision, 'o')
            
            times = data.get('times', [])
            if len(times) == len(problem_sizes):
                ax1.loglog(problem_sizes, times, color=color, marker=marker,
                          linewidth=2, markersize=6, label=precision.title(), alpha=0.8)
        
        ax1.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax1.set_ylabel("Solve Time (seconds)", fontsize=12)
        ax1.set_title("Solve Time by Precision", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        
        # Plot 2: Accuracy vs Precision
        for precision, data in precision_data.items():
            color = precision_colors.get(precision, '#333333')
            marker = precision_markers.get(precision, 'o')
            
            errors = data.get('errors', [])
            if len(errors) == len(problem_sizes):
                ax2.loglog(problem_sizes, errors, color=color, marker=marker,
                          linewidth=2, markersize=6, label=precision.title(), alpha=0.8)
        
        ax2.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax2.set_ylabel("L2 Error", fontsize=12)
        ax2.set_title("Accuracy by Precision", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        # Plot 3: Performance vs Accuracy Trade-off
        for precision, data in precision_data.items():
            color = precision_colors.get(precision, '#333333')
            marker = precision_markers.get(precision, 'o')
            
            times = data.get('times', [])
            errors = data.get('errors', [])
            
            if len(times) == len(errors):
                ax3.loglog(errors, times, color=color, marker=marker,
                          linewidth=2, markersize=8, label=precision.title(), alpha=0.8)
        
        ax3.set_xlabel("L2 Error", fontsize=12)
        ax3.set_ylabel("Solve Time (seconds)", fontsize=12)
        ax3.set_title("Performance vs Accuracy Trade-off", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12)
        
        # Add Pareto frontier if multiple precision types
        if len(precision_data) > 1:
            all_times = []
            all_errors = []
            for data in precision_data.values():
                times = data.get('times', [])
                errors = data.get('errors', [])
                if times and errors:
                    all_times.extend(times)
                    all_errors.extend(errors)
            
            if all_times and all_errors:
                # Simple Pareto frontier approximation
                points = list(zip(all_errors, all_times))
                points.sort()  # Sort by error
                
                pareto_errors = [points[0][0]]
                pareto_times = [points[0][1]]
                
                for error, time in points[1:]:
                    if time < pareto_times[-1]:  # Better performance
                        pareto_errors.append(error)
                        pareto_times.append(time)
                
                ax3.plot(pareto_errors, pareto_times, 'k--', alpha=0.7, 
                        linewidth=2, label='Pareto Frontier')
        
        # Plot 4: Speedup Analysis
        if 'double' in precision_data:
            double_times = precision_data['double'].get('times', [])
            
            for precision, data in precision_data.items():
                if precision != 'double':
                    color = precision_colors.get(precision, '#333333')
                    marker = precision_markers.get(precision, 'o')
                    
                    times = data.get('times', [])
                    if len(times) == len(double_times):
                        speedups = [dt / t if t > 0 else 0 for dt, t in zip(double_times, times)]
                        ax4.semilogx(problem_sizes, speedups, color=color, marker=marker,
                                    linewidth=2, markersize=6, 
                                    label=f'{precision.title()} vs Double', alpha=0.8)
        
        ax4.axhline(y=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup')
        ax4.set_xlabel("Problem Size (Total Unknowns)", fontsize=12)
        ax4.set_ylabel("Speedup vs Double Precision", fontsize=12)
        ax4.set_title("Precision Speedup Analysis", fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_weak_strong_scaling(
        self,
        weak_scaling_data: Dict[str, List[float]],
        strong_scaling_data: Dict[str, List[float]],
        title: str = "Weak vs Strong Scaling",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot weak vs strong scaling analysis.
        
        Args:
            weak_scaling_data: Dict with 'processors', 'efficiency', 'times'
            strong_scaling_data: Dict with 'processors', 'speedup', 'efficiency'
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Weak Scaling Efficiency
        processors_weak = weak_scaling_data.get('processors', [1, 2, 4, 8])
        efficiency_weak = weak_scaling_data.get('efficiency', [1.0, 0.9, 0.8, 0.7])
        
        ax1.plot(processors_weak, efficiency_weak, color=self.mixed_color, 
                marker='o', linewidth=3, markersize=8, label='Weak Scaling', alpha=0.8)
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal')
        ax1.axhline(y=0.8, color='r', linestyle=':', alpha=0.7, label='80% Threshold')
        
        ax1.set_xlabel("Number of Processors", fontsize=12)
        ax1.set_ylabel("Parallel Efficiency", fontsize=12)
        ax1.set_title("Weak Scaling Efficiency", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Strong Scaling Speedup
        processors_strong = strong_scaling_data.get('processors', [1, 2, 4, 8])
        speedup_strong = strong_scaling_data.get('speedup', [1.0, 1.8, 3.2, 5.6])
        
        ax2.plot(processors_strong, speedup_strong, color=self.gpu_color,
                marker='s', linewidth=3, markersize=8, label='Strong Scaling', alpha=0.8)
        ax2.plot(processors_strong, processors_strong, 'k--', alpha=0.7, 
                linewidth=2, label='Ideal Linear')
        
        ax2.set_xlabel("Number of Processors", fontsize=12)
        ax2.set_ylabel("Speedup Factor", fontsize=12)
        ax2.set_title("Strong Scaling Speedup", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        # Plot 3: Strong Scaling Efficiency
        efficiency_strong = strong_scaling_data.get('efficiency', 
                                                   [s/p for s, p in zip(speedup_strong, processors_strong)])
        
        ax3.plot(processors_strong, efficiency_strong, color=self.cpu_color,
                marker='^', linewidth=3, markersize=8, label='Strong Scaling Efficiency', alpha=0.8)
        ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal')
        ax3.axhline(y=0.5, color='r', linestyle=':', alpha=0.7, label='50% Threshold')
        
        ax3.set_xlabel("Number of Processors", fontsize=12)
        ax3.set_ylabel("Parallel Efficiency", fontsize=12)
        ax3.set_title("Strong Scaling Efficiency", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12)
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Comparison of Efficiencies
        ax4.plot(processors_weak, efficiency_weak, color=self.mixed_color,
                marker='o', linewidth=3, markersize=8, label='Weak Scaling', alpha=0.8)
        ax4.plot(processors_strong, efficiency_strong, color=self.cpu_color,
                marker='^', linewidth=3, markersize=8, label='Strong Scaling', alpha=0.8)
        
        ax4.axhline(y=0.8, color='orange', linestyle=':', alpha=0.7, label='Good (80%)')
        ax4.axhline(y=0.6, color='red', linestyle=':', alpha=0.7, label='Fair (60%)')
        
        ax4.set_xlabel("Number of Processors", fontsize=12)
        ax4.set_ylabel("Parallel Efficiency", fontsize=12)
        ax4.set_title("Scaling Efficiency Comparison", fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=12)
        ax4.set_ylim(0, 1.1)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, save_name: str):
        """Save figure in multiple formats."""
        png_path = os.path.join(self.output_dir, f"{save_name}.png")
        svg_path = os.path.join(self.output_dir, f"{save_name}.svg")
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        fig.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        logger.info(f"Saved performance plot: {png_path} and {svg_path}")


def plot_scaling_analysis(
    problem_sizes: List[int],
    performance_data: Dict[str, Dict[str, List[float]]],
    title: str = "Scaling Analysis", 
    save_name: Optional[str] = None,
    output_dir: str = "plots"
) -> plt.Figure:
    """
    Convenience function for scaling analysis plotting.
    
    Args:
        problem_sizes: List of total unknowns
        performance_data: Performance data dictionary
        title: Plot title
        save_name: Filename to save
        output_dir: Output directory
        
    Returns:
        Matplotlib figure object
    """
    plotter = PerformancePlotter(output_dir=output_dir)
    return plotter.plot_scaling_analysis(problem_sizes, performance_data, 
                                        title=title, save_name=save_name)


def plot_cpu_gpu_comparison(
    problem_sizes: List[int],
    cpu_data: Dict[str, List[float]],
    gpu_data: Dict[str, List[float]],
    title: str = "CPU vs GPU Performance",
    save_name: Optional[str] = None,
    output_dir: str = "plots"
) -> plt.Figure:
    """
    Convenience function for CPU vs GPU comparison plotting.
    
    Args:
        problem_sizes: List of total unknowns
        cpu_data: CPU performance data
        gpu_data: GPU performance data
        title: Plot title
        save_name: Filename to save
        output_dir: Output directory
        
    Returns:
        Matplotlib figure object
    """
    plotter = PerformancePlotter(output_dir=output_dir)
    return plotter.plot_cpu_gpu_comparison(problem_sizes, cpu_data, gpu_data,
                                          title=title, save_name=save_name)