"""
Analysis Plots Module

Comprehensive analysis and comparison plotting for the Mixed-Precision Multigrid Solvers project.
Focuses on error analysis, statistical validation, and method comparisons.

Classes:
    AnalysisVisualizer: Main class for analysis visualization
    
Functions:
    plot_error_decomposition: Visualize different error components
    plot_statistical_validation: Statistical analysis with confidence intervals
    plot_method_comparison: Compare different solver methods
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union, Any
import warnings


class AnalysisVisualizer:
    """
    Comprehensive analysis visualization for mixed-precision multigrid methods.
    
    Provides tools for error analysis, statistical validation, method comparison,
    and comprehensive performance analysis with confidence intervals.
    """
    
    def __init__(self, style='publication', dpi=300):
        """
        Initialize analysis visualizer with specified style.
        
        Args:
            style: Plot style ('publication', 'presentation', 'minimal')
            dpi: Resolution for saved figures
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style for publication-quality plots."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'Times'],
                'font.size': 12,
                'axes.linewidth': 1.2,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.dpi': self.dpi,
                'savefig.dpi': self.dpi,
                'savefig.bbox': 'tight',
                'text.usetex': False,
                'mathtext.fontset': 'cm'
            })
        
        # Color schemes for different components
        self.primary_color = '#2563eb'
        self.secondary_color = '#dc2626'
        self.accent_color = '#059669'
        self.warning_color = '#d97706'
        self.error_color = '#ef4444'
        
        self.method_colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0891b2']
    
    def plot_error_decomposition(self, grid_sizes: List[int], 
                               discretization_errors: List[float],
                               iteration_errors: List[float],
                               roundoff_errors: List[float],
                               title: str = "Error Component Analysis",
                               save_name: Optional[str] = None):
        """
        Plot decomposition of total error into different components.
        
        Args:
            grid_sizes: List of grid sizes
            discretization_errors: Discretization error values  
            iteration_errors: Iteration error values
            roundoff_errors: Roundoff error values
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        h_values = [1.0 / (n - 1) for n in grid_sizes]
        total_errors = [np.sqrt(d**2 + i**2 + r**2) 
                       for d, i, r in zip(discretization_errors, iteration_errors, roundoff_errors)]
        
        # Plot 1: Individual error components
        ax1.loglog(h_values, discretization_errors, 'o-', color=self.primary_color, 
                   linewidth=2, markersize=8, label='Discretization Error')
        ax1.loglog(h_values, iteration_errors, 's-', color=self.secondary_color, 
                   linewidth=2, markersize=8, label='Iteration Error')
        ax1.loglog(h_values, roundoff_errors, '^-', color=self.accent_color, 
                   linewidth=2, markersize=8, label='Roundoff Error')
        ax1.loglog(h_values, total_errors, 'D-', color=self.error_color, 
                   linewidth=2, markersize=8, label='Total Error')
        
        # Add theoretical slopes
        if len(h_values) >= 2:
            h_theory = np.array([h_values[0], h_values[-1]])
            # O(h^2) slope for discretization error
            disc_theory = discretization_errors[0] * (h_theory / h_values[0]) ** 2
            ax1.loglog(h_theory, disc_theory, '--', color=self.primary_color, 
                      alpha=0.7, linewidth=1, label='O(h²) slope')
        
        ax1.set_xlabel('Grid Spacing (h)', fontsize=12)
        ax1.set_ylabel('Error Magnitude', fontsize=12)
        ax1.set_title('Error Components vs Grid Spacing', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Relative contribution of each error component
        discretization_fraction = [d/t for d, t in zip(discretization_errors, total_errors)]
        iteration_fraction = [i/t for i, t in zip(iteration_errors, total_errors)]
        roundoff_fraction = [r/t for r, t in zip(roundoff_errors, total_errors)]
        
        width = 0.6
        x_pos = np.arange(len(grid_sizes))
        
        ax2.bar(x_pos, discretization_fraction, width, color=self.primary_color, 
               alpha=0.8, label='Discretization')
        ax2.bar(x_pos, iteration_fraction, width, bottom=discretization_fraction,
               color=self.secondary_color, alpha=0.8, label='Iteration')
        
        # Calculate bottom for roundoff error
        bottom_roundoff = [d + i for d, i in zip(discretization_fraction, iteration_fraction)]
        ax2.bar(x_pos, roundoff_fraction, width, bottom=bottom_roundoff,
               color=self.accent_color, alpha=0.8, label='Roundoff')
        
        ax2.set_xlabel('Grid Size', fontsize=12)
        ax2.set_ylabel('Relative Contribution', fontsize=12)
        ax2.set_title('Error Component Contributions', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{n}×{n}' for n in grid_sizes])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig
    
    def plot_statistical_validation(self, problem_sizes: List[int], 
                                  mean_errors: List[float], 
                                  std_errors: List[float],
                                  n_samples: int = 10,
                                  confidence_level: float = 0.95,
                                  title: str = "Statistical Validation",
                                  save_name: Optional[str] = None):
        """
        Plot statistical validation with confidence intervals and regression analysis.
        
        Args:
            problem_sizes: List of problem sizes
            mean_errors: Mean error values
            std_errors: Standard deviation of errors
            n_samples: Number of samples for statistical analysis
            confidence_level: Confidence level for intervals (0.95 for 95%)
            title: Plot title  
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        h_values = [1.0 / (n - 1) for n in problem_sizes]
        
        # Calculate confidence intervals
        t_value = stats.t.ppf((1 + confidence_level) / 2, n_samples - 1)
        margin_error = [t_value * std / np.sqrt(n_samples) for std in std_errors]
        
        # Plot 1: Error with confidence intervals
        ax1.errorbar(h_values, mean_errors, yerr=margin_error, 
                    fmt='o-', color=self.primary_color, linewidth=2, markersize=8,
                    capsize=5, capthick=2, elinewidth=2,
                    label=f'{confidence_level*100:.0f}% Confidence Interval')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Grid Spacing (h)', fontsize=12)
        ax1.set_ylabel('Mean L² Error', fontsize=12)
        ax1.set_title('Convergence with Confidence Intervals', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add theoretical convergence line
        if len(h_values) >= 2:
            h_theory = np.array([h_values[0], h_values[-1]])
            error_theory = mean_errors[0] * (h_theory / h_values[0]) ** 2
            ax1.loglog(h_theory, error_theory, '--', color=self.secondary_color, 
                      linewidth=2, alpha=0.8, label='Theoretical O(h²)')
            ax1.legend()
        
        # Plot 2: Convergence rate analysis
        if len(h_values) >= 3:
            rates = []
            rate_positions = []
            for i in range(1, len(h_values)):
                rate = np.log(mean_errors[i]/mean_errors[i-1]) / np.log(h_values[i]/h_values[i-1])
                rates.append(rate)
                rate_positions.append((h_values[i] + h_values[i-1]) / 2)
            
            ax2.semilogx(rate_positions, rates, 'o-', color=self.accent_color, 
                        linewidth=2, markersize=8, label='Observed Rate')
            ax2.axhline(y=2.0, color=self.secondary_color, linestyle='--', 
                       linewidth=2, alpha=0.8, label='Theoretical Rate (2.0)')
            ax2.set_xlabel('Grid Spacing (h)', fontsize=12)
            ax2.set_ylabel('Convergence Rate', fontsize=12)
            ax2.set_title('Local Convergence Rate', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: Distribution analysis (box plot for different grid sizes)
        # Simulate individual samples for demonstration
        np.random.seed(42)
        sample_data = []
        positions = []
        labels = []
        
        for i, (mean, std) in enumerate(zip(mean_errors, std_errors)):
            if std > 0:
                samples = np.random.normal(mean, std, n_samples)
                sample_data.append(samples)
                positions.append(i)
                labels.append(f'{problem_sizes[i]}²')
        
        if sample_data:
            bp = ax3.boxplot(sample_data, positions=positions, widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=self.primary_color, alpha=0.7),
                           medianprops=dict(color='white', linewidth=2))
            ax3.set_yscale('log')
            ax3.set_xlabel('Problem Size', fontsize=12)
            ax3.set_ylabel('Error Distribution', fontsize=12)
            ax3.set_title('Error Distribution Analysis', fontsize=14)
            ax3.set_xticks(positions)
            ax3.set_xticklabels(labels)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: R² goodness of fit analysis
        if len(h_values) >= 3:
            # Linear regression in log space
            log_h = np.log(h_values)
            log_err = np.log(mean_errors)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept = np.polyfit(log_h, log_err, 1)
                predicted_log = slope * log_h + intercept
                predicted_err = np.exp(predicted_log)
            
            # Calculate R²
            ss_res = np.sum((log_err - predicted_log) ** 2)
            ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            ax4.loglog(h_values, mean_errors, 'o', color=self.primary_color, 
                      markersize=8, label='Observed')
            ax4.loglog(h_values, predicted_err, '-', color=self.secondary_color, 
                      linewidth=2, label=f'Fitted (R² = {r_squared:.4f})')
            ax4.set_xlabel('Grid Spacing (h)', fontsize=12)
            ax4.set_ylabel('L² Error', fontsize=12)
            ax4.set_title('Regression Fit Quality', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add text with fitted parameters
            ax4.text(0.05, 0.95, f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig
    
    def plot_method_comparison(self, methods: Dict[str, Dict], 
                              metric: str = 'solve_time',
                              title: str = "Method Comparison",
                              save_name: Optional[str] = None):
        """
        Compare different solver methods across multiple metrics.
        
        Args:
            methods: Dictionary with method names as keys and data dictionaries as values
            metric: Primary metric for comparison ('solve_time', 'iterations', 'error')
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_methods = len(methods)
        method_names = list(methods.keys())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Determine problem sizes (assume all methods have same sizes)
        first_method = next(iter(methods.values()))
        problem_sizes = first_method.get('problem_sizes', [])
        
        # Plot 1: Primary metric comparison
        for i, (method_name, data) in enumerate(methods.items()):
            values = data.get(metric, [])
            color = self.method_colors[i % len(self.method_colors)]
            
            if metric in ['solve_time', 'error']:
                ax1.loglog(problem_sizes, values, 'o-', color=color, 
                          linewidth=2, markersize=8, label=method_name)
            else:
                ax1.plot(problem_sizes, values, 'o-', color=color, 
                        linewidth=2, markersize=8, label=method_name)
        
        ax1.set_xlabel('Problem Size (N)', fontsize=12)
        ax1.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax1.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Speedup analysis (relative to first method)
        if len(methods) > 1:
            baseline_method = method_names[0]
            baseline_times = methods[baseline_method].get('solve_time', [])
            
            for i, (method_name, data) in enumerate(methods.items()):
                if method_name == baseline_method:
                    continue
                    
                solve_times = data.get('solve_time', [])
                speedups = [bt / st if st > 0 else 0 
                           for bt, st in zip(baseline_times, solve_times)]
                
                color = self.method_colors[i % len(self.method_colors)]
                ax2.semilogx(problem_sizes, speedups, 'o-', color=color, 
                           linewidth=2, markersize=8, label=f'{method_name} vs {baseline_method}')
            
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Problem Size (N)', fontsize=12)
            ax2.set_ylabel('Speedup Factor', fontsize=12)
            ax2.set_title('Speedup Analysis', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: Iteration count comparison
        for i, (method_name, data) in enumerate(methods.items()):
            iterations = data.get('iterations', [])
            if iterations:
                color = self.method_colors[i % len(self.method_colors)]
                ax3.semilogx(problem_sizes, iterations, 'o-', color=color, 
                            linewidth=2, markersize=8, label=method_name)
        
        ax3.set_xlabel('Problem Size (N)', fontsize=12)
        ax3.set_ylabel('Iterations to Convergence', fontsize=12)
        ax3.set_title('Iteration Count Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Efficiency analysis (work per problem size)
        for i, (method_name, data) in enumerate(methods.items()):
            solve_times = data.get('solve_time', [])
            if solve_times:
                efficiency = [n / t if t > 0 else 0 
                             for n, t in zip(problem_sizes, solve_times)]
                color = self.method_colors[i % len(self.method_colors)]
                ax4.loglog(problem_sizes, efficiency, 'o-', color=color, 
                          linewidth=2, markersize=8, label=method_name)
        
        ax4.set_xlabel('Problem Size (N)', fontsize=12)
        ax4.set_ylabel('Efficiency (N/time)', fontsize=12)
        ax4.set_title('Computational Efficiency', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig
    
    def plot_precision_impact(self, precision_data: Dict[str, Dict],
                             title: str = "Mixed-Precision Impact Analysis", 
                             save_name: Optional[str] = None):
        """
        Analyze the impact of different precision strategies.
        
        Args:
            precision_data: Dictionary with precision types as keys
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        precision_types = list(precision_data.keys())
        colors = self.method_colors[:len(precision_types)]
        
        # Extract problem sizes (assume consistent across precision types)
        first_precision = next(iter(precision_data.values()))
        problem_sizes = first_precision.get('problem_sizes', [])
        
        # Plot 1: Performance comparison
        for i, (precision_type, data) in enumerate(precision_data.items()):
            solve_times = data.get('solve_times', [])
            ax1.loglog(problem_sizes, solve_times, 'o-', color=colors[i], 
                      linewidth=2, markersize=8, label=precision_type)
        
        ax1.set_xlabel('Problem Size (N)', fontsize=12)
        ax1.set_ylabel('Solve Time (s)', fontsize=12)
        ax1.set_title('Performance vs Precision', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Accuracy comparison
        for i, (precision_type, data) in enumerate(precision_data.items()):
            errors = data.get('errors', [])
            ax2.loglog(problem_sizes, errors, 'o-', color=colors[i], 
                      linewidth=2, markersize=8, label=precision_type)
        
        ax2.set_xlabel('Problem Size (N)', fontsize=12)
        ax2.set_ylabel('L² Error', fontsize=12)
        ax2.set_title('Accuracy vs Precision', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Memory usage comparison
        for i, (precision_type, data) in enumerate(precision_data.items()):
            memory_usage = data.get('memory_usage', [])
            if memory_usage:
                ax3.loglog(problem_sizes, memory_usage, 'o-', color=colors[i], 
                          linewidth=2, markersize=8, label=precision_type)
        
        ax3.set_xlabel('Problem Size (N)', fontsize=12)
        ax3.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax3.set_title('Memory Usage vs Precision', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Performance-accuracy trade-off (Pareto frontier)
        for i, (precision_type, data) in enumerate(precision_data.items()):
            solve_times = data.get('solve_times', [])
            errors = data.get('errors', [])
            
            if solve_times and errors:
                ax4.loglog(solve_times, errors, 'o', color=colors[i], 
                          markersize=10, alpha=0.7, label=precision_type)
        
        ax4.set_xlabel('Solve Time (s)', fontsize=12)
        ax4.set_ylabel('L² Error', fontsize=12)
        ax4.set_title('Performance-Accuracy Trade-off', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add arrow indicating "better" direction
        ax4.annotate('Better', xy=(0.1, 0.9), xytext=(0.3, 0.7),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red', fontweight='bold')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig


# Convenience functions for quick plotting
def plot_error_analysis(grid_sizes: List[int], discretization_errors: List[float],
                       iteration_errors: List[float], roundoff_errors: List[float], **kwargs):
    """
    Quick function to plot error component analysis.
    
    Args:
        grid_sizes: List of grid sizes
        discretization_errors: Discretization error values
        iteration_errors: Iteration error values  
        roundoff_errors: Roundoff error values
        **kwargs: Additional arguments passed to AnalysisVisualizer.plot_error_decomposition
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    visualizer = AnalysisVisualizer()
    return visualizer.plot_error_decomposition(grid_sizes, discretization_errors, 
                                             iteration_errors, roundoff_errors, **kwargs)


def plot_method_comparison_quick(methods: Dict[str, Dict], **kwargs):
    """
    Quick function to compare different solver methods.
    
    Args:
        methods: Dictionary with method data
        **kwargs: Additional arguments passed to AnalysisVisualizer.plot_method_comparison
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    visualizer = AnalysisVisualizer()
    return visualizer.plot_method_comparison(methods, **kwargs)