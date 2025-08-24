"""Convergence visualization tools for residual history and convergence rates."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os

logger = logging.getLogger(__name__)


class ConvergencePlotter:
    """
    Professional convergence visualization for multigrid solvers.
    
    Creates publication-quality plots for residual histories,
    convergence rates, and error analysis.
    """
    
    def __init__(self, output_dir: str = "plots", style: str = "publication"):
        """Initialize convergence plotter."""
        self.output_dir = output_dir
        self.style = style
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette for different solvers/methods
        self.colors = sns.color_palette("husl", 8)
        self.markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p']
        self.linestyles = ['-', '--', '-.', ':']
        
        logger.info("Convergence plotter initialized")
    
    def plot_residual_history(
        self,
        residual_histories: Dict[str, np.ndarray],
        title: str = "Multigrid Convergence",
        xlabel: str = "Iteration",
        ylabel: str = "Residual",
        save_name: Optional[str] = None,
        semilogy: bool = True,
        show_theory: bool = True,
        convergence_factor: float = 0.1
    ) -> plt.Figure:
        """
        Plot residual convergence history for multiple solvers.
        
        Args:
            residual_histories: Dict of {solver_name: residual_array}
            title: Plot title
            xlabel, ylabel: Axis labels
            save_name: Filename to save
            semilogy: Use logarithmic y-axis
            show_theory: Show theoretical convergence line
            convergence_factor: Expected convergence factor
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for i, (solver_name, residuals) in enumerate(residual_histories.items()):
            iterations = np.arange(len(residuals))
            
            # Plot residual history
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            if semilogy:
                ax.semilogy(iterations, residuals, color=color, marker=marker,
                           linewidth=2, markersize=6, markevery=max(1, len(residuals)//20),
                           label=solver_name, alpha=0.8)
            else:
                ax.plot(iterations, residuals, color=color, marker=marker,
                       linewidth=2, markersize=6, markevery=max(1, len(residuals)//20),
                       label=solver_name, alpha=0.8)
        
        # Show theoretical convergence if requested
        if show_theory and len(residual_histories) > 0:
            # Use first solver as reference
            first_residuals = list(residual_histories.values())[0]
            if len(first_residuals) > 1:
                theory_iterations = np.arange(len(first_residuals))
                initial_residual = first_residuals[0]
                theory_residuals = initial_residual * (convergence_factor ** theory_iterations)
                
                ax.semilogy(theory_iterations, theory_residuals, 'k--', 
                           linewidth=2, alpha=0.7, 
                           label=f'Theory (ρ={convergence_factor})')
        
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # Add convergence rate annotations
        self._add_convergence_annotations(ax, residual_histories, semilogy)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def _add_convergence_annotations(
        self,
        ax: plt.Axes,
        residual_histories: Dict[str, np.ndarray],
        semilogy: bool
    ):
        """Add convergence factor annotations to the plot."""
        y_pos = 0.95  # Start position for annotations
        
        for solver_name, residuals in residual_histories.items():
            if len(residuals) > 10:  # Need enough points to calculate factor
                # Calculate average convergence factor
                ratios = []
                for i in range(5, len(residuals)-5):  # Skip initial and final points
                    if residuals[i] > 0 and residuals[i+1] > 0:
                        ratio = residuals[i+1] / residuals[i]
                        if 0.01 < ratio < 1.0:  # Reasonable range
                            ratios.append(ratio)
                
                if ratios:
                    avg_factor = np.mean(ratios)
                    annotation = f"{solver_name}: ρ ≈ {avg_factor:.3f}"
                    
                    ax.text(0.02, y_pos, annotation, transform=ax.transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                                 facecolor='white', alpha=0.8))
                    y_pos -= 0.05
    
    def plot_convergence_rates(
        self,
        grid_sizes: List[Tuple[int, int]],
        errors: Dict[str, List[float]],
        error_types: List[str] = None,
        title: str = "Grid Convergence Study",
        save_name: Optional[str] = None,
        theoretical_slopes: Dict[str, float] = None
    ) -> plt.Figure:
        """
        Plot convergence rates for different error norms.
        
        Args:
            grid_sizes: List of (nx, ny) grid sizes
            errors: Dict of {error_type: [error_values]}
            error_types: Types of errors to plot
            title: Plot title
            save_name: Filename to save
            theoretical_slopes: Expected convergence slopes
            
        Returns:
            Matplotlib figure object
        """
        if error_types is None:
            error_types = list(errors.keys())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate grid spacings (assuming square grids)
        h_values = [1.0 / (nx - 1) for nx, ny in grid_sizes]
        
        for i, error_type in enumerate(error_types):
            if error_type not in errors:
                continue
            
            error_values = errors[error_type]
            if len(error_values) != len(h_values):
                logger.warning(f"Mismatch in data length for {error_type}")
                continue
            
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            # Plot error vs grid spacing
            ax.loglog(h_values, error_values, color=color, marker=marker,
                     linewidth=2, markersize=8, label=error_type.replace('_', ' ').title())
            
            # Fit and plot trend line
            if len(h_values) >= 2:
                log_h = np.log10(h_values)
                log_errors = np.log10(error_values)
                
                # Linear regression in log space
                coeffs = np.polyfit(log_h, log_errors, 1)
                slope = coeffs[0]
                
                # Create trend line
                h_trend = np.logspace(np.log10(min(h_values)), np.log10(max(h_values)), 100)
                error_trend = 10**(coeffs[1]) * h_trend**slope
                
                ax.loglog(h_trend, error_trend, '--', color=color, alpha=0.7,
                         linewidth=1.5, label=f'{error_type} slope: {slope:.2f}')
        
        # Add theoretical lines if provided
        if theoretical_slopes:
            for error_type, slope in theoretical_slopes.items():
                if error_type in errors and error_type in error_types:
                    # Use first point to normalize
                    error_values = errors[error_type]
                    h_ref = h_values[0]
                    error_ref = error_values[0]
                    
                    h_theory = np.logspace(np.log10(min(h_values)), np.log10(max(h_values)), 100)
                    error_theory = error_ref * (h_theory / h_ref)**slope
                    
                    ax.loglog(h_theory, error_theory, ':', color='black', alpha=0.8,
                             linewidth=2, label=f'Theory O(h^{slope})')
        
        ax.set_xlabel("Grid Spacing h", fontsize=14)
        ax.set_ylabel("Error", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Add grid size annotations
        for i, (h, (nx, ny)) in enumerate(zip(h_values, grid_sizes)):
            if i % 2 == 0:  # Annotate every other point to avoid crowding
                ax.annotate(f'{nx}×{ny}', (h, min([errors[et][i] for et in error_types if et in errors])),
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_multigrid_efficiency(
        self,
        grid_sizes: List[Tuple[int, int]],
        iteration_counts: Dict[str, List[int]],
        solve_times: Dict[str, List[float]],
        title: str = "Multigrid Efficiency Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot multigrid efficiency metrics.
        
        Args:
            grid_sizes: List of grid sizes
            iteration_counts: Dict of {solver: [iterations]}
            solve_times: Dict of {solver: [times]}
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Calculate total unknowns
        total_unknowns = [nx * ny for nx, ny in grid_sizes]
        
        # Plot 1: Iterations vs Problem Size
        for i, (solver, iterations) in enumerate(iteration_counts.items()):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            ax1.semilogx(total_unknowns, iterations, color=color, marker=marker,
                        linewidth=2, markersize=6, label=solver)
        
        ax1.set_xlabel("Total Unknowns", fontsize=12)
        ax1.set_ylabel("Iterations to Convergence", fontsize=12)
        ax1.set_title("Iteration Count vs Problem Size", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Solve Time vs Problem Size
        for i, (solver, times) in enumerate(solve_times.items()):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            ax2.loglog(total_unknowns, times, color=color, marker=marker,
                      linewidth=2, markersize=6, label=solver)
        
        # Add theoretical scaling lines
        if len(total_unknowns) >= 2:
            n_min, n_max = min(total_unknowns), max(total_unknowns)
            n_theory = np.logspace(np.log10(n_min), np.log10(n_max), 100)
            
            # O(N) scaling
            t_ref = min([min(times) for times in solve_times.values()])
            n_ref = min(total_unknowns)
            t_linear = t_ref * (n_theory / n_ref)
            ax2.loglog(n_theory, t_linear, 'k--', alpha=0.7, label='O(N) Theory')
            
            # O(N log N) scaling
            t_nlogn = t_ref * (n_theory / n_ref) * np.log(n_theory / n_ref)
            ax2.loglog(n_theory, t_nlogn, 'k:', alpha=0.7, label='O(N log N)')
        
        ax2.set_xlabel("Total Unknowns", fontsize=12)
        ax2.set_ylabel("Solve Time (seconds)", fontsize=12)
        ax2.set_title("Solve Time vs Problem Size", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Throughput Analysis
        for i, (solver, times) in enumerate(solve_times.items()):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            throughputs = [n / t for n, t in zip(total_unknowns, times)]
            ax3.semilogx(total_unknowns, throughputs, color=color, marker=marker,
                        linewidth=2, markersize=6, label=solver)
        
        ax3.set_xlabel("Total Unknowns", fontsize=12)
        ax3.set_ylabel("Throughput (unknowns/sec)", fontsize=12)
        ax3.set_title("Computational Throughput", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_error_components(
        self,
        iterations: np.ndarray,
        error_components: Dict[str, np.ndarray],
        title: str = "Error Component Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot different error components (discretization, iteration, etc.).
        
        Args:
            iterations: Iteration numbers
            error_components: Dict of {component_name: error_values}
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Stack the error components
        bottom = np.zeros(len(iterations))
        colors = sns.color_palette("Set2", len(error_components))
        
        for i, (component, errors) in enumerate(error_components.items()):
            ax.fill_between(iterations, bottom, bottom + errors, 
                          color=colors[i], alpha=0.7, label=component)
            bottom += errors
        
        # Also plot individual components
        for i, (component, errors) in enumerate(error_components.items()):
            ax.plot(iterations, errors, color=colors[i], linewidth=2, alpha=0.9)
        
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Error Magnitude", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_convergence_comparison(
        self,
        solvers_data: Dict[str, Dict[str, Any]],
        title: str = "Solver Convergence Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare convergence behavior of different solvers.
        
        Args:
            solvers_data: Dict of {solver_name: {'residuals': array, 'iterations': int, 'time': float}}
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Residual histories
        for i, (solver, data) in enumerate(solvers_data.items()):
            color = self.colors[i % len(self.colors)]
            residuals = data.get('residuals', [])
            if len(residuals) > 0:
                iterations = np.arange(len(residuals))
                ax1.semilogy(iterations, residuals, color=color, linewidth=2,
                           marker=self.markers[i % len(self.markers)], 
                           markersize=4, markevery=max(1, len(residuals)//10),
                           label=solver)
        
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Residual")
        ax1.set_title("Residual Convergence")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Final iteration count comparison
        solver_names = list(solvers_data.keys())
        final_iterations = [data.get('iterations', 0) for data in solvers_data.values()]
        
        bars = ax2.bar(solver_names, final_iterations, color=self.colors[:len(solver_names)],
                      alpha=0.8)
        ax2.set_ylabel("Iterations to Convergence")
        ax2.set_title("Final Iteration Count")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_iterations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom')
        
        # Subplot 3: Solve time comparison
        solve_times = [data.get('time', 0) for data in solvers_data.values()]
        
        bars = ax3.bar(solver_names, solve_times, color=self.colors[:len(solver_names)],
                      alpha=0.8)
        ax3.set_ylabel("Solve Time (seconds)")
        ax3.set_title("Total Solve Time")
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, solve_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(solve_times)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Subplot 4: Convergence factor comparison
        convergence_factors = []
        for solver, data in solvers_data.items():
            residuals = data.get('residuals', [])
            if len(residuals) > 10:
                # Calculate average convergence factor
                ratios = []
                for i in range(5, len(residuals)-1):
                    if residuals[i] > 0 and residuals[i+1] > 0:
                        ratio = residuals[i+1] / residuals[i]
                        if 0.01 < ratio < 1.0:
                            ratios.append(ratio)
                
                factor = np.mean(ratios) if ratios else 1.0
                convergence_factors.append(factor)
            else:
                convergence_factors.append(1.0)
        
        bars = ax4.bar(solver_names, convergence_factors, 
                      color=self.colors[:len(solver_names)], alpha=0.8)
        ax4.set_ylabel("Convergence Factor")
        ax4.set_title("Average Convergence Factor")
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, 
                   label='Target (0.1)')
        ax4.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, convergence_factors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
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
        
        logger.info(f"Saved convergence plot: {png_path} and {svg_path}")


def plot_residual_history(
    residual_histories: Dict[str, np.ndarray],
    title: str = "Residual Convergence",
    save_name: Optional[str] = None,
    output_dir: str = "plots"
) -> plt.Figure:
    """
    Convenience function for plotting residual histories.
    
    Args:
        residual_histories: Dict of {solver_name: residual_array}
        title: Plot title
        save_name: Filename to save
        output_dir: Output directory
        
    Returns:
        Matplotlib figure object
    """
    plotter = ConvergencePlotter(output_dir=output_dir)
    return plotter.plot_residual_history(residual_histories, title=title, 
                                        save_name=save_name)


def plot_convergence_rates(
    grid_sizes: List[Tuple[int, int]],
    errors: Dict[str, List[float]],
    title: str = "Grid Convergence Study",
    save_name: Optional[str] = None,
    output_dir: str = "plots"
) -> plt.Figure:
    """
    Convenience function for plotting convergence rates.
    
    Args:
        grid_sizes: List of (nx, ny) grid sizes
        errors: Dict of {error_type: [error_values]}
        title: Plot title
        save_name: Filename to save
        output_dir: Output directory
        
    Returns:
        Matplotlib figure object
    """
    plotter = ConvergencePlotter(output_dir=output_dir)
    return plotter.plot_convergence_rates(grid_sizes, errors, title=title,
                                         save_name=save_name)