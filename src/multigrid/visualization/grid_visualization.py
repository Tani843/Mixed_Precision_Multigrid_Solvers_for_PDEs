"""
Grid Visualization Module

Visualizes multigrid hierarchies, grid structures, and adaptive refinement patterns
for the Mixed-Precision Multigrid Solvers project.

Classes:
    GridVisualizer: Main class for grid structure visualization
    
Functions:
    plot_grid_hierarchy: Visualize multigrid grid levels
    plot_adaptive_grid: Display adaptive mesh refinement
    plot_grid_convergence: Show convergence across grid levels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import warnings

class GridVisualizer:
    """
    Comprehensive grid visualization for multigrid methods.
    
    Provides tools for visualizing grid hierarchies, mesh structures,
    adaptive refinement patterns, and convergence behavior across 
    different grid levels.
    """
    
    def __init__(self, style='publication', dpi=300):
        """
        Initialize grid visualizer with specified style.
        
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
        
        # Color schemes for different grid levels
        self.level_colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed']
        self.grid_color = '#6b7280'
        self.highlight_color = '#ef4444'
    
    def plot_grid_hierarchy(self, grid_sizes: List[int], domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
                           title: str = "Multigrid Hierarchy", save_name: Optional[str] = None):
        """
        Visualize multigrid grid hierarchy showing coarsening pattern.
        
        Args:
            grid_sizes: List of grid sizes for each level (nx values)
            domain: Domain boundaries (x_min, x_max, y_min, y_max)
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_levels = len(grid_sizes)
        fig, axes = plt.subplots(1, n_levels, figsize=(4*n_levels, 4))
        
        if n_levels == 1:
            axes = [axes]
        
        x_min, x_max, y_min, y_max = domain
        
        for level, (ax, nx) in enumerate(zip(axes, grid_sizes)):
            ny = nx  # Assume square grids
            
            # Create grid lines
            x = np.linspace(x_min, x_max, nx + 1)
            y = np.linspace(y_min, y_max, ny + 1)
            
            # Plot vertical lines
            for xi in x:
                ax.axvline(xi, color=self.level_colors[level % len(self.level_colors)], 
                          linewidth=1.5 if level == 0 else 1.0, alpha=0.8)
            
            # Plot horizontal lines
            for yi in y:
                ax.axhline(yi, color=self.level_colors[level % len(self.level_colors)], 
                          linewidth=1.5 if level == 0 else 1.0, alpha=0.8)
            
            # Formatting
            ax.set_xlim(x_min - 0.05, x_max + 0.05)
            ax.set_ylim(y_min - 0.05, y_max + 0.05)
            ax.set_aspect('equal')
            ax.set_title(f'Level {level}\n{nx}×{ny} grid', fontsize=12)
            
            # Add grid statistics
            h = (x_max - x_min) / nx
            ax.text(0.02, 0.98, f'h = {h:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        fig.suptitle(title, fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig
    
    def plot_grid_convergence_study(self, grid_sizes: List[int], errors: List[float], 
                                   error_type: str = "L2 Error", theoretical_rate: float = 2.0,
                                   title: str = "Grid Convergence Study", save_name: Optional[str] = None):
        """
        Visualize grid convergence study with theoretical comparison.
        
        Args:
            grid_sizes: List of grid sizes (number of points in each direction)
            errors: Corresponding error values
            error_type: Type of error being plotted
            theoretical_rate: Theoretical convergence rate
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate grid spacing
        h_values = [1.0 / (n - 1) for n in grid_sizes]
        
        # Plot 1: Error vs Grid Size
        ax1.loglog(grid_sizes, errors, 'o-', color=self.level_colors[0], 
                   linewidth=2, markersize=8, label='Computed Error')
        
        ax1.set_xlabel('Grid Size (N)', fontsize=12)
        ax1.set_ylabel(f'{error_type}', fontsize=12)
        ax1.set_title('Error vs Grid Size', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Error vs Grid Spacing with theoretical rate
        ax2.loglog(h_values, errors, 'o-', color=self.level_colors[0], 
                   linewidth=2, markersize=8, label='Computed Error')
        
        # Add theoretical slope
        if len(h_values) >= 2:
            h_theory = np.array([h_values[0], h_values[-1]])
            error_theory = errors[0] * (h_theory / h_values[0]) ** theoretical_rate
            ax2.loglog(h_theory, error_theory, '--', color=self.highlight_color, 
                      linewidth=2, label=f'Theory: O(h^{theoretical_rate})')
        
        ax2.set_xlabel('Grid Spacing (h)', fontsize=12)
        ax2.set_ylabel(f'{error_type}', fontsize=12)
        ax2.set_title('Error vs Grid Spacing', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Calculate and display convergence rate
        if len(h_values) >= 2:
            # Linear regression in log space
            log_h = np.log(h_values)
            log_err = np.log(errors)
            
            # Handle potential issues with log of very small numbers
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, intercept = np.polyfit(log_h, log_err, 1)
            
            # Add convergence rate annotation
            ax2.text(0.05, 0.95, f'Computed Rate: {slope:.2f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig
    
    def plot_adaptive_grid(self, refined_regions: List[Tuple[float, float, float, float]],
                          base_grid_size: int = 32, domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
                          title: str = "Adaptive Mesh Refinement", save_name: Optional[str] = None):
        """
        Visualize adaptive mesh refinement pattern.
        
        Args:
            refined_regions: List of refined regions as (x_min, x_max, y_min, y_max)
            base_grid_size: Base grid resolution
            domain: Domain boundaries
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        x_min, x_max, y_min, y_max = domain
        
        # Draw base grid
        x_base = np.linspace(x_min, x_max, base_grid_size + 1)
        y_base = np.linspace(y_min, y_max, base_grid_size + 1)
        
        for xi in x_base:
            ax.axvline(xi, color=self.grid_color, linewidth=0.5, alpha=0.6)
        for yi in y_base:
            ax.axhline(yi, color=self.grid_color, linewidth=0.5, alpha=0.6)
        
        # Draw refined regions
        for i, (rx_min, rx_max, ry_min, ry_max) in enumerate(refined_regions):
            # Create finer grid in refined region
            n_refined = base_grid_size * 2  # Double resolution
            x_refined = np.linspace(rx_min, rx_max, n_refined + 1)
            y_refined = np.linspace(ry_min, ry_max, n_refined + 1)
            
            # Draw refined grid lines
            for xi in x_refined:
                if rx_min <= xi <= rx_max:
                    ax.axvline(xi, ymin=(ry_min - y_min)/(y_max - y_min), 
                              ymax=(ry_max - y_min)/(y_max - y_min),
                              color=self.level_colors[i % len(self.level_colors)], 
                              linewidth=1.0, alpha=0.8)
            
            for yi in y_refined:
                if ry_min <= yi <= ry_max:
                    ax.axhline(yi, xmin=(rx_min - x_min)/(x_max - x_min), 
                              xmax=(rx_max - x_min)/(x_max - x_min),
                              color=self.level_colors[i % len(self.level_colors)], 
                              linewidth=1.0, alpha=0.8)
            
            # Highlight refined region boundary
            rect = patches.Rectangle((rx_min, ry_min), rx_max - rx_min, ry_max - ry_min,
                                   linewidth=3, edgecolor=self.level_colors[i % len(self.level_colors)],
                                   facecolor='none', linestyle='--', alpha=0.8)
            ax.add_patch(rect)
            
            # Add region label
            ax.text((rx_min + rx_max)/2, (ry_min + ry_max)/2, f'R{i+1}',
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add legend for different refinement levels
        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], color=self.grid_color, linewidth=2, 
                                        label=f'Base Grid ({base_grid_size}×{base_grid_size})'))
        for i in range(len(refined_regions)):
            legend_elements.append(plt.Line2D([0], [0], 
                                            color=self.level_colors[i % len(self.level_colors)], 
                                            linewidth=2, label=f'Refined Region {i+1}'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig
    
    def plot_grid_quality_metrics(self, grid_sizes: List[int], metrics: Dict[str, List[float]],
                                 title: str = "Grid Quality Metrics", save_name: Optional[str] = None):
        """
        Plot various grid quality metrics across different grid levels.
        
        Args:
            grid_sizes: List of grid sizes
            metrics: Dictionary with metric names as keys and lists of values
            title: Plot title
            save_name: Optional filename to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        h_values = [1.0 / (n - 1) for n in grid_sizes]
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            if 'ratio' in metric_name.lower() or 'efficiency' in metric_name.lower():
                # Linear scale for ratios and efficiency
                ax.plot(grid_sizes, values, 'o-', color=self.level_colors[0], 
                       linewidth=2, markersize=8)
                ax.set_xlabel('Grid Size (N)')
            else:
                # Log scale for errors and other metrics
                ax.loglog(h_values, values, 'o-', color=self.level_colors[0], 
                         linewidth=2, markersize=8)
                ax.set_xlabel('Grid Spacing (h)')
            
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, format='png', dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_name.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        
        return fig


# Convenience functions for quick plotting
def plot_grid_hierarchy(grid_sizes: List[int], **kwargs):
    """
    Quick function to plot multigrid hierarchy.
    
    Args:
        grid_sizes: List of grid sizes for each level
        **kwargs: Additional arguments passed to GridVisualizer.plot_grid_hierarchy
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    visualizer = GridVisualizer()
    return visualizer.plot_grid_hierarchy(grid_sizes, **kwargs)


def plot_grid_convergence(grid_sizes: List[int], errors: List[float], **kwargs):
    """
    Quick function to plot grid convergence study.
    
    Args:
        grid_sizes: List of grid sizes
        errors: Corresponding error values
        **kwargs: Additional arguments passed to GridVisualizer.plot_grid_convergence_study
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    visualizer = GridVisualizer()
    return visualizer.plot_grid_convergence_study(grid_sizes, errors, **kwargs)


def plot_adaptive_refinement(refined_regions: List[Tuple[float, float, float, float]], **kwargs):
    """
    Quick function to plot adaptive mesh refinement.
    
    Args:
        refined_regions: List of refined regions as (x_min, x_max, y_min, y_max)
        **kwargs: Additional arguments passed to GridVisualizer.plot_adaptive_grid
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    visualizer = GridVisualizer()
    return visualizer.plot_adaptive_grid(refined_regions, **kwargs)