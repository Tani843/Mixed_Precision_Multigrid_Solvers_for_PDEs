"""Solution visualization tools for 2D and 3D plotting."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os

logger = logging.getLogger(__name__)

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False,  # Set to True if LaTeX is available
    'mathtext.fontset': 'stix'
})

class SolutionVisualizer:
    """
    Professional-quality solution visualization for 2D and 3D problems.
    
    Provides publication-ready plots with customizable styling,
    error analysis, and comparison capabilities.
    """
    
    def __init__(self, output_dir: str = "plots", style: str = "publication"):
        """
        Initialize solution visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Plotting style ('publication', 'presentation', 'web')
        """
        self.output_dir = output_dir
        self.style = style
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define color schemes
        self.colormap_solution = 'RdBu_r'  # Red-Blue reversed for solutions
        self.colormap_error = 'plasma'      # Plasma for errors
        self.colormap_residual = 'viridis'  # Viridis for residuals
        
        # Set style-specific parameters
        self._configure_style()
        
        logger.info(f"Solution visualizer initialized with {style} style")
    
    def _configure_style(self):
        """Configure matplotlib parameters for different styles."""
        if self.style == "publication":
            plt.rcParams.update({
                'font.size': 12,
                'figure.figsize': (8, 6),
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11
            })
        elif self.style == "presentation":
            plt.rcParams.update({
                'font.size': 14,
                'figure.figsize': (10, 7.5),
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 14
            })
        elif self.style == "web":
            plt.rcParams.update({
                'font.size': 11,
                'figure.figsize': (9, 6),
                'axes.titlesize': 13,
                'axes.labelsize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 10
            })
    
    def plot_solution_2d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        solution: np.ndarray,
        title: str = "Solution",
        xlabel: str = "x",
        ylabel: str = "y",
        colorbar_label: str = "u(x,y)",
        save_name: Optional[str] = None,
        analytical_solution: Optional[np.ndarray] = None,
        plot_type: str = "contourf"  # 'contourf', 'imshow', 'surface'
    ) -> plt.Figure:
        """
        Create 2D solution plot with professional styling.
        
        Args:
            X, Y: Coordinate meshgrids
            solution: Solution values
            title: Plot title
            xlabel, ylabel: Axis labels
            colorbar_label: Colorbar label
            save_name: Filename to save (without extension)
            analytical_solution: Optional analytical solution for comparison
            plot_type: Type of 2D plot
            
        Returns:
            Matplotlib figure object
        """
        if plot_type == "contourf":
            return self._plot_contourf(
                X, Y, solution, title, xlabel, ylabel, 
                colorbar_label, save_name, analytical_solution
            )
        elif plot_type == "imshow":
            return self._plot_imshow(
                solution, title, xlabel, ylabel,
                colorbar_label, save_name, X, Y
            )
        elif plot_type == "surface":
            return self._plot_surface_2d(
                X, Y, solution, title, xlabel, ylabel,
                colorbar_label, save_name
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    def _plot_contourf(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        solution: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        colorbar_label: str,
        save_name: Optional[str],
        analytical_solution: Optional[np.ndarray]
    ) -> plt.Figure:
        """Create filled contour plot."""
        if analytical_solution is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Numerical solution
            levels = 20
            cs1 = ax1.contourf(X, Y, solution, levels=levels, cmap=self.colormap_solution)
            ax1.set_title("Numerical Solution")
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            ax1.set_aspect('equal')
            plt.colorbar(cs1, ax=ax1, label=colorbar_label)
            
            # Analytical solution  
            cs2 = ax2.contourf(X, Y, analytical_solution, levels=levels, cmap=self.colormap_solution)
            ax2.set_title("Analytical Solution")
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabel)
            ax2.set_aspect('equal')
            plt.colorbar(cs2, ax=ax2, label=colorbar_label)
            
            # Error
            error = np.abs(solution - analytical_solution)
            cs3 = ax3.contourf(X, Y, error, levels=levels, cmap=self.colormap_error)
            ax3.set_title("Absolute Error")
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel(ylabel)
            ax3.set_aspect('equal')
            plt.colorbar(cs3, ax=ax3, label="|Error|")
            
            fig.suptitle(title, fontsize=16)
            
        else:
            fig, ax = plt.subplots(figsize=plt.rcParams['figure.figsize'])
            
            levels = 20
            cs = ax.contourf(X, Y, solution, levels=levels, cmap=self.colormap_solution)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect('equal')
            
            cbar = plt.colorbar(cs, ax=ax, label=colorbar_label)
            cbar.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'])
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def _plot_imshow(
        self,
        solution: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        colorbar_label: str,
        save_name: Optional[str],
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """Create image plot."""
        fig, ax = plt.subplots(figsize=plt.rcParams['figure.figsize'])
        
        # Flip solution for correct orientation (origin at bottom-left)
        im = ax.imshow(solution[::-1], cmap=self.colormap_solution, 
                      aspect='equal', interpolation='bilinear')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Set proper tick locations if coordinates provided
        if X is not None and Y is not None:
            ny, nx = solution.shape
            x_ticks = np.linspace(0, nx-1, 5)
            y_ticks = np.linspace(0, ny-1, 5)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([f"{X[0, int(i)]:.2f}" for i in x_ticks])
            ax.set_yticklabels([f"{Y[int(i), 0]:.2f}" for i in y_ticks[::-1]])
        
        cbar = plt.colorbar(im, ax=ax, label=colorbar_label)
        cbar.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'])
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def _plot_surface_2d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        solution: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        colorbar_label: str,
        save_name: Optional[str]
    ) -> plt.Figure:
        """Create 2D surface plot (3D visualization)."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, solution, cmap=self.colormap_solution,
                              alpha=0.9, linewidth=0, antialiased=True)
        
        # Add contour projection
        ax.contour(X, Y, solution, zdir='z', offset=np.min(solution), 
                  cmap=self.colormap_solution, alpha=0.5, linewidths=1)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(colorbar_label)
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.6, label=colorbar_label)
        cbar.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'])
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_solution_comparison(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        solutions: Dict[str, np.ndarray],
        title: str = "Solution Comparison",
        save_name: Optional[str] = None,
        layout: str = "grid"  # 'grid', 'horizontal', 'vertical'
    ) -> plt.Figure:
        """
        Compare multiple solutions side by side.
        
        Args:
            X, Y: Coordinate meshgrids
            solutions: Dictionary of {label: solution_array}
            title: Overall title
            save_name: Filename to save
            layout: Subplot layout
            
        Returns:
            Matplotlib figure object
        """
        n_solutions = len(solutions)
        
        if layout == "horizontal":
            fig, axes = plt.subplots(1, n_solutions, figsize=(5*n_solutions, 5))
        elif layout == "vertical":
            fig, axes = plt.subplots(n_solutions, 1, figsize=(8, 4*n_solutions))
        else:  # grid
            cols = int(np.ceil(np.sqrt(n_solutions)))
            rows = int(np.ceil(n_solutions / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if n_solutions == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        # Find global min/max for consistent color scaling
        all_values = np.concatenate([sol.flatten() for sol in solutions.values()])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        for i, (label, solution) in enumerate(solutions.items()):
            ax = axes[i] if n_solutions > 1 else axes[0]
            
            cs = ax.contourf(X, Y, solution, levels=20, cmap=self.colormap_solution,
                           vmin=vmin, vmax=vmax)
            ax.set_title(label)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            
            cbar = plt.colorbar(cs, ax=ax)
            cbar.ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for i in range(n_solutions, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_solution_evolution(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        time_steps: np.ndarray,
        solution_history: List[np.ndarray],
        title: str = "Solution Evolution",
        save_name: Optional[str] = None,
        n_snapshots: int = 6
    ) -> plt.Figure:
        """
        Plot solution evolution over time.
        
        Args:
            X, Y: Coordinate meshgrids
            time_steps: Array of time values
            solution_history: List of solution arrays
            title: Plot title
            save_name: Filename to save
            n_snapshots: Number of time snapshots to show
            
        Returns:
            Matplotlib figure object
        """
        # Select time indices for snapshots
        total_steps = len(solution_history)
        if total_steps <= n_snapshots:
            indices = list(range(total_steps))
        else:
            indices = np.linspace(0, total_steps-1, n_snapshots, dtype=int)
        
        # Create subplot layout
        cols = 3
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if len(indices) == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        # Find global min/max for consistent color scaling
        all_values = np.concatenate([sol.flatten() for sol in solution_history])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        for i, idx in enumerate(indices):
            ax = axes[i] if len(indices) > 1 else axes[0]
            
            cs = ax.contourf(X, Y, solution_history[idx], levels=20, 
                           cmap=self.colormap_solution, vmin=vmin, vmax=vmax)
            ax.set_title(f"t = {time_steps[idx]:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            
            if i % cols == cols - 1 or i == len(indices) - 1:  # Last in row
                cbar = plt.colorbar(cs, ax=ax)
                cbar.ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_cross_sections(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        solution: np.ndarray,
        analytical_solution: Optional[np.ndarray] = None,
        cross_sections: List[Dict[str, Any]] = None,
        title: str = "Cross-Sections",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cross-sectional views of the solution.
        
        Args:
            X, Y: Coordinate meshgrids
            solution: Numerical solution
            analytical_solution: Optional analytical solution
            cross_sections: List of cross-section specifications
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure object
        """
        if cross_sections is None:
            # Default cross-sections
            ny, nx = solution.shape
            cross_sections = [
                {'type': 'horizontal', 'index': ny//4, 'label': 'y = 0.25'},
                {'type': 'horizontal', 'index': ny//2, 'label': 'y = 0.50'},
                {'type': 'horizontal', 'index': 3*ny//4, 'label': 'y = 0.75'},
                {'type': 'vertical', 'index': nx//2, 'label': 'x = 0.50'}
            ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, cs in enumerate(cross_sections):
            ax = axes[i]
            
            if cs['type'] == 'horizontal':
                idx = cs['index']
                x_coords = X[idx, :]
                num_values = solution[idx, :]
                
                ax.plot(x_coords, num_values, 'b-', linewidth=2, 
                       label='Numerical', marker='o', markersize=4)
                
                if analytical_solution is not None:
                    anal_values = analytical_solution[idx, :]
                    ax.plot(x_coords, anal_values, 'r--', linewidth=2, 
                           label='Analytical', alpha=0.8)
                
                ax.set_xlabel("x")
                ax.set_ylabel("u")
                ax.set_title(cs['label'])
                
            elif cs['type'] == 'vertical':
                idx = cs['index']
                y_coords = Y[:, idx]
                num_values = solution[:, idx]
                
                ax.plot(y_coords, num_values, 'b-', linewidth=2,
                       label='Numerical', marker='o', markersize=4)
                
                if analytical_solution is not None:
                    anal_values = analytical_solution[:, idx]
                    ax.plot(y_coords, anal_values, 'r--', linewidth=2,
                           label='Analytical', alpha=0.8)
                
                ax.set_xlabel("y")
                ax.set_ylabel("u")
                ax.set_title(cs['label'])
            
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, save_name: str):
        """Save figure in multiple formats."""
        # Save as PNG (high quality)
        png_path = os.path.join(self.output_dir, f"{save_name}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Save as SVG (vector format)
        svg_path = os.path.join(self.output_dir, f"{save_name}.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        logger.info(f"Saved plot: {png_path} and {svg_path}")


def plot_solution_2d(
    X: np.ndarray,
    Y: np.ndarray,
    solution: np.ndarray,
    title: str = "2D Solution",
    save_name: Optional[str] = None,
    analytical_solution: Optional[np.ndarray] = None,
    output_dir: str = "plots"
) -> plt.Figure:
    """
    Convenience function for 2D solution plotting.
    
    Args:
        X, Y: Coordinate meshgrids
        solution: Solution values
        title: Plot title
        save_name: Filename to save
        analytical_solution: Optional analytical solution
        output_dir: Output directory
        
    Returns:
        Matplotlib figure object
    """
    visualizer = SolutionVisualizer(output_dir=output_dir)
    return visualizer.plot_solution_2d(
        X, Y, solution, title=title, save_name=save_name,
        analytical_solution=analytical_solution
    )


def plot_solution_3d(
    X: np.ndarray,
    Y: np.ndarray,
    solution: np.ndarray,
    title: str = "3D Solution Surface",
    save_name: Optional[str] = None,
    output_dir: str = "plots"
) -> plt.Figure:
    """
    Convenience function for 3D surface plotting.
    
    Args:
        X, Y: Coordinate meshgrids
        solution: Solution values
        title: Plot title
        save_name: Filename to save
        output_dir: Output directory
        
    Returns:
        Matplotlib figure object
    """
    visualizer = SolutionVisualizer(output_dir=output_dir)
    return visualizer._plot_surface_2d(
        X, Y, solution, title, "x", "y", "u(x,y)", save_name
    )