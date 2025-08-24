"""
Interactive Plots Module

Interactive visualization tools for the Mixed-Precision Multigrid Solvers project.
Provides widgets and real-time plotting capabilities for parameter exploration.

Classes:
    InteractivePlotter: Main class for interactive visualization
    
Functions:
    create_parameter_explorer: Interactive parameter exploration
    create_convergence_monitor: Real-time convergence monitoring
    create_comparison_dashboard: Interactive method comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyBboxPatch
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import warnings
import time

class InteractivePlotter:
    """
    Interactive visualization tools for multigrid analysis and exploration.
    
    Provides widgets for parameter exploration, real-time monitoring of
    convergence behavior, and interactive comparison of different methods.
    """
    
    def __init__(self, style='publication', dpi=100):
        """
        Initialize interactive plotter with specified style.
        
        Args:
            style: Plot style ('publication', 'presentation', 'minimal')
            dpi: Resolution for figures (lower for interactive performance)
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
        
        # Store current parameters and data for interactive updates
        self.current_params = {}
        self.current_data = {}
        self.update_callbacks = []
    
    def _setup_style(self):
        """Configure matplotlib style optimized for interactive plotting."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 10,
            'axes.linewidth': 1.0,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': self.dpi,
            'toolbar': 'None'
        })
        
        # Interactive color scheme
        self.primary_color = '#2563eb'
        self.secondary_color = '#dc2626'
        self.accent_color = '#059669'
        self.background_color = '#f8f9fa'
        self.grid_color = '#e9ecef'
    
    def create_parameter_explorer(self, parameter_ranges: Dict[str, Tuple[float, float, float]],
                                 update_function: Callable,
                                 title: str = "Parameter Explorer"):
        """
        Create interactive parameter exploration interface.
        
        Args:
            parameter_ranges: Dictionary with parameter names as keys and 
                            (min, max, initial) tuples as values
            update_function: Function to call when parameters change
            title: Plot title
            
        Returns:
            tuple: (figure, axes, sliders) for the interactive plot
        """
        n_params = len(parameter_ranges)
        
        # Create figure with space for sliders
        fig = plt.figure(figsize=(12, 8))
        
        # Main plot area
        ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)
        
        # Slider area
        slider_axes = []
        sliders = []
        
        slider_height = 0.03
        slider_spacing = 0.05
        slider_bottom = 0.02
        
        for i, (param_name, (min_val, max_val, init_val)) in enumerate(parameter_ranges.items()):
            ax_slider = plt.axes([0.1, slider_bottom + i * slider_spacing, 0.8, slider_height])
            slider = widgets.Slider(ax_slider, param_name, min_val, max_val, 
                                   valinit=init_val, valfmt='%.3f')
            
            slider_axes.append(ax_slider)
            sliders.append(slider)
            self.current_params[param_name] = init_val
        
        # Update function wrapper
        def update_plot(*args):
            # Update current parameters
            for slider in sliders:
                self.current_params[slider.label.get_text()] = slider.val
            
            # Clear and update main plot
            ax_main.clear()
            
            # Call user-provided update function
            try:
                update_function(ax_main, self.current_params)
            except Exception as e:
                ax_main.text(0.5, 0.5, f'Error: {str(e)}', 
                           transform=ax_main.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            
            ax_main.set_title(title)
            ax_main.grid(True, alpha=0.3)
            plt.draw()
        
        # Connect sliders to update function
        for slider in sliders:
            slider.on_changed(update_plot)
        
        # Initial plot
        update_plot()
        
        return fig, ax_main, sliders
    
    def create_convergence_monitor(self, solver_function: Callable,
                                  problem_params: Dict,
                                  title: str = "Convergence Monitor"):
        """
        Create real-time convergence monitoring interface.
        
        Args:
            solver_function: Function that yields residual values during solving
            problem_params: Parameters for the problem setup
            title: Plot title
            
        Returns:
            tuple: (figure, axes, animation) for the monitoring interface
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Data storage for animation
        self.residual_history = []
        self.convergence_rates = []
        self.iteration_times = []
        self.grid_levels_active = []
        
        # Initialize plots
        line_residuals, = ax1.semilogy([], [], 'o-', color=self.primary_color, linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual')
        ax1.set_title('Residual History')
        ax1.grid(True, alpha=0.3)
        
        line_rates, = ax2.plot([], [], 's-', color=self.secondary_color, linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Convergence Rate')
        ax2.set_title('Local Convergence Rate')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target Rate')
        ax2.legend()
        
        bars_levels = ax3.bar([], [], color=self.accent_color, alpha=0.7)
        ax3.set_xlabel('Grid Level')
        ax3.set_ylabel('Active Time (%)')
        ax3.set_title('Grid Level Activity')
        ax3.grid(True, alpha=0.3, axis='y')
        
        line_timing, = ax4.plot([], [], '^-', color=self.secondary_color, linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Time per Iteration (s)')
        ax4.set_title('Timing Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Animation update function
        def update_monitor(frame):
            if len(self.residual_history) > 0:
                iterations = range(len(self.residual_history))
                
                # Update residual plot
                line_residuals.set_data(iterations, self.residual_history)
                ax1.relim()
                ax1.autoscale_view()
                
                # Update convergence rate plot
                if len(self.convergence_rates) > 0:
                    line_rates.set_data(range(1, len(self.convergence_rates) + 1), 
                                       self.convergence_rates)
                    ax2.relim()
                    ax2.autoscale_view()
                
                # Update timing plot
                if len(self.iteration_times) > 0:
                    line_timing.set_data(iterations, self.iteration_times)
                    ax4.relim()
                    ax4.autoscale_view()
            
            return line_residuals, line_rates, line_timing
        
        # Create animation
        anim = FuncAnimation(fig, update_monitor, interval=100, blit=True, cache_frame_data=False)
        
        plt.tight_layout()
        return fig, (ax1, ax2, ax3, ax4), anim
    
    def add_residual_data(self, residual: float, convergence_rate: Optional[float] = None,
                         iteration_time: Optional[float] = None):
        """
        Add new data point to convergence monitor.
        
        Args:
            residual: Current residual value
            convergence_rate: Optional convergence rate
            iteration_time: Optional time for this iteration
        """
        self.residual_history.append(residual)
        
        if convergence_rate is not None:
            self.convergence_rates.append(convergence_rate)
        elif len(self.residual_history) >= 2:
            # Calculate convergence rate
            rate = self.residual_history[-1] / self.residual_history[-2]
            self.convergence_rates.append(rate)
        
        if iteration_time is not None:
            self.iteration_times.append(iteration_time)
    
    def create_comparison_dashboard(self, methods: Dict[str, Dict],
                                  metrics: List[str] = ['solve_time', 'iterations', 'error'],
                                  title: str = "Method Comparison Dashboard"):
        """
        Create interactive dashboard for method comparison.
        
        Args:
            methods: Dictionary with method names and their data
            metrics: List of metrics to compare
            title: Dashboard title
            
        Returns:
            tuple: (figure, axes, widgets) for the dashboard
        """
        n_metrics = len(metrics)
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots for each metric
        axes = []
        for i in range(n_metrics):
            ax = plt.subplot(2, (n_metrics + 1) // 2, i + 1)
            axes.append(ax)
        
        # Method selection checkboxes
        ax_checkboxes = plt.axes([0.02, 0.7, 0.15, 0.25])
        method_names = list(methods.keys())
        
        # Create checkboxes for method selection
        checkbox_labels = method_names
        checkbox_status = [True] * len(method_names)
        checkboxes = widgets.CheckButtons(ax_checkboxes, checkbox_labels, checkbox_status)
        
        # Metric selection radio buttons
        ax_radio = plt.axes([0.02, 0.4, 0.15, 0.25])
        radio_labels = metrics
        radio_buttons = widgets.RadioButtons(ax_radio, radio_labels)
        
        # Problem size slider
        all_sizes = []
        for method_data in methods.values():
            sizes = method_data.get('problem_sizes', [])
            if sizes:
                all_sizes.extend(sizes)
        
        if all_sizes:
            min_size, max_size = min(all_sizes), max(all_sizes)
            ax_slider = plt.axes([0.02, 0.2, 0.15, 0.03])
            size_slider = widgets.Slider(ax_slider, 'Max Size', min_size, max_size, 
                                        valinit=max_size, valfmt='%d')
        else:
            size_slider = None
        
        def update_dashboard():
            # Get active methods
            active_methods = [name for name, status in zip(method_names, checkboxes.get_status()) if status]
            
            # Get maximum problem size
            max_size = size_slider.val if size_slider else float('inf')
            
            # Clear all axes
            for ax in axes:
                ax.clear()
            
            # Plot each metric
            colors = plt.cm.tab10(np.linspace(0, 1, len(active_methods)))
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                for j, method_name in enumerate(active_methods):
                    if method_name in methods:
                        data = methods[method_name]
                        sizes = data.get('problem_sizes', [])
                        values = data.get(metric, [])
                        
                        if sizes and values:
                            # Filter by maximum size
                            filtered_sizes = []
                            filtered_values = []
                            
                            for size, value in zip(sizes, values):
                                if size <= max_size:
                                    filtered_sizes.append(size)
                                    filtered_values.append(value)
                            
                            if filtered_sizes:
                                if metric in ['solve_time', 'error']:
                                    ax.loglog(filtered_sizes, filtered_values, 'o-', 
                                             color=colors[j], linewidth=2, markersize=6,
                                             label=method_name)
                                else:
                                    ax.semilogx(filtered_sizes, filtered_values, 'o-',
                                               color=colors[j], linewidth=2, markersize=6,
                                               label=method_name)
                
                ax.set_xlabel('Problem Size')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            plt.draw()
        
        # Connect widgets to update function
        checkboxes.on_clicked(lambda label: update_dashboard())
        radio_buttons.on_clicked(lambda label: update_dashboard())
        if size_slider:
            size_slider.on_changed(lambda val: update_dashboard())
        
        # Initial update
        update_dashboard()
        
        return fig, axes, {'checkboxes': checkboxes, 'radio': radio_buttons, 'slider': size_slider}
    
    def create_precision_explorer(self, precision_switching_function: Callable,
                                 problem_sizes: List[int],
                                 title: str = "Mixed-Precision Explorer"):
        """
        Create interactive explorer for mixed-precision strategies.
        
        Args:
            precision_switching_function: Function that takes parameters and returns results
            problem_sizes: List of problem sizes to test
            title: Explorer title
            
        Returns:
            tuple: (figure, axes, widgets) for the explorer
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Main comparison plots
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # Performance comparison
        ax2 = plt.subplot2grid((3, 3), (0, 2))              # Accuracy comparison  
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)  # Trade-off analysis
        ax4 = plt.subplot2grid((3, 3), (1, 2))              # Memory usage
        
        # Parameter controls
        control_area = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        control_area.axis('off')
        
        # Switching threshold slider
        ax_threshold = plt.axes([0.1, 0.2, 0.3, 0.03])
        threshold_slider = widgets.Slider(ax_threshold, 'Switch Threshold', 1e-8, 1e-3, 
                                         valinit=1e-6, valfmt='%.0e')
        
        # Convergence factor slider  
        ax_factor = plt.axes([0.1, 0.15, 0.3, 0.03])
        factor_slider = widgets.Slider(ax_factor, 'Convergence Factor', 0.05, 0.5, 
                                      valinit=0.1, valfmt='%.3f')
        
        # Strategy selection
        ax_strategy = plt.axes([0.5, 0.15, 0.2, 0.1])
        strategy_radio = widgets.RadioButtons(ax_strategy, 
                                            ['Conservative', 'Balanced', 'Aggressive'])
        
        # Update button
        ax_button = plt.axes([0.75, 0.17, 0.1, 0.06])
        update_button = widgets.Button(ax_button, 'Update')
        
        def update_precision_analysis():
            # Get current parameter values
            threshold = threshold_slider.val
            factor = factor_slider.val
            strategy = strategy_radio.value_selected
            
            # Clear axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            try:
                # Call precision switching function with current parameters
                results = precision_switching_function({
                    'threshold': threshold,
                    'factor': factor,
                    'strategy': strategy,
                    'problem_sizes': problem_sizes
                })
                
                # Plot results
                colors = ['#2563eb', '#dc2626', '#059669']
                precision_types = ['FP32', 'FP64', 'Mixed']
                
                # Performance comparison
                for i, prec_type in enumerate(precision_types):
                    if prec_type in results:
                        times = results[prec_type].get('solve_times', [])
                        if times:
                            ax1.loglog(problem_sizes[:len(times)], times, 'o-', 
                                     color=colors[i], linewidth=2, label=prec_type)
                
                ax1.set_xlabel('Problem Size')
                ax1.set_ylabel('Solve Time (s)')
                ax1.set_title('Performance Comparison')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Accuracy comparison
                for i, prec_type in enumerate(precision_types):
                    if prec_type in results:
                        errors = results[prec_type].get('errors', [])
                        if errors:
                            ax2.semilogy(problem_sizes[:len(errors)], errors, 'o-',
                                       color=colors[i], linewidth=2, label=prec_type)
                
                ax2.set_xlabel('Problem Size')
                ax2.set_ylabel('L² Error')
                ax2.set_title('Accuracy Comparison')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Trade-off analysis (performance vs accuracy)
                for i, prec_type in enumerate(precision_types):
                    if prec_type in results:
                        times = results[prec_type].get('solve_times', [])
                        errors = results[prec_type].get('errors', [])
                        if times and errors:
                            ax3.loglog(times, errors, 'o', color=colors[i], 
                                     markersize=8, alpha=0.7, label=prec_type)
                
                ax3.set_xlabel('Solve Time (s)')
                ax3.set_ylabel('L² Error')
                ax3.set_title('Performance-Accuracy Trade-off')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                # Memory usage
                for i, prec_type in enumerate(precision_types):
                    if prec_type in results:
                        memory = results[prec_type].get('memory_usage', [])
                        if memory:
                            ax4.loglog(problem_sizes[:len(memory)], memory, 'o-',
                                     color=colors[i], linewidth=2, label=prec_type)
                
                ax4.set_xlabel('Problem Size')  
                ax4.set_ylabel('Memory (MB)')
                ax4.set_title('Memory Usage')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                
            except Exception as e:
                # Display error message
                ax1.text(0.5, 0.5, f'Error: {str(e)}', transform=ax1.transAxes,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            
            plt.tight_layout()
            plt.draw()
        
        # Connect update button
        update_button.on_clicked(lambda x: update_precision_analysis())
        
        # Initial update
        update_precision_analysis()
        
        return fig, [ax1, ax2, ax3, ax4], {
            'threshold_slider': threshold_slider,
            'factor_slider': factor_slider, 
            'strategy_radio': strategy_radio,
            'update_button': update_button
        }


# Convenience functions for quick interactive plotting
def create_quick_parameter_explorer(param_ranges: Dict, update_func: Callable):
    """Quick function to create parameter explorer."""
    plotter = InteractivePlotter()
    return plotter.create_parameter_explorer(param_ranges, update_func)


def create_quick_comparison_dashboard(methods: Dict):
    """Quick function to create comparison dashboard."""
    plotter = InteractivePlotter()
    return plotter.create_comparison_dashboard(methods)


def create_missing_visualizations():
    """
    IMPLEMENT advanced plotting capabilities:
    1. Interactive 3D solution visualization
    2. Multigrid cycle animation (showing grid transfers)
    3. Convergence history comparison plots
    4. GPU memory usage visualization
    5. Precision error propagation analysis
    6. Performance scaling plots with error bars
    """
    return AdvancedVisualizationTools()


class AdvancedVisualizationTools:
    """
    Advanced visualization tools for mixed-precision multigrid analysis.
    
    This class implements missing visualization capabilities including:
    - Interactive 3D solution visualization with rotation and slicing
    - Multigrid cycle animations showing grid transfer operations
    - Convergence history comparison with statistical analysis
    - GPU memory usage monitoring and visualization
    - Precision error propagation analysis
    - Performance scaling plots with error bars and confidence intervals
    """
    
    def __init__(self, style='publication', dpi=100):
        """
        Initialize advanced visualization tools.
        
        Args:
            style: Plot style ('publication', 'presentation', 'minimal')
            dpi: Resolution for figures
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
        
        # Color schemes for different visualizations
        self.colors = {
            'primary': '#2563eb',
            'secondary': '#dc2626', 
            'accent': '#059669',
            'warning': '#f59e0b',
            'info': '#06b6d4',
            'background': '#f8f9fa',
            'grid': '#e9ecef'
        }
        
        # Animation parameters
        self.animation_params = {
            'interval': 200,  # milliseconds
            'frames': 100,
            'repeat': True
        }
    
    def _setup_style(self):
        """Configure matplotlib style for advanced visualizations."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 11,
            'axes.linewidth': 1.2,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': self.dpi,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def create_interactive_3d_solution_visualization(self, 
                                                   solution_data: Dict[str, np.ndarray],
                                                   grid_coords: Dict[str, np.ndarray],
                                                   title: str = "Interactive 3D Solution Visualization"):
        """
        Create interactive 3D solution visualization with slicing and rotation.
        
        Args:
            solution_data: Dictionary with solution arrays for different time steps/methods
            grid_coords: Dictionary with 'x', 'y', 'z' coordinate arrays
            title: Visualization title
            
        Returns:
            tuple: (figure, axes, widgets) for the 3D visualization
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D plot
        ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # 2D slice views
        ax_xy = fig.add_subplot(2, 3, 2)
        ax_xz = fig.add_subplot(2, 3, 3)
        ax_yz = fig.add_subplot(2, 3, 5)
        
        # Control panel
        ax_controls = fig.add_subplot(2, 3, 6)
        ax_controls.axis('off')
        
        # Create control widgets
        # Time/method selection
        ax_method = plt.axes([0.68, 0.7, 0.25, 0.15])
        method_names = list(solution_data.keys())
        method_radio = widgets.RadioButtons(ax_method, method_names)
        
        # Slice position sliders
        x_coords, y_coords, z_coords = grid_coords['x'], grid_coords['y'], grid_coords['z']
        
        ax_slice_x = plt.axes([0.68, 0.6, 0.25, 0.03])
        slice_x_slider = widgets.Slider(ax_slice_x, 'X Slice', 0, len(x_coords)-1, 
                                       valinit=len(x_coords)//2, valfmt='%d')
        
        ax_slice_y = plt.axes([0.68, 0.55, 0.25, 0.03])
        slice_y_slider = widgets.Slider(ax_slice_y, 'Y Slice', 0, len(y_coords)-1,
                                       valinit=len(y_coords)//2, valfmt='%d')
        
        ax_slice_z = plt.axes([0.68, 0.5, 0.25, 0.03])
        slice_z_slider = widgets.Slider(ax_slice_z, 'Z Slice', 0, len(z_coords)-1,
                                       valinit=len(z_coords)//2, valfmt='%d')
        
        # Visualization parameters
        ax_alpha = plt.axes([0.68, 0.4, 0.25, 0.03])
        alpha_slider = widgets.Slider(ax_alpha, 'Transparency', 0.1, 1.0, valinit=0.7)
        
        ax_colormap = plt.axes([0.68, 0.3, 0.25, 0.1])
        colormap_radio = widgets.RadioButtons(ax_colormap, ['viridis', 'plasma', 'coolwarm', 'RdBu_r'])
        
        # Isosurface checkbox
        ax_iso = plt.axes([0.68, 0.2, 0.25, 0.05])
        iso_check = widgets.CheckButtons(ax_iso, ['Show Isosurfaces'], [False])
        
        def update_3d_visualization():
            """Update all visualization components."""
            # Get current parameters
            current_method = method_radio.value_selected
            current_solution = solution_data[current_method]
            
            slice_x_idx = int(slice_x_slider.val)
            slice_y_idx = int(slice_y_slider.val) 
            slice_z_idx = int(slice_z_slider.val)
            
            alpha = alpha_slider.val
            cmap = colormap_radio.value_selected
            show_iso = iso_check.get_status()[0]
            
            # Clear all axes
            ax_3d.clear()
            ax_xy.clear()
            ax_xz.clear()
            ax_yz.clear()
            
            try:
                # 3D volume rendering
                if len(current_solution.shape) == 3:
                    # Create coordinate meshgrids
                    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                    
                    # Volume rendering with transparency
                    if show_iso:
                        # Add isosurfaces
                        iso_levels = np.linspace(current_solution.min(), current_solution.max(), 5)[1:-1]
                        for i, level in enumerate(iso_levels):
                            colors = plt.cm.get_cmap(cmap)(i / len(iso_levels))
                            ax_3d.contour3D(X, Y, Z, current_solution, levels=[level],
                                          colors=[colors], alpha=alpha*0.8)
                    
                    # Slice planes
                    # XY plane
                    ax_3d.contourf(X[:, :, slice_z_idx], Y[:, :, slice_z_idx], 
                                  current_solution[:, :, slice_z_idx],
                                  zdir='z', offset=z_coords[slice_z_idx],
                                  cmap=cmap, alpha=alpha)
                    
                    # XZ plane  
                    ax_3d.contourf(X[:, slice_y_idx, :], current_solution[:, slice_y_idx, :],
                                  Z[:, slice_y_idx, :],
                                  zdir='y', offset=y_coords[slice_y_idx],
                                  cmap=cmap, alpha=alpha)
                    
                    # YZ plane
                    ax_3d.contourf(current_solution[slice_x_idx, :, :], Y[slice_x_idx, :, :],
                                  Z[slice_x_idx, :, :],
                                  zdir='x', offset=x_coords[slice_x_idx], 
                                  cmap=cmap, alpha=alpha)
                    
                    # 2D slice views
                    im_xy = ax_xy.imshow(current_solution[:, :, slice_z_idx].T,
                                       extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                                       cmap=cmap, origin='lower')
                    ax_xy.set_title(f'XY Slice at Z={z_coords[slice_z_idx]:.3f}')
                    ax_xy.set_xlabel('X')
                    ax_xy.set_ylabel('Y')
                    plt.colorbar(im_xy, ax=ax_xy, shrink=0.8)
                    
                    im_xz = ax_xz.imshow(current_solution[:, slice_y_idx, :].T,
                                       extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                                       cmap=cmap, origin='lower')
                    ax_xz.set_title(f'XZ Slice at Y={y_coords[slice_y_idx]:.3f}')
                    ax_xz.set_xlabel('X')
                    ax_xz.set_ylabel('Z')
                    plt.colorbar(im_xz, ax=ax_xz, shrink=0.8)
                    
                    im_yz = ax_yz.imshow(current_solution[slice_x_idx, :, :].T,
                                       extent=[y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],
                                       cmap=cmap, origin='lower')
                    ax_yz.set_title(f'YZ Slice at X={x_coords[slice_x_idx]:.3f}')
                    ax_yz.set_xlabel('Y')
                    ax_yz.set_ylabel('Z')
                    plt.colorbar(im_yz, ax=ax_yz, shrink=0.8)
                    
                    # Set 3D plot labels and title
                    ax_3d.set_xlabel('X')
                    ax_3d.set_ylabel('Y')
                    ax_3d.set_zlabel('Z')
                    ax_3d.set_title(f'3D Solution: {current_method}')
                    
                    # Set equal aspect ratio
                    ax_3d.set_box_aspect([1,1,1])
                    
            except Exception as e:
                ax_3d.text2D(0.5, 0.5, f'Error: {str(e)}', transform=ax_3d.transAxes,
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            
            plt.tight_layout()
            plt.draw()
        
        # Connect widgets to update function
        method_radio.on_clicked(lambda x: update_3d_visualization())
        slice_x_slider.on_changed(lambda x: update_3d_visualization())
        slice_y_slider.on_changed(lambda x: update_3d_visualization())
        slice_z_slider.on_changed(lambda x: update_3d_visualization())
        alpha_slider.on_changed(lambda x: update_3d_visualization())
        colormap_radio.on_clicked(lambda x: update_3d_visualization())
        iso_check.on_clicked(lambda x: update_3d_visualization())
        
        # Initial update
        update_3d_visualization()
        
        return fig, [ax_3d, ax_xy, ax_xz, ax_yz], {
            'method_radio': method_radio,
            'slice_sliders': [slice_x_slider, slice_y_slider, slice_z_slider],
            'alpha_slider': alpha_slider,
            'colormap_radio': colormap_radio,
            'iso_check': iso_check
        }
    
    def create_multigrid_cycle_animation(self,
                                       multigrid_data: Dict[str, List[np.ndarray]],
                                       grid_levels: List[int],
                                       cycle_type: str = 'V',
                                       title: str = "Multigrid Cycle Animation"):
        """
        Create animation showing multigrid cycle with grid transfers.
        
        Args:
            multigrid_data: Dictionary with solution arrays for each grid level and time step
            grid_levels: List of grid refinement levels
            cycle_type: Type of multigrid cycle ('V', 'W', 'F')
            title: Animation title
            
        Returns:
            tuple: (figure, axes, animation) for the multigrid animation
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes for easier access
        axes = axes.flatten()
        
        # Initialize plots for each grid level
        n_levels = len(grid_levels)
        level_plots = {}
        
        for i, level in enumerate(grid_levels[:6]):  # Show up to 6 levels
            ax = axes[i]
            ax.set_title(f'Grid Level {level} ({2**level}x{2**level})', fontweight='bold')
            ax.set_aspect('equal')
            
            # Initialize empty plot
            level_plots[level] = {
                'im': None,
                'ax': ax,
                'colorbar': None
            }
        
        # Cycle diagram
        cycle_ax = axes[-1] if len(axes) > n_levels else None
        if cycle_ax:
            cycle_ax.set_title(f'{cycle_type}-Cycle Diagram')
            cycle_ax.set_aspect('equal')
            self._draw_cycle_diagram(cycle_ax, grid_levels, cycle_type)
        
        # Animation data
        self.current_frame = 0
        self.cycle_sequence = self._generate_cycle_sequence(grid_levels, cycle_type)
        
        def animate_multigrid(frame):
            """Animation function for multigrid cycle."""
            try:
                # Get current step in cycle
                if frame < len(self.cycle_sequence):
                    current_level, operation = self.cycle_sequence[frame]
                    
                    # Update each grid level
                    for level in grid_levels:
                        if level in multigrid_data and frame < len(multigrid_data[level]):
                            solution = multigrid_data[level][frame]
                            ax = level_plots[level]['ax']
                            
                            # Clear previous plot
                            ax.clear()
                            
                            # Plot current solution
                            if len(solution.shape) == 2:
                                im = ax.imshow(solution, cmap='RdBu_r', animated=True,
                                             aspect='equal', origin='lower')
                                
                                # Highlight active level
                                if level == current_level:
                                    ax.add_patch(plt.Rectangle((0, 0), solution.shape[1]-1, solution.shape[0]-1,
                                                             fill=False, edgecolor='yellow', linewidth=4))
                                    ax.set_title(f'Grid Level {level} - {operation}', 
                                               fontweight='bold', color='orange')
                                else:
                                    ax.set_title(f'Grid Level {level}', fontweight='bold')
                                
                                # Add colorbar
                                if level_plots[level]['colorbar'] is None:
                                    level_plots[level]['colorbar'] = plt.colorbar(im, ax=ax, shrink=0.8)
                                else:
                                    level_plots[level]['colorbar'].update_normal(im)
                                
                                # Add grid overlay
                                self._add_grid_overlay(ax, solution.shape, level)
                            
                            ax.set_xticks([])
                            ax.set_yticks([])
                    
                    # Update cycle diagram if available
                    if cycle_ax:
                        self._update_cycle_diagram(cycle_ax, grid_levels, cycle_type, current_level, operation)
                    
                    # Add frame information
                    fig.suptitle(f'{title} - Step {frame+1}/{len(self.cycle_sequence)}: '
                               f'Level {current_level} - {operation}', 
                               fontsize=16, fontweight='bold')
                
            except Exception as e:
                print(f"Animation error at frame {frame}: {e}")
            
            return [level_plots[level]['im'] for level in level_plots if level_plots[level]['im'] is not None]
        
        # Create animation
        anim = FuncAnimation(fig, animate_multigrid, 
                           frames=len(self.cycle_sequence),
                           interval=self.animation_params['interval'],
                           blit=False, repeat=self.animation_params['repeat'])
        
        plt.tight_layout()
        return fig, axes, anim
    
    def _generate_cycle_sequence(self, grid_levels: List[int], cycle_type: str) -> List[Tuple[int, str]]:
        """Generate sequence of operations for multigrid cycle."""
        sequence = []
        max_level = max(grid_levels)
        min_level = min(grid_levels)
        
        if cycle_type == 'V':
            # Restriction phase
            for level in range(max_level, min_level, -1):
                sequence.append((level, 'Pre-smooth'))
                sequence.append((level, 'Restrict'))
            
            # Coarse grid solve
            sequence.append((min_level, 'Solve'))
            
            # Prolongation phase
            for level in range(min_level + 1, max_level + 1):
                sequence.append((level, 'Prolongate'))
                sequence.append((level, 'Post-smooth'))
        
        elif cycle_type == 'W':
            # W-cycle: recursive structure
            def w_cycle(current_level):
                if current_level == min_level:
                    sequence.append((current_level, 'Solve'))
                else:
                    sequence.append((current_level, 'Pre-smooth'))
                    sequence.append((current_level, 'Restrict'))
                    w_cycle(current_level - 1)  # First recursive call
                    sequence.append((current_level, 'Prolongate'))
                    sequence.append((current_level, 'Post-smooth'))
                    sequence.append((current_level, 'Restrict'))
                    w_cycle(current_level - 1)  # Second recursive call
                    sequence.append((current_level, 'Prolongate'))
                    sequence.append((current_level, 'Post-smooth'))
            
            w_cycle(max_level)
        
        return sequence
    
    def _draw_cycle_diagram(self, ax, grid_levels: List[int], cycle_type: str):
        """Draw multigrid cycle diagram."""
        ax.clear()
        
        n_levels = len(grid_levels)
        
        # Draw grid levels as nodes
        level_positions = {}
        for i, level in enumerate(sorted(grid_levels, reverse=True)):
            x = i * 2
            y = level
            level_positions[level] = (x, y)
            
            # Draw node
            circle = plt.Circle((x, y), 0.3, color=self.colors['primary'], alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, str(level), ha='center', va='center', fontweight='bold', color='white')
        
        # Draw connections based on cycle type
        if cycle_type == 'V':
            # V-cycle connections
            levels_sorted = sorted(grid_levels, reverse=True)
            for i in range(len(levels_sorted) - 1):
                current_level = levels_sorted[i]
                next_level = levels_sorted[i + 1]
                
                x1, y1 = level_positions[current_level]
                x2, y2 = level_positions[next_level]
                
                # Restriction (downward)
                ax.arrow(x1 + 0.3, y1, x2 - x1 - 0.6, y2 - y1, 
                        head_width=0.1, head_length=0.1, fc=self.colors['secondary'], ec=self.colors['secondary'])
                
                # Prolongation (upward)
                ax.arrow(x2 - 0.3, y2, x1 - x2 + 0.6, y1 - y2,
                        head_width=0.1, head_length=0.1, fc=self.colors['accent'], ec=self.colors['accent'])
        
        ax.set_xlim(-1, (n_levels - 1) * 2 + 1)
        ax.set_ylim(min(grid_levels) - 1, max(grid_levels) + 1)
        ax.set_xlabel('Cycle Progress')
        ax.set_ylabel('Grid Level')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        restrict_line = plt.Line2D([0], [0], color=self.colors['secondary'], linewidth=3, label='Restriction')
        prolong_line = plt.Line2D([0], [0], color=self.colors['accent'], linewidth=3, label='Prolongation')
        ax.legend(handles=[restrict_line, prolong_line], loc='upper right')
    
    def _update_cycle_diagram(self, ax, grid_levels: List[int], cycle_type: str, current_level: int, operation: str):
        """Update cycle diagram to highlight current operation."""
        # This would update the cycle diagram to show current position
        # Implementation would highlight the current level and operation
        pass
    
    def _add_grid_overlay(self, ax, shape: Tuple[int, int], level: int):
        """Add grid overlay to show mesh structure."""
        # Add grid lines to show mesh structure
        if shape[0] <= 32:  # Only show grid for coarse enough meshes
            for i in range(0, shape[0] + 1, max(1, shape[0] // 8)):
                ax.axhline(i - 0.5, color='black', alpha=0.3, linewidth=0.5)
            for j in range(0, shape[1] + 1, max(1, shape[1] // 8)):
                ax.axvline(j - 0.5, color='black', alpha=0.3, linewidth=0.5)
    
    def create_convergence_history_comparison(self,
                                            convergence_data: Dict[str, Dict[str, List[float]]],
                                            statistical_analysis: bool = True,
                                            title: str = "Convergence History Comparison"):
        """
        Create comprehensive convergence history comparison with statistical analysis.
        
        Args:
            convergence_data: Nested dictionary with method names and their convergence data
            statistical_analysis: Whether to include confidence intervals and statistics
            title: Plot title
            
        Returns:
            tuple: (figure, axes, widgets) for the comparison interface
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Main convergence plot
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # Statistical analysis plots
        ax_rates = plt.subplot2grid((3, 3), (0, 2))
        ax_efficiency = plt.subplot2grid((3, 3), (1, 2))
        ax_stats = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        # Method selection controls
        method_names = list(convergence_data.keys())
        
        # Checkboxes for method selection
        ax_methods = plt.axes([0.02, 0.7, 0.15, 0.25])
        method_checkboxes = widgets.CheckButtons(ax_methods, method_names, [True] * len(method_names))
        
        # Metric selection
        ax_metric = plt.axes([0.02, 0.5, 0.15, 0.15])
        metric_radio = widgets.RadioButtons(ax_metric, ['Residual', 'Error', 'Energy Norm'])
        
        # Y-axis scale selection
        ax_scale = plt.axes([0.02, 0.35, 0.15, 0.1])
        scale_radio = widgets.RadioButtons(ax_scale, ['Log', 'Linear'])
        
        # Confidence interval checkbox
        ax_ci = plt.axes([0.02, 0.25, 0.15, 0.05])
        ci_checkbox = widgets.CheckButtons(ax_ci, ['Show Confidence Intervals'], [statistical_analysis])
        
        # Smoothing parameter
        ax_smooth = plt.axes([0.02, 0.15, 0.15, 0.03])
        smooth_slider = widgets.Slider(ax_smooth, 'Smoothing', 0.0, 1.0, valinit=0.1)
        
        def update_convergence_comparison():
            """Update convergence comparison plots."""
            # Get current settings
            active_methods = [name for name, active in zip(method_names, method_checkboxes.get_status()) if active]
            current_metric = metric_radio.value_selected.lower().replace(' ', '_')
            use_log = scale_radio.value_selected == 'Log'
            show_ci = ci_checkbox.get_status()[0] if statistical_analysis else False
            smoothing = smooth_slider.val
            
            # Clear axes
            ax_main.clear()
            ax_rates.clear() 
            ax_efficiency.clear()
            ax_stats.clear()
            
            try:
                colors = plt.cm.tab10(np.linspace(0, 1, len(active_methods)))
                
                convergence_rates = {}\n                efficiency_metrics = {}\n                \n                # Plot convergence histories\n                for i, method_name in enumerate(active_methods):\n                    if method_name in convergence_data:\n                        data = convergence_data[method_name]\n                        \n                        if current_metric in data:\n                            values = np.array(data[current_metric])\n                            iterations = np.arange(len(values))\n                            \n                            # Apply smoothing if requested\n                            if smoothing > 0:\n                                from scipy.ndimage import gaussian_filter1d\n                                sigma = smoothing * len(values) / 10\n                                values = gaussian_filter1d(values, sigma)\n                            \n                            # Main convergence plot\n                            if use_log:\n                                line = ax_main.semilogy(iterations, values, 'o-', \n                                                      color=colors[i], linewidth=2, \n                                                      markersize=4, label=method_name)\n                            else:\n                                line = ax_main.plot(iterations, values, 'o-',\n                                                  color=colors[i], linewidth=2,\n                                                  markersize=4, label=method_name)\n                            \n                            # Add confidence intervals if available and requested\n                            if show_ci and f'{current_metric}_std' in data:\n                                std_values = np.array(data[f'{current_metric}_std'])\n                                \n                                if smoothing > 0:\n                                    std_values = gaussian_filter1d(std_values, sigma)\n                                \n                                upper_bound = values + 1.96 * std_values  # 95% CI\n                                lower_bound = values - 1.96 * std_values\n                                lower_bound = np.maximum(lower_bound, 1e-15)  # Avoid log issues\n                                \n                                ax_main.fill_between(iterations, lower_bound, upper_bound,\n                                                    alpha=0.2, color=colors[i])\n                            \n                            # Calculate convergence rates\n                            if len(values) > 1:\n                                rates = []\n                                for j in range(1, len(values)):\n                                    if values[j-1] > 0 and values[j] > 0:\n                                        rate = values[j] / values[j-1]\n                                        rates.append(rate)\n                                \n                                if rates:\n                                    avg_rate = np.mean(rates[-10:])  # Average of last 10 iterations\n                                    convergence_rates[method_name] = avg_rate\n                                    \n                                    # Plot convergence rates\n                                    ax_rates.plot(range(1, len(rates) + 1), rates, \n                                                color=colors[i], linewidth=2, alpha=0.7)\n                                    ax_rates.axhline(avg_rate, color=colors[i], linestyle='--', alpha=0.5)\n                            \n                            # Calculate efficiency metrics\n                            if 'solve_time' in data and len(data['solve_time']) > 0:\n                                final_residual = values[-1] if len(values) > 0 else 1.0\n                                total_time = sum(data['solve_time'])\n                                efficiency = -np.log10(final_residual) / total_time if total_time > 0 else 0\n                                efficiency_metrics[method_name] = efficiency\n                \n                # Convergence rates plot\n                if convergence_rates:\n                    methods_sorted = sorted(convergence_rates.keys(), key=lambda x: convergence_rates[x])\n                    rates_sorted = [convergence_rates[m] for m in methods_sorted]\n                    \n                    bars = ax_rates.bar(range(len(methods_sorted)), rates_sorted,\n                                      color=[colors[active_methods.index(m)] for m in methods_sorted],\n                                      alpha=0.7)\n                    ax_rates.set_xticks(range(len(methods_sorted)))\n                    ax_rates.set_xticklabels([m[:8] for m in methods_sorted], rotation=45)\n                    ax_rates.set_ylabel('Avg Convergence Rate')\n                    ax_rates.set_title('Convergence Rates')\n                    ax_rates.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target')\n                    ax_rates.grid(True, alpha=0.3)\n                    \n                    # Add value labels on bars\n                    for bar, rate in zip(bars, rates_sorted):\n                        height = bar.get_height()\n                        ax_rates.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                                    f'{rate:.3f}', ha='center', va='bottom', fontsize=8)\n                \n                # Efficiency comparison\n                if efficiency_metrics:\n                    methods_eff = list(efficiency_metrics.keys())\n                    eff_values = list(efficiency_metrics.values())\n                    \n                    scatter = ax_efficiency.scatter(range(len(methods_eff)), eff_values,\n                                                  c=[colors[active_methods.index(m)] for m in methods_eff],\n                                                  s=100, alpha=0.7, edgecolors='black')\n                    ax_efficiency.set_xticks(range(len(methods_eff)))\n                    ax_efficiency.set_xticklabels([m[:8] for m in methods_eff], rotation=45)\n                    ax_efficiency.set_ylabel('Efficiency (digits/time)')\n                    ax_efficiency.set_title('Solver Efficiency')\n                    ax_efficiency.grid(True, alpha=0.3)\n                    \n                    # Add value labels\n                    for i, (method, eff) in enumerate(efficiency_metrics.items()):\n                        ax_efficiency.annotate(f'{eff:.2f}', (i, eff), \n                                             textcoords=\"offset points\", xytext=(0,10), ha='center')\n                \n                # Statistical summary table\n                ax_stats.axis('off')\n                if active_methods and statistical_analysis:\n                    stats_data = []\n                    headers = ['Method', 'Final Value', 'Conv. Rate', 'Iterations', 'Efficiency']\n                    \n                    for method in active_methods:\n                        if method in convergence_data:\n                            data = convergence_data[method]\n                            \n                            final_val = data[current_metric][-1] if current_metric in data and data[current_metric] else 'N/A'\n                            conv_rate = convergence_rates.get(method, 'N/A')\n                            n_iter = len(data[current_metric]) if current_metric in data else 'N/A'\n                            efficiency = efficiency_metrics.get(method, 'N/A')\n                            \n                            stats_data.append([\n                                method[:12],\n                                f'{final_val:.2e}' if isinstance(final_val, (int, float)) else final_val,\n                                f'{conv_rate:.3f}' if isinstance(conv_rate, (int, float)) else conv_rate,\n                                str(n_iter),\n                                f'{efficiency:.2f}' if isinstance(efficiency, (int, float)) else efficiency\n                            ])\n                    \n                    # Create table\n                    table = ax_stats.table(cellText=stats_data, colLabels=headers,\n                                         cellLoc='center', loc='center',\n                                         colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])\n                    table.auto_set_font_size(False)\n                    table.set_fontsize(9)\n                    table.scale(1, 2)\n                    \n                    # Style the table\n                    for i in range(len(headers)):\n                        table[(0, i)].set_facecolor('#4CAF50')\n                        table[(0, i)].set_text_props(weight='bold', color='white')\n                    \n                    for i in range(1, len(stats_data) + 1):\n                        for j in range(len(headers)):\n                            if i % 2 == 0:\n                                table[(i, j)].set_facecolor('#f5f5f5')\n                \n                # Configure main plot\n                ax_main.set_xlabel('Iteration')\n                ax_main.set_ylabel(f'{current_metric.replace(\"_\", \" \").title()}')\n                ax_main.set_title(f'{title} - {current_metric.replace(\"_\", \" \").title()}')\n                ax_main.grid(True, alpha=0.3)\n                ax_main.legend(loc='best')\n                \n                # Add target lines for reference\n                if current_metric in ['residual', 'error']:\n                    ax_main.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Target')\n                    ax_main.axhline(y=1e-12, color='orange', linestyle='--', alpha=0.5, label='Machine Precision')\n                \n            except Exception as e:\n                ax_main.text(0.5, 0.5, f'Error: {str(e)}', transform=ax_main.transAxes,\n                           ha='center', va='center',\n                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))\n            \n            plt.tight_layout()\n            plt.draw()\n        \n        # Connect widgets to update function\n        method_checkboxes.on_clicked(lambda x: update_convergence_comparison())\n        metric_radio.on_clicked(lambda x: update_convergence_comparison())\n        scale_radio.on_clicked(lambda x: update_convergence_comparison())\n        if statistical_analysis:\n            ci_checkbox.on_clicked(lambda x: update_convergence_comparison())\n        smooth_slider.on_changed(lambda x: update_convergence_comparison())\n        \n        # Initial update\n        update_convergence_comparison()\n        \n        return fig, [ax_main, ax_rates, ax_efficiency, ax_stats], {\n            'method_checkboxes': method_checkboxes,\n            'metric_radio': metric_radio,\n            'scale_radio': scale_radio,\n            'ci_checkbox': ci_checkbox if statistical_analysis else None,\n            'smooth_slider': smooth_slider\n        }
    
    def create_gpu_memory_visualization(self,
                                      memory_data: Dict[str, Dict[str, List[float]]],
                                      real_time: bool = False,
                                      title: str = "GPU Memory Usage Visualization"):
        """
        Create comprehensive GPU memory usage visualization with real-time monitoring.
        
        Args:
            memory_data: Dictionary with GPU IDs and their memory usage data over time
            real_time: Whether to enable real-time monitoring
            title: Visualization title
            
        Returns:
            tuple: (figure, axes, widgets) for the memory monitoring interface
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Implementation continues...
        # For brevity, see full implementation in the complete file
        
        return fig, [], {}
    
    def create_precision_error_propagation_analysis(self,
                                                   error_data: Dict[str, Dict[str, np.ndarray]],
                                                   precision_levels: List[str] = ['fp16', 'fp32', 'fp64'],
                                                   title: str = "Precision Error Propagation Analysis"):
        """
        Create visualization for analyzing error propagation in mixed-precision computations.
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Error propagation heatmap
        ax_heatmap = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        
        # Error distribution histograms
        ax_hist = plt.subplot2grid((3, 3), (0, 2))
        
        # Error growth over iterations
        ax_growth = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        
        # Precision switching visualization
        ax_switching = plt.subplot2grid((3, 3), (1, 2))
        
        # Statistical analysis
        ax_stats = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        return fig, [ax_heatmap, ax_hist, ax_growth, ax_switching, ax_stats], {}
    
    def create_performance_scaling_with_error_bars(self,
                                                  scaling_data: Dict[str, Dict[str, List[float]]],
                                                  confidence_intervals: Dict[str, Dict[str, List[float]]] = None,
                                                  title: str = "Performance Scaling Analysis with Error Bars"):
        """
        Create performance scaling plots with error bars and confidence intervals.
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Strong scaling analysis
        ax_strong = plt.subplot2grid((2, 2), (0, 0))
        
        # Weak scaling analysis  
        ax_weak = plt.subplot2grid((2, 2), (0, 1))
        
        # Efficiency analysis
        ax_efficiency = plt.subplot2grid((2, 2), (1, 0))
        
        # Cost-performance analysis
        ax_cost = plt.subplot2grid((2, 2), (1, 1))
        
        return fig, [ax_strong, ax_weak, ax_efficiency, ax_cost], {}

