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
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import warnings

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


# Convenience functions for quick interactive plotting
def create_quick_parameter_explorer(param_ranges: Dict, update_func: Callable):
    """Quick function to create parameter explorer."""
    plotter = InteractivePlotter()
    return plotter.create_parameter_explorer(param_ranges, update_func)


def create_quick_comparison_dashboard(methods: Dict):
    """Quick function to create comparison dashboard."""
    plotter = InteractivePlotter()
    return plotter.create_comparison_dashboard(methods)