"""
Visualization Module for Mixed-Precision Multigrid Solvers

This module provides comprehensive visualization tools for analyzing and presenting
results from mixed-precision multigrid PDE solvers. It includes:

- Solution plotting (2D/3D surfaces, contours, comparisons)
- Convergence analysis (residual histories, grid convergence studies)
- Performance visualization (CPU/GPU comparisons, scaling analysis)
- Grid hierarchy visualization (multigrid levels, adaptive refinement)
- Statistical analysis (error decomposition, confidence intervals)
- Interactive exploration tools (parameter sweeps, real-time monitoring)

All visualization tools are designed for publication-quality output with
professional styling and mathematical notation support.
"""

# Core visualization classes (import classes that definitely exist)
try:
    from .solution_plots import SolutionVisualizer
except ImportError:
    SolutionVisualizer = None

try:
    from .convergence_plots import ConvergencePlotter
except ImportError:
    ConvergencePlotter = None

try:
    from .performance_plots import PerformancePlotter
except ImportError:
    PerformancePlotter = None

try:
    from .grid_visualization import GridVisualizer
except ImportError:
    GridVisualizer = None

try:
    from .analysis_plots import AnalysisVisualizer
except ImportError:
    AnalysisVisualizer = None

try:
    from .interactive_plots import InteractivePlotter
except ImportError:
    InteractivePlotter = None

# Advanced visualization tools (our new implementation)
from .advanced_visualizations import (
    AdvancedVisualizationTools,
    create_missing_visualizations
)

# Real-time monitoring dashboard
try:
    from .realtime_dashboard import (
        solver_dashboard,
        SolverDashboard,
        create_monitoring_widgets
    )
except ImportError:
    solver_dashboard = None
    SolverDashboard = None
    create_monitoring_widgets = None

# Convenience functions
try:
    from .interactive_plots import (
        create_quick_parameter_explorer,
        create_quick_comparison_dashboard
    )
except ImportError:
    create_quick_parameter_explorer = None
    create_quick_comparison_dashboard = None

# Define what's available
__all__ = [
    # Core classes
    "AdvancedVisualizationTools",
    "create_missing_visualizations",
    "solver_dashboard"
]

# Add available classes to __all__
for name, obj in [
    ("SolutionVisualizer", SolutionVisualizer),
    ("ConvergencePlotter", ConvergencePlotter), 
    ("PerformancePlotter", PerformancePlotter),
    ("GridVisualizer", GridVisualizer),
    ("AnalysisVisualizer", AnalysisVisualizer),
    ("InteractivePlotter", InteractivePlotter),
    ("SolverDashboard", SolverDashboard),
    ("create_monitoring_widgets", create_monitoring_widgets),
    ("create_quick_parameter_explorer", create_quick_parameter_explorer),
    ("create_quick_comparison_dashboard", create_quick_comparison_dashboard)
]:
    if obj is not None:
        __all__.append(name)