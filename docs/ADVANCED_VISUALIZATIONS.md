# Advanced Visualization Tools

This document describes the comprehensive set of advanced visualization tools implemented for the Mixed-Precision Multigrid Solvers project. These tools provide interactive, publication-quality visualizations for analyzing solver performance, convergence behavior, and computational efficiency.

## Overview

The advanced visualization module (`src/multigrid/visualization/advanced_visualizations.py`) implements six key visualization capabilities that were previously missing:

1. **Interactive 3D Solution Visualization**
2. **Multigrid Cycle Animation** 
3. **Convergence History Comparison with Statistical Analysis**
4. **GPU Memory Usage Visualization**
5. **Precision Error Propagation Analysis**
6. **Performance Scaling Plots with Error Bars**

## Installation and Dependencies

The advanced visualization tools require the following Python packages:

```python
# Core dependencies (automatically installed with the project)
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.9.0

# Optional for enhanced visualizations
plotly>=5.0.0  # For web-based interactive plots
seaborn>=0.11.0  # For statistical visualizations
```

## Usage

### Basic Usage

```python
from multigrid.visualization.advanced_visualizations import create_missing_visualizations

# Create visualization tools instance
viz_tools = create_missing_visualizations()

# Use any of the advanced visualization methods
fig, axes, widgets = viz_tools.create_interactive_3d_solution_visualization(
    solution_data, grid_coords
)
```

### Running the Demonstration

```bash
# From the project root directory
cd examples/
python advanced_visualizations_demo.py
```

## Visualization Components

### 1. Interactive 3D Solution Visualization

**Purpose**: Visualize 3D PDE solutions with interactive slicing and parameter control.

**Features**:
- Real-time slice positioning through 3D volume
- Multiple 2D slice views (XY, XZ, YZ planes)
- Isosurface rendering for level sets
- Multiple colormap options
- Transparency control
- Method/time step comparison

**Usage Example**:
```python
# Prepare 3D solution data
solution_data = {
    'Method_1': np.array(...),  # 3D numpy array
    'Method_2': np.array(...),  # 3D numpy array
}
grid_coords = {
    'x': np.linspace(0, 1, nx),
    'y': np.linspace(0, 1, ny), 
    'z': np.linspace(0, 1, nz)
}

# Create interactive visualization
fig, axes, widgets = viz_tools.create_interactive_3d_solution_visualization(
    solution_data, grid_coords,
    title="3D Poisson Solution Comparison"
)
```

**Interactive Controls**:
- **Method Selection**: Radio buttons to switch between different solution methods
- **Slice Position**: Sliders to adjust X, Y, Z slice positions
- **Transparency**: Control volume rendering transparency
- **Colormap**: Choose from scientific colormaps (viridis, plasma, coolwarm, RdBu_r)
- **Isosurfaces**: Toggle isosurface rendering on/off

### 2. Multigrid Cycle Animation

**Purpose**: Animate multigrid cycles showing grid transfer operations and solver progression.

**Features**:
- V-cycle, W-cycle, and F-cycle animations
- Grid level activity visualization
- Operation sequence display (restriction, prolongation, smoothing)
- Multi-level solution evolution
- Cycle diagram with current position highlighting

**Usage Example**:
```python
# Prepare multigrid data for each level and time step
multigrid_data = {
    'level_0': [solution_t0, solution_t1, ...],  # Finest level
    'level_1': [solution_t0, solution_t1, ...],  # Coarser level
    # ... more levels
}
grid_levels = [0, 1, 2, 3]  # Grid level hierarchy

# Create animation
fig, axes, animation = viz_tools.create_multigrid_cycle_animation(
    multigrid_data, grid_levels, cycle_type='V',
    title="V-Cycle Multigrid Animation"
)
```

**Animation Features**:
- **Real-time Grid Highlighting**: Shows which grid level is currently active
- **Operation Labels**: Displays current operation (pre-smooth, restrict, solve, prolongate, post-smooth)
- **Progress Tracking**: Shows current step in the multigrid cycle
- **Grid Overlay**: Visualizes mesh structure on coarse grids
- **Cycle Diagram**: Interactive diagram showing cycle progress

### 3. Convergence History Comparison

**Purpose**: Compare convergence behavior across multiple methods with statistical analysis.

**Features**:
- Multiple metric support (residual, error, energy norm)
- Statistical confidence intervals
- Convergence rate calculation
- Solver efficiency metrics
- Interactive method selection
- Data smoothing options
- Comprehensive summary tables

**Usage Example**:
```python
# Prepare convergence data
convergence_data = {
    'Jacobi': {
        'residual': [1.0, 0.8, 0.64, ...],
        'error': [1.0, 0.7, 0.49, ...],
        'solve_time': [0.01, 0.01, 0.01, ...],
        'residual_std': [0.1, 0.08, 0.064, ...]  # Optional: for confidence intervals
    },
    'Multigrid_V': {
        'residual': [1.0, 0.1, 0.01, ...],
        'error': [1.0, 0.15, 0.02, ...],
        'solve_time': [0.05, 0.05, 0.05, ...]
    },
    # ... more methods
}

# Create comparison visualization
fig, axes, widgets = viz_tools.create_convergence_history_comparison(
    convergence_data, statistical_analysis=True,
    title="Solver Convergence Comparison"
)
```

**Interactive Controls**:
- **Method Selection**: Checkboxes to enable/disable specific methods
- **Metric Selection**: Radio buttons to switch between residual, error, energy norm
- **Scale Selection**: Linear vs logarithmic y-axis
- **Confidence Intervals**: Toggle 95% confidence intervals
- **Smoothing**: Gaussian smoothing parameter for noisy data

**Output Components**:
- **Main Plot**: Convergence curves with optional confidence bands
- **Rate Analysis**: Average convergence rates with target lines
- **Efficiency Plot**: Performance vs accuracy scatter plot
- **Statistical Table**: Summary statistics for all methods

### 4. GPU Memory Usage Visualization

**Purpose**: Monitor and analyze GPU memory usage patterns during computations.

**Features**:
- Multi-GPU monitoring support
- Real-time memory tracking
- Memory type breakdown (allocated, cached, free)
- Alert system for high usage
- Performance impact analysis
- Memory distribution visualization

**Usage Example**:
```python
# Prepare GPU memory data
memory_data = {
    '0': {  # GPU ID
        'allocated': [1000, 1200, 1100, ...],  # MB over time
        'cached': [200, 240, 220, ...],
        'free': [6792, 6552, 6672, ...],
        'total': [8192, 8192, 8192, ...],
        'max_memory': [8192, 8192, 8192, ...]
    },
    '1': { /* GPU 1 data */ },
    # ... more GPUs
}

# Create memory visualization
fig, axes, widgets = viz_tools.create_gpu_memory_visualization(
    memory_data, real_time=False,
    title="Multi-GPU Memory Analysis"
)
```

**Interactive Controls**:
- **GPU Selection**: Checkboxes to select which GPUs to monitor
- **Memory Type**: Radio buttons for total, allocated, cached, free memory
- **Time Window**: Slider to adjust monitoring time window
- **Alert Threshold**: Set memory usage alert percentage
- **Real-time Toggle**: Enable/disable real-time monitoring
- **Reset Button**: Clear alerts and reset view

**Visualization Components**:
- **Usage Timeline**: Memory usage over time with threshold lines
- **Distribution Pie Chart**: Current memory distribution across GPUs
- **Stacked Timeline**: Memory allocation breakdown over time
- **Statistics Panel**: Real-time statistics and memory alerts

### 5. Precision Error Propagation Analysis

**Purpose**: Analyze how numerical errors propagate through mixed-precision computations.

**Features**:
- Multi-precision comparison (FP16, FP32, FP64)
- Error evolution tracking
- Statistical error distribution
- Precision switching visualization
- Error growth rate analysis

**Usage Example**:
```python
# Prepare error propagation data
error_data = {
    'fp32': {
        'error_matrix': np.array(...),  # [n_operations, n_variables]
        'error_evolution': np.array(...),  # Mean error over operations
        'max_errors': np.array(...),
        'std_errors': np.array(...)
    },
    'fp64': { /* FP64 error data */ },
    # ... more precision levels
}

# Create error analysis
fig, axes, widgets = viz_tools.create_precision_error_propagation_analysis(
    error_data, precision_levels=['fp16', 'fp32', 'fp64'],
    title="Mixed-Precision Error Analysis"
)
```

**Analysis Components**:
- **Error Heatmap**: 2D visualization of error propagation across operations and variables
- **Distribution Histograms**: Error distribution for each precision level
- **Growth Curves**: Error evolution over computational operations
- **Switching Visualization**: Optimal precision switching points
- **Statistical Summary**: Error statistics and precision efficiency metrics

### 6. Performance Scaling with Error Bars

**Purpose**: Analyze parallel performance scaling with statistical confidence measures.

**Features**:
- Strong and weak scaling analysis
- Multiple parallelization methods comparison
- Confidence intervals and error bars
- Efficiency and speedup calculations
- Cost-performance analysis

**Usage Example**:
```python
# Prepare scaling data
scaling_data = {
    'OpenMP': {
        'solve_time': [100, 52, 28, 16, ...],  # Times for [1, 2, 4, 8, ...] cores
        'efficiency': [1.0, 0.96, 0.89, 0.78, ...],
        'speedup': [1.0, 1.92, 3.57, 6.25, ...]
    },
    'CUDA': { /* GPU scaling data */ },
    # ... more methods
}

# Optional: confidence intervals
confidence_intervals = {
    'OpenMP': {
        'solve_time': [5, 3, 2, 1.5, ...],  # Error bars
        'efficiency': [0.02, 0.03, 0.04, 0.05, ...]
    },
    # ... more methods
}

# Create scaling analysis
fig, axes, widgets = viz_tools.create_performance_scaling_with_error_bars(
    scaling_data, confidence_intervals,
    title="Parallel Performance Scaling Analysis"
)
```

**Scaling Analysis Components**:
- **Strong Scaling**: Fixed problem size, varying core count
- **Weak Scaling**: Fixed problem per core, varying core count  
- **Efficiency Analysis**: Parallel efficiency vs core count
- **Cost-Performance**: Performance per computational cost unit

## Best Practices

### Data Preparation

1. **Consistent Formatting**: Ensure all data dictionaries use consistent key names
2. **Error Handling**: Include try-catch blocks when loading experimental data
3. **Data Validation**: Verify array dimensions match expected formats
4. **Missing Data**: Handle missing data points gracefully

### Performance Optimization

1. **Data Sampling**: For large datasets, consider sampling for interactive performance
2. **Animation Frame Rate**: Adjust animation intervals based on data complexity
3. **Memory Management**: Clear previous plots when updating to prevent memory leaks
4. **Widget Responsiveness**: Debounce rapid widget updates for better user experience

### Publication Quality

1. **Color Schemes**: Use colorblind-friendly palettes for publications
2. **Font Sizing**: Ensure text is readable at publication resolution
3. **Vector Graphics**: Save as PDF/SVG for scalable publication figures
4. **Figure Dimensions**: Match journal requirements for figure sizes

## Customization

### Extending Visualizations

```python
class CustomVisualizationTools(AdvancedVisualizationTools):
    def create_custom_analysis(self, data, title="Custom Analysis"):
        """Add your own visualization methods."""
        fig = plt.figure(figsize=(12, 8))
        # Custom implementation
        return fig, axes, widgets

# Use custom tools
custom_viz = CustomVisualizationTools()
```

### Style Customization

```python
# Initialize with custom style
viz_tools = create_missing_visualizations()

# Modify color scheme
viz_tools.colors.update({
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c'
})

# Adjust animation parameters
viz_tools.animation_params.update({
    'interval': 100,  # Faster animation
    'frames': 200     # More frames
})
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce data size or increase system memory
3. **Animation Performance**: Decrease frame rate or reduce data resolution
4. **Widget Not Responding**: Check for matplotlib backend compatibility

### Performance Tips

1. **Use appropriate data types**: Float32 vs Float64 based on precision needs
2. **Optimize animation data**: Pre-compute expensive operations
3. **Batch widget updates**: Group related parameter changes
4. **Clear axes properly**: Use ax.clear() before redrawing

## Future Extensions

Planned enhancements for future versions:

1. **WebGL Support**: Browser-based interactive 3D visualizations
2. **HDF5 Integration**: Direct support for large dataset formats
3. **Jupyter Widgets**: Native Jupyter notebook integration
4. **Export Capabilities**: Animation and interactive plot export
5. **Performance Profiling**: Built-in performance monitoring tools

## References

- [Matplotlib Animation Documentation](https://matplotlib.org/stable/api/animation_api.html)
- [NumPy Array Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Scientific Visualization Best Practices](https://ieeexplore.ieee.org/document/7539069)

For more examples and detailed API documentation, see the `examples/advanced_visualizations_demo.py` script.