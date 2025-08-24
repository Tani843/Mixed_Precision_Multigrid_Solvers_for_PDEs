# Real-time Solver Monitoring Dashboard

This document describes the comprehensive real-time monitoring dashboard implemented for the Mixed-Precision Multigrid Solvers project. The dashboard provides live visualization of solver convergence, system resource usage, and performance metrics.

## Overview

The real-time dashboard (`src/multigrid/visualization/realtime_dashboard.py`) implements a comprehensive monitoring system with:

- **Live convergence plotting** - Real-time visualization of solver residuals and convergence behavior
- **Memory usage tracking** - System and GPU memory monitoring with configurable alerts
- **GPU utilization monitoring** - Real-time GPU usage and memory tracking
- **Precision switching visualization** - Live display of precision level changes during mixed-precision solving

## Quick Start

### Basic Usage

```python
from multigrid.visualization import solver_dashboard

# Create the dashboard
dashboard = solver_dashboard()

# Set up solver callback (see Integration section)
dashboard.set_solver_callback(your_solver_callback)

# Start monitoring
dashboard.start_monitoring()

# Show the dashboard
dashboard.show()
```

### Running the Demo

```bash
# Interactive demo with simulated data
python examples/realtime_dashboard_demo.py --mode demo

# Test functionality
python examples/realtime_dashboard_demo.py --mode test

# Integrated solver demonstration  
python examples/realtime_dashboard_demo.py --mode solver
```

## Dashboard Components

### 1. Live Convergence Plotting

**Location**: Top-left panel
**Features**:
- Real-time residual plotting on logarithmic scale
- Iteration-based x-axis with automatic scaling
- Multiple solver comparison support
- Convergence stall detection and alerts

**Data Requirements**:
```python
solver_metrics = {
    'residual': float,     # Current residual value
    'iteration': int,      # Current iteration number
    'precision_level': int # Current precision (16, 32, 64)
}
```

### 2. Memory Usage Tracking

**Location**: Top-center panel
**Features**:
- System memory usage percentage
- Configurable alert thresholds (default: 80%)
- Historical tracking with time-based x-axis
- Visual threshold indicators

**Monitoring Capabilities**:
- System RAM usage via `psutil`
- Memory alerts and notifications
- Trend analysis and peak detection

### 3. GPU Utilization Monitoring

**Location**: Top-right panel
**Features**:
- GPU compute utilization (%)
- GPU memory usage (%)
- Multi-GPU support (primary GPU displayed)
- Real-time usage curves

**Requirements**:
- Optional: `pip install GPUtil` for GPU monitoring
- Falls back gracefully if GPU monitoring unavailable

### 4. Precision Switching Visualization

**Location**: Middle-left panel
**Features**:
- Real-time precision level display
- Precision change annotations
- Historical precision timeline
- Mixed-precision strategy analysis

**Supported Precision Levels**:
- FP16 (Half precision)
- FP32 (Single precision) 
- FP64 (Double precision)

### 5. System Metrics Panel

**Location**: Middle-center panel
**Features**:
- Current CPU usage percentage
- Current memory usage
- GPU utilization summary
- System uptime
- Real-time metric updates

### 6. Alerts Panel

**Location**: Middle-right panel
**Features**:
- High memory usage alerts (configurable threshold)
- High CPU usage warnings
- GPU memory alerts
- Convergence stall detection
- Color-coded alert levels

### 7. Interactive Controls

**Location**: Bottom panel
**Features**:
- Start/Stop monitoring toggle
- Reset data button
- Update interval slider (50-1000ms)
- Data window size slider (50-500 points)
- Data export functionality

## Integration with Solvers

### Callback Pattern

The dashboard uses a callback pattern to integrate with solvers:

```python
def your_solver_callback():
    """
    Your solver callback function.
    
    Returns:
        dict: Current solver metrics
    """
    return {
        'residual': current_residual,
        'iteration': current_iteration, 
        'precision_level': current_precision
    }

# Set up the callback
dashboard.set_solver_callback(your_solver_callback)
```

### Example Integration

```python
class YourSolver:
    def __init__(self):
        self.iteration = 0
        self.residual = 1.0
        self.precision = 32
        
        # Set up dashboard
        self.dashboard = solver_dashboard()
        self.dashboard.set_solver_callback(self.get_metrics)
        self.dashboard.start_monitoring()
    
    def get_metrics(self):
        return {
            'residual': self.residual,
            'iteration': self.iteration,
            'precision_level': self.precision
        }
    
    def solve(self):
        while not self.converged():
            # Perform solver iteration
            self.step()
            
            # Metrics are automatically collected via callback
            
        # Stop monitoring when done
        self.dashboard.stop_monitoring()
```

### Threading and Performance

The dashboard uses multi-threading for real-time data collection:

- **GUI Thread**: Handles matplotlib updates and user interactions
- **Data Collection Thread**: Collects system and solver metrics
- **Thread-Safe Queues**: Communicate between threads safely

**Performance Considerations**:
- Default update interval: 100ms (configurable)
- Automatic data windowing (default: 200 points)
- Efficient memory management for long runs

## Configuration Options

### Alert Thresholds

```python
dashboard.alert_thresholds = {
    'memory_percent': 80.0,      # System memory alert threshold
    'gpu_memory_percent': 90.0,  # GPU memory alert threshold  
    'cpu_percent': 90.0,         # CPU usage alert threshold
    'convergence_stall': 50      # Iterations without improvement
}
```

### Update Settings

```python
dashboard = SolverDashboard(
    update_interval=0.1,    # Update every 100ms
    window_size=200         # Show last 200 data points
)
```

### Monitoring Widgets

```python
from multigrid.visualization import create_monitoring_widgets

# Create additional control widgets
widgets = create_monitoring_widgets(dashboard)

# Available widgets:
# - Memory alert threshold slider
# - GPU alert threshold slider  
# - Data export button
```

## Data Export and Analysis

### Export Functionality

```python
# Export current data to JSON
dashboard.export_data('monitoring_data.json')

# Programmatic data access
recent_data = dashboard.data_streamer.metrics_collector.get_recent_data(100)
```

### Data Format

Exported data contains:
```python
{
    "timestamps": [...],           # Unix timestamps
    "cpu_usage": [...],           # CPU usage percentages
    "memory_usage": [...],        # Memory usage percentages  
    "gpu_usage": [...],           # GPU usage percentages
    "gpu_memory": [...],          # GPU memory percentages
    "solver_residuals": [...],    # Solver residual values
    "solver_iterations": [...],   # Iteration numbers
    "precision_switches": [...]   # Precision levels over time
}
```

## Advanced Features

### Custom Metrics Collection

```python
class CustomMetricsCollector(MetricsCollector):
    def collect_custom_metrics(self):
        # Add your custom metrics here
        return {
            'custom_metric_1': value1,
            'custom_metric_2': value2
        }
```

### Multi-Solver Monitoring

```python
# Monitor multiple solvers simultaneously
dashboard1 = solver_dashboard()
dashboard2 = solver_dashboard()

dashboard1.set_solver_callback(solver1.get_metrics)
dashboard2.set_solver_callback(solver2.get_metrics)
```

### Dashboard Customization

```python
# Customize colors
dashboard.colors.update({
    'convergence': '#custom_color',
    'memory': '#another_color'
})

# Customize alert thresholds
dashboard.alert_thresholds['memory_percent'] = 85.0
```

## System Requirements

### Required Dependencies

- `numpy >= 1.21.0`
- `matplotlib >= 3.5.0`
- `threading` (built-in)
- `queue` (built-in)
- `time` (built-in)

### Optional Dependencies

- `psutil >= 5.8.0` - For system resource monitoring
- `GPUtil >= 1.4.0` - For GPU monitoring

### Installation

```bash
# Required packages (automatically installed with project)
pip install numpy matplotlib

# Optional system monitoring  
pip install psutil

# Optional GPU monitoring
pip install GPUtil
```

## Troubleshooting

### Common Issues

**1. "GPUtil not available" warning**
```bash
# Install GPU monitoring support
pip install GPUtil

# Or ignore if GPU monitoring not needed
```

**2. Dashboard not updating**
- Check that solver callback is returning valid data
- Verify monitoring has been started with `dashboard.start_monitoring()`
- Check update interval is reasonable (50-1000ms)

**3. High CPU usage**
- Increase update interval: `dashboard.update_interval = 0.5`
- Reduce window size: `dashboard.window_size = 100`
- Check solver callback efficiency

**4. Memory leaks during long runs**
- Use data windowing (automatically enabled)
- Periodically reset data: `dashboard._reset_dashboard()`
- Monitor system resources during extended runs

### Performance Optimization

```python
# For long-running solvers
dashboard = SolverDashboard(
    update_interval=0.2,    # Slower updates
    window_size=100         # Smaller data window
)

# Disable expensive features if needed
dashboard.alert_thresholds = {}  # Disable alerting
```

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging for troubleshooting
dashboard = solver_dashboard()
```

## API Reference

### Core Classes

#### `SolverDashboard`

Main dashboard class providing real-time monitoring interface.

```python
class SolverDashboard:
    def __init__(self, update_interval=0.1, window_size=200):
        """Initialize dashboard with specified parameters."""
    
    def start_monitoring(self):
        """Start real-time data collection and display."""
    
    def stop_monitoring(self):
        """Stop monitoring and data collection."""
    
    def set_solver_callback(self, callback: Callable):
        """Set function to collect solver metrics."""
    
    def show(self):
        """Display the dashboard interface."""
    
    def save_screenshot(self, filename: str):
        """Save dashboard screenshot."""
```

#### `MetricsCollector`

Handles collection and storage of system and solver metrics.

```python
class MetricsCollector:
    def __init__(self, max_history=1000):
        """Initialize with maximum data history."""
    
    def update_metrics(self, solver_metrics=None):
        """Update all metrics with current values."""
    
    def get_recent_data(self, n_points=100):
        """Get recent metric data."""
    
    def reset_metrics(self):
        """Reset all metric histories."""
```

#### `LiveDataStreamer`

Manages real-time data streaming using threading.

```python
class LiveDataStreamer:
    def __init__(self, update_interval=0.1):
        """Initialize data streamer."""
    
    def start_streaming(self):
        """Start real-time data collection thread."""
    
    def stop_streaming(self):
        """Stop data collection thread."""
    
    def set_solver_callback(self, callback: Callable):
        """Set solver metrics callback function."""
```

### Main Functions

#### `solver_dashboard()`

```python
def solver_dashboard() -> SolverDashboard:
    """
    Create real-time solver monitoring dashboard.
    
    Features:
    - Live convergence plotting
    - Memory usage tracking
    - GPU utilization monitoring
    - Precision switching visualization
    
    Returns:
        SolverDashboard: Configured dashboard instance
    """
```

#### `create_monitoring_widgets()`

```python
def create_monitoring_widgets(dashboard: SolverDashboard) -> Dict:
    """
    Create additional monitoring control widgets.
    
    Args:
        dashboard: SolverDashboard instance
        
    Returns:
        dict: Dictionary of widget controls
    """
```

## Examples

### Basic Real-time Monitoring

```python
from multigrid.visualization import solver_dashboard

# Create and start dashboard
dashboard = solver_dashboard()

# Define solver callback
def solver_metrics():
    return {
        'residual': current_residual,
        'iteration': current_iteration,
        'precision_level': 32
    }

dashboard.set_solver_callback(solver_metrics)
dashboard.start_monitoring()
dashboard.show()  # Blocks until window closed
```

### Advanced Configuration

```python
# Create custom dashboard
dashboard = SolverDashboard(
    update_interval=0.05,  # 50ms updates
    window_size=500        # Show more history
)

# Customize alert thresholds
dashboard.alert_thresholds.update({
    'memory_percent': 75.0,
    'convergence_stall': 100
})

# Add monitoring widgets
widgets = create_monitoring_widgets(dashboard)

# Start with custom callback
dashboard.set_solver_callback(lambda: {
    'residual': get_current_residual(),
    'iteration': get_current_iteration(),
    'precision_level': get_current_precision()
})

dashboard.start_monitoring()
dashboard.show()
```

For more examples, see:
- `examples/realtime_dashboard_demo.py` - Comprehensive demonstration
- `test_realtime_dashboard.py` - Testing examples
- `test_solver_dashboard_function.py` - Integration examples