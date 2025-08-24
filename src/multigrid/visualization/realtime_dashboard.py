"""
Real-time Solver Monitoring Dashboard

This module implements a comprehensive real-time monitoring dashboard for
multigrid solver analysis with live updates, memory tracking, GPU monitoring,
and precision switching visualization.

Classes:
    SolverDashboard: Main real-time monitoring dashboard
    LiveDataStreamer: Data streaming and threading support
    MetricsCollector: System metrics collection
    
Functions:
    solver_dashboard: Main dashboard creation function
    create_monitoring_widgets: Dashboard control widgets
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import queue
import json
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Optional imports for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    GPUtil = None

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collect system metrics including memory, CPU, and GPU utilization.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of data points to store
        """
        self.max_history = max_history
        self.reset_metrics()
        
        # Check available monitoring capabilities
        self.can_monitor_system = HAS_PSUTIL
        self.can_monitor_gpu = HAS_GPUTIL
        
        if not self.can_monitor_system:
            logger.warning("psutil not available - system monitoring disabled")
        if not self.can_monitor_gpu:
            logger.warning("GPUtil not available - GPU monitoring disabled")
    
    def reset_metrics(self):
        """Reset all metric histories."""
        self.timestamps = deque(maxlen=self.max_history)
        self.cpu_usage = deque(maxlen=self.max_history)
        self.memory_usage = deque(maxlen=self.max_history)
        self.gpu_usage = deque(maxlen=self.max_history)
        self.gpu_memory = deque(maxlen=self.max_history)
        self.solver_residuals = deque(maxlen=self.max_history)
        self.solver_iterations = deque(maxlen=self.max_history)
        self.precision_switches = deque(maxlen=self.max_history)
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_mb': 0.0,
            'gpu_percent': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_memory_used_mb': 0.0
        }
        
        if self.can_monitor_system:
            try:
                metrics['cpu_percent'] = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                metrics['memory_percent'] = memory.percent
                metrics['memory_used_mb'] = memory.used / (1024 * 1024)
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
        
        if self.can_monitor_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics['gpu_percent'] = gpu.load * 100
                    metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                    metrics['gpu_memory_used_mb'] = gpu.memoryUsed
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
        
        return metrics
    
    def update_metrics(self, solver_metrics: Optional[Dict[str, float]] = None):
        """
        Update all metrics with current values.
        
        Args:
            solver_metrics: Optional solver-specific metrics
        """
        # Collect system metrics
        sys_metrics = self.collect_system_metrics()
        
        # Store system metrics
        self.timestamps.append(sys_metrics['timestamp'])
        self.cpu_usage.append(sys_metrics['cpu_percent'])
        self.memory_usage.append(sys_metrics['memory_percent'])
        self.gpu_usage.append(sys_metrics['gpu_percent'])
        self.gpu_memory.append(sys_metrics['gpu_memory_percent'])
        
        # Store solver metrics if provided
        if solver_metrics:
            self.solver_residuals.append(solver_metrics.get('residual', 0.0))
            self.solver_iterations.append(solver_metrics.get('iteration', 0))
            self.precision_switches.append(solver_metrics.get('precision_level', 32))
    
    def get_recent_data(self, n_points: int = 100) -> Dict[str, List]:
        """
        Get recent metric data.
        
        Args:
            n_points: Number of recent points to return
            
        Returns:
            Dictionary containing recent metric arrays
        """
        def get_recent(data, n):
            return list(data)[-n:] if len(data) >= n else list(data)
        
        return {
            'timestamps': get_recent(self.timestamps, n_points),
            'cpu_usage': get_recent(self.cpu_usage, n_points),
            'memory_usage': get_recent(self.memory_usage, n_points),
            'gpu_usage': get_recent(self.gpu_usage, n_points),
            'gpu_memory': get_recent(self.gpu_memory, n_points),
            'solver_residuals': get_recent(self.solver_residuals, n_points),
            'solver_iterations': get_recent(self.solver_iterations, n_points),
            'precision_switches': get_recent(self.precision_switches, n_points)
        }

class LiveDataStreamer:
    """
    Handle real-time data streaming using threading.
    """
    
    def __init__(self, update_interval: float = 0.1):
        """
        Initialize data streamer.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.data_queue = queue.Queue()
        self.metrics_collector = MetricsCollector()
        self.running = False
        self.thread = None
        self.solver_callback = None
    
    def set_solver_callback(self, callback: Callable):
        """Set callback function to get solver metrics."""
        self.solver_callback = callback
    
    def start_streaming(self):
        """Start real-time data streaming."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        logger.info("Data streaming started")
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Data streaming stopped")
    
    def _stream_loop(self):
        """Main streaming loop (runs in separate thread)."""
        while self.running:
            try:
                # Get solver metrics if callback is available
                solver_metrics = None
                if self.solver_callback:
                    try:
                        solver_metrics = self.solver_callback()
                    except Exception as e:
                        logger.warning(f"Solver callback error: {e}")
                
                # Update metrics
                self.metrics_collector.update_metrics(solver_metrics)
                
                # Put data in queue for GUI thread
                recent_data = self.metrics_collector.get_recent_data(200)
                self.data_queue.put(recent_data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get latest data from queue (non-blocking)."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

class SolverDashboard:
    """
    Real-time solver monitoring dashboard with live plots and metrics.
    """
    
    def __init__(self, update_interval: float = 0.1, window_size: int = 200):
        """
        Initialize solver dashboard.
        
        Args:
            update_interval: GUI update interval in seconds
            window_size: Number of data points to display
        """
        self.update_interval = update_interval
        self.window_size = window_size
        self.data_streamer = LiveDataStreamer(update_interval)
        
        # Dashboard state
        self.is_running = False
        self.start_time = None
        
        # Color scheme
        self.colors = {
            'convergence': '#2563eb',
            'memory': '#dc2626',
            'gpu': '#059669',
            'cpu': '#f59e0b',
            'precision': '#8b5cf6',
            'background': '#f8f9fa',
            'grid': '#e5e7eb',
            'alert': '#ef4444',
            'success': '#10b981'
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'memory_percent': 80.0,
            'gpu_memory_percent': 90.0,
            'cpu_percent': 90.0,
            'convergence_stall': 50  # iterations without improvement
        }
        
        # Create dashboard
        self._create_dashboard()
    
    def _create_dashboard(self):
        """Create the main dashboard interface."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Real-time Solver Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Define subplot layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Convergence plot (top-left, spans 2 columns)
        self.ax_convergence = self.fig.add_subplot(gs[0, :2])
        self.ax_convergence.set_title('Live Convergence')
        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('Residual')
        self.ax_convergence.set_yscale('log')
        self.ax_convergence.grid(True, alpha=0.3)
        
        # Memory usage (top-right)
        self.ax_memory = self.fig.add_subplot(gs[0, 2])
        self.ax_memory.set_title('Memory Usage')
        self.ax_memory.set_ylabel('Usage (%)')
        self.ax_memory.set_ylim(0, 100)
        self.ax_memory.grid(True, alpha=0.3)
        
        # GPU utilization (top-far-right)
        self.ax_gpu = self.fig.add_subplot(gs[0, 3])
        self.ax_gpu.set_title('GPU Utilization')
        self.ax_gpu.set_ylabel('Usage (%)')
        self.ax_gpu.set_ylim(0, 100)
        self.ax_gpu.grid(True, alpha=0.3)
        
        # Precision switching (middle-left, spans 2 columns)
        self.ax_precision = self.fig.add_subplot(gs[1, :2])
        self.ax_precision.set_title('Precision Switching')
        self.ax_precision.set_xlabel('Time (s)')
        self.ax_precision.set_ylabel('Precision Level')
        self.ax_precision.grid(True, alpha=0.3)
        
        # System metrics (middle-right)
        self.ax_system = self.fig.add_subplot(gs[1, 2])
        self.ax_system.set_title('System Metrics')
        self.ax_system.axis('off')
        
        # Alerts panel (middle-far-right)
        self.ax_alerts = self.fig.add_subplot(gs[1, 3])
        self.ax_alerts.set_title('Alerts')
        self.ax_alerts.axis('off')
        
        # Controls panel (bottom, spans all columns)
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.set_title('Dashboard Controls')
        self.ax_controls.axis('off')
        
        # Initialize plot lines
        self.lines = {}
        self._setup_plot_lines()
        
        # Create control widgets
        self._create_control_widgets()
        
        # Initialize animation
        self.animation = FuncAnimation(
            self.fig, self._update_plots, interval=int(self.update_interval * 1000),
            blit=False, cache_frame_data=False
        )
    
    def _setup_plot_lines(self):
        """Initialize plot lines for real-time updates."""
        # Convergence plot
        self.lines['residual'], = self.ax_convergence.plot(
            [], [], 'o-', color=self.colors['convergence'], 
            linewidth=2, markersize=3, label='Residual'
        )
        self.ax_convergence.legend()
        
        # Memory usage
        self.lines['memory'], = self.ax_memory.plot(
            [], [], '-', color=self.colors['memory'], linewidth=2, label='Memory'
        )
        self.lines['memory_threshold'] = self.ax_memory.axhline(
            y=self.alert_thresholds['memory_percent'], 
            color=self.colors['alert'], linestyle='--', alpha=0.7, label='Threshold'
        )
        self.ax_memory.legend()
        
        # GPU utilization
        self.lines['gpu'], = self.ax_gpu.plot(
            [], [], '-', color=self.colors['gpu'], linewidth=2, label='GPU'
        )
        self.lines['gpu_memory'], = self.ax_gpu.plot(
            [], [], '--', color=self.colors['gpu'], linewidth=2, alpha=0.7, label='GPU Mem'
        )
        self.ax_gpu.legend()
        
        # Precision switching
        self.lines['precision'], = self.ax_precision.plot(
            [], [], 's-', color=self.colors['precision'], 
            linewidth=2, markersize=4, label='Precision'
        )
        self.ax_precision.legend()
    
    def _create_control_widgets(self):
        """Create dashboard control widgets."""
        # Start/Stop button
        ax_start = plt.axes([0.1, 0.02, 0.08, 0.04])
        self.btn_start = widgets.Button(ax_start, 'Start', color='lightgreen')
        self.btn_start.on_clicked(self._toggle_monitoring)
        
        # Reset button
        ax_reset = plt.axes([0.2, 0.02, 0.08, 0.04])
        self.btn_reset = widgets.Button(ax_reset, 'Reset', color='lightblue')
        self.btn_reset.on_clicked(self._reset_dashboard)
        
        # Update interval slider
        ax_interval = plt.axes([0.4, 0.02, 0.2, 0.03])
        self.slider_interval = widgets.Slider(
            ax_interval, 'Update (ms)', 50, 1000, 
            valinit=self.update_interval * 1000, valfmt='%d'
        )
        self.slider_interval.on_changed(self._update_interval_changed)
        
        # Window size slider
        ax_window = plt.axes([0.65, 0.02, 0.2, 0.03])
        self.slider_window = widgets.Slider(
            ax_window, 'Window Size', 50, 500, 
            valinit=self.window_size, valfmt='%d'
        )
        self.slider_window.on_changed(self._window_size_changed)
    
    def _toggle_monitoring(self, event):
        """Toggle monitoring on/off."""
        if not self.is_running:
            self.start_monitoring()
            self.btn_start.label.set_text('Stop')
            self.btn_start.color = 'lightcoral'
        else:
            self.stop_monitoring()
            self.btn_start.label.set_text('Start')
            self.btn_start.color = 'lightgreen'
    
    def _reset_dashboard(self, event):
        """Reset dashboard data."""
        self.data_streamer.metrics_collector.reset_metrics()
        self._clear_plots()
        logger.info("Dashboard reset")
    
    def _update_interval_changed(self, val):
        """Handle update interval change."""
        self.update_interval = val / 1000.0
        if self.is_running:
            # Restart streaming with new interval
            self.data_streamer.stop_streaming()
            self.data_streamer.update_interval = self.update_interval
            self.data_streamer.start_streaming()
    
    def _window_size_changed(self, val):
        """Handle window size change."""
        self.window_size = int(val)
    
    def _clear_plots(self):
        """Clear all plot data."""
        for line in self.lines.values():
            if hasattr(line, 'set_data'):
                line.set_data([], [])
        
        # Clear text displays
        self.ax_system.clear()
        self.ax_system.axis('off')
        self.ax_alerts.clear()
        self.ax_alerts.axis('off')
    
    def _update_plots(self, frame):
        """Update all plots with latest data (called by animation)."""
        if not self.is_running:
            return []
        
        # Get latest data
        data = self.data_streamer.get_latest_data()
        if not data or not data.get('timestamps'):
            return []
        
        try:
            # Get time series for x-axis
            timestamps = np.array(data['timestamps'])
            if len(timestamps) == 0:
                return []
            
            # Convert to relative time from start
            if self.start_time is None:
                self.start_time = timestamps[0]
            
            relative_times = timestamps - self.start_time
            
            # Limit to window size
            if len(relative_times) > self.window_size:
                start_idx = len(relative_times) - self.window_size
                relative_times = relative_times[start_idx:]
                for key in data:
                    if key != 'timestamps' and len(data[key]) > self.window_size:
                        data[key] = data[key][start_idx:]
            
            # Update convergence plot
            if data.get('solver_residuals') and len(data['solver_residuals']) > 0:
                residuals = np.array(data['solver_residuals'])
                iterations = np.arange(len(residuals))
                self.lines['residual'].set_data(iterations, residuals)
                self.ax_convergence.relim()
                self.ax_convergence.autoscale_view()
            
            # Update memory plot
            if data.get('memory_usage'):
                memory_data = np.array(data['memory_usage'])
                self.lines['memory'].set_data(relative_times[-len(memory_data):], memory_data)
                self.ax_memory.relim()
                self.ax_memory.autoscale_view()
            
            # Update GPU plots
            if data.get('gpu_usage'):
                gpu_data = np.array(data['gpu_usage'])
                self.lines['gpu'].set_data(relative_times[-len(gpu_data):], gpu_data)
                
                if data.get('gpu_memory'):
                    gpu_mem_data = np.array(data['gpu_memory'])
                    self.lines['gpu_memory'].set_data(relative_times[-len(gpu_mem_data):], gpu_mem_data)
                
                self.ax_gpu.relim()
                self.ax_gpu.autoscale_view()
            
            # Update precision plot
            if data.get('precision_switches'):
                precision_data = np.array(data['precision_switches'])
                self.lines['precision'].set_data(relative_times[-len(precision_data):], precision_data)
                self.ax_precision.relim()
                self.ax_precision.autoscale_view()
            
            # Update system metrics display
            self._update_system_display(data)
            
            # Update alerts
            self._update_alerts_display(data)
            
        except Exception as e:
            logger.error(f"Plot update error: {e}")
        
        return []
    
    def _update_system_display(self, data):
        """Update system metrics text display."""
        self.ax_system.clear()
        self.ax_system.axis('off')
        
        if not data.get('timestamps'):
            return
        
        # Get latest values
        latest_cpu = data['cpu_usage'][-1] if data.get('cpu_usage') else 0
        latest_memory = data['memory_usage'][-1] if data.get('memory_usage') else 0
        latest_gpu = data['gpu_usage'][-1] if data.get('gpu_usage') else 0
        latest_gpu_mem = data['gpu_memory'][-1] if data.get('gpu_memory') else 0
        
        metrics_text = [
            f"CPU: {latest_cpu:.1f}%",
            f"Memory: {latest_memory:.1f}%",
            f"GPU: {latest_gpu:.1f}%",
            f"GPU Mem: {latest_gpu_mem:.1f}%",
            f"Uptime: {time.time() - (self.start_time or time.time()):.1f}s"
        ]
        
        self.ax_system.text(0.1, 0.8, '\n'.join(metrics_text), 
                          transform=self.ax_system.transAxes,
                          fontsize=12, fontfamily='monospace',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _update_alerts_display(self, data):
        """Update alerts display."""
        self.ax_alerts.clear()
        self.ax_alerts.axis('off')
        
        alerts = []
        
        # Check memory threshold
        if data.get('memory_usage') and data['memory_usage'][-1] > self.alert_thresholds['memory_percent']:
            alerts.append(f"⚠ High Memory: {data['memory_usage'][-1]:.1f}%")
        
        # Check GPU memory threshold
        if data.get('gpu_memory') and data['gpu_memory'][-1] > self.alert_thresholds['gpu_memory_percent']:
            alerts.append(f"⚠ High GPU Memory: {data['gpu_memory'][-1]:.1f}%")
        
        # Check CPU threshold
        if data.get('cpu_usage') and data['cpu_usage'][-1] > self.alert_thresholds['cpu_percent']:
            alerts.append(f"⚠ High CPU: {data['cpu_usage'][-1]:.1f}%")
        
        # Check convergence stall
        if (data.get('solver_residuals') and len(data['solver_residuals']) > self.alert_thresholds['convergence_stall']):
            recent_residuals = data['solver_residuals'][-self.alert_thresholds['convergence_stall']:]
            if len(set(recent_residuals)) == 1:  # No change in residuals
                alerts.append("⚠ Convergence Stalled")
        
        if not alerts:
            alerts = ["✓ All systems normal"]
        
        alert_text = '\n'.join(alerts[:5])  # Show max 5 alerts
        color = 'lightcoral' if len(alerts) > 1 or alerts[0].startswith('⚠') else 'lightgreen'
        
        self.ax_alerts.text(0.1, 0.8, alert_text,
                          transform=self.ax_alerts.transAxes,
                          fontsize=10, fontfamily='monospace',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.data_streamer.start_streaming()
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.data_streamer.stop_streaming()
        logger.info("Monitoring stopped")
    
    def set_solver_callback(self, callback: Callable):
        """
        Set callback function to get solver metrics.
        
        Args:
            callback: Function that returns dict with solver metrics
                     Should return: {'residual': float, 'iteration': int, 'precision_level': int}
        """
        self.data_streamer.set_solver_callback(callback)
    
    def show(self):
        """Display the dashboard."""
        plt.show()
    
    def save_screenshot(self, filename: str):
        """Save dashboard screenshot."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard screenshot saved: {filename}")


def solver_dashboard():
    """
    IMPLEMENT: Real-time solver monitoring
    - Live convergence plotting
    - Memory usage tracking  
    - GPU utilization monitoring
    - Precision switching visualization
    """
    # Create and return the dashboard
    dashboard = SolverDashboard()
    
    logger.info("Real-time solver monitoring dashboard created")
    logger.info("Features:")
    logger.info("- Live convergence plotting")
    logger.info("- Memory usage tracking")
    logger.info("- GPU utilization monitoring") 
    logger.info("- Precision switching visualization")
    
    return dashboard


def create_monitoring_widgets(dashboard: SolverDashboard) -> Dict[str, widgets.Widget]:
    """
    Create additional monitoring control widgets.
    
    Args:
        dashboard: SolverDashboard instance
        
    Returns:
        Dictionary of widget controls
    """
    widget_controls = {}
    
    # Alert threshold controls
    ax_mem_threshold = plt.axes([0.1, 0.1, 0.2, 0.03])
    widget_controls['memory_threshold'] = widgets.Slider(
        ax_mem_threshold, 'Memory Alert (%)', 50, 95, 
        valinit=dashboard.alert_thresholds['memory_percent']
    )
    
    ax_gpu_threshold = plt.axes([0.35, 0.1, 0.2, 0.03])  
    widget_controls['gpu_threshold'] = widgets.Slider(
        ax_gpu_threshold, 'GPU Alert (%)', 50, 95,
        valinit=dashboard.alert_thresholds['gpu_memory_percent']
    )
    
    # Export controls
    ax_export = plt.axes([0.7, 0.1, 0.08, 0.04])
    widget_controls['export_btn'] = widgets.Button(ax_export, 'Export', color='lightyellow')
    
    def export_data(event):
        filename = f"dashboard_data_{int(time.time())}.json"
        data = dashboard.data_streamer.metrics_collector.get_recent_data()
        with open(filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    json_data[key] = list(value)
                else:
                    json_data[key] = value
            json.dump(json_data, f, indent=2)
        logger.info(f"Data exported to {filename}")
    
    widget_controls['export_btn'].on_clicked(export_data)
    
    return widget_controls


# Example solver callback for demonstration
def example_solver_callback() -> Dict[str, float]:
    """
    Example solver callback function.
    
    This should be replaced with actual solver integration.
    
    Returns:
        Dictionary with current solver metrics
    """
    # Simulate solver progress
    current_time = time.time()
    
    # Simulate decreasing residual
    residual = 1.0 * np.exp(-current_time / 10.0) + 0.01 * np.random.randn()
    
    # Simulate iteration count
    iteration = int(current_time * 5)  # 5 iterations per second
    
    # Simulate precision switching
    precision_level = 32 if (iteration % 100) < 80 else 64
    
    return {
        'residual': max(residual, 1e-12),
        'iteration': iteration,
        'precision_level': precision_level
    }