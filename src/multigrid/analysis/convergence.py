"""Convergence analysis and monitoring tools."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceData:
    """Container for convergence data."""
    residual_norms: List[float] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    precision_levels: List[str] = field(default_factory=list)
    error_norms: List[float] = field(default_factory=list)
    energy_norms: List[float] = field(default_factory=list)
    
    def add_iteration(
        self,
        residual_norm: float,
        iteration_time: float,
        precision_level: str = "unknown",
        error_norm: Optional[float] = None,
        energy_norm: Optional[float] = None
    ) -> None:
        """Add data for one iteration."""
        self.residual_norms.append(residual_norm)
        self.iteration_times.append(iteration_time)
        self.precision_levels.append(precision_level)
        
        if error_norm is not None:
            self.error_norms.append(error_norm)
        if energy_norm is not None:
            self.energy_norms.append(energy_norm)
    
    def get_convergence_rate(self, window_size: int = 5) -> float:
        """Compute average convergence rate over recent iterations."""
        if len(self.residual_norms) < window_size + 1:
            return 0.0
        
        recent_residuals = self.residual_norms[-window_size-1:]
        rates = []
        
        for i in range(1, len(recent_residuals)):
            if recent_residuals[i-1] > 0:
                rate = recent_residuals[i] / recent_residuals[i-1]
                if 0 < rate < 1:  # Valid convergent rate
                    rates.append(rate)
        
        return np.mean(rates) if rates else 1.0
    
    def clear(self) -> None:
        """Clear all convergence data."""
        self.residual_norms.clear()
        self.iteration_times.clear()
        self.precision_levels.clear()
        self.error_norms.clear()
        self.energy_norms.clear()


class ConvergenceAnalyzer:
    """
    Analyze convergence behavior of iterative solvers.
    
    Provides tools for:
    - Convergence rate analysis
    - Asymptotic behavior estimation
    - Stagnation detection
    - Performance characterization
    """
    
    def __init__(self):
        """Initialize convergence analyzer."""
        self.data = ConvergenceData()
        
    def analyze_convergence(
        self,
        residual_history: List[float],
        error_history: Optional[List[float]] = None,
        iteration_times: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive convergence analysis.
        
        Args:
            residual_history: History of residual norms
            error_history: History of error norms (optional)
            iteration_times: History of iteration times (optional)
            
        Returns:
            Dictionary with convergence analysis results
        """
        if not residual_history:
            raise ValueError("Empty residual history")
        
        analysis = {}
        
        # Basic convergence metrics
        analysis.update(self._compute_basic_metrics(residual_history))
        
        # Convergence rate analysis
        analysis.update(self._analyze_convergence_rates(residual_history))
        
        # Asymptotic behavior
        analysis.update(self._analyze_asymptotic_behavior(residual_history))
        
        # Stagnation detection
        analysis.update(self._detect_stagnation(residual_history))
        
        # Error analysis (if available)
        if error_history:
            analysis.update(self._analyze_error_behavior(error_history, residual_history))
        
        # Performance analysis (if available)
        if iteration_times:
            analysis.update(self._analyze_performance(iteration_times, residual_history))
        
        return analysis
    
    def _compute_basic_metrics(self, residual_history: List[float]) -> Dict[str, Any]:
        """Compute basic convergence metrics."""
        residuals = np.array(residual_history)
        
        metrics = {
            'total_iterations': len(residuals),
            'initial_residual': residuals[0],
            'final_residual': residuals[-1],
            'reduction_factor': residuals[-1] / residuals[0] if residuals[0] > 0 else 0,
            'log_reduction': np.log10(residuals[-1] / residuals[0]) if residuals[0] > 0 else -np.inf
        }
        
        return metrics
    
    def _analyze_convergence_rates(self, residual_history: List[float]) -> Dict[str, Any]:
        """Analyze convergence rates."""
        residuals = np.array(residual_history)
        
        # Compute iteration-to-iteration rates
        rates = []
        for i in range(1, len(residuals)):
            if residuals[i-1] > 0:
                rate = residuals[i] / residuals[i-1]
                if 0 < rate < 2:  # Filter out unrealistic rates
                    rates.append(rate)
        
        if not rates:
            return {'convergence_rates': {'mean': 1.0, 'std': 0.0, 'asymptotic': 1.0}}
        
        rates = np.array(rates)
        
        # Asymptotic rate (average of last 25% of rates)
        n_asymptotic = max(1, len(rates) // 4)
        asymptotic_rate = np.mean(rates[-n_asymptotic:]) if n_asymptotic <= len(rates) else np.mean(rates)
        
        convergence_analysis = {
            'convergence_rates': {
                'mean': float(np.mean(rates)),
                'std': float(np.std(rates)),
                'median': float(np.median(rates)),
                'asymptotic': float(asymptotic_rate),
                'min': float(np.min(rates)),
                'max': float(np.max(rates))
            }
        }
        
        # Classify convergence behavior
        if asymptotic_rate < 0.1:
            convergence_type = "superlinear"
        elif asymptotic_rate < 0.9:
            convergence_type = "linear"
        elif asymptotic_rate < 0.99:
            convergence_type = "slow"
        else:
            convergence_type = "stagnating"
        
        convergence_analysis['convergence_type'] = convergence_type
        
        return convergence_analysis
    
    def _analyze_asymptotic_behavior(self, residual_history: List[float]) -> Dict[str, Any]:
        """Analyze asymptotic convergence behavior."""
        residuals = np.array(residual_history)
        
        if len(residuals) < 10:
            return {'asymptotic_analysis': {'insufficient_data': True}}
        
        # Take last 50% for asymptotic analysis
        n_asymptotic = max(5, len(residuals) // 2)
        asymptotic_residuals = residuals[-n_asymptotic:]
        
        # Fit exponential decay: r_k = C * ρ^k
        iterations = np.arange(len(asymptotic_residuals))
        log_residuals = np.log(asymptotic_residuals + 1e-16)  # Avoid log(0)
        
        try:
            # Linear fit to log(residual) vs iteration
            coeffs = np.polyfit(iterations, log_residuals, 1)
            asymptotic_rate = np.exp(coeffs[0])
            
            # R-squared for goodness of fit
            predicted = coeffs[1] + coeffs[0] * iterations
            ss_res = np.sum((log_residuals - predicted) ** 2)
            ss_tot = np.sum((log_residuals - np.mean(log_residuals)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        except (np.linalg.LinAlgError, ValueError):
            asymptotic_rate = 1.0
            r_squared = 0.0
        
        return {
            'asymptotic_analysis': {
                'asymptotic_rate': float(asymptotic_rate),
                'fit_quality': float(r_squared),
                'data_points_used': n_asymptotic
            }
        }
    
    def _detect_stagnation(self, residual_history: List[float]) -> Dict[str, Any]:
        """Detect convergence stagnation."""
        residuals = np.array(residual_history)
        
        if len(residuals) < 5:
            return {'stagnation_analysis': {'insufficient_data': True}}
        
        # Check for stagnation in last 20% of iterations
        window_size = max(3, len(residuals) // 5)
        recent_residuals = residuals[-window_size:]
        
        # Compute relative changes
        relative_changes = []
        for i in range(1, len(recent_residuals)):
            if recent_residuals[i-1] > 0:
                rel_change = abs(recent_residuals[i] - recent_residuals[i-1]) / recent_residuals[i-1]
                relative_changes.append(rel_change)
        
        if not relative_changes:
            return {'stagnation_analysis': {'no_change_data': True}}
        
        avg_relative_change = np.mean(relative_changes)
        
        # Stagnation thresholds
        stagnation_detected = avg_relative_change < 1e-6
        slow_progress = avg_relative_change < 1e-3
        
        return {
            'stagnation_analysis': {
                'stagnation_detected': stagnation_detected,
                'slow_progress': slow_progress,
                'avg_relative_change': float(avg_relative_change),
                'window_size': window_size
            }
        }
    
    def _analyze_error_behavior(
        self,
        error_history: List[float],
        residual_history: List[float]
    ) -> Dict[str, Any]:
        """Analyze relationship between error and residual."""
        errors = np.array(error_history)
        residuals = np.array(residual_history)
        
        if len(errors) != len(residuals):
            logger.warning("Error and residual histories have different lengths")
            min_len = min(len(errors), len(residuals))
            errors = errors[:min_len]
            residuals = residuals[:min_len]
        
        # Error reduction factor
        error_reduction = errors[-1] / errors[0] if errors[0] > 0 else 0
        
        # Correlation between error and residual
        try:
            correlation = np.corrcoef(np.log(errors + 1e-16), np.log(residuals + 1e-16))[0, 1]
        except (ValueError, RuntimeWarning):
            correlation = 0.0
        
        return {
            'error_analysis': {
                'initial_error': float(errors[0]),
                'final_error': float(errors[-1]),
                'error_reduction_factor': float(error_reduction),
                'error_residual_correlation': float(correlation)
            }
        }
    
    def _analyze_performance(
        self,
        iteration_times: List[float],
        residual_history: List[float]
    ) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        times = np.array(iteration_times)
        residuals = np.array(residual_history)
        
        total_time = np.sum(times)
        avg_time_per_iteration = np.mean(times)
        
        # Time to convergence efficiency
        final_accuracy = residuals[-1] / residuals[0] if residuals[0] > 0 else 1
        time_to_accuracy = total_time / (-np.log10(final_accuracy)) if final_accuracy > 0 else total_time
        
        return {
            'performance_analysis': {
                'total_solve_time': float(total_time),
                'avg_iteration_time': float(avg_time_per_iteration),
                'time_std': float(np.std(times)),
                'time_to_accuracy': float(time_to_accuracy),
                'iterations_per_second': float(len(times) / total_time) if total_time > 0 else 0
            }
        }


class ConvergenceMonitor:
    """
    Real-time convergence monitoring with adaptive behavior.
    
    Monitors convergence during solver execution and can provide
    feedback for adaptive strategies.
    """
    
    def __init__(
        self,
        stagnation_window: int = 10,
        stagnation_threshold: float = 1e-6,
        divergence_threshold: float = 1e3,
        enable_plotting: bool = False
    ):
        """
        Initialize convergence monitor.
        
        Args:
            stagnation_window: Window size for stagnation detection
            stagnation_threshold: Threshold for stagnation detection
            divergence_threshold: Threshold for divergence detection
            enable_plotting: Enable real-time plotting
        """
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold
        self.divergence_threshold = divergence_threshold
        self.enable_plotting = enable_plotting
        
        self.data = ConvergenceData()
        self.callbacks: List[Callable] = []
        
        # Status tracking
        self.is_stagnating = False
        self.is_diverging = False
        self.best_residual = float('inf')
        self.best_iteration = 0
        
        # Plotting setup
        if self.enable_plotting:
            self.fig = None
            self.ax = None
            self._setup_plot()
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add callback function to be called on status changes.
        
        Args:
            callback: Function to call with monitor status
        """
        self.callbacks.append(callback)
    
    def update(
        self,
        iteration: int,
        residual_norm: float,
        iteration_time: float,
        precision_level: str = "unknown",
        error_norm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update monitor with new iteration data.
        
        Args:
            iteration: Current iteration number
            residual_norm: Current residual norm
            iteration_time: Time for this iteration
            precision_level: Precision level used
            error_norm: Error norm (if available)
            
        Returns:
            Dictionary with current monitor status
        """
        # Add data
        self.data.add_iteration(residual_norm, iteration_time, precision_level, error_norm)
        
        # Update best residual tracking
        if residual_norm < self.best_residual:
            self.best_residual = residual_norm
            self.best_iteration = iteration
        
        # Check for stagnation
        self._check_stagnation()
        
        # Check for divergence
        self._check_divergence(residual_norm)
        
        # Update plot
        if self.enable_plotting:
            self._update_plot()
        
        # Prepare status
        status = {
            'iteration': iteration,
            'residual_norm': residual_norm,
            'convergence_rate': self.data.get_convergence_rate(),
            'is_stagnating': self.is_stagnating,
            'is_diverging': self.is_diverging,
            'best_residual': self.best_residual,
            'best_iteration': self.best_iteration
        }
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")
        
        return status
    
    def _check_stagnation(self) -> None:
        """Check for convergence stagnation."""
        if len(self.data.residual_norms) < self.stagnation_window:
            return
        
        # Check relative changes in recent window
        recent_residuals = self.data.residual_norms[-self.stagnation_window:]
        relative_changes = []
        
        for i in range(1, len(recent_residuals)):
            if recent_residuals[i-1] > 0:
                rel_change = abs(recent_residuals[i] - recent_residuals[i-1]) / recent_residuals[i-1]
                relative_changes.append(rel_change)
        
        if relative_changes:
            avg_change = np.mean(relative_changes)
            self.is_stagnating = avg_change < self.stagnation_threshold
    
    def _check_divergence(self, current_residual: float) -> None:
        """Check for solver divergence."""
        if len(self.data.residual_norms) < 3:
            return
        
        initial_residual = self.data.residual_norms[0]
        
        # Divergence if residual grows significantly
        self.is_diverging = (current_residual > self.divergence_threshold * initial_residual)
    
    def _setup_plot(self) -> None:
        """Setup real-time plotting."""
        try:
            import matplotlib.pyplot as plt
            plt.ion()  # Interactive mode
            
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Residual Norm')
            self.ax.set_yscale('log')
            self.ax.set_title('Convergence Monitor')
            self.ax.grid(True, alpha=0.3)
            
        except ImportError:
            logger.warning("Matplotlib not available for real-time plotting")
            self.enable_plotting = False
    
    def _update_plot(self) -> None:
        """Update real-time plot."""
        if not self.enable_plotting or self.fig is None:
            return
        
        try:
            iterations = list(range(len(self.data.residual_norms)))
            
            self.ax.clear()
            self.ax.semilogy(iterations, self.data.residual_norms, 'b-', marker='o', markersize=3)
            
            # Highlight best iteration
            if self.best_iteration < len(iterations):
                self.ax.plot(self.best_iteration, self.best_residual, 'ro', markersize=8, 
                           label=f'Best: iter {self.best_iteration}')
            
            # Add status indicators
            if self.is_stagnating:
                self.ax.axhline(y=self.data.residual_norms[-1], color='orange', 
                               linestyle='--', alpha=0.7, label='Stagnating')
            
            if self.is_diverging:
                self.ax.text(0.7, 0.9, 'DIVERGING', transform=self.ax.transAxes, 
                            color='red', fontweight='bold', fontsize=14)
            
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Residual Norm')
            self.ax.set_title(f'Convergence Monitor - Rate: {self.data.get_convergence_rate():.3f}')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            logger.warning(f"Plot update failed: {e}")
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on current convergence behavior."""
        recommendations = []
        
        if self.is_diverging:
            recommendations.append("Solver is diverging - consider reducing time step or relaxation parameter")
            recommendations.append("Check initial conditions and boundary conditions")
        
        elif self.is_stagnating:
            recommendations.append("Convergence is stagnating - consider:")
            recommendations.append("  - Switching to higher precision")
            recommendations.append("  - Using different preconditioning")
            recommendations.append("  - Adjusting solver parameters")
        
        elif len(self.data.residual_norms) > 5:
            rate = self.data.get_convergence_rate()
            if rate > 0.95:
                recommendations.append("Slow convergence - consider better preconditioning")
            elif rate < 0.3:
                recommendations.append("Fast convergence - solver parameters are well-tuned")
        
        return recommendations
    
    def save_convergence_plot(self, filename: str) -> None:
        """Save convergence plot to file."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            iterations = list(range(len(self.data.residual_norms)))
            ax.semilogy(iterations, self.data.residual_norms, 'b-', marker='o', markersize=3)
            
            # Add precision level information if available
            precision_changes = []
            current_precision = None
            
            for i, precision in enumerate(self.data.precision_levels):
                if precision != current_precision:
                    precision_changes.append((i, precision))
                    current_precision = precision
            
            # Mark precision changes
            for i, precision in precision_changes:
                ax.axvline(x=i, color='red', linestyle=':', alpha=0.7)
                ax.text(i, max(self.data.residual_norms) * 0.5, precision, 
                       rotation=90, verticalalignment='center')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual Norm')
            ax.set_title('Convergence History')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved convergence plot to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save convergence plot: {e}")
    
    def reset(self) -> None:
        """Reset monitor state."""
        self.data.clear()
        self.is_stagnating = False
        self.is_diverging = False
        self.best_residual = float('inf')
        self.best_iteration = 0
        
        if self.enable_plotting and self.ax:
            self.ax.clear()