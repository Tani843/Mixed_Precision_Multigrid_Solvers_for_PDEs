"""Grid convergence study tools and comprehensive error analysis."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import optimize
import json

from .poisson_solver import PoissonSolver2D, PoissonProblem
from .heat_solver import HeatSolver2D, HeatProblem, TimeSteppingConfig
from .test_problems import PoissonTestProblems, HeatTestProblems
from ..core.grid import Grid

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceData:
    """Data structure for convergence study results."""
    problem_name: str
    solver_type: str
    grid_sizes: List[Tuple[int, int]]
    grid_spacings: List[float]
    errors: Dict[str, List[float]]  # Error type -> list of errors
    convergence_rates: Dict[str, List[float]]  # Error type -> list of rates
    theoretical_order: float
    achieved_order: Dict[str, float]  # Error type -> achieved order
    regression_data: Dict[str, Dict[str, float]]  # Regression statistics
    solver_info: List[Dict[str, Any]]
    additional_metrics: Dict[str, Any] = None


class GridConvergenceAnalyzer:
    """
    Comprehensive grid convergence analysis tool.
    
    Performs systematic convergence studies with error analysis,
    rate calculations, and statistical validation.
    """
    
    def __init__(self):
        """Initialize convergence analyzer."""
        self.convergence_studies: List[ConvergenceData] = []
        self.default_grid_sequences = {
            'geometric_2': [(9, 9), (17, 17), (33, 33), (65, 65), (129, 129)],
            'geometric_3': [(9, 9), (25, 25), (73, 73), (217, 217)],  # factor of ~3
            'geometric_sqrt2': [(9, 9), (13, 13), (19, 19), (27, 27), (39, 39), (55, 55)],
            'uniform': [(16, 16), (32, 32), (64, 64), (128, 128)],
            'anisotropic_x': [(9, 33), (17, 65), (33, 129), (65, 257)],
            'anisotropic_y': [(33, 9), (65, 17), (129, 33), (257, 65)]
        }
        
        logger.info("Grid convergence analyzer initialized")
    
    def run_poisson_convergence_study(
        self,
        solver_config: Dict[str, Any],
        problem_name: str,
        grid_sequence: Optional[List[Tuple[int, int]]] = None,
        theoretical_order: float = 2.0,
        custom_problem: Optional[PoissonProblem] = None
    ) -> ConvergenceData:
        """
        Run comprehensive convergence study for Poisson equation.
        
        Args:
            solver_config: Configuration for PoissonSolver2D
            problem_name: Name of test problem or 'custom'
            grid_sequence: Sequence of (nx, ny) grid sizes
            theoretical_order: Expected convergence order
            custom_problem: Custom problem if problem_name is 'custom'
            
        Returns:
            Convergence analysis data
        """
        logger.info(f"Running Poisson convergence study: {problem_name}")
        
        # Get test problem
        if problem_name == 'custom':
            if custom_problem is None:
                raise ValueError("Custom problem must be provided")
            problem = custom_problem
        else:
            test_problems = PoissonTestProblems()
            problem = test_problems.get_problem(problem_name)
        
        if not problem.analytical_solution:
            raise ValueError("Convergence study requires analytical solution")
        
        # Use default grid sequence if none provided
        if grid_sequence is None:
            grid_sequence = self.default_grid_sequences['geometric_2']
        
        # Create solver
        solver = PoissonSolver2D(**solver_config)
        
        # Run convergence study
        results = []
        for nx, ny in grid_sequence:
            logger.debug(f"Solving on {nx}x{ny} grid")
            result = solver.solve_poisson_problem(problem, nx, ny)
            results.append(result)
        
        # Analyze convergence
        convergence_data = self._analyze_poisson_convergence(
            results, problem_name, solver_config['solver_type'], 
            theoretical_order, grid_sequence
        )
        
        # Store results
        self.convergence_studies.append(convergence_data)
        
        logger.info(f"Poisson convergence study completed. Achieved orders: "
                   f"L2={convergence_data.achieved_order['l2_error']:.2f}, "
                   f"Max={convergence_data.achieved_order['max_error']:.2f}")
        
        return convergence_data
    
    def run_heat_convergence_study(
        self,
        solver_config: Dict[str, Any],
        time_config: TimeSteppingConfig,
        problem_name: str,
        grid_sequence: Optional[List[Tuple[int, int]]] = None,
        theoretical_order: float = 2.0,
        custom_problem: Optional[HeatProblem] = None
    ) -> ConvergenceData:
        """
        Run comprehensive convergence study for Heat equation.
        
        Args:
            solver_config: Configuration for HeatSolver2D
            time_config: Time stepping configuration
            problem_name: Name of test problem or 'custom'
            grid_sequence: Sequence of (nx, ny) grid sizes
            theoretical_order: Expected spatial convergence order
            custom_problem: Custom problem if problem_name is 'custom'
            
        Returns:
            Convergence analysis data
        """
        logger.info(f"Running Heat convergence study: {problem_name}")
        
        # Get test problem
        if problem_name == 'custom':
            if custom_problem is None:
                raise ValueError("Custom problem must be provided")
            problem = custom_problem
        else:
            test_problems = HeatTestProblems()
            problem = test_problems.get_problem(problem_name)
        
        if not problem.analytical_solution:
            raise ValueError("Convergence study requires analytical solution")
        
        # Use default grid sequence if none provided
        if grid_sequence is None:
            grid_sequence = self.default_grid_sequences['geometric_2'][:4]  # Smaller for heat
        
        # Create solver
        solver = HeatSolver2D(**solver_config)
        
        # Run convergence study
        results = []
        for nx, ny in grid_sequence:
            logger.debug(f"Solving heat equation on {nx}x{ny} grid")
            result = solver.solve_heat_problem(problem, nx, ny, time_config)
            results.append(result)
        
        # Analyze convergence
        convergence_data = self._analyze_heat_convergence(
            results, problem_name, solver_config['solver_type'], 
            theoretical_order, grid_sequence, time_config
        )
        
        # Store results
        self.convergence_studies.append(convergence_data)
        
        logger.info(f"Heat convergence study completed. Achieved orders: "
                   f"L2={convergence_data.achieved_order['l2_error']:.2f}, "
                   f"Max={convergence_data.achieved_order['max_error']:.2f}")
        
        return convergence_data
    
    def _analyze_poisson_convergence(
        self,
        results: List[Dict[str, Any]],
        problem_name: str,
        solver_type: str,
        theoretical_order: float,
        grid_sequence: List[Tuple[int, int]]
    ) -> ConvergenceData:
        """Analyze convergence data for Poisson equation."""
        
        # Extract grid spacings and errors
        grid_spacings = []
        errors = {'l2_error': [], 'max_error': [], 'h1_semi_error': []}
        solver_info = []
        
        for result in results:
            # Grid spacing (minimum of hx, hy)
            hx, hy = result['errors']['grid_spacing']
            h = min(hx, hy)
            grid_spacings.append(h)
            
            # Errors
            for error_type in errors.keys():
                if error_type in result['errors']:
                    errors[error_type].append(result['errors'][error_type])
                else:
                    errors[error_type].append(np.nan)
            
            # Solver info
            solver_info.append(result['solver_info'])
        
        # Calculate convergence rates
        convergence_rates = {}
        achieved_order = {}
        regression_data = {}
        
        for error_type, error_values in errors.items():
            if len(error_values) >= 2 and not any(np.isnan(error_values)):
                rates = self._calculate_convergence_rates(grid_spacings, error_values)
                regression_stats = self._perform_regression_analysis(grid_spacings, error_values)
                
                convergence_rates[error_type] = rates
                achieved_order[error_type] = regression_stats['slope']
                regression_data[error_type] = regression_stats
            else:
                convergence_rates[error_type] = []
                achieved_order[error_type] = 0.0
                regression_data[error_type] = {}
        
        # Additional metrics
        additional_metrics = {
            'total_solve_time': sum(r['solve_time'] for r in results),
            'avg_iterations': np.mean([r['solver_info']['iterations'] for r in results]),
            'iteration_efficiency': self._calculate_iteration_efficiency(results, grid_sequence)
        }
        
        return ConvergenceData(
            problem_name=problem_name,
            solver_type=solver_type,
            grid_sizes=grid_sequence,
            grid_spacings=grid_spacings,
            errors=errors,
            convergence_rates=convergence_rates,
            theoretical_order=theoretical_order,
            achieved_order=achieved_order,
            regression_data=regression_data,
            solver_info=solver_info,
            additional_metrics=additional_metrics
        )
    
    def _analyze_heat_convergence(
        self,
        results: List[Dict[str, Any]],
        problem_name: str,
        solver_type: str,
        theoretical_order: float,
        grid_sequence: List[Tuple[int, int]],
        time_config: TimeSteppingConfig
    ) -> ConvergenceData:
        """Analyze convergence data for Heat equation."""
        
        # Extract grid spacings and errors
        grid_spacings = []
        errors = {'l2_error': [], 'max_error': []}
        solver_info = []
        
        for result in results:
            # Grid spacing
            hx, hy = result['errors']['grid_spacing']
            h = min(hx, hy)
            grid_spacings.append(h)
            
            # Errors
            for error_type in errors.keys():
                if error_type in result['errors']:
                    errors[error_type].append(result['errors'][error_type])
                else:
                    errors[error_type].append(np.nan)
            
            # Solver info (aggregate time stepping info)
            solver_info.append({
                'total_steps': result['total_steps'],
                'avg_mg_iterations': result['avg_mg_iterations'],
                'total_time': result['total_time']
            })
        
        # Calculate convergence rates
        convergence_rates = {}
        achieved_order = {}
        regression_data = {}
        
        for error_type, error_values in errors.items():
            if len(error_values) >= 2 and not any(np.isnan(error_values)):
                rates = self._calculate_convergence_rates(grid_spacings, error_values)
                regression_stats = self._perform_regression_analysis(grid_spacings, error_values)
                
                convergence_rates[error_type] = rates
                achieved_order[error_type] = regression_stats['slope']
                regression_data[error_type] = regression_stats
            else:
                convergence_rates[error_type] = []
                achieved_order[error_type] = 0.0
                regression_data[error_type] = {}
        
        # Additional metrics for heat equation
        additional_metrics = {
            'total_solve_time': sum(r['total_time'] for r in results),
            'total_time_steps': sum(r['total_steps'] for r in results),
            'avg_mg_iterations_per_step': np.mean([r['avg_mg_iterations'] for r in results]),
            'time_stepping_method': time_config.method.value,
            'time_step_size': time_config.dt
        }
        
        return ConvergenceData(
            problem_name=f"heat_{problem_name}",
            solver_type=solver_type,
            grid_sizes=grid_sequence,
            grid_spacings=grid_spacings,
            errors=errors,
            convergence_rates=convergence_rates,
            theoretical_order=theoretical_order,
            achieved_order=achieved_order,
            regression_data=regression_data,
            solver_info=solver_info,
            additional_metrics=additional_metrics
        )
    
    def _calculate_convergence_rates(
        self, 
        grid_spacings: List[float], 
        errors: List[float]
    ) -> List[float]:
        """Calculate pointwise convergence rates between consecutive grids."""
        if len(grid_spacings) < 2:
            return []
        
        rates = []
        for i in range(1, len(grid_spacings)):
            h_ratio = grid_spacings[i-1] / grid_spacings[i]
            error_ratio = errors[i-1] / errors[i]
            
            if error_ratio > 0 and h_ratio > 1:
                rate = np.log(error_ratio) / np.log(h_ratio)
                rates.append(rate)
            else:
                rates.append(0.0)
        
        return rates
    
    def _perform_regression_analysis(
        self, 
        grid_spacings: List[float], 
        errors: List[float]
    ) -> Dict[str, float]:
        """Perform linear regression on log(error) vs log(h) to find convergence rate."""
        if len(grid_spacings) < 2:
            return {}
        
        # Convert to log space
        log_h = np.log(grid_spacings)
        log_error = np.log(errors)
        
        # Remove any inf or nan values
        valid_mask = np.isfinite(log_h) & np.isfinite(log_error)
        if np.sum(valid_mask) < 2:
            return {}
        
        log_h_clean = log_h[valid_mask]
        log_error_clean = log_error[valid_mask]
        
        # Linear regression: log(error) = slope * log(h) + intercept
        slope, intercept, r_value, p_value, std_err = optimize.linregress(log_h_clean, log_error_clean)
        
        # Calculate coefficient of determination and confidence intervals
        n = len(log_h_clean)
        residuals = log_error_clean - (slope * log_h_clean + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_error_clean - np.mean(log_error_clean))**2)
        
        return {
            'slope': slope,  # This is the convergence rate
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'standard_error': std_err,
            'n_points': n,
            'residual_sum_squares': ss_res,
            'confidence_95': 1.96 * std_err  # Approximate 95% confidence interval
        }
    
    def _calculate_iteration_efficiency(
        self, 
        results: List[Dict[str, Any]], 
        grid_sequence: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """Calculate multigrid iteration efficiency metrics."""
        grid_sizes = [nx * ny for nx, ny in grid_sequence]
        iterations = [r['solver_info']['iterations'] for r in results]
        
        # Check if iterations are approximately O(1) (multigrid optimality)
        if len(grid_sizes) >= 2:
            # Fit linear model: iterations vs log(grid_size)
            log_grid_sizes = np.log(grid_sizes)
            try:
                slope, intercept, r_value, _, _ = optimize.linregress(log_grid_sizes, iterations)
                
                return {
                    'iteration_growth_rate': slope,  # Should be ~0 for optimal multigrid
                    'r_squared': r_value**2,
                    'avg_iterations': np.mean(iterations),
                    'iteration_variation': np.std(iterations) / np.mean(iterations)
                }
            except:
                pass
        
        return {
            'iteration_growth_rate': 0.0,
            'r_squared': 0.0,
            'avg_iterations': np.mean(iterations) if iterations else 0.0,
            'iteration_variation': 0.0
        }
    
    def compare_convergence_studies(
        self, 
        study_names: List[str], 
        error_type: str = 'l2_error'
    ) -> Dict[str, Any]:
        """
        Compare multiple convergence studies side by side.
        
        Args:
            study_names: Names/identifiers of studies to compare
            error_type: Type of error to compare ('l2_error', 'max_error', etc.)
            
        Returns:
            Comparison analysis
        """
        studies = []
        for study in self.convergence_studies:
            if any(name in study.problem_name for name in study_names):
                studies.append(study)
        
        if len(studies) < 2:
            logger.warning(f"Need at least 2 studies to compare, found {len(studies)}")
            return {}
        
        comparison_data = {
            'studies': [],
            'error_type': error_type,
            'convergence_comparison': {},
            'performance_comparison': {}
        }
        
        for study in studies:
            if error_type in study.achieved_order:
                comparison_data['studies'].append({
                    'name': study.problem_name,
                    'solver_type': study.solver_type,
                    'achieved_order': study.achieved_order[error_type],
                    'theoretical_order': study.theoretical_order,
                    'r_squared': study.regression_data.get(error_type, {}).get('r_squared', 0.0),
                    'total_solve_time': study.additional_metrics.get('total_solve_time', 0.0)
                })
        
        # Rank studies by convergence rate accuracy
        sorted_studies = sorted(
            comparison_data['studies'], 
            key=lambda x: abs(x['achieved_order'] - x['theoretical_order'])
        )
        
        comparison_data['convergence_comparison'] = {
            'best_convergence': sorted_studies[0]['name'] if sorted_studies else None,
            'convergence_ranking': [s['name'] for s in sorted_studies]
        }
        
        # Performance comparison
        if sorted_studies:
            fastest_study = min(sorted_studies, key=lambda x: x['total_solve_time'])
            comparison_data['performance_comparison'] = {
                'fastest_solver': fastest_study['name'],
                'performance_ranking': sorted(sorted_studies, key=lambda x: x['total_solve_time'])
            }
        
        return comparison_data
    
    def generate_convergence_report(
        self, 
        study: ConvergenceData, 
        save_to_file: Optional[str] = None
    ) -> str:
        """
        Generate detailed convergence analysis report.
        
        Args:
            study: Convergence study data
            save_to_file: Optional filename to save report
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "GRID CONVERGENCE ANALYSIS REPORT",
            "=" * 50,
            "",
            f"Problem: {study.problem_name}",
            f"Solver: {study.solver_type}",
            f"Theoretical Order: {study.theoretical_order}",
            ""
        ]
        
        # Grid sequence information
        report_lines.extend([
            "GRID SEQUENCE:",
            "-" * 20
        ])
        
        for i, (grid_size, h) in enumerate(zip(study.grid_sizes, study.grid_spacings)):
            report_lines.append(f"Grid {i+1}: {grid_size[0]}x{grid_size[1]} (h = {h:.6f})")
        
        report_lines.append("")
        
        # Convergence analysis for each error type
        for error_type, achieved_order in study.achieved_order.items():
            if error_type in study.errors and len(study.errors[error_type]) > 0:
                report_lines.extend([
                    f"{error_type.upper()} CONVERGENCE ANALYSIS:",
                    "-" * (len(error_type) + 22)
                ])
                
                # Error values
                errors = study.errors[error_type]
                for i, (grid_size, error) in enumerate(zip(study.grid_sizes, errors)):
                    if not np.isnan(error):
                        report_lines.append(f"Grid {grid_size[0]}x{grid_size[1]:3d}: {error:.6e}")
                
                # Convergence rates
                if error_type in study.convergence_rates:
                    rates = study.convergence_rates[error_type]
                    report_lines.append("")
                    report_lines.append("Pointwise convergence rates:")
                    for i, rate in enumerate(rates):
                        grid_from = study.grid_sizes[i]
                        grid_to = study.grid_sizes[i+1]
                        report_lines.append(f"  {grid_from[0]}x{grid_from[1]} -> {grid_to[0]}x{grid_to[1]}: {rate:.2f}")
                
                # Regression analysis
                if error_type in study.regression_data:
                    reg_data = study.regression_data[error_type]
                    if reg_data:
                        report_lines.extend([
                            "",
                            "Regression analysis (log-log fit):",
                            f"  Achieved order: {achieved_order:.3f}",
                            f"  R-squared: {reg_data.get('r_squared', 0):.4f}",
                            f"  Standard error: {reg_data.get('standard_error', 0):.4f}",
                            f"  95% confidence: ±{reg_data.get('confidence_95', 0):.3f}",
                        ])
                        
                        # Convergence quality assessment
                        order_error = abs(achieved_order - study.theoretical_order)
                        if order_error < 0.1:
                            quality = "EXCELLENT"
                        elif order_error < 0.3:
                            quality = "GOOD"
                        elif order_error < 0.5:
                            quality = "ACCEPTABLE"
                        else:
                            quality = "POOR"
                        
                        report_lines.append(f"  Convergence quality: {quality}")
                
                report_lines.append("")
        
        # Solver performance analysis
        if study.solver_info:
            report_lines.extend([
                "SOLVER PERFORMANCE:",
                "-" * 20
            ])
            
            avg_iterations = np.mean([info.get('iterations', 0) for info in study.solver_info])
            std_iterations = np.std([info.get('iterations', 0) for info in study.solver_info])
            
            report_lines.extend([
                f"Average iterations: {avg_iterations:.1f} ± {std_iterations:.1f}",
                f"Total solve time: {study.additional_metrics.get('total_solve_time', 0):.3f}s"
            ])
            
            # Multigrid efficiency
            if 'iteration_efficiency' in study.additional_metrics:
                eff = study.additional_metrics['iteration_efficiency']
                growth_rate = eff.get('iteration_growth_rate', 0)
                
                if abs(growth_rate) < 0.1:
                    efficiency = "OPTIMAL (O(1) iterations)"
                elif abs(growth_rate) < 0.5:
                    efficiency = "GOOD"
                else:
                    efficiency = "SUBOPTIMAL"
                
                report_lines.append(f"Multigrid efficiency: {efficiency}")
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write(report)
                logger.info(f"Convergence report saved to {save_to_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def plot_convergence_study(
        self, 
        study: ConvergenceData, 
        error_types: Optional[List[str]] = None,
        save_figure: Optional[str] = None
    ) -> None:
        """
        Plot convergence study results.
        
        Args:
            study: Convergence study data
            error_types: List of error types to plot
            save_figure: Optional filename to save figure
        """
        if error_types is None:
            error_types = ['l2_error', 'max_error']
        
        # Filter available error types
        available_types = [et for et in error_types if et in study.errors and len(study.errors[et]) > 0]
        
        if not available_types:
            logger.warning("No error data available for plotting")
            return
        
        try:
            fig, axes = plt.subplots(1, len(available_types), figsize=(6*len(available_types), 5))
            if len(available_types) == 1:
                axes = [axes]
            
            for i, error_type in enumerate(available_types):
                ax = axes[i]
                
                # Plot error vs grid spacing
                h = study.grid_spacings
                errors = study.errors[error_type]
                
                # Remove NaN values
                valid_mask = ~np.isnan(errors)
                h_valid = np.array(h)[valid_mask]
                errors_valid = np.array(errors)[valid_mask]
                
                if len(h_valid) >= 2:
                    ax.loglog(h_valid, errors_valid, 'o-', linewidth=2, markersize=8, 
                             label=f'{error_type.replace("_", " ").title()}')
                    
                    # Plot theoretical line if regression data available
                    if error_type in study.regression_data and study.regression_data[error_type]:
                        reg_data = study.regression_data[error_type]
                        theoretical_slope = study.theoretical_order
                        achieved_slope = reg_data['slope']
                        
                        # Theoretical line
                        h_theory = np.array([h_valid[0], h_valid[-1]])
                        error_theory = errors_valid[0] * (h_theory / h_valid[0])**theoretical_slope
                        ax.loglog(h_theory, error_theory, '--', alpha=0.7, 
                                 label=f'Theoretical O(h^{theoretical_slope})')
                        
                        # Achieved line
                        error_achieved = errors_valid[0] * (h_theory / h_valid[0])**achieved_slope
                        ax.loglog(h_theory, error_achieved, ':', alpha=0.8,
                                 label=f'Achieved O(h^{achieved_slope:.2f})')
                
                ax.set_xlabel('Grid Spacing (h)')
                ax.set_ylabel(f'{error_type.replace("_", " ").title()}')
                ax.set_title(f'{study.problem_name}: {error_type.replace("_", " ").title()} Convergence')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            
            if save_figure:
                plt.savefig(save_figure, dpi=300, bbox_inches='tight')
                logger.info(f"Convergence plot saved to {save_figure}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to create convergence plot: {e}")
    
    def export_convergence_data(
        self, 
        study: ConvergenceData, 
        filename: str, 
        format: str = 'json'
    ) -> None:
        """
        Export convergence data to file.
        
        Args:
            study: Convergence study data
            filename: Output filename
            format: Export format ('json', 'csv')
        """
        try:
            if format.lower() == 'json':
                # Convert numpy arrays to lists for JSON serialization
                export_data = {
                    'problem_name': study.problem_name,
                    'solver_type': study.solver_type,
                    'grid_sizes': study.grid_sizes,
                    'grid_spacings': study.grid_spacings,
                    'errors': {k: [float(v) if not np.isnan(v) else None for v in vals] 
                              for k, vals in study.errors.items()},
                    'convergence_rates': {k: [float(v) for v in vals] 
                                        for k, vals in study.convergence_rates.items()},
                    'theoretical_order': study.theoretical_order,
                    'achieved_order': {k: float(v) for k, v in study.achieved_order.items()},
                    'regression_data': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                                           for kk, vv in v.items()} 
                                       for k, v in study.regression_data.items()},
                    'additional_metrics': study.additional_metrics
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
            elif format.lower() == 'csv':
                import pandas as pd
                
                # Create CSV data
                csv_data = []
                for i, (grid_size, h) in enumerate(zip(study.grid_sizes, study.grid_spacings)):
                    row = {
                        'grid_nx': grid_size[0],
                        'grid_ny': grid_size[1],
                        'grid_spacing': h
                    }
                    
                    for error_type, errors in study.errors.items():
                        if i < len(errors):
                            row[error_type] = errors[i]
                    
                    csv_data.append(row)
                
                df = pd.DataFrame(csv_data)
                df.to_csv(filename, index=False)
            
            logger.info(f"Convergence data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export convergence data: {e}")


def run_comprehensive_convergence_analysis() -> Dict[str, Any]:
    """
    Run comprehensive convergence analysis for key test problems.
    
    Returns:
        Dictionary of convergence analysis results
    """
    logger.info("Starting comprehensive convergence analysis")
    
    analyzer = GridConvergenceAnalyzer()
    
    # Poisson solver configuration
    poisson_config = {
        'solver_type': 'multigrid',
        'max_levels': 6,
        'tolerance': 1e-10,
        'use_gpu': False
    }
    
    # Heat solver configuration
    heat_config = {
        'solver_type': 'multigrid',
        'max_levels': 5,
        'tolerance': 1e-8,
        'use_gpu': False
    }
    
    # Test problems to analyze
    poisson_problems = ['trigonometric', 'polynomial', 'high_frequency']
    heat_problems = ['pure_diffusion', 'separable']
    
    results = {
        'poisson_studies': [],
        'heat_studies': [],
        'summary': {}
    }
    
    # Poisson convergence studies
    for problem_name in poisson_problems:
        try:
            study = analyzer.run_poisson_convergence_study(
                poisson_config, problem_name, theoretical_order=2.0
            )
            results['poisson_studies'].append(study)
            logger.info(f"Completed Poisson study: {problem_name}")
        except Exception as e:
            logger.error(f"Failed Poisson study {problem_name}: {e}")
    
    # Heat convergence studies
    from .heat_solver import TimeSteppingConfig, TimeSteppingMethod
    
    time_config = TimeSteppingConfig(
        method=TimeSteppingMethod.BACKWARD_EULER,
        dt=0.01,
        t_final=0.1
    )
    
    for problem_name in heat_problems:
        try:
            study = analyzer.run_heat_convergence_study(
                heat_config, time_config, problem_name, theoretical_order=2.0
            )
            results['heat_studies'].append(study)
            logger.info(f"Completed Heat study: {problem_name}")
        except Exception as e:
            logger.error(f"Failed Heat study {problem_name}: {e}")
    
    # Generate summary
    all_studies = results['poisson_studies'] + results['heat_studies']
    if all_studies:
        results['summary'] = {
            'total_studies': len(all_studies),
            'poisson_studies': len(results['poisson_studies']),
            'heat_studies': len(results['heat_studies']),
            'convergence_analyzer': analyzer
        }
    
    logger.info("Comprehensive convergence analysis completed")
    
    return results