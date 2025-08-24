"""Mixed-precision effectiveness analysis and precision trade-off studies."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import time
import matplotlib.pyplot as plt

from .poisson_solver import PoissonSolver2D, PoissonProblem
from .heat_solver import HeatSolver2D, HeatProblem, TimeSteppingConfig
from .test_problems import PoissonTestProblems, HeatTestProblems
from .performance_analysis import PerformanceResult

logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Precision levels for analysis."""
    SINGLE = "float32"
    DOUBLE = "float64"
    MIXED = "mixed"
    ADAPTIVE = "adaptive"


@dataclass
class MixedPrecisionResult:
    """Results from mixed-precision analysis."""
    problem_name: str
    solver_type: str
    precision_config: Dict[str, Any]
    grid_size: Tuple[int, int]
    solve_time: float
    memory_usage: float
    accuracy_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    performance_gain: float
    accuracy_loss: float
    precision_efficiency: float


@dataclass
class PrecisionComparison:
    """Comparison between different precision levels."""
    problem_name: str
    grid_size: Tuple[int, int]
    single_result: Optional[MixedPrecisionResult]
    double_result: Optional[MixedPrecisionResult]
    mixed_result: Optional[MixedPrecisionResult]
    speedup_mixed_vs_double: float
    speedup_single_vs_double: float
    memory_saving_mixed: float
    memory_saving_single: float
    accuracy_comparison: Dict[str, Dict[str, float]]


class MixedPrecisionAnalyzer:
    """
    Comprehensive mixed-precision analysis tool.
    
    Analyzes the trade-offs between computational performance 
    and numerical accuracy in mixed-precision solvers.
    """
    
    def __init__(self):
        """Initialize mixed-precision analyzer."""
        self.precision_results: List[MixedPrecisionResult] = []
        self.comparison_results: List[PrecisionComparison] = []
        
        # Default precision configurations
        self.precision_configs = {
            'double': {
                'use_mixed_precision': False,
                'primary_precision': 'float64',
                'secondary_precision': None,
                'precision_threshold': None
            },
            'single': {
                'use_mixed_precision': False,
                'primary_precision': 'float32',
                'secondary_precision': None,
                'precision_threshold': None
            },
            'mixed_conservative': {
                'use_mixed_precision': True,
                'primary_precision': 'float32',
                'secondary_precision': 'float64',
                'precision_threshold': 1e-6,
                'refinement_strategy': 'conservative'
            },
            'mixed_aggressive': {
                'use_mixed_precision': True,
                'primary_precision': 'float32',
                'secondary_precision': 'float64',
                'precision_threshold': 1e-4,
                'refinement_strategy': 'aggressive'
            },
            'adaptive': {
                'use_mixed_precision': True,
                'primary_precision': 'float32',
                'secondary_precision': 'float64',
                'precision_threshold': 'adaptive',
                'refinement_strategy': 'adaptive'
            }
        }
        
        logger.info("Mixed-precision analyzer initialized")
    
    def analyze_precision_trade_offs(
        self,
        problem_names: List[str],
        grid_sizes: List[Tuple[int, int]],
        solver_type: str = "multigrid",
        precision_levels: Optional[List[str]] = None,
        num_runs: int = 3
    ) -> List[PrecisionComparison]:
        """
        Comprehensive analysis of precision trade-offs.
        
        Args:
            problem_names: List of problems to analyze
            grid_sizes: List of grid sizes to test
            solver_type: Base solver type
            precision_levels: List of precision configurations to compare
            num_runs: Number of runs for averaging
            
        Returns:
            List of precision comparison results
        """
        if precision_levels is None:
            precision_levels = ['double', 'single', 'mixed_conservative', 'mixed_aggressive']
        
        logger.info(f"Analyzing precision trade-offs: {len(problem_names)} problems, "
                   f"{len(grid_sizes)} grid sizes, {len(precision_levels)} precision levels")
        
        test_problems = PoissonTestProblems()
        comparison_results = []
        
        for problem_name in problem_names:
            for grid_size in grid_sizes:
                logger.debug(f"Analyzing {problem_name} on {grid_size[0]}x{grid_size[1]} grid")
                
                problem = test_problems.get_problem(problem_name)
                precision_results = {}
                
                # Run analysis for each precision level
                for precision_level in precision_levels:
                    try:
                        result = self._run_precision_analysis(
                            solver_type, problem, grid_size, precision_level, num_runs
                        )
                        precision_results[precision_level] = result
                        self.precision_results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Precision analysis failed for {precision_level}: {e}")
                        precision_results[precision_level] = None
                
                # Create comparison
                comparison = self._create_precision_comparison(
                    problem_name, grid_size, precision_results
                )
                comparison_results.append(comparison)
                self.comparison_results.append(comparison)
        
        logger.info(f"Precision trade-off analysis completed: {len(comparison_results)} comparisons")
        return comparison_results
    
    def _run_precision_analysis(
        self,
        solver_type: str,
        problem: PoissonProblem,
        grid_size: Tuple[int, int],
        precision_level: str,
        num_runs: int
    ) -> MixedPrecisionResult:
        """Run analysis for a specific precision configuration."""
        
        if precision_level not in self.precision_configs:
            raise ValueError(f"Unknown precision level: {precision_level}")
        
        precision_config = self.precision_configs[precision_level].copy()
        
        # Create solver configuration
        solver_config = {
            'solver_type': solver_type,
            'max_levels': 6,
            'tolerance': 1e-8,
            'use_gpu': False,  # CPU analysis first
            'enable_mixed_precision': precision_config.get('use_mixed_precision', False)
        }
        
        # Add precision-specific configurations
        if 'primary_precision' in precision_config:
            solver_config['primary_precision'] = precision_config['primary_precision']
        if 'secondary_precision' in precision_config:
            solver_config['secondary_precision'] = precision_config['secondary_precision']
        if 'precision_threshold' in precision_config:
            solver_config['precision_threshold'] = precision_config['precision_threshold']
        
        # Create solver
        solver = PoissonSolver2D(**solver_config)
        
        # Monitor memory before
        memory_before = self._estimate_memory_usage()
        
        # Run multiple times for averaging
        solve_times = []
        accuracy_results = []
        convergence_results = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            
            # Solve problem
            result = solver.solve_poisson_problem(problem, grid_size[0], grid_size[1])
            
            end_time = time.perf_counter()
            solve_times.append(end_time - start_time)
            
            # Collect accuracy and convergence info
            if 'errors' in result:
                accuracy_results.append(result['errors'])
            
            if 'solver_info' in result:
                convergence_results.append(result['solver_info'])
        
        # Monitor memory after
        memory_after = self._estimate_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Calculate averages
        avg_solve_time = np.mean(solve_times)
        
        # Aggregate accuracy metrics
        accuracy_metrics = self._aggregate_accuracy_metrics(accuracy_results)
        
        # Aggregate convergence info
        convergence_info = self._aggregate_convergence_info(convergence_results)
        
        # Calculate performance metrics (compared to double precision baseline)
        performance_gain = self._calculate_performance_gain(precision_level, avg_solve_time)
        accuracy_loss = self._calculate_accuracy_loss(precision_level, accuracy_metrics)
        precision_efficiency = self._calculate_precision_efficiency(performance_gain, accuracy_loss)
        
        return MixedPrecisionResult(
            problem_name=problem.name,
            solver_type=solver_type,
            precision_config=precision_config,
            grid_size=grid_size,
            solve_time=avg_solve_time,
            memory_usage=memory_usage,
            accuracy_metrics=accuracy_metrics,
            convergence_info=convergence_info,
            performance_gain=performance_gain,
            accuracy_loss=accuracy_loss,
            precision_efficiency=precision_efficiency
        )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)  # MB
        except:
            return 0.0
    
    def _aggregate_accuracy_metrics(self, accuracy_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate accuracy metrics from multiple runs."""
        if not accuracy_results:
            return {}
        
        metrics = {}
        for key in accuracy_results[0].keys():
            if key != 'grid_spacing':  # Skip non-numeric fields
                values = [result[key] for result in accuracy_results if key in result]
                if values:
                    metrics[f"avg_{key}"] = np.mean(values)
                    metrics[f"std_{key}"] = np.std(values)
        
        return metrics
    
    def _aggregate_convergence_info(self, convergence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate convergence information from multiple runs."""
        if not convergence_results:
            return {}
        
        iterations_list = [r.get('iterations', 0) for r in convergence_results]
        residuals_list = [r.get('final_residual', 0) for r in convergence_results if 'final_residual' in r]
        
        info = {
            'avg_iterations': np.mean(iterations_list) if iterations_list else 0,
            'std_iterations': np.std(iterations_list) if iterations_list else 0,
            'max_iterations': np.max(iterations_list) if iterations_list else 0,
            'min_iterations': np.min(iterations_list) if iterations_list else 0
        }
        
        if residuals_list:
            info['avg_final_residual'] = np.mean(residuals_list)
            info['std_final_residual'] = np.std(residuals_list)
        
        return info
    
    def _calculate_performance_gain(self, precision_level: str, solve_time: float) -> float:
        """Calculate performance gain relative to double precision."""
        # This would ideally compare against a reference double precision run
        # For now, estimate based on theoretical speedups
        
        if precision_level == 'single':
            return 1.8  # Typical single precision speedup
        elif precision_level.startswith('mixed'):
            return 1.4  # Conservative mixed precision speedup
        elif precision_level == 'double':
            return 1.0  # Baseline
        else:
            return 1.0
    
    def _calculate_accuracy_loss(self, precision_level: str, accuracy_metrics: Dict[str, float]) -> float:
        """Calculate accuracy loss relative to double precision."""
        # This would ideally compare against reference solution
        # For now, estimate based on precision level
        
        base_error = accuracy_metrics.get('avg_l2_error', 1e-8)
        
        if precision_level == 'single':
            # Single precision has ~7 digits accuracy
            return max(0.0, np.log10(base_error / 1e-6))
        elif precision_level.startswith('mixed'):
            # Mixed precision should maintain most accuracy
            return max(0.0, np.log10(base_error / 1e-7))
        else:
            return 0.0  # Double precision is reference
    
    def _calculate_precision_efficiency(self, performance_gain: float, accuracy_loss: float) -> float:
        """Calculate overall precision efficiency score."""
        # Efficiency metric: performance gain per unit of accuracy loss
        if accuracy_loss <= 0:
            return performance_gain * 10  # Very high efficiency if no accuracy loss
        else:
            return performance_gain / (1 + accuracy_loss)
    
    def _create_precision_comparison(
        self,
        problem_name: str,
        grid_size: Tuple[int, int],
        precision_results: Dict[str, Optional[MixedPrecisionResult]]
    ) -> PrecisionComparison:
        """Create comparison between different precision levels."""
        
        double_result = precision_results.get('double')
        single_result = precision_results.get('single')
        mixed_result = precision_results.get('mixed_conservative') or precision_results.get('mixed_aggressive')
        
        # Calculate speedups
        speedup_mixed_vs_double = 0.0
        speedup_single_vs_double = 0.0
        
        if double_result:
            if mixed_result:
                speedup_mixed_vs_double = double_result.solve_time / mixed_result.solve_time
            if single_result:
                speedup_single_vs_double = double_result.solve_time / single_result.solve_time
        
        # Calculate memory savings
        memory_saving_mixed = 0.0
        memory_saving_single = 0.0
        
        if double_result:
            if mixed_result and double_result.memory_usage > 0:
                memory_saving_mixed = (double_result.memory_usage - mixed_result.memory_usage) / double_result.memory_usage
            if single_result and double_result.memory_usage > 0:
                memory_saving_single = (double_result.memory_usage - single_result.memory_usage) / double_result.memory_usage
        
        # Accuracy comparison
        accuracy_comparison = {}
        
        if double_result:
            accuracy_comparison['double'] = double_result.accuracy_metrics
        if single_result:
            accuracy_comparison['single'] = single_result.accuracy_metrics
        if mixed_result:
            accuracy_comparison['mixed'] = mixed_result.accuracy_metrics
        
        return PrecisionComparison(
            problem_name=problem_name,
            grid_size=grid_size,
            single_result=single_result,
            double_result=double_result,
            mixed_result=mixed_result,
            speedup_mixed_vs_double=speedup_mixed_vs_double,
            speedup_single_vs_double=speedup_single_vs_double,
            memory_saving_mixed=memory_saving_mixed,
            memory_saving_single=memory_saving_single,
            accuracy_comparison=accuracy_comparison
        )
    
    def analyze_precision_convergence(
        self,
        problem_name: str,
        grid_sizes: List[Tuple[int, int]],
        precision_levels: List[str],
        tolerance_range: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze how precision affects convergence behavior across tolerances.
        
        Args:
            problem_name: Problem to analyze
            grid_sizes: Grid sizes to test
            precision_levels: Precision configurations
            tolerance_range: Range of solver tolerances to test
            
        Returns:
            Convergence analysis results
        """
        logger.info(f"Analyzing precision convergence for {problem_name}")
        
        test_problems = PoissonTestProblems()
        problem = test_problems.get_problem(problem_name)
        
        convergence_study = {
            'problem_name': problem_name,
            'grid_sizes': grid_sizes,
            'precision_levels': precision_levels,
            'tolerance_range': tolerance_range,
            'results': {}
        }
        
        for precision_level in precision_levels:
            precision_results = {}
            
            for grid_size in grid_sizes:
                tolerance_results = []
                
                for tolerance in tolerance_range:
                    # Create solver with specific tolerance
                    precision_config = self.precision_configs[precision_level].copy()
                    solver_config = {
                        'solver_type': 'multigrid',
                        'max_levels': 6,
                        'tolerance': tolerance,
                        'max_iterations': 200,
                        'use_gpu': False,
                        'enable_mixed_precision': precision_config.get('use_mixed_precision', False)
                    }
                    
                    try:
                        solver = PoissonSolver2D(**solver_config)
                        result = solver.solve_poisson_problem(problem, grid_size[0], grid_size[1])
                        
                        # Extract convergence metrics
                        convergence_metrics = {
                            'tolerance': tolerance,
                            'iterations': result['solver_info'].get('iterations', 0),
                            'final_residual': result['solver_info'].get('final_residual', 0),
                            'solve_time': result['solve_time'],
                            'converged': result['solver_info'].get('converged', False)
                        }
                        
                        if 'errors' in result:
                            convergence_metrics['l2_error'] = result['errors'].get('l2_error', 0)
                            convergence_metrics['max_error'] = result['errors'].get('max_error', 0)
                        
                        tolerance_results.append(convergence_metrics)
                        
                    except Exception as e:
                        logger.warning(f"Convergence analysis failed for tolerance {tolerance}: {e}")
                        tolerance_results.append({
                            'tolerance': tolerance,
                            'converged': False,
                            'error': str(e)
                        })
                
                precision_results[f"{grid_size[0]}x{grid_size[1]}"] = tolerance_results
            
            convergence_study['results'][precision_level] = precision_results
        
        # Analyze convergence patterns
        convergence_study['analysis'] = self._analyze_convergence_patterns(convergence_study['results'])
        
        return convergence_study
    
    def _analyze_convergence_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence patterns across precision levels."""
        analysis = {
            'convergence_reliability': {},
            'performance_vs_accuracy': {},
            'precision_limitations': {}
        }
        
        for precision_level, precision_data in results.items():
            reliability_scores = []
            
            for grid_size, tolerance_results in precision_data.items():
                converged_count = sum(1 for r in tolerance_results if r.get('converged', False))
                total_count = len(tolerance_results)
                reliability = converged_count / total_count if total_count > 0 else 0
                reliability_scores.append(reliability)
            
            analysis['convergence_reliability'][precision_level] = np.mean(reliability_scores)
        
        return analysis
    
    def generate_mixed_precision_report(
        self,
        comparison_results: Optional[List[PrecisionComparison]] = None,
        save_to_file: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive mixed-precision analysis report.
        
        Args:
            comparison_results: Specific comparison results to report
            save_to_file: Optional filename to save report
            
        Returns:
            Formatted report string
        """
        if comparison_results is None:
            comparison_results = self.comparison_results
        
        if not comparison_results:
            return "No mixed-precision analysis results available"
        
        report_lines = [
            "MIXED-PRECISION EFFECTIVENESS ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Analysis Summary:",
            f"Total comparisons: {len(comparison_results)}",
            f"Problems analyzed: {len(set(c.problem_name for c in comparison_results))}",
            f"Grid sizes tested: {len(set(c.grid_size for c in comparison_results))}",
            ""
        ]
        
        # Overall performance summary
        valid_comparisons = [c for c in comparison_results if c.double_result and c.mixed_result]
        
        if valid_comparisons:
            avg_speedup_mixed = np.mean([c.speedup_mixed_vs_double for c in valid_comparisons])
            avg_memory_saving = np.mean([c.memory_saving_mixed for c in valid_comparisons if c.memory_saving_mixed >= 0])
            
            report_lines.extend([
                "OVERALL MIXED-PRECISION PERFORMANCE:",
                "-" * 40,
                f"Average speedup (mixed vs double): {avg_speedup_mixed:.2f}x",
                f"Average memory saving: {avg_memory_saving:.1%}",
                ""
            ])
        
        # Single vs double precision comparison
        single_comparisons = [c for c in comparison_results if c.double_result and c.single_result]
        
        if single_comparisons:
            avg_speedup_single = np.mean([c.speedup_single_vs_double for c in single_comparisons])
            avg_memory_saving_single = np.mean([c.memory_saving_single for c in single_comparisons if c.memory_saving_single >= 0])
            
            report_lines.extend([
                "SINGLE vs DOUBLE PRECISION COMPARISON:",
                "-" * 40,
                f"Average speedup (single vs double): {avg_speedup_single:.2f}x",
                f"Average memory saving: {avg_memory_saving_single:.1%}",
                ""
            ])
        
        # Problem-specific analysis
        problems = set(c.problem_name for c in comparison_results)
        
        for problem in problems:
            problem_comparisons = [c for c in comparison_results if c.problem_name == problem]
            
            if not problem_comparisons:
                continue
            
            report_lines.extend([
                f"PROBLEM: {problem.upper()}",
                "-" * (10 + len(problem))
            ])
            
            for comp in problem_comparisons:
                grid_str = f"{comp.grid_size[0]}x{comp.grid_size[1]}"
                report_lines.append(f"Grid {grid_str}:")
                
                if comp.double_result and comp.mixed_result:
                    report_lines.extend([
                        f"  Mixed precision speedup: {comp.speedup_mixed_vs_double:.2f}x",
                        f"  Memory saving: {comp.memory_saving_mixed:.1%}",
                        f"  Solve time (double): {comp.double_result.solve_time:.4f}s",
                        f"  Solve time (mixed): {comp.mixed_result.solve_time:.4f}s"
                    ])
                
                # Accuracy comparison
                if comp.accuracy_comparison:
                    report_lines.append("  Accuracy comparison:")
                    
                    for precision, metrics in comp.accuracy_comparison.items():
                        l2_error = metrics.get('avg_l2_error', 0)
                        max_error = metrics.get('avg_max_error', 0)
                        report_lines.append(f"    {precision}: L2={l2_error:.2e}, Max={max_error:.2e}")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 15,
            ""
        ])
        
        if valid_comparisons:
            best_mixed_performance = max(valid_comparisons, key=lambda x: x.speedup_mixed_vs_double)
            worst_mixed_performance = min(valid_comparisons, key=lambda x: x.speedup_mixed_vs_double)
            
            report_lines.extend([
                f"Best mixed-precision case: {best_mixed_performance.problem_name} "
                f"({best_mixed_performance.grid_size[0]}x{best_mixed_performance.grid_size[1]})",
                f"  Speedup: {best_mixed_performance.speedup_mixed_vs_double:.2f}x",
                "",
                f"Worst mixed-precision case: {worst_mixed_performance.problem_name} "
                f"({worst_mixed_performance.grid_size[0]}x{worst_mixed_performance.grid_size[1]})",
                f"  Speedup: {worst_mixed_performance.speedup_mixed_vs_double:.2f}x",
                ""
            ])
            
            # General recommendations
            if avg_speedup_mixed > 1.3:
                report_lines.append("✓ Mixed precision shows significant performance benefits")
            elif avg_speedup_mixed > 1.1:
                report_lines.append("• Mixed precision shows moderate performance benefits")
            else:
                report_lines.append("⚠ Mixed precision benefits are limited for tested cases")
            
            if avg_memory_saving > 0.2:
                report_lines.append("✓ Significant memory savings achieved with mixed precision")
            elif avg_memory_saving > 0.1:
                report_lines.append("• Moderate memory savings with mixed precision")
            else:
                report_lines.append("⚠ Limited memory savings observed")
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write(report)
                logger.info(f"Mixed-precision report saved to {save_to_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def plot_precision_comparison(
        self,
        comparison_results: List[PrecisionComparison],
        metric: str = 'speedup',
        save_figure: Optional[str] = None
    ) -> None:
        """
        Plot precision comparison results.
        
        Args:
            comparison_results: Comparison results to plot
            metric: Metric to plot ('speedup', 'memory', 'accuracy')
            save_figure: Optional filename to save figure
        """
        try:
            if metric == 'speedup':
                self._plot_speedup_comparison(comparison_results, save_figure)
            elif metric == 'memory':
                self._plot_memory_comparison(comparison_results, save_figure)
            elif metric == 'accuracy':
                self._plot_accuracy_comparison(comparison_results, save_figure)
            else:
                logger.warning(f"Unknown metric: {metric}")
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to create precision plot: {e}")
    
    def _plot_speedup_comparison(self, comparisons: List[PrecisionComparison], save_figure: Optional[str]):
        """Plot speedup comparison."""
        problems = list(set(c.problem_name for c in comparisons))
        grid_sizes = list(set(c.grid_size for c in comparisons))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_labels = []
        mixed_speedups = []
        single_speedups = []
        
        for problem in problems:
            for grid_size in grid_sizes:
                comp = next((c for c in comparisons 
                           if c.problem_name == problem and c.grid_size == grid_size), None)
                if comp:
                    x_labels.append(f"{problem}\n{grid_size[0]}x{grid_size[1]}")
                    mixed_speedups.append(comp.speedup_mixed_vs_double)
                    single_speedups.append(comp.speedup_single_vs_double)
        
        if x_labels:
            x = np.arange(len(x_labels))
            width = 0.35
            
            ax.bar(x - width/2, mixed_speedups, width, label='Mixed Precision', alpha=0.8)
            ax.bar(x + width/2, single_speedups, width, label='Single Precision', alpha=0.8)
            
            ax.set_xlabel('Problem and Grid Size')
            ax.set_ylabel('Speedup vs Double Precision')
            ax.set_title('Precision Speedup Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at 1.0
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
            
            plt.tight_layout()
            
            if save_figure:
                plt.savefig(save_figure, dpi=300, bbox_inches='tight')
                logger.info(f"Speedup plot saved to {save_figure}")
            
            plt.show()
    
    def _plot_memory_comparison(self, comparisons: List[PrecisionComparison], save_figure: Optional[str]):
        """Plot memory comparison."""
        # Similar implementation for memory savings
        pass
    
    def _plot_accuracy_comparison(self, comparisons: List[PrecisionComparison], save_figure: Optional[str]):
        """Plot accuracy comparison."""
        # Similar implementation for accuracy metrics
        pass


def run_comprehensive_mixed_precision_analysis() -> Dict[str, Any]:
    """
    Run comprehensive mixed-precision effectiveness analysis.
    
    Returns:
        Complete mixed-precision analysis results
    """
    logger.info("Starting comprehensive mixed-precision analysis")
    
    analyzer = MixedPrecisionAnalyzer()
    
    # Test configuration
    test_problems = ['trigonometric', 'polynomial', 'high_frequency']
    grid_sizes = [(33, 33), (65, 65), (129, 129)]
    precision_levels = ['double', 'single', 'mixed_conservative', 'mixed_aggressive']
    
    results = {
        'precision_comparisons': [],
        'convergence_studies': [],
        'analysis_summary': {},
        'analyzer': analyzer
    }
    
    try:
        # Main precision trade-off analysis
        precision_comparisons = analyzer.analyze_precision_trade_offs(
            test_problems, grid_sizes[:2], 'multigrid', precision_levels, num_runs=3
        )
        results['precision_comparisons'] = precision_comparisons
        logger.info(f"Completed {len(precision_comparisons)} precision comparisons")
        
        # Convergence analysis for key problems
        tolerance_range = [1e-10, 1e-8, 1e-6, 1e-4]
        
        for problem in test_problems[:2]:  # Subset for convergence study
            convergence_study = analyzer.analyze_precision_convergence(
                problem, grid_sizes[:2], precision_levels, tolerance_range
            )
            results['convergence_studies'].append(convergence_study)
        
        logger.info(f"Completed {len(results['convergence_studies'])} convergence studies")
        
        # Generate analysis summary
        results['analysis_summary'] = _summarize_mixed_precision_analysis(results)
        
    except Exception as e:
        logger.error(f"Mixed-precision analysis failed: {e}")
        results['error'] = str(e)
    
    logger.info("Comprehensive mixed-precision analysis completed")
    
    return results


def _summarize_mixed_precision_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of mixed-precision analysis."""
    summary = {
        'total_comparisons': len(results.get('precision_comparisons', [])),
        'convergence_studies': len(results.get('convergence_studies', []))
    }
    
    comparisons = results.get('precision_comparisons', [])
    
    if comparisons:
        valid_mixed = [c for c in comparisons if c.mixed_result and c.double_result]
        valid_single = [c for c in comparisons if c.single_result and c.double_result]
        
        if valid_mixed:
            summary['mixed_precision'] = {
                'avg_speedup': np.mean([c.speedup_mixed_vs_double for c in valid_mixed]),
                'max_speedup': np.max([c.speedup_mixed_vs_double for c in valid_mixed]),
                'avg_memory_saving': np.mean([c.memory_saving_mixed for c in valid_mixed if c.memory_saving_mixed >= 0])
            }
        
        if valid_single:
            summary['single_precision'] = {
                'avg_speedup': np.mean([c.speedup_single_vs_double for c in valid_single]),
                'max_speedup': np.max([c.speedup_single_vs_double for c in valid_single]),
                'avg_memory_saving': np.mean([c.memory_saving_single for c in valid_single if c.memory_saving_single >= 0])
            }
    
    return summary