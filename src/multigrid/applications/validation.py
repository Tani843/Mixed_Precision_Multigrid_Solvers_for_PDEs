"""Validation framework using Method of Manufactured Solutions and analytical tests."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import special
import sympy as sp
from sympy import symbols, diff, sin, cos, exp, pi, lambdify

from .poisson_solver import PoissonSolver2D, PoissonProblem
from .heat_solver import HeatSolver2D, HeatProblem, TimeSteppingConfig, TimeSteppingMethod
from ..core.grid import Grid

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a validation test."""
    test_name: str
    problem_type: str  # 'poisson' or 'heat'
    grid_sizes: List[Tuple[int, int]]
    errors: List[Dict[str, float]]
    convergence_rates: List[Dict[str, float]]
    expected_order: float
    achieved_order: Dict[str, float]
    passed: bool
    tolerance: float = 0.1  # Tolerance for convergence rate
    additional_metrics: Dict[str, Any] = None


class MethodOfManufacturedSolutions:
    """
    Method of Manufactured Solutions (MMS) for systematic validation.
    
    Generates exact solutions and corresponding source terms for validation testing.
    """
    
    def __init__(self):
        """Initialize MMS framework."""
        # Symbolic variables
        self.x, self.y, self.t = symbols('x y t', real=True)
        
        # Common manufactured solutions
        self.manufactured_solutions = {
            'polynomial_2d': self.x**2 + self.y**2,
            'trigonometric_2d': sin(pi * self.x) * sin(pi * self.y),
            'exponential_2d': exp(self.x + self.y),
            'mixed_2d': self.x**2 * self.y + sin(pi * self.x) * cos(pi * self.y),
            'high_freq_2d': sin(4*pi*self.x) * sin(4*pi*self.y),
            'anisotropic_2d': sin(pi*self.x) * sin(2*pi*self.y)
        }
        
        # Time-dependent solutions for heat equation
        self.heat_solutions = {
            'decay_trig': sin(pi * self.x) * sin(pi * self.y) * exp(-2*pi**2*self.t),
            'gaussian_diffusion': exp(-(self.x-0.5)**2/self.t - (self.y-0.5)**2/self.t) / self.t,
            'separable': sin(pi*self.x) * sin(pi*self.y) * (1 + self.t),
            'polynomial_time': (self.x**2 + self.y**2) * (1 + self.t**2)
        }
        
        logger.info("Method of Manufactured Solutions framework initialized")
    
    def generate_poisson_problem(
        self,
        solution_name: str,
        domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
        custom_solution: Optional[sp.Expr] = None
    ) -> PoissonProblem:
        """
        Generate Poisson problem using manufactured solution.
        
        Args:
            solution_name: Name of manufactured solution or 'custom'
            domain: Computational domain
            custom_solution: Custom symbolic solution if solution_name is 'custom'
            
        Returns:
            PoissonProblem with exact solution and corresponding source
        """
        if solution_name == 'custom':
            if custom_solution is None:
                raise ValueError("Custom solution must be provided")
            u_exact = custom_solution
        else:
            if solution_name not in self.manufactured_solutions:
                raise ValueError(f"Unknown solution: {solution_name}")
            u_exact = self.manufactured_solutions[solution_name]
        
        # Compute source term: f = -∇²u
        u_xx = diff(u_exact, self.x, 2)
        u_yy = diff(u_exact, self.y, 2)
        source_term = -(u_xx + u_yy)
        
        # Convert to numerical functions
        source_func = lambdify((self.x, self.y), source_term, 'numpy')
        solution_func = lambdify((self.x, self.y), u_exact, 'numpy')
        
        # Create problem
        problem = PoissonProblem(
            name=f"MMS_{solution_name}",
            source_function=source_func,
            analytical_solution=solution_func,
            boundary_conditions={'type': 'dirichlet', 'value': solution_func},
            domain=domain,
            description=f"Manufactured solution: {u_exact}"
        )
        
        logger.debug(f"Generated MMS Poisson problem: {solution_name}")
        logger.debug(f"Exact solution: {u_exact}")
        logger.debug(f"Source term: {source_term}")
        
        return problem
    
    def generate_heat_problem(
        self,
        solution_name: str,
        thermal_diffusivity: float = 1.0,
        domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
        custom_solution: Optional[sp.Expr] = None
    ) -> HeatProblem:
        """
        Generate heat equation problem using manufactured solution.
        
        Args:
            solution_name: Name of manufactured solution or 'custom'
            thermal_diffusivity: Thermal diffusivity parameter
            domain: Computational domain
            custom_solution: Custom symbolic solution if solution_name is 'custom'
            
        Returns:
            HeatProblem with exact solution and corresponding source
        """
        if solution_name == 'custom':
            if custom_solution is None:
                raise ValueError("Custom solution must be provided")
            u_exact = custom_solution
        else:
            if solution_name not in self.heat_solutions:
                raise ValueError(f"Unknown heat solution: {solution_name}")
            u_exact = self.heat_solutions[solution_name]
        
        # Compute source term: f = ∂u/∂t - α∇²u
        u_t = diff(u_exact, self.t)
        u_xx = diff(u_exact, self.x, 2)
        u_yy = diff(u_exact, self.y, 2)
        source_term = u_t - thermal_diffusivity * (u_xx + u_yy)
        
        # Convert to numerical functions
        source_func = lambdify((self.x, self.y, self.t), source_term, 'numpy')
        solution_func = lambdify((self.x, self.y, self.t), u_exact, 'numpy')
        initial_func = lambdify((self.x, self.y), u_exact.subs(self.t, 0), 'numpy')
        
        # Boundary conditions (Dirichlet with exact solution)
        boundary_func = lambda x, y, t: solution_func(x, y, t)
        
        # Create problem
        problem = HeatProblem(
            name=f"MMS_heat_{solution_name}",
            initial_condition=initial_func,
            source_function=source_func,
            analytical_solution=solution_func,
            boundary_conditions={'type': 'dirichlet', 'value': boundary_func},
            thermal_diffusivity=thermal_diffusivity,
            domain=domain,
            description=f"Manufactured heat solution: {u_exact}"
        )
        
        logger.debug(f"Generated MMS heat problem: {solution_name}")
        logger.debug(f"Exact solution: {u_exact}")
        logger.debug(f"Source term: {source_term}")
        
        return problem


class ValidationSuite:
    """
    Comprehensive validation suite for Poisson and Heat equation solvers.
    
    Performs systematic validation using manufactured solutions, analytical tests,
    and convergence studies.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.mms = MethodOfManufacturedSolutions()
        self.validation_results: List[ValidationResult] = []
        
        # Default grid sizes for convergence studies
        self.default_grid_sizes = [
            (17, 17), (33, 33), (65, 65), (129, 129)
        ]
        
        logger.info("Validation suite initialized")
    
    def validate_poisson_solver(
        self,
        solver_config: Dict[str, Any],
        test_problems: Optional[List[str]] = None,
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
        expected_order: float = 2.0
    ) -> List[ValidationResult]:
        """
        Validate Poisson solver with manufactured solutions.
        
        Args:
            solver_config: Configuration for PoissonSolver2D
            test_problems: List of test problem names
            grid_sizes: Grid sizes for convergence study
            expected_order: Expected convergence order
            
        Returns:
            List of validation results
        """
        if test_problems is None:
            test_problems = ['trigonometric_2d', 'polynomial_2d', 'mixed_2d']
        
        if grid_sizes is None:
            grid_sizes = self.default_grid_sizes
        
        logger.info(f"Validating Poisson solver with {len(test_problems)} test problems")
        
        results = []
        
        for problem_name in test_problems:
            logger.info(f"Testing Poisson problem: {problem_name}")
            
            # Create manufactured problem
            mms_problem = self.mms.generate_poisson_problem(problem_name)
            
            # Create solver
            solver = PoissonSolver2D(**solver_config)
            
            # Run convergence study
            convergence_study = solver.run_convergence_study(
                mms_problem, grid_sizes, expected_order
            )
            
            # Analyze results
            achieved_l2_order = convergence_study['achieved_order']['l2']
            achieved_max_order = convergence_study['achieved_order']['max']
            
            # Check if convergence rate meets expectations
            tolerance = 0.3  # Allow some deviation
            l2_passed = abs(achieved_l2_order - expected_order) < tolerance
            max_passed = abs(achieved_max_order - expected_order) < tolerance
            overall_passed = l2_passed and max_passed
            
            # Create validation result
            validation_result = ValidationResult(
                test_name=f"poisson_{problem_name}",
                problem_type='poisson',
                grid_sizes=grid_sizes,
                errors=[result['errors'] for result in convergence_study['results']],
                convergence_rates=convergence_study['convergence_rates'],
                expected_order=expected_order,
                achieved_order={
                    'l2': achieved_l2_order,
                    'max': achieved_max_order
                },
                passed=overall_passed,
                tolerance=tolerance,
                additional_metrics={
                    'solver_config': solver_config,
                    'convergence_study': convergence_study
                }
            )
            
            results.append(validation_result)
            self.validation_results.append(validation_result)
            
            logger.info(f"Poisson {problem_name}: L2 order = {achieved_l2_order:.2f}, "
                       f"Max order = {achieved_max_order:.2f}, Passed = {overall_passed}")
        
        return results
    
    def validate_heat_solver(
        self,
        solver_config: Dict[str, Any],
        time_config: TimeSteppingConfig,
        test_problems: Optional[List[str]] = None,
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
        expected_order: float = 2.0
    ) -> List[ValidationResult]:
        """
        Validate heat solver with manufactured solutions.
        
        Args:
            solver_config: Configuration for HeatSolver2D
            time_config: Time stepping configuration
            test_problems: List of test problem names
            grid_sizes: Grid sizes for convergence study
            expected_order: Expected spatial convergence order
            
        Returns:
            List of validation results
        """
        if test_problems is None:
            test_problems = ['decay_trig', 'separable']
        
        if grid_sizes is None:
            grid_sizes = self.default_grid_sizes[:3]  # Smaller for time-dependent problems
        
        logger.info(f"Validating heat solver with {len(test_problems)} test problems")
        
        results = []
        
        for problem_name in test_problems:
            logger.info(f"Testing heat problem: {problem_name}")
            
            # Create manufactured problem
            mms_problem = self.mms.generate_heat_problem(problem_name)
            
            # Create solver
            solver = HeatSolver2D(**solver_config)
            
            # Run convergence study
            convergence_errors = []
            convergence_results = []
            
            for nx, ny in grid_sizes:
                result = solver.solve_heat_problem(mms_problem, nx, ny, time_config)
                convergence_results.append(result)
                convergence_errors.append(result['errors'])
            
            # Calculate convergence rates
            convergence_rates = self._calculate_heat_convergence_rates(convergence_results)
            
            # Analyze results
            if convergence_rates:
                achieved_l2_order = np.mean([rate['l2'] for rate in convergence_rates])
                achieved_max_order = np.mean([rate['max'] for rate in convergence_rates])
            else:
                achieved_l2_order = 0.0
                achieved_max_order = 0.0
            
            # Check convergence
            tolerance = 0.4  # More tolerant for time-dependent problems
            l2_passed = abs(achieved_l2_order - expected_order) < tolerance
            max_passed = abs(achieved_max_order - expected_order) < tolerance
            overall_passed = l2_passed and max_passed
            
            # Create validation result
            validation_result = ValidationResult(
                test_name=f"heat_{problem_name}",
                problem_type='heat',
                grid_sizes=grid_sizes,
                errors=convergence_errors,
                convergence_rates=convergence_rates,
                expected_order=expected_order,
                achieved_order={
                    'l2': achieved_l2_order,
                    'max': achieved_max_order
                },
                passed=overall_passed,
                tolerance=tolerance,
                additional_metrics={
                    'solver_config': solver_config,
                    'time_config': time_config.__dict__,
                    'convergence_results': convergence_results
                }
            )
            
            results.append(validation_result)
            self.validation_results.append(validation_result)
            
            logger.info(f"Heat {problem_name}: L2 order = {achieved_l2_order:.2f}, "
                       f"Max order = {achieved_max_order:.2f}, Passed = {overall_passed}")
        
        return results
    
    def _calculate_heat_convergence_rates(self, results: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Calculate convergence rates for heat equation results."""
        if len(results) < 2:
            return []
        
        rates = []
        for i in range(1, len(results)):
            current = results[i]
            previous = results[i-1]
            
            # Grid refinement ratio
            h_current = min(current['errors']['grid_spacing'])
            h_previous = min(previous['errors']['grid_spacing'])
            h_ratio = h_previous / h_current
            
            # Error ratios
            l2_ratio = previous['errors']['l2_error'] / current['errors']['l2_error']
            max_ratio = previous['errors']['max_error'] / current['errors']['max_error']
            
            # Convergence rates
            l2_rate = np.log(l2_ratio) / np.log(h_ratio) if l2_ratio > 0 and h_ratio > 1 else 0
            max_rate = np.log(max_ratio) / np.log(h_ratio) if max_ratio > 0 and h_ratio > 1 else 0
            
            rates.append({
                'grid_transition': f"{previous['grid_size']} -> {current['grid_size']}",
                'h_ratio': h_ratio,
                'l2': l2_rate,
                'max': max_rate,
                'l2_error_ratio': l2_ratio,
                'max_error_ratio': max_ratio
            })
        
        return rates
    
    def run_comprehensive_validation(
        self,
        poisson_configs: List[Dict[str, Any]],
        heat_configs: List[Dict[str, Any]],
        time_configs: List[TimeSteppingConfig]
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation across multiple solver configurations.
        
        Args:
            poisson_configs: List of Poisson solver configurations
            heat_configs: List of Heat solver configurations  
            time_configs: List of time stepping configurations
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive validation suite")
        
        all_results = {
            'poisson_results': [],
            'heat_results': [],
            'summary': {}
        }
        
        # Validate Poisson solvers
        for i, poisson_config in enumerate(poisson_configs):
            logger.info(f"Validating Poisson configuration {i+1}/{len(poisson_configs)}")
            
            poisson_results = self.validate_poisson_solver(poisson_config)
            all_results['poisson_results'].extend(poisson_results)
        
        # Validate Heat solvers
        for i, heat_config in enumerate(heat_configs):
            for j, time_config in enumerate(time_configs):
                logger.info(f"Validating Heat configuration {i+1}/{len(heat_configs)}, "
                           f"time method {j+1}/{len(time_configs)}")
                
                heat_results = self.validate_heat_solver(heat_config, time_config)
                all_results['heat_results'].extend(heat_results)
        
        # Generate summary
        all_results['summary'] = self._generate_validation_summary(all_results)
        
        logger.info("Comprehensive validation completed")
        
        return all_results
    
    def _generate_validation_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        poisson_results = all_results['poisson_results']
        heat_results = all_results['heat_results']
        
        all_validation_results = poisson_results + heat_results
        
        summary = {
            'total_tests': len(all_validation_results),
            'poisson_tests': len(poisson_results),
            'heat_tests': len(heat_results),
            'passed_tests': sum(1 for r in all_validation_results if r.passed),
            'failed_tests': sum(1 for r in all_validation_results if not r.passed),
            'pass_rate': 0.0,
            'poisson_summary': {},
            'heat_summary': {}
        }
        
        if summary['total_tests'] > 0:
            summary['pass_rate'] = summary['passed_tests'] / summary['total_tests']
        
        # Poisson summary
        if poisson_results:
            poisson_passed = sum(1 for r in poisson_results if r.passed)
            poisson_l2_orders = [r.achieved_order['l2'] for r in poisson_results]
            poisson_max_orders = [r.achieved_order['max'] for r in poisson_results]
            
            summary['poisson_summary'] = {
                'total': len(poisson_results),
                'passed': poisson_passed,
                'pass_rate': poisson_passed / len(poisson_results),
                'avg_l2_order': np.mean(poisson_l2_orders),
                'avg_max_order': np.mean(poisson_max_orders),
                'std_l2_order': np.std(poisson_l2_orders),
                'std_max_order': np.std(poisson_max_orders)
            }
        
        # Heat summary
        if heat_results:
            heat_passed = sum(1 for r in heat_results if r.passed)
            heat_l2_orders = [r.achieved_order['l2'] for r in heat_results]
            heat_max_orders = [r.achieved_order['max'] for r in heat_results]
            
            summary['heat_summary'] = {
                'total': len(heat_results),
                'passed': heat_passed,
                'pass_rate': heat_passed / len(heat_results),
                'avg_l2_order': np.mean(heat_l2_orders),
                'avg_max_order': np.mean(heat_max_orders),
                'std_l2_order': np.std(heat_l2_orders),
                'std_max_order': np.std(heat_max_orders)
            }
        
        return summary
    
    def generate_validation_report(self, save_to_file: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            save_to_file: Optional filename to save report
            
        Returns:
            Formatted validation report
        """
        if not self.validation_results:
            return "No validation results available"
        
        report_lines = [
            "VALIDATION SUITE REPORT",
            "=" * 50,
            ""
        ]
        
        # Summary statistics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report_lines.extend([
            "SUMMARY:",
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {total_tests - passed_tests}",
            f"Pass Rate: {pass_rate:.1%}",
            ""
        ])
        
        # Detailed results by problem type
        poisson_results = [r for r in self.validation_results if r.problem_type == 'poisson']
        heat_results = [r for r in self.validation_results if r.problem_type == 'heat']
        
        if poisson_results:
            report_lines.extend([
                "POISSON EQUATION VALIDATION:",
                "-" * 30
            ])
            
            for result in poisson_results:
                status = "PASS" if result.passed else "FAIL"
                report_lines.append(
                    f"{result.test_name:25s}: {status:4s} "
                    f"(L2: {result.achieved_order['l2']:5.2f}, "
                    f"Max: {result.achieved_order['max']:5.2f})"
                )
            
            report_lines.append("")
        
        if heat_results:
            report_lines.extend([
                "HEAT EQUATION VALIDATION:",
                "-" * 25
            ])
            
            for result in heat_results:
                status = "PASS" if result.passed else "FAIL"
                report_lines.append(
                    f"{result.test_name:25s}: {status:4s} "
                    f"(L2: {result.achieved_order['l2']:5.2f}, "
                    f"Max: {result.achieved_order['max']:5.2f})"
                )
            
            report_lines.append("")
        
        # Failed tests details
        failed_results = [r for r in self.validation_results if not r.passed]
        if failed_results:
            report_lines.extend([
                "FAILED TESTS ANALYSIS:",
                "-" * 25
            ])
            
            for result in failed_results:
                expected = result.expected_order
                achieved_l2 = result.achieved_order['l2']
                achieved_max = result.achieved_order['max']
                
                report_lines.extend([
                    f"Test: {result.test_name}",
                    f"  Expected order: {expected:.1f}",
                    f"  Achieved L2 order: {achieved_l2:.2f} (error: {abs(achieved_l2 - expected):.2f})",
                    f"  Achieved Max order: {achieved_max:.2f} (error: {abs(achieved_max - expected):.2f})",
                    f"  Tolerance: {result.tolerance:.2f}",
                    ""
                ])
        
        report = "\n".join(report_lines)
        
        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write(report)
                logger.info(f"Validation report saved to {save_to_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def plot_convergence_results(
        self,
        result: ValidationResult,
        save_figure: Optional[str] = None
    ) -> None:
        """
        Plot convergence results for a validation test.
        
        Args:
            result: Validation result to plot
            save_figure: Optional filename to save figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Extract data
            grid_sizes = [np.prod(size) for size in result.grid_sizes]
            l2_errors = [error['l2_error'] for error in result.errors]
            max_errors = [error['max_error'] for error in result.errors]
            
            # Plot L2 errors
            ax1.loglog(grid_sizes, l2_errors, 'o-', label='L2 Error')
            ax1.set_xlabel('Number of Grid Points')
            ax1.set_ylabel('L2 Error')
            ax1.set_title(f'{result.test_name}: L2 Error Convergence')
            ax1.grid(True)
            
            # Add theoretical line
            theoretical_slope = result.expected_order / 2  # h^p -> N^(-p/2)
            theoretical_errors = l2_errors[0] * (grid_sizes[0] / np.array(grid_sizes))**(theoretical_slope)
            ax1.loglog(grid_sizes, theoretical_errors, '--', alpha=0.7, 
                      label=f'Expected O(h^{result.expected_order})')
            ax1.legend()
            
            # Plot Max errors
            ax2.loglog(grid_sizes, max_errors, 's-', color='red', label='Max Error')
            ax2.set_xlabel('Number of Grid Points')
            ax2.set_ylabel('Max Error')
            ax2.set_title(f'{result.test_name}: Max Error Convergence')
            ax2.grid(True)
            
            # Add theoretical line
            theoretical_max_errors = max_errors[0] * (grid_sizes[0] / np.array(grid_sizes))**(theoretical_slope)
            ax2.loglog(grid_sizes, theoretical_max_errors, '--', alpha=0.7,
                      label=f'Expected O(h^{result.expected_order})')
            ax2.legend()
            
            plt.tight_layout()
            
            if save_figure:
                plt.savefig(save_figure, dpi=300, bbox_inches='tight')
                logger.info(f"Convergence plot saved to {save_figure}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to create convergence plot: {e}")


def run_quick_validation() -> Dict[str, Any]:
    """
    Run quick validation with default settings.
    
    Returns:
        Quick validation results
    """
    validation_suite = ValidationSuite()
    
    # Quick Poisson validation
    poisson_config = {
        'solver_type': 'multigrid',
        'max_levels': 4,
        'tolerance': 1e-8,
        'use_gpu': False
    }
    
    # Quick Heat validation
    heat_config = {
        'solver_type': 'multigrid', 
        'max_levels': 4,
        'tolerance': 1e-6,
        'use_gpu': False
    }
    
    time_config = TimeSteppingConfig(
        method=TimeSteppingMethod.BACKWARD_EULER,
        dt=0.01,
        t_final=0.1
    )
    
    # Run validations
    quick_grids = [(17, 17), (33, 33), (65, 65)]
    
    poisson_results = validation_suite.validate_poisson_solver(
        poisson_config,
        test_problems=['trigonometric_2d'],
        grid_sizes=quick_grids
    )
    
    heat_results = validation_suite.validate_heat_solver(
        heat_config,
        time_config,
        test_problems=['decay_trig'],
        grid_sizes=quick_grids[:2]  # Even smaller for heat
    )
    
    # Generate report
    report = validation_suite.generate_validation_report()
    
    return {
        'poisson_results': poisson_results,
        'heat_results': heat_results,
        'validation_report': report,
        'passed': all(r.passed for r in poisson_results + heat_results)
    }