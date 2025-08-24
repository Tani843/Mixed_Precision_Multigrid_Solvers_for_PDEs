"""Comprehensive validation test suite for mixed-precision multigrid solvers."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

from .validation import ValidationSuite, ValidationResult
from .convergence_analysis import GridConvergenceAnalyzer, ConvergenceData
from .performance_analysis import PerformanceAnalyzer, ScalabilityAnalyzer, ComparisonResult
from .mixed_precision_analysis import MixedPrecisionAnalyzer, PrecisionComparison
from .test_problems import PoissonTestProblems, HeatTestProblems, BenchmarkProblems
from .heat_solver import TimeSteppingConfig, TimeSteppingMethod

logger = logging.getLogger(__name__)


@dataclass
class ValidationTestResult:
    """Result from a single validation test."""
    test_name: str
    test_type: str  # 'correctness', 'convergence', 'performance', 'precision'
    passed: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ValidationSuiteResult:
    """Results from complete validation suite."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_execution_time: float
    test_results: List[ValidationTestResult]
    summary: Dict[str, Any]
    detailed_results: Dict[str, Any]


class ComprehensiveValidationSuite:
    """
    Comprehensive validation test suite for mixed-precision multigrid solvers.
    
    Integrates all validation components: correctness, convergence, performance,
    and mixed-precision analysis in a unified testing framework.
    """
    
    def __init__(self):
        """Initialize comprehensive validation suite."""
        self.validation_suite = ValidationSuite()
        self.convergence_analyzer = GridConvergenceAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.scalability_analyzer = ScalabilityAnalyzer(self.performance_analyzer)
        self.precision_analyzer = MixedPrecisionAnalyzer()
        
        self.validation_results: List[ValidationSuiteResult] = []
        
        # Test configurations
        self.solver_configs = {
            'cpu_multigrid': {
                'solver_type': 'multigrid',
                'max_levels': 6,
                'tolerance': 1e-8,
                'use_gpu': False,
                'enable_mixed_precision': False
            },
            'cpu_multigrid_mixed': {
                'solver_type': 'multigrid',
                'max_levels': 6,
                'tolerance': 1e-8,
                'use_gpu': False,
                'enable_mixed_precision': True
            }
        }
        
        # Add GPU configs if available
        try:
            from ..gpu.gpu_solver import GPUMultigridSolver
            self.solver_configs.update({
                'gpu_multigrid': {
                    'solver_type': 'gpu_multigrid',
                    'max_levels': 6,
                    'tolerance': 1e-8,
                    'use_gpu': True,
                    'enable_mixed_precision': False
                },
                'gpu_multigrid_mixed': {
                    'solver_type': 'gpu_multigrid',
                    'max_levels': 6,
                    'tolerance': 1e-8,
                    'use_gpu': True,
                    'enable_mixed_precision': True
                }
            })
            logger.info("GPU solver configurations added")
        except ImportError:
            logger.info("GPU solvers not available, using CPU-only configurations")
        
        logger.info("Comprehensive validation suite initialized")
    
    def run_full_validation_suite(
        self,
        quick_mode: bool = False,
        parallel_execution: bool = False,
        save_results: bool = True
    ) -> ValidationSuiteResult:
        """
        Run the complete validation suite.
        
        Args:
            quick_mode: Run reduced test set for faster execution
            parallel_execution: Run tests in parallel where possible
            save_results: Save detailed results to files
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive validation suite")
        start_time = time.time()
        
        test_results = []
        detailed_results = {}
        
        # Define test sequence
        test_sequence = [
            ('correctness_validation', self._run_correctness_validation),
            ('convergence_analysis', self._run_convergence_analysis),
            ('performance_benchmarking', self._run_performance_benchmarking),
            ('mixed_precision_analysis', self._run_precision_analysis),
            ('scalability_studies', self._run_scalability_studies),
            ('integration_tests', self._run_integration_tests)
        ]
        
        if quick_mode:
            # Reduce test scope for quick mode
            test_sequence = test_sequence[:3]  # Only first 3 test types
        
        # Run tests
        for test_name, test_function in test_sequence:
            logger.info(f"Running {test_name}")
            
            try:
                test_start = time.time()
                
                if parallel_execution and test_name in ['performance_benchmarking', 'convergence_analysis']:
                    result = test_function(quick_mode, use_parallel=True)
                else:
                    result = test_function(quick_mode)
                
                test_time = time.time() - test_start
                
                test_result = ValidationTestResult(
                    test_name=test_name,
                    test_type=test_name.split('_')[0],
                    passed=result.get('passed', True),
                    execution_time=test_time,
                    details=result
                )
                
                test_results.append(test_result)
                detailed_results[test_name] = result
                
                logger.info(f"Completed {test_name}: {'PASSED' if test_result.passed else 'FAILED'} "
                           f"({test_time:.2f}s)")
                
            except Exception as e:
                test_time = time.time() - test_start
                error_msg = f"Test {test_name} failed: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                
                test_result = ValidationTestResult(
                    test_name=test_name,
                    test_type=test_name.split('_')[0],
                    passed=False,
                    execution_time=test_time,
                    details={'error': str(e)},
                    error_message=error_msg
                )
                
                test_results.append(test_result)
                detailed_results[test_name] = {'error': str(e)}
        
        total_time = time.time() - start_time
        
        # Generate summary
        passed_tests = sum(1 for t in test_results if t.passed)
        failed_tests = len(test_results) - passed_tests
        
        summary = self._generate_validation_summary(test_results, detailed_results)
        
        suite_result = ValidationSuiteResult(
            suite_name="Comprehensive Mixed-Precision Multigrid Validation",
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_execution_time=total_time,
            test_results=test_results,
            summary=summary,
            detailed_results=detailed_results
        )
        
        self.validation_results.append(suite_result)
        
        # Save results if requested
        if save_results:
            self._save_validation_results(suite_result)
        
        logger.info(f"Comprehensive validation completed: {passed_tests}/{len(test_results)} tests passed "
                   f"({total_time:.2f}s)")
        
        return suite_result
    
    def _run_correctness_validation(self, quick_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """Run correctness validation tests."""
        logger.info("Running correctness validation")
        
        # Test problems for correctness validation
        test_problems = ['trigonometric', 'polynomial', 'mixed'] if not quick_mode else ['trigonometric']
        grid_sizes = [(33, 33), (65, 65)] if not quick_mode else [(33, 33)]
        
        results = {
            'poisson_results': [],
            'heat_results': [],
            'passed': True,
            'summary': {}
        }
        
        # Poisson solver validation
        for config_name, config in list(self.solver_configs.items())[:2]:  # Test first 2 configs
            try:
                poisson_results = self.validation_suite.validate_poisson_solver(
                    config, test_problems, grid_sizes, expected_order=2.0
                )
                results['poisson_results'].extend(poisson_results)
                
                # Check if all tests passed
                if not all(r.passed for r in poisson_results):
                    results['passed'] = False
                    
            except Exception as e:
                logger.error(f"Poisson validation failed for {config_name}: {e}")
                results['passed'] = False
        
        # Heat solver validation
        time_config = TimeSteppingConfig(
            method=TimeSteppingMethod.BACKWARD_EULER,
            dt=0.01,
            t_final=0.05 if quick_mode else 0.1
        )
        
        heat_problems = ['pure_diffusion'] if quick_mode else ['pure_diffusion', 'separable']
        
        for config_name, config in list(self.solver_configs.items())[:2]:
            try:
                heat_results = self.validation_suite.validate_heat_solver(
                    config, time_config, heat_problems, grid_sizes[:1], expected_order=2.0
                )
                results['heat_results'].extend(heat_results)
                
                # Check if all tests passed
                if not all(r.passed for r in heat_results):
                    results['passed'] = False
                    
            except Exception as e:
                logger.error(f"Heat validation failed for {config_name}: {e}")
                results['passed'] = False
        
        # Generate summary
        all_results = results['poisson_results'] + results['heat_results']
        passed_count = sum(1 for r in all_results if r.passed)
        
        results['summary'] = {
            'total_tests': len(all_results),
            'passed_tests': passed_count,
            'pass_rate': passed_count / len(all_results) if all_results else 0,
            'poisson_tests': len(results['poisson_results']),
            'heat_tests': len(results['heat_results'])
        }
        
        logger.info(f"Correctness validation completed: {passed_count}/{len(all_results)} passed")
        
        return results
    
    def _run_convergence_analysis(
        self, 
        quick_mode: bool = False, 
        use_parallel: bool = False, 
        **kwargs
    ) -> Dict[str, Any]:
        """Run convergence analysis tests."""
        logger.info("Running convergence analysis")
        
        test_problems = ['trigonometric', 'polynomial'] if not quick_mode else ['trigonometric']
        grid_sequence = [(17, 17), (33, 33), (65, 65)] if quick_mode else [(9, 9), (17, 17), (33, 33), (65, 65)]
        
        results = {
            'poisson_studies': [],
            'heat_studies': [],
            'passed': True,
            'summary': {}
        }
        
        # Poisson convergence studies
        for problem in test_problems:
            for config_name, config in list(self.solver_configs.items())[:2]:
                try:
                    study = self.convergence_analyzer.run_poisson_convergence_study(
                        config, problem, grid_sequence, theoretical_order=2.0
                    )
                    results['poisson_studies'].append(study)
                    
                    # Check convergence quality
                    achieved_order = study.achieved_order.get('l2_error', 0)
                    if abs(achieved_order - 2.0) > 0.5:  # Tolerance for convergence
                        results['passed'] = False
                        
                except Exception as e:
                    logger.error(f"Poisson convergence study failed for {problem}: {e}")
                    results['passed'] = False
        
        # Heat convergence studies (smaller subset)
        if not quick_mode:
            time_config = TimeSteppingConfig(
                method=TimeSteppingMethod.BACKWARD_EULER,
                dt=0.005,
                t_final=0.05
            )
            
            for problem in ['pure_diffusion']:
                for config_name, config in list(self.solver_configs.items())[:1]:
                    try:
                        study = self.convergence_analyzer.run_heat_convergence_study(
                            config, time_config, problem, grid_sequence[:3], theoretical_order=2.0
                        )
                        results['heat_studies'].append(study)
                        
                    except Exception as e:
                        logger.error(f"Heat convergence study failed for {problem}: {e}")
                        results['passed'] = False
        
        # Generate summary
        all_studies = results['poisson_studies'] + results['heat_studies']
        good_convergence_count = sum(
            1 for s in all_studies 
            if abs(s.achieved_order.get('l2_error', 0) - s.theoretical_order) < 0.5
        )
        
        results['summary'] = {
            'total_studies': len(all_studies),
            'good_convergence': good_convergence_count,
            'poisson_studies': len(results['poisson_studies']),
            'heat_studies': len(results['heat_studies'])
        }
        
        logger.info(f"Convergence analysis completed: {good_convergence_count}/{len(all_studies)} "
                   f"studies achieved good convergence")
        
        return results
    
    def _run_performance_benchmarking(
        self, 
        quick_mode: bool = False, 
        use_parallel: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        logger.info("Running performance benchmarking")
        
        test_problems = ['trigonometric'] if quick_mode else ['trigonometric', 'polynomial']
        grid_sizes = [(33, 33), (65, 65)] if quick_mode else [(33, 33), (65, 65), (129, 129)]
        
        results = {
            'benchmark_results': [],
            'comparison_results': [],
            'passed': True,
            'summary': {}
        }
        
        # Run benchmarks
        try:
            configs_to_test = list(self.solver_configs.values())[:2] if quick_mode else list(self.solver_configs.values())
            
            benchmark_results = self.performance_analyzer.benchmark_poisson_solver(
                configs_to_test, test_problems, grid_sizes, 
                num_runs=2 if quick_mode else 3, warmup_runs=1
            )
            results['benchmark_results'] = benchmark_results
            
            # Basic performance check: all solvers should complete successfully
            if not benchmark_results:
                results['passed'] = False
                
        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            results['passed'] = False
        
        # CPU vs GPU comparison if applicable
        if len(self.solver_configs) > 2 and not quick_mode:
            try:
                comparison_results = self.performance_analyzer.compare_cpu_gpu_performance(
                    test_problems, grid_sizes[:2], num_runs=2
                )
                results['comparison_results'] = comparison_results
                
            except Exception as e:
                logger.error(f"CPU vs GPU comparison failed: {e}")
        
        # Generate summary
        results['summary'] = {
            'total_benchmarks': len(results['benchmark_results']),
            'total_comparisons': len(results['comparison_results']),
            'avg_throughput': np.mean([r.throughput for r in results['benchmark_results']]) if results['benchmark_results'] else 0
        }
        
        if results['comparison_results']:
            speedups = [c.speedup for c in results['comparison_results'] if c.gpu_result]
            if speedups:
                results['summary']['avg_gpu_speedup'] = np.mean(speedups)
        
        logger.info(f"Performance benchmarking completed: {len(results['benchmark_results'])} benchmarks")
        
        return results
    
    def _run_precision_analysis(self, quick_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """Run mixed-precision analysis tests."""
        logger.info("Running mixed-precision analysis")
        
        test_problems = ['trigonometric'] if quick_mode else ['trigonometric', 'polynomial']
        grid_sizes = [(33, 33)] if quick_mode else [(33, 33), (65, 65)]
        precision_levels = ['double', 'mixed_conservative'] if quick_mode else ['double', 'single', 'mixed_conservative']
        
        results = {
            'precision_comparisons': [],
            'passed': True,
            'summary': {}
        }
        
        try:
            precision_comparisons = self.precision_analyzer.analyze_precision_trade_offs(
                test_problems, grid_sizes, 'multigrid', precision_levels, num_runs=2
            )
            results['precision_comparisons'] = precision_comparisons
            
            # Check if mixed precision provides reasonable performance
            valid_comparisons = [c for c in precision_comparisons if c.mixed_result and c.double_result]
            if valid_comparisons:
                avg_speedup = np.mean([c.speedup_mixed_vs_double for c in valid_comparisons])
                if avg_speedup < 1.05:  # At least 5% speedup expected
                    logger.warning("Mixed precision showing limited performance benefits")
            
        except Exception as e:
            logger.error(f"Mixed-precision analysis failed: {e}")
            results['passed'] = False
        
        # Generate summary
        results['summary'] = {
            'total_comparisons': len(results['precision_comparisons']),
            'precision_levels_tested': len(precision_levels)
        }
        
        if results['precision_comparisons']:
            valid_mixed = [c for c in results['precision_comparisons'] if c.mixed_result and c.double_result]
            if valid_mixed:
                results['summary']['avg_mixed_speedup'] = np.mean([c.speedup_mixed_vs_double for c in valid_mixed])
        
        logger.info(f"Mixed-precision analysis completed: {len(results['precision_comparisons'])} comparisons")
        
        return results
    
    def _run_scalability_studies(self, quick_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """Run scalability studies."""
        logger.info("Running scalability studies")
        
        results = {
            'scaling_studies': [],
            'passed': True,
            'summary': {}
        }
        
        # Weak scaling study
        try:
            base_size = (17, 17) if quick_mode else (13, 13)
            scaling_factors = [1, 4] if quick_mode else [1, 4, 9]
            
            weak_scaling = self.scalability_analyzer.run_weak_scaling_study(
                base_size, scaling_factors, self.solver_configs['cpu_multigrid']
            )
            results['scaling_studies'].append(weak_scaling)
            
            # Check scaling efficiency
            efficiency = weak_scaling['analysis'].get('overall_efficiency', 0)
            if efficiency < 0.7:  # Expect at least 70% efficiency
                logger.warning(f"Weak scaling efficiency low: {efficiency:.2f}")
            
        except Exception as e:
            logger.error(f"Weak scaling study failed: {e}")
            results['passed'] = False
        
        # Strong scaling study (if multiple configs available)
        if len(self.solver_configs) > 1 and not quick_mode:
            try:
                strong_scaling = self.scalability_analyzer.run_strong_scaling_study(
                    'trigonometric', [(65, 65)], list(self.solver_configs.values())[:2]
                )
                results['scaling_studies'].append(strong_scaling)
                
            except Exception as e:
                logger.error(f"Strong scaling study failed: {e}")
        
        # Generate summary
        results['summary'] = {
            'total_scaling_studies': len(results['scaling_studies'])
        }
        
        logger.info(f"Scalability studies completed: {len(results['scaling_studies'])} studies")
        
        return results
    
    def _run_integration_tests(self, quick_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """Run integration tests combining multiple components."""
        logger.info("Running integration tests")
        
        results = {
            'integration_tests': [],
            'passed': True,
            'summary': {}
        }
        
        # Test 1: End-to-end Poisson problem solving with validation
        try:
            test_problems = PoissonTestProblems()
            problem = test_problems.get_problem('trigonometric')
            
            from .poisson_solver import PoissonSolver2D
            solver = PoissonSolver2D(**self.solver_configs['cpu_multigrid'])
            
            result = solver.solve_poisson_problem(problem, 65, 65)
            
            # Verify solution quality
            if 'errors' in result:
                l2_error = result['errors']['l2_error']
                if l2_error > 1e-5:  # Should achieve good accuracy
                    results['passed'] = False
                    logger.warning(f"Integration test: High L2 error {l2_error}")
                else:
                    logger.info(f"Integration test: Good accuracy achieved (L2 error: {l2_error:.2e})")
            
            results['integration_tests'].append({
                'name': 'poisson_end_to_end',
                'passed': 'errors' in result and result['errors']['l2_error'] < 1e-5,
                'details': result
            })
            
        except Exception as e:
            logger.error(f"Poisson integration test failed: {e}")
            results['passed'] = False
        
        # Test 2: Heat equation with time stepping
        if not quick_mode:
            try:
                heat_problems = HeatTestProblems()
                heat_problem = heat_problems.get_problem('pure_diffusion')
                
                from .heat_solver import HeatSolver2D
                heat_solver = HeatSolver2D(**self.solver_configs['cpu_multigrid'])
                
                time_config = TimeSteppingConfig(
                    method=TimeSteppingMethod.BACKWARD_EULER,
                    dt=0.01,
                    t_final=0.05
                )
                
                heat_result = heat_solver.solve_heat_problem(heat_problem, 33, 33, time_config)
                
                # Verify heat solution
                if 'errors' in heat_result:
                    heat_l2_error = heat_result['errors']['l2_error']
                    heat_passed = heat_l2_error < 1e-3  # Relaxed for time-dependent
                else:
                    heat_passed = False
                
                results['integration_tests'].append({
                    'name': 'heat_equation_time_stepping',
                    'passed': heat_passed,
                    'details': heat_result
                })
                
                if not heat_passed:
                    results['passed'] = False
                
            except Exception as e:
                logger.error(f"Heat equation integration test failed: {e}")
                results['passed'] = False
        
        # Generate summary
        results['summary'] = {
            'total_integration_tests': len(results['integration_tests']),
            'passed_integration_tests': sum(1 for t in results['integration_tests'] if t['passed'])
        }
        
        logger.info(f"Integration tests completed: "
                   f"{results['summary']['passed_integration_tests']}/{results['summary']['total_integration_tests']} passed")
        
        return results
    
    def _generate_validation_summary(
        self, 
        test_results: List[ValidationTestResult],
        detailed_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        summary = {
            'overall_pass_rate': sum(1 for t in test_results if t.passed) / len(test_results) if test_results else 0,
            'test_type_summary': {},
            'performance_metrics': {},
            'key_findings': []
        }
        
        # Summarize by test type
        test_types = set(t.test_type for t in test_results)
        for test_type in test_types:
            type_tests = [t for t in test_results if t.test_type == test_type]
            type_passed = sum(1 for t in type_tests if t.passed)
            
            summary['test_type_summary'][test_type] = {
                'total': len(type_tests),
                'passed': type_passed,
                'pass_rate': type_passed / len(type_tests) if type_tests else 0,
                'avg_execution_time': np.mean([t.execution_time for t in type_tests])
            }
        
        # Extract key performance metrics
        if 'performance_benchmarking' in detailed_results:
            perf_data = detailed_results['performance_benchmarking']
            summary['performance_metrics'] = perf_data.get('summary', {})
        
        # Generate key findings
        if summary['overall_pass_rate'] >= 0.95:
            summary['key_findings'].append("✓ Excellent overall validation performance")
        elif summary['overall_pass_rate'] >= 0.8:
            summary['key_findings'].append("• Good overall validation performance")
        else:
            summary['key_findings'].append("⚠ Some validation tests failed - review required")
        
        # Specific findings from each test type
        if 'correctness' in summary['test_type_summary']:
            correctness_rate = summary['test_type_summary']['correctness']['pass_rate']
            if correctness_rate == 1.0:
                summary['key_findings'].append("✓ All correctness tests passed")
            elif correctness_rate >= 0.8:
                summary['key_findings'].append("• Most correctness tests passed")
            else:
                summary['key_findings'].append("⚠ Correctness issues detected")
        
        if 'convergence' in summary['test_type_summary']:
            convergence_rate = summary['test_type_summary']['convergence']['pass_rate']
            if convergence_rate >= 0.9:
                summary['key_findings'].append("✓ Good convergence behavior verified")
            else:
                summary['key_findings'].append("• Some convergence issues observed")
        
        if 'performance' in summary['test_type_summary']:
            if summary['performance_metrics'].get('avg_gpu_speedup', 0) > 2.0:
                summary['key_findings'].append("✓ Significant GPU performance gains")
            elif summary['performance_metrics'].get('avg_gpu_speedup', 0) > 1.3:
                summary['key_findings'].append("• Moderate GPU performance gains")
        
        return summary
    
    def _save_validation_results(self, suite_result: ValidationSuiteResult) -> None:
        """Save validation results to files."""
        try:
            # Generate timestamp for unique filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save summary report
            report = self.generate_validation_report(suite_result)
            report_filename = f"validation_report_{timestamp}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            
            # Save detailed results as JSON
            results_filename = f"validation_results_{timestamp}.json"
            
            # Convert results to JSON-serializable format
            json_data = {
                'suite_name': suite_result.suite_name,
                'total_tests': suite_result.total_tests,
                'passed_tests': suite_result.passed_tests,
                'failed_tests': suite_result.failed_tests,
                'total_execution_time': suite_result.total_execution_time,
                'summary': suite_result.summary,
                'test_results': [
                    {
                        'test_name': tr.test_name,
                        'test_type': tr.test_type,
                        'passed': tr.passed,
                        'execution_time': tr.execution_time,
                        'error_message': tr.error_message
                    }
                    for tr in suite_result.test_results
                ]
            }
            
            with open(results_filename, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to {report_filename} and {results_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    def generate_validation_report(self, suite_result: ValidationSuiteResult) -> str:
        """Generate comprehensive validation report."""
        report_lines = [
            "COMPREHENSIVE VALIDATION SUITE REPORT",
            "=" * 60,
            "",
            f"Suite: {suite_result.suite_name}",
            f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Execution Time: {suite_result.total_execution_time:.2f} seconds",
            "",
            "SUMMARY:",
            f"Total Tests: {suite_result.total_tests}",
            f"Passed: {suite_result.passed_tests}",
            f"Failed: {suite_result.failed_tests}",
            f"Overall Pass Rate: {suite_result.passed_tests/suite_result.total_tests:.1%}",
            ""
        ]
        
        # Test type breakdown
        if 'test_type_summary' in suite_result.summary:
            report_lines.extend([
                "TEST TYPE BREAKDOWN:",
                "-" * 25
            ])
            
            for test_type, summary in suite_result.summary['test_type_summary'].items():
                report_lines.extend([
                    f"{test_type.upper()}:",
                    f"  Tests: {summary['total']}",
                    f"  Passed: {summary['passed']}",
                    f"  Pass Rate: {summary['pass_rate']:.1%}",
                    f"  Avg Time: {summary['avg_execution_time']:.2f}s",
                    ""
                ])
        
        # Key findings
        if 'key_findings' in suite_result.summary:
            report_lines.extend([
                "KEY FINDINGS:",
                "-" * 15
            ])
            
            for finding in suite_result.summary['key_findings']:
                report_lines.append(f"  {finding}")
            
            report_lines.append("")
        
        # Failed tests details
        failed_tests = [t for t in suite_result.test_results if not t.passed]
        if failed_tests:
            report_lines.extend([
                "FAILED TESTS:",
                "-" * 15
            ])
            
            for test in failed_tests:
                report_lines.extend([
                    f"Test: {test.test_name}",
                    f"  Type: {test.test_type}",
                    f"  Execution Time: {test.execution_time:.2f}s",
                    f"  Error: {test.error_message or 'Details in test results'}",
                    ""
                ])
        
        # Performance metrics
        if 'performance_metrics' in suite_result.summary and suite_result.summary['performance_metrics']:
            report_lines.extend([
                "PERFORMANCE METRICS:",
                "-" * 20
            ])
            
            perf_metrics = suite_result.summary['performance_metrics']
            for metric, value in perf_metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {metric}: {value:.3f}")
                else:
                    report_lines.append(f"  {metric}: {value}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 15
        ])
        
        overall_pass_rate = suite_result.summary.get('overall_pass_rate', 0)
        
        if overall_pass_rate >= 0.95:
            report_lines.append("  ✓ Validation suite passed successfully")
            report_lines.append("  ✓ System is ready for production use")
        elif overall_pass_rate >= 0.8:
            report_lines.append("  • Validation mostly successful")
            report_lines.append("  • Review failed tests before production deployment")
        else:
            report_lines.append("  ⚠ Significant validation issues detected")
            report_lines.append("  ⚠ System requires investigation before deployment")
        
        if failed_tests:
            report_lines.append("  • Address failed test issues")
            report_lines.append("  • Rerun validation after fixes")
        
        return "\n".join(report_lines)


def run_comprehensive_validation(
    quick_mode: bool = False,
    parallel_execution: bool = False,
    save_results: bool = True
) -> ValidationSuiteResult:
    """
    Run comprehensive validation suite with all components.
    
    Args:
        quick_mode: Run reduced test set for faster execution
        parallel_execution: Run tests in parallel where possible
        save_results: Save detailed results to files
        
    Returns:
        Complete validation results
    """
    logger.info("Starting comprehensive validation")
    
    validation_suite = ComprehensiveValidationSuite()
    
    result = validation_suite.run_full_validation_suite(
        quick_mode=quick_mode,
        parallel_execution=parallel_execution,
        save_results=save_results
    )
    
    logger.info(f"Comprehensive validation completed: {result.passed_tests}/{result.total_tests} tests passed")
    
    return result


def run_quick_validation() -> ValidationSuiteResult:
    """Run quick validation for development/CI purposes."""
    logger.info("Running quick validation suite")
    
    return run_comprehensive_validation(
        quick_mode=True,
        parallel_execution=False,
        save_results=False
    )