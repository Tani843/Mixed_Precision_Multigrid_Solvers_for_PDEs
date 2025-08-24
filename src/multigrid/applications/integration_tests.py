"""Integration tests and performance demonstrations for the complete system."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json

from .poisson_solver import PoissonSolver2D, PoissonProblem
from .heat_solver import HeatSolver2D, HeatProblem, TimeSteppingConfig, TimeSteppingMethod
from .test_problems import PoissonTestProblems, HeatTestProblems, BenchmarkProblems
from .validation import ValidationSuite
from .convergence_analysis import GridConvergenceAnalyzer
from .performance_analysis import PerformanceAnalyzer, run_comprehensive_performance_analysis
from .mixed_precision_analysis import MixedPrecisionAnalyzer, run_comprehensive_mixed_precision_analysis
from .comprehensive_validation import run_comprehensive_validation

logger = logging.getLogger(__name__)


@dataclass
class DemoResult:
    """Result from a demonstration."""
    demo_name: str
    description: str
    success: bool
    execution_time: float
    key_metrics: Dict[str, Any]
    visualizations: List[str]
    summary: str


class SystemIntegrationTests:
    """
    Complete system integration tests and demonstrations.
    
    Tests the full pipeline from problem definition to solution
    and demonstrates the capabilities of the mixed-precision
    multigrid solver framework.
    """
    
    def __init__(self):
        """Initialize integration test suite."""
        self.demo_results: List[DemoResult] = []
        
        # Test data storage
        self.test_data = {
            'solutions': {},
            'convergence_data': {},
            'performance_data': {},
            'validation_results': {}
        }
        
        logger.info("System integration test suite initialized")
    
    def run_all_integration_tests(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete integration test suite.
        
        Args:
            save_results: Save results and visualizations
            
        Returns:
            Complete integration test results
        """
        logger.info("Starting complete system integration tests")
        start_time = time.time()
        
        # Define test sequence
        integration_tests = [
            ('poisson_solver_demo', self.demonstrate_poisson_solver),
            ('heat_solver_demo', self.demonstrate_heat_solver),
            ('convergence_study_demo', self.demonstrate_convergence_studies),
            ('performance_comparison_demo', self.demonstrate_performance_analysis),
            ('mixed_precision_demo', self.demonstrate_mixed_precision_analysis),
            ('full_validation_demo', self.demonstrate_comprehensive_validation),
            ('scaling_analysis_demo', self.demonstrate_scaling_analysis),
            ('real_world_problem_demo', self.demonstrate_real_world_problems)
        ]
        
        # Run all demonstrations
        for demo_name, demo_function in integration_tests:
            logger.info(f"Running {demo_name}")
            
            try:
                demo_start = time.time()
                result = demo_function()
                demo_time = time.time() - demo_start
                
                demo_result = DemoResult(
                    demo_name=demo_name,
                    description=result.get('description', ''),
                    success=result.get('success', True),
                    execution_time=demo_time,
                    key_metrics=result.get('metrics', {}),
                    visualizations=result.get('visualizations', []),
                    summary=result.get('summary', '')
                )
                
                self.demo_results.append(demo_result)
                
                logger.info(f"Completed {demo_name}: {'SUCCESS' if demo_result.success else 'FAILED'} "
                           f"({demo_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Demo {demo_name} failed: {e}")
                
                demo_result = DemoResult(
                    demo_name=demo_name,
                    description=f"Demo failed: {str(e)}",
                    success=False,
                    execution_time=time.time() - demo_start,
                    key_metrics={},
                    visualizations=[],
                    summary=f"ERROR: {str(e)}"
                )
                
                self.demo_results.append(demo_result)
        
        total_time = time.time() - start_time
        
        # Generate final report
        results = self._generate_integration_report(total_time)
        
        # Save results if requested
        if save_results:
            self._save_integration_results(results)
        
        logger.info(f"Integration tests completed in {total_time:.2f}s")
        
        return results
    
    def demonstrate_poisson_solver(self) -> Dict[str, Any]:
        """Demonstrate Poisson solver capabilities."""
        logger.info("Demonstrating Poisson solver")
        
        # Test multiple problems and configurations
        test_problems = PoissonTestProblems()
        problems_to_test = ['trigonometric', 'polynomial', 'high_frequency', 'boundary_layer']
        
        solver_configs = [
            {
                'name': 'CPU Multigrid',
                'config': {
                    'solver_type': 'multigrid',
                    'max_levels': 6,
                    'tolerance': 1e-8,
                    'use_gpu': False,
                    'enable_mixed_precision': False
                }
            },
            {
                'name': 'CPU Mixed Precision',
                'config': {
                    'solver_type': 'multigrid',
                    'max_levels': 6,
                    'tolerance': 1e-8,
                    'use_gpu': False,
                    'enable_mixed_precision': True
                }
            }
        ]
        
        results = {
            'description': 'Comprehensive Poisson solver demonstration with multiple problems and configurations',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'solutions': {}
        }
        
        for problem_name in problems_to_test:
            problem = test_problems.get_problem(problem_name)
            problem_results = {}
            
            for solver_info in solver_configs:
                solver = PoissonSolver2D(**solver_info['config'])
                
                try:
                    # Solve on multiple grid sizes
                    for nx, ny in [(65, 65), (129, 129)]:
                        result = solver.solve_poisson_problem(problem, nx, ny)
                        
                        key = f"{solver_info['name']}_{nx}x{ny}"
                        problem_results[key] = {
                            'solve_time': result['solve_time'],
                            'iterations': result['solver_info']['iterations'],
                            'l2_error': result['errors'].get('l2_error', 0) if 'errors' in result else 0,
                            'solution': result['solution']
                        }
                        
                        # Store solution for visualization
                        self.test_data['solutions'][f"{problem_name}_{key}"] = result
                        
                except Exception as e:
                    logger.error(f"Failed to solve {problem_name} with {solver_info['name']}: {e}")
                    results['success'] = False
            
            results['solutions'][problem_name] = problem_results
        
        # Calculate key metrics
        all_times = []
        all_errors = []
        
        for problem_name, problem_results in results['solutions'].items():
            for config_key, config_results in problem_results.items():
                all_times.append(config_results['solve_time'])
                if config_results['l2_error'] > 0:
                    all_errors.append(config_results['l2_error'])
        
        results['metrics'] = {
            'problems_tested': len(problems_to_test),
            'configurations_tested': len(solver_configs),
            'total_solves': len(all_times),
            'avg_solve_time': np.mean(all_times) if all_times else 0,
            'avg_accuracy': np.mean(all_errors) if all_errors else 0,
            'max_accuracy': np.min(all_errors) if all_errors else 0
        }
        
        results['summary'] = (f"Successfully demonstrated Poisson solver on {len(problems_to_test)} problems "
                             f"with {len(solver_configs)} configurations. "
                             f"Average solve time: {results['metrics']['avg_solve_time']:.4f}s, "
                             f"Average L2 error: {results['metrics']['avg_accuracy']:.2e}")
        
        return results
    
    def demonstrate_heat_solver(self) -> Dict[str, Any]:
        """Demonstrate heat equation solver capabilities."""
        logger.info("Demonstrating heat equation solver")
        
        # Test multiple heat equation problems
        heat_problems = HeatTestProblems()
        problems_to_test = ['pure_diffusion', 'heat_source', 'separable']
        
        # Test different time stepping methods
        time_configs = [
            TimeSteppingConfig(
                method=TimeSteppingMethod.BACKWARD_EULER,
                dt=0.01,
                t_final=0.1,
                adaptive_dt=False
            ),
            TimeSteppingConfig(
                method=TimeSteppingMethod.CRANK_NICOLSON,
                dt=0.01,
                t_final=0.1,
                adaptive_dt=False
            )
        ]
        
        solver_config = {
            'solver_type': 'multigrid',
            'max_levels': 5,
            'tolerance': 1e-8,
            'use_gpu': False,
            'enable_mixed_precision': False
        }
        
        results = {
            'description': 'Heat equation solver demonstration with multiple time stepping methods',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'heat_solutions': {}
        }
        
        solver = HeatSolver2D(**solver_config)
        
        for problem_name in problems_to_test:
            problem = heat_problems.get_problem(problem_name)
            problem_results = {}
            
            for i, time_config in enumerate(time_configs):
                method_name = time_config.method.value
                
                try:
                    # Solve on moderate grid size
                    result = solver.solve_heat_problem(
                        problem, 65, 65, time_config, save_solution_history=True
                    )
                    
                    problem_results[method_name] = {
                        'total_time': result['total_time'],
                        'total_steps': result['total_steps'],
                        'avg_mg_iterations': result['avg_mg_iterations'],
                        'final_l2_error': result['errors'].get('l2_error', 0) if 'errors' in result else 0,
                        'solution_history': result.get('solution_history', [])
                    }
                    
                    # Store for visualization
                    self.test_data['solutions'][f"heat_{problem_name}_{method_name}"] = result
                    
                except Exception as e:
                    logger.error(f"Failed to solve heat {problem_name} with {method_name}: {e}")
                    results['success'] = False
            
            results['heat_solutions'][problem_name] = problem_results
        
        # Calculate metrics
        all_times = []
        all_steps = []
        all_errors = []
        
        for problem_name, problem_results in results['heat_solutions'].items():
            for method_name, method_results in problem_results.items():
                all_times.append(method_results['total_time'])
                all_steps.append(method_results['total_steps'])
                if method_results['final_l2_error'] > 0:
                    all_errors.append(method_results['final_l2_error'])
        
        results['metrics'] = {
            'problems_tested': len(problems_to_test),
            'time_methods_tested': len(time_configs),
            'avg_solve_time': np.mean(all_times) if all_times else 0,
            'avg_time_steps': np.mean(all_steps) if all_steps else 0,
            'avg_final_error': np.mean(all_errors) if all_errors else 0
        }
        
        results['summary'] = (f"Successfully demonstrated heat solver on {len(problems_to_test)} problems "
                             f"with {len(time_configs)} time stepping methods. "
                             f"Average solve time: {results['metrics']['avg_solve_time']:.4f}s")
        
        return results
    
    def demonstrate_convergence_studies(self) -> Dict[str, Any]:
        """Demonstrate grid convergence analysis."""
        logger.info("Demonstrating convergence studies")
        
        analyzer = GridConvergenceAnalyzer()
        
        solver_config = {
            'solver_type': 'multigrid',
            'max_levels': 6,
            'tolerance': 1e-10,
            'use_gpu': False,
            'enable_mixed_precision': False
        }
        
        results = {
            'description': 'Grid convergence analysis demonstration',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'convergence_studies': {}
        }
        
        # Test Poisson convergence
        poisson_problems = ['trigonometric', 'polynomial']
        grid_sequence = [(17, 17), (33, 33), (65, 65), (129, 129)]
        
        for problem_name in poisson_problems:
            try:
                study = analyzer.run_poisson_convergence_study(
                    solver_config, problem_name, grid_sequence, theoretical_order=2.0
                )
                
                results['convergence_studies'][f"poisson_{problem_name}"] = {
                    'achieved_l2_order': study.achieved_order['l2_error'],
                    'achieved_max_order': study.achieved_order['max_error'],
                    'theoretical_order': study.theoretical_order,
                    'r_squared_l2': study.regression_data['l2_error'].get('r_squared', 0),
                    'grid_sizes': study.grid_sizes
                }
                
                # Store for visualization
                self.test_data['convergence_data'][f"poisson_{problem_name}"] = study
                
            except Exception as e:
                logger.error(f"Convergence study failed for {problem_name}: {e}")
                results['success'] = False
        
        # Test Heat convergence (smaller study)
        time_config = TimeSteppingConfig(
            method=TimeSteppingMethod.BACKWARD_EULER,
            dt=0.005,
            t_final=0.05
        )
        
        try:
            heat_study = analyzer.run_heat_convergence_study(
                solver_config, time_config, 'pure_diffusion', 
                grid_sequence[:3], theoretical_order=2.0
            )
            
            results['convergence_studies']['heat_pure_diffusion'] = {
                'achieved_l2_order': heat_study.achieved_order['l2_error'],
                'achieved_max_order': heat_study.achieved_order['max_error'],
                'theoretical_order': heat_study.theoretical_order
            }
            
            self.test_data['convergence_data']['heat_pure_diffusion'] = heat_study
            
        except Exception as e:
            logger.error(f"Heat convergence study failed: {e}")
            results['success'] = False
        
        # Calculate metrics
        achieved_orders_l2 = [
            study['achieved_l2_order'] for study in results['convergence_studies'].values()
            if 'achieved_l2_order' in study
        ]
        
        results['metrics'] = {
            'convergence_studies': len(results['convergence_studies']),
            'avg_l2_order': np.mean(achieved_orders_l2) if achieved_orders_l2 else 0,
            'convergence_quality': sum(1 for order in achieved_orders_l2 if abs(order - 2.0) < 0.3)
        }
        
        results['summary'] = (f"Completed {len(results['convergence_studies'])} convergence studies. "
                             f"Average L2 convergence order: {results['metrics']['avg_l2_order']:.2f}, "
                             f"{results['metrics']['convergence_quality']} studies achieved good convergence")
        
        return results
    
    def demonstrate_performance_analysis(self) -> Dict[str, Any]:
        """Demonstrate performance analysis capabilities."""
        logger.info("Demonstrating performance analysis")
        
        results = {
            'description': 'Comprehensive performance analysis demonstration',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'performance_data': {}
        }
        
        try:
            # Run comprehensive performance analysis
            perf_results = run_comprehensive_performance_analysis()
            
            # Extract key metrics
            benchmark_results = perf_results.get('benchmark_results', [])
            comparison_results = perf_results.get('comparison_results', [])
            
            if benchmark_results:
                cpu_results = [r for r in benchmark_results if r.device_info.get('device_type') != 'GPU']
                gpu_results = [r for r in benchmark_results if r.device_info.get('device_type') == 'GPU']
                
                results['performance_data']['cpu_benchmarks'] = len(cpu_results)
                results['performance_data']['gpu_benchmarks'] = len(gpu_results)
                
                if cpu_results:
                    results['performance_data']['avg_cpu_throughput'] = np.mean([r.throughput for r in cpu_results])
                if gpu_results:
                    results['performance_data']['avg_gpu_throughput'] = np.mean([r.throughput for r in gpu_results])
            
            if comparison_results:
                speedups = [c.speedup for c in comparison_results if c.gpu_result]
                if speedups:
                    results['performance_data']['avg_gpu_speedup'] = np.mean(speedups)
                    results['performance_data']['max_gpu_speedup'] = np.max(speedups)
            
            # Store for analysis
            self.test_data['performance_data'] = perf_results
            
            results['metrics'] = results['performance_data']
            results['summary'] = (f"Performance analysis completed with "
                                 f"{len(benchmark_results)} benchmarks and "
                                 f"{len(comparison_results)} CPU vs GPU comparisons")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            results['success'] = False
            results['summary'] = f"Performance analysis failed: {e}"
        
        return results
    
    def demonstrate_mixed_precision_analysis(self) -> Dict[str, Any]:
        """Demonstrate mixed-precision effectiveness analysis."""
        logger.info("Demonstrating mixed-precision analysis")
        
        results = {
            'description': 'Mixed-precision effectiveness analysis demonstration',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'precision_data': {}
        }
        
        try:
            # Run mixed-precision analysis
            precision_results = run_comprehensive_mixed_precision_analysis()
            
            # Extract key metrics
            precision_comparisons = precision_results.get('precision_comparisons', [])
            
            if precision_comparisons:
                valid_mixed = [c for c in precision_comparisons if c.mixed_result and c.double_result]
                valid_single = [c for c in precision_comparisons if c.single_result and c.double_result]
                
                if valid_mixed:
                    mixed_speedups = [c.speedup_mixed_vs_double for c in valid_mixed]
                    memory_savings = [c.memory_saving_mixed for c in valid_mixed if c.memory_saving_mixed >= 0]
                    
                    results['precision_data']['avg_mixed_speedup'] = np.mean(mixed_speedups)
                    results['precision_data']['max_mixed_speedup'] = np.max(mixed_speedups)
                    
                    if memory_savings:
                        results['precision_data']['avg_memory_saving'] = np.mean(memory_savings)
                
                if valid_single:
                    single_speedups = [c.speedup_single_vs_double for c in valid_single]
                    results['precision_data']['avg_single_speedup'] = np.mean(single_speedups)
            
            # Store for analysis
            self.test_data['precision_data'] = precision_results
            
            results['metrics'] = results['precision_data']
            results['summary'] = (f"Mixed-precision analysis completed with "
                                 f"{len(precision_comparisons)} comparisons. "
                                 f"Average mixed-precision speedup: "
                                 f"{results['precision_data'].get('avg_mixed_speedup', 0):.2f}x")
            
        except Exception as e:
            logger.error(f"Mixed-precision analysis failed: {e}")
            results['success'] = False
            results['summary'] = f"Mixed-precision analysis failed: {e}"
        
        return results
    
    def demonstrate_comprehensive_validation(self) -> Dict[str, Any]:
        """Demonstrate comprehensive validation suite."""
        logger.info("Demonstrating comprehensive validation")
        
        results = {
            'description': 'Comprehensive validation suite demonstration',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'validation_data': {}
        }
        
        try:
            # Run comprehensive validation in quick mode
            validation_result = run_comprehensive_validation(quick_mode=True, save_results=False)
            
            results['validation_data'] = {
                'total_tests': validation_result.total_tests,
                'passed_tests': validation_result.passed_tests,
                'failed_tests': validation_result.failed_tests,
                'pass_rate': validation_result.passed_tests / validation_result.total_tests if validation_result.total_tests > 0 else 0,
                'execution_time': validation_result.total_execution_time
            }
            
            # Store detailed results
            self.test_data['validation_results'] = validation_result
            
            results['metrics'] = results['validation_data']
            results['summary'] = (f"Validation suite completed: "
                                 f"{validation_result.passed_tests}/{validation_result.total_tests} tests passed "
                                 f"({results['validation_data']['pass_rate']:.1%} pass rate)")
            
            if validation_result.failed_tests > 0:
                results['success'] = False
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            results['success'] = False
            results['summary'] = f"Validation failed: {e}"
        
        return results
    
    def demonstrate_scaling_analysis(self) -> Dict[str, Any]:
        """Demonstrate scalability analysis."""
        logger.info("Demonstrating scaling analysis")
        
        analyzer = PerformanceAnalyzer()
        
        results = {
            'description': 'Scalability analysis demonstration',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'scaling_data': {}
        }
        
        try:
            # Scaling analysis
            grid_sizes = [(33, 33), (65, 65), (129, 129)]
            solver_configs = [
                {
                    'solver_type': 'multigrid',
                    'max_levels': 6,
                    'tolerance': 1e-8,
                    'use_gpu': False,
                    'enable_mixed_precision': False
                }
            ]
            
            scaling_analysis = analyzer.analyze_scaling_performance(
                'trigonometric', grid_sizes, solver_configs
            )
            
            # Extract scaling metrics
            for config_name, analysis in scaling_analysis['scaling_results'].items():
                if 'time_scaling_exponent' in analysis:
                    results['scaling_data'][f'{config_name}_time_exponent'] = analysis['time_scaling_exponent']
                if 'parallel_efficiency' in analysis:
                    results['scaling_data'][f'{config_name}_efficiency'] = analysis['parallel_efficiency']
            
            results['metrics'] = results['scaling_data']
            results['summary'] = (f"Scaling analysis completed for {len(grid_sizes)} grid sizes. "
                                 f"Results stored for detailed analysis.")
            
        except Exception as e:
            logger.error(f"Scaling analysis failed: {e}")
            results['success'] = False
            results['summary'] = f"Scaling analysis failed: {e}"
        
        return results
    
    def demonstrate_real_world_problems(self) -> Dict[str, Any]:
        """Demonstrate solver on realistic problems."""
        logger.info("Demonstrating real-world problems")
        
        results = {
            'description': 'Real-world problem demonstration',
            'success': True,
            'metrics': {},
            'visualizations': [],
            'real_world_data': {}
        }
        
        # Create realistic test problems
        problems = BenchmarkProblems()
        
        # Test challenging problems
        challenging_problems = problems.get_accuracy_problems()['challenging_solutions']
        
        solver_config = {
            'solver_type': 'multigrid',
            'max_levels': 7,
            'tolerance': 1e-10,
            'use_gpu': False,
            'enable_mixed_precision': True
        }
        
        solver = PoissonSolver2D(**solver_config)
        
        for i, problem in enumerate(challenging_problems[:2]):  # Test first 2 challenging problems
            problem_name = f"challenging_problem_{i+1}"
            
            try:
                # Solve on fine grid
                result = solver.solve_poisson_problem(problem, 129, 129)
                
                results['real_world_data'][problem_name] = {
                    'solve_time': result['solve_time'],
                    'iterations': result['solver_info']['iterations'],
                    'converged': result['solver_info'].get('converged', True),
                    'final_residual': result['solver_info'].get('final_residual', 0),
                    'l2_error': result['errors'].get('l2_error', 0) if 'errors' in result else 0
                }
                
                if not result['solver_info'].get('converged', True):
                    results['success'] = False
                
            except Exception as e:
                logger.error(f"Real-world problem {problem_name} failed: {e}")
                results['success'] = False
        
        # Calculate metrics
        if results['real_world_data']:
            solve_times = [data['solve_time'] for data in results['real_world_data'].values()]
            iterations = [data['iterations'] for data in results['real_world_data'].values()]
            converged_count = sum(1 for data in results['real_world_data'].values() if data['converged'])
            
            results['metrics'] = {
                'problems_solved': len(results['real_world_data']),
                'avg_solve_time': np.mean(solve_times),
                'avg_iterations': np.mean(iterations),
                'convergence_rate': converged_count / len(results['real_world_data'])
            }
        
        results['summary'] = (f"Solved {len(results['real_world_data'])} challenging problems. "
                             f"Average solve time: {results['metrics'].get('avg_solve_time', 0):.4f}s, "
                             f"Convergence rate: {results['metrics'].get('convergence_rate', 0):.1%}")
        
        return results
    
    def _generate_integration_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive integration test report."""
        successful_demos = sum(1 for demo in self.demo_results if demo.success)
        
        results = {
            'integration_summary': {
                'total_demonstrations': len(self.demo_results),
                'successful_demonstrations': successful_demos,
                'success_rate': successful_demos / len(self.demo_results) if self.demo_results else 0,
                'total_execution_time': total_time
            },
            'demonstration_results': self.demo_results,
            'test_data': self.test_data,
            'system_capabilities': self._assess_system_capabilities(),
            'recommendations': self._generate_recommendations()
        }
        
        return results
    
    def _assess_system_capabilities(self) -> Dict[str, Any]:
        """Assess overall system capabilities based on integration tests."""
        capabilities = {
            'poisson_solving': 'unknown',
            'heat_equation_solving': 'unknown',
            'convergence_behavior': 'unknown',
            'performance_characteristics': 'unknown',
            'mixed_precision_effectiveness': 'unknown',
            'scalability': 'unknown',
            'overall_reliability': 'unknown'
        }
        
        # Assess based on demonstration results
        for demo in self.demo_results:
            if demo.demo_name == 'poisson_solver_demo' and demo.success:
                avg_error = demo.key_metrics.get('avg_accuracy', 1.0)
                if avg_error < 1e-6:
                    capabilities['poisson_solving'] = 'excellent'
                elif avg_error < 1e-4:
                    capabilities['poisson_solving'] = 'good'
                else:
                    capabilities['poisson_solving'] = 'acceptable'
            
            elif demo.demo_name == 'heat_solver_demo' and demo.success:
                avg_error = demo.key_metrics.get('avg_final_error', 1.0)
                if avg_error < 1e-4:
                    capabilities['heat_equation_solving'] = 'excellent'
                elif avg_error < 1e-2:
                    capabilities['heat_equation_solving'] = 'good'
                else:
                    capabilities['heat_equation_solving'] = 'acceptable'
            
            elif demo.demo_name == 'convergence_study_demo' and demo.success:
                convergence_quality = demo.key_metrics.get('convergence_quality', 0)
                total_studies = demo.key_metrics.get('convergence_studies', 1)
                if convergence_quality / total_studies > 0.8:
                    capabilities['convergence_behavior'] = 'excellent'
                elif convergence_quality / total_studies > 0.6:
                    capabilities['convergence_behavior'] = 'good'
                else:
                    capabilities['convergence_behavior'] = 'needs_improvement'
            
            elif demo.demo_name == 'mixed_precision_demo' and demo.success:
                speedup = demo.key_metrics.get('avg_mixed_speedup', 1.0)
                if speedup > 1.5:
                    capabilities['mixed_precision_effectiveness'] = 'excellent'
                elif speedup > 1.2:
                    capabilities['mixed_precision_effectiveness'] = 'good'
                else:
                    capabilities['mixed_precision_effectiveness'] = 'limited'
        
        # Overall reliability
        success_rate = sum(1 for demo in self.demo_results if demo.success) / len(self.demo_results) if self.demo_results else 0
        if success_rate >= 0.9:
            capabilities['overall_reliability'] = 'excellent'
        elif success_rate >= 0.7:
            capabilities['overall_reliability'] = 'good'
        else:
            capabilities['overall_reliability'] = 'needs_improvement'
        
        return capabilities
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on integration test results."""
        recommendations = []
        
        failed_demos = [demo for demo in self.demo_results if not demo.success]
        
        if not failed_demos:
            recommendations.append("✓ All integration tests passed - system is production ready")
        else:
            recommendations.append(f"⚠ {len(failed_demos)} demonstration(s) failed - investigate before deployment")
        
        # Performance-specific recommendations
        perf_demo = next((demo for demo in self.demo_results if demo.demo_name == 'performance_comparison_demo'), None)
        if perf_demo and perf_demo.success:
            gpu_speedup = perf_demo.key_metrics.get('avg_gpu_speedup', 0)
            if gpu_speedup > 2.0:
                recommendations.append("✓ GPU acceleration provides significant benefits - recommend GPU deployment")
            elif gpu_speedup > 1.3:
                recommendations.append("• GPU acceleration provides moderate benefits - consider for large problems")
            else:
                recommendations.append("• GPU acceleration limited - evaluate cost/benefit for deployment")
        
        # Mixed precision recommendations
        precision_demo = next((demo for demo in self.demo_results if demo.demo_name == 'mixed_precision_demo'), None)
        if precision_demo and precision_demo.success:
            mixed_speedup = precision_demo.key_metrics.get('avg_mixed_speedup', 0)
            if mixed_speedup > 1.3:
                recommendations.append("✓ Mixed precision recommended for production - good performance/accuracy balance")
            else:
                recommendations.append("• Mixed precision benefits limited - use double precision for critical applications")
        
        # Validation recommendations
        validation_demo = next((demo for demo in self.demo_results if demo.demo_name == 'full_validation_demo'), None)
        if validation_demo and validation_demo.success:
            pass_rate = validation_demo.key_metrics.get('pass_rate', 0)
            if pass_rate >= 0.95:
                recommendations.append("✓ Validation suite passed - high confidence in system reliability")
            elif pass_rate >= 0.8:
                recommendations.append("• Most validation tests passed - review failed tests before deployment")
            else:
                recommendations.append("⚠ Significant validation issues - requires investigation")
        
        return recommendations
    
    def _save_integration_results(self, results: Dict[str, Any]) -> None:
        """Save integration test results to files."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save summary report
            report = self._generate_integration_summary_report(results)
            report_filename = f"integration_test_report_{timestamp}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            
            # Save detailed results
            # Convert to JSON-serializable format
            json_results = {
                'integration_summary': results['integration_summary'],
                'demonstration_results': [
                    {
                        'demo_name': demo.demo_name,
                        'description': demo.description,
                        'success': demo.success,
                        'execution_time': demo.execution_time,
                        'key_metrics': demo.key_metrics,
                        'summary': demo.summary
                    }
                    for demo in results['demonstration_results']
                ],
                'system_capabilities': results['system_capabilities'],
                'recommendations': results['recommendations']
            }
            
            results_filename = f"integration_test_results_{timestamp}.json"
            with open(results_filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Integration test results saved to {report_filename} and {results_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save integration results: {e}")
    
    def _generate_integration_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable integration test report."""
        summary = results['integration_summary']
        capabilities = results['system_capabilities']
        recommendations = results['recommendations']
        
        report_lines = [
            "MIXED-PRECISION MULTIGRID SOLVER",
            "INTEGRATION TEST REPORT",
            "=" * 60,
            "",
            f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Execution Time: {summary['total_execution_time']:.2f} seconds",
            "",
            "OVERALL RESULTS:",
            f"Demonstrations Run: {summary['total_demonstrations']}",
            f"Successful: {summary['successful_demonstrations']}",
            f"Failed: {summary['total_demonstrations'] - summary['successful_demonstrations']}",
            f"Success Rate: {summary['success_rate']:.1%}",
            "",
            "SYSTEM CAPABILITIES ASSESSMENT:",
            "-" * 35
        ]
        
        for capability, assessment in capabilities.items():
            capability_name = capability.replace('_', ' ').title()
            report_lines.append(f"{capability_name:30s}: {assessment.upper()}")
        
        report_lines.extend([
            "",
            "DEMONSTRATION RESULTS:",
            "-" * 25
        ])
        
        for demo in results['demonstration_results']:
            status = "PASS" if demo.success else "FAIL"
            report_lines.extend([
                f"{demo.demo_name:30s}: {status:4s} ({demo.execution_time:.2f}s)",
                f"  {demo.summary}",
                ""
            ])
        
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 15,
            ""
        ])
        
        for rec in recommendations:
            report_lines.append(f"  {rec}")
        
        report_lines.extend([
            "",
            "CONCLUSION:",
            "-" * 11,
            ""
        ])
        
        if summary['success_rate'] >= 0.9:
            report_lines.extend([
                "✓ System integration tests passed successfully",
                "✓ Mixed-precision multigrid solver is ready for production use",
                "✓ All major components validated and working correctly"
            ])
        elif summary['success_rate'] >= 0.7:
            report_lines.extend([
                "• System integration mostly successful",
                "• Review failed demonstrations before production deployment",
                "• Core functionality validated"
            ])
        else:
            report_lines.extend([
                "⚠ Significant integration issues detected",
                "⚠ System requires investigation and fixes",
                "⚠ Not recommended for production use in current state"
            ])
        
        return "\n".join(report_lines)


def run_complete_system_demonstration(save_results: bool = True) -> Dict[str, Any]:
    """
    Run complete system demonstration and integration tests.
    
    Args:
        save_results: Save results and visualizations
        
    Returns:
        Complete demonstration results
    """
    logger.info("Starting complete system demonstration")
    
    integration_tests = SystemIntegrationTests()
    
    results = integration_tests.run_all_integration_tests(save_results=save_results)
    
    logger.info("Complete system demonstration finished")
    
    return results


def run_quick_integration_tests() -> Dict[str, Any]:
    """Run quick integration tests for development/CI."""
    logger.info("Running quick integration tests")
    
    integration_tests = SystemIntegrationTests()
    
    # Run subset of tests for quick feedback
    quick_results = {
        'poisson_demo': integration_tests.demonstrate_poisson_solver(),
        'heat_demo': integration_tests.demonstrate_heat_solver(),
        'validation_demo': integration_tests.demonstrate_comprehensive_validation()
    }
    
    return quick_results