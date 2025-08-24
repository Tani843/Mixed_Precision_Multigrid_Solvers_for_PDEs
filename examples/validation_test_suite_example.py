#!/usr/bin/env python3
"""
Validation Test Suite Example
Demonstrates comprehensive MMS validation and performance baseline establishment
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multigrid.validation.mms_validation import MMSValidator, complete_mms_validation
from multigrid.validation.performance_baselines import PerformanceBaselines, establish_performance_baselines
from multigrid.core.precision import PrecisionLevel


def run_mms_validation_example():
    """Run Method of Manufactured Solutions validation example."""
    
    print("="*80)
    print("METHOD OF MANUFACTURED SOLUTIONS (MMS) VALIDATION EXAMPLE")
    print("="*80)
    
    # Initialize validator
    validator = MMSValidator()
    
    # Show available test problems
    print(f"\nAvailable MMS Test Problems ({len(validator.test_problems)}):")
    print("-" * 60)
    
    for name, problem in validator.test_problems.items():
        dimensions = "2D" if problem.spatial_dimensions == 2 else "3D"
        time_dep = "Time-dependent" if problem.time_dependent else "Steady-state"
        print(f"  {name:25s}: {problem.name}")
        print(f"    Type: {problem.problem_type.value}, {dimensions}, {time_dep}")
        print(f"    Expected convergence rate: {problem.expected_convergence_rate}")
        print()
    
    # Run specific test examples
    test_examples = [
        'quadratic_2d',           # Simple polynomial test
        'trigonometric_single_2d', # Basic trigonometric test
        'heat_2d_exponential'     # Time-dependent test
    ]
    
    print(f"Running detailed validation on selected problems:")
    for problem_name in test_examples:
        print(f"  • {problem_name}")
    print()
    
    # Run convergence studies
    grid_sizes = [17, 33, 65]  # Small sizes for example
    precision_levels = [PrecisionLevel.SINGLE, PrecisionLevel.DOUBLE]
    
    detailed_results = {}
    
    for problem_name in test_examples:
        print(f"\n{'='*60}")
        print(f"TESTING: {validator.test_problems[problem_name].name}")
        print(f"{'='*60}")
        
        try:
            result = validator.run_convergence_study(
                problem_name=problem_name,
                grid_sizes=grid_sizes,
                precision_levels=precision_levels
            )
            
            detailed_results[problem_name] = result
            
            # Print summary
            print(f"Problem: {result['problem_info'].name}")
            print(f"Type: {result['problem_info'].problem_type.value}")
            print(f"Expected rate: {result['theoretical_rate']}")
            
            for precision_str, data in result['convergence_data'].items():
                if data['l2_errors']:
                    print(f"\n{precision_str.upper()} PRECISION:")
                    print("  Grid    h       L2 Error    Solve Time  Iterations")
                    print("  " + "-"*50)
                    
                    for i, size in enumerate(data['grid_sizes']):
                        h = data['h_values'][i]
                        error = data['l2_errors'][i]
                        time_val = data['solve_times'][i]
                        iters = data['solver_iterations'][i]
                        print(f"  {size:4d}  {h:.4f}  {error:.2e}   {time_val:.3f}s     {iters:3d}")
                    
                    if 'average_convergence_rate' in data:
                        avg_rate = data['average_convergence_rate']
                        rate_std = data['convergence_rate_std']
                        expected = result['theoretical_rate']
                        status = "✅ PASS" if abs(avg_rate - expected) < 0.3 else "❌ FAIL"
                        
                        print(f"\n  Convergence rate: {avg_rate:.2f} ± {rate_std:.3f} {status}")
                        print(f"  Expected rate:    {expected:.2f}")
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
            continue
    
    # Create convergence plots for successful tests
    if detailed_results:
        print(f"\nCreating convergence plots...")
        validator.create_convergence_plots({'all_results': detailed_results}, 
                                         save_dir="mms_example_plots")
        print(f"Plots saved to: mms_example_plots/")
    
    return detailed_results


def run_performance_baseline_example():
    """Run performance baseline establishment example."""
    
    print("\n" + "="*80)  
    print("PERFORMANCE BASELINE ESTABLISHMENT EXAMPLE")
    print("="*80)
    
    # Initialize baseline suite
    baseline_suite = PerformanceBaselines()
    
    # Show system information
    print("System Information:")
    print(f"  CPU: {baseline_suite.system_info['cpu_info']}")
    print(f"  Cores: {baseline_suite.system_info['cpu_cores']} physical, {baseline_suite.system_info['cpu_threads']} logical")
    print(f"  Memory: {baseline_suite.system_info['total_memory_gb']:.1f} GB")
    print(f"  Python: {baseline_suite.system_info['python_version']}")
    print(f"  NumPy: {baseline_suite.system_info['numpy_version']}")
    
    # Show available external solvers
    print(f"\nExternal Solver Availability:")
    for solver_name, available in baseline_suite.external_solvers.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {solver_name.upper():10s}: {status}")
    
    # Run baseline establishment with smaller problem sizes for example
    problem_sizes = [33, 65, 129]
    
    print(f"\nRunning baseline establishment...")
    print(f"Problem sizes: {problem_sizes}")
    
    baselines = baseline_suite.establish_performance_baselines(
        problem_sizes=problem_sizes,
        include_external=True
    )
    
    # Print detailed results
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Solver comparison results
    if 'solver_comparisons' in baselines:
        print(f"\nSOLVER PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        for size_key, results in baselines['solver_comparisons'].items():
            if size_key.startswith('problem_size_'):
                size = size_key.split('_')[-1]
                print(f"\nProblem size {size}×{size}:")
                print(f"  {'Solver':25s} {'Time':>8s} {'Iter':>6s} {'Error':>12s} {'Status':>10s}")
                print(f"  {'-'*60}")
                
                for solver_key, data in results.items():
                    if 'error' not in data:  # Skip failed solvers
                        name = data['solver_name']
                        time_str = f"{data['avg_time']:.3f}s"
                        iter_str = f"{data['iterations']}"
                        error_str = f"{data['l2_error']:.2e}"
                        status = "✅ Pass" if data['converged'] else "❌ Fail"
                        
                        print(f"  {name:25s} {time_str:>8s} {iter_str:>6s} {error_str:>12s} {status:>10s}")
    
    # Computational complexity analysis
    if 'scaling_analysis' in baselines and 'computational_complexity' in baselines['scaling_analysis']:
        comp = baselines['scaling_analysis']['computational_complexity']
        print(f"\nCOMPUTATIONAL COMPLEXITY ANALYSIS:")
        print(f"  Measured exponent: {comp['exponent']:.2f}")
        print(f"  Interpretation: {comp['interpretation']}")
        print(f"  Goodness of fit (R²): {comp['r_squared']:.3f}")
    
    # Precision effectiveness
    if 'precision_effectiveness' in baselines:
        print(f"\nMIXED-PRECISION EFFECTIVENESS:")
        print("-" * 50)
        
        for size_key, results in baselines['precision_effectiveness'].items():
            if 'effectiveness' in results:
                size = size_key.split('_')[1]
                eff = results['effectiveness']
                
                print(f"\nSize {size}×{size}:")
                print(f"  Error improvement (double vs single): {eff['error_improvement_double']:.1f}×")
                print(f"  Time overhead (double vs single): {eff['time_overhead_double']:.1f}×")
                print(f"  Adaptive error vs single: {eff['adaptive_error_vs_single']:.1f}×")
                print(f"  Adaptive time vs single: {eff['adaptive_time_vs_single']:.1f}×")
                
                # Calculate effectiveness ratio
                if eff['time_overhead_double'] > 0:
                    effectiveness = eff['error_improvement_double'] / eff['time_overhead_double']
                    print(f"  Overall effectiveness (improvement/overhead): {effectiveness:.1f}")
    
    # Memory analysis
    if 'memory_analysis' in baselines:
        memory = baselines['memory_analysis']
        print(f"\nMEMORY USAGE ANALYSIS:")
        print("-" * 50)
        print(f"  {'Size':>6s} {'Theoretical (MB)':>15s} {'Actual (MB)':>12s} {'Efficiency':>12s}")
        print(f"  {'-'*50}")
        
        for i, size in enumerate(memory['problem_sizes']):
            theo = memory['theoretical_memory_mb'][i]
            actual = memory['actual_memory_mb'][i]
            efficiency = memory['memory_efficiency'][i]
            
            print(f"  {size:4d}×{size:<4d} {theo:>12.1f} {actual:>12.1f} {efficiency:>10.1%}")
    
    # Generate plots
    print(f"\nGenerating performance plots...")
    baseline_suite.create_baseline_plots(baselines, save_dir="baseline_example_plots")
    print(f"Plots saved to: baseline_example_plots/")
    
    # Save detailed report
    report_file = "baseline_example_report.json"
    baseline_suite.generate_baseline_report(baselines, output_file=report_file)
    print(f"Detailed report saved to: {report_file}")
    
    return baselines


def create_comprehensive_validation_summary(mms_results, baseline_results):
    """Create a comprehensive validation summary."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("="*80)
    
    # MMS Validation Summary
    print(f"\n📊 METHOD OF MANUFACTURED SOLUTIONS:")
    successful_mms = len(mms_results)
    total_tests = len(mms_results) * 2  # Two precision levels per problem
    
    print(f"  Problems tested: {len(mms_results)}")
    print(f"  Total test cases: {total_tests}")
    
    convergence_passed = 0
    convergence_total = 0
    
    for problem_name, result in mms_results.items():
        problem_info = result['problem_info']
        expected_rate = result['theoretical_rate']
        
        print(f"\n  {problem_info.name}:")
        for precision_str, data in result['convergence_data'].items():
            if 'average_convergence_rate' in data:
                rate = data['average_convergence_rate']
                rate_ok = abs(rate - expected_rate) < 0.3
                status = "✅ PASS" if rate_ok else "❌ FAIL"
                
                print(f"    {precision_str}: Rate {rate:.2f} (expected {expected_rate:.2f}) {status}")
                
                convergence_total += 1
                if rate_ok:
                    convergence_passed += 1
    
    convergence_success_rate = convergence_passed / convergence_total if convergence_total > 0 else 0
    print(f"\n  Overall convergence rate success: {convergence_success_rate:.1%} ({convergence_passed}/{convergence_total})")
    
    # Performance Baseline Summary
    print(f"\n🚀 PERFORMANCE BASELINES:")
    
    if 'solver_comparisons' in baseline_results:
        # Find our solver's performance relative to others
        our_solver_times = []
        external_solver_times = []
        
        for size_key, results in baseline_results['solver_comparisons'].items():
            if size_key.startswith('problem_size_'):
                our_time = None
                other_times = []
                
                for solver_key, data in results.items():
                    if 'error' not in data:
                        if 'mixed_precision_mg' in solver_key:
                            our_time = data['avg_time']
                        else:
                            other_times.append(data['avg_time'])
                
                if our_time is not None:
                    our_solver_times.append(our_time)
                    if other_times:
                        external_solver_times.append(min(other_times))  # Best competing time
        
        if our_solver_times and external_solver_times:
            avg_our_time = np.mean(our_solver_times)
            avg_external_time = np.mean(external_solver_times)
            relative_performance = avg_external_time / avg_our_time
            
            if relative_performance > 1.2:
                performance_status = "✅ Superior"
            elif relative_performance > 0.8:
                performance_status = "✅ Competitive"
            else:
                performance_status = "⚠️  Needs improvement"
            
            print(f"  Relative to external solvers: {relative_performance:.1f}× {performance_status}")
    
    # Scaling assessment
    if 'scaling_analysis' in baseline_results:
        scaling = baseline_results['scaling_analysis']
        if 'computational_complexity' in scaling:
            exponent = scaling['computational_complexity']['exponent']
            r_squared = scaling['computational_complexity']['r_squared']
            
            if exponent < 1.5 and r_squared > 0.9:
                scaling_status = "✅ Excellent (Near-linear)"
            elif exponent < 2.2 and r_squared > 0.8:
                scaling_status = "✅ Good (Expected for 2D)"
            elif exponent < 3.0:
                scaling_status = "⚠️  Acceptable"
            else:
                scaling_status = "❌ Poor scaling"
            
            print(f"  Computational complexity: O(N^{exponent:.2f}) {scaling_status}")
    
    # Precision effectiveness assessment
    if 'precision_effectiveness' in baseline_results:
        total_improvement = 0
        total_overhead = 0
        count = 0
        
        for size_key, results in baseline_results['precision_effectiveness'].items():
            if 'effectiveness' in results:
                eff = results['effectiveness']
                total_improvement += eff['error_improvement_double']
                total_overhead += eff['time_overhead_double']
                count += 1
        
        if count > 0:
            avg_improvement = total_improvement / count
            avg_overhead = total_overhead / count
            effectiveness_ratio = avg_improvement / avg_overhead
            
            if effectiveness_ratio > 2.0:
                precision_status = "✅ Highly effective"
            elif effectiveness_ratio > 1.0:
                precision_status = "✅ Effective"
            else:
                precision_status = "⚠️  Limited effectiveness"
            
            print(f"  Mixed precision effectiveness: {avg_improvement:.1f}× improvement, {avg_overhead:.1f}× overhead {precision_status}")
    
    # Overall assessment
    print(f"\n📈 OVERALL VALIDATION ASSESSMENT:")
    
    validation_scores = []
    
    # MMS score (0-100)
    mms_score = convergence_success_rate * 100
    validation_scores.append(mms_score)
    print(f"  Mathematical Validation Score: {mms_score:.0f}/100")
    
    # Performance score (0-100, relative to theoretical optimal)
    if 'scaling_analysis' in baseline_results:
        scaling = baseline_results['scaling_analysis']
        if 'computational_complexity' in scaling:
            exponent = scaling['computational_complexity']['exponent']
            # Score based on how close to optimal O(N) scaling
            performance_score = max(0, 100 - (exponent - 1.0) * 50)
            validation_scores.append(performance_score)
            print(f"  Performance Score: {performance_score:.0f}/100")
    
    # Overall score
    if validation_scores:
        overall_score = np.mean(validation_scores)
        
        if overall_score >= 90:
            grade = "A+"
            status = "🏆 Excellent"
        elif overall_score >= 80:
            grade = "A"
            status = "✅ Very Good"
        elif overall_score >= 70:
            grade = "B"
            status = "✅ Good"
        elif overall_score >= 60:
            grade = "C"
            status = "⚠️  Acceptable"
        else:
            grade = "D"
            status = "❌ Needs Improvement"
        
        print(f"\n  🎯 OVERALL VALIDATION SCORE: {overall_score:.0f}/100 (Grade: {grade}) {status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if convergence_success_rate < 0.8:
        print("  • Improve convergence rate validation - check numerical implementation")
    
    if 'scaling_analysis' in baseline_results:
        scaling = baseline_results['scaling_analysis']
        if 'computational_complexity' in scaling:
            exponent = scaling['computational_complexity']['exponent']
            if exponent > 2.5:
                print("  • Optimize algorithmic complexity - current scaling is suboptimal")
    
    print("  • Consider additional test problems for more comprehensive validation")
    print("  • Run larger-scale benchmarks when computational resources allow")
    print("  • Compare against additional external solvers if available")


def main():
    """Run comprehensive validation test suite example."""
    
    print("Validation Test Suite Comprehensive Example")
    print("Mixed-Precision Multigrid Solvers")
    print("=" * 80)
    
    # Run MMS validation
    mms_results = run_mms_validation_example()
    
    # Run performance baseline establishment
    baseline_results = run_performance_baseline_example()
    
    # Create comprehensive summary
    create_comprehensive_validation_summary(mms_results, baseline_results)
    
    print(f"\n📁 FILES GENERATED:")
    print(f"  • mms_example_plots/         - MMS convergence plots")
    print(f"  • baseline_example_plots/    - Performance comparison plots")
    print(f"  • baseline_example_report.json - Detailed performance data")
    
    print(f"\n🎉 Validation test suite example completed successfully!")
    
    return mms_results, baseline_results


if __name__ == "__main__":
    main()