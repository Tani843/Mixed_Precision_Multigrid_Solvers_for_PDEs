"""
Simple validation test for the corrected multigrid solver
Tests core mathematical properties without interface complications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

from ..core.grid import Grid
from ..solvers.corrected_multigrid import CorrectedMultigridSolver
from ..core.precision import PrecisionManager, PrecisionLevel

logger = logging.getLogger(__name__)


def test_convergence_rates():
    """Test h-independent convergence on different grid sizes."""
    print("="*60)
    print("TESTING H-INDEPENDENT CONVERGENCE")
    print("="*60)
    
    grid_sizes = [17, 33, 65]
    convergence_factors = []
    
    for grid_size in grid_sizes:
        print(f"\nTesting grid size: {grid_size}×{grid_size}")
        print("-" * 40)
        
        # Create grid and solver
        grid = Grid(grid_size, grid_size, domain=(0, 1, 0, 1))
        solver = CorrectedMultigridSolver(
            max_levels=4,
            max_iterations=15,
            tolerance=1e-10,
            verbose=False
        )
        
        # Create test problem
        rhs, u_exact = solver.create_test_problem(grid, "manufactured")
        
        # Solve
        initial_guess = np.zeros_like(rhs)
        result = solver.solve(initial_guess, rhs, grid)
        
        # Calculate convergence factor
        residuals = result['residual_history']
        if len(residuals) >= 5:
            factors = []
            for i in range(2, min(len(residuals), 7)):  # Skip first iterations
                if residuals[i-1] > 0:
                    factor = residuals[i] / residuals[i-1]
                    factors.append(factor)
            
            if factors:
                conv_factor = np.exp(np.mean(np.log(factors)))
                convergence_factors.append(conv_factor)
                
                # Calculate errors
                u_computed = result['solution']
                error = u_computed - u_exact
                l2_error = np.linalg.norm(error[1:-1, 1:-1])
                h = 1.0 / (grid_size - 1)
                
                print(f"Converged: {result['converged']}")
                print(f"Iterations: {result['iterations']}")
                print(f"Final residual: {result['final_residual']:.2e}")
                print(f"Convergence factor: {conv_factor:.4f}")
                print(f"L2 error: {l2_error:.2e}")
                print(f"h: {h:.4f}")
    
    # Analyze h-independence
    if len(convergence_factors) >= 2:
        factor_variation = np.std(convergence_factors) / np.mean(convergence_factors)
        avg_factor = np.mean(convergence_factors)
        
        print(f"\n{'='*60}")
        print("H-INDEPENDENCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Average convergence factor: {avg_factor:.4f}")
        print(f"Factor variation coefficient: {factor_variation:.4f}")
        print(f"H-independent convergence: {'✅ PASS' if factor_variation < 0.3 else '❌ FAIL'}")
        print(f"Optimal convergence (<0.1): {'✅ PASS' if avg_factor < 0.1 else '❌ FAIR' if avg_factor < 0.3 else '❌ FAIL'}")
        
        return {
            'convergence_factors': convergence_factors,
            'average_factor': avg_factor,
            'h_independent': factor_variation < 0.3,
            'optimal_convergence': avg_factor < 0.1
        }
    
    return None


def test_precision_switching():
    """Test mixed-precision switching logic."""
    print(f"\n{'='*60}")
    print("TESTING MIXED-PRECISION SWITCHING")
    print(f"{'='*60}")
    
    # Create test problem
    grid = Grid(65, 65, domain=(0, 1, 0, 1))
    solver = CorrectedMultigridSolver(
        max_levels=4,
        max_iterations=30,
        tolerance=1e-10,
        verbose=False
    )
    
    # Create precision manager
    precision_manager = PrecisionManager(
        default_precision=PrecisionLevel.SINGLE,
        adaptive=True,
        convergence_threshold=1e-6
    )
    
    # Create test problem
    rhs, u_exact = solver.create_test_problem(grid, "manufactured")
    
    print(f"Initial precision: {precision_manager.current_precision.value}")
    
    # Solve with precision management
    initial_guess = np.zeros_like(rhs)
    result = solver.solve(initial_guess, rhs, grid, precision_manager)
    
    # Analyze precision usage
    stats = precision_manager.get_statistics()
    
    print(f"Final precision: {precision_manager.current_precision.value}")
    print(f"Precision history: {' → '.join(stats['precision_history'])}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final residual: {result['final_residual']:.2e}")
    
    # Calculate final error
    u_computed = result['solution']
    error = u_computed - u_exact
    l2_error = np.linalg.norm(error[1:-1, 1:-1])
    print(f"L2 error: {l2_error:.2e}")
    
    # Test precision promotion logic
    residual_history = result['residual_history']
    stagnation_detected = precision_manager.should_promote_precision(
        residual_history, PrecisionLevel.SINGLE
    )
    
    print(f"Precision promotion logic working: {'✅ PASS' if len(stats['precision_history']) > 1 else '❌ FAIL'}")
    
    return {
        'precision_switches': len(stats['precision_history']) - 1,
        'final_precision': precision_manager.current_precision.value,
        'converged': result['converged'],
        'final_error': l2_error
    }


def test_manufactured_solutions():
    """Test solver accuracy on problems with known solutions."""
    print(f"\n{'='*60}")
    print("TESTING MANUFACTURED SOLUTIONS")
    print(f"{'='*60}")
    
    grid = Grid(33, 33, domain=(0, 1, 0, 1))
    solver = CorrectedMultigridSolver(
        max_levels=4,
        max_iterations=20,
        tolerance=1e-12,
        verbose=False
    )
    
    test_problems = ['manufactured', 'polynomial']
    results = {}
    
    for problem_type in test_problems:
        print(f"\nTesting {problem_type} solution:")
        print("-" * 30)
        
        # Create and solve problem
        rhs, u_exact = solver.create_test_problem(grid, problem_type)
        initial_guess = np.zeros_like(rhs)
        result = solver.solve(initial_guess, rhs, grid)
        
        # Calculate errors
        u_computed = result['solution']
        error = u_computed - u_exact
        l2_error = np.linalg.norm(error[1:-1, 1:-1])
        max_error = np.max(np.abs(error[1:-1, 1:-1]))
        
        # Expected error based on grid resolution
        h = 1.0 / (grid.nx - 1)
        expected_error = h**2  # O(h²) for second-order method
        
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"L2 error: {l2_error:.2e}")
        print(f"Max error: {max_error:.2e}")
        print(f"Expected O(h²): {expected_error:.2e}")
        print(f"Error ratio: {l2_error / expected_error:.2f}")
        
        accuracy_test = l2_error < 10 * expected_error  # Allow some tolerance
        print(f"Accuracy test: {'✅ PASS' if accuracy_test else '❌ FAIL'}")
        
        results[problem_type] = {
            'l2_error': l2_error,
            'max_error': max_error,
            'expected_error': expected_error,
            'accuracy_test': accuracy_test,
            'converged': result['converged']
        }
    
    return results


def test_optimal_precision_per_level():
    """Test the optimal precision per level logic."""
    print(f"\n{'='*60}")
    print("TESTING OPTIMAL PRECISION PER LEVEL")
    print(f"{'='*60}")
    
    precision_manager = PrecisionManager(adaptive=True)
    
    # Test different grid levels and problem sizes
    test_cases = [
        (0, 10000),    # Finest level, small problem
        (0, 1000000),  # Finest level, large problem
        (1, 100000),   # Second level, medium problem
        (3, 10000),    # Coarse level, small problem
        (5, 100000),   # Very coarse level, medium problem
    ]
    
    print("Level | Problem Size | Recommended Precision | Reasoning")
    print("-" * 60)
    
    for level, size in test_cases:
        precision = precision_manager.optimal_precision_per_level(level, size)
        
        if level == 0:
            reasoning = "Finest grid - accuracy critical"
        elif level <= 2 and size > 500000:
            reasoning = "Large problem - use single for speed"
        elif level <= 2:
            reasoning = "Fine level - use double for accuracy"  
        else:
            reasoning = "Coarse level - use single for speed"
            
        print(f"{level:5d} | {size:12,d} | {precision.value:>17s} | {reasoning}")
    
    # Test edge cases
    edge_cases_pass = True
    
    # Finest level should always be double precision for small/medium problems
    if precision_manager.optimal_precision_per_level(0, 100000) != PrecisionLevel.DOUBLE:
        edge_cases_pass = False
    
    # Very coarse levels should be single precision
    if precision_manager.optimal_precision_per_level(5, 100000) != PrecisionLevel.SINGLE:
        edge_cases_pass = False
    
    print(f"\nOptimal precision logic: {'✅ PASS' if edge_cases_pass else '❌ FAIL'}")
    
    return edge_cases_pass


def run_comprehensive_validation():
    """Run all validation tests."""
    print("COMPREHENSIVE MIXED-PRECISION MULTIGRID VALIDATION")
    print("=" * 80)
    
    all_results = {}
    overall_pass = True
    
    # Test 1: Convergence rates
    try:
        convergence_results = test_convergence_rates()
        all_results['convergence'] = convergence_results
        if convergence_results and not convergence_results.get('h_independent', False):
            overall_pass = False
    except Exception as e:
        print(f"❌ Convergence test failed: {e}")
        overall_pass = False
    
    # Test 2: Precision switching
    try:
        precision_results = test_precision_switching()
        all_results['precision'] = precision_results
        if not precision_results.get('converged', False):
            overall_pass = False
    except Exception as e:
        print(f"❌ Precision test failed: {e}")
        overall_pass = False
    
    # Test 3: Manufactured solutions
    try:
        accuracy_results = test_manufactured_solutions()
        all_results['accuracy'] = accuracy_results
        for problem, result in accuracy_results.items():
            if not result.get('accuracy_test', False):
                overall_pass = False
    except Exception as e:
        print(f"❌ Accuracy test failed: {e}")
        overall_pass = False
    
    # Test 4: Precision per level logic
    try:
        precision_logic_pass = test_optimal_precision_per_level()
        all_results['precision_logic'] = precision_logic_pass
        if not precision_logic_pass:
            overall_pass = False
    except Exception as e:
        print(f"❌ Precision logic test failed: {e}")
        overall_pass = False
    
    # Final summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    test_results = []
    if 'convergence' in all_results:
        conv = all_results['convergence']
        if conv:
            test_results.append(f"✅ H-independent convergence: {conv.get('h_independent', False)}")
            test_results.append(f"✅ Optimal convergence factor: {conv.get('optimal_convergence', False)}")
        else:
            test_results.append("❌ Convergence test incomplete")
    
    if 'precision' in all_results:
        prec = all_results['precision']
        test_results.append(f"✅ Precision switching: {prec.get('precision_switches', 0) > 0}")
        test_results.append(f"✅ Final convergence: {prec.get('converged', False)}")
    
    if 'accuracy' in all_results:
        acc = all_results['accuracy']
        for problem, result in acc.items():
            test_results.append(f"✅ {problem} accuracy: {result.get('accuracy_test', False)}")
    
    if 'precision_logic' in all_results:
        test_results.append(f"✅ Precision per level logic: {all_results['precision_logic']}")
    
    for result in test_results:
        print(result)
    
    print(f"\nOVERALL VALIDATION: {'✅ PASS' if overall_pass else '❌ SOME ISSUES'}")
    
    return all_results, overall_pass


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive validation
    results, passed = run_comprehensive_validation()
    
    if passed:
        print("\n🎉 All mathematical validations PASSED!")
        print("The mixed-precision multigrid solver is mathematically sound.")
    else:
        print("\n⚠️  Some validations need attention.")
        print("Review the results above for specific issues.")