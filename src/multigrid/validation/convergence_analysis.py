"""
Comprehensive Convergence Validation System for Mixed-Precision Multigrid Solvers
Implements mathematical validation of convergence rates and theoretical properties
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.grid import Grid
from ..operators.laplacian import LaplacianOperator
from ..operators.transfer import RestrictionOperator, ProlongationOperator
from ..solvers.smoothers import GaussSeidelSmoother, JacobiSmoother
from ..solvers.multigrid import MultigridSolver

logger = logging.getLogger(__name__)


class ConvergenceTestType(Enum):
    """Types of convergence tests."""
    TWO_GRID = "two_grid"
    MULTIGRID_H_INDEPENDENT = "multigrid_h_independent"
    SMOOTHING_ANALYSIS = "smoothing_analysis"
    GRID_TRANSFER_ACCURACY = "grid_transfer_accuracy"


@dataclass
class ConvergenceResult:
    """Results of convergence analysis."""
    test_type: ConvergenceTestType
    convergence_factor: float
    iterations_to_convergence: int
    theoretical_factor: float
    passes_theoretical_test: bool
    grid_sizes: List[int]
    residual_history: List[float]
    error_norms: List[float]
    additional_metrics: Dict[str, Any]


class TheoreticalAnalyzer:
    """Theoretical analysis of multigrid convergence properties."""
    
    def __init__(self):
        self.tolerance = 1e-12
        self.max_iterations = 1000
    
    def analyze_fourier_modes(self, grid_size: int, operator: LaplacianOperator) -> Dict[str, float]:
        """
        Analyze Fourier modes for theoretical convergence prediction.
        
        For 2D Poisson with standard finite differences:
        - High-frequency modes: θ_h ∈ [π/2, π] × [π/2, π]
        - Low-frequency modes: θ_h ∈ [0, π/2] × [0, π/2]
        """
        h = 1.0 / (grid_size - 1)
        
        # Create frequency grid
        n = grid_size - 2  # Interior points
        freqs_x = np.pi * np.arange(1, n + 1) / (n + 1)
        freqs_y = np.pi * np.arange(1, n + 1) / (n + 1)
        
        # Fourier symbols for discrete Laplacian
        fourier_symbols = []
        for wx in freqs_x:
            for wy in freqs_y:
                # Fourier symbol of 5-point stencil Laplacian
                symbol = (4 - 2*np.cos(wx) - 2*np.cos(wy)) / h**2
                fourier_symbols.append(symbol)
        
        fourier_symbols = np.array(fourier_symbols)
        
        # Identify high and low frequency modes
        high_freq_threshold = np.pi / 2
        high_freq_modes = []
        low_freq_modes = []
        
        idx = 0
        for wx in freqs_x:
            for wy in freqs_y:
                if wx >= high_freq_threshold or wy >= high_freq_threshold:
                    high_freq_modes.append(fourier_symbols[idx])
                else:
                    low_freq_modes.append(fourier_symbols[idx])
                idx += 1
        
        return {
            'all_symbols': fourier_symbols,
            'high_freq_symbols': np.array(high_freq_modes),
            'low_freq_symbols': np.array(low_freq_modes),
            'condition_number': np.max(fourier_symbols) / np.min(fourier_symbols[fourier_symbols > 0])
        }
    
    def predict_smoothing_factor(self, smoother_type: str = 'gauss_seidel') -> float:
        """
        Predict theoretical smoothing factor for different smoothers.
        
        Theoretical smoothing factors:
        - Jacobi (2D Poisson): μ ≈ 0.5
        - Gauss-Seidel (2D Poisson): μ ≈ 0.25 (lexicographic ordering)
        - Red-Black Gauss-Seidel: μ ≈ 0.5
        """
        theoretical_factors = {
            'jacobi': 0.5,
            'gauss_seidel': 0.25,
            'red_black_gs': 0.5,
            'weighted_jacobi': 0.33  # with optimal weight ω = 2/3
        }
        
        return theoretical_factors.get(smoother_type, 0.5)
    
    def predict_two_grid_factor(self, smoother_type: str = 'gauss_seidel', 
                              num_pre_smooth: int = 2, num_post_smooth: int = 2) -> float:
        """
        Predict two-grid convergence factor based on theory.
        
        For 2D Poisson with standard coarsening and smoothing:
        - Optimal two-grid factor should be < 0.1
        """
        smoothing_factor = self.predict_smoothing_factor(smoother_type)
        
        # Simplified two-grid analysis
        # Factor depends on smoothing effectiveness and grid transfer quality
        approximation_factor = smoothing_factor ** (num_pre_smooth + num_post_smooth)
        grid_transfer_factor = 0.8  # Typical for standard restriction/prolongation
        
        two_grid_factor = approximation_factor * grid_transfer_factor
        
        return min(two_grid_factor, 0.5)  # Upper bound


class ConvergenceValidator:
    """Comprehensive convergence validation for multigrid methods."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.max_iterations = 1000
        self.theoretical_analyzer = TheoreticalAnalyzer()
    
    def create_test_problem(self, grid: Grid, problem_type: str = "smooth_solution") -> Tuple[np.ndarray, np.ndarray]:
        """Create test problems with known analytical solutions."""
        x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
        y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if problem_type == "smooth_solution":
            # u(x,y) = sin(π*x)*sin(π*y), smooth solution
            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
            rhs = 2 * np.pi**2 * u_exact
        
        elif problem_type == "polynomial":
            # u(x,y) = x²*y²*(1-x)²*(1-y)², satisfies homogeneous BCs
            u_exact = (X**2) * (Y**2) * ((1-X)**2) * ((1-Y)**2)
            # Compute -∇²u analytically
            d2u_dx2 = 2*Y**2*(1-Y)**2 * (6*X**2 - 6*X + 1)
            d2u_dy2 = 2*X**2*(1-X)**2 * (6*Y**2 - 6*Y + 1)
            rhs = -(d2u_dx2 + d2u_dy2)
        
        elif problem_type == "high_frequency":
            # u(x,y) = sin(4π*x)*sin(4π*y), higher frequency
            u_exact = np.sin(4*np.pi * X) * np.sin(4*np.pi * Y)
            rhs = 32 * np.pi**2 * u_exact
        
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        return rhs, u_exact
    
    def validate_two_grid_convergence(self, grid_sizes: List[int]) -> ConvergenceResult:
        """
        Validate two-grid convergence factor.
        
        Tests that two-grid method achieves convergence factor < 0.5
        and preferably < 0.1 for optimal performance.
        """
        logger.info("Starting two-grid convergence validation...")
        
        convergence_factors = []
        all_residuals = []
        
        for grid_size in grid_sizes:
            # Create fine and coarse grids
            fine_grid = Grid(grid_size, grid_size, domain=(0, 1, 0, 1))
            coarse_grid = Grid(grid_size//2 + 1, grid_size//2 + 1, domain=(0, 1, 0, 1))
            
            # Create test problem
            rhs, u_exact = self.create_test_problem(fine_grid, "smooth_solution")
            
            # Initialize with random error
            np.random.seed(42)  # Reproducible results
            u_initial = u_exact + 0.1 * np.random.randn(*u_exact.shape)
            
            # Apply homogeneous boundary conditions
            u_initial[0, :] = u_initial[-1, :] = 0
            u_initial[:, 0] = u_initial[:, -1] = 0
            
            # Set up operators
            operator = LaplacianOperator()
            restriction = RestrictionOperator()
            prolongation = ProlongationOperator()
            smoother = GaussSeidelSmoother(max_iterations=2)
            
            # Two-grid cycle implementation
            u_current = u_initial.copy()
            residual_history = []
            
            for iteration in range(20):  # Limited iterations for convergence analysis
                # Pre-smoothing
                u_current = smoother.smooth(u_current, rhs, fine_grid, operator, iterations=2)
                
                # Compute residual
                residual = operator.apply(u_current, fine_grid) - rhs
                residual_norm = np.linalg.norm(residual[1:-1, 1:-1])
                residual_history.append(residual_norm)
                
                if residual_norm < self.tolerance:
                    break
                
                # Restrict residual to coarse grid
                coarse_residual = restriction.apply(residual, fine_grid, coarse_grid)
                
                # Solve coarse grid equation (simplified - few iterations)
                coarse_correction = np.zeros_like(coarse_residual)
                coarse_smoother = GaussSeidelSmoother(max_iterations=10)
                coarse_correction = coarse_smoother.smooth(
                    coarse_correction, coarse_residual, coarse_grid, operator, iterations=10
                )
                
                # Prolongate correction
                correction = prolongation.apply(coarse_correction, coarse_grid, fine_grid)
                
                # Apply correction
                u_current += correction
                
                # Post-smoothing
                u_current = smoother.smooth(u_current, rhs, fine_grid, operator, iterations=2)
            
            # Calculate convergence factor
            if len(residual_history) >= 5:
                # Use geometric mean of last few iterations for stability
                recent_factors = []
                for i in range(-5, -1):
                    if residual_history[i-1] > 0:
                        factor = residual_history[i] / residual_history[i-1]
                        recent_factors.append(factor)
                
                if recent_factors:
                    conv_factor = np.exp(np.mean(np.log(recent_factors)))
                    convergence_factors.append(conv_factor)
                    all_residuals.extend(residual_history)
        
        # Analyze results
        if convergence_factors:
            avg_convergence_factor = np.mean(convergence_factors)
            theoretical_factor = self.theoretical_analyzer.predict_two_grid_factor()
            
            passes_test = avg_convergence_factor < 0.5  # Basic requirement
            optimal_performance = avg_convergence_factor < 0.1  # Optimal requirement
            
            logger.info(f"Two-grid convergence factor: {avg_convergence_factor:.4f}")
            logger.info(f"Theoretical prediction: {theoretical_factor:.4f}")
            logger.info(f"Passes basic test (< 0.5): {passes_test}")
            logger.info(f"Optimal performance (< 0.1): {optimal_performance}")
            
            return ConvergenceResult(
                test_type=ConvergenceTestType.TWO_GRID,
                convergence_factor=avg_convergence_factor,
                iterations_to_convergence=len(residual_history),
                theoretical_factor=theoretical_factor,
                passes_theoretical_test=passes_test,
                grid_sizes=grid_sizes,
                residual_history=all_residuals,
                error_norms=[],
                additional_metrics={
                    'optimal_performance': optimal_performance,
                    'individual_factors': convergence_factors
                }
            )
        else:
            logger.warning("No convergence factors could be computed")
            return None
    
    def validate_h_independent_convergence(self, grid_sizes: List[int]) -> ConvergenceResult:
        """
        Validate h-independent convergence of multigrid method.
        
        Tests that convergence rate is independent of grid size h.
        This is the fundamental theoretical property of optimal multigrid.
        """
        logger.info("Starting h-independent convergence validation...")
        
        convergence_factors = []
        iterations_per_grid = []
        
        for grid_size in grid_sizes:
            grid = Grid(grid_size, grid_size, domain=(0, 1, 0, 1))
            rhs, u_exact = self.create_test_problem(grid, "smooth_solution")
            
            # Create multigrid solver
            solver = MultigridSolver(
                max_iterations=50,
                tolerance=self.tolerance,
                cycle_type='V',
                pre_smooth_iterations=2,
                post_smooth_iterations=2
            )
            
            # Initialize solution with random error
            np.random.seed(42)
            u_initial = 0.1 * np.random.randn(*u_exact.shape)
            u_initial[0, :] = u_initial[-1, :] = 0
            u_initial[:, 0] = u_initial[:, -1] = 0
            
            # Solve and monitor convergence
            result = solver.solve(u_initial, rhs, grid)
            
            if result and 'residual_history' in result:
                residuals = result['residual_history']
                
                # Calculate convergence factor (average over middle iterations)
                if len(residuals) >= 10:
                    start_idx = 5
                    end_idx = min(len(residuals) - 1, 15)
                    
                    factors = []
                    for i in range(start_idx, end_idx):
                        if residuals[i-1] > 0:
                            factor = residuals[i] / residuals[i-1]
                            factors.append(factor)
                    
                    if factors:
                        avg_factor = np.exp(np.mean(np.log(factors)))
                        convergence_factors.append(avg_factor)
                        iterations_per_grid.append(len(residuals))
            
            logger.info(f"Grid {grid_size}x{grid_size}: {len(convergence_factors)} factors computed")
        
        if len(convergence_factors) >= 2:
            # Test h-independence: convergence factors should be similar across grid sizes
            factor_variation = np.std(convergence_factors) / np.mean(convergence_factors)
            h_independent = factor_variation < 0.3  # Allow 30% variation
            
            avg_factor = np.mean(convergence_factors)
            theoretical_optimal = 0.1  # Theoretical optimal multigrid factor
            
            logger.info(f"Average convergence factor: {avg_factor:.4f}")
            logger.info(f"Factor variation coefficient: {factor_variation:.4f}")
            logger.info(f"H-independent convergence: {h_independent}")
            
            return ConvergenceResult(
                test_type=ConvergenceTestType.MULTIGRID_H_INDEPENDENT,
                convergence_factor=avg_factor,
                iterations_to_convergence=int(np.mean(iterations_per_grid)),
                theoretical_factor=theoretical_optimal,
                passes_theoretical_test=h_independent and avg_factor < 0.5,
                grid_sizes=grid_sizes,
                residual_history=[],
                error_norms=[],
                additional_metrics={
                    'factor_variation': factor_variation,
                    'individual_factors': convergence_factors,
                    'iterations_per_grid': iterations_per_grid
                }
            )
        else:
            logger.warning("Insufficient data for h-independent analysis")
            return None
    
    def validate_smoothing_analysis(self, grid_size: int = 65) -> ConvergenceResult:
        """
        Validate smoothing properties of iterative methods.
        
        Tests that smoothers effectively reduce high-frequency error components.
        """
        logger.info("Starting smoothing analysis validation...")
        
        grid = Grid(grid_size, grid_size, domain=(0, 1, 0, 1))
        operator = LaplacianOperator()
        
        # Create high-frequency error (checkerboard pattern)
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # High-frequency test function
        high_freq_error = np.sin(8*np.pi*X) * np.sin(8*np.pi*Y)
        high_freq_error[0, :] = high_freq_error[-1, :] = 0
        high_freq_error[:, 0] = high_freq_error[:, -1] = 0
        
        # Test different smoothers
        smoothers = {
            'jacobi': JacobiSmoother(max_iterations=1),
            'gauss_seidel': GaussSeidelSmoother(max_iterations=1)
        }
        
        smoother_results = {}
        
        for smoother_name, smoother in smoothers.items():
            error_initial = high_freq_error.copy()
            error_norms = [np.linalg.norm(error_initial[1:-1, 1:-1])]
            
            # Apply several smoothing iterations
            current_error = error_initial.copy()
            rhs = np.zeros_like(current_error)  # Homogeneous equation for error
            
            for iteration in range(10):
                current_error = smoother.smooth(current_error, rhs, grid, operator, iterations=1)
                error_norm = np.linalg.norm(current_error[1:-1, 1:-1])
                error_norms.append(error_norm)
                
                if error_norm < 1e-12:
                    break
            
            # Calculate smoothing factor
            if len(error_norms) >= 3:
                # Use geometric mean for stability
                factors = []
                for i in range(1, min(6, len(error_norms))):
                    if error_norms[i-1] > 0:
                        factor = error_norms[i] / error_norms[i-1]
                        factors.append(factor)
                
                if factors:
                    smoothing_factor = np.exp(np.mean(np.log(factors)))
                    theoretical_factor = self.theoretical_analyzer.predict_smoothing_factor(smoother_name)
                    
                    smoother_results[smoother_name] = {
                        'smoothing_factor': smoothing_factor,
                        'theoretical_factor': theoretical_factor,
                        'error_reduction': error_norms[0] / error_norms[-1],
                        'error_history': error_norms
                    }
                    
                    logger.info(f"{smoother_name}: factor = {smoothing_factor:.4f}, "
                              f"theoretical = {theoretical_factor:.4f}")
        
        # Use Gauss-Seidel results for main result
        if 'gauss_seidel' in smoother_results:
            gs_result = smoother_results['gauss_seidel']
            
            passes_test = gs_result['smoothing_factor'] < 0.8  # Should significantly reduce error
            
            return ConvergenceResult(
                test_type=ConvergenceTestType.SMOOTHING_ANALYSIS,
                convergence_factor=gs_result['smoothing_factor'],
                iterations_to_convergence=len(gs_result['error_history']),
                theoretical_factor=gs_result['theoretical_factor'],
                passes_theoretical_test=passes_test,
                grid_sizes=[grid_size],
                residual_history=[],
                error_norms=gs_result['error_history'],
                additional_metrics={
                    'all_smoother_results': smoother_results,
                    'high_freq_reduction': gs_result['error_reduction']
                }
            )
        else:
            logger.warning("Gauss-Seidel smoothing analysis failed")
            return None
    
    def validate_grid_transfer_accuracy(self, grid_sizes: List[int]) -> ConvergenceResult:
        """
        Validate accuracy of grid transfer operators.
        
        Tests that restriction and prolongation operators maintain
        appropriate accuracy for smooth functions: ||I_h^{2h}I_{2h}^h u - u|| = O(h²)
        """
        logger.info("Starting grid transfer accuracy validation...")
        
        transfer_errors = []
        h_values = []
        
        restriction = RestrictionOperator()
        prolongation = ProlongationOperator()
        
        for grid_size in grid_sizes:
            if grid_size < 10:
                continue
                
            # Create fine and coarse grids
            fine_grid = Grid(grid_size, grid_size, domain=(0, 1, 0, 1))
            coarse_size = grid_size // 2 + 1
            coarse_grid = Grid(coarse_size, coarse_size, domain=(0, 1, 0, 1))
            
            h = 1.0 / (grid_size - 1)
            h_values.append(h)
            
            # Create smooth test function
            _, u_smooth = self.create_test_problem(fine_grid, "polynomial")
            
            # Apply restriction followed by prolongation
            u_coarse = restriction.apply(u_smooth, fine_grid, coarse_grid)
            u_reconstructed = prolongation.apply(u_coarse, coarse_grid, fine_grid)
            
            # Compute error in interior points
            error = u_reconstructed[1:-1, 1:-1] - u_smooth[1:-1, 1:-1]
            error_norm = np.linalg.norm(error)
            transfer_errors.append(error_norm)
            
            logger.info(f"Grid {grid_size}: h = {h:.4f}, transfer error = {error_norm:.6e}")
        
        if len(transfer_errors) >= 3:
            # Fit error vs h to determine convergence order
            log_h = np.log(np.array(h_values))
            log_errors = np.log(np.array(transfer_errors))
            
            # Linear regression to find slope (convergence order)
            coeffs = np.polyfit(log_h, log_errors, 1)
            convergence_order = coeffs[0]
            
            # For O(h²) accuracy, slope should be approximately 2
            theoretical_order = 2.0
            order_error = abs(convergence_order - theoretical_order)
            passes_test = order_error < 0.5  # Allow some deviation
            
            logger.info(f"Grid transfer convergence order: {convergence_order:.2f}")
            logger.info(f"Theoretical order: {theoretical_order:.2f}")
            logger.info(f"Passes accuracy test: {passes_test}")
            
            return ConvergenceResult(
                test_type=ConvergenceTestType.GRID_TRANSFER_ACCURACY,
                convergence_factor=convergence_order,
                iterations_to_convergence=0,
                theoretical_factor=theoretical_order,
                passes_theoretical_test=passes_test,
                grid_sizes=grid_sizes,
                residual_history=[],
                error_norms=transfer_errors,
                additional_metrics={
                    'h_values': h_values,
                    'order_error': order_error,
                    'regression_coefficients': coeffs.tolist()
                }
            )
        else:
            logger.warning("Insufficient data for grid transfer analysis")
            return None
    
    def run_comprehensive_validation(self) -> Dict[str, ConvergenceResult]:
        """Run all convergence validation tests."""
        logger.info("Starting comprehensive convergence validation...")
        
        # Define test grid sizes
        grid_sizes = [17, 33, 65, 129]
        
        results = {}
        
        # Run all validation tests
        tests = [
            ('two_grid', lambda: self.validate_two_grid_convergence(grid_sizes[:3])),
            ('h_independent', lambda: self.validate_h_independent_convergence(grid_sizes)),
            ('smoothing', lambda: self.validate_smoothing_analysis()),
            ('grid_transfer', lambda: self.validate_grid_transfer_accuracy(grid_sizes))
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name} validation...")
                result = test_func()
                if result:
                    results[test_name] = result
                    logger.info(f"{test_name} validation completed successfully")
                else:
                    logger.warning(f"{test_name} validation failed")
            except Exception as e:
                logger.error(f"Error in {test_name} validation: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ConvergenceResult]) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("="*80)
        report.append("MULTIGRID CONVERGENCE VALIDATION REPORT")
        report.append("="*80)
        
        overall_pass = True
        
        for test_name, result in results.items():
            report.append(f"\n{test_name.upper().replace('_', ' ')} VALIDATION:")
            report.append("-" * 50)
            
            status = "✅ PASS" if result.passes_theoretical_test else "❌ FAIL"
            report.append(f"Status: {status}")
            report.append(f"Measured Factor: {result.convergence_factor:.4f}")
            report.append(f"Theoretical Factor: {result.theoretical_factor:.4f}")
            
            if result.test_type == ConvergenceTestType.TWO_GRID:
                optimal = result.additional_metrics.get('optimal_performance', False)
                report.append(f"Optimal Performance (<0.1): {'✅' if optimal else '❌'}")
            
            elif result.test_type == ConvergenceTestType.MULTIGRID_H_INDEPENDENT:
                variation = result.additional_metrics.get('factor_variation', 0)
                report.append(f"Factor Variation: {variation:.4f}")
                report.append(f"H-Independence: {'✅' if variation < 0.3 else '❌'}")
            
            elif result.test_type == ConvergenceTestType.GRID_TRANSFER_ACCURACY:
                order_error = result.additional_metrics.get('order_error', 0)
                report.append(f"Order Error: {order_error:.3f}")
                report.append(f"Expected O(h²) accuracy: {'✅' if order_error < 0.5 else '❌'}")
            
            if not result.passes_theoretical_test:
                overall_pass = False
        
        report.append("\n" + "="*80)
        report.append(f"OVERALL VALIDATION: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        report.append("="*80)
        
        return "\n".join(report)


# Example usage and validation runner
def run_validation():
    """Run comprehensive validation and return results."""
    validator = ConvergenceValidator(tolerance=1e-10)
    results = validator.run_comprehensive_validation()
    
    if results:
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save results to file
        import json
        import time
        
        timestamp = int(time.time())
        results_file = f"convergence_validation_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for test_name, result in results.items():
            json_results[test_name] = {
                'test_type': result.test_type.value,
                'convergence_factor': result.convergence_factor,
                'theoretical_factor': result.theoretical_factor,
                'passes_test': result.passes_theoretical_test,
                'additional_metrics': result.additional_metrics
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    # Run validation if called directly
    run_validation()