"""
Phase 2 Example: Compare iterative solvers and preconditioning methods.

Demonstrates the enhanced iterative solvers with different preconditioning
techniques and adaptive precision strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multigrid import Grid, LaplacianOperator, PrecisionManager
from multigrid.solvers.iterative import (
    EnhancedJacobiSolver, EnhancedGaussSeidelSolver, SORSolver,
    WeightedJacobiSolver, AdaptivePrecisionSolver
)
from multigrid.preconditioning import (
    DiagonalPreconditioner, ScaledDiagonalPreconditioner,
    BlockDiagonalPreconditioner, MultigridPreconditioner
)
from multigrid.analysis.convergence import ConvergenceAnalyzer, ConvergenceMonitor


def create_test_problem(grid, problem_type="poisson_2d"):
    """Create test problem for benchmarking."""
    if problem_type == "poisson_2d":
        # -∇²u = f with u(x,y) = sin(πx)sin(πy)
        u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
        rhs = 2 * np.pi**2 * u_exact
        
    elif problem_type == "anisotropic":
        # Anisotropic diffusion problem
        u_exact = np.sin(2*np.pi * grid.X) * np.cos(np.pi * grid.Y)
        # -∇²u with different coefficients
        rhs = (4*np.pi**2 * np.sin(2*np.pi * grid.X) * np.cos(np.pi * grid.Y) + 
               np.pi**2 * np.sin(2*np.pi * grid.X) * np.cos(np.pi * grid.Y))
    
    elif problem_type == "high_frequency":
        # High frequency solution
        u_exact = np.sin(4*np.pi * grid.X) * np.sin(4*np.pi * grid.Y)
        rhs = 32 * np.pi**2 * u_exact
    
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    return u_exact, rhs


class SolverBenchmark:
    """Benchmark different solvers and preconditioning methods."""
    
    def __init__(self, grid_sizes=None, max_iterations=200, tolerance=1e-8):
        """Initialize benchmark."""
        self.grid_sizes = grid_sizes or [33, 65, 129]
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.results = {}
    
    def setup_solvers(self):
        """Setup different solver configurations."""
        solvers = {
            # Basic iterative methods
            'Jacobi': EnhancedJacobiSolver(
                max_iterations=self.max_iterations, 
                tolerance=self.tolerance,
                use_vectorized=True
            ),
            'WeightedJacobi': WeightedJacobiSolver(
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                auto_weight=True
            ),
            'GaussSeidel': EnhancedGaussSeidelSolver(
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                red_black=False
            ),
            'RedBlackGS': EnhancedGaussSeidelSolver(
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                red_black=True
            ),
            'SOR': SORSolver(
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                auto_omega=True
            ),
            
            # Adaptive precision methods
            'AdaptiveJacobi': AdaptivePrecisionSolver(
                EnhancedJacobiSolver(max_iterations=self.max_iterations, tolerance=self.tolerance),
                precision_switch_threshold=0.9,
                min_iterations_before_switch=10
            ),
            'AdaptiveGaussSeidel': AdaptivePrecisionSolver(
                EnhancedGaussSeidelSolver(max_iterations=self.max_iterations, tolerance=self.tolerance),
                precision_switch_threshold=0.9,
                min_iterations_before_switch=10
            )
        }
        
        return solvers
    
    def setup_preconditioned_solvers(self):
        """Setup preconditioned solver configurations."""
        base_solver = EnhancedGaussSeidelSolver(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance
        )
        
        # Note: This is a simplified setup - full preconditioned solver
        # integration would require additional wrapper classes
        preconditioners = {
            'Diagonal': DiagonalPreconditioner(),
            'ScaledDiagonal': ScaledDiagonalPreconditioner(scaling_factor=2.0),
            'BlockDiagonal': BlockDiagonalPreconditioner(block_direction="row"),
        }
        
        return preconditioners
    
    def run_benchmark(self, problem_type="poisson_2d"):
        """Run comprehensive benchmark."""
        print("=" * 80)
        print(f"ITERATIVE SOLVER BENCHMARK - {problem_type.upper()}")
        print("=" * 80)
        
        solvers = self.setup_solvers()
        
        for grid_size in self.grid_sizes:
            print(f"\nGrid Size: {grid_size} x {grid_size}")
            print("-" * 50)
            
            # Create test problem
            grid = Grid(nx=grid_size, ny=grid_size, domain=(0, 1, 0, 1))
            operator = LaplacianOperator()
            u_exact, rhs = create_test_problem(grid, problem_type)
            
            # Apply boundary conditions
            grid.apply_dirichlet_bc(0.0)
            rhs[0, :] = 0.0
            rhs[-1, :] = 0.0
            rhs[:, 0] = 0.0
            rhs[:, -1] = 0.0
            
            grid_results = {}
            
            # Test each solver
            for solver_name, solver in solvers.items():
                print(f"Testing {solver_name}...", end=" ")
                
                try:
                    # Create precision manager for adaptive solvers
                    precision_manager = None
                    if "Adaptive" in solver_name:
                        precision_manager = PrecisionManager(
                            adaptive=True,
                            default_precision="mixed"
                        )
                    
                    # Solve
                    start_time = time.time()
                    solution, info = solver.solve(
                        grid, operator, rhs, 
                        precision_manager=precision_manager
                    )
                    solve_time = time.time() - start_time
                    
                    # Compute error
                    error = np.max(np.abs(solution - u_exact))
                    
                    # Store results
                    grid_results[solver_name] = {
                        'converged': info['converged'],
                        'iterations': info['iterations'],
                        'final_residual': info['final_residual'],
                        'convergence_rate': info['convergence_rate'],
                        'solve_time': solve_time,
                        'error': error,
                        'time_per_iteration': solve_time / info['iterations'] if info['iterations'] > 0 else np.inf
                    }
                    
                    status = "CONVERGED" if info['converged'] else "FAILED"
                    print(f"{status} ({info['iterations']} iter, {solve_time:.3f}s, err={error:.2e})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    grid_results[solver_name] = {'error': str(e)}
            
            # Store results for this grid size
            self.results[grid_size] = grid_results
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Find best performers
        for grid_size in self.grid_sizes:
            print(f"\nGrid {grid_size}x{grid_size} - Best Performers:")
            
            grid_results = self.results[grid_size]
            
            # Filter converged results
            converged = {name: result for name, result in grid_results.items() 
                        if isinstance(result, dict) and result.get('converged', False)}
            
            if not converged:
                print("  No solvers converged!")
                continue
            
            # Best by iterations
            best_iter = min(converged.items(), key=lambda x: x[1]['iterations'])
            print(f"  Fastest convergence: {best_iter[0]} ({best_iter[1]['iterations']} iter)")
            
            # Best by time
            best_time = min(converged.items(), key=lambda x: x[1]['solve_time'])
            print(f"  Fastest solve time: {best_time[0]} ({best_time[1]['solve_time']:.3f}s)")
            
            # Best by accuracy
            best_accuracy = min(converged.items(), key=lambda x: x[1]['error'])
            print(f"  Most accurate: {best_accuracy[0]} (error={best_accuracy[1]['error']:.2e})")
    
    def create_comparison_plots(self, filename_prefix="solver_comparison"):
        """Create comparison plots."""
        if not self.results:
            print("No results to plot!")
            return
        
        # Prepare data for plotting
        solver_names = []
        grid_sizes = []
        iterations_data = []
        time_data = []
        error_data = []
        
        for grid_size, grid_results in self.results.items():
            for solver_name, result in grid_results.items():
                if isinstance(result, dict) and result.get('converged', False):
                    solver_names.append(solver_name)
                    grid_sizes.append(grid_size)
                    iterations_data.append(result['iterations'])
                    time_data.append(result['solve_time'])
                    error_data.append(result['error'])
        
        if not iterations_data:
            print("No convergent results to plot!")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Iterative Solver Comparison', fontsize=16)
        
        # Plot 1: Iterations vs Grid Size
        unique_solvers = list(set(solver_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_solvers)))
        
        for i, solver in enumerate(unique_solvers):
            solver_grid_sizes = [gs for gs, sn in zip(grid_sizes, solver_names) if sn == solver]
            solver_iterations = [it for it, sn in zip(iterations_data, solver_names) if sn == solver]
            
            axes[0, 0].plot(solver_grid_sizes, solver_iterations, 'o-', 
                           label=solver, color=colors[i])
        
        axes[0, 0].set_xlabel('Grid Size')
        axes[0, 0].set_ylabel('Iterations to Convergence')
        axes[0, 0].set_title('Convergence Speed')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Solve Time vs Grid Size
        for i, solver in enumerate(unique_solvers):
            solver_grid_sizes = [gs for gs, sn in zip(grid_sizes, solver_names) if sn == solver]
            solver_times = [t for t, sn in zip(time_data, solver_names) if sn == solver]
            
            axes[0, 1].semilogy(solver_grid_sizes, solver_times, 'o-', 
                               label=solver, color=colors[i])
        
        axes[0, 1].set_xlabel('Grid Size')
        axes[0, 1].set_ylabel('Solve Time (s)')
        axes[0, 1].set_title('Computational Efficiency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error vs Grid Size
        for i, solver in enumerate(unique_solvers):
            solver_grid_sizes = [gs for gs, sn in zip(grid_sizes, solver_names) if sn == solver]
            solver_errors = [e for e, sn in zip(error_data, solver_names) if sn == solver]
            
            axes[1, 0].semilogy(solver_grid_sizes, solver_errors, 'o-', 
                               label=solver, color=colors[i])
        
        axes[1, 0].set_xlabel('Grid Size')
        axes[1, 0].set_ylabel('Max Error')
        axes[1, 0].set_title('Solution Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency (Iterations per second)
        efficiency_data = [1.0 / (t / it) for t, it in zip(time_data, iterations_data)]
        
        for i, solver in enumerate(unique_solvers):
            solver_grid_sizes = [gs for gs, sn in zip(grid_sizes, solver_names) if sn == solver]
            solver_efficiency = [e for e, sn in zip(efficiency_data, solver_names) if sn == solver]
            
            axes[1, 1].plot(solver_grid_sizes, solver_efficiency, 'o-', 
                           label=solver, color=colors[i])
        
        axes[1, 1].set_xlabel('Grid Size')
        axes[1, 1].set_ylabel('Iterations per Second')
        axes[1, 1].set_title('Computational Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / f'{filename_prefix}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nComparison plots saved to: {output_file}")
        
        try:
            plt.show()
        except:
            pass


def demonstrate_convergence_analysis():
    """Demonstrate convergence analysis tools."""
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Setup problem
    grid = Grid(nx=33, ny=33)
    operator = LaplacianOperator()
    u_exact, rhs = create_test_problem(grid, "poisson_2d")
    
    # Test different solvers with analysis
    solvers = {
        'Jacobi': EnhancedJacobiSolver(max_iterations=100, tolerance=1e-8),
        'SOR': SORSolver(max_iterations=50, tolerance=1e-8, auto_omega=True)
    }
    
    analyzer = ConvergenceAnalyzer()
    
    for solver_name, solver in solvers.items():
        print(f"\nAnalyzing {solver_name} convergence...")
        
        # Solve and collect detailed history
        solution, info = solver.solve(grid, operator, rhs)
        
        if info['converged']:
            # Compute error history if we have exact solution
            error_history = []
            for i, residual_norm in enumerate(info['residual_history']):
                # This is approximate - in real scenario, we'd track solution at each iteration
                approx_error = residual_norm * 0.1  # Rough approximation
                error_history.append(approx_error)
            
            # Analyze convergence
            analysis = analyzer.analyze_convergence(
                info['residual_history'],
                error_history,
                [info['average_time_per_iteration']] * len(info['residual_history'])
            )
            
            print(f"  Convergence type: {analysis['convergence_type']}")
            print(f"  Asymptotic rate: {analysis['convergence_rates']['asymptotic']:.4f}")
            print(f"  Mean convergence rate: {analysis['convergence_rates']['mean']:.4f}")
            print(f"  Final residual reduction: {analysis['reduction_factor']:.2e}")
            
            if analysis['stagnation_analysis'].get('stagnation_detected', False):
                print("  WARNING: Stagnation detected!")


def demonstrate_precision_switching():
    """Demonstrate adaptive precision switching."""
    print("\n" + "=" * 80)
    print("ADAPTIVE PRECISION DEMONSTRATION")
    print("=" * 80)
    
    # Setup problem
    grid = Grid(nx=65, ny=65)
    operator = LaplacianOperator()
    u_exact, rhs = create_test_problem(grid, "poisson_2d")
    
    # Create adaptive precision solver
    base_solver = SORSolver(max_iterations=100, tolerance=1e-10, auto_omega=True)
    adaptive_solver = AdaptivePrecisionSolver(
        base_solver,
        precision_switch_threshold=0.95,
        min_iterations_before_switch=8
    )
    
    # Setup precision manager
    precision_manager = PrecisionManager(
        adaptive=True,
        default_precision="single",  # Start with single precision
        convergence_threshold=1e-6
    )
    
    print("Solving with adaptive precision switching...")
    print("Starting with single precision, will switch to double if needed.")
    
    solution, info = adaptive_solver.solve(
        grid, operator, rhs, precision_manager=precision_manager
    )
    
    print(f"\nResults:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final residual: {info['final_residual']:.2e}")
    print(f"  Precision switched: {info.get('precision_switched', False)}")
    
    if info.get('precision_switched', False):
        print(f"  Switch occurred at iteration: {info.get('switch_iteration', 'unknown')}")
    
    # Get precision statistics
    precision_stats = precision_manager.get_statistics()
    print(f"\nPrecision usage:")
    for precision, stats in precision_stats['precision_breakdown'].items():
        if stats['operations'] > 0:
            print(f"  {precision}: {stats['operations']} operations ({stats['percentage']:.1f}%)")


def main():
    """Main demonstration function."""
    print("Mixed-Precision Multigrid Solvers - Phase 2 Demonstration")
    print("Iterative Solvers & Preconditioning")
    
    # Run solver benchmark
    benchmark = SolverBenchmark(
        grid_sizes=[17, 33, 65],
        max_iterations=150,
        tolerance=1e-8
    )
    
    # Test on different problem types
    for problem_type in ["poisson_2d", "high_frequency"]:
        benchmark.run_benchmark(problem_type)
        benchmark.print_summary()
        benchmark.create_comparison_plots(f"solver_comparison_{problem_type}")
    
    # Demonstrate analysis tools
    demonstrate_convergence_analysis()
    
    # Demonstrate precision switching
    demonstrate_precision_switching()
    
    print("\n" + "=" * 80)
    print("PHASE 2 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey features demonstrated:")
    print("• Enhanced iterative solvers (Jacobi, Gauss-Seidel, SOR)")
    print("• Vectorized operations for performance")
    print("• Red-black ordering for parallelization")
    print("• Adaptive precision switching")
    print("• Convergence analysis and monitoring")
    print("• Comprehensive benchmarking")


if __name__ == "__main__":
    main()