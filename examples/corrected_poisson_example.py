"""
Corrected Poisson 2D Example with Mixed-Precision Multigrid
Demonstrates the fixed numerical implementation with proper convergence
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multigrid.core.grid import Grid
from multigrid.core.precision import PrecisionManager, PrecisionLevel
from multigrid.solvers.corrected_multigrid import CorrectedMultigridSolver


def main():
    print("="*70)
    print("CORRECTED MIXED-PRECISION MULTIGRID SOLVER DEMONSTRATION")
    print("="*70)
    
    # Problem setup
    grid_size = 65
    domain = (0.0, 1.0, 0.0, 1.0)
    
    print(f"Creating {grid_size}x{grid_size} grid on domain {domain}")
    grid = Grid(grid_size, grid_size, domain=domain)
    
    # Create precision manager with adaptive switching
    precision_manager = PrecisionManager(
        default_precision=PrecisionLevel.SINGLE,
        adaptive=True,
        convergence_threshold=1e-6,
        memory_threshold_gb=2.0
    )
    
    # Create solver
    solver = CorrectedMultigridSolver(
        max_levels=4,
        max_iterations=25,
        tolerance=1e-10,
        cycle_type="V",
        pre_smooth_iterations=2,
        post_smooth_iterations=2,
        verbose=True
    )
    
    print(f"Multigrid hierarchy: {len(solver.grids)} levels")
    for i, g in enumerate(solver.grids):
        h = 1.0 / (g.nx - 1)
        print(f"  Level {i}: {g.nx}x{g.ny} (h={h:.4f})")
    
    print("\nSetting up test problems...")
    
    # Test multiple problems
    problems = {
        "Manufactured Solution": "manufactured",
        "Polynomial Solution": "polynomial"
    }
    
    for problem_name, problem_type in problems.items():
        print(f"\n{'='*50}")
        print(f"SOLVING: {problem_name}")
        print(f"{'='*50}")
        
        # Create problem
        rhs, u_exact = solver.create_test_problem(grid, problem_type)
        
        print(f"Problem type: {problem_type}")
        print(f"RHS norm: {np.linalg.norm(rhs):.2e}")
        print(f"Starting precision: {precision_manager.current_precision.value}")
        
        # Reset precision manager for each problem
        precision_manager.current_precision = precision_manager.default_precision
        precision_manager.reset_statistics()
        
        # Solve the problem
        start_time = time.time()
        initial_guess = np.zeros_like(rhs)
        
        result = solver.solve(initial_guess, rhs, grid, precision_manager)
        
        solve_time = time.time() - start_time
        
        # Extract results
        u_computed = result['solution']
        converged = result['converged']
        iterations = result['iterations']
        final_residual = result['final_residual']
        residual_history = result['residual_history']
        
        print(f"\n{'='*40}")
        print("SOLUTION RESULTS")
        print(f"{'='*40}")
        print(f"Converged: {converged}")
        print(f"Iterations: {iterations}")
        print(f"Final residual: {final_residual:.2e}")
        print(f"Total solve time: {solve_time:.3f}s")
        print(f"Average time per iteration: {solve_time/iterations:.3f}s")
        
        # Compute errors
        error = u_computed - u_exact
        l2_error = np.linalg.norm(error[1:-1, 1:-1])
        max_error = np.max(np.abs(error[1:-1, 1:-1]))
        h = 1.0 / (grid_size - 1)
        relative_l2_error = l2_error / np.linalg.norm(u_exact[1:-1, 1:-1])
        
        print(f"\nError Analysis:")
        print(f"L2 error: {l2_error:.2e}")
        print(f"Max error: {max_error:.2e}")
        print(f"Relative L2 error: {relative_l2_error:.2e}")
        print(f"Grid spacing h: {h:.4f}")
        print(f"Error/h²: {l2_error/h**2:.2e}")
        
        # Precision usage statistics
        stats = precision_manager.get_statistics()
        print(f"\nPrecision Usage:")
        print(f"Final precision: {stats['current_precision']}")
        print(f"Precision history: {' → '.join(stats['precision_history'])}")
        print(f"Precision changes: {len(stats['precision_history']) - 1}")
        
        # Convergence analysis
        if len(residual_history) >= 5:
            convergence_factors = []
            for i in range(2, min(len(residual_history), 8)):
                if residual_history[i-1] > 0:
                    factor = residual_history[i] / residual_history[i-1]
                    convergence_factors.append(factor)
            
            if convergence_factors:
                avg_factor = np.exp(np.mean(np.log(convergence_factors)))
                print(f"Average convergence factor: {avg_factor:.4f}")
                
                # Theoretical assessment
                if avg_factor < 0.1:
                    print("✅ Optimal multigrid convergence achieved!")
                elif avg_factor < 0.3:
                    print("✅ Good multigrid convergence")
                else:
                    print("⚠️  Suboptimal convergence - check problem setup")
        
        # Create visualization
        create_solution_plot(u_computed, u_exact, error, problem_name, 
                           grid, save_path=f"corrected_{problem_type}_results.png")
        
        # Create convergence plot
        create_convergence_plot(residual_history, problem_name,
                              save_path=f"corrected_{problem_type}_convergence.png")
        
        print(f"Plots saved: corrected_{problem_type}_results.png")
        print(f"              corrected_{problem_type}_convergence.png")


def create_solution_plot(u_computed, u_exact, error, title, grid, save_path):
    """Create solution visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
    y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
    X, Y = np.meshgrid(x, y)
    
    # Computed solution
    im1 = axes[0, 0].contourf(X, Y, u_computed, levels=20, cmap='viridis')
    axes[0, 0].set_title('Computed Solution')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Exact solution
    im2 = axes[0, 1].contourf(X, Y, u_exact, levels=20, cmap='viridis')
    axes[0, 1].set_title('Exact Solution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Error
    im3 = axes[1, 0].contourf(X, Y, error, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Error (Computed - Exact)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Cross-section comparison
    mid_j = grid.ny // 2
    axes[1, 1].plot(x, u_computed[:, mid_j], 'b-', label='Computed', linewidth=2)
    axes[1, 1].plot(x, u_exact[:, mid_j], 'r--', label='Exact', linewidth=2)
    axes[1, 1].set_title(f'Cross-section at y = {y[mid_j]:.2f}')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Mixed-Precision Multigrid Solution: {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_convergence_plot(residual_history, title, save_path):
    """Create convergence history plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(residual_history))
    ax.semilogy(iterations, residual_history, 'b-o', linewidth=2, markersize=4)
    
    # Add convergence rate reference lines
    if len(residual_history) > 5:
        # Linear convergence reference (factor = 0.1)
        linear_ref = [residual_history[0] * (0.1)**i for i in iterations]
        ax.semilogy(iterations, linear_ref, 'g--', alpha=0.7, label='Factor = 0.1')
        
        # Calculate actual average factor
        factors = []
        for i in range(1, min(len(residual_history), 8)):
            if residual_history[i-1] > 0:
                factor = residual_history[i] / residual_history[i-1]
                factors.append(factor)
        
        if factors:
            avg_factor = np.exp(np.mean(np.log(factors)))
            actual_ref = [residual_history[0] * (avg_factor)**i for i in iterations]
            ax.semilogy(iterations, actual_ref, 'r:', alpha=0.7, 
                       label=f'Actual factor ≈ {avg_factor:.3f}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual Norm')
    ax.set_title(f'Convergence History: {title}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with convergence info
    if len(residual_history) >= 2:
        reduction_factor = residual_history[-1] / residual_history[0]
        textstr = f'Initial: {residual_history[0]:.2e}\n'
        textstr += f'Final: {residual_history[-1]:.2e}\n'
        textstr += f'Reduction: {reduction_factor:.2e}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()