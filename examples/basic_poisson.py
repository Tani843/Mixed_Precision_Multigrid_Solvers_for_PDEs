"""
Basic example: Solve 2D Poisson equation using multigrid method.

Problem: -∇²u = f in Ω = [0,1]²
         u = 0 on ∂Ω

Exact solution: u(x,y) = sin(πx)sin(πy)
RHS: f(x,y) = 2π²sin(πx)sin(πy)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multigrid import Grid, LaplacianOperator, MultigridSolver
from multigrid import RestrictionOperator, ProlongationOperator, PrecisionManager
from multigrid.solvers import GaussSeidelSmoother


def main():
    """Solve 2D Poisson equation with multigrid method."""
    
    print("=" * 60)
    print("Mixed-Precision Multigrid Solver - Poisson 2D Example")
    print("=" * 60)
    
    # Problem parameters
    nx, ny = 65, 65
    domain = (0.0, 1.0, 0.0, 1.0)
    
    # Create computational grid
    print(f"Creating {nx}x{ny} grid on domain {domain}")
    grid = Grid(nx=nx, ny=ny, domain=domain)
    
    # Setup operators
    laplacian = LaplacianOperator()
    restriction = RestrictionOperator("full_weighting")
    prolongation = ProlongationOperator("bilinear")
    
    # Create precision manager for adaptive precision
    precision_mgr = PrecisionManager(
        adaptive=True,
        default_precision="mixed",
        convergence_threshold=1e-6
    )
    
    # Setup smoother
    smoother = GaussSeidelSmoother(
        max_iterations=1000,
        tolerance=1e-12,
        verbose=False
    )
    
    # Setup multigrid solver
    solver = MultigridSolver(
        max_levels=4,
        max_iterations=50,
        tolerance=1e-8,
        cycle_type="V",
        pre_smooth_iterations=2,
        post_smooth_iterations=2,
        verbose=True
    )
    
    solver.setup(grid, laplacian, restriction, prolongation, smoother)
    
    print(f"Multigrid hierarchy: {len(solver.grids)} levels")
    for i, g in enumerate(solver.grids):
        print(f"  Level {i}: {g.nx}x{g.ny} (h={g.h:.4f})")
    
    # Define exact solution and RHS
    print("\nSetting up Poisson problem...")
    u_exact = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
    rhs = 2 * np.pi**2 * np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
    
    # Apply homogeneous Dirichlet boundary conditions
    grid.apply_dirichlet_bc(0.0)
    
    # Solve the system
    print(f"\nSolving with {solver.cycle_type}-cycle multigrid...")
    print(f"Tolerance: {solver.tolerance:.2e}")
    
    solution, info = solver.solve(
        grid, laplacian, rhs, 
        precision_manager=precision_mgr
    )
    
    # Apply boundary conditions to solution
    solution[0, :] = 0.0   # Left
    solution[-1, :] = 0.0  # Right  
    solution[:, 0] = 0.0   # Bottom
    solution[:, -1] = 0.0  # Top
    
    # Results
    print("\n" + "=" * 40)
    print("SOLUTION RESULTS")
    print("=" * 40)
    
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final residual: {info['final_residual']:.2e}")
    print(f"Convergence rate: {info['convergence_rate']:.3f}")
    print(f"Total solve time: {info['total_time']:.3f}s")
    print(f"Average time per iteration: {info['average_time_per_iteration']:.3f}s")
    
    # Compute error compared to exact solution
    error = solution - u_exact
    l2_error = grid.l2_norm(error)
    max_error = grid.max_norm(error)
    relative_l2_error = l2_error / grid.l2_norm(u_exact)
    
    print(f"\nError Analysis:")
    print(f"L2 error: {l2_error:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"Relative L2 error: {relative_l2_error:.2e}")
    
    # Precision statistics
    precision_stats = precision_mgr.get_statistics()
    print(f"\nPrecision Usage:")
    print(f"Current precision: {precision_stats['current_precision']}")
    print(f"Precision changes: {len(precision_stats['precision_history']) - 1}")
    for prec, stats in precision_stats['precision_breakdown'].items():
        if stats['operations'] > 0:
            print(f"  {prec}: {stats['operations']} ops ({stats['percentage']:.1f}%)")
    
    # Create visualization
    create_plots(grid, solution, u_exact, error, info)
    
    return solution, info


def create_plots(grid, solution, u_exact, error, info):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('2D Poisson Equation - Multigrid Solution', fontsize=14)
    
    # Plot solution
    im1 = axes[0, 0].contourf(grid.X, grid.Y, solution, levels=20, cmap='viridis')
    axes[0, 0].set_title('Numerical Solution')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot exact solution
    im2 = axes[0, 1].contourf(grid.X, grid.Y, u_exact, levels=20, cmap='viridis')
    axes[0, 1].set_title('Exact Solution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot error
    im3 = axes[1, 0].contourf(grid.X, grid.Y, error, levels=20, cmap='RdBu')
    axes[1, 0].set_title('Error (Numerical - Exact)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot convergence history
    axes[1, 1].semilogy(info['residual_history'], 'b-', marker='o', markersize=4)
    axes[1, 1].axhline(y=info['final_residual'], color='r', linestyle='--', 
                       label=f'Final: {info["final_residual"]:.2e}')
    axes[1, 1].set_title('Convergence History')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Residual Norm')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(__file__).parent / 'poisson_2d_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass  # Non-interactive environment


if __name__ == "__main__":
    solution, info = main()