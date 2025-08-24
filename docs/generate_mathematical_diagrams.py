#!/usr/bin/env python3
"""
Generate Mathematical Diagrams for Methodology Documentation

This script creates supporting visualizations for the mathematical concepts
described in the methodology.md file, including:
- Multigrid hierarchy visualization
- Convergence analysis plots
- Mixed-precision strategy diagrams
- Performance scaling visualizations
- GPU occupancy models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_multigrid_hierarchy():
    """Create multigrid grid hierarchy visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Multigrid Grid Hierarchy', fontsize=16, fontweight='bold')
    
    # Grid sizes: 8x8, 4x4, 2x2, 1x1
    sizes = [8, 4, 2, 1]
    titles = ['Fine Grid (h)', 'Grid (2h)', 'Grid (4h)', 'Coarse Grid (8h)']
    
    for i, (ax, size, title) in enumerate(zip(axes, sizes, titles)):
        # Create grid
        x = np.linspace(0, 1, size+1)
        y = np.linspace(0, 1, size+1)
        
        # Draw grid lines
        for xi in x:
            ax.axvline(xi, color='black', linewidth=0.5)
        for yi in y:
            ax.axhline(yi, color='black', linewidth=0.5)
        
        # Add grid points
        X, Y = np.meshgrid(x, y)
        ax.scatter(X, Y, c='red', s=20, zorder=5)
        
        # Highlight center if not finest grid
        if i > 0:
            center_x, center_y = x[size//2], y[size//2]
            ax.scatter(center_x, center_y, c='blue', s=50, zorder=6)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f'Grid size: {size}×{size}')
    
    plt.tight_layout()
    return fig

def create_convergence_analysis():
    """Create convergence analysis visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multigrid Convergence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Two-grid convergence factor vs frequency
    k = np.linspace(0, np.pi, 100)
    
    # Jacobi smoothing factor
    omega = 2/3
    jacobi_factor = np.abs(1 - omega * 2 * (1 - np.cos(k)))
    
    # Two-grid convergence factor (simplified model)
    smoothing_factor = np.where(k > np.pi/2, jacobi_factor, 0.1)
    coarse_correction = np.where(k <= np.pi/2, 0.1, jacobi_factor)
    two_grid_factor = np.maximum(smoothing_factor, coarse_correction)
    
    ax1.plot(k/np.pi, jacobi_factor, label='Jacobi Smoothing', linewidth=2)
    ax1.plot(k/np.pi, two_grid_factor, label='Two-Grid Factor', linewidth=2)
    ax1.axhline(y=1/3, color='red', linestyle='--', label='Optimal (1/3)')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='High/Low Freq Split')
    ax1.set_xlabel('Normalized Frequency (k/π)')
    ax1.set_ylabel('Convergence Factor')
    ax1.set_title('Two-Grid Convergence Factor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual reduction over iterations
    iterations = np.arange(0, 10)
    rho = 0.1  # convergence factor
    
    # Different methods
    jacobi_residual = (0.9)**iterations
    gauss_seidel_residual = (0.5)**iterations
    multigrid_residual = rho**iterations
    
    ax2.semilogy(iterations, jacobi_residual, 'o-', label='Jacobi (ρ=0.9)', linewidth=2)
    ax2.semilogy(iterations, gauss_seidel_residual, 's-', label='Gauss-Seidel (ρ=0.5)', linewidth=2)
    ax2.semilogy(iterations, multigrid_residual, '^-', label='Multigrid (ρ=0.1)', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual (log scale)')
    ax2.set_title('Convergence Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Smoothing property visualization
    x = np.linspace(0, 2*np.pi, 100)
    high_freq_error = np.sin(8*x) + 0.3*np.sin(12*x)
    low_freq_error = 0.5*np.sin(x) + 0.3*np.sin(2*x)
    
    ax3.plot(x, high_freq_error, label='High Frequency Error', linewidth=2)
    ax3.plot(x, 0.3*high_freq_error, label='After Smoothing', linewidth=2, linestyle='--')
    ax3.set_xlabel('Spatial Position')
    ax3.set_ylabel('Error Amplitude')
    ax3.set_title('Smoothing Property')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. V-cycle work complexity
    grid_levels = np.arange(1, 8)
    work_per_level = 4.0**(-grid_levels)
    cumulative_work = np.cumsum(work_per_level)
    theoretical_limit = 4/3
    
    ax4.bar(grid_levels, work_per_level, alpha=0.7, label='Work per Level')
    ax4.plot(grid_levels, cumulative_work, 'ro-', label='Cumulative Work', linewidth=2)
    ax4.axhline(y=theoretical_limit, color='green', linestyle='--', 
               label=f'Theoretical Limit (4/3)', linewidth=2)
    ax4.set_xlabel('Grid Level')
    ax4.set_ylabel('Normalized Work')
    ax4.set_title('V-Cycle Work Complexity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_precision_strategy_diagram():
    """Create mixed-precision strategy visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mixed-Precision Strategy Analysis', fontsize=16, fontweight='bold')
    
    # 1. IEEE 754 precision comparison
    precisions = ['FP16', 'FP32', 'FP64']
    machine_eps = [2**(-10), 2**(-23), 2**(-52)]
    colors = ['red', 'blue', 'green']
    
    ax1.bar(precisions, machine_eps, color=colors, alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Machine Epsilon (log scale)')
    ax1.set_title('Floating-Point Precision Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add precision values as text
    for i, (prec, eps) in enumerate(zip(precisions, machine_eps)):
        ax1.text(i, eps*1.5, f'{eps:.2e}', ha='center', fontweight='bold')
    
    # 2. Error propagation in mixed precision
    iterations = np.arange(0, 50)
    
    # FP32 phase (first 30 iterations)
    fp32_phase = iterations[:30]
    fp32_residual = 1e-1 * (0.1)**fp32_phase
    fp32_roundoff = 1e-7 * np.ones_like(fp32_phase)
    fp32_total = np.maximum(fp32_residual, fp32_roundoff)
    
    # FP64 phase (remaining iterations)
    fp64_phase = iterations[30:]
    switch_residual = fp32_total[-1]
    fp64_residual = switch_residual * (0.1)**(fp64_phase - 30)
    fp64_roundoff = 1e-16 * np.ones_like(fp64_phase)
    fp64_total = np.maximum(fp64_residual, fp64_roundoff)
    
    # Combined
    total_residual = np.concatenate([fp32_total, fp64_total])
    
    ax2.semilogy(fp32_phase, fp32_residual, 'b-', label='FP32 Iteration Error', linewidth=2)
    ax2.semilogy(fp32_phase, fp32_roundoff, 'b--', label='FP32 Roundoff Floor', linewidth=2)
    ax2.semilogy(fp64_phase, fp64_residual, 'g-', label='FP64 Iteration Error', linewidth=2)
    ax2.semilogy(fp64_phase, fp64_roundoff, 'g--', label='FP64 Roundoff Floor', linewidth=2)
    ax2.semilogy(iterations, total_residual, 'r-', label='Total Error', linewidth=3, alpha=0.8)
    ax2.axvline(x=30, color='black', linestyle=':', label='Precision Switch', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual (log scale)')
    ax2.set_title('Mixed-Precision Error Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision switching criteria
    residual_values = np.logspace(-1, -12, 100)
    fp32_threshold = 1e-7 * np.ones_like(residual_values)
    switch_zone = np.logical_and(residual_values > 1e-8, residual_values < 1e-6)
    
    ax3.loglog(residual_values, residual_values, 'k--', label='Iteration Error', linewidth=2)
    ax3.loglog(residual_values, fp32_threshold, 'r-', label='FP32 Roundoff Threshold', linewidth=2)
    ax3.fill_between(residual_values, 1e-8, 1e-6, alpha=0.3, color='yellow', 
                     label='Optimal Switch Zone')
    ax3.set_xlabel('Current Residual')
    ax3.set_ylabel('Error Bound')
    ax3.set_title('Precision Switching Criteria')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cost-benefit analysis
    switch_points = np.arange(0, 41, 5)
    total_iterations = 40
    
    # Cost model: FP32 is 2x faster than FP64
    fp32_cost = 1.0
    fp64_cost = 2.0
    conversion_cost = 0.1
    
    total_costs = []
    for switch in switch_points:
        cost = switch * fp32_cost + (total_iterations - switch) * fp64_cost + conversion_cost
        total_costs.append(cost)
    
    optimal_switch = switch_points[np.argmin(total_costs)]
    
    ax4.plot(switch_points, total_costs, 'bo-', linewidth=2, markersize=8)
    ax4.axvline(x=optimal_switch, color='red', linestyle='--', 
               label=f'Optimal Switch (iter {optimal_switch})', linewidth=2)
    ax4.set_xlabel('Switch Point (Iteration)')
    ax4.set_ylabel('Total Computational Cost')
    ax4.set_title('Mixed-Precision Cost Optimization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_scaling_plots():
    """Create performance and scalability visualizations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Modeling and Scalability', fontsize=16, fontweight='bold')
    
    # 1. Multigrid complexity analysis
    problem_sizes = np.array([64, 128, 256, 512, 1024, 2048])**2  # N = n^2
    
    # Different method complexities
    direct_solver = problem_sizes**(3/2)  # O(N^(3/2)) for sparse direct
    jacobi_iterations = problem_sizes * np.log(problem_sizes)  # O(N log N) iterations * O(N) per iter
    multigrid_work = problem_sizes.astype(float)  # O(N) optimal
    
    # Normalize to smallest problem
    direct_solver = direct_solver / direct_solver[0]
    jacobi_iterations = jacobi_iterations / jacobi_iterations[0]
    multigrid_work = multigrid_work / multigrid_work[0]
    
    ax1.loglog(problem_sizes, direct_solver, 'r^-', label='Direct Solver O(N^(3/2))', 
               linewidth=2, markersize=8)
    ax1.loglog(problem_sizes, jacobi_iterations, 'bs-', label='Jacobi O(N log N)', 
               linewidth=2, markersize=8)
    ax1.loglog(problem_sizes, multigrid_work, 'go-', label='Multigrid O(N)', 
               linewidth=2, markersize=8)
    ax1.set_xlabel('Problem Size (N)')
    ax1.set_ylabel('Relative Work')
    ax1.set_title('Computational Complexity Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory bandwidth analysis
    grid_sizes = np.array([128, 256, 512, 1024, 2048])
    
    # Memory traffic per V-cycle (normalized)
    smoothing_ops = 2 * 1 * 5 * grid_sizes**2  # 2*nu*5*N for nu=1
    restriction_ops = 9 * grid_sizes**2 / 4
    prolongation_ops = 4 * grid_sizes**2 / 4
    residual_ops = 5 * grid_sizes**2
    
    total_memory = smoothing_ops + restriction_ops + prolongation_ops + residual_ops
    
    # Different precision memory requirements (bytes per V-cycle)
    fp64_memory = total_memory * 8
    fp32_memory = total_memory * 4
    mixed_memory = 0.7 * fp32_memory + 0.3 * fp64_memory  # 70% FP32, 30% FP64
    
    ax2.plot(grid_sizes, fp64_memory/1e6, 'r-o', label='FP64 Only', linewidth=2, markersize=8)
    ax2.plot(grid_sizes, fp32_memory/1e6, 'b-s', label='FP32 Only', linewidth=2, markersize=8)
    ax2.plot(grid_sizes, mixed_memory/1e6, 'g-^', label='Mixed Precision', linewidth=2, markersize=8)
    ax2.set_xlabel('Grid Size (n)')
    ax2.set_ylabel('Memory Traffic (MB per V-cycle)')
    ax2.set_title('Memory Bandwidth Requirements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. GPU occupancy analysis
    block_sizes = np.arange(32, 513, 32)
    
    # Occupancy model for V100 (simplified)
    max_blocks_per_sm = 32
    max_threads_per_sm = 2048
    shared_memory_per_sm = 48 * 1024  # 48 KB
    
    occupancy = []
    for bs in block_sizes:
        threads_per_block = bs
        shared_per_block = (bs + 2) * 4  # Simple model: (block_size + halo) * 4 bytes
        
        # Limiting factors
        blocks_by_threads = max_threads_per_sm // threads_per_block
        blocks_by_shared = shared_memory_per_sm // shared_per_block if shared_per_block > 0 else max_blocks_per_sm
        blocks_by_register = max_blocks_per_sm  # Simplified
        
        max_blocks = min(blocks_by_threads, blocks_by_shared, blocks_by_register, max_blocks_per_sm)
        occ = (max_blocks * threads_per_block) / max_threads_per_sm
        occupancy.append(min(occ, 1.0))
    
    ax3.plot(block_sizes, occupancy, 'b-o', linewidth=2, markersize=6)
    ax3.axhline(y=1.0, color='red', linestyle='--', label='100% Occupancy', alpha=0.7)
    ax3.axvline(x=256, color='green', linestyle=':', label='Typical Optimal', alpha=0.7)
    ax3.set_xlabel('Block Size (Threads)')
    ax3.set_ylabel('Theoretical Occupancy')
    ax3.set_title('GPU Kernel Occupancy Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Strong scaling efficiency
    num_gpus = np.array([1, 2, 4, 8, 16, 32, 64])
    
    # Perfect scaling
    perfect_scaling = 1.0 / num_gpus
    
    # Realistic scaling with communication overhead
    communication_overhead = 0.05 * np.log2(num_gpus)  # Log scaling with communication
    realistic_scaling = 1.0 / (num_gpus * (1 + communication_overhead))
    
    # With load imbalance
    load_imbalance = 0.02 * (num_gpus - 1)
    practical_scaling = realistic_scaling * (1 - load_imbalance)
    
    efficiency_perfect = perfect_scaling * num_gpus[0]
    efficiency_realistic = realistic_scaling * num_gpus[0]
    efficiency_practical = practical_scaling * num_gpus[0]
    
    ax4.plot(num_gpus, efficiency_perfect, 'g--', label='Perfect Scaling', linewidth=2)
    ax4.plot(num_gpus, efficiency_realistic, 'b-o', label='With Communication', linewidth=2, markersize=6)
    ax4.plot(num_gpus, efficiency_practical, 'r-s', label='Realistic (Load Imbalance)', linewidth=2, markersize=6)
    ax4.set_xlabel('Number of GPUs')
    ax4.set_ylabel('Parallel Efficiency')
    ax4.set_title('Strong Scaling Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig

def create_algorithm_flowchart():
    """Create V-cycle algorithm flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'start': 'lightgreen',
        'process': 'lightblue',
        'decision': 'yellow',
        'coarse': 'orange',
        'end': 'lightcoral'
    }
    
    # Helper function to create boxes
    def create_box(ax, x, y, width, height, text, color, box_type='rect'):
        if box_type == 'diamond':
            # Create diamond shape for decisions
            diamond = patches.FancyBboxPatch((x-width/2, y-height/2), width, height,
                                           boxstyle="round,pad=0.1", 
                                           facecolor=color, edgecolor='black')
        else:
            # Create rectangle for processes
            diamond = patches.FancyBboxPatch((x-width/2, y-height/2), width, height,
                                           boxstyle="round,pad=0.1",
                                           facecolor=color, edgecolor='black')
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, 
                weight='bold', wrap=True)
        
    def create_arrow(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.2, mid_y, label, fontsize=9, color='red')
    
    # Create flowchart elements
    create_box(ax, 5, 9, 2, 0.5, 'Start V-Cycle\nInitial guess u⁰', colors['start'])
    create_box(ax, 5, 8, 2.5, 0.5, 'Pre-smooth: ν₁ steps\nGauss-Seidel', colors['process'])
    create_box(ax, 5, 7, 2.5, 0.5, 'Compute residual\nr = f - Au', colors['process'])
    create_box(ax, 5, 6, 2.5, 0.5, 'Restrict to coarse grid\nr₂ₕ = I₂ₕʰ rʰ', colors['process'])
    
    create_box(ax, 5, 5, 2, 0.6, 'Coarsest\nGrid?', colors['decision'], 'diamond')
    
    # Coarse grid solve branch
    create_box(ax, 2.5, 4, 2, 0.5, 'Direct solve\nA₂ₕe₂ₕ = r₂ₕ', colors['coarse'])
    
    # Recursive branch
    create_box(ax, 7.5, 4, 2, 0.5, 'Recursive V-cycle\non coarse grid', colors['process'])
    
    create_box(ax, 5, 3, 2.5, 0.5, 'Prolongate correction\neʰ = Iʰ₂ₕ e₂ₕ', colors['process'])
    create_box(ax, 5, 2, 2.5, 0.5, 'Correct solution\nuʰ = uʰ + eʰ', colors['process'])
    create_box(ax, 5, 1, 2.5, 0.5, 'Post-smooth: ν₂ steps\nGauss-Seidel', colors['process'])
    create_box(ax, 5, 0.2, 2, 0.4, 'Return corrected\nsolution', colors['end'])
    
    # Create arrows
    create_arrow(ax, 5, 8.7, 5, 8.3)
    create_arrow(ax, 5, 7.7, 5, 7.3)
    create_arrow(ax, 5, 6.7, 5, 6.3)
    create_arrow(ax, 5, 5.7, 5, 5.3)
    
    # Decision branches
    create_arrow(ax, 4.2, 4.8, 2.8, 4.3, 'Yes')
    create_arrow(ax, 5.8, 4.8, 7.2, 4.3, 'No')
    
    # Return from branches
    create_arrow(ax, 2.5, 3.7, 4.5, 3.3)
    create_arrow(ax, 7.5, 3.7, 5.5, 3.3)
    
    create_arrow(ax, 5, 2.7, 5, 2.3)
    create_arrow(ax, 5, 1.7, 5, 1.3)
    create_arrow(ax, 5, 0.7, 5, 0.4)
    
    ax.set_title('V-Cycle Multigrid Algorithm Flowchart', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    """Generate all mathematical diagrams."""
    print("Generating mathematical diagrams for methodology documentation...")
    
    # Create output directory
    output_dir = Path('docs/images')
    output_dir.mkdir(exist_ok=True)
    
    # Generate diagrams
    diagrams = {
        'multigrid_hierarchy': create_multigrid_hierarchy(),
        'convergence_analysis': create_convergence_analysis(),
        'precision_strategy': create_precision_strategy_diagram(),
        'performance_scaling': create_performance_scaling_plots(),
        'v_cycle_flowchart': create_algorithm_flowchart()
    }
    
    # Save diagrams
    for name, fig in diagrams.items():
        filename = output_dir / f'{name}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved {filename}")
        
        # Also save as SVG for vector graphics
        svg_filename = output_dir / f'{name}.svg'
        fig.savefig(svg_filename, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved {svg_filename}")
        
        plt.close(fig)
    
    print(f"\n✅ Generated {len(diagrams)} mathematical diagrams in {output_dir}/")
    print("\nDiagrams created:")
    for name in diagrams.keys():
        print(f"  • {name}.png/.svg")
    
    print("\nThese diagrams can be embedded in methodology.md using:")
    print("![Diagram Description](images/diagram_name.png)")

if __name__ == "__main__":
    main()