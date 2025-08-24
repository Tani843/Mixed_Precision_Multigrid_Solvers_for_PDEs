"""Comprehensive Poisson equation solver implementation."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import logging
from dataclasses import dataclass

from ..core.grid import Grid
from ..operators.laplacian import LaplacianOperator
from ..operators.transfer import RestrictionOperator, ProlongationOperator
from ..solvers.multigrid import MultigridSolver

# Try to import GPU modules
try:
    from ..gpu.gpu_solver import GPUMultigridSolver, GPUCommunicationAvoidingMultigrid
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PoissonProblem:
    """Definition of a Poisson problem."""
    name: str
    source_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    analytical_solution: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    boundary_conditions: Optional[Dict[str, Any]] = None
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1)
    description: str = ""


class PoissonSolver2D:
    """
    Comprehensive 2D Poisson equation solver.
    
    Solves -∇²u = f in 2D with various boundary conditions
    using multigrid methods with mixed-precision support.
    """
    
    def __init__(
        self,
        solver_type: str = "multigrid",
        max_levels: int = 6,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        cycle_type: str = "V",
        use_gpu: bool = False,
        device_id: int = 0,
        enable_mixed_precision: bool = True
    ):
        """
        Initialize Poisson solver.
        
        Args:
            solver_type: Type of solver ("multigrid", "gpu_multigrid", "gpu_ca_multigrid")
            max_levels: Maximum multigrid levels
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
            cycle_type: Multigrid cycle type ("V", "W", "F")
            use_gpu: Use GPU acceleration
            device_id: GPU device ID
            enable_mixed_precision: Enable mixed precision
        """
        self.solver_type = solver_type
        self.max_levels = max_levels
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cycle_type = cycle_type
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.enable_mixed_precision = enable_mixed_precision
        
        # Initialize solver
        self.solver = self._create_solver()
        self.operator = LaplacianOperator()
        self.restriction = RestrictionOperator("full_weighting")
        self.prolongation = ProlongationOperator("bilinear")
        
        # Problem tracking
        self.current_problem: Optional[PoissonProblem] = None
        self.solve_history: List[Dict[str, Any]] = []
        
        logger.info(f"PoissonSolver2D initialized: {solver_type}, GPU={use_gpu}")
    
    def _create_solver(self):
        """Create the appropriate solver based on configuration."""
        if self.use_gpu and GPU_AVAILABLE:
            if self.solver_type == "gpu_ca_multigrid":
                return GPUCommunicationAvoidingMultigrid(
                    device_id=self.device_id,
                    max_levels=self.max_levels,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    cycle_type=self.cycle_type,
                    enable_mixed_precision=self.enable_mixed_precision,
                    use_fmg=True
                )
            else:
                return GPUMultigridSolver(
                    device_id=self.device_id,
                    max_levels=self.max_levels,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    cycle_type=self.cycle_type,
                    enable_mixed_precision=self.enable_mixed_precision
                )
        else:
            return MultigridSolver(
                max_levels=self.max_levels,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                cycle_type=self.cycle_type
            )
    
    def solve_poisson_problem(
        self,
        problem: PoissonProblem,
        nx: int,
        ny: int,
        initial_guess: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Solve a specific Poisson problem.
        
        Args:
            problem: Poisson problem definition
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            initial_guess: Initial solution guess
            
        Returns:
            Solution dictionary with results and metrics
        """
        self.current_problem = problem
        
        # Create computational grid
        grid = Grid(nx=nx, ny=ny, domain=problem.domain)
        
        # Generate source term
        rhs = problem.source_function(grid.X, grid.Y)
        
        # Apply boundary conditions
        if problem.boundary_conditions:
            self._apply_boundary_conditions(grid, rhs, problem.boundary_conditions)
        else:
            # Default: homogeneous Dirichlet BC
            grid.apply_dirichlet_bc(0.0)
        
        # Setup solver
        self.solver.setup(grid, self.operator, self.restriction, self.prolongation)
        
        # Solve
        start_time = time.time()
        solution, solve_info = self.solver.solve(grid, self.operator, rhs, initial_guess)
        solve_time = time.time() - start_time
        
        # Calculate errors if analytical solution available
        errors = {}
        if problem.analytical_solution:
            u_exact = problem.analytical_solution(grid.X, grid.Y)
            errors = self._compute_errors(solution, u_exact, grid)
        
        # Compile results
        results = {
            'problem_name': problem.name,
            'grid_size': (nx, ny),
            'domain': problem.domain,
            'solution': solution,
            'solve_time': solve_time,
            'solver_info': solve_info,
            'errors': errors,
            'solver_type': self.solver_type,
            'use_gpu': self.use_gpu,
            'mixed_precision': self.enable_mixed_precision
        }
        
        if problem.analytical_solution:
            results['analytical_solution'] = problem.analytical_solution(grid.X, grid.Y)
        
        # Store in history
        self.solve_history.append(results)
        
        logger.info(f"Solved {problem.name}: {nx}x{ny} grid, "
                   f"time={solve_time:.3f}s, iterations={solve_info['iterations']}")
        
        return results
    
    def _apply_boundary_conditions(self, grid: Grid, rhs: np.ndarray, bc_config: Dict[str, Any]):
        """Apply boundary conditions to grid and RHS."""
        bc_type = bc_config.get('type', 'dirichlet')
        
        if bc_type == 'dirichlet':
            # Homogeneous or inhomogeneous Dirichlet BC
            value = bc_config.get('value', 0.0)
            if callable(value):
                # Function-defined boundary conditions
                # Apply on boundaries
                nx, ny = grid.shape
                for i in range(nx):
                    grid.data[i, 0] = value(grid.X[i, 0], grid.Y[i, 0])  # bottom
                    grid.data[i, -1] = value(grid.X[i, -1], grid.Y[i, -1])  # top
                for j in range(ny):
                    grid.data[0, j] = value(grid.X[0, j], grid.Y[0, j])  # left  
                    grid.data[-1, j] = value(grid.X[-1, j], grid.Y[-1, j])  # right
            else:
                # Constant boundary conditions
                grid.apply_dirichlet_bc(value)
        
        elif bc_type == 'neumann':
            # Neumann boundary conditions - modify RHS
            self._apply_neumann_bc(grid, rhs, bc_config)
        
        elif bc_type == 'mixed':
            # Mixed boundary conditions
            self._apply_mixed_bc(grid, rhs, bc_config)
    
    def _apply_neumann_bc(self, grid: Grid, rhs: np.ndarray, bc_config: Dict[str, Any]):
        """Apply Neumann boundary conditions."""
        # Neumann BC: ∂u/∂n = g on boundary
        # Modify the discrete equations at boundary points
        nx, ny = grid.shape
        hx, hy = grid.hx, grid.hy
        
        g_value = bc_config.get('value', 0.0)
        
        # Modify RHS for Neumann conditions
        # This is a simplified implementation - full implementation would modify operator
        if callable(g_value):
            # Left boundary (x = 0)
            for j in range(1, ny-1):
                rhs[0, j] -= g_value(grid.X[0, j], grid.Y[0, j]) / hx
            # Right boundary (x = 1) 
            for j in range(1, ny-1):
                rhs[-1, j] += g_value(grid.X[-1, j], grid.Y[-1, j]) / hx
            # Bottom boundary (y = 0)
            for i in range(1, nx-1):
                rhs[i, 0] -= g_value(grid.X[i, 0], grid.Y[i, 0]) / hy
            # Top boundary (y = 1)
            for i in range(1, nx-1):
                rhs[i, -1] += g_value(grid.X[i, -1], grid.Y[i, -1]) / hy
        else:
            # Constant Neumann condition
            rhs[0, 1:-1] -= g_value / hx    # Left
            rhs[-1, 1:-1] += g_value / hx   # Right
            rhs[1:-1, 0] -= g_value / hy    # Bottom
            rhs[1:-1, -1] += g_value / hy   # Top
    
    def _apply_mixed_bc(self, grid: Grid, rhs: np.ndarray, bc_config: Dict[str, Any]):
        """Apply mixed boundary conditions (different types on different boundaries)."""
        boundaries = bc_config.get('boundaries', {})
        
        for boundary, bc_info in boundaries.items():
            if boundary == 'left':
                self._apply_boundary_segment(grid, rhs, bc_info, 'left')
            elif boundary == 'right':
                self._apply_boundary_segment(grid, rhs, bc_info, 'right')
            elif boundary == 'bottom':
                self._apply_boundary_segment(grid, rhs, bc_info, 'bottom')
            elif boundary == 'top':
                self._apply_boundary_segment(grid, rhs, bc_info, 'top')
    
    def _apply_boundary_segment(self, grid: Grid, rhs: np.ndarray, bc_info: Dict[str, Any], segment: str):
        """Apply boundary condition to specific segment."""
        bc_type = bc_info.get('type', 'dirichlet')
        value = bc_info.get('value', 0.0)
        nx, ny = grid.shape
        
        if bc_type == 'dirichlet':
            if segment == 'left':
                grid.data[0, :] = value if not callable(value) else [value(grid.X[0, j], grid.Y[0, j]) for j in range(ny)]
            elif segment == 'right':
                grid.data[-1, :] = value if not callable(value) else [value(grid.X[-1, j], grid.Y[-1, j]) for j in range(ny)]
            elif segment == 'bottom':
                grid.data[:, 0] = value if not callable(value) else [value(grid.X[i, 0], grid.Y[i, 0]) for i in range(nx)]
            elif segment == 'top':
                grid.data[:, -1] = value if not callable(value) else [value(grid.X[i, -1], grid.Y[i, -1]) for i in range(nx)]
    
    def _compute_errors(self, numerical: np.ndarray, analytical: np.ndarray, grid: Grid) -> Dict[str, float]:
        """Compute various error norms."""
        error = numerical - analytical
        
        # L2 norm
        l2_error = np.sqrt(np.sum(error**2) * grid.hx * grid.hy)
        l2_norm = np.sqrt(np.sum(analytical**2) * grid.hx * grid.hy)
        relative_l2_error = l2_error / l2_norm if l2_norm > 0 else l2_error
        
        # Maximum (L∞) norm
        max_error = np.max(np.abs(error))
        max_norm = np.max(np.abs(analytical))
        relative_max_error = max_error / max_norm if max_norm > 0 else max_error
        
        # H1 semi-norm (gradient error)
        # Approximate gradients with finite differences
        grad_error_x = np.diff(error, axis=0) / grid.hx
        grad_error_y = np.diff(error, axis=1) / grid.hy
        
        # H1 semi-norm calculation (simplified)
        h1_semi_error = np.sqrt(
            np.sum(grad_error_x[:-1, :]**2) * grid.hx * grid.hy +
            np.sum(grad_error_y[:, :-1]**2) * grid.hx * grid.hy
        )
        
        return {
            'l2_error': l2_error,
            'relative_l2_error': relative_l2_error,
            'max_error': max_error,
            'relative_max_error': relative_max_error,
            'h1_semi_error': h1_semi_error,
            'grid_spacing': (grid.hx, grid.hy)
        }
    
    def run_convergence_study(
        self,
        problem: PoissonProblem,
        grid_sizes: List[Tuple[int, int]],
        expected_order: float = 2.0
    ) -> Dict[str, Any]:
        """
        Run grid convergence study for a Poisson problem.
        
        Args:
            problem: Poisson problem to study
            grid_sizes: List of (nx, ny) grid sizes
            expected_order: Expected convergence order
            
        Returns:
            Convergence study results
        """
        if not problem.analytical_solution:
            raise ValueError("Convergence study requires analytical solution")
        
        logger.info(f"Running convergence study for {problem.name}")
        
        results = []
        for nx, ny in grid_sizes:
            result = self.solve_poisson_problem(problem, nx, ny)
            results.append(result)
        
        # Calculate convergence rates
        convergence_rates = self._calculate_convergence_rates(results)
        
        # Analyze convergence
        convergence_analysis = {
            'problem_name': problem.name,
            'grid_sizes': grid_sizes,
            'results': results,
            'convergence_rates': convergence_rates,
            'expected_order': expected_order,
            'achieved_order': {
                'l2': np.mean([rate['l2'] for rate in convergence_rates]) if convergence_rates else 0,
                'max': np.mean([rate['max'] for rate in convergence_rates]) if convergence_rates else 0
            }
        }
        
        logger.info(f"Convergence study completed. Achieved order: "
                   f"L2={convergence_analysis['achieved_order']['l2']:.2f}, "
                   f"Max={convergence_analysis['achieved_order']['max']:.2f}")
        
        return convergence_analysis
    
    def _calculate_convergence_rates(self, results: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Calculate convergence rates from multiple grid results."""
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
            
            # Convergence rates: error ~ h^p => rate = log(error_ratio) / log(h_ratio)
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
    
    def benchmark_solver_performance(
        self,
        problem: PoissonProblem,
        grid_sizes: List[Tuple[int, int]],
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark solver performance across different grid sizes.
        
        Args:
            problem: Problem to benchmark
            grid_sizes: Grid sizes to test
            num_runs: Number of runs per grid size
            
        Returns:
            Performance benchmark results
        """
        logger.info(f"Benchmarking {problem.name} across {len(grid_sizes)} grid sizes")
        
        benchmark_results = []
        
        for nx, ny in grid_sizes:
            run_times = []
            iterations_list = []
            
            for run in range(num_runs):
                result = self.solve_poisson_problem(problem, nx, ny)
                run_times.append(result['solve_time'])
                iterations_list.append(result['solver_info']['iterations'])
            
            # Statistics
            avg_time = np.mean(run_times)
            std_time = np.std(run_times)
            avg_iterations = np.mean(iterations_list)
            
            # Throughput metrics
            total_unknowns = nx * ny
            throughput = total_unknowns / avg_time  # unknowns per second
            
            benchmark_results.append({
                'grid_size': (nx, ny),
                'total_unknowns': total_unknowns,
                'num_runs': num_runs,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min(run_times),
                'max_time': max(run_times),
                'avg_iterations': avg_iterations,
                'throughput': throughput,
                'solver_type': self.solver_type
            })
        
        return {
            'problem_name': problem.name,
            'solver_configuration': {
                'solver_type': self.solver_type,
                'use_gpu': self.use_gpu,
                'mixed_precision': self.enable_mixed_precision,
                'max_levels': self.max_levels,
                'cycle_type': self.cycle_type
            },
            'benchmark_results': benchmark_results
        }
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """Get comprehensive solver statistics."""
        stats = {
            'solver_type': self.solver_type,
            'configuration': {
                'max_levels': self.max_levels,
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance,
                'cycle_type': self.cycle_type,
                'use_gpu': self.use_gpu,
                'mixed_precision': self.enable_mixed_precision
            },
            'solve_history_count': len(self.solve_history)
        }
        
        if hasattr(self.solver, 'get_performance_statistics'):
            stats['performance_statistics'] = self.solver.get_performance_statistics()
        
        return stats


class PoissonSolver3D:
    """
    3D Poisson equation solver implementation.
    
    Extends the 2D solver to handle 3D problems: -∇²u = f in 3D.
    """
    
    def __init__(
        self,
        solver_type: str = "multigrid",
        max_levels: int = 5,  # Fewer levels for 3D due to memory
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        cycle_type: str = "V",
        use_gpu: bool = False,
        device_id: int = 0,
        enable_mixed_precision: bool = True
    ):
        """Initialize 3D Poisson solver."""
        self.solver_type = solver_type
        self.max_levels = max_levels
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cycle_type = cycle_type
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.enable_mixed_precision = enable_mixed_precision
        
        # Note: 3D implementation would require 3D grid and operators
        # For now, we provide the structure and interface
        logger.info(f"PoissonSolver3D initialized: {solver_type}, GPU={use_gpu}")
        logger.warning("3D implementation is a framework - full 3D operators needed")
    
    def solve_poisson_problem_3d(
        self,
        problem: PoissonProblem,
        nx: int,
        ny: int,
        nz: int,
        initial_guess: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Solve 3D Poisson problem.
        
        Note: This is a framework implementation. Full 3D would require:
        - 3D Grid class
        - 3D Laplacian operator
        - 3D transfer operators
        - 3D boundary condition handling
        """
        raise NotImplementedError("Full 3D implementation requires 3D grid and operators")
    
    def get_memory_requirements_3d(self, nx: int, ny: int, nz: int, num_levels: int = 5) -> Dict[str, float]:
        """Estimate memory requirements for 3D problem."""
        total_points = nx * ny * nz
        bytes_per_point = 8  # double precision
        
        # Estimate total memory for grid hierarchy
        total_memory = 0
        current_points = total_points
        for level in range(num_levels):
            # Multiple arrays per level (solution, RHS, residual, etc.)
            level_memory = current_points * bytes_per_point * 4  # 4 arrays per level
            total_memory += level_memory
            current_points //= 8  # 3D coarsening reduces by factor of 8
            if current_points < 1000:
                break
        
        return {
            'total_points': total_points,
            'estimated_memory_mb': total_memory / (1024**2),
            'estimated_memory_gb': total_memory / (1024**3),
            'bytes_per_point': bytes_per_point,
            'levels_estimated': level + 1
        }