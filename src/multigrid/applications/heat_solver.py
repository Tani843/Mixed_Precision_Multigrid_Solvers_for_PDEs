"""Heat equation solver with implicit time stepping methods."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.grid import Grid
from ..operators.laplacian import LaplacianOperator
from ..operators.transfer import RestrictionOperator, ProlongationOperator
from ..solvers.multigrid import MultigridSolver
from .poisson_solver import PoissonSolver2D

# Try to import GPU modules
try:
    from ..gpu.gpu_solver import GPUMultigridSolver, GPUCommunicationAvoidingMultigrid
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class TimeSteppingMethod(Enum):
    """Time stepping methods for heat equation."""
    BACKWARD_EULER = "backward_euler"
    CRANK_NICOLSON = "crank_nicolson"
    THETA_METHOD = "theta_method"


@dataclass
class HeatProblem:
    """Definition of a heat equation problem."""
    name: str
    initial_condition: Callable[[np.ndarray, np.ndarray], np.ndarray]
    source_function: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
    analytical_solution: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
    boundary_conditions: Optional[Dict[str, Any]] = None
    thermal_diffusivity: float = 1.0
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1)
    description: str = ""


@dataclass
class TimeSteppingConfig:
    """Configuration for time stepping."""
    method: TimeSteppingMethod
    dt: float
    t_final: float
    theta: float = 0.5  # For theta method (0.5 = Crank-Nicolson)
    adaptive_dt: bool = False
    cfl_max: float = 0.5
    save_frequency: int = 1  # Save every nth time step


class HeatSolver2D:
    """
    2D Heat equation solver with implicit time stepping.
    
    Solves: ∂u/∂t - α∇²u = f
    using implicit time stepping methods (Backward Euler, Crank-Nicolson).
    """
    
    def __init__(
        self,
        solver_type: str = "multigrid",
        max_levels: int = 6,
        max_iterations: int = 50,  # Fewer iterations per time step
        tolerance: float = 1e-6,   # Relaxed tolerance for time stepping
        cycle_type: str = "V",
        use_gpu: bool = False,
        device_id: int = 0,
        enable_mixed_precision: bool = True
    ):
        """
        Initialize Heat equation solver.
        
        Args:
            solver_type: Solver type for implicit systems
            max_levels: Maximum multigrid levels
            max_iterations: Max iterations per time step
            tolerance: Convergence tolerance per time step
            cycle_type: Multigrid cycle type
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
        
        # Create Poisson solver for implicit systems
        self.poisson_solver = PoissonSolver2D(
            solver_type=solver_type,
            max_levels=max_levels,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cycle_type=cycle_type,
            use_gpu=use_gpu,
            device_id=device_id,
            enable_mixed_precision=enable_mixed_precision
        )
        
        # Problem tracking
        self.current_problem: Optional[HeatProblem] = None
        self.time_history: List[Dict[str, Any]] = []
        
        logger.info(f"HeatSolver2D initialized: {solver_type}, GPU={use_gpu}")
    
    def solve_heat_problem(
        self,
        problem: HeatProblem,
        nx: int,
        ny: int,
        time_config: TimeSteppingConfig,
        save_solution_history: bool = False
    ) -> Dict[str, Any]:
        """
        Solve heat equation problem with time stepping.
        
        Args:
            problem: Heat problem definition
            nx: Grid points in x direction
            ny: Grid points in y direction
            time_config: Time stepping configuration
            save_solution_history: Whether to save solution at all time steps
            
        Returns:
            Solution results including time evolution
        """
        self.current_problem = problem
        
        logger.info(f"Solving heat problem: {problem.name}")
        logger.info(f"Grid: {nx}x{ny}, Method: {time_config.method.value}, "
                   f"dt={time_config.dt}, T={time_config.t_final}")
        
        # Create computational grid
        grid = Grid(nx=nx, ny=ny, domain=problem.domain)
        
        # Initialize solution with initial condition
        u_current = problem.initial_condition(grid.X, grid.Y)
        
        # Apply boundary conditions to initial solution
        if problem.boundary_conditions:
            u_current = self._apply_boundary_conditions(u_current, grid, problem.boundary_conditions, 0.0)
        else:
            # Default: homogeneous Dirichlet BC
            u_current[0, :] = u_current[-1, :] = u_current[:, 0] = u_current[:, -1] = 0.0
        
        # Time stepping parameters
        dt = time_config.dt
        t_current = 0.0
        time_steps = []
        solutions = []
        
        if save_solution_history:
            time_steps.append(t_current)
            solutions.append(u_current.copy())
        
        # Adaptive time stepping
        if time_config.adaptive_dt:
            dt = self._compute_adaptive_dt(grid, problem.thermal_diffusivity, time_config.cfl_max)
            logger.info(f"Using adaptive time step: dt = {dt:.6f}")
        
        # Time stepping loop
        step = 0
        total_solver_time = 0.0
        total_mg_iterations = 0
        
        start_time = time.time()
        
        while t_current < time_config.t_final:
            # Adjust final time step
            if t_current + dt > time_config.t_final:
                dt = time_config.t_final - t_current
            
            step += 1
            t_new = t_current + dt
            
            # Solve implicit system for next time step
            step_start = time.time()
            u_new, step_info = self._time_step(
                u_current, grid, problem, dt, t_current, t_new, time_config.method, time_config.theta
            )
            step_time = time.time() - step_start
            
            total_solver_time += step_time
            total_mg_iterations += step_info['mg_iterations']
            
            # Update solution
            u_current = u_new
            t_current = t_new
            
            # Save solution if requested
            if save_solution_history and step % time_config.save_frequency == 0:
                time_steps.append(t_current)
                solutions.append(u_current.copy())
            
            # Adaptive time stepping update
            if time_config.adaptive_dt and step_info['mg_iterations'] < 5:
                # If solving too easily, try larger time step
                dt = min(dt * 1.1, self._compute_adaptive_dt(grid, problem.thermal_diffusivity, time_config.cfl_max))
            elif time_config.adaptive_dt and step_info['mg_iterations'] > 20:
                # If struggling to converge, reduce time step
                dt = max(dt * 0.8, time_config.dt * 0.1)
            
            if step % 100 == 0 or step <= 5:
                logger.debug(f"Step {step}: t={t_current:.6f}, dt={dt:.6f}, "
                           f"mg_iters={step_info['mg_iterations']}, time={step_time:.3f}s")
        
        total_time = time.time() - start_time
        
        # Calculate errors if analytical solution available
        errors = {}
        if problem.analytical_solution:
            u_exact = problem.analytical_solution(grid.X, grid.Y, t_current)
            errors = self._compute_errors(u_current, u_exact, grid)
        
        # Compile results
        results = {
            'problem_name': problem.name,
            'grid_size': (nx, ny),
            'time_config': {
                'method': time_config.method.value,
                'dt_initial': time_config.dt,
                'dt_final': dt,
                't_final': time_config.t_final,
                'adaptive_dt': time_config.adaptive_dt
            },
            'final_solution': u_current,
            'final_time': t_current,
            'total_steps': step,
            'total_time': total_time,
            'total_solver_time': total_solver_time,
            'avg_mg_iterations': total_mg_iterations / step if step > 0 else 0,
            'total_mg_iterations': total_mg_iterations,
            'errors': errors,
            'solver_type': self.solver_type,
            'use_gpu': self.use_gpu
        }
        
        if save_solution_history:
            results['time_steps'] = np.array(time_steps)
            results['solution_history'] = solutions
        
        if problem.analytical_solution:
            results['analytical_solution'] = problem.analytical_solution(grid.X, grid.Y, t_current)
        
        # Store in history
        self.time_history.append(results)
        
        logger.info(f"Heat solve completed: {step} steps, total_time={total_time:.3f}s, "
                   f"avg_mg_iters={results['avg_mg_iterations']:.1f}")
        
        return results
    
    def _time_step(
        self,
        u_current: np.ndarray,
        grid: Grid,
        problem: HeatProblem,
        dt: float,
        t_current: float,
        t_new: float,
        method: TimeSteppingMethod,
        theta: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single time step using specified method.
        
        Args:
            u_current: Current solution
            grid: Computational grid
            problem: Heat problem
            dt: Time step size
            t_current: Current time
            t_new: New time
            method: Time stepping method
            theta: Theta parameter for theta method
            
        Returns:
            Tuple of (new_solution, step_info)
        """
        alpha = problem.thermal_diffusivity
        
        if method == TimeSteppingMethod.BACKWARD_EULER:
            # Backward Euler: (I - α*dt*∇²)u^{n+1} = u^n + dt*f^{n+1}
            return self._backward_euler_step(u_current, grid, problem, dt, t_new)
        
        elif method == TimeSteppingMethod.CRANK_NICOLSON:
            # Crank-Nicolson: (I - α*dt/2*∇²)u^{n+1} = (I + α*dt/2*∇²)u^n + dt/2*(f^n + f^{n+1})
            return self._crank_nicolson_step(u_current, grid, problem, dt, t_current, t_new)
        
        elif method == TimeSteppingMethod.THETA_METHOD:
            # θ-method: (I - α*dt*θ*∇²)u^{n+1} = (I + α*dt*(1-θ)*∇²)u^n + dt*θ*f^{n+1} + dt*(1-θ)*f^n
            return self._theta_method_step(u_current, grid, problem, dt, t_current, t_new, theta)
        
        else:
            raise ValueError(f"Unknown time stepping method: {method}")
    
    def _backward_euler_step(
        self,
        u_current: np.ndarray,
        grid: Grid,
        problem: HeatProblem,
        dt: float,
        t_new: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform Backward Euler time step."""
        alpha = problem.thermal_diffusivity
        
        # Create modified Poisson problem: (I - α*dt*∇²)u = rhs
        # This becomes: -α*dt*∇²u + u = rhs
        # Or: ∇²u - (1/α/dt)*u = -rhs/(α*dt)
        
        # Build RHS
        rhs = u_current.copy()
        
        # Add source term if present
        if problem.source_function:
            source = problem.source_function(grid.X, grid.Y, t_new)
            rhs += dt * source
        
        # Create modified Poisson problem
        # We need to solve: (1/dt - α∇²)u = rhs/dt
        # Rearranging: -α∇²u + (1/dt)u = rhs/dt
        # This is equivalent to: -∇²u + (1/(α*dt))u = rhs/(α*dt)
        
        # For this, we need a modified operator that includes the mass term
        # For now, use iterative approach with the Poisson solver
        
        # Initial guess
        u_new = u_current.copy()
        
        # Fixed point iteration: solve implicit system
        max_fp_iterations = 10
        fp_tolerance = self.tolerance * 0.1
        
        for fp_iter in range(max_fp_iterations):
            # Solve: -α∇²u_new = (u_current + dt*source - u_new)/dt
            # Rearranged: -α∇²u_new = rhs_modified
            rhs_modified = (u_current - u_new) / dt
            if problem.source_function:
                rhs_modified += problem.source_function(grid.X, grid.Y, t_new)
            rhs_modified *= -alpha
            
            # Apply boundary conditions
            if problem.boundary_conditions:
                u_new = self._apply_boundary_conditions(u_new, grid, problem.boundary_conditions, t_new)
                rhs_modified = self._apply_boundary_conditions(rhs_modified, grid, problem.boundary_conditions, t_new, is_rhs=True)
            else:
                u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
                rhs_modified[0, :] = rhs_modified[-1, :] = rhs_modified[:, 0] = rhs_modified[:, -1] = 0.0
            
            # Solve Poisson system
            result = self.poisson_solver.solve_poisson_problem(
                problem=type('TempProblem', (), {
                    'name': f'heat_backward_euler_t{t_new:.6f}',
                    'source_function': lambda x, y: rhs_modified,
                    'boundary_conditions': problem.boundary_conditions,
                    'domain': problem.domain
                })(),
                nx=grid.nx,
                ny=grid.ny,
                initial_guess=u_new
            )
            
            u_new_iteration = result['solution']
            
            # Check convergence
            change = np.max(np.abs(u_new_iteration - u_new))
            u_new = u_new_iteration
            
            if change < fp_tolerance:
                break
        
        step_info = {
            'mg_iterations': result['solver_info']['iterations'],
            'fp_iterations': fp_iter + 1,
            'method': 'backward_euler'
        }
        
        return u_new, step_info
    
    def _crank_nicolson_step(
        self,
        u_current: np.ndarray,
        grid: Grid,
        problem: HeatProblem,
        dt: float,
        t_current: float,
        t_new: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform Crank-Nicolson time step."""
        alpha = problem.thermal_diffusivity
        
        # Crank-Nicolson: (I - α*dt/2*∇²)u^{n+1} = (I + α*dt/2*∇²)u^n + dt/2*(f^n + f^{n+1})
        
        # Build explicit part: (I + α*dt/2*∇²)u^n
        # This requires applying the Laplacian to current solution
        laplacian_u = self._apply_laplacian(u_current, grid)
        rhs = u_current + alpha * (dt/2) * laplacian_u
        
        # Add source terms
        if problem.source_function:
            source_current = problem.source_function(grid.X, grid.Y, t_current)
            source_new = problem.source_function(grid.X, grid.Y, t_new)
            rhs += (dt/2) * (source_current + source_new)
        
        # Create modified Poisson problem for implicit part
        # (I - α*dt/2*∇²)u^{n+1} = rhs
        # Rearranged: α*dt/2*∇²u^{n+1} - u^{n+1} = -rhs
        # Or: ∇²u^{n+1} - (2/(α*dt))*u^{n+1} = -2*rhs/(α*dt)
        
        # Use similar approach as Backward Euler but with modified coefficients
        u_new = u_current.copy()
        
        max_fp_iterations = 10
        fp_tolerance = self.tolerance * 0.1
        
        for fp_iter in range(max_fp_iterations):
            # Solve implicit part
            rhs_modified = (rhs - u_new) * (2.0 / (alpha * dt))
            
            # Apply boundary conditions
            if problem.boundary_conditions:
                u_new = self._apply_boundary_conditions(u_new, grid, problem.boundary_conditions, t_new)
                rhs_modified = self._apply_boundary_conditions(rhs_modified, grid, problem.boundary_conditions, t_new, is_rhs=True)
            else:
                u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
                rhs_modified[0, :] = rhs_modified[-1, :] = rhs_modified[:, 0] = rhs_modified[:, -1] = 0.0
            
            # Solve Poisson system
            result = self.poisson_solver.solve_poisson_problem(
                problem=type('TempProblem', (), {
                    'name': f'heat_crank_nicolson_t{t_new:.6f}',
                    'source_function': lambda x, y: rhs_modified,
                    'boundary_conditions': problem.boundary_conditions,
                    'domain': problem.domain
                })(),
                nx=grid.nx,
                ny=grid.ny,
                initial_guess=u_new
            )
            
            u_new_iteration = result['solution']
            
            # Check convergence
            change = np.max(np.abs(u_new_iteration - u_new))
            u_new = u_new_iteration
            
            if change < fp_tolerance:
                break
        
        step_info = {
            'mg_iterations': result['solver_info']['iterations'],
            'fp_iterations': fp_iter + 1,
            'method': 'crank_nicolson'
        }
        
        return u_new, step_info
    
    def _theta_method_step(
        self,
        u_current: np.ndarray,
        grid: Grid,
        problem: HeatProblem,
        dt: float,
        t_current: float,
        t_new: float,
        theta: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform θ-method time step."""
        if theta == 0:
            # Forward Euler (explicit)
            return self._forward_euler_step(u_current, grid, problem, dt, t_current)
        elif theta == 1:
            # Backward Euler
            return self._backward_euler_step(u_current, grid, problem, dt, t_new)
        elif theta == 0.5:
            # Crank-Nicolson
            return self._crank_nicolson_step(u_current, grid, problem, dt, t_current, t_new)
        else:
            # General θ-method
            alpha = problem.thermal_diffusivity
            
            # θ-method: (I - α*dt*θ*∇²)u^{n+1} = (I + α*dt*(1-θ)*∇²)u^n + dt*θ*f^{n+1} + dt*(1-θ)*f^n
            
            # Explicit part
            laplacian_u = self._apply_laplacian(u_current, grid)
            rhs = u_current + alpha * dt * (1 - theta) * laplacian_u
            
            # Add source terms
            if problem.source_function:
                source_current = problem.source_function(grid.X, grid.Y, t_current)
                source_new = problem.source_function(grid.X, grid.Y, t_new)
                rhs += dt * ((1 - theta) * source_current + theta * source_new)
            
            # Solve implicit part similar to other methods
            u_new = u_current.copy()
            
            max_fp_iterations = 10
            fp_tolerance = self.tolerance * 0.1
            
            for fp_iter in range(max_fp_iterations):
                rhs_modified = (rhs - u_new) / (alpha * dt * theta)
                
                # Apply boundary conditions
                if problem.boundary_conditions:
                    u_new = self._apply_boundary_conditions(u_new, grid, problem.boundary_conditions, t_new)
                    rhs_modified = self._apply_boundary_conditions(rhs_modified, grid, problem.boundary_conditions, t_new, is_rhs=True)
                else:
                    u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
                    rhs_modified[0, :] = rhs_modified[-1, :] = rhs_modified[:, 0] = rhs_modified[:, -1] = 0.0
                
                # Solve Poisson system
                result = self.poisson_solver.solve_poisson_problem(
                    problem=type('TempProblem', (), {
                        'name': f'heat_theta_method_t{t_new:.6f}',
                        'source_function': lambda x, y: rhs_modified,
                        'boundary_conditions': problem.boundary_conditions,
                        'domain': problem.domain
                    })(),
                    nx=grid.nx,
                    ny=grid.ny,
                    initial_guess=u_new
                )
                
                u_new_iteration = result['solution']
                
                # Check convergence
                change = np.max(np.abs(u_new_iteration - u_new))
                u_new = u_new_iteration
                
                if change < fp_tolerance:
                    break
            
            step_info = {
                'mg_iterations': result['solver_info']['iterations'],
                'fp_iterations': fp_iter + 1,
                'method': f'theta_method_{theta}'
            }
            
            return u_new, step_info
    
    def _forward_euler_step(
        self,
        u_current: np.ndarray,
        grid: Grid,
        problem: HeatProblem,
        dt: float,
        t_current: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform Forward Euler time step (explicit)."""
        alpha = problem.thermal_diffusivity
        
        # Forward Euler: u^{n+1} = u^n + dt*(α*∇²u^n + f^n)
        laplacian_u = self._apply_laplacian(u_current, grid)
        u_new = u_current + dt * alpha * laplacian_u
        
        # Add source term if present
        if problem.source_function:
            source = problem.source_function(grid.X, grid.Y, t_current)
            u_new += dt * source
        
        # Apply boundary conditions
        if problem.boundary_conditions:
            u_new = self._apply_boundary_conditions(u_new, grid, problem.boundary_conditions, t_current + dt)
        else:
            u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
        
        step_info = {
            'mg_iterations': 0,  # No multigrid solve needed
            'fp_iterations': 1,
            'method': 'forward_euler'
        }
        
        return u_new, step_info
    
    def _apply_laplacian(self, u: np.ndarray, grid: Grid) -> np.ndarray:
        """Apply discrete Laplacian operator."""
        laplacian = np.zeros_like(u)
        hx2_inv = 1.0 / (grid.hx * grid.hx)
        hy2_inv = 1.0 / (grid.hy * grid.hy)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            hx2_inv * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
            hy2_inv * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
        )
        
        return laplacian
    
    def _apply_boundary_conditions(
        self,
        u: np.ndarray,
        grid: Grid,
        bc_config: Dict[str, Any],
        t: float,
        is_rhs: bool = False
    ) -> np.ndarray:
        """Apply boundary conditions at time t."""
        result = u.copy()
        
        bc_type = bc_config.get('type', 'dirichlet')
        
        if bc_type == 'dirichlet':
            value = bc_config.get('value', 0.0)
            if callable(value):
                # Time-dependent boundary conditions
                nx, ny = grid.shape
                for i in range(nx):
                    result[i, 0] = value(grid.X[i, 0], grid.Y[i, 0], t)    # bottom
                    result[i, -1] = value(grid.X[i, -1], grid.Y[i, -1], t)  # top
                for j in range(ny):
                    result[0, j] = value(grid.X[0, j], grid.Y[0, j], t)     # left
                    result[-1, j] = value(grid.X[-1, j], grid.Y[-1, j], t)  # right
            else:
                # Constant boundary conditions
                if not is_rhs:
                    result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = value
                else:
                    result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = 0.0
        
        return result
    
    def _compute_adaptive_dt(self, grid: Grid, alpha: float, cfl_max: float) -> float:
        """Compute adaptive time step based on CFL condition."""
        # CFL condition for diffusion: dt ≤ CFL_max * h² / (2*d*α)
        # where d is spatial dimension (2 for 2D)
        h_min = min(grid.hx, grid.hy)
        dt_max = cfl_max * h_min**2 / (2 * 2 * alpha)  # 2D
        return dt_max
    
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
        
        return {
            'l2_error': l2_error,
            'relative_l2_error': relative_l2_error,
            'max_error': max_error,
            'relative_max_error': relative_max_error,
            'grid_spacing': (grid.hx, grid.hy)
        }
    
    def analyze_stability(
        self,
        problem: HeatProblem,
        nx: int,
        ny: int,
        dt_range: List[float],
        method: TimeSteppingMethod,
        t_final: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze stability of time stepping method.
        
        Args:
            problem: Heat problem
            nx, ny: Grid dimensions
            dt_range: Range of time steps to test
            method: Time stepping method
            t_final: Final time for stability test
            
        Returns:
            Stability analysis results
        """
        logger.info(f"Analyzing stability for {method.value}")
        
        grid = Grid(nx=nx, ny=ny, domain=problem.domain)
        
        stability_results = []
        
        for dt in dt_range:
            try:
                time_config = TimeSteppingConfig(
                    method=method,
                    dt=dt,
                    t_final=t_final,
                    adaptive_dt=False
                )
                
                result = self.solve_heat_problem(problem, nx, ny, time_config, save_solution_history=False)
                
                # Check for stability (solution shouldn't blow up)
                max_solution = np.max(np.abs(result['final_solution']))
                is_stable = max_solution < 1e6 and not np.any(np.isnan(result['final_solution']))
                
                stability_results.append({
                    'dt': dt,
                    'cfl_number': dt * problem.thermal_diffusivity / min(grid.hx, grid.hy)**2,
                    'max_solution': max_solution,
                    'is_stable': is_stable,
                    'total_steps': result['total_steps'],
                    'avg_mg_iterations': result['avg_mg_iterations']
                })
                
                logger.debug(f"dt={dt:.6f}, CFL={stability_results[-1]['cfl_number']:.3f}, "
                           f"stable={is_stable}, max_val={max_solution:.2e}")
                
            except Exception as e:
                stability_results.append({
                    'dt': dt,
                    'cfl_number': dt * problem.thermal_diffusivity / min(grid.hx, grid.hy)**2,
                    'max_solution': np.inf,
                    'is_stable': False,
                    'error': str(e)
                })
        
        # Determine stability threshold
        stable_dts = [r['dt'] for r in stability_results if r['is_stable']]
        max_stable_dt = max(stable_dts) if stable_dts else 0.0
        
        return {
            'method': method.value,
            'grid_size': (nx, ny),
            'problem_name': problem.name,
            'stability_results': stability_results,
            'max_stable_dt': max_stable_dt,
            'theoretical_cfl_limit': self._get_theoretical_cfl_limit(method),
            'grid_spacing': (grid.hx, grid.hy)
        }
    
    def _get_theoretical_cfl_limit(self, method: TimeSteppingMethod) -> float:
        """Get theoretical CFL stability limit for method."""
        if method == TimeSteppingMethod.FORWARD_EULER:
            return 0.25  # dt ≤ h²/(4*α) for 2D
        elif method in [TimeSteppingMethod.BACKWARD_EULER, TimeSteppingMethod.CRANK_NICOLSON]:
            return float('inf')  # Unconditionally stable
        else:
            return 1.0  # General case


class HeatSolver3D:
    """
    3D Heat equation solver framework.
    
    Note: This is a framework implementation. Full 3D would require 3D operators.
    """
    
    def __init__(self, **kwargs):
        """Initialize 3D heat solver."""
        logger.info("HeatSolver3D initialized (framework implementation)")
        logger.warning("3D implementation requires 3D grid and operators")
    
    def get_memory_requirements_3d(self, nx: int, ny: int, nz: int, num_time_steps: int) -> Dict[str, float]:
        """Estimate memory requirements for 3D heat problem."""
        total_points = nx * ny * nz
        bytes_per_point = 8  # double precision
        
        # Memory for current and previous solutions
        solution_memory = total_points * bytes_per_point * 2
        
        # Memory for multigrid hierarchy (estimate 5 levels)
        mg_memory = 0
        current_points = total_points
        for level in range(5):
            mg_memory += current_points * bytes_per_point * 4  # 4 arrays per level
            current_points //= 8  # 3D coarsening
            if current_points < 1000:
                break
        
        # Memory for time history if saved
        history_memory = total_points * bytes_per_point * num_time_steps
        
        total_memory = solution_memory + mg_memory + history_memory
        
        return {
            'total_points': total_points,
            'solution_memory_gb': solution_memory / (1024**3),
            'multigrid_memory_gb': mg_memory / (1024**3),
            'history_memory_gb': history_memory / (1024**3),
            'total_memory_gb': total_memory / (1024**3),
            'estimated_levels': 5
        }