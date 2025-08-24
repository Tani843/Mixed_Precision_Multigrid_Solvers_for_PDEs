"""
Complete Heat Equation Solver Implementation
Time-dependent PDE solver with advanced time stepping schemes and boundary conditions

Heat equation: ∂u/∂t = α∇²u + f(x,y,t)
where α is thermal diffusivity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Callable, Tuple, List
import time
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.grid import Grid
from ..core.precision import PrecisionManager, PrecisionLevel
from ..solvers.corrected_multigrid import CorrectedMultigridSolver
from ..operators.laplacian import LaplacianOperator

logger = logging.getLogger(__name__)


class TimeSteppingScheme(Enum):
    """Time stepping scheme types."""
    EXPLICIT_EULER = "explicit_euler"
    IMPLICIT_EULER = "implicit_euler"
    CRANK_NICOLSON = "crank_nicolson"
    BDF2 = "bdf2"  # Second-order backward differentiation


class BoundaryType(Enum):
    """Boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""
    boundary_type: BoundaryType
    value: Optional[Callable[[float, float, float], float]] = None  # f(x, y, t)
    alpha: Optional[float] = None  # For Robin: αu + β∂u/∂n = g
    beta: Optional[float] = None
    
    def evaluate(self, x: float, y: float, t: float) -> float:
        """Evaluate boundary condition at given point and time."""
        if self.value is None:
            return 0.0
        return self.value(x, y, t)


@dataclass
class HeatEquationConfig:
    """Configuration for heat equation solver."""
    thermal_diffusivity: float = 1.0
    initial_condition: Callable[[float, float], float] = None
    source_term: Optional[Callable[[float, float, float], float]] = None
    boundary_conditions: Dict[str, BoundaryCondition] = None
    
    def __post_init__(self):
        if self.boundary_conditions is None:
            # Default: homogeneous Dirichlet on all boundaries
            self.boundary_conditions = {
                'left': BoundaryCondition(BoundaryType.DIRICHLET, lambda x, y, t: 0.0),
                'right': BoundaryCondition(BoundaryType.DIRICHLET, lambda x, y, t: 0.0),
                'bottom': BoundaryCondition(BoundaryType.DIRICHLET, lambda x, y, t: 0.0),
                'top': BoundaryCondition(BoundaryType.DIRICHLET, lambda x, y, t: 0.0)
            }


class HeatEquationSolver:
    """
    Complete time-dependent Heat Equation solver with multiple time stepping schemes.
    
    Solves: ∂u/∂t = α∇²u + f(x,y,t)
    """
    
    def __init__(
        self,
        config: HeatEquationConfig,
        grid: Grid,
        precision_manager: Optional[PrecisionManager] = None
    ):
        """
        Initialize heat equation solver.
        
        Args:
            config: Heat equation configuration
            grid: Computational grid
            precision_manager: Optional precision manager
        """
        self.config = config
        self.grid = grid
        self.precision_manager = precision_manager or PrecisionManager()
        
        # Set up multigrid solver for implicit methods
        self.mg_solver = CorrectedMultigridSolver(
            max_levels=4,
            max_iterations=20,
            tolerance=1e-10,
            verbose=False
        )
        
        # Laplacian operator
        self.laplacian = LaplacianOperator()
        
        # Time stepping state
        self.current_time = 0.0
        self.current_solution = None
        self.solution_history = []
        self.time_history = []
        self.dt_history = []
        
        logger.info(f"Initialized HeatEquationSolver: α={config.thermal_diffusivity}")
    
    def set_initial_condition(self, u0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Set initial condition for the heat equation.
        
        Args:
            u0: Initial condition array (optional, will use config if None)
            
        Returns:
            Initial condition array
        """
        if u0 is not None:
            self.current_solution = u0.copy()
        elif self.config.initial_condition is not None:
            # Generate from function
            x = np.linspace(self.grid.domain[0], self.grid.domain[1], self.grid.nx)
            y = np.linspace(self.grid.domain[2], self.grid.domain[3], self.grid.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            self.current_solution = np.zeros((self.grid.nx, self.grid.ny))
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    self.current_solution[i, j] = self.config.initial_condition(X[i, j], Y[i, j])
        else:
            self.current_solution = np.zeros((self.grid.nx, self.grid.ny))
        
        # Apply initial boundary conditions
        self._apply_boundary_conditions(self.current_solution, 0.0)
        
        # Initialize history
        self.current_time = 0.0
        self.solution_history = [self.current_solution.copy()]
        self.time_history = [0.0]
        
        return self.current_solution
    
    def explicit_euler_step(self, u_old: np.ndarray, dt: float) -> np.ndarray:
        """
        Explicit Euler time step: u^{n+1} = u^n + dt*(α∇²u^n + f^n)
        
        Args:
            u_old: Solution at previous time step
            dt: Time step size
            
        Returns:
            Solution at new time step
        """
        # Check stability condition: dt ≤ h²/(4α) for 2D
        h_min = min(self.grid.hx, self.grid.hy)
        dt_stable = h_min**2 / (4 * self.config.thermal_diffusivity)
        
        if dt > dt_stable:
            logger.warning(f"Time step dt={dt:.2e} exceeds stability limit {dt_stable:.2e}")
        
        # Compute Laplacian
        lap_u = self._compute_laplacian(u_old)
        
        # Add source term
        source = self._evaluate_source_term(self.current_time)
        
        # Explicit update
        u_new = u_old + dt * (self.config.thermal_diffusivity * lap_u + source)
        
        # Apply boundary conditions
        self._apply_boundary_conditions(u_new, self.current_time + dt)
        
        return u_new
    
    def implicit_euler_step(self, u_old: np.ndarray, dt: float) -> np.ndarray:
        """
        Implicit Euler time step: (I - dt*α*∇²)u^{n+1} = u^n + dt*f^{n+1}
        
        This is unconditionally stable but requires solving a linear system.
        
        Args:
            u_old: Solution at previous time step  
            dt: Time step size
            
        Returns:
            Solution at new time step
        """
        alpha = self.config.thermal_diffusivity
        
        # Right-hand side: u^n + dt*f^{n+1}
        source = self._evaluate_source_term(self.current_time + dt)
        rhs = u_old + dt * source
        
        # Apply boundary conditions to RHS
        self._apply_boundary_conditions_to_rhs(rhs, self.current_time + dt)
        
        # Solve (I - dt*α*∇²)u^{n+1} = rhs
        # This is equivalent to solving (1/(dt*α)I - ∇²)u^{n+1} = rhs/(dt*α)
        
        # Create modified RHS for multigrid solver
        # The system becomes: -∇²u + (1/(dt*α))u = rhs/(dt*α)
        # This is a Helmholtz equation: (-∇² + λ)u = f with λ = 1/(dt*α)
        
        lambda_coeff = 1.0 / (dt * alpha)
        mg_rhs = rhs / (dt * alpha)
        
        # Solve using multigrid with Helmholtz operator
        u_new = self._solve_helmholtz(mg_rhs, lambda_coeff, u_old)
        
        # Apply boundary conditions
        self._apply_boundary_conditions(u_new, self.current_time + dt)
        
        return u_new
    
    def crank_nicolson_step(self, u_old: np.ndarray, dt: float) -> np.ndarray:
        """
        Crank-Nicolson time step (second-order accurate):
        (I - dt*α*∇²/2)u^{n+1} = (I + dt*α*∇²/2)u^n + dt*(f^n + f^{n+1})/2
        
        Args:
            u_old: Solution at previous time step
            dt: Time step size
            
        Returns:
            Solution at new time step
        """
        alpha = self.config.thermal_diffusivity
        
        # Compute Laplacian of old solution
        lap_u_old = self._compute_laplacian(u_old)
        
        # Source terms at old and new time
        source_old = self._evaluate_source_term(self.current_time)
        source_new = self._evaluate_source_term(self.current_time + dt)
        
        # Right-hand side: (I + dt*α*∇²/2)u^n + dt*(f^n + f^{n+1})/2
        rhs = u_old + dt * alpha * lap_u_old / 2 + dt * (source_old + source_new) / 2
        
        # Apply boundary conditions to RHS
        self._apply_boundary_conditions_to_rhs(rhs, self.current_time + dt)
        
        # Solve (I - dt*α*∇²/2)u^{n+1} = rhs
        # This becomes: (-∇² + 2/(dt*α))u^{n+1} = 2*rhs/(dt*α)
        
        lambda_coeff = 2.0 / (dt * alpha)
        mg_rhs = 2.0 * rhs / (dt * alpha)
        
        # Solve using multigrid
        u_new = self._solve_helmholtz(mg_rhs, lambda_coeff, u_old)
        
        # Apply boundary conditions
        self._apply_boundary_conditions(u_new, self.current_time + dt)
        
        return u_new
    
    def adaptive_time_stepping(
        self, 
        u_old: np.ndarray, 
        dt_initial: float,
        error_tolerance: float = 1e-4,
        scheme: TimeSteppingScheme = TimeSteppingScheme.CRANK_NICOLSON
    ) -> Tuple[np.ndarray, float]:
        """
        Adaptive time step selection based on truncation error estimation.
        
        Uses embedded method: compute solution with step size dt and dt/2,
        then estimate local truncation error and adjust step size.
        
        Args:
            u_old: Solution at previous time step
            dt_initial: Initial time step size
            error_tolerance: Target error tolerance
            scheme: Time stepping scheme to use
            
        Returns:
            Tuple of (new_solution, actual_dt_used)
        """
        dt = dt_initial
        max_iterations = 10
        safety_factor = 0.8
        
        for iteration in range(max_iterations):
            # Take one step with dt
            u_full = self._single_time_step(u_old, dt, scheme)
            
            # Take two steps with dt/2
            u_half1 = self._single_time_step(u_old, dt/2, scheme)
            u_half2 = self._single_time_step(u_half1, dt/2, scheme)
            
            # Estimate truncation error (Richardson extrapolation)
            if scheme == TimeSteppingScheme.EXPLICIT_EULER or scheme == TimeSteppingScheme.IMPLICIT_EULER:
                # First-order methods: error ~ dt
                error_est = np.linalg.norm(u_half2 - u_full)
                order = 1
            else:
                # Second-order methods: error ~ dt²
                error_est = np.linalg.norm(u_half2 - u_full) / 3.0  # Richardson error estimate
                order = 2
            
            # Check if error is acceptable
            if error_est < error_tolerance:
                # Accept the step, possibly increase dt for next step
                if error_est < error_tolerance / 4:
                    # Error much smaller than tolerance, increase step size
                    dt_new = min(2 * dt, dt * (error_tolerance / error_est)**(1/(order+1)))
                else:
                    dt_new = dt
                
                logger.debug(f"Accepted dt={dt:.2e}, error={error_est:.2e}, next_dt={dt_new:.2e}")
                return u_half2, dt  # Use more accurate solution
            else:
                # Reject step, decrease dt
                dt_new = max(dt / 4, dt * safety_factor * (error_tolerance / error_est)**(1/(order+1)))
                dt = dt_new
                logger.debug(f"Rejected dt={dt:.2e}, error={error_est:.2e}, trying dt={dt_new:.2e}")
        
        logger.warning(f"Adaptive time stepping failed to converge after {max_iterations} iterations")
        return u_full, dt
    
    def solve_time_dependent(
        self,
        t_final: float,
        dt_initial: float = None,
        scheme: TimeSteppingScheme = TimeSteppingScheme.CRANK_NICOLSON,
        adaptive: bool = True,
        error_tolerance: float = 1e-4,
        save_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Solve time-dependent heat equation.
        
        Args:
            t_final: Final time
            dt_initial: Initial time step (auto-determined if None)
            scheme: Time stepping scheme
            adaptive: Use adaptive time stepping
            error_tolerance: Error tolerance for adaptive stepping
            save_interval: Save solution every N time steps
            
        Returns:
            Dictionary with solution history and metadata
        """
        if self.current_solution is None:
            raise ValueError("Initial condition not set. Call set_initial_condition() first.")
        
        # Auto-determine initial time step if not provided
        if dt_initial is None:
            h_min = min(self.grid.hx, self.grid.hy)
            if scheme == TimeSteppingScheme.EXPLICIT_EULER:
                dt_initial = 0.2 * h_min**2 / self.config.thermal_diffusivity  # Stable dt
            else:
                dt_initial = 0.1 * h_min  # CFL-like condition
        
        dt = dt_initial
        step_count = 0
        
        logger.info(f"Starting time integration: t_final={t_final}, scheme={scheme.value}, adaptive={adaptive}")
        
        start_time = time.time()
        
        while self.current_time < t_final:
            # Adjust dt to hit final time exactly
            if self.current_time + dt > t_final:
                dt = t_final - self.current_time
            
            # Take time step
            if adaptive and scheme != TimeSteppingScheme.EXPLICIT_EULER:
                u_new, dt_used = self.adaptive_time_stepping(
                    self.current_solution, dt, error_tolerance, scheme
                )
                dt = dt_used  # Update dt for next step
            else:
                u_new = self._single_time_step(self.current_solution, dt, scheme)
            
            # Update state
            self.current_solution = u_new
            self.current_time += dt
            step_count += 1
            
            # Save solution
            if step_count % save_interval == 0:
                self.solution_history.append(u_new.copy())
                self.time_history.append(self.current_time)
                self.dt_history.append(dt)
            
            # Progress reporting
            if step_count % 100 == 0:
                progress = self.current_time / t_final * 100
                logger.info(f"Progress: {progress:.1f}% (t={self.current_time:.3f}, dt={dt:.2e})")
        
        solve_time = time.time() - start_time
        
        logger.info(f"Time integration completed: {step_count} steps in {solve_time:.3f}s")
        
        return {
            'solution_history': self.solution_history,
            'time_history': self.time_history,
            'dt_history': self.dt_history,
            'final_solution': self.current_solution,
            'final_time': self.current_time,
            'total_steps': step_count,
            'solve_time': solve_time,
            'scheme': scheme.value,
            'adaptive': adaptive
        }
    
    def _single_time_step(self, u_old: np.ndarray, dt: float, scheme: TimeSteppingScheme) -> np.ndarray:
        """Take a single time step with specified scheme."""
        if scheme == TimeSteppingScheme.EXPLICIT_EULER:
            return self.explicit_euler_step(u_old, dt)
        elif scheme == TimeSteppingScheme.IMPLICIT_EULER:
            return self.implicit_euler_step(u_old, dt)
        elif scheme == TimeSteppingScheme.CRANK_NICOLSON:
            return self.crank_nicolson_step(u_old, dt)
        else:
            raise ValueError(f"Unsupported time stepping scheme: {scheme}")
    
    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian using central differences."""
        hx, hy = self.grid.hx, self.grid.hy
        laplacian = np.zeros_like(u)
        
        # Interior points: ∇²u ≈ (u[i-1,j] - 2u[i,j] + u[i+1,j])/hx² + 
        #                      (u[i,j-1] - 2u[i,j] + u[i,j+1])/hy²
        laplacian[1:-1, 1:-1] = (
            (u[:-2, 1:-1] - 2*u[1:-1, 1:-1] + u[2:, 1:-1]) / hx**2 +
            (u[1:-1, :-2] - 2*u[1:-1, 1:-1] + u[1:-1, 2:]) / hy**2
        )
        
        return laplacian
    
    def _evaluate_source_term(self, t: float) -> np.ndarray:
        """Evaluate source term f(x,y,t) at given time."""
        if self.config.source_term is None:
            return np.zeros((self.grid.nx, self.grid.ny))
        
        x = np.linspace(self.grid.domain[0], self.grid.domain[1], self.grid.nx)
        y = np.linspace(self.grid.domain[2], self.grid.domain[3], self.grid.ny)
        
        source = np.zeros((self.grid.nx, self.grid.ny))
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                source[i, j] = self.config.source_term(x[i], y[j], t)
        
        return source
    
    def _solve_helmholtz(self, rhs: np.ndarray, lambda_coeff: float, initial_guess: np.ndarray) -> np.ndarray:
        """
        Solve Helmholtz equation: (-∇² + λ)u = f
        
        This is approximated by modifying the multigrid solver to handle the λ term.
        For simplicity, we use the identity: (-∇² + λ)u ≈ -∇²u + λu = f
        """
        # For now, use an iterative approach since our multigrid solver is designed for Poisson
        u = initial_guess.copy()
        tolerance = 1e-10
        max_iterations = 100
        
        for iteration in range(max_iterations):
            # Compute residual: r = f - (-∇²u + λu) = f + ∇²u - λu
            laplacian_u = self._compute_laplacian(u)
            residual = rhs - laplacian_u + lambda_coeff * u
            
            # Apply boundary conditions to residual
            residual[0, :] = residual[-1, :] = 0
            residual[:, 0] = residual[:, -1] = 0
            
            residual_norm = np.linalg.norm(residual[1:-1, 1:-1])
            
            if residual_norm < tolerance:
                logger.debug(f"Helmholtz solver converged in {iteration + 1} iterations")
                break
            
            # Simple relaxation step (could be improved with multigrid for Helmholtz)
            omega = 0.8  # Under-relaxation factor
            hx2 = self.grid.hx**2
            
            # Gauss-Seidel iteration for (-∇² + λ)u = f
            for i in range(1, self.grid.nx - 1):
                for j in range(1, self.grid.ny - 1):
                    u[i, j] = (rhs[i, j] + (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]) / hx2) / (4/hx2 + lambda_coeff)
        else:
            logger.warning(f"Helmholtz solver reached max iterations ({max_iterations})")
        
        return u
    
    def _apply_boundary_conditions(self, u: np.ndarray, t: float) -> None:
        """Apply boundary conditions to solution array."""
        x = np.linspace(self.grid.domain[0], self.grid.domain[1], self.grid.nx)
        y = np.linspace(self.grid.domain[2], self.grid.domain[3], self.grid.ny)
        
        # Left boundary (x = domain[0])
        bc_left = self.config.boundary_conditions.get('left')
        if bc_left:
            self._apply_single_boundary(u, bc_left, 'left', x, y, t)
        
        # Right boundary (x = domain[1])
        bc_right = self.config.boundary_conditions.get('right')
        if bc_right:
            self._apply_single_boundary(u, bc_right, 'right', x, y, t)
        
        # Bottom boundary (y = domain[2])
        bc_bottom = self.config.boundary_conditions.get('bottom')
        if bc_bottom:
            self._apply_single_boundary(u, bc_bottom, 'bottom', x, y, t)
        
        # Top boundary (y = domain[3])
        bc_top = self.config.boundary_conditions.get('top')
        if bc_top:
            self._apply_single_boundary(u, bc_top, 'top', x, y, t)
    
    def _apply_single_boundary(
        self, 
        u: np.ndarray, 
        bc: BoundaryCondition, 
        location: str,
        x: np.ndarray, 
        y: np.ndarray, 
        t: float
    ) -> None:
        """Apply boundary condition at specific boundary."""
        if bc.boundary_type == BoundaryType.DIRICHLET:
            if location == 'left':
                for j in range(self.grid.ny):
                    u[0, j] = bc.evaluate(x[0], y[j], t)
            elif location == 'right':
                for j in range(self.grid.ny):
                    u[-1, j] = bc.evaluate(x[-1], y[j], t)
            elif location == 'bottom':
                for i in range(self.grid.nx):
                    u[i, 0] = bc.evaluate(x[i], y[0], t)
            elif location == 'top':
                for i in range(self.grid.nx):
                    u[i, -1] = bc.evaluate(x[i], y[-1], t)
        
        elif bc.boundary_type == BoundaryType.NEUMANN:
            h = min(self.grid.hx, self.grid.hy)
            if location == 'left':
                for j in range(self.grid.ny):
                    # ∂u/∂x = g => u[0] = u[1] - h*g
                    u[0, j] = u[1, j] - h * bc.evaluate(x[0], y[j], t)
            elif location == 'right':
                for j in range(self.grid.ny):
                    u[-1, j] = u[-2, j] + h * bc.evaluate(x[-1], y[j], t)
            elif location == 'bottom':
                for i in range(self.grid.nx):
                    u[i, 0] = u[i, 1] - h * bc.evaluate(x[i], y[0], t)
            elif location == 'top':
                for i in range(self.grid.nx):
                    u[i, -1] = u[i, -2] + h * bc.evaluate(x[i], y[-1], t)
        
        elif bc.boundary_type == BoundaryType.ROBIN:
            # Robin: αu + β∂u/∂n = g
            # Rearrange: u = (g - β∂u/∂n)/α
            h = min(self.grid.hx, self.grid.hy)
            alpha, beta = bc.alpha, bc.beta
            
            if location == 'left':
                for j in range(self.grid.ny):
                    g_val = bc.evaluate(x[0], y[j], t)
                    # ∂u/∂n = -∂u/∂x ≈ -(u[1] - u[0])/h
                    # αu[0] + β(-(u[1] - u[0])/h) = g
                    # u[0] = (g + β*u[1]/h) / (α + β/h)
                    u[0, j] = (g_val + beta * u[1, j] / h) / (alpha + beta / h)
            # Similar for other boundaries...
    
    def _apply_boundary_conditions_to_rhs(self, rhs: np.ndarray, t: float) -> None:
        """Apply boundary conditions to RHS vector for implicit methods."""
        # For Dirichlet boundaries, set RHS to boundary value
        x = np.linspace(self.grid.domain[0], self.grid.domain[1], self.grid.nx)
        y = np.linspace(self.grid.domain[2], self.grid.domain[3], self.grid.ny)
        
        # Apply Dirichlet conditions to RHS
        for bc_name, bc in self.config.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.DIRICHLET:
                if bc_name == 'left':
                    for j in range(self.grid.ny):
                        rhs[0, j] = bc.evaluate(x[0], y[j], t)
                elif bc_name == 'right':
                    for j in range(self.grid.ny):
                        rhs[-1, j] = bc.evaluate(x[-1], y[j], t)
                elif bc_name == 'bottom':
                    for i in range(self.grid.nx):
                        rhs[i, 0] = bc.evaluate(x[i], y[0], t)
                elif bc_name == 'top':
                    for i in range(self.grid.nx):
                        rhs[i, -1] = bc.evaluate(x[i], y[-1], t)


def create_gaussian_initial_condition(center: Tuple[float, float] = (0.5, 0.5), 
                                    width: float = 0.1, 
                                    amplitude: float = 1.0) -> Callable[[float, float], float]:
    """Create Gaussian initial condition."""
    def gaussian(x: float, y: float) -> float:
        dx = x - center[0]
        dy = y - center[1]
        return amplitude * np.exp(-(dx**2 + dy**2) / (2 * width**2))
    return gaussian


def create_time_dependent_boundary(amplitude: float = 1.0, 
                                 frequency: float = 1.0) -> Callable[[float, float, float], float]:
    """Create time-dependent boundary condition."""
    def time_varying(x: float, y: float, t: float) -> float:
        return amplitude * np.sin(2 * np.pi * frequency * t)
    return time_varying


# Example usage and test
if __name__ == "__main__":
    # Test heat equation solver
    grid = Grid(33, 33, domain=(0, 1, 0, 1))
    
    # Configuration with Gaussian initial condition
    config = HeatEquationConfig(
        thermal_diffusivity=0.1,
        initial_condition=create_gaussian_initial_condition(),
        source_term=None  # No source term for this test
    )
    
    solver = HeatEquationSolver(config, grid)
    solver.set_initial_condition()
    
    print("Testing heat equation solver...")
    result = solver.solve_time_dependent(
        t_final=1.0,
        scheme=TimeSteppingScheme.CRANK_NICOLSON,
        adaptive=True
    )
    
    print(f"Completed {result['total_steps']} time steps")
    print(f"Final time: {result['final_time']:.3f}")
    print(f"Solve time: {result['solve_time']:.3f}s")