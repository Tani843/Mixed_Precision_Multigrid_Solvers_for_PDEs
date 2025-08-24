"""
Advanced Boundary Conditions Framework
Implements Robin, Periodic, and Mixed boundary conditions for PDEs
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

from ..core.grid import Grid

logger = logging.getLogger(__name__)


class BoundaryLocation(Enum):
    """Boundary location identifiers."""
    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    TOP = "top"
    INTERIOR = "interior"  # For mixed boundary conditions


class BoundaryConditionType(Enum):
    """Types of boundary conditions."""
    DIRICHLET = "dirichlet"      # u = g
    NEUMANN = "neumann"          # ∂u/∂n = g
    ROBIN = "robin"              # αu + β∂u/∂n = g
    PERIODIC = "periodic"        # u(x1) = u(x2), ∂u/∂n(x1) = ∂u/∂n(x2)
    MIXED = "mixed"              # Different conditions on different parts


@dataclass
class BoundarySegment:
    """Defines a segment of boundary with specific conditions."""
    start_coord: float  # Parameter along boundary (0 to 1)
    end_coord: float    # Parameter along boundary (0 to 1)
    condition_type: BoundaryConditionType
    value_function: Optional[Callable[[float, float, float], float]] = None
    alpha: Optional[float] = None  # Robin condition coefficient
    beta: Optional[float] = None   # Robin condition coefficient
    
    def contains_point(self, param: float) -> bool:
        """Check if parameter is within this segment."""
        return self.start_coord <= param <= self.end_coord


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    
    def __init__(self, location: BoundaryLocation):
        self.location = location
    
    @abstractmethod
    def apply(self, u: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply boundary condition to solution array."""
        pass
    
    @abstractmethod
    def apply_to_rhs(self, rhs: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply boundary condition to RHS for implicit methods."""
        pass


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition: u = g(x, y, t)"""
    
    def __init__(self, location: BoundaryLocation, 
                 value_function: Callable[[float, float, float], float]):
        super().__init__(location)
        self.value_function = value_function
    
    def apply(self, u: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply Dirichlet condition: u = g on boundary."""
        x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
        y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
        
        if self.location == BoundaryLocation.LEFT:
            for j in range(grid.ny):
                u[0, j] = self.value_function(x[0], y[j], time)
        elif self.location == BoundaryLocation.RIGHT:
            for j in range(grid.ny):
                u[-1, j] = self.value_function(x[-1], y[j], time)
        elif self.location == BoundaryLocation.BOTTOM:
            for i in range(grid.nx):
                u[i, 0] = self.value_function(x[i], y[0], time)
        elif self.location == BoundaryLocation.TOP:
            for i in range(grid.nx):
                u[i, -1] = self.value_function(x[i], y[-1], time)
    
    def apply_to_rhs(self, rhs: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply Dirichlet condition to RHS."""
        self.apply(rhs, grid, time)  # Same as applying to solution


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition: ∂u/∂n = g(x, y, t)"""
    
    def __init__(self, location: BoundaryLocation,
                 gradient_function: Callable[[float, float, float], float]):
        super().__init__(location)
        self.gradient_function = gradient_function
    
    def apply(self, u: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply Neumann condition: ∂u/∂n = g."""
        x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
        y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
        hx, hy = grid.hx, grid.hy
        
        if self.location == BoundaryLocation.LEFT:
            # ∂u/∂n = -∂u/∂x at left boundary
            for j in range(grid.ny):
                g_val = self.gradient_function(x[0], y[j], time)
                u[0, j] = u[1, j] + hx * g_val  # Forward difference: (u[1] - u[0])/hx = -g
        elif self.location == BoundaryLocation.RIGHT:
            # ∂u/∂n = ∂u/∂x at right boundary  
            for j in range(grid.ny):
                g_val = self.gradient_function(x[-1], y[j], time)
                u[-1, j] = u[-2, j] + hx * g_val  # Backward difference
        elif self.location == BoundaryLocation.BOTTOM:
            # ∂u/∂n = -∂u/∂y at bottom boundary
            for i in range(grid.nx):
                g_val = self.gradient_function(x[i], y[0], time)
                u[i, 0] = u[i, 1] + hy * g_val
        elif self.location == BoundaryLocation.TOP:
            # ∂u/∂n = ∂u/∂y at top boundary
            for i in range(grid.nx):
                g_val = self.gradient_function(x[i], y[-1], time)
                u[i, -1] = u[i, -2] + hy * g_val
    
    def apply_to_rhs(self, rhs: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply Neumann condition to RHS (no modification needed for pure Neumann)."""
        pass  # Neumann conditions are naturally incorporated into the discretization


class RobinBC(BoundaryCondition):
    """Robin boundary condition: αu + β∂u/∂n = g(x, y, t)"""
    
    def __init__(self, location: BoundaryLocation,
                 alpha: float, beta: float,
                 value_function: Callable[[float, float, float], float]):
        super().__init__(location)
        self.alpha = alpha
        self.beta = beta
        self.value_function = value_function
    
    def apply(self, u: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply Robin condition: αu + β∂u/∂n = g."""
        x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
        y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
        hx, hy = grid.hx, grid.hy
        
        if self.location == BoundaryLocation.LEFT:
            # αu[0] + β(-∂u/∂x) = g => αu[0] - β(u[1] - u[0])/hx = g
            # u[0] = (g + β*u[1]/hx) / (α + β/hx)
            for j in range(grid.ny):
                g_val = self.value_function(x[0], y[j], time)
                u[0, j] = (g_val + self.beta * u[1, j] / hx) / (self.alpha + self.beta / hx)
        elif self.location == BoundaryLocation.RIGHT:
            for j in range(grid.ny):
                g_val = self.value_function(x[-1], y[j], time)
                u[-1, j] = (g_val + self.beta * u[-2, j] / hx) / (self.alpha + self.beta / hx)
        elif self.location == BoundaryLocation.BOTTOM:
            for i in range(grid.nx):
                g_val = self.value_function(x[i], y[0], time)
                u[i, 0] = (g_val + self.beta * u[i, 1] / hy) / (self.alpha + self.beta / hy)
        elif self.location == BoundaryLocation.TOP:
            for i in range(grid.nx):
                g_val = self.value_function(x[i], y[-1], time)
                u[i, -1] = (g_val + self.beta * u[i, -2] / hy) / (self.alpha + self.beta / hy)
    
    def apply_to_rhs(self, rhs: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply Robin condition to RHS."""
        # Robin conditions require special treatment in implicit methods
        # For now, apply same as to solution (could be improved)
        self.apply(rhs, grid, time)


class PeriodicBC(BoundaryCondition):
    """Periodic boundary conditions: u(x1) = u(x2), ∂u/∂n(x1) = ∂u/∂n(x2)"""
    
    def __init__(self, direction: str = "both"):
        """
        Initialize periodic boundary conditions.
        
        Args:
            direction: 'x', 'y', or 'both' for periodic directions
        """
        super().__init__(BoundaryLocation.INTERIOR)  # Affects multiple boundaries
        self.direction = direction
    
    def apply(self, u: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply periodic boundary conditions."""
        if self.direction in ['x', 'both']:
            # Periodic in x-direction: u(0, y) = u(Lx, y)
            u[0, :] = u[-2, :]  # u[0] = u[nx-2] (avoiding overlap)
            u[-1, :] = u[1, :]  # u[nx-1] = u[1]
        
        if self.direction in ['y', 'both']:
            # Periodic in y-direction: u(x, 0) = u(x, Ly)
            u[:, 0] = u[:, -2]  # u[:, 0] = u[:, ny-2]
            u[:, -1] = u[:, 1]  # u[:, ny-1] = u[:, 1]
    
    def apply_to_rhs(self, rhs: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply periodic conditions to RHS."""
        self.apply(rhs, grid, time)
    
    def modify_operator_for_periodicity(self, matrix: np.ndarray, grid: Grid) -> np.ndarray:
        """Modify discrete operator matrix to handle periodic boundaries."""
        # This would modify the discrete Laplacian matrix for periodic BCs
        # Implementation depends on specific matrix structure
        logger.warning("Periodic BC operator modification not yet implemented")
        return matrix


class MixedBC(BoundaryCondition):
    """Mixed boundary conditions: different conditions on different boundary segments."""
    
    def __init__(self, location: BoundaryLocation, segments: List[BoundarySegment]):
        super().__init__(location)
        self.segments = segments
        
        # Validate segments
        self._validate_segments()
    
    def _validate_segments(self) -> None:
        """Validate that segments cover boundary completely without overlap."""
        # Sort segments by start coordinate
        sorted_segments = sorted(self.segments, key=lambda s: s.start_coord)
        
        # Check coverage and overlaps
        current_pos = 0.0
        for segment in sorted_segments:
            if segment.start_coord > current_pos + 1e-10:
                logger.warning(f"Gap in boundary coverage at {current_pos}")
            elif segment.start_coord < current_pos - 1e-10:
                logger.warning(f"Overlap in boundary segments at {segment.start_coord}")
            current_pos = max(current_pos, segment.end_coord)
        
        if current_pos < 1.0 - 1e-10:
            logger.warning(f"Boundary not fully covered, ends at {current_pos}")
    
    def apply(self, u: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply mixed boundary conditions."""
        x = np.linspace(grid.domain[0], grid.domain[1], grid.nx)
        y = np.linspace(grid.domain[2], grid.domain[3], grid.ny)
        
        if self.location == BoundaryLocation.LEFT:
            self._apply_to_boundary(u, x, y, grid, time, "left", grid.ny)
        elif self.location == BoundaryLocation.RIGHT:
            self._apply_to_boundary(u, x, y, grid, time, "right", grid.ny)
        elif self.location == BoundaryLocation.BOTTOM:
            self._apply_to_boundary(u, x, y, grid, time, "bottom", grid.nx)
        elif self.location == BoundaryLocation.TOP:
            self._apply_to_boundary(u, x, y, grid, time, "top", grid.nx)
    
    def _apply_to_boundary(self, u: np.ndarray, x: np.ndarray, y: np.ndarray,
                          grid: Grid, time: float, boundary: str, n_points: int) -> None:
        """Apply conditions to specific boundary."""
        for i in range(n_points):
            param = i / (n_points - 1)  # Parameter along boundary (0 to 1)
            
            # Find which segment this point belongs to
            segment = self._find_segment(param)
            if segment is None:
                continue
            
            # Get coordinates
            if boundary in ["left", "right"]:
                coord_x = x[0] if boundary == "left" else x[-1]
                coord_y = y[i]
                boundary_idx = (0, i) if boundary == "left" else (-1, i)
            else:  # bottom or top
                coord_x = x[i]
                coord_y = y[0] if boundary == "bottom" else y[-1]
                boundary_idx = (i, 0) if boundary == "bottom" else (i, -1)
            
            # Apply appropriate condition
            self._apply_segment_condition(u, segment, boundary_idx, coord_x, coord_y, time, grid)
    
    def _find_segment(self, param: float) -> Optional[BoundarySegment]:
        """Find segment containing given parameter."""
        for segment in self.segments:
            if segment.contains_point(param):
                return segment
        return None
    
    def _apply_segment_condition(self, u: np.ndarray, segment: BoundarySegment,
                               idx: Tuple[int, int], x: float, y: float, t: float, grid: Grid) -> None:
        """Apply condition from specific segment."""
        i, j = idx
        
        if segment.condition_type == BoundaryConditionType.DIRICHLET:
            u[i, j] = segment.value_function(x, y, t) if segment.value_function else 0.0
        
        elif segment.condition_type == BoundaryConditionType.NEUMANN:
            # Implement Neumann condition
            h = min(grid.hx, grid.hy)
            g_val = segment.value_function(x, y, t) if segment.value_function else 0.0
            
            # Apply based on boundary location
            if i == 0:  # left boundary
                u[0, j] = u[1, j] + h * g_val
            elif i == grid.nx - 1:  # right boundary
                u[-1, j] = u[-2, j] + h * g_val
            elif j == 0:  # bottom boundary
                u[i, 0] = u[i, 1] + h * g_val
            elif j == grid.ny - 1:  # top boundary
                u[i, -1] = u[i, -2] + h * g_val
        
        elif segment.condition_type == BoundaryConditionType.ROBIN:
            # Implement Robin condition
            alpha, beta = segment.alpha or 1.0, segment.beta or 1.0
            g_val = segment.value_function(x, y, t) if segment.value_function else 0.0
            h = min(grid.hx, grid.hy)
            
            # Robin: αu + β∂u/∂n = g
            if i == 0:  # left boundary
                u[0, j] = (g_val + beta * u[1, j] / h) / (alpha + beta / h)
            elif i == grid.nx - 1:  # right boundary
                u[-1, j] = (g_val + beta * u[-2, j] / h) / (alpha + beta / h)
            elif j == 0:  # bottom boundary
                u[i, 0] = (g_val + beta * u[i, 1] / h) / (alpha + beta / h)
            elif j == grid.ny - 1:  # top boundary
                u[i, -1] = (g_val + beta * u[i, -2] / h) / (alpha + beta / h)
    
    def apply_to_rhs(self, rhs: np.ndarray, grid: Grid, time: float = 0.0) -> None:
        """Apply mixed boundary conditions to RHS."""
        # For mixed conditions, apply each segment appropriately
        self.apply(rhs, grid, time)


class BoundaryConditionManager:
    """Manages multiple boundary conditions for a PDE problem."""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.conditions: Dict[BoundaryLocation, BoundaryCondition] = {}
        self.has_periodic = False
    
    def add_condition(self, condition: BoundaryCondition) -> None:
        """Add boundary condition."""
        self.conditions[condition.location] = condition
        
        if isinstance(condition, PeriodicBC):
            self.has_periodic = True
            logger.info("Periodic boundary conditions detected")
    
    def apply_all_conditions(self, u: np.ndarray, time: float = 0.0) -> None:
        """Apply all boundary conditions to solution."""
        for condition in self.conditions.values():
            condition.apply(u, self.grid, time)
    
    def apply_all_to_rhs(self, rhs: np.ndarray, time: float = 0.0) -> None:
        """Apply all boundary conditions to RHS."""
        for condition in self.conditions.values():
            condition.apply_to_rhs(rhs, self.grid, time)
    
    def get_condition(self, location: BoundaryLocation) -> Optional[BoundaryCondition]:
        """Get boundary condition for specific location."""
        return self.conditions.get(location)
    
    def has_dirichlet_boundary(self) -> bool:
        """Check if any Dirichlet conditions are present."""
        return any(isinstance(bc, DirichletBC) for bc in self.conditions.values())
    
    def has_neumann_boundary(self) -> bool:
        """Check if any Neumann conditions are present."""
        return any(isinstance(bc, NeumannBC) for bc in self.conditions.values())
    
    def validate_compatibility(self) -> bool:
        """Validate that boundary conditions are mathematically compatible."""
        # Check for pure Neumann problem (requires solvability condition)
        all_neumann = all(isinstance(bc, NeumannBC) for bc in self.conditions.values())
        if all_neumann and len(self.conditions) == 4:
            logger.warning("Pure Neumann problem detected - ensure compatibility condition is satisfied")
        
        # Check periodic compatibility
        if self.has_periodic:
            for location, bc in self.conditions.items():
                if location != BoundaryLocation.INTERIOR and not isinstance(bc, PeriodicBC):
                    logger.warning("Mixing periodic and non-periodic conditions may cause issues")
        
        return True


# Factory functions for common boundary condition setups

def create_homogeneous_dirichlet(locations: List[str] = None) -> List[BoundaryCondition]:
    """Create homogeneous Dirichlet conditions (u = 0)."""
    if locations is None:
        locations = ["left", "right", "bottom", "top"]
    
    conditions = []
    for loc_str in locations:
        location = BoundaryLocation(loc_str)
        bc = DirichletBC(location, lambda x, y, t: 0.0)
        conditions.append(bc)
    
    return conditions


def create_robin_radiation_bc(location: str, heat_transfer_coeff: float, 
                            ambient_temp: float = 0.0) -> RobinBC:
    """
    Create Robin boundary condition for heat transfer (radiation/convection).
    
    Models: -k∂T/∂n = h(T - T_ambient)
    Rearranged: h*T + k∂T/∂n = h*T_ambient
    
    Args:
        location: Boundary location string
        heat_transfer_coeff: Heat transfer coefficient h
        ambient_temp: Ambient temperature
    """
    location_enum = BoundaryLocation(location)
    return RobinBC(
        location_enum,
        alpha=heat_transfer_coeff,
        beta=1.0,  # Assuming thermal conductivity k = 1
        value_function=lambda x, y, t: heat_transfer_coeff * ambient_temp
    )


def create_mixed_bc_example(location: str) -> MixedBC:
    """Create example mixed boundary condition with multiple segments."""
    location_enum = BoundaryLocation(location)
    
    segments = [
        # First third: Dirichlet u = 1
        BoundarySegment(0.0, 1/3, BoundaryConditionType.DIRICHLET, 
                       lambda x, y, t: 1.0),
        
        # Middle third: Neumann ∂u/∂n = 0
        BoundarySegment(1/3, 2/3, BoundaryConditionType.NEUMANN,
                       lambda x, y, t: 0.0),
        
        # Last third: Robin αu + β∂u/∂n = 0
        BoundarySegment(2/3, 1.0, BoundaryConditionType.ROBIN,
                       lambda x, y, t: 0.0, alpha=1.0, beta=1.0)
    ]
    
    return MixedBC(location_enum, segments)


def create_time_dependent_dirichlet(location: str, amplitude: float = 1.0, 
                                   frequency: float = 1.0) -> DirichletBC:
    """Create time-dependent Dirichlet boundary condition."""
    location_enum = BoundaryLocation(location)
    
    def time_varying(x: float, y: float, t: float) -> float:
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    return DirichletBC(location_enum, time_varying)


# Test and example usage
if __name__ == "__main__":
    # Test boundary conditions
    grid = Grid(10, 10, domain=(0, 1, 0, 1))
    u = np.random.rand(grid.nx, grid.ny)
    
    print("Testing boundary conditions...")
    
    # Test Dirichlet
    bc_dirichlet = DirichletBC(BoundaryLocation.LEFT, lambda x, y, t: 1.0)
    bc_dirichlet.apply(u, grid)
    print(f"Left boundary after Dirichlet: {u[0, :]}")
    
    # Test periodic
    bc_periodic = PeriodicBC("x")
    bc_periodic.apply(u, grid)
    print(f"Periodic x: u[0,:] = {u[0, 0]:.3f}, u[-1,:] = {u[-1, 0]:.3f}")
    
    # Test manager
    manager = BoundaryConditionManager(grid)
    manager.add_condition(DirichletBC(BoundaryLocation.LEFT, lambda x, y, t: 0.0))
    manager.add_condition(DirichletBC(BoundaryLocation.RIGHT, lambda x, y, t: 1.0))
    manager.add_condition(NeumannBC(BoundaryLocation.BOTTOM, lambda x, y, t: 0.0))
    manager.add_condition(RobinBC(BoundaryLocation.TOP, 1.0, 1.0, lambda x, y, t: 0.5))
    
    print(f"Has Dirichlet: {manager.has_dirichlet_boundary()}")
    print(f"Has Neumann: {manager.has_neumann_boundary()}")
    
    manager.apply_all_conditions(u)
    print("Applied all boundary conditions successfully")