"""Base class for preconditioning matrices."""

from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..operators.base import BaseOperator

logger = logging.getLogger(__name__)


class BasePreconditioner(ABC):
    """Abstract base class for preconditioning matrices."""
    
    def __init__(self, name: str = "BasePreconditioner"):
        """
        Initialize base preconditioner.
        
        Args:
            name: Human-readable name for the preconditioner
        """
        self.name = name
        self.setup_completed = False
    
    @abstractmethod
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """
        Setup the preconditioner for the given operator.
        
        Args:
            grid: Computational grid
            operator: Linear operator to precondition
        """
        pass
    
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the preconditioner: solve M*z = x for z.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector z
        """
        pass
    
    @abstractmethod
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the transpose of the preconditioner: solve M^T*z = x for z.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector z
        """
        pass
    
    def is_setup(self) -> bool:
        """Check if preconditioner has been setup."""
        return self.setup_completed
    
    def reset(self) -> None:
        """Reset preconditioner state."""
        self.setup_completed = False
        logger.debug(f"Reset {self.name} preconditioner")
    
    def __str__(self) -> str:
        """String representation of the preconditioner."""
        return self.name
    
    def __repr__(self) -> str:
        """Detailed representation of the preconditioner."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class IdentityPreconditioner(BasePreconditioner):
    """Identity preconditioner (no preconditioning)."""
    
    def __init__(self):
        """Initialize identity preconditioner."""
        super().__init__("Identity")
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """Setup identity preconditioner (no-op)."""
        self.setup_completed = True
        logger.debug("Identity preconditioner setup (no-op)")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply identity preconditioner (return input unchanged)."""
        return x.copy()
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """Apply transpose of identity (return input unchanged)."""
        return x.copy()


class CompositePreconditioner(BasePreconditioner):
    """Composite preconditioner combining multiple preconditioners."""
    
    def __init__(self, preconditioners: list, name: Optional[str] = None):
        """
        Initialize composite preconditioner.
        
        Args:
            preconditioners: List of preconditioners to compose
            name: Optional name for the composite preconditioner
        """
        if name is None:
            name = "Composite[" + "+".join([p.name for p in preconditioners]) + "]"
        
        super().__init__(name)
        self.preconditioners = preconditioners
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """Setup all component preconditioners."""
        for preconditioner in self.preconditioners:
            preconditioner.setup(grid, operator)
        
        self.setup_completed = True
        logger.debug(f"Setup composite preconditioner with {len(self.preconditioners)} components")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply all preconditioners in sequence."""
        result = x.copy()
        
        for preconditioner in self.preconditioners:
            result = preconditioner.apply(result)
        
        return result
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """Apply transpose of all preconditioners in reverse order."""
        result = x.copy()
        
        for preconditioner in reversed(self.preconditioners):
            result = preconditioner.apply_transpose(result)
        
        return result
    
    def reset(self) -> None:
        """Reset all component preconditioners."""
        super().reset()
        
        for preconditioner in self.preconditioners:
            preconditioner.reset()


class AdaptivePreconditioner(BasePreconditioner):
    """
    Adaptive preconditioner that switches based on convergence behavior.
    """
    
    def __init__(
        self,
        preconditioners: dict,
        switch_threshold: float = 0.9,
        evaluation_window: int = 5
    ):
        """
        Initialize adaptive preconditioner.
        
        Args:
            preconditioners: Dictionary of {name: preconditioner}
            switch_threshold: Convergence rate threshold for switching
            evaluation_window: Window size for convergence evaluation
        """
        super().__init__("Adaptive")
        self.preconditioners = preconditioners
        self.switch_threshold = switch_threshold
        self.evaluation_window = evaluation_window
        
        self.current_preconditioner = None
        self.convergence_history = []
        self.current_name = None
    
    def setup(self, grid: 'Grid', operator: 'BaseOperator') -> None:
        """Setup all available preconditioners."""
        for name, preconditioner in self.preconditioners.items():
            preconditioner.setup(grid, operator)
        
        # Start with first preconditioner
        self.current_name = list(self.preconditioners.keys())[0]
        self.current_preconditioner = self.preconditioners[self.current_name]
        
        self.setup_completed = True
        logger.info(f"Adaptive preconditioner initialized with {self.current_name}")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply current preconditioner."""
        if self.current_preconditioner is None:
            raise RuntimeError("Adaptive preconditioner not setup")
        
        return self.current_preconditioner.apply(x)
    
    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        """Apply transpose of current preconditioner."""
        if self.current_preconditioner is None:
            raise RuntimeError("Adaptive preconditioner not setup")
        
        return self.current_preconditioner.apply_transpose(x)
    
    def update_convergence(self, residual_norm: float) -> None:
        """
        Update convergence history and potentially switch preconditioner.
        
        Args:
            residual_norm: Current residual norm
        """
        self.convergence_history.append(residual_norm)
        
        # Check if we should switch preconditioners
        if len(self.convergence_history) >= self.evaluation_window:
            recent_residuals = self.convergence_history[-self.evaluation_window:]
            
            # Calculate convergence rate
            if len(recent_residuals) >= 2:
                convergence_rate = recent_residuals[-1] / recent_residuals[0]
                
                if convergence_rate > self.switch_threshold:
                    self._switch_preconditioner()
    
    def _switch_preconditioner(self) -> None:
        """Switch to next available preconditioner."""
        preconditioner_names = list(self.preconditioners.keys())
        current_index = preconditioner_names.index(self.current_name)
        
        # Switch to next preconditioner (cycle back to start if at end)
        next_index = (current_index + 1) % len(preconditioner_names)
        next_name = preconditioner_names[next_index]
        
        if next_name != self.current_name:
            self.current_name = next_name
            self.current_preconditioner = self.preconditioners[next_name]
            
            logger.info(f"Switched to {next_name} preconditioner")
            
            # Reset convergence history after switch
            self.convergence_history = []