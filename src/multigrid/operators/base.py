"""Base class for discrete operators."""

from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid


class BaseOperator(ABC):
    """Abstract base class for discrete operators."""
    
    def __init__(self, name: str = "BaseOperator"):
        """
        Initialize base operator.
        
        Args:
            name: Human-readable name for the operator
        """
        self.name = name
    
    @abstractmethod
    def apply(self, grid: 'Grid', field: np.ndarray) -> np.ndarray:
        """
        Apply the operator to a field.
        
        Args:
            grid: Computational grid
            field: Input field
            
        Returns:
            Result of operator application
        """
        pass
    
    @abstractmethod
    def can_apply(self, grid: 'Grid') -> bool:
        """
        Check if operator can be applied to the given grid.
        
        Args:
            grid: Grid to check compatibility with
            
        Returns:
            True if operator can be applied, False otherwise
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the operator."""
        return self.name
    
    def __repr__(self) -> str:
        """Detailed representation of the operator."""
        return f"{self.__class__.__name__}(name='{self.name}')"