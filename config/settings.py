"""Configuration classes for multigrid solver settings."""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for computational grids."""
    nx: int = 65
    ny: int = 65
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    dtype: str = "float64"
    
    def validate(self) -> None:
        """Validate grid configuration."""
        if self.nx < 3 or self.ny < 3:
            raise ValueError("Grid must have at least 3 points in each direction")
        
        if len(self.domain) != 4:
            raise ValueError("Domain must be (x_min, x_max, y_min, y_max)")
        
        if self.domain[1] <= self.domain[0] or self.domain[3] <= self.domain[2]:
            raise ValueError("Invalid domain bounds")
        
        if self.dtype not in ["float32", "float64"]:
            raise ValueError(f"Unsupported dtype: {self.dtype}")


@dataclass 
class PrecisionConfig:
    """Configuration for precision management."""
    default_precision: str = "double"
    adaptive: bool = True
    convergence_threshold: float = 1e-6
    memory_threshold_gb: float = 4.0
    
    def validate(self) -> None:
        """Validate precision configuration."""
        valid_precisions = ["single", "double", "mixed", "float32", "float64"]
        if self.default_precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.default_precision}")
        
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")
        
        if self.memory_threshold_gb <= 0:
            raise ValueError("Memory threshold must be positive")


@dataclass
class SolverConfig:
    """Configuration for multigrid solver."""
    max_levels: int = 4
    max_iterations: int = 50
    tolerance: float = 1e-8
    cycle_type: str = "V"
    pre_smooth_iterations: int = 2
    post_smooth_iterations: int = 2
    smoother_type: str = "gauss_seidel"
    smoother_relaxation: float = 1.0
    coarse_tolerance: float = 1e-12
    coarse_max_iterations: int = 1000
    restriction_method: str = "full_weighting"
    prolongation_method: str = "bilinear"
    
    def validate(self) -> None:
        """Validate solver configuration."""
        if self.max_levels < 2:
            raise ValueError("Must have at least 2 grid levels")
        
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        
        if self.cycle_type not in ["V", "W", "F"]:
            raise ValueError(f"Invalid cycle type: {self.cycle_type}")
        
        if self.pre_smooth_iterations < 0 or self.post_smooth_iterations < 0:
            raise ValueError("Smoothing iterations must be non-negative")
        
        valid_smoothers = ["jacobi", "gauss_seidel", "weighted_jacobi", "symmetric_gauss_seidel"]
        if self.smoother_type not in valid_smoothers:
            raise ValueError(f"Invalid smoother type: {self.smoother_type}")
        
        if not 0 < self.smoother_relaxation <= 2:
            logger.warning(f"Relaxation parameter {self.smoother_relaxation} may cause instability")
        
        valid_restrictions = ["injection", "full_weighting", "half_weighting"]
        if self.restriction_method not in valid_restrictions:
            raise ValueError(f"Invalid restriction method: {self.restriction_method}")
        
        valid_prolongations = ["injection", "bilinear"]
        if self.prolongation_method not in valid_prolongations:
            raise ValueError(f"Invalid prolongation method: {self.prolongation_method}")


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_output: Optional[str] = None
    console_output: bool = True
    
    def validate(self) -> None:
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"Invalid logging level: {self.level}")


@dataclass
class MultigridConfig:
    """Complete configuration for multigrid solver."""
    grid: GridConfig = None
    precision: PrecisionConfig = None
    solver: SolverConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.grid is None:
            self.grid = GridConfig()
        if self.precision is None:
            self.precision = PrecisionConfig()
        if self.solver is None:
            self.solver = SolverConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.grid.validate()
        self.precision.validate()
        self.solver.validate()
        self.logging.validate()
        
        # Cross-validation
        min_nx = 3 * (2 ** (self.solver.max_levels - 1)) + 1
        min_ny = 3 * (2 ** (self.solver.max_levels - 1)) + 1
        
        if self.grid.nx < min_nx or self.grid.ny < min_ny:
            logger.warning(f"Grid size ({self.grid.nx}x{self.grid.ny}) may be too small "
                          f"for {self.solver.max_levels} levels. "
                          f"Recommend at least {min_nx}x{min_ny}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultigridConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'grid' in config_dict:
            config.grid = GridConfig(**config_dict['grid'])
        
        if 'precision' in config_dict:
            config.precision = PrecisionConfig(**config_dict['precision'])
        
        if 'solver' in config_dict:
            config.solver = SolverConfig(**config_dict['solver'])
        
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        return config
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'MultigridConfig':
        """Load configuration from JSON file."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        config.validate()
        
        logger.info(f"Loaded configuration from {json_path}")
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'MultigridConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls.from_dict(config_dict)
        config.validate()
        
        logger.info(f"Loaded configuration from {yaml_path}")
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'grid': asdict(self.grid),
            'precision': asdict(self.precision),
            'solver': asdict(self.solver),
            'logging': asdict(self.logging)
        }
    
    def to_json(self, json_path: Union[str, Path], indent: int = 2) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
        
        logger.info(f"Saved configuration to {json_path}")
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {yaml_path}")
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        # Convert string level to logging constant
        numeric_level = getattr(logging, self.logging.level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {self.logging.level}')
        
        # Create logger configuration
        log_config = {
            'level': numeric_level,
            'format': self.logging.format
        }
        
        # Setup handlers
        handlers = []
        
        if self.logging.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(console_handler)
        
        if self.logging.file_output:
            file_path = Path(self.logging.file_output)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format=self.logging.format,
            handlers=handlers,
            force=True  # Override existing configuration
        )
        
        logger.info(f"Logging configured: level={self.logging.level}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"MultigridConfig(grid={self.grid.nx}x{self.grid.ny}, "
                f"precision={self.precision.default_precision}, "
                f"solver={self.solver.cycle_type}-cycle)")
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return (f"MultigridConfig(grid={self.grid}, precision={self.precision}, "
                f"solver={self.solver}, logging={self.logging})")


def create_default_config() -> MultigridConfig:
    """Create default configuration."""
    return MultigridConfig()


def create_performance_config() -> MultigridConfig:
    """Create configuration optimized for performance."""
    config = MultigridConfig()
    config.precision.default_precision = "mixed"
    config.precision.adaptive = True
    config.solver.smoother_type = "weighted_jacobi"
    config.solver.pre_smooth_iterations = 1
    config.solver.post_smooth_iterations = 1
    config.logging.level = "WARNING"
    
    return config


def create_accuracy_config() -> MultigridConfig:
    """Create configuration optimized for accuracy."""
    config = MultigridConfig()
    config.precision.default_precision = "double"
    config.precision.adaptive = False
    config.solver.tolerance = 1e-12
    config.solver.coarse_tolerance = 1e-14
    config.solver.pre_smooth_iterations = 3
    config.solver.post_smooth_iterations = 3
    config.solver.smoother_type = "symmetric_gauss_seidel"
    
    return config