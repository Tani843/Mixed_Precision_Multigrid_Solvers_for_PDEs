"""Logging utilities for multigrid solvers."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import time
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Add color to log messages."""
        if record.levelname in self.COLORS:
            record.levelname = (self.COLORS[record.levelname] + 
                              record.levelname + 
                              self.COLORS['RESET'])
        return super().format(record)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record):
        """Add elapsed time to log record."""
        record.elapsed_time = time.time() - self.start_time
        return True


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    colored_console: bool = True,
    include_performance: bool = False
) -> None:
    """
    Setup comprehensive logging for multigrid solvers.
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Path to log file (optional)
        console_output: Enable console output
        colored_console: Use colored console output
        include_performance: Include performance metrics
    """
    # Default format string
    if format_string is None:
        if include_performance:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - [%(elapsed_time:.3f)s] - %(message)s'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if colored_console:
            console_formatter = ColoredFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)
        
        console_handler.setFormatter(console_formatter)
        
        if include_performance:
            console_handler.addFilter(PerformanceFilter())
        
        root_logger.addHandler(console_handler)
    
    # Setup file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        
        if include_performance:
            file_handler.addFilter(PerformanceFilter())
        
        root_logger.addHandler(file_handler)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: level={logging.getLevelName(level)}, "
                f"console={console_output}, file={log_file is not None}")


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional override level for this logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    return logger


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, level: Union[str, int], logger_name: Optional[str] = None):
        """
        Initialize logging context.
        
        Args:
            level: Temporary logging level
            logger_name: Specific logger to modify (None for root)
        """
        self.new_level = level
        self.logger_name = logger_name
        self.original_level = None
        self.logger = None
    
    def __enter__(self):
        """Enter context - set new logging level."""
        if self.logger_name:
            self.logger = logging.getLogger(self.logger_name)
        else:
            self.logger = logging.getLogger()
        
        self.original_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original logging level."""
        if self.logger and self.original_level is not None:
            self.logger.setLevel(self.original_level)


@contextmanager
def silence_logger(logger_name: str):
    """Context manager to temporarily silence a specific logger."""
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield logger
    finally:
        logger.setLevel(original_level)


@contextmanager
def debug_logging(logger_name: Optional[str] = None):
    """Context manager to temporarily enable debug logging."""
    with LoggingContext(logging.DEBUG, logger_name) as logger:
        yield logger


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, total_steps: int, logger_name: str = __name__, log_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            total_steps: Total number of steps
            logger_name: Logger name
            log_interval: Log every N steps
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = time.time()
        self.logger = get_logger(logger_name)
    
    def update(self, step: Optional[int] = None, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Current step (if None, increment by 1)
            message: Additional message to log
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if (self.current_step % self.log_interval == 0 or 
            self.current_step == self.total_steps):
            
            elapsed_time = time.time() - self.start_time
            progress_pct = (self.current_step / self.total_steps) * 100
            
            if self.current_step > 0:
                avg_time_per_step = elapsed_time / self.current_step
                eta = avg_time_per_step * (self.total_steps - self.current_step)
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""
            
            log_msg = (f"Progress: {self.current_step}/{self.total_steps} "
                      f"({progress_pct:.1f}%), elapsed: {elapsed_time:.1f}s{eta_str}")
            
            if message:
                log_msg += f" - {message}"
            
            self.logger.info(log_msg)
    
    def finish(self, message: str = "Complete") -> None:
        """Mark progress as finished."""
        total_time = time.time() - self.start_time
        self.logger.info(f"{message} - Total time: {total_time:.1f}s, "
                        f"avg: {total_time/self.total_steps:.3f}s/step")


def log_function_call(func):
    """Decorator to log function calls with timing."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        # Log function entry
        logger.debug(f"Entering {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"Exiting {func.__name__} (elapsed: {elapsed_time:.3f}s)")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Exception in {func.__name__} after {elapsed_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_memory_usage(logger_name: str = __name__) -> None:
    """Log current memory usage (requires psutil)."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        logger = get_logger(logger_name)
        logger.info(f"Memory usage: {memory_mb:.1f} MB")
        
    except ImportError:
        logger = get_logger(logger_name)
        logger.warning("psutil not available for memory monitoring")


def create_logger_config(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> Dict[str, Any]:
    """
    Create a logging configuration dictionary.
    
    Args:
        level: Logging level
        format_string: Log format string
        log_file: Log file path
        console_output: Enable console output
        
    Returns:
        Logging configuration dictionary
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': format_string
            }
        },
        'handlers': {},
        'loggers': {
            '': {  # Root logger
                'handlers': [],
                'level': level,
                'propagate': False
            }
        }
    }
    
    if console_output:
        config['handlers']['console'] = {
            'level': level,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
        config['loggers']['']['handlers'].append('console')
    
    if log_file:
        config['handlers']['file'] = {
            'level': level,
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': log_file,
            'mode': 'a'
        }
        config['loggers']['']['handlers'].append('file')
    
    return config