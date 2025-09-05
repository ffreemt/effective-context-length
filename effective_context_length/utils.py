"""
Utility functions and classes for the Effective Context Length CLI Tool.

This module provides logging configuration, error handling utilities,
and other common functionality used throughout the application.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    verbose_level: int = 0,
    quiet: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration based on verbosity level.
    
    Args:
        verbose_level: Verbosity level (0=INFO simple, 1=DEBUG with location, 2+=detailed)
        quiet: If True, suppress console output
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
        
    Format Details:
        - Level 0 (default): Simple format - '%(levelname)s: %(message)s'
        - Level 1 (debug): With filename and line - '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        - Level 2+ (detailed): Full format with logger name - '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        - File logging: Always uses detailed format with full location information
    """
    # Determine log level based on verbosity
    if quiet:
        console_level = logging.CRITICAL + 1  # Effectively disable console logging
    elif verbose_level == 0:
        console_level = logging.INFO
    elif verbose_level == 1:
        console_level = logging.DEBUG
    else:
        console_level = logging.DEBUG
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        
        if verbose_level >= 2:
            # Detailed format for high verbosity
            console_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        elif verbose_level == 1:
            # Debug format with filename and line number
            console_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        else:
            # Simple format for normal use
            console_format = '%(levelname)s: %(message)s'
        
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always capture all levels in file
        
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressReporter:
    """Simple progress reporting utility."""
    
    def __init__(self, total: int, description: str = "Progress", quiet: bool = False):
        self.total = total
        self.current = 0
        self.description = description
        self.quiet = quiet
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress and optionally display message."""
        self.current += increment
        
        if not self.quiet:
            percentage = (self.current / self.total) * 100 if self.total > 0 else 0
            status_msg = f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)"
            
            if message:
                status_msg += f" - {message}"
            
            self.logger.info(status_msg)
    
    def finish(self, message: Optional[str] = None):
        """Mark progress as complete."""
        if not self.quiet:
            final_msg = f"{self.description}: Complete"
            if message:
                final_msg += f" - {message}"
            self.logger.info(final_msg)


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class APIError(Exception):
    """Raised when there's an API-related error."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class TokenCountError(Exception):
    """Raised when there's an error with token counting."""
    pass


def validate_url(url: str) -> bool:
    """
    Validate that a URL is properly formatted.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not url:
        return False
    
    return url.startswith(('http://', 'https://'))


def validate_positive_int(value: Any, name: str) -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Returns:
        Validated integer value
        
    Raises:
        ConfigurationError: If value is not a positive integer
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ConfigurationError(f"{name} must be positive, got {int_value}")
        return int_value
    except (ValueError, TypeError):
        raise ConfigurationError(f"{name} must be an integer, got {type(value).__name__}")


def validate_float_range(value: Any, name: str, min_val: float, max_val: float) -> float:
    """
    Validate that a value is a float within a specified range.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        Validated float value
        
    Raises:
        ConfigurationError: If value is not within the specified range
    """
    try:
        float_value = float(value)
        if not (min_val <= float_value <= max_val):
            raise ConfigurationError(f"{name} must be between {min_val} and {max_val}, got {float_value}")
        return float_value
    except (ValueError, TypeError):
        raise ConfigurationError(f"{name} must be a number, got {type(value).__name__}")


def safe_json_dump(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Safely serialize data to JSON, handling datetime and other non-serializable objects.
    
    Args:
        data: Data to serialize
        indent: JSON indentation level
        
    Returns:
        JSON string
    """
    def json_serializer(obj):
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    return json.dumps(data, indent=indent, default=json_serializer, ensure_ascii=False)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(number: int) -> str:
    """
    Format large numbers with thousand separators.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{number:,}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_error_summary(exception: Exception) -> Dict[str, Any]:
    """
    Get a structured summary of an exception.
    
    Args:
        exception: Exception to summarize
        
    Returns:
        Dictionary with error details
    """
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "module": getattr(exception, '__module__', 'unknown'),
        "args": exception.args if hasattr(exception, 'args') else []
    }


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.debug(f"Completed {self.description} in {format_duration(duration)}")
        else:
            self.logger.error(f"Failed {self.description} after {format_duration(duration)}: {exc_val}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration in seconds, if timing is complete."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None