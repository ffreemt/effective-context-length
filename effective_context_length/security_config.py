"""
Security configuration and settings for the effective_context_length package.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # API Key Security
    api_key_min_length: int = 10
    api_key_env_var: str = "OPENAI_API_KEY"
    allow_api_key_in_logs: bool = False
    
    # Input Validation
    max_target_tokens: int = 1000000
    max_text_length: int = 100000
    allowed_model_pattern: str = r'^[a-zA-Z0-9\-._]+$'
    temperature_min: float = 0.0
    temperature_max: float = 2.0
    
    # Rate Limiting
    max_requests_per_second: float = 10.0
    max_concurrent_connections: int = 20
    
    # Logging Security
    log_sensitive_data: bool = False
    max_error_message_length: int = 200
    redaction_patterns: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default redaction patterns."""
        if self.redaction_patterns is None:
            self.redaction_patterns = {
                'api_key': r'api[_-]?key[s]?\s*[:=]\s*[^\s]+',
                'password': r'password[s]?\s*[:=]\s*[^\s]+',
                'token': r'token[s]?\s*[:=]\s*[^\s]+',
                'secret': r'secret[s]?\s*[:=]\s*[^\s]+',
                'authorization': r'authorization\s*[:=]\s*[^\s]+',
                'bearer': r'bearer\s+[^\s]+'
            }


# Global security configuration
security_config = SecurityConfig()


def get_security_config() -> SecurityConfig:
    """Get the current security configuration."""
    return security_config


def update_security_config(**kwargs) -> None:
    """Update security configuration settings."""
    for key, value in kwargs.items():
        if hasattr(security_config, key):
            setattr(security_config, key, value)
        else:
            raise ValueError(f"Unknown security configuration parameter: {key}")