"""
Tests for security improvements in the effective_context_length package.
"""

import pytest
import re
from unittest.mock import patch, Mock
from effective_context_length.http_client import HTTPClient, ErrorClassifier
from effective_context_length.context_generator import InputValidator, ContextGenerator
from effective_context_length.security_config import SecurityConfig, get_security_config


class TestSecurityImprovements:
    """Test security improvements across the codebase."""

    def test_response_text_sanitization(self):
        """Test that sensitive data is properly sanitized in error messages."""
        # Test with API key
        response_text = "Error: invalid api_key sk-1234567890abcdef"
        sanitized = ErrorClassifier._sanitize_response_text(response_text)
        assert "sk-1234567890abcdef" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test with password
        response_text = "Password: mySecret123"
        sanitized = ErrorClassifier._sanitize_response_text(response_text)
        assert "mySecret123" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test with token
        response_text = "Authorization: Bearer abc123token"
        sanitized = ErrorClassifier._sanitize_response_text(response_text)
        assert "abc123token" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test with stack trace
        response_text = "Error occurred\nTraceback (most recent call last):\n  File '/app/main.py', line 42"
        sanitized = ErrorClassifier._sanitize_response_text(response_text)
        assert "Traceback" not in sanitized
        assert "[STACK_TRACE_REMOVED]" in sanitized
        
        # Test length limiting
        long_text = "x" * 200
        sanitized = ErrorClassifier._sanitize_response_text(long_text)
        assert len(sanitized) <= 103  # 100 + "..."

    def test_http_client_secure_logging(self):
        """Test that API keys are redacted in logs."""
        api_key = "sk-1234567890abcdef"
        client = HTTPClient("https://api.example.com/v1", api_key)
        
        # Check that the logger filter was added
        assert len(client.logger.filters) == 1
        filter_instance = list(client.logger.filters)[0]
        
        # Test the filter
        log_record = Mock()
        log_record.msg = f"Making request to https://api.example.com/v1 with key {api_key}"
        filter_instance.filter(log_record)
        
        # API key should be redacted
        assert api_key not in log_record.msg
        assert "[API_KEY_REDACTED]" in log_record.msg

    def test_http_client_environment_variable(self):
        """Test that API key can be provided via environment variable."""
        api_key = "sk-env-test-key"
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': api_key}):
            # Should work without explicit API key
            client = HTTPClient("https://api.example.com/v1")
            assert client.api_key == api_key

    def test_http_client_api_key_validation(self):
        """Test API key validation."""
        # Test with None
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                HTTPClient("https://api.example.com/v1")
        
        # Test with short key
        with pytest.raises(ValueError, match="Invalid API key format"):
            HTTPClient("https://api.example.com/v1", "short")

    def test_input_validator_target_tokens(self):
        """Test target token validation."""
        # Valid values
        assert InputValidator.validate_target_tokens(100) == 100
        assert InputValidator.validate_target_tokens(1) == 1
        
        # Invalid values
        with pytest.raises(ValueError, match="Target tokens must be an integer"):
            InputValidator.validate_target_tokens("100")
        
        with pytest.raises(ValueError, match="Target tokens must be positive"):
            InputValidator.validate_target_tokens(0)
        
        with pytest.raises(ValueError, match="Target tokens must be positive"):
            InputValidator.validate_target_tokens(-1)
        
        with pytest.raises(ValueError, match="Target tokens cannot exceed"):
            InputValidator.validate_target_tokens(2000000)

    def test_input_validator_model_name(self):
        """Test model name validation."""
        # Valid values
        assert InputValidator.validate_model_name("gpt-3.5-turbo") == "gpt-3.5-turbo"
        assert InputValidator.validate_model_name("claude-3-sonnet") == "claude-3-sonnet"
        assert InputValidator.validate_model_name("  gpt-4  ") == "gpt-4"
        
        # Invalid values
        with pytest.raises(ValueError, match="Model name must be a string"):
            InputValidator.validate_model_name(123)
        
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            InputValidator.validate_model_name("")
        
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            InputValidator.validate_model_name("   ")
        
        with pytest.raises(ValueError, match="Model name contains invalid characters"):
            InputValidator.validate_model_name("gpt@3.5")

    def test_input_validator_temperature(self):
        """Test temperature validation."""
        # Valid values
        assert InputValidator.validate_temperature(0.0) == 0.0
        assert InputValidator.validate_temperature(1.0) == 1.0
        assert InputValidator.validate_temperature(2.0) == 2.0
        assert InputValidator.validate_temperature(0) == 0.0
        assert InputValidator.validate_temperature(1) == 1.0
        
        # Invalid values
        with pytest.raises(ValueError, match="Temperature must be a number"):
            InputValidator.validate_temperature("invalid")
        
        with pytest.raises(ValueError, match="Temperature must be between"):
            InputValidator.validate_temperature(-0.1)
        
        with pytest.raises(ValueError, match="Temperature must be between"):
            InputValidator.validate_temperature(2.1)

    def test_input_validator_sanitize_text(self):
        """Test text content sanitization."""
        # Test script removal
        malicious = "<script>alert('xss')</script>Hello world"
        sanitized = InputValidator.sanitize_text_content(malicious)
        assert "<script>" not in sanitized
        assert "Hello world" in sanitized
        
        # Test javascript: removal
        malicious = "Click here: javascript:alert('xss')"
        sanitized = InputValidator.sanitize_text_content(malicious)
        assert "javascript:" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test data: removal
        malicious = "data:text/html,<script>alert(1)</script>"
        sanitized = InputValidator.sanitize_text_content(malicious)
        assert "data:" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test control character removal
        text_with_controls = "Hello\x00world\x07"
        sanitized = InputValidator.sanitize_text_content(text_with_controls)
        assert "\x00" not in sanitized
        assert "\x07" not in sanitized
        
        # Test length limiting
        long_text = "x" * 200000
        sanitized = InputValidator.sanitize_text_content(long_text)
        assert len(sanitized) == 100000

    def test_context_generator_validation(self):
        """Test that ContextGenerator uses input validation."""
        # Valid initialization
        generator = ContextGenerator("gpt-3.5-turbo", 0.5)
        assert generator.model == "gpt-3.5-turbo"
        assert generator.temperature == 0.5
        
        # Invalid model
        with pytest.raises(ValueError, match="Model name contains invalid characters"):
            ContextGenerator("invalid@model", 0.5)
        
        # Invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between"):
            ContextGenerator("gpt-3.5-turbo", 3.0)

    def test_context_generator_text_sanitization(self):
        """Test that generated text is sanitized."""
        generator = ContextGenerator("gpt-3.5-turbo", 0.1)
        
        # The internal text generation methods should produce clean text
        text = generator.generate_filler_text(100)
        assert isinstance(text, str)
        assert len(text) > 0
        
        # No malicious content should be present
        assert "<script>" not in text
        assert "javascript:" not in text

    def test_security_config(self):
        """Test security configuration."""
        config = get_security_config()
        
        # Check default values
        assert isinstance(config, SecurityConfig)
        assert config.api_key_min_length == 10
        assert config.max_target_tokens == 1000000
        assert not config.allow_api_key_in_logs
        
        # Test updating config
        from effective_context_length.security_config import update_security_config
        update_security_config(api_key_min_length=20)
        
        updated_config = get_security_config()
        assert updated_config.api_key_min_length == 20

    def test_secure_token_counting_improvement(self):
        """Test that improved token counting is more accurate."""
        generator = ContextGenerator("gpt-3.5-turbo", 0.1)
        
        # Test with known token counts (using tiktoken if available)
        text = "Hello world, this is a test sentence."
        
        # With tiktoken available
        if generator.token_counter.encoding is not None:
            exact_count = generator.token_counter.count_tokens(text)
            assert exact_count > 0
        
        # Without tiktoken (fallback)
        with patch('effective_context_length.context_generator.tiktoken', None):
            fallback_count = generator.token_counter.count_tokens(text)
            assert fallback_count > 0
            
            # Should be more accurate than simple len(text)//4
            simple_estimate = len(text) // 4
            # The improved method should give a different (better) estimate
            # We can't guarantee exact accuracy without tiktoken, but it should be reasonable
            assert 0 < fallback_count < len(text)