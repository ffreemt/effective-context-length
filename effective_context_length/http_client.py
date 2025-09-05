"""
HTTP Client for Effective Context Length Testing

This module provides an async HTTP client with retry logic, rate limiting,
and error handling specifically designed for testing OpenAI-compatible LLM endpoints.
"""

import asyncio
import time
import logging
import re
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json

import httpx
from httpx import Response, RequestError, HTTPStatusError, TimeoutException


class ErrorType(Enum):
    """Classification of different error types that can occur during API requests."""
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    RATE_LIMIT = "rate_limit_exceeded"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    AUTHENTICATION = "authentication_error"
    NETWORK = "network_error"
    UNKNOWN = "unknown_error"


@dataclass
class APIResponse:
    """
    Structured response from API requests with comprehensive error handling.
    
    Provides a standardized response format that includes success status,
    timing information, token counts, and detailed error categorization.
    
    Attributes:
        success: Whether the request completed successfully
        status_code: HTTP status code (None for network errors)
        response_time_ms: Total response time in milliseconds
        content: Parsed JSON response content (None on error)
        error_type: Categorized error type if request failed
        error_message: Detailed error description if request failed
        tokens_sent: Number of tokens in the request payload
        tokens_received: Number of tokens in the response payload
    """
    success: bool
    status_code: Optional[int] = None
    response_time_ms: float = 0.0
    content: Optional[Dict[str, Any]] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    tokens_sent: Optional[int] = None
    tokens_received: Optional[int] = None


class RateLimiter:
    """
    Token bucket rate limiter for controlling request frequency.
    
    Implements the token bucket algorithm to provide smooth rate limiting
    with burst tolerance. This prevents overwhelming API endpoints while
    allowing for occasional bursts of requests.
    
    Attributes:
        rate: Maximum requests per second
        tokens: Current available tokens
        last_update: Last time tokens were updated
        lock: Async lock for thread-safe operations
    """
    
    def __init__(self, requests_per_second: float):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests allowed per second
        """
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire permission to make a request, blocking if necessary.
        
        Uses the token bucket algorithm to determine if a request can proceed
        immediately or must wait. Implements exponential token accumulation
        and precise timing calculations.
        """
        async with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
            else:
                # Wait until we have enough tokens
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, max_backoff: float = 60.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.logger = logging.getLogger(__name__)
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Determine if a request should be retried based on error type and attempt count."""
        if attempt >= self.max_retries:
            return False
        
        # Retry on rate limits, timeouts, and server errors
        retryable_errors = {
            ErrorType.RATE_LIMIT,
            ErrorType.TIMEOUT,
            ErrorType.SERVER_ERROR,
            ErrorType.NETWORK
        }
        
        return error_type in retryable_errors
    
    def get_backoff_delay(self, attempt: int) -> float:
        """Calculate backoff delay for the given attempt."""
        delay = min(self.backoff_factor ** attempt, self.max_backoff)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
        return delay + jitter


class ErrorClassifier:
    """Classifies API errors into specific error types."""
    
    @staticmethod
    def classify_error(response: Optional[Response] = None, exception: Optional[Exception] = None) -> tuple[ErrorType, str]:
        """Classify an error based on response or exception."""
        
        if exception:
            if isinstance(exception, TimeoutException):
                return ErrorType.TIMEOUT, f"Request timeout: {str(exception)}"
            elif isinstance(exception, RequestError):
                return ErrorType.NETWORK, f"Network error: {str(exception)}"
            else:
                return ErrorType.UNKNOWN, f"Unknown error: {str(exception)}"
        
        if response is None:
            return ErrorType.UNKNOWN, "No response received"
        
        status_code = response.status_code
        
        # Authentication errors
        if status_code == 401:
            return ErrorType.AUTHENTICATION, "Invalid API key or unauthorized access"
        elif status_code == 403:
            return ErrorType.AUTHENTICATION, "Forbidden - insufficient permissions"
        
        # Rate limiting
        elif status_code == 429:
            return ErrorType.RATE_LIMIT, "Rate limit exceeded"
        
        # Server errors
        elif 500 <= status_code < 600:
            return ErrorType.SERVER_ERROR, f"Server error: HTTP {status_code}"
        
        # Context length errors - check response content
        elif status_code == 400:
            try:
                content = response.json()
                error_message = content.get('error', {}).get('message', '').lower()
                if any(keyword in error_message for keyword in ['context', 'length', 'token', 'maximum']):
                    return ErrorType.CONTEXT_LENGTH_EXCEEDED, f"Context length exceeded: {error_message}"
                else:
                    return ErrorType.UNKNOWN, f"Bad request: {error_message}"
            except (json.JSONDecodeError, AttributeError):
                return ErrorType.UNKNOWN, f"Bad request: HTTP {status_code}"
        
        else:
            # Sanitize response text to prevent information leakage
            sanitized_text = self._sanitize_response_text(response.text)
            return ErrorType.UNKNOWN, f"HTTP {status_code}: {sanitized_text}"

    @staticmethod
    def _sanitize_response_text(text: str, max_length: int = 100) -> str:
        """Sanitize response text to prevent information leakage."""
        if not text:
            return ""
        
        # Remove potential sensitive information patterns
        text = re.sub(r'api[_-]?key[s]?\s*[=:]?\s*[a-zA-Z0-9\-_.]+', '[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'password[s]?\s*[:=]\s*[^\s]+', '[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'Bearer\s+[a-zA-Z0-9\-_.]+', '[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'token[s]?\s*[:=]\s*[^\s]+', '[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'secret[s]?\s*[:=]\s*[^\s]+', '[REDACTED]', text, flags=re.IGNORECASE)
        
        # Remove stack traces and detailed error messages
        text = re.sub(r'Traceback.*?\n', '[STACK_TRACE_REMOVED]', text, flags=re.DOTALL)
        text = re.sub(r'File ".*?", line \d+', '[FILE_PATH_REMOVED]', text)
        
        # Limit length and truncate at word boundaries
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
        
        return text.strip()


class HTTPClient:
    """
    Async HTTP client for OpenAI-compatible API endpoints with comprehensive
    error handling, retry logic, and rate limiting.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        rate_limit: float = 1.0,
        max_connections: int = 10,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the HTTP client.
        
        Args:
            base_url: Base URL of the API endpoint
            api_key: API key for authentication (can be None if using environment variable)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit: Maximum requests per second
            max_connections: Maximum concurrent connections
            model: Model name to use for health check and requests
        """
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        # Validate API key format (basic validation)
        if not isinstance(api_key, str) or len(api_key) < 10:
            raise ValueError("Invalid API key format")
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model
        
        # Initialize components
        self.rate_limiter = RateLimiter(rate_limit)
        self.retry_handler = RetryHandler(max_retries)
        self.error_classifier = ErrorClassifier()
        self.logger = logging.getLogger(__name__)
        
        # Configure httpx client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=5),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "effective-context-length/1.0.0"
            }
        )
        
        # Add redaction for logging
        self._setup_secure_logging()
    
    def _setup_secure_logging(self):
        """Setup secure logging with API key redaction."""
        # Create a custom filter to redact sensitive information
        class SensitiveDataFilter(logging.Filter):
            def __init__(self, api_key):
                super().__init__()
                self.api_key = api_key
                self.api_key_pattern = re.compile(re.escape(api_key))
            
            def filter(self, record):
                if hasattr(record, 'msg'):
                    record.msg = self.api_key_pattern.sub('[API_KEY_REDACTED]', str(record.msg))
                return True
        
        # Add filter to logger
        sensitive_filter = SensitiveDataFilter(self.api_key)
        self.logger.addFilter(sensitive_filter)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client and clean up resources."""
        await self.client.aclose()
    
    async def make_chat_completion_request(
        self,
        payload: Dict[str, Any],
        tokens_sent: Optional[int] = None
    ) -> APIResponse:
        """
        Make a chat completion request to the API with full error handling and retry logic.
        
        Args:
            payload: The request payload (messages, model, etc.)
            tokens_sent: Number of tokens in the request (for tracking)
            
        Returns:
            APIResponse object with success status and relevant data
        """
        url = f"{self.base_url}/chat/completions"
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                # Record start time
                start_time = time.time()
                
                # Make the request
                self.logger.debug(f"Making request attempt {attempt + 1} to {url}")
                response = await self.client.post(url, json=payload)
                
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Handle successful response
                if response.status_code == 200:
                    try:
                        content = response.json()
                        tokens_received = self._extract_token_count(content)
                        
                        self.logger.debug(f"Successful request: {response.status_code}, "
                                        f"response_time: {response_time_ms:.2f}ms")
                        
                        return APIResponse(
                            success=True,
                            status_code=response.status_code,
                            response_time_ms=response_time_ms,
                            content=content,
                            tokens_sent=tokens_sent,
                            tokens_received=tokens_received
                        )
                    except json.JSONDecodeError as e:
                        error_type, error_message = ErrorType.UNKNOWN, f"Invalid JSON response: {str(e)}"
                        return APIResponse(
                            success=False,
                            status_code=response.status_code,
                            response_time_ms=response_time_ms,
                            error_type=error_type,
                            error_message=error_message,
                            tokens_sent=tokens_sent
                        )
                
                # Handle error response
                else:
                    error_type, error_message = self.error_classifier.classify_error(response=response)
                    
                    self.logger.warning(f"Request failed: {response.status_code}, "
                                      f"error_type: {error_type.value}, "
                                      f"message: {error_message}")
                    
                    # Check if we should retry
                    if self.retry_handler.should_retry(error_type, attempt):
                        backoff_delay = self.retry_handler.get_backoff_delay(attempt)
                        self.logger.info(f"Retrying in {backoff_delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(backoff_delay)
                        continue
                    
                    return APIResponse(
                        success=False,
                        status_code=response.status_code,
                        response_time_ms=response_time_ms,
                        error_type=error_type,
                        error_message=error_message,
                        tokens_sent=tokens_sent
                    )
            
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
                error_type, error_message = self.error_classifier.classify_error(exception=e)
                
                self.logger.error(f"Request exception: {error_type.value}, message: {error_message}")
                
                # Check if we should retry
                if self.retry_handler.should_retry(error_type, attempt):
                    backoff_delay = self.retry_handler.get_backoff_delay(attempt)
                    self.logger.info(f"Retrying in {backoff_delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(backoff_delay)
                    continue
                
                return APIResponse(
                    success=False,
                    response_time_ms=response_time_ms,
                    error_type=error_type,
                    error_message=error_message,
                    tokens_sent=tokens_sent
                )
        
        # This should never be reached, but just in case
        return APIResponse(
            success=False,
            error_type=ErrorType.UNKNOWN,
            error_message="Maximum retries exceeded",
            tokens_sent=tokens_sent
        )
    
    def _extract_token_count(self, response_content: Dict[str, Any]) -> Optional[int]:
        """Extract token count from API response."""
        try:
            usage = response_content.get('usage', {})
            return usage.get('completion_tokens', 0)
        except (KeyError, TypeError):
            return None
    
    async def health_check(self) -> bool:
        """
        Perform a simple health check against the API endpoint.
        
        Returns:
            True if the endpoint is accessible, False otherwise
        """
        try:
            # Simple test payload using the configured model
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1
            }
            
            response = await self.make_chat_completion_request(test_payload)
            return response.success or response.error_type == ErrorType.CONTEXT_LENGTH_EXCEEDED
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics and configuration."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "rate_limit": self.rate_limiter.rate,
            "current_tokens": self.rate_limiter.tokens
        }