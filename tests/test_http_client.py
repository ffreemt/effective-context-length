"""
Unit tests for the HTTP client functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import httpx
from effective_context_length.http_client import (
    HTTPClient, 
    ErrorType, 
    APIResponse, 
    RateLimiter, 
    RetryHandler, 
    ErrorClassifier
)


class TestErrorClassifier:
    """Test the error classification functionality."""
    
    def test_classify_timeout_exception(self):
        """Test classification of timeout exceptions."""
        exception = httpx.TimeoutException("Request timeout")
        error_type, message = ErrorClassifier.classify_error(exception=exception)
        
        assert error_type == ErrorType.TIMEOUT
        assert "timeout" in message.lower()
    
    def test_classify_network_exception(self):
        """Test classification of network exceptions."""
        exception = httpx.RequestError("Connection failed")
        error_type, message = ErrorClassifier.classify_error(exception=exception)
        
        assert error_type == ErrorType.NETWORK
        assert "network error" in message.lower()
    
    def test_classify_401_response(self):
        """Test classification of 401 unauthorized response."""
        response = Mock()
        response.status_code = 401
        
        error_type, message = ErrorClassifier.classify_error(response=response)
        
        assert error_type == ErrorType.AUTHENTICATION
        assert "unauthorized" in message.lower()
    
    def test_classify_429_response(self):
        """Test classification of 429 rate limit response."""
        response = Mock()
        response.status_code = 429
        
        error_type, message = ErrorClassifier.classify_error(response=response)
        
        assert error_type == ErrorType.RATE_LIMIT
        assert "rate limit" in message.lower()
    
    def test_classify_500_response(self):
        """Test classification of 500 server error response."""
        response = Mock()
        response.status_code = 500
        
        error_type, message = ErrorClassifier.classify_error(response=response)
        
        assert error_type == ErrorType.SERVER_ERROR
        assert "server error" in message.lower()
    
    def test_classify_context_length_error(self):
        """Test classification of context length exceeded error."""
        response = Mock()
        response.status_code = 400
        response.json.return_value = {
            "error": {
                "message": "Maximum context length exceeded"
            }
        }
        
        error_type, message = ErrorClassifier.classify_error(response=response)
        
        assert error_type == ErrorType.CONTEXT_LENGTH_EXCEEDED
        assert "context length exceeded" in message.lower()


class TestRateLimiter:
    """Test the rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_immediate_request(self):
        """Test that rate limiter allows immediate first request."""
        limiter = RateLimiter(requests_per_second=1.0)
        
        # First request should be immediate
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should take minimal time (less than 0.1 seconds)
        assert (end_time - start_time) < 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_delay(self):
        """Test that rate limiter enforces delay between requests."""
        limiter = RateLimiter(requests_per_second=2.0)  # 0.5 second intervals
        
        # First request
        await limiter.acquire()
        
        # Second request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should take approximately 0.5 seconds (with some tolerance)
        delay = end_time - start_time
        assert 0.4 < delay < 0.6


class TestRetryHandler:
    """Test the retry handling functionality."""
    
    def test_should_retry_retryable_errors(self):
        """Test that retryable errors are marked for retry."""
        handler = RetryHandler(max_retries=3)
        
        retryable_errors = [
            ErrorType.RATE_LIMIT,
            ErrorType.TIMEOUT,
            ErrorType.SERVER_ERROR,
            ErrorType.NETWORK
        ]
        
        for error_type in retryable_errors:
            assert handler.should_retry(error_type, 0)  # First attempt
            assert handler.should_retry(error_type, 2)  # Within max retries
            assert not handler.should_retry(error_type, 3)  # At max retries
    
    def test_should_not_retry_non_retryable_errors(self):
        """Test that non-retryable errors are not marked for retry."""
        handler = RetryHandler(max_retries=3)
        
        non_retryable_errors = [
            ErrorType.CONTEXT_LENGTH_EXCEEDED,
            ErrorType.AUTHENTICATION,
            ErrorType.UNKNOWN
        ]
        
        for error_type in non_retryable_errors:
            assert not handler.should_retry(error_type, 0)
    
    def test_backoff_delay_calculation(self):
        """Test that backoff delay increases exponentially."""
        handler = RetryHandler(max_retries=3, backoff_factor=2.0)
        
        delay_0 = handler.get_backoff_delay(0)
        delay_1 = handler.get_backoff_delay(1)
        delay_2 = handler.get_backoff_delay(2)
        
        # Each delay should be roughly double the previous (with jitter tolerance)
        assert delay_0 < delay_1 < delay_2
        assert delay_1 > delay_0 * 1.5  # Account for jitter
        assert delay_2 > delay_1 * 1.5  # Account for jitter


class TestHTTPClient:
    """Test the HTTP client functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock HTTP client for testing."""
        with patch('effective_context_length.http_client.httpx.AsyncClient') as mock:
            client = HTTPClient(
                base_url="https://api.example.com/v1",
                api_key="test-key-long-enough",
                timeout=30,
                max_retries=2,
                rate_limit=1.0
            )
            yield client, mock
    
    @pytest.mark.asyncio
    async def test_successful_request(self, mock_client):
        """Test successful API request."""
        client, mock_httpx = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"completion_tokens": 10}
        }
        
        mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = await client.make_chat_completion_request(payload, tokens_sent=100)
        
        assert response.success is True
        assert response.status_code == 200
        assert response.tokens_sent == 100
        assert response.tokens_received == 10
        assert response.error_type is None
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_with_retry(self, mock_client):
        """Test rate limit error handling with retry."""
        client, mock_httpx = mock_client
        
        # Mock rate limit response, then success
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}],
            "usage": {"completion_tokens": 5}
        }
        
        mock_httpx.return_value.post = AsyncMock(side_effect=[rate_limit_response, success_response])
        
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}
        
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
            response = await client.make_chat_completion_request(payload)
        
        assert response.success is True
        assert response.status_code == 200
        assert mock_httpx.return_value.post.call_count == 2  # Initial + 1 retry
    
    @pytest.mark.asyncio
    async def test_context_length_exceeded_no_retry(self, mock_client):
        """Test that context length exceeded errors are not retried."""
        client, mock_httpx = mock_client
        
        # Mock context length exceeded response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Maximum context length exceeded"}
        }
        
        mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
        
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}
        
        response = await client.make_chat_completion_request(payload)
        
        assert response.success is False
        assert response.error_type == ErrorType.CONTEXT_LENGTH_EXCEEDED
        assert mock_httpx.return_value.post.call_count == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, mock_client):
        """Test authentication error handling."""
        client, mock_httpx = mock_client
        
        # Mock authentication error response
        mock_response = Mock()
        mock_response.status_code = 401
        
        mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
        
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}
        
        response = await client.make_chat_completion_request(payload)
        
        assert response.success is False
        assert response.error_type == ErrorType.AUTHENTICATION
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_network_exception_with_retry(self, mock_client):
        """Test network exception handling with retry."""
        client, mock_httpx = mock_client
        
        # Mock network exception, then success
        network_exception = httpx.RequestError("Connection failed")
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success after network retry"}}],
            "usage": {"completion_tokens": 8}
        }
        
        mock_httpx.return_value.post = AsyncMock(side_effect=[network_exception, success_response])
        
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}
        
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
            response = await client.make_chat_completion_request(payload)
        
        assert response.success is True
        assert response.status_code == 200
        assert mock_httpx.return_value.post.call_count == 2  # Initial + 1 retry
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, mock_client):
        """Test behavior when max retries are exceeded."""
        client, mock_httpx = mock_client
        
        # Mock consistent server errors
        server_error_response = Mock()
        server_error_response.status_code = 500
        
        mock_httpx.return_value.post = AsyncMock(return_value=server_error_response)
        
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}
        
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
            response = await client.make_chat_completion_request(payload)
        
        assert response.success is False
        assert response.error_type == ErrorType.SERVER_ERROR
        # Should try initial + max_retries (2) = 3 total attempts
        assert mock_httpx.return_value.post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_client):
        """Test successful health check."""
        client, mock_httpx = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"completion_tokens": 1}
        }
        
        mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
        
        is_healthy = await client.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_context_length_still_healthy(self, mock_client):
        """Test that context length errors still indicate a healthy endpoint."""
        client, mock_httpx = mock_client
        
        # Mock context length exceeded response (endpoint is working, just hit limit)
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Maximum context length exceeded"}
        }
        
        mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
        
        is_healthy = await client.health_check()
        
        assert is_healthy is True  # Context length error means endpoint is working
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_client):
        """Test failed health check."""
        client, mock_httpx = mock_client
        
        # Mock network exception
        network_exception = httpx.RequestError("Connection failed")
        mock_httpx.return_value.post = AsyncMock(side_effect=network_exception)
        
        is_healthy = await client.health_check()
        
        assert is_healthy is False
    
    def test_get_stats(self, mock_client):
        """Test getting client statistics."""
        client, _ = mock_client
        
        stats = client.get_stats()
        
        assert stats["base_url"] == "https://api.example.com/v1"
        assert stats["timeout"] == 30
        assert stats["max_retries"] == 2
        assert stats["rate_limit"] == 1.0
        assert "current_tokens" in stats


if __name__ == "__main__":
    pytest.main([__file__])