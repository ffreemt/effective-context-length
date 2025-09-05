"""
Integration tests for the effective context length testing tool.

These tests demonstrate the basic functionality and integration between components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effective_context_length.http_client import HTTPClient, ErrorType
from effective_context_length.context_generator import ContextGenerator, ContextStrategy
from effective_context_length.utils import setup_logging, Timer


class TestBasicIntegration:
    """Test basic integration between HTTP client and context generator."""
    
    @pytest.fixture
    def mock_successful_client(self):
        """Create a mock HTTP client that always returns successful responses."""
        with patch('effective_context_length.http_client.httpx.AsyncClient') as mock_httpx:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "This is a test response."}}],
                "usage": {"completion_tokens": 15, "prompt_tokens": 100, "total_tokens": 115}
            }
            
            mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
            
            client = HTTPClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                timeout=30,
                max_retries=2,
                rate_limit=10.0  # High rate limit for testing
            )
            
            yield client
    
    @pytest.fixture
    def context_generator(self):
        """Create a context generator for testing."""
        with patch('effective_context_length.context_generator.tiktoken') as mock_tiktoken:
            # Mock tiktoken to return predictable token counts
            mock_encoding = Mock()
            mock_encoding.encode.side_effect = lambda text: list(range(len(text) // 4 + 1))
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            return ContextGenerator("gpt-3.5-turbo", temperature=0.1)
    
    @pytest.mark.asyncio
    async def test_basic_workflow(self, mock_successful_client, context_generator):
        """Test the basic workflow of generating context and making requests."""
        # Generate test payload
        target_tokens = 1000
        payload = context_generator.generate_test_payload(
            target_tokens, 
            ContextStrategy.PADDING, 
            max_response_tokens=50
        )
        
        # Verify payload structure
        assert isinstance(payload, dict)
        assert "model" in payload
        assert "messages" in payload
        assert len(payload["messages"]) >= 2
        
        # Get token count for the payload
        token_count = context_generator.get_payload_token_count(payload)
        assert token_count.total_tokens > 0
        
        # Make API request
        response = await mock_successful_client.make_chat_completion_request(
            payload, 
            tokens_sent=token_count.total_tokens
        )
        
        # Verify successful response
        assert response.success is True
        assert response.status_code == 200
        assert response.tokens_sent == token_count.total_tokens
        assert response.tokens_received == 15
        assert response.error_type is None
    
    @pytest.mark.asyncio
    async def test_multiple_context_sizes(self, mock_successful_client, context_generator):
        """Test generating and testing multiple context sizes."""
        context_sizes = [500, 1000, 2000, 5000]
        results = []
        
        for size in context_sizes:
            # Generate payload
            payload = context_generator.generate_test_payload(size, ContextStrategy.REPETITION)
            token_count = context_generator.get_payload_token_count(payload)
            
            # Make request
            response = await mock_successful_client.make_chat_completion_request(
                payload, 
                tokens_sent=token_count.total_tokens
            )
            
            results.append({
                "target_size": size,
                "actual_tokens": token_count.total_tokens,
                "success": response.success,
                "response_time": response.response_time_ms
            })
        
        # Verify all requests succeeded
        assert len(results) == len(context_sizes)
        for result in results:
            assert result["success"] is True
            assert result["actual_tokens"] > 0
            assert result["response_time"] >= 0
    
    @pytest.mark.asyncio
    async def test_different_strategies(self, mock_successful_client, context_generator):
        """Test different context generation strategies."""
        strategies = [
            ContextStrategy.PADDING,
            ContextStrategy.REPETITION,
            ContextStrategy.RANDOM_TEXT,
            ContextStrategy.STRUCTURED,
            ContextStrategy.MIXED
        ]
        
        target_tokens = 1500
        results = {}
        
        for strategy in strategies:
            # Generate payload with specific strategy
            payload = context_generator.generate_test_payload(target_tokens, strategy)
            token_count = context_generator.get_payload_token_count(payload)
            
            # Make request
            response = await mock_successful_client.make_chat_completion_request(payload)
            
            results[strategy.value] = {
                "payload_tokens": token_count.total_tokens,
                "success": response.success,
                "content_length": len(payload["messages"][-1]["content"])
            }
        
        # Verify all strategies worked
        for strategy_name, result in results.items():
            assert result["success"] is True, f"Strategy {strategy_name} failed"
            assert result["payload_tokens"] > 0
            assert result["content_length"] > 0
        
        # Verify different strategies produce different content lengths
        content_lengths = [result["content_length"] for result in results.values()]
        assert len(set(content_lengths)) > 1, "All strategies produced identical content lengths"


class TestErrorHandlingIntegration:
    """Test error handling integration between components."""
    
    @pytest.fixture
    def mock_failing_client(self):
        """Create a mock HTTP client that simulates various failures."""
        with patch('effective_context_length.http_client.httpx.AsyncClient') as mock_httpx:
            client = HTTPClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                timeout=30,
                max_retries=1,  # Low retry count for faster tests
                rate_limit=10.0
            )
            
            yield client, mock_httpx
    
    @pytest.mark.asyncio
    async def test_context_length_exceeded_handling(self, mock_failing_client, context_generator):
        """Test handling of context length exceeded errors."""
        client, mock_httpx = mock_failing_client
        
        # Mock context length exceeded response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Maximum context length exceeded"}
        }
        mock_httpx.return_value.post = AsyncMock(return_value=mock_response)
        
        # Generate a large payload
        payload = context_generator.generate_test_payload(10000, ContextStrategy.PADDING)
        
        # Make request
        response = await client.make_chat_completion_request(payload)
        
        # Verify error handling
        assert response.success is False
        assert response.error_type == ErrorType.CONTEXT_LENGTH_EXCEEDED
        assert "context length exceeded" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_failing_client, context_generator):
        """Test handling of rate limit errors with retry."""
        client, mock_httpx = mock_failing_client
        
        # Mock rate limit response, then success
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}],
            "usage": {"completion_tokens": 10}
        }
        
        mock_httpx.return_value.post = AsyncMock(side_effect=[rate_limit_response, success_response])
        
        payload = context_generator.generate_test_payload(500, ContextStrategy.PADDING)
        
        # Speed up the test by mocking sleep
        with patch('asyncio.sleep', new_callable=AsyncMock):
            response = await client.make_chat_completion_request(payload)
        
        # Should succeed after retry
        assert response.success is True
        assert response.status_code == 200
        assert mock_httpx.return_value.post.call_count == 2


class TestUtilitiesIntegration:
    """Test integration with utility functions."""
    
    def test_logging_setup(self):
        """Test logging setup utility."""
        # Test different verbosity levels
        logger = setup_logging(verbose_level=0, quiet=False)
        assert logger is not None
        
        logger = setup_logging(verbose_level=2, quiet=False)
        assert logger is not None
        
        logger = setup_logging(verbose_level=0, quiet=True)
        assert logger is not None
    
    def test_timer_utility(self):
        """Test timer utility for performance measurement."""
        import time
        
        with Timer("Test operation") as timer:
            time.sleep(0.1)  # Simulate work
        
        assert timer.duration is not None
        assert timer.duration >= 0.1
        assert timer.duration < 0.2  # Should be close to 0.1 seconds
    
    @pytest.mark.asyncio
    async def test_performance_measurement(self, context_generator):
        """Test performance measurement during context generation."""
        with Timer("Context generation") as timer:
            # Generate multiple payloads
            payloads = context_generator.generate_test_series(
                [100, 500, 1000], 
                ContextStrategy.PADDING
            )
        
        assert len(payloads) == 3
        assert timer.duration is not None
        assert timer.duration > 0


class TestEndToEndScenario:
    """Test end-to-end scenarios that simulate real usage."""
    
    @pytest.mark.asyncio
    async def test_simulated_context_length_discovery(self):
        """Simulate discovering the effective context length of an endpoint."""
        # Mock an endpoint that fails at 8000 tokens but succeeds below that
        with patch('effective_context_length.http_client.httpx.AsyncClient') as mock_httpx:
            def mock_response_based_on_tokens(payload):
                # Estimate tokens from content length (rough approximation)
                content = payload.get("messages", [{}])[-1].get("content", "")
                estimated_tokens = len(content) // 4
                
                if estimated_tokens > 2000:  # Fail above 2000 tokens
                    response = Mock()
                    response.status_code = 400
                    response.json.return_value = {
                        "error": {"message": "Maximum context length exceeded"}
                    }
                    return response
                else:
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}],
                        "usage": {"completion_tokens": 10}
                    }
                    return response
            
            mock_httpx.return_value.post = AsyncMock(side_effect=lambda url, json: mock_response_based_on_tokens(json))
            
            # Create client and generator
            client = HTTPClient("https://api.example.com/v1", "test-key", rate_limit=10.0)
            generator = ContextGenerator("gpt-3.5-turbo")
            
            # Test different context sizes to find the limit
            test_sizes = [500, 1000, 1500, 2000, 2500, 3000]
            results = []
            
            for size in test_sizes:
                payload = generator.generate_test_payload(size, ContextStrategy.PADDING)
                response = await client.make_chat_completion_request(payload)
                
                results.append({
                    "size": size,
                    "success": response.success,
                    "error_type": response.error_type
                })
            
            # Analyze results to find the effective limit
            successful_sizes = [r["size"] for r in results if r["success"]]
            failed_sizes = [r["size"] for r in results if not r["success"]]
            
            # Should have some successes and some failures
            assert len(successful_sizes) > 0, "No requests succeeded"
            assert len(failed_sizes) > 0, "No requests failed"
            
            # The boundary should be around 2000 tokens
            max_successful = max(successful_sizes) if successful_sizes else 0
            min_failed = min(failed_sizes) if failed_sizes else float('inf')
            
            assert max_successful < min_failed, "Results don't show a clear boundary"
            assert max_successful >= 1500, "Effective limit too low"
            assert min_failed <= 2500, "Effective limit too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])