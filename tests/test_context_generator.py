"""
Unit tests for the context generator functionality.
"""

import pytest
from unittest.mock import Mock, patch
from effective_context_length.context_generator import (
    ContextGenerator,
    TokenCounter,
    ContextStrategy,
    TokenCount
)


class TestTokenCounter:
    """Test the token counting functionality."""
    
    def test_token_counter_with_tiktoken(self):
        """Test token counter when tiktoken is available."""
        with patch('effective_context_length.context_generator.tiktoken') as mock_tiktoken:
            # Mock tiktoken encoding
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            counter = TokenCounter("gpt-3.5-turbo")
            
            token_count = counter.count_tokens("Hello world")
            assert token_count == 5
            mock_encoding.encode.assert_called_once_with("Hello world")
    
    def test_token_counter_fallback(self):
        """Test token counter fallback when tiktoken is not available."""
        with patch('effective_context_length.context_generator.tiktoken', None):
            counter = TokenCounter("gpt-3.5-turbo")
            
            # Should use improved fallback estimation
            # "Hello world" = 2 words, 11 chars
            # Improved estimate: (11/6.3 * 0.6) + (2 * 0.75 * 0.4) ≈ 1.4 → 1
            token_count = counter.count_tokens("Hello world")  
            assert token_count == 1
            
            # Test with longer text
            long_text = "This is a longer sentence with many words to test the improved fallback estimation method."
            # Should give a more accurate estimate than simple len(text)//4
            token_count = counter.count_tokens(long_text)
            assert token_count > 0
    
    def test_count_message_tokens_with_tiktoken(self):
        """Test message token counting with tiktoken."""
        with patch('effective_context_length.context_generator.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            # Mock different token counts for different strings
            def mock_encode(text):
                if "system" in text:
                    return [1, 2]  # 2 tokens
                elif "user" in text:
                    return [1, 2]  # 2 tokens
                elif "You are a helpful assistant" in text:
                    return [1, 2, 3, 4, 5]  # 5 tokens
                elif "Hello" in text:
                    return [1, 2, 3]  # 3 tokens
                else:
                    return [1]  # 1 token
            
            mock_encoding.encode.side_effect = mock_encode
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            counter = TokenCounter("gpt-3.5-turbo")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]
            
            token_count = counter.count_message_tokens(messages)
            
            assert isinstance(token_count, TokenCount)
            assert token_count.system_tokens == 5
            assert token_count.user_tokens == 3
            assert token_count.total_tokens > 0
    
    def test_count_message_tokens_fallback(self):
        """Test message token counting fallback."""
        with patch('effective_context_length.context_generator.tiktoken', None):
            counter = TokenCounter("gpt-3.5-turbo")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]
            
            token_count = counter.count_message_tokens(messages)
            
            assert isinstance(token_count, TokenCount)
            assert token_count.total_tokens > 0
            assert token_count.message_tokens > 0
    
    def test_estimate_response_tokens(self):
        """Test response token estimation."""
        counter = TokenCounter("gpt-3.5-turbo")
        
        # Should return min of max_tokens and 100
        assert counter.estimate_response_tokens(50) == 50
        assert counter.estimate_response_tokens(150) == 100


class TestContextGenerator:
    """Test the context generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create a context generator for testing."""
        with patch('effective_context_length.context_generator.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4]  # 4 tokens for any text
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            return ContextGenerator("gpt-3.5-turbo", temperature=0.1)
    
    def test_generate_padding_text(self, generator):
        """Test padding text generation."""
        text = generator.generate_filler_text(100, ContextStrategy.PADDING)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "test content" in text.lower()
    
    def test_generate_repetition_text(self, generator):
        """Test repetition text generation."""
        text = generator.generate_filler_text(50, ContextStrategy.REPETITION)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "fox" in text.lower() or "intelligence" in text.lower()
    
    def test_generate_random_text(self, generator):
        """Test random text generation."""
        text = generator.generate_filler_text(30, ContextStrategy.RANDOM_TEXT)
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Random text should contain lowercase letters and spaces
        assert any(c.islower() for c in text)
    
    def test_generate_structured_text(self, generator):
        """Test structured text generation."""
        text = generator.generate_filler_text(40, ContextStrategy.STRUCTURED)
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain dictionary-like structure
        assert "id" in text.lower() or "name" in text.lower()
    
    def test_generate_mixed_text(self, generator):
        """Test mixed text generation."""
        text = generator.generate_filler_text(60, ContextStrategy.MIXED)
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain content from different strategies
        assert len(text.split('\n\n')) > 1  # Multiple sections
    
    def test_generate_test_payload(self, generator):
        """Test complete payload generation."""
        payload = generator.generate_test_payload(1000, ContextStrategy.PADDING, 50)
        
        assert isinstance(payload, dict)
        assert "model" in payload
        assert "messages" in payload
        assert "max_tokens" in payload
        assert "temperature" in payload
        
        assert payload["model"] == "gpt-3.5-turbo"
        assert payload["max_tokens"] == 50
        assert payload["temperature"] == 0.1
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) >= 2  # At least system and user messages
    
    def test_get_payload_token_count(self, generator):
        """Test payload token counting."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ]
        }
        
        token_count = generator.get_payload_token_count(payload)
        
        assert isinstance(token_count, TokenCount)
        assert token_count.total_tokens > 0
    
    def test_generate_test_series(self, generator):
        """Test generating a series of test payloads."""
        token_sizes = [100, 500, 1000]
        payloads = generator.generate_test_series(token_sizes, ContextStrategy.PADDING, 25)
        
        assert len(payloads) == 3
        for payload in payloads:
            assert isinstance(payload, dict)
            assert "messages" in payload
            assert payload["max_tokens"] == 25
    
    def test_validate_token_count(self, generator):
        """Test token count validation."""
        # Create a payload with known token count
        payload = generator.generate_test_payload(100, ContextStrategy.PADDING, 10)
        
        # Should validate within tolerance
        assert generator.validate_token_count(payload, 100, tolerance=0.5)  # 50% tolerance
        
        # Should fail with very strict tolerance
        assert not generator.validate_token_count(payload, 100, tolerance=0.01)  # 1% tolerance
    
    def test_get_stats(self, generator):
        """Test getting generator statistics."""
        stats = generator.get_stats()
        
        assert isinstance(stats, dict)
        assert "model" in stats
        assert "temperature" in stats
        assert "tokenizer_available" in stats
        assert "base_message_tokens" in stats
        
        assert stats["model"] == "gpt-3.5-turbo"
        assert stats["temperature"] == 0.1
        assert isinstance(stats["tokenizer_available"], bool)
        assert isinstance(stats["base_message_tokens"], int)
    
    def test_generate_filler_text_zero_tokens(self, generator):
        """Test generating filler text with zero tokens."""
        text = generator.generate_filler_text(0, ContextStrategy.PADDING)
        assert text == ""
    
    def test_generate_filler_text_negative_tokens(self, generator):
        """Test generating filler text with negative tokens."""
        text = generator.generate_filler_text(-10, ContextStrategy.PADDING)
        assert text == ""
    
    def test_generate_test_payload_insufficient_tokens(self, generator):
        """Test payload generation with insufficient tokens for base messages."""
        # This should handle the case gracefully
        payload = generator.generate_test_payload(10, ContextStrategy.PADDING, 5)
        
        assert isinstance(payload, dict)
        assert "messages" in payload
        # Should still generate a valid payload even with minimal tokens


class TestContextStrategies:
    """Test different context generation strategies."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator with mocked tiktoken."""
        with patch('effective_context_length.context_generator.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            # Mock token counting to return predictable results
            mock_encoding.encode.side_effect = lambda text: list(range(len(text) // 4 + 1))
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            return ContextGenerator("gpt-3.5-turbo")
    
    def test_all_strategies_produce_text(self, generator):
        """Test that all strategies produce non-empty text."""
        strategies = [
            ContextStrategy.PADDING,
            ContextStrategy.REPETITION,
            ContextStrategy.RANDOM_TEXT,
            ContextStrategy.STRUCTURED,
            ContextStrategy.MIXED
        ]
        
        for strategy in strategies:
            text = generator.generate_filler_text(50, strategy)
            assert isinstance(text, str)
            assert len(text) > 0, f"Strategy {strategy} produced empty text"
    
    def test_strategies_produce_different_content(self, generator):
        """Test that different strategies produce different content."""
        target_tokens = 100
        
        padding_text = generator.generate_filler_text(target_tokens, ContextStrategy.PADDING)
        repetition_text = generator.generate_filler_text(target_tokens, ContextStrategy.REPETITION)
        random_text = generator.generate_filler_text(target_tokens, ContextStrategy.RANDOM_TEXT)
        
        # Different strategies should produce different content
        texts = [padding_text, repetition_text, random_text]
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i != j:
                    assert text1 != text2, f"Strategies {i} and {j} produced identical content"


if __name__ == "__main__":
    pytest.main([__file__])