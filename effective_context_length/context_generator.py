"""
Context Generator for Effective Context Length Testing

This module provides context generation and token counting functionality
for testing different context lengths with OpenAI-compatible LLM endpoints.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random
import string

try:
    import tiktoken
except ImportError:
    tiktoken = None


class TokenizerType(Enum):
    """Supported tokenizer types."""
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4O = "gpt-4o"
    CLAUDE = "claude"  # Fallback estimation
    DEFAULT = "cl100k_base"  # Default encoding


class InputValidator:
    """Validates and sanitizes input parameters."""
    
    @staticmethod
    def validate_target_tokens(target_tokens: int) -> int:
        """Validate target token count."""
        if not isinstance(target_tokens, int):
            raise ValueError("Target tokens must be an integer")
        
        if target_tokens <= 0:
            raise ValueError("Target tokens must be positive")
        
        if target_tokens > 1000000:  # 1M token limit
            raise ValueError("Target tokens cannot exceed 1,000,000")
        
        return target_tokens
    
    @staticmethod
    def validate_model_name(model: str) -> str:
        """Validate model name."""
        if not isinstance(model, str):
            raise ValueError("Model name must be a string")
        
        model = model.strip()
        
        if not model:
            raise ValueError("Model name cannot be empty")
        
        # Allow only alphanumeric characters, hyphens, and dots
        if not re.match(r'^[a-zA-Z0-9\-._]+$', model):
            raise ValueError("Model name contains invalid characters")
        
        return model
    
    @staticmethod
    def validate_temperature(temperature: float) -> float:
        """Validate temperature parameter."""
        if not isinstance(temperature, (int, float)):
            raise ValueError("Temperature must be a number")
        
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        return float(temperature)
    
    @staticmethod
    def sanitize_text_content(text: str) -> str:
        """Sanitize text content to prevent injection attacks."""
        if not isinstance(text, str):
            return ""
        
        # Remove potential malicious content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'data:', '[REDACTED]', text, flags=re.IGNORECASE)
        
        # Control character removal
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Length limiting
        if len(text) > 100000:  # 100KB limit
            text = text[:100000]
        
        return text.strip()


class ContextStrategy(Enum):
    """Different strategies for generating test context."""
    PADDING = "padding"  # Repeat simple text patterns
    REPETITION = "repetition"  # Repeat meaningful content
    RANDOM_TEXT = "random_text"  # Generate random text
    STRUCTURED = "structured"  # Generate structured data
    MIXED = "mixed"  # Mix of different content types


@dataclass
class TokenCount:
    """Token count information for a message or payload."""
    total_tokens: int
    message_tokens: int
    system_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    overhead_tokens: int = 0  # Tokens used for message formatting


class TokenCounter:
    """Accurate token counting using tiktoken library."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter for the specified model.
        
        Args:
            model: Model name to determine the appropriate tokenizer
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer
        if tiktoken is None:
            self.logger.warning("tiktoken not available, using fallback estimation")
            self.encoding = None
        else:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.logger.warning(f"Unknown model {model}, using default encoding")
                self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if self.encoding is None:
            # Improved fallback estimation with better accuracy
            # Based on empirical analysis of English text:
            # - Average word length: 4.7 characters
            # - Average tokens per word: 0.75 (for GPT models)
            # - Therefore: ~6.3 characters per token
            words = text.split()
            if not words:
                return 0
            
            # Use a weighted approach for better accuracy
            char_count = len(text)
            word_count = len(words)
            
            # Estimate based on both character and word count
            char_estimate = char_count / 6.3
            word_estimate = word_count * 0.75
            
            # Weighted average (60% chars, 40% words)
            estimated_tokens = (char_estimate * 0.6 + word_estimate * 0.4)
            
            return max(1, int(estimated_tokens))
        
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> TokenCount:
        """
        Count tokens in a list of messages, accounting for OpenAI's message formatting.
        
        Based on OpenAI's token counting guidelines:
        https://github.com/openai/openai-python/blob/main/chatml.md
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            TokenCount object with detailed token breakdown
        """
        if self.encoding is None:
            # Fallback estimation
            total_text = " ".join(msg.get("content", "") for msg in messages)
            estimated_tokens = self.count_tokens(total_text)
            return TokenCount(
                total_tokens=estimated_tokens + len(messages) * 4,  # Add overhead
                message_tokens=estimated_tokens
            )
        
        # Tokens for message formatting (varies by model)
        tokens_per_message = 3  # Default for most models
        tokens_per_name = 1
        
        # Model-specific adjustments
        if "gpt-3.5-turbo" in self.model:
            tokens_per_message = 4
            tokens_per_name = -1  # If there's a name, the role is omitted
        elif "gpt-4" in self.model:
            tokens_per_message = 3
            tokens_per_name = 1
        
        total_tokens = 0
        system_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        overhead_tokens = 0
        
        for message in messages:
            message_tokens = tokens_per_message
            role = message.get("role", "")
            content = message.get("content", "")
            name = message.get("name", "")
            
            # Count content tokens
            content_tokens = self.count_tokens(content)
            
            # Count role tokens
            role_tokens = self.count_tokens(role)
            message_tokens += role_tokens + content_tokens
            
            # Count name tokens if present
            if name:
                message_tokens += tokens_per_name + self.count_tokens(name)
            
            total_tokens += message_tokens
            overhead_tokens += tokens_per_message + role_tokens
            
            # Track tokens by role
            if role == "system":
                system_tokens += content_tokens
            elif role == "user":
                user_tokens += content_tokens
            elif role == "assistant":
                assistant_tokens += content_tokens
        
        # Add final tokens for assistant response
        total_tokens += 3
        overhead_tokens += 3
        
        return TokenCount(
            total_tokens=total_tokens,
            message_tokens=total_tokens - overhead_tokens,
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens,
            overhead_tokens=overhead_tokens
        )
    
    def estimate_response_tokens(self, max_tokens: int) -> int:
        """
        Estimate tokens that will be used for the response.
        
        Args:
            max_tokens: Maximum tokens requested for response
            
        Returns:
            Estimated response tokens (usually much less than max_tokens)
        """
        # Most responses are much shorter than max_tokens
        # Use a conservative estimate
        return min(max_tokens, 100)


class ContextGenerator:
    """
    Generates test contexts of specific token lengths for testing LLM endpoints.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize context generator.
        
        Args:
            model: Model name for token counting
            temperature: Temperature for API requests
        """
        # Validate inputs
        self.model = InputValidator.validate_model_name(model)
        self.temperature = InputValidator.validate_temperature(temperature)
        self.token_counter = TokenCounter(self.model)
        self.logger = logging.getLogger(__name__)
        self.validator = InputValidator()
        
        # Base messages that will be used in all requests
        self.base_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please provide a brief response to the user's request."
            },
            {
                "role": "user",
                "content": "Please analyze the following text and provide a brief summary: "
            }
        ]
    
    def generate_filler_text(self, target_tokens: int, strategy: ContextStrategy = ContextStrategy.PADDING) -> str:
        """
        Generate filler text with approximately the target number of tokens.
        
        Args:
            target_tokens: Target number of tokens to generate
            strategy: Strategy to use for text generation
            
        Returns:
            Generated text with approximately target_tokens tokens
        """
        # Validate input
        target_tokens = self.validator.validate_target_tokens(target_tokens)
        
        if target_tokens <= 0:
            return ""
        
        # Generate text based on strategy
        if strategy == ContextStrategy.PADDING:
            text = self._generate_padding_text(target_tokens)
        elif strategy == ContextStrategy.REPETITION:
            text = self._generate_repetition_text(target_tokens)
        elif strategy == ContextStrategy.RANDOM_TEXT:
            text = self._generate_random_text(target_tokens)
        elif strategy == ContextStrategy.STRUCTURED:
            text = self._generate_structured_text(target_tokens)
        elif strategy == ContextStrategy.MIXED:
            text = self._generate_mixed_text(target_tokens)
        else:
            text = self._generate_padding_text(target_tokens)
        
        # Sanitize output
        return self.validator.sanitize_text_content(text)
    
    def _generate_padding_text(self, target_tokens: int) -> str:
        """Generate simple padding text by repeating a pattern."""
        # Simple pattern that's roughly 1 token per 4 characters
        pattern = "This is test content for context length testing. "
        pattern_tokens = self.token_counter.count_tokens(pattern)
        
        if pattern_tokens == 0:
            pattern_tokens = 1
        
        repetitions = max(1, target_tokens // pattern_tokens)
        remainder_tokens = target_tokens % pattern_tokens
        
        text = pattern * repetitions
        
        # Add partial pattern for remainder
        if remainder_tokens > 0:
            partial_chars = min(len(pattern), remainder_tokens * 4)
            text += pattern[:partial_chars]
        
        return text
    
    def _generate_repetition_text(self, target_tokens: int) -> str:
        """Generate text by repeating meaningful content."""
        base_content = """
        The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
        In the realm of artificial intelligence, large language models have revolutionized how we interact with technology.
        These models can understand context, generate human-like text, and perform complex reasoning tasks.
        However, they also have limitations, including context length constraints that affect their performance.
        """
        
        base_tokens = self.token_counter.count_tokens(base_content)
        if base_tokens == 0:
            base_tokens = 1
        
        repetitions = max(1, target_tokens // base_tokens)
        remainder_tokens = target_tokens % base_tokens
        
        text = base_content * repetitions
        
        # Add partial content for remainder
        if remainder_tokens > 0:
            words = base_content.split()
            partial_text = ""
            for word in words:
                test_text = partial_text + " " + word if partial_text else word
                if self.token_counter.count_tokens(test_text) > remainder_tokens:
                    break
                partial_text = test_text
            text += " " + partial_text
        
        return text.strip()
    
    def _generate_random_text(self, target_tokens: int) -> str:
        """Generate random text with approximately target tokens."""
        # Generate random words of varying lengths
        words = []
        current_tokens = 0
        
        while current_tokens < target_tokens:
            # Generate random word (3-12 characters)
            word_length = random.randint(3, 12)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            
            test_text = " ".join(words + [word])
            test_tokens = self.token_counter.count_tokens(test_text)
            
            if test_tokens > target_tokens:
                break
            
            words.append(word)
            current_tokens = test_tokens
        
        return " ".join(words)
    
    def _generate_structured_text(self, target_tokens: int) -> str:
        """Generate structured text (JSON-like) with target tokens."""
        entries = []
        current_tokens = 0
        entry_id = 1
        
        while current_tokens < target_tokens:
            entry = {
                "id": entry_id,
                "name": f"Item_{entry_id}",
                "description": f"This is a test item number {entry_id} for context length testing.",
                "value": random.randint(1, 1000),
                "active": random.choice([True, False])
            }
            
            entry_text = str(entry)
            test_text = "\n".join([str(e) for e in entries + [entry]])
            test_tokens = self.token_counter.count_tokens(test_text)
            
            if test_tokens > target_tokens:
                break
            
            entries.append(entry)
            current_tokens = test_tokens
            entry_id += 1
        
        return "\n".join(str(entry) for entry in entries)
    
    def _generate_mixed_text(self, target_tokens: int) -> str:
        """Generate mixed content using different strategies."""
        # Divide tokens among different strategies
        strategies = [
            ContextStrategy.PADDING,
            ContextStrategy.REPETITION,
            ContextStrategy.RANDOM_TEXT,
            ContextStrategy.STRUCTURED
        ]
        
        tokens_per_strategy = target_tokens // len(strategies)
        remainder = target_tokens % len(strategies)
        
        parts = []
        for i, strategy in enumerate(strategies):
            strategy_tokens = tokens_per_strategy
            if i < remainder:
                strategy_tokens += 1
            
            if strategy_tokens > 0:
                part = self.generate_filler_text(strategy_tokens, strategy)
                parts.append(part)
        
        return "\n\n".join(parts)
    
    def generate_test_payload(
        self,
        target_tokens: int,
        strategy: ContextStrategy = ContextStrategy.PADDING,
        max_response_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Generate a complete test payload with approximately target_tokens.
        
        Args:
            target_tokens: Target total tokens for the request
            strategy: Strategy for generating filler content
            max_response_tokens: Maximum tokens for the response
            
        Returns:
            Complete API request payload
        """
        # Calculate base tokens from system and user messages
        base_token_count = self.token_counter.count_message_tokens(self.base_messages)
        base_tokens = base_token_count.total_tokens
        
        # Reserve tokens for response
        available_tokens = target_tokens - base_tokens - max_response_tokens
        
        if available_tokens <= 0:
            self.logger.warning(f"Target tokens {target_tokens} too small for base messages and response")
            available_tokens = 10  # Minimum filler
        
        # Generate filler content
        filler_text = self.generate_filler_text(available_tokens, strategy)
        
        # Create messages with filler content
        messages = self.base_messages.copy()
        messages[-1]["content"] += filler_text
        
        # Verify actual token count
        actual_token_count = self.token_counter.count_message_tokens(messages)
        
        self.logger.debug(f"Generated payload: target={target_tokens}, "
                         f"actual={actual_token_count.total_tokens}, "
                         f"strategy={strategy.value}")
        
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_response_tokens,
            "temperature": self.temperature,
            "stream": False
        }
    
    def get_payload_token_count(self, payload: Dict[str, Any]) -> TokenCount:
        """
        Get accurate token count for a payload.
        
        Args:
            payload: API request payload
            
        Returns:
            TokenCount object with detailed breakdown
        """
        messages = payload.get("messages", [])
        return self.token_counter.count_message_tokens(messages)
    
    def generate_test_series(
        self,
        token_sizes: List[int],
        strategy: ContextStrategy = ContextStrategy.PADDING,
        max_response_tokens: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate a series of test payloads for different token sizes.
        
        Args:
            token_sizes: List of target token sizes
            max_response_tokens: Maximum tokens for responses
            
        Returns:
            List of API request payloads
        """
        payloads = []
        for size in token_sizes:
            payload = self.generate_test_payload(size, strategy, max_response_tokens)
            payloads.append(payload)
        
        return payloads
    
    def validate_token_count(self, payload: Dict[str, Any], expected_tokens: int, tolerance: float = 0.1) -> bool:
        """
        Validate that a payload has approximately the expected token count.
        
        Args:
            payload: API request payload
            expected_tokens: Expected token count
            tolerance: Acceptable deviation as a fraction (0.1 = 10%)
            
        Returns:
            True if token count is within tolerance
        """
        actual_count = self.get_payload_token_count(payload)
        actual_tokens = actual_count.total_tokens
        
        min_tokens = expected_tokens * (1 - tolerance)
        max_tokens = expected_tokens * (1 + tolerance)
        
        return min_tokens <= actual_tokens <= max_tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics and configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "tokenizer_available": self.token_counter.encoding is not None,
            "base_message_tokens": self.token_counter.count_message_tokens(self.base_messages).total_tokens
        }