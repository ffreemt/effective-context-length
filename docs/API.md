# API Documentation

## Overview

The Effective Context Length tool provides a comprehensive API for testing and determining the effective context length limits of OpenAI-compatible LLM endpoints. This document covers the main classes, interfaces, and usage patterns.

## Core Classes

### EffectiveContextLengthTester

The main testing orchestrator that manages test sessions and executes testing strategies.

```python
class EffectiveContextLengthTester:
    """
    Main testing class for determining effective context length limits.
    
    This class provides a context manager interface for safe resource management
    and supports multiple testing strategies through a strategy pattern.
    """
    
    async def __aenter__(self) -> 'EffectiveContextLengthTester':
        """Enter async context manager and initialize resources."""
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and cleanup resources."""
        
    async def run_test(self, progress_callback: Optional[Callable] = None) -> TestSession:
        """
        Execute the context length test using the configured strategy.
        
        Args:
            progress_callback: Optional callback for progress reporting
            
        Returns:
            TestSession: Complete test session with all results
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If test execution fails
        """
```

#### Constructor Parameters

- `base_url` (str): Base URL of the OpenAI-compatible API endpoint
- `api_key` (str): API key for authentication  
- `model` (str): Model name to test
- `strategy` (TestStrategy): Testing strategy to use
- `min_tokens` (int): Minimum context length to test (default: 1000)
- `max_tokens` (int): Maximum context length to test (default: 200000)
- `step_size` (int): Step size for gradual strategy (default: 2000)
- `samples_per_size` (int): Number of test samples per context size (default: 5)
- `error_threshold` (float): Error rate threshold for determining effective limit (default: 0.1)
- `timeout` (int): Request timeout in seconds (default: 300)
- `rate_limit` (float): Rate limit in requests per second (default: 1.0)
- `max_retries` (int): Maximum number of retries per request (default: 3)
- `temperature` (float): Temperature for API requests (default: 0.1)
- `min_samples` (int): Minimum samples required for statistical analysis (default: 3)
- `predefined_sizes` (List[int], optional): Predefined context sizes for predefined strategy

### HTTPClient

Handles all HTTP communication with OpenAI-compatible APIs, including retry logic, rate limiting, and error categorization.

```python
class HTTPClient:
    """
    Async HTTP client with retry logic and rate limiting for API requests.
    
    Provides robust error handling and automatic retry for transient failures.
    """
    
    async def make_request(self, messages: List[Dict[str, str]], **kwargs) -> APIResponse:
        """
        Make a request to the OpenAI-compatible API.
        
        Args:
            messages: List of message dictionaries for the conversation
            **kwargs: Additional request parameters
            
        Returns:
            APIResponse: Structured response with success/error information
            
        Raises:
            RequestError: For network-related failures
            AuthenticationError: For authentication failures
        """
        
    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
```

#### Constructor Parameters

- `base_url` (str): Base URL for API requests
- `api_key` (str): API key for authentication
- `model` (str): Model name to use for requests
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum number of retry attempts
- `rate_limit` (float): Maximum requests per second
- `temperature` (float): Temperature parameter for generation

### ContextGenerator

Generates test payloads of varying context lengths with accurate token counting.

```python
class ContextGenerator:
    """
    Generates context payloads for testing different context lengths.
    
    Supports multiple content strategies and provides accurate token counting.
    """
    
    def generate_context(self, target_tokens: int, strategy: ContextStrategy = ContextStrategy.TOKENS) -> Tuple[List[Dict[str, str]], int]:
        """
        Generate context content with specified token count.
        
        Args:
            target_tokens: Target number of tokens to generate
            strategy: Content generation strategy to use
            
        Returns:
            Tuple[List[Dict[str, str]], int]: Generated messages and actual token count
            
        Raises:
            ValueError: If target_tokens is invalid
        """
        
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages using tiktoken.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            int: Total token count
        """
```

#### Constructor Parameters

- `model` (str): Model name for token encoding
- `encoding_name` (str, optional): Specific encoding name to use

### ResultsAnalyzer

Performs statistical analysis of test results to determine effective context length limits.

```python
class ResultsAnalyzer:
    """
    Analyzes test results to determine effective context length limits.
    
    Provides statistical analysis with confidence intervals and error rate calculations.
    """
    
    def analyze_session(self, session: TestSession, error_threshold: float, min_samples: int) -> AnalysisResult:
        """
        Analyze a complete test session.
        
        Args:
            session: Test session with all results
            error_threshold: Maximum acceptable error rate
            min_samples: Minimum samples required for analysis
            
        Returns:
            AnalysisResult: Complete analysis with recommendations
        """
        
    def export_detailed_results(self, analysis: AnalysisResult) -> Dict[str, Any]:
        """
        Export analysis results in structured format.
        
        Args:
            analysis: Analysis result to export
            
        Returns:
            Dict[str, Any]: Structured results suitable for serialization
        """
```

#### Constructor Parameters

- `confidence_level` (float): Confidence level for statistical analysis (default: 0.95)

## Data Classes

### TestSession

```python
@dataclass
class TestSession:
    """Complete test session with all results and metadata."""
    base_url: str
    model: str
    strategy: TestStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[ContextLengthResults] = field(default_factory=list)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    configuration: Dict[str, Any] = field(default_factory=dict)
```

### TestResult

```python
@dataclass
class TestResult:
    """Individual test result."""
    context_length: int
    success: bool
    response_time_ms: float
    tokens_sent: Optional[int] = None
    tokens_received: Optional[int] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt_number: int = 1
```

### APIResponse

```python
@dataclass
class APIResponse:
    """Structured response from API requests."""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: float = 0.0
    content: Optional[Dict[str, Any]] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    tokens_sent: Optional[int] = None
    tokens_received: Optional[int] = None
```

## Enums

### TestStrategy

```python
class TestStrategy(Enum):
    """Available testing strategies."""
    HYBRID = "hybrid"           # Combines gradual and binary search approaches
    GRADUAL = "gradual"         # Step-by-step increasing context lengths
    BINARY_SEARCH = "binary-search"  # Binary search for efficient finding
    PREDEFINED = "predefined"   # Test specific predefined sizes
```

### ContextStrategy

```python
class ContextStrategy(Enum):
    """Content generation strategies."""
    TOKENS = "tokens"           # Generate exact token counts
    WORDS = "words"             # Word-based content generation
    SENTENCES = "sentences"     # Sentence-based content generation
    MIXED = "mixed"             # Mixed content types
```

### ErrorType

```python
class ErrorType(Enum):
    """Classification of different error types."""
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    RATE_LIMIT = "rate_limit_exceeded"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    AUTHENTICATION = "authentication_error"
    NETWORK = "network_error"
    UNKNOWN = "unknown_error"
```

## Usage Examples

### Basic Usage

```python
import asyncio
from effective_context_length import EffectiveContextLengthTester, TestStrategy

async def basic_test():
    async with EffectiveContextLengthTester(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        model="gpt-3.5-turbo",
        strategy=TestStrategy.HYBRID
    ) as tester:
        session = await tester.run_test()
        print(f"Effective context length: {session.effective_context_length}")

asyncio.run(basic_test())
```

### Custom Configuration

```python
async def custom_test():
    config = {
        'min_tokens': 1000,
        'max_tokens': 100000,
        'step_size': 5000,
        'samples_per_size': 3,
        'error_threshold': 0.15,
        'timeout': 600,
        'rate_limit': 2.0
    }
    
    async with EffectiveContextLengthTester(
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        model="custom-model",
        strategy=TestStrategy.GRADUAL,
        **config
    ) as tester:
        session = await tester.run_test()
        return session
```

### Progress Tracking

```python
def progress_callback(completed: int, total: int, message: str = ""):
    """Custom progress callback function."""
    percentage = (completed / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% - {message}")

async def test_with_progress():
    async with EffectiveContextLengthTester(...) as tester:
        session = await tester.run_test(progress_callback=progress_callback)
```

### Error Handling

```python
async def robust_test():
    try:
        async with EffectiveContextLengthTester(...) as tester:
            session = await tester.run_test()
            return session
    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"Test execution error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Best Practices

1. **Resource Management**: Always use the async context manager for proper resource cleanup
2. **Error Handling**: Implement appropriate error handling for production use
3. **Rate Limiting**: Configure appropriate rate limits to avoid overwhelming the API
4. **Timeout Settings**: Set reasonable timeouts based on your network conditions
5. **Progress Tracking**: Use progress callbacks for long-running tests
6. **Configuration**: Choose appropriate testing strategies based on your needs:
   - Use `HYBRID` for balanced efficiency and accuracy
   - Use `BINARY_SEARCH` for quick approximation
   - Use `GRADUAL` for detailed analysis
   - Use `PREDEFINED` for testing specific sizes