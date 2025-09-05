
# Effective Context Length CLI Tool - Technical Architecture Specification

## 1. Overview

The Effective Context Length CLI Tool is a Python-based command-line utility designed to test and determine the effective context length limits for OpenAI-compatible LLM endpoints. The tool systematically tests various context lengths, collects error rate data, and provides comprehensive reporting on the practical limits of LLM endpoints.

## 2. Overall Tool Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   ArgumentParser │  │  Configuration  │  │    Logger    │ │
│  │     Handler     │  │    Manager      │  │   Manager    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Core Engine Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Test Runner   │  │  Context Length │  │   Results    │ │
│  │    Manager      │  │   Generator     │  │  Collector   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  API Communication Layer                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   HTTP Client   │  │   Token Counter │  │   Response   │ │
│  │    (httpx)      │  │                 │  │   Parser     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Data Analysis Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Error Rate     │  │   Statistical   │  │   Report     │ │
│  │  Calculator     │  │   Analyzer      │  │  Generator   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

#### CLI Interface Layer
- **ArgumentParser Handler**: Processes command-line arguments and validates input
- **Configuration Manager**: Manages configuration files and environment variables
- **Logger Manager**: Handles logging configuration and output formatting

#### Core Engine Layer
- **Test Runner Manager**: Orchestrates the testing process and manages test execution
- **Context Length Generator**: Generates test payloads of varying context lengths
- **Results Collector**: Aggregates and stores test results

#### API Communication Layer
- **HTTP Client**: Manages API requests using httpx with retry logic and rate limiting
- **Token Counter**: Accurately counts tokens in requests and responses
- **Response Parser**: Parses API responses and extracts relevant metrics

#### Data Analysis Layer
- **Error Rate Calculator**: Computes error rates and success metrics
- **Statistical Analyzer**: Performs statistical analysis on collected data
- **Report Generator**: Creates formatted reports and visualizations

## 3. CLI Interface Design

### 3.1 Command Structure

```bash
effective-context-length [OPTIONS] BASE_URL
```

### 3.2 Argument Parser Structure

```python
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        prog='effective-context-length',
        description='Test effective context length for OpenAI-compatible LLM endpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://api.openai.com/v1
  %(prog)s http://localhost:8000/v1 --model custom-model --max-tokens 128000
  %(prog)s https://api.example.com/v1 --strategy binary-search --output results.json
        """
    )
    
    # Positional arguments
    parser.add_argument('base_url', help='Base URL of the OpenAI-compatible API endpoint')
    
    # Authentication
    auth_group = parser.add_argument_group('Authentication')
    auth_group.add_argument('--api-key', help='API key for authentication')
    auth_group.add_argument('--api-key-env', default='OPENAI_API_KEY', 
                           help='Environment variable containing API key (default: OPENAI_API_KEY)')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', default='gpt-3.5-turbo', 
                            help='Model name to test (default: gpt-3.5-turbo)')
    model_group.add_argument('--max-tokens', type=int, default=200000,
                            help='Maximum context length to test (default: 200000)')
    model_group.add_argument('--min-tokens', type=int, default=1000,
                            help='Minimum context length to start testing (default: 1000)')
    
    # Testing strategy
    strategy_group = parser.add_argument_group('Testing Strategy')
    strategy_group.add_argument('--strategy', choices=['gradual', 'binary-search', 'predefined', 'hybrid'],
                               default='hybrid', help='Testing strategy (default: hybrid)')
    strategy_group.add_argument('--step-size', type=int, default=2000,
                               help='Step size for gradual strategy (default: 2000)')
    strategy_group.add_argument('--predefined-sizes', nargs='+', type=int,
                               help='Predefined context sizes to test')
    strategy_group.add_argument('--samples-per-size', type=int, default=5,
                               help='Number of test samples per context size (default: 5)')
    
    # Request configuration
    request_group = parser.add_argument_group('Request Configuration')
    request_group.add_argument('--timeout', type=int, default=300,
                              help='Request timeout in seconds (default: 300)')
    request_group.add_argument('--max-retries', type=int, default=3,
                              help='Maximum number of retries per request (default: 3)')
    request_group.add_argument('--rate-limit', type=float, default=1.0,
                              help='Rate limit in requests per second (default: 1.0)')
    request_group.add_argument('--temperature', type=float, default=0.1,
                              help='Temperature for API requests (default: 0.1)')
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output', '-o', help='Output file path for results')
    output_group.add_argument('--format', choices=['json', 'csv', 'yaml', 'txt'],
                             default='json', help='Output format (default: json)')
    output_group.add_argument('--verbose', '-v', action='count', default=0,
                             help='Increase verbosity level')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Suppress progress output')
    output_group.add_argument('--no-progress', action='store_true',
                             help='Disable progress bar')
    
    # Analysis configuration
    analysis_group = parser.add_argument_group('Analysis Configuration')
    analysis_group.add_argument('--error-threshold', type=float, default=0.1,
                               help='Error rate threshold for determining effective limit (default: 0.1)')
    analysis_group.add_argument('--confidence-level', type=float, default=0.95,
                               help='Confidence level for statistical analysis (default: 0.95)')
    
    return parser
```

## 4. Data Collection Strategy

### 4.1 Testing Strategies

#### 4.1.1 Hybrid Strategy (Default)
1. **Phase 1 - Binary Search**: Quickly identify approximate upper bound
2. **Phase 2 - Fine-tuning**: Use gradual increases around the identified range
3. **Phase 3 - Validation**: Multiple samples at critical points

#### 4.1.2 Gradual Strategy
- Start from `min_tokens`
- Increase by `step_size` until `max_tokens` or error threshold reached
- Collect multiple samples per context length

#### 4.1.3 Binary Search Strategy
- Efficiently find maximum working context length
- Logarithmic time complexity
- Fewer total requests but less granular data

#### 4.1.4 Predefined Strategy
- Test specific context lengths provided by user
- Useful for comparing specific configurations
- Allows targeted testing of known problematic ranges

### 4.2 Test Payload Generation

```python
class ContextGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please analyze the following text: "}
        ]
    
    def generate_payload(self, target_tokens: int) -> dict:
        """Generate a payload with approximately target_tokens"""
        # Calculate tokens needed for filler content
        base_tokens = self.count_tokens(self.base_messages)
        filler_tokens = target_tokens - base_tokens - 50  # Reserve for response
        
        # Generate filler content
        filler_content = self.generate_filler_text(filler_tokens)
        
        messages = self.base_messages.copy()
        messages[-1]["content"] += filler_content
        
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": 50,
            "temperature": self.temperature
        }
```

### 4.3 Data Collection Points

For each test request, collect:
- **Request metadata**: timestamp, context_length, attempt_number
- **Response data**: success/failure, response_time, token_counts
- **Error information**: error_type, error_message, http_status_code
- **Performance metrics**: latency, throughput, memory_usage

## 5. Error Rate Calculation Methodology

### 5.1 Error Classification

```python
class ErrorClassifier:
    ERROR_TYPES = {
        'CONTEXT_LENGTH_EXCEEDED': ['context_length_exceeded', 'maximum context length'],
        'RATE_LIMIT': ['rate_limit_exceeded', 'too_many_requests'],
        'TIMEOUT': ['timeout', 'request_timeout'],
        'SERVER_ERROR': ['internal_server_error', '5xx'],
        'AUTHENTICATION': ['unauthorized', 'invalid_api_key'],
        'NETWORK': ['connection_error', 'dns_error'],
        'OTHER': ['unknown_error']
    }
```

### 5.2 Error Rate Calculation

```python
def calculate_error_rates(results: List[TestResult]) -> Dict[str, float]:
    """Calculate various error rate metrics"""
    total_requests = len(results)
    
    # Overall error rate
    failed_requests = sum(1 for r in results if not r.success)
    overall_error_rate = failed_requests / total_requests
    
    # Context-specific error rate
    context_errors = sum(1 for r in results 
                        if not r.success and 
                        r.error_type == 'CONTEXT_LENGTH_EXCEEDED')
    context_error_rate = context_errors / total_requests
    
    # Success rate by context length
    success_by_length = {}
    for length in set(r.context_length for r in results):
        length_results = [r for r in results if r.context_length == length]
        success_count = sum(1 for r in length_results if r.success)
        success_by_length[length] = success_count / len(length_results)
    
    return {
        'overall_error_rate': overall_error_rate,
        'context_error_rate': context_error_rate,
        'success_by_length': success_by_length,
        'total_requests': total_requests,
        'failed_requests': failed_requests
    }
```

### 5.3 Effective Context Length Determination

```python
def determine_effective_context_length(
    success_rates: Dict[int, float], 
    error_threshold: float = 0.1
) -> Dict[str, Any]:
    """Determine effective context length based on error threshold"""
    
    sorted_lengths = sorted(success_rates.keys())
    
    # Find the largest context length with error rate below threshold
    effective_length = None
    for length in reversed(sorted_lengths):
        error_rate = 1 - success_rates[length]
        if error_rate <= error_threshold:
            effective_length = length
            break
    
    # Calculate confidence intervals
    confidence_interval = calculate_confidence_interval(
        success_rates, effective_length
    )
    
    return {
        'effective_context_length': effective_length,
        'confidence_interval': confidence_interval,
        'error_threshold_used': error_threshold,
        'success_rate_at_effective_length': success_rates.get(effective_length, 0)
    }
```

## 6. Output Format and Reporting Structure

### 6.1 JSON Output Format

```json
{
  "metadata": {
    "tool_version": "1.0.0",
    "test_timestamp": "2024-01-15T10:30:00Z",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-3.5-turbo",
    "strategy": "hybrid",
    "total_duration_seconds": 1234.56
  },
  "configuration": {
    "min_tokens": 1000,
    "max_tokens": 200000,
    "step_size": 2000,
    "samples_per_size": 5,
    "error_threshold": 0.1,
    "timeout": 300,
    "rate_limit": 1.0
  },
  "results": {
    "effective_context_length": 128000,
    "confidence_interval": {
      "lower": 125000,
      "upper": 130000,
      "confidence_level": 0.95
    },
    "overall_statistics": {
      "total_requests": 150,
      "successful_requests": 142,
      "failed_requests": 8,
      "overall_success_rate": 0.947,
      "average_response_time": 2.34
    },
    "error_analysis": {
      "error_types": {
        "CONTEXT_LENGTH_EXCEEDED": 6,
        "TIMEOUT": 1,
        "SERVER_ERROR": 1
      },
      "error_rate_by_context_length": {
        "100000": 0.0,
        "120000": 0.0,
        "130000": 0.2,
        "140000": 0.8
,
        "150000": 1.0
      }
    },
    "performance_metrics": {
      "average_latency_ms": 2340,
      "p95_latency_ms": 4500,
      "p99_latency_ms": 8000,
      "requests_per_second": 0.85
    },
    "context_length_analysis": {
      "tested_lengths": [1000, 5000, 10000, 50000, 100000, 120000, 130000, 140000, 150000],
      "success_rate_by_length": {
        "1000": 1.0,
        "5000": 1.0,
        "10000": 1.0,
        "50000": 1.0,
        "100000": 1.0,
        "120000": 1.0,
        "130000": 0.8,
        "140000": 0.2,
        "150000": 0.0
      }
    }
  },
  "raw_data": [
    {
      "timestamp": "2024-01-15T10:30:15Z",
      "context_length": 128000,
      "success": true,
      "response_time_ms": 2100,
      "tokens_sent": 128000,
      "tokens_received": 45,
      "error_type": null,
      "error_message": null
    }
  ]
}
```

### 6.2 CSV Output Format

```csv
context_length,success,response_time_ms,tokens_sent,tokens_received,error_type,error_message,timestamp
1000,true,1200,1000,50,,,2024-01-15T10:30:15Z
5000,true,1500,5000,48,,,2024-01-15T10:30:20Z
128000,false,0,128000,0,CONTEXT_LENGTH_EXCEEDED,Maximum context length exceeded,2024-01-15T10:35:15Z
```

### 6.3 Text Report Format

```
Effective Context Length Analysis Report
========================================

Test Configuration:
- Endpoint: https://api.openai.com/v1
- Model: gpt-3.5-turbo
- Strategy: hybrid
- Test Duration: 20m 34s

Results Summary:
- Effective Context Length: 128,000 tokens
- Confidence Interval: 125,000 - 130,000 tokens (95% confidence)
- Overall Success Rate: 94.7%
- Total Requests: 150

Performance Metrics:
- Average Response Time: 2.34s
- 95th Percentile Latency: 4.5s
- Requests per Second: 0.85

Error Analysis:
- Context Length Exceeded: 6 requests (4.0%)
- Timeouts: 1 request (0.7%)
- Server Errors: 1 request (0.7%)

Recommendations:
- Safe operating limit: 125,000 tokens
- Monitor error rates above 120,000 tokens
- Consider implementing retry logic for timeouts
```

## 7. Configuration Options and Parameters

### 7.1 Configuration File Support

```yaml
# effective-context-config.yaml
default:
  model: "gpt-3.5-turbo"
  strategy: "hybrid"
  min_tokens: 1000
  max_tokens: 200000
  step_size: 2000
  samples_per_size: 5
  error_threshold: 0.1
  timeout: 300
  rate_limit: 1.0
  
authentication:
  api_key_env: "OPENAI_API_KEY"
  
output:
  format: "json"
  verbose: 1
  
analysis:
  confidence_level: 0.95
  
profiles:
  quick_test:
    strategy: "binary-search"
    samples_per_size: 2
    max_tokens: 50000
    
  thorough_test:
    strategy: "gradual"
    step_size: 1000
    samples_per_size: 10
    
  production_validation:
    strategy: "predefined"
    predefined_sizes: [4000, 8000, 16000, 32000, 64000, 128000]
    samples_per_size: 20
```

### 7.2 Environment Variables

```bash
# Authentication
OPENAI_API_KEY=your_api_key_here
EFFECTIVE_CONTEXT_API_KEY=your_api_key_here

# Configuration
EFFECTIVE_CONTEXT_CONFIG_FILE=./config.yaml
EFFECTIVE_CONTEXT_LOG_LEVEL=INFO
EFFECTIVE_CONTEXT_OUTPUT_DIR=./results

# Rate limiting
EFFECTIVE_CONTEXT_RATE_LIMIT=1.0
EFFECTIVE_CONTEXT_MAX_RETRIES=3
```

## 8. Implementation Architecture Details

### 8.1 Core Classes Structure

```python
# Main application class
class EffectiveContextLengthTester:
    def __init__(self, config: Config):
        self.config = config
        self.client = HTTPClient(config)
        self.generator = ContextGenerator(config)
        self.analyzer = ResultsAnalyzer(config)
        self.reporter = ReportGenerator(config)
    
    async def run_test(self) -> TestResults:
        """Main test execution method"""
        pass

# Configuration management
class Config:
    def __init__(self, args: argparse.Namespace):
        self.base_url = args.base_url
        self.model = args.model
        # ... other configuration
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from file"""
        pass

# HTTP client with retry logic
class HTTPClient:
    def __init__(self, config: Config):
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            limits=httpx.Limits(max_connections=1)
        )
        self.rate_limiter = RateLimiter(config.rate_limit)
    
    async def make_request(self, payload: dict) -> Response:
        """Make API request with retry logic"""
        pass

# Test strategy implementations
class TestStrategy(ABC):
    @abstractmethod
    async def generate_test_plan(self) -> List[int]:
        """Generate list of context lengths to test"""
        pass

class HybridStrategy(TestStrategy):
    async def generate_test_plan(self) -> List[int]:
        # Phase 1: Binary search
        # Phase 2: Fine-tuning
        # Phase 3: Validation
        pass

# Results analysis
class ResultsAnalyzer:
    def analyze_results(self, results: List[TestResult]) -> AnalysisResults:
        """Perform statistical analysis on test results"""
        pass
    
    def calculate_effective_length(self, results: List[TestResult]) -> int:
        """Determine effective context length"""
        pass

# Report generation
class ReportGenerator:
    def generate_report(self, results: AnalysisResults, format: str) -> str:
        """Generate formatted report"""
        pass
```

### 8.2 Token Counting Strategy

```python
class TokenCounter:
    def __init__(self, model: str):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, messages: List[dict]) -> int:
        """Accurate token counting for messages"""
        # Implementation based on OpenAI's token counting guidelines
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Quick token estimation for text"""
        return len(self.encoding.encode(text))
```

### 8.3 Progress Tracking

```python
class ProgressTracker:
    def __init__(self, total_tests: int, quiet: bool = False):
        self.progress_bar = rich.progress.Progress() if not quiet else None
        self.task = self.progress_bar.add_task("Testing...", total=total_tests)
    
    def update(self, completed: int, description: str = None):
        """Update progress display"""
        if self.progress_bar:
            self.progress_bar.update(self.task, completed=completed, description=description)
```

## 9. Error Handling and Resilience

### 9.1 Retry Logic

```python
class RetryHandler:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RetryableError as e:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(self.backoff_factor ** attempt)
```

### 9.2 Rate Limiting

```python
class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.last_request_time = 0
    
    async def acquire(self):
        """Acquire permission to make request"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
```

## 10. Testing and Validation

### 10.1 Unit Testing Strategy

- Test each component in isolation
- Mock HTTP responses for consistent testing
- Validate token counting accuracy
- Test error handling scenarios

### 10.2 Integration Testing

- Test against known API endpoints
- Validate end-to-end workflows
- Test different configuration combinations
- Performance testing under various loads

### 10.3 Validation Metrics

- Accuracy of effective context length determination
- Performance benchmarks
- Error handling robustness
- Output format correctness

## 11. Deployment and Distribution

### 11.1 Package Structure

```
effective-context-length/
├── src/
│   └── effective_context_length/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── client.py
│       ├── strategies.py
│       ├── analyzer.py
│       ├── reporter.py
│       └── utils.py
├── tests/
├── docs/
├── examples/
├── pyproject.toml
├── README.md
└── LICENSE
```

### 11.2 Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.8"
httpx = "^0.25.0"
rich = "^13.0.0"
tiktoken = "^0.5.0"
pydantic = "^2.0.0"
pyyaml = "^6.0"
numpy = "^1.24.0"
scipy = "^1.10.0"
```

### 11.3 Installation Methods

- PyPI package: `pip install effective-context-length`
- Docker container: `docker run effective-context-length`
- Standalone executable: Platform-specific binaries

## 12. Future Enhancements

### 12.1 Planned Features

- Support for streaming responses
- Multi-model comparison testing
- Historical data tracking and trending
- Integration with monitoring systems
- Web-based dashboard for results visualization

### 12.2 Extensibility Points

- Plugin system for custom strategies
- Custom error classifiers
- Additional output formats
- Integration with CI/CD pipelines

## 13. Security Considerations

### 13.1 API Key Management

- Support for multiple authentication methods
- Secure storage of credentials
- Environment variable validation
- Key rotation support

### 13.2 Data Privacy

- No logging of sensitive request content
- Configurable data retention policies
- Option to exclude response content from logs
- GDPR compliance considerations

## 14. Performance Considerations

### 14.1 Optimization Strategies

- Async/await for concurrent requests
- Connection pooling and reuse
- Efficient memory management for large contexts
- Streaming for large responses

### 14.2 Resource Management

- Memory usage monitoring
- Disk space management for logs
- CPU usage optimization
- Network bandwidth considerations

---

This comprehensive technical specification provides the foundation for implementing a robust, scalable, and user-friendly CLI tool for testing effective context lengths of OpenAI-compatible LLM endpoints. The architecture emphasizes modularity, extensibility, and reliability while providing comprehensive data collection and analysis capabilities.