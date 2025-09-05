# Effective Context Length

A comprehensive CLI tool for testing and determining the effective context length limits of OpenAI-compatible Large Language Model (LLM) endpoints.

## 🎯 What This Tool Does

This tool systematically tests various context lengths, collects error rate data, and provides comprehensive reporting on practical context length limits. It helps you understand:

- **Maximum reliable context length** for your specific model/endpoint
- **Error rates** at different context lengths
- **Performance characteristics** across the context range
- **Recommended safe limits** for production use

## ✨ Key Features

- **Multiple Testing Strategies**: Hybrid, Binary Search, Gradual, and Predefined approaches
- **Comprehensive Error Analysis**: Categorizes and analyzes different types of failures
- **Statistical Analysis**: Confidence intervals and error rate calculations
- **Accurate Token Counting**: Uses tiktoken for precise token measurements
- **Robust HTTP Client**: Retry logic, rate limiting, and error handling
- **Multiple Output Formats**: JSON, CSV, YAML, and text reports
- **Enhanced Logging**: Detailed debugging with filename and line numbers
- **Progress Tracking**: Real-time progress reporting during tests

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd effective-context-length

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for enhanced functionality
pip install pyyaml scipy numpy
```

### Basic Usage

```bash
# Test with default configuration (uses your configured defaults)
python effective_context_length.py

# Test a specific endpoint
python effective_context_length.py https://api.openai.com/v1

# Test with custom model
python effective_context_length.py https://api.openai.com/v1 --model gpt-4

# Save results to file
python effective_context_length.py https://api.openai.com/v1 --output results.json
```

## 📖 Usage Examples

### Example 1: Quick Endpoint Test

```bash
# Test OpenAI's GPT-3.5-turbo with quick binary search
python effective_context_length.py https://api.openai.com/v1 \
  --model gpt-3.5-turbo \
  --strategy binary-search \
  --max-tokens 16000 \
  --samples-per-size 2 \
  --output gpt35-results.json
```

### Example 2: Comprehensive Analysis

```bash
# Thorough testing of a local model with detailed reporting
python effective_context_length.py http://localhost:8000/v1 \
  --model custom-llm \
  --strategy hybrid \
  --max-tokens 100000 \
  --min-tokens 1000 \
  --samples-per-size 5 \
  --error-threshold 0.1 \
  --confidence-level 0.95 \
  --timeout 600 \
  --rate-limit 0.5 \
  --output comprehensive-analysis.json \
  --format json \
  --verbose
```

### Example 3: Production Safety Testing

```bash
# Conservative testing for production deployment
python effective_context_length.py https://your-api-endpoint.com/v1 \
  --model production-model \
  --strategy gradual \
  --step-size 1000 \
  --max-tokens 50000 \
  --samples-per-size 7 \
  --error-threshold 0.05 \
  --min-samples 5 \
  --output production-safety-report.csv \
  --format csv
```

### Example 4: Specific Context Lengths

```bash
# Test specific context lengths for compliance testing
python effective_context_length.py https://api.example.com/v1 \
  --model compliance-model \
  --strategy predefined \
  --predefined-sizes 1000 2000 4000 8000 16000 32000 64000 128000 \
  --samples-per-size 3 \
  --output compliance-test.yaml \
  --format yaml
```

## 🔧 Configuration

### Default Configuration

The tool comes pre-configured with sensible defaults:
- **Base URL**: `http://127.0.0.1:3300/v1`
- **API Key**: `sk-aMjTHX0fJGIC7jb34HRZSLC8QTtZACQw92VewXqJYiiP4Knb`
- **Model**: `v0-1.0-md`

### Environment Variables

```bash
# Set API key via environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or use custom environment variable
python effective_context_length.py --api-key-env CUSTOM_API_KEY_VAR
```

### Common Configuration Options

```bash
# Rate limiting and timeouts
python effective_context_length.py https://api.example.com/v1 \
  --rate-limit 0.3 \
  --timeout 600 \
  --max-retries 5

# Output control
python effective_context_length.py https://api.example.com/v1 \
  --quiet \
  --no-progress \
  --output results.json

# Verbose logging for debugging
python effective_context_length.py https://api.example.com/v1 \
  --verbose --verbose
```

## 📊 Understanding Results

### Sample Output

```json
{
  "summary": {
    "effective_context_length": 16384,
    "recommended_safe_length": 14746,
    "overall_success_rate": 0.87,
    "error_rate": 0.13,
    "average_response_time_ms": 1250.5
  },
  "recommendations": [
    "Use recommended safe length for production workloads",
    "Monitor error rates at high context lengths",
    "Consider model upgrades for larger context requirements"
  ]
}
```

### Key Metrics

- **Effective Context Length**: Maximum reliable context length
- **Recommended Safe Length**: Conservative limit (typically 80-90% of effective)
- **Success Rate**: Percentage of successful requests
- **Error Rate**: Percentage of failed requests
- **Response Time**: Average time for successful requests

## 🛠️ Testing Strategies

### Hybrid (Recommended)
Combines gradual and binary search approaches for balanced efficiency and accuracy.

```bash
python effective_context_length.py https://api.example.com/v1 --strategy hybrid
```

### Binary Search
Efficient logarithmic approach for quick approximation.

```bash
python effective_context_length.py https://api.example.com/v1 --strategy binary-search
```

### Gradual
Step-by-step testing with configurable increments for detailed analysis.

```bash
python effective_context_length.py https://api.example.com/v1 --strategy gradual --step-size 2000
```

### Predefined
Test specific context lengths for compliance or specific use cases.

```bash
python effective_context_length.py https://api.example.com/v1 --strategy predefined --predefined-sizes 1000 4000 8000
```

## 🔍 Troubleshooting

### Common Issues

**Authentication Errors**
```bash
# Check API key
export OPENAI_API_KEY="your-key"
python effective_context_length.py --verbose
```

**Timeout Issues**
```bash
# Increase timeout
python effective_context_length.py --timeout 600
```

**Rate Limiting**
```bash
# Reduce request rate
python effective_context_length.py --rate-limit 0.3
```

**Memory Issues**
```bash
# Reduce sample size
python effective_context_length.py --samples-per-size 2
```

### Debug Mode

```bash
# Enable detailed logging
python effective_context_length.py --verbose --verbose --verbose

# Test with minimal configuration
python effective_context_length.py https://api.example.com/v1 \
  --max-tokens 5000 \
  --samples-per-size 1 \
  --strategy predefined \
  --predefined-sizes 1000 2000 3000
```

## 📚 Documentation

- [API Documentation](docs/API.md) - Detailed API reference
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples
- [Configuration Reference](docs/CONFIGURATION.md) - All configuration options

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_http_client.py

# Run with coverage
pytest --cov=effective_context_length

# Run with verbose output
pytest -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Python and modern async/await patterns
- Uses [tiktoken](https://github.com/openai/tiktoken) for accurate token counting
- Implements robust HTTP client with [httpx](https://www.python-httpx.org/)
- Statistical analysis with [scipy](https://scipy.org/) and [numpy](https://numpy.org/)

---

**Made with ❤️ for the LLM development community**