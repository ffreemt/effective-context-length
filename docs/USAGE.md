# Usage Guide

## Overview

This guide provides comprehensive instructions for using the Effective Context Length tool to test and determine the practical context length limits of OpenAI-compatible LLM endpoints.

## Quick Start

### Basic Usage

```bash
# Test with default settings (uses your configured defaults)
python effective_context_length.py

# Test a specific endpoint
python effective_context_length.py https://api.openai.com/v1

# Test with custom model
python effective_context_length.py https://api.openai.com/v1 --model gpt-4

# Save results to file
python effective_context_length.py https://api.openai.com/v1 --output results.json
```

### Common Examples

```bash
# Quick test of local endpoint
python effective_context_length.py http://localhost:8000/v1 --model custom-model

# Comprehensive test with binary search
python effective_context_length.py https://api.example.com/v1 \
  --strategy binary-search \
  --max-tokens 128000 \
  --samples-per-size 3 \
  --output comprehensive_results.json

# Test with specific API key
python effective_context_length.py https://api.openai.com/v1 \
  --api-key sk-your-api-key \
  --model gpt-4 \
  --verbose
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality, install these optional packages:

```bash
# For CSV output format
pip install pyyaml

# For statistical analysis
pip install scipy numpy
```

## Configuration

### Default Configuration

The tool now uses sensible defaults:

- **Base URL**: `http://127.0.0.1:3300/v1`
- **API Key**: `sk-aMjTHX0fJGIC7jb34HRZSLC8QTtZACQw92VewXqJYiiP4Knb`
- **Model**: `v0-1.0-md`

### Environment Variables

```bash
# Set API key via environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or use custom environment variable name
python effective_context_length.py --api-key-env CUSTOM_API_KEY_VAR
```

## Testing Strategies

### 1. Hybrid Strategy (Recommended)

**Default strategy that combines gradual and binary search approaches**

```bash
python effective_context_length.py https://api.openai.com/v1 --strategy hybrid
```

**When to use:**
- General purpose testing
- Balanced between speed and accuracy
- Good for most endpoints

### 2. Binary Search Strategy

**Efficient strategy using binary search algorithm**

```bash
python effective_context_length.py https://api.openai.com/v1 --strategy binary-search
```

**When to use:**
- Quick approximation of limits
- Large context ranges (100K+ tokens)
- Limited testing time

### 3. Gradual Strategy

**Step-by-step testing with configurable increments**

```bash
python effective_context_length.py https://api.openai.com/v1 \
  --strategy gradual \
  --step-size 5000 \
  --min-tokens 1000 \
  --max-tokens 50000
```

**When to use:**
- Detailed analysis of behavior
- Small context ranges
- Understanding gradual degradation

### 4. Predefined Strategy

**Test specific context lengths**

```bash
python effective_context_length.py https://api.openai.com/v1 \
  --strategy predefined \
  --predefined-sizes 1000 4000 8000 16000 32000 64000 128000
```

**When to use:**
- Testing known breakpoints
- Compliance testing
- Specific use case validation

## Output Formats

### JSON Format (Default)

```bash
python effective_context_length.py https://api.openai.com/v1 --format json
```

**Features:**
- Machine-readable format
- Complete test data
- Metadata and analysis
- Easy integration with other tools

### CSV Format

```bash
python effective_context_length.py https://api.openai.com/v1 --format csv --output results.csv
```

**Features:**
- Spreadsheet-compatible
- Summary statistics
- Easy data analysis
- Reporting-friendly

### YAML Format

```bash
python effective_context_length.py https://api.openai.com/v1 --format yaml
```

**Features:**
- Human-readable
- Structured data
- Configuration-friendly

### Text Format

```bash
python effective_context_length.py https://api.openai.com/v1 --format txt
```

**Features:**
- Easy to read
- Printable reports
- Summary-focused
- No special tools required

## Advanced Usage

### Custom Testing Parameters

```bash
python effective_context_length.py https://api.openai.com/v1 \
  --min-tokens 500 \
  --max-tokens 100000 \
  --step-size 2500 \
  --samples-per-size 5 \
  --error-threshold 0.15 \
  --confidence-level 0.99 \
  --min-samples 3
```

### Rate Limiting and Timeouts

```bash
python effective_context_length.py https://api.openai.com/v1 \
  --rate-limit 0.5 \
  --timeout 600 \
  --max-retries 5
```

### Request Configuration

```bash
python effective_context_length.py https://api.openai.com/v1 \
  --temperature 0.0 \
  --max-retries 3 \
  --timeout 300
```

### Progress and Output Control

```bash
# Verbose output with detailed logging
python effective_context_length.py https://api.openai.com/v1 --verbose

# Quiet mode (only final results)
python effective_context_length.py https://api.openai.com/v1 --quiet

# Disable progress bar
python effective_context_length.py https://api.openai.com/v1 --no-progress

# Custom output location
python effective_context_length.py https://api.openai.com/v1 --output /path/to/results.json
```

## Understanding Results

### Key Metrics

**Effective Context Length**: The maximum context length that maintains acceptable error rates

**Recommended Safe Length**: A conservative limit (typically 80-90% of effective length)

**Error Rate**: Percentage of failed requests at each context length

**Response Time**: Average time for successful requests

**Confidence Interval**: Statistical confidence in the results

### Interpreting Results

```json
{
  "summary": {
    "effective_context_length": 16384,
    "recommended_safe_length": 14746,
    "overall_success_rate": 0.87,
    "error_rate": 0.13
  },
  "recommendations": [
    "Use recommended safe length for production workloads",
    "Monitor error rates at high context lengths",
    "Consider model upgrades for larger context requirements"
  ]
}
```

## Troubleshooting

### Common Issues

#### Authentication Errors

```bash
# Check API key
python effective_context_length.py https://api.openai.com/v1 --api-key your-key

# Use environment variable
export OPENAI_API_KEY="your-key"
python effective_context_length.py https://api.openai.com/v1
```

#### Network Issues

```bash
# Increase timeout
python effective_context_length.py https://api.openai.com/v1 --timeout 600

# Reduce rate limit
python effective_context_length.py https://api.openai.com/v1 --rate-limit 0.2

# Increase retries
python effective_context_length.py https://api.openai.com/v1 --max-retries 5
```

#### Memory/Performance Issues

```bash
# Reduce sample size
python effective_context_length.py https://api.openai.com/v1 --samples-per-size 2

# Use binary search for faster testing
python effective_context_length.py https://api.openai.com/v1 --strategy binary-search

# Lower max tokens
python effective_context_length.py https://api.openai.com/v1 --max-tokens 50000
```

### Debug Mode

```bash
# Enable verbose logging
python effective_context_length.py https://api.openai.com/v1 --verbose --verbose --verbose

# Test with minimal configuration
python effective_context_length.py https://api.openai.com/v1 \
  --max-tokens 5000 \
  --samples-per-size 1 \
  --min-tokens 1000 \
  --strategy predefined \
  --predefined-sizes 1000 2000 3000 4000 5000
```

## Best Practices

### 1. Choose Appropriate Strategy

- **Development**: Use `binary-search` for quick feedback
- **Testing**: Use `hybrid` for balanced results
- **Production**: Use `gradual` for comprehensive analysis
- **Compliance**: Use `predefined` for specific requirements

### 2. Configure Realistic Parameters

```bash
# For production testing
python effective_context_length.py https://api.openai.com/v1 \
  --samples-per-size 5 \
  --error-threshold 0.1 \
  --confidence-level 0.95

# For quick testing
python effective_context_length.py https://api.openai.com/v1 \
  --samples-per-size 2 \
  --error-threshold 0.2 \
  --strategy binary-search
```

### 3. Monitor Resource Usage

- Start with smaller context ranges
- Monitor memory usage during tests
- Use appropriate rate limits to avoid overwhelming APIs
- Save results for later analysis

### 4. Interpret Results Carefully

- Consider error rates in context of your use case
- Use recommended safe lengths for production
- Monitor performance degradation at high context lengths
- Test with realistic workloads when possible

## Integration Examples

### CI/CD Pipeline

```yaml
# GitHub Actions example
name: Context Length Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run context length test
      run: |
        python effective_context_length.py ${{ secrets.API_URL }} \
          --api-key ${{ secrets.API_KEY }} \
          --model ${{ secrets.MODEL_NAME }} \
          --output test_results.json \
          --max-tokens 50000
      env:
        API_URL: ${{ secrets.API_URL }}
        API_KEY: ${{ secrets.API_KEY }}
        MODEL_NAME: ${{ secrets.MODEL_NAME }}
```

### Python Script Integration

```python
import asyncio
import subprocess
import json

async def run_context_test():
    """Run context length test from Python script."""
    cmd = [
        'python', 'effective_context_length.py',
        'https://api.openai.com/v1',
        '--model', 'gpt-3.5-turbo',
        '--strategy', 'hybrid',
        '--output', 'test_results.json',
        '--quiet'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        with open('test_results.json', 'r') as f:
            data = json.load(f)
            return data['summary']['effective_context_length']
    else:
        raise Exception(f"Test failed: {result.stderr}")

# Usage
effective_length = asyncio.run(run_context_test())
print(f"Effective context length: {effective_length}")
```

## Performance Tips

1. **Start Small**: Begin with limited token ranges to estimate limits
2. **Use Binary Search**: For large ranges, binary search is most efficient
3. **Adjust Sample Size**: Balance accuracy with testing time
4. **Monitor Resources**: Watch memory and CPU usage during tests
5. **Cache Results**: Save results for comparison over time

## Support

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review the API documentation
- Examine test results for error patterns
- Consider network conditions and API status