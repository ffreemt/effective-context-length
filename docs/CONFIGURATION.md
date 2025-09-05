# Configuration Reference

## Overview

This document provides a comprehensive reference for all configuration options available in the Effective Context Length tool. Options can be set via command-line arguments or environment variables.

## Command Line Options

### Positional Arguments

#### `base_url`
- **Type**: String (optional)
- **Default**: `http://127.0.0.1:3300/v1`
- **Description**: Base URL of the OpenAI-compatible API endpoint
- **Example**: `https://api.openai.com/v1`, `http://localhost:8000/v1`

### Authentication Options

#### `--api-key`
- **Type**: String
- **Default**: `sk-aMjTHX0fJGIC7jb34HRZSLC8QTtZACQw92VewXqJYiiP4Knb`
- **Description**: API key for authentication. Can also be set via environment variable.
- **Environment Variable**: `OPENAI_API_KEY` (customizable with `--api-key-env`)
- **Example**: `--api-key sk-your-api-key-here`

#### `--api-key-env`
- **Type**: String
- **Default**: `OPENAI_API_KEY`
- **Description**: Environment variable name containing the API key
- **Example**: `--api-key-env CUSTOM_API_KEY`

### Model Configuration

#### `--model`
- **Type**: String
- **Default**: `v0-1.0-md`
- **Description**: Model name to test for context length limits
- **Example**: `--model gpt-4`, `--model claude-3-opus`

#### `--max-tokens`
- **Type**: Integer
- **Default**: `200000`
- **Description**: Maximum context length to test (in tokens)
- **Range**: Must be greater than `--min-tokens`
- **Example**: `--max-tokens 128000`

#### `--min-tokens`
- **Type**: Integer
- **Default**: `1000`
- **Description**: Minimum context length to start testing (in tokens)
- **Range**: Must be positive and less than `--max-tokens`
- **Example**: `--min-tokens 500`

### Testing Strategy

#### `--strategy`
- **Type**: Choice
- **Default**: `hybrid`
- **Options**: `gradual`, `binary-search`, `predefined`, `hybrid`
- **Description**: Testing strategy to use for determining context length limits

**Strategy Details:**

- **`hybrid`**: Combines gradual and binary search approaches for balanced efficiency and accuracy
- **`gradual`**: Step-by-step increasing context lengths with configurable increments
- **`binary-search`**: Efficient logarithmic approach for quick approximation
- **`predefined`**: Test specific context lengths provided by the user

#### `--step-size`
- **Type**: Integer
- **Default**: `2000`
- **Description**: Step size for gradual strategy (in tokens)
- **Relevant for**: `gradual` strategy
- **Example**: `--step-size 5000`

#### `--predefined-sizes`
- **Type**: List of integers
- **Default**: None
- **Description**: Predefined context sizes to test (space-separated list)
- **Relevant for**: `predefined` strategy
- **Required for**: `predefined` strategy
- **Example**: `--predefined-sizes 1000 4000 8000 16000 32000 64000`

#### `--samples-per-size`
- **Type**: Integer
- **Default**: `5`
- **Description**: Number of test samples per context size
- **Range**: Must be positive
- **Example**: `--samples-per-size 3`

### Request Configuration

#### `--timeout`
- **Type**: Integer
- **Default**: `300`
- **Description**: Request timeout in seconds
- **Range**: Must be positive
- **Example**: `--timeout 600`

#### `--max-retries`
- **Type**: Integer
- **Default**: `3`
- **Description**: Maximum number of retries per request
- **Range**: Must be non-negative
- **Example**: `--max-retries 5`

#### `--rate-limit`
- **Type**: Float
- **Default**: `1.0`
- **Description**: Rate limit in requests per second
- **Range**: Must be positive
- **Example**: `--rate-limit 0.5`

#### `--temperature`
- **Type**: Float
- **Default**: `0.1`
- **Description**: Temperature parameter for API requests (affects response randomness)
- **Range**: `0.0` to `2.0`
- **Example**: `--temperature 0.0`

### Output Configuration

#### `--output`, `-o`
- **Type**: String
- **Default**: None (prints to stdout)
- **Description**: Output file path for results
- **Example**: `--output results.json`, `-o report.csv`

#### `--format`
- **Type**: Choice
- **Default**: `json`
- **Options**: `json`, `csv`, `yaml`, `txt`
- **Description**: Output format for results

**Format Details:**

- **`json`**: Machine-readable format with complete data structure
- **`csv`**: Spreadsheet-compatible format with summary statistics
- **`yaml`**: Human-readable structured format
- **`txt`**: Printable report format with key metrics

#### `--verbose`, `-v`
- **Type**: Count
- **Default**: `0`
- **Description**: Increase verbosity level
- **Usage**: 
  - `-v`: Basic verbose output
  - `-vv`: Detailed verbose output
  - `-vvv`: Debug-level output

#### `--quiet`, `-q`
- **Type**: Flag
- **Default**: False
- **Description**: Suppress progress output (only show final results)
- **Mutually exclusive with**: `--verbose`

#### `--no-progress`
- **Type**: Flag
- **Default**: False
- **Description**: Disable progress bar display
- **Example**: `--no-progress`

### Analysis Configuration

#### `--error-threshold`
- **Type**: Float
- **Default**: `0.1`
- **Description**: Error rate threshold for determining effective limit (0.0 to 1.0)
- **Range**: `0.0` to `1.0`
- **Example**: `--error-threshold 0.15`

#### `--confidence-level`
- **Type**: Float
- **Default**: `0.95`
- **Description**: Confidence level for statistical analysis
- **Range**: `0.0` to `1.0` (exclusive)
- **Example**: `--confidence-level 0.99`

#### `--min-samples`
- **Type**: Integer
- **Default**: `3`
- **Description**: Minimum number of samples required per context length
- **Range**: Must be positive
- **Example**: `--min-samples 5`

### Utility Options

#### `--version`
- **Type**: Flag
- **Description**: Show program version information and exit

#### `--help`, `-h`
- **Type**: Flag
- **Description**: Show help message and exit

## Environment Variables

### Primary Environment Variables

#### `OPENAI_API_KEY`
- **Purpose**: Default API key for authentication
- **Override**: Use `--api-key-env` to specify a different variable name
- **Example**: `export OPENAI_API_KEY="sk-your-key"`

### Custom Environment Variables

You can use any environment variable name by specifying it with `--api-key-env`:

```bash
export MY_API_KEY="sk-custom-key"
python effective_context_length.py --api-key-env MY_API_KEY
```

## Configuration Files

The tool does not currently support configuration files, but you can create shell scripts or aliases for common configurations:

### Bash Script Example

```bash
#!/bin/bash
# test_config.sh - Common testing configurations

# Quick test for local development
quick_test() {
    python effective_context_length.py \
        --strategy binary-search \
        --max-tokens 50000 \
        --samples-per-size 2 \
        --quiet
}

# Comprehensive test for production
comprehensive_test() {
    python effective_context_length.py \
        --strategy hybrid \
        --samples-per-size 5 \
        --error-threshold 0.1 \
        --confidence-level 0.95 \
        --output comprehensive_results.json
}

# Test specific model
test_model() {
    local model=$1
    python effective_context_length.py \
        --model "$model" \
        --strategy gradual \
        --step-size 1000 \
        --max-tokens 100000 \
        --output "${model}_results.json"
}
```

## Configuration Precedence

The tool uses the following precedence for configuration (highest to lowest):

1. **Command line arguments** (explicitly provided)
2. **Environment variables** (via `--api-key-env`)
3. **Default values** (built into the tool)

## Strategy-Specific Configurations

### Hybrid Strategy Configuration
```bash
python effective_context_length.py \
    --strategy hybrid \
    --samples-per-size 5 \
    --error-threshold 0.1 \
    --confidence-level 0.95
```

### Binary Search Configuration
```bash
python effective_context_length.py \
    --strategy binary-search \
    --samples-per-size 3 \
    --min-tokens 1000 \
    --max-tokens 200000
```

### Gradual Strategy Configuration
```bash
python effective_context_length.py \
    --strategy gradual \
    --step-size 2000 \
    --min-tokens 1000 \
    --max-tokens 50000
```

### Predefined Strategy Configuration
```bash
python effective_context_length.py \
    --strategy predefined \
    --predefined-sizes 1000 2000 4000 8000 16000 32000 64000 128000
```

## Performance Tuning

### For Fast Testing
```bash
python effective_context_length.py \
    --strategy binary-search \
    --samples-per-size 2 \
    --max-tokens 50000 \
    --timeout 60 \
    --rate-limit 2.0
```

### For Accurate Results
```bash
python effective_context_length.py \
    --strategy hybrid \
    --samples-per-size 7 \
    --error-threshold 0.05 \
    --confidence-level 0.99 \
    --min-samples 5
```

### For Resource-Constrained Environments
```bash
python effective_context_length.py \
    --strategy gradual \
    --step-size 5000 \
    --samples-per-size 2 \
    --rate-limit 0.3 \
    --timeout 180
```

## Error Handling Configuration

### Retry Configuration
```bash
python effective_context_length.py \
    --max-retries 5 \
    --timeout 600 \
    --rate-limit 0.5
```

### Error Threshold Configuration
```bash
# Lenient threshold (for development)
python effective_context_length.py --error-threshold 0.2

# Strict threshold (for production)
python effective_context_length.py --error-threshold 0.05

# Custom threshold (for specific use cases)
python effective_context_length.py --error-threshold 0.15
```

## Configuration Validation

The tool validates all configurations and will report errors for:

- Invalid URL formats
- Inconsistent token ranges (`min-tokens` >= `max-tokens`)
- Negative values for positive-only parameters
- Invalid strategy/predefined-sizes combinations
- Conflicting options (e.g., `--quiet` and `--verbose`)
- Out-of-range values (temperature, confidence level, etc.)

## Best Practices

1. **Start Conservative**: Begin with smaller token ranges to estimate limits
2. **Adjust for Environment**: Use longer timeouts for slower networks
3. **Respect Rate Limits**: Configure appropriate rate limits to avoid API throttling
4. **Balance Speed vs Accuracy**: More samples = more accurate but slower testing
5. **Use Appropriate Strategies**: Choose strategies based on your testing goals
6. **Save Results**: Always save results for later analysis and comparison

## Troubleshooting Configuration Issues

### Common Issues and Solutions

1. **Authentication Failures**
   ```bash
   # Check API key
   export OPENAI_API_KEY="your-key"
   python effective_context_length.py --verbose
   
   # Or specify directly
   python effective_context_length.py --api-key your-key
   ```

2. **Timeout Errors**
   ```bash
   # Increase timeout
   python effective_context_length.py --timeout 600
   
   # Reduce rate limit
   python effective_context_length.py --rate-limit 0.5
   ```

3. **Memory Issues**
   ```bash
   # Reduce sample size
   python effective_context_length.py --samples-per-size 2
   
   # Use binary search
   python effective_context_length.py --strategy binary-search
   
   # Lower max tokens
   python effective_context_length.py --max-tokens 50000
   ```

4. **Rate Limiting**
   ```bash
   # Reduce request rate
   python effective_context_length.py --rate-limit 0.3
   
   # Increase retry attempts
   python effective_context_length.py --max-retries 5
   ```