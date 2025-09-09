# Unit Tests for GraphJudge Chat Module

This directory contains unit tests for the GraphJudge chat module, specifically testing the OpenAI API configuration functionality.

## Setup

### Prerequisites

1. **Python Dependencies**: Install the required testing dependencies:
   ```bash
   pip install pytest pytest-mock
   ```

2. **Environment Variables**: The configuration system uses environment variables for security. Before running scripts or tests, set up your API credentials:

   ```bash
   # Required
   export OPENAI_API_KEY="your_actual_openai_api_key_here"
   
   # Optional (defaults to https://api.openai.com/v1)
   export OPENAI_API_BASE="https://api.openai.com/v1"
   ```

   Or create a `.env` file in the chat directory using the provided `env_example.txt` as a template.

## Running Tests

### Run All Tests
```bash
# From the unit_test directory
pytest test_config.py -v

# Or run from the chat directory
pytest unit_test/test_config.py -v
```

### Run Specific Test Classes
```bash
# Test only the main config function
pytest test_config.py::TestGetApiConfig -v

# Test only validation functions
pytest test_config.py::TestValidateApiConfig -v

# Test only individual getter functions
pytest test_config.py::TestIndividualGetters -v
```

### Run with Coverage (if pytest-cov is installed)
```bash
pip install pytest-cov
pytest test_config.py --cov=config --cov-report=html
```

## Test Structure

### `test_config.py`

This file contains comprehensive tests for the `config.py` module:

- **TestGetApiConfig**: Tests for the main configuration loading function
  - Valid environment variable loading
  - Default value handling
  - Error handling for missing/empty keys
  - URL formatting and standardization

- **TestValidateApiConfig**: Tests for configuration validation
  - Valid configuration detection
  - Invalid configuration detection
  - Edge cases (short keys, malformed URLs)

- **TestIndividualGetters**: Tests for convenience functions
  - `get_api_key()` function
  - `get_api_base()` function
  - Error handling

- **TestIntegration**: Integration and edge case tests
  - Module standalone execution
  - Environment variable isolation

## Expected Test Results

When all tests pass, you should see output similar to:
```
test_config.py::TestGetApiConfig::test_get_api_config_with_valid_env_vars PASSED
test_config.py::TestGetApiConfig::test_get_api_config_with_default_base_url PASSED
test_config.py::TestGetApiConfig::test_get_api_config_missing_api_key PASSED
...
========================= 20 passed in 0.15s =========================
```

## Configuration Module Features Tested

1. **Environment Variable Loading**: Validates that API credentials are properly loaded from environment variables
2. **Error Handling**: Ensures appropriate errors are raised for missing or invalid configuration
3. **URL Formatting**: Tests automatic formatting of API base URLs
4. **Validation Functions**: Tests configuration validation without raising exceptions
5. **Security**: Ensures no hardcoded credentials are used

## Troubleshooting

### Common Issues

1. **Import Errors**: If you get import errors, ensure you're running tests from the correct directory and that the parent directory is in the Python path.

2. **Environment Variable Issues**: The tests use mocking to avoid requiring actual API keys, but if you see environment-related errors, ensure your test environment is clean.

3. **Pytest Not Found**: Install pytest with `pip install pytest`

### Running Individual Tests

You can run specific tests for debugging:
```bash
# Test only API key validation
pytest test_config.py::TestGetApiConfig::test_get_api_config_missing_api_key -v

# Test only URL formatting
pytest test_config.py::TestGetApiConfig::test_api_base_url_formatting -v
```

## Security Notes

- Tests use mocked environment variables to avoid requiring real API keys
- The configuration system is designed to prevent hardcoded credentials
- All API keys should be stored in environment variables or `.env` files
- Never commit actual API keys to version control 
