# GraphJudge Phase Modular Test Suite

## 📋 Overview

This is the complete test suite for the GraphJudge Phase modular system, ensuring full compatibility with the original `run_gj.py` functionality while testing each component of the modular architecture.

## 🏗️ Test Structure

### Test Module Organization

tests/
├── init.py # Test suite initialization
├── conftest.py # Shared test configuration and fixtures
├── test_config.py # Configuration module tests
├── test_data_structures.py # Data structures tests
├── test_graph_judge_core.py # Core judgment logic tests
├── test_prompt_engineering.py # Prompt engineering tests
├── test_gold_label_bootstrapping.py # Gold label bootstrapping tests
├── test_utilities.py # Utility functions tests
├── test_processing_pipeline.py # Processing pipeline tests
├── test_integration.py # Integration tests
├── run_tests.py # Test runner
└── README.md # This document


## 📊 Test Coverage

### Correspondence to original `test_run_gj.py`

| Original Test Class               | Corresponding Modular Test           | Test File                             |
|-----------------------------------|--------------------------------------|---------------------------------------|
| `TestPerplexityGraphJudge`        | Core judgment functionality tests    | `test_graph_judge_core.py`            |
| `TestPerplexityCompletion`        | Processing pipeline tests            | `test_processing_pipeline.py`         |
| `TestInputValidation`             | Input validation tests               | `test_utilities.py`                   |
| `TestExplainableOutputHandling`   | Explainable output tests             | `test_prompt_engineering.py`          |
| `TestResponseProcessing`          | Response handling tests              | `test_processing_pipeline.py`         |
| `TestErrorHandling`               | Error handling tests                 | `test_integration.py`                 |
| `TestIntegration`                 | Integration tests                    | `test_integration.py`                 |
| Gold label bootstrapping tests    | Bootstrapping system tests           | `test_gold_label_bootstrapping.py`    |

### Test Feature Completeness

#### ✅ Core Functionality Tests (`test_graph_judge_core.py`)
- ✅ PerplexityGraphJudge initialization  
- ✅ Basic graph judgment (mock and real modes)  
- ✅ Explainable judgment  
- ✅ Streaming judgment  
- ✅ Citation handling  
- ✅ Error handling and retry logic  
- ✅ Compatibility across judgment modes  

#### ✅ Prompt Engineering Tests (`test_prompt_engineering.py`)
- ✅ Basic judgment prompt creation  
- ✅ Explainable judgment prompt creation  
- ✅ Response parsing (Yes/No classification)  
- ✅ Explainable response parsing  
- ✅ Citation extraction and handling  
- ✅ HTML cleaning and text processing  
- ✅ Multilingual content handling  

#### ✅ Gold Label Bootstrapping Tests (`test_gold_label_bootstrapping.py`)
- ✅ Triple data loading  
- ✅ Source text processing  
- ✅ Stage 1: RapidFuzz string matching  
- ✅ Stage 2: LLM semantic evaluation  
- ✅ Sampling uncertain cases  
- ✅ Result saving and statistics  
- ✅ Full bootstrapping workflow  

#### ✅ Processing Pipeline Tests (`test_processing_pipeline.py`)
- ✅ Standard processing mode  
- ✅ Explainable processing mode  
- ✅ Concurrency control and rate limiting  
- ✅ Statistics calculation  
- ✅ File operations (CSV and JSON)  
- ✅ Error handling  

#### ✅ Data Structures Tests (`test_data_structures.py`)
- ✅ `TripleData` structure  
- ✅ `BootstrapResult` structure  
- ✅ `ExplainableJudgment` structure  
- ✅ Citation-related structures  
- ✅ Processing and statistics structures  
- ✅ JSON serialization compatibility  

#### ✅ Utilities Tests (`test_utilities.py`)
- ✅ File validation  
- ✅ Directory operations  
- ✅ Environment validation  
- ✅ Instruction format validation  
- ✅ Text processing  
- ✅ File operations  
- ✅ System utilities  

#### ✅ Configuration Tests (`test_config.py`)
- ✅ Configuration constants validation  
- ✅ Environment variable handling  
- ✅ File path generation  
- ✅ Pipeline integration  

#### ✅ Integration Tests (`test_integration.py`)
- ✅ Module-to-module integration  
- ✅ End-to-end workflows  
- ✅ Compatibility with original functionality  
- ✅ Error handling integration  
- ✅ Performance tests  
- ✅ Cross-module data flow  

## 🚀 Running the Tests

### Basic Usage

```bash
# Run all tests
python tests/run_tests.py

# Verbose output
python tests/run_tests.py -v

# Include coverage report
python tests/run_tests.py --coverage

# Run a specific test file
python tests/run_tests.py -t test_config.py

# Verbose + coverage
python tests/run_tests.py -v --coverage
```

### Using pytest Directly

```bash
# Run all tests
pytest tests/ -v

# Run a specific test
pytest tests/test_graph_judge_core.py -v

# Include coverage
pytest tests/ --cov=graphJudge_Phase --cov-report=html
```

## 📁 Test Output

Running the tests will generate a structured output directory:

test_results/
└── run_YYYYMMDD_HHMMSS/
├── reports/
│ ├── html_report.html # HTML test report
│ ├── junit_report.xml # JUnit XML report
│ └── test_summary.json # Test summary
├── logs/ # Test logs
├── coverage/ # Coverage reports
│ └── html/ # HTML coverage report
└── artifacts/ # Test artifacts


## 🧪 Test Types

### Unit Tests
- Isolated tests for each module  
- Mocked and isolated scenarios  
- Boundary and error case coverage  

### Integration Tests
- Inter-module interaction tests  
- End-to-end workflow tests  
- Data flow consistency tests  

### Compatibility Tests
- Compatibility with original `run_gj.py` output format  
- Environment variable handling compatibility  
- API behavior consistency  

### Performance Tests
- Batch processing performance  
- Memory usage stability  
- Concurrency efficiency  

## 📝 Test Development Guide

### Adding New Tests

1. Choose or create the appropriate test file  
2. Use shared fixtures from `conftest.py`  
3. Follow existing naming conventions  
4. Include docstrings describing each test  
5. Cover both positive and negative cases  

### Test Naming Convention

```python
class TestModuleName:
    """Test cases for ModuleName functionality."""

    def test_function_specific_case(self):
        """Test specific behavior of function."""
        pass

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async function behavior."""
        pass
```

### Mocking Guidelines

- Use `mock_async_perplexity_judge` fixture for async tests  
- Use `MockPerplexityResponse` to simulate API responses  
- Use `PerplexityTestBase` for full setup and teardown  

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you run tests from the `tests/` directory with correct environment variables.  
2. **API errors**: Tests use mock mode—no real API key needed.  
3. **File permission issues**: Ensure write permissions for test directories.  
4. **Missing dependencies**: Install test requirements:  
   ```bash
   pip install pytest pytest-cov pytest-html pytest-asyncio
   ```

### Debugging Tips

- Use `-v` and `-s` flags for detailed output  
- Insert `pytest.set_trace()` to drop into a debugger  
- Check generated reports for failure details  

## ✅ Validation Checklist

Before committing, ensure:

- [ ] All tests pass  
- [ ] Coverage meets target (>90%)  
- [ ] New features have corresponding tests  
- [ ] Test documentation is up to date  
- [ ] Compatibility with original functionality verified  

## 📋 Test Specification Comparison

### Coverage vs. original `test_run_gj.py`

| Test Category               | Original Count | Modular Count | Status      |
|-----------------------------|----------------|---------------|-------------|
| Core judgment functionality| 15             | 18            | ✅ Extended |
| Explainable judgment        | 12             | 15            | ✅ Extended |
| Input validation            | 8              | 12            | ✅ Extended |
| File handling               | 6              | 10            | ✅ Extended |
| Error handling              | 5              | 8             | ✅ Extended |
| Integration tests           | 4              | 6             | ✅ Extended |
| Gold label bootstrapping    | 8              | 12            | ✅ Extended |
| **Total**                   | **58**         | **81**        | ✅ Full coverage & extension |

The modular test suite not only covers all original tests but also adds extra cases to ensure the correctness and stability of the modular architecture.