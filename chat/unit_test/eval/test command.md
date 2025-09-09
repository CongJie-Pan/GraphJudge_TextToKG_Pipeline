# Comprehensive Test Suite for GraphJudge Evaluation

## Updated Test Commands (2025-01-25)

### 1. Run All Tests
```bash
# Run all evaluation tests with detailed output
pytest . -v --tb=short

# Run with coverage report
pytest . -v --cov=. --cov-report=html
```

### 2. Individual Test Files

#### Core Evaluation Tests
```bash
# Main eval.py functionality tests
pytest test_eval.py -v

# Data preprocessing tests  
pytest test_eval_preDataProcess.py -v

# Device detection and integration tests
pytest test_eval_device_integration.py -v
```

### 3. Specific Test Categories

#### Device Detection Tests
```bash
# Test CUDA/CPU device detection for BERTScore
pytest test_eval.py::TestDeviceDetection -v

# Integration tests for device detection
pytest test_eval_device_integration.py::TestEvalDeviceIntegration -v
```

#### Error Handling Tests
```bash
# Test error handling and edge cases
pytest test_eval.py::TestGraphMatchingErrorHandling -v
```

#### End-to-End Tests
```bash
# Test complete pipeline from CSV to evaluation
pytest test_eval.py::TestEvalIntegration -v
```

### 4. Performance Tests

#### Quick Tests (CPU only, no BERTScore)
```bash
# Run tests with mocked BERTScore for speed
pytest test_eval.py::TestMainEvaluation::test_main_evaluation_pipeline -v
```

#### Full Integration Tests
```bash
# Test actual eval.py execution (slower but comprehensive)  
pytest test_eval_device_integration.py -v -s
```

### 5. Test with Different Hardware Configurations

#### Force CPU Testing
```bash
# Test with CUDA disabled (simulates CPU-only environment)
CUDA_VISIBLE_DEVICES="" pytest test_eval_device_integration.py -v
```

#### Mock GPU Testing
```bash
# Test device detection logic without actual GPU
pytest test_eval.py::TestDeviceDetection -v
```

### 6. Generate Test Reports

#### JSON Report
```bash
pytest . -v --json-report --json-report-file=test_report.json
```

#### HTML Coverage Report
```bash
pytest . --cov=. --cov-report=html --cov-report-dir=coverage_html
```

### 7. Test Data Requirements

The tests use temporary directories and mock data, but for integration tests ensure:
- Python packages: `pytest`, `numpy`, `torch` (optional), `bert-score`, `rouge-score`, `spacy`, `nltk`
- No external files required (tests create their own data)

### 8. Common Issues and Solutions

#### Import Errors
```bash
# If module import fails, run from correct directory:
cd /path/to/2025-IM-senior-project/Miscellaneous/KgGen/GraphJudge/chat/unit_test/eval
python -m pytest . -v
```

#### CUDA/Device Issues
```bash
# If CUDA tests fail on CPU-only machines:
pytest . -v -k "not cuda and not gpu"
```

#### Timeout Issues
```bash
# For slower machines, increase timeout:
pytest . -v --timeout=300
```

---

## Legacy Command (still works)
```bash
pytest test_eval_preDataProcess.py --cov=. --cov-report=json:coverage.json --cov-report=term -v
```
