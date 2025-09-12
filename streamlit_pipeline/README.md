# GraphJudge Streamlit Pipeline

A refactored, modular implementation of the GraphJudge text-to-knowledge-graph pipeline optimized for Streamlit integration. This project extracts and refactors core functionality from the main GraphJudge system into clean, testable modules suitable for web application deployment.

## Quick Start

### Prerequisites

- Python 3.8+ (tested on 3.8-3.11)
- API keys for OpenAI (or Azure OpenAI) and Perplexity

### Installation

1. **Clone the repository and navigate to the streamlit pipeline:**
   ```bash
   cd streamlit_pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API credentials:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

4. **Run tests to verify setup:**
   ```bash
   python run_tests.py --coverage
   ```

## Project Structure

```
streamlit_pipeline/
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── .env.example             # Environment configuration template
├── run_tests.py             # Test runner script (entry point)
├── pytest.ini              # Test framework configuration
│
├── core/                    # Core pipeline modules
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── models.py           # Data models (Triple, EntityResult, etc.)
│   └── entity_processor.py # Entity extraction and text denoising
│
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── api_client.py       # LLM API client wrapper
│   └── validation.py       # Input validation and sanitization
│
└── tests/                   # Test suite
    ├── __init__.py
    ├── conftest.py         # Pytest fixtures and configuration
    ├── test_models.py      # Data model tests
    ├── test_config.py      # Configuration tests
    ├── test_validation.py  # Validation tests
    ├── test_integration.py # Integration tests
    ├── test_utils.py       # Testing utilities
    └── fixtures/           # Test data and mock API responses
        └── api_fixtures.py
```

### Why is `run_tests.py` in the Main Folder?

The test runner script (`run_tests.py`) is located in the main project folder rather than in `/tests` for several important reasons:

- **Entry Point Convention**: Test runners are entry points, not test files themselves, and belong at the package root level
- **Path Resolution**: Being at the root simplifies import paths and working directory management
- **Industry Standard**: Most projects (Django, Node.js, etc.) place test runners at the root level
- **CI/CD Friendly**: Automation systems expect test runners at predictable root locations
- **User Experience**: Developers expect to run `python run_tests.py` from the main project directory

The `/tests` folder contains only actual test files (`test_*.py`), fixtures, and test utilities, maintaining clear separation of concerns.

## Configuration

### API Setup

The pipeline requires API keys for:
1. **OpenAI GPT-5-mini** (for entity extraction and text denoising)
2. **Perplexity Sonar** (for graph judgment)

Configure using `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys (choose one option):

# Option 1: Azure OpenAI (Recommended for enterprise)
AZURE_OPENAI_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Option 2: Standard OpenAI
OPENAI_API_KEY=your_openai_key_here

# Required: Perplexity for graph judgment
PERPLEXITY_API_KEY=your_perplexity_key_here
```

### Model Configuration

The pipeline uses these default models:
- **Entity Extraction**: GPT-5-mini (temperature: 0.0, max_tokens: 4000)
- **Triple Generation**: GPT-5-mini (deterministic output)
- **Graph Judgment**: Perplexity Sonar Reasoning

Settings can be customized in `core/config.py`.

## Testing

### Test Framework

The project uses **pytest** with comprehensive testing including:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-module interaction testing  
- **Mock API Tests**: Simulated API responses for reliable testing
- **Performance Tests**: Timing and scalability validation
- **Coverage Reporting**: 90%+ code coverage requirement

### Running Tests

```bash
# Run all unit tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run specific test types
python run_tests.py --integration  # Include integration tests
python run_tests.py --performance  # Include performance tests
python run_tests.py --smoke       # Quick smoke tests only

# Run with different options
python run_tests.py --verbose     # Detailed output
python run_tests.py --fail-fast   # Stop on first failure
python run_tests.py --parallel 4  # Run in parallel with 4 processes

# Generate HTML coverage report
python run_tests.py --html-coverage

# Run specific test pattern
python run_tests.py --pattern "test_config"
```

### Test Structure

- **Test Files**: Located in `/tests` folder, following `test_*.py` naming
- **Fixtures**: Shared test data and mocks in `tests/fixtures/`
- **Configuration**: `pytest.ini` and `conftest.py` for test setup
- **Utilities**: Common testing patterns in `tests/test_utils.py`

## Core Components

### Entity Processor (`core/entity_processor.py`)
- Extracts named entities from input text
- Performs text denoising based on identified entities
- Handles Chinese and English text processing
- Async operation with error handling and retry logic

### Data Models (`core/models.py`)
- **Triple**: Represents knowledge graph relationships (subject-predicate-object)
- **EntityResult**: Encapsulates entity extraction results
- **TripleResult**: Contains generated triples with metadata
- **JudgmentResult**: Holds graph judgment decisions with confidence scores

### API Client (`utils/api_client.py`)
- Unified interface for multiple LLM APIs (OpenAI, Azure OpenAI, Perplexity)
- Rate limiting and error handling
- Retry mechanisms with exponential backoff
- Response validation and parsing

### Validation System (`utils/validation.py`)
- Input text validation (length, content, encoding)
- Data model validation using Pydantic
- Security checks for malicious content
- Cross-platform compatibility validation

## Pipeline Workflow

1. **Input Validation**: Text preprocessing and sanitization
2. **Entity Extraction**: Identify key entities using GPT-5-mini
3. **Text Denoising**: Clean and structure text based on entities
4. **Triple Generation**: Create knowledge graph relationships
5. **Graph Judgment**: Validate triples using Perplexity reasoning
6. **Output Generation**: Return filtered, high-quality knowledge graph

## Integration with Streamlit

This pipeline is designed for easy Streamlit integration:

```python
import asyncio
from core.entity_processor import EntityProcessor
from core.models import EntityResult

# Initialize processor
processor = EntityProcessor()

# Process text (async)
async def process_text(text: str) -> EntityResult:
    return await processor.extract_entities(text)

# Streamlit app integration
result = asyncio.run(process_text(user_input))
```

## Development

### Code Quality Standards

- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings (Google style)
- **Error Handling**: Graceful failure with meaningful messages
- **Testing**: TDD approach with 90%+ coverage requirement
- **Security**: Input validation and API key protection

### Adding New Features

1. **Write Tests First**: Follow TDD principles
2. **Update Models**: Add/modify data structures in `core/models.py`
3. **Implement Logic**: Add functionality in appropriate module
4. **Add Validation**: Update validation rules if needed
5. **Update Tests**: Ensure comprehensive test coverage
6. **Update Documentation**: Keep README and docstrings current

### Performance Guidelines

- **Async Operations**: Use async/await for I/O operations
- **Rate Limiting**: Respect API limits and implement backoff
- **Caching**: Consider caching for repeated operations
- **Memory Management**: Handle large text inputs efficiently

## Monitoring and Logging

The pipeline includes comprehensive logging:
- API request/response logging
- Performance metrics tracking
- Error reporting with context
- Processing time measurements

Configure logging levels through environment variables or `core/config.py`.

## Security Considerations

- **API Key Management**: Use environment variables, never commit keys
- **Input Validation**: All user input is validated and sanitized  
- **Rate Limiting**: Implemented to prevent API abuse
- **Error Handling**: Sensitive information is not exposed in error messages

## Troubleshooting

### Common Issues

1. **"No valid API configuration found"**
   - Check `.env` file exists and has correct keys
   - Verify API keys are valid and have sufficient quota
   - For Azure OpenAI, ensure both KEY and ENDPOINT are set

2. **Import Errors**
   - Run from the `streamlit_pipeline` directory
   - Check Python path includes current directory
   - Verify all dependencies are installed

3. **Test Failures**
   - Run `python run_tests.py --verbose` for detailed output
   - Check API connectivity if integration tests fail
   - Clear pytest cache: `rm -rf .pytest_cache __pycache__`

4. **Performance Issues**
   - Monitor API rate limits
   - Use `--performance` flag to run performance tests
   - Consider async processing for batch operations

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python run_tests.py --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests first (TDD approach)
4. Implement your feature
5. Ensure tests pass: `python run_tests.py --coverage`
6. Update documentation as needed
7. Submit a pull request

### Code Style

- Follow PEP 8 style guidelines
- Use type hints throughout
- Write comprehensive docstrings
- Maintain test coverage above 90%

## License

This project is part of the GraphJudge research system. See the main repository for license information.

## Acknowledgments

- Built on the foundation of the original GraphJudge system
- Utilizes OpenAI GPT models for entity extraction
- Leverages Perplexity AI for graph reasoning and judgment

---

**For more information about the broader GraphJudge project, see the main repository README.**