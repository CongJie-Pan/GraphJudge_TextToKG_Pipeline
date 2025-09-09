# GraphJudge Phase - Modularized Perplexity API Graph Judge System

## Overview

GraphJudge Phase is a modular knowledge graph triple validation system that uses Perplexity API's sonar-reasoning model for graph judgment. The system has been refactored from the original single-file architecture to a modular design, providing better maintainability, testability, and extensibility.

## Module Architecture

```
graphJudge_Phase/
├── __init__.py                    # Module initialization and exports
├── config.py                      # Configuration constants and settings
├── data_structures.py             # Data structure definitions
├── logging_system.py              # Logging system
├── graph_judge_core.py            # Core GraphJudge class
├── prompt_engineering.py          # Prompt engineering
├── gold_label_bootstrapping.py    # Gold label bootstrapping functionality
├── processing_pipeline.py         # Processing pipeline
├── utilities.py                   # Utility functions
├── main.py                        # Main entry point
└── README.md                      # This document
```

## Main Features

### 1. Graph Judgment
- Knowledge graph triple validation using Perplexity API
- Support for standard and explainable modes
- Concurrent processing and rate limiting
- Error handling and retry mechanisms

### 2. Gold Label Bootstrapping
- Two-stage automatic label assignment
- RapidFuzz string similarity matching
- LLM semantic evaluation
- Manual review sampling

### 3. Explainable Reasoning
- Detailed reasoning process
- Confidence scoring
- Evidence source tracking
- Alternative suggestion generation

## Installation and Setup

### 1. Environment Requirements
```bash
# Python 3.7+
pip install litellm python-dotenv
pip install rapidfuzz  # Optional, for gold label bootstrapping
pip install datasets   # Optional, for dataset processing
```

### 2. Environment Variables
```bash
export PERPLEXITYAI_API_KEY="your_perplexity_api_key_here"
```

### 3. Optional Environment Variables
```bash
export PIPELINE_ITERATION="2"
export PIPELINE_INPUT_FILE="path/to/input.json"
export PIPELINE_OUTPUT_FILE="path/to/output.csv"
```

## Usage

### 1. As a Module Import

```python
from graphJudge_Phase import (
    PerplexityGraphJudge,
    GoldLabelBootstrapper,
    ProcessingPipeline
)

# Initialize graph judge
judge = PerplexityGraphJudge(
    model_name="perplexity/sonar-reasoning",
    reasoning_effort="medium",
    enable_console_logging=True
)

# Perform graph judgment
result = await judge.judge_graph_triple("Is this true: 曹雪芹 創作 紅樓夢 ?")
print(result)  # "Yes" or "No"

# Explainable judgment
explainable_result = await judge.judge_graph_triple_with_explanation(
    "Is this true: 曹雪芹 創作 紅樓夢 ?"
)
print(f"Judgment: {explainable_result.judgment}")
print(f"Confidence: {explainable_result.confidence}")
print(f"Reasoning: {explainable_result.reasoning}")
```

### 2. Gold Label Bootstrapping

```python
from graphJudge_Phase import GoldLabelBootstrapper

# Initialize bootstrapper
bootstrapper = GoldLabelBootstrapper(judge)

# Run gold label bootstrapping
success = await bootstrapper.bootstrap_gold_labels(
    triples_file="path/to/triples.txt",
    source_file="path/to/source.txt",
    output_file="path/to/output.csv"
)
```

### 3. Processing Pipeline

```python
from graphJudge_Phase import ProcessingPipeline

# Initialize processing pipeline
pipeline = ProcessingPipeline(judge)

# Process instructions
stats = await pipeline.process_instructions(
    data_eval=dataset,
    explainable_mode=True,
    reasoning_file_path="path/to/reasoning.json"
)
```

### 4. Command Line Usage

#### Navigate to the parent directory
```bash
cd Miscellaneous\KgGen\GraphJudge\chat
```

#### Standard Graph Judgment Mode
```bash
python -m graphJudge_Phase.main
```

#### Explainable Mode(Use this to get more detail)
```bash
python -m graphJudge_Phase.main --explainable
```

#### Gold Label Bootstrapping Mode
```bash
python -m graphJudge_Phase.main --bootstrap \
    --triples-file ../datasets/triples.txt \
    --source-file ../datasets/source.txt \
    --output ../datasets/gold_bootstrap.csv \
    --threshold 0.8 \
    --sample-rate 0.15
```

#### Custom Model and Parameters
```bash
python -m graphJudge_Phase.main \
    --model sonar-reasoning-pro \
    --reasoning-effort high \
    --explainable \
    --verbose
```

## Configuration Options

### Perplexity API Configuration
```python
# Modify in config.py
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"
PERPLEXITY_CONCURRENT_LIMIT = 3
PERPLEXITY_RETRY_ATTEMPTS = 3
PERPLEXITY_BASE_DELAY = 0.5
PERPLEXITY_REASONING_EFFORT = "medium"
```

### Gold Label Bootstrapping Configuration
```python
GOLD_BOOTSTRAP_CONFIG = {
    'fuzzy_threshold': 0.8,      # RapidFuzz similarity threshold
    'sample_rate': 0.15,         # Manual review sampling rate
    'llm_batch_size': 10,        # LLM batch size
    'max_source_lines': 1000,    # Maximum source text lines
    'random_seed': 42            # Random seed
}
```

## Data Structures

### TripleData
```python
class TripleData(NamedTuple):
    subject: str          # Subject
    predicate: str        # Predicate/Relation
    object: str          # Object
    source_line: str     # Source text line
    line_number: int     # Line number
```

### ExplainableJudgment
```python
class ExplainableJudgment(NamedTuple):
    judgment: str                    # Judgment result ("Yes" or "No")
    confidence: float               # Confidence (0.0-1.0)
    reasoning: str                  # Detailed reasoning
    evidence_sources: List[str]     # Evidence sources
    alternative_suggestions: List[Dict]  # Alternative suggestions
    error_type: Optional[str]       # Error type
    processing_time: float          # Processing time
```

### BootstrapResult
```python
class BootstrapResult(NamedTuple):
    triple: TripleData              # Triple
    source_idx: int                 # Best matching source index
    fuzzy_score: float             # Fuzzy matching score
    auto_expected: Optional[bool]   # Auto-expected value
    llm_evaluation: Optional[str]   # LLM evaluation result
    expected: Optional[bool]        # Final expected value
    note: str                      # Note
```

## Backward Compatibility

To ensure compatibility with existing code, we provide the `run_gj_compatibility.py` file:

```python
# Old import method still works
import run_gj_compatibility as run_gj

# Usage remains the same
result = await run_gj.get_perplexity_completion("Is this true: Test ?")
```

## Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_graph_judge_core.py -v
```

### Integration Tests
```bash
# Test complete pipeline
python -m graphJudge_Phase.main --explainable --verbose
```

## Performance Optimization

### 1. Concurrency Control
- Default concurrency limit: 3 requests
- Adjustable via `PERPLEXITY_CONCURRENT_LIMIT`

### 2. Batch Processing
- Gold label bootstrapping supports batch processing
- Configurable via `llm_batch_size`

### 3. Caching
- Response caching (optional)
- Local result storage

## Error Handling

### 1. API Errors
- Automatic retry mechanism
- Exponential backoff strategy
- Error classification and logging

### 2. Data Validation
- Input format validation
- File existence checks
- Data integrity validation

### 3. Fallback Mechanisms
- Mock mode support
- Graceful degradation
- Error recovery

## Logging and Monitoring

### 1. Logging System
```python
from graphJudge_Phase import setup_terminal_logging, TerminalLogger

# Set up logging
log_filepath = setup_terminal_logging()
logger = TerminalLogger(log_filepath)

# Log information
logger.log_info("Processing started")
logger.log_error(error, "API call failed")
```

### 2. Statistics
```python
# Processing statistics
stats = ProcessingStatistics(
    total_instructions=100,
    successful_responses=95,
    error_responses=5,
    yes_judgments=60,
    no_judgments=35,
    success_rate=95.0,
    positive_rate=63.2,
    avg_confidence=0.85,
    unique_error_types=2
)
```

## Extension Development

### 1. Adding New Model Support
```python
# Add new models in config.py
PERPLEXITY_MODELS = {
    "sonar-pro": "perplexity/sonar-pro",
    "sonar-reasoning": "perplexity/sonar-reasoning",
    "sonar-reasoning-pro": "perplexity/sonar-reasoning-pro",
    "new-model": "perplexity/new-model"  # New model
}
```

### 2. Custom Prompts
```python
# Add new methods in prompt_engineering.py
class PromptEngineer:
    @staticmethod
    def create_custom_prompt(instruction: str) -> str:
        # Custom prompt logic
        pass
```

### 3. New Data Structures
```python
# Add new structures in data_structures.py
class CustomResult(NamedTuple):
    # Custom result structure
    pass
```

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Solution: Check PERPLEXITYAI_API_KEY environment variable
   ```

2. **Import Error**
   ```
   Solution: Ensure all dependencies are installed
   pip install litellm python-dotenv
   ```

3. **Insufficient Memory**
   ```
   Solution: Reduce batch size or concurrency limit
   ```

4. **Rate Limiting**
   ```
   Solution: Increase delay time or reduce concurrency
   ```

### Debug Mode
```bash
# Enable verbose logging
python -m graphJudge_Phase.main --verbose --explainable

# Disable file logging
python -m graphJudge_Phase.main --no-logging
```

## Contributing

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Version History

- **v2.0.0**: Modular Refactoring
  - Split single file into multiple modules
  - Improved error handling and logging
  - Added backward compatibility support
  - Enhanced configuration management

- **v1.0.0**: Initial Version
  - Basic graph judgment functionality
  - Perplexity API integration
  - Gold label bootstrapping

## License

This project is licensed under the MIT License.

## Contact Information

For questions or suggestions, please contact the development team.
