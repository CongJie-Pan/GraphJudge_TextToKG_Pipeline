# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
- Verify the file status before editing
- If encountering storage issues, please reload the file

## Repository Overview

This is a research codebase for **GraphJudge**, a system that uses LLMs to construct and evaluate knowledge graphs. The project implements a multi-stage pipeline for text-to-KG conversion with LLM-based graph judgment capabilities.

## Key Commands

### Core Pipeline Execution
```bash
# ECTD Stage: Entity extraction and text denoising
python chat/run_entity.py

# Triple Generation Stage: Generate knowledge graph triples
python chat/run_triple.py

# Graph Judge Stage: Evaluate and filter generated triples
python chat/run_gj.py

# Evaluation: Assess final knowledge graph quality
cd graph_evaluation && bash eval.sh
```

### LoRA Fine-tuning for Graph Judgment
```bash
cd graph_judger
python lora_finetune_genwiki_context.py    # For GenWiki dataset
python lora_finetune_rebel_context.py      # For REBEL dataset
python lora_finetune_scierc_context.py     # For SciERC dataset
```

### Model Inference
```bash
cd graph_judger
python lora_infer_batch.py          # Batch inference with fine-tuned model
python lora_infer.py                # Single inference
python lora_infer_batch_naive.py    # Naive baseline inference
```

### Testing
```bash
# Run specific test modules
python -m pytest chat/unit_test/test_*.py
python -m pytest chat/extractEntity_Phase/tests/
python -m pytest chat/graphJudge_Phase/tests/

# Run comprehensive test suites
python chat/unit_test/run_ectd_tests.py
```

## Architecture Overview

### Pipeline Stages (ECTD → Triple Generation → Graph Judge)

1. **ECTD Stage** (`chat/run_entity.py`, `chat/extractEntity_Phase/`):
   - **Entity Extraction**: Identifies key entities from input text
   - **Text Denoising**: Cleans and restructures text based on extracted entities
   - Uses GPT-5-mini model via LiteLLM for Chinese text processing

2. **Triple Generation Stage** (`chat/run_triple.py`):
   - Generates RDF-style knowledge graph triples from processed text
   - Supports JSON output with schema validation
   - Implements text chunking for large inputs and relation vocabulary standardization

3. **Graph Judge Stage** (`chat/run_gj.py`, `chat/graphJudge_Phase/`):
   - Evaluates correctness of generated triples using LLM judgment
   - Supports multiple APIs (Perplexity, Gemini, KIMI)
   - Includes explainable reasoning and confidence scoring

### Key Directories

- **`chat/`**: Main pipeline scripts and modularized components
  - `run_*.py`: Main execution scripts for each pipeline stage
  - `extractEntity_Phase/`: Modular entity extraction system
  - `graphJudge_Phase/`: Modular graph judgment system
  - `unit_test/`: Comprehensive test suite

- **`graph_judger/`**: LoRA fine-tuning and inference for specialized graph judgment models
  - `lora_finetune_*.py`: Dataset-specific fine-tuning scripts
  - `lora_infer*.py`: Inference scripts for trained models

- **`graph_evaluation/`**: Evaluation metrics and benchmarking
  - `metrics/eval.py`: Core evaluation logic
  - `eval.sh`: Evaluation runner script

- **`datasets/`**: Training data and results storage
- **`config/`**: Pipeline configuration files
- **`tools/`**: Utility scripts for data processing

### Configuration System

The pipeline uses YAML-based configuration (`config/pipeline_config.yaml`) with:
- Stage-specific settings for ECTD, triple generation, and graph judgment
- API rate limiting and error handling configuration
- Quality assurance and monitoring settings

### Data Flow

1. **Input**: Raw text (supports Chinese classical literature like 紅樓夢)
2. **ECTD Processing**: Entities extracted → Text denoised
3. **Triple Generation**: Structured knowledge graph triples in JSON format
4. **Graph Judgment**: LLM-based triple validation with reasoning
5. **Output**: Filtered, high-quality knowledge graph

## Development Notes

- The codebase primarily uses Python with async processing patterns
- Supports multiple LLM APIs through LiteLLM integration
- Implements comprehensive caching and rate limiting
- Uses Pydantic for data validation and PyTest for testing
- Configuration is centralized in YAML files with environment-specific overrides
- Results are stored in timestamped iteration directories under `datasets/`

## Working with Tests

- Test files follow the pattern `test_*.py` and are distributed across component directories
- Use `python -m pytest` for running tests with proper module resolution
- Some tests require API keys to be configured in environment variable

## Coding Guidelines

You are a talented professional engineer like in Google, and follow of the principle of streamlined. Please adhere to the following guidelines:

### 🗨️ Communication & Language
- Respond to general user messages in Traditional Chinese (繁體中文).
- All code-including comments, variable names, function names, logs, and documentation-must be written in English.
- Provide detailed comments for each code block, explaining the purpose and logic of every function, class, and significant code section.
- Include thorough explanations of why specific approaches were chosen.
- All error logging, debugging information, and technical documentation should be in English.
- User interface text should be in Traditional Chinese when specified.
- Ensure code is well-structured, follows best practices, and includes comprehensive complete error handling with clear English error messages.

### 🔄 Project Awareness & Context
- Always read `PLANNING.md` at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- Check `TASK.md` before starting a new task. If the task isn’t listed, add it with a brief description and today’s date.
- Use consistent naming conventions, file structure, and architecture patterns as described in `PLANNING.md`.

### 🧱 Code Structure & Modularity
- Never create a file longer than 500 lines of code. If a file approaches this limit, refactor by splitting it into modules or helper files.
- Organize code into clearly separated modules, grouped by feature or responsibility.
- Use clear, consistent imports (prefer relative imports within packages).

### 🧪 Testing & Reliability
- **Write tests first, implement later** - Follow TDD principles to ensure interface design consistency
- Define expected behavior through tests before writing implementation code
- Use tests as living documentation of system requirements and expected behavior

Architectural Consistency
- **Ensure alignment between tests and implementation** - Tests must use the same calling patterns as the actual implementation
- Document class relationships clearly (composition vs inheritance vs static calls) in architecture design
- Conduct code review checkpoints to verify test-implementation consistency
- If tests expect `object.attribute.method()`, implementation must provide that exact interface
- If implementation uses static methods `Class.method()`, tests should mirror this pattern

Test Coverage Requirements
- **Always create unit tests for new features** (functions, classes, routes, etc)
- **Update existing tests when logic changes** - After any logic modification, verify and update related tests
- Tests should live in a `/tests` folder mirroring the main application structure

Minimum Test Cases
Each feature must include at least:
- **1 test for expected use case** - Normal, successful operation
- **1 edge case test** - Boundary conditions, unusual but valid inputs
- **1 failure case test** - Invalid inputs, error conditions, exception handling

est Design Best Practices

Mock and Test-Friendly Design
- **Use module-level variables for aligned test mock paths** - Ensure mocks target the correct import paths
- **Access and call objects meaningfully** - Don't just import to confirm existence; actually invoke methods to match mock design
- **Trigger actual mock behavior** - Use real calls/await statements to activate mock responses
- **Design for testability** - Structure code to allow easy mocking and testing

Cross-Platform Consistency
- **Path checking independence** - Don't rely on OS-specific path interpretation
- **Explicitly reject special paths** for testing purposes
- Ensure tests behave identically across different operating systems

Simplicity and Robustness
- **Apply the "condense principle"** - Make minimal, targeted changes
- **Implement clear error handling** with predictable error messages
- **Ensure predictable return values** - Avoid non-deterministic test outcomes
- **Handle initialization order carefully** - Critical attributes must be initialized before validation that might raise exceptions

Cache-Related Issues
- **Python module caching** can hide code modifications during development
- Clear caches regularly during development: `__pycache__`, `*.pyc`, `.pytest_cache`
- Force module reloading in test environments when necessary
- Be aware that unsaved changes in editors may not be reflected in test runs

Attribute and Object Testing
- **Test attribute existence explicitly** - Use `hasattr()` checks for critical attributes
- **Verify object initialization completeness** - Ensure all required attributes are properly initialized
- **Test object state consistency** - Verify objects maintain expected state throughout operations

Integration Testing
- **Test module interactions** - Verify that different modules work together correctly
- **Data flow consistency** - Ensure data remains consistent as it flows between modules
- **Configuration propagation** - Test that settings properly propagate through integrated systems

Error Handling and Debugging

Comprehensive Error Testing
- Test error propagation across module boundaries
- Verify graceful handling of file system errors
- Test configuration error scenarios (missing API keys, invalid settings)
- Ensure error messages are clear and actionable

Debugging Support
- Structure tests to provide meaningful failure information
- Include debug information in test assertions
- Design tests that help isolate problems when they occur
- Use descriptive test names that clearly indicate what is being tested

Performance and Memory Considerations

Performance Testing
- Test batch processing performance with realistic data sizes
- Verify memory usage stability during extended operations
- Test concurrent operations and race conditions where applicable
- Ensure acceptable throughput for expected workloads

 Resource Management
- Test proper cleanup of resources (files, connections, etc.)
- Verify that temporary files and directories are properly removed
- Test behavior under resource constraints

Documentation and Maintenance

Test Documentation
- Use clear, descriptive test method names
- Include docstrings explaining test purpose and expected behavior
- Document any special setup or teardown requirements
- Maintain test documentation alongside code changes

Test Maintenance
- Regular review of test relevance and effectiveness
- Remove or update obsolete tests
- Ensure test suite execution time remains reasonable
- Monitor test flakiness and address unstable tests promptly

Quality Assurance Checklist

Before considering any feature complete:
- [ ] All new functionality has corresponding tests
- [ ] Tests cover normal, edge, and failure cases
- [ ] Test-implementation architectural consistency verified
- [ ] Cross-platform compatibility confirmed
- [ ] Error handling thoroughly tested
- [ ] Performance impact assessed
- [ ] Documentation updated accordingly
- [ ] Code review completed with focus on test quality

Remember: **Prevention is better than debugging** - Well-designed tests and consistent development practices prevent most debugging scenarios.

### ✅ Task Completion
- Mark completed tasks in `TASK.md` immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a “Discovered During Work” section.

### 📎 Style & Conventions (Here need to modify depends on different project.)
- Use Python as the primary language.
- Follow PEP8, use type hints, and format with `black`.
- Use `pydantic` for data validation.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write docstrings for every function using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability
- Update `README.md` when new features are added, dependencies change, or setup steps are modified.
- Comment non-obvious code and ensure everything is understandable to a mid-level developer.
- When writing complex logic, add an inline `# Reason:` comment explaining the why, not just the what.

### 🧠 AI Behavior Rules
- Never assume missing context. Ask questions if uncertain.
- Never hallucinate libraries or functions-only use known, verified Python packages.
- Always confirm file paths and module names exist before referencing them in code or tests.
- Never delete or overwrite existing code unless explicitly instructed.
- While in debugging, Except necessary, otherwise don't create new file while in debuging, just edited in the original file.