# GraphJudge Streamlit Pipeline - API Reference

**Version:** 2.0  
**Last Updated:** 2025-09-13  
**Status:** Production Ready

This document provides comprehensive API reference for all modules, classes, and functions in the GraphJudge Streamlit Pipeline.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Utility Modules](#utility-modules)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

---

## Core Modules

### `streamlit_pipeline.core.entity_processor`

Main module for entity extraction and text denoising using GPT-5-mini.

#### Functions

##### `extract_entities(text: str, api_client: Optional[APIClient] = None) -> EntityResult`

Extracts named entities from input text and performs text denoising.

**Parameters:**
- `text` (str): Input text to process (Chinese or English)
- `api_client` (Optional[APIClient]): Custom API client, uses default if None

**Returns:**
- `EntityResult`: Object containing extracted entities and denoised text

**Raises:**
- `ValueError`: If input text is empty or invalid
- `APIError`: If API call fails

**Example:**
```python
from streamlit_pipeline.core.entity_processor import extract_entities

text = "林黛玉是賈府的親戚，從江南來到榮國府。"
result = extract_entities(text)

if result.success:
    print(f"Entities: {result.entities}")
    print(f"Denoised text: {result.denoised_text}")
else:
    print(f"Error: {result.error}")
```

**Performance:**
- Typical processing time: 2-8 seconds
- Input length limit: 50,000 characters
- Rate limit: Follows OpenAI API limits

---

### `streamlit_pipeline.core.triple_generator`

Module for generating knowledge graph triples from entities and text.

#### Functions

##### `generate_triples(entities: List[str], text: str, api_client: Optional[APIClient] = None) -> TripleResult`

Generates knowledge graph triples from extracted entities and processed text.

**Parameters:**
- `entities` (List[str]): List of entity names to focus on
- `text` (str): Processed/denoised text
- `api_client` (Optional[APIClient]): Custom API client, uses default if None

**Returns:**
- `TripleResult`: Object containing generated triples and metadata

**Raises:**
- `ValueError`: If entities list is empty or text is invalid
- `APIError`: If API call fails
- `ValidationError`: If generated triples don't match schema

**Example:**
```python
from streamlit_pipeline.core.triple_generator import generate_triples

entities = ["林黛玉", "賈府", "榮國府"]
text = "林黛玉來到榮國府居住。"
result = generate_triples(entities, text)

if result.success:
    for triple in result.triples:
        print(f"{triple.subject} -> {triple.predicate} -> {triple.object}")
        print(f"Confidence: {triple.confidence}")
else:
    print(f"Error: {result.error}")
```

**Features:**
- Automatic text chunking for large inputs
- JSON schema validation
- Confidence scoring
- Relationship vocabulary standardization

---

### `streamlit_pipeline.core.graph_judge`

Module for judging the quality and correctness of generated triples.

#### Functions

##### `judge_triples(triples: List[Triple], api_client: Optional[APIClient] = None) -> JudgmentResult`

Evaluates the correctness of knowledge graph triples using AI reasoning.

**Parameters:**
- `triples` (List[Triple]): List of triples to evaluate
- `api_client` (Optional[APIClient]): Custom API client, uses default if None

**Returns:**
- `JudgmentResult`: Object containing judgment decisions and confidence scores

**Example:**
```python
from streamlit_pipeline.core.graph_judge import judge_triples
from streamlit_pipeline.core.models import Triple

triples = [
    Triple(subject="林黛玉", predicate="居住於", object="榮國府"),
    Triple(subject="賈寶玉", predicate="初見", object="林黛玉")
]

result = judge_triples(triples)

if result.success:
    for i, (triple, judgment, confidence) in enumerate(
        zip(triples, result.judgments, result.confidence)
    ):
        status = "APPROVED" if judgment else "REJECTED"
        print(f"Triple {i+1}: {status} (confidence: {confidence:.2f})")
        if result.explanations:
            print(f"Explanation: {result.explanations[i]}")
else:
    print(f"Error: {result.error}")
```

##### `judge_triples_with_explanations(triples: List[Triple], api_client: Optional[APIClient] = None) -> JudgmentResult`

Same as `judge_triples` but includes detailed explanations for each judgment.

**Parameters:**
- Same as `judge_triples`

**Returns:**
- `JudgmentResult`: Object with judgments, confidence, and explanations

**Note:** This function may take longer due to explanation generation.

---

### `streamlit_pipeline.core.pipeline`

Main orchestrator for the complete three-stage pipeline.

#### Classes

##### `class PipelineOrchestrator`

Main class for coordinating the complete pipeline execution.

**Methods:**

###### `run_pipeline(input_text: str, progress_callback: Optional[Callable] = None) -> PipelineResult`

Executes the complete three-stage pipeline (Entity → Triple → Judge).

**Parameters:**
- `input_text` (str): Raw text to process
- `progress_callback` (Optional[Callable]): Function to call for progress updates

**Returns:**
- `PipelineResult`: Complete results from all stages

**Example:**
```python
from streamlit_pipeline.core.pipeline import PipelineOrchestrator

def progress_callback(stage: int, message: str):
    print(f"Stage {stage}: {message}")

orchestrator = PipelineOrchestrator()
result = orchestrator.run_pipeline(
    "林黛玉是賈府的親戚，從江南來到榮國府。",
    progress_callback
)

if result.success:
    print(f"Pipeline completed in {result.total_time:.2f} seconds")
    print(f"Final approval rate: {result.stats['approval_rate']:.1%}")
else:
    print(f"Pipeline failed at stage: {result.error_stage}")
    print(f"Error: {result.error}")
```

###### `get_pipeline_state() -> PipelineState`

Returns the current pipeline state for monitoring and debugging.

###### `reset_pipeline() -> None`

Resets the pipeline state for a new run.

---

## Utility Modules

### `streamlit_pipeline.utils.api_client`

Unified API client for managing OpenAI and Perplexity API interactions.

#### Functions

##### `get_api_client() -> APIClient`

Returns the global API client instance (singleton pattern).

**Returns:**
- `APIClient`: Configured API client

**Example:**
```python
from streamlit_pipeline.utils.api_client import get_api_client

client = get_api_client()
response = client.complete(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.0
)
```

#### Classes

##### `class APIClient`

Main API client class with rate limiting and error handling.

**Methods:**

###### `complete(model: str, messages: List[Dict], **kwargs) -> APIResponse`

Sends completion request to the appropriate API.

**Parameters:**
- `model` (str): Model name (e.g., "gpt-5-mini", "perplexity/sonar-reasoning")
- `messages` (List[Dict]): Chat messages in OpenAI format
- `**kwargs`: Additional API parameters (temperature, max_tokens, etc.)

**Returns:**
- `APIResponse`: Response object with choices and usage information

---

### `streamlit_pipeline.utils.validation`

Input validation and data sanitization utilities.

#### Functions

##### `validate_text_input(text: str, max_length: int = 50000) -> ValidationResult`

Validates and sanitizes text input.

**Parameters:**
- `text` (str): Input text to validate
- `max_length` (int): Maximum allowed length

**Returns:**
- `ValidationResult`: Validation outcome with sanitized text

**Example:**
```python
from streamlit_pipeline.utils.validation import validate_text_input

result = validate_text_input("Some input text...")
if result.is_valid:
    processed_text = result.sanitized_text
else:
    print(f"Validation errors: {result.errors}")
```

##### `validate_api_response(response: Dict[str, Any], expected_schema: Dict) -> bool`

Validates API response against expected schema.

##### `sanitize_input_text(text: str) -> str`

Sanitizes input text by removing potentially harmful content.

---

### `streamlit_pipeline.utils.error_handling`

Comprehensive error handling and logging system.

#### Classes

##### `class ErrorHandler`

Main error handling class for creating and managing errors.

**Methods:**

###### `create_error(error_type: ErrorType, message: str = None, **kwargs) -> ErrorInfo`

Creates standardized error information.

**Parameters:**
- `error_type` (ErrorType): Type of error from ErrorType enum
- `message` (str): Custom error message (optional)
- `**kwargs`: Additional error context

**Returns:**
- `ErrorInfo`: Structured error information

##### `class StreamlitLogger`

Specialized logger for Streamlit applications.

**Methods:**

###### `log_info(message: str, extra: Dict = None, stage: str = None) -> None`

Logs informational message.

###### `log_error(message: str, extra: Dict = None, stage: str = None) -> None`

Logs error message.

###### `log_warning(message: str, extra: Dict = None, stage: str = None) -> None`

Logs warning message.

#### Functions

##### `safe_execute(func: Callable, error_message: str, error_handler: ErrorHandler, **kwargs) -> Any`

Safely executes a function with error handling.

**Parameters:**
- `func` (Callable): Function to execute
- `error_message` (str): Error message for failures
- `error_handler` (ErrorHandler): Error handler instance
- `**kwargs`: Additional parameters

**Returns:**
- Function result or error dictionary

---

### `streamlit_pipeline.utils.session_state`

Session state management for Streamlit applications.

#### Functions

##### `get_session_manager() -> SessionStateManager`

Returns the global session state manager.

#### Classes

##### `class SessionStateManager`

Manages session state, caching, and progress tracking.

**Methods:**

###### `set_current_result(result: PipelineResult) -> None`

Stores the current pipeline result.

###### `get_current_result() -> Optional[PipelineResult]`

Retrieves the current pipeline result.

###### `get_pipeline_results() -> List[PipelineResult]`

Gets history of pipeline results.

###### `get_session_metadata() -> SessionMetadata`

Returns session metadata and statistics.

---

## Data Models

### Core Data Classes

All data models are defined in `streamlit_pipeline.core.models` and use Python dataclasses with type hints.

#### `Triple`

Represents a knowledge graph triple (subject-predicate-object).

```python
@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    source_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Properties:**
- `confidence_level`: Returns ConfidenceLevel enum based on confidence score

**Methods:**
- `to_dict()`: Converts to dictionary
- `from_dict(data)`: Creates from dictionary
- `__str__()`: String representation

#### `EntityResult`

Result of entity extraction operations.

```python
@dataclass
class EntityResult:
    entities: List[str]
    denoised_text: str
    success: bool
    processing_time: float
    error: Optional[str] = None
```

#### `TripleResult`

Result of triple generation operations.

```python
@dataclass
class TripleResult:
    triples: List[Triple]
    metadata: Dict[str, Any]
    success: bool
    processing_time: float
    error: Optional[str] = None
```

#### `JudgmentResult`

Result of graph judgment operations.

```python
@dataclass
class JudgmentResult:
    judgments: List[bool]
    confidence: List[float]
    explanations: Optional[List[str]] = None
    success: bool = True
    processing_time: float = 0.0
    error: Optional[str] = None
```

#### `PipelineResult`

Complete result from pipeline execution.

```python
@dataclass
class PipelineResult:
    success: bool
    stage_reached: int
    total_time: float
    entity_result: Optional[EntityResult] = None
    triple_result: Optional[TripleResult] = None
    judgment_result: Optional[JudgmentResult] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None
    stats: Dict[str, Any] = None
```

### Enums

#### `ErrorType`

Enumeration of error types for structured error handling.

```python
class ErrorType(Enum):
    CONFIGURATION = "configuration"
    API_AUTH = "api_auth"
    API_RATE_LIMIT = "api_rate_limit"
    API_SERVER = "api_server"
    API_TIMEOUT = "api_timeout"
    API_QUOTA = "api_quota"
    VALIDATION = "validation"
    PROCESSING = "processing"
    INTERNAL = "internal"
```

#### `ErrorSeverity`

Error severity levels.

```python
class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

#### `PipelineStage`

Pipeline processing stages.

```python
class PipelineStage(Enum):
    ENTITY_EXTRACTION = "entity_extraction"
    TRIPLE_GENERATION = "triple_generation"
    GRAPH_JUDGMENT = "graph_judgment"
```

---

## Error Handling

### Exception Hierarchy

```
APIError
├── AuthenticationError
├── RateLimitError
├── ServerError
└── TimeoutError

ValidationError
├── InputValidationError
├── SchemaValidationError
└── ResponseValidationError

ConfigurationError
├── MissingAPIKeyError
├── InvalidConfigError
└── ModelNotAvailableError
```

### Error Response Format

All errors follow a consistent format:

```python
{
    "error_type": "api_auth",
    "severity": "high",
    "message": "Authentication failed with OpenAI API",
    "technical_details": "401 Unauthorized: Invalid API key",
    "suggestions": [
        "Check your API key is correct",
        "Verify API key has sufficient permissions",
        "Contact support if issue persists"
    ],
    "stage": "entity_extraction",
    "timestamp": "2025-09-13T10:30:00Z",
    "context": {
        "model": "gpt-5-mini",
        "retry_count": 3
    }
}
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | None | OpenAI API key |
| `AZURE_OPENAI_KEY` | Yes* | None | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | No | None | Azure endpoint URL |
| `PERPLEXITY_API_KEY` | Yes | None | Perplexity API key |
| `LOG_LEVEL` | No | INFO | Logging level |
| `API_TIMEOUT` | No | 60 | API timeout in seconds |
| `MAX_RETRIES` | No | 3 | Maximum retry attempts |
| `CACHE_ENABLED` | No | true | Enable response caching |

*Either OPENAI_API_KEY or AZURE_OPENAI_KEY is required

### Configuration Loading

```python
from streamlit_pipeline.core.config import get_model_config, get_api_config

# Get model configuration
model_config = get_model_config()
print(f"Entity model: {model_config.entity_model}")
print(f"Judge model: {model_config.judge_model}")

# Get API configuration
api_key, api_base = get_api_config()
print(f"Using API base: {api_base}")
```

---

## Usage Examples

### Basic Pipeline Usage

```python
from streamlit_pipeline.core.pipeline import PipelineOrchestrator

# Simple usage
orchestrator = PipelineOrchestrator()
result = orchestrator.run_pipeline("林黛玉來到榮國府。")

if result.success:
    print("Pipeline completed successfully!")
    print(f"Entities found: {len(result.entity_result.entities)}")
    print(f"Triples generated: {len(result.triple_result.triples)}")
    print(f"Approved triples: {sum(result.judgment_result.judgments)}")
else:
    print(f"Pipeline failed: {result.error}")
```

### Advanced Usage with Progress Tracking

```python
import time
from streamlit_pipeline.core.pipeline import PipelineOrchestrator
from streamlit_pipeline.utils.session_state import get_session_manager

def progress_callback(stage: int, message: str):
    stage_names = ["Entity Extraction", "Triple Generation", "Graph Judgment", "Complete"]
    print(f"[{stage_names[stage]}] {message}")

# Initialize pipeline with session state
session_manager = get_session_manager()
orchestrator = PipelineOrchestrator()

# Execute pipeline
text = """
林黛玉是賈府的親戚，從江南來到榮國府。她聰明伶俐，才情出眾，
深受賈母喜愛。賈寶玉初見林黛玉時，覺得這個妹妹似曾相識。
"""

start_time = time.time()
result = orchestrator.run_pipeline(text, progress_callback)
end_time = time.time()

# Store result in session
session_manager.set_current_result(result)

# Display results
if result.success:
    print(f"\n✅ Pipeline completed in {end_time - start_time:.2f} seconds")
    
    # Entity extraction results
    print(f"\n🔍 Entities ({len(result.entity_result.entities)}):")
    for entity in result.entity_result.entities:
        print(f"  - {entity}")
    
    # Triple generation results
    print(f"\n🔗 Generated Triples ({len(result.triple_result.triples)}):")
    for i, triple in enumerate(result.triple_result.triples):
        print(f"  {i+1}. {triple}")
    
    # Judgment results
    approved = sum(result.judgment_result.judgments)
    total = len(result.judgment_result.judgments)
    print(f"\n⚖️ Graph Judgment: {approved}/{total} approved ({approved/total:.1%})")
    
    if result.judgment_result.explanations:
        print("\n📝 Explanations:")
        for i, (triple, judgment, explanation) in enumerate(
            zip(result.triple_result.triples, result.judgment_result.judgments, result.judgment_result.explanations)
        ):
            status = "✅" if judgment else "❌"
            print(f"  {status} {triple.subject} → {triple.predicate} → {triple.object}")
            print(f"     {explanation}")
    
    # Performance statistics
    print(f"\n📊 Statistics:")
    print(f"  - Entity extraction: {result.entity_result.processing_time:.2f}s")
    print(f"  - Triple generation: {result.triple_result.processing_time:.2f}s")
    print(f"  - Graph judgment: {result.judgment_result.processing_time:.2f}s")
    print(f"  - Total processing: {result.total_time:.2f}s")
    print(f"  - Approval rate: {result.stats['approval_rate']:.1%}")
    
else:
    print(f"\n❌ Pipeline failed at {result.error_stage}: {result.error}")
```

### Error Handling Example

```python
from streamlit_pipeline.core.pipeline import PipelineOrchestrator
from streamlit_pipeline.utils.error_handling import ErrorHandler, ErrorType

def robust_pipeline_execution(text: str):
    """Execute pipeline with comprehensive error handling."""
    orchestrator = PipelineOrchestrator()
    error_handler = ErrorHandler()
    
    try:
        result = orchestrator.run_pipeline(text)
        
        if result.success:
            return result
        else:
            # Handle specific error types
            if result.error_stage == "entity_extraction":
                print("Entity extraction failed - check API configuration")
            elif result.error_stage == "triple_generation":
                print("Triple generation failed - check input quality")
            elif result.error_stage == "graph_judgment":
                print("Graph judgment failed - check Perplexity API")
            
            # Create structured error
            error_info = error_handler.create_error(
                error_type=ErrorType.PROCESSING,
                message=result.error,
                stage=result.error_stage,
                input_length=len(text)
            )
            
            print(f"Error: {error_info.message}")
            print(f"Suggestions: {', '.join(error_info.suggestions)}")
            
            return None
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Usage
result = robust_pipeline_execution("Your text here...")
if result:
    print("Pipeline completed successfully!")
else:
    print("Pipeline failed - check logs for details")
```

### Batch Processing Example

```python
from typing import List
from streamlit_pipeline.core.pipeline import PipelineOrchestrator
from streamlit_pipeline.utils.session_state import get_session_manager

def batch_process_texts(texts: List[str]) -> List[PipelineResult]:
    """Process multiple texts in batch."""
    orchestrator = PipelineOrchestrator()
    session_manager = get_session_manager()
    results = []
    
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}...")
        
        result = orchestrator.run_pipeline(text)
        results.append(result)
        
        # Store in session for tracking
        session_manager.set_current_result(result)
        
        if result.success:
            print(f"  ✅ Success: {len(result.triple_result.triples)} triples generated")
        else:
            print(f"  ❌ Failed: {result.error}")
        
        # Reset pipeline for next iteration
        orchestrator.reset_pipeline()
    
    return results

# Usage
texts = [
    "林黛玉來到榮國府。",
    "賈寶玉初見林黛玉。",
    "賈母疼愛林黛玉。"
]

results = batch_process_texts(texts)

# Analyze batch results
successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"\nBatch Processing Summary:")
print(f"Successful: {len(successful)}/{len(results)}")
print(f"Failed: {len(failed)}/{len(results)}")

if successful:
    avg_time = sum(r.total_time for r in successful) / len(successful)
    total_triples = sum(len(r.triple_result.triples) for r in successful)
    print(f"Average processing time: {avg_time:.2f}s")
    print(f"Total triples generated: {total_triples}")
```

---

## Performance Considerations

### Rate Limits

**OpenAI API:**
- Free tier: 3 requests/minute
- Pay-as-you-use: 3,500 requests/minute
- Consider upgrading for production use

**Perplexity API:**
- Check current rate limits in Perplexity documentation
- Implement proper rate limiting in production

### Optimization Tips

1. **Use caching for repeated inputs**
2. **Implement request batching where possible**
3. **Monitor API costs and usage**
4. **Use appropriate model parameters (temperature, max_tokens)**
5. **Implement proper error handling and retries**

### Memory Usage

- Entity results: ~1KB per entity
- Triple results: ~500B per triple
- Session state: Monitor for large datasets
- Clear session periodically in long-running applications

---

**Documentation Version:** 2.0  
**Last Updated:** 2025-09-13  
**Next Review:** Monthly