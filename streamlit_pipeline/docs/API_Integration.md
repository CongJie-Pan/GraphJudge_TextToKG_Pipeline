# GraphJudge Entity Processor API Integration Guide

**Version:** 1.0  
**Date:** 2025-09-12  
**Module:** `streamlit_pipeline.core.entity_processor`  
**Task ID:** REF-001

## Overview

This document provides comprehensive API integration guidance for the Entity Processor module, which has been refactored from the original `chat/run_entity.py` script following the specifications in `spec.md`.

## Module Architecture

### Core Function Interface

The main interface follows spec.md Section 8 requirements:

```python
def extract_entities(text: str) -> EntityResult
```

**Simplifications from Original:**
- **Synchronous execution** (no async/await complexity)
- **In-memory data handling** (no file I/O operations)
- **Unified error handling** (errors returned as data, not exceptions)
- **No complex caching system**
- **No intricate logging to files**

## API Dependencies

### External APIs Used

1. **GPT-5-mini via LiteLLM**
   - **Purpose**: Entity extraction and text denoising
   - **Model**: `gpt-5-mini`
   - **Configuration**: Via `streamlit_pipeline.core.config`
   - **Rate Limiting**: Basic rate limiting implemented in `utils.api_client`

### Required Environment Variables

```bash
# OpenAI Standard API (primary)
OPENAI_API_KEY=your_openai_api_key_here

# OR Azure OpenAI (takes priority if set)
AZURE_OPENAI_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Usage Examples

### Basic Usage

```python
from streamlit_pipeline.core.entity_processor import extract_entities

# Process Chinese text
text = "林黛玉進入榮國府後，與賈寶玉初次相遇，兩人情投意合。"
result = extract_entities(text)

if result.success:
    print(f"Entities: {result.entities}")
    print(f"Denoised text: {result.denoised_text}")
    print(f"Processing time: {result.processing_time:.2f}s")
else:
    print(f"Error: {result.error}")
```

### Batch Processing

```python
from streamlit_pipeline.core.entity_processor import batch_extract_entities

texts = [
    "林黛玉進入榮國府後，與賈寶玉初次相遇。",
    "王熙鳳管理榮國府的日常事務。"
]

results = batch_extract_entities(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: {len(result.entities)} entities extracted")
```

### Error Handling

```python
from streamlit_pipeline.core.entity_processor import extract_entities

# Handle various error conditions
result = extract_entities("")  # Empty text
if not result.success:
    print(f"Expected error: {result.error}")

# Handle API failures gracefully
result = extract_entities("some text")  # Will handle API errors internally
if result.success:
    # Process results
    pass
else:
    # Handle error gracefully
    print(f"Processing failed: {result.error}")
```

## Data Models

### EntityResult

```python
@dataclass
class EntityResult:
    entities: List[str]           # Extracted entity names
    denoised_text: str           # Cleaned and restructured text  
    success: bool                # Operation success status
    processing_time: float       # Time taken in seconds
    error: Optional[str] = None  # Error message if failed
```

### Processing Timer

```python
from streamlit_pipeline.core.models import ProcessingTimer

with ProcessingTimer() as timer:
    # Your processing code here
    pass

print(f"Elapsed time: {timer.elapsed:.2f}s")
```

## API Configuration

### Model Configuration

```python
from streamlit_pipeline.core.config import get_model_config

config = get_model_config()
print(config)
# Output:
# {
#     'entity_model': 'gpt-5-mini',
#     'triple_model': 'gpt-5-mini', 
#     'judgment_model': 'perplexity/sonar-reasoning',
#     'temperature': 0.0,
#     'max_tokens': 4000,
#     'timeout': 60,
#     'max_retries': 3
# }
```

### API Key Management

```python
from streamlit_pipeline.core.config import get_api_config, get_api_key

# Get full API configuration
api_key, api_base = get_api_config()

# Get just the API key (compatible with original interface)
api_key = get_api_key()
```

## Performance Characteristics

### Typical Performance Metrics

- **Processing Time**: 2-5 seconds per text (depending on length and API response time)
- **Entity Extraction**: Handles 100-2000 character Chinese texts effectively
- **Text Denoising**: Maintains classical Chinese style while improving clarity
- **Rate Limiting**: Basic 0.1 second interval between requests

### Optimization Recommendations

1. **Batch Processing**: Use `batch_extract_entities()` for multiple texts
2. **Text Length**: Optimal performance with 100-1000 character texts
3. **API Limits**: Respect OpenAI rate limits (handled automatically)

## Error Handling Patterns

### Error Types and Handling

1. **Input Validation Errors**
   ```python
   # Empty or invalid input
   result = extract_entities("")
   assert not result.success
   assert "empty" in result.error.lower()
   ```

2. **API Connection Errors**
   ```python
   # Network issues, API failures
   # Handled automatically with retries
   # Returns error result instead of raising exceptions
   ```

3. **Response Parsing Errors**
   ```python
   # Malformed API responses handled gracefully
   # Returns best-effort results or clean error messages
   ```

### Error Recovery Strategies

- **Automatic Retries**: Up to 3 attempts with exponential backoff
- **Graceful Degradation**: Returns partial results when possible
- **Clear Error Messages**: User-friendly error descriptions
- **No Exceptions**: All errors returned as data in result objects

## Testing Integration

### Running Tests

```bash
cd streamlit_pipeline
pytest tests/test_entity_processor.py -v
```

### Mock Testing

```python
import pytest
from unittest.mock import patch
from streamlit_pipeline.core.entity_processor import extract_entities

@patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini')
def test_entity_extraction(mock_api_call):
    mock_api_call.side_effect = ['["實體1", "實體2"]', "去噪文本"]
    
    result = extract_entities("測試文本")
    assert result.success
    assert result.entities == ["實體1", "實體2"]
```

## Migration from Original Script

### Key Differences from `chat/run_entity.py`

| Aspect | Original Script | Refactored Module |
|--------|----------------|------------------|
| Execution | Async/await | Synchronous |
| Data Flow | File-based I/O | In-memory objects |
| Error Handling | Exceptions | Return values |
| Caching | Complex file cache | No caching |
| Logging | File-based logs | Simple return data |
| Dependencies | Many complex deps | Minimal deps |
| Lines of Code | ~800 lines | ~200 lines |

### Migration Steps

1. **Replace async calls**:
   ```python
   # Old
   entities_list = await extract_entities(texts)
   
   # New
   results = batch_extract_entities(texts)
   entities_list = [r.entities for r in results if r.success]
   ```

2. **Update error handling**:
   ```python
   # Old
   try:
       result = await some_function()
   except Exception as e:
       handle_error(e)
   
   # New
   result = extract_entities(text)
   if not result.success:
       handle_error(result.error)
   ```

3. **Update data access patterns**:
   ```python
   # Old
   with open(entity_file, 'r') as f:
       entities = json.load(f)
   
   # New
   result = extract_entities(text)
   entities = result.entities
   ```

## Troubleshooting

### Common Issues

1. **"No API key found in configuration"**
   - **Cause**: Missing environment variables
   - **Solution**: Set `OPENAI_API_KEY` or Azure credentials
   - **Check**: `get_api_config()` should return valid credentials

2. **"API connection failed"**
   - **Cause**: Network issues or API downtime
   - **Solution**: Check internet connection and API status
   - **Retry**: Automatic retries included, but may need manual retry

3. **"Failed to parse entity response"**
   - **Cause**: Unexpected API response format
   - **Solution**: Usually self-correcting, check API model version
   - **Fallback**: Returns empty list rather than crashing

4. **Empty entity results**
   - **Cause**: Text may not contain clear entities
   - **Solution**: Normal behavior for some texts
   - **Check**: Verify text is in Chinese and contains entities

### Debug Information

```python
from streamlit_pipeline.core.entity_processor import extract_entities
from streamlit_pipeline.core.config import get_api_config

# Check API configuration
try:
    api_key, api_base = get_api_config()
    print(f"API configured: {bool(api_key)}")
except Exception as e:
    print(f"API config error: {e}")

# Test with simple text
result = extract_entities("林黛玉讀書")
print(f"Success: {result.success}")
print(f"Processing time: {result.processing_time}")
if result.error:
    print(f"Error: {result.error}")
```

## Future Enhancements

### Planned Improvements (Out of Scope for REF-001)

1. **Caching Layer**: Optional result caching for repeated requests
2. **Advanced Rate Limiting**: More sophisticated API usage management  
3. **Streaming Responses**: Support for real-time processing updates
4. **Multi-model Support**: Support for additional Chinese language models

### Integration Points

- **REF-004**: Triple Generator will consume `EntityResult` objects
- **REF-008**: Streamlit UI will call `extract_entities()` directly
- **REF-009**: Session state will store `EntityResult` objects

---

## Summary

The Entity Processor module successfully extracts core functionality from the original 800-line script into a clean, testable 200-line module. It maintains all essential GPT-5-mini capabilities while providing a Streamlit-compatible interface with proper error handling and comprehensive test coverage.

**Key Achievements:**
- ✅ 75% code reduction (800 → 200 lines)
- ✅ Preserved GPT-5-mini functionality  
- ✅ Clean synchronous interface
- ✅ Comprehensive test coverage
- ✅ Proper error handling
- ✅ Full API integration documentation

This module is ready for integration with the broader Streamlit pipeline and serves as a foundation for the remaining refactoring tasks.