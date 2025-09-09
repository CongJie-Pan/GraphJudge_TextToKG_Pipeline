# Entity Extraction and Text Denoising (ECTD) Pipeline

A modular, testable, and extensible implementation of the GPT-5-mini Entity Extraction and Text Denoising pipeline for classical Chinese text processing.

## 🏗️ Architecture Overview

The ECTD pipeline is built with a modular architecture that separates concerns and promotes maintainability:

```
extractEntity_Phase/
├── api/                    # API client layer
│   └── gpt5mini_client.py # GPT-5-mini API client with caching & rate limiting
├── core/                   # Business logic layer
│   ├── entity_extractor.py    # Entity extraction from text
│   ├── text_denoiser.py       # Text denoising with entity context
│   └── pipeline_orchestrator.py # Pipeline coordination
├── models/                 # Data models
│   ├── entities.py            # Entity data structures
│   └── pipeline_state.py      # Pipeline execution state
├── utils/                  # Utility functions
│   ├── logger.py              # Logging utilities
│   └── cache_manager.py       # Response caching
├── tests/                  # Test suite
│   ├── test_entity_extractor.py
│   ├── test_text_denoiser.py
│   ├── test_pipeline_orchestrator.py
│   └── test_pipeline_integration.py
└── docs/                   # Documentation
    └── run_entity_modulize_plan.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key with GPT-5-mini access
- Required Python packages (see requirements below)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Miscellaneous/KgGen/GraphJudge/chat/extractEntity_Phase
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install litellm pydantic asyncio aiofiles
   ```

3. **Set up environment variables:**
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```

### Basic Usage

#### 1. Simple Entity Extraction

```python
import asyncio
from core.entity_extractor import EntityExtractor
from api.gpt5mini_client import GPT5MiniClient

async def main():
    # Initialize the API client
    client = GPT5MiniClient()
    
    # Create entity extractor
    extractor = EntityExtractor(client)
    
    # Extract entities from texts
    texts = [
        "賈寶玉是《紅樓夢》中的主要人物，他與林黛玉有著深厚的感情。",
        "薛寶釵是金陵十二釵之一，她聰明能幹，深受賈母喜愛。"
    ]
    
    results = await extractor.extract_entities_from_texts(texts)
    
    for i, collection in enumerate(results):
        print(f"Text {i+1}: {texts[i]}")
        print(f"Entities: {[e.text for e in collection.entities]}")
        print("---")

# Run the async function
asyncio.run(main())
```

#### 2. Text Denoising with Entity Context

```python
import asyncio
from core.text_denoiser import TextDenoiser
from api.gpt5mini_client import GPT5MiniClient

async def main():
    # Initialize components
    client = GPT5MiniClient()
    denoiser = TextDenoiser(client)
    
    # Denoise texts with entity context
    texts = [
        "賈寶玉是《紅樓夢》中的主要人物，他與林黛玉有著深厚的感情。",
        "薛寶釵是金陵十二釵之一，她聰明能幹，深受賈母喜愛。"
    ]
    
    entities_list = [
        ["賈寶玉", "林黛玉", "紅樓夢"],
        ["薛寶釵", "金陵十二釵", "賈母"]
    ]
    
    denoised_texts = await denoiser.denoise_texts(texts, entities_list)
    
    for i, (original, denoised) in enumerate(zip(texts, denoised_texts)):
        print(f"Original {i+1}: {original}")
        print(f"Denoised {i+1}: {denoised}")
        print("---")

asyncio.run(main())
```

#### 3. Full Pipeline Execution

```python
import asyncio
from core.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig

async def main():
    # Configure the pipeline
    config = PipelineConfig(
        batch_size=5,
        max_retries=3,
        output_dir="./output",
        enable_logging=True
    )
    
    # Create and run the pipeline
    orchestrator = PipelineOrchestrator(config)
    
    # Option 1: Provide texts directly
    input_texts = [
        "賈寶玉是《紅樓夢》中的主要人物。",
        "林黛玉是賈寶玉的表妹。",
        "薛寶釵是金陵十二釵之一。"
    ]
    
    success = await orchestrator.run_pipeline(input_texts)
    
    if success:
        print("Pipeline completed successfully!")
        print(f"Statistics: {orchestrator.get_pipeline_statistics()}")
    else:
        print("Pipeline failed. Check logs for details.")

# Run the pipeline
asyncio.run(main())
```

## 🔧 Configuration

### Entity Extractor Configuration

```python
from core.entity_extractor import ExtractionConfig

config = ExtractionConfig(
    batch_size=10,           # Process 10 texts at once
    max_retries=3,           # Retry failed API calls up to 3 times
    use_examples=True,       # Include examples in prompts
    enable_deduplication=True # Remove duplicate entities
)

extractor = EntityExtractor(client, config)
```

### Text Denoiser Configuration

```python
from core.text_denoiser import DenoisingConfig

config = DenoisingConfig(
    batch_size=8,            # Process 8 texts at once
    max_retries=2,           # Retry failed API calls up to 2 times
    use_examples=True,       # Include examples in prompts
    similarity_threshold=0.7  # Minimum similarity for validation
)

denoiser = TextDenoiser(client, config)
```

### Pipeline Configuration

```python
from core.pipeline_orchestrator import PipelineConfig

config = PipelineConfig(
    batch_size=5,            # Overall batch size for pipeline stages
    max_retries=3,           # Maximum retries for any stage
    output_dir="./results",  # Output directory for results
    enable_logging=True,     # Enable detailed logging
    save_intermediate=True   # Save intermediate results
)

orchestrator = PipelineOrchestrator(config)
```

## 📊 Monitoring and Statistics

### Entity Extraction Statistics

```python
# Get extraction statistics
stats = extractor.get_statistics()
print(f"Processed texts: {stats['total_texts']}")
print(f"Total entities: {stats['total_entities']}")
print(f"Average entities per text: {stats['avg_entities_per_text']}")
print(f"Success rate: {stats['success_rate']:.2%}")

# Reset statistics
extractor.reset_statistics()
```

### Text Denoising Statistics

```python
# Get denoising statistics
stats = denoiser.get_statistics()
print(f"Processed texts: {stats['total_texts']}")
print(f"Average compression ratio: {stats['avg_compression_ratio']:.2%}")
print(f"Success rate: {stats['success_rate']:.2%}")

# Reset statistics
denoiser.reset_statistics()
```

### Pipeline Statistics

```python
# Get overall pipeline statistics
pipeline_stats = orchestrator.get_pipeline_statistics()
print(f"Total execution time: {pipeline_stats['total_execution_time']:.2f}s")
print(f"Entities extracted: {pipeline_stats['total_entities']}")
print(f"Texts denoised: {pipeline_stats['total_texts_denoised']}")

# Get pipeline state
state = orchestrator.get_pipeline_state()
print(f"Current status: {state.status}")
print(f"Current stage: {state.current_stage}")
```

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=core --cov=api --cov=models -v

# Run specific test file
python -m pytest tests/test_entity_extractor.py -v
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete pipeline workflows

## 📁 Output Structure

The pipeline generates the following output structure:

```
output/
├── entities_YYYYMMDD_HHMMSS.json      # Extracted entities
├── denoised_texts_YYYYMMDD_HHMMSS.txt # Denoised texts
├── pipeline_stats_YYYYMMDD_HHMMSS.json # Pipeline statistics
└── pipeline_state_YYYYMMDD_HHMMSS.json # Pipeline execution state
```

## 🔍 Error Handling

The system provides comprehensive error handling:

```python
try:
    results = await extractor.extract_entities_from_texts(texts)
except Exception as e:
    print(f"Extraction failed: {e}")
    # Check detailed error information
    state = extractor.get_pipeline_state()
    if state.errors:
        for error in state.errors:
            print(f"Error: {error.message} (Severity: {error.severity})")
```

## 🚀 Advanced Usage

### Custom Entity Types

```python
from models.entities import EntityType

# The system automatically classifies entities, but you can customize
def custom_classifier(entity_text: str, source_text: str) -> EntityType:
    if "寶玉" in entity_text:
        return EntityType.PERSON
    elif "紅樓夢" in entity_text:
        return EntityType.WORK
    return EntityType.OTHER

# Use in extractor
extractor._classify_entity_type = custom_classifier
```

### Progress Callbacks

```python
def progress_callback(current: int, total: int, stage: str):
    print(f"{stage}: {current}/{total} ({current/total*100:.1f}%)")

# Use in extraction
results = await extractor.extract_entities_from_texts(
    texts, 
    progress_callback=progress_callback
)
```

### Custom Prompt Templates

```python
# Override prompt building methods
def custom_prompt(text: str) -> str:
    return f"請從以下古典中文文本中提取實體：\n\n{text}\n\n請以JSON格式返回結果。"

extractor._build_extraction_prompt = custom_prompt
```

## 🔧 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure `OPENAI_API_KEY` is set correctly
   - Check API key permissions for GPT-5-mini

2. **Rate Limiting**
   - The system includes built-in rate limiting
   - Adjust batch sizes if hitting API limits

3. **Memory Issues**
   - Reduce batch sizes for large text collections
   - Process texts in smaller chunks

4. **Text Encoding**
   - Ensure texts are properly encoded (UTF-8)
   - Handle special characters appropriately

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or use the built-in logger
from utils.logger import TerminalLogger
logger = TerminalLogger(level="DEBUG")
```

## 📚 API Reference

### Core Classes

- **`EntityExtractor`**: Main entity extraction logic
- **`TextDenoiser`**: Text denoising with entity context
- **`PipelineOrchestrator`**: Complete pipeline coordination
- **`GPT5MiniClient`**: API client with caching and rate limiting

### Key Methods

- **`extract_entities_from_texts()`**: Extract entities from multiple texts
- **`denoise_texts()`**: Denoise texts using entity context
- **`run_pipeline()`**: Execute complete ECTD pipeline
- **`get_statistics()`**: Retrieve processing statistics

## 🤝 Contributing

When contributing to this module:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for new features
4. Ensure all tests pass before submitting changes

## 📄 License

This module is part of the larger project and follows the same licensing terms.

## 🆘 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Check the original `run_entity.py` for reference implementation
4. Review the modularization plan in `docs/run_entity_modulize_plan.md`
