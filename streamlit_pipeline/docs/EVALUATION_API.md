# GraphJudge Evaluation System - API Reference

**Version:** 1.0
**Date:** 2025-09-23
**Status:** Production Ready

This document provides comprehensive API reference documentation for the GraphJudge evaluation system, including all components, configuration options, and integration patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Data Models](#data-models)
4. [Main API Classes](#main-api-classes)
5. [Utility Components](#utility-components)
6. [Configuration](#configuration)
7. [Integration Patterns](#integration-patterns)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Examples](#examples)

---

## Overview

The GraphJudge evaluation system provides comprehensive graph quality assessment capabilities for knowledge graphs generated through the text-to-KG pipeline. It implements multiple evaluation dimensions including exact matching, text similarity, semantic similarity, and structural distance metrics.

### Key Features

- **Multi-Metric Assessment**: Triple Match F1, Graph Match Accuracy, G-BLEU/G-ROUGE, G-BertScore, Graph Edit Distance
- **Real-time and Batch Evaluation**: Support for both single graph and batch evaluation modes
- **Graceful Fallbacks**: Optional dependencies with fallback implementations
- **Performance Optimization**: <500ms typical evaluation time with lazy loading and timeouts
- **Research-Grade Metrics**: Proven algorithms with structured result objects

---

## Core Components

### Module Structure

```
eval/
├── graph_evaluator.py        # Main evaluation engine
├── metrics/                  # Modular metric implementations
│   ├── __init__.py
│   ├── exact_matching.py     # Triple and graph matching
│   ├── text_similarity.py    # G-BLEU and G-ROUGE
│   ├── semantic_similarity.py # G-BertScore
│   └── structural_distance.py # Graph Edit Distance
└── __init__.py
```

### Component Dependencies

```python
# Required dependencies
from typing import List, Dict, Any, Optional, Tuple
import time
import logging

# Optional dependencies (with graceful fallbacks)
try:
    import nltk              # For G-BLEU metrics
    import rouge_score       # For G-ROUGE metrics
    import bert_score        # For G-BertScore metrics
    import networkx as nx    # For Graph Edit Distance
except ImportError:
    # Graceful fallbacks implemented
    pass
```

---

## Data Models

### Triple

Core data structure representing knowledge graph relationships.

```python
@dataclass
class Triple:
    """
    Represents a single knowledge graph triple (subject, predicate, object).
    """
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    source_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary representation."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        """Create triple from dictionary representation."""

    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        """Get confidence level enum for this triple."""
```

### GraphMetrics

Comprehensive evaluation metrics container.

```python
@dataclass
class GraphMetrics:
    """
    Comprehensive graph evaluation metrics based on multiple assessment dimensions.
    """
    # Exact matching metrics
    triple_match_f1: float           # F1 score for exact triple matching
    graph_match_accuracy: float      # Structural graph isomorphism accuracy

    # Text similarity metrics (G-BLEU)
    g_bleu_precision: float          # BLEU precision for graph edges
    g_bleu_recall: float             # BLEU recall for graph edges
    g_bleu_f1: float                 # BLEU F1 score for graph edges

    # Text similarity metrics (G-ROUGE)
    g_rouge_precision: float         # ROUGE precision for graph edges
    g_rouge_recall: float            # ROUGE recall for graph edges
    g_rouge_f1: float                # ROUGE F1 score for graph edges

    # Semantic similarity metrics (G-BertScore)
    g_bert_precision: float          # BertScore precision for graph edges
    g_bert_recall: float             # BertScore recall for graph edges
    g_bert_f1: float                 # BertScore F1 score for graph edges

    # Optional structural distance metric
    graph_edit_distance: Optional[float] = None  # Average graph edit distance

    def get_overall_score(self) -> float:
        """Calculate overall quality score by averaging key metrics."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format for export."""
```

### EvaluationResult

Complete evaluation result container.

```python
@dataclass
class EvaluationResult:
    """
    Complete evaluation result containing metrics, metadata, and processing information.
    """
    metrics: GraphMetrics                           # Computed evaluation metrics
    metadata: Dict[str, Any]                        # Evaluation parameters, timestamps, etc.
    success: bool                                   # Whether evaluation completed successfully
    processing_time: float                          # Time taken for evaluation in seconds
    error: Optional[str] = None                     # Error message if evaluation failed
    reference_graph_info: Optional[Dict[str, Any]] = None  # Reference graph statistics
    predicted_graph_info: Optional[Dict[str, Any]] = None  # Predicted graph statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary format for export."""

    def export_summary(self) -> str:
        """Generate a human-readable summary of evaluation results."""
```

---

## Main API Classes

### GraphEvaluator

Primary evaluation engine implementing comprehensive graph quality assessment.

```python
class GraphEvaluator:
    """
    Comprehensive graph quality evaluator implementing multiple assessment metrics.
    """

    def __init__(self,
                 enable_ged: bool = False,
                 enable_bert_score: bool = True,
                 max_evaluation_time: float = 30.0):
        """
        Initialize the graph evaluator with configuration options.

        Args:
            enable_ged: Whether to enable Graph Edit Distance computation (expensive)
            enable_bert_score: Whether to enable BertScore semantic similarity
            max_evaluation_time: Maximum time allowed for evaluation in seconds
        """

    def evaluate_graph(self,
                      predicted_graph: List[Triple],
                      reference_graph: List[Triple]) -> EvaluationResult:
        """
        Evaluate a predicted graph against a reference graph.

        Args:
            predicted_graph: List of predicted Triple objects
            reference_graph: List of reference/gold standard Triple objects

        Returns:
            EvaluationResult: Comprehensive evaluation results with metrics

        Raises:
            ValueError: If input graphs are invalid
            TimeoutError: If evaluation exceeds max_evaluation_time
        """

    def evaluate_batch(self,
                      graph_pairs: List[Tuple[List[Triple], List[Triple]]]) -> List[EvaluationResult]:
        """
        Evaluate multiple graph pairs in batch mode.

        Args:
            graph_pairs: List of (predicted_graph, reference_graph) tuples

        Returns:
            List[EvaluationResult]: Results for each graph pair
        """

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get current evaluator configuration."""

    def set_evaluation_config(self, config: Dict[str, Any]) -> None:
        """Update evaluator configuration."""
```

#### Key Methods

##### evaluate_graph()

The primary evaluation method for single graph pairs.

```python
# Basic usage
evaluator = GraphEvaluator()
result = evaluator.evaluate_graph(predicted_triples, reference_triples)

# With custom configuration
evaluator = GraphEvaluator(
    enable_ged=True,           # Enable expensive Graph Edit Distance
    enable_bert_score=True,    # Enable semantic similarity
    max_evaluation_time=60.0   # Allow longer evaluation time
)
result = evaluator.evaluate_graph(predicted_triples, reference_triples)

# Check results
if result.success:
    print(f"Overall Score: {result.metrics.get_overall_score():.3f}")
    print(f"Triple Match F1: {result.metrics.triple_match_f1:.3f}")
    print(f"Processing Time: {result.processing_time:.2f}s")
else:
    print(f"Evaluation failed: {result.error}")
```

##### evaluate_batch()

Batch evaluation for multiple graph pairs.

```python
# Prepare multiple graph pairs
graph_pairs = [
    (predicted_graph_1, reference_graph_1),
    (predicted_graph_2, reference_graph_2),
    (predicted_graph_3, reference_graph_3)
]

# Evaluate in batch
results = evaluator.evaluate_batch(graph_pairs)

# Process results
for i, result in enumerate(results):
    if result.success:
        print(f"Graph {i+1}: Score = {result.metrics.get_overall_score():.3f}")
    else:
        print(f"Graph {i+1}: Failed - {result.error}")
```

---

## Utility Components

### ReferenceGraphManager

Manages reference graph loading and format conversion.

```python
class ReferenceGraphManager:
    """
    Manages reference graph loading, validation, and format conversion.
    """

    def __init__(self, max_graph_size: int = 1000):
        """
        Initialize reference graph manager.

        Args:
            max_graph_size: Maximum number of triples allowed in reference graph
        """

    def load_reference_graph(self, file_path: str) -> List[Triple]:
        """
        Load reference graph from file (JSON, CSV, or TXT format).

        Args:
            file_path: Path to reference graph file

        Returns:
            List[Triple]: Loaded reference graph

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid
            Exception: If graph exceeds maximum size limit
        """

    def validate_reference_graph(self, graph: List[Triple]) -> Dict[str, Any]:
        """
        Validate reference graph and return statistics.

        Args:
            graph: Reference graph to validate

        Returns:
            Dict[str, Any]: Validation statistics and metadata
        """

    def convert_format(self, input_path: str, output_path: str, output_format: str) -> None:
        """
        Convert reference graph between supported formats.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            output_format: Target format ('json', 'csv', 'txt')
        """
```

#### Supported Formats

##### JSON Format
```json
[
    {
        "subject": "Paris",
        "predicate": "capital_of",
        "object": "France",
        "confidence": 0.95,
        "metadata": {}
    },
    {
        "subject": "France",
        "predicate": "located_in",
        "object": "Europe"
    }
]
```

##### CSV Format
```csv
subject,predicate,object,confidence
"Paris","capital_of","France",0.95
"France","located_in","Europe",
```

##### TXT Format
```
(Paris, capital_of, France)
(France, located_in, Europe)
```

---

## Configuration

### Evaluation Configuration

Configuration options for the evaluation system.

```python
def get_evaluation_config() -> Dict[str, Any]:
    """
    Get evaluation configuration with defaults and environment overrides.

    Returns:
        Dictionary containing evaluation configuration options
    """
    return {
        'enable_evaluation': bool,         # Whether evaluation is enabled
        'enable_ged': bool,               # Graph Edit Distance (expensive)
        'enable_bert_score': bool,        # Semantic similarity using BERT
        'max_evaluation_time': float,     # Maximum evaluation time in seconds
        'batch_size': int,                # Maximum graphs to evaluate in batch
        'reference_graph_max_size': int,  # Maximum reference graph size
        'supported_formats': List[str],   # Supported file formats
        'temp_dir': str                   # Temporary storage directory
    }
```

### Environment Variables

```bash
# Evaluation system configuration
EVALUATION_ENABLED=true                    # Enable/disable evaluation
EVALUATION_ENABLE_GED=false               # Enable Graph Edit Distance
EVALUATION_ENABLE_BERT_SCORE=true         # Enable BertScore
EVALUATION_TIMEOUT=30.0                   # Evaluation timeout in seconds
EVALUATION_BATCH_SIZE=10                  # Batch evaluation size
REFERENCE_GRAPH_MAX_SIZE=1000             # Max reference graph size
REFERENCE_GRAPH_TEMP_DIR=temp/ref_graphs  # Temporary directory
```

### Runtime Configuration

```python
# Configure evaluation at runtime
from streamlit_pipeline.core.config import get_evaluation_config

config = get_evaluation_config()
evaluator = GraphEvaluator(
    enable_ged=config['enable_ged'],
    enable_bert_score=config['enable_bert_score'],
    max_evaluation_time=config['max_evaluation_time']
)
```

---

## Integration Patterns

### Pipeline Integration

Integration with the main GraphJudge pipeline.

```python
from streamlit_pipeline.core.pipeline import PipelineOrchestrator

# Enable evaluation in pipeline
orchestrator = PipelineOrchestrator(enable_evaluation=True)

# Run pipeline with evaluation
result = orchestrator.run_pipeline(
    input_text="Your input text here",
    reference_graph_path="path/to/reference.json"
)

# Access evaluation results
if result.evaluation_success:
    metrics = result.evaluation_metrics
    print(f"Graph Quality Score: {metrics.get_overall_score():.3f}")
```

### Streamlit Integration

Integration with Streamlit UI components.

```python
import streamlit as st
from streamlit_pipeline.ui.evaluation_display import (
    display_evaluation_dashboard,
    display_evaluation_configuration
)

# Display evaluation configuration
evaluation_config = display_evaluation_configuration()

# Display evaluation results
if st.session_state.get('evaluation_result'):
    display_evaluation_dashboard(st.session_state.evaluation_result)
```

### Session State Integration

Integration with session state management.

```python
from streamlit_pipeline.utils.session_state import SessionStateManager

# Initialize session manager
session_manager = SessionStateManager()

# Store evaluation results
session_manager.set_evaluation_result(evaluation_result)

# Retrieve evaluation results
stored_result = session_manager.get_evaluation_result()
```

---

## Error Handling

### Error Types

The evaluation system defines several error types for comprehensive error handling:

```python
from streamlit_pipeline.utils.error_handling import ErrorType

# Configuration errors
ErrorType.CONFIGURATION_ERROR    # Missing or invalid configuration
ErrorType.DEPENDENCY_ERROR       # Missing optional dependencies

# Input validation errors
ErrorType.VALIDATION_ERROR       # Invalid input data
ErrorType.FILE_FORMAT_ERROR      # Unsupported file format

# Runtime errors
ErrorType.TIMEOUT_ERROR          # Evaluation timeout exceeded
ErrorType.MEMORY_ERROR           # Insufficient memory
ErrorType.PROCESSING_ERROR       # General processing failure
```

### Error Handling Patterns

```python
from streamlit_pipeline.utils.error_handling import safe_execute

# Safe evaluation with error handling
def safe_evaluate_graph(predicted, reference):
    def evaluation_operation():
        evaluator = GraphEvaluator()
        return evaluator.evaluate_graph(predicted, reference)

    return safe_execute(
        operation=evaluation_operation,
        operation_name="Graph Evaluation",
        expected_errors=[ValueError, TimeoutError, ImportError]
    )

# Usage
result, error = safe_evaluate_graph(predicted_graph, reference_graph)
if error:
    print(f"Evaluation failed: {error}")
else:
    print(f"Evaluation succeeded: {result.metrics.get_overall_score():.3f}")
```

### Graceful Degradation

The system provides graceful degradation when optional dependencies are unavailable:

```python
# Example: BertScore fallback
try:
    import bert_score
    bert_available = True
except ImportError:
    bert_available = False

if bert_available:
    # Use actual BertScore
    g_bert_f1 = calculate_bert_score(predicted, reference)
else:
    # Fallback to default value
    g_bert_f1 = 0.0
    logger.warning("BertScore not available, using fallback value")
```

---

## Performance Considerations

### Performance Requirements

- **Evaluation Overhead**: <500ms for typical graphs (100 triples)
- **Memory Usage**: <100MB additional memory for large graphs
- **Concurrent Evaluation**: Support for multiple simultaneous evaluations
- **Batch Processing**: Efficient processing of multiple graph pairs

### Optimization Strategies

#### Lazy Loading
```python
# Dependencies are loaded only when needed
class GraphEvaluator:
    def __init__(self):
        self._bert_scorer = None

    @property
    def bert_scorer(self):
        if self._bert_scorer is None and self.enable_bert_score:
            from bert_score import BERTScorer
            self._bert_scorer = BERTScorer(lang="en")
        return self._bert_scorer
```

#### Caching
```python
# Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_graph_embedding(graph_hash: str) -> np.ndarray:
    """Cache graph embeddings for repeated evaluations."""
    pass
```

#### Timeout Management
```python
# Set appropriate timeouts for different graph sizes
def get_adaptive_timeout(graph_size: int) -> float:
    """Calculate adaptive timeout based on graph size."""
    base_timeout = 10.0
    size_factor = max(1.0, graph_size / 100.0)
    return min(base_timeout * size_factor, 300.0)  # Cap at 5 minutes
```

### Performance Monitoring

```python
# Monitor evaluation performance
def monitor_evaluation_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        logger.info(f"Evaluation completed in {end_time - start_time:.3f}s")
        logger.info(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")

        return result
    return wrapper
```

---

## Examples

### Basic Evaluation

```python
from streamlit_pipeline.eval.graph_evaluator import GraphEvaluator
from streamlit_pipeline.core.models import Triple

# Create sample graphs
predicted_graph = [
    Triple("Paris", "capital_of", "France"),
    Triple("London", "capital_of", "UK"),
    Triple("France", "located_in", "Europe")
]

reference_graph = [
    Triple("Paris", "capital_of", "France"),
    Triple("Berlin", "capital_of", "Germany"),
    Triple("France", "located_in", "Europe")
]

# Evaluate
evaluator = GraphEvaluator()
result = evaluator.evaluate_graph(predicted_graph, reference_graph)

# Display results
if result.success:
    print(f"Evaluation Results:")
    print(f"  Overall Score: {result.metrics.get_overall_score():.3f}")
    print(f"  Triple Match F1: {result.metrics.triple_match_f1:.3f}")
    print(f"  Graph Match Accuracy: {result.metrics.graph_match_accuracy:.3f}")
    print(f"  Processing Time: {result.processing_time:.3f}s")
else:
    print(f"Evaluation failed: {result.error}")
```

### Advanced Configuration

```python
# Configure evaluation with all options
evaluator = GraphEvaluator(
    enable_ged=True,                    # Enable Graph Edit Distance
    enable_bert_score=True,             # Enable semantic similarity
    max_evaluation_time=60.0            # Allow longer evaluation
)

# Evaluate with comprehensive metrics
result = evaluator.evaluate_graph(predicted_graph, reference_graph)

if result.success:
    metrics = result.metrics
    print(f"Comprehensive Evaluation Results:")
    print(f"  Triple Match F1: {metrics.triple_match_f1:.3f}")
    print(f"  G-BLEU F1: {metrics.g_bleu_f1:.3f}")
    print(f"  G-ROUGE F1: {metrics.g_rouge_f1:.3f}")
    print(f"  G-BertScore F1: {metrics.g_bert_f1:.3f}")
    if metrics.graph_edit_distance is not None:
        print(f"  Graph Edit Distance: {metrics.graph_edit_distance:.3f}")
```

### File-Based Evaluation

```python
from streamlit_pipeline.utils.reference_graph_manager import ReferenceGraphManager

# Load reference graph from file
ref_manager = ReferenceGraphManager()
reference_graph = ref_manager.load_reference_graph("reference.json")

# Validate reference graph
validation_stats = ref_manager.validate_reference_graph(reference_graph)
print(f"Reference graph statistics: {validation_stats}")

# Evaluate against loaded reference
result = evaluator.evaluate_graph(predicted_graph, reference_graph)
```

### Batch Evaluation

```python
# Prepare multiple graph pairs
graph_pairs = [
    (predicted_graph_1, reference_graph_1),
    (predicted_graph_2, reference_graph_2),
    (predicted_graph_3, reference_graph_3)
]

# Batch evaluation
results = evaluator.evaluate_batch(graph_pairs)

# Analyze batch results
successful_evaluations = [r for r in results if r.success]
average_score = sum(r.metrics.get_overall_score() for r in successful_evaluations) / len(successful_evaluations)

print(f"Batch Evaluation Results:")
print(f"  Successful evaluations: {len(successful_evaluations)}/{len(results)}")
print(f"  Average quality score: {average_score:.3f}")
```

### Export and Reporting

```python
# Export evaluation results
result_dict = result.to_dict()

# Save to JSON file
import json
with open("evaluation_results.json", "w") as f:
    json.dump(result_dict, f, indent=2)

# Generate human-readable summary
summary = result.export_summary()
print(summary)

# Save summary to file
with open("evaluation_summary.txt", "w") as f:
    f.write(summary)
```

---

This API reference provides comprehensive documentation for integrating and using the GraphJudge evaluation system. For research usage guidelines and best practices, see the [Research Usage Guidelines](EVALUATION_RESEARCH_GUIDE.md).

For performance optimization and troubleshooting, refer to the inline documentation and error handling patterns described above.