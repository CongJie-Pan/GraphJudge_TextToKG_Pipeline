"""
Shared data models for GraphJudge Streamlit Pipeline.

This module defines the core data structures used across all pipeline stages,
following the specifications in spec.md Section 8. These models replace
the complex file-based data exchange from the original CLI scripts.

Key principles:
- All data passed as Python objects in memory
- Unified error handling (errors returned as data, not exceptions)
- Self-contained results with metadata
- Optional features with simple flags
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum, auto
from datetime import datetime
import time
import json


class PipelineStage(Enum):
    """
    Enumeration of pipeline processing stages.
    
    Used for tracking progress and managing state transitions
    across the three-stage NLP pipeline.
    """
    ENTITY_EXTRACTION = "entity_extraction"
    TRIPLE_GENERATION = "triple_generation" 
    GRAPH_JUDGMENT = "graph_judgment"


class ProcessingStatus(Enum):
    """
    Enumeration of processing status states.
    
    Corresponds to the state machine defined in spec.md Section 9.
    """
    DRAFT = "draft"                    # User editing input
    QUEUED = "queued"                  # Ready to start processing
    RUNNING_ENTITY = "running_entity"  # Entity extraction in progress
    RUNNING_TRIPLE = "running_triple"  # Triple generation in progress
    RUNNING_GJ = "running_gj"          # Graph judgment in progress
    SUCCEEDED = "succeeded"            # All stages completed successfully
    FAILED = "failed"                  # Unrecoverable error occurred
    ARCHIVED = "archived"              # Job complete, results displayed




@dataclass
class Triple:
    """
    Represents a single knowledge graph triple (subject, predicate, object).
    
    Used by triple generator and graph judge modules for consistent
    representation of extracted knowledge facts.
    
    Attributes:
        subject: The subject entity of the triple
        predicate: The relationship/predicate connecting subject and object
        object: The object entity of the triple
        source_text: Optional original text that generated this triple
        metadata: Additional information about the triple
    """
    subject: str
    predicate: str
    object: str
    source_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary representation."""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'source_text': self.source_text,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        """Create triple from dictionary representation."""
        return cls(
            subject=data['subject'],
            predicate=data['predicate'],
            object=data['object'],
            source_text=data.get('source_text'),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation of the triple."""
        return f"({self.subject}, {self.predicate}, {self.object})"


@dataclass
class EntityResult:
    """
    Result object for entity extraction and text denoising operations.
    
    Contains both extracted entities and denoised text, along with
    processing metadata and error information.
    
    Attributes:
        entities: List of extracted entity names
        denoised_text: Cleaned and restructured text
        success: Whether the operation completed successfully
        processing_time: Time taken for the operation in seconds
        error: Error message if operation failed, None otherwise
    """
    entities: List[str]
    denoised_text: str
    success: bool
    processing_time: float
    error: Optional[str] = None


@dataclass
class TripleResult:
    """
    Result object for triple generation operations.
    
    Contains generated triples along with validation metadata
    and processing information.
    
    Attributes:
        triples: List of extracted Triple objects
        metadata: Processing statistics and validation info
        success: Whether the operation completed successfully
        processing_time: Time taken for the operation in seconds
        error: Error message if operation failed, None otherwise
    """
    triples: List[Triple]
    metadata: Dict[str, Any]  # Contains validation stats, chunk info, etc.
    success: bool
    processing_time: float
    error: Optional[str] = None


@dataclass
class JudgmentResult:
    """
    Result object for graph judgment operations.
    
    Contains judgment decisions and optional
    explanations for each evaluated triple.
    
    Attributes:
        judgments: List of True/False decisions for each triple
        explanations: Optional detailed explanations for explainable mode
        success: Whether the operation completed successfully
        processing_time: Time taken for the operation in seconds
        error: Error message if operation failed, None otherwise
    """
    judgments: List[bool]  # True/False for each triple
    explanations: Optional[List[str]] = None  # For explainable mode
    success: bool = True
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            'judgments': self.judgments,
            'explanations': self.explanations,
            'success': self.success,
            'processing_time': self.processing_time,
            'error': self.error
        }


@dataclass
class PipelineState:
    """
    Represents the complete state of a pipeline execution.
    
    This class tracks the progress through all three stages and maintains
    intermediate results for session state management in Streamlit.
    """
    # Input data
    input_text: str = ""
    
    # Processing status
    status: ProcessingStatus = ProcessingStatus.DRAFT
    current_stage: Optional[PipelineStage] = None
    
    # Stage results
    entity_result: Optional[EntityResult] = None
    triple_result: Optional[TripleResult] = None
    judgment_result: Optional[JudgmentResult] = None
    
    # Overall timing and metadata
    total_processing_time: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Error information
    error_stage: Optional[PipelineStage] = None
    error_message: Optional[str] = None
    
    # Session metadata
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline has completed successfully."""
        return self.status == ProcessingStatus.SUCCEEDED
    
    @property
    def has_error(self) -> bool:
        """Check if pipeline has encountered an error."""
        return self.status == ProcessingStatus.FAILED
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage based on completed stages."""
        if self.status in [ProcessingStatus.DRAFT, ProcessingStatus.QUEUED]:
            return 0.0
        elif self.status == ProcessingStatus.RUNNING_ENTITY:
            return 10.0
        elif self.entity_result and self.status == ProcessingStatus.RUNNING_TRIPLE:
            return 40.0
        elif self.triple_result and self.status == ProcessingStatus.RUNNING_GJ:
            return 70.0
        elif self.status == ProcessingStatus.SUCCEEDED:
            return 100.0
        elif self.status == ProcessingStatus.FAILED:
            # Return progress up to failed stage
            if self.error_stage == PipelineStage.ENTITY_EXTRACTION:
                return 10.0
            elif self.error_stage == PipelineStage.TRIPLE_GENERATION:
                return 40.0
            elif self.error_stage == PipelineStage.GRAPH_JUDGMENT:
                return 70.0
        return 0.0
    
    def get_completed_stages(self) -> List[PipelineStage]:
        """Get list of successfully completed stages."""
        completed = []
        if self.entity_result and self.entity_result.success:
            completed.append(PipelineStage.ENTITY_EXTRACTION)
        if self.triple_result and self.triple_result.success:
            completed.append(PipelineStage.TRIPLE_GENERATION)
        if self.judgment_result and self.judgment_result.success:
            completed.append(PipelineStage.GRAPH_JUDGMENT)
        return completed
    
    def reset(self):
        """Reset pipeline state to initial draft state."""
        self.status = ProcessingStatus.DRAFT
        self.current_stage = None
        self.entity_result = None
        self.triple_result = None
        self.judgment_result = None
        self.total_processing_time = 0.0
        self.started_at = None
        self.completed_at = None
        self.error_stage = None
        self.error_message = None


def create_error_result(result_type: type, error_message: str, processing_time: float = 0.0):
    """
    Helper function to create error result objects with consistent structure.
    
    Args:
        result_type: The result class to instantiate (EntityResult, TripleResult, etc.)
        error_message: Description of the error that occurred
        processing_time: Time spent before error occurred
        
    Returns:
        Error result object with success=False and appropriate default values
    """
    if result_type == EntityResult:
        return EntityResult(
            entities=[],
            denoised_text="",
            success=False,
            processing_time=processing_time,
            error=error_message
        )
    elif result_type == TripleResult:
        return TripleResult(
            triples=[],
            metadata={},
            success=False,
            processing_time=processing_time,
            error=error_message
        )
    elif result_type == JudgmentResult:
        return JudgmentResult(
            judgments=[],
            explanations=None,
            success=False,
            processing_time=processing_time,
            error=error_message
        )
    else:
        raise ValueError(f"Unsupported result type: {result_type}")


class ProcessingTimer:
    """
    Context manager for timing operations and creating result objects.
    
    Usage:
        with ProcessingTimer() as timer:
            # do processing
            pass
        result = EntityResult(
            entities=entities,
            denoised_text=text,
            success=True,
            processing_time=timer.elapsed
        )
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


# ============================================================================
# Graph Quality Evaluation Models
# ============================================================================

@dataclass
class GraphMetrics:
    """
    Comprehensive graph evaluation metrics based on multiple assessment dimensions.

    This class encapsulates all evaluation metrics computed by the graph evaluator,
    providing a structured representation of graph quality assessment results.
    Based on metrics from graph_evaluation/metrics/eval.py.
    """
    # Exact matching metrics
    triple_match_f1: float  # F1 score for exact triple matching
    graph_match_accuracy: float  # Structural graph isomorphism accuracy

    # Text similarity metrics (G-BLEU)
    g_bleu_precision: float  # BLEU precision for graph edges
    g_bleu_recall: float     # BLEU recall for graph edges
    g_bleu_f1: float         # BLEU F1 score for graph edges

    # Text similarity metrics (G-ROUGE)
    g_rouge_precision: float # ROUGE precision for graph edges
    g_rouge_recall: float    # ROUGE recall for graph edges
    g_rouge_f1: float        # ROUGE F1 score for graph edges

    # Semantic similarity metrics (G-BertScore)
    g_bert_precision: float  # BertScore precision for graph edges
    g_bert_recall: float     # BertScore recall for graph edges
    g_bert_f1: float         # BertScore F1 score for graph edges

    # Optional structural distance metric
    graph_edit_distance: Optional[float] = None  # Average graph edit distance (computationally expensive)

    def get_overall_score(self) -> float:
        """
        Calculate an overall quality score by averaging key metrics.

        Returns:
            float: Overall quality score between 0.0 and 1.0
        """
        key_metrics = [
            self.triple_match_f1,
            self.graph_match_accuracy,
            self.g_bleu_f1,
            self.g_rouge_f1,
            self.g_bert_f1
        ]
        return sum(key_metrics) / len(key_metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format for export."""
        return {
            "exact_matching": {
                "triple_match_f1": self.triple_match_f1,
                "graph_match_accuracy": self.graph_match_accuracy
            },
            "text_similarity": {
                "g_bleu": {
                    "precision": self.g_bleu_precision,
                    "recall": self.g_bleu_recall,
                    "f1": self.g_bleu_f1
                },
                "g_rouge": {
                    "precision": self.g_rouge_precision,
                    "recall": self.g_rouge_recall,
                    "f1": self.g_rouge_f1
                }
            },
            "semantic_similarity": {
                "g_bert_score": {
                    "precision": self.g_bert_precision,
                    "recall": self.g_bert_recall,
                    "f1": self.g_bert_f1
                }
            },
            "structural_distance": {
                "graph_edit_distance": self.graph_edit_distance
            },
            "overall_score": self.get_overall_score()
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result containing metrics, metadata, and processing information.

    This class provides a comprehensive container for graph evaluation results,
    including detailed metrics, processing metadata, and error information.
    """
    metrics: GraphMetrics                    # Computed evaluation metrics
    metadata: Dict[str, Any]                 # Evaluation parameters, timestamps, etc.
    success: bool                            # Whether evaluation completed successfully
    processing_time: float                   # Time taken for evaluation in seconds
    error: Optional[str] = None              # Error message if evaluation failed
    reference_graph_info: Optional[Dict[str, Any]] = None  # Reference graph statistics
    predicted_graph_info: Optional[Dict[str, Any]] = None  # Predicted graph statistics

    def __post_init__(self):
        """Initialize metadata with default values if not provided."""
        if not self.metadata:
            self.metadata = {}

        # Add timestamp if not present
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()

        # Add evaluation version if not present
        if 'evaluation_version' not in self.metadata:
            self.metadata['evaluation_version'] = '1.0'

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary format for export."""
        result = {
            "evaluation_summary": {
                "success": self.success,
                "processing_time": self.processing_time,
                "overall_score": self.metrics.get_overall_score() if self.success else 0.0,
                "timestamp": self.metadata.get('timestamp'),
                "evaluation_version": self.metadata.get('evaluation_version')
            },
            "metrics": self.metrics.to_dict() if self.success else {},
            "metadata": self.metadata
        }

        if self.error:
            result["error"] = self.error

        if self.reference_graph_info:
            result["reference_graph"] = self.reference_graph_info

        if self.predicted_graph_info:
            result["predicted_graph"] = self.predicted_graph_info

        return result

    def export_summary(self) -> str:
        """Generate a human-readable summary of evaluation results."""
        if not self.success:
            return f"Evaluation failed: {self.error}"

        summary_lines = [
            f"Graph Evaluation Summary",
            f"========================",
            f"Overall Score: {self.metrics.get_overall_score():.3f}",
            f"Processing Time: {self.processing_time:.2f}s",
            f"",
            f"Exact Matching Metrics:",
            f"  Triple Match F1: {self.metrics.triple_match_f1:.3f}",
            f"  Graph Match Accuracy: {self.metrics.graph_match_accuracy:.3f}",
            f"",
            f"Text Similarity Metrics:",
            f"  G-BLEU F1: {self.metrics.g_bleu_f1:.3f}",
            f"  G-ROUGE F1: {self.metrics.g_rouge_f1:.3f}",
            f"",
            f"Semantic Similarity Metrics:",
            f"  G-BertScore F1: {self.metrics.g_bert_f1:.3f}",
        ]

        if self.metrics.graph_edit_distance is not None:
            summary_lines.extend([
                f"",
                f"Structural Distance:",
                f"  Graph Edit Distance: {self.metrics.graph_edit_distance:.3f}"
            ])

        return "\n".join(summary_lines)