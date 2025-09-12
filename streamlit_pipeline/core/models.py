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


class ConfidenceLevel(Enum):
    """
    Enumeration for confidence levels in judgments and extractions.
    """
    VERY_LOW = (0.0, 0.2)      # 0-20% confidence
    LOW = (0.2, 0.4)           # 20-40% confidence  
    MEDIUM = (0.4, 0.6)        # 40-60% confidence
    HIGH = (0.6, 0.8)          # 60-80% confidence
    VERY_HIGH = (0.8, 1.0)     # 80-100% confidence
    
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val
    
    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        """Convert numerical confidence score to confidence level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


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
        confidence: Optional confidence score (0.0-1.0)
        source_text: Optional original text that generated this triple
        metadata: Additional information about the triple
    """
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    source_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate triple data after initialization."""
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        """Get confidence level enum for this triple."""
        if self.confidence is not None:
            return ConfidenceLevel.from_score(self.confidence)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary representation."""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
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
            confidence=data.get('confidence'),
            source_text=data.get('source_text'),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation of the triple."""
        conf_str = f" (confidence: {self.confidence:.2f})" if self.confidence else ""
        return f"({self.subject}, {self.predicate}, {self.object}){conf_str}"


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
    
    Contains judgment decisions, confidence scores, and optional
    explanations for each evaluated triple.
    
    Attributes:
        judgments: List of True/False decisions for each triple
        confidence: List of confidence scores (0-1) for each judgment
        explanations: Optional detailed explanations for explainable mode
        success: Whether the operation completed successfully
        processing_time: Time taken for the operation in seconds
        error: Error message if operation failed, None otherwise
    """
    judgments: List[bool]  # True/False for each triple
    confidence: List[float]  # Confidence scores (0-1)
    explanations: Optional[List[str]] = None  # For explainable mode
    success: bool = True
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            'judgments': self.judgments,
            'confidence': self.confidence,
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
            confidence=[],
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