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

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time


@dataclass
class Triple:
    """
    Represents a single knowledge graph triple (subject, predicate, object).
    
    Used by triple generator and graph judge modules for consistent
    representation of extracted knowledge facts.
    """
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None


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