"""
Data Models Module

This module defines the core data structures used throughout the ECTD pipeline:
- Entity: Individual entity representation with metadata
- EntityList: Collection of entities with deduplication
- EntityType: Enumeration of supported entity types
- PipelineState: Execution state and progress tracking
- ProcessingStatus: Status enumeration for pipeline stages

All models use Pydantic for validation and serialization, ensuring data integrity.
"""

from .entities import Entity, EntityList, EntityType
from .pipeline_state import PipelineState, ProcessingStatus

__all__ = [
    "Entity",
    "EntityList", 
    "EntityType",
    "PipelineState",
    "ProcessingStatus",
]
