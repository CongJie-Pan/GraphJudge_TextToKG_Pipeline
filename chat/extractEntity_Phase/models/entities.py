"""
Entity Data Structures Module

This module defines the core data structures for entities extracted from Chinese text,
including individual entities, entity collections, and entity type classifications.

All models use Pydantic for validation and serialization, ensuring data integrity
and providing a clean interface for entity processing throughout the pipeline.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re


class EntityType(str, Enum):
    """Enumeration of supported entity types for Chinese text."""
    
    PERSON = "person"           # 人物 (e.g., 賈寶玉, 林黛玉)
    LOCATION = "location"       # 地點 (e.g., 大觀園, 榮國府)
    ORGANIZATION = "organization"  # 組織 (e.g., 賈府, 榮國府)
    OBJECT = "object"           # 物品 (e.g., 通靈寶玉, 金鎖)
    CONCEPT = "concept"         # 概念 (e.g., 太虛幻境, 木石前盟)
    EVENT = "event"             # 事件 (e.g., 葬花, 結社)
    TIME = "time"               # 時間 (e.g., 春, 秋)
    OTHER = "other"             # 其他未分類實體


class Entity(BaseModel):
    """
    Individual entity representation with metadata.
    
    This class represents a single entity extracted from Chinese text,
    including its text content, type, confidence, and contextual information.
    """
    
    text: str = Field(..., description="Entity text content")
    type: EntityType = Field(..., description="Entity type classification")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    start_pos: Optional[int] = Field(None, ge=0, description="Starting position in source text")
    end_pos: Optional[int] = Field(None, ge=0, description="Ending position in source text")
    source_text: Optional[str] = Field(None, description="Source text context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional entity metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Entity creation timestamp")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate entity text content."""
        if not v or not v.strip():
            raise ValueError("Entity text cannot be empty")
        
        # Ensure text is properly encoded for Chinese characters
        if not v.strip():
            raise ValueError("Entity text cannot be whitespace only")
        
        return v.strip()
    
    @validator('end_pos')
    def validate_positions(cls, v, values):
        """Validate position consistency."""
        if v is not None and 'start_pos' in values and values['start_pos'] is not None:
            if v <= values['start_pos']:
                raise ValueError("End position must be greater than start position")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 3)
    
    def __str__(self) -> str:
        """String representation of entity."""
        return f"{self.text} ({self.type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of entity."""
        return f"Entity(text='{self.text}', type={self.type.value}, confidence={self.confidence})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "text": self.text,
            "type": self.type.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "source_text": self.source_text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary."""
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def is_chinese(self) -> bool:
        """Check if entity text contains Chinese characters."""
        # Pattern for Chinese characters (including traditional)
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        return bool(chinese_pattern.search(self.text))
    
    def get_length(self) -> int:
        """Get entity text length in characters."""
        return len(self.text)
    
    def get_word_count(self) -> int:
        """Get entity text word count (approximate for Chinese)."""
        # For Chinese text, approximate word count by character count
        # This is a simplified approach; more sophisticated tokenization could be used
        return len(self.text.strip())


class EntityList(BaseModel):
    """
    Collection of entities with deduplication and validation.
    
    This class manages a collection of entities, providing methods for
    deduplication, filtering, and analysis of entity collections.
    """
    
    entities: List[Entity] = Field(default_factory=list, description="List of entities")
    source_text: Optional[str] = Field(None, description="Source text for all entities")
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
    
    @validator('entities')
    def validate_entities(cls, v):
        """Validate entity list."""
        if not isinstance(v, list):
            raise ValueError("Entities must be a list")
        return v
    
    def __len__(self) -> int:
        """Get number of entities in collection."""
        return len(self.entities)
    
    def __getitem__(self, index: int) -> Entity:
        """Get entity by index."""
        return self.entities[index]
    
    def __iter__(self):
        """Iterate over entities."""
        return iter(self.entities)
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add entity to collection with deduplication.
        
        Args:
            entity: Entity to add
        """
        if not isinstance(entity, Entity):
            raise ValueError("Can only add Entity objects")
        
        # Check for duplicates (same text and type)
        if not self._is_duplicate(entity):
            self.entities.append(entity)
    
    def _is_duplicate(self, entity: Entity) -> bool:
        """
        Check if entity is duplicate based on text and type.
        
        Args:
            entity: Entity to check for duplication
            
        Returns:
            True if duplicate found, False otherwise
        """
        for existing_entity in self.entities:
            if (existing_entity.text == entity.text and 
                existing_entity.type == entity.type):
                return True
        return False
    
    def remove_duplicates(self) -> None:
        """Remove duplicate entities from collection."""
        seen = set()
        unique_entities = []
        
        for entity in self.entities:
            # Create unique identifier for deduplication
            identifier = (entity.text, entity.type)
            
            if identifier not in seen:
                seen.add(identifier)
                unique_entities.append(entity)
        
        self.entities = unique_entities
    
    def filter_by_type(self, entity_type: EntityType) -> 'EntityList':
        """
        Filter entities by type.
        
        Args:
            entity_type: Entity type to filter by
            
        Returns:
            New EntityList with filtered entities
        """
        filtered_entities = [e for e in self.entities if e.type == entity_type]
        return EntityList(
            entities=filtered_entities,
            source_text=self.source_text,
            extraction_timestamp=self.extraction_timestamp,
            metadata=self.metadata
        )
    
    def filter_by_confidence(self, min_confidence: float) -> 'EntityList':
        """
        Filter entities by minimum confidence.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            New EntityList with filtered entities
        """
        filtered_entities = [e for e in self.entities if e.confidence >= min_confidence]
        return EntityList(
            entities=filtered_entities,
            source_text=self.source_text,
            extraction_timestamp=self.extraction_timestamp,
            metadata=self.metadata
        )
    
    def get_entity_types(self) -> List[EntityType]:
        """Get list of unique entity types in collection."""
        return list(set(entity.type for entity in self.entities))
    
    def get_type_counts(self) -> Dict[EntityType, int]:
        """Get count of entities by type."""
        type_counts = {}
        for entity in self.entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        return type_counts
    
    def get_average_confidence(self) -> float:
        """Get average confidence score for all entities."""
        if not self.entities:
            return 0.0
        
        total_confidence = sum(entity.confidence for entity in self.entities)
        return round(total_confidence / len(self.entities), 3)
    
    def sort_by_confidence(self, reverse: bool = True) -> None:
        """
        Sort entities by confidence score.
        
        Args:
            reverse: If True, sort in descending order (highest first)
        """
        self.entities.sort(key=lambda x: x.confidence, reverse=reverse)
    
    def sort_by_text_length(self, reverse: bool = False) -> None:
        """
        Sort entities by text length.
        
        Args:
            reverse: If True, sort in descending order (longest first)
        """
        self.entities.sort(key=lambda x: len(x.text), reverse=reverse)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity list to dictionary for serialization."""
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "source_text": self.source_text,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "metadata": self.metadata,
            "entity_count": len(self.entities),
            "entity_types": [t.value for t in self.get_entity_types()],
            "type_counts": {k.value: v for k, v in self.get_type_counts().items()},
            "average_confidence": self.get_average_confidence()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityList':
        """Create entity list from dictionary."""
        # Handle datetime conversion
        if 'extraction_timestamp' in data and isinstance(data['extraction_timestamp'], str):
            data['extraction_timestamp'] = datetime.fromisoformat(data['extraction_timestamp'])
        
        # Convert entity dictionaries to Entity objects
        if 'entities' in data:
            data['entities'] = [Entity.from_dict(e) for e in data['entities']]
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of entity list."""
        return f"EntityList({len(self.entities)} entities, {len(self.get_entity_types())} types)"
    
    def __repr__(self) -> str:
        """Detailed string representation of entity list."""
        return f"EntityList(entities={self.entities}, source_text='{self.source_text}', entity_count={len(self.entities)})"


class EntityExtractionResult(BaseModel):
    """
    Result of entity extraction operation.
    
    This class represents the complete result of an entity extraction operation,
    including the extracted entities, processing statistics, and metadata.
    """
    
    entities: EntityList = Field(..., description="Extracted entities")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    text_length: int = Field(..., ge=0, description="Length of processed text")
    extraction_method: str = Field(..., description="Method used for extraction")
    model_version: Optional[str] = Field(None, description="AI model version used")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        """Validate processing time."""
        if v < 0:
            raise ValueError("Processing time cannot be negative")
        return round(v, 3)
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics."""
        return {
            "total_entities": len(self.entities),
            "entity_types": self.entities.get_type_counts(),
            "average_confidence": self.entities.get_average_confidence(),
            "processing_time": self.processing_time,
            "text_length": self.text_length,
            "entities_per_character": len(self.entities) / self.text_length if self.text_length > 0 else 0,
            "extraction_method": self.extraction_method,
            "model_version": self.model_version,
            "confidence_threshold": self.confidence_threshold
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to dictionary for serialization."""
        return {
            "entities": self.entities.to_dict(),
            "processing_time": self.processing_time,
            "text_length": self.text_length,
            "extraction_method": self.extraction_method,
            "model_version": self.model_version,
            "confidence_threshold": self.confidence_threshold,
            "metadata": self.metadata,
            "statistics": self.get_extraction_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityExtractionResult':
        """Create extraction result from dictionary."""
        # Convert entities dictionary to EntityList
        if 'entities' in data and isinstance(data['entities'], dict):
            data['entities'] = EntityList.from_dict(data['entities'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of extraction result."""
        return f"EntityExtractionResult({len(self.entities)} entities, {self.processing_time}s)"
    
    def __repr__(self) -> str:
        """Detailed string representation of extraction result."""
        return f"EntityExtractionResult(entities={self.entities}, processing_time={self.processing_time}, text_length={self.text_length})"
