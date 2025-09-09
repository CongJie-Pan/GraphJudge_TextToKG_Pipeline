"""
Tests for entity data structures.
"""

import pytest
from datetime import datetime
from extractEntity_Phase.models.entities import (
    Entity, EntityList, EntityExtractionResult, EntityType
)


class TestEntityType:
    """Test entity type enumeration."""
    
    def test_entity_type_values(self):
        """Test entity type enum values."""
        assert EntityType.PERSON.value == "person"
        assert EntityType.LOCATION.value == "location"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.OBJECT.value == "object"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.EVENT.value == "event"
        assert EntityType.TIME.value == "time"
        assert EntityType.OTHER.value == "other"


class TestEntity:
    """Test individual entity class."""
    
    def test_entity_creation(self):
        """Test entity creation with required fields."""
        entity = Entity(
            text="賈寶玉",
            type=EntityType.PERSON
        )
        assert entity.text == "賈寶玉"
        assert entity.type == EntityType.PERSON
        assert entity.confidence == 1.0
        assert entity.start_pos is None
        assert entity.end_pos is None
    
    def test_entity_creation_with_optional_fields(self):
        """Test entity creation with optional fields."""
        entity = Entity(
            text="大觀園",
            type=EntityType.LOCATION,
            confidence=0.95,
            start_pos=10,
            end_pos=13,
            source_text="紅樓夢中的大觀園",
            metadata={"importance": "high"}
        )
        assert entity.text == "大觀園"
        assert entity.type == EntityType.LOCATION
        assert entity.confidence == 0.95
        assert entity.start_pos == 10
        assert entity.end_pos == 13
        assert entity.source_text == "紅樓夢中的大觀園"
        assert entity.metadata["importance"] == "high"
    
    def test_entity_validation_text_empty(self):
        """Test entity validation with empty text."""
        with pytest.raises(ValueError, match="Entity text cannot be empty"):
            Entity(text="", type=EntityType.PERSON)
    
    def test_entity_validation_text_whitespace(self):
        """Test entity validation with whitespace-only text."""
        with pytest.raises(ValueError, match="Entity text cannot be empty"):
            Entity(text="   ", type=EntityType.PERSON)
    
    def test_entity_validation_positions(self):
        """Test entity validation with invalid positions."""
        with pytest.raises(ValueError, match="End position must be greater than start position"):
            Entity(
                text="test",
                type=EntityType.PERSON,
                start_pos=10,
                end_pos=5
            )
    
    def test_entity_validation_confidence(self):
        """Test entity validation with invalid confidence."""
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            Entity(
                text="test",
                type=EntityType.PERSON,
                confidence=1.5
            )
    
    def test_entity_string_representation(self):
        """Test entity string representation."""
        entity = Entity(text="賈寶玉", type=EntityType.PERSON)
        assert str(entity) == "賈寶玉 (person)"
    
    def test_entity_repr_representation(self):
        """Test entity detailed string representation."""
        entity = Entity(text="賈寶玉", type=EntityType.PERSON, confidence=0.95)
        repr_str = repr(entity)
        assert "Entity" in repr_str
        assert "賈寶玉" in repr_str
        assert "person" in repr_str
        assert "0.95" in repr_str
    
    def test_entity_to_dict(self):
        """Test entity serialization to dictionary."""
        entity = Entity(
            text="賈寶玉",
            type=EntityType.PERSON,
            confidence=0.95,
            start_pos=10,
            end_pos=13
        )
        entity_dict = entity.to_dict()
        
        assert entity_dict["text"] == "賈寶玉"
        assert entity_dict["type"] == "person"
        assert entity_dict["confidence"] == 0.95
        assert entity_dict["start_pos"] == 10
        assert entity_dict["end_pos"] == 13
        assert "created_at" in entity_dict
    
    def test_entity_from_dict(self):
        """Test entity creation from dictionary."""
        entity_dict = {
            "text": "賈寶玉",
            "type": "person",
            "confidence": 0.95,
            "start_pos": 10,
            "end_pos": 13,
            "created_at": "2025-01-27T10:00:00"
        }
        entity = Entity.from_dict(entity_dict)
        
        assert entity.text == "賈寶玉"
        assert entity.type == EntityType.PERSON
        assert entity.confidence == 0.95
        assert entity.start_pos == 10
        assert entity.end_pos == 13
    
    def test_entity_chinese_detection(self):
        """Test Chinese character detection."""
        chinese_entity = Entity(text="賈寶玉", type=EntityType.PERSON)
        english_entity = Entity(text="John", type=EntityType.PERSON)
        
        assert chinese_entity.is_chinese() == True
        assert english_entity.is_chinese() == False
    
    def test_entity_length(self):
        """Test entity text length."""
        entity = Entity(text="賈寶玉", type=EntityType.PERSON)
        assert entity.get_length() == 3
    
    def test_entity_word_count(self):
        """Test entity word count."""
        entity = Entity(text="賈寶玉", type=EntityType.PERSON)
        assert entity.get_word_count() == 3


class TestEntityList:
    """Test entity list collection."""
    
    def test_entity_list_creation(self):
        """Test entity list creation."""
        entity_list = EntityList()
        assert len(entity_list) == 0
        assert entity_list.entities == []
    
    def test_entity_list_creation_with_entities(self):
        """Test entity list creation with entities."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="林黛玉", type=EntityType.PERSON)
        ]
        entity_list = EntityList(entities=entities)
        assert len(entity_list) == 2
        assert entity_list[0].text == "賈寶玉"
        assert entity_list[1].text == "林黛玉"
    
    def test_entity_list_iteration(self):
        """Test entity list iteration."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="林黛玉", type=EntityType.PERSON)
        ]
        entity_list = EntityList(entities=entities)
        
        texts = [entity.text for entity in entity_list]
        assert texts == ["賈寶玉", "林黛玉"]
    
    def test_entity_list_add_entity(self):
        """Test adding entity to list."""
        entity_list = EntityList()
        entity = Entity(text="賈寶玉", type=EntityType.PERSON)
        
        entity_list.add_entity(entity)
        assert len(entity_list) == 1
        assert entity_list[0] == entity
    
    def test_entity_list_add_duplicate(self):
        """Test adding duplicate entity."""
        entity_list = EntityList()
        entity1 = Entity(text="賈寶玉", type=EntityType.PERSON)
        entity2 = Entity(text="賈寶玉", type=EntityType.PERSON)
        
        entity_list.add_entity(entity1)
        entity_list.add_entity(entity2)
        
        # Should not add duplicate
        assert len(entity_list) == 1
    
    def test_entity_list_remove_duplicates(self):
        """Test removing duplicates from list."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="林黛玉", type=EntityType.PERSON)
        ]
        entity_list = EntityList(entities=entities)
        
        entity_list.remove_duplicates()
        assert len(entity_list) == 2
    
    def test_entity_list_filter_by_type(self):
        """Test filtering entities by type."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="大觀園", type=EntityType.LOCATION),
            Entity(text="林黛玉", type=EntityType.PERSON)
        ]
        entity_list = EntityList(entities=entities)
        
        person_entities = entity_list.filter_by_type(EntityType.PERSON)
        assert len(person_entities) == 2
        assert all(entity.type == EntityType.PERSON for entity in person_entities)
    
    def test_entity_list_filter_by_confidence(self):
        """Test filtering entities by confidence."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON, confidence=0.9),
            Entity(text="大觀園", type=EntityType.LOCATION, confidence=0.7),
            Entity(text="林黛玉", type=EntityType.PERSON, confidence=0.8)
        ]
        entity_list = EntityList(entities=entities)
        
        high_confidence = entity_list.filter_by_confidence(0.8)
        assert len(high_confidence) == 2
        assert all(entity.confidence >= 0.8 for entity in high_confidence)
    
    def test_entity_list_get_entity_types(self):
        """Test getting unique entity types."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="大觀園", type=EntityType.LOCATION),
            Entity(text="林黛玉", type=EntityType.PERSON)
        ]
        entity_list = EntityList(entities=entities)
        
        types = entity_list.get_entity_types()
        assert len(types) == 2
        assert EntityType.PERSON in types
        assert EntityType.LOCATION in types
    
    def test_entity_list_get_type_counts(self):
        """Test getting entity type counts."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="大觀園", type=EntityType.LOCATION),
            Entity(text="林黛玉", type=EntityType.PERSON)
        ]
        entity_list = EntityList(entities=entities)
        
        type_counts = entity_list.get_type_counts()
        assert type_counts[EntityType.PERSON] == 2
        assert type_counts[EntityType.LOCATION] == 1
    
    def test_entity_list_average_confidence(self):
        """Test calculating average confidence."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON, confidence=0.8),
            Entity(text="大觀園", type=EntityType.LOCATION, confidence=0.9),
            Entity(text="林黛玉", type=EntityType.PERSON, confidence=0.7)
        ]
        entity_list = EntityList(entities=entities)
        
        avg_confidence = entity_list.get_average_confidence()
        assert avg_confidence == 0.8
    
    def test_entity_list_sort_by_confidence(self):
        """Test sorting entities by confidence."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON, confidence=0.8),
            Entity(text="大觀園", type=EntityType.LOCATION, confidence=0.9),
            Entity(text="林黛玉", type=EntityType.PERSON, confidence=0.7)
        ]
        entity_list = EntityList(entities=entities)
        
        entity_list.sort_by_confidence()
        assert entity_list[0].confidence == 0.9
        assert entity_list[1].confidence == 0.8
        assert entity_list[2].confidence == 0.7
    
    def test_entity_list_to_dict(self):
        """Test entity list serialization to dictionary."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON),
            Entity(text="大觀園", type=EntityType.LOCATION)
        ]
        entity_list = EntityList(entities=entities)
        
        entity_list_dict = entity_list.to_dict()
        assert "entities" in entity_list_dict
        assert "entity_count" in entity_list_dict
        assert "entity_types" in entity_list_dict
        assert "type_counts" in entity_list_dict
        assert "average_confidence" in entity_list_dict
    
    def test_entity_list_from_dict(self):
        """Test entity list creation from dictionary."""
        entity_list_dict = {
            "entities": [
                {"text": "賈寶玉", "type": "person", "confidence": 0.9},
                {"text": "大觀園", "type": "location", "confidence": 0.8}
            ],
            "source_text": "Test source",
            "extraction_timestamp": "2025-01-27T10:00:00"
        }
        entity_list = EntityList.from_dict(entity_list_dict)
        
        assert len(entity_list) == 2
        assert entity_list[0].text == "賈寶玉"
        assert entity_list[1].text == "大觀園"


class TestEntityExtractionResult:
    """Test entity extraction result."""
    
    def test_extraction_result_creation(self):
        """Test extraction result creation."""
        entities = EntityList()
        result = EntityExtractionResult(
            entities=entities,
            processing_time=1.5,
            text_length=100,
            extraction_method="GPT-5-mini"
        )
        assert result.entities == entities
        assert result.processing_time == 1.5
        assert result.text_length == 100
        assert result.extraction_method == "GPT-5-mini"
    
    def test_extraction_result_validation(self):
        """Test extraction result validation."""
        entities = EntityList()
        
        # Test invalid processing time
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            EntityExtractionResult(
                entities=entities,
                processing_time=-1.0,
                text_length=100,
                extraction_method="GPT-5-mini"
            )
    
    def test_extraction_result_statistics(self):
        """Test extraction result statistics."""
        entities = [
            Entity(text="賈寶玉", type=EntityType.PERSON, confidence=0.9),
            Entity(text="大觀園", type=EntityType.LOCATION, confidence=0.8)
        ]
        entity_list = EntityList(entities=entities)
        
        result = EntityExtractionResult(
            entities=entity_list,
            processing_time=1.5,
            text_length=100,
            extraction_method="GPT-5-mini"
        )
        
        stats = result.get_extraction_statistics()
        assert stats["total_entities"] == 2
        assert stats["processing_time"] == 1.5
        assert stats["text_length"] == 100
        assert stats["extraction_method"] == "GPT-5-mini"
    
    def test_extraction_result_to_dict(self):
        """Test extraction result serialization to dictionary."""
        entities = EntityList()
        result = EntityExtractionResult(
            entities=entities,
            processing_time=1.5,
            text_length=100,
            extraction_method="GPT-5-mini"
        )
        
        result_dict = result.to_dict()
        assert "entities" in result_dict
        assert "processing_time" in result_dict
        assert "text_length" in result_dict
        assert "extraction_method" in result_dict
        assert "statistics" in result_dict
    
    def test_extraction_result_from_dict(self):
        """Test extraction result creation from dictionary."""
        result_dict = {
            "entities": {"entities": []},
            "processing_time": 1.5,
            "text_length": 100,
            "extraction_method": "GPT-5-mini"
        }
        result = EntityExtractionResult.from_dict(result_dict)
        
        assert result.processing_time == 1.5
        assert result.text_length == 100
        assert result.extraction_method == "GPT-5-mini"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_entity_list(self):
        """Test empty entity list operations."""
        entity_list = EntityList()
        assert entity_list.get_average_confidence() == 0.0
        assert entity_list.get_type_counts() == {}
        assert entity_list.get_entity_types() == []
    
    def test_single_entity_list(self):
        """Test single entity list operations."""
        entity = Entity(text="賈寶玉", type=EntityType.PERSON, confidence=0.9)
        entity_list = EntityList(entities=[entity])
        
        assert entity_list.get_average_confidence() == 0.9
        assert entity_list.get_type_counts()[EntityType.PERSON] == 1
    
    def test_entity_with_extreme_confidence(self):
        """Test entity with extreme confidence values."""
        entity = Entity(text="test", type=EntityType.PERSON, confidence=0.0)
        assert entity.confidence == 0.0
        
        entity = Entity(text="test", type=EntityType.PERSON, confidence=1.0)
        assert entity.confidence == 1.0
    
    def test_entity_with_special_characters(self):
        """Test entity with special characters."""
        entity = Entity(text="!@#$%^&*()", type=EntityType.OTHER)
        assert entity.text == "!@#$%^&*()"
        assert entity.is_chinese() == False


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_entity_list_add_invalid_entity(self):
        """Test adding invalid entity to list."""
        entity_list = EntityList()
        
        with pytest.raises(ValueError, match="Can only add Entity objects"):
            entity_list.add_entity("not an entity")
    
    def test_entity_list_validation(self):
        """Test entity list validation."""
        with pytest.raises(ValueError, match="Input should be a valid list"):
            EntityList(entities="not a list")
    
    def test_entity_creation_with_invalid_type(self):
        """Test entity creation with invalid type."""
        with pytest.raises(ValueError):
            Entity(text="test", type="invalid_type")
