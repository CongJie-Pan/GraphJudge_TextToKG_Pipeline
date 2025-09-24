"""
Unit tests for data models in the GraphJudge Streamlit Pipeline.

This module tests all data structures defined in core/models.py,
following the TDD principles outlined in docs/Testing_Demands.md.

Test coverage includes:
- Data model creation and validation
- Enum conversions and edge cases
- Result object consistency
- Pipeline state management
- Error handling and edge cases
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.models import (
    Triple, EntityResult, TripleResult, JudgmentResult, PipelineState,
    PipelineStage, ProcessingStatus,
    create_error_result, ProcessingTimer
)


class TestTriple:
    """Test cases for Triple data model."""
    
    def test_triple_creation_basic(self):
        """Test basic triple creation with required fields."""
        triple = Triple(
            subject="John",
            predicate="works_at",
            object="Google"
        )
        
        assert triple.subject == "John"
        assert triple.predicate == "works_at"
        assert triple.object == "Google"
        assert triple.source_text is None
        assert triple.metadata == {}
    
    def test_triple_to_dict(self):
        """Test triple serialization to dictionary."""
        triple = Triple(
            subject="John",
            predicate="knows",
            object="Mary",
            source_text="John knows Mary well",
            metadata={"context": "friendship"}
        )

        result = triple.to_dict()

        expected = {
            'subject': "John",
            'predicate': "knows",
            'object': "Mary",
            'source_text': "John knows Mary well",
            'metadata': {"context": "friendship"}
        }
        
        assert result == expected
    
    def test_triple_from_dict(self):
        """Test triple creation from dictionary."""
        data = {
            'subject': "Alice",
            'predicate': "works_at",
            'object': "Microsoft",
            'source_text': "Alice works at Microsoft",
            'metadata': {"department": "engineering"}
        }

        triple = Triple.from_dict(data)

        assert triple.subject == "Alice"
        assert triple.predicate == "works_at"
        assert triple.object == "Microsoft"
        assert triple.source_text == "Alice works at Microsoft"
        assert triple.metadata == {"department": "engineering"}
    
    def test_triple_from_dict_minimal(self):
        """Test triple creation from minimal dictionary."""
        data = {
            'subject': "Bob",
            'predicate': "lives_in",
            'object': "London"
        }
        
        triple = Triple.from_dict(data)
        
        assert triple.subject == "Bob"
        assert triple.predicate == "lives_in"
        assert triple.object == "London"
        assert triple.source_text is None
        assert triple.metadata == {}
    
    def test_triple_str_representation(self):
        """Test string representation of triples."""
        triple = Triple("A", "B", "C")
        assert str(triple) == "(A, B, C)"


class TestEntityResult:
    """Test cases for EntityResult data model."""
    
    def test_entity_result_creation_success(self):
        """Test successful EntityResult creation."""
        result = EntityResult(
            entities=["John", "Mary", "Google"],
            denoised_text="John and Mary work at Google.",
            success=True,
            processing_time=1.5
        )
        
        assert result.entities == ["John", "Mary", "Google"]
        assert result.denoised_text == "John and Mary work at Google."
        assert result.success is True
        assert result.processing_time == 1.5
        assert result.error is None
    
    def test_entity_result_creation_error(self):
        """Test EntityResult creation with error."""
        result = EntityResult(
            entities=[],
            denoised_text="",
            success=False,
            processing_time=0.5,
            error="API connection failed"
        )
        
        assert result.entities == []
        assert result.denoised_text == ""
        assert result.success is False
        assert result.processing_time == 0.5
        assert result.error == "API connection failed"


class TestTripleResult:
    """Test cases for TripleResult data model."""
    
    def test_triple_result_creation_success(self):
        """Test successful TripleResult creation."""
        triples = [
            Triple("John", "works_at", "Google"),
            Triple("Mary", "lives_in", "NYC")
        ]
        metadata = {
            "total_chunks": 1,
            "validation_passed": True,
            "extraction_method": "GPT-5-mini"
        }
        
        result = TripleResult(
            triples=triples,
            metadata=metadata,
            success=True,
            processing_time=2.3
        )
        
        assert len(result.triples) == 2
        assert result.triples[0].subject == "John"
        assert result.triples[1].subject == "Mary"
        assert result.metadata == metadata
        assert result.success is True
        assert result.processing_time == 2.3
        assert result.error is None
    
    def test_triple_result_creation_empty(self):
        """Test TripleResult with no triples."""
        result = TripleResult(
            triples=[],
            metadata={"reason": "no_valid_triples"},
            success=False,
            processing_time=1.0,
            error="No valid triples could be extracted"
        )
        
        assert result.triples == []
        assert result.success is False
        assert result.error == "No valid triples could be extracted"


class TestJudgmentResult:
    """Test cases for JudgmentResult data model."""
    
    def test_judgment_result_creation_basic(self):
        """Test basic JudgmentResult creation."""
        result = JudgmentResult(
            judgments=[True, False, True]
        )

        assert result.judgments == [True, False, True]
        assert result.explanations is None
        assert result.success is True
        assert result.processing_time == 0.0
        assert result.error is None
    
    def test_judgment_result_with_explanations(self):
        """Test JudgmentResult with explanations."""
        explanations = [
            "Strong evidence in text",
            "Contradicted by context", 
            "Explicitly stated"
        ]
        
        result = JudgmentResult(
            judgments=[True, False, True],
            explanations=explanations,
            success=True,
            processing_time=3.5
        )
        
        assert result.explanations == explanations
        assert result.processing_time == 3.5
    
    def test_judgment_result_to_dict(self):
        """Test JudgmentResult serialization."""
        result = JudgmentResult(
            judgments=[True, False],
            explanations=["Good", "Bad"],
            success=True,
            processing_time=2.0
        )

        data = result.to_dict()

        expected = {
            'judgments': [True, False],
            'explanations': ["Good", "Bad"],
            'success': True,
            'processing_time': 2.0,
            'error': None
        }
        
        assert data == expected


class TestPipelineState:
    """Test cases for PipelineState data model."""
    
    def test_pipeline_state_initial(self):
        """Test initial pipeline state."""
        state = PipelineState()
        
        assert state.input_text == ""
        assert state.status == ProcessingStatus.DRAFT
        assert state.current_stage is None
        assert state.entity_result is None
        assert state.triple_result is None
        assert state.judgment_result is None
        assert state.total_processing_time == 0.0
        assert not state.is_complete
        assert not state.has_error
        assert state.progress_percentage == 0.0
    
    def test_pipeline_state_progress_calculation(self):
        """Test progress percentage calculation."""
        state = PipelineState()
        
        # Draft state
        assert state.progress_percentage == 0.0
        
        # Running entity
        state.status = ProcessingStatus.RUNNING_ENTITY
        assert state.progress_percentage == 10.0
        
        # Entity completed, running triple
        state.entity_result = EntityResult(["A"], "text", True, 1.0)
        state.status = ProcessingStatus.RUNNING_TRIPLE
        assert state.progress_percentage == 40.0
        
        # Triple completed, running judgment
        state.triple_result = TripleResult([Triple("A", "B", "C")], {}, True, 1.0)
        state.status = ProcessingStatus.RUNNING_GJ
        assert state.progress_percentage == 70.0
        
        # All completed
        state.judgment_result = JudgmentResult([True])
        state.status = ProcessingStatus.SUCCEEDED
        assert state.progress_percentage == 100.0
        assert state.is_complete
    
    def test_pipeline_state_error_handling(self):
        """Test pipeline state error handling."""
        state = PipelineState()
        state.status = ProcessingStatus.FAILED
        state.error_stage = PipelineStage.TRIPLE_GENERATION
        state.error_message = "Triple generation failed"
        
        assert state.has_error
        assert state.progress_percentage == 40.0  # Failed at triple stage
    
    def test_pipeline_state_completed_stages(self):
        """Test getting completed stages."""
        state = PipelineState()
        
        # No completed stages initially
        assert state.get_completed_stages() == []
        
        # Entity stage completed
        state.entity_result = EntityResult(["A"], "text", True, 1.0)
        assert state.get_completed_stages() == [PipelineStage.ENTITY_EXTRACTION]
        
        # Triple stage completed
        state.triple_result = TripleResult([Triple("A", "B", "C")], {}, True, 1.0)
        completed = state.get_completed_stages()
        assert PipelineStage.ENTITY_EXTRACTION in completed
        assert PipelineStage.TRIPLE_GENERATION in completed
        
        # All stages completed
        state.judgment_result = JudgmentResult([True])
        completed = state.get_completed_stages()
        assert len(completed) == 3
        assert PipelineStage.GRAPH_JUDGMENT in completed
    
    def test_pipeline_state_reset(self):
        """Test pipeline state reset functionality."""
        state = PipelineState(input_text="test")
        state.status = ProcessingStatus.SUCCEEDED
        state.entity_result = EntityResult(["A"], "text", True, 1.0)
        state.total_processing_time = 5.0
        state.started_at = "2023-01-01"
        
        state.reset()
        
        assert state.status == ProcessingStatus.DRAFT
        assert state.current_stage is None
        assert state.entity_result is None
        assert state.triple_result is None
        assert state.judgment_result is None
        assert state.total_processing_time == 0.0
        assert state.started_at is None


class TestEnums:
    """Test cases for enum classes."""
    
    def test_pipeline_stage_values(self):
        """Test pipeline stage enum values."""
        assert PipelineStage.ENTITY_EXTRACTION.value == "entity_extraction"
        assert PipelineStage.TRIPLE_GENERATION.value == "triple_generation"
        assert PipelineStage.GRAPH_JUDGMENT.value == "graph_judgment"
    
    def test_processing_status_values(self):
        """Test processing status enum values."""
        assert ProcessingStatus.DRAFT.value == "draft"
        assert ProcessingStatus.QUEUED.value == "queued"
        assert ProcessingStatus.SUCCEEDED.value == "succeeded"
        assert ProcessingStatus.FAILED.value == "failed"


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_error_result_entity(self):
        """Test creating error EntityResult."""
        result = create_error_result(EntityResult, "Test error", 1.5)
        
        assert isinstance(result, EntityResult)
        assert result.entities == []
        assert result.denoised_text == ""
        assert result.success is False
        assert result.processing_time == 1.5
        assert result.error == "Test error"
    
    def test_create_error_result_triple(self):
        """Test creating error TripleResult."""
        result = create_error_result(TripleResult, "Triple error", 2.0)
        
        assert isinstance(result, TripleResult)
        assert result.triples == []
        assert result.metadata == {}
        assert result.success is False
        assert result.processing_time == 2.0
        assert result.error == "Triple error"
    
    def test_create_error_result_judgment(self):
        """Test creating error JudgmentResult."""
        result = create_error_result(JudgmentResult, "Judgment error")

        assert isinstance(result, JudgmentResult)
        assert result.judgments == []
        assert result.explanations is None
        assert result.success is False
        assert result.processing_time == 0.0
        assert result.error == "Judgment error"
    
    def test_create_error_result_invalid_type(self):
        """Test create_error_result with invalid type."""
        with pytest.raises(ValueError, match="Unsupported result type"):
            create_error_result(dict, "error")


class TestProcessingTimer:
    """Test cases for ProcessingTimer context manager."""
    
    def test_processing_timer_basic(self):
        """Test basic timer functionality."""
        with ProcessingTimer() as timer:
            time.sleep(0.01)  # Small delay
        
        assert timer.elapsed > 0
        assert timer.elapsed < 1.0  # Should be much less than 1 second
    
    def test_processing_timer_with_result_creation(self):
        """Test timer usage with result object creation."""
        entities = ["John", "Mary"]
        
        with ProcessingTimer() as timer:
            # Simulate some processing
            time.sleep(0.01)
            denoised_text = "Processed text"
        
        result = EntityResult(
            entities=entities,
            denoised_text=denoised_text,
            success=True,
            processing_time=timer.elapsed
        )
        
        assert result.processing_time > 0
        assert result.entities == entities
        assert result.success is True
    
    def test_processing_timer_no_context(self):
        """Test timer behavior when not used as context manager."""
        timer = ProcessingTimer()
        assert timer.elapsed == 0.0
        
        # Manually start timer
        timer.start_time = time.time()
        time.sleep(0.01)
        
        assert timer.elapsed > 0


# Integration tests
class TestDataModelIntegration:
    """Integration tests for data model interactions."""
    
    def test_full_pipeline_data_flow(self):
        """Test complete data flow through all models."""
        # Start with entity extraction
        entity_result = EntityResult(
            entities=["Alice", "Bob", "Company X"],
            denoised_text="Alice and Bob work at Company X",
            success=True,
            processing_time=1.2
        )
        
        # Generate triples
        triples = [
            Triple("Alice", "works_at", "Company X"),
            Triple("Bob", "works_at", "Company X")
        ]
        triple_result = TripleResult(
            triples=triples,
            metadata={"extraction_method": "gpt-5-mini", "chunks": 1},
            success=True,
            processing_time=2.5
        )
        
        # Judge triples
        judgment_result = JudgmentResult(
            judgments=[True, True],
            explanations=["Clear employment relationship", "Explicit in text"],
            success=True,
            processing_time=1.8
        )
        
        # Create pipeline state
        state = PipelineState(input_text="Alice and Bob work at Company X")
        state.entity_result = entity_result
        state.triple_result = triple_result
        state.judgment_result = judgment_result
        state.status = ProcessingStatus.SUCCEEDED
        state.total_processing_time = 5.5
        
        # Verify complete pipeline state
        assert state.is_complete
        assert not state.has_error
        assert state.progress_percentage == 100.0
        assert len(state.get_completed_stages()) == 3
        
        # Verify data consistency
        assert len(triple_result.triples) == len(judgment_result.judgments)
        assert all(isinstance(t, Triple) for t in triple_result.triples)
    
    def test_error_propagation_through_pipeline(self):
        """Test error handling across pipeline stages."""
        # Successful entity extraction
        entity_result = EntityResult(
            entities=["John"],
            denoised_text="John works somewhere",
            success=True,
            processing_time=1.0
        )
        
        # Failed triple generation
        triple_result = create_error_result(
            TripleResult, 
            "Failed to parse JSON response from API",
            0.5
        )
        
        # Pipeline state with error
        state = PipelineState()
        state.entity_result = entity_result
        state.triple_result = triple_result
        state.status = ProcessingStatus.FAILED
        state.error_stage = PipelineStage.TRIPLE_GENERATION
        state.error_message = triple_result.error
        
        assert state.has_error
        assert not state.is_complete
        assert state.progress_percentage == 40.0  # Failed at triple stage
        assert len(state.get_completed_stages()) == 1  # Only entity stage completed