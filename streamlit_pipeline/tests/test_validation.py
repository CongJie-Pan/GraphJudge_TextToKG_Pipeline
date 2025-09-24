"""
Unit tests for validation utilities in the GraphJudge Streamlit Pipeline.

This module tests the validation functions in utils/validation.py,
following the TDD principles outlined in docs/Testing_Demands.md.

Test coverage includes:
- Input text validation
- Entity list validation  
- Triple validation
- Judgment consistency validation
- API response format validation
- Edge cases and error conditions
"""

import pytest
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.validation import (
    ValidationResult, validate_input_text, validate_entities, 
    validate_triple, validate_triples_list, validate_judgment_consistency,
    validate_api_response_format, _is_highly_repetitive
)
from core.models import Triple


class TestValidationResult:
    """Test ValidationResult data class."""
    
    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.error_message is None
        assert result.warnings == []
        assert result.metadata == {}
    
    def test_validation_result_error(self):
        """Test error validation result."""
        result = ValidationResult(
            is_valid=False,
            error_message="Test error",
            warnings=["Warning 1", "Warning 2"],
            metadata={"context": "test"}
        )
        
        assert result.is_valid is False
        assert result.error_message == "Test error"
        assert result.warnings == ["Warning 1", "Warning 2"]
        assert result.metadata == {"context": "test"}
    
    def test_validation_result_post_init(self):
        """Test ValidationResult initialization of None fields."""
        result = ValidationResult(is_valid=True)
        
        # Should initialize None fields to empty collections
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)


class TestInputTextValidation:
    """Test input text validation."""
    
    def test_validate_input_text_valid(self):
        """Test validation of valid input text."""
        text = "This is a valid input text for processing. It has enough content to be meaningful."
        
        result = validate_input_text(text)
        
        assert result.is_valid is True
        assert result.error_message is None
        assert 'length' in result.metadata
        assert 'lines' in result.metadata
    
    def test_validate_input_text_empty(self):
        """Test validation of empty input text."""
        result = validate_input_text("")
        
        assert result.is_valid is False
        assert "cannot be empty" in result.error_message
    
    def test_validate_input_text_none(self):
        """Test validation of None input."""
        result = validate_input_text(None)
        
        assert result.is_valid is False
        assert "cannot be empty" in result.error_message
    
    def test_validate_input_text_non_string(self):
        """Test validation of non-string input."""
        result = validate_input_text(123)
        
        assert result.is_valid is False
        assert "cannot be empty" in result.error_message
    
    def test_validate_input_text_too_short(self):
        """Test validation of text below minimum length."""
        result = validate_input_text("short", min_length=10)
        
        assert result.is_valid is False
        assert "too short" in result.error_message
        assert "Minimum 10 characters required, got 5" in result.error_message
    
    def test_validate_input_text_too_long(self):
        """Test validation of text above maximum length."""
        long_text = "a" * 1001
        result = validate_input_text(long_text, max_length=1000)
        
        assert result.is_valid is False
        assert "too long" in result.error_message
        assert "Maximum 1000 characters allowed, got 1001" in result.error_message
    
    def test_validate_input_text_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled."""
        text = "   Valid text with whitespace padding   "
        
        result = validate_input_text(text, min_length=20)
        
        # Should be valid after stripping whitespace
        assert result.is_valid is True
    
    def test_validate_input_text_repetitive_warning(self):
        """Test warning for highly repetitive text."""
        repetitive_text = "aaaa" * 100  # Very repetitive - high trigram overlap
        
        result = validate_input_text(repetitive_text)
        
        assert result.is_valid is True
        assert any("repetitive" in warning for warning in result.warnings)
    
    def test_validate_input_text_long_lines_warning(self):
        """Test warning for very long lines."""
        long_line = "x" * 1500
        text = f"Normal line\n{long_line}\nAnother normal line"
        
        result = validate_input_text(text)
        
        assert result.is_valid is True
        assert any("long lines" in warning for warning in result.warnings)
        assert result.metadata['long_lines'] == 1
    
    def test_validate_input_text_metadata_content(self):
        """Test metadata content accuracy."""
        text = "Line 1\nLine 2\nLine 3"
        
        result = validate_input_text(text)
        
        assert result.metadata['length'] == len(text)
        assert result.metadata['lines'] == 3
        assert result.metadata['long_lines'] == 0


class TestEntityValidation:
    """Test entity validation."""
    
    def test_validate_entities_valid(self):
        """Test validation of valid entity list."""
        entities = ["John Doe", "Microsoft", "New York", "Software Engineer"]
        
        result = validate_entities(entities)
        
        assert result.is_valid is True
        assert result.metadata['total_entities'] == 4
        assert result.metadata['unique_entities'] == 4
    
    def test_validate_entities_empty_list(self):
        """Test validation of empty entity list."""
        result = validate_entities([])
        
        assert result.is_valid is False
        assert "No entities found" in result.error_message
    
    def test_validate_entities_not_list(self):
        """Test validation of non-list input."""
        result = validate_entities("not a list")
        
        assert result.is_valid is False
        assert "must be a list" in result.error_message
    
    def test_validate_entities_with_empty_entities(self):
        """Test validation with empty/whitespace entities."""
        entities = ["John", "", "   ", "Microsoft"]  # Removed None to avoid type issues
        
        result = validate_entities(entities)
        
        assert result.is_valid is True
        assert any("empty or whitespace-only" in warning for warning in result.warnings)
    
    def test_validate_entities_with_duplicates(self):
        """Test validation with duplicate entities."""
        entities = ["John", "Microsoft", "John", "Google", "Microsoft"]
        
        result = validate_entities(entities)
        
        assert result.is_valid is True
        assert any("duplicate entities" in warning for warning in result.warnings)
        assert result.metadata['total_entities'] == 5
        assert result.metadata['unique_entities'] == 3
    
    def test_validate_entities_very_long_entities(self):
        """Test validation with unusually long entities."""
        long_entity = "x" * 150
        entities = ["Normal entity", long_entity, "Another normal entity"]
        
        result = validate_entities(entities)
        
        assert result.is_valid is True
        assert any("unusually long entities" in warning for warning in result.warnings)
        assert result.metadata['long_entities'] == 1
    
    def test_validate_entities_malformed_entities(self):
        """Test validation with potentially malformed entities."""
        entities = ["Normal Entity", "X", "123", "!!!@@@###", "Good Entity"]
        
        result = validate_entities(entities)
        
        assert result.is_valid is True
        assert any("malformed entities" in warning for warning in result.warnings)
        # Should detect single character and excessive punctuation


class TestTripleValidation:
    """Test single triple validation."""
    
    def test_validate_triple_valid(self):
        """Test validation of valid triple."""
        triple = Triple("John", "works_at", "Google")

        result = validate_triple(triple)

        assert result.is_valid is True
    
    def test_validate_triple_not_triple_instance(self):
        """Test validation of non-Triple object."""
        fake_triple = {"subject": "John", "predicate": "works_at", "object": "Google"}
        
        result = validate_triple(fake_triple)
        
        assert result.is_valid is False
        assert "not a valid Triple instance" in result.error_message
    
    def test_validate_triple_empty_subject(self):
        """Test validation with empty subject."""
        triple = Triple("", "works_at", "Google")
        
        result = validate_triple(triple)
        
        assert result.is_valid is False
        assert "subject cannot be empty" in result.error_message
    
    def test_validate_triple_empty_predicate(self):
        """Test validation with empty predicate."""
        triple = Triple("John", "", "Google")
        
        result = validate_triple(triple)
        
        assert result.is_valid is False
        assert "predicate cannot be empty" in result.error_message
    
    def test_validate_triple_empty_object(self):
        """Test validation with empty object."""
        triple = Triple("John", "works_at", "")
        
        result = validate_triple(triple)
        
        assert result.is_valid is False
        assert "object cannot be empty" in result.error_message
    
    def test_validate_triple_whitespace_only_components(self):
        """Test validation with whitespace-only components."""
        triple = Triple("   ", "works_at", "Google")
        
        result = validate_triple(triple)
        
        assert result.is_valid is False
        assert "subject cannot be empty" in result.error_message
    
    def test_validate_triple_long_components_warnings(self):
        """Test warnings for unusually long components."""
        long_subject = "x" * 250
        long_predicate = "y" * 150
        long_object = "z" * 250
        
        triple = Triple(long_subject, long_predicate, long_object)
        
        result = validate_triple(triple)
        
        assert result.is_valid is True
        assert any("Subject is unusually long" in warning for warning in result.warnings)
        assert any("Predicate is unusually long" in warning for warning in result.warnings)
        assert any("Object is unusually long" in warning for warning in result.warnings)
    
    def test_validate_triple_basic_validation(self):
        """Test basic triple validation."""
        # Valid triple
        triple1 = Triple("A", "B", "C")
        result1 = validate_triple(triple1)
        assert result1.is_valid is True
    


class TestTriplesListValidation:
    """Test validation of triple lists."""
    
    def test_validate_triples_list_valid(self):
        """Test validation of valid triple list."""
        triples = [
            Triple("John", "works_at", "Google"),
            Triple("Mary", "lives_in", "NYC"),
            Triple("Bob", "knows", "Alice")
        ]
        
        result = validate_triples_list(triples)
        
        assert result.is_valid is True
        assert result.metadata['total_triples'] == 3
        assert result.metadata['unique_triples'] == 3
        assert result.metadata['duplicate_triples'] == 0
    
    def test_validate_triples_list_empty(self):
        """Test validation of empty triple list."""
        result = validate_triples_list([])
        
        assert result.is_valid is False
        assert "No triples found" in result.error_message
    
    def test_validate_triples_list_not_list(self):
        """Test validation of non-list input."""
        result = validate_triples_list("not a list")
        
        assert result.is_valid is False
        assert "must be a list" in result.error_message
    
    def test_validate_triples_list_with_invalid_triples(self):
        """Test validation with some invalid triples."""
        triples = [
            Triple("John", "works_at", "Google"),  # Valid
            Triple("", "works_at", "Microsoft"),   # Invalid - empty subject
            Triple("Mary", "lives_in", "NYC")      # Valid
        ]
        
        result = validate_triples_list(triples)
        
        assert result.is_valid is False
        assert "invalid triples" in result.error_message
        assert "Triple 1:" in result.error_message
    
    def test_validate_triples_list_with_duplicates(self):
        """Test validation with duplicate triples."""
        triples = [
            Triple("John", "works_at", "Google"),
            Triple("Mary", "lives_in", "NYC"),
            Triple("John", "works_at", "Google")  # Duplicate
        ]
        
        result = validate_triples_list(triples)
        
        assert result.is_valid is True
        assert any("duplicate triples" in warning for warning in result.warnings)
        assert result.metadata['duplicate_triples'] == 1


class TestJudgmentConsistencyValidation:
    """Test judgment consistency validation."""
    
    def test_validate_judgment_consistency_valid(self):
        """Test validation of consistent judgment data."""
        triples = [
            Triple("John", "works_at", "Google"),
            Triple("Mary", "lives_in", "NYC")
        ]
        judgments = [True, False]

        result = validate_judgment_consistency(triples, judgments, None)

        assert result.is_valid is True
        assert result.metadata['total_judgments'] == 2
        assert result.metadata['true_judgments'] == 1
        assert result.metadata['false_judgments'] == 1
    
    def test_validate_judgment_consistency_length_mismatch_judgments(self):
        """Test validation with mismatched triples and judgments length."""
        triples = [Triple("A", "B", "C"), Triple("X", "Y", "Z")]
        judgments = [True]  # Wrong length

        result = validate_judgment_consistency(triples, judgments, None)

        assert result.is_valid is False
        assert "Mismatch between triples (2) and judgments (1)" in result.error_message
    
    
    def test_validate_judgment_consistency_non_lists(self):
        """Test validation with non-list inputs."""
        result = validate_judgment_consistency("not list", [True], None)

        assert result.is_valid is False
        assert "All inputs must be lists" in result.error_message
    
    
    
    
    def test_validate_judgment_consistency_all_true_warning(self):
        """Test warning when all judgments are true."""
        triples = [Triple("A", "B", "C"), Triple("X", "Y", "Z")]
        judgments = [True, True]

        result = validate_judgment_consistency(triples, judgments, None)

        assert result.is_valid is True
        assert any("All triples were judged as true" in warning for warning in result.warnings)
    
    def test_validate_judgment_consistency_all_false_warning(self):
        """Test warning when all judgments are false."""
        triples = [Triple("A", "B", "C"), Triple("X", "Y", "Z")]
        judgments = [False, False]

        result = validate_judgment_consistency(triples, judgments, None)

        assert result.is_valid is True
        assert any("All triples were judged as false" in warning for warning in result.warnings)
    
    def test_validate_judgment_consistency_metadata_calculations(self):
        """Test metadata calculations accuracy."""
        triples = [Triple("A", "B", "C") for _ in range(5)]
        judgments = [True, True, False, True, False]

        result = validate_judgment_consistency(triples, judgments, None)

        assert result.is_valid is True
        assert result.metadata['total_judgments'] == 5
        assert result.metadata['true_judgments'] == 3
        assert result.metadata['false_judgments'] == 2


class TestApiResponseValidation:
    """Test API response format validation."""
    
    def test_validate_api_response_format_valid(self):
        """Test validation of valid API response."""
        response = {
            "entities": ["John", "Google"],
            "status": "success",
            "processing_time": 1.5
        }
        expected_fields = ["entities", "status", "processing_time"]
        
        result = validate_api_response_format(response, expected_fields)
        
        assert result.is_valid is True
        assert result.metadata['response_fields'] == list(response.keys())
        assert result.metadata['expected_fields'] == expected_fields
    
    def test_validate_api_response_format_not_dict(self):
        """Test validation of non-dictionary response."""
        result = validate_api_response_format("not a dict", ["field1"])
        
        assert result.is_valid is False
        assert "not a dictionary" in result.error_message
    
    def test_validate_api_response_format_missing_fields(self):
        """Test validation with missing required fields."""
        response = {"field1": "value1"}
        expected_fields = ["field1", "field2", "field3"]
        
        result = validate_api_response_format(response, expected_fields)
        
        assert result.is_valid is False
        assert "missing required fields" in result.error_message
        assert "field2" in result.error_message
        assert "field3" in result.error_message
    
    def test_validate_api_response_format_extra_fields_warning(self):
        """Test warning for unexpected extra fields."""
        response = {
            "required_field": "value",
            "extra_field1": "extra1",
            "extra_field2": "extra2"
        }
        expected_fields = ["required_field"]
        
        result = validate_api_response_format(response, expected_fields)
        
        assert result.is_valid is True
        assert any("unexpected fields" in warning for warning in result.warnings)
        assert set(result.metadata['extra_fields']) == {"extra_field1", "extra_field2"}


class TestRepetitiveTextDetection:
    """Test repetitive text detection helper function."""
    
    def test_is_highly_repetitive_normal_text(self):
        """Test normal text is not detected as repetitive."""
        normal_text = """
        This is a normal text with varied content and different words.
        It contains multiple sentences with diverse vocabulary and structure.
        There should be no excessive repetition detected in this text.
        """
        
        result = _is_highly_repetitive(normal_text)
        
        assert result is False
    
    def test_is_highly_repetitive_repetitive_text(self):
        """Test highly repetitive text is detected."""
        repetitive_text = "aaa" * 200  # Very repetitive - single trigram dominates
        
        result = _is_highly_repetitive(repetitive_text)
        
        assert result is True
    
    def test_is_highly_repetitive_short_text(self):
        """Test short text is not analyzed for repetition."""
        short_text = "Short"
        
        result = _is_highly_repetitive(short_text)
        
        assert result is False  # Too short to analyze
    
    def test_is_highly_repetitive_custom_threshold(self):
        """Test repetitive text detection with custom threshold."""
        moderately_repetitive = "xyz" * 50  # Moderately repetitive
        
        # Should not be detected with high threshold
        assert _is_highly_repetitive(moderately_repetitive, threshold=0.9) is False
        
        # Should be detected with low threshold  
        assert _is_highly_repetitive(moderately_repetitive, threshold=0.3) is True
    
    def test_is_highly_repetitive_long_text_sampling(self):
        """Test that very long texts are sampled for performance."""
        # Create text longer than 2000 characters
        long_text = "varied content with different words " * 100  # > 2000 chars
        
        # Should handle long text without performance issues
        result = _is_highly_repetitive(long_text)
        
        # The function should complete quickly and return a result
        assert isinstance(result, bool)


# Edge cases and integration tests
class TestValidationIntegration:
    """Integration tests for validation functions."""
    
    def test_full_validation_pipeline_success(self):
        """Test complete validation pipeline with valid data."""
        # Input text
        text = "John works at Google. Mary lives in New York. Bob knows Alice well."
        text_result = validate_input_text(text)
        assert text_result.is_valid
        
        # Entities
        entities = ["John", "Google", "Mary", "New York", "Bob", "Alice"]
        entity_result = validate_entities(entities)
        assert entity_result.is_valid
        
        # Triples
        triples = [
            Triple("John", "works_at", "Google"),
            Triple("Mary", "lives_in", "New York"),
            Triple("Bob", "knows", "Alice")
        ]
        triple_result = validate_triples_list(triples)
        assert triple_result.is_valid

        # Judgments
        judgments = [True, True, False]
        judgment_result = validate_judgment_consistency(triples, judgments, None)
        assert judgment_result.is_valid
        
        # All validations should pass
        assert all([
            text_result.is_valid,
            entity_result.is_valid,
            triple_result.is_valid,
            judgment_result.is_valid
        ])
    
    def test_validation_error_propagation(self):
        """Test how validation errors propagate through pipeline."""
        # Start with invalid input text
        invalid_text = "x"  # Too short
        text_result = validate_input_text(invalid_text, min_length=10)
        
        assert text_result.is_valid is False
        assert "too short" in text_result.error_message
        
        # Invalid entities
        invalid_entities = []  # Empty
        entity_result = validate_entities(invalid_entities)
        
        assert entity_result.is_valid is False
        assert "No entities found" in entity_result.error_message
        
        # Invalid triples
        invalid_triples = [Triple("", "predicate", "object")]  # Empty subject
        triple_result = validate_triples_list(invalid_triples)
        
        assert triple_result.is_valid is False
        assert "subject cannot be empty" in triple_result.error_message
    
    def test_validation_warning_accumulation(self):
        """Test accumulation of warnings across validations."""
        # Text with warnings
        repetitive_text = "repeat " * 50 + "x" * 1200  # Repetitive + long line
        text_result = validate_input_text(repetitive_text)
        
        assert text_result.is_valid is True
        assert len(text_result.warnings) >= 2  # Should have multiple warnings
        
        # Entities with warnings
        problematic_entities = ["John", "John", "x", "y" * 150]  # Duplicates + short + long
        entity_result = validate_entities(problematic_entities)
        
        assert entity_result.is_valid is True
        assert len(entity_result.warnings) >= 2  # Multiple warnings