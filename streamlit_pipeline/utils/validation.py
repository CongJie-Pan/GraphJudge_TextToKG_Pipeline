"""
Input validation utilities for GraphJudge Streamlit Pipeline.

This module provides validation functions for user inputs, API responses,
and data model consistency checks. All validation functions follow the
pattern of returning errors as data (not raising exceptions) to maintain
compatibility with the Streamlit error handling strategy.

Key principles:
- Return validation results as data, not exceptions
- Provide clear, user-friendly error messages
- Support both strict and lenient validation modes
- Maintain consistency with data models in models.py
"""

import re
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.models import Triple, EntityResult, TripleResult, JudgmentResult


@dataclass
class ValidationResult:
    """
    Result object for validation operations.
    
    Attributes:
        is_valid: Whether the validation passed
        error_message: Human-readable error description if validation failed
        warnings: List of non-fatal validation warnings
        metadata: Additional validation information
    """
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


def validate_input_text(text: str, min_length: int = 10, max_length: int = 50000) -> ValidationResult:
    """
    Validate user input text for pipeline processing.
    
    Args:
        text: Input text to validate
        min_length: Minimum allowed text length in characters
        max_length: Maximum allowed text length in characters
        
    Returns:
        ValidationResult indicating whether text is suitable for processing
    """
    if not text or not isinstance(text, str):
        return ValidationResult(
            is_valid=False,
            error_message="Input text cannot be empty"
        )
    
    text = text.strip()
    
    if len(text) < min_length:
        return ValidationResult(
            is_valid=False,
            error_message=f"Input text too short. Minimum {min_length} characters required, got {len(text)}"
        )
    
    if len(text) > max_length:
        return ValidationResult(
            is_valid=False,
            error_message=f"Input text too long. Maximum {max_length} characters allowed, got {len(text)}"
        )
    
    warnings = []
    
    # Check for potential encoding issues
    try:
        text.encode('utf-8')
    except UnicodeError:
        warnings.append("Text contains characters that may cause encoding issues")
    
    # Check for overly repetitive content
    if _is_highly_repetitive(text):
        warnings.append("Text appears highly repetitive, which may affect extraction quality")
    
    # Check for very long lines (may indicate formatting issues)
    lines = text.split('\n')
    long_lines = [i for i, line in enumerate(lines) if len(line) > 1000]
    if long_lines:
        warnings.append(f"Found {len(long_lines)} very long lines that may need formatting")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        metadata={
            'length': len(text),
            'lines': len(lines),
            'long_lines': len(long_lines)
        }
    )


def validate_entities(entities: List[str]) -> ValidationResult:
    """
    Validate extracted entities list.
    
    Args:
        entities: List of entity names to validate
        
    Returns:
        ValidationResult indicating entity list quality
    """
    if not isinstance(entities, list):
        return ValidationResult(
            is_valid=False,
            error_message="Entities must be a list"
        )
    
    if len(entities) == 0:
        return ValidationResult(
            is_valid=False,
            error_message="No entities found in the input text"
        )
    
    warnings = []
    
    # Check for empty or whitespace-only entities
    empty_entities = [i for i, entity in enumerate(entities) if not entity or not entity.strip()]
    if empty_entities:
        warnings.append(f"Found {len(empty_entities)} empty or whitespace-only entities")
    
    # Check for very long entities (may indicate extraction errors)
    long_entities = [entity for entity in entities if entity and isinstance(entity, str) and len(entity) > 100]
    if long_entities:
        warnings.append(f"Found {len(long_entities)} unusually long entities that may be extraction errors")
    
    # Check for potential duplicates
    unique_entities = set(entities)
    if len(unique_entities) < len(entities):
        duplicate_count = len(entities) - len(unique_entities)
        warnings.append(f"Found {duplicate_count} duplicate entities")
    
    # Check entity format (basic checks)
    malformed_entities = []
    for entity in entities:
        if isinstance(entity, str) and entity.strip():
            # Check for entities that are just numbers or single characters
            if len(entity.strip()) < 2:
                malformed_entities.append(entity)
            # Check for entities with excessive punctuation
            elif re.search(r'[^\w\s\u4e00-\u9fff]{3,}', entity):
                malformed_entities.append(entity)
    
    if malformed_entities:
        warnings.append(f"Found {len(malformed_entities)} potentially malformed entities")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        metadata={
            'total_entities': len(entities),
            'unique_entities': len(unique_entities),
            'empty_entities': len(empty_entities),
            'long_entities': len(long_entities),
            'malformed_entities': len(malformed_entities)
        }
    )


def validate_triple(triple: Triple) -> ValidationResult:
    """
    Validate a single triple object.
    
    Args:
        triple: Triple object to validate
        
    Returns:
        ValidationResult for the triple
    """
    if not isinstance(triple, Triple):
        return ValidationResult(
            is_valid=False,
            error_message="Object is not a valid Triple instance"
        )
    
    warnings = []
    
    # Check for empty components
    if not triple.subject or not triple.subject.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Triple subject cannot be empty"
        )
    
    if not triple.predicate or not triple.predicate.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Triple predicate cannot be empty"
        )
    
    if not triple.object or not triple.object.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Triple object cannot be empty"
        )
    
    # Check for reasonable component lengths
    if len(triple.subject) > 200:
        warnings.append("Subject is unusually long")
    
    if len(triple.predicate) > 100:
        warnings.append("Predicate is unusually long")
    
    if len(triple.object) > 200:
        warnings.append("Object is unusually long")
    
    # Check confidence score if present
    if triple.confidence is not None:
        if not (0.0 <= triple.confidence <= 1.0):
            return ValidationResult(
                is_valid=False,
                error_message=f"Confidence score must be between 0.0 and 1.0, got {triple.confidence}"
            )
        if triple.confidence < 0.3:
            warnings.append("Triple has low confidence score")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        metadata={
            'subject_length': len(triple.subject),
            'predicate_length': len(triple.predicate),
            'object_length': len(triple.object),
            'has_confidence': triple.confidence is not None,
            'confidence_score': triple.confidence
        }
    )


def validate_triples_list(triples: List[Triple]) -> ValidationResult:
    """
    Validate a list of triples.
    
    Args:
        triples: List of Triple objects to validate
        
    Returns:
        ValidationResult for the entire list
    """
    if not isinstance(triples, list):
        return ValidationResult(
            is_valid=False,
            error_message="Triples must be a list"
        )
    
    if len(triples) == 0:
        return ValidationResult(
            is_valid=False,
            error_message="No triples found"
        )
    
    warnings = []
    invalid_triples = []
    
    # Validate each triple
    for i, triple in enumerate(triples):
        result = validate_triple(triple)
        if not result.is_valid:
            invalid_triples.append(f"Triple {i}: {result.error_message}")
        elif result.warnings:
            warnings.extend([f"Triple {i}: {warning}" for warning in result.warnings])
    
    if invalid_triples:
        return ValidationResult(
            is_valid=False,
            error_message=f"Found {len(invalid_triples)} invalid triples: " + "; ".join(invalid_triples[:3])
        )
    
    # Check for duplicate triples
    triple_strings = [str(triple) for triple in triples]
    unique_triples = set(triple_strings)
    duplicate_count = len(triple_strings) - len(unique_triples)
    
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate triples")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        metadata={
            'total_triples': len(triples),
            'unique_triples': len(unique_triples),
            'duplicate_triples': duplicate_count
        }
    )


def validate_judgment_consistency(triples: List[Triple], judgments: List[bool], 
                                confidence: List[float]) -> ValidationResult:
    """
    Validate that judgment results are consistent with input triples.
    
    Args:
        triples: Original triples that were judged
        judgments: List of boolean judgments
        confidence: List of confidence scores
        
    Returns:
        ValidationResult for consistency
    """
    if not isinstance(triples, list) or not isinstance(judgments, list) or not isinstance(confidence, list):
        return ValidationResult(
            is_valid=False,
            error_message="All inputs must be lists"
        )
    
    if len(triples) != len(judgments):
        return ValidationResult(
            is_valid=False,
            error_message=f"Mismatch between triples ({len(triples)}) and judgments ({len(judgments)})"
        )
    
    if len(triples) != len(confidence):
        return ValidationResult(
            is_valid=False,
            error_message=f"Mismatch between triples ({len(triples)}) and confidence scores ({len(confidence)})"
        )
    
    warnings = []
    
    # Validate confidence scores
    for i, conf in enumerate(confidence):
        if not isinstance(conf, (int, float)):
            return ValidationResult(
                is_valid=False,
                error_message=f"Confidence score {i} is not numeric: {conf}"
            )
        if not (0.0 <= conf <= 1.0):
            return ValidationResult(
                is_valid=False,
                error_message=f"Confidence score {i} out of range [0,1]: {conf}"
            )
        if conf < 0.5:
            warnings.append(f"Low confidence score ({conf:.2f}) for judgment {i}")
    
    # Check judgment distribution
    true_count = sum(judgments)
    false_count = len(judgments) - true_count
    
    if true_count == 0:
        warnings.append("All triples were judged as false - this may indicate an issue")
    elif false_count == 0:
        warnings.append("All triples were judged as true - this may indicate low selectivity")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        metadata={
            'total_judgments': len(judgments),
            'true_judgments': true_count,
            'false_judgments': false_count,
            'average_confidence': sum(confidence) / len(confidence) if confidence else 0.0,
            'min_confidence': min(confidence) if confidence else 0.0,
            'max_confidence': max(confidence) if confidence else 0.0
        }
    )


def validate_api_response_format(response: Dict[str, Any], expected_fields: List[str]) -> ValidationResult:
    """
    Validate that an API response contains expected fields.
    
    Args:
        response: API response dictionary
        expected_fields: List of required field names
        
    Returns:
        ValidationResult for the API response format
    """
    if not isinstance(response, dict):
        return ValidationResult(
            is_valid=False,
            error_message="API response is not a dictionary"
        )
    
    missing_fields = []
    for field in expected_fields:
        if field not in response:
            missing_fields.append(field)
    
    if missing_fields:
        return ValidationResult(
            is_valid=False,
            error_message=f"API response missing required fields: {', '.join(missing_fields)}"
        )
    
    warnings = []
    extra_fields = set(response.keys()) - set(expected_fields)
    if extra_fields:
        warnings.append(f"API response contains unexpected fields: {', '.join(extra_fields)}")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        metadata={
            'response_fields': list(response.keys()),
            'expected_fields': expected_fields,
            'extra_fields': list(extra_fields)
        }
    )


def _is_highly_repetitive(text: str, threshold: float = 0.7) -> bool:
    """
    Check if text is highly repetitive by analyzing character n-gram frequencies.
    
    Args:
        text: Text to analyze
        threshold: Repetition threshold (0-1)
        
    Returns:
        True if text appears highly repetitive
    """
    if len(text) < 100:  # Too short to analyze meaningfully
        return False
    
    # Sample the text to avoid performance issues with very long texts
    sample_text = text[:2000] if len(text) > 2000 else text
    
    # Count 3-character sequences
    trigrams = {}
    for i in range(len(sample_text) - 2):
        trigram = sample_text[i:i+3]
        trigrams[trigram] = trigrams.get(trigram, 0) + 1
    
    if not trigrams:
        return False
    
    # Calculate repetition ratio
    total_trigrams = len(sample_text) - 2
    most_common_count = max(trigrams.values())
    
    repetition_ratio = most_common_count / total_trigrams
    return repetition_ratio > threshold