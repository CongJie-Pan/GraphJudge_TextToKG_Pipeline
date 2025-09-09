"""
Unit tests for the data_structures module.

Tests all data structures used in the GraphJudge system,
including TripleData, BootstrapResult, ExplainableJudgment,
and other supporting structures.
"""

import pytest
from typing import List, Dict

from ..data_structures import (
    TripleData,
    BootstrapResult,
    ExplainableJudgment,
    CitationData,
    CitationSummary,
    ProcessingResult,
    BootstrapStatistics,
    ProcessingStatistics
)


class TestTripleData:
    """Test cases for TripleData structure."""
    
    def test_triple_data_creation_full(self):
        """Test TripleData creation with all parameters."""
        triple = TripleData(
            subject="賈寶玉",
            predicate="喜歡",
            object="林黛玉",
            source_line="賈寶玉深深地喜歡林黛玉",
            line_number=5
        )
        
        assert triple.subject == "賈寶玉"
        assert triple.predicate == "喜歡"
        assert triple.object == "林黛玉"
        assert triple.source_line == "賈寶玉深深地喜歡林黛玉"
        assert triple.line_number == 5
    
    def test_triple_data_creation_minimal(self):
        """Test TripleData creation with minimal parameters."""
        triple = TripleData("作者", "創作", "紅樓夢")
        
        assert triple.subject == "作者"
        assert triple.predicate == "創作"
        assert triple.object == "紅樓夢"
        assert triple.source_line == ""
        assert triple.line_number == 0
    
    def test_triple_data_immutability(self):
        """Test that TripleData is immutable (NamedTuple property)."""
        triple = TripleData("Apple", "Founded by", "Steve Jobs")
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            triple.subject = "Microsoft"
    
    def test_triple_data_english_content(self):
        """Test TripleData with English content."""
        triple = TripleData(
            subject="Apple",
            predicate="Founded by",
            object="Steve Jobs",
            source_line="Apple was founded by Steve Jobs in 1976.",
            line_number=1
        )
        
        assert triple.subject == "Apple"
        assert triple.predicate == "Founded by"
        assert triple.object == "Steve Jobs"
        assert "Apple" in triple.source_line
        assert "Steve Jobs" in triple.source_line
    
    def test_triple_data_mixed_content(self):
        """Test TripleData with mixed Chinese/English content."""
        triple = TripleData(
            subject="Apple公司",
            predicate="創立於",
            object="1976年",
            source_line="Apple公司創立於1976年，由Steve Jobs等人創辦。"
        )
        
        assert "Apple" in triple.subject
        assert "創立" in triple.predicate
        assert "1976" in triple.object
        assert "Apple" in triple.source_line and "創立" in triple.source_line


class TestBootstrapResult:
    """Test cases for BootstrapResult structure."""
    
    def test_bootstrap_result_creation(self):
        """Test BootstrapResult creation with all parameters."""
        triple = TripleData("賈寶玉", "喜歡", "林黛玉")
        result = BootstrapResult(
            triple=triple,
            source_idx=5,
            fuzzy_score=0.85,
            auto_expected=True,
            llm_evaluation="Yes",
            expected=True,
            note="High similarity test"
        )
        
        assert result.triple == triple
        assert result.source_idx == 5
        assert result.fuzzy_score == 0.85
        assert result.auto_expected == True
        assert result.llm_evaluation == "Yes"
        assert result.expected == True
        assert result.note == "High similarity test"
    
    def test_bootstrap_result_uncertain_case(self):
        """Test BootstrapResult for uncertain cases."""
        triple = TripleData("Unknown", "Relation", "Entity")
        result = BootstrapResult(
            triple=triple,
            source_idx=10,
            fuzzy_score=0.6,
            auto_expected=None,  # Uncertain
            llm_evaluation=None,
            expected=None,
            note="Low similarity, requires semantic evaluation"
        )
        
        assert result.auto_expected is None
        assert result.llm_evaluation is None
        assert result.expected is None
        assert "semantic evaluation" in result.note
    
    def test_bootstrap_result_negative_case(self):
        """Test BootstrapResult for negative judgment cases."""
        triple = TripleData("Microsoft", "Founded by", "Mark Zuckerberg")
        result = BootstrapResult(
            triple=triple,
            source_idx=0,
            fuzzy_score=0.3,
            auto_expected=False,
            llm_evaluation="No",
            expected=False,
            note="LLM semantic evaluation: No"
        )
        
        assert result.auto_expected == False
        assert result.llm_evaluation == "No"
        assert result.expected == False
        assert "No" in result.note
    
    def test_bootstrap_result_manual_review_case(self):
        """Test BootstrapResult marked for manual review."""
        triple = TripleData("Test", "Subject", "Object")
        result = BootstrapResult(
            triple=triple,
            source_idx=3,
            fuzzy_score=0.75,
            auto_expected=True,
            llm_evaluation="Yes",
            expected=None,  # Manual review needed
            note="LLM evaluation: Yes | SAMPLED FOR MANUAL REVIEW"
        )
        
        assert result.auto_expected == True
        assert result.expected is None
        assert "MANUAL REVIEW" in result.note


class TestExplainableJudgment:
    """Test cases for ExplainableJudgment structure."""
    
    def test_explainable_judgment_positive(self):
        """Test ExplainableJudgment for positive cases."""
        judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.95,
            reasoning="這是一個已知的歷史事實，有充分的文獻證據支持。",
            evidence_sources=["domain_knowledge", "historical_records"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=1.2
        )
        
        assert judgment.judgment == "Yes"
        assert judgment.confidence == 0.95
        assert "歷史事實" in judgment.reasoning
        assert "domain_knowledge" in judgment.evidence_sources
        assert len(judgment.alternative_suggestions) == 0
        assert judgment.error_type is None
        assert judgment.processing_time == 1.2
    
    def test_explainable_judgment_negative_with_alternatives(self):
        """Test ExplainableJudgment for negative cases with alternatives."""
        alternatives = [
            {"subject": "Bill Gates", "relation": "創立", "object": "Microsoft", "confidence": 0.95},
            {"subject": "Paul Allen", "relation": "共同創立", "object": "Microsoft", "confidence": 0.95}
        ]
        
        judgment = ExplainableJudgment(
            judgment="No",
            confidence=0.90,
            reasoning="這個陳述是錯誤的。Microsoft是由Bill Gates和Paul Allen創立的。",
            evidence_sources=["domain_knowledge", "tech_history"],
            alternative_suggestions=alternatives,
            error_type="factual_error",
            processing_time=1.5
        )
        
        assert judgment.judgment == "No"
        assert judgment.confidence == 0.90
        assert "錯誤" in judgment.reasoning
        assert len(judgment.alternative_suggestions) == 2
        assert judgment.alternative_suggestions[0]["subject"] == "Bill Gates"
        assert judgment.error_type == "factual_error"
    
    def test_explainable_judgment_error_case(self):
        """Test ExplainableJudgment for error cases."""
        judgment = ExplainableJudgment(
            judgment="No",
            confidence=0.0,
            reasoning="Error during processing: API timeout",
            evidence_sources=[],
            alternative_suggestions=[],
            error_type="processing_error",
            processing_time=5.0
        )
        
        assert judgment.judgment == "No"
        assert judgment.confidence == 0.0
        assert "Error" in judgment.reasoning
        assert judgment.error_type == "processing_error"
        assert judgment.processing_time == 5.0
    
    def test_explainable_judgment_validation(self):
        """Test ExplainableJudgment field validation."""
        judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.8,
            reasoning="Test reasoning",
            evidence_sources=["source1", "source2"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=0.5
        )
        
        # Validate types
        assert isinstance(judgment.judgment, str)
        assert isinstance(judgment.confidence, float)
        assert isinstance(judgment.reasoning, str)
        assert isinstance(judgment.evidence_sources, list)
        assert isinstance(judgment.alternative_suggestions, list)
        assert isinstance(judgment.processing_time, float)
        
        # Validate ranges
        assert judgment.judgment in ["Yes", "No"]
        assert 0.0 <= judgment.confidence <= 1.0
        assert judgment.processing_time >= 0.0


class TestCitationStructures:
    """Test cases for citation-related structures."""
    
    def test_citation_data_creation(self):
        """Test CitationData structure creation."""
        citation = CitationData(
            number="1",
            title="Red Chamber Wikipedia",
            url="https://en.wikipedia.org/wiki/Dream_of_the_Red_Chamber",
            type="perplexity_citation",
            source="direct"
        )
        
        assert citation.number == "1"
        assert "Red Chamber" in citation.title
        assert citation.url.startswith("https://")
        assert citation.type == "perplexity_citation"
        assert citation.source == "direct"
    
    def test_citation_summary_creation(self):
        """Test CitationSummary structure creation."""
        citations = [
            CitationData("1", "Source 1", "https://example.com/1", "web", "direct"),
            CitationData("2", "Source 2", "https://example.com/2", "web", "reference")
        ]
        
        summary = CitationSummary(
            total_citations=2,
            citations=citations,
            has_citations=True,
            citation_types=["web"]
        )
        
        assert summary.total_citations == 2
        assert len(summary.citations) == 2
        assert summary.has_citations == True
        assert "web" in summary.citation_types
    
    def test_citation_summary_empty(self):
        """Test CitationSummary for empty case."""
        summary = CitationSummary(
            total_citations=0,
            citations=[],
            has_citations=False,
            citation_types=[]
        )
        
        assert summary.total_citations == 0
        assert len(summary.citations) == 0
        assert summary.has_citations == False
        assert len(summary.citation_types) == 0


class TestProcessingStructures:
    """Test cases for processing-related structures."""
    
    def test_processing_result_success(self):
        """Test ProcessingResult for successful processing."""
        reasoning_data = {
            "judgment": "Yes",
            "confidence": 0.9,
            "reasoning": "Test reasoning",
            "evidence_sources": ["test"],
            "processing_time": 1.0
        }
        
        result = ProcessingResult(
            index=0,
            prompt="Is this true: Test Statement ?",
            response="Yes",
            reasoning_data=reasoning_data,
            processing_time=1.0,
            error=None
        )
        
        assert result.index == 0
        assert "Test Statement" in result.prompt
        assert result.response == "Yes"
        assert result.reasoning_data["judgment"] == "Yes"
        assert result.processing_time == 1.0
        assert result.error is None
    
    def test_processing_result_error(self):
        """Test ProcessingResult for error cases."""
        result = ProcessingResult(
            index=1,
            prompt="Is this true: Invalid Statement ?",
            response="Error: Processing failed",
            reasoning_data=None,
            processing_time=0.1,
            error="API timeout"
        )
        
        assert result.index == 1
        assert result.response.startswith("Error:")
        assert result.reasoning_data is None
        assert result.error == "API timeout"
    
    def test_bootstrap_statistics_calculation(self):
        """Test BootstrapStatistics structure."""
        stats = BootstrapStatistics(
            total_triples=100,
            auto_confirmed=60,
            auto_rejected=25,
            manual_review=15,
            coverage_percentage=85.0
        )
        
        assert stats.total_triples == 100
        assert stats.auto_confirmed == 60
        assert stats.auto_rejected == 25
        assert stats.manual_review == 15
        assert stats.coverage_percentage == 85.0
        
        # Verify math consistency
        assert stats.auto_confirmed + stats.auto_rejected + stats.manual_review == stats.total_triples
        assert abs(stats.coverage_percentage - ((stats.auto_confirmed + stats.auto_rejected) / stats.total_triples * 100)) < 0.1
    
    def test_processing_statistics_calculation(self):
        """Test ProcessingStatistics structure."""
        stats = ProcessingStatistics(
            total_instructions=50,
            successful_responses=45,
            error_responses=5,
            yes_judgments=25,
            no_judgments=20,
            success_rate=90.0,
            positive_rate=55.6,
            avg_confidence=0.85,
            unique_error_types=2
        )
        
        assert stats.total_instructions == 50
        assert stats.successful_responses == 45
        assert stats.error_responses == 5
        assert stats.yes_judgments == 25
        assert stats.no_judgments == 20
        assert stats.success_rate == 90.0
        assert stats.positive_rate == 55.6
        assert 0.0 <= stats.avg_confidence <= 1.0
        assert stats.unique_error_types == 2
        
        # Verify math consistency
        assert stats.successful_responses + stats.error_responses == stats.total_instructions
        assert stats.yes_judgments + stats.no_judgments == stats.successful_responses


class TestDataStructureInteractions:
    """Test interactions between different data structures."""
    
    def test_triple_data_in_bootstrap_result(self):
        """Test TripleData used within BootstrapResult."""
        triple = TripleData("Subject", "Predicate", "Object", "source", 1)
        result = BootstrapResult(
            triple=triple,
            source_idx=0,
            fuzzy_score=0.8,
            auto_expected=True,
            llm_evaluation="Yes",
            expected=True,
            note="Test"
        )
        
        # Verify triple is properly embedded
        assert result.triple.subject == "Subject"
        assert result.triple.predicate == "Predicate"
        assert result.triple.object == "Object"
        assert result.triple.line_number == 1
    
    def test_explainable_judgment_with_citations(self):
        """Test ExplainableJudgment with citation integration."""
        judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.9,
            reasoning="Based on reliable sources",
            evidence_sources=["perplexity_citations(2)", "domain_knowledge"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=1.5
        )
        
        # Verify citation integration in evidence sources
        citation_sources = [source for source in judgment.evidence_sources if "perplexity_citations" in source]
        assert len(citation_sources) > 0
        assert "perplexity_citations(2)" in judgment.evidence_sources
    
    def test_data_structure_json_serialization_compatibility(self):
        """Test that data structures are compatible with JSON serialization."""
        # Test TripleData serialization
        triple = TripleData("Test", "Subject", "Object")
        triple_dict = triple._asdict()
        assert isinstance(triple_dict, dict)
        assert triple_dict["subject"] == "Test"
        
        # Test ExplainableJudgment serialization
        judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.8,
            reasoning="Test",
            evidence_sources=["test"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=1.0
        )
        judgment_dict = judgment._asdict()
        assert isinstance(judgment_dict, dict)
        assert judgment_dict["judgment"] == "Yes"
