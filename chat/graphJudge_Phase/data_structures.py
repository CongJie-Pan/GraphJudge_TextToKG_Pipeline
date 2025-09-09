"""
Data structures and models for the GraphJudge system.

This module defines all data structures used throughout the GraphJudge system,
including NamedTuple classes for knowledge graph triples, bootstrap results,
and explainable judgment results.
"""

from typing import NamedTuple, List, Optional, Dict, Any


class TripleData(NamedTuple):
    """
    Data structure for knowledge graph triples.
    
    Attributes:
        subject (str): The subject of the triple
        predicate (str): The predicate/relation of the triple
        object (str): The object of the triple
        source_line (str): The source line where this triple was extracted
        line_number (int): The line number in the source file
    """
    subject: str
    predicate: str
    object: str
    source_line: str = ""
    line_number: int = 0


class BootstrapResult(NamedTuple):
    """
    Result of gold label bootstrapping for a single triple.
    
    Attributes:
        triple (TripleData): The triple being evaluated
        source_idx (int): Index of the best matching source line
        fuzzy_score (float): RapidFuzz similarity score (0.0-1.0)
        auto_expected (Optional[bool]): Auto-assigned expected value (None for uncertain)
        llm_evaluation (Optional[str]): LLM's semantic evaluation if performed
        expected (Optional[bool]): Final expected value (None for manual review cases)
        note (str): Additional notes or error messages
    """
    triple: TripleData
    source_idx: int
    fuzzy_score: float
    auto_expected: Optional[bool]  # None for uncertain, True/False for confident
    llm_evaluation: Optional[str]  # LLM's semantic evaluation if performed
    expected: Optional[bool]       # Final expected value (for manual review cases)
    note: str                      # Additional notes or error messages


class ExplainableJudgment(NamedTuple):
    """
    Data structure for explainable graph judgment results.
    
    This structure provides comprehensive information about a graph judgment,
    including the binary decision, confidence level, detailed reasoning,
    evidence sources, and alternative suggestions.
    
    Attributes:
        judgment (str): Binary decision ("Yes" or "No")
        confidence (float): Confidence score (0.0-1.0)
        reasoning (str): Detailed reasoning explanation
        evidence_sources (List[str]): Sources of evidence used
        alternative_suggestions (List[Dict]): Alternative suggestions if judgment is "No"
        error_type (Optional[str]): Error classification (if applicable)
        processing_time (float): Time taken to process (seconds)
    """
    judgment: str                    # "Yes" or "No" binary decision
    confidence: float                # Confidence score (0.0-1.0)
    reasoning: str                   # Detailed reasoning explanation
    evidence_sources: List[str]      # Sources of evidence used
    alternative_suggestions: List[Dict]  # Alternative suggestions if judgment is "No"
    error_type: Optional[str]        # Error classification (if applicable)
    processing_time: float           # Time taken to process (seconds)


class CitationData(NamedTuple):
    """
    Data structure for citation information from Perplexity API.
    
    Attributes:
        number (str): Citation number/identifier
        title (str): Title or description of the citation
        url (str): URL of the citation source
        type (str): Type of citation (e.g., "perplexity_citation")
        source (str): Source of the citation (e.g., "direct", "content_reference")
    """
    number: str
    title: str
    url: str
    type: str
    source: str


class CitationSummary(NamedTuple):
    """
    Summary of citations from a Perplexity response.
    
    Attributes:
        total_citations (int): Total number of citations
        citations (List[CitationData]): List of citation data
        has_citations (bool): Whether any citations were found
        citation_types (List[str]): Types of citations found
    """
    total_citations: int
    citations: List[CitationData]
    has_citations: bool
    citation_types: List[str]


class ProcessingResult(NamedTuple):
    """
    Result of processing a single instruction.
    
    Attributes:
        index (int): Index of the instruction in the dataset
        prompt (str): The original instruction/prompt
        response (str): The binary response ("Yes" or "No")
        reasoning_data (Optional[Dict]): Detailed reasoning data if in explainable mode
        processing_time (float): Time taken to process
        error (Optional[str]): Error message if processing failed
    """
    index: int
    prompt: str
    response: str
    reasoning_data: Optional[Dict[str, Any]]
    processing_time: float
    error: Optional[str] = None


class BootstrapStatistics(NamedTuple):
    """
    Statistics from gold label bootstrapping process.
    
    Attributes:
        total_triples (int): Total number of triples processed
        auto_confirmed (int): Number of triples auto-confirmed as True
        auto_rejected (int): Number of triples auto-rejected as False
        manual_review (int): Number of triples requiring manual review
        coverage_percentage (float): Percentage of triples auto-labeled
    """
    total_triples: int
    auto_confirmed: int
    auto_rejected: int
    manual_review: int
    coverage_percentage: float


class ProcessingStatistics(NamedTuple):
    """
    Statistics from graph judgment processing.
    
    Attributes:
        total_instructions (int): Total number of instructions processed
        successful_responses (int): Number of successful responses
        error_responses (int): Number of error responses
        yes_judgments (int): Number of "Yes" judgments
        no_judgments (int): Number of "No" judgments
        success_rate (float): Success rate percentage
        positive_rate (float): Positive judgment rate percentage
        avg_confidence (float): Average confidence score (explainable mode)
        unique_error_types (int): Number of unique error types
    """
    total_instructions: int
    successful_responses: int
    error_responses: int
    yes_judgments: int
    no_judgments: int
    success_rate: float
    positive_rate: float
    avg_confidence: float
    unique_error_types: int
