"""
Integration tests for explanation button functionality in GraphJudge Streamlit Pipeline.

This module tests the end-to-end integration of the explanation toggle feature,
ensuring that the UI checkbox properly controls the generation of explanations
in the pipeline judgment stage.
"""

import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any

# Import core pipeline components
from streamlit_pipeline.core.pipeline import PipelineOrchestrator
from streamlit_pipeline.core.models import Triple, JudgmentResult


@pytest.fixture
def sample_triples():
    """Provide sample triples for testing."""
    return [
        Triple(
            subject="賈寶玉",
            predicate="居住於",
            object="榮國府",
            source_text="賈寶玉住在榮國府"
        ),
        Triple(
            subject="林黛玉",
            predicate="來自",
            object="江南",
            source_text="林黛玉從江南來"
        )
    ]


@pytest.fixture
def mock_judge_triples():
    """Mock the judge_triples function for testing."""
    with patch('streamlit_pipeline.core.pipeline.judge_triples') as mock:
        mock.return_value = JudgmentResult(
            judgments=[True, True],
            confidence=[0.9, 0.8],
            explanations=None,  # Basic function returns no explanations
            success=True,
            processing_time=1.5
        )
        yield mock


@pytest.fixture
def mock_judge_triples_with_explanations():
    """Mock the judge_triples_with_explanations function for testing."""
    with patch('streamlit_pipeline.core.pipeline.judge_triples_with_explanations') as mock:
        mock.return_value = {
            "judgments": [True, True],
            "confidence": [0.9, 0.8],
            "explanations": [
                {
                    "reasoning": "This triple is correct based on the classical text context",
                    "evidence_sources": ["原文描述"],
                    "actual_citations": ["https://example.com/source1"],
                    "error_type": None
                },
                {
                    "reasoning": "The relationship is well-supported by textual evidence",
                    "evidence_sources": ["文本證據"],
                    "actual_citations": ["https://example.com/source2"],
                    "error_type": None
                }
            ],
            "success": True,
            "processing_time": 2.3
        }
        yield mock


class TestExplanationIntegration:
    """Test explanation functionality integration."""

    def test_pipeline_with_explanations_enabled(self, sample_triples, mock_judge_triples_with_explanations):
        """Test pipeline execution with explanations enabled."""
        orchestrator = PipelineOrchestrator()

        # Configuration with explanations enabled
        config_options = {'enable_explanations': True}

        # Test the _execute_judgment_stage method directly
        result = orchestrator._execute_judgment_stage(sample_triples, config_options)

        # Verify the correct function was called
        mock_judge_triples_with_explanations.assert_called_once_with(
            sample_triples, include_reasoning=True
        )

        # Verify result structure
        assert isinstance(result, JudgmentResult)
        assert result.success is True
        assert result.judgments == [True, True]
        assert result.confidence == [0.9, 0.8]
        assert result.explanations is not None
        assert len(result.explanations) == 2
        assert result.explanations[0]["reasoning"] == "This triple is correct based on the classical text context"

    def test_pipeline_with_explanations_disabled(self, sample_triples, mock_judge_triples):
        """Test pipeline execution with explanations disabled."""
        orchestrator = PipelineOrchestrator()

        # Configuration with explanations disabled
        config_options = {'enable_explanations': False}

        # Test the _execute_judgment_stage method directly
        result = orchestrator._execute_judgment_stage(sample_triples, config_options)

        # Verify the correct function was called
        mock_judge_triples.assert_called_once_with(sample_triples)

        # Verify result structure
        assert isinstance(result, JudgmentResult)
        assert result.success is True
        assert result.judgments == [True, True]
        assert result.confidence == [0.9, 0.8]
        assert result.explanations is None  # No explanations when disabled

    def test_pipeline_with_no_config(self, sample_triples, mock_judge_triples):
        """Test pipeline execution with no config (default behavior)."""
        orchestrator = PipelineOrchestrator()

        # No configuration provided
        result = orchestrator._execute_judgment_stage(sample_triples, None)

        # Should default to basic judgment without explanations
        mock_judge_triples.assert_called_once_with(sample_triples)

        # Verify result structure
        assert isinstance(result, JudgmentResult)
        assert result.success is True
        assert result.explanations is None

    def test_pipeline_with_empty_config(self, sample_triples, mock_judge_triples):
        """Test pipeline execution with empty config dictionary."""
        orchestrator = PipelineOrchestrator()

        # Empty configuration
        config_options = {}

        result = orchestrator._execute_judgment_stage(sample_triples, config_options)

        # Should default to basic judgment without explanations
        mock_judge_triples.assert_called_once_with(sample_triples)

        # Verify result structure
        assert isinstance(result, JudgmentResult)
        assert result.success is True
        assert result.explanations is None

    def test_config_parameter_backward_compatibility(self):
        """Test that the new config parameter doesn't break existing code."""
        orchestrator = PipelineOrchestrator()

        # Test that run_pipeline can be called with old signature (2 parameters)
        with patch('streamlit_pipeline.core.pipeline.extract_entities') as mock_entity, \
             patch('streamlit_pipeline.core.pipeline.generate_triples') as mock_triple, \
             patch('streamlit_pipeline.core.pipeline.judge_triples') as mock_judge:

            # Mock successful stages
            mock_entity.return_value = Mock(
                success=True,
                entities=["賈寶玉", "林黛玉"],
                denoised_text="測試文本",
                processing_time=1.0
            )

            mock_triple.return_value = Mock(
                success=True,
                triples=[Mock(subject="賈寶玉", predicate="居住於", object="榮國府")],
                processing_time=1.0,
                metadata={}
            )

            mock_judge.return_value = JudgmentResult(
                judgments=[True],
                confidence=[0.9],
                explanations=None,
                success=True,
                processing_time=1.0
            )

            # Call with old signature (should work)
            result = orchestrator.run_pipeline("測試文本")
            assert result.success is True

    def test_explanation_content_structure(self, sample_triples, mock_judge_triples_with_explanations):
        """Test that explanation content has the expected structure."""
        orchestrator = PipelineOrchestrator()
        config_options = {'enable_explanations': True}

        result = orchestrator._execute_judgment_stage(sample_triples, config_options)

        # Verify explanation structure
        assert result.explanations is not None
        for explanation in result.explanations:
            assert "reasoning" in explanation
            assert "evidence_sources" in explanation
            assert "error_type" in explanation
            assert isinstance(explanation["reasoning"], str)
            assert isinstance(explanation["evidence_sources"], list)

    def test_performance_difference_logged(self, sample_triples):
        """Test that there's a measurable performance difference between modes."""
        orchestrator = PipelineOrchestrator()

        with patch('streamlit_pipeline.core.pipeline.judge_triples') as mock_basic, \
             patch('streamlit_pipeline.core.pipeline.judge_triples_with_explanations') as mock_detailed:

            # Mock different processing times
            mock_basic.return_value = JudgmentResult(
                judgments=[True, True],
                confidence=[0.9, 0.8],
                explanations=None,
                success=True,
                processing_time=1.0  # Faster
            )

            mock_detailed.return_value = {
                "judgments": [True, True],
                "confidence": [0.9, 0.8],
                "explanations": [{"reasoning": "test"}, {"reasoning": "test"}],
                "success": True,
                "processing_time": 3.0  # Slower due to explanations
            }

            # Test basic mode
            result_basic = orchestrator._execute_judgment_stage(
                sample_triples, {'enable_explanations': False}
            )

            # Test explanation mode
            result_detailed = orchestrator._execute_judgment_stage(
                sample_triples, {'enable_explanations': True}
            )

            # Verify performance difference is captured
            assert result_detailed.processing_time > result_basic.processing_time