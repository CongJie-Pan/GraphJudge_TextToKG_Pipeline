"""
Tests to verify that confidence and quality grade features have been completely removed.

This test suite ensures that:
1. ConfidenceLevel enum no longer exists
2. Triple objects have no confidence field
3. JudgmentResult objects have no confidence field
4. UI components don't display confidence/quality
5. API responses don't include confidence data
6. Graph styling is not confidence-based
"""

import pytest
import sys
import os
import json
import inspect
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add the parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.models import Triple, JudgmentResult
from core.graph_judge import GraphJudge
from core.graph_converter import GraphConverter


class TestConfidenceRemoval:
    """Test suite to verify complete removal of confidence features."""

    def test_confidence_level_enum_removed(self):
        """Verify ConfidenceLevel enum no longer exists."""
        try:
            from core.models import ConfidenceLevel
            pytest.fail("ConfidenceLevel enum still exists and should be removed")
        except ImportError:
            # Expected - ConfidenceLevel should not exist
            pass

    def test_triple_no_confidence_field(self):
        """Verify Triple class has no confidence field."""
        # Create a basic triple
        triple = Triple(
            subject="Test Subject",
            predicate="test relation",
            object="Test Object"
        )

        # Check that confidence attribute doesn't exist
        assert not hasattr(triple, 'confidence'), "Triple still has confidence attribute"

        # Check that confidence is not in the Triple class signature
        init_signature = inspect.signature(Triple.__init__)
        param_names = list(init_signature.parameters.keys())
        assert 'confidence' not in param_names, "Triple.__init__ still has confidence parameter"

    def test_triple_to_dict_no_confidence(self):
        """Verify Triple.to_dict() doesn't include confidence."""
        triple = Triple(
            subject="Test Subject",
            predicate="test relation",
            object="Test Object"
        )

        triple_dict = triple.to_dict()
        assert 'confidence' not in triple_dict, "Triple.to_dict() still includes confidence"

    def test_triple_from_dict_no_confidence(self):
        """Verify Triple.from_dict() doesn't expect confidence."""
        test_data = {
            'subject': 'Test Subject',
            'predicate': 'test relation',
            'object': 'Test Object'
        }

        # Should work without confidence field
        triple = Triple.from_dict(test_data)
        assert triple.subject == 'Test Subject'
        assert triple.predicate == 'test relation'
        assert triple.object == 'Test Object'

    def test_triple_str_no_confidence(self):
        """Verify Triple.__str__() doesn't display confidence."""
        triple = Triple(
            subject="Test Subject",
            predicate="test relation",
            object="Test Object"
        )

        str_repr = str(triple)
        assert 'confidence' not in str_repr.lower(), "Triple.__str__() still shows confidence"

    def test_judgment_result_no_confidence_field(self):
        """Verify JudgmentResult class has no confidence field."""
        # Create basic judgment result
        result = JudgmentResult(
            judgments=[True, False, True],
            success=True,
            processing_time=1.5
        )

        # Check that confidence attribute doesn't exist
        assert not hasattr(result, 'confidence'), "JudgmentResult still has confidence attribute"

        # Check that confidence is not in the JudgmentResult class signature
        init_signature = inspect.signature(JudgmentResult.__init__)
        param_names = list(init_signature.parameters.keys())
        assert 'confidence' not in param_names, "JudgmentResult.__init__ still has confidence parameter"

    def test_judgment_result_to_dict_no_confidence(self):
        """Verify JudgmentResult.to_dict() doesn't include confidence."""
        result = JudgmentResult(
            judgments=[True, False],
            success=True,
            processing_time=1.0
        )

        result_dict = result.to_dict()
        assert 'confidence' not in result_dict, "JudgmentResult.to_dict() still includes confidence"

    def test_graph_judge_no_confidence_scoring(self):
        """Verify GraphJudge doesn't generate confidence scores."""
        # Mock the necessary dependencies
        with patch('core.graph_judge.get_api_client') as mock_client, \
             patch('core.graph_judge.DetailedLogger') as mock_logger:

            # Set up mocks
            mock_api = Mock()
            mock_api.create_completion.return_value = "Yes"
            mock_client.return_value = mock_api

            judge = GraphJudge(model_name="test-model")

            # Create test triples
            triples = [
                Triple(subject="A", predicate="relates to", object="B"),
                Triple(subject="B", predicate="connects to", object="C")
            ]

            # Mock the internal methods
            judge._judge_single_triple = Mock(return_value="Yes")

            # Test judgment
            result = judge.judge_triples(triples)

            # Verify no confidence in result
            assert not hasattr(result, 'confidence'), "JudgmentResult still has confidence field"

            # Verify result structure
            assert hasattr(result, 'judgments'), "JudgmentResult missing judgments field"
            assert hasattr(result, 'success'), "JudgmentResult missing success field"
            assert hasattr(result, 'processing_time'), "JudgmentResult missing processing_time field"

    def test_graph_judge_explainable_no_confidence(self):
        """Verify explainable judgment doesn't include confidence."""
        with patch('core.graph_judge.get_api_client') as mock_client, \
             patch('core.graph_judge.DetailedLogger') as mock_logger:

            # Set up mocks
            mock_api = Mock()
            mock_client.return_value = mock_api

            judge = GraphJudge(model_name="test-model")

            # Create test triples
            triples = [Triple(subject="A", predicate="relates to", object="B")]

            # Mock the explanation method to return structure without confidence
            mock_explanation = Mock()
            mock_explanation.judgment = "Yes"
            mock_explanation.reasoning = "Test reasoning"
            mock_explanation.evidence_sources = []
            mock_explanation.actual_citations = []
            mock_explanation.error_type = None
            mock_explanation.processing_time = 1.0
            # Note: no confidence attribute

            judge._judge_with_explanation = Mock(return_value=mock_explanation)

            # Test explainable judgment
            result = judge.judge_triples_with_explanations(triples)

            # Verify no confidence in explainable result
            assert 'confidence' not in result, "Explainable judgment result still includes confidence"

    def test_graph_converter_no_confidence_stats(self):
        """Verify GraphConverter doesn't track confidence statistics."""
        converter = GraphConverter()

        # Check that confidence_scores attribute doesn't exist
        assert not hasattr(converter, 'confidence_scores'), "GraphConverter still has confidence_scores attribute"

        # Check that stats don't include average_confidence
        assert 'average_confidence' not in converter.stats, "GraphConverter stats still include average_confidence"

    def test_graph_converter_no_confidence_styling(self):
        """Verify GraphConverter doesn't use confidence for styling."""
        converter = GraphConverter()

        # Create test triples
        triples = [
            Triple(subject="A", predicate="relates to", object="B"),
            Triple(subject="B", predicate="connects to", object="C")
        ]

        # Convert to graph
        graph_data = converter.convert_triples_to_graph(triples)

        # Check edges don't have confidence-based styling
        if 'edges' in graph_data:
            for edge in graph_data['edges']:
                # All edges should have the same fixed color and width
                assert edge.get('color') == '#1f77b4', f"Edge color is confidence-based: {edge.get('color')}"
                assert edge.get('width') == 3, f"Edge width is confidence-based: {edge.get('width')}"

        # Check Pyvis data doesn't have confidence-based styling
        if 'pyvis_data' in graph_data:
            pyvis_edges = graph_data['pyvis_data'].get('edges', [])
            for edge in pyvis_edges:
                assert edge.get('color') == '#1f77b4', f"Pyvis edge color is confidence-based: {edge.get('color')}"
                assert edge.get('width') == 3, f"Pyvis edge width is confidence-based: {edge.get('width')}"
                # Title should not mention confidence
                title = edge.get('title', '')
                assert 'confidence' not in title.lower(), f"Pyvis edge title still mentions confidence: {title}"
                assert '置信度' not in title, f"Pyvis edge title still mentions confidence in Chinese: {title}"

    def test_graph_converter_report_no_confidence(self):
        """Verify graph report doesn't include confidence metrics."""
        converter = GraphConverter()

        # Create test triples
        triples = [Triple(subject="A", predicate="relates to", object="B")]
        graph_data = converter.convert_triples_to_graph(triples)

        # Check that report doesn't include confidence metrics
        if 'report' in graph_data:
            report = graph_data['report']

            # Check summary section
            if 'summary' in report:
                summary = report['summary']
                assert 'average_confidence' not in summary, "Graph report summary still includes average_confidence"

            # Check quality_metrics section
            if 'quality_metrics' in report:
                quality_metrics = report['quality_metrics']
                assert 'average_confidence' not in quality_metrics, "Graph report quality_metrics still includes average_confidence"

    def test_no_confidence_methods_exist(self):
        """Verify confidence-related methods have been removed."""
        # Check GraphJudge doesn't have confidence estimation method
        judge = GraphJudge.__new__(GraphJudge)  # Create without calling __init__
        assert not hasattr(judge, '_estimate_confidence'), "GraphJudge still has _estimate_confidence method"

        # Check GraphConverter doesn't have confidence color method
        converter = GraphConverter()
        assert not hasattr(converter, '_get_edge_color'), "GraphConverter still has _get_edge_color method"

    def test_create_error_result_no_confidence(self):
        """Verify error result creation doesn't include confidence."""
        from core.models import create_error_result

        # Test JudgmentResult error creation
        error_result = create_error_result(JudgmentResult, "Test error", 1.0)

        assert isinstance(error_result, JudgmentResult)
        assert not hasattr(error_result, 'confidence'), "Error JudgmentResult still has confidence field"
        assert error_result.success == False
        assert error_result.error == "Test error"
        assert error_result.processing_time == 1.0


class TestConfidenceUIRemoval:
    """Test suite to verify UI components don't display confidence features."""

    def test_no_confidence_display_functions(self):
        """Verify confidence display functions have been removed."""
        try:
            from ui.display import get_quality_grade
            pytest.fail("get_quality_grade function still exists and should be removed")
        except ImportError:
            # Expected - function should not exist
            pass

    def test_ui_components_import_successfully(self):
        """Verify UI components can be imported without confidence dependencies."""
        try:
            from ui import components
            from ui import display
            # If imports succeed without errors, confidence references have been removed
            assert True
        except ImportError as e:
            pytest.fail(f"UI components failed to import after confidence removal: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])