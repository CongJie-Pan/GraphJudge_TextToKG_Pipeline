"""
Comprehensive test suite for graph evaluation system.

This module provides thorough testing of the graph quality evaluation functionality,
including all metric implementations and the main evaluation engine.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.models import Triple, GraphMetrics, EvaluationResult
from eval.graph_evaluator import GraphEvaluator, evaluate_graph, evaluate_batch


class TestGraphEvaluator:
    """Test class for GraphEvaluator functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.evaluator = GraphEvaluator(enable_ged=False, enable_bert_score=True)

        # Sample test graphs
        self.predicted_graph = [
            Triple("Paris", "capital_of", "France"),
            Triple("London", "capital_of", "UK"),
            Triple("France", "located_in", "Europe")
        ]

        self.reference_graph = [
            Triple("Paris", "capital_of", "France"),
            Triple("Berlin", "capital_of", "Germany"),
            Triple("France", "located_in", "Europe")
        ]

        self.empty_graph = []

    def test_evaluator_initialization(self):
        """Test GraphEvaluator initialization with different configurations."""
        # Test default initialization
        evaluator = GraphEvaluator()
        assert evaluator.enable_ged == False
        assert evaluator.enable_bert_score == True
        assert evaluator.max_evaluation_time == 30.0
        assert evaluator.evaluation_count == 0
        assert evaluator.total_evaluation_time == 0.0

        # Test custom initialization
        evaluator = GraphEvaluator(enable_ged=True, enable_bert_score=False, max_evaluation_time=60.0)
        assert evaluator.enable_ged == True
        assert evaluator.enable_bert_score == False
        assert evaluator.max_evaluation_time == 60.0

    def test_input_validation_success(self):
        """Test successful input validation."""
        result = self.evaluator._validate_inputs(self.predicted_graph, self.reference_graph)
        assert result[0] == True
        assert result[1] is None

    def test_input_validation_failures(self):
        """Test various input validation failure cases."""
        # Test non-list inputs
        result = self.evaluator._validate_inputs("not a list", self.reference_graph)
        assert result[0] == False
        assert "must be a list" in result[1]

        result = self.evaluator._validate_inputs(self.predicted_graph, "not a list")
        assert result[0] == False
        assert "must be a list" in result[1]

        # Test empty graphs
        result = self.evaluator._validate_inputs([], self.reference_graph)
        assert result[0] == False
        assert "cannot be empty" in result[1]

        result = self.evaluator._validate_inputs(self.predicted_graph, [])
        assert result[0] == False
        assert "cannot be empty" in result[1]

        # Test invalid triple objects
        invalid_graph = ["not a triple", Triple("s", "p", "o")]
        result = self.evaluator._validate_inputs(invalid_graph, self.reference_graph)
        assert result[0] == False
        assert "not a Triple object" in result[1]

    def test_graph_format_conversion(self):
        """Test conversion of Triple objects to evaluation format."""
        result = self.evaluator._convert_graph_format(self.predicted_graph)

        expected = [
            ["Paris", "capital_of", "France"],
            ["London", "capital_of", "UK"],
            ["France", "located_in", "Europe"]
        ]

        assert result == expected
        assert len(result) == len(self.predicted_graph)

    def test_graph_info_generation(self):
        """Test graph statistics generation."""
        info = self.evaluator._get_graph_info(self.predicted_graph)

        assert info["size"] == 3
        assert info["unique_subjects"] == 3
        assert info["unique_predicates"] == 2
        assert info["unique_objects"] == 3
        assert len(info["subject_examples"]) <= 5
        assert len(info["predicate_examples"]) <= 5
        assert len(info["object_examples"]) <= 5

        # Test empty graph
        empty_info = self.evaluator._get_graph_info([])
        assert empty_info["size"] == 0
        assert empty_info["unique_subjects"] == 0

    def test_empty_metrics_generation(self):
        """Test empty metrics object creation."""
        metrics = self.evaluator._get_empty_metrics()

        assert isinstance(metrics, GraphMetrics)
        assert metrics.triple_match_f1 == 0.0
        assert metrics.graph_match_accuracy == 0.0
        assert metrics.g_bleu_f1 == 0.0
        assert metrics.g_rouge_f1 == 0.0
        assert metrics.g_bert_f1 == 0.0
        assert metrics.graph_edit_distance is None

    @patch('streamlit_pipeline.eval.metrics.exact_matching.get_triple_match_f1')
    @patch('streamlit_pipeline.eval.metrics.exact_matching.get_graph_match_accuracy')
    @patch('streamlit_pipeline.eval.metrics.text_similarity.get_bleu_rouge_scores')
    @patch('streamlit_pipeline.eval.metrics.semantic_similarity.get_bert_score')
    def test_metric_computation_success(self, mock_bert, mock_text, mock_accuracy, mock_f1):
        """Test successful metric computation with mocked dependencies."""
        # Setup mock returns
        mock_f1.return_value = 0.8
        mock_accuracy.return_value = 0.75
        mock_text.return_value = {
            'bleu': {'precision': 0.7, 'recall': 0.6, 'f1': 0.65},
            'rouge': {'precision': 0.72, 'recall': 0.68, 'f1': 0.7}
        }
        mock_bert.return_value = {'precision': 0.85, 'recall': 0.8, 'f1': 0.825}

        # Test metric computation
        result = self.evaluator._compute_all_metrics(
            [["Paris", "capital_of", "France"]],
            [["Paris", "capital_of", "France"]],
            enable_ged=False
        )

        assert result[0] == True
        metrics = result[1]
        assert isinstance(metrics, GraphMetrics)
        assert metrics.triple_match_f1 == 0.8
        assert metrics.graph_match_accuracy == 0.75
        assert metrics.g_bleu_f1 == 0.65
        assert metrics.g_rouge_f1 == 0.7
        assert metrics.g_bert_f1 == 0.825

    @patch('streamlit_pipeline.eval.metrics.exact_matching.get_triple_match_f1')
    def test_metric_computation_failure(self, mock_f1):
        """Test metric computation failure handling."""
        # Setup mock to raise exception
        mock_f1.side_effect = Exception("Computation failed")

        result = self.evaluator._compute_all_metrics(
            [["Paris", "capital_of", "France"]],
            [["Paris", "capital_of", "France"]],
            enable_ged=False
        )

        assert result[0] == False
        assert "Failed to compute metrics" in result[1]

    @patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator._compute_all_metrics')
    def test_evaluate_graph_success(self, mock_compute):
        """Test successful graph evaluation."""
        # Setup mock return
        mock_metrics = GraphMetrics(
            triple_match_f1=0.8, graph_match_accuracy=0.75,
            g_bleu_precision=0.7, g_bleu_recall=0.6, g_bleu_f1=0.65,
            g_rouge_precision=0.72, g_rouge_recall=0.68, g_rouge_f1=0.7,
            g_bert_precision=0.85, g_bert_recall=0.8, g_bert_f1=0.825
        )
        mock_compute.return_value = (True, mock_metrics)

        result = self.evaluator.evaluate_graph(self.predicted_graph, self.reference_graph)

        assert isinstance(result, EvaluationResult)
        assert result.success == True
        assert result.error is None
        assert result.metrics == mock_metrics
        assert result.processing_time > 0
        assert "predicted_graph_size" in result.metadata
        assert "reference_graph_size" in result.metadata

    def test_evaluate_graph_validation_failure(self):
        """Test graph evaluation with validation failure."""
        result = self.evaluator.evaluate_graph([], self.reference_graph)

        assert isinstance(result, EvaluationResult)
        assert result.success == False
        assert result.error is not None
        assert "cannot be empty" in result.error
        assert result.metadata["error_type"] == "validation_error"

    @patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator._compute_all_metrics')
    def test_evaluate_graph_computation_failure(self, mock_compute):
        """Test graph evaluation with computation failure."""
        mock_compute.return_value = (False, "Metric computation failed")

        result = self.evaluator.evaluate_graph(self.predicted_graph, self.reference_graph)

        assert isinstance(result, EvaluationResult)
        assert result.success == False
        assert result.error == "Metric computation failed"
        assert result.metadata["error_type"] == "computation_error"

    @patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator.evaluate_graph')
    def test_evaluate_batch_success(self, mock_evaluate):
        """Test successful batch evaluation."""
        # Setup mock return
        mock_result = EvaluationResult(
            metrics=self.evaluator._get_empty_metrics(),
            metadata={}, success=True, processing_time=0.1
        )
        mock_evaluate.return_value = mock_result

        graph_pairs = [
            (self.predicted_graph, self.reference_graph),
            (self.reference_graph, self.predicted_graph)
        ]

        results = self.evaluator.evaluate_batch(graph_pairs, show_progress=False)

        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert mock_evaluate.call_count == 2

    def test_evaluation_statistics(self):
        """Test evaluation statistics tracking."""
        # Initial stats
        stats = self.evaluator.get_evaluation_stats()
        assert stats["evaluation_count"] == 0
        assert stats["total_evaluation_time"] == 0.0
        assert stats["average_evaluation_time"] == 0.0

        # Mock a successful evaluation to update stats
        with patch.object(self.evaluator, '_compute_all_metrics') as mock_compute:
            mock_compute.return_value = (True, self.evaluator._get_empty_metrics())

            # Perform evaluation
            self.evaluator.evaluate_graph(self.predicted_graph, self.reference_graph)

            # Check updated stats
            updated_stats = self.evaluator.get_evaluation_stats()
            assert updated_stats["evaluation_count"] == 1
            assert updated_stats["total_evaluation_time"] > 0
            assert updated_stats["average_evaluation_time"] > 0

    def test_ged_override(self):
        """Test GED computation override functionality."""
        evaluator = GraphEvaluator(enable_ged=False)  # Default disabled

        with patch.object(evaluator, '_compute_all_metrics') as mock_compute:
            mock_compute.return_value = (True, evaluator._get_empty_metrics())

            # Test override to enable GED
            evaluator.evaluate_graph(
                self.predicted_graph,
                self.reference_graph,
                enable_ged_override=True
            )

            # Verify GED was enabled for this call
            call_args = mock_compute.call_args[0]
            assert call_args[2] == True  # enable_ged parameter

    def test_performance_timing(self):
        """Test that evaluation timing is properly measured."""
        with patch.object(self.evaluator, '_compute_all_metrics') as mock_compute:
            mock_compute.return_value = (True, self.evaluator._get_empty_metrics())

            # Add artificial delay to computation
            def delayed_computation(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return (True, self.evaluator._get_empty_metrics())

            mock_compute.side_effect = delayed_computation

            result = self.evaluator.evaluate_graph(self.predicted_graph, self.reference_graph)

            # Verify timing was captured
            assert result.processing_time >= 0.01
            assert result.metadata["performance"]["processing_time"] >= 0.01


class TestConvenienceFunctions:
    """Test convenience functions for direct usage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predicted_graph = [Triple("A", "rel", "B")]
        self.reference_graph = [Triple("A", "rel", "B")]

    @patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator')
    def test_evaluate_graph_function(self, mock_evaluator_class):
        """Test the convenience evaluate_graph function."""
        mock_evaluator = Mock()
        mock_result = EvaluationResult(
            metrics=GraphMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            metadata={}, success=True, processing_time=0.1
        )
        mock_evaluator.evaluate_graph.return_value = mock_result
        mock_evaluator_class.return_value = mock_evaluator

        result = evaluate_graph(self.predicted_graph, self.reference_graph, enable_ged=True)

        mock_evaluator_class.assert_called_once_with(enable_ged=True)
        mock_evaluator.evaluate_graph.assert_called_once_with(
            self.predicted_graph, self.reference_graph
        )
        assert result == mock_result

    @patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator')
    def test_evaluate_batch_function(self, mock_evaluator_class):
        """Test the convenience evaluate_batch function."""
        mock_evaluator = Mock()
        mock_results = [EvaluationResult(
            metrics=GraphMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            metadata={}, success=True, processing_time=0.1
        )]
        mock_evaluator.evaluate_batch.return_value = mock_results
        mock_evaluator_class.return_value = mock_evaluator

        graph_pairs = [(self.predicted_graph, self.reference_graph)]
        results = evaluate_batch(graph_pairs, enable_ged=False, show_progress=True)

        mock_evaluator_class.assert_called_once_with(enable_ged=False)
        mock_evaluator.evaluate_batch.assert_called_once_with(graph_pairs, True)
        assert results == mock_results


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = GraphEvaluator()
        self.valid_graph = [Triple("A", "rel", "B")]

    def test_exception_during_evaluation(self):
        """Test handling of unexpected exceptions during evaluation."""
        with patch.object(self.evaluator, '_validate_inputs') as mock_validate:
            mock_validate.side_effect = Exception("Unexpected error")

            result = self.evaluator.evaluate_graph(self.valid_graph, self.valid_graph)

            assert result.success == False
            assert "Evaluation failed" in result.error
            assert result.metadata["error_type"] == "computation_error"

    def test_invalid_triple_attributes(self):
        """Test handling of Triple objects with invalid attributes."""
        # Create triple with None values
        invalid_graph = [Triple(None, "predicate", "object")]

        # The evaluation should handle this gracefully
        result = self.evaluator.evaluate_graph(invalid_graph, self.valid_graph)

        # Should either succeed with converted values or fail gracefully
        assert isinstance(result, EvaluationResult)
        if not result.success:
            assert result.error is not None


class TestIntegration:
    """Integration tests for the complete evaluation system."""

    def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline with real metric computation."""
        evaluator = GraphEvaluator(enable_ged=False, enable_bert_score=False)

        # Create realistic test graphs
        predicted = [
            Triple("Paris", "capital_of", "France"),
            Triple("France", "located_in", "Europe"),
            Triple("Europe", "is_a", "continent")
        ]

        reference = [
            Triple("Paris", "capital_of", "France"),  # Exact match
            Triple("London", "capital_of", "UK"),     # Different
            Triple("Europe", "is_a", "continent")     # Exact match
        ]

        # This test requires the actual metric implementations to work
        try:
            result = evaluator.evaluate_graph(predicted, reference)

            # Basic validation of result structure
            assert isinstance(result, EvaluationResult)
            assert isinstance(result.metrics, GraphMetrics)
            assert result.processing_time > 0

            # Check that some metrics are reasonable
            if result.success:
                assert 0.0 <= result.metrics.triple_match_f1 <= 1.0
                assert 0.0 <= result.metrics.graph_match_accuracy <= 1.0
                assert result.metrics.get_overall_score() >= 0.0

        except ImportError as e:
            # Skip if required dependencies are not available
            pytest.skip(f"Skipping integration test due to missing dependency: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])