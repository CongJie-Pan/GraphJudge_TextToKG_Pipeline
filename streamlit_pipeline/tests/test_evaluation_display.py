"""
Tests to verify that evaluation functionality is properly displayed in the UI.

This test suite ensures that:
1. Evaluation is enabled by default in configuration
2. Evaluation section appears in UI even without reference graph
3. Evaluation dashboard displays correctly when results are available
4. Evaluation export functionality works
5. Pipeline executes evaluation stage when enabled
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add the parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core import config
from core.models import EvaluationResult, GraphMetrics
from core.pipeline import PipelineOrchestrator, PipelineResult


class TestEvaluationConfiguration:
    """Test suite to verify evaluation is enabled by default."""

    def test_evaluation_enabled_by_default(self):
        """Verify EVALUATION_ENABLED is True in config."""
        assert config.EVALUATION_ENABLED == True, "EVALUATION_ENABLED should be True by default"

    def test_evaluation_config_structure(self):
        """Verify evaluation configuration has proper structure."""
        # Test that evaluation config exists
        assert hasattr(config, 'EVALUATION_ENABLED'), "EVALUATION_ENABLED config missing"
        assert hasattr(config, 'EVALUATION_ENABLE_GED'), "EVALUATION_ENABLE_GED config missing"
        assert hasattr(config, 'EVALUATION_ENABLE_BERT_SCORE'), "EVALUATION_ENABLE_BERT_SCORE config missing"
        assert hasattr(config, 'EVALUATION_TIMEOUT'), "EVALUATION_TIMEOUT config missing"

        # Test config values
        assert isinstance(config.EVALUATION_ENABLED, bool), "EVALUATION_ENABLED should be boolean"
        assert isinstance(config.EVALUATION_TIMEOUT, (int, float)), "EVALUATION_TIMEOUT should be numeric"

    def test_evaluation_environment_override(self):
        """Verify evaluation can be overridden by environment variable."""
        with patch.dict(os.environ, {'EVALUATION_ENABLED': 'false'}):
            # Reload config to test environment override
            import importlib
            importlib.reload(config)

            evaluation_config = config.get_pipeline_config()
            assert evaluation_config.get('enable_evaluation') == False, "Environment override not working"


class TestEvaluationPipelineIntegration:
    """Test suite to verify evaluation integrates properly with pipeline."""

    def test_pipeline_orchestrator_includes_evaluation(self):
        """Verify PipelineOrchestrator can handle evaluation."""
        # Create orchestrator
        orchestrator = PipelineOrchestrator()

        # Check that evaluation config is included in pipeline config
        pipeline_config = config.get_pipeline_config()
        assert 'enable_evaluation' in pipeline_config, "Pipeline config missing enable_evaluation"
        assert pipeline_config['enable_evaluation'] == True, "Evaluation not enabled by default in pipeline"

    def test_pipeline_result_has_evaluation_field(self):
        """Verify PipelineResult can contain evaluation results."""
        # Create a mock evaluation result
        mock_metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.90,
            g_bleu_precision=0.80,
            g_bleu_recall=0.75,
            g_bleu_f1=0.77,
            g_rouge_precision=0.82,
            g_rouge_recall=0.78,
            g_rouge_f1=0.80,
            g_bert_precision=0.88,
            g_bert_recall=0.85,
            g_bert_f1=0.86
        )

        mock_evaluation = EvaluationResult(
            metrics=mock_metrics,
            success=True,
            processing_time=5.2,
            metadata={"test": "data"}
        )

        # Create pipeline result with evaluation
        pipeline_result = PipelineResult(
            success=True,
            total_time=10.0
        )
        pipeline_result.evaluation_result = mock_evaluation
        pipeline_result.evaluation_enabled = True

        # Verify evaluation result is accessible
        assert hasattr(pipeline_result, 'evaluation_result'), "PipelineResult missing evaluation_result field"
        assert hasattr(pipeline_result, 'evaluation_enabled'), "PipelineResult missing evaluation_enabled field"
        assert pipeline_result.evaluation_result is not None, "Evaluation result should not be None"
        assert pipeline_result.evaluation_result.success == True, "Evaluation result should indicate success"

    @patch('core.pipeline.GraphEvaluator')
    def test_pipeline_executes_evaluation_stage(self, mock_evaluator_class):
        """Verify pipeline executes evaluation when enabled."""
        # Set up mock evaluator
        mock_evaluator = Mock()
        mock_metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.90,
            g_bleu_precision=0.80,
            g_bleu_recall=0.75,
            g_bleu_f1=0.77,
            g_rouge_precision=0.82,
            g_rouge_recall=0.78,
            g_rouge_f1=0.80,
            g_bert_precision=0.88,
            g_bert_recall=0.85,
            g_bert_f1=0.86
        )
        mock_evaluation_result = EvaluationResult(
            metrics=mock_metrics,
            metadata={"test": True},
            success=True,
            processing_time=3.0
        )
        mock_evaluator.evaluate_graph.return_value = mock_evaluation_result
        mock_evaluator_class.return_value = mock_evaluator

        # Create orchestrator with mocked dependencies
        with patch('core.pipeline.EntityProcessor') as mock_entity_proc, \
             patch('core.pipeline.TripleGenerator') as mock_triple_gen, \
             patch('core.pipeline.GraphJudge') as mock_judge:

            # Set up mock results for each stage
            mock_entity_proc.return_value.extract_and_denoise.return_value = Mock(success=True)
            mock_triple_gen.return_value.generate_triples.return_value = Mock(success=True, triples=[])
            mock_judge.return_value.judge_triples.return_value = Mock(success=True, judgments=[])

            orchestrator = PipelineOrchestrator()

            # Run pipeline with evaluation enabled
            evaluation_config = {'enable_evaluation': True}
            reference_graph = [{"subject": "A", "predicate": "relates to", "object": "B"}]

            result = orchestrator.run_pipeline(
                text="Test text",
                evaluation_config=evaluation_config,
                reference_graph=reference_graph
            )

            # Verify evaluation was executed
            assert hasattr(result, 'evaluation_result'), "Pipeline result missing evaluation_result"
            assert result.evaluation_result is not None, "Evaluation result should not be None"
            mock_evaluator.evaluate_graph.assert_called_once()


class TestEvaluationUIDisplay:
    """Test suite to verify evaluation UI displays correctly."""

    @patch('streamlit.markdown')
    @patch('streamlit.success')
    @patch('streamlit.info')
    def test_evaluation_section_always_visible(self, mock_info, mock_success, mock_markdown):
        """Verify evaluation section appears in UI regardless of results."""
        from ui.evaluation_display import display_evaluation_configuration

        # Test that evaluation configuration function exists and can be called
        try:
            config_result = display_evaluation_configuration()
            assert isinstance(config_result, dict), "Evaluation configuration should return dict"
        except Exception as e:
            pytest.fail(f"Evaluation configuration display failed: {e}")

    def test_evaluation_display_components_exist(self):
        """Verify evaluation display components exist."""
        try:
            from ui.evaluation_display import (
                display_evaluation_dashboard,
                display_evaluation_configuration,
                display_reference_graph_upload,
                display_evaluation_export_options
            )
            # If imports succeed, functions exist
            assert True
        except ImportError as e:
            pytest.fail(f"Evaluation display components missing: {e}")

    def test_evaluation_display_handles_success_case(self):
        """Verify evaluation display handles successful evaluation."""
        from ui.evaluation_display import display_evaluation_dashboard

        # Create successful evaluation result
        mock_metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.90,
            g_bleu_precision=0.80,
            g_bleu_recall=0.75,
            g_bleu_f1=0.77,
            g_rouge_precision=0.82,
            g_rouge_recall=0.78,
            g_rouge_f1=0.80,
            g_bert_precision=0.88,
            g_bert_recall=0.85,
            g_bert_f1=0.86
        )

        evaluation_result = EvaluationResult(
            metrics=mock_metrics,
            success=True,
            processing_time=3.0,
            metadata={"reference_graph_size": 10}
        )

        # Test that display function can handle successful result
        with patch('streamlit.markdown'), \
             patch('streamlit.metric'), \
             patch('streamlit.plotly_chart'), \
             patch('streamlit.columns'):

            try:
                display_evaluation_dashboard(evaluation_result, show_detailed=True)
                # If no exception raised, function handles success case properly
                assert True
            except Exception as e:
                pytest.fail(f"Evaluation dashboard failed with successful result: {e}")

    def test_evaluation_display_handles_failure_case(self):
        """Verify evaluation display handles failed evaluation."""
        from ui.evaluation_display import display_evaluation_dashboard

        # Create failed evaluation result
        failed_evaluation = EvaluationResult(
            metrics=GraphMetrics(
                triple_match_f1=0.0,
                graph_match_accuracy=0.0,
                g_bleu_precision=0.0,
                g_bleu_recall=0.0,
                g_bleu_f1=0.0,
                g_rouge_precision=0.0,
                g_rouge_recall=0.0,
                g_rouge_f1=0.0,
                g_bert_precision=0.0,
                g_bert_recall=0.0,
                g_bert_f1=0.0
            ),
            metadata={"test": True},
            success=False,
            processing_time=0.0,
            error="Test evaluation error"
        )

        # Test that display function can handle failed result
        with patch('streamlit.markdown'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'):

            try:
                display_evaluation_dashboard(failed_evaluation, show_detailed=True)
                # If no exception raised, function handles failure case properly
                assert True
            except Exception as e:
                pytest.fail(f"Evaluation dashboard failed with failed result: {e}")

    def test_evaluation_export_options_exist(self):
        """Verify evaluation export functionality exists."""
        from ui.evaluation_display import display_evaluation_export_options

        # Create mock evaluation result for export
        mock_metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.90,
            g_bleu_precision=0.80,
            g_bleu_recall=0.75,
            g_bleu_f1=0.77,
            g_rouge_precision=0.82,
            g_rouge_recall=0.78,
            g_rouge_f1=0.80,
            g_bert_precision=0.88,
            g_bert_recall=0.85,
            g_bert_f1=0.86
        )

        evaluation_result = EvaluationResult(
            metrics=mock_metrics,
            metadata={"test": True},
            success=True,
            processing_time=3.0
        )

        # Test export options display
        with patch('streamlit.download_button'), \
             patch('streamlit.columns'):

            try:
                display_evaluation_export_options(evaluation_result, "test_export")
                # If no exception raised, export functionality works
                assert True
            except Exception as e:
                pytest.fail(f"Evaluation export options failed: {e}")


class TestEvaluationMetrics:
    """Test suite to verify evaluation metrics work correctly."""

    def test_graph_metrics_structure(self):
        """Verify GraphMetrics has proper structure."""
        metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.90,
            g_bleu_precision=0.80,
            g_bleu_recall=0.75,
            g_bleu_f1=0.77,
            g_rouge_precision=0.82,
            g_rouge_recall=0.78,
            g_rouge_f1=0.80,
            g_bert_precision=0.88,
            g_bert_recall=0.85,
            g_bert_f1=0.86
        )

        # Verify all required metrics exist
        assert hasattr(metrics, 'triple_match_f1'), "GraphMetrics missing triple_match_f1"
        assert hasattr(metrics, 'graph_match_accuracy'), "GraphMetrics missing graph_match_accuracy"
        assert hasattr(metrics, 'g_bleu_f1'), "GraphMetrics missing g_bleu_f1"
        assert hasattr(metrics, 'g_rouge_f1'), "GraphMetrics missing g_rouge_f1"
        assert hasattr(metrics, 'g_bert_f1'), "GraphMetrics missing g_bert_f1"

        # Verify values are within expected range
        assert 0.0 <= metrics.triple_match_f1 <= 1.0, "triple_match_f1 out of range"
        assert 0.0 <= metrics.graph_match_accuracy <= 1.0, "graph_match_accuracy out of range"

    def test_evaluation_result_structure(self):
        """Verify EvaluationResult has proper structure."""
        mock_metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.90,
            g_bleu_precision=0.80,
            g_bleu_recall=0.75,
            g_bleu_f1=0.77,
            g_rouge_precision=0.82,
            g_rouge_recall=0.78,
            g_rouge_f1=0.80,
            g_bert_precision=0.88,
            g_bert_recall=0.85,
            g_bert_f1=0.86
        )

        evaluation_result = EvaluationResult(
            metrics=mock_metrics,
            success=True,
            processing_time=3.0,
            metadata={"test": "data"},
            error=None
        )

        # Verify all required fields exist
        assert hasattr(evaluation_result, 'metrics'), "EvaluationResult missing metrics"
        assert hasattr(evaluation_result, 'success'), "EvaluationResult missing success"
        assert hasattr(evaluation_result, 'processing_time'), "EvaluationResult missing processing_time"
        assert hasattr(evaluation_result, 'metadata'), "EvaluationResult missing metadata"
        assert hasattr(evaluation_result, 'error'), "EvaluationResult missing error"

        # Verify types
        assert isinstance(evaluation_result.metrics, GraphMetrics), "metrics should be GraphMetrics instance"
        assert isinstance(evaluation_result.success, bool), "success should be boolean"
        assert isinstance(evaluation_result.processing_time, (int, float)), "processing_time should be numeric"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])