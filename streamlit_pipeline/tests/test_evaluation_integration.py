"""
Integration tests for the evaluation pipeline system.

This module tests the complete integration of the evaluation system with the
existing pipeline, including reference graph management, session state integration,
and end-to-end evaluation workflow.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from pathlib import Path

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.models import Triple, GraphMetrics, EvaluationResult
from core.pipeline import PipelineOrchestrator, PipelineResult, run_full_pipeline
from core.config import get_evaluation_config
from utils.reference_graph_manager import ReferenceGraphManager, upload_reference_graph

# Skip SessionStateManager import due to relative import issues in test environment
try:
    from utils.session_state import SessionStateManager
    SESSION_STATE_AVAILABLE = True
except ImportError:
    SESSION_STATE_AVAILABLE = False


class TestEvaluationPipelineIntegration:
    """Test complete evaluation pipeline integration."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_text = "巴黎是法国的首都。法国位于欧洲。"

        # Sample graphs for testing
        self.predicted_triples = [
            Triple("巴黎", "首都", "法国"),
            Triple("法国", "位于", "欧洲")
        ]

        self.reference_triples = [
            Triple("巴黎", "首都", "法国"),
            Triple("柏林", "首都", "德国"),
            Triple("法国", "位于", "欧洲")
        ]

        # Sample evaluation configuration
        self.evaluation_config = {
            'enable_evaluation': True,
            'enable_ged': False,
            'enable_bert_score': False,  # Disable for testing
            'max_evaluation_time': 10.0
        }

    def test_evaluation_config_loading(self):
        """Test evaluation configuration loading."""
        config = get_evaluation_config()

        assert isinstance(config, dict)
        assert 'enable_evaluation' in config
        assert 'enable_ged' in config
        assert 'enable_bert_score' in config
        assert 'max_evaluation_time' in config
        assert 'supported_formats' in config

    @patch('streamlit_pipeline.core.pipeline.extract_entities')
    @patch('streamlit_pipeline.core.pipeline.generate_triples')
    @patch('streamlit_pipeline.core.pipeline.judge_triples')
    @patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator')
    def test_pipeline_with_evaluation_enabled(self, mock_evaluator_class, mock_judge, mock_triples, mock_entities):
        """Test pipeline execution with evaluation enabled."""
        # Setup mocks for pipeline stages
        mock_entities.return_value = Mock(
            entities=["巴黎", "法国", "欧洲"],
            denoised_text=self.test_text,
            success=True,
            processing_time=0.5
        )

        mock_triples.return_value = Mock(
            triples=self.predicted_triples,
            metadata={},
            success=True,
            processing_time=1.0
        )

        mock_judge.return_value = Mock(
            judgments=[True, True],
            success=True,
            processing_time=0.8
        )

        # Setup evaluation mock
        mock_evaluator = Mock()
        mock_metrics = GraphMetrics(
            triple_match_f1=0.75,
            graph_match_accuracy=0.5,
            g_bleu_precision=0.8, g_bleu_recall=0.7, g_bleu_f1=0.75,
            g_rouge_precision=0.85, g_rouge_recall=0.72, g_rouge_f1=0.78,
            g_bert_precision=0.0, g_bert_recall=0.0, g_bert_f1=0.0
        )
        mock_evaluation_result = EvaluationResult(
            metrics=mock_metrics,
            metadata={},
            success=True,
            processing_time=0.3
        )
        mock_evaluator.evaluate_graph.return_value = mock_evaluation_result
        mock_evaluator_class.return_value = mock_evaluator

        # Run pipeline with evaluation
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run_pipeline(
            input_text=self.test_text,
            evaluation_config=self.evaluation_config,
            reference_graph=self.reference_triples
        )

        # Verify pipeline completed successfully
        assert result.success == True
        assert result.evaluation_enabled == True
        assert result.evaluation_result is not None
        assert result.evaluation_result.success == True
        assert result.evaluation_result.metrics.triple_match_f1 == 0.75

        # Verify evaluator was called correctly
        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate_graph.assert_called_once()

    @patch('streamlit_pipeline.core.pipeline.extract_entities')
    @patch('streamlit_pipeline.core.pipeline.generate_triples')
    @patch('streamlit_pipeline.core.pipeline.judge_triples')
    def test_pipeline_with_evaluation_disabled(self, mock_judge, mock_triples, mock_entities):
        """Test pipeline execution with evaluation disabled."""
        # Setup mocks
        mock_entities.return_value = Mock(
            entities=["巴黎", "法国"],
            denoised_text=self.test_text,
            success=True,
            processing_time=0.5
        )

        mock_triples.return_value = Mock(
            triples=self.predicted_triples,
            metadata={},
            success=True,
            processing_time=1.0
        )

        mock_judge.return_value = Mock(
            judgments=[True, True],
            success=True,
            processing_time=0.8
        )

        # Run pipeline without evaluation
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run_pipeline(
            input_text=self.test_text,
            evaluation_config={'enable_evaluation': False}
        )

        # Verify evaluation was skipped
        assert result.success == True
        assert result.evaluation_enabled == False
        assert result.evaluation_result is None

    def test_pipeline_evaluation_graceful_failure(self):
        """Test that evaluation failures don't break the pipeline."""
        with patch('streamlit_pipeline.core.pipeline.extract_entities') as mock_entities, \
             patch('streamlit_pipeline.core.pipeline.generate_triples') as mock_triples, \
             patch('streamlit_pipeline.core.pipeline.judge_triples') as mock_judge, \
             patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator') as mock_evaluator_class:

            # Setup successful pipeline mocks
            mock_entities.return_value = Mock(
                entities=["巴黎"], denoised_text=self.test_text,
                success=True, processing_time=0.5
            )
            mock_triples.return_value = Mock(
                triples=[self.predicted_triples[0]], metadata={},
                success=True, processing_time=1.0
            )
            mock_judge.return_value = Mock(
                judgments=[True],
                success=True, processing_time=0.8
            )

            # Make evaluation fail
            mock_evaluator_class.side_effect = Exception("Evaluation failed")

            # Run pipeline with failing evaluation
            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_pipeline(
                input_text=self.test_text,
                evaluation_config=self.evaluation_config,
                reference_graph=self.reference_triples
            )

            # Verify pipeline still succeeds despite evaluation failure
            assert result.success == True
            assert result.evaluation_enabled == True
            assert result.evaluation_result is not None
            assert result.evaluation_result.success == False
            assert "Evaluation stage failed" in result.evaluation_result.error


class TestReferenceGraphManager:
    """Test reference graph management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ReferenceGraphManager(temp_dir=self.temp_dir)

    def test_json_format_parsing(self):
        """Test parsing JSON format reference graphs."""
        json_content = json.dumps([
            ["巴黎", "首都", "法国"],
            ["法国", "位于", "欧洲"]
        ])

        # Mock uploaded file
        mock_file = Mock()
        mock_file.read.return_value = json_content.encode('utf-8')
        mock_file.name = "test.json"

        success, triples, error = self.manager.upload_reference_graph(mock_file, "json")

        assert success == True
        assert error is None
        assert len(triples) == 2
        assert triples[0].subject == "巴黎"
        assert triples[0].predicate == "首都"
        assert triples[0].object == "法国"

    def test_csv_format_parsing(self):
        """Test parsing CSV format reference graphs."""
        csv_content = "subject,predicate,object\n巴黎,首都,法国\n法国,位于,欧洲"

        mock_file = Mock()
        mock_file.read.return_value = csv_content.encode('utf-8')
        mock_file.name = "test.csv"

        success, triples, error = self.manager.upload_reference_graph(mock_file, "csv")

        assert success == True
        assert error is None
        assert len(triples) == 2

    def test_txt_format_parsing(self):
        """Test parsing text format reference graphs."""
        txt_content = "巴黎\t首都\t法国\n法国\t位于\t欧洲"

        mock_file = Mock()
        mock_file.read.return_value = txt_content.encode('utf-8')
        mock_file.name = "test.txt"

        success, triples, error = self.manager.upload_reference_graph(mock_file, "txt")

        assert success == True
        assert error is None
        assert len(triples) == 2

    def test_auto_format_detection(self):
        """Test automatic format detection."""
        json_content = json.dumps([["A", "rel", "B"]])

        mock_file = Mock()
        mock_file.read.return_value = json_content.encode('utf-8')
        mock_file.name = "unknown_format.data"

        success, triples, error = self.manager.upload_reference_graph(mock_file, "auto")

        assert success == True
        assert len(triples) == 1

    def test_file_size_validation(self):
        """Test file size validation."""
        large_content = "A,B,C\n" * 100000  # Large content

        mock_file = Mock()
        mock_file.read.return_value = large_content.encode('utf-8')
        mock_file.name = "large.csv"

        success, triples, error = self.manager.upload_reference_graph(mock_file, "csv")

        assert success == False
        assert "too large" in error

    def test_graph_statistics(self):
        """Test graph statistics generation."""
        triples = [
            Triple("A", "rel1", "B"),
            Triple("B", "rel2", "C"),
            Triple("A", "rel1", "C")
        ]

        stats = self.manager.get_graph_statistics(triples)

        assert stats["size"] == 3
        assert stats["unique_subjects"] == 2
        assert stats["unique_predicates"] == 2
        assert stats["unique_objects"] == 2
        assert len(stats["most_common_predicates"]) > 0

    def test_graph_validation(self):
        """Test graph validation functionality."""
        # Valid triples
        valid_triples = [Triple("A", "rel", "B")]
        is_valid, error = self.manager._validate_triples(valid_triples)
        assert is_valid == True
        assert error is None

        # Invalid triples (empty field)
        invalid_triples = [Triple("", "rel", "B")]
        is_valid, error = self.manager._validate_triples(invalid_triples)
        assert is_valid == False
        assert error is not None


class TestSessionStateIntegration:
    """Test session state integration for evaluation system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock streamlit session state
        self.mock_session_state = {}
        self.patcher = patch('streamlit_pipeline.utils.session_state.st')
        self.mock_st = self.patcher.start()
        self.mock_st.session_state = self.mock_session_state

    def teardown_method(self):
        """Clean up after tests."""
        self.patcher.stop()

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_session_state_initialization(self):
        """Test session state initialization with evaluation keys."""
        manager = SessionStateManager()

        # Check that evaluation keys are initialized
        assert 'evaluation_config' in self.mock_session_state
        assert 'reference_graph' in self.mock_session_state
        assert 'evaluation_enabled' in self.mock_session_state
        assert 'evaluation_results' in self.mock_session_state

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_evaluation_config_management(self):
        """Test evaluation configuration management."""
        manager = SessionStateManager()

        config = {'enable_evaluation': True, 'enable_ged': False}
        manager.set_evaluation_config(config)

        stored_config = manager.get_evaluation_config()
        assert stored_config == config
        assert self.mock_session_state['evaluation_enabled'] == True

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_reference_graph_management(self):
        """Test reference graph management in session state."""
        manager = SessionStateManager()

        triples = [Triple("A", "rel", "B")]
        graph_info = {"size": 1, "format": "test"}

        manager.set_reference_graph(triples, graph_info)

        stored_triples = manager.get_reference_graph()
        stored_info = manager.get_reference_graph_info()

        assert stored_triples == triples
        assert stored_info == graph_info

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_evaluation_results_management(self):
        """Test evaluation results management."""
        manager = SessionStateManager()

        # Create mock evaluation result
        metrics = GraphMetrics(
            triple_match_f1=0.8, graph_match_accuracy=0.7,
            g_bleu_precision=0.75, g_bleu_recall=0.7, g_bleu_f1=0.72,
            g_rouge_precision=0.8, g_rouge_recall=0.75, g_rouge_f1=0.77,
            g_bert_precision=0.85, g_bert_recall=0.8, g_bert_f1=0.82
        )
        eval_result = EvaluationResult(
            metrics=metrics, metadata={}, success=True, processing_time=1.0
        )

        manager.add_evaluation_result(eval_result)

        results = manager.get_evaluation_results()
        assert len(results) == 1
        assert results[0] == eval_result

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_evaluation_readiness_check(self):
        """Test evaluation readiness checking."""
        manager = SessionStateManager()

        # Initially not ready
        assert manager.is_evaluation_ready() == False

        # Enable evaluation but no reference graph
        manager.set_evaluation_config({'enable_evaluation': True})
        assert manager.is_evaluation_ready() == False

        # Add reference graph
        manager.set_reference_graph([Triple("A", "rel", "B")], {})
        assert manager.is_evaluation_ready() == True

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_evaluation_data_cleanup(self):
        """Test evaluation data cleanup."""
        manager = SessionStateManager()

        # Set some evaluation data
        manager.set_evaluation_config({'enable_evaluation': True})
        manager.set_reference_graph([Triple("A", "rel", "B")], {"size": 1})
        manager.add_evaluation_result(Mock())

        # Clear all evaluation data
        manager.clear_evaluation_data()

        # Verify cleanup
        assert manager.get_evaluation_config() == {}
        assert manager.get_reference_graph() is None
        assert manager.get_reference_graph_info() == {}
        assert len(manager.get_evaluation_results()) == 0
        assert self.mock_session_state['evaluation_enabled'] == False


class TestEndToEndEvaluationWorkflow:
    """Test complete end-to-end evaluation workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    @pytest.mark.skipif(not SESSION_STATE_AVAILABLE, reason="SessionStateManager not available")
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from upload to results."""
        # Step 1: Upload reference graph
        json_content = json.dumps([
            ["Paris", "capital_of", "France"],
            ["France", "located_in", "Europe"]
        ])

        mock_file = Mock()
        mock_file.read.return_value = json_content.encode('utf-8')
        mock_file.name = "reference.json"

        success, reference_triples, error = upload_reference_graph(mock_file, "json")
        assert success == True
        assert len(reference_triples) == 2

        # Step 2: Mock session state for integration
        with patch('streamlit_pipeline.utils.session_state.st') as mock_st:
            mock_session_state = {}
            mock_st.session_state = mock_session_state

            session_manager = SessionStateManager()

            # Step 3: Configure evaluation
            eval_config = {
                'enable_evaluation': True,
                'enable_ged': False,
                'enable_bert_score': False,
                'max_evaluation_time': 5.0
            }
            session_manager.set_evaluation_config(eval_config)
            session_manager.set_reference_graph(reference_triples, {"size": 2})

            # Step 4: Verify evaluation readiness
            assert session_manager.is_evaluation_ready() == True

            # Step 5: Mock pipeline execution with evaluation
            with patch('streamlit_pipeline.core.pipeline.extract_entities') as mock_entities, \
                 patch('streamlit_pipeline.core.pipeline.generate_triples') as mock_triples, \
                 patch('streamlit_pipeline.core.pipeline.judge_triples') as mock_judge, \
                 patch('streamlit_pipeline.eval.graph_evaluator.GraphEvaluator') as mock_evaluator_class:

                # Setup pipeline mocks
                mock_entities.return_value = Mock(
                    entities=["Paris", "France"],
                    denoised_text="Paris is the capital of France",
                    success=True, processing_time=0.5
                )

                predicted_triples = [Triple("Paris", "capital_of", "France")]
                mock_triples.return_value = Mock(
                    triples=predicted_triples, metadata={},
                    success=True, processing_time=1.0
                )

                mock_judge.return_value = Mock(
                    judgments=[True],
                    success=True, processing_time=0.8
                )

                # Setup evaluation mock
                mock_evaluator = Mock()
                mock_metrics = GraphMetrics(
                    triple_match_f1=0.5, graph_match_accuracy=0.5,
                    g_bleu_precision=1.0, g_bleu_recall=0.5, g_bleu_f1=0.67,
                    g_rouge_precision=1.0, g_rouge_recall=0.5, g_rouge_f1=0.67,
                    g_bert_precision=0.0, g_bert_recall=0.0, g_bert_f1=0.0
                )
                mock_evaluation_result = EvaluationResult(
                    metrics=mock_metrics, metadata={}, success=True, processing_time=0.2
                )
                mock_evaluator.evaluate_graph.return_value = mock_evaluation_result
                mock_evaluator_class.return_value = mock_evaluator

                # Run complete pipeline
                result = run_full_pipeline(
                    input_text="Paris is the capital of France",
                    evaluation_config=eval_config,
                    reference_graph=reference_triples
                )

                # Step 6: Verify complete workflow
                assert result.success == True
                assert result.evaluation_enabled == True
                assert result.evaluation_result is not None
                assert result.evaluation_result.success == True
                assert result.evaluation_result.metrics.triple_match_f1 == 0.5

                # Step 7: Store evaluation result in session state
                session_manager.add_evaluation_result(result.evaluation_result)
                stored_results = session_manager.get_evaluation_results()
                assert len(stored_results) == 1
                assert stored_results[0].metrics.triple_match_f1 == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])