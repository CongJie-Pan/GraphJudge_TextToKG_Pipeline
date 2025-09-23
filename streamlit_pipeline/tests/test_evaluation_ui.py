"""
Unit tests for evaluation UI components.

Tests the functionality of evaluation display components including
metrics dashboard, comparative analysis, export functionality, and
UI configuration options.
"""

import pytest
import unittest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Import models and UI components
try:
    from streamlit_pipeline.core.models import GraphMetrics, EvaluationResult, Triple
    from streamlit_pipeline.ui.evaluation_display import (
        _get_quality_indicator, _create_comparison_table,
        display_evaluation_configuration, display_reference_graph_upload
    )
except ImportError:
    # Fallback for testing from different directory
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import GraphMetrics, EvaluationResult, Triple
    from ui.evaluation_display import (
        _get_quality_indicator, _create_comparison_table,
        display_evaluation_configuration, display_reference_graph_upload
    )


class TestEvaluationUIComponents(unittest.TestCase):
    """Test evaluation UI components functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample metrics
        self.sample_metrics = GraphMetrics(
            triple_match_f1=0.85,
            graph_match_accuracy=0.78,
            g_bleu_precision=0.72,
            g_bleu_recall=0.68,
            g_bleu_f1=0.70,
            g_rouge_precision=0.74,
            g_rouge_recall=0.71,
            g_rouge_f1=0.725,
            g_bert_precision=0.88,
            g_bert_recall=0.82,
            g_bert_f1=0.85,
            graph_edit_distance=0.15
        )

        # Create sample evaluation result
        self.sample_evaluation_result = EvaluationResult(
            metrics=self.sample_metrics,
            success=True,
            processing_time=2.5,
            metadata={
                'evaluation_time': datetime.now().isoformat(),
                'config': {
                    'enable_bert_score': True,
                    'enable_ged': True
                }
            },
            reference_graph_info={
                'size': 100,
                'unique_subjects': 25,
                'unique_predicates': 15,
                'unique_objects': 35
            },
            predicted_graph_info={
                'size': 95,
                'unique_subjects': 23,
                'unique_predicates': 14,
                'unique_objects': 33
            }
        )

        # Create sample triples
        self.sample_triples = [
            Triple("主角", "居住在", "大观园"),
            Triple("林黛玉", "是", "贾宝玉的表妹"),
            Triple("贾宝玉", "喜欢", "诗词")
        ]

    def test_quality_indicator_function(self):
        """Test quality indicator classification."""
        # Test excellent quality
        self.assertEqual(_get_quality_indicator(0.9), "Excellent")
        self.assertEqual(_get_quality_indicator(0.85), "Excellent")

        # Test good quality
        self.assertEqual(_get_quality_indicator(0.75), "Good")
        self.assertEqual(_get_quality_indicator(0.65), "Good")

        # Test fair quality
        self.assertEqual(_get_quality_indicator(0.55), "Fair")
        self.assertEqual(_get_quality_indicator(0.45), "Fair")

        # Test poor quality
        self.assertEqual(_get_quality_indicator(0.35), "Poor")
        self.assertEqual(_get_quality_indicator(0.15), "Poor")

        # Test boundary conditions
        self.assertEqual(_get_quality_indicator(0.8), "Excellent")
        self.assertEqual(_get_quality_indicator(0.6), "Good")
        self.assertEqual(_get_quality_indicator(0.4), "Fair")

    def test_evaluation_result_data_structure(self):
        """Test evaluation result data structure integrity."""
        # Test successful evaluation result
        self.assertTrue(self.sample_evaluation_result.success)
        self.assertEqual(self.sample_evaluation_result.processing_time, 2.5)
        self.assertIsNone(self.sample_evaluation_result.error)

        # Test metrics are properly set
        metrics = self.sample_evaluation_result.metrics
        self.assertEqual(metrics.triple_match_f1, 0.85)
        self.assertEqual(metrics.g_bert_f1, 0.85)
        self.assertEqual(metrics.graph_edit_distance, 0.15)

        # Test metadata structure
        self.assertIn('evaluation_time', self.sample_evaluation_result.metadata)
        self.assertIn('config', self.sample_evaluation_result.metadata)

    def test_failed_evaluation_result(self):
        """Test handling of failed evaluation results."""
        failed_result = EvaluationResult(
            metrics=GraphMetrics(
                triple_match_f1=0.0, graph_match_accuracy=0.0,
                g_bleu_precision=0.0, g_bleu_recall=0.0, g_bleu_f1=0.0,
                g_rouge_precision=0.0, g_rouge_recall=0.0, g_rouge_f1=0.0,
                g_bert_precision=0.0, g_bert_recall=0.0, g_bert_f1=0.0
            ),
            success=False,
            processing_time=0.5,
            error="Reference graph not available",
            metadata={}
        )

        self.assertFalse(failed_result.success)
        self.assertEqual(failed_result.error, "Reference graph not available")
        self.assertEqual(failed_result.processing_time, 0.5)

    @patch('streamlit_pipeline.ui.evaluation_display.st')
    def test_evaluation_configuration_display(self, mock_st):
        """Test evaluation configuration UI component."""
        # Mock streamlit components
        mock_st.checkbox.side_effect = [True, True, False]  # enable_evaluation, enable_bert_score, enable_ged
        mock_st.slider.return_value = 45.0
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()

        # Call the function
        config = display_evaluation_configuration()

        # Verify configuration returned
        expected_config = {
            'enable_evaluation': True,
            'enable_bert_score': True,
            'enable_ged': False,
            'max_evaluation_time': 45.0
        }

        self.assertEqual(config, expected_config)

        # Verify UI calls
        mock_st.markdown.assert_called()
        self.assertEqual(mock_st.checkbox.call_count, 3)
        mock_st.slider.assert_called_once()

    @patch('streamlit_pipeline.ui.evaluation_display.st')
    def test_evaluation_configuration_disabled(self, mock_st):
        """Test evaluation configuration when disabled."""
        # Mock evaluation disabled
        mock_st.checkbox.return_value = False

        # Call the function
        config = display_evaluation_configuration()

        # Verify configuration returned
        expected_config = {'enable_evaluation': False}
        self.assertEqual(config, expected_config)

        # Verify only one checkbox call (enable_evaluation)
        self.assertEqual(mock_st.checkbox.call_count, 1)

    def test_comparison_table_creation(self):
        """Test creation of comparison table data."""
        # Create multiple evaluation results for comparison
        results = [
            ("Run 1", self.sample_evaluation_result),
            ("Run 2", EvaluationResult(
                metrics=GraphMetrics(
                    triple_match_f1=0.75, graph_match_accuracy=0.68,
                    g_bleu_precision=0.62, g_bleu_recall=0.58, g_bleu_f1=0.60,
                    g_rouge_precision=0.64, g_rouge_recall=0.61, g_rouge_f1=0.625,
                    g_bert_precision=0.78, g_bert_recall=0.72, g_bert_f1=0.75
                ),
                success=True, processing_time=3.2, metadata={}
            ))
        ]

        # Mock streamlit for table creation
        with patch('streamlit_pipeline.ui.evaluation_display.st') as mock_st:
            mock_st.dataframe = MagicMock()

            # Call the function
            _create_comparison_table(results)

            # Verify dataframe was called
            mock_st.dataframe.assert_called_once()

    def test_comparison_table_with_errors(self):
        """Test comparison table handling results with errors."""
        # Create results with one successful and one failed
        results = [
            ("Successful Run", self.sample_evaluation_result),
            ("Failed Run", EvaluationResult(
                metrics=GraphMetrics(
                    triple_match_f1=0.0, graph_match_accuracy=0.0,
                    g_bleu_precision=0.0, g_bleu_recall=0.0, g_bleu_f1=0.0,
                    g_rouge_precision=0.0, g_rouge_recall=0.0, g_rouge_f1=0.0,
                    g_bert_precision=0.0, g_bert_recall=0.0, g_bert_f1=0.0
                ),
                success=False, processing_time=1.0, error="Test error", metadata={}
            ))
        ]

        # Mock streamlit for table creation
        with patch('streamlit_pipeline.ui.evaluation_display.st') as mock_st:
            mock_st.dataframe = MagicMock()

            # Call the function
            _create_comparison_table(results)

            # Verify dataframe was called
            mock_st.dataframe.assert_called_once()

            # Get the dataframe call arguments
            call_args = mock_st.dataframe.call_args[0][0]

            # Verify error handling in table
            self.assertIn("Error", call_args.iloc[1].values)  # Failed run should have "Error"

    @patch('streamlit_pipeline.utils.reference_graph_manager.upload_reference_graph')
    @patch('streamlit_pipeline.ui.evaluation_display.st')
    def test_reference_graph_upload_success(self, mock_st, mock_upload):
        """Test successful reference graph upload."""
        # Mock successful upload
        mock_upload.return_value = (True, self.sample_triples, None)
        mock_st.file_uploader.return_value = MagicMock()  # Simulate file upload

        # Mock columns to return proper values
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]

        # Mock expander context manager
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        # Mock stats function
        with patch('streamlit_pipeline.utils.reference_graph_manager.get_reference_graph_stats') as mock_stats:
            mock_stats.return_value = {
                'size': 3,
                'unique_subjects': 3,
                'unique_predicates': 3,
                'unique_objects': 3,
                'subject_examples': ['主角', '林黛玉', '贾宝玉'],
                'predicate_examples': ['居住在', '是', '喜欢']
            }

            # Call the function
            result = display_reference_graph_upload()

            # Verify successful upload
            self.assertEqual(result, self.sample_triples)
            mock_st.success.assert_called()

    @patch('streamlit_pipeline.utils.reference_graph_manager.upload_reference_graph')
    @patch('streamlit_pipeline.ui.evaluation_display.st')
    def test_reference_graph_upload_failure(self, mock_st, mock_upload):
        """Test failed reference graph upload."""
        # Mock failed upload
        mock_upload.return_value = (False, None, "Invalid file format")
        mock_st.file_uploader.return_value = MagicMock()  # Simulate file upload

        # Call the function
        result = display_reference_graph_upload()

        # Verify failed upload
        self.assertIsNone(result)
        mock_st.error.assert_called()

    @patch('streamlit_pipeline.ui.evaluation_display.st')
    def test_reference_graph_upload_no_file(self, mock_st):
        """Test reference graph upload with no file selected."""
        # Mock no file uploaded
        mock_st.file_uploader.return_value = None

        # Call the function
        result = display_reference_graph_upload()

        # Verify no processing occurred
        self.assertIsNone(result)

    def test_metrics_data_completeness(self):
        """Test that all required metrics are present."""
        metrics = self.sample_metrics

        # Test exact matching metrics
        self.assertIsInstance(metrics.triple_match_f1, float)
        self.assertIsInstance(metrics.graph_match_accuracy, float)

        # Test text similarity metrics
        self.assertIsInstance(metrics.g_bleu_precision, float)
        self.assertIsInstance(metrics.g_bleu_recall, float)
        self.assertIsInstance(metrics.g_bleu_f1, float)
        self.assertIsInstance(metrics.g_rouge_precision, float)
        self.assertIsInstance(metrics.g_rouge_recall, float)
        self.assertIsInstance(metrics.g_rouge_f1, float)

        # Test semantic similarity metrics
        self.assertIsInstance(metrics.g_bert_precision, float)
        self.assertIsInstance(metrics.g_bert_recall, float)
        self.assertIsInstance(metrics.g_bert_f1, float)

        # Test structural similarity (optional)
        self.assertIsInstance(metrics.graph_edit_distance, (float, type(None)))

    def test_evaluation_export_data_structure(self):
        """Test the data structure prepared for export."""
        # Test that evaluation result can be serialized for export
        result = self.sample_evaluation_result

        # Verify all required fields exist
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.success)
        self.assertIsNotNone(result.processing_time)
        self.assertIsNotNone(result.metadata)

        # Test metadata structure for export
        metadata = result.metadata
        self.assertIn('evaluation_time', metadata)
        self.assertIn('config', metadata)

        # Test graph info for export
        self.assertIsNotNone(result.reference_graph_info)
        self.assertIsNotNone(result.predicted_graph_info)


if __name__ == '__main__':
    unittest.main()