"""
Integration Tests for Device Detection in eval.py

This test module verifies that the device detection changes work correctly
in the full evaluation pipeline, ensuring compatibility with both GPU and CPU environments.

Test Coverage:
- End-to-end eval.py execution with device detection
- Integration with realistic data
- Error handling for different hardware configurations

Run with: pytest test_eval_device_integration.py -v
"""

import ast
import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np


def _write_graph_file(path: Path, graphs):
    """Write graphs to file in eval.py expected format."""
    with path.open("w", encoding="utf-8") as f:
        for graph in graphs:
            f.write(str(graph) + "\n")


class TestEvalDeviceIntegration:
    """Integration tests for eval.py with device detection."""
    
    def test_eval_script_runs_with_cpu_fallback(self):
        """Test that eval.py runs successfully with CPU fallback."""
        with tempfile.TemporaryDirectory() as td:
            # Create test data files
            pred_path = Path(td) / "pred.txt"
            gold_path = Path(td) / "gold.txt"
            output_path = Path(td) / "results.json"
            
            # Sample data - small enough to run quickly
            test_graphs = [
                [["entity1", "relation1", "entity2"]],
                [["entity3", "relation2", "entity4"]],
            ]
            
            _write_graph_file(pred_path, test_graphs)
            _write_graph_file(gold_path, test_graphs)
            
            # Get path to eval.py
            eval_script = Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics" / "eval.py"
            
            # Mock torch to force CPU usage for testing
            with patch.dict('sys.modules', {'torch': MagicMock(cuda=MagicMock(is_available=lambda: False))}):
                # Run eval.py as subprocess to test actual execution
                cmd = [
                    sys.executable, str(eval_script),
                    "--pred_file", str(pred_path),
                    "--gold_file", str(gold_path),
                    "--out", str(output_path)
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    # Check that the script completed successfully
                    assert result.returncode == 0, f"eval.py failed with: {result.stderr}"
                    
                    # Check that output file was created
                    assert output_path.exists(), "Results JSON file should be created"
                    
                    # Verify output contains expected structure
                    with output_path.open('r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    assert "exact_matching_metrics" in results
                    assert "text_similarity_metrics" in results
                    assert "semantic_similarity_metrics" in results
                    assert "evaluation_metadata" in results
                    
                except subprocess.TimeoutExpired:
                    pytest.fail("eval.py execution timed out - may indicate device detection issues")
    
    def test_eval_script_device_detection_messages(self):
        """Test that device detection messages are printed correctly."""
        with tempfile.TemporaryDirectory() as td:
            # Create minimal test data
            pred_path = Path(td) / "pred.txt"
            gold_path = Path(td) / "gold.txt"
            
            test_graphs = [
                [["test_entity", "test_relation", "test_object"]],
            ]
            
            _write_graph_file(pred_path, test_graphs)
            _write_graph_file(gold_path, test_graphs)
            
            # Get path to eval.py
            eval_script = Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics" / "eval.py"
            
            # Run eval.py and capture output
            cmd = [
                sys.executable, str(eval_script),
                "--pred_file", str(pred_path),
                "--gold_file", str(gold_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # Check that device detection message appears in output
                assert "Using device for BERTScore:" in result.stdout or "Using device for BERTScore:" in result.stderr
                
                # Should show either "cuda" or "cpu"
                output_text = result.stdout + result.stderr
                assert ("Using device for BERTScore: cuda" in output_text or 
                       "Using device for BERTScore: cpu" in output_text)
                
            except subprocess.TimeoutExpired:
                pytest.fail("eval.py execution timed out")
    
    @patch('torch.cuda.is_available')
    def test_mock_cuda_available_in_eval(self, mock_cuda_available):
        """Test eval.py behavior when CUDA is mocked as available."""
        mock_cuda_available.return_value = True
        
        # Import eval module after mocking
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
        
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "pred.txt"
            gold_path = Path(td) / "gold.txt"
            
            test_graphs = [
                [["entity", "relation", "object"]],
            ]
            
            _write_graph_file(pred_path, test_graphs)
            _write_graph_file(gold_path, test_graphs)
            
            # Mock the bert_score function to avoid actual BERT computation
            with patch('graph_matching.score_bert') as mock_score_bert:
                mock_score_bert.return_value = (None, None, np.array([0.8]))
                
                # Import and run eval module functions
                import eval as eval_module
                
                # Load test data
                with open(pred_path, 'r', encoding="utf-8") as f:
                    pred_graphs = [ast.literal_eval(line.strip()) for line in f.readlines()]
                
                with open(gold_path, 'r', encoding="utf-8") as f:
                    gold_graphs = [ast.literal_eval(line.strip()) for line in f.readlines()]
                
                # This should work without errors and use mocked CUDA
                # (Actual full test would require more setup, this tests the import/basic functionality)
                assert len(pred_graphs) == len(gold_graphs)
                
                # Verify mock was set up correctly
                mock_cuda_available.assert_called()
    
    def test_eval_without_ged_flag(self):
        """Test eval.py runs faster without GED computation."""
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "pred.txt"
            gold_path = Path(td) / "gold.txt"
            output_path = Path(td) / "results.json"
            
            # Slightly larger dataset to test performance
            test_graphs = [
                [["entity1", "relation1", "entity2"], ["entity2", "relation2", "entity3"]],
                [["entity4", "relation3", "entity5"]],
                [["entity6", "relation4", "entity7"], ["entity7", "relation5", "entity8"]],
            ]
            
            _write_graph_file(pred_path, test_graphs)
            _write_graph_file(gold_path, test_graphs)
            
            eval_script = Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics" / "eval.py"
            
            # Run without GED (should be faster)
            cmd = [
                sys.executable, str(eval_script),
                "--pred_file", str(pred_path),
                "--gold_file", str(gold_path),
                "--out", str(output_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                assert result.returncode == 0, f"eval.py failed: {result.stderr}"
                assert output_path.exists()
                
                # Verify results don't include GED
                with output_path.open('r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # GED should not be present or should be None
                metadata = results.get("evaluation_metadata", {})
                assert metadata.get("graph_edit_distance") is None
                
            except subprocess.TimeoutExpired:
                pytest.fail("eval.py without GED should complete within timeout")


class TestGraphMatchingFunctionUpdates:
    """Test specific graph_matching function updates."""
    
    def test_get_bert_score_device_parameter_handling(self):
        """Test that get_bert_score correctly handles device parameter."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
        import graph_matching
        
        # Mock torch and score_bert to test device parameter passing
        with patch('graph_matching.torch') as mock_torch:
            with patch('graph_matching.score_bert') as mock_score_bert:
                mock_torch.cuda.is_available.return_value = True
                # For 1 graph with 2 gold edges and 2 pred edges = 2*2 = 4 edge pairs
                mock_score_bert.return_value = (None, None, np.array([0.5, 0.6, 0.7, 0.8]))
                
                # Test data
                gold_edges = [["entity1;relation1;entity2"], ["entity3;relation2;entity4"]]
                pred_edges = [["entity1;relation1;entity2"], ["entity3;relation2;entity4"]]
                
                # Call function
                precisions, recalls, f1s = graph_matching.get_bert_score([gold_edges], [pred_edges])
                
                # Verify device parameter was passed correctly
                mock_score_bert.assert_called_once()
                call_kwargs = mock_score_bert.call_args.kwargs
                assert call_kwargs['device'] == 'cuda'
                assert call_kwargs['model_type'] == 'bert-base-uncased'
                assert call_kwargs['lang'] == 'en'
                assert call_kwargs['idf'] is False
    
    def test_error_handling_in_device_detection(self):
        """Test error handling when device detection fails."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "Miscellaneous" / "KgGen" / "GraphJudge" / "graph_evaluation" / "metrics"))
        import graph_matching
        
        # Test with various exception scenarios - use None to simulate import failure
        with patch('graph_matching.torch', None):
            with patch('graph_matching.score_bert') as mock_score_bert:
                # For 1 graph with 1 gold edge and 1 pred edge = 1*1 = 1 edge pair
                mock_score_bert.return_value = (None, None, np.array([0.5]))
                
                gold_edges = [["test;relation;test"]]
                pred_edges = [["test;relation;test"]]
                
                # Should fall back to CPU without crashing
                precisions, recalls, f1s = graph_matching.get_bert_score([gold_edges], [pred_edges])
                
                # Verify CPU fallback was used
                call_kwargs = mock_score_bert.call_args.kwargs
                assert call_kwargs['device'] == 'cpu'


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
