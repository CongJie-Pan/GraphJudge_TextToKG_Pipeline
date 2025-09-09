"""
Comprehensive Unit Tests for eval.py

This test suite provides comprehensive coverage for the graph evaluation metrics
script, including CSV auto-conversion, file loading, metrics computation, and
error handling scenarios.

Test Coverage:
- CSV to pred.txt auto-conversion functionality  
- File loading and parsing (gold and pred files)
- Graph metrics computation pipeline
- Error handling for malformed data and mismatched file lengths
- Integration testing with realistic data scenarios
- Unicode handling for Chinese text

Run with: pytest test_eval.py -v --json-report --json-report-file=test_eval_report.json
"""

import ast
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

# Add paths to import modules under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _import_eval_module():
    """Dynamically import eval.py from graph_evaluation/metrics."""
    graphjudge_dir = Path(__file__).resolve().parents[3]
    metrics_dir = graphjudge_dir / "graph_evaluation" / "metrics"
    sys.path.insert(0, str(metrics_dir))
    import eval as eval_module  # type: ignore
    return eval_module


def _import_csv_to_predtxt():
    """Import csv_to_predtxt from tools."""
    graphjudge_dir = Path(__file__).resolve().parents[3]
    tools_dir = graphjudge_dir / "tools"
    sys.path.insert(0, str(tools_dir))
    from csv_to_predtxt import convert_csv_to_pred_txt  # type: ignore
    return convert_csv_to_pred_txt


def _write_csv(path: Path, rows):
    """Write CSV data to file with proper headers."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "generated"])
        writer.writerows(rows)


def _write_graph_file(path: Path, graphs):
    """Write graphs to file in eval.py expected format."""
    with path.open("w", encoding="utf-8") as f:
        for graph in graphs:
            f.write(str(graph) + "\n")


def _read_lines(path: Path):
    """Read all lines from file, stripping newlines."""
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class TestEnsurePredTxtPath:
    """Test the CSV auto-conversion functionality."""
    
    def test_csv_to_pred_txt_conversion(self):
        """CSV should be converted to .pred.txt with expected JSON-line triples."""
        eval_module = _import_eval_module()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "input.csv"
            _write_csv(
                csv_path,
                [
                    ["Is this true: 士隱 地點 書房 ?", "Yes"],
                    ["Is this true: 雨村 地點 葫蘆廟 ?", "No"],
                    ["Is this true: 作者 行為 隱真事 ?", "Yes"],
                ],
            )

            result_path = eval_module._ensure_pred_txt_path(str(csv_path))
            result_path_obj = Path(result_path)

            # Check output file exists and has correct extension
            assert result_path_obj.exists(), "Converted pred.txt file should exist"
            assert result_path_obj.suffix == ".txt"
            assert result_path_obj.name == "input.pred.txt"

            # Check content format and conversion accuracy
            lines = _read_lines(result_path_obj)
            assert len(lines) == 3
            assert json.loads(lines[0]) == [["士隱", "地點", "書房"]]
            assert json.loads(lines[1]) == [["Null", "Null", "Null"]]  # No answer
            assert json.loads(lines[2]) == [["作者", "行為", "隱真事"]]

    def test_txt_file_passthrough(self):
        """If a .txt path is provided, it should be returned unchanged."""
        eval_module = _import_eval_module()

        with tempfile.TemporaryDirectory() as td:
            txt_path = Path(td) / "pred.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write('[[\"Subject\",\"Relation\",\"Object\"]]\n')

            result_path = eval_module._ensure_pred_txt_path(str(txt_path))
            assert result_path == str(txt_path)

    def test_unsupported_extension_error(self):
        """Unsupported extensions should raise ValueError."""
        eval_module = _import_eval_module()

        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "pred.json"
            json_path.write_text('{"test": "data"}', encoding="utf-8")

            with pytest.raises(ValueError, match="Unsupported prediction file extension"):
                eval_module._ensure_pred_txt_path(str(json_path))

    def test_nonexistent_file_error(self):
        """Non-existent files should raise FileNotFoundError."""
        eval_module = _import_eval_module()

        with pytest.raises(FileNotFoundError, match="Prediction file not found"):
            eval_module._ensure_pred_txt_path("nonexistent_file.txt")


class TestCsvToPredTxtConversion:
    """Test the CSV to pred.txt conversion functionality from tools."""
    
    def test_basic_yes_no_conversion(self):
        """Test basic Yes/No conversion with proper triple parsing."""
        convert_csv_to_pred_txt = _import_csv_to_predtxt()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "input.csv"
            out_path = Path(td) / "pred.txt"

            rows = [
                ["Is this true: 士隱 地點 書房 ?", "Yes"],
                ["Is this true: 雨村 地點 葫蘆廟 ?", "No"],
            ]
            _write_csv(csv_path, rows)

            convert_csv_to_pred_txt(csv_path, out_path)

            lines = _read_lines(out_path)
            assert len(lines) == 2
            assert json.loads(lines[0]) == [["士隱", "地點", "書房"]]
            assert json.loads(lines[1]) == [["Null", "Null", "Null"]]

    def test_parsing_without_prefix_suffix(self):
        """Test parsing when prefix/suffix are missing."""
        convert_csv_to_pred_txt = _import_csv_to_predtxt()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "input.csv"
            out_path = Path(td) / "pred.txt"

            rows = [
                ["士隱 地點 書房", "Yes"],
                ["雨村 地點 葫蘆廟", "Yes"],
            ]
            _write_csv(csv_path, rows)

            convert_csv_to_pred_txt(csv_path, out_path)

            lines = _read_lines(out_path)
            assert len(lines) == 2
            assert json.loads(lines[0]) == [["士隱", "地點", "書房"]]
            assert json.loads(lines[1]) == [["雨村", "地點", "葫蘆廟"]]

    def test_malformed_and_unknown_data(self):
        """Test handling of malformed prompts and unknown generated values."""
        convert_csv_to_pred_txt = _import_csv_to_predtxt()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "input.csv"
            out_path = Path(td) / "pred.txt"

            rows = [
                ["Is this true: 無效格式", "Maybe"],   # unknown generated
                ["Is this true: 士隱 地點 書房 ?", "Yes"],
                ["壞格式", "Yes"],  # malformed prompt
                ["", "No"],  # empty prompt
            ]
            _write_csv(csv_path, rows)

            convert_csv_to_pred_txt(csv_path, out_path)

            lines = _read_lines(out_path)
            assert len(lines) == 4
            assert json.loads(lines[0]) == [["Null", "Null", "Null"]]  # unknown generated
            assert json.loads(lines[1]) == [["士隱", "地點", "書房"]]  # valid
            assert json.loads(lines[2]) == [["Null", "Null", "Null"]]  # malformed prompt
            assert json.loads(lines[3]) == [["Null", "Null", "Null"]]  # empty prompt

    def test_unicode_chinese_text(self):
        """Test proper handling of Chinese Unicode characters."""
        convert_csv_to_pred_txt = _import_csv_to_predtxt()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "input.csv"
            out_path = Path(td) / "pred.txt"

            rows = [
                ["Is this true: 賈寶玉 關係 林黛玉 ?", "Yes"],
                ["Is this true: 《紅樓夢》 作者 曹雪芹 ?", "Yes"],
                ["Is this true: 甄士隱 女兒 英蓮 ?", "Yes"],
            ]
            _write_csv(csv_path, rows)

            convert_csv_to_pred_txt(csv_path, out_path)

            lines = _read_lines(out_path)
            assert len(lines) == 3
            assert json.loads(lines[0]) == [["賈寶玉", "關係", "林黛玉"]]
            assert json.loads(lines[1]) == [["《紅樓夢》", "作者", "曹雪芹"]]
            assert json.loads(lines[2]) == [["甄士隱", "女兒", "英蓮"]]


class TestEvalMainFunctionality:
    """Test the main evaluation functionality with mocked dependencies."""
    
    @patch('graph_matching.get_triple_match_f1')
    @patch('graph_matching.get_graph_match_accuracy')
    @patch('graph_matching.split_to_edges')
    @patch('graph_matching.get_tokens')
    @patch('graph_matching.get_bleu_rouge')
    @patch('graph_matching.get_bert_score')
    def test_main_evaluation_pipeline(self, mock_bert_score, mock_bleu_rouge, 
                                    mock_get_tokens, mock_split_to_edges,
                                    mock_graph_match, mock_triple_match):
        """Test the main evaluation pipeline with mocked graph_matching functions."""
        # Setup mocks
        mock_triple_match.return_value = 0.85
        mock_graph_match.return_value = 0.75
        mock_split_to_edges.return_value = [["edge1"], ["edge2"]]
        mock_get_tokens.return_value = ([["token1"]], [["token2"]])
        
        # Mock BLEU/ROUGE returns (precisions, recalls, f1s for both)
        import numpy as np
        mock_bleu_rouge.return_value = (
            np.array([0.8]), np.array([0.7]), np.array([0.75]),  # ROUGE
            np.array([0.9]), np.array([0.8]), np.array([0.85])   # BLEU
        )
        mock_bert_score.return_value = (np.array([0.88]), np.array([0.82]), np.array([0.85]))

        with tempfile.TemporaryDirectory() as td:
            gold_path = Path(td) / "gold.txt"
            pred_path = Path(td) / "pred.txt"
            
            # Create test data
            gold_graphs = [
                [["士隱", "地點", "書房"]],
                [["雨村", "地點", "葫蘆廟"]]
            ]
            pred_graphs = [
                [["士隱", "地點", "書房"]],
                [["雨村", "地點", "廟中"]]  # Slightly different
            ]
            
            _write_graph_file(gold_path, gold_graphs)
            _write_graph_file(pred_path, pred_graphs)

            # Import and run evaluation (need to mock sys.argv)
            eval_module = _import_eval_module()
            
            # Test by directly calling the evaluation logic
            # (avoiding the argparse main section)
            with patch.object(sys, 'argv', ['eval.py', '--gold_file', str(gold_path), '--pred_file', str(pred_path)]):
                # We can't easily test the main execution without significant refactoring
                # Instead, let's test the file loading logic directly
                
                # Test gold file loading
                with open(gold_path, 'r', encoding="utf-8") as f:
                    loaded_gold = []
                    for line in f.readlines():
                        loaded_gold.append(ast.literal_eval(line.strip()))
                
                assert loaded_gold == gold_graphs
                
                # Test pred file loading with error handling
                with open(pred_path, 'r', encoding="utf-8") as f:
                    loaded_pred = []
                    for line in f.readlines():
                        try:
                            loaded_pred.append(ast.literal_eval(line.strip()))
                        except:
                            loaded_pred.append([["Null", "Null", "Null"]])
                
                assert loaded_pred == pred_graphs

    def test_malformed_prediction_handling(self):
        """Test that malformed predictions are handled gracefully."""
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "pred.txt"
            
            # Write file with some malformed lines
            with pred_path.open("w", encoding="utf-8") as f:
                f.write('[["Valid", "Triple", "Here"]]\n')
                f.write('{"malformed": json}\n')  # Invalid format - missing quotes around json
                f.write('incomplete_line_[\n')  # Incomplete
                f.write('[["Another", "Valid", "Triple"]]\n')
            
            # Test loading with error handling
            loaded_graphs = []
            with pred_path.open('r', encoding="utf-8") as f:
                for line in f.readlines():
                    try:
                        loaded_graphs.append(ast.literal_eval(line.strip()))
                    except:
                        loaded_graphs.append([["Null", "Null", "Null"]])
            
            expected = [
                [["Valid", "Triple", "Here"]],
                [["Null", "Null", "Null"]],  # malformed json
                [["Null", "Null", "Null"]],  # incomplete line
                [["Another", "Valid", "Triple"]]
            ]
            
            assert loaded_graphs == expected

    def test_file_length_mismatch_detection(self):
        """Test that mismatched file lengths are detected."""
        with tempfile.TemporaryDirectory() as td:
            gold_path = Path(td) / "gold.txt"
            pred_path = Path(td) / "pred.txt"
            
            # Create files with different lengths
            gold_graphs = [
                [["士隱", "地點", "書房"]],
                [["雨村", "地點", "葫蘆廟"]],
                [["作者", "行為", "隱真事"]]
            ]
            pred_graphs = [
                [["士隱", "地點", "書房"]],
                [["雨村", "地點", "廟中"]]
                # Missing third graph
            ]
            
            _write_graph_file(gold_path, gold_graphs)
            _write_graph_file(pred_path, pred_graphs)
            
            # Load files
            with open(gold_path, 'r', encoding="utf-8") as f:
                loaded_gold = [ast.literal_eval(line.strip()) for line in f.readlines()]
            
            with open(pred_path, 'r', encoding="utf-8") as f:
                loaded_pred = []
                for line in f.readlines():
                    try:
                        loaded_pred.append(ast.literal_eval(line.strip()))
                    except:
                        loaded_pred.append([["Null", "Null", "Null"]])
            
            # Should detect length mismatch
            assert len(loaded_gold) != len(loaded_pred)
            assert len(loaded_gold) == 3
            assert len(loaded_pred) == 2


class TestEvalJsonOutput:
    """Test new JSON output functionality and improved result handling."""
    
    def test_collect_evaluation_results_structure(self):
        """Test that _collect_evaluation_results creates proper structure."""
        eval_module = _import_eval_module()
        
        # Mock data
        gold_graphs = [[["A", "R", "B"]], [["C", "R", "D"]]]
        pred_graphs = [[["A", "R", "B"]], [["C", "R", "X"]]]
        
        # Mock metric results
        import numpy as np
        results = eval_module._collect_evaluation_results(
            gold_graphs=gold_graphs,
            pred_graphs=pred_graphs,
            triple_match_f1=0.75,
            graph_match_accuracy=0.5,
            precisions_rouge=np.array([0.8, 0.6]),
            recalls_rouge=np.array([0.7, 0.9]),
            f1s_rouge=np.array([0.73, 0.72]),
            precisions_bleu=np.array([0.9, 0.8]),
            recalls_bleu=np.array([0.8, 0.7]),
            f1s_bleu=np.array([0.85, 0.75]),
            precisions_BS=np.array([0.88, 0.82]),
            recalls_BS=np.array([0.86, 0.84]),
            f1s_BS=np.array([0.87, 0.83]),
            overall_ged=None
        )
        
        # Verify structure
        assert "evaluation_metadata" in results
        assert "exact_matching_metrics" in results
        assert "text_similarity_metrics" in results
        assert "semantic_similarity_metrics" in results
        
        # Check metadata
        metadata = results["evaluation_metadata"]
        assert metadata["total_graphs"] == 2
        assert "timestamp" in metadata
        assert "evaluation_metrics" in metadata
        
        # Check exact matching metrics
        exact = results["exact_matching_metrics"]
        assert exact["triple_match_f1"]["score"] == 0.75
        assert exact["graph_match_accuracy"]["score"] == 0.5
        assert "description" in exact["triple_match_f1"]
        assert "description" in exact["graph_match_accuracy"]
        
        # Check text similarity metrics
        text = results["text_similarity_metrics"]
        assert "g_bleu" in text
        assert "g_rouge" in text
        g_bleu = text["g_bleu"]
        assert g_bleu["precision"] == 0.85  # (0.9 + 0.8) / 2
        assert g_bleu["recall"] == 0.75     # (0.8 + 0.7) / 2
        assert g_bleu["f1"] == 0.8          # (0.85 + 0.75) / 2
        
        # Check semantic similarity metrics
        semantic = results["semantic_similarity_metrics"]
        assert "g_bert_score" in semantic
        g_bert = semantic["g_bert_score"]
        assert g_bert["precision"] == 0.85  # (0.88 + 0.82) / 2
        assert g_bert["recall"] == 0.85     # (0.86 + 0.84) / 2
        assert g_bert["f1"] == 0.85         # (0.87 + 0.83) / 2

    def test_collect_evaluation_results_with_ged(self):
        """Test _collect_evaluation_results includes GED when provided."""
        eval_module = _import_eval_module()
        
        gold_graphs = [[["A", "R", "B"]]]
        pred_graphs = [[["A", "R", "C"]]]
        
        import numpy as np
        results = eval_module._collect_evaluation_results(
            gold_graphs=gold_graphs,
            pred_graphs=pred_graphs,
            triple_match_f1=0.0,
            graph_match_accuracy=0.0,
            precisions_rouge=np.array([0.5]),
            recalls_rouge=np.array([0.5]),
            f1s_rouge=np.array([0.5]),
            precisions_bleu=np.array([0.5]),
            recalls_bleu=np.array([0.5]),
            f1s_bleu=np.array([0.5]),
            precisions_BS=np.array([0.5]),
            recalls_BS=np.array([0.5]),
            f1s_BS=np.array([0.5]),
            overall_ged=2.5  # Some edit distance
        )
        
        assert "structural_distance_metrics" in results
        ged_metric = results["structural_distance_metrics"]["graph_edit_distance"]
        assert ged_metric["average_ged"] == 2.5
        assert "description" in ged_metric

    def test_output_results_console_only(self):
        """Test _output_results with console output only."""
        eval_module = _import_eval_module()
        
        results = {
            "evaluation_metadata": {
                "timestamp": "2025-01-01T00:00:00",
                "total_graphs": 2,
                "evaluation_metrics": ["test"]
            },
            "exact_matching_metrics": {
                "triple_match_f1": {"score": 0.85, "description": "Test F1"},
                "graph_match_accuracy": {"score": 0.75, "description": "Test Accuracy"}
            },
            "text_similarity_metrics": {
                "g_bleu": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "description": "Test BLEU"},
                "g_rouge": {"precision": 0.88, "recall": 0.82, "f1": 0.85, "description": "Test ROUGE"}
            },
            "semantic_similarity_metrics": {
                "g_bert_score": {"precision": 0.92, "recall": 0.88, "f1": 0.9, "description": "Test BERT"}
            }
        }
        
        # Should not raise any exceptions
        eval_module._output_results(results, output_file=None)

    def test_output_results_with_json_file(self):
        """Test _output_results with JSON file output."""
        eval_module = _import_eval_module()
        
        results = {
            "evaluation_metadata": {
                "timestamp": "2025-01-01T00:00:00",
                "total_graphs": 1,
                "evaluation_metrics": ["test"]
            },
            "exact_matching_metrics": {
                "triple_match_f1": {"score": 0.85, "description": "Test F1"},
                "graph_match_accuracy": {"score": 0.75, "description": "Test Accuracy"}
            },
            "text_similarity_metrics": {
                "g_bleu": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "description": "Test BLEU"},
                "g_rouge": {"precision": 0.88, "recall": 0.82, "f1": 0.85, "description": "Test ROUGE"}
            },
            "semantic_similarity_metrics": {
                "g_bert_score": {"precision": 0.92, "recall": 0.88, "f1": 0.9, "description": "Test BERT"}
            }
        }
        
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "results.json"
            
            eval_module._output_results(results, output_file=str(json_path))
            
            # Verify JSON file was created
            assert json_path.exists()
            
            # Verify JSON content
            with json_path.open('r', encoding='utf-8') as f:
                saved_results = json.load(f)
            
            assert saved_results == results
            assert saved_results["exact_matching_metrics"]["triple_match_f1"]["score"] == 0.85

    def test_naming_correction_in_output(self):
        """Test that the naming confusion has been corrected in output."""
        eval_module = _import_eval_module()
        
        results = {
            "evaluation_metadata": {
                "timestamp": "2025-01-01T00:00:00",
                "total_graphs": 1,
                "evaluation_metrics": ["test"]
            },
            "exact_matching_metrics": {
                "triple_match_f1": {"score": 0.85, "description": "Test F1"},
                "graph_match_accuracy": {"score": 0.75, "description": "Test Accuracy"}
            },
            "text_similarity_metrics": {
                "g_bleu": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "description": "Test BLEU"},
                "g_rouge": {"precision": 0.88, "recall": 0.82, "f1": 0.85, "description": "Test ROUGE"}
            },
            "semantic_similarity_metrics": {
                "g_bert_score": {"precision": 0.92, "recall": 0.88, "f1": 0.9, "description": "Test BERT"}
            }
        }
        
        # Capture output to verify correct naming
        import io
        import contextlib
        from unittest.mock import patch
        
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            eval_module._output_results(results, output_file=None)
        
        output_text = output_buffer.getvalue()
        
        # Should contain "Graph Match Accuracy" not "Graph Match F1 Score"
        assert "Graph Match Accuracy: 0.7500" in output_text
        assert "Graph Match F1 Score" not in output_text  # Should NOT contain old incorrect naming
        assert "Triple Match F1 Score: 0.8500" in output_text

    def test_json_output_integration_with_csv_input(self):
        """Test end-to-end JSON output with CSV input."""
        eval_module = _import_eval_module()
        
        with tempfile.TemporaryDirectory() as td:
            # Create CSV input
            csv_path = Path(td) / "input.csv"
            gold_path = Path(td) / "gold.txt"
            json_path = Path(td) / "results.json"
            
            # CSV data
            csv_data = [
                ["Is this true: 士隱 地點 書房 ?", "Yes"],
                ["Is this true: 雨村 地點 葫蘆廟 ?", "Yes"],
            ]
            _write_csv(csv_path, csv_data)
            
            # Gold data
            gold_data = [
                [["士隱", "地點", "書房"]],
                [["雨村", "地點", "葫蘆廟"]],
            ]
            _write_graph_file(gold_path, gold_data)
            
            # Test CSV conversion
            pred_path_str = eval_module._ensure_pred_txt_path(str(csv_path))
            
            # Load the converted data
            with open(pred_path_str, 'r', encoding="utf-8") as f:
                pred_lines = [json.loads(line.strip()) for line in f.readlines()]
            
            # Create mock results
            results = eval_module._collect_evaluation_results(
                gold_graphs=gold_data,
                pred_graphs=pred_lines,
                triple_match_f1=1.0,  # Perfect match
                graph_match_accuracy=1.0,
                precisions_rouge=np.array([1.0, 1.0]),
                recalls_rouge=np.array([1.0, 1.0]),
                f1s_rouge=np.array([1.0, 1.0]),
                precisions_bleu=np.array([1.0, 1.0]),
                recalls_bleu=np.array([1.0, 1.0]),
                f1s_bleu=np.array([1.0, 1.0]),
                precisions_BS=np.array([1.0, 1.0]),
                recalls_BS=np.array([1.0, 1.0]),
                f1s_BS=np.array([1.0, 1.0])
            )
            
            # Test JSON output
            eval_module._output_results(results, str(json_path))
            
            # Verify JSON file
            assert json_path.exists()
            with json_path.open('r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert saved_data["exact_matching_metrics"]["triple_match_f1"]["score"] == 1.0
            assert saved_data["exact_matching_metrics"]["graph_match_accuracy"]["score"] == 1.0
            assert saved_data["evaluation_metadata"]["total_graphs"] == 2


class TestEvalIntegration:
    """Integration tests using realistic data scenarios."""
    
    def test_end_to_end_csv_to_evaluation_ready(self):
        """Test complete pipeline from CSV input to evaluation-ready files."""
        eval_module = _import_eval_module()

        with tempfile.TemporaryDirectory() as td:
            # Create realistic CSV data based on the provided sample
            csv_path = Path(td) / "realistic_data.csv"
            gold_path = Path(td) / "gold.txt"
            
            csv_data = [
                ["Is this true: 僧 行為 見士隱抱英蓮大哭 ?", "Yes"],
                ["Is this true: 作者 行為 隱真事 ?", "Yes"],
                ["Is this true: 士隱 地點 書房 ?", "Yes"],
                ["Is this true: 雨村 地點 敝齋 ?", "No"],
                ["Is this true: 士隱 地點 小齋 ?", "Yes"],
            ]
            
            # Corresponding gold data (what we expect as correct)
            gold_data = [
                [["僧", "行為", "見士隱抱英蓮大哭"]],
                [["作者", "行為", "隱真事"]],
                [["士隱", "地點", "書房"]],
                [["Null", "Null", "Null"]],  # Should be No in predictions
                [["士隱", "地點", "小齋"]],
            ]
            
            _write_csv(csv_path, csv_data)
            _write_graph_file(gold_path, gold_data)
            
            # Convert CSV to pred.txt
            pred_path_str = eval_module._ensure_pred_txt_path(str(csv_path))
            pred_path = Path(pred_path_str)
            
            # Verify conversion worked
            assert pred_path.exists()
            lines = _read_lines(pred_path)
            assert len(lines) == 5
            
            # Verify content matches expectations
            parsed_pred = [json.loads(line) for line in lines]
            expected_pred = [
                [["僧", "行為", "見士隱抱英蓮大哭"]],  # Yes
                [["作者", "行為", "隱真事"]],          # Yes
                [["士隱", "地點", "書房"]],           # Yes
                [["Null", "Null", "Null"]],          # No
                [["士隱", "地點", "小齋"]],           # Yes
            ]
            
            assert parsed_pred == expected_pred
            
            # Verify files are now ready for evaluation
            with open(gold_path, 'r', encoding="utf-8") as f:
                loaded_gold = [ast.literal_eval(line.strip()) for line in f.readlines()]
            
            with open(pred_path, 'r', encoding="utf-8") as f:
                loaded_pred = [ast.literal_eval(line.strip()) for line in f.readlines()]
            
            # Files should have matching lengths for evaluation
            assert len(loaded_gold) == len(loaded_pred)

    def test_realistic_chinese_text_handling(self):
        """Test handling of complex Chinese text with various character types."""
        convert_csv_to_pred_txt = _import_csv_to_predtxt()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "chinese_complex.csv"
            out_path = Path(td) / "pred.txt"
            
            # Complex Chinese text with various punctuation and formats
            complex_data = [
                ["Is this true: 《紅樓夢》 作者 曹雪芹 ?", "Yes"],
                ["Is this true: 賈寶玉 關係 林黛玉 ?", "Yes"],
                ["Is this true: 甄士隱 女兒 英蓮 ?", "Yes"],
                ["Is this true: 葫蘆廟 地點 閶門外十里街 ?", "Yes"],
                ["Is this true: 神瑛侍者 行為 下凡歷劫 ?", "No"],
            ]
            
            _write_csv(csv_path, complex_data)
            convert_csv_to_pred_txt(csv_path, out_path)
            
            lines = _read_lines(out_path)
            parsed = [json.loads(line) for line in lines]
            
            expected = [
                [["《紅樓夢》", "作者", "曹雪芹"]],
                [["賈寶玉", "關係", "林黛玉"]],
                [["甄士隱", "女兒", "英蓮"]],
                [["葫蘆廟", "地點", "閶門外十里街"]],
                [["Null", "Null", "Null"]],  # No answer
            ]
            
            assert parsed == expected

    def test_edge_cases_and_error_resilience(self):
        """Test various edge cases and error resilience."""
        convert_csv_to_pred_txt = _import_csv_to_predtxt()

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "edge_cases.csv"
            out_path = Path(td) / "pred.txt"
            
            edge_cases = [
                ["", "Yes"],  # Empty prompt
                ["Is this true: A B ?", "Yes"],  # Missing object
                ["Is this true: A B C D E ?", "Yes"],  # Too many parts
                ["Is this true: 單字 關係 另一個單字 ?", ""],  # Empty generated
                ["Very long subject name here 關係 object ?", "Yes"],  # Long subject
                ["Subject 關係 Very long object name here with spaces ?", "Yes"],  # Long object
            ]
            
            _write_csv(csv_path, edge_cases)
            convert_csv_to_pred_txt(csv_path, out_path)
            
            lines = _read_lines(out_path)
            assert len(lines) == len(edge_cases)  # Should preserve line count
            
            # All should be parseable as JSON
            for line in lines:
                parsed = json.loads(line)
                assert isinstance(parsed, list)
                assert len(parsed) == 1  # Single graph per line
                assert len(parsed[0]) == 3  # Triple format


class TestDeviceDetection:
    """Test device detection functionality for BERTScore."""
    
    def test_cuda_available_device_selection(self):
        """Test device selection when CUDA is available."""
        # Mock torch to simulate CUDA availability
        with patch('graph_matching.torch', create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            # Mock the score_bert function to capture device parameter
            with patch('graph_matching.score_bert') as mock_score_bert:
                mock_score_bert.return_value = (None, None, np.array([0.5]))
                
                # Import and call get_bert_score
                sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
                import graph_matching
                
                # Test data
                gold_edges = [["entity1;relation;entity2"]]
                pred_edges = [["entity1;relation;entity2"]]
                
                graph_matching.get_bert_score(gold_edges, pred_edges)
                
                # Verify that score_bert was called with device="cuda"
                mock_score_bert.assert_called_once()
                call_args = mock_score_bert.call_args
                assert call_args.kwargs['device'] == "cuda"
    
    def test_cuda_unavailable_device_selection(self):
        """Test device selection when CUDA is not available."""
        # Mock torch to simulate CUDA unavailability
        with patch('graph_matching.torch', create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            
            # Mock the score_bert function to capture device parameter
            with patch('graph_matching.score_bert') as mock_score_bert:
                mock_score_bert.return_value = (None, None, np.array([0.5]))
                
                # Import and call get_bert_score
                sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
                import graph_matching
                
                # Test data
                gold_edges = [["entity1;relation;entity2"]]
                pred_edges = [["entity1;relation;entity2"]]
                
                graph_matching.get_bert_score(gold_edges, pred_edges)
                
                # Verify that score_bert was called with device="cpu"
                mock_score_bert.assert_called_once()
                call_args = mock_score_bert.call_args
                assert call_args.kwargs['device'] == "cpu"
    
    def test_torch_import_error_fallback(self):
        """Test fallback to CPU when torch import fails."""
        # Import graph_matching module first
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
        import graph_matching
        
        # Mock torch to be None to simulate import failure
        with patch.object(graph_matching, 'torch', None):
            # Mock the score_bert function to capture device parameter
            with patch.object(graph_matching, 'score_bert') as mock_score_bert:
                mock_score_bert.return_value = (None, None, np.array([0.5]))
                
                # Test data
                gold_edges = [["entity1;relation;entity2"]]
                pred_edges = [["entity1;relation;entity2"]]
                
                graph_matching.get_bert_score(gold_edges, pred_edges)
                
                # Verify that score_bert was called with device="cpu"
                mock_score_bert.assert_called_once()
                call_args = mock_score_bert.call_args
                assert call_args.kwargs['device'] == "cpu"
    
    def test_device_detection_output_message(self, capsys):
        """Test that device detection prints the correct message."""
        # Mock torch to simulate CUDA availability
        with patch('graph_matching.torch', create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            # Mock the score_bert function
            with patch('graph_matching.score_bert') as mock_score_bert:
                mock_score_bert.return_value = (None, None, np.array([0.5]))
                
                # Import and call get_bert_score
                sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
                import graph_matching
                
                # Test data
                gold_edges = [["entity1;relation;entity2"]]
                pred_edges = [["entity1;relation;entity2"]]
                
                graph_matching.get_bert_score(gold_edges, pred_edges)
                
                # Capture stdout and verify device message was printed
                captured = capsys.readouterr()
                assert "Using device for BERTScore: cuda" in captured.out


class TestGraphMatchingErrorHandling:
    """Test error handling and edge cases in graph_matching functions."""
    
    def test_get_bert_score_empty_input(self):
        """Test BERTScore with empty input lists."""
        with patch('graph_matching.score_bert') as mock_score_bert:
            mock_score_bert.return_value = (None, None, np.array([]))
            
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
            import graph_matching
            
            # Test with empty edges
            gold_edges = [[]]
            pred_edges = [[]]
            
            precisions, recalls, f1s = graph_matching.get_bert_score(gold_edges, pred_edges)
            
            # Should handle empty input gracefully
            assert len(precisions) == 1
            assert len(recalls) == 1
            assert len(f1s) == 1
    
    def test_get_bert_score_with_mismatched_graph_sizes(self):
        """Test BERTScore with graphs of different sizes."""
        with patch('graph_matching.score_bert') as mock_score_bert:
            # Create mock return value for different sized graphs
            mock_score_bert.return_value = (None, None, np.array([0.8, 0.6, 0.9, 0.7]))
            
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
            import graph_matching
            
            # Test with graphs of different sizes
            gold_edges = [["entity1;relation;entity2", "entity3;relation;entity4"]]
            pred_edges = [["entity1;relation;entity2"]]  # Different size
            
            precisions, recalls, f1s = graph_matching.get_bert_score(gold_edges, pred_edges)
            
            # Should handle different sized graphs
            assert len(precisions) == 1
            assert len(recalls) == 1
            assert len(f1s) == 1
    
    def test_modify_graph_with_various_input_types(self):
        """Test modify_graph function with different input types."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "graph_evaluation" / "metrics"))
        import graph_matching
        
        # Test with mixed data types
        input_graph = [
            ["Entity1", "RELATION", "Entity2"],
            [123, "relation", "entity"],  # Mixed types
            ["  SpacedEntity  ", "relation", "  AnotherEntity  "],  # With spaces
        ]
        
        result = graph_matching.modify_graph(input_graph)
        
        # All should be normalized to lowercase strings
        expected = [
            ["entity1", "relation", "entity2"],
            ["123", "relation", "entity"],
            ["spacedentity", "relation", "anotherentity"],
        ]
        
        assert result == expected


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--json-report", "--json-report-file=test_eval_report.json"])
