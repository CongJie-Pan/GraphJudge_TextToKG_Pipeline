"""
Comprehensive Test Suite for eval_preDataProcess.py

This test suite validates all functionality of the preprocessing pipeline
including edge cases, error handling, and integration with real data.

Test Coverage:
1. Triple parsing functionality with various Chinese text patterns
2. CSV to pred.txt conversion with different response types
3. Pred.txt to gold.txt conversion logic
4. Complete pipeline integration
5. Error handling and edge cases
6. Real data validation with the provided KIMI dataset

Usage:
    python test_eval_preDataProcess.py
    python -m pytest test_eval_preDataProcess.py -v

Authors: AI Assistant (Google Engineer Standards)
Created: 2024
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
import json
import csv
import ast
import logging

# Add the correct path to eval_preDataProcess.py module
# From test location: chat/unit_test/eval/ -> graph_evaluation/metrics/
eval_module_path = Path(__file__).parent.parent.parent.parent / "graph_evaluation" / "metrics"
sys.path.insert(0, str(eval_module_path))

try:
    from eval_preDataProcess import (
        parse_triple_from_prompt,
        normalize_generated_response,
        convert_csv_to_pred_txt,
        convert_pred_to_gold_txt,
        process_csv_to_evaluation_files,
        DataProcessingError,
        TripleParsingError
    )
except ImportError as e:
    print(f"Error importing eval_preDataProcess: {e}")
    print("Make sure eval_preDataProcess.py is in the same directory")
    sys.exit(1)

# Configure test logging
logging.basicConfig(level=logging.DEBUG)

class TestTripleParsing(unittest.TestCase):
    """Test cases for triple parsing functionality."""
    
    def test_basic_triple_parsing(self):
        """Test basic triple parsing with standard format."""
        prompt = "Is this true: 僧 行為 見士隱抱英蓮大哭 ?"
        result = parse_triple_from_prompt(prompt)
        expected = ("僧", "行為", "見士隱抱英蓮大哭")
        self.assertEqual(result, expected)
    
    def test_complex_chinese_text(self):
        """Test parsing with complex Chinese entities and relationships."""
        test_cases = [
            ("Is this true: 曹雪芹 行為 纂成目錄 ?", ("曹雪芹", "行為", "纂成目錄")),
            ("Is this true: 絳珠草 地點 西方靈河岸上三生石畔 ?", ("絳珠草", "地點", "西方靈河岸上三生石畔")),
            ("Is this true: 《好了歌》 作者 士隱 ?", ("《好了歌》", "作者", "士隱")),
            ("Is this true: 士隱 囑咐 雨村十九日買舟西上赴神京 ?", ("士隱", "囑咐", "雨村十九日買舟西上赴神京"))
        ]
        
        for prompt, expected in test_cases:
            with self.subTest(prompt=prompt):
                result = parse_triple_from_prompt(prompt)
                self.assertEqual(result, expected, f"Failed for prompt: {prompt}")
    
    def test_parsing_without_prefix(self):
        """Test parsing when 'Is this true:' prefix is missing."""
        prompt = "僧 行為 見士隱抱英蓮大哭 ?"
        result = parse_triple_from_prompt(prompt)
        expected = ("僧", "行為", "見士隱抱英蓮大哭")
        self.assertEqual(result, expected)
    
    def test_parsing_without_question_mark(self):
        """Test parsing when question mark is missing."""
        prompt = "Is this true: 僧 行為 見士隱抱英蓮大哭"
        result = parse_triple_from_prompt(prompt)
        expected = ("僧", "行為", "見士隱抱英蓮大哭")
        self.assertEqual(result, expected)
    
    def test_invalid_formats(self):
        """Test parsing with invalid or malformed prompts."""
        invalid_prompts = [
            "",
            "Invalid format",
            "Is this true: 僧 行為",  # Missing object
            "Is this true: 僧",  # Missing relation and object
            "Is this true:  行為 見士隱抱英蓮大哭",  # Missing subject
            None,
            123,  # Non-string input
        ]
        
        for prompt in invalid_prompts:
            with self.subTest(prompt=prompt):
                result = parse_triple_from_prompt(prompt)
                self.assertIsNone(result, f"Should return None for: {prompt}")
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        prompt = "  Is this true:   僧   行為   見士隱抱英蓮大哭   ?  "
        result = parse_triple_from_prompt(prompt)
        expected = ("僧", "行為", "見士隱抱英蓮大哭")
        self.assertEqual(result, expected)

class TestResponseNormalization(unittest.TestCase):
    """Test cases for response normalization."""
    
    def test_yes_responses(self):
        """Test various 'Yes' response formats."""
        yes_variants = ["Yes", "YES", "yes", "  Yes  ", "  YES  "]
        for response in yes_variants:
            with self.subTest(response=response):
                result = normalize_generated_response(response)
                self.assertTrue(result)
    
    def test_no_responses(self):
        """Test various 'No' response formats."""
        no_variants = ["No", "NO", "no", "  No  ", "  NO  "]
        for response in no_variants:
            with self.subTest(response=response):
                result = normalize_generated_response(response)
                self.assertFalse(result)
    
    def test_invalid_responses(self):
        """Test invalid or unknown responses."""
        invalid_responses = ["Maybe", "Unknown", "", "True", "False", None, 123]
        for response in invalid_responses:
            with self.subTest(response=response):
                result = normalize_generated_response(response)
                self.assertIsNone(result)

class TestCSVToPredTxt(unittest.TestCase):
    """Test cases for CSV to pred.txt conversion."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_path = self.temp_dir / "test.csv"
        self.pred_path = self.temp_dir / "pred.txt"
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self, data: List[Dict[str, str]]) -> None:
        """Helper to create test CSV files."""
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
            writer.writeheader()
            writer.writerows(data)
    
    def test_basic_conversion(self):
        """Test basic CSV to pred.txt conversion."""
        test_data = [
            {"prompt": "Is this true: 僧 行為 見士隱抱英蓮大哭 ?", "generated": "Yes"},
            {"prompt": "Is this true: 雨村 地點 敝齋 ?", "generated": "No"},
            {"prompt": "Is this true: 士隱 地點 小齋 ?", "generated": "Yes"}
        ]
        
        self.create_test_csv(test_data)
        stats = convert_csv_to_pred_txt(self.csv_path, self.pred_path)
        
        # Check statistics
        self.assertEqual(stats["total_rows"], 3)
        self.assertEqual(stats["yes_responses"], 2)
        self.assertEqual(stats["no_responses"], 1)
        self.assertEqual(stats["successful_extractions"], 2)
        
        # Check file contents
        with self.pred_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        
        self.assertEqual(len(lines), 3)
        
        # Parse and validate first line (Yes response)
        first_obj = ast.literal_eval(lines[0])
        expected_first = [["僧", "行為", "見士隱抱英蓮大哭"]]
        self.assertEqual(first_obj, expected_first)
        
        # Parse and validate second line (No response)
        second_obj = ast.literal_eval(lines[1])
        expected_second = [["Null", "Null", "Null"]]
        self.assertEqual(second_obj, expected_second)
    
    def test_malformed_prompts(self):
        """Test handling of malformed prompts."""
        test_data = [
            {"prompt": "Invalid format", "generated": "Yes"},
            {"prompt": "Is this true: 僧 行為 見士隱抱英蓮大哭 ?", "generated": "Yes"},
            {"prompt": "", "generated": "Yes"}
        ]
        
        self.create_test_csv(test_data)
        stats = convert_csv_to_pred_txt(self.csv_path, self.pred_path)
        
        self.assertEqual(stats["total_rows"], 3)
        self.assertEqual(stats["yes_responses"], 3)
        self.assertEqual(stats["successful_extractions"], 1)  # Only one valid extraction
        self.assertEqual(stats["failed_extractions"], 2)
    
    def test_missing_csv_file(self):
        """Test error handling for missing CSV file."""
        non_existent_path = self.temp_dir / "nonexistent.csv"
        
        with self.assertRaises(FileNotFoundError):
            convert_csv_to_pred_txt(non_existent_path, self.pred_path)
    
    def test_invalid_csv_format(self):
        """Test error handling for invalid CSV format."""
        # Create CSV with wrong columns
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wrong", "columns"])
            writer.writeheader()
            writer.writerow({"wrong": "data", "columns": "here"})
        
        with self.assertRaises(DataProcessingError):
            convert_csv_to_pred_txt(self.csv_path, self.pred_path)

class TestPredToGoldTxt(unittest.TestCase):
    """Test cases for pred.txt to gold.txt conversion."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_path = self.temp_dir / "test.csv"
        self.pred_path = self.temp_dir / "pred.txt"
        self.gold_path = self.temp_dir / "gold.txt"
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self, csv_data: List[Dict[str, str]], pred_lines: List[str]) -> None:
        """Helper to create test CSV and pred.txt files."""
        # Create CSV
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
            writer.writeheader()
            writer.writerows(csv_data)
        
        # Create pred.txt
        with self.pred_path.open("w", encoding="utf-8") as f:
            for line in pred_lines:
                f.write(line + "\n")
    
    def test_basic_gold_conversion(self):
        """Test basic pred.txt to gold.txt conversion."""
        csv_data = [
            {"prompt": "Is this true: 僧 行為 見士隱抱英蓮大哭 ?", "generated": "Yes"},
            {"prompt": "Is this true: 雨村 地點 敝齋 ?", "generated": "No"},
            {"prompt": "Is this true: 士隱 地點 小齋 ?", "generated": "Yes"}
        ]
        
        pred_lines = [
            '[["僧", "行為", "見士隱抱英蓮大哭"]]',
            '[["Null", "Null", "Null"]]',
            '[["士隱", "地點", "小齋"]]'
        ]
        
        self.create_test_files(csv_data, pred_lines)
        stats = convert_pred_to_gold_txt(self.pred_path, self.gold_path, self.csv_path)
        
        # Check statistics
        self.assertEqual(stats["total_lines"], 3)
        self.assertEqual(stats["true_labels"], 2)
        self.assertEqual(stats["false_labels"], 1)
        
        # Check file contents
        with self.gold_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        
        self.assertEqual(len(lines), 3)
        
        # Validate gold.txt content
        first_obj = ast.literal_eval(lines[0])  # Yes -> keep triple
        second_obj = ast.literal_eval(lines[1])  # No -> null
        third_obj = ast.literal_eval(lines[2])  # Yes -> keep triple
        
        self.assertEqual(first_obj, [["僧", "行為", "見士隱抱英蓮大哭"]])
        self.assertEqual(second_obj, [["Null", "Null", "Null"]])
        self.assertEqual(third_obj, [["士隱", "地點", "小齋"]])
    
    def test_file_length_mismatch(self):
        """Test error handling for mismatched file lengths."""
        csv_data = [
            {"prompt": "Is this true: 僧 行為 見士隱抱英蓮大哭 ?", "generated": "Yes"}
        ]
        
        pred_lines = [
            '[["僧", "行為", "見士隱抱英蓮大哭"]]',
            '[["Extra", "Line", "Here"]]'  # Extra line
        ]
        
        self.create_test_files(csv_data, pred_lines)
        
        with self.assertRaises(DataProcessingError):
            convert_pred_to_gold_txt(self.pred_path, self.gold_path, self.csv_path)

class TestCompleteIntegtation(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_path = self.temp_dir / "test.csv"
        self.output_dir = self.temp_dir / "output"
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline(self):
        """Test the complete preprocessing pipeline."""
        # Create test CSV with realistic data
        test_data = [
            {"prompt": "Is this true: 僧 行為 見士隱抱英蓮大哭 ?", "generated": "Yes"},
            {"prompt": "Is this true: 作者 行為 隱真事 ?", "generated": "Yes"},
            {"prompt": "Is this true: 雨村 地點 敝齋 ?", "generated": "No"},
            {"prompt": "Is this true: 士隱 地點 小齋 ?", "generated": "Yes"},
            {"prompt": "Invalid format", "generated": "Yes"},  # Should fail parsing
            {"prompt": "Is this true: 曹雪芹 行為 纂成目錄 ?", "generated": "Maybe"}  # Unknown response
        ]
        
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
            writer.writeheader()
            writer.writerows(test_data)
        
        # Run complete pipeline
        results = process_csv_to_evaluation_files(
            csv_path=self.csv_path,
            output_dir=self.output_dir
        )
        
        # Verify results structure
        self.assertIn("output_files", results)
        self.assertIn("processing_stats", results)
        self.assertIn("summary", results)
        
        # Check output files exist
        pred_path = Path(results["output_files"]["pred_txt"])
        gold_path = Path(results["output_files"]["gold_txt"])
        
        self.assertTrue(pred_path.exists())
        self.assertTrue(gold_path.exists())
        
        # Verify file contents
        with pred_path.open("r", encoding="utf-8") as f:
            pred_lines = [line.strip() for line in f.readlines()]
        
        with gold_path.open("r", encoding="utf-8") as f:
            gold_lines = [line.strip() for line in f.readlines()]
        
        self.assertEqual(len(pred_lines), 6)  # Should match input rows
        self.assertEqual(len(gold_lines), 6)  # Should match pred.txt
        
        # Validate specific content
        # First line: Yes response with valid triple
        pred_obj_1 = ast.literal_eval(pred_lines[0])
        gold_obj_1 = ast.literal_eval(gold_lines[0])
        self.assertEqual(pred_obj_1, [["僧", "行為", "見士隱抱英蓮大哭"]])
        self.assertEqual(gold_obj_1, [["僧", "行為", "見士隱抱英蓮大哭"]])
        
        # Third line: No response
        pred_obj_3 = ast.literal_eval(pred_lines[2])
        gold_obj_3 = ast.literal_eval(gold_lines[2])
        self.assertEqual(pred_obj_3, [["Null", "Null", "Null"]])
        self.assertEqual(gold_obj_3, [["Null", "Null", "Null"]])

class TestRealDataValidation(unittest.TestCase):
    """Test with the actual KIMI dataset provided by the user."""
    
    def setUp(self):
        """Set up paths to real data."""
        # Relative path from test file to CSV data
        # From test location: chat/unit_test/eval/ -> datasets/
        self.real_csv_path = Path(__file__).parent.parent.parent.parent / "datasets" / "KIMI_result_DreamOf_RedChamber" / "Graph_Iteration2" / "pred_instructions_context_gemini_itr2.csv"
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "real_data_test"
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_data_processing(self):
        """Test processing with the actual KIMI dataset."""
        # Check if real data file exists
        if not self.real_csv_path.exists():
            self.skipTest(f"Real data file not found: {self.real_csv_path}")
        
        # Process real data
        try:
            results = process_csv_to_evaluation_files(
                csv_path=self.real_csv_path,
                output_dir=self.output_dir
            )
            
            # Verify processing completed successfully
            self.assertTrue(results["summary"]["ready_for_evaluation"])
            self.assertGreater(results["summary"]["total_processed"], 0)
            
            # Check that files were created
            pred_path = Path(results["output_files"]["pred_txt"])
            gold_path = Path(results["output_files"]["gold_txt"])
            
            self.assertTrue(pred_path.exists())
            self.assertTrue(gold_path.exists())
            
            # Validate file format by reading a few lines
            with pred_path.open("r", encoding="utf-8") as f:
                sample_lines = [f.readline().strip() for _ in range(3)]
            
            for line in sample_lines:
                if line:  # Skip empty lines
                    # Should be parseable as Python literal
                    parsed = ast.literal_eval(line)
                    self.assertIsInstance(parsed, list)
                    self.assertGreater(len(parsed), 0)
                    self.assertIsInstance(parsed[0], list)
                    self.assertEqual(len(parsed[0]), 3)  # Should be triples
            
            print(f"\n✅ Real data processing test passed!")
            print(f"   📊 Processed {results['summary']['total_processed']} rows")
            print(f"   📈 Success rate: {results['summary']['extraction_success_rate']:.2%}")
            print(f"   📁 Files created at: {self.output_dir}")
            
        except Exception as e:
            self.fail(f"Real data processing failed: {e}")

def run_performance_test():
    """Run a performance test with a large dataset."""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    # Create a large test dataset
    temp_dir = Path(tempfile.mkdtemp())
    large_csv = temp_dir / "large_test.csv"
    output_dir = temp_dir / "performance_output"
    
    try:
        # Generate 1000 test rows
        import time
        
        print("Generating large test dataset (1000 rows)...")
        test_data = []
        for i in range(1000):
            prompt = f"Is this true: 實體{i} 關係{i%10} 對象{i} ?"
            generated = "Yes" if i % 3 == 0 else "No"  # Mix of responses
            test_data.append({"prompt": prompt, "generated": generated})
        
        with large_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
            writer.writeheader()
            writer.writerows(test_data)
        
        # Time the processing
        start_time = time.time()
        results = process_csv_to_evaluation_files(large_csv, output_dir)
        end_time = time.time()
        
        processing_time = end_time - start_time
        rows_per_second = results["summary"]["total_processed"] / processing_time
        
        print(f"✅ Performance test completed!")
        print(f"   📊 Processed: {results['summary']['total_processed']} rows")
        print(f"   ⏱️  Time: {processing_time:.2f} seconds")
        print(f"   🚀 Speed: {rows_per_second:.1f} rows/second")
        print(f"   📈 Success rate: {results['summary']['extraction_success_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run all tests."""
    print("🧪 Running comprehensive test suite for eval_preDataProcess.py")
    print("="*80)
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTripleParsing,
        TestResponseNormalization,
        TestCSVToPredTxt,
        TestPredToGoldTxt,
        TestCompleteIntegtation,
        TestRealDataValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance test
    if result.wasSuccessful():
        run_performance_test()
    
    # Print summary
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print(f"   ✅ Ran {result.testsRun} tests successfully")
        print(f"   📋 Test coverage: All core functionality validated")
        print(f"   🔧 Ready for production use!")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"   ❌ Failures: {len(result.failures)}")
        print(f"   💥 Errors: {len(result.errors)}")
        print(f"   📋 Total tests: {result.testsRun}")
    print("="*80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
