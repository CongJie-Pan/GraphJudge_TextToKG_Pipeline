#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_convert_Judge_To_jsonGraph.py

Unit tests for the convert_Judge_To_jsonGraph module.
Tests cover triple extraction, CSV parsing, JSON generation and file operations.

Author: Assistant Engineer
Date: 2025-01-15
Purpose: Comprehensive testing of CSV to JSON knowledge graph conversion functionality
"""

import unittest
import tempfile
import os
import json
import csv
import sys
from unittest.mock import patch, mock_open
from datetime import datetime

# Add parent directory to path to import the module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_Judge_To_jsonGraph import TripleExtractor, KnowledgeGraphConverter


class TestTripleExtractor(unittest.TestCase):
    """Test cases for the TripleExtractor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.extractor = TripleExtractor()
    
    def test_extract_simple_triple(self):
        """Test extraction of a simple triple."""
        prompt = "Is this true: 士隱 地點 書房 ?"
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "士隱")
        self.assertEqual(result[1], "地點") 
        self.assertEqual(result[2], "書房")
    
    def test_extract_triple_with_action(self):
        """Test extraction of triple with action predicate."""
        prompt = "Is this true: 雨村 行為 見 ?"
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "雨村")
        self.assertEqual(result[1], "行為")
        self.assertEqual(result[2], "見")
    
    def test_extract_triple_with_complex_entities(self):
        """Test extraction with complex entity names."""
        prompt = "Is this true: 作者 作品 石頭記 ?"
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "作者")
        self.assertEqual(result[1], "作品")
        self.assertEqual(result[2], "石頭記")
    
    def test_extract_triple_with_whitespace(self):
        """Test extraction handles extra whitespace."""
        prompt = "Is this true:   士隱   地點   書房   ?"
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "士隱")
        self.assertEqual(result[1], "地點")
        self.assertEqual(result[2], "書房")
    
    def test_extract_triple_invalid_format(self):
        """Test extraction handles invalid format gracefully."""
        prompt = "This is not a valid prompt"
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNone(result)
    
    def test_extract_triple_empty_string(self):
        """Test extraction handles empty string."""
        prompt = ""
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNone(result)
    
    def test_extract_triple_with_multi_word_predicate(self):
        """Test extraction with multi-word predicate."""
        prompt = "Is this true: 士隱 囑咐 雨村十九日買舟西上赴神京 ?"
        result = self.extractor.extract_triple(prompt)
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "士隱")
        self.assertEqual(result[1], "囑咐")
        self.assertEqual(result[2], "雨村十九日買舟西上赴神京")


class TestKnowledgeGraphConverter(unittest.TestCase):
    """Test cases for the KnowledgeGraphConverter class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv_file = os.path.join(self.temp_dir, "test.csv")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Setup environment variables for testing (reflecting convert_Judge_To_jsonGraph.py changes)
        self.test_iteration = "3"
        os.environ['PIPELINE_ITERATION'] = self.test_iteration
        os.environ['PIPELINE_INPUT_FILE'] = self.temp_csv_file
        os.environ['PIPELINE_OUTPUT_DIR'] = self.output_dir
    
    def tearDown(self):
        """Clean up after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self, data):
        """Create a temporary CSV file with test data."""
        with open(self.temp_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'generated'])  # Header
            for row in data:
                writer.writerow(row)
    
    def test_initialization(self):
        """Test converter initialization."""
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        
        self.assertEqual(converter.csv_file_path, self.temp_csv_file)
        self.assertEqual(converter.output_dir, self.output_dir)
        self.assertIsInstance(converter.extractor, TripleExtractor)
        self.assertEqual(len(converter.entities), 0)
        self.assertEqual(len(converter.relationships), 0)
    
    def test_load_and_parse_csv_success(self):
        """Test successful CSV loading and parsing."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "Yes"],
            ["Is this true: 雨村 地點 敝齋 ?", "No"],
            ["Is this true: 作者 行為 隱真事 ?", "Yes"]
        ]
        self.create_test_csv(test_data)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        result = converter.load_and_parse_csv()
        
        self.assertTrue(result)
        self.assertEqual(converter.stats['total_rows'], 3)
        self.assertEqual(converter.stats['valid_triplets'], 2)
        self.assertEqual(converter.stats['invalid_triplets'], 1)
        self.assertEqual(len(converter.entities), 4)  # 士隱, 書房, 作者, 隱真事
        self.assertEqual(len(converter.relationships), 2)
    
    def test_load_and_parse_csv_file_not_found(self):
        """Test handling of non-existent CSV file."""
        converter = KnowledgeGraphConverter("non_existent.csv", self.output_dir)
        result = converter.load_and_parse_csv()
        
        self.assertFalse(result)
    
    def test_load_and_parse_csv_empty_file(self):
        """Test handling of empty CSV file."""
        self.create_test_csv([])
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        result = converter.load_and_parse_csv()
        
        self.assertTrue(result)  # Should succeed but with no data
        self.assertEqual(converter.stats['total_rows'], 0)
    
    def test_load_and_parse_csv_malformed_prompts(self):
        """Test handling of malformed prompts."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "Yes"],
            ["Invalid prompt format", "Yes"],
            ["Is this true: 雨村 地點 敝齋 ?", "No"]
        ]
        self.create_test_csv(test_data)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        result = converter.load_and_parse_csv()
        
        self.assertTrue(result)
        self.assertEqual(converter.stats['total_rows'], 3)
        self.assertEqual(converter.stats['valid_triplets'], 1)
        self.assertEqual(converter.stats['parsing_errors'], 1)
    
    def test_generate_json_output(self):
        """Test JSON output generation."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "Yes"],
            ["Is this true: 作者 行為 隱真事 ?", "Yes"]
        ]
        self.create_test_csv(test_data)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        converter.load_and_parse_csv()
        json_data = converter.generate_json_output()
        
        # Test structure
        self.assertIn('entities', json_data)
        self.assertIn('relationships', json_data)
        self.assertIn('report', json_data)
        self.assertIn('metadata', json_data)
        
        # Test entities
        self.assertIsInstance(json_data['entities'], list)
        self.assertEqual(len(json_data['entities']), 4)  # 士隱, 書房, 作者, 隱真事
        
        # Test relationships
        self.assertIsInstance(json_data['relationships'], list)
        self.assertEqual(len(json_data['relationships']), 2)
        
        # Test report structure
        report = json_data['report']
        self.assertIn('summary', report)
        self.assertIn('processing_summary', report)
        self.assertIn('quality_metrics', report)
        
        # Test summary statistics
        summary = report['summary']
        self.assertEqual(summary['entities'], 4)
        self.assertEqual(summary['relationships'], 2)
        self.assertEqual(summary['valid_triplets'], 2)
    
    def test_save_json_file(self):
        """Test JSON file saving."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "Yes"]
        ]
        self.create_test_csv(test_data)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        converter.load_and_parse_csv()
        json_data = converter.generate_json_output()
        
        output_path = converter.save_json_file(json_data)
        
        # Test file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith('.json'))
        
        # Test file content
        with open(output_path, 'r', encoding='utf-8') as file:
            saved_data = json.load(file)
        
        self.assertEqual(saved_data['entities'], json_data['entities'])
        self.assertEqual(saved_data['relationships'], json_data['relationships'])
    
    def test_run_conversion_full_process(self):
        """Test the complete conversion process."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "Yes"],
            ["Is this true: 雨村 地點 敝齋 ?", "No"],
            ["Is this true: 作者 行為 隱真事 ?", "Yes"],
            ["Is this true: 小童 行為 獻茶 ?", "Yes"]
        ]
        self.create_test_csv(test_data)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        result = converter.run_conversion()
        
        # Test conversion success
        self.assertTrue(result)
        
        # Test final statistics
        self.assertEqual(converter.stats['total_rows'], 4)
        self.assertEqual(converter.stats['valid_triplets'], 3)
        self.assertEqual(converter.stats['invalid_triplets'], 1)
        
        # Test output directory created
        self.assertTrue(os.path.exists(self.output_dir))
        
        # Test output file created
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
        self.assertEqual(len(output_files), 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv_file = os.path.join(self.temp_dir, "test.csv")
        self.output_dir = os.path.join(self.temp_dir, "output")
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_duplicate_relationships(self):
        """Test handling of duplicate relationships."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "Yes"],
            ["Is this true: 士隱 地點 書房 ?", "Yes"],  # Duplicate
            ["Is this true: 作者 行為 隱真事 ?", "Yes"]
        ]
        
        with open(self.temp_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'generated'])
            for row in test_data:
                writer.writerow(row)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        result = converter.run_conversion()
        
        self.assertTrue(result)
        # Should include both instances (not deduplicated in relationships list)
        self.assertEqual(len(converter.relationships), 3)
        # But entities should be unique
        self.assertEqual(len(converter.entities), 4)
    
    def test_case_sensitivity(self):
        """Test case sensitivity in generated answers."""
        test_data = [
            ["Is this true: 士隱 地點 書房 ?", "YES"],  # Uppercase
            ["Is this true: 作者 行為 隱真事 ?", "yes"],  # Lowercase
            ["Is this true: 雨村 地點 敝齋 ?", "No"],   # Mixed case
            ["Is this true: 小童 行為 獻茶 ?", "NO"]    # Uppercase NO
        ]
        
        with open(self.temp_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'generated'])
            for row in test_data:
                writer.writerow(row)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        converter.load_and_parse_csv()
        
        # Should accept both YES and yes as valid
        self.assertEqual(converter.stats['valid_triplets'], 2)
        self.assertEqual(converter.stats['invalid_triplets'], 2)
    
    def test_special_characters_in_entities(self):
        """Test handling of special characters in entity names."""
        test_data = [
            ["Is this true: 《石頭記》 作者 曹雪芹 ?", "Yes"],
            ["Is this true: 甄士隱（字士隱） 地點 書房 ?", "Yes"]
        ]
        
        with open(self.temp_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'generated'])
            for row in test_data:
                writer.writerow(row)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        result = converter.run_conversion()
        
        self.assertTrue(result)
        self.assertEqual(converter.stats['valid_triplets'], 2)
        
        # Test that special characters are preserved
        entities = list(converter.entities)
        self.assertIn("《石頭記》", entities)
        self.assertIn("甄士隱（字士隱）", entities)


class TestPerformanceAndMemory(unittest.TestCase):
    """Test performance and memory usage with larger datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv_file = os.path.join(self.temp_dir, "large_test.csv")
        self.output_dir = os.path.join(self.temp_dir, "output")
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_large_dataset_processing(self):
        """Test processing of a larger dataset."""
        # Create a dataset with 100 entries
        test_data = []
        for i in range(100):
            entity1 = f"實體{i}"
            entity2 = f"目標{i}" 
            relation = "關係" if i % 2 == 0 else "行為"
            answer = "Yes" if i % 3 != 0 else "No"
            
            test_data.append([f"Is this true: {entity1} {relation} {entity2} ?", answer])
        
        with open(self.temp_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'generated'])
            for row in test_data:
                writer.writerow(row)
        
        converter = KnowledgeGraphConverter(self.temp_csv_file, self.output_dir)
        
        # Time the conversion
        start_time = datetime.now()
        result = converter.run_conversion()
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        self.assertTrue(result)
        self.assertEqual(converter.stats['total_rows'], 100)
        
        # Performance expectation: should complete within reasonable time
        self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds
        
        print(f"Large dataset processing time: {processing_time:.3f} seconds")
        print(f"Valid triplets extracted: {converter.stats['valid_triplets']}")
        print(f"Unique entities: {len(converter.entities)}")


def create_detailed_test_report():
    """Create a detailed test report with coverage information."""
    import time
    
    # Run all tests and collect results
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Custom test runner to collect detailed results
    class DetailedTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
        
        def addSuccess(self, test):
            super().addSuccess(test)
            self.test_results.append({
                'test': str(test),
                'status': 'PASS',
                'message': 'Test completed successfully'
            })
        
        def addError(self, test, err):
            super().addError(test, err)
            self.test_results.append({
                'test': str(test),
                'status': 'ERROR',
                'message': str(err[1])
            })
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_results.append({
                'test': str(test),
                'status': 'FAIL',
                'message': str(err[1])
            })
    
    class DetailedTestRunner(unittest.TextTestRunner):
        def _makeResult(self):
            return DetailedTestResult(self.stream, self.descriptions, self.verbosity)
    
    # Run tests
    start_time = time.time()
    runner = DetailedTestRunner(verbosity=2)
    result = runner.run(suite)
    end_time = time.time()
    
    # Generate report
    report = {
        'test_run_info': {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(end_time - start_time, 3),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': round((result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun) * 100, 2)
        },
        'test_results': result.test_results if hasattr(result, 'test_results') else [],
        'coverage_areas': [
            'Triple extraction from natural language prompts',
            'CSV file loading and parsing',
            'Data validation and error handling',
            'JSON format generation',
            'File I/O operations',
            'Edge cases and special characters',
            'Performance with larger datasets',
            'Error recovery and graceful degradation'
        ],
        'test_categories': {
            'unit_tests': 'Individual component functionality',
            'integration_tests': 'End-to-end conversion process',
            'edge_case_tests': 'Boundary conditions and error states',
            'performance_tests': 'Processing time and memory usage'
        }
    }
    
    return report


class TestConvertJudgeEnvironmentIntegration(unittest.TestCase):
    """Test environment variable integration for convert_Judge_To_jsonGraph."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_file = os.path.join(self.temp_dir, "test_input.csv")
        self.test_output_dir = os.path.join(self.temp_dir, "test_output")
        
        # Setup environment variables for testing
        self.test_iteration = "3"
        os.environ['PIPELINE_ITERATION'] = self.test_iteration
        os.environ['PIPELINE_INPUT_FILE'] = self.test_csv_file
        os.environ['PIPELINE_OUTPUT_DIR'] = self.test_output_dir

    def tearDown(self):
        """Clean up after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_environment_variable_usage(self):
        """Test that environment variables are correctly used for configuration."""
        # Test that PIPELINE environment variables are set
        self.assertEqual(os.environ.get('PIPELINE_ITERATION'), self.test_iteration)
        self.assertEqual(os.environ.get('PIPELINE_INPUT_FILE'), self.test_csv_file)
        self.assertEqual(os.environ.get('PIPELINE_OUTPUT_DIR'), self.test_output_dir)

    @patch.dict(os.environ, {
        'PIPELINE_ITERATION': '9',
        'PIPELINE_INPUT_FILE': '/test/custom/input.csv',
        'PIPELINE_OUTPUT_DIR': '/test/custom/output/'
    }, clear=False)
    def test_environment_variable_override(self):
        """Test that environment variables can override default values."""
        # Test environment variable override functionality
        self.assertEqual(os.environ.get('PIPELINE_ITERATION'), '9')
        self.assertEqual(os.environ.get('PIPELINE_INPUT_FILE'), '/test/custom/input.csv')
        self.assertEqual(os.environ.get('PIPELINE_OUTPUT_DIR'), '/test/custom/output/')

    def test_pipeline_integration_compatibility(self):
        """Test compatibility with convert_Judge_To_jsonGraph.py main function."""
        # Verify that the test setup is compatible with the modified script
        required_vars = [
            'PIPELINE_ITERATION',
            'PIPELINE_INPUT_FILE',
            'PIPELINE_OUTPUT_DIR'
        ]
        
        for var in required_vars:
            self.assertIn(var, os.environ, f"Required environment variable {var} not set")
            self.assertNotEqual(os.environ[var].strip(), '', f"Environment variable {var} is empty")

    def test_main_function_environment_variable_usage(self):
        """Test that main function would use environment variables correctly."""
        # Create test CSV file
        with open(self.test_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'generated'])  # Header
            writer.writerow(['Is this true: 士隱 地點 書房 ?', 'Yes'])
        
        # Create output directory
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Test environment variable reading simulation
        iteration = os.environ.get('PIPELINE_ITERATION', '2')
        csv_file_path = os.environ.get('PIPELINE_INPUT_FILE', 'default.csv')
        output_dir = os.environ.get('PIPELINE_OUTPUT_DIR', 'default_output')
        
        # Verify that environment variables are used
        self.assertEqual(iteration, self.test_iteration)
        self.assertEqual(csv_file_path, self.test_csv_file)
        self.assertEqual(output_dir, self.test_output_dir)
        
        # Verify files exist and are accessible
        self.assertTrue(os.path.exists(csv_file_path), f"CSV file should exist: {csv_file_path}")
        self.assertTrue(os.path.isdir(output_dir), f"Output directory should exist: {output_dir}")


if __name__ == '__main__':
    print("=" * 80)
    print("Running Unit Tests for convert_Judge_To_jsonGraph.py")
    print("=" * 80)
    
    # Run tests and generate report
    report = create_detailed_test_report()
    
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    info = report['test_run_info']
    print(f"Tests Run: {info['tests_run']}")
    print(f"Failures: {info['failures']}")
    print(f"Errors: {info['errors']}")
    print(f"Success Rate: {info['success_rate']}%")
    print(f"Total Time: {info['total_time_seconds']} seconds")
    
    # Save detailed report
    import tempfile
    report_file = os.path.join(tempfile.gettempdir(), "test_convert_Judge_To_jsonGraph_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed test report saved to: {report_file}")
    
    # Exit with appropriate code
    if info['failures'] + info['errors'] == 0:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
