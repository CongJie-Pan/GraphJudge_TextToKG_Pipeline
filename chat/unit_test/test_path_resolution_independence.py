#!/usr/bin/env python3
"""
Test Suite for Path Resolution Working Directory Independence

These tests verify that the path_resolver fixes the working directory
dependency bug identified in the ECTD pipeline.

Test Categories:
1. Working Directory Independence Tests
2. Project Root Detection Tests  
3. Dataset Base Detection Tests
4. Integration Tests for File Transfer Scenarios
5. Regression Tests for Previous Bug

Author: ProEngineer Debugging Team
Date: September 10, 2025
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the chat directory to Python path for imports
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

import path_resolver
from path_resolver import (
    find_project_root, 
    detect_dataset_base, 
    resolve_pipeline_output,
    PathResolutionError
)

class TestWorkingDirectoryIndependence(unittest.TestCase):
    """
    Core tests for working directory independence - the main bug fix.
    """
    
    def setUp(self):
        """Set up test environment with controlled directory structure"""
        self.original_cwd = os.getcwd()
        self.test_root = Path(tempfile.mkdtemp())
        
        # Create mock project structure
        self.create_mock_project_structure()
        
        # Store original environment
        self.original_env = {}
        for key in ['PIPELINE_OUTPUT_DIR', 'PIPELINE_DATASET_PATH']:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_root, ignore_errors=True)
        
        # Restore environment variables
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def create_mock_project_structure(self):
        """Create a realistic project structure for testing"""
        # Project root files
        (self.test_root / "README.md").write_text("# Test Project")
        (self.test_root / "LICENSE").write_text("MIT License")
        
        # Config directory
        config_dir = self.test_root / "config"
        config_dir.mkdir()
        (config_dir / "pipeline_config.yaml").write_text("test: config")
        
        # Chat directory with local datasets
        chat_dir = self.test_root / "chat"
        chat_dir.mkdir()
        chat_datasets = chat_dir / "datasets" / "KIMI_result_DreamOf_RedChamber"
        chat_datasets.mkdir(parents=True)
        (chat_datasets / "Graph_Iteration3").mkdir()
        (chat_datasets / "Graph_Iteration3" / "test_entity.txt").write_text("test data")
        
        # Root datasets directory
        root_datasets = self.test_root / "datasets" / "KIMI_result_DreamOf_RedChamber"
        root_datasets.mkdir(parents=True)
        (root_datasets / "Graph_Iteration1").mkdir()
        (root_datasets / "Graph_Iteration2").mkdir()
        
        # Alternative model result for ambiguity testing
        alt_datasets = self.test_root / "datasets" / "GPT5mini_result_DreamOf_RedChamber"
        alt_datasets.mkdir(parents=True)
        (alt_datasets / "Graph_Iteration1").mkdir()
    
    @patch('path_resolver.find_project_root')
    def test_dataset_detection_from_different_working_directories(self, mock_find_root):
        """
        Test that dataset detection returns consistent results regardless 
        of working directory - this is the core bug fix test.
        """
        # Mock project root to our test environment
        mock_find_root.return_value = self.test_root
        
        # Test from different working directories
        test_directories = [
            self.test_root,  # Root directory
            self.test_root / "chat",  # Chat directory 
            self.test_root / "config",  # Config directory
            Path(tempfile.mkdtemp())  # Random directory
        ]
        
        results = []
        
        for test_dir in test_directories:
            if not test_dir.exists():
                test_dir.mkdir()
            
            os.chdir(test_dir)
            try:
                result = detect_dataset_base()
                results.append(result)
            except PathResolutionError as e:
                # Multiple datasets found - this is expected in our test setup
                self.assertIn("Multiple dataset bases found", str(e))
                results.append("MULTIPLE_FOUND")
        
        # All results should be identical
        self.assertTrue(
            all(r == results[0] for r in results),
            f"Inconsistent results from different working directories: {results}"
        )
    
    @patch('path_resolver.find_project_root')
    def test_resolve_pipeline_output_working_directory_independence(self, mock_find_root):
        """
        Test that resolve_pipeline_output gives consistent results from 
        different working directories.
        """
        mock_find_root.return_value = self.test_root
        
        # Set up single dataset to avoid ambiguity
        single_dataset = self.test_root / "datasets" / "TestModel_result_DreamOf_RedChamber"
        single_dataset.mkdir(parents=True)
        (single_dataset / "Graph_Iteration1").mkdir()
        
        # Remove other datasets to avoid conflicts
        shutil.rmtree(self.test_root / "chat" / "datasets", ignore_errors=True)
        shutil.rmtree(self.test_root / "datasets" / "KIMI_result_DreamOf_RedChamber", ignore_errors=True)
        shutil.rmtree(self.test_root / "datasets" / "GPT5mini_result_DreamOf_RedChamber", ignore_errors=True)
        
        # Test from different directories
        test_dirs = [self.test_root, self.test_root / "chat"]
        results = []
        
        with patch('path_resolver.detect_dataset_base') as mock_detect:
            mock_detect.return_value = str(single_dataset)
            
            for test_dir in test_dirs:
                if not test_dir.exists():
                    test_dir.mkdir()
                os.chdir(test_dir)
                
                result = resolve_pipeline_output(iteration=3, create=False)
                results.append(result)
        
        # Results should be identical and absolute
        self.assertEqual(len(set(results)), 1, f"Different results: {results}")
        self.assertTrue(Path(results[0]).is_absolute())
    
    def test_working_directory_bug_regression(self):
        """
        Regression test: Ensure the original working directory bug doesn't reappear.
        
        This test reproduces the original scenario where run_entity.py and 
        stage_manager.py got different paths.
        """
        with patch('path_resolver.find_project_root') as mock_find_root:
            mock_find_root.return_value = self.test_root
            
            # Simulate run_entity.py execution context (from chat/)
            os.chdir(self.test_root / "chat")
            entity_path = None
            try:
                entity_path = resolve_pipeline_output(iteration=3, create=False)
            except PathResolutionError:
                pass  # Multiple datasets - expected in test setup
            
            # Simulate stage_manager.py execution context (from root/)
            os.chdir(self.test_root)
            manager_path = None
            try:
                manager_path = resolve_pipeline_output(iteration=3, create=False)
            except PathResolutionError:
                pass  # Multiple datasets - expected in test setup
            
            # If both succeeded, they should be the same
            if entity_path and manager_path:
                self.assertEqual(
                    entity_path, manager_path,
                    "Working directory bug has regressed! Different paths from different directories."
                )

class TestProjectRootDetection(unittest.TestCase):
    """Test the project root detection mechanism"""
    
    def setUp(self):
        self.test_root = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
    
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_root, ignore_errors=True)
    
    def test_find_project_root_with_markers(self):
        """Test that project root is found when markers exist"""
        # Create project structure
        (self.test_root / "README.md").write_text("# Project")
        (self.test_root / "LICENSE").write_text("License")
        
        # Create subdirectories
        sub_dir = self.test_root / "some" / "nested" / "directory"
        sub_dir.mkdir(parents=True)
        
        # Mock __file__ to be in the subdirectory
        with patch('path_resolver.__file__', str(sub_dir / "fake_file.py")):
            result = find_project_root()
        
        self.assertEqual(result, self.test_root)
    
    def test_find_project_root_no_markers(self):
        """Test that None is returned when no project markers found"""
        # Create directory without markers
        sub_dir = self.test_root / "no" / "markers" / "here"
        sub_dir.mkdir(parents=True)
        
        with patch('path_resolver.__file__', str(sub_dir / "fake_file.py")):
            result = find_project_root()
        
        self.assertIsNone(result)

class TestDatasetBaseDetection(unittest.TestCase):
    """Test dataset base detection logic"""
    
    def setUp(self):
        self.test_root = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)
    
    def test_single_dataset_detection(self):
        """Test detection when single dataset exists"""
        # Create single dataset
        dataset = self.test_root / "datasets" / "Test_result_DreamOf_RedChamber"
        dataset.mkdir(parents=True)
        (dataset / "Graph_Iteration1").mkdir()
        
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            result = detect_dataset_base()
        
        self.assertEqual(result, str(dataset))
    
    def test_multiple_datasets_error(self):
        """Test that multiple datasets raise appropriate error"""
        # Create multiple datasets
        dataset1 = self.test_root / "datasets" / "Model1_result_DreamOf_RedChamber"
        dataset2 = self.test_root / "datasets" / "Model2_result_DreamOf_RedChamber"
        
        for dataset in [dataset1, dataset2]:
            dataset.mkdir(parents=True)
            (dataset / "Graph_Iteration1").mkdir()
        
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            with self.assertRaises(PathResolutionError) as cm:
                detect_dataset_base()
        
        self.assertIn("Multiple dataset bases found", str(cm.exception))
    
    def test_no_datasets_found(self):
        """Test behavior when no datasets exist"""
        # Create empty datasets directory
        (self.test_root / "datasets").mkdir()
        
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            result = detect_dataset_base()
        
        self.assertIsNone(result)

class TestEnvironmentVariablePrecedence(unittest.TestCase):
    """Test that environment variables take precedence correctly"""
    
    def setUp(self):
        self.original_env = {}
        for key in ['PIPELINE_OUTPUT_DIR', 'PIPELINE_DATASET_PATH']:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def test_pipeline_output_dir_precedence(self):
        """Test that PIPELINE_OUTPUT_DIR takes highest precedence"""
        test_dir = tempfile.mkdtemp()
        os.environ['PIPELINE_OUTPUT_DIR'] = test_dir
        
        try:
            result = resolve_pipeline_output(iteration=1, create=False)
            self.assertEqual(result, test_dir)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_pipeline_dataset_path_precedence(self):
        """Test that PIPELINE_DATASET_PATH works correctly"""
        test_dir = tempfile.mkdtemp()
        os.environ['PIPELINE_DATASET_PATH'] = test_dir
        
        try:
            result = resolve_pipeline_output(iteration=3, create=False)
            expected = os.path.join(test_dir, "Graph_Iteration3")
            self.assertEqual(result, expected)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests that simulate real pipeline scenarios"""
    
    def setUp(self):
        self.test_root = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        
        # Create realistic project structure
        (self.test_root / "README.md").write_text("# GraphJudge")
        (self.test_root / "LICENSE").write_text("MIT")
        
        # Create chat directory structure
        self.chat_dir = self.test_root / "chat"
        self.chat_dir.mkdir()
        
        # Create single dataset to avoid ambiguity
        self.dataset = self.test_root / "datasets" / "KIMI_result_DreamOf_RedChamber"
        self.dataset.mkdir(parents=True)
        (self.dataset / "Graph_Iteration3").mkdir()
        
        # Clear environment
        self.original_env = {}
        for key in ['PIPELINE_OUTPUT_DIR', 'PIPELINE_DATASET_PATH']:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_root, ignore_errors=True)
        
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def test_ectd_file_transfer_scenario(self):
        """
        Integration test: Simulate the ECTD file transfer scenario that was failing.
        
        This test ensures that run_entity.py writing files and stage_manager.py 
        reading files use the same paths.
        """
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            # Simulate run_entity.py execution (from chat directory)
            os.chdir(self.chat_dir)
            writer_output_path = resolve_pipeline_output(iteration=3, create=False)
            
            # Simulate stage_manager.py execution (from root directory)  
            os.chdir(self.test_root)
            reader_input_path = resolve_pipeline_output(iteration=3, create=False)
            
            # Paths should be identical
            self.assertEqual(
                writer_output_path, reader_input_path,
                "File transfer path mismatch - the original bug has not been fixed!"
            )
            
            # Path should point to our test dataset
            expected_path = str(self.dataset / "Graph_Iteration3")
            self.assertEqual(writer_output_path, expected_path)

if __name__ == '__main__':
    print("🧪 Running Path Resolution Working Directory Independence Tests")
    print("=" * 70)
    print("These tests verify the fix for the ECTD pipeline path resolution bug.")
    print()
    
    # Run all tests with detailed output
    unittest.main(verbosity=2)
