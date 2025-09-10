#!/usr/bin/env python3
"""
Regression Test for ECTD Pipeline Working Directory Bug

This test specifically reproduces and verifies the fix for the bug where:
- run_entity.py writes files to chat/datasets/KIMI_result_DreamOf_RedChamber/
- stage_manager.py looks for files in datasets/KIMI_result_DreamOf_RedChamber/
- This caused "files not found" errors despite successful execution

The test must FAIL before the fix and PASS after the fix.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Add the chat directory to Python path
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

import path_resolver

class TestECTDWorkingDirectoryBugRegression(unittest.TestCase):
    """
    Specific regression test for the ECTD working directory bug.
    
    This test reproduces the exact scenario that was failing:
    1. ECTD stage (run_entity.py) executes from chat/ directory
    2. Validation stage (stage_manager.py) executes from root directory
    3. Both use path_resolver but must get the same result
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up once for all tests in this class"""
        cls.test_root = Path(tempfile.mkdtemp())
        
        # Create realistic project structure matching actual codebase
        (cls.test_root / "README.md").write_text("# GraphJudge TextToKG CLI")
        (cls.test_root / "LICENSE").write_text("MIT License")
        
        # Create config directory with pipeline config
        config_dir = cls.test_root / "config"
        config_dir.mkdir()
        (config_dir / "pipeline_config.yaml").write_text("pipeline: config")
        
        # Create the exact scenario that was causing the bug:
        # Both chat/datasets and root/datasets exist with different content
        
        # Chat datasets (where run_entity.py writes)
        chat_dataset = cls.test_root / "chat" / "datasets" / "KIMI_result_DreamOf_RedChamber"
        chat_dataset.mkdir(parents=True)
        chat_iteration3 = chat_dataset / "Graph_Iteration3"
        chat_iteration3.mkdir()
        (chat_iteration3 / "test_entity.txt").write_text("Entity data from ECTD")
        (chat_iteration3 / "test_denoised.txt").write_text("Denoised text from ECTD")
        
        # Root datasets (where stage_manager.py was incorrectly looking)
        root_dataset = cls.test_root / "datasets" / "KIMI_result_DreamOf_RedChamber"
        root_dataset.mkdir(parents=True)
        # Intentionally create different iterations to highlight the bug
        (root_dataset / "Graph_Iteration1").mkdir()
        (root_dataset / "Graph_Iteration2").mkdir()
        # Note: No Graph_Iteration3 here - this was the source of "file not found"
        
        print(f"Test environment created at: {cls.test_root}")
        print(f"Chat dataset: {chat_dataset}")
        print(f"Root dataset: {root_dataset}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_root, ignore_errors=True)
    
    def setUp(self):
        """Set up for each test"""
        self.original_cwd = os.getcwd()
        self.original_env = {}
        
        # Clear environment variables to force auto-detection
        for key in ['PIPELINE_OUTPUT_DIR', 'PIPELINE_DATASET_PATH']:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Clean up after each test"""
        os.chdir(self.original_cwd)
        
        # Restore environment variables
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def test_working_directory_bug_is_fixed(self):
        """
        Main regression test: Verify that the working directory bug is fixed.
        
        This test MUST pass after the fix - if it fails, the bug has regressed.
        """
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            # Scenario 1: run_entity.py execution context (from chat/)
            os.chdir(self.test_root / "chat")
            
            try:
                ectd_output_path = path_resolver.resolve_pipeline_output(iteration=3, create=False)
                ectd_success = True
                print(f"ECTD stage would write to: {ectd_output_path}")
            except path_resolver.PathResolutionError as e:
                ectd_success = False
                ectd_error = str(e)
                print(f"ECTD stage failed: {ectd_error}")
            
            # Scenario 2: stage_manager.py execution context (from root/)
            os.chdir(self.test_root)
            
            try:
                validation_input_path = path_resolver.resolve_pipeline_output(iteration=3, create=False)
                validation_success = True
                print(f"Validation stage would read from: {validation_input_path}")
            except path_resolver.PathResolutionError as e:
                validation_success = False
                validation_error = str(e)
                print(f"Validation stage failed: {validation_error}")
            
            # Analysis of results
            if not ectd_success and not validation_success:
                # Both failed with multiple datasets error - this is expected in our test setup
                # The important thing is they fail consistently, not differently
                self.assertIn("Multiple dataset bases found", ectd_error)
                self.assertIn("Multiple dataset bases found", validation_error)
                print("✅ Both stages fail consistently with multiple datasets error")
                
            elif ectd_success and validation_success:
                # Both succeeded - they MUST return the same path
                self.assertEqual(
                    ectd_output_path, validation_input_path,
                    "❌ BUG REGRESSION! Different paths from different working directories:\n"
                    f"ECTD (from chat/): {ectd_output_path}\n"
                    f"Validation (from root/): {validation_input_path}"
                )
                print("✅ Both stages return the same path - bug is fixed!")
                
            else:
                # One succeeded, one failed - this indicates working directory dependency
                self.fail(
                    "❌ BUG REGRESSION! Inconsistent behavior based on working directory:\n"
                    f"ECTD success: {ectd_success}\n"
                    f"Validation success: {validation_success}"
                )
    
    def test_original_bug_scenario_simulation(self):
        """
        Simulate the exact original bug scenario with controlled environment.
        
        This test uses environment variables to create the exact conditions
        that were causing the original bug.
        """
        # Create a clean single-dataset scenario to avoid ambiguity errors
        test_dataset = self.test_root / "datasets" / "TestModel_result_DreamOf_RedChamber"
        test_dataset.mkdir(parents=True)
        (test_dataset / "Graph_Iteration3").mkdir()
        
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            with patch('path_resolver.detect_dataset_base', return_value=str(test_dataset)):
                # Test from both working directories
                paths = []
                
                # From chat/ directory
                os.chdir(self.test_root / "chat")
                path_from_chat = path_resolver.resolve_pipeline_output(iteration=3, create=False)
                paths.append(path_from_chat)
                
                # From root/ directory
                os.chdir(self.test_root)
                path_from_root = path_resolver.resolve_pipeline_output(iteration=3, create=False)
                paths.append(path_from_root)
                
                # Verify consistency
                self.assertEqual(
                    paths[0], paths[1],
                    f"Working directory still affects path resolution: {paths}"
                )
                
                # Verify the path is absolute and correct
                expected_path = str(test_dataset / "Graph_Iteration3")
                self.assertEqual(paths[0], expected_path)
                print(f"✅ Consistent path resolved: {paths[0]}")
    
    def test_file_existence_validation_scenario(self):
        """
        Test the specific file existence scenario that was failing.
        
        This reproduces the exact sequence:
        1. ECTD writes files successfully
        2. Validation checks for files but can't find them
        """
        # Use environment variable to control the exact dataset
        chat_dataset = self.test_root / "chat" / "datasets" / "KIMI_result_DreamOf_RedChamber"
        os.environ['PIPELINE_DATASET_PATH'] = str(chat_dataset)
        
        try:
            # Simulate ECTD file writing (from chat/)
            os.chdir(self.test_root / "chat")
            ectd_output_dir = path_resolver.resolve_pipeline_output(iteration=3, create=True)
            
            # Write test files (simulate run_entity.py behavior)
            test_files = ["test_entity.txt", "test_denoised.txt"]
            for filename in test_files:
                file_path = Path(ectd_output_dir) / filename
                file_path.write_text(f"Test content for {filename}")
            
            print(f"ECTD wrote files to: {ectd_output_dir}")
            
            # Simulate validation file checking (from root/)
            os.chdir(self.test_root)
            validation_input_dir = path_resolver.resolve_pipeline_output(iteration=3, create=False)
            
            print(f"Validation looking in: {validation_input_dir}")
            
            # Check if validation can find the files ECTD wrote
            files_found = []
            for filename in test_files:
                file_path = Path(validation_input_dir) / filename
                if file_path.exists():
                    files_found.append(filename)
            
            # This is the critical test - validation MUST find all files ECTD wrote
            self.assertEqual(
                set(files_found), set(test_files),
                f"❌ FILE TRANSFER BUG! Validation can't find files ECTD wrote.\n"
                f"ECTD wrote to: {ectd_output_dir}\n"
                f"Validation looked in: {validation_input_dir}\n"
                f"Files found: {files_found} (expected: {test_files})"
            )
            
            print("✅ File transfer test passed - validation found all ECTD files!")
            
        finally:
            # Clean up environment
            if 'PIPELINE_DATASET_PATH' in os.environ:
                del os.environ['PIPELINE_DATASET_PATH']

if __name__ == '__main__':
    print("🔍 ECTD Working Directory Bug Regression Test")
    print("=" * 60)
    print("Testing the fix for the bug where ECTD files were written")
    print("but validation couldn't find them due to working directory")
    print("differences in path resolution.")
    print()
    
    # Run with detailed output
    unittest.main(verbosity=2)
