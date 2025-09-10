#!/usr/bin/env python3
"""
Permanent Regression Test for ECTD Working Directory Bug

This test MUST be included in the permanent test suite to prevent
the working directory dependency bug from reappearing.

Requirements:
- This test must FAIL before the fix
- This test must PASS after the fix  
- This test must be run in CI/CD pipeline
- This test must catch any regression of the working directory bug

Bug Summary:
The original bug caused run_entity.py (ECTD stage) and stage_manager.py 
(validation stage) to use different paths when executed from different 
working directories, leading to "files not found" errors despite 
successful file creation.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Ensure we can import path_resolver
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

import path_resolver

class TestECTDWorkingDirectoryBugPrevention(unittest.TestCase):
    """
    PERMANENT REGRESSION TEST
    
    This test ensures the ECTD working directory bug never returns.
    If this test fails, the working directory dependency has been reintroduced.
    """
    
    def setUp(self):
        """Set up controlled test environment"""
        self.original_cwd = os.getcwd()
        self.test_root = Path(tempfile.mkdtemp())
        
        # Create project structure
        (self.test_root / "README.md").write_text("# Test Project")
        (self.test_root / "LICENSE").write_text("MIT License")
        
        # Create chat directory
        self.chat_dir = self.test_root / "chat"
        self.chat_dir.mkdir()
        
        # Create single dataset to avoid ambiguity
        self.dataset = self.test_root / "datasets" / "Test_result_DreamOf_RedChamber"
        self.dataset.mkdir(parents=True)
        (self.dataset / "Graph_Iteration1").mkdir()
        
        # Clear environment variables
        self.original_env = {}
        for key in ['PIPELINE_OUTPUT_DIR', 'PIPELINE_DATASET_PATH']:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_root, ignore_errors=True)
        
        # Restore environment
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def test_working_directory_independence_requirement(self):
        """
        CRITICAL TEST: Path resolution must be working directory independent
        
        This is the core requirement that prevents the original bug.
        If this test fails, the bug has returned.
        """
        with patch('path_resolver.find_project_root', return_value=self.test_root):
            # Test from chat/ directory (run_entity.py context)
            os.chdir(self.chat_dir)
            try:
                path_from_chat = path_resolver.resolve_pipeline_output(iteration=1, create=False)
                chat_success = True
            except Exception as e:
                chat_success = False
                chat_error = str(e)
            
            # Test from root/ directory (stage_manager.py context)
            os.chdir(self.test_root)
            try:
                path_from_root = path_resolver.resolve_pipeline_output(iteration=1, create=False)
                root_success = True
            except Exception as e:
                root_success = False
                root_error = str(e)
            
            # CRITICAL ASSERTION: Both must succeed with same result OR both must fail with same error
            if chat_success and root_success:
                self.assertEqual(
                    path_from_chat, path_from_root,
                    f"❌ WORKING DIRECTORY BUG HAS RETURNED!\n"
                    f"Different paths from different directories:\n"
                    f"From chat/: {path_from_chat}\n"
                    f"From root/: {path_from_root}\n"
                    f"This is the exact bug that was fixed!"
                )
            elif not chat_success and not root_success:
                self.assertEqual(
                    chat_error, root_error,
                    f"❌ INCONSISTENT ERROR BEHAVIOR!\n"
                    f"Different errors from different directories:\n"
                    f"From chat/: {chat_error}\n"
                    f"From root/: {root_error}"
                )
            else:
                self.fail(
                    f"❌ WORKING DIRECTORY DEPENDENCY DETECTED!\n"
                    f"Inconsistent success/failure based on working directory:\n"
                    f"Chat success: {chat_success}\n"
                    f"Root success: {root_success}\n"
                    f"This indicates working directory dependency!"
                )
    
    def test_file_transfer_scenario_prevention(self):
        """
        CRITICAL TEST: File transfer between ECTD and validation must work
        
        This test prevents the specific scenario that was failing:
        ECTD writes files but validation can't find them.
        """
        # Set environment variable for consistent test
        os.environ['PIPELINE_OUTPUT_DIR'] = str(self.dataset / "Graph_Iteration1")
        
        try:
            # Simulate ECTD file writing (from chat/)
            os.chdir(self.chat_dir)
            ectd_output_path = path_resolver.resolve_pipeline_output(iteration=1, create=True)
            
            # Write test files
            test_files = ["test_entity.txt", "test_denoised.txt"]
            for filename in test_files:
                (Path(ectd_output_path) / filename).write_text(f"Test content: {filename}")
            
            # Simulate validation file checking (from root/)
            os.chdir(self.test_root)
            validation_input_path = path_resolver.resolve_pipeline_output(iteration=1, create=False)
            
            # CRITICAL ASSERTION: Validation must find all files ECTD wrote
            missing_files = []
            for filename in test_files:
                if not (Path(validation_input_path) / filename).exists():
                    missing_files.append(filename)
            
            self.assertEqual(
                len(missing_files), 0,
                f"❌ FILE TRANSFER BUG HAS RETURNED!\n"
                f"ECTD wrote files to: {ectd_output_path}\n"
                f"Validation looked in: {validation_input_path}\n"
                f"Missing files: {missing_files}\n"
                f"This is the exact file transfer failure that was fixed!"
            )
            
            # Also verify paths are identical
            self.assertEqual(
                ectd_output_path, validation_input_path,
                f"❌ PATH INCONSISTENCY DETECTED!\n"
                f"ECTD and validation use different paths:\n"
                f"ECTD: {ectd_output_path}\n"
                f"Validation: {validation_input_path}"
            )
            
        finally:
            if 'PIPELINE_OUTPUT_DIR' in os.environ:
                del os.environ['PIPELINE_OUTPUT_DIR']
    
    def test_project_root_detection_stability(self):
        """
        SUPPORT TEST: Project root detection must be stable
        
        This test ensures the foundation of our fix remains solid.
        """
        # Test from multiple subdirectories
        test_dirs = [
            self.test_root,
            self.chat_dir,
            self.test_root / "config",  # Create if needed
        ]
        
        # Create missing directories
        for test_dir in test_dirs:
            if not test_dir.exists():
                test_dir.mkdir()
        
        detected_roots = []
        for test_dir in test_dirs:
            os.chdir(test_dir)
            with patch('path_resolver.__file__', str(test_dir / "fake_file.py")):
                root = path_resolver.find_project_root()
                detected_roots.append(root)
        
        # All detections should return the same project root
        unique_roots = set(str(r) if r else None for r in detected_roots)
        self.assertEqual(
            len(unique_roots), 1,
            f"❌ PROJECT ROOT DETECTION IS UNSTABLE!\n"
            f"Different roots detected: {unique_roots}\n"
            f"This could cause working directory dependencies!"
        )
    
    def test_environment_variable_precedence_maintained(self):
        """
        SUPPORT TEST: Environment variable precedence must be maintained
        
        This ensures our fix doesn't break existing configuration methods.
        """
        test_output_dir = str(self.test_root / "custom_output")
        os.environ['PIPELINE_OUTPUT_DIR'] = test_output_dir
        
        try:
            # Test from different working directories
            paths = []
            for test_dir in [self.chat_dir, self.test_root]:
                os.chdir(test_dir)
                path = path_resolver.resolve_pipeline_output(iteration=1, create=False)
                paths.append(path)
            
            # All paths should be identical and match environment variable
            self.assertTrue(
                all(p == test_output_dir for p in paths),
                f"❌ ENVIRONMENT VARIABLE PRECEDENCE BROKEN!\n"
                f"Expected all paths to be: {test_output_dir}\n"
                f"Got paths: {paths}\n"
                f"Environment variables must take precedence!"
            )
            
        finally:
            del os.environ['PIPELINE_OUTPUT_DIR']

class TestBugPreventionIntegration(unittest.TestCase):
    """
    Integration tests to ensure the bug prevention works in realistic scenarios
    """
    
    def test_real_project_structure_compatibility(self):
        """
        Test that the fix works with the actual project structure
        """
        # This test uses the real project environment
        original_cwd = os.getcwd()
        
        try:
            # Test that project root detection works in real environment
            real_project_root = path_resolver.find_project_root()
            self.assertIsNotNone(
                real_project_root,
                "❌ Cannot detect project root in real environment!"
            )
            
            # Test from different real directories
            test_dirs = [
                Path(real_project_root) / "chat",
                Path(real_project_root),
            ]
            
            results = []
            for test_dir in test_dirs:
                if test_dir.exists():
                    os.chdir(test_dir)
                    try:
                        # This may fail with multiple datasets, but should fail consistently
                        result = path_resolver.detect_dataset_base()
                        results.append(("success", result))
                    except path_resolver.PathResolutionError as e:
                        results.append(("error", str(e)))
            
            # Results should be consistent
            if len(results) >= 2:
                result_types = [r[0] for r in results]
                self.assertTrue(
                    all(rt == result_types[0] for rt in result_types),
                    f"❌ Inconsistent behavior in real environment: {results}"
                )
                
                if result_types[0] == "success":
                    result_values = [r[1] for r in results]
                    self.assertTrue(
                        all(rv == result_values[0] for rv in result_values),
                        f"❌ Different results in real environment: {result_values}"
                    )
                elif result_types[0] == "error":
                    error_messages = [r[1] for r in results]
                    self.assertTrue(
                        all(em == error_messages[0] for em in error_messages),
                        f"❌ Different errors in real environment: {error_messages}"
                    )
        
        finally:
            os.chdir(original_cwd)

def run_regression_tests():
    """
    Run all regression tests and provide detailed report
    """
    print("🛡️  RUNNING PERMANENT REGRESSION TESTS")
    print("=" * 60)
    print("These tests prevent the ECTD working directory bug from returning.")
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all regression test classes
    test_classes = [
        TestECTDWorkingDirectoryBugPrevention,
        TestBugPreventionIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    print("\n" + "🛡️ " * 20)
    if success:
        print("✅ ALL REGRESSION TESTS PASSED!")
        print("The working directory bug prevention is ACTIVE and WORKING!")
        print("This bug will be caught if it ever reappears!")
    else:
        print("❌ REGRESSION TESTS FAILED!")
        print("The working directory bug may have returned!")
        print("IMMEDIATE ACTION REQUIRED!")
    print("🛡️ " * 20)
    
    return success

if __name__ == '__main__':
    success = run_regression_tests()
    sys.exit(0 if success else 1)
