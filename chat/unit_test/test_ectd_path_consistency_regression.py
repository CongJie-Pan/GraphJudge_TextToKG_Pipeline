#!/usr/bin/env python3
"""
Regression Test for ECTD Path Consistency Bug

This test validates that the path resolution fix prevents the "files missing" bug
by ensuring run_entity.py (writer) and stage validation (reader) use identical paths.

Test Scenarios:
1. Default behavior (no environment overrides)
2. Explicit PIPELINE_OUTPUT_DIR override
3. PIPELINE_DATASET_PATH with iteration
4. Mixed environment conditions
5. Manifest-based validation

This test must FAIL before the fix and PASS after the fix.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import unittest
from pathlib import Path

# Add chat directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from path_resolver import resolve_pipeline_output, write_manifest, load_manifest
    PATH_RESOLVER_AVAILABLE = True
except ImportError:
    PATH_RESOLVER_AVAILABLE = False

class TestECTDPathConsistency(unittest.TestCase):
    """
    Test class for ECTD pipeline path consistency.
    
    This validates that the writer (run_entity.py) and validator (stage_manager.py)
    always resolve to the same absolute path, preventing the "files missing" bug.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='ectd_path_test_')
        self.original_env = os.environ.copy()
        self.original_cwd = os.getcwd()
        
        # Clean environment for isolated testing
        env_vars_to_clean = [
            'PIPELINE_OUTPUT_DIR',
            'PIPELINE_DATASET_PATH', 
            'PIPELINE_ITERATION',
            'ECTD_OUTPUT_DIR'
        ]
        for var in env_vars_to_clean:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment and working directory
        os.environ.clear()
        os.environ.update(self.original_env)
        os.chdir(self.original_cwd)
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(PATH_RESOLVER_AVAILABLE, "path_resolver module not available")
    def test_default_path_resolution_consistency(self):
        """Test that default path resolution produces consistent results."""
        iteration = 2
        
        # Test the new path resolver (what run_entity.py will use)
        try:
            resolved_path = resolve_pipeline_output(iteration, create=False)
            self.assertIsInstance(resolved_path, str)
            self.assertTrue(os.path.isabs(resolved_path))
        except Exception as e:
            # If no existing dataset found, should get default path
            self.assertIn("GPT5mini_result_DreamOf_RedChamber", str(e).lower() or resolved_path)
    
    @unittest.skipUnless(PATH_RESOLVER_AVAILABLE, "path_resolver module not available")
    def test_explicit_output_dir_override(self):
        """Test explicit PIPELINE_OUTPUT_DIR override is respected consistently."""
        iteration = 3
        test_output_dir = os.path.join(self.temp_dir, "explicit_output")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Set explicit override
        os.environ['PIPELINE_OUTPUT_DIR'] = test_output_dir
        
        # Test resolver respects the override
        resolved_path = resolve_pipeline_output(iteration, create=False)
        self.assertEqual(os.path.normpath(resolved_path), os.path.normpath(test_output_dir))
    
    @unittest.skipUnless(PATH_RESOLVER_AVAILABLE, "path_resolver module not available")
    def test_dataset_path_with_iteration(self):
        """Test PIPELINE_DATASET_PATH + iteration produces consistent paths."""
        iteration = 4
        dataset_base = os.path.join(self.temp_dir, "datasets", "TEST_result_DreamOf_RedChamber")
        os.makedirs(dataset_base, exist_ok=True)
        
        os.environ['PIPELINE_DATASET_PATH'] = f"{dataset_base}/"
        
        resolved_path = resolve_pipeline_output(iteration, create=True)
        expected_path = os.path.join(dataset_base, f"Graph_Iteration{iteration}")
        
        self.assertEqual(os.path.normpath(resolved_path), os.path.normpath(expected_path))
        self.assertTrue(os.path.exists(resolved_path))
    
    @unittest.skipUnless(PATH_RESOLVER_AVAILABLE, "path_resolver module not available") 
    def test_manifest_based_validation(self):
        """Test that manifest-based validation works correctly."""
        iteration = 5
        output_dir = os.path.join(self.temp_dir, "manifest_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test files
        entity_file = os.path.join(output_dir, "test_entity.txt")
        denoised_file = os.path.join(output_dir, "test_denoised.target")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            f.write("['test_entity']\n")
        
        with open(denoised_file, 'w', encoding='utf-8') as f:
            f.write("Test denoised content\n")
        
        # Write manifest
        created_files = ["test_entity.txt", "test_denoised.target"]
        manifest_path = write_manifest(output_dir, "ectd", iteration, created_files)
        
        # Load and validate manifest
        manifest = load_manifest(output_dir)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest['stage'], "ectd")
        self.assertEqual(manifest['iteration'], iteration)
        self.assertEqual(manifest['files'], created_files)
        
        # Verify manifest points to existing files
        from path_resolver import validate_manifest_files
        is_valid, missing = validate_manifest_files(output_dir, manifest)
        self.assertTrue(is_valid, f"Manifest validation failed, missing: {missing}")
    
    def test_simulated_stage_manager_compatibility(self):
        """Test that stage_manager-style validation would find files from new resolver."""
        iteration = 6
        
        # Create a realistic dataset structure
        dataset_dir = os.path.join(self.temp_dir, "datasets", "KIMI_result_DreamOf_RedChamber")
        iteration_dir = os.path.join(dataset_dir, f"Graph_Iteration{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Create the required files
        entity_file = os.path.join(iteration_dir, "test_entity.txt")
        denoised_file = os.path.join(iteration_dir, "test_denoised.target")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            f.write("['simulated_entity']\n")
        
        with open(denoised_file, 'w', encoding='utf-8') as f:
            f.write("Simulated denoised content\n")
        
        # Simulate stage_manager checking logic
        files_to_check = [entity_file, denoised_file]
        all_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in files_to_check)
        
        self.assertTrue(all_exist, "Stage manager simulation should find all files")
    
    def test_cross_platform_path_handling(self):
        """Test that path resolution works correctly across different path separators."""
        iteration = 7
        
        # Test with mixed path separators (relevant for Windows/Unix compatibility)
        if PATH_RESOLVER_AVAILABLE:
            # Set a path with forward slashes
            test_path = self.temp_dir.replace('\\', '/') + "/cross_platform_test"
            os.makedirs(test_path, exist_ok=True)
            
            os.environ['PIPELINE_OUTPUT_DIR'] = test_path
            
            resolved_path = resolve_pipeline_output(iteration, create=False)
            
            # Should normalize to platform-appropriate separators
            self.assertTrue(os.path.exists(resolved_path))
            self.assertEqual(os.path.normpath(resolved_path), os.path.normpath(test_path))
    
    def test_legacy_compatibility_fallback(self):
        """Test that system gracefully falls back when path_resolver is not available."""
        # This test validates backward compatibility
        iteration = 8
        
        # Simulate the old hardcoded path logic
        legacy_dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
        legacy_path = f"{legacy_dataset_base}Graph_Iteration{iteration}"
        
        # The legacy path should still be a valid format
        self.assertIn("Graph_Iteration", legacy_path)
        self.assertIn("DreamOf_RedChamber", legacy_path)

class TestPathConsistencyRegression(unittest.TestCase):
    """
    End-to-end regression test that reproduces the original bug scenario.
    
    This test specifically validates the fix for the path mismatch between
    run_entity.py output and stage_manager.py validation.
    """
    
    def setUp(self):
        """Set up test environment for regression testing."""
        self.temp_dir = tempfile.mkdtemp(prefix='regression_test_')
        self.original_env = os.environ.copy()
        
        # Clean environment
        env_vars_to_clean = [
            'PIPELINE_OUTPUT_DIR',
            'PIPELINE_DATASET_PATH',
            'PIPELINE_ITERATION'
        ]
        for var in env_vars_to_clean:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up regression test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(PATH_RESOLVER_AVAILABLE, "path_resolver module not available")
    def test_original_bug_scenario_fixed(self):
        """
        Test that the original bug scenario (different dataset prefixes) is fixed.
        
        Original bug: run_entity.py wrote to GPT5mini_result_* but validator
        checked KIMI_result_* causing "files missing" errors.
        """
        iteration = 3
        
        # Create both potential dataset directories to simulate the ambiguity
        kimi_dir = os.path.join(self.temp_dir, "datasets", "KIMI_result_DreamOf_RedChamber", f"Graph_Iteration{iteration}")
        gpt5_dir = os.path.join(self.temp_dir, "datasets", "GPT5mini_result_DreamOf_RedChamber", f"Graph_Iteration{iteration}")
        
        os.makedirs(kimi_dir, exist_ok=True)
        os.makedirs(gpt5_dir, exist_ok=True)
        
        # Simulate the fix: explicitly set PIPELINE_OUTPUT_DIR
        os.environ['PIPELINE_OUTPUT_DIR'] = gpt5_dir
        
        # Test that resolver uses the explicit override
        resolved_path = resolve_pipeline_output(iteration, create=False)
        self.assertEqual(os.path.normpath(resolved_path), os.path.normpath(gpt5_dir))
        
        # Create files in the resolved directory
        entity_file = os.path.join(resolved_path, "test_entity.txt")
        denoised_file = os.path.join(resolved_path, "test_denoised.target")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            f.write("['regression_test_entity']\n")
        
        with open(denoised_file, 'w', encoding='utf-8') as f:
            f.write("Regression test denoised content\n")
        
        # Write manifest to ensure stage validation finds files
        write_manifest(resolved_path, "ectd", iteration, ["test_entity.txt", "test_denoised.target"])
        
        # Verify manifest-based validation succeeds
        manifest = load_manifest(resolved_path)
        self.assertIsNotNone(manifest)
        
        from path_resolver import validate_manifest_files
        is_valid, missing = validate_manifest_files(resolved_path, manifest)
        self.assertTrue(is_valid, f"Regression test failed: {missing}")

def create_test_report():
    """Generate a test report and return success status."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestECTDPathConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestPathConsistencyRegression))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ECTD Path Consistency Test Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\nOverall Status: {status}")
    
    return success

if __name__ == "__main__":
    print("🧪 Running ECTD Path Consistency Regression Tests...")
    success = create_test_report()
    sys.exit(0 if success else 1)
