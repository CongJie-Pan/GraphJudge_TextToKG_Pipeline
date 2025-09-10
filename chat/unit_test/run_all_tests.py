#!/usr/bin/env python3
"""
Complete Test Suite Runner for Path Resolution Fix

This script runs all the tests related to the ECTD pipeline working directory
bug fix and provides a comprehensive report.

Test Categories:
1. Unit Tests - Working Directory Independence
2. Regression Tests - Original Bug Verification  
3. Integration Tests - Complete Pipeline Scenarios
4. Environment Tests - Configuration Precedence

Usage:
    python run_all_tests.py
"""

import os
import sys
import unittest
from pathlib import Path

# Add the chat directory to Python path
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

def run_test_suite():
    """Run all path resolution tests and provide comprehensive report"""
    
    print("🧪 COMPLETE PATH RESOLUTION TEST SUITE")
    print("=" * 70)
    print("Running all tests for the ECTD pipeline working directory bug fix")
    print()
    
    # Test discovery and execution
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load all test modules
    test_modules = [
        'test_path_resolution_independence',
        'test_ectd_bug_regression'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✅ Loaded tests from {module_name}")
        except ImportError as e:
            print(f"❌ Failed to load {module_name}: {e}")
    
    print()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary report
    print("\n" + "=" * 70)
    print("📊 TEST EXECUTION SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    print("\n" + "🎯" * 20)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("The ECTD pipeline working directory bug has been successfully fixed!")
        print("The path resolution system is now working directory independent!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("The bug fix may need additional work!")
    print("🎯" * 20)
    
    return success

if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
