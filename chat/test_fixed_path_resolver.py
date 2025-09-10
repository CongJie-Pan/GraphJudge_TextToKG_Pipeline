#!/usr/bin/env python3
"""
Test New Path Resolver Fix in Real Environment

This test verifies that the updated path_resolver.py actually fixes
the working directory dependency issue in the real project environment.
"""

import os
import sys
from pathlib import Path

# Add the chat directory to the Python path so we can import path_resolver
chat_dir = Path(__file__).parent
sys.path.insert(0, str(chat_dir))

import path_resolver

def test_fixed_path_resolver():
    """Test that the new path_resolver is working directory independent"""
    
    print("🧪 Testing Fixed Path Resolver in Real Environment")
    print("=" * 60)
    
    # Store original working directory
    original_cwd = os.getcwd()
    
    try:
        print("Testing updated path_resolver.resolve_pipeline_output()...")
        print("This should return consistent results regardless of working directory.")
        print()
        
        # Test 1: From chat/ directory (run_entity.py scenario)
        print("🗂️  Test 1: From chat/ directory")
        os.chdir(chat_dir)
        print(f"   Working directory: {os.getcwd()}")
        
        try:
            result_from_chat = path_resolver.resolve_pipeline_output(iteration=3, create=False)
            print(f"   ✅ Resolved path: {result_from_chat}")
            chat_success = True
        except path_resolver.PathResolutionError as e:
            print(f"   ⚠️  Multiple datasets error: {e}")
            chat_success = False
            chat_error = str(e)
        
        # Test 2: From root/ directory (stage_manager.py scenario)
        print("\n🗂️  Test 2: From root/ directory")
        root_dir = chat_dir.parent
        os.chdir(root_dir)
        print(f"   Working directory: {os.getcwd()}")
        
        try:
            result_from_root = path_resolver.resolve_pipeline_output(iteration=3, create=False)
            print(f"   ✅ Resolved path: {result_from_root}")
            root_success = True
        except path_resolver.PathResolutionError as e:
            print(f"   ⚠️  Multiple datasets error: {e}")
            root_success = False
            root_error = str(e)
        
        # Analysis
        print("\n" + "=" * 60)
        print("📊 ANALYSIS:")
        
        if chat_success and root_success:
            # Both succeeded - check if paths are the same
            if result_from_chat == result_from_root:
                print("✅ SUCCESS: Both directories resolve to the same path!")
                print(f"   Consistent path: {result_from_chat}")
                return True
            else:
                print("❌ FAILURE: Different paths from different directories!")
                print(f"   From chat/: {result_from_chat}")
                print(f"   From root/: {result_from_root}")
                return False
                
        elif not chat_success and not root_success:
            # Both failed - check if they fail with the same error
            if chat_error == root_error:
                print("✅ SUCCESS: Both directories fail consistently!")
                print("   This is expected when multiple datasets exist.")
                print("   The important thing is consistent behavior.")
                return True
            else:
                print("❌ FAILURE: Different error messages from different directories!")
                print(f"   From chat/: {chat_error}")
                print(f"   From root/: {root_error}")
                return False
                
        else:
            # One succeeded, one failed - this is definitely a bug
            print("❌ FAILURE: Inconsistent behavior based on working directory!")
            print(f"   Chat success: {chat_success}")
            print(f"   Root success: {root_success}")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def test_with_environment_variable():
    """Test with environment variable set to ensure precedence works"""
    
    print("\n🧪 Testing with PIPELINE_OUTPUT_DIR Environment Variable")
    print("=" * 60)
    
    original_cwd = os.getcwd()
    original_env = os.environ.get('PIPELINE_OUTPUT_DIR')
    
    try:
        # Set a specific output directory
        test_output = str(chat_dir / "datasets" / "KIMI_result_DreamOf_RedChamber" / "Graph_Iteration3")
        os.environ['PIPELINE_OUTPUT_DIR'] = test_output
        
        # Test from both directories
        results = []
        
        for test_dir, name in [(chat_dir, "chat"), (chat_dir.parent, "root")]:
            os.chdir(test_dir)
            result = path_resolver.resolve_pipeline_output(iteration=3, create=False)
            results.append(result)
            print(f"From {name}/ directory: {result}")
        
        if results[0] == results[1] == test_output:
            print("✅ SUCCESS: Environment variable takes precedence consistently!")
            return True
        else:
            print("❌ FAILURE: Environment variable precedence not working correctly!")
            return False
            
    finally:
        os.chdir(original_cwd)
        if original_env is not None:
            os.environ['PIPELINE_OUTPUT_DIR'] = original_env
        else:
            os.environ.pop('PIPELINE_OUTPUT_DIR', None)

def test_project_root_detection():
    """Test that project root detection works from any directory"""
    
    print("\n🧪 Testing Project Root Detection")
    print("=" * 60)
    
    original_cwd = os.getcwd()
    
    try:
        # Test from various directories
        test_dirs = [
            chat_dir,  # chat/
            chat_dir.parent,  # root/
            chat_dir / "unit_test",  # chat/unit_test/
        ]
        
        consistent_root = None
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
                
            os.chdir(test_dir)
            root = path_resolver.find_project_root()
            
            print(f"From {test_dir.name}/: {root}")
            
            if consistent_root is None:
                consistent_root = root
            elif consistent_root != root:
                print("❌ FAILURE: Inconsistent project root detection!")
                return False
        
        if consistent_root:
            print("✅ SUCCESS: Project root detection is working directory independent!")
            return True
        else:
            print("❌ FAILURE: Could not detect project root!")
            return False
            
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    print("🔧 Testing Path Resolver Fix in Real Project Environment")
    print("This test verifies that our fix actually works in the real codebase.")
    print()
    
    # Run all tests
    tests = [
        test_fixed_path_resolver,
        test_with_environment_variable,
        test_project_root_detection
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Overall result
    print("\n" + "🎯" * 20)
    print("OVERALL TEST RESULTS:")
    
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("The working directory dependency bug has been FIXED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("The bug may not be completely fixed yet.")
        sys.exit(1)
