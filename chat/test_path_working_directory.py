#!/usr/bin/env python3
"""
Dynamic Verification Test for Path Resolution Working Directory Bug

This test reproduces the path inconsistency issue where path_resolver.py
returns different paths when executed from different working directories.

Expected behavior: path_resolver should return consistent paths regardless 
of working directory.

Actual behavior: detect_dataset_base() returns different paths when called
from chat/ vs root directory due to relative path patterns.
"""

import os
import sys
from pathlib import Path

# Add the chat directory to the Python path so we can import path_resolver
chat_dir = Path(__file__).parent
sys.path.insert(0, str(chat_dir))

def test_path_resolution_from_different_directories():
    """
    Test that demonstrates the working directory path resolution bug.
    
    This test simulates the scenario where:
    1. run_entity.py executes from chat/ directory
    2. stage_manager.py may execute from root directory  
    3. Both use path_resolver but get different results
    """
    print("=== Dynamic Verification: Path Resolution Working Directory Bug ===\n")
    
    # Store original working directory
    original_cwd = os.getcwd()
    
    try:
        # Test 1: Import and test from chat/ directory (run_entity.py scenario)
        print("Test 1: Path resolution from chat/ directory")
        print(f"Current working directory: {os.getcwd()}")
        
        # Change to chat directory
        os.chdir(chat_dir)
        print(f"Changed to: {os.getcwd()}")
        
        # Import path_resolver (fresh import)
        if 'path_resolver' in sys.modules:
            del sys.modules['path_resolver']
        import path_resolver
        
        # Test dataset base detection
        dataset_base_from_chat = path_resolver.detect_dataset_base()
        print(f"Dataset base from chat/: {dataset_base_from_chat}")
        
        # Test pipeline output resolution
        pipeline_output_from_chat = path_resolver.resolve_pipeline_output()
        print(f"Pipeline output from chat/: {pipeline_output_from_chat}")
        print()
        
        # Test 2: Import and test from root directory (stage_manager.py scenario)
        print("Test 2: Path resolution from root directory")
        root_dir = chat_dir.parent
        os.chdir(root_dir)
        print(f"Changed to: {os.getcwd()}")
        
        # Force reimport of path_resolver to test from new working directory
        if 'path_resolver' in sys.modules:
            del sys.modules['path_resolver']
        
        # Add chat to path again since we're in a different directory
        if str(chat_dir) not in sys.path:
            sys.path.insert(0, str(chat_dir))
            
        import path_resolver
        
        # Test dataset base detection
        dataset_base_from_root = path_resolver.detect_dataset_base()
        print(f"Dataset base from root/: {dataset_base_from_root}")
        
        # Test pipeline output resolution
        pipeline_output_from_root = path_resolver.resolve_pipeline_output()
        print(f"Pipeline output from root/: {pipeline_output_from_root}")
        print()
        
        # Test 3: Compare results and verify hypothesis
        print("=== Comparison Results ===")
        print(f"Dataset base from chat/: {dataset_base_from_chat}")
        print(f"Dataset base from root/: {dataset_base_from_root}")
        print(f"Are they the same? {dataset_base_from_chat == dataset_base_from_root}")
        print()
        
        print(f"Pipeline output from chat/: {pipeline_output_from_chat}")
        print(f"Pipeline output from root/: {pipeline_output_from_root}")
        print(f"Are they the same? {pipeline_output_from_chat == pipeline_output_from_root}")
        print()
        
        # Test 4: Check if paths actually exist
        print("=== Path Existence Check ===")
        chat_path_exists = Path(dataset_base_from_chat).exists() if dataset_base_from_chat else False
        root_path_exists = Path(dataset_base_from_root).exists() if dataset_base_from_root else False
        
        print(f"Path from chat/ exists: {chat_path_exists}")
        print(f"Path from root/ exists: {root_path_exists}")
        print()
        
        # Test 5: Show the actual search patterns being used
        print("=== Debug: Search Pattern Analysis ===")
        
        # Manually test the patterns to see what they resolve to
        from glob import glob
        
        # Test from chat directory
        os.chdir(chat_dir)
        chat_pattern1 = glob("../datasets/*_result_DreamOf_RedChamber")
        chat_pattern2 = glob("datasets/*_result_DreamOf_RedChamber")
        print(f"From chat/ - '../datasets/*_result_DreamOf_RedChamber': {chat_pattern1}")
        print(f"From chat/ - 'datasets/*_result_DreamOf_RedChamber': {chat_pattern2}")
        
        # Test from root directory  
        os.chdir(root_dir)
        root_pattern1 = glob("../datasets/*_result_DreamOf_RedChamber")
        root_pattern2 = glob("datasets/*_result_DreamOf_RedChamber")
        print(f"From root/ - '../datasets/*_result_DreamOf_RedChamber': {root_pattern1}")
        print(f"From root/ - 'datasets/*_result_DreamOf_RedChamber': {root_pattern2}")
        print()
        
        # Conclusion
        print("=== Test Conclusion ===")
        if dataset_base_from_chat != dataset_base_from_root:
            print("❌ HYPOTHESIS CONFIRMED: Path resolution is working-directory dependent!")
            print("This explains why files are written to one location but validation looks in another.")
            return False  # Test fails, confirming the bug
        else:
            print("✅ Paths are consistent across working directories.")
            print("Hypothesis not confirmed - may need to investigate other causes.")
            return True
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        print(f"\nRestored working directory to: {os.getcwd()}")

if __name__ == "__main__":
    print("Starting dynamic verification test...")
    print("This test will prove whether path_resolver has working directory dependencies.\n")
    
    success = test_path_resolution_from_different_directories()
    
    if not success:
        print("\n🔍 DYNAMIC VERIFICATION COMPLETE")
        print("Root cause confirmed: path_resolver.py uses relative patterns that resolve")
        print("differently based on working directory, causing file location inconsistencies")
        print("between ECTD writer (run_entity.py) and validator (stage_manager.py).")
        sys.exit(1)
    else:
        print("\n✅ Path resolution appears consistent - investigate other causes.")
        sys.exit(0)
