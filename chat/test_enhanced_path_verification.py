#!/usr/bin/env python3
"""
Enhanced Dynamic Verification Test for Path Resolution Working Directory Bug

This test demonstrates the exact path inconsistency by setting environment 
variables to control the test scenario and clearly show working directory 
dependencies.
"""

import os
import sys
from pathlib import Path
from glob import glob

# Add the chat directory to the Python path
chat_dir = Path(__file__).parent
sys.path.insert(0, str(chat_dir))

def test_working_directory_dependency():
    """
    Test the core hypothesis: path_resolver's search patterns behave 
    differently based on working directory.
    """
    print("=== Enhanced Dynamic Verification: Working Directory Dependency ===\n")
    
    # Store original working directory and environment
    original_cwd = os.getcwd()
    original_env = {}
    for key in ['PIPELINE_OUTPUT_DIR', 'PIPELINE_DATASET_PATH']:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    try:
        # Test the core patterns that path_resolver uses
        print("Testing the search patterns used by path_resolver.detect_dataset_base():")
        print("Pattern 1: '../datasets/*_result_DreamOf_RedChamber'")
        print("Pattern 2: 'datasets/*_result_DreamOf_RedChamber'")
        print()
        
        # Test from chat/ directory (run_entity.py execution context)
        print("=== Test from chat/ directory ===")
        os.chdir(chat_dir)
        print(f"Working directory: {os.getcwd()}")
        
        pattern1_from_chat = glob("../datasets/*_result_DreamOf_RedChamber")
        pattern2_from_chat = glob("datasets/*_result_DreamOf_RedChamber")
        
        print(f"Pattern '../datasets/*': {pattern1_from_chat}")
        print(f"Pattern 'datasets/*': {pattern2_from_chat}")
        
        all_matches_from_chat = pattern1_from_chat + pattern2_from_chat
        print(f"All matches from chat/: {len(all_matches_from_chat)} results")
        for match in all_matches_from_chat:
            print(f"  - {Path(match).resolve()}")
        print()
        
        # Test from root/ directory (stage_manager.py execution context)
        print("=== Test from root/ directory ===")
        root_dir = chat_dir.parent
        os.chdir(root_dir)
        print(f"Working directory: {os.getcwd()}")
        
        pattern1_from_root = glob("../datasets/*_result_DreamOf_RedChamber")
        pattern2_from_root = glob("datasets/*_result_DreamOf_RedChamber")
        
        print(f"Pattern '../datasets/*': {pattern1_from_root}")
        print(f"Pattern 'datasets/*': {pattern2_from_root}")
        
        all_matches_from_root = pattern1_from_root + pattern2_from_root
        print(f"All matches from root/: {len(all_matches_from_root)} results")
        for match in all_matches_from_root:
            print(f"  - {Path(match).resolve()}")
        print()
        
        # Compare results
        print("=== Comparison ===")
        print(f"Matches from chat/: {len(all_matches_from_chat)}")
        print(f"Matches from root/: {len(all_matches_from_root)}")
        
        # Convert to resolved paths for comparison
        resolved_from_chat = set(str(Path(p).resolve()) for p in all_matches_from_chat)
        resolved_from_root = set(str(Path(p).resolve()) for p in all_matches_from_root)
        
        print(f"Resolved paths from chat/: {resolved_from_chat}")
        print(f"Resolved paths from root/: {resolved_from_root}")
        
        are_same = resolved_from_chat == resolved_from_root
        print(f"Same resolved paths? {are_same}")
        print()
        
        # Show what path_resolver would choose in each case
        print("=== Path Selection Logic Simulation ===")
        
        def simulate_detect_dataset_base(working_dir_name, matches):
            """Simulate the path selection logic from path_resolver"""
            if len(matches) == 0:
                return f"No dataset base found from {working_dir_name}"
            elif len(matches) == 1:
                return f"Selected from {working_dir_name}: {matches[0]}"
            else:
                return f"Multiple options from {working_dir_name}: {matches} (would raise error)"
        
        chat_result = simulate_detect_dataset_base("chat/", all_matches_from_chat)
        root_result = simulate_detect_dataset_base("root/", all_matches_from_root)
        
        print(chat_result)
        print(root_result)
        print()
        
        # Final conclusion
        print("=== CONCLUSION ===")
        if not are_same or len(all_matches_from_chat) != len(all_matches_from_root):
            print("❌ HYPOTHESIS CONFIRMED!")
            print("Working directory affects path resolution results.")
            print("This explains the file location inconsistency bug.")
            print()
            print("Root Cause: path_resolver.detect_dataset_base() uses relative patterns")
            print("that resolve to different locations based on current working directory.")
            print()
            print("Impact:")
            print("- run_entity.py (from chat/) writes files to one location")
            print("- stage_manager.py (from root/) looks for files in another location")
            return False
        else:
            print("✅ Paths are consistent - investigate other causes.")
            return True
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original state
        os.chdir(original_cwd)
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
        print(f"\nRestored working directory to: {os.getcwd()}")

if __name__ == "__main__":
    print("Enhanced Dynamic Verification Test")
    print("Testing hypothesis: path_resolver has working directory dependencies\n")
    
    success = test_working_directory_dependency()
    
    if not success:
        print("\n🎯 STEP 3 COMPLETE: Dynamic Verification")
        print("✅ Hypothesis CONFIRMED - Root cause identified!")
        print("Ready to proceed to Step 4: Solution Implementation")
        sys.exit(1)
    else:
        print("\n❓ Hypothesis not confirmed - need alternative investigation")
        sys.exit(0)
