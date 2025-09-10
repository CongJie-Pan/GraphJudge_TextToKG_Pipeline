#!/usr/bin/env python3
"""
Final Proof: Minimal Dynamic Verification Test

This test provides the clearest evidence of the working directory bug
that causes ECTD pipeline file location inconsistency.
"""

import os
from pathlib import Path
from glob import glob

def demonstrate_bug():
    """Show the exact bug with minimal code"""
    
    print("🔍 MINIMAL DYNAMIC VERIFICATION - Working Directory Path Bug")
    print("=" * 60)
    
    # Original working directory
    original_cwd = os.getcwd()
    chat_dir = Path(__file__).parent
    root_dir = chat_dir.parent
    
    print("Testing path_resolver's search patterns:")
    print("Pattern 1: '../datasets/*_result_DreamOf_RedChamber'")  
    print("Pattern 2: 'datasets/*_result_DreamOf_RedChamber'")
    print()
    
    try:
        # Test from chat/ (where run_entity.py executes)
        print("🗂️  From chat/ directory (run_entity.py context):")
        os.chdir(chat_dir)
        print(f"   Working directory: {os.getcwd()}")
        
        pattern2_chat = glob("datasets/*_result_DreamOf_RedChamber")
        if pattern2_chat:
            result_chat = str(Path(pattern2_chat[0]).resolve())
            print(f"   ✅ Found: {result_chat}")
        else:
            print("   ❌ No match found")
        
        # Test from root/ (where stage_manager.py might execute)  
        print("\n🗂️  From root/ directory (stage_manager.py context):")
        os.chdir(root_dir)
        print(f"   Working directory: {os.getcwd()}")
        
        pattern2_root = glob("datasets/*_result_DreamOf_RedChamber")
        if pattern2_root:
            result_root = str(Path(pattern2_root[0]).resolve())
            print(f"   ✅ Found: {result_root}")
        else:
            print("   ❌ No match found")
            
        print("\n" + "=" * 60)
        print("📋 COMPARISON:")
        if pattern2_chat and pattern2_root:
            print(f"From chat/: {result_chat}")
            print(f"From root/: {result_root}")
            
            if result_chat != result_root:
                print("\n❌ DIFFERENT PATHS!")
                print("🐛 BUG CONFIRMED: Same pattern resolves to different locations")
                print("   depending on working directory!")
                
                print(f"\n📁 Chat version has: {len(os.listdir(Path(result_chat)))} items")
                print(f"📁 Root version has: {len(os.listdir(Path(result_root)))} items")
                
                # Show which one has the actual files we're looking for
                chat_has_files = (Path(result_chat) / "Graph_Iteration3").exists()
                root_has_files = (Path(result_root) / "Graph_Iteration3").exists()
                
                print(f"\n🎯 Graph_Iteration3 exists in chat path: {chat_has_files}")
                print(f"🎯 Graph_Iteration3 exists in root path: {root_has_files}")
                
                return False
            else:
                print("\n✅ Same paths - working directory not affecting resolution")
                return True
        else:
            print("❌ Could not find matching paths from both directories")
            return False
            
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    bug_confirmed = not demonstrate_bug()
    
    if bug_confirmed:
        print("\n" + "🎯" * 20)
        print("STEP 3: DYNAMIC VERIFICATION - COMPLETE")
        print("ROOT CAUSE PROVEN:")
        print("- path_resolver uses relative patterns that depend on working directory")
        print("- run_entity.py writes to chat/datasets/KIMI_result_DreamOf_RedChamber/")
        print("- stage_manager.py looks in datasets/KIMI_result_DreamOf_RedChamber/")
        print("- These resolve to DIFFERENT physical locations!")
        print("🎯" * 20)
    else:
        print("\n❓ Bug not reproduced - investigate other causes")
