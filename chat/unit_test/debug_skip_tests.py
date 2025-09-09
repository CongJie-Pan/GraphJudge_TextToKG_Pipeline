#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to understand why tests are being skipped
調試腳本以了解為什麼測試被跳過
"""

import os
import sys
import tempfile
from pathlib import Path

# Add CLI directory to path
current_dir = Path(__file__).parent
cli_dir = current_dir.parent / "cli"

print("=== Debug Skip Tests ===")
print(f"Current dir: {current_dir}")
print(f"CLI dir: {cli_dir}")
print(f"CLI dir exists: {cli_dir.exists()}")

# Test 1: Check file existence
cli_files = [
    "cli.py",
    "stage_manager.py", 
    "config_manager.py",
    "pipeline_monitor.py",
    "iteration_manager.py"
]

print("\n=== File Existence Check ===")
for filename in cli_files:
    file_path = cli_dir / filename
    exists = file_path.exists()
    print(f"{filename}: {'EXISTS' if exists else 'NOT FOUND'} - {file_path}")

# Test 2: Try reading files
print("\n=== File Reading Test ===")
for filename in cli_files:
    file_path = cli_dir / filename
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"{filename}: READ SUCCESS ({len(content)} chars)")
        except Exception as e:
            print(f"{filename}: READ FAILED - {e}")
    else:
        print(f"{filename}: FILE NOT FOUND")

# Test 3: Try subprocess execution (CP950 test simulation)
print("\n=== CP950 Subprocess Test ===")
cli_file = cli_dir / "cli.py"
if cli_file.exists():
    try:
        import subprocess
        
        # Test basic execution
        result = subprocess.run(
            [sys.executable, str(cli_file), "--help"],
            capture_output=True,
            text=True,
            cwd=str(cli_file.parent),
            timeout=10
        )
        
        print(f"Basic CLI execution: return_code={result.returncode}")
        print(f"Stderr length: {len(result.stderr)}")
        if result.stderr:
            print(f"Stderr sample: {result.stderr[:200]}...")
        
        # Test with CP950 environment
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'cp950'
        
        result_cp950 = subprocess.run(
            [sys.executable, str(cli_file), "status"],
            capture_output=True,
            text=True,
            cwd=str(cli_file.parent),
            env=env,
            timeout=10
        )
        
        print(f"CP950 CLI execution: return_code={result_cp950.returncode}")
        print(f"CP950 Stderr length: {len(result_cp950.stderr)}")
        if result_cp950.stderr:
            print(f"CP950 Stderr sample: {result_cp950.stderr[:200]}...")
            
    except Exception as e:
        print(f"Subprocess test failed: {e}")
else:
    print("CLI file not found for subprocess test")

# Test 4: Import test
print("\n=== Import Test ===")
if str(cli_dir) not in sys.path:
    sys.path.insert(0, str(cli_dir))

modules_to_test = ["iteration_manager", "config_manager", "stage_manager", "pipeline_monitor"]

for module_name in modules_to_test:
    try:
        module = __import__(module_name)
        print(f"{module_name}: IMPORT SUCCESS")
    except Exception as e:
        print(f"{module_name}: IMPORT FAILED - {e}")

print("\n=== Debug Complete ===")
