#!/usr/bin/env python3
"""
Dynamic Verification Script for ECTD Path Consistency Bug

This script reproduces the "files missing" validation failure by:
1. Running run_entity.py and capturing its exact output directory
2. Scanning all potential dataset directories for actual file locations
3. Simulating stage_manager validation logic to show the mismatch
4. Providing concrete evidence of which hypothesis is correct

Purpose: Prove root cause before implementing any fixes (Step 3 of debugging methodology)
"""

import os
import sys
import subprocess
import json
import glob
import datetime
from pathlib import Path

def log_with_timestamp(message, level="INFO"):
    """Log message with timestamp and level."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def capture_environment_state():
    """Capture current environment variables relevant to pipeline."""
    env_vars = [
        'PIPELINE_OUTPUT_DIR',
        'PIPELINE_DATASET_PATH', 
        'PIPELINE_ITERATION',
        'PIPELINE_INPUT_ITERATION',
        'PIPELINE_GRAPH_ITERATION',
        'ECTD_OUTPUT_DIR'
    ]
    
    env_state = {}
    for var in env_vars:
        env_state[var] = os.environ.get(var, '<not set>')
    
    log_with_timestamp("Environment State Captured:")
    for var, value in env_state.items():
        log_with_timestamp(f"  {var} = {value}")
    
    return env_state

def scan_dataset_directories():
    """Scan for all possible dataset directories and check for target files."""
    log_with_timestamp("Scanning for dataset directories...")
    
    # Define search patterns for different model prefixes
    search_patterns = [
        "../datasets/*_result_DreamOf_RedChamber/Graph_Iteration*",
        "datasets/*_result_DreamOf_RedChamber/Graph_Iteration*",
        "../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration*",
        "../datasets/GPT5mini_result_DreamOf_RedChamber/Graph_Iteration*",
        "../datasets/GPT5Mini_result_DreamOf_RedChamber/Graph_Iteration*"
    ]
    
    found_directories = []
    target_files = ["test_entity.txt", "test_denoised.target"]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            abs_path = os.path.abspath(match)
            
            file_status = {}
            for target_file in target_files:
                file_path = os.path.join(abs_path, target_file)
                exists = os.path.exists(file_path)
                size = os.path.getsize(file_path) if exists else 0
                file_status[target_file] = {
                    "exists": exists,
                    "size": size,
                    "path": file_path
                }
            
            found_directories.append({
                "directory": abs_path,
                "pattern_matched": pattern,
                "files": file_status,
                "has_all_files": all(f["exists"] and f["size"] > 0 for f in file_status.values())
            })
    
    log_with_timestamp(f"Found {len(found_directories)} potential directories")
    return found_directories

def simulate_stage_manager_validation(iteration=3):
    """Simulate the stage_manager validation logic to show what it checks."""
    log_with_timestamp("Simulating stage_manager validation logic...")
    
    # Replicate the logic from stage_manager._validate_stage_output
    current_working_dir = os.getcwd()
    
    # Mock environment as stage_manager would set it
    mock_env = {
        'PIPELINE_OUTPUT_DIR': os.environ.get('PIPELINE_OUTPUT_DIR', ''),
        'ECTD_OUTPUT_DIR': os.environ.get('ECTD_OUTPUT_DIR', ''),
        'PIPELINE_ITERATION_PATH': os.environ.get('PIPELINE_ITERATION_PATH', ''),
        'PIPELINE_ITERATION': str(iteration)
    }
    
    # Replicate the candidate location logic
    primary_output_dir = mock_env.get('ECTD_OUTPUT_DIR', mock_env.get('PIPELINE_OUTPUT_DIR', ''))
    legacy_output_dir = os.path.join(mock_env.get('PIPELINE_ITERATION_PATH', ''), "results", "ectd")
    
    # The "Actual" path that stage_manager constructs
    actual_output_locations = [
        os.path.join(current_working_dir, "../datasets/KIMI_result_DreamOf_RedChamber", f"Graph_Iteration{iteration}"),
        primary_output_dir,
        legacy_output_dir
    ]
    
    # Normalize paths as stage_manager does
    normalized_locations = []
    for location in actual_output_locations:
        if location:
            if os.path.isabs(location):
                normalized_locations.append(os.path.normpath(location))
            else:
                normalized_locations.append(os.path.abspath(os.path.normpath(location)))
    
    log_with_timestamp("Stage manager would check these locations:")
    for i, location in enumerate(normalized_locations):
        label = ["Actual", "Primary", "Legacy"][i] if i < 3 else f"Location {i+1}"
        log_with_timestamp(f"  {label}: {location}")
    
    return normalized_locations

def run_entity_with_diagnostics():
    """Run run_entity.py and capture diagnostic output."""
    log_with_timestamp("Running run_entity.py with diagnostic capture...")
    
    # Change to chat directory for execution
    original_cwd = os.getcwd()
    chat_dir = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(chat_dir)
    
    try:
        # Run with current environment
        result = subprocess.run(
            [sys.executable, "run_entity.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        log_with_timestamp(f"run_entity.py completed with return code: {result.returncode}")
        
        if result.stdout:
            log_with_timestamp("STDOUT from run_entity.py:")
            for line in result.stdout.splitlines()[-20:]:  # Last 20 lines
                log_with_timestamp(f"  {line}")
        
        if result.stderr:
            log_with_timestamp("STDERR from run_entity.py:")
            for line in result.stderr.splitlines():
                log_with_timestamp(f"  {line}")
        
        return result
        
    except subprocess.TimeoutExpired:
        log_with_timestamp("run_entity.py timed out after 5 minutes", "ERROR")
        return None
    except Exception as e:
        log_with_timestamp(f"Error running run_entity.py: {e}", "ERROR")
        return None
    finally:
        os.chdir(original_cwd)

def analyze_path_mismatch(scan_results, validator_locations):
    """Analyze the mismatch between actual file locations and validator expectations."""
    log_with_timestamp("Analyzing path mismatch...")
    
    # Find directories that actually contain the files
    actual_file_locations = [d for d in scan_results if d["has_all_files"]]
    
    log_with_timestamp(f"Files actually found in {len(actual_file_locations)} locations:")
    for location in actual_file_locations:
        log_with_timestamp(f"  ✓ {location['directory']}")
        for filename, file_info in location['files'].items():
            if file_info['exists']:
                log_with_timestamp(f"    - {filename}: {file_info['size']} bytes")
    
    # Check if any actual location matches validator expectations
    matches = []
    for actual_loc in actual_file_locations:
        actual_path = actual_loc['directory']
        for validator_path in validator_locations:
            if os.path.normpath(actual_path) == os.path.normpath(validator_path):
                matches.append(actual_path)
    
    if matches:
        log_with_timestamp(f"✅ MATCH FOUND: Validator would find files in {len(matches)} locations")
        for match in matches:
            log_with_timestamp(f"  {match}")
    else:
        log_with_timestamp("❌ NO MATCH: Validator locations don't overlap with actual file locations")
        log_with_timestamp("This confirms the path consistency bug!")
    
    return len(matches) > 0

def main():
    """Main diagnostic execution."""
    log_with_timestamp("=" * 60)
    log_with_timestamp("ECTD Path Consistency Diagnostic Script")
    log_with_timestamp("=" * 60)
    
    # Step 1: Capture initial state
    log_with_timestamp("Step 1: Capturing environment state...")
    env_state = capture_environment_state()
    
    # Step 2: Scan existing directories
    log_with_timestamp("\nStep 2: Scanning for existing dataset directories...")
    initial_scan = scan_dataset_directories()
    
    # Step 3: Run entity extraction
    log_with_timestamp("\nStep 3: Running entity extraction...")
    entity_result = run_entity_with_diagnostics()
    
    if entity_result is None:
        log_with_timestamp("Entity extraction failed, cannot continue diagnostic", "ERROR")
        return False
    
    # Step 4: Scan again to see what was created
    log_with_timestamp("\nStep 4: Scanning for files after entity extraction...")
    post_run_scan = scan_dataset_directories()
    
    # Step 5: Simulate validator
    log_with_timestamp("\nStep 5: Simulating stage_manager validation...")
    validator_locations = simulate_stage_manager_validation()
    
    # Step 6: Analyze mismatch
    log_with_timestamp("\nStep 6: Analyzing path consistency...")
    is_consistent = analyze_path_mismatch(post_run_scan, validator_locations)
    
    # Step 7: Generate report
    log_with_timestamp("\n" + "=" * 60)
    log_with_timestamp("DIAGNOSTIC SUMMARY")
    log_with_timestamp("=" * 60)
    
    if is_consistent:
        log_with_timestamp("✅ Path consistency: PASS - Validator would find the files")
    else:
        log_with_timestamp("❌ Path consistency: FAIL - This reproduces the reported bug")
    
    # Save detailed report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": env_state,
        "initial_scan": initial_scan,
        "post_run_scan": post_run_scan,
        "validator_locations": validator_locations,
        "is_consistent": is_consistent,
        "entity_exit_code": entity_result.returncode if entity_result else None
    }
    
    report_file = f"path_diagnostic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Detailed report saved to: {report_file}")
    
    return is_consistent

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
