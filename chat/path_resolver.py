"""
Pipeline Path Resolution Utilities

This module provides centralized path resolution for the GraphJudge pipeline,
eliminating the inconsistent path construction that causes file transfer failures
between stages.

Key Features:
- Single source of truth for output directory resolution
- Automatic dataset prefix detection and validation
- Path manifest system for stage-to-stage contracts
- Backward compatibility with existing environment variables

Usage:
    from chat.path_resolver import resolve_pipeline_output, write_manifest
    
    output_dir = resolve_pipeline_output(iteration=3)
    write_manifest(output_dir, stage="ectd", files=["test_entity.txt"])
"""

import os
import json
import glob
import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Version for manifest compatibility
RESOLVER_VERSION = "1.0"

class PathResolutionError(Exception):
    """Raised when path resolution fails or is ambiguous."""
    pass

class ManifestError(Exception):
    """Raised when manifest operations fail."""
    pass

def find_project_root() -> Optional[Path]:
    """
    Find the project root directory by looking for characteristic files.
    
    This method is working-directory independent - it starts from the current
    file's location and searches upward for project markers.
    
    Returns:
        Path: Project root directory, or None if not found
    """
    # Start from this file's directory
    current = Path(__file__).resolve().parent
    
    # Project markers that indicate the root directory
    project_markers = ["README.md", "LICENSE", "config/pipeline_config.yaml"]
    
    # Search upward through parent directories
    for parent in [current] + list(current.parents):
        # Check if this directory contains project markers
        if all((parent / marker).exists() for marker in project_markers[:2]):  # README + LICENSE
            return parent
    
    return None

def detect_dataset_base() -> Optional[str]:
    """
    Working-directory independent dataset base detection.
    
    Uses project root as reference point instead of relative paths,
    eliminating the working directory dependency bug.
    
    Returns:
        str: The detected dataset base path, or None if not found
        
    Raises:
        PathResolutionError: If multiple conflicting bases are found
    """
    # Find project root first (working directory independent)
    project_root = find_project_root()
    if not project_root:
        raise PathResolutionError(
            "Cannot find project root directory. Expected to find README.md and LICENSE files."
        )
    
    # Search for datasets in known locations relative to project root
    search_locations = [
        project_root / "datasets",
        project_root / "chat" / "datasets"
    ]
    
    found_bases = set()
    pattern = "*_result_DreamOf_RedChamber"
    
    for search_dir in search_locations:
        if not search_dir.exists():
            continue
            
        # Find matching dataset directories
        matches = list(search_dir.glob(pattern))
        for match in matches:
            # Check if this directory has active iterations
            iteration_dirs = list(match.glob("Graph_Iteration*"))
            if iteration_dirs:
                found_bases.add(str(match.resolve()))
    
    if len(found_bases) == 0:
        return None
    elif len(found_bases) == 1:
        return list(found_bases)[0]
    else:
        # Multiple bases found - provide helpful error
        base_list = sorted(found_bases)
        raise PathResolutionError(
            f"Multiple dataset bases found: {base_list}. "
            f"Set PIPELINE_OUTPUT_DIR or PIPELINE_DATASET_PATH to disambiguate."
        )

def resolve_pipeline_output(iteration: int, create: bool = True) -> str:
    """
    Working-directory independent output directory resolution.
    
    Uses the precedence order:
    1. PIPELINE_OUTPUT_DIR (explicit override)
    2. PIPELINE_DATASET_PATH + Graph_Iteration{iteration}
    3. Auto-detected dataset base + Graph_Iteration{iteration}
    
    Args:
        iteration: The pipeline iteration number
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        str: Absolute path to the output directory
        
    Raises:
        PathResolutionError: If resolution fails or is ambiguous
    """
    # Priority 1: Explicit PIPELINE_OUTPUT_DIR
    if 'PIPELINE_OUTPUT_DIR' in os.environ:
        output_dir = os.environ['PIPELINE_OUTPUT_DIR']
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
        
        if create and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    # Priority 2: PIPELINE_DATASET_PATH + iteration
    if 'PIPELINE_DATASET_PATH' in os.environ:
        dataset_path = os.environ['PIPELINE_DATASET_PATH']
        output_dir = os.path.join(dataset_path, f"Graph_Iteration{iteration}")
        
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
        
        if create and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    # Priority 3: Auto-detect dataset base (now working-directory independent)
    try:
        dataset_base = detect_dataset_base()
        if dataset_base:
            output_dir = os.path.join(dataset_base, f"Graph_Iteration{iteration}")
            
            if not os.path.isabs(output_dir):
                output_dir = os.path.abspath(output_dir)
            
            if create and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            return output_dir
    except PathResolutionError:
        # Re-raise auto-detection errors with additional context
        raise
    
    # If all else fails, provide a helpful error message
    project_root = find_project_root()
    if project_root:
        suggested_path = project_root / "datasets" / "KIMI_result_DreamOf_RedChamber"
        raise PathResolutionError(
            f"Cannot auto-detect dataset base. Please set PIPELINE_OUTPUT_DIR environment variable.\n"
            f"Suggested value: {suggested_path / f'Graph_Iteration{iteration}'}"
        )
    else:
        raise PathResolutionError(
            "Cannot find project root or resolve output path. "
            "Please set PIPELINE_OUTPUT_DIR environment variable."
        )

def load_manifest(directory: str) -> Optional[Dict]:
    """
    Load path manifest from a directory.
    
    Args:
        directory: Directory to load manifest from
        
    Returns:
        Dict: Manifest data, or None if not found
        
    Raises:
        ManifestError: If manifest exists but is corrupted
    """
    manifest_path = os.path.join(directory, "path_manifest.json")
    
    if not os.path.exists(manifest_path):
        return None
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Validate required fields
        required_fields = ['stage', 'iteration', 'output_dir', 'created_at']
        for field in required_fields:
            if field not in manifest:
                raise ManifestError(f"Manifest missing required field: {field}")
        
        return manifest
    
    except (json.JSONDecodeError, IOError) as e:
        raise ManifestError(f"Failed to load manifest from {manifest_path}: {e}")

def write_manifest(directory: str, stage: str, iteration: int, 
                  files: List[str], metadata: Optional[Dict] = None) -> str:
    """
    Write path manifest to a directory.
    
    Args:
        directory: Directory to write manifest to
        stage: Pipeline stage name (e.g., "ectd", "triple_generation")
        iteration: Pipeline iteration number
        files: List of files created by this stage
        metadata: Optional additional metadata
        
    Returns:
        str: Path to the written manifest file
        
    Raises:
        ManifestError: If manifest cannot be written
    """
    manifest_path = os.path.join(directory, "path_manifest.json")
    
    manifest = {
        "stage": stage,
        "iteration": iteration,
        "output_dir": os.path.abspath(directory),
        "files": files,
        "created_at": datetime.datetime.now().isoformat(),
        "resolver_version": RESOLVER_VERSION
    }
    
    if metadata:
        manifest["metadata"] = metadata
    
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return manifest_path
    
    except IOError as e:
        raise ManifestError(f"Failed to write manifest to {manifest_path}: {e}")

def validate_manifest_files(directory: str, manifest: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that files listed in manifest actually exist and have non-zero size.
    
    Args:
        directory: Directory containing the files
        manifest: Loaded manifest data
        
    Returns:
        Tuple[bool, List[str]]: (all_valid, list_of_missing_files)
    """
    missing_files = []
    
    for filename in manifest.get('files', []):
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files

def get_iteration_output_path(dataset_path: str, iteration: int) -> str:
    """
    Legacy compatibility function for existing code.
    
    Args:
        dataset_path: Base dataset path
        iteration: Iteration number
        
    Returns:
        str: Resolved output path
    """
    # Temporarily set PIPELINE_DATASET_PATH for resolution
    original_value = os.environ.get('PIPELINE_DATASET_PATH')
    os.environ['PIPELINE_DATASET_PATH'] = dataset_path
    
    try:
        return resolve_pipeline_output(iteration)
    finally:
        # Restore original value
        if original_value is not None:
            os.environ['PIPELINE_DATASET_PATH'] = original_value
        else:
            os.environ.pop('PIPELINE_DATASET_PATH', None)

def get_iteration_input_path(dataset_path: str, iteration: int) -> str:
    """
    Legacy compatibility function for existing code.
    Input path is the same as output path for the same iteration.
    
    Args:
        dataset_path: Base dataset path
        iteration: Iteration number
        
    Returns:
        str: Resolved input path
    """
    return get_iteration_output_path(dataset_path, iteration)

def log_path_diagnostics(stage: str, iteration: int, resolved_path: str):
    """
    Log standardized path diagnostic information.
    
    Args:
        stage: Current pipeline stage
        iteration: Pipeline iteration
        resolved_path: The resolved output path
    """
    print(f"[PATH_DIAG] Stage: {stage}")
    print(f"[PATH_DIAG] Iteration: {iteration}")
    print(f"[PATH_DIAG] CWD: {os.getcwd()}")
    print(f"[PATH_DIAG] PIPELINE_OUTPUT_DIR: {os.environ.get('PIPELINE_OUTPUT_DIR', '<not set>')}")
    print(f"[PATH_DIAG] PIPELINE_DATASET_PATH: {os.environ.get('PIPELINE_DATASET_PATH', '<not set>')}")
    print(f"[PATH_DIAG] Resolved path: {resolved_path}")
    
    # Check if manifest exists
    manifest = load_manifest(resolved_path)
    print(f"[PATH_DIAG] Manifest loaded: {manifest is not None}")
    
    if manifest:
        print(f"[PATH_DIAG] Manifest stage: {manifest.get('stage', 'unknown')}")
        print(f"[PATH_DIAG] Manifest iteration: {manifest.get('iteration', 'unknown')}")
        
        # Validate manifest files
        is_valid, missing = validate_manifest_files(resolved_path, manifest)
        print(f"[PATH_DIAG] Manifest files valid: {is_valid}")
        if missing:
            print(f"[PATH_DIAG] Missing files: {missing}")

# Backward compatibility aliases
resolve_output_path = resolve_pipeline_output
