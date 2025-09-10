#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage Manager for Unified CLI Pipeline Architecture

This module wraps existing pipeline scripts and provides a unified interface
for executing the four main stages of the knowledge graph generation pipeline.

Stages:
1. ECTD (Entity Extraction & Text Denoising) - run_entity.py
2. Triple Generation - run_triple.py  
3. Graph Judge - run_gj.py
4. Evaluation - convert_Judge_To_jsonGraph.py

Author: Engineering Team
Date: 2025-01-15
Version: 1.0.0
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import pipeline state management system
try:
    # NOTE: extractEntity_Phase module is not developed yet per docs
    # from ..extractEntity_Phase.models.pipeline_state import (
    #     PipelineState, PipelineStateManager, PipelineStage as PipelineStageEnum,
    #     ProcessingStatus, PipelineError, ErrorSeverity
    # )
    PIPELINE_STATE_AVAILABLE = False  # Set to False since module doesn't exist
    print("WARNING: Pipeline state management not available (module not developed)")
    
    # Create mock classes for interface consistency
    class MockPipelineStageEnum:
        ENTITY_EXTRACTION = "entity_extraction"
        POST_PROCESSING = "post_processing"
        VALIDATION = "validation"
    
    class MockPipelineError:
        def __init__(self, stage, message, severity):
            self.stage = stage
            self.message = message
            self.severity = severity
    
    class MockErrorSeverity:
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    PipelineStageEnum = MockPipelineStageEnum()
    PipelineError = MockPipelineError
    ErrorSeverity = MockErrorSeverity()
    
except ImportError:
    PIPELINE_STATE_AVAILABLE = False
    print("WARNING: Pipeline state management not available")
    
    # Create mock classes for interface consistency
    class MockPipelineStageEnum:
        ENTITY_EXTRACTION = "entity_extraction"
        POST_PROCESSING = "post_processing"
        VALIDATION = "validation"
    
    class MockPipelineError:
        def __init__(self, stage, message, severity):
            self.stage = stage
            self.message = message
            self.severity = severity
    
    class MockErrorSeverity:
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    PipelineStageEnum = MockPipelineStageEnum()
    PipelineError = MockPipelineError
    ErrorSeverity = MockErrorSeverity()

# Import standardized environment management
try:
    from environment_manager import EnvironmentManager
except ImportError:
    try:
        from .environment_manager import EnvironmentManager
    except ImportError:
        EnvironmentManager = None
        print("WARNING: EnvironmentManager not available")

# Import enhanced stage implementations
try:
    from enhanced_ectd_stage import EnhancedECTDStage
    from enhanced_triple_stage import EnhancedTripleGenerationStage
    from graph_judge_phase_stage import GraphJudgePhaseStage
    ENHANCED_STAGES_AVAILABLE = True
except ImportError:
    print("WARNING: Enhanced stages not available, using legacy stages")
    ENHANCED_STAGES_AVAILABLE = False
    EnhancedECTDStage = None
    EnhancedTripleGenerationStage = None
    GraphJudgePhaseStage = None


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage implements the execute method to run its specific functionality
    while maintaining consistent interface and error handling.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the pipeline stage.
        
        Args:
            name: Stage name
            config: Stage-specific configuration
        """
        self.name = name
        self.config = config
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.error_message = None
        
        # Initialize env_manager to None - will be set by StageManager if available
        self.env_manager = None
        
        # Pipeline state integration
        self.pipeline_state = None
        self.stage_enum = self._get_pipeline_stage_enum(name)
    
    @abstractmethod
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """
        Execute the pipeline stage.
        
        Args:
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            **kwargs: Additional stage-specific arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def _get_pipeline_stage_enum(self, name: str):
        """
        Map stage name to pipeline stage enum.
        
        Args:
            name: Stage name
            
        Returns:
            PipelineStageEnum value or None if not available
        """
        if not PIPELINE_STATE_AVAILABLE:
            return None
            
        stage_mapping = {
            "ECTD": PipelineStageEnum.ENTITY_EXTRACTION,
            "Triple Generation": PipelineStageEnum.POST_PROCESSING,  # Closest match
            "Graph Judge": PipelineStageEnum.VALIDATION,
            "Evaluation": PipelineStageEnum.VALIDATION
        }
        return stage_mapping.get(name)
    
    def _log_stage_start(self):
        """Log stage start with pipeline state integration."""
        self.start_time = datetime.now()
        self.status = "running"
        print(f"Starting {self.name} stage at {self.start_time.strftime('%H:%M:%S')}")
        
        # Update pipeline state if available
        if PIPELINE_STATE_AVAILABLE and self.pipeline_state and self.stage_enum:
            self.pipeline_state.start_stage(self.stage_enum)
    
    def _log_stage_end(self, success: bool):
        """Log stage completion with pipeline state integration."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if success:
            self.status = "completed"
            print(f"{self.name} stage completed successfully in {duration:.1f}s")
        else:
            self.status = "failed"
            print(f"{self.name} stage failed after {duration:.1f}s")
        
        # Update pipeline state if available
        if PIPELINE_STATE_AVAILABLE and self.pipeline_state and self.stage_enum:
            self.pipeline_state.complete_stage(self.stage_enum, success)
            
            # Log error details if failed
            if not success and self.error_message:
                error = PipelineError(
                    stage=self.stage_enum,
                    message=self.error_message,
                    severity=ErrorSeverity.HIGH
                )
                self.pipeline_state.add_stage_error(self.stage_enum, error)
    
    def _validate_model_configuration(self):
        """
        Validate and enforce correct model configuration for ECTD stage.
        """
        if self.name == "ECTD":
            primary_model = self.config.get('model', 'gpt5-mini')
            force_primary = self.config.get('force_primary_model', True)
            
            if primary_model != 'gpt5-mini':
                print(f"⚠️  WARNING: Expected GPT-5-mini but configured for {primary_model}")
                if force_primary:
                    self.config['model'] = 'gpt5-mini'
                    print(f"✅ Corrected model configuration to: gpt5-mini")
                else:
                    print(f"❌ Model configuration validation failed")
                    raise ValueError(f"Invalid model configuration: {primary_model}")
            else:
                print(f"✅ Model configuration validated: {primary_model}")
    
    def _log_stage_environment(self, stage_name: str, env: Dict[str, str]):
        """
        Log stage environment configuration for debugging.
        
        Args:
            stage_name: Name of the stage
            env: Environment variables
        """
        print(f"\n🔧 {stage_name} Environment Configuration:")
        print(f"   Pipeline Iteration: {env.get('PIPELINE_ITERATION', 'unknown')}")
        print(f"   Output Directory: {env.get('PIPELINE_OUTPUT_DIR', 'unknown')}")
        print(f"   Dataset Path: {env.get('PIPELINE_DATASET_PATH', 'unknown')}")
        if stage_name == "ECTD":
            print(f"   ECTD Output Directory: {env.get('ECTD_OUTPUT_DIR', 'unknown')}")
        print(f"   Encoding: {env.get('PYTHONIOENCODING', 'default')}")
    
    def _get_script_path(self, script_name: str) -> str:
        """
        Get absolute path to a script in the chat directory.
        
        Args:
            script_name: Name of the script file
            
        Returns:
            Absolute path to the script
        """
        current_dir = Path(__file__).parent
        chat_dir = current_dir.parent
        script_path = chat_dir / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        return str(script_path)
    
    def _create_dynamic_wrapper(self, original_script: str, iteration_path: str, 
                               stage_name: str, additional_vars: Dict[str, Any] = None) -> str:
        """
        Create a dynamic wrapper script with proper encoding and variable injection.
        
        This method addresses multiple Unicode and Windows path issues:
        1. UTF-8 BOM encoding for wrapper files to prevent encoding errors
        2. Proper Windows path escaping to avoid escape sequence warnings
        3. RuntimeError prevention when iterating over locals()
        
        Args:
            original_script: Path to the original script to wrap
            iteration_path: Path to iteration directory
            stage_name: Name of the current stage
            additional_vars: Additional variables to inject
            
        Returns:
            Path to the created wrapper script
        """
        # Reason: Escape backslashes in Windows paths to prevent escape sequence warnings
        original_script_escaped = original_script.replace('\\', '\\\\')
        iteration_path_escaped = iteration_path.replace('\\', '\\\\')
        
        # Create wrapper content with proper variable injection
        wrapper_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic wrapper for {stage_name} stage
Auto-generated wrapper to handle encoding and path issues
"""

import os
import sys
from pathlib import Path
import importlib.util

# Set up environment
current_dir = Path(__file__).parent
chat_dir = current_dir if current_dir.name == "cli" else current_dir.parent

# Add chat directory to path
sys.path.insert(0, str(chat_dir))

# Set up additional environment variables
if {additional_vars is not None}:
    additional_vars = {additional_vars or {}}
    for key, value in additional_vars.items():
        os.environ[key] = str(value)

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Execute the original script
if __name__ == "__main__":
    # Import and run the original script's main function
    script_path = Path("{original_script_escaped}")
    
    # Add the script's directory to sys.path
    sys.path.insert(0, str(script_path.parent))
    
    # Import the module
    spec = importlib.util.spec_from_file_location("target_module", script_path)
    target_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(target_module)
        
        # Try to run main function if it exists
        if hasattr(target_module, 'main'):
            target_module.main()
        elif hasattr(target_module, '__main__'):
            # Execute as script
            exec(compile(open(script_path).read(), script_path, 'exec'))
    except Exception as e:
        print(f"Error executing {{script_path}}: {{e}}")
        sys.exit(1)
'''
        
        # Create temporary wrapper file
        wrapper_fd, wrapper_path = tempfile.mkstemp(suffix='.py', prefix=f'{stage_name.lower().replace(" ", "_")}_wrapper_')
        
        try:
            # Write wrapper content with UTF-8 encoding
            with os.fdopen(wrapper_fd, 'w', encoding='utf-8') as wrapper_file:
                wrapper_file.write(wrapper_content)
        except Exception:
            # Clean up on error
            os.close(wrapper_fd)
            if os.path.exists(wrapper_path):
                os.remove(wrapper_path)
            raise
        
        return wrapper_path

    def _setup_stage_environment(self, stage_name: str, iteration: int, iteration_path: str) -> Dict[str, str]:
        """
        Setup standardized environment variables for stage execution.
        
        Args:
            stage_name: Name of the stage
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            
        Returns:
            Dictionary of environment variables
        """
        if self.env_manager:
            # Use standardized environment management
            return self.env_manager.setup_stage_environment(stage_name, iteration, iteration_path)
        else:
            # Fallback to manual setup if environment manager not available
            env = os.environ.copy()
            
            # Set UTF-8 encoding for subprocess to handle Unicode paths
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            
            # Common environment variables for all stages
            env['PIPELINE_ITERATION'] = str(iteration)
            env['PIPELINE_ITERATION_PATH'] = iteration_path
            env['PIPELINE_DATASET_PATH'] = f"../datasets/KIMI_result_DreamOf_RedChamber/"
            
            # Unified output directory configuration based on original scripts
            if stage_name == "ectd":
                # Use centralized path resolver to eliminate hardcoded dataset prefixes
                try:
                    import sys
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                    from path_resolver import resolve_pipeline_output
                    resolved_output = resolve_pipeline_output(iteration, create=True)
                    env['PIPELINE_OUTPUT_DIR'] = resolved_output
                    env['ECTD_OUTPUT_DIR'] = resolved_output  # Backward compatibility
                except Exception as e:
                    print(f"Warning: Path resolution failed, using legacy method: {e}")
                    # Fallback to legacy hardcoded path
                    dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                    env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                    env['ECTD_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                # Also set the iteration path for backward compatibility
                env['PIPELINE_ITERATION_PATH'] = iteration_path
            elif stage_name == "triple_generation":
                dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                env['TRIPLE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
            elif stage_name == "graph_judge":
                dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                env['GRAPH_JUDGE_OUTPUT_FILE'] = f"{dataset_base}Graph_Iteration{iteration}/judgment_results.csv"
            else:  # evaluation
                env['EVALUATION_OUTPUT_DIR'] = os.path.join(iteration_path, "results", "evaluation")
            
            return env
    
    def _validate_stage_output(self, stage_name: str, env: Dict[str, str]) -> bool:
        """
        Enhanced validation with multiple path checking for better compatibility.
        
        Args:
            stage_name: Name of the stage
            env: Environment variables used during execution
            
        Returns:
            True if all expected files exist, False otherwise
        """
        expected_files = []
        
        if stage_name == "ectd":
            # Enhanced validation using manifest-first approach
            try:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                from path_resolver import load_manifest, validate_manifest_files
                
                # Try to load manifest first (highest priority)
                primary_output_dir = env.get('ECTD_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', ''))
                manifest = None
                working_location = None
                
                if primary_output_dir:
                    manifest = load_manifest(primary_output_dir)
                    if manifest and manifest.get('stage') == 'ectd':
                        is_valid, missing = validate_manifest_files(primary_output_dir, manifest)
                        if is_valid:
                            working_location = primary_output_dir
                            expected_files = [os.path.join(primary_output_dir, f) for f in manifest.get('files', [])]
                            print(f"✅ Found valid manifest in: {primary_output_dir}")
                
                if not working_location:
                    # Fallback to legacy multi-location checking
                    legacy_output_dir = os.path.join(env.get('PIPELINE_ITERATION_PATH', ''), "results", "ectd")
                    
                    # Also check the actual working directory where run_entity.py puts files
                    current_working_dir = os.getcwd()
                    actual_output_locations = [
                        os.path.join(current_working_dir, "../datasets/KIMI_result_DreamOf_RedChamber", f"Graph_Iteration{env.get('PIPELINE_ITERATION', '1')}"),
                        primary_output_dir,
                        legacy_output_dir
                    ]
                    
                    # Normalize paths consistently - preserve relative paths if they were relative
                    normalized_locations = []
                    for location in actual_output_locations:
                        if location:
                            if os.path.isabs(location):
                                normalized_locations.append(os.path.normpath(location))
                            else:
                                # Convert relative to absolute for better checking
                                normalized_locations.append(os.path.abspath(os.path.normpath(location)))
                    
                    # Add delay to ensure file system operations complete
                    import time
                    time.sleep(0.5)  # 500ms buffer for file system operations
                    
                    # Check all potential locations
                    file_found = False
                    
                    for location in normalized_locations:
                        if not location:
                            continue
                            
                        files_to_check = [
                            os.path.join(location, "test_entity.txt"),
                            os.path.join(location, "test_denoised.target")
                        ]
                        
                        print(f"🔍 Checking location: {location}")
                        location_files_exist = True
                        for f in files_to_check:
                            exists = os.path.exists(f)
                            size = os.path.getsize(f) if exists else 0
                            print(f"   {'✓' if exists and size > 0 else '✗'} {os.path.basename(f)} ({'exists' if exists else 'missing'}, {size} bytes)")
                            if not exists or size == 0:
                                location_files_exist = False
                        
                        if location_files_exist:
                            expected_files = files_to_check
                            working_location = location
                            file_found = True
                            print(f"✅ Found all output files in location: {location}")
                            break
                    
                    if not file_found:
                        print(f"❌ Output files not found in any checked locations:")
                        for i, location in enumerate(normalized_locations):
                            if location:
                                label = ["Actual", "Primary", "Legacy"][i] if i < 3 else f"Location {i+1}"
                                print(f"   {label}: {location}")
                        print(f"   Note: Files must exist AND have non-zero size")
                        return False
                
                # Update environment with working location for next stages
                if working_location:
                    env['VALIDATED_ECTD_OUTPUT_DIR'] = working_location
                    # Store in pipeline state for subsequent stages
                    self.pipeline_environment_state['VALIDATED_ECTD_OUTPUT_DIR'] = working_location
                    print(f"🔗 Stored validated ECTD output path for next stages: {working_location}")
                    return True
                
            except Exception as e:
                print(f"Warning: Enhanced validation failed, falling back to legacy: {e}")
                # Fall through to legacy validation logic below
                pass
                
        elif stage_name == "triple_generation":
            # Use validated ECTD output directory if available, otherwise fall back to configured paths
            primary_output_dir = env.get('VALIDATED_ECTD_OUTPUT_DIR') or env.get('TRIPLE_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', ''))
            legacy_output_dir = os.path.join(env.get('PIPELINE_ITERATION_PATH', ''), "results", "triple_generation")
            
            # Check for GPT5-mini output format first
            gpt5_files = [
                os.path.join(primary_output_dir, "test_instructions_context_gpt5mini_v2.json")
            ]
            
            # Check for legacy Kimi format
            kimi_files = [
                os.path.join(primary_output_dir, "test_instructions_context_kimi_v2.json"),
                os.path.join(legacy_output_dir, "test_instructions_context_kimi_v2.json")
            ]
            
            print(f"🔍 Checking triple generation files in: {primary_output_dir}")
            for f in gpt5_files + kimi_files[:1]:  # Check main files
                exists = os.path.exists(f)
                size = os.path.getsize(f) if exists else 0
                print(f"   {'✓' if exists and size > 0 else '✗'} {os.path.basename(f)} ({'exists' if exists else 'missing'}, {size} bytes)")
            
            if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in gpt5_files):
                expected_files = gpt5_files
                print(f"✅ Found GPT5-mini triple generation output")
            elif any(os.path.exists(f) and os.path.getsize(f) > 0 for f in kimi_files):
                expected_files = [f for f in kimi_files if os.path.exists(f) and os.path.getsize(f) > 0]
                print(f"✅ Found Kimi triple generation output")
            else:
                print(f"❌ No valid triple generation output files found")
                print(f"   Expected GPT5-mini format: {gpt5_files}")
                print(f"   Expected Kimi format: {kimi_files}")
                return False
                
        elif stage_name == "graph_judge":
            output_file = env.get('GRAPH_JUDGE_OUTPUT_FILE', env.get('PIPELINE_OUTPUT_FILE', ''))
            if output_file and os.path.exists(output_file):
                expected_files = [output_file]
            else:
                print(f"❌ Graph judge output file not found: {output_file}")
                return False
                
        elif stage_name == "evaluation":
            output_dir = env.get('EVALUATION_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', ''))
            # Check for any JSON files created by the evaluation stage
            if os.path.exists(output_dir):
                json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
                if json_files:
                    expected_files = [os.path.join(output_dir, json_files[0])]
        
        # Log successful validation with file sizes
        if expected_files:
            print(f"✅ {stage_name} stage - All expected output files validated:")
            for file_path in expected_files:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"   ✓ {file_path} ({file_size:,} bytes)")
            return True
        
        return False
        
        # Validate file existence and log results
        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"WARNING: {stage_name} stage - Missing expected output files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            return False
        else:
            print(f"SUCCESS: {stage_name} stage - All expected output files created successfully:")
            for file_path in expected_files:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"   - {file_path} ({file_size:,} bytes)")
            return True
    
    def _log_stage_environment(self, stage_name: str, env: Dict[str, str]) -> None:
        """
        Log detailed environment variables for debugging.
        
        Args:
            stage_name: Name of the stage
            env: Environment variables dictionary
        """
        print(f"\n{stage_name} Stage Environment Configuration:")
        
        # Log pipeline-specific environment variables
        pipeline_vars = {k: v for k, v in env.items() if k.startswith('PIPELINE_')}
        for var_name, var_value in pipeline_vars.items():
            print(f"   {var_name}: {var_value}")
        
        # Log other relevant environment variables
        other_vars = ['PYTHONIOENCODING', 'LANG', 'ECTD_CONCURRENT_LIMIT', 'TRIPLE_BATCH_SIZE', 
                     'GRAPH_JUDGE_CONCURRENT_LIMIT', 'LOG_LEVEL', 'CACHE_ENABLED']
        for var_name in other_vars:
            if var_name in env:
                print(f"   {var_name}: {env[var_name]}")

    
    async def _safe_subprocess_exec(self, cmd_args: List[str], env: Dict[str, str], 
                                  cwd: str, stage_name: str) -> tuple[int, str]:
        """
        Execute subprocess with real-time output streaming and robust error handling.
        
        Implements timeout protection with asyncio.wait_for(process.communicate(), timeout=1800)
        and fallback encoding using stdout.decode('latin-1', errors='replace') for Unicode errors.
        
        Args:
            cmd_args: Command arguments list
            env: Environment variables
            cwd: Working directory
            stage_name: Name of the stage for logging
            
        Returns:
            Tuple of (return_code, output_text)
        """
        try:
            print(f"🚀 Starting {stage_name} stage execution...")
            print(f" [DEBUG] Executing command: {' '.join(cmd_args[:2])}...")
            print(f" [DEBUG] Working directory: {cwd}")
            print(f" [DEBUG] Environment encoding: {env.get('PYTHONIOENCODING', 'default')}")
            print(f"📝 Real-time progress will be displayed below:")
            print(f"{'='*80}")
            
            # Create subprocess with real-time output
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=cwd
            )
            
            output_lines = []
            start_time = datetime.now()
            
            # Stream output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                    
                try:
                    decoded_line = line.decode('utf-8').rstrip()
                except UnicodeDecodeError:
                    # Fallback encoding for handling stdout.decode('latin-1', errors='replace')
                    decoded_line = line.decode('latin-1', errors='replace').rstrip()
                
                if decoded_line:
                    # Display real-time progress with timestamp
                    current_time = datetime.now()
                    elapsed = (current_time - start_time).total_seconds()
                    timestamp = current_time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] [Elapsed: {elapsed:.1f}s] {decoded_line}")
                    output_lines.append(decoded_line)
                    
                    # Flush output immediately for real-time display
                    sys.stdout.flush()
            
            # Wait for process completion with timeout protection
            try:
                await asyncio.wait_for(process.communicate(), timeout=1800)
            except asyncio.TimeoutError:
                print(f"⚠️ Process timeout after 1800 seconds, terminating...")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                return 1, "Process terminated due to timeout"
            
            full_output = '\n'.join(output_lines)
            
            print(f"{'='*80}")
            print(f"✅ {stage_name} stage execution completed with return code: {process.returncode}")
            
            return process.returncode, full_output
            
        except Exception as e:
            error_msg = f"Real-time subprocess execution failed for {stage_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return 1, error_msg


class ECTDStage(PipelineStage):
    """
    Entity Extraction & Text Denoising Stage
    
    Wraps run_entity.py to provide ECTD functionality within the unified pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ECTD", config)
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """Execute ECTD stage using run_entity.py."""
        self._log_stage_start()
        
        try:
            # Validate model configuration before execution
            self._validate_model_configuration()
            
            # Get script path
            script_path = self._get_script_path("run_entity.py")
            
            # Setup environment variables for direct script execution
            env = self._setup_stage_environment("ectd", iteration, iteration_path)
            
            # Create output directory if it doesn't exist
            output_dir = env.get('ECTD_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', './results/ectd'))
            os.makedirs(output_dir, exist_ok=True)
            
            # Only set default ECTD_OUTPUT_DIR if it's completely missing from environment
            if 'ECTD_OUTPUT_DIR' not in env:
                # Set default based on iteration path structure
                default_output_dir = os.path.join(os.path.dirname(iteration_path), f"datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration{iteration}")
                env['ECTD_OUTPUT_DIR'] = default_output_dir
                os.makedirs(default_output_dir, exist_ok=True)
                print(f"⚙️  Set ECTD_OUTPUT_DIR to: {default_output_dir}")
            else:
                print(f"⚙️  Using existing ECTD_OUTPUT_DIR: {env['ECTD_OUTPUT_DIR']}")
            
            # Add any stage-specific environment variables
            if 'parallel_workers' in kwargs:
                env['ECTD_CONCURRENT_LIMIT'] = str(kwargs['parallel_workers'])
            
            # Log environment configuration for debugging
            print(f"🔧 ECTD Environment Variables:")
            print(f"   ECTD_OUTPUT_DIR: {env.get('ECTD_OUTPUT_DIR')}")
            print(f"   PIPELINE_ITERATION_PATH: {env.get('PIPELINE_ITERATION_PATH')}")
            print(f"   PIPELINE_OUTPUT_DIR: {env.get('PIPELINE_OUTPUT_DIR')}")
            self._log_stage_environment("ECTD", env)
            
            # Execute the script directly
            print(f"\n🔧 Executing: python {script_path}")
            print(f"⚙️  Configuration: {self.config}")
            
            # Run the script directly using safe subprocess execution
            cmd_args = [sys.executable, script_path]
            return_code, output_text = await self._safe_subprocess_exec(
                cmd_args=cmd_args,
                env=env,
                cwd=str(Path(script_path).parent),
                stage_name="ECTD"
            )
            
            # Check if successful
            if return_code == 0:
                print("\nECTD Stage Output:")
                print(output_text)
                
                # Validate output files were created
                validation_success = self._validate_stage_output("ectd", env)
                
                if validation_success:
                    print("\n ECTD stage completed successfully with all expected outputs!")
                    self._log_stage_end(True)
                    return True
                else:
                    print("\n ECTD stage completed but missing expected output files!")
                    self.error_message = "Missing expected output files"
                    self._log_stage_end(False)
                    return False
            else:
                self.error_message = f"Script exited with code {return_code}"
                print(f"\n ECTD Error: {self.error_message}")
                print(output_text)
                self._log_stage_end(False)
                return False
                
        except Exception as e:
            self.error_message = str(e)
            print(f"ERROR: ECTD Exception: {e}")
            self._log_stage_end(False)
            return False


class TripleGenerationStage(PipelineStage):
    """
    Triple Generation Stage
    
    Wraps run_triple.py to provide enhanced triple generation within the unified pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Triple Generation", config)
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """Execute Triple Generation stage using run_triple.py."""
        self._log_stage_start()
        
        try:
            # Get script path
            script_path = self._get_script_path("run_triple.py")
            
            # Setup environment variables for direct script execution
            env = self._setup_stage_environment("triple_generation", iteration, iteration_path)
            
            # Add any stage-specific environment variables
            if 'batch_size' in kwargs:
                env['TRIPLE_BATCH_SIZE'] = str(kwargs['batch_size'])
            
            # Create output directory if it doesn't exist
            output_dir = env.get('TRIPLE_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', './results/triple_generation'))
            os.makedirs(output_dir, exist_ok=True)
            
            # Log environment configuration for debugging
            self._log_stage_environment("Triple Generation", env)
            
            # Execute the script directly
            print(f"\n Executing: python {script_path}")
            print(f" Configuration: {self.config}")
            
            # Run the script directly using safe subprocess execution
            cmd_args = [sys.executable, script_path]
            return_code, output_text = await self._safe_subprocess_exec(
                cmd_args=cmd_args,
                env=env,
                cwd=str(Path(script_path).parent),
                stage_name="Triple Generation"
            )
            
            # Check if successful
            if return_code == 0:
                print("\nTriple Generation Stage Output:")
                print(output_text)
                
                # Validate output files were created
                validation_success = self._validate_stage_output("triple_generation", env)
                
                if validation_success:
                    print("\n Triple Generation stage completed successfully with all expected outputs!")
                    self._log_stage_end(True)
                    return True
                else:
                    print("\n Triple Generation stage completed but missing expected output files!")
                    self.error_message = "Missing expected output files"
                    self._log_stage_end(False)
                    return False
            else:
                self.error_message = f"Script exited with code {return_code}"
                print(f"\n Triple Generation Error: {self.error_message}")
                print(output_text)
                self._log_stage_end(False)
                return False
                
        except Exception as e:
            self.error_message = str(e)
            print(f"ERROR: Triple Generation Exception: {e}")
            self._log_stage_end(False)
            return False


class GraphJudgeStage(PipelineStage):
    """
    Graph Judge Stage
    
    Integrates the modular graphJudge_Phase system to provide graph judgment 
    with explainable reasoning within the unified pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Graph Judge", config)
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """Execute Graph Judge stage using modular graphJudge_Phase system."""
        self._log_stage_start()
        
        try:
            # Setup environment variables
            env = self._setup_stage_environment("graph_judge", iteration, iteration_path)
            
            # Create output directory if it doesn't exist
            output_file = env.get('GRAPH_JUDGE_OUTPUT_FILE', env.get('PIPELINE_OUTPUT_FILE', './results/graph_judge/judgment_results.csv'))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Set up environment variables for the process
            for key, value in env.items():
                os.environ[key] = value
            
            # Log environment configuration for debugging
            self._log_stage_environment("Graph Judge", env)
            
            # Execute using modular graphJudge_Phase system via main module
            cmd_args = [sys.executable, "-m", "graphJudge_Phase.main"]
            
            # Add explainable mode if specified
            if kwargs.get('explainable', self.config.get('explainable_mode', False)):
                cmd_args.append('--explainable')
            
            # Add reasoning file path if specified
            if 'reasoning_file' in kwargs:
                cmd_args.extend(['--reasoning-file', kwargs['reasoning_file']])
            
            # Add bootstrap mode if specified
            if kwargs.get('bootstrap', self.config.get('bootstrap_mode', False)):
                cmd_args.append('--bootstrap')
                if 'triples_file' in kwargs:
                    cmd_args.extend(['--triples-file', kwargs['triples_file']])
                if 'source_file' in kwargs:
                    cmd_args.extend(['--source-file', kwargs['source_file']])
                if 'bootstrap_output' in kwargs:
                    cmd_args.extend(['--output', kwargs['bootstrap_output']])
            
            # Add model configuration if specified
            if 'model' in kwargs:
                cmd_args.extend(['--model', kwargs['model']])
            
            # Add verbose logging if specified
            if kwargs.get('verbose', self.config.get('verbose', False)):
                cmd_args.append('--verbose')
            
            print(f"\n Executing modular GraphJudge: {' '.join(cmd_args)}")
            print(f" Configuration: {self.config}")
            
            # Get working directory (parent of graphJudge_Phase)
            current_dir = Path(__file__).parent.parent  # chat directory
            
            # Run the modular system using safe subprocess execution
            return_code, output_text = await self._safe_subprocess_exec(
                cmd_args=cmd_args,
                env=env,
                cwd=str(current_dir),
                stage_name="Graph Judge (Modular)"
            )
            
            # Check if successful
            if return_code == 0:
                print("\nGraph Judge Stage Output:")
                print(output_text)
                
                # Validate output files were created
                validation_success = self._validate_stage_output("graph_judge", env)
                
                if validation_success:
                    print("\n Graph Judge stage completed successfully with all expected outputs!")
                    self._log_stage_end(True)
                    return True
                else:
                    print("\n Graph Judge stage completed but missing expected output files!")
                    self.error_message = "Missing expected output files"
                    self._log_stage_end(False)
                    return False
            else:
                self.error_message = f"Modular system exited with code {return_code}"
                print(f"\n Graph Judge Error: {self.error_message}")
                print(output_text)
                self._log_stage_end(False)
                return False
                
        except Exception as e:
            self.error_message = str(e)
            print(f"ERROR: Graph Judge Exception: {e}")
            self._log_stage_end(False)
            return False


class EvaluationStage(PipelineStage):
    """
    Evaluation Stage
    
    Wraps convert_Judge_To_jsonGraph.py and other evaluation scripts
    to provide comprehensive evaluation within the unified pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Evaluation", config)
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """Execute Evaluation stage using convert_Judge_To_jsonGraph.py."""
        self._log_stage_start()
        
        try:
            # Get script path
            script_path = self._get_script_path("convert_Judge_To_jsonGraph.py")
            
            # Setup environment variables for direct script execution
            env = self._setup_stage_environment("evaluation", iteration, iteration_path)
            
            # Create output directory if it doesn't exist  
            output_dir = env.get('EVALUATION_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', './results/evaluation'))
            os.makedirs(output_dir, exist_ok=True)
            
            # Log environment configuration for debugging
            self._log_stage_environment("Evaluation", env)
            
            # Execute the script directly
            print(f"\n Executing: python {script_path}")
            print(f" Configuration: {self.config}")
            
            # Run the script directly using safe subprocess execution
            cmd_args = [sys.executable, script_path]
            return_code, output_text = await self._safe_subprocess_exec(
                cmd_args=cmd_args,
                env=env,
                cwd=str(Path(script_path).parent),
                stage_name="Evaluation"
            )
            
            # Check if successful
            if return_code == 0:
                print("\nEvaluation Stage Output:")
                print(output_text)
                
                # Validate output files were created
                validation_success = self._validate_stage_output("evaluation", env)
                
                if validation_success:
                    print("\n Evaluation stage completed successfully with all expected outputs!")
                    self._log_stage_end(True)
                    return True
                else:
                    print("\n Evaluation stage completed but missing expected output files!")
                    self.error_message = "Missing expected output files"
                    self._log_stage_end(False)
                    return False
            else:
                self.error_message = f"Script exited with code {return_code}"
                print(f"\n Evaluation Error: {self.error_message}")
                print(output_text)
                self._log_stage_end(False)
                return False
                
        except Exception as e:
            self.error_message = str(e)
            print(f"ERROR: Evaluation Exception: {e}")
            self._log_stage_end(False)
            return False


class StageManager:
    """
    Manages pipeline stage execution.
    
    This class orchestrates the execution of all pipeline stages with
    proper dependency handling, error recovery, and progress tracking.
    Enhanced to support both legacy and enhanced stage implementations.
    """
    
    def __init__(self, config):
        """
        Initialize the stage manager with pipeline state and environment integration.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        
        # Initialize pipeline state for inter-stage communication
        self.pipeline_environment_state = {}
        
        # Initialize standardized environment management
        if EnvironmentManager:
            try:
                self.env_manager = EnvironmentManager()
            except Exception as e:
                print(f"WARNING: Failed to initialize EnvironmentManager: {e}")
                # Create a mock EnvironmentManager to maintain interface consistency
                class MockEnvironmentManager:
                    def __init__(self):
                        self.variables = {}
                        self._current_values = {}
                    def get(self, name, default=None):
                        # Check if value was set via set() method first
                        if name in self._current_values:
                            return self._current_values[name]
                        # Fall back to checking actual environment, then default
                        return os.getenv(name, default)
                    def set(self, name, value, persist=False):
                        self._current_values[name] = value
                        if persist:
                            os.environ[name] = str(value)
                    def validate_all(self):
                        return []
                    def refresh_environment(self):
                        pass
                    def setup_stage_environment(self, stage_name, iteration, iteration_path):
                        """Mock implementation of setup_stage_environment."""
                        import os
                        env = os.environ.copy()
                        
                        # Set UTF-8 encoding for subprocess to handle Unicode paths
                        env['PYTHONIOENCODING'] = 'utf-8'
                        env['LANG'] = 'en_US.UTF-8'
                        
                        env['PIPELINE_ITERATION'] = str(iteration)
                        env['PIPELINE_ITERATION_PATH'] = iteration_path
                        env['PIPELINE_DATASET_PATH'] = f"../datasets/KIMI_result_DreamOf_RedChamber/"
                        
                        # Unified output directory configuration to match Phase 3 expectations
                        dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                        unified_output_dir = f"{dataset_base}Graph_Iteration{iteration}"
                        
                        # Stage-specific environment variables with unified paths
                        if stage_name == "ectd":
                            env['ECTD_OUTPUT_DIR'] = unified_output_dir
                            env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                        elif stage_name == "triple_generation":
                            env['TRIPLE_OUTPUT_DIR'] = unified_output_dir
                            env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                        elif stage_name == "graph_judge":
                            env['GRAPH_JUDGE_OUTPUT_FILE'] = f"{unified_output_dir}/judgment_results.csv"
                            env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                        else:  # evaluation
                            env['EVALUATION_OUTPUT_DIR'] = unified_output_dir
                            env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                        
                        return env
                self.env_manager = MockEnvironmentManager()
                print("WARNING: Using mock EnvironmentManager")
        else:
            # Create a mock EnvironmentManager to maintain interface consistency
            class MockEnvironmentManager:
                def __init__(self):
                    self.variables = {}
                    self._current_values = {}
                def get(self, name, default=None):
                    # Check if value was set via set() method first
                    if name in self._current_values:
                        return self._current_values[name]
                    # Fall back to checking actual environment, then default
                    return os.getenv(name, default)
                def set(self, name, value, persist=False):
                    self._current_values[name] = value
                    if persist:
                        os.environ[name] = str(value)
                def validate_all(self):
                    return []
                def refresh_environment(self):
                    pass
                def get_environment_dict(self):
                    return os.environ.copy()
                def setup_stage_environment(self, stage_name, iteration, iteration_path):
                    """Mock implementation of setup_stage_environment."""
                    import os
                    env = os.environ.copy()
                    
                    # Set UTF-8 encoding for subprocess to handle Unicode paths
                    env['PYTHONIOENCODING'] = 'utf-8'
                    env['LANG'] = 'en_US.UTF-8'
                    
                    env['PIPELINE_ITERATION'] = str(iteration)
                    env['PIPELINE_ITERATION_PATH'] = iteration_path
                    env['PIPELINE_DATASET_PATH'] = f"../datasets/KIMI_result_DreamOf_RedChamber/"
                    
                    # Unified output directory configuration to match Phase 3 expectations
                    dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                    unified_output_dir = f"{dataset_base}Graph_Iteration{iteration}"
                    
                    # Stage-specific environment variables with unified paths
                    if stage_name == "ectd":
                        env['ECTD_OUTPUT_DIR'] = unified_output_dir
                        env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                    elif stage_name == "triple_generation":
                        env['TRIPLE_OUTPUT_DIR'] = unified_output_dir
                        env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                    elif stage_name == "graph_judge":
                        env['GRAPH_JUDGE_OUTPUT_FILE'] = f"{unified_output_dir}/judgment_results.csv"
                        env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                    else:  # evaluation
                        env['EVALUATION_OUTPUT_DIR'] = unified_output_dir
                        env['PIPELINE_OUTPUT_DIR'] = unified_output_dir
                    
                    return env
            self.env_manager = MockEnvironmentManager()
            print("WARNING: Using mock EnvironmentManager")
        
        # Initialize stages with enhanced implementations if available
        self.stages = self._initialize_stages()
        
        # Define stage execution order
        self.stage_order = ['ectd', 'triple_generation', 'graph_judge', 'evaluation']
        
        print(f"✓ Stage Manager initialized with {len(self.stages)} stages")
        if ENHANCED_STAGES_AVAILABLE:
            print("  - Enhanced stages available and loaded")
        else:
            print("  - Using legacy stage implementations")
    
    def _initialize_stages(self) -> Dict[str, PipelineStage]:
        """
        Initialize pipeline stages with enhanced implementations if available.
        
        Returns:
            Dict[str, PipelineStage]: Dictionary of initialized stages
        """
        stages = {}
        
        # Initialize ECTD stage
        if ENHANCED_STAGES_AVAILABLE and EnhancedECTDStage:
            stages['ectd'] = EnhancedECTDStage(self.config.ectd_config or {})
            print("  ✓ Enhanced ECTD Stage loaded")
        else:
            stages['ectd'] = ECTDStage(self.config.ectd_config or {})
            print("  ✓ Legacy ECTD Stage loaded")
        
        # Initialize Triple Generation stage
        if ENHANCED_STAGES_AVAILABLE and EnhancedTripleGenerationStage:
            stages['triple_generation'] = EnhancedTripleGenerationStage(self.config.triple_generation_config or {})
            print("  ✓ Enhanced Triple Generation Stage loaded")
        else:
            stages['triple_generation'] = TripleGenerationStage(self.config.triple_generation_config or {})
            print("  ✓ Legacy Triple Generation Stage loaded")
        
        # Initialize Graph Judge stage
        if ENHANCED_STAGES_AVAILABLE and GraphJudgePhaseStage:
            stages['graph_judge'] = GraphJudgePhaseStage(self.config.graph_judge_phase_config or {})
            print("  ✓ Enhanced Graph Judge Phase Stage loaded")
        else:
            stages['graph_judge'] = GraphJudgeStage(self.config.graph_judge_phase_config or {})
            print("  ✓ Legacy Graph Judge Stage loaded")
        
        # Initialize Evaluation stage (no enhanced version yet)
        stages['evaluation'] = EvaluationStage(self.config.evaluation_config or {})
        print("  ✓ Evaluation Stage loaded")
        
        # Set env_manager reference for all stages
        for stage_name, stage in stages.items():
            stage.env_manager = self.env_manager
            print(f"  ✓ Environment manager assigned to {stage_name} stage")
        
        return stages
    
    def _setup_stage_environment(self, stage_name: str, iteration: int, iteration_path: str) -> Dict[str, str]:
        """
        Setup standardized environment variables for stage execution.
        
        This method provides unified environment setup for all stages, ensuring
        consistent environment variable configuration across the pipeline with
        inter-stage communication support.
        
        Args:
            stage_name: Name of the stage
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            
        Returns:
            Dictionary of environment variables
        """
        if self.env_manager:
            # Use standardized environment management
            env = self.env_manager.setup_stage_environment(stage_name, iteration, iteration_path)
        else:
            # Fallback to manual setup if environment manager not available
            import os
            env = os.environ.copy()
            
            # Set UTF-8 encoding for subprocess to handle Unicode paths
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            
            # Common environment variables for all stages
            env['PIPELINE_ITERATION'] = str(iteration)
            env['PIPELINE_ITERATION_PATH'] = iteration_path
            env['PIPELINE_DATASET_PATH'] = f"../datasets/KIMI_result_DreamOf_RedChamber/"
        
        # Apply pipeline state from previous stages
        env.update(self.pipeline_environment_state)
        
        # Use validated output directory from previous stage if available
        if stage_name == "triple_generation" and 'VALIDATED_ECTD_OUTPUT_DIR' in self.pipeline_environment_state:
            env['TRIPLE_INPUT_DIR'] = self.pipeline_environment_state['VALIDATED_ECTD_OUTPUT_DIR']
            env['TRIPLE_OUTPUT_DIR'] = self.pipeline_environment_state['VALIDATED_ECTD_OUTPUT_DIR']
            env['PIPELINE_OUTPUT_DIR'] = self.pipeline_environment_state['VALIDATED_ECTD_OUTPUT_DIR']
            print(f"🔗 Using validated ECTD output for triple generation: {env['TRIPLE_INPUT_DIR']}")
        
        # Continue with original stage-specific setup
        if not self.env_manager:  # Only do manual setup if no environment manager
            if stage_name == "ectd":
                # Match run_entity.py expected output structure
                dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                env['ECTD_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                # Also set the iteration path for backward compatibility
                env['PIPELINE_ITERATION_PATH'] = iteration_path
            elif stage_name == "triple_generation":
                dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                env['TRIPLE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
            elif stage_name == "graph_judge":
                dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                env['GRAPH_JUDGE_OUTPUT_FILE'] = f"{dataset_base}Graph_Iteration{iteration}/judgment_results.csv"
            else:  # evaluation
                dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
                env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
                env['EVALUATION_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
        
        return env
    
    def select_stage_variant(self, stage_name: str, model_type: str = None, mode: str = None) -> str:
        """
        Select appropriate stage variant based on configuration.
        
        Args:
            stage_name: Base stage name
            model_type: Model type preference
            mode: Operation mode
            
        Returns:
            str: Actual stage name to use
        """
        if not ENHANCED_STAGES_AVAILABLE:
            return stage_name
        
        # Map stage names to enhanced versions when appropriate
        if stage_name == 'ectd' and model_type in ['gpt5-mini', 'enhanced']:
            return 'ectd'  # Enhanced ECTD stage handles model selection internally
        elif stage_name == 'triple_generation':
            return 'triple_generation'  # Enhanced triple generation
        elif stage_name == 'graph_judge':
            return 'graph_judge'  # Enhanced graph judge phase
        
        return stage_name
    
    def configure_stage_mode(self, stage_name: str, mode_config: Dict[str, Any]) -> None:
        """
        Configure stage operation mode for enhanced stages.
        
        Args:
            stage_name: Name of the stage
            mode_config: Mode configuration dictionary
        """
        stage = self.stages.get(stage_name)
        if not stage:
            return
        
        # Configure enhanced stages with specific integration for GraphJudge Phase
        if hasattr(stage, 'update_configuration'):
            stage.update_configuration(mode_config)
        elif stage_name == 'graph_judge' and isinstance(stage, GraphJudgePhaseStage):
            # Enhanced GraphJudge Phase integration with modular architecture support
            if hasattr(stage, 'configure_modular_system'):
                stage.configure_modular_system(mode_config)
                print(f"  ✓ GraphJudge Phase modular system configured")
            # Direct configuration for modular components
            if hasattr(stage, 'graph_judge') and hasattr(stage.graph_judge, 'update_config'):
                stage.graph_judge.update_config(mode_config)
                print(f"  ✓ PerplexityGraphJudge component configured")
            if hasattr(stage, 'bootstrapper') and hasattr(stage.bootstrapper, 'update_config'):
                stage.bootstrapper.update_config(mode_config)
                print(f"  ✓ GoldLabelBootstrapper component configured")
        elif stage_name == 'graph_judge' and hasattr(stage, 'graph_judge'):
            # Fallback for legacy GraphJudge integration
            if hasattr(stage.graph_judge, 'update_config'):
                stage.graph_judge.update_config(mode_config)
    
    def get_enhanced_stage_info(self) -> Dict[str, Any]:
        """
        Get information about enhanced stage availability and capabilities.
        
        Returns:
            Dict[str, Any]: Enhanced stage information
        """
        info = {
            'enhanced_stages_available': ENHANCED_STAGES_AVAILABLE,
            'stage_capabilities': {}
        }
        
        for stage_name, stage in self.stages.items():
            capabilities = []
            
            if isinstance(stage, EnhancedECTDStage):
                capabilities.extend(['gpt5-mini', 'caching', 'rate-limiting', 'validation'])
            elif isinstance(stage, EnhancedTripleGenerationStage):
                capabilities.extend(['schema-validation', 'text-chunking', 'post-processing', 'quality-metrics'])
            elif isinstance(stage, GraphJudgePhaseStage):
                capabilities.extend(['explainable-reasoning', 'gold-label-bootstrapping', 'streaming', 'modular-architecture'])
            else:
                capabilities.append('legacy')
            
            info['stage_capabilities'][stage_name] = capabilities
        
        return info
    
    def validate_configuration(self) -> bool:
        """
        Validate pipeline configuration for all stages.
        
        Returns:
            bool: True if all configurations are valid, False otherwise
        """
        print(f"\n🔍 Validating Pipeline Configuration...")
        
        validation_results = []
        
        # Validate ECTD configuration
        if hasattr(self.config, 'validate_model_configuration'):
            try:
                ectd_valid = self.config.validate_model_configuration()
                validation_results.append(('ECTD Model', ectd_valid))
            except Exception as e:
                print(f"❌ ECTD configuration validation failed: {e}")
                validation_results.append(('ECTD Model', False))
        
        # Validate environment manager
        if self.env_manager:
            try:
                env_issues = self.env_manager.validate_all() if hasattr(self.env_manager, 'validate_all') else []
                env_valid = len(env_issues) == 0
                validation_results.append(('Environment', env_valid))
                if not env_valid:
                    print(f"⚠️ Environment issues found: {env_issues}")
            except Exception as e:
                print(f"❌ Environment validation failed: {e}")
                validation_results.append(('Environment', False))
        
        # Validate stage availability
        for stage_name, stage in self.stages.items():
            stage_valid = stage is not None
            validation_results.append((f'Stage {stage_name}', stage_valid))
        
        # Log validation summary
        all_valid = all(result[1] for result in validation_results)
        if all_valid:
            print(f"✅ All configuration validations passed")
        else:
            failed_components = [name for name, valid in validation_results if not valid]
            print(f"❌ Configuration validation failed for: {', '.join(failed_components)}")
        
        return all_valid
    
    def get_stage_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report for all stages.
        
        Returns:
            Dict[str, Any]: Stage status report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'enhanced_stages_available': ENHANCED_STAGES_AVAILABLE,
            'stages': {}
        }
        
        for stage_name, stage in self.stages.items():
            stage_info = {
                'name': stage.name,
                'status': stage.status,
                'config': stage.config,
                'error_message': stage.error_message,
                'start_time': stage.start_time.isoformat() if stage.start_time else None,
                'end_time': stage.end_time.isoformat() if stage.end_time else None,
                'duration': None
            }
            
            # Calculate duration if both times are available
            if stage.start_time and stage.end_time:
                duration = (stage.end_time - stage.start_time).total_seconds()
                stage_info['duration'] = f"{duration:.1f}s"
            
            # Add enhanced stage specific info
            if hasattr(stage, 'get_capabilities'):
                stage_info['capabilities'] = stage.get_capabilities()
            
            report['stages'][stage_name] = stage_info
        
        return report
    
    def log_configuration_summary(self):
        """Log a comprehensive configuration summary."""
        print(f"\n📊 Pipeline Configuration Summary:")
        
        if hasattr(self.config, 'log_configuration_summary'):
            self.config.log_configuration_summary()
        else:
            print(f"   Configuration object: {type(self.config).__name__}")
        
        # Log stage information
        print(f"\n🔧 Stage Information:")
        for stage_name, stage in self.stages.items():
            stage_type = "Enhanced" if "Enhanced" in type(stage).__name__ else "Legacy"
            print(f"   {stage_name}: {stage_type} ({type(stage).__name__})")
        
        # Log environment manager status
        env_status = "Available" if self.env_manager and not isinstance(self.env_manager, type(None)) else "Mock"
        print(f"   Environment Manager: {env_status}")
        
        print(f"   Enhanced Stages Available: {ENHANCED_STAGES_AVAILABLE}")
    
    async def execute_stage(self, stage_name: str, iteration: int, iteration_path: str, **kwargs) -> bool:
        """
        Execute a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage to execute
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            **kwargs: Additional stage-specific arguments
            
        Returns:
            bool: True if successful, False otherwise
        """
        if stage_name not in self.stages:
            print(f"ERROR: Unknown stage: {stage_name}")
            return False
        
        stage = self.stages[stage_name]
        print(f"\n{'='*60}")
        print(f" EXECUTING STAGE: {stage.name}")
        print(f"{'='*60}")
        
        try:
            success = await stage.execute(iteration, iteration_path, **kwargs)
            if success:
                print(f"✅ {stage.name} completed successfully")
            else:
                print(f"❌ {stage.name} failed: {stage.error_message}")
            return success
        except Exception as e:
            print(f"❌ {stage.name} exception: {str(e)}")
            return False
    
    async def run_stages(self, input_file: str, iteration: int, iteration_path: str, 
                        start_from_stage: Optional[str] = None) -> bool:
        """
        Run all pipeline stages in sequence.
        
        Args:
            input_file: Path to input file
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            start_from_stage: Stage to start from (for recovery)
            
        Returns:
            True if all stages completed successfully
        """
        print(f"🚀 Starting pipeline execution for Iteration {iteration}")
        print(f"  - Working directory: {iteration_path}")
        print(f"  - Input file: {input_file}")
        
        # Determine starting point
        start_index = 0
        if start_from_stage:
            try:
                start_index = self.stage_order.index(start_from_stage)
                print(f"  - Resuming from stage: {self.stage_order[start_index]}")
            except (ValueError, IndexError):
                print(f"  - Invalid start stage '{start_from_stage}', starting from beginning")
                start_index = 0
        
        # Execute stages in order
        failed_stage = None
        for i in range(start_index, len(self.stage_order)):
            stage_name = self.stage_order[i]
            stage = self.stages[stage_name]
            
            print(f"\n{'='*60}")
            print(f"📋 STAGE {i+1}/{len(self.stage_order)}: {stage.name}")
            print(f"{'='*60}")
            
            # Execute stage
            success = await self.execute_stage(stage_name, iteration, iteration_path, input_file=input_file)
            
            if not success:
                failed_stage = stage_name
                print(f"\n❌ Pipeline failed at stage: {stage_name}")
                break
            
            print(f"\n✅ Stage {i+1}/{len(self.stage_order)} completed successfully")
        
        # Final summary
        if failed_stage:
            print(f"\n{'='*60}")
            print(f"❌ PIPELINE FAILED")
            print(f"📍 Failed at stage: {failed_stage}")
            print(f"🔄 To resume: use --start-from-stage {failed_stage}")
            print(f"{'='*60}")
            return False
        else:
            print(f"\n{'='*60}")
            print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY")
            print(f"📁 Results location: {iteration_path}")
            print(f"{'='*60}")
            return True
    
    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get a stage instance by name."""
        return self.stages.get(stage_name)
    
    def get_stage_status(self, stage_name: str) -> Dict[str, Any]:
        """
        Get status information for a stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dictionary with status information
        """
        if stage_name not in self.stages:
            return {"error": "Stage not found"}
        
        stage = self.stages[stage_name]
        return {
            "name": stage.name,
            "status": stage.status,
            "start_time": stage.start_time.isoformat() if stage.start_time else None,
            "end_time": stage.end_time.isoformat() if stage.end_time else None,
            "error_message": stage.error_message
        }
    
    def get_all_stage_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all stages.
        
        Returns:
            Dictionary mapping stage names to status information
        """
        return {name: self.get_stage_status(name) for name in self.stages.keys()}

    async def run_single_stage(self, stage_name: str, iteration: int, iteration_path: str, **kwargs) -> bool:
        """
        Run a single pipeline stage.
        
        This method provides the interface expected by the CLI for running individual stages.
        It delegates to the execute_stage method for actual execution.
        
        Args:
            stage_name: Name of the stage to run
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            **kwargs: Additional stage-specific arguments
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🔧 Running single stage: {stage_name}")
        return await self.execute_stage(stage_name, iteration, iteration_path, **kwargs)
    
    def reset_stage_status(self, stage_name: Optional[str] = None):
        """
        Reset status for a stage or all stages.
        
        Args:
            stage_name: Name of stage to reset, or None for all stages
        """
        if stage_name:
            if stage_name in self.stages:
                stage = self.stages[stage_name]
                stage.status = "pending"
                stage.start_time = None
                stage.end_time = None
                stage.error_message = None
                print(f"🔄 Reset status for stage: {stage_name}")
            else:
                print(f"❌ Unknown stage: {stage_name}")
        else:
            for stage in self.stages.values():
                stage.status = "pending"
                stage.start_time = None
                stage.end_time = None
                stage.error_message = None
            print("🔄 Reset status for all stages")
    
    def validate_stage_dependencies(self) -> List[str]:
        """
        Validate that all stage dependencies are met.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check if all required scripts/modules exist
        script_mapping = {
            'ectd': 'run_entity.py',
            'triple_generation': 'run_triple.py',
            'graph_judge': 'graphJudge_Phase',  # Now uses modular system
            'evaluation': 'convert_Judge_To_jsonGraph.py'
        }
        
        for stage_name, script_name in script_mapping.items():
            try:
                if stage_name == 'graph_judge':
                    # Check if graphJudge_Phase module exists
                    module_path = Path(__file__).parent.parent / "graphJudge_Phase"
                    if not module_path.exists() or not (module_path / "__init__.py").exists():
                        errors.append(f"Missing graphJudge_Phase module for {stage_name}")
                else:
                    # For enhanced stages, check if they can access the original scripts
                    stage = self.stages.get(stage_name)
                    if hasattr(stage, '_get_script_path'):
                        stage._get_script_path(script_name)
            except (FileNotFoundError, AttributeError) as e:
                errors.append(f"Missing script for {stage_name}: {e}")
        
        return errors

    def create_dynamic_wrapper(self, script_path: str, iteration_path: str, **kwargs) -> str:
        """
        Create dynamic wrapper with UTF-8 BOM encoding.
        
        Args:
            script_path: Path to the original script
            iteration_path: Path to iteration directory
            **kwargs: Additional parameters
            
        Returns:
            str: Content of the dynamic wrapper
        """
        # Path escaping for Windows compatibility
        original_script_escaped = script_path.replace('\\', '\\\\')
        iteration_path_escaped = iteration_path.replace('\\', '\\\\')
        
        # Create wrapper content with UTF-8 BOM encoding support
        wrapper_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic wrapper for script execution with enhanced error handling.
Auto-generated by StageManager for safe script execution.
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'

def execute_with_error_handling():
    """Execute script with comprehensive error handling."""
    try:
        # Prevent RuntimeError: dictionary changed size during iteration
        local_vars = dict(locals())
        
        # Set up execution environment
        script_path = r"{original_script_escaped}"
        iteration_path = r"{iteration_path_escaped}"
        
        # Inject variables into global scope (excluding internal vars)
        for var_name, var_value in local_vars.items():
            if var_name not in ['local_vars', 'var_name', 'var_value']:
                globals()[var_name] = var_value
        
        # Execute the original script
        with open(script_path, 'r', encoding='utf-8-sig') as f:
            script_content = f.read()
        
        exec(script_content, globals())
        
    except Exception as e:
        print(f"ERROR in dynamic wrapper: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    execute_with_error_handling()
'''
        
        return wrapper_content

    def _escape_paths(self, original_script: str, iteration_path: str) -> tuple:
        """
        Escape Windows paths properly for use in dynamic wrappers.
        
        Args:
            original_script: Original script path
            iteration_path: Iteration path
            
        Returns:
            tuple: (escaped_script_path, escaped_iteration_path)
        """
        original_script_escaped = original_script.replace('\\', '\\\\')
        iteration_path_escaped = iteration_path.replace('\\', '\\\\')
        
        return original_script_escaped, iteration_path_escaped

    def _prevent_runtime_error(self):
        """
        Prevent RuntimeError in dynamic wrapper by using local variable snapshot.
        
        This method demonstrates the pattern used in dynamic wrappers to prevent
        'RuntimeError: dictionary changed size during iteration' when iterating
        over locals().
        """
        # Create a snapshot of locals() to prevent RuntimeError
        local_vars = dict(locals())
        
        # Safe iteration over the snapshot
        for var_name, var_value in local_vars.items():
            if var_name not in ['local_vars', 'var_name', 'var_value']:
                # Safe to use var_name and var_value here
                pass
        
        return local_vars
