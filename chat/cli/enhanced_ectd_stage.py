#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced ECTD Stage Implementation for Unified CLI Pipeline

This module implements the EnhancedECTDStage class that integrates run_entity.py
functionality with advanced features including GPT-5-mini support, intelligent
caching, rate limiting, and comprehensive error handling.

Features:
- GPT-5-mini and Kimi model support with fallback
- SHA256-based intelligent caching system
- Rate limiting and retry mechanisms
- Comprehensive terminal logging
- Input validation and error handling
- Pipeline state integration

Author: Engineering Team
Date: 2025-09-07
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the chat directory to the path to import run_entity functionality
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

from stage_manager import PipelineStage
from environment_manager import EnvironmentManager

# Import run_entity components
try:
    from run_entity import TerminalLogger, EntityExtractor, main as run_entity_main
except ImportError:
    # Handle case where run_entity is not directly importable
    TerminalLogger = None
    EntityExtractor = None
    run_entity_main = None


class EnhancedECTDStage(PipelineStage):
    """
    Enhanced ECTD Stage with GPT-5-mini support and advanced caching.
    
    This stage integrates the run_entity.py functionality into the unified CLI
    pipeline with enhanced features for production use.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Enhanced ECTD Stage.
        
        Args:
            config: Stage-specific configuration dictionary
        """
        super().__init__("Enhanced ECTD", config)
        
        # Model configuration
        self.model_type = config.get('model_type', 'gpt5-mini')
        self.fallback_model = config.get('fallback_model', 'kimi-k2')
        self.temperature = config.get('temperature', 0.3)
        
        # Performance configuration
        self.batch_size = config.get('batch_size', 20)
        self.parallel_workers = config.get('parallel_workers', 5)
        self.concurrent_limit = config.get('concurrent_limit', 3)
        
        # Caching configuration
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_type = config.get('cache_type', 'sha256')
        self.cache_path = config.get('cache_path', './cache/ectd')
        
        # Rate limiting configuration
        self.rate_limiting_enabled = config.get('rate_limiting_enabled', True)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.base_delay = config.get('base_delay', 0.5)
        
        # Input validation configuration
        self.max_text_length = config.get('max_text_length', 8000)
        self.encoding_method = config.get('encoding_method', 'utf-8')
        
        # Initialize cache directory
        if self.cache_enabled:
            Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize terminal logger
        self.terminal_logger = None
        
        print(f"✓ Enhanced ECTD Stage initialized with model: {self.model_type}")
        print(f"  - Cache enabled: {self.cache_enabled}")
        print(f"  - Parallel workers: {self.parallel_workers}")
        print(f"  - Rate limiting: {self.rate_limiting_enabled}")
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """
        Execute the Enhanced ECTD stage.
        
        Args:
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            **kwargs: Additional execution parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_stage_start()
        
        try:
            # Setup environment for this execution
            env_vars = self._setup_stage_environment(iteration, iteration_path, **kwargs)
            
            # Initialize terminal logger
            log_dir = os.path.join(iteration_path, "logs", "stages")
            os.makedirs(log_dir, exist_ok=True)
            self.terminal_logger = self._create_terminal_logger(log_dir)
            
            # Determine input file
            input_file = kwargs.get('input_file') or env_vars.get('PIPELINE_INPUT_FILE')
            if not input_file:
                input_file = self._find_input_file(iteration_path)
            
            if not input_file or not os.path.exists(input_file):
                self.error_message = f"Input file not found: {input_file}"
                self.terminal_logger.log(f"❌ Error: {self.error_message}")
                return False
            
            # Validate input file
            if not self._validate_input_file(input_file):
                self.error_message = "Input file validation failed"
                return False
            
            # Setup output paths
            output_dir = os.path.join(iteration_path, "results", "ectd")
            os.makedirs(output_dir, exist_ok=True)
            
            entity_output = os.path.join(output_dir, "test_entity.txt")
            denoised_output = os.path.join(output_dir, "test_denoised.target")
            
            # Execute ECTD with model selection
            if self.model_type == 'gpt5-mini':
                success = await self._execute_gpt5mini_ectd(
                    input_file, entity_output, denoised_output, env_vars
                )
            else:
                success = await self._execute_kimi_ectd(
                    input_file, entity_output, denoised_output, env_vars
                )
            
            if success:
                # Validate outputs
                if not self._validate_outputs(entity_output, denoised_output):
                    self.error_message = "Output validation failed"
                    success = False
                else:
                    self.terminal_logger.log(f"✅ ECTD stage completed successfully")
                    self.terminal_logger.log(f"  - Entity output: {entity_output}")
                    self.terminal_logger.log(f"  - Denoised output: {denoised_output}")
            
            # Cleanup
            if self.terminal_logger:
                self.terminal_logger.close()
            
            self._log_stage_end(success)
            return success
            
        except Exception as e:
            self.error_message = f"ECTD stage execution failed: {str(e)}"
            if self.terminal_logger:
                self.terminal_logger.log(f"❌ Error: {self.error_message}")
                self.terminal_logger.close()
            
            self._log_stage_end(False)
            return False
    
    def _setup_stage_environment(self, iteration: int, iteration_path: str, **kwargs) -> Dict[str, str]:
        """
        Setup environment variables for stage execution.
        
        Args:
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, str]: Environment variables
        """
        # Get base environment from EnvironmentManager
        if EnvironmentManager:
            env_manager = EnvironmentManager()
            env_vars = env_manager.get_environment_dict()
        else:
            env_vars = os.environ.copy()
        
        # Set stage-specific variables
        env_vars.update({
            'PIPELINE_ITERATION': str(iteration),
            'PIPELINE_ITERATION_PATH': iteration_path,
            'ECTD_MODEL': self.model_type,
            'ECTD_TEMPERATURE': str(self.temperature),
            'ECTD_BATCH_SIZE': str(self.batch_size),
            'ECTD_CACHE_ENABLED': str(self.cache_enabled),
            'ECTD_CACHE_PATH': self.cache_path,
            'PYTHONIOENCODING': 'utf-8',
            'LANG': 'en_US.UTF-8'
        })
        
        return env_vars
    
    def _create_terminal_logger(self, log_dir: str):
        """
        Create a terminal logger for this stage execution.
        
        Args:
            log_dir: Directory for log files
            
        Returns:
            Terminal logger instance
        """
        # Create a simple logger if TerminalLogger is not available
        class SimpleLogger:
            def __init__(self, log_path):
                self.log_path = log_path
                self.log_file = open(log_path, 'w', encoding='utf-8')
                self.log_file.write(f"Enhanced ECTD Stage Log - {datetime.now()}\n")
                self.log_file.write("="*50 + "\n")
                
            def log(self, message):
                print(message)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"[{timestamp}] {message}\n")
                self.log_file.flush()
                
            def close(self):
                if self.log_file:
                    self.log_file.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"enhanced_ectd_{timestamp}.log")
        
        if TerminalLogger:
            return TerminalLogger(log_dir)
        else:
            return SimpleLogger(log_path)
    
    def _find_input_file(self, iteration_path: str) -> Optional[str]:
        """
        Find the input file for ECTD processing.
        
        Args:
            iteration_path: Path to iteration directory
            
        Returns:
            Optional[str]: Path to input file or None
        """
        # Look for common input file patterns
        possible_inputs = [
            os.path.join(iteration_path, "input.txt"),
            os.path.join(iteration_path, "raw_text.txt"),
            os.path.join(iteration_path, "source.txt")
        ]
        
        for input_file in possible_inputs:
            if os.path.exists(input_file):
                return input_file
        
        # Look in parent directories
        parent_dir = Path(iteration_path).parent
        dataset_dirs = ["datasets", "data", "input"]
        
        for dataset_dir in dataset_dirs:
            dataset_path = parent_dir / dataset_dir
            if dataset_path.exists():
                for file_path in dataset_path.glob("*.txt"):
                    return str(file_path)
        
        return None
    
    def _validate_input_file(self, input_file: str) -> bool:
        """
        Validate the input file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check file exists and is readable
            if not os.path.exists(input_file):
                return False
            
            # Check file size
            file_size = os.path.getsize(input_file)
            if file_size == 0:
                return False
            
            # Check encoding
            with open(input_file, 'r', encoding=self.encoding_method) as f:
                content = f.read()
                
            # Check content length
            if len(content) > self.max_text_length:
                self.terminal_logger.log(f"⚠️  Warning: Input file exceeds max length ({len(content)} > {self.max_text_length})")
                # Don't fail, just warn
            
            return True
            
        except Exception as e:
            if self.terminal_logger:
                self.terminal_logger.log(f"❌ Input validation error: {str(e)}")
            return False
    
    async def _execute_gpt5mini_ectd(self, input_file: str, entity_output: str, 
                                   denoised_output: str, env_vars: Dict[str, str]) -> bool:
        """
        Execute ECTD using GPT-5-mini model.
        
        Args:
            input_file: Path to input file
            entity_output: Path to entity output file
            denoised_output: Path to denoised output file
            env_vars: Environment variables
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting GPT-5-mini ECTD processing")
            self.terminal_logger.log(f"  - Input: {input_file}")
            self.terminal_logger.log(f"  - Model: {self.model_type}")
            self.terminal_logger.log(f"  - Cache enabled: {self.cache_enabled}")
            
            # Use subprocess to run the original script with environment
            import subprocess
            
            # Create wrapper script that calls run_entity.py
            wrapper_script = self._create_dynamic_wrapper(
                original_script=str(chat_dir / "run_entity.py"),
                iteration_path=os.path.dirname(entity_output),
                stage_name="Enhanced ECTD",
                additional_vars={
                    'INPUT_FILE': input_file,
                    'ENTITY_OUTPUT': entity_output,
                    'DENOISED_OUTPUT': denoised_output
                }
            )
            
            # Execute the wrapper script
            process = await asyncio.create_subprocess_exec(
                sys.executable, wrapper_script,
                env=env_vars,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(chat_dir)
            )
            
            stdout, stderr = await process.communicate()
            
            # Log output
            if stdout:
                self.terminal_logger.log(f"📄 STDOUT:\n{stdout.decode('utf-8', errors='ignore')}")
            if stderr:
                self.terminal_logger.log(f"⚠️  STDERR:\n{stderr.decode('utf-8', errors='ignore')}")
            
            # Clean up wrapper script
            if os.path.exists(wrapper_script):
                os.remove(wrapper_script)
            
            success = process.returncode == 0
            
            if success:
                self.terminal_logger.log(f"✅ GPT-5-mini ECTD completed successfully")
            else:
                self.terminal_logger.log(f"❌ GPT-5-mini ECTD failed with return code: {process.returncode}")
            
            return success
            
        except Exception as e:
            self.terminal_logger.log(f"❌ GPT-5-mini ECTD execution error: {str(e)}")
            return False
    
    async def _execute_kimi_ectd(self, input_file: str, entity_output: str, 
                               denoised_output: str, env_vars: Dict[str, str]) -> bool:
        """
        Execute ECTD using Kimi model (fallback).
        
        Args:
            input_file: Path to input file
            entity_output: Path to entity output file
            denoised_output: Path to denoised output file
            env_vars: Environment variables
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting Kimi ECTD processing (fallback)")
            self.terminal_logger.log(f"  - Input: {input_file}")
            self.terminal_logger.log(f"  - Model: {self.fallback_model}")
            
            # Set Kimi-specific environment variables
            env_vars['ECTD_MODEL'] = self.fallback_model
            
            # Use subprocess to run the original script
            import subprocess
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(chat_dir / "run_entity.py"),
                env=env_vars,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(chat_dir)
            )
            
            stdout, stderr = await process.communicate()
            
            # Log output
            if stdout:
                self.terminal_logger.log(f"📄 STDOUT:\n{stdout.decode('utf-8', errors='ignore')}")
            if stderr:
                self.terminal_logger.log(f"⚠️  STDERR:\n{stderr.decode('utf-8', errors='ignore')}")
            
            success = process.returncode == 0
            
            if success:
                self.terminal_logger.log(f"✅ Kimi ECTD completed successfully")
            else:
                self.terminal_logger.log(f"❌ Kimi ECTD failed with return code: {process.returncode}")
            
            return success
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Kimi ECTD execution error: {str(e)}")
            return False
    
    def _validate_outputs(self, entity_output: str, denoised_output: str) -> bool:
        """
        Validate the generated outputs.
        
        Args:
            entity_output: Path to entity output file
            denoised_output: Path to denoised output file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check entity output
            if not os.path.exists(entity_output):
                self.terminal_logger.log(f"❌ Entity output file not found: {entity_output}")
                return False
            
            # Check denoised output
            if not os.path.exists(denoised_output):
                self.terminal_logger.log(f"❌ Denoised output file not found: {denoised_output}")
                return False
            
            # Check file sizes
            entity_size = os.path.getsize(entity_output)
            denoised_size = os.path.getsize(denoised_output)
            
            if entity_size == 0:
                self.terminal_logger.log(f"❌ Entity output file is empty")
                return False
            
            if denoised_size == 0:
                self.terminal_logger.log(f"❌ Denoised output file is empty")
                return False
            
            # Check content quality (basic validation)
            with open(entity_output, 'r', encoding='utf-8') as f:
                entity_content = f.read()
            
            with open(denoised_output, 'r', encoding='utf-8') as f:
                denoised_content = f.read()
            
            # Log statistics
            self.terminal_logger.log(f"📊 Output Statistics:")
            self.terminal_logger.log(f"  - Entity output: {len(entity_content)} characters")
            self.terminal_logger.log(f"  - Denoised output: {len(denoised_content)} characters")
            
            return True
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Output validation error: {str(e)}")
            return False
