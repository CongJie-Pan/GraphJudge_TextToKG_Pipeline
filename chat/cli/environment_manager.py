#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Manager for Unified CLI Pipeline Architecture

This module provides centralized environment variable management with
standardized naming conventions, validation, and default values for
all pipeline modules.

Features:
- Standardized environment variable naming
- Centralized validation and type conversion
- Default value management
- Environment variable documentation
- Cross-module compatibility

Author: Engineering Team
Date: 2025-01-15
Version: 1.0.0
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum


class EnvironmentGroup(Enum):
    """Environment variable groupings for organization."""
    PIPELINE = "pipeline"
    ECTD = "ectd"
    TRIPLE_GENERATION = "triple_generation"
    GRAPH_JUDGE_PHASE = "graph_judge_phase"
    EVALUATION = "evaluation"
    API_KEYS = "api_keys"
    SYSTEM = "system"
    LOGGING = "logging"
    CACHE = "cache"


@dataclass
class EnvironmentVariable:
    """Environment variable definition with metadata."""
    name: str
    description: str
    default: Any
    var_type: type
    required: bool = False
    group: EnvironmentGroup = EnvironmentGroup.SYSTEM
    validation_func: Optional[callable] = None


class EnvironmentManager:
    """
    Centralized environment variable management system.
    
    Provides standardized access to all environment variables used across
    the pipeline modules with consistent naming, validation, and defaults.
    """
    
    def __init__(self):
        """Initialize the environment manager with standardized variables."""
        self.variables = self._define_environment_variables()
        self._current_values = {}
        
        # Load and validate all variables
        self.refresh_environment()
        
        print("Environment Manager initialized with standardized variables")
    
    def _define_environment_variables(self) -> Dict[str, EnvironmentVariable]:
        """
        Define all standardized environment variables.
        
        Returns:
            Dictionary mapping variable names to their definitions
        """
        variables = {}
        
        # ============ PIPELINE CORE VARIABLES ============
        variables.update({
            "PIPELINE_ITERATION": EnvironmentVariable(
                name="PIPELINE_ITERATION",
                description="Current pipeline iteration number",
                default=1,
                var_type=int,
                required=True,
                group=EnvironmentGroup.PIPELINE,
                validation_func=lambda x: x > 0
            ),
            "PIPELINE_ITERATION_PATH": EnvironmentVariable(
                name="PIPELINE_ITERATION_PATH",
                description="Base path for current iteration",
                default="./docs/Iteration_Report/Iteration1",
                var_type=str,
                required=True,
                group=EnvironmentGroup.PIPELINE
            ),
            "PIPELINE_DATASET": EnvironmentVariable(
                name="PIPELINE_DATASET",
                description="Name of the dataset being processed",
                default="DreamOf_RedChamber",
                var_type=str,
                group=EnvironmentGroup.PIPELINE
            ),
            "PIPELINE_DATASET_PATH": EnvironmentVariable(
                name="PIPELINE_DATASET_PATH",
                description="Path to the dataset directory",
                default="../datasets/KIMI_result_DreamOf_RedChamber/",
                var_type=str,
                group=EnvironmentGroup.PIPELINE
            ),
            "PIPELINE_OUTPUT_BASE": EnvironmentVariable(
                name="PIPELINE_OUTPUT_BASE",
                description="Base output directory for all pipeline results",
                default="./docs/Iteration_Report",
                var_type=str,
                group=EnvironmentGroup.PIPELINE
            ),
            "PIPELINE_PARALLEL_WORKERS": EnvironmentVariable(
                name="PIPELINE_PARALLEL_WORKERS",
                description="Number of parallel workers for processing",
                default=5,
                var_type=int,
                group=EnvironmentGroup.PIPELINE,
                validation_func=lambda x: 1 <= x <= 20
            ),
        })
        
        # ============ ECTD STAGE VARIABLES ============
        variables.update({
            "ECTD_MODEL": EnvironmentVariable(
                name="ECTD_MODEL",
                description="Primary model for ECTD stage",
                default="gpt5-mini",
                var_type=str,
                group=EnvironmentGroup.ECTD,
                validation_func=lambda x: x in ["gpt5-mini", "kimi-k2", "gpt-4", "claude-3"]
            ),
            "ECTD_FALLBACK_MODEL": EnvironmentVariable(
                name="ECTD_FALLBACK_MODEL",
                description="Fallback model for ECTD stage",
                default="kimi-k2",
                var_type=str,
                group=EnvironmentGroup.ECTD
            ),
            "ECTD_TEMPERATURE": EnvironmentVariable(
                name="ECTD_TEMPERATURE",
                description="Temperature setting for ECTD model",
                default=0.3,
                var_type=float,
                group=EnvironmentGroup.ECTD,
                validation_func=lambda x: 0.0 <= x <= 2.0
            ),
            "ECTD_BATCH_SIZE": EnvironmentVariable(
                name="ECTD_BATCH_SIZE",
                description="Batch size for ECTD processing",
                default=20,
                var_type=int,
                group=EnvironmentGroup.ECTD,
                validation_func=lambda x: 1 <= x <= 100
            ),
            "ECTD_CONCURRENT_LIMIT": EnvironmentVariable(
                name="ECTD_CONCURRENT_LIMIT",
                description="Concurrent request limit for ECTD",
                default=3,
                var_type=int,
                group=EnvironmentGroup.ECTD,
                validation_func=lambda x: 1 <= x <= 10
            ),
            "ECTD_OUTPUT_DIR": EnvironmentVariable(
                name="ECTD_OUTPUT_DIR",
                description="Output directory for ECTD results",
                default="./results/ectd",
                var_type=str,
                group=EnvironmentGroup.ECTD
            ),
            "ECTD_CACHE_ENABLED": EnvironmentVariable(
                name="ECTD_CACHE_ENABLED",
                description="Enable caching for ECTD stage",
                default=True,
                var_type=bool,
                group=EnvironmentGroup.ECTD
            ),
            "ECTD_CACHE_PATH": EnvironmentVariable(
                name="ECTD_CACHE_PATH",
                description="Cache directory for ECTD stage",
                default="./cache/ectd",
                var_type=str,
                group=EnvironmentGroup.ECTD
            ),
            "ECTD_MAX_TEXT_LENGTH": EnvironmentVariable(
                name="ECTD_MAX_TEXT_LENGTH",
                description="Maximum text length for ECTD processing",
                default=8000,
                var_type=int,
                group=EnvironmentGroup.ECTD,
                validation_func=lambda x: x > 0
            )
        })
        
        # ============ TRIPLE GENERATION VARIABLES ============
        variables.update({
            "TRIPLE_BATCH_SIZE": EnvironmentVariable(
                name="TRIPLE_BATCH_SIZE",
                description="Batch size for triple generation",
                default=10,
                var_type=int,
                group=EnvironmentGroup.TRIPLE_GENERATION,
                validation_func=lambda x: 1 <= x <= 50
            ),
            "TRIPLE_SCHEMA_VALIDATION_ENABLED": EnvironmentVariable(
                name="TRIPLE_SCHEMA_VALIDATION_ENABLED",
                description="Enable schema validation for triples",
                default=True,
                var_type=bool,
                group=EnvironmentGroup.TRIPLE_GENERATION
            ),
            "TRIPLE_TEXT_CHUNKING_ENABLED": EnvironmentVariable(
                name="TRIPLE_TEXT_CHUNKING_ENABLED",
                description="Enable text chunking for triple generation",
                default=True,
                var_type=bool,
                group=EnvironmentGroup.TRIPLE_GENERATION
            ),
            "TRIPLE_CHUNK_SIZE": EnvironmentVariable(
                name="TRIPLE_CHUNK_SIZE",
                description="Size of text chunks for processing",
                default=1000,
                var_type=int,
                group=EnvironmentGroup.TRIPLE_GENERATION,
                validation_func=lambda x: x > 0
            ),
            "TRIPLE_CHUNK_OVERLAP": EnvironmentVariable(
                name="TRIPLE_CHUNK_OVERLAP",
                description="Overlap size between text chunks",
                default=200,
                var_type=int,
                group=EnvironmentGroup.TRIPLE_GENERATION,
                validation_func=lambda x: x >= 0
            ),
            "TRIPLE_OUTPUT_DIR": EnvironmentVariable(
                name="TRIPLE_OUTPUT_DIR",
                description="Output directory for triple generation results",
                default="./results/triple_generation",
                var_type=str,
                group=EnvironmentGroup.TRIPLE_GENERATION
            ),
            "TRIPLE_POST_PROCESSING_ENABLED": EnvironmentVariable(
                name="TRIPLE_POST_PROCESSING_ENABLED",
                description="Enable post-processing for triples",
                default=True,
                var_type=bool,
                group=EnvironmentGroup.TRIPLE_GENERATION
            )
        })
        
        # ============ GRAPH JUDGE PHASE VARIABLES ============ 
        variables.update({
            "GRAPH_JUDGE_MODEL": EnvironmentVariable(
                name="GRAPH_JUDGE_MODEL",
                description="Model for graph judgment (Perplexity API)",
                default="perplexity/sonar-reasoning",
                var_type=str,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: x.startswith("perplexity/")
            ),
            "GRAPH_JUDGE_TEMPERATURE": EnvironmentVariable(
                name="GRAPH_JUDGE_TEMPERATURE",
                description="Temperature for graph judgment model",
                default=0.2,
                var_type=float,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: 0.0 <= x <= 2.0
            ),
            "GRAPH_JUDGE_MAX_TOKENS": EnvironmentVariable(
                name="GRAPH_JUDGE_MAX_TOKENS",
                description="Maximum tokens for graph judgment",
                default=2000,
                var_type=int,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: x > 0
            ),
            "GRAPH_JUDGE_CONCURRENT_LIMIT": EnvironmentVariable(
                name="GRAPH_JUDGE_CONCURRENT_LIMIT",
                description="Concurrent request limit for graph judgment",
                default=3,
                var_type=int,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: 1 <= x <= 10
            ),
            "GRAPH_JUDGE_EXPLAINABLE_MODE": EnvironmentVariable(
                name="GRAPH_JUDGE_EXPLAINABLE_MODE",
                description="Enable explainable reasoning mode",
                default=False,
                var_type=bool,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE
            ),
            "GRAPH_JUDGE_BOOTSTRAP_MODE": EnvironmentVariable(
                name="GRAPH_JUDGE_BOOTSTRAP_MODE",
                description="Enable bootstrap mode for gold labels",
                default=False,
                var_type=bool,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE
            ),
            "GRAPH_JUDGE_STREAMING_MODE": EnvironmentVariable(
                name="GRAPH_JUDGE_STREAMING_MODE",
                description="Enable streaming processing mode",
                default=False,
                var_type=bool,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE
            ),
            "GRAPH_JUDGE_ENABLE_CITATIONS": EnvironmentVariable(
                name="GRAPH_JUDGE_ENABLE_CITATIONS",
                description="Enable citation tracking",
                default=True,
                var_type=bool,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE
            ),
            "GRAPH_JUDGE_CONFIDENCE_THRESHOLD": EnvironmentVariable(
                name="GRAPH_JUDGE_CONFIDENCE_THRESHOLD",
                description="Confidence threshold for judgments",
                default=0.7,
                var_type=float,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: 0.0 <= x <= 1.0
            ),
            "GRAPH_JUDGE_FUZZY_THRESHOLD": EnvironmentVariable(
                name="GRAPH_JUDGE_FUZZY_THRESHOLD",
                description="Fuzzy matching threshold for bootstrapping",
                default=0.8,
                var_type=float,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: 0.0 <= x <= 1.0
            ),
            "GRAPH_JUDGE_SAMPLE_RATE": EnvironmentVariable(
                name="GRAPH_JUDGE_SAMPLE_RATE",
                description="Sampling rate for bootstrap mode",
                default=0.15,
                var_type=float,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE,
                validation_func=lambda x: 0.0 <= x <= 1.0
            ),
            "GRAPH_JUDGE_OUTPUT_FILE": EnvironmentVariable(
                name="GRAPH_JUDGE_OUTPUT_FILE",
                description="Output CSV file for graph judgment results",
                default="./results/graph_judge/judgment_results.csv",
                var_type=str,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE
            ),
            "GRAPH_JUDGE_REASONING_FILE": EnvironmentVariable(
                name="GRAPH_JUDGE_REASONING_FILE",
                description="Output JSON file for reasoning explanations",
                default="./results/graph_judge/reasoning_explanations.json",
                var_type=str,
                group=EnvironmentGroup.GRAPH_JUDGE_PHASE
            )
        })
        
        # ============ EVALUATION VARIABLES ============
        variables.update({
            "EVALUATION_METRICS": EnvironmentVariable(
                name="EVALUATION_METRICS",
                description="Comma-separated list of evaluation metrics",
                default="triple_match_f1,graph_match_accuracy,g_bleu,g_rouge,g_bert_score",
                var_type=str,
                group=EnvironmentGroup.EVALUATION
            ),
            "EVALUATION_GOLD_STANDARD": EnvironmentVariable(
                name="EVALUATION_GOLD_STANDARD",
                description="Path to gold standard dataset",
                default="./datasets/gold_standard.json",
                var_type=str,
                group=EnvironmentGroup.EVALUATION
            ),
            "EVALUATION_OUTPUT_DIR": EnvironmentVariable(
                name="EVALUATION_OUTPUT_DIR",
                description="Output directory for evaluation results",
                default="./results/evaluation",
                var_type=str,
                group=EnvironmentGroup.EVALUATION
            )
        })
        
        # ============ API KEY VARIABLES ============
        variables.update({
            "OPENAI_API_KEY": EnvironmentVariable(
                name="OPENAI_API_KEY",
                description="OpenAI API key for GPT models",
                default="",
                var_type=str,
                required=True,
                group=EnvironmentGroup.API_KEYS
            ),
            "PERPLEXITY_API_KEY": EnvironmentVariable(
                name="PERPLEXITY_API_KEY",
                description="Perplexity API key for graph judgment",
                default="",
                var_type=str,
                required=True,
                group=EnvironmentGroup.API_KEYS
            ),
            "KIMI_API_KEY": EnvironmentVariable(
                name="KIMI_API_KEY",
                description="Kimi API key for fallback processing",
                default="",
                var_type=str,
                group=EnvironmentGroup.API_KEYS
            )
        })
        
        # ============ SYSTEM VARIABLES ============
        variables.update({
            "PYTHONIOENCODING": EnvironmentVariable(
                name="PYTHONIOENCODING",
                description="Python I/O encoding for Unicode handling",
                default="utf-8",
                var_type=str,
                group=EnvironmentGroup.SYSTEM
            ),
            "LANG": EnvironmentVariable(
                name="LANG",
                description="System language setting",
                default="en_US.UTF-8",
                var_type=str,
                group=EnvironmentGroup.SYSTEM
            ),
            "RUNTIME_ENVIRONMENT": EnvironmentVariable(
                name="RUNTIME_ENVIRONMENT",
                description="Runtime environment (development/testing/production)",
                default="development",
                var_type=str,
                group=EnvironmentGroup.SYSTEM,
                validation_func=lambda x: x in ["development", "testing", "production"]
            )
        })
        
        # ============ LOGGING VARIABLES ============
        variables.update({
            "LOG_LEVEL": EnvironmentVariable(
                name="LOG_LEVEL",
                description="Logging level for all modules",
                default="INFO",
                var_type=str,
                group=EnvironmentGroup.LOGGING,
                validation_func=lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            "LOG_DIR": EnvironmentVariable(
                name="LOG_DIR",
                description="Base directory for all log files",
                default="./logs",
                var_type=str,
                group=EnvironmentGroup.LOGGING
            ),
            "LOG_MAX_FILE_SIZE_MB": EnvironmentVariable(
                name="LOG_MAX_FILE_SIZE_MB",
                description="Maximum log file size in MB",
                default=10,
                var_type=int,
                group=EnvironmentGroup.LOGGING,
                validation_func=lambda x: x > 0
            )
        })
        
        # ============ CACHE VARIABLES ============
        variables.update({
            "CACHE_ENABLED": EnvironmentVariable(
                name="CACHE_ENABLED",
                description="Global cache enable/disable flag",
                default=True,
                var_type=bool,
                group=EnvironmentGroup.CACHE
            ),
            "CACHE_BASE_DIR": EnvironmentVariable(
                name="CACHE_BASE_DIR",
                description="Base directory for all cache files",
                default="./cache",
                var_type=str,
                group=EnvironmentGroup.CACHE
            ),
            "CACHE_MAX_SIZE_MB": EnvironmentVariable(
                name="CACHE_MAX_SIZE_MB",
                description="Maximum total cache size in MB",
                default=1000,
                var_type=int,
                group=EnvironmentGroup.CACHE,
                validation_func=lambda x: x > 0
            ),
            "CACHE_TTL_HOURS": EnvironmentVariable(
                name="CACHE_TTL_HOURS",
                description="Cache time-to-live in hours",
                default=24,
                var_type=int,
                group=EnvironmentGroup.CACHE,
                validation_func=lambda x: x > 0
            )
        })
        
        return variables
    
    def get(self, name: str, default: Any = None) -> Any:
        """
        Get environment variable value with type conversion and validation.
        
        Args:
            name: Environment variable name
            default: Default value if not found and no default defined
            
        Returns:
            Converted and validated environment variable value
        """
        # First check if value was set via set() method for any variable
        if name in self._current_values:
            return self._current_values[name]
            
        if name not in self.variables:
            # For undefined variables, return from environment or default
            return os.getenv(name, default)
        
        var_def = self.variables[name]
        
        # Get raw value from environment or use default
        raw_value = os.getenv(name)
        if raw_value is None:
            if default is not None:
                return default
            if var_def.required:
                raise ValueError(f"Required environment variable '{name}' is not set")
            return var_def.default
        
        # Convert to appropriate type
        try:
            if var_def.var_type == bool:
                converted_value = raw_value.lower() in ['true', '1', 'yes', 'on']
            elif var_def.var_type == int:
                converted_value = int(raw_value)
            elif var_def.var_type == float:
                converted_value = float(raw_value)
            else:
                converted_value = raw_value
        except (ValueError, TypeError) as e:
            print(f"WARNING: Warning: Invalid value for {name}: {raw_value}. Using default: {var_def.default}")
            return var_def.default
        
        # Validate if validation function is provided
        if var_def.validation_func and not var_def.validation_func(converted_value):
            print(f"WARNING: Warning: Invalid value for {name}: {converted_value}. Using default: {var_def.default}")
            return var_def.default
        
        return converted_value
    
    def set(self, name: str, value: Any, persist: bool = False):
        """
        Set environment variable value.
        
        Args:
            name: Environment variable name
            value: Value to set
            persist: Whether to persist to actual environment
        """
        self._current_values[name] = value
        
        if persist:
            os.environ[name] = str(value)
    
    def get_group_variables(self, group: EnvironmentGroup) -> Dict[str, Any]:
        """
        Get all variables for a specific group.
        
        Args:
            group: Environment variable group
            
        Returns:
            Dictionary of variable names to values for the group
        """
        group_vars = {}
        for name, var_def in self.variables.items():
            if var_def.group == group:
                group_vars[name] = self.get(name)
        return group_vars
    
    def refresh_environment(self):
        """Refresh all environment variables from current environment."""
        # Preserve values that were set via set() method for variables not in self.variables
        preserved_values = {}
        for name, value in self._current_values.items():
            if name not in self.variables:
                preserved_values[name] = value
        
        self._current_values = {}
        
        # Validate all required variables (directly from environment, not via get())
        missing_required = []
        for name, var_def in self.variables.items():
            if var_def.required and not os.getenv(name):
                missing_required.append(name)
        
        # Restore preserved values for undefined variables
        self._current_values.update(preserved_values)
        
        if missing_required:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_required)}")
    
    def export_template(self, file_path: str, groups: Optional[List[EnvironmentGroup]] = None):
        """
        Export environment variable template file.
        
        Args:
            file_path: Path to save template file
            groups: Optional list of groups to include (default: all)
        """
        if groups is None:
            groups = list(EnvironmentGroup)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# Unified CLI Pipeline Environment Variables\n")
            f.write("# Generated template with standardized variable names\n\n")
            
            for group in groups:
                f.write(f"# ============ {group.value.upper()} VARIABLES ============\n")
                
                group_vars = {name: var_def for name, var_def in self.variables.items() 
                             if var_def.group == group}
                
                for name, var_def in sorted(group_vars.items()):
                    f.write(f"# {var_def.description}\n")
                    if var_def.required:
                        f.write(f"# REQUIRED\n")
                    f.write(f"{name}={var_def.default}\n\n")
        
        print(f" Environment template exported to: {file_path}")
    
    def validate_all(self) -> List[str]:
        """
        Validate all environment variables and return any errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        for name, var_def in self.variables.items():
            try:
                self.get(name)
            except ValueError as e:
                errors.append(str(e))
        
        return errors
    
    def get_documentation(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive documentation for all environment variables.
        
        Returns:
            Dictionary with variable documentation
        """
        docs = {}
        
        for group in EnvironmentGroup:
            group_vars = {}
            for name, var_def in self.variables.items():
                if var_def.group == group:
                    group_vars[name] = {
                        'description': var_def.description,
                        'default': var_def.default,
                        'type': var_def.var_type.__name__,
                        'required': var_def.required,
                        'current_value': self.get(name)
                    }
            
            if group_vars:
                docs[group.value] = group_vars
        
        return docs
    
    def setup_stage_environment(self, stage_name: str, iteration: int, iteration_path: str) -> Dict[str, str]:
        """
        Setup environment variables for a specific pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            
        Returns:
            Dictionary of environment variables for the stage
        """
        # Start with system environment
        env = os.environ.copy()
        
        # Set common pipeline variables
        env.update({
            'PIPELINE_ITERATION': str(iteration),
            'PIPELINE_ITERATION_PATH': iteration_path,
            'PIPELINE_DATASET_PATH': self.get('PIPELINE_DATASET_PATH'),
            'PYTHONIOENCODING': self.get('PYTHONIOENCODING'),
            'LANG': self.get('LANG'),
            'LOG_LEVEL': self.get('LOG_LEVEL'),
            'CACHE_ENABLED': str(self.get('CACHE_ENABLED')).lower()
        })
        
        # Set stage-specific variables
        if stage_name == "ectd":
            env.update({
                'ECTD_MODEL': self.get('ECTD_MODEL'),
                'ECTD_TEMPERATURE': str(self.get('ECTD_TEMPERATURE')),
                'ECTD_BATCH_SIZE': str(self.get('ECTD_BATCH_SIZE')),
                'ECTD_CONCURRENT_LIMIT': str(self.get('ECTD_CONCURRENT_LIMIT')),
                'ECTD_OUTPUT_DIR': os.path.join(iteration_path, "results", "ectd"),
                'ECTD_CACHE_ENABLED': str(self.get('ECTD_CACHE_ENABLED')).lower()
            })
        
        elif stage_name == "triple_generation":
            env.update({
                'TRIPLE_BATCH_SIZE': str(self.get('TRIPLE_BATCH_SIZE')),
                'TRIPLE_SCHEMA_VALIDATION_ENABLED': str(self.get('TRIPLE_SCHEMA_VALIDATION_ENABLED')).lower(),
                'TRIPLE_TEXT_CHUNKING_ENABLED': str(self.get('TRIPLE_TEXT_CHUNKING_ENABLED')).lower(),
                'TRIPLE_OUTPUT_DIR': os.path.join(iteration_path, "results", "triple_generation"),
                'TRIPLE_POST_PROCESSING_ENABLED': str(self.get('TRIPLE_POST_PROCESSING_ENABLED')).lower()
            })
        
        elif stage_name == "graph_judge":
            env.update({
                'GRAPH_JUDGE_MODEL': self.get('GRAPH_JUDGE_MODEL'),
                'GRAPH_JUDGE_TEMPERATURE': str(self.get('GRAPH_JUDGE_TEMPERATURE')),
                'GRAPH_JUDGE_CONCURRENT_LIMIT': str(self.get('GRAPH_JUDGE_CONCURRENT_LIMIT')),
                'GRAPH_JUDGE_OUTPUT_FILE': os.path.join(iteration_path, "results", "graph_judge", "judgment_results.csv"),
                'GRAPH_JUDGE_EXPLAINABLE_MODE': str(self.get('GRAPH_JUDGE_EXPLAINABLE_MODE')).lower(),
                'GRAPH_JUDGE_ENABLE_CITATIONS': str(self.get('GRAPH_JUDGE_ENABLE_CITATIONS')).lower()
            })
        
        elif stage_name == "evaluation":
            env.update({
                'EVALUATION_METRICS': self.get('EVALUATION_METRICS'),
                'EVALUATION_OUTPUT_DIR': os.path.join(iteration_path, "results", "evaluation")
            })
        
        return env
