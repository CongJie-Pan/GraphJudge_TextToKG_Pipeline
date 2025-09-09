#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Manager for Unified CLI Pipeline Architecture

This module handles configuration file creation, loading, and management
for iteration-specific pipeline configurations.

Features:
- Base configuration template management
- Iteration-specific configuration generation
- Configuration validation and merging
- Dynamic path resolution

Author: Engineering Team
Date: 2025-01-15
Version: 1.0.0
"""

import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class PipelineConfig:
    """Enhanced pipeline configuration data structure with comprehensive module support."""
    iteration: Optional[int] = None
    parallel_workers: int = 5
    checkpoint_frequency: int = 10
    error_tolerance: float = 0.1
    auto_create_directories: bool = True
    base_output_path: str = "./docs/Iteration_Report"
    
    # Stage configurations
    ectd_config: Dict[str, Any] = None
    triple_generation_config: Dict[str, Any] = None
    graph_judge_phase_config: Dict[str, Any] = None  # Updated for modular system
    evaluation_config: Dict[str, Any] = None
    
    # Pipeline state management integration
    pipeline_state_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize comprehensive default stage configurations if not provided."""
        if self.ectd_config is None:
            # Enhanced ECTD configuration with GPT-5-mini and Kimi support
            self.ectd_config = {
                'model': 'gpt5-mini',  # Primary model for enhanced processing
                'fallback_model': 'kimi-k2',  # Only for fallback scenarios
                'model_priority': ['gpt5-mini', 'kimi-k2'],  # Explicit priority order
                'force_primary_model': True,  # Prevent automatic fallback
                'validate_model_availability': True,  # Check model accessibility
                'temperature': 0.3,
                'batch_size': 20,
                'parallel_workers': 5,
                'cache_enabled': True,
                'cache_type': 'sha256',  # SHA256-based cache keys
                'cache_path': './cache/ectd',
                'rate_limiting_enabled': True,
                'retry_attempts': 3,
                'base_delay': 0.5,
                'max_text_length': 8000,
                'encoding_method': 'utf-8'
            }
        
        if self.triple_generation_config is None:
            # Enhanced triple generation with schema validation and chunking
            self.triple_generation_config = {
                'output_format': 'json',
                'validation_enabled': True,
                'schema_validation_enabled': True,
                'relation_mapping': './config/relation_map.json',
                'text_chunking_enabled': True,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'post_processing_enabled': True,
                'duplicate_removal_enabled': True,
                'entity_linking_enabled': True,
                'confidence_scoring_enabled': True,
                'batch_processing_size': 10
            }
        
        if self.graph_judge_phase_config is None:
            # Comprehensive graphJudge_Phase modular system configuration
            self.graph_judge_phase_config = {
                # Core GraphJudge configuration
                'explainable_mode': False,
                'bootstrap_mode': False,
                'streaming_mode': False,
                'model_name': 'perplexity/sonar-reasoning',
                'reasoning_effort': 'medium',
                'enable_console_logging': False,
                'temperature': 0.2,
                'max_tokens': 2000,
                
                # API and performance settings
                'concurrent_limit': 3,
                'retry_attempts': 3,
                'base_delay': 0.5,
                'enable_citations': True,
                
                # Gold label bootstrapping specific settings
                'fuzzy_threshold': 0.8,
                'sample_rate': 0.15,
                'llm_batch_size': 10,
                'max_source_lines': 1000,
                'random_seed': 42,
                
                # Advanced processing options
                'enable_progress_tracking': True,
                'output_reasoning_files': False,
                'confidence_threshold': 0.7,
                'evidence_sources': ['source_text', 'domain_knowledge'],
                'alternative_suggestions_enabled': False
            }
        
        if self.evaluation_config is None:
            self.evaluation_config = {
                'metrics': ['triple_match_f1', 'graph_match_accuracy', 'g_bleu', 'g_rouge', 'g_bert_score'],
                'gold_standard': './datasets/gold_standard.json',
                'comprehensive_reporting': True,
                'export_formats': ['json', 'csv', 'html']
            }
        
        if self.pipeline_state_config is None:
            # Pipeline state management integration
            self.pipeline_state_config = {
                'enabled': True,
                'state_file': './pipeline_state.json',
                'auto_save_enabled': True,
                'auto_save_interval': 30,  # seconds
                'error_tracking_enabled': True,
                'progress_monitoring_enabled': True,
                'recovery_mode_enabled': True,
                'detailed_logging_enabled': True
            }
    
    def validate_model_configuration(self) -> bool:
        """
        Validate and enforce correct model configuration.
        
        Returns:
            True if model configuration is valid, False otherwise
        """
        primary_model = self.ectd_config.get('model', 'gpt5-mini')
        force_primary = self.ectd_config.get('force_primary_model', True)
        
        if primary_model != 'gpt5-mini':
            print(f"⚠️  WARNING: Expected GPT-5-mini but configured for {primary_model}")
            if force_primary:
                self.ectd_config['model'] = 'gpt5-mini'
                print(f"✅ Corrected model configuration to: gpt5-mini")
                return True
            else:
                print(f"❌ Model configuration validation failed")
                return False
        
        print(f"✅ Model configuration validated: {primary_model}")
        return True
    
    def log_configuration_summary(self):
        """Log a summary of the current configuration for debugging."""
        print(f"\n📊 Configuration Summary:")
        print(f"   ECTD Model: {self.ectd_config.get('model', 'unknown')}")
        print(f"   Force Primary Model: {self.ectd_config.get('force_primary_model', False)}")
        print(f"   Cache Enabled: {self.ectd_config.get('cache_enabled', False)}")
        print(f"   Parallel Workers: {self.ectd_config.get('parallel_workers', 'unknown')}")
        print(f"   Graph Judge Model: {self.graph_judge_phase_config.get('model_name', 'unknown')}")
        print(f"   Explainable Mode: {self.graph_judge_phase_config.get('explainable_mode', False)}")
        print(f"   Bootstrap Mode: {self.graph_judge_phase_config.get('bootstrap_mode', False)}")


class ConfigManager:
    """
    Manages configuration files for the KG pipeline.
    
    This class handles loading, creating, and managing configuration files
    with support for both base templates and iteration-specific configurations.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        # Find the config directory relative to the project structure
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # chat/cli -> chat -> GraphJudge
        self.config_dir = project_root / "config"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Base configuration file path
        self.base_config_path = self.config_dir / "pipeline_config.yaml"
        
        print(f" Config Manager initialized: {self.config_dir}")
    
    def create_base_config_template(self) -> str:
        """
        Create the base configuration template if it doesn't exist.
        
        Returns:
            Path to the created configuration file
        """
        if self.base_config_path.exists():
            print(f" Base config template already exists: {self.base_config_path}")
            return str(self.base_config_path)
        
        # Base template configuration based on improvement_plan2.md
        base_config = {
            'pipeline': {
                'iteration': 'auto-prompt',  # Will prompt user if not specified
                'parallel_workers': 5,
                'checkpoint_frequency': 10,
                'error_tolerance': 0.1,
                'auto_create_directories': True,
                'base_output_path': './docs/Iteration_Report'
            },
            'stages': {
                'ectd': {
                    'model': 'gpt5-mini',
                    'fallback_model': 'kimi-k2',
                    'temperature': 0.3,
                    'batch_size': 20,
                    'parallel_workers': 5,
                    'cache_enabled': True,
                    'cache_type': 'sha256',
                    'rate_limiting_enabled': True
                },
                'triple_generation': {
                    'output_format': 'json',
                    'validation_enabled': True,
                    'schema_validation_enabled': True,
                    'relation_mapping': './config/relation_map.json',
                    'text_chunking_enabled': True,
                    'post_processing_enabled': True
                },
                'graph_judge_phase': {
                    'explainable_mode': False,
                    'bootstrap_mode': False,
                    'model_name': 'perplexity/sonar-reasoning',
                    'temperature': 0.2,
                    'concurrent_limit': 3,
                    'enable_citations': True,
                    'confidence_threshold': 0.7,
                    'evidence_sources': ['source_text', 'domain_knowledge']
                },
                'evaluation': {
                    'metrics': ['triple_match_f1', 'graph_match_accuracy', 'g_bleu', 'g_rouge', 'g_bert_score'],
                    'gold_standard': './datasets/gold_standard.json',
                    'comprehensive_reporting': True
                }
            },
            'pipeline_state': {
                'enabled': True,
                'state_file': './pipeline_state.json',
                'auto_save_enabled': True,
                'error_tracking_enabled': True
            }
        }
        
        # Save base configuration
        with open(self.base_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[SUCCESS] Created base config template: {self.base_config_path}")
        return str(self.base_config_path)
    
    def load_config(self, config_path: str) -> PipelineConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            PipelineConfig object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f" Config file not found: {config_path}")
            print("Creating default configuration...")
            return PipelineConfig()
        
        try:
            # Determine file format and load accordingly
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Convert loaded data to PipelineConfig
            config = self._dict_to_config(data)
            print(f"[SUCCESS] Loaded configuration: {config_path}")
            return config
            
        except Exception as e:
            print(f" Error loading config file {config_path}: {e}")
            print("Using default configuration...")
            return PipelineConfig()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> PipelineConfig:
        """
        Convert dictionary data to PipelineConfig object.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            PipelineConfig object
        """
        # Handle nested structure from YAML/JSON
        pipeline_data = data.get('pipeline', {})
        stages_data = data.get('stages', {})
        pipeline_state_data = data.get('pipeline_state', {})
        
        # Extract pipeline-level settings
        config = PipelineConfig(
            iteration=pipeline_data.get('iteration'),
            parallel_workers=pipeline_data.get('parallel_workers', 5),
            checkpoint_frequency=pipeline_data.get('checkpoint_frequency', 10),
            error_tolerance=pipeline_data.get('error_tolerance', 0.1),
            auto_create_directories=pipeline_data.get('auto_create_directories', True),
            base_output_path=pipeline_data.get('base_output_path', './docs/Iteration_Report'),
            
            # Stage-specific configurations
            ectd_config=stages_data.get('ectd', {}),
            triple_generation_config=stages_data.get('triple_generation', {}),
            graph_judge_phase_config=stages_data.get('graph_judge_phase', {}),  # Updated key name
            evaluation_config=stages_data.get('evaluation', {}),
            
            # Pipeline state management
            pipeline_state_config=pipeline_state_data
        )
        
        return config
    
    def create_iteration_config(self, config_path: str, iteration: int, base_config: PipelineConfig):
        """
        Create iteration-specific configuration file.
        
        Args:
            config_path: Path where to save the configuration
            iteration: Iteration number
            base_config: Base configuration to extend
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Create iteration-specific configuration based on improvement_plan2.md
        iteration_config = {
            'pipeline': {
                'iteration': iteration,
                'parallel_workers': base_config.parallel_workers,
                'checkpoint_frequency': base_config.checkpoint_frequency,
                'error_tolerance': base_config.error_tolerance,
                'output_base_path': f"./Iteration{iteration}/results",
                'log_base_path': f"./Iteration{iteration}/logs",
                'created_at': datetime.now().isoformat(),
                'base_path': f"./docs/Iteration_Report/Iteration{iteration}"
            },
            'stages': {
                'ectd': {
                    **base_config.ectd_config,
                    'output_path': f"./Iteration{iteration}/results/ectd_output.json",
                    'log_path': f"./Iteration{iteration}/logs/ectd.log"
                },
                'triple_generation': {
                    **base_config.triple_generation_config,
                    'output_path': f"./Iteration{iteration}/results/triples_output.json",
                    'log_path': f"./Iteration{iteration}/logs/triple_generation.log"
                },
                'graph_judge_phase': {
                    **base_config.graph_judge_phase_config,
                    'csv_output_path': f"./Iteration{iteration}/results/judgment_results.csv",
                    'reasoning_output_path': f"./Iteration{iteration}/results/judgment_reasoning.json",
                    'log_path': f"./Iteration{iteration}/logs/graph_judge.log"
                },
                'evaluation': {
                    **base_config.evaluation_config,
                    'report_output_path': f"./Iteration{iteration}/reports/evaluation_report.json",
                    'log_path': f"./Iteration{iteration}/logs/evaluation.log"
                }
            }
        }
        
        # Save configuration file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(iteration_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f" Created iteration config: {config_path}")
    
    def update_config(self, config_path: str, updates: Dict[str, Any]):
        """
        Update an existing configuration file.
        
        Args:
            config_path: Path to configuration file
            updates: Dictionary of updates to apply
        """
        config_path = Path(config_path)
        
        # Load existing configuration
        if config_path.exists():
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        else:
            data = {}
        
        # Apply updates recursively
        self._deep_update(data, updates)
        
        # Save updated configuration
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f" Updated configuration: {config_path}")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """
        Recursively update a dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self, config: PipelineConfig) -> List[str]:
        """
        Validate configuration and return any errors.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate pipeline-level settings
        if config.parallel_workers <= 0:
            errors.append("parallel_workers must be positive")
        
        if config.checkpoint_frequency <= 0:
            errors.append("checkpoint_frequency must be positive")
        
        if not (0.0 <= config.error_tolerance <= 1.0):
            errors.append("error_tolerance must be between 0.0 and 1.0")
        
        # Validate stage configurations
        if not config.ectd_config:
            errors.append("ectd_config is required")
        
        if not config.triple_generation_config:
            errors.append("triple_generation_config is required")
        
        if not config.graph_judge_phase_config:
            errors.append("graph_judge_phase_config is required")
        
        if not config.evaluation_config:
            errors.append("evaluation_config is required")
        
        # Validate graph judge phase confidence threshold
        if 'confidence_threshold' in config.graph_judge_phase_config:
            threshold = config.graph_judge_phase_config['confidence_threshold']
            if not (0.0 <= threshold <= 1.0):
                errors.append("graph_judge_phase confidence_threshold must be between 0.0 and 1.0")
        
        # Validate ECTD model configuration
        if 'model' in config.ectd_config:
            valid_models = ['gpt5-mini', 'kimi-k2', 'gpt-4', 'claude-3']
            if config.ectd_config['model'] not in valid_models:
                errors.append(f"ECTD model must be one of: {valid_models}")
        
        # Validate triple generation batch size
        if 'batch_processing_size' in config.triple_generation_config:
            batch_size = config.triple_generation_config['batch_processing_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append("triple_generation batch_processing_size must be a positive integer")
        
        # Validate graph judge phase concurrent limit
        if 'concurrent_limit' in config.graph_judge_phase_config:
            concurrent_limit = config.graph_judge_phase_config['concurrent_limit']
            if not isinstance(concurrent_limit, int) or concurrent_limit <= 0:
                errors.append("graph_judge_phase concurrent_limit must be a positive integer")
        
        return errors
    
    def get_stage_config(self, config: PipelineConfig, stage_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific stage.
        
        Args:
            config: Pipeline configuration
            stage_name: Name of the stage
            
        Returns:
            Stage-specific configuration dictionary
        """
        stage_configs = {
            'ectd': config.ectd_config,
            'triple_generation': config.triple_generation_config,
            'graph_judge_phase': config.graph_judge_phase_config,
            'evaluation': config.evaluation_config
        }
        
        return stage_configs.get(stage_name, {})
    
    def save_config(self, config: PipelineConfig, config_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary format
        config_dict = {
            'pipeline': {
                'iteration': config.iteration,
                'parallel_workers': config.parallel_workers,
                'checkpoint_frequency': config.checkpoint_frequency,
                'error_tolerance': config.error_tolerance,
                'auto_create_directories': config.auto_create_directories,
                'base_output_path': config.base_output_path
            },
            'stages': {
                'ectd': config.ectd_config,
                'triple_generation': config.triple_generation_config,
                'graph_judge_phase': config.graph_judge_phase_config,
                'evaluation': config.evaluation_config
            },
            'pipeline_state': config.pipeline_state_config
        }
        
        # Save based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Saved configuration: {config_path}")
    
    def list_configs(self) -> Dict[str, List[str]]:
        """
        List available configuration files.
        
        Returns:
            Dictionary mapping config types to file paths
        """
        configs = {
            'base': [],
            'iteration': []
        }
        
        # Find base configurations
        for config_file in self.config_dir.glob('*.{yaml,yml,json}'):
            configs['base'].append(str(config_file))
        
        # Find iteration-specific configurations
        # Look in the standard iteration directory structure
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent.parent
        iteration_base = project_root / "Miscellaneous" / "KgGen" / "GraphJudge" / "docs" / "Iteration_Report"
        
        if iteration_base.exists():
            for iteration_dir in iteration_base.glob('Iteration*'):
                config_dir = iteration_dir / 'configs'
                if config_dir.exists():
                    for config_file in config_dir.glob('*.{yaml,yml,json}'):
                        configs['iteration'].append(str(config_file))
        
        return configs
