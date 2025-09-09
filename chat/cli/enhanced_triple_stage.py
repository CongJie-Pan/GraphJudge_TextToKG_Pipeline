#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Triple Generation Stage Implementation for Unified CLI Pipeline

This module implements the EnhancedTripleGenerationStage class that integrates
run_triple.py functionality with schema validation, text chunking, and
comprehensive post-processing capabilities.

Features:
- Structured JSON output with Pydantic validation
- Text chunking for large inputs with pagination support
- Post-processing with triple parser for standardization
- Multiple output formats (JSON, TXT, enhanced)
- Quality metrics and statistics tracking
- Comprehensive error handling and logging

Author: Engineering Team
Date: 2025-09-07
Version: 1.0.0
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add the chat directory to the path to import run_triple functionality
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

from stage_manager import PipelineStage
from environment_manager import EnvironmentManager

# Import run_triple components
try:
    from run_triple import TerminalProgressLogger, TripleGenerationPipeline, main as run_triple_main
except ImportError:
    # Handle case where run_triple is not directly importable
    TerminalProgressLogger = None
    TripleGenerationPipeline = None
    run_triple_main = None


class EnhancedTripleGenerationStage(PipelineStage):
    """
    Enhanced Triple Generation Stage with schema validation and chunking.
    
    This stage integrates the run_triple.py functionality into the unified CLI
    pipeline with enhanced features for production use including schema validation,
    text chunking, and comprehensive post-processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Enhanced Triple Generation Stage.
        
        Args:
            config: Stage-specific configuration dictionary
        """
        super().__init__("Enhanced Triple Generation", config)
        
        # Output configuration
        self.output_format = config.get('output_format', 'json')
        self.validation_enabled = config.get('validation_enabled', True)
        self.schema_validation_enabled = config.get('schema_validation_enabled', True)
        
        # Processing configuration
        self.text_chunking_enabled = config.get('text_chunking_enabled', True)
        self.post_processing_enabled = config.get('post_processing_enabled', True)
        self.duplicate_removal_enabled = config.get('duplicate_removal_enabled', True)
        
        # Performance configuration
        self.chunk_size = config.get('chunk_size', 1000)
        self.max_tokens_per_chunk = self.chunk_size  # Alias for test compatibility
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.batch_processing_size = config.get('batch_processing_size', 10)
        
        # Advanced features
        self.entity_linking_enabled = config.get('entity_linking_enabled', True)
        self.confidence_scoring_enabled = config.get('confidence_scoring_enabled', True)
        self.relation_mapping_path = config.get('relation_mapping', './config/relation_map.json')
        self.multiple_formats = config.get('multiple_formats', ['json', 'txt'])
        
        # Initialize terminal logger
        self.terminal_logger = None
        
        print(f"✓ Enhanced Triple Generation Stage initialized")
        print(f"  - Schema validation: {self.schema_validation_enabled}")
        print(f"  - Text chunking: {self.text_chunking_enabled}")
        print(f"  - Post-processing: {self.post_processing_enabled}")
        print(f"  - Output format: {self.output_format}")
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """
        Execute the Enhanced Triple Generation stage.
        
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
            
            # Determine input files (entities and denoised text)
            input_iteration = kwargs.get('input_iteration', iteration)
            entity_file, denoised_file = self._find_input_files(iteration_path, input_iteration)
            
            if not entity_file or not denoised_file:
                self.error_message = f"Input files not found: entity={entity_file}, denoised={denoised_file}"
                if self.terminal_logger:
                    self.terminal_logger.log(f"❌ Error: {self.error_message}")
                else:
                    print(f"❌ Error: {self.error_message}")
                return False
            
            # Validate input files
            if not self._validate_input_files(entity_file, denoised_file):
                self.error_message = "Input file validation failed"
                return False
            
            # Setup output paths
            output_dir = os.path.join(iteration_path, "results", "triple_generation")
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output files based on configuration
            output_files = self._define_output_files(output_dir, iteration)
            
            # Execute enhanced triple generation with all features
            success = await self._execute_enhanced_triple_generation(
                entity_file, denoised_file, output_files, env_vars
            )
            
            if success:
                # Validate outputs
                if not self._validate_outputs(output_files):
                    self.error_message = "Output validation failed"
                    success = False
                else:
                    # Generate statistics and quality metrics
                    self._generate_quality_metrics(output_files)
                    self.terminal_logger.log(f"✅ Enhanced Triple Generation completed successfully")
            
            # Cleanup
            if self.terminal_logger:
                self.terminal_logger.close()
            
            self._log_stage_end(success)
            return success
            
        except Exception as e:
            self.error_message = f"Enhanced Triple Generation stage execution failed: {str(e)}"
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
            'TRIPLE_OUTPUT_FORMAT': self.output_format,
            'TRIPLE_SCHEMA_VALIDATION': str(self.schema_validation_enabled),
            'TRIPLE_TEXT_CHUNKING': str(self.text_chunking_enabled),
            'TRIPLE_POST_PROCESSING': str(self.post_processing_enabled),
            'TRIPLE_CHUNK_SIZE': str(self.chunk_size),
            'TRIPLE_CHUNK_OVERLAP': str(self.chunk_overlap),
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
        # Create a simple logger if TerminalProgressLogger is not available
        class SimpleTripleLogger:
            def __init__(self, log_path):
                self.log_path = log_path
                self.log_file = open(log_path, 'w', encoding='utf-8')
                self.log_file.write(f"Enhanced Triple Generation Stage Log - {datetime.now()}\n")
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
        log_path = os.path.join(log_dir, f"enhanced_triple_generation_{timestamp}.log")
        
        if TerminalProgressLogger:
            return TerminalProgressLogger(log_dir)
        else:
            return SimpleTripleLogger(log_path)
    
    def _find_input_files(self, iteration_path: str, input_iteration: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the input files for triple generation (entity and denoised text).
        
        Args:
            iteration_path: Path to current iteration directory
            input_iteration: Iteration number to look for inputs
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (entity_file, denoised_file)
        """
        # Look for ECTD outputs from current or specified iteration
        ectd_dir = os.path.join(iteration_path, "results", "ectd")
        
        # Check if ECTD results exist in current iteration
        entity_file = os.path.join(ectd_dir, "test_entity.txt")
        denoised_file = os.path.join(ectd_dir, "test_denoised.target")
        
        if os.path.exists(entity_file) and os.path.exists(denoised_file):
            return entity_file, denoised_file
        
        # Look in other iteration directories if not found
        base_dir = Path(iteration_path).parent
        for i in range(input_iteration, 0, -1):  # Look backwards from input_iteration
            iter_dir = base_dir / f"Iteration{i}" / "results" / "ectd"
            entity_file = iter_dir / "test_entity.txt"
            denoised_file = iter_dir / "test_denoised.target"
            
            if entity_file.exists() and denoised_file.exists():
                return str(entity_file), str(denoised_file)
        
        # Look for alternative file names
        alternative_patterns = [
            ("entities.txt", "denoised.txt"),
            ("entity_output.txt", "denoised_output.txt"),
            ("extracted_entities.txt", "cleaned_text.txt")
        ]
        
        for entity_pattern, denoised_pattern in alternative_patterns:
            entity_file = os.path.join(ectd_dir, entity_pattern)
            denoised_file = os.path.join(ectd_dir, denoised_pattern)
            
            if os.path.exists(entity_file) and os.path.exists(denoised_file):
                return entity_file, denoised_file
        
        return None, None
    
    def _validate_input_files(self, entity_file: str, denoised_file: str) -> bool:
        """
        Validate the input files for triple generation.
        
        Args:
            entity_file: Path to entity file
            denoised_file: Path to denoised text file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check entity file
            if not os.path.exists(entity_file):
                if self.terminal_logger:
                    self.terminal_logger.log(f"❌ Entity file not found: {entity_file}")
                else:
                    print(f"❌ Entity file not found: {entity_file}")
                return False
            
            # Check denoised file
            if not os.path.exists(denoised_file):
                if self.terminal_logger:
                    self.terminal_logger.log(f"❌ Denoised file not found: {denoised_file}")
                else:
                    print(f"❌ Denoised file not found: {denoised_file}")
                return False
            
            # Check file sizes
            entity_size = os.path.getsize(entity_file)
            denoised_size = os.path.getsize(denoised_file)
            
            if entity_size == 0:
                if self.terminal_logger:
                    self.terminal_logger.log(f"❌ Entity file is empty: {entity_file}")
                else:
                    print(f"❌ Entity file is empty: {entity_file}")
                return False
            
            if denoised_size == 0:
                if self.terminal_logger:
                    self.terminal_logger.log(f"❌ Denoised file is empty: {denoised_file}")
                else:
                    print(f"❌ Denoised file is empty: {denoised_file}")
                return False
            
            # Log file information
            if self.terminal_logger:
                self.terminal_logger.log(f"✅ Input files validated:")
                self.terminal_logger.log(f"  - Entity file: {entity_file} ({entity_size} bytes)")
                self.terminal_logger.log(f"  - Denoised file: {denoised_file} ({denoised_size} bytes)")
            else:
                print(f"✅ Input files validated:")
                print(f"  - Entity file: {entity_file} ({entity_size} bytes)")
                print(f"  - Denoised file: {denoised_file} ({denoised_size} bytes)")
            
            return True
            
        except Exception as e:
            if self.terminal_logger:
                self.terminal_logger.log(f"❌ Input validation error: {str(e)}")
            else:
                print(f"❌ Input validation error: {str(e)}")
            return False
    
    def _define_output_files(self, output_dir: str, iteration: int) -> Dict[str, str]:
        """
        Define output files based on configuration.
        
        Args:
            output_dir: Output directory
            iteration: Current iteration number
            
        Returns:
            Dict[str, str]: Dictionary of output file paths
        """
        output_files = {}
        
        # Define output files based on multiple_formats configuration
        for format_type in self.multiple_formats:
            if format_type == 'json':
                if self.output_format == 'json':
                    output_files['json'] = os.path.join(
                        output_dir, f"test_instructions_context_gpt5mini_v2.json"
                    )
                else:
                    output_files['json'] = os.path.join(
                        output_dir, f"triples_iter{iteration}.json"
                    )
            elif format_type == 'txt':
                output_files['txt'] = os.path.join(
                    output_dir, f"triples_iter{iteration}.txt"
                )
            elif format_type == 'enhanced':
                output_files['enhanced'] = os.path.join(
                    output_dir, f"enhanced_triples_iter{iteration}.json"
                )
        
        # Ensure at least primary output exists for backward compatibility
        if not output_files and self.output_format == 'json':
            output_files['primary'] = os.path.join(
                output_dir, f"test_instructions_context_gpt5mini_v2.json"
            )
        
        # Additional metadata files
        output_files['statistics'] = os.path.join(
            output_dir, f"generation_stats_iter{iteration}.json"
        )
        
        if self.post_processing_enabled:
            output_files['processed'] = os.path.join(
                output_dir, f"processed_triples_iter{iteration}.json"
            )
        
        return output_files
    
    async def _execute_enhanced_triple_generation(self, entity_file: str, denoised_file: str, 
                                                output_files: Dict[str, str], env_vars: Dict[str, str]) -> bool:
        """
        Execute enhanced triple generation with all features.
        
        Args:
            entity_file: Path to entity file
            denoised_file: Path to denoised text file
            output_files: Dictionary of output file paths
            env_vars: Environment variables
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting Enhanced Triple Generation")
            self.terminal_logger.log(f"  - Entity input: {entity_file}")
            self.terminal_logger.log(f"  - Denoised input: {denoised_file}")
            self.terminal_logger.log(f"  - Schema validation: {self.schema_validation_enabled}")
            self.terminal_logger.log(f"  - Text chunking: {self.text_chunking_enabled}")
            self.terminal_logger.log(f"  - Post-processing: {self.post_processing_enabled}")
            
            # Set up input files in environment
            env_vars.update({
                'ENTITY_FILE': entity_file,
                'DENOISED_FILE': denoised_file,
                'PRIMARY_OUTPUT': output_files['primary']
            })
            
            # Use subprocess to run the original script with environment
            import subprocess
            
            # Create wrapper script that calls run_triple.py
            wrapper_script = self._create_dynamic_wrapper(
                original_script=str(chat_dir / "run_triple.py"),
                iteration_path=os.path.dirname(output_files['primary']),
                stage_name="Enhanced Triple Generation",
                additional_vars={
                    'ENTITY_INPUT': entity_file,
                    'DENOISED_INPUT': denoised_file,
                    'OUTPUT_DIR': os.path.dirname(output_files['primary'])
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
                self.terminal_logger.log(f"✅ Enhanced Triple Generation completed successfully")
                
                # Post-process if enabled
                if self.post_processing_enabled:
                    await self._post_process_triples(output_files)
                
                # Generate additional formats if needed
                await self._generate_additional_formats(output_files)
                
            else:
                self.terminal_logger.log(f"❌ Enhanced Triple Generation failed with return code: {process.returncode}")
            
            return success
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Enhanced Triple Generation execution error: {str(e)}")
            return False
    
    async def _post_process_triples(self, output_files: Dict[str, str]) -> bool:
        """
        Post-process generated triples for standardization and cleanup.
        
        Args:
            output_files: Dictionary of output file paths
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🔄 Starting post-processing...")
            
            # Read primary output
            primary_file = output_files['primary']
            if not os.path.exists(primary_file):
                self.terminal_logger.log(f"❌ Primary output not found for post-processing: {primary_file}")
                return False
            
            with open(primary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as JSON
            try:
                triples_data = json.loads(content)
            except json.JSONDecodeError:
                self.terminal_logger.log(f"⚠️  Primary output is not valid JSON, attempting text parsing")
                triples_data = self._parse_text_triples(content)
            
            # Apply post-processing steps
            processed_triples = self._apply_triple_processing(triples_data)
            
            # Save processed results
            if 'processed' in output_files:
                with open(output_files['processed'], 'w', encoding='utf-8') as f:
                    json.dump(processed_triples, f, indent=2, ensure_ascii=False)
                
                self.terminal_logger.log(f"✅ Post-processed triples saved to: {output_files['processed']}")
            
            return True
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Post-processing error: {str(e)}")
            return False
    
    def _parse_text_triples(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse triples from text content when JSON parsing fails.
        
        Args:
            content: Text content to parse
            
        Returns:
            List[Dict[str, Any]]: Parsed triples
        """
        triples = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract subject, predicate, object from various formats
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    triples.append({
                        'subject': parts[0].strip(),
                        'predicate': parts[1].strip(),
                        'object': parts[2].strip()
                    })
        
        return triples
    
    def _apply_triple_processing(self, triples_data: Any) -> Dict[str, Any]:
        """
        Apply comprehensive triple processing and standardization.
        
        Args:
            triples_data: Raw triples data
            
        Returns:
            Dict[str, Any]: Processed triples with metadata
        """
        processed = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'processing_version': '1.0.0',
                'features_enabled': {
                    'duplicate_removal': self.duplicate_removal_enabled,
                    'entity_linking': self.entity_linking_enabled,
                    'confidence_scoring': self.confidence_scoring_enabled
                }
            },
            'statistics': {
                'raw_triples_count': 0,
                'processed_triples_count': 0,
                'duplicates_removed': 0,
                'confidence_scores_added': 0
            },
            'triples': []
        }
        
        # Extract triples from various data formats
        raw_triples = self._extract_triples_from_data(triples_data)
        processed['statistics']['raw_triples_count'] = len(raw_triples)
        
        # Remove duplicates if enabled
        if self.duplicate_removal_enabled:
            unique_triples = self._remove_duplicate_triples(raw_triples)
            processed['statistics']['duplicates_removed'] = len(raw_triples) - len(unique_triples)
            raw_triples = unique_triples
        
        # Process each triple
        for triple in raw_triples:
            processed_triple = self._process_single_triple(triple)
            if processed_triple:
                processed['triples'].append(processed_triple)
        
        processed['statistics']['processed_triples_count'] = len(processed['triples'])
        
        self.terminal_logger.log(f"📊 Post-processing statistics:")
        self.terminal_logger.log(f"  - Raw triples: {processed['statistics']['raw_triples_count']}")
        self.terminal_logger.log(f"  - Processed triples: {processed['statistics']['processed_triples_count']}")
        self.terminal_logger.log(f"  - Duplicates removed: {processed['statistics']['duplicates_removed']}")
        
        return processed
    
    def _extract_triples_from_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Extract triples from various data formats.
        
        Args:
            data: Input data in various formats
            
        Returns:
            List[Dict[str, Any]]: Extracted triples
        """
        triples = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    triples.append(item)
        elif isinstance(data, dict):
            if 'triples' in data:
                triples.extend(self._extract_triples_from_data(data['triples']))
            elif 'subject' in data and 'predicate' in data and 'object' in data:
                triples.append(data)
        
        return triples
    
    def _remove_duplicate_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate triples based on subject-predicate-object.
        
        Args:
            triples: List of triples
            
        Returns:
            List[Dict[str, Any]]: Unique triples
        """
        seen = set()
        unique_triples = []
        
        for triple in triples:
            # Create a unique key for the triple
            key = (
                triple.get('subject', '').strip().lower(),
                triple.get('predicate', '').strip().lower(),
                triple.get('object', '').strip().lower()
            )
            
            if key not in seen and all(key):  # Ensure no empty components
                seen.add(key)
                unique_triples.append(triple)
        
        return unique_triples
    
    def _process_single_triple(self, triple: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single triple with enhancements.
        
        Args:
            triple: Input triple
            
        Returns:
            Optional[Dict[str, Any]]: Processed triple or None if invalid
        """
        # Validate required fields
        if not all(key in triple for key in ['subject', 'predicate', 'object']):
            return None
        
        processed_triple = {
            'subject': triple['subject'].strip(),
            'predicate': triple['predicate'].strip(),
            'object': triple['object'].strip()
        }
        
        # Add confidence scoring if enabled
        if self.confidence_scoring_enabled:
            processed_triple['confidence'] = self._calculate_confidence_score(processed_triple)
        
        # Add entity linking if enabled
        if self.entity_linking_enabled:
            processed_triple['entity_types'] = self._identify_entity_types(processed_triple)
        
        return processed_triple
    
    def _calculate_confidence_score(self, triple: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for the triple.
        
        Args:
            triple: Triple to score
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Simple heuristic-based confidence scoring
        score = 0.5  # Base score
        
        # Length-based scoring
        subject_len = len(triple['subject'])
        predicate_len = len(triple['predicate'])
        object_len = len(triple['object'])
        
        if 2 <= subject_len <= 50:
            score += 0.1
        if 2 <= predicate_len <= 30:
            score += 0.1
        if 2 <= object_len <= 100:
            score += 0.1
        
        # Character quality scoring
        if triple['subject'].strip() and not triple['subject'].startswith(('_', '?')):
            score += 0.1
        if triple['predicate'].strip() and not triple['predicate'].startswith(('_', '?')):
            score += 0.1
        if triple['object'].strip() and not triple['object'].startswith(('_', '?')):
            score += 0.1
        
        return min(1.0, score)
    
    def _identify_entity_types(self, triple: Dict[str, Any]) -> Dict[str, str]:
        """
        Identify entity types for triple components.
        
        Args:
            triple: Triple to analyze
            
        Returns:
            Dict[str, str]: Entity type mappings
        """
        # Simple entity type identification
        entity_types = {
            'subject_type': 'entity',
            'predicate_type': 'relation',
            'object_type': 'entity'
        }
        
        # Basic heuristics for Chinese text
        subject = triple['subject']
        object_val = triple['object']
        
        # Check for person names (Chinese characters with common surname patterns)
        if any(char in subject for char in ['王', '李', '張', '劉', '陳', '楊', '黃', '吳']):
            entity_types['subject_type'] = 'person'
        
        if any(char in object_val for char in ['王', '李', '張', '劉', '陳', '楊', '黃', '吳']):
            entity_types['object_type'] = 'person'
        
        # Check for location indicators
        if any(suffix in subject for suffix in ['城', '縣', '省', '國', '府', '園', '院']):
            entity_types['subject_type'] = 'location'
        
        if any(suffix in object_val for suffix in ['城', '縣', '省', '國', '府', '園', '院']):
            entity_types['object_type'] = 'location'
        
        return entity_types
    
    async def _generate_additional_formats(self, output_files: Dict[str, str]) -> bool:
        """
        Generate additional output formats.
        
        Args:
            output_files: Dictionary of output file paths
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read primary output for format conversion
            primary_file = output_files['primary']
            if not os.path.exists(primary_file):
                return False
            
            with open(primary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate TXT format
            if 'txt_format' in output_files:
                await self._generate_txt_format(content, output_files['txt_format'])
            
            # Generate enhanced JSON format
            if 'enhanced_json' in output_files:
                await self._generate_enhanced_json(content, output_files['enhanced_json'])
            
            return True
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Additional format generation error: {str(e)}")
            return False
    
    async def _generate_txt_format(self, content: str, output_file: str):
        """Generate simple TXT format output."""
        try:
            # Parse JSON content
            try:
                data = json.loads(content)
                triples = self._extract_triples_from_data(data)
            except:
                # Fallback to text parsing
                triples = self._parse_text_triples(content)
            
            # Write TXT format
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Enhanced Triple Generation Output (TXT Format)\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total triples: {len(triples)}\n\n")
                
                for i, triple in enumerate(triples, 1):
                    f.write(f"{i:04d}: {triple.get('subject', '')} | {triple.get('predicate', '')} | {triple.get('object', '')}\n")
            
            self.terminal_logger.log(f"✅ TXT format generated: {output_file}")
            
        except Exception as e:
            self.terminal_logger.log(f"❌ TXT format generation error: {str(e)}")
    
    async def _generate_enhanced_json(self, content: str, output_file: str):
        """Generate enhanced JSON format with metadata."""
        try:
            # Parse original content
            try:
                original_data = json.loads(content)
            except:
                original_data = {"raw_content": content}
            
            # Create enhanced format
            enhanced_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "stage": "Enhanced Triple Generation",
                    "version": "1.0.0",
                    "features": {
                        "schema_validation": self.schema_validation_enabled,
                        "text_chunking": self.text_chunking_enabled,
                        "post_processing": self.post_processing_enabled
                    }
                },
                "original_data": original_data,
                "processing_summary": {
                    "total_triples": len(self._extract_triples_from_data(original_data)),
                    "output_format": self.output_format
                }
            }
            
            # Write enhanced JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            self.terminal_logger.log(f"✅ Enhanced JSON generated: {output_file}")
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Enhanced JSON generation error: {str(e)}")
    
    def _validate_outputs(self, output_files: Dict[str, str]) -> bool:
        """
        Validate the generated outputs.
        
        Args:
            output_files: Dictionary of output file paths
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check primary output
            primary_file = output_files['primary']
            if not os.path.exists(primary_file):
                self.terminal_logger.log(f"❌ Primary output file not found: {primary_file}")
                return False
            
            # Check file size
            file_size = os.path.getsize(primary_file)
            if file_size == 0:
                self.terminal_logger.log(f"❌ Primary output file is empty")
                return False
            
            # Validate content format
            with open(primary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as JSON if expected
            if self.output_format == 'json':
                try:
                    json.loads(content)
                    self.terminal_logger.log(f"✅ Primary output is valid JSON")
                except json.JSONDecodeError:
                    self.terminal_logger.log(f"⚠️  Primary output is not valid JSON, but content exists")
            
            # Log file statistics
            self.terminal_logger.log(f"📊 Output validation completed:")
            self.terminal_logger.log(f"  - Primary output: {primary_file} ({file_size} bytes)")
            
            for name, path in output_files.items():
                if name != 'primary' and os.path.exists(path):
                    size = os.path.getsize(path)
                    self.terminal_logger.log(f"  - {name}: {path} ({size} bytes)")
            
            return True
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Output validation error: {str(e)}")
            return False
    
    def _generate_quality_metrics(self, output_files: Dict[str, str]):
        """
        Generate quality metrics and statistics.
        
        Args:
            output_files: Dictionary of output file paths
        """
        try:
            # Read primary output for analysis
            primary_file = output_files['primary']
            with open(primary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse content for analysis
            try:
                data = json.loads(content)
                triples = self._extract_triples_from_data(data)
            except:
                triples = self._parse_text_triples(content)
            
            # Calculate metrics
            metrics = {
                'generation_timestamp': datetime.now().isoformat(),
                'total_triples': len(triples),
                'unique_subjects': len(set(t.get('subject', '') for t in triples)),
                'unique_predicates': len(set(t.get('predicate', '') for t in triples)),
                'unique_objects': len(set(t.get('object', '') for t in triples)),
                'average_subject_length': sum(len(t.get('subject', '')) for t in triples) / max(1, len(triples)),
                'average_predicate_length': sum(len(t.get('predicate', '')) for t in triples) / max(1, len(triples)),
                'average_object_length': sum(len(t.get('object', '')) for t in triples) / max(1, len(triples)),
                'features_enabled': {
                    'schema_validation': self.schema_validation_enabled,
                    'text_chunking': self.text_chunking_enabled,
                    'post_processing': self.post_processing_enabled,
                    'duplicate_removal': self.duplicate_removal_enabled
                }
            }
            
            # Save metrics
            if 'statistics' in output_files:
                with open(output_files['statistics'], 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                self.terminal_logger.log(f"📊 Quality metrics saved to: {output_files['statistics']}")
            
            # Log key metrics
            self.terminal_logger.log(f"📊 Generation Quality Metrics:")
            self.terminal_logger.log(f"  - Total triples: {metrics['total_triples']}")
            self.terminal_logger.log(f"  - Unique subjects: {metrics['unique_subjects']}")
            self.terminal_logger.log(f"  - Unique predicates: {metrics['unique_predicates']}")
            self.terminal_logger.log(f"  - Unique objects: {metrics['unique_objects']}")
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Quality metrics generation error: {str(e)}")
