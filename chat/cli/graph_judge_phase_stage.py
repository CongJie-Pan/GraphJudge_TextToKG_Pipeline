#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Graph Judge Phase Stage Implementation for Unified CLI Pipeline

This module implements the GraphJudgePhaseStage class that integrates the modular
graphJudge_Phase system with full feature support including explainable reasoning,
gold label bootstrapping, and streaming capabilities.

Features:
- Full integration with modular graphJudge_Phase system
- PerplexityGraphJudge with multiple model support
- GoldLabelBootstrapper for two-stage automatic label assignment
- ProcessingPipeline for configurable batch processing
- Explainable reasoning with confidence scores
- Gold label bootstrapping functionality
- Streaming capabilities for real-time processing
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

# Add the chat directory to the path to import graphJudge_Phase
current_dir = Path(__file__).parent
chat_dir = current_dir.parent
sys.path.insert(0, str(chat_dir))

from stage_manager import PipelineStage
from environment_manager import EnvironmentManager

# Import graphJudge_Phase components
try:
    from graphJudge_Phase import (
        PerplexityGraphJudge,
        GoldLabelBootstrapper,
        ProcessingPipeline,
        TripleData,
        ExplainableJudgment,
        BootstrapResult,
        TerminalLogger,
        validate_input_file,
        create_output_directory
    )
    GRAPHJUDGE_PHASE_AVAILABLE = True
except ImportError:
    print("Warning: graphJudge_Phase module not available")
    GRAPHJUDGE_PHASE_AVAILABLE = False
    PerplexityGraphJudge = None
    GoldLabelBootstrapper = None
    ProcessingPipeline = None


class GraphJudgePhaseStage(PipelineStage):
    """
    Enhanced Graph Judge Phase Stage with full modular integration.
    
    This stage integrates the complete graphJudge_Phase modular system into the
    unified CLI pipeline with support for all operation modes and advanced features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Graph Judge Phase Stage.
        
        Args:
            config: Stage-specific configuration dictionary
        """
        super().__init__("GraphJudge Phase", config)
        
        # Core GraphJudge configuration
        self.explainable_mode = config.get('explainable_mode', False)
        self.bootstrap_mode = config.get('bootstrap_mode', False)
        self.streaming_mode = config.get('streaming_mode', False)
        
        # Model configuration
        self.model_name = config.get('model_name', 'perplexity/sonar-reasoning')
        self.reasoning_effort = config.get('reasoning_effort', 'medium')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 2000)
        
        # Performance configuration
        self.concurrent_limit = config.get('concurrent_limit', 3)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.base_delay = config.get('base_delay', 0.5)
        
        # Logging configuration
        self.enable_console_logging = config.get('enable_console_logging', False)
        self.enable_citations = config.get('enable_citations', True)
        
        # Gold label bootstrapping configuration
        self.gold_bootstrap_config = config.get('gold_bootstrap_config', {
            'fuzzy_threshold': 0.8,
            'sample_rate': 0.15,
            'llm_batch_size': 10,
            'max_source_lines': 1000,
            'random_seed': 42
        })
        
        # Processing pipeline configuration
        self.processing_config = config.get('processing_config', {
            'enable_statistics': True,
            'save_reasoning_files': True,
            'batch_processing': True
        })
        
        # Advanced features
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.evidence_sources = config.get('evidence_sources', ['source_text', 'domain_knowledge'])
        self.alternative_suggestions_enabled = config.get('alternative_suggestions_enabled', False)
        
        # Initialize modular components if available
        if GRAPHJUDGE_PHASE_AVAILABLE:
            self._initialize_components()
        else:
            self.graph_judge = None
            self.bootstrapper = None
            self.pipeline = None
        
        # Initialize terminal logger
        self.terminal_logger = None
        
        print(f"✓ GraphJudge Phase Stage initialized")
        print(f"  - Explainable mode: {self.explainable_mode}")
        print(f"  - Bootstrap mode: {self.bootstrap_mode}")
        print(f"  - Streaming mode: {self.streaming_mode}")
        print(f"  - Model: {self.model_name}")
        print(f"  - Components available: {GRAPHJUDGE_PHASE_AVAILABLE}")
    
    def _initialize_components(self):
        """Initialize the modular graphJudge_Phase components."""
        try:
            # Initialize PerplexityGraphJudge
            self.graph_judge = PerplexityGraphJudge(
                model_name=self.model_name,
                reasoning_effort=self.reasoning_effort,
                enable_console_logging=self.enable_console_logging,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                concurrent_limit=self.concurrent_limit,
                retry_attempts=self.retry_attempts,
                base_delay=self.base_delay
            )
            
            # Initialize GoldLabelBootstrapper
            self.bootstrapper = GoldLabelBootstrapper(
                graph_judge=self.graph_judge,
                **self.gold_bootstrap_config
            )
            
            # Initialize ProcessingPipeline
            self.pipeline = ProcessingPipeline(
                graph_judge=self.graph_judge,
                **self.processing_config
            )
            
            print("✓ GraphJudge Phase components initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize graphJudge_Phase components: {str(e)}")
            self.graph_judge = None
            self.bootstrapper = None
            self.pipeline = None

    def configure_modular_system(self, mode_config: Dict[str, Any]) -> None:
        """
        Configure the modular GraphJudge system with runtime settings.
        
        Args:
            mode_config: Configuration dictionary for modular components
        """
        if not GRAPHJUDGE_PHASE_AVAILABLE:
            print("⚠️  GraphJudge Phase module not available for configuration")
            return
        
        try:
            # Update core configuration
            if 'explainable_mode' in mode_config:
                self.explainable_mode = mode_config['explainable_mode']
            if 'bootstrap_mode' in mode_config:
                self.bootstrap_mode = mode_config['bootstrap_mode']
            if 'streaming_mode' in mode_config:
                self.streaming_mode = mode_config['streaming_mode']
            
            # Update model configuration
            if 'model_name' in mode_config:
                self.model_name = mode_config['model_name']
            if 'temperature' in mode_config:
                self.temperature = mode_config['temperature']
            if 'reasoning_effort' in mode_config:
                self.reasoning_effort = mode_config['reasoning_effort']
            
            # Reinitialize components with updated configuration
            self._initialize_components()
            
            print(f"✓ GraphJudge Phase modular system configured with {len(mode_config)} settings")
            
        except Exception as e:
            print(f"❌ Failed to configure modular system: {str(e)}")

    def get_modular_capabilities(self) -> List[str]:
        """
        Get list of capabilities supported by the modular GraphJudge system.
        
        Returns:
            List[str]: List of supported capabilities
        """
        capabilities = [
            'explainable-reasoning',
            'gold-label-bootstrapping',
            'streaming',
            'modular-architecture'
        ]
        
        if GRAPHJUDGE_PHASE_AVAILABLE:
            capabilities.extend([
                'perplexity-integration',
                'batch-processing',
                'confidence-scoring',
                'citation-support'
            ])
        
        return capabilities

    def update_configuration(self, config_update: Dict[str, Any]) -> None:
        """
        Update stage configuration dynamically.
        
        Args:
            config_update: Dictionary of configuration updates
        """
        self.config.update(config_update)
        self.configure_modular_system(config_update)
    
    async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
        """
        Execute the Graph Judge Phase stage.
        
        Args:
            iteration: Current iteration number
            iteration_path: Path to iteration directory
            **kwargs: Additional execution parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_stage_start()
        
        try:
            # Check if components are available
            if not GRAPHJUDGE_PHASE_AVAILABLE:
                self.error_message = "graphJudge_Phase module not available"
                return False
            
            # Setup environment for this execution
            env_vars = self._setup_stage_environment(iteration, iteration_path, **kwargs)
            
            # Initialize terminal logger
            log_dir = os.path.join(iteration_path, "logs", "stages")
            os.makedirs(log_dir, exist_ok=True)
            self.terminal_logger = self._create_terminal_logger(log_dir)
            
            # Find input files (triples from previous stage)
            triples_file = self._find_triples_input(iteration_path, iteration)
            if not triples_file:
                self.error_message = "Triples input file not found"
                self.terminal_logger.log(f"❌ Error: {self.error_message}")
                return False
            
            # Setup output directory
            output_dir = os.path.join(iteration_path, "results", "graph_judge")
            os.makedirs(output_dir, exist_ok=True)
            
            # Execute based on operation mode
            if self.bootstrap_mode:
                success = await self._execute_gold_label_bootstrapping(
                    triples_file, output_dir, iteration_path, **kwargs
                )
            elif self.explainable_mode:
                success = await self._execute_explainable_judgment(
                    triples_file, output_dir, **kwargs
                )
            elif self.streaming_mode:
                success = await self._execute_streaming_judgment(
                    triples_file, output_dir, **kwargs
                )
            else:
                success = await self._execute_standard_judgment(
                    triples_file, output_dir, **kwargs
                )
            
            if success:
                self.terminal_logger.log(f"✅ GraphJudge Phase completed successfully")
            
            # Cleanup
            if self.terminal_logger:
                self.terminal_logger.close()
            
            self._log_stage_end(success)
            return success
            
        except Exception as e:
            self.error_message = f"GraphJudge Phase stage execution failed: {str(e)}"
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
            'GRAPHJUDGE_MODEL': self.model_name,
            'GRAPHJUDGE_EXPLAINABLE': str(self.explainable_mode),
            'GRAPHJUDGE_BOOTSTRAP': str(self.bootstrap_mode),
            'GRAPHJUDGE_STREAMING': str(self.streaming_mode),
            'GRAPHJUDGE_REASONING_EFFORT': self.reasoning_effort,
            'GRAPHJUDGE_TEMPERATURE': str(self.temperature),
            'GRAPHJUDGE_CONCURRENT_LIMIT': str(self.concurrent_limit),
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
        # Use the graphJudge_Phase TerminalLogger if available
        if TerminalLogger:
            return TerminalLogger(log_dir, "GraphJudgePhase")
        
        # Fallback to simple logger
        class SimpleGraphJudgeLogger:
            def __init__(self, log_path):
                self.log_path = log_path
                self.log_file = open(log_path, 'w', encoding='utf-8')
                self.log_file.write(f"GraphJudge Phase Stage Log - {datetime.now()}\n")
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
        log_path = os.path.join(log_dir, f"graph_judge_phase_{timestamp}.log")
        return SimpleGraphJudgeLogger(log_path)
    
    def _find_triples_input(self, iteration_path: str, iteration: int) -> Optional[str]:
        """
        Find the triples input file from previous stage.
        
        Args:
            iteration_path: Path to current iteration directory
            iteration: Current iteration number
            
        Returns:
            Optional[str]: Path to triples file or None
        """
        # Look for triple generation outputs
        triple_dir = os.path.join(iteration_path, "results", "triple_generation")
        
        # Common triple file patterns
        possible_files = [
            os.path.join(triple_dir, "test_instructions_context_gpt5mini_v2.json"),
            os.path.join(triple_dir, f"enhanced_triples_iter{iteration}.json"),
            os.path.join(triple_dir, f"triples_iter{iteration}.txt"),
            os.path.join(triple_dir, "triples.json"),
            os.path.join(triple_dir, "triples.txt")
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                self.terminal_logger.log(f"📄 Found triples input: {file_path}")
                return file_path
        
        # Look in other iteration directories if not found
        base_dir = Path(iteration_path).parent
        for i in range(iteration, 0, -1):  # Look backwards from current iteration
            iter_dir = base_dir / f"Iteration{i}" / "results" / "triple_generation"
            
            for pattern in ["test_instructions_context_gpt5mini_v2.json", "triples.json", "triples.txt"]:
                file_path = iter_dir / pattern
                if file_path.exists() and file_path.stat().st_size > 0:
                    self.terminal_logger.log(f"📄 Found triples input from Iteration {i}: {file_path}")
                    return str(file_path)
        
        return None
    
    async def _execute_standard_judgment(self, triples_file: str, output_dir: str, **kwargs) -> bool:
        """
        Execute standard graph judgment without explainable reasoning.
        
        Args:
            triples_file: Path to triples input file
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting Standard Graph Judgment")
            self.terminal_logger.log(f"  - Input: {triples_file}")
            self.terminal_logger.log(f"  - Model: {self.model_name}")
            self.terminal_logger.log(f"  - Output: {output_dir}")
            
            # Load triples from file
            triples_data = self._load_triples_data(triples_file)
            if not triples_data:
                self.terminal_logger.log(f"❌ Failed to load triples data")
                return False
            
            # Create output file
            output_file = os.path.join(output_dir, f"judgment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Use ProcessingPipeline for standard judgment
            if self.pipeline:
                results = await self.pipeline.process_triples_batch(
                    triples_data=triples_data,
                    output_file=output_file,
                    enable_statistics=self.processing_config.get('enable_statistics', True)
                )
                
                if results:
                    self.terminal_logger.log(f"✅ Standard judgment completed")
                    self.terminal_logger.log(f"  - Results saved to: {output_file}")
                    self.terminal_logger.log(f"  - Processed {len(results)} triples")
                    return True
                else:
                    self.terminal_logger.log(f"❌ Standard judgment failed")
                    return False
            else:
                # Fallback execution using subprocess
                return await self._execute_fallback_judgment(triples_file, output_file)
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Standard judgment error: {str(e)}")
            return False
    
    async def _execute_explainable_judgment(self, triples_file: str, output_dir: str, **kwargs) -> bool:
        """
        Execute explainable graph judgment with reasoning.
        
        Args:
            triples_file: Path to triples input file
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting Explainable Graph Judgment")
            self.terminal_logger.log(f"  - Input: {triples_file}")
            self.terminal_logger.log(f"  - Model: {self.model_name}")
            self.terminal_logger.log(f"  - Reasoning effort: {self.reasoning_effort}")
            self.terminal_logger.log(f"  - Output: {output_dir}")
            
            # Load triples from file
            triples_data = self._load_triples_data(triples_file)
            if not triples_data:
                self.terminal_logger.log(f"❌ Failed to load triples data")
                return False
            
            # Create output files
            csv_output = os.path.join(output_dir, f"explainable_judgment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            reasoning_output = os.path.join(output_dir, f"reasoning_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Process triples with explainable reasoning
            if self.graph_judge:
                explainable_results = []
                
                for i, triple in enumerate(triples_data):
                    self.terminal_logger.log(f"🔍 Processing triple {i+1}/{len(triples_data)}: {triple}")
                    
                    # Create TripleData object
                    triple_obj = TripleData(
                        subject=triple.get('subject', ''),
                        predicate=triple.get('predicate', ''),
                        object=triple.get('object', '')
                    )
                    
                    # Get explainable judgment
                    result = await self.graph_judge.judge_triple_explainable(
                        triple_obj,
                        confidence_threshold=self.confidence_threshold,
                        evidence_sources=self.evidence_sources
                    )
                    
                    if result:
                        explainable_results.append(result)
                        self.terminal_logger.log(f"  ✓ Judgment: {result.judgment}, Confidence: {result.confidence:.2f}")
                    else:
                        self.terminal_logger.log(f"  ❌ Failed to get judgment for triple")
                
                # Save results
                await self._save_explainable_results(explainable_results, csv_output, reasoning_output)
                
                self.terminal_logger.log(f"✅ Explainable judgment completed")
                self.terminal_logger.log(f"  - CSV results: {csv_output}")
                self.terminal_logger.log(f"  - Reasoning details: {reasoning_output}")
                self.terminal_logger.log(f"  - Processed {len(explainable_results)} triples")
                
                return True
            else:
                # Fallback execution
                return await self._execute_fallback_explainable(triples_file, csv_output, reasoning_output)
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Explainable judgment error: {str(e)}")
            return False
    
    async def _execute_gold_label_bootstrapping(self, triples_file: str, output_dir: str, 
                                              iteration_path: str, **kwargs) -> bool:
        """
        Execute gold label bootstrapping functionality.
        
        Args:
            triples_file: Path to triples input file
            output_dir: Output directory
            iteration_path: Path to iteration directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting Gold Label Bootstrapping")
            self.terminal_logger.log(f"  - Triples input: {triples_file}")
            self.terminal_logger.log(f"  - Fuzzy threshold: {self.gold_bootstrap_config['fuzzy_threshold']}")
            self.terminal_logger.log(f"  - Sample rate: {self.gold_bootstrap_config['sample_rate']}")
            self.terminal_logger.log(f"  - Output: {output_dir}")
            
            # Find source text file for bootstrapping
            source_file = self._find_source_text(iteration_path)
            if not source_file:
                self.terminal_logger.log(f"❌ Source text file not found for bootstrapping")
                return False
            
            # Load triples from file
            triples_data = self._load_triples_data(triples_file)
            if not triples_data:
                self.terminal_logger.log(f"❌ Failed to load triples data")
                return False
            
            # Create output file
            bootstrap_output = os.path.join(output_dir, f"bootstrap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Execute bootstrapping if components available
            if self.bootstrapper:
                self.terminal_logger.log(f"🔄 Executing two-stage bootstrapping...")
                
                # Convert triples to TripleData objects
                triple_objects = [
                    TripleData(
                        subject=triple.get('subject', ''),
                        predicate=triple.get('predicate', ''),
                        object=triple.get('object', '')
                    )
                    for triple in triples_data
                ]
                
                # Execute bootstrapping
                bootstrap_results = await self.bootstrapper.bootstrap_gold_labels(
                    triples=triple_objects,
                    source_file=source_file,
                    output_file=bootstrap_output,
                    **self.gold_bootstrap_config
                )
                
                if bootstrap_results:
                    self.terminal_logger.log(f"✅ Gold label bootstrapping completed")
                    self.terminal_logger.log(f"  - Bootstrap results: {bootstrap_output}")
                    self.terminal_logger.log(f"  - Processed {len(bootstrap_results)} triples")
                    
                    # Save detailed bootstrap results
                    detailed_output = os.path.join(output_dir, f"bootstrap_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    await self._save_bootstrap_details(bootstrap_results, detailed_output)
                    
                    return True
                else:
                    self.terminal_logger.log(f"❌ Gold label bootstrapping failed")
                    return False
            else:
                # Fallback execution
                return await self._execute_fallback_bootstrapping(triples_file, source_file, bootstrap_output)
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Gold label bootstrapping error: {str(e)}")
            return False
    
    async def _execute_streaming_judgment(self, triples_file: str, output_dir: str, **kwargs) -> bool:
        """
        Execute streaming graph judgment for real-time processing.
        
        Args:
            triples_file: Path to triples input file
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.terminal_logger.log(f"🚀 Starting Streaming Graph Judgment")
            self.terminal_logger.log(f"  - Input: {triples_file}")
            self.terminal_logger.log(f"  - Streaming mode enabled")
            self.terminal_logger.log(f"  - Output: {output_dir}")
            
            # Load triples from file
            triples_data = self._load_triples_data(triples_file)
            if not triples_data:
                self.terminal_logger.log(f"❌ Failed to load triples data")
                return False
            
            # Create streaming output file
            streaming_output = os.path.join(output_dir, f"streaming_judgment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            
            # Process triples in streaming mode
            processed_count = 0
            with open(streaming_output, 'w', encoding='utf-8') as f:
                for i, triple in enumerate(triples_data):
                    self.terminal_logger.log(f"🔄 Streaming triple {i+1}/{len(triples_data)}")
                    
                    # Create TripleData object
                    triple_obj = TripleData(
                        subject=triple.get('subject', ''),
                        predicate=triple.get('predicate', ''),
                        object=triple.get('object', '')
                    )
                    
                    # Get judgment
                    if self.graph_judge:
                        result = await self.graph_judge.judge_triple(triple_obj)
                        
                        # Write result immediately (streaming)
                        result_data = {
                            'triple_id': i,
                            'triple': triple,
                            'judgment': result,
                            'timestamp': datetime.now().isoformat(),
                            'confidence': getattr(result, 'confidence', None)
                        }
                        
                        f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                        f.flush()  # Ensure immediate write
                        
                        processed_count += 1
                    
                    # Add small delay to simulate real-time processing
                    await asyncio.sleep(0.1)
            
            self.terminal_logger.log(f"✅ Streaming judgment completed")
            self.terminal_logger.log(f"  - Streaming results: {streaming_output}")
            self.terminal_logger.log(f"  - Processed {processed_count} triples")
            
            return True
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Streaming judgment error: {str(e)}")
            return False
    
    def _load_triples_data(self, triples_file: str) -> List[Dict[str, Any]]:
        """
        Load triples data from file.
        
        Args:
            triples_file: Path to triples file
            
        Returns:
            List[Dict[str, Any]]: Loaded triples data
        """
        try:
            with open(triples_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'triples' in data:
                    return data['triples']
                else:
                    # Extract triples from complex structure
                    return self._extract_triples_from_complex_data(data)
            except json.JSONDecodeError:
                # Parse as text format
                return self._parse_text_triples(content)
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Error loading triples data: {str(e)}")
            return []
    
    def _extract_triples_from_complex_data(self, data: Any) -> List[Dict[str, Any]]:
        """Extract triples from complex nested data structures."""
        triples = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'triples' and isinstance(value, list):
                    triples.extend(value)
                elif isinstance(value, (dict, list)):
                    triples.extend(self._extract_triples_from_complex_data(value))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if all(k in item for k in ['subject', 'predicate', 'object']):
                        triples.append(item)
                    else:
                        triples.extend(self._extract_triples_from_complex_data(item))
        
        return triples
    
    def _parse_text_triples(self, content: str) -> List[Dict[str, Any]]:
        """Parse triples from text content."""
        triples = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try different separator patterns
            for separator in ['|', '\t', ',']:
                if separator in line:
                    parts = line.split(separator)
                    if len(parts) >= 3:
                        triples.append({
                            'subject': parts[0].strip(),
                            'predicate': parts[1].strip(),
                            'object': parts[2].strip()
                        })
                        break
        
        return triples
    
    def _find_source_text(self, iteration_path: str) -> Optional[str]:
        """Find source text file for bootstrapping."""
        # Look for source text in various locations
        possible_locations = [
            os.path.join(iteration_path, "source.txt"),
            os.path.join(iteration_path, "input.txt"),
            os.path.join(iteration_path, "results", "ectd", "test_denoised.target"),
            os.path.join(iteration_path, "raw_text.txt")
        ]
        
        for location in possible_locations:
            if os.path.exists(location) and os.path.getsize(location) > 0:
                return location
        
        # Look in parent directories
        parent_dir = Path(iteration_path).parent
        for pattern in ["source.txt", "input.txt", "*.txt"]:
            for file_path in parent_dir.glob(pattern):
                if file_path.is_file() and file_path.stat().st_size > 0:
                    return str(file_path)
        
        return None
    
    async def _save_explainable_results(self, results: List, csv_output: str, reasoning_output: str):
        """Save explainable judgment results."""
        try:
            # Save CSV format
            import csv
            with open(csv_output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Predicate', 'Object', 'Judgment', 'Confidence', 'Reasoning'])
                
                for result in results:
                    writer.writerow([
                        result.triple.subject,
                        result.triple.predicate,
                        result.triple.object,
                        result.judgment,
                        result.confidence,
                        result.reasoning[:200] + '...' if len(result.reasoning) > 200 else result.reasoning
                    ])
            
            # Save detailed reasoning JSON
            reasoning_data = []
            for result in results:
                reasoning_data.append({
                    'triple': {
                        'subject': result.triple.subject,
                        'predicate': result.triple.predicate,
                        'object': result.triple.object
                    },
                    'judgment': result.judgment,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'evidence': result.evidence if hasattr(result, 'evidence') else [],
                    'alternatives': result.alternatives if hasattr(result, 'alternatives') else []
                })
            
            with open(reasoning_output, 'w', encoding='utf-8') as f:
                json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Error saving explainable results: {str(e)}")
    
    async def _save_bootstrap_details(self, results: List, output_file: str):
        """Save detailed bootstrap results."""
        try:
            bootstrap_data = []
            for result in results:
                bootstrap_data.append({
                    'triple': {
                        'subject': result.triple.subject,
                        'predicate': result.triple.predicate,
                        'object': result.triple.object
                    },
                    'fuzzy_match_score': result.fuzzy_score,
                    'llm_judgment': result.llm_judgment,
                    'confidence': result.confidence,
                    'bootstrap_stage': result.stage,  # Stage 1 (fuzzy) or Stage 2 (LLM)
                    'reasoning': result.reasoning if hasattr(result, 'reasoning') else ''
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(bootstrap_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Error saving bootstrap details: {str(e)}")
    
    async def _execute_fallback_judgment(self, triples_file: str, output_file: str) -> bool:
        """Execute fallback judgment using subprocess."""
        try:
            self.terminal_logger.log(f"🔄 Using fallback judgment execution")
            
            # Use subprocess to run graphJudge_Phase main
            import subprocess
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(chat_dir / "graphJudge_Phase" / "main.py"),
                "--input", triples_file,
                "--output", output_file,
                "--model", self.model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(chat_dir)
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                self.terminal_logger.log(f"📄 STDOUT:\n{stdout.decode('utf-8', errors='ignore')}")
            if stderr:
                self.terminal_logger.log(f"⚠️  STDERR:\n{stderr.decode('utf-8', errors='ignore')}")
            
            return process.returncode == 0
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Fallback execution error: {str(e)}")
            return False
    
    async def _execute_fallback_explainable(self, triples_file: str, csv_output: str, reasoning_output: str) -> bool:
        """Execute fallback explainable judgment."""
        try:
            self.terminal_logger.log(f"🔄 Using fallback explainable execution")
            
            import subprocess
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(chat_dir / "graphJudge_Phase" / "main.py"),
                "--input", triples_file,
                "--output", csv_output,
                "--explainable",
                "--reasoning-file", reasoning_output,
                "--model", self.model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(chat_dir)
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                self.terminal_logger.log(f"📄 STDOUT:\n{stdout.decode('utf-8', errors='ignore')}")
            if stderr:
                self.terminal_logger.log(f"⚠️  STDERR:\n{stderr.decode('utf-8', errors='ignore')}")
            
            return process.returncode == 0
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Fallback explainable execution error: {str(e)}")
            return False
    
    async def _execute_fallback_bootstrapping(self, triples_file: str, source_file: str, output_file: str) -> bool:
        """Execute fallback bootstrapping."""
        try:
            self.terminal_logger.log(f"🔄 Using fallback bootstrapping execution")
            
            import subprocess
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(chat_dir / "graphJudge_Phase" / "main.py"),
                "--bootstrap",
                "--triples-file", triples_file,
                "--source-file", source_file,
                "--output", output_file,
                "--threshold", str(self.gold_bootstrap_config['fuzzy_threshold']),
                "--sample-rate", str(self.gold_bootstrap_config['sample_rate']),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(chat_dir)
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                self.terminal_logger.log(f"📄 STDOUT:\n{stdout.decode('utf-8', errors='ignore')}")
            if stderr:
                self.terminal_logger.log(f"⚠️  STDERR:\n{stderr.decode('utf-8', errors='ignore')}")
            
            return process.returncode == 0
            
        except Exception as e:
            self.terminal_logger.log(f"❌ Fallback bootstrapping execution error: {str(e)}")
            return False
