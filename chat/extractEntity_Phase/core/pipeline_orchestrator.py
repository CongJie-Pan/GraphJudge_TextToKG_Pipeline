"""
Pipeline Orchestrator Module

This module implements the main pipeline coordination for the ECTD (Entity Extraction
and Text Denoising) pipeline, orchestrating the complete workflow from input to output.

The module provides a high-level interface for running the complete pipeline with
comprehensive error handling, progress tracking, and result management.
"""

import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from extractEntity_Phase.core.entity_extractor import EntityExtractor, ExtractionConfig
from extractEntity_Phase.core.text_denoiser import TextDenoiser, DenoisingConfig
from extractEntity_Phase.api.gpt5mini_client import GPT5MiniClient, GPT5MiniConfig
from extractEntity_Phase.infrastructure.logging import get_logger
from extractEntity_Phase.infrastructure.config import get_config
from extractEntity_Phase.models.entities import EntityList
from extractEntity_Phase.models.pipeline_state import (
    PipelineState, PipelineStage, ProcessingStatus, StageProgress, PipelineError
)
from extractEntity_Phase.utils.chinese_text import ChineseTextProcessor


@dataclass
class PipelineConfig:
    """Configuration for the complete ECTD pipeline."""
    
    # Input/Output settings
    input_file: Optional[str] = None
    output_dir: Optional[str] = None
    iteration: int = 1
    
    # Entity extraction settings
    extraction_config: ExtractionConfig = None
    
    # Text denoising settings
    denoising_config: DenoisingConfig = None
    
    # GPT-5-mini settings
    gpt5mini_config: GPT5MiniConfig = None
    
    # Processing settings
    batch_size: int = 10
    max_concurrent: int = 3
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    
    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True
    create_backup: bool = True
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        if self.extraction_config is None:
            self.extraction_config = ExtractionConfig()
        
        if self.denoising_config is None:
            self.denoising_config = DenoisingConfig()
        
        if self.gpt5mini_config is None:
            self.gpt5mini_config = GPT5MiniConfig()


class PipelineOrchestrator:
    """
    Main pipeline orchestrator for the ECTD workflow.
    
    This class coordinates the complete entity extraction and text denoising pipeline,
    managing the workflow, error handling, and result generation.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger()
        self.text_processor = ChineseTextProcessor()
        
        # Initialize components
        self.gpt5mini_client = None
        self.entity_extractor = None
        self.text_denoiser = None
        
        # Pipeline state
        self.pipeline_state = PipelineState()
        self.current_stage = None
        
        # Results storage
        self.extracted_entities: List[EntityList] = []
        self.denoised_texts: List[str] = []
        self.original_texts: List[str] = []
        
        # Statistics
        self.pipeline_stats = {
            "start_time": None,
            "end_time": None,
            "total_duration": 0.0,
            "total_texts_processed": 0,
            "total_entities_extracted": 0,
            "total_texts_denoised": 0,
            "success_rate": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def run_pipeline(self, input_texts: Optional[List[str]] = None) -> bool:
        """
        Run the complete ECTD pipeline.
        
        Args:
            input_texts: Optional list of input texts (if not using input file)
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            self.logger.log("🚀 Starting ECTD Pipeline...")
            self.pipeline_state.start_pipeline()
            self.pipeline_stats["start_time"] = datetime.now()
            
            # Initialize pipeline components
            await self._initialize_components()
            
            # Load input texts
            texts = await self._load_input_texts(input_texts)
            if not texts:
                raise ValueError("No input texts available")
            
            self.original_texts = texts
            self.pipeline_stats["total_texts_processed"] = len(texts)
            
            # Stage 1: Entity Extraction
            await self._run_entity_extraction_stage(texts)
            
            # Stage 2: Text Denoising
            await self._run_text_denoising_stage(texts)
            
            # Stage 3: Generate Outputs
            await self._run_output_generation_stage()
            
            # Stage 4: Finalize Pipeline
            await self._run_finalization_stage()
            
            self.logger.log("✅ ECTD Pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.log(f"❌ Pipeline failed: {str(e)}")
            self.pipeline_state.mark_failed(str(e))
            await self._handle_pipeline_failure(e)
            return False
        
        finally:
            self.pipeline_stats["end_time"] = datetime.now()
            if self.pipeline_stats["start_time"]:
                duration = self.pipeline_stats["end_time"] - self.pipeline_stats["start_time"]
                self.pipeline_stats["total_duration"] = duration.total_seconds()
    
    async def _initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.log("🔧 Initializing pipeline components...")
        
        # Initialize GPT-5-mini client
        self.gpt5mini_client = GPT5MiniClient(self.config.gpt5mini_config)
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(
            self.gpt5mini_client, 
            self.config.extraction_config
        )
        
        # Initialize text denoiser
        self.text_denoiser = TextDenoiser(
            self.gpt5mini_client, 
            self.config.denoising_config
        )
        
        self.logger.log("✅ Pipeline components initialized")
    
    async def _load_input_texts(self, input_texts: Optional[List[str]] = None) -> List[str]:
        """
        Load input texts from file or provided list.
        
        Args:
            input_texts: Optional list of input texts
            
        Returns:
            List of input text strings
        """
        if input_texts:
            self.logger.log(f"📖 Using {len(input_texts)} provided input texts")
            return input_texts
        
        # Load from input file
        if not self.config.input_file:
            raise ValueError("No input file specified and no texts provided")
        
        if not os.path.exists(self.config.input_file):
            raise ValueError(f"Input file not found: {self.config.input_file}")
        
        self.logger.log(f"📖 Loading input texts from: {self.config.input_file}")
        
        try:
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
            
            self.logger.log(f"✅ Loaded {len(texts)} input texts")
            return texts
            
        except Exception as e:
            raise ValueError(f"Failed to load input file: {str(e)}")
    
    async def _run_entity_extraction_stage(self, texts: List[str]):
        """Run the entity extraction stage."""
        self.logger.log("🔍 Starting Entity Extraction Stage...")
        
        stage_progress = StageProgress(
            stage=PipelineStage.ENTITY_EXTRACTION,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now(),
            total_items=len(texts)
        )
        
        self.current_stage = stage_progress
        self.pipeline_state.update_stage_progress(stage_progress)
        
        try:
            # Progress callback for monitoring
            def progress_callback(progress: float, message: str):
                stage_progress.progress_percentage = progress * 100
                stage_progress.current_item = message
                self.pipeline_state.update_stage_progress(stage_progress)
            
            # Extract entities
            self.extracted_entities = await self.entity_extractor.extract_entities_from_texts(
                texts, progress_callback
            )
            
            # Validate extraction results
            successful_extractions = sum(1 for ec in self.extracted_entities if ec.entities)
            success_rate = successful_extractions / len(texts) if texts else 0
            
            self.logger.log(f"✅ Entity extraction completed. "
                           f"Success rate: {success_rate:.1%} ({successful_extractions}/{len(texts)})")
            
            # Update stage progress
            stage_progress.status = ProcessingStatus.COMPLETED
            stage_progress.end_time = datetime.now()
            stage_progress.progress_percentage = 100.0
            stage_progress.items_processed = len(texts)
            self.pipeline_state.update_stage_progress(stage_progress)
            
            # Update pipeline statistics
            total_entities = sum(len(ec.entities) for ec in self.extracted_entities)
            self.pipeline_stats["total_entities_extracted"] = total_entities
            
        except Exception as e:
            self.logger.log(f"❌ Entity extraction stage failed: {str(e)}")
            stage_progress.status = ProcessingStatus.FAILED
            stage_progress.errors.append(PipelineError(
                stage=PipelineStage.ENTITY_EXTRACTION,
                severity="ERROR",
                message=str(e),
                exception=e
            ))
            self.pipeline_state.update_stage_progress(stage_progress)
            raise
    
    async def _run_text_denoising_stage(self, texts: List[str]):
        """Run the text denoising stage."""
        self.logger.log("🧹 Starting Text Denoising Stage...")
        
        stage_progress = StageProgress(
            stage=PipelineStage.TEXT_DENOISING,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now(),
            total_items=len(texts)
        )
        
        self.current_stage = stage_progress
        self.pipeline_state.update_stage_progress(stage_progress)
        
        try:
            # Progress callback for monitoring
            def progress_callback(progress: float, message: str):
                stage_progress.progress_percentage = progress * 100
                stage_progress.current_item = message
                self.pipeline_state.update_stage_progress(stage_progress)
            
            # Denoise texts using extracted entities
            self.denoised_texts = await self.text_denoiser.denoise_texts(
                texts, self.extracted_entities, progress_callback
            )
            
            # Validate denoising results
            successful_denoising = sum(1 for text in self.denoised_texts if text)
            success_rate = successful_denoising / len(texts) if texts else 0
            
            self.logger.log(f"✅ Text denoising completed. "
                           f"Success rate: {success_rate:.1%} ({successful_denoising}/{len(texts)})")
            
            # Update stage progress
            stage_progress.status = ProcessingStatus.COMPLETED
            stage_progress.end_time = datetime.now()
            stage_progress.progress_percentage = 100.0
            stage_progress.items_processed = len(texts)
            self.pipeline_state.update_stage_progress(stage_progress)
            
            # Update pipeline statistics
            self.pipeline_stats["total_texts_denoised"] = successful_denoising
            
        except Exception as e:
            self.logger.log(f"❌ Text denoising stage failed: {str(e)}")
            stage_progress.status = ProcessingStatus.FAILED
            stage_progress.errors.append(PipelineError(
                stage=PipelineStage.TEXT_DENOISING,
                severity="ERROR",
                message=str(e),
                exception=e
            ))
            self.pipeline_state.update_stage_progress(stage_progress)
            raise
    
    async def _run_output_generation_stage(self):
        """Run the output generation stage."""
        self.logger.log("📤 Starting Output Generation Stage...")
        
        stage_progress = StageProgress(
            stage=PipelineStage.OUTPUT_GENERATION,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now(),
            total_items=1
        )
        
        self.current_stage = stage_progress
        self.pipeline_state.update_stage_progress(stage_progress)
        
        try:
            # Create output directory
            output_dir = self._create_output_directory()
            
            # Save extracted entities
            entity_file = os.path.join(output_dir, "test_entity.txt")
            await self._save_entities(entity_file)
            
            # Save denoised texts
            denoised_file = os.path.join(output_dir, "test_denoised.target")
            await self._save_denoised_texts(denoised_file)
            
            # Save pipeline statistics
            stats_file = os.path.join(output_dir, "pipeline_statistics.json")
            await self._save_pipeline_statistics(stats_file)
            
            # Save pipeline state
            state_file = os.path.join(output_dir, "pipeline_state.json")
            await self._save_pipeline_state(state_file)
            
            self.logger.log(f"✅ Output generation completed. Results saved to: {output_dir}")
            
            # Update stage progress
            stage_progress.status = ProcessingStatus.COMPLETED
            stage_progress.end_time = datetime.now()
            stage_progress.progress_percentage = 100.0
            stage_progress.items_processed = 1
            self.pipeline_state.update_stage_progress(stage_progress)
            
        except Exception as e:
            self.logger.log(f"❌ Output generation stage failed: {str(e)}")
            stage_progress.status = ProcessingStatus.FAILED
            stage_progress.errors.append(PipelineError(
                stage=PipelineStage.OUTPUT_GENERATION,
                severity="ERROR",
                message=str(e),
                exception=e
            ))
            self.pipeline_state.update_stage_progress(stage_progress)
            raise
    
    async def _run_finalization_stage(self):
        """Run the pipeline finalization stage."""
        self.logger.log("🎯 Starting Pipeline Finalization Stage...")
        
        stage_progress = StageProgress(
            stage=PipelineStage.CLEANUP,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.now(),
            total_items=1
        )
        
        self.current_stage = stage_progress
        self.pipeline_state.update_stage_progress(stage_progress)
        
        try:
            # Calculate final statistics
            self._calculate_final_statistics()
            
            # Mark pipeline as completed
            self.pipeline_state.mark_completed()
            
            # Log final summary
            self._log_pipeline_summary()
            
            self.logger.log("✅ Pipeline finalization completed")
            
            # Update stage progress
            stage_progress.status = ProcessingStatus.COMPLETED
            stage_progress.end_time = datetime.now()
            stage_progress.progress_percentage = 100.0
            stage_progress.items_processed = 1
            self.pipeline_state.update_stage_progress(stage_progress)
            
        except Exception as e:
            self.logger.log(f"❌ Pipeline finalization failed: {str(e)}")
            stage_progress.status = ProcessingStatus.FAILED
            stage_progress.errors.append(PipelineError(
                stage=PipelineStage.CLEANUP,
                severity="ERROR",
                message=str(e),
                exception=e
            ))
            self.pipeline_state.update_stage_progress(stage_progress)
            raise
    
    def _create_output_directory(self) -> str:
        """
        Create output directory for pipeline results.
        
        Returns:
            Path to created output directory
        """
        if self.config.output_dir:
            output_dir = self.config.output_dir
        else:
            # Use default output directory
            base_dir = os.environ.get('PIPELINE_OUTPUT_DIR', './output')
            output_dir = os.path.join(base_dir, f"Graph_Iteration{self.config.iteration}")
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.log(f"📁 Created output directory: {output_dir}")
        
        return output_dir
    
    async def _save_entities(self, entity_file: str):
        """Save extracted entities to file."""
        try:
            with open(entity_file, 'w', encoding='utf-8') as f:
                for entity_collection in self.extracted_entities:
                    if entity_collection.entities:
                        # Convert entities to string representation
                        entity_strings = [entity.text for entity in entity_collection.entities]
                        f.write(str(entity_strings).strip().replace('\n', '') + '\n')
                    else:
                        f.write('[]\n')
            
            self.logger.log(f"✅ Entities saved to: {entity_file}")
            
        except Exception as e:
            raise ValueError(f"Failed to save entities: {str(e)}")
    
    async def _save_denoised_texts(self, denoised_file: str):
        """Save denoised texts to file."""
        try:
            with open(denoised_file, 'w', encoding='utf-8') as f:
                for denoised_text in self.denoised_texts:
                    if denoised_text:
                        # Clean and save denoised text
                        cleaned_text = str(denoised_text).strip().replace('\n', ' ')
                        f.write(cleaned_text + '\n')
                    else:
                        # Use original text if denoising failed
                        f.write('\n')
            
            self.logger.log(f"✅ Denoised texts saved to: {denoised_file}")
            
        except Exception as e:
            raise ValueError(f"Failed to save denoised texts: {str(e)}")
    
    async def _save_pipeline_statistics(self, stats_file: str):
        """Save pipeline statistics to file."""
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_stats, f, indent=2, default=str)
            
            self.logger.log(f"✅ Pipeline statistics saved to: {stats_file}")
            
        except Exception as e:
            raise ValueError(f"Failed to save pipeline statistics: {str(e)}")
    
    async def _save_pipeline_state(self, state_file: str):
        """Save pipeline state to file."""
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_state.to_dict(), f, indent=2, default=str)
            
            self.logger.log(f"✅ Pipeline state saved to: {state_file}")
            
        except Exception as e:
            raise ValueError(f"Failed to save pipeline state: {str(e)}")
    
    def _calculate_final_statistics(self):
        """Calculate final pipeline statistics."""
        # Calculate success rate
        total_stages = len(self.pipeline_state.stage_progress)
        completed_stages = sum(1 for stage in self.pipeline_state.stage_progress 
                             if stage.status == ProcessingStatus.COMPLETED)
        
        self.pipeline_stats["success_rate"] = completed_stages / total_stages if total_stages > 0 else 0.0
        
        # Calculate cache hit rate
        total_cache_requests = (
            self.entity_extractor.stats.get("cache_hits", 0) + 
            self.entity_extractor.stats.get("cache_misses", 0) +
            self.text_denoiser.stats.get("cache_hits", 0) + 
            self.text_denoiser.stats.get("cache_misses", 0)
        )
        
        total_cache_hits = (
            self.entity_extractor.stats.get("cache_hits", 0) + 
            self.text_denoiser.stats.get("cache_hits", 0)
        )
        
        self.pipeline_stats["cache_hit_rate"] = (
            total_cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        )
    
    def _log_pipeline_summary(self):
        """Log final pipeline summary."""
        self.logger.log("\n" + "=" * 60)
        self.logger.log("📊 ECTD Pipeline Summary")
        self.logger.log("=" * 60)
        self.logger.log(f"📖 Input texts processed: {self.pipeline_stats['total_texts_processed']}")
        self.logger.log(f"🔍 Entities extracted: {self.pipeline_stats['total_entities_extracted']}")
        self.logger.log(f"🧹 Texts denoised: {self.pipeline_stats['total_texts_denoised']}")
        self.logger.log(f"✅ Success rate: {self.pipeline_stats['success_rate']:.1%}")
        self.logger.log(f"📦 Cache hit rate: {self.pipeline_stats['cache_hit_rate']:.1%}")
        self.logger.log(f"⏱️ Total duration: {self.pipeline_stats['total_duration']:.1f} seconds")
        self.logger.log("=" * 60)
    
    async def _handle_pipeline_failure(self, error: Exception):
        """Handle pipeline failure and cleanup."""
        self.logger.log(f"💥 Pipeline failed with error: {str(error)}")
        
        # Try to save partial results if possible
        try:
            if self.extracted_entities or self.denoised_texts:
                output_dir = self._create_output_directory()
                
                # Save partial results
                if self.extracted_entities:
                    entity_file = os.path.join(output_dir, "partial_entities.txt")
                    await self._save_entities(entity_file)
                
                if self.denoised_texts:
                    denoised_file = os.path.join(output_dir, "partial_denoised.txt")
                    await self._save_denoised_texts(denoised_file)
                
                # Save error information
                error_file = os.path.join(output_dir, "pipeline_error.txt")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Pipeline failed at: {datetime.now()}\n")
                    f.write(f"Error: {str(error)}\n")
                    f.write(f"Pipeline state: {self.pipeline_state.status.value}\n")
                
                self.logger.log(f"💾 Partial results saved to: {output_dir}")
        
        except Exception as save_error:
            self.logger.log(f"⚠️ Failed to save partial results: {str(save_error)}")
    
    def get_pipeline_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.pipeline_state
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.pipeline_stats.copy()
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get entity extraction statistics."""
        if self.entity_extractor:
            return self.entity_extractor.get_statistics()
        return {}
    
    def get_denoising_statistics(self) -> Dict[str, Any]:
        """Get text denoising statistics."""
        if self.text_denoiser:
            return self.text_denoiser.get_statistics()
        return {}
    
    def reset_pipeline(self):
        """Reset pipeline state and statistics."""
        self.pipeline_state = PipelineState()
        self.current_stage = None
        self.extracted_entities = []
        self.denoised_texts = []
        self.original_texts = []
        
        self.pipeline_stats = {
            "start_time": None,
            "end_time": None,
            "total_duration": 0.0,
            "total_texts_processed": 0,
            "total_entities_extracted": 0,
            "total_texts_denoised": 0,
            "success_rate": 0.0,
            "cache_hit_rate": 0.0
        }
        
        if self.entity_extractor:
            self.entity_extractor.reset_statistics()
        
        if self.text_denoiser:
            self.text_denoiser.reset_statistics()
        
        self.logger.log("🔄 Pipeline reset completed")
