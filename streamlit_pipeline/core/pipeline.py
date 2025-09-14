"""
Pipeline Orchestrator for GraphJudge Streamlit Application.

This module provides the main orchestration logic for the three-stage
GraphJudge pipeline: Entity Extraction → Triple Generation → Graph Judgment.

Implements the user flows from spec.md Section 5 and system architecture 
from Section 6, with proper error handling and progress tracking.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

try:
    from .entity_processor import extract_entities
    from .triple_generator import generate_triples
    from .graph_judge import judge_triples
    from .models import EntityResult, TripleResult, JudgmentResult, Triple, PipelineState
    from ..utils.error_handling import ErrorHandler, ErrorType, safe_execute
    from ..utils.api_client import get_api_client
    from ..utils.storage_manager import get_storage_manager, create_new_pipeline_iteration, save_phase_result
    from ..utils.detailed_logger import DetailedLogger
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.entity_processor import extract_entities
    from core.triple_generator import generate_triples
    from core.graph_judge import judge_triples
    from core.models import EntityResult, TripleResult, JudgmentResult, Triple, PipelineState
    from utils.error_handling import ErrorHandler, ErrorType, safe_execute
    from utils.api_client import get_api_client
    from utils.storage_manager import get_storage_manager, create_new_pipeline_iteration, save_phase_result
    from utils.detailed_logger import DetailedLogger


@dataclass
class PipelineResult:
    """Complete pipeline result containing all stage outputs."""
    success: bool
    stage_reached: int  # 0=entity, 1=triple, 2=judgment, 3=complete
    total_time: float
    
    # Stage results
    entity_result: Optional[EntityResult] = None
    triple_result: Optional[TripleResult] = None
    judgment_result: Optional[JudgmentResult] = None
    
    # Error information
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    # Summary statistics
    stats: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize stats if not provided."""
        if self.stats is None:
            self.stats = {}


class PipelineOrchestrator:
    """
    Main orchestrator for the GraphJudge three-stage pipeline.
    
    Coordinates execution of entity extraction, triple generation, and
    graph judgment stages with comprehensive error handling and progress tracking.
    
    Following spec.md requirements for clean interfaces and session state management.
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.error_handler = ErrorHandler()
        self.logger = logging.getLogger(__name__)
        self.detailed_logger = DetailedLogger(phase="pipeline")
        self.current_stage = -1
        self.pipeline_state = PipelineState()
        self.storage_manager = get_storage_manager()
        
    def run_pipeline(self, input_text: str, progress_callback=None) -> PipelineResult:
        """
        Execute the complete three-stage GraphJudge pipeline.

        Args:
            input_text: The raw Chinese text to process
            progress_callback: Optional callback function for progress updates

        Returns:
            PipelineResult containing all stage outputs and metadata
        """
        start_time = time.time()

        self.detailed_logger.log_pipeline_start({
            "input_length": len(input_text) if input_text else 0,
            "timestamp": datetime.now().isoformat()
        })

        # Input validation
        if not input_text or not input_text.strip():
            error_msg = "Input text is empty or contains only whitespace"
            self.detailed_logger.log_error("PIPELINE", "Input validation failed", {"error": error_msg})
            return PipelineResult(
                success=False,
                stage_reached=0,
                total_time=0.0,
                error=error_msg,
                error_stage="input_validation"
            )

        # Create new iteration folder for this pipeline run
        iteration_path = self.storage_manager.create_new_iteration(input_text.strip())
        self.detailed_logger.log_info("STORAGE", f"Created iteration folder: {iteration_path}")

        # Initialize pipeline state
        self.pipeline_state = PipelineState(
            input_text=input_text.strip(),
            started_at=datetime.now().isoformat()
        )

        result = PipelineResult(
            success=False,
            stage_reached=0,
            total_time=0.0
        )

        try:
            # Stage 1: Entity Extraction
            self.current_stage = 0
            self.detailed_logger.log_info("PIPELINE", "Starting Stage 1: Entity Extraction", {
                "stage": "entity_extraction",
                "input_preview": input_text[:200] + ("..." if len(input_text) > 200 else "")
            })

            if progress_callback:
                progress_callback(0, "Starting entity extraction...")

            entity_result = self._execute_entity_stage(input_text)
            result.entity_result = entity_result

            if not entity_result.success:
                self.detailed_logger.log_error("ENTITY", "Entity extraction failed", {
                    "error": entity_result.error,
                    "processing_time": entity_result.processing_time
                })
                result.error = entity_result.error
                result.error_stage = "entity_extraction"
                return result

            # Check for empty entities
            if not entity_result.entities:
                error_msg = "No entities were found in the input text"
                self.detailed_logger.log_error("ENTITY", error_msg, {
                    "denoised_text_length": len(entity_result.denoised_text) if entity_result.denoised_text else 0,
                    "processing_time": entity_result.processing_time
                })
                result.error = error_msg
                result.error_stage = "entity_extraction"
                return result

            self.detailed_logger.log_info("ENTITY", "Entity extraction completed successfully", {
                "entity_count": len(entity_result.entities),
                "entities": entity_result.entities[:10],  # First 10 entities for debugging
                "denoised_text_length": len(entity_result.denoised_text) if entity_result.denoised_text else 0,
                "processing_time": entity_result.processing_time
            })

            # Save ECTD results to storage
            try:
                saved_path = self.storage_manager.save_entity_result(entity_result)
                self.detailed_logger.log_info("STORAGE", f"ECTD results saved to: {saved_path}")
            except Exception as e:
                self.detailed_logger.log_error("STORAGE", f"Failed to save ECTD results: {e}")

            result.stage_reached = 1
            self.pipeline_state.entity_result = entity_result

            if progress_callback:
                progress_callback(1, f"Found {len(entity_result.entities)} entities. Starting triple generation...")

            # Stage 2: Triple Generation
            self.current_stage = 1
            self.detailed_logger.log_info("PIPELINE", "Starting Stage 2: Triple Generation", {
                "stage": "triple_generation",
                "input_entities": entity_result.entities[:5],  # First 5 entities for debugging
                "denoised_text_preview": entity_result.denoised_text[:300] + ("..." if len(entity_result.denoised_text) > 300 else "")
            })

            triple_result = self._execute_triple_stage(entity_result.entities, entity_result.denoised_text)
            result.triple_result = triple_result

            if not triple_result.success:
                self.detailed_logger.log_error("TRIPLE", "Triple generation failed", {
                    "error": triple_result.error,
                    "processing_time": triple_result.processing_time,
                    "metadata": triple_result.metadata
                })
                result.error = triple_result.error
                result.error_stage = "triple_generation"
                return result

            # Check for empty triples
            if not triple_result.triples:
                error_msg = "No triples were generated from the extracted entities"
                self.detailed_logger.log_error("TRIPLE", error_msg, {
                    "entities_count": len(entity_result.entities),
                    "entities": entity_result.entities,
                    "processing_time": triple_result.processing_time,
                    "metadata": triple_result.metadata
                })
                result.error = error_msg
                result.error_stage = "triple_generation"
                return result

            self.detailed_logger.log_info("TRIPLE", "Triple generation completed successfully", {
                "triple_count": len(triple_result.triples),
                "triples": [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in triple_result.triples[:5]],  # First 5 triples
                "processing_time": triple_result.processing_time,
                "metadata": triple_result.metadata
            })

            # Save Triple Generation results to storage
            try:
                saved_path = self.storage_manager.save_triple_result(triple_result)
                self.detailed_logger.log_info("STORAGE", f"Triple results saved to: {saved_path}")
            except Exception as e:
                self.detailed_logger.log_error("STORAGE", f"Failed to save Triple results: {e}")

            result.stage_reached = 2
            self.pipeline_state.triple_result = triple_result

            if progress_callback:
                progress_callback(2, f"Generated {len(triple_result.triples)} triples. Starting graph judgment...")

            # Stage 3: Graph Judgment
            self.current_stage = 2
            self.detailed_logger.log_info("PIPELINE", "Starting Stage 3: Graph Judgment", {
                "stage": "graph_judgment",
                "input_triples_count": len(triple_result.triples)
            })

            judgment_result = self._execute_judgment_stage(triple_result.triples)
            result.judgment_result = judgment_result

            if not judgment_result.success:
                self.detailed_logger.log_error("JUDGMENT", "Graph judgment failed", {
                    "error": judgment_result.error,
                    "processing_time": judgment_result.processing_time
                })
                result.error = judgment_result.error
                result.error_stage = "graph_judgment"
                return result

            approved_count = sum(1 for j in judgment_result.judgments if j)
            self.detailed_logger.log_info("JUDGMENT", "Graph judgment completed successfully", {
                "total_judgments": len(judgment_result.judgments),
                "approved_triples": approved_count,
                "rejected_triples": len(judgment_result.judgments) - approved_count,
                "processing_time": judgment_result.processing_time
            })

            # Save Graph Judgment results to storage
            try:
                saved_path = self.storage_manager.save_judgment_result(judgment_result, triple_result.triples)
                self.detailed_logger.log_info("STORAGE", f"Judgment results saved to: {saved_path}")
            except Exception as e:
                self.detailed_logger.log_error("STORAGE", f"Failed to save Judgment results: {e}")

            result.stage_reached = 3
            self.pipeline_state.judgment_result = judgment_result

            if progress_callback:
                progress_callback(3, "Pipeline completed successfully!")

            # Pipeline completed successfully
            result.success = True
            result.total_time = time.time() - start_time

            # Generate summary statistics
            result.stats = self._generate_stats(result)

            self.detailed_logger.log_pipeline_complete({
                "success": True,
                "total_time": result.total_time,
                "stats": result.stats
            })

            return result

        except Exception as e:
            self.detailed_logger.log_error("PIPELINE", f"Unexpected error in pipeline", {
                "error": str(e),
                "error_type": type(e).__name__,
                "current_stage": self.current_stage,
                "traceback": str(e.__traceback__)
            })
            self.logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            result.error = f"Unexpected error: {str(e)}"
            result.error_stage = f"stage_{self.current_stage}"
            result.total_time = time.time() - start_time
            return result
    
    def _execute_entity_stage(self, input_text: str) -> EntityResult:
        """
        Execute the entity extraction stage.
        
        Args:
            input_text: Raw text input
            
        Returns:
            EntityResult containing extracted entities and denoised text
        """
        def entity_execution():
            return extract_entities(input_text)
        
        entity_result, error_info = safe_execute(
            entity_execution,
            logger=self.logger,
            stage="entity_extraction"
        )
        
        # If safe_execute returned an error, convert to EntityResult
        if error_info is not None:
            return EntityResult(
                entities=[],
                denoised_text=input_text,
                success=False,
                processing_time=0.0,
                error=error_info.message
            )
        
        return entity_result
    
    def _execute_triple_stage(self, entities: List[str], denoised_text: str) -> TripleResult:
        """
        Execute the triple generation stage.
        
        Args:
            entities: List of extracted entities
            denoised_text: Cleaned text from entity stage
            
        Returns:
            TripleResult containing generated triples
        """
        def triple_execution():
            return generate_triples(entities, denoised_text)
        
        triple_result, error_info = safe_execute(
            triple_execution,
            logger=self.logger,
            stage="triple_generation"
        )
        
        # If safe_execute returned an error, convert to TripleResult
        if error_info is not None:
            return TripleResult(
                triples=[],
                metadata={},
                success=False,
                processing_time=0.0,
                error=error_info.message
            )
        
        return triple_result
    
    def _execute_judgment_stage(self, triples: List[Triple]) -> JudgmentResult:
        """
        Execute the graph judgment stage.
        
        Args:
            triples: List of triples to judge
            
        Returns:
            JudgmentResult containing judgment outcomes
        """
        def judgment_execution():
            return judge_triples(triples)
        
        judgment_result, error_info = safe_execute(
            judgment_execution,
            logger=self.logger,
            stage="graph_judgment"
        )
        
        # If safe_execute returned an error, convert to JudgmentResult
        if error_info is not None:
            return JudgmentResult(
                judgments=[],
                confidence=[],
                explanations=None,
                success=False,
                processing_time=0.0,
                error=error_info.message
            )
        
        return judgment_result
    
    def _generate_stats(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Generate summary statistics for the pipeline run.
        
        Args:
            result: The pipeline result to analyze
            
        Returns:
            Dictionary containing summary statistics
        """
        stats = {
            'input_length': len(self.pipeline_state.input_text),
            'total_time': result.total_time
        }
        
        if result.entity_result:
            stats.update({
                'entity_count': len(result.entity_result.entities),
                'entity_time': result.entity_result.processing_time
            })
        
        if result.triple_result:
            stats.update({
                'triple_count': len(result.triple_result.triples),
                'triple_time': result.triple_result.processing_time
            })
            
        if result.judgment_result:
            stats.update({
                'judgment_count': len(result.judgment_result.judgments),
                'judgment_time': result.judgment_result.processing_time,
                'approved_triples': sum(1 for j in result.judgment_result.judgments if j),
                'rejected_triples': sum(1 for j in result.judgment_result.judgments if not j)
            })
            
            # Calculate approval rate
            if result.judgment_result.judgments:
                stats['approval_rate'] = stats['approved_triples'] / len(result.judgment_result.judgments)
        
        return stats
    
    def get_pipeline_state(self) -> PipelineState:
        """
        Get the current pipeline state.
        
        Returns:
            Current PipelineState object
        """
        return self.pipeline_state
    
    def reset_pipeline(self):
        """Reset the pipeline state for a new run."""
        self.current_stage = -1
        self.pipeline_state = PipelineState()


def run_full_pipeline(input_text: str, progress_callback=None) -> PipelineResult:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        input_text: The raw Chinese text to process
        progress_callback: Optional progress callback function
        
    Returns:
        PipelineResult with all stage outputs
    """
    orchestrator = PipelineOrchestrator()
    return orchestrator.run_pipeline(input_text, progress_callback)