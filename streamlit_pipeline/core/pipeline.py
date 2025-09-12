"""
Pipeline Orchestrator for GraphJudge Streamlit Application.

This module provides the main orchestration logic for the three-stage
GraphJudge pipeline: Entity Extraction → Triple Generation → Graph Judgment.

Implements the user flows from spec.md Section 5 and system architecture 
from Section 6, with proper error handling and progress tracking.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from .entity_processor import extract_entities
from .triple_generator import generate_triples  
from .graph_judge import judge_triples
from .models import EntityResult, TripleResult, JudgmentResult, Triple, PipelineState
from ..utils.error_handling import ErrorHandler, ErrorType, safe_execute
from ..utils.api_client import get_api_client


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
        self.current_stage = -1
        self.pipeline_state = PipelineState()
        
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
        
        # Input validation
        if not input_text or not input_text.strip():
            return PipelineResult(
                success=False,
                stage_reached=0,
                total_time=0.0,
                error="Input text is empty or contains only whitespace",
                error_stage="input_validation"
            )
        
        # Initialize pipeline state
        self.pipeline_state = PipelineState(
            input_text=input_text.strip(),
            start_time=start_time
        )
        
        result = PipelineResult(
            success=False,
            stage_reached=0,
            total_time=0.0
        )
        
        try:
            # Stage 1: Entity Extraction
            self.current_stage = 0
            if progress_callback:
                progress_callback(0, "Starting entity extraction...")
            
            entity_result = self._execute_entity_stage(input_text)
            result.entity_result = entity_result
            
            if not entity_result.success:
                result.error = entity_result.error
                result.error_stage = "entity_extraction"
                return result
            
            # Check for empty entities
            if not entity_result.entities:
                result.error = "No entities were found in the input text"
                result.error_stage = "entity_extraction"
                return result
            
            result.stage_reached = 1
            self.pipeline_state.entity_result = entity_result
            
            if progress_callback:
                progress_callback(1, f"Found {len(entity_result.entities)} entities. Starting triple generation...")
            
            # Stage 2: Triple Generation
            self.current_stage = 1
            triple_result = self._execute_triple_stage(entity_result.entities, entity_result.denoised_text)
            result.triple_result = triple_result
            
            if not triple_result.success:
                result.error = triple_result.error
                result.error_stage = "triple_generation"
                return result
            
            # Check for empty triples
            if not triple_result.triples:
                result.error = "No triples were generated from the extracted entities"
                result.error_stage = "triple_generation"
                return result
            
            result.stage_reached = 2
            self.pipeline_state.triple_result = triple_result
            
            if progress_callback:
                progress_callback(2, f"Generated {len(triple_result.triples)} triples. Starting graph judgment...")
            
            # Stage 3: Graph Judgment
            self.current_stage = 2
            judgment_result = self._execute_judgment_stage(triple_result.triples)
            result.judgment_result = judgment_result
            
            if not judgment_result.success:
                result.error = judgment_result.error
                result.error_stage = "graph_judgment"
                return result
            
            result.stage_reached = 3
            self.pipeline_state.judgment_result = judgment_result
            
            if progress_callback:
                progress_callback(3, "Pipeline completed successfully!")
            
            # Pipeline completed successfully
            result.success = True
            result.total_time = time.time() - start_time
            
            # Generate summary statistics
            result.stats = self._generate_stats(result)
            
            return result
            
        except Exception as e:
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
        
        result = safe_execute(
            entity_execution,
            "Entity extraction failed",
            self.error_handler,
            stage="entity_extraction"
        )
        
        # If safe_execute returns a dict with error, convert to EntityResult
        if isinstance(result, dict) and result.get('error'):
            return EntityResult(
                entities=[],
                denoised_text=input_text,
                success=False,
                processing_time=0.0,
                error=result['error']
            )
        
        return result
    
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
        
        result = safe_execute(
            triple_execution,
            "Triple generation failed",
            self.error_handler,
            stage="triple_generation"
        )
        
        # If safe_execute returns a dict with error, convert to TripleResult
        if isinstance(result, dict) and result.get('error'):
            return TripleResult(
                triples=[],
                metadata={},
                success=False,
                processing_time=0.0,
                error=result['error']
            )
        
        return result
    
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
        
        result = safe_execute(
            judgment_execution,
            "Graph judgment failed",
            self.error_handler,
            stage="graph_judgment"
        )
        
        # If safe_execute returns a dict with error, convert to JudgmentResult
        if isinstance(result, dict) and result.get('error'):
            return JudgmentResult(
                judgments=[],
                confidence=[],
                explanations=None,
                success=False,
                processing_time=0.0,
                error=result['error']
            )
        
        return result
    
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