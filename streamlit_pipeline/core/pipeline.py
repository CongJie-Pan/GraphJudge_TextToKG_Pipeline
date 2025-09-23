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
    from .graph_judge import judge_triples, judge_triples_with_explanations
    from .models import EntityResult, TripleResult, JudgmentResult, Triple, PipelineState, EvaluationResult
    from .graph_converter import (
        create_graph_from_judgment_result,
        create_graph_from_triples,
        create_pyvis_graph_from_judgment_result,
        create_kgshows_graph_from_judgment_result,
        validate_graph_data,
        get_graph_statistics
    )
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
    from core.models import EntityResult, TripleResult, JudgmentResult, Triple, PipelineState, EvaluationResult
    from core.graph_converter import (
        create_graph_from_judgment_result,
        create_graph_from_triples,
        create_pyvis_graph_from_judgment_result,
        create_kgshows_graph_from_judgment_result,
        validate_graph_data,
        get_graph_statistics
    )
    from utils.error_handling import ErrorHandler, ErrorType, safe_execute
    from utils.api_client import get_api_client
    from utils.storage_manager import get_storage_manager, create_new_pipeline_iteration, save_phase_result
    from utils.detailed_logger import DetailedLogger


@dataclass
class PipelineResult:
    """Complete pipeline result containing all stage outputs."""
    success: bool
    stage_reached: int  # 0=entity, 1=triple, 2=judgment, 3=evaluation, 4=complete
    total_time: float

    # Stage results
    entity_result: Optional[EntityResult] = None
    triple_result: Optional[TripleResult] = None
    judgment_result: Optional[JudgmentResult] = None

    # Graph data in multiple formats
    graph_data: Optional[Dict[str, Any]] = None  # Plotly format (backward compatibility)
    pyvis_data: Optional[Dict[str, Any]] = None  # Pyvis format (primary viewer)
    kgshows_data: Optional[Dict[str, Any]] = None  # kgGenShows format (for other projects)

    # Evaluation results (optional)
    evaluation_result: Optional['EvaluationResult'] = None  # Forward reference since EvaluationResult is in models
    evaluation_enabled: bool = False
    reference_graph_info: Optional[Dict[str, Any]] = None

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
        
    def run_pipeline(self, input_text: str, progress_callback=None, config_options=None,
                     evaluation_config=None, reference_graph=None) -> PipelineResult:
        """
        Execute the complete GraphJudge pipeline with optional evaluation.

        Args:
            input_text: The raw Chinese text to process
            progress_callback: Optional callback function for progress updates
            config_options: Optional configuration options from UI
            evaluation_config: Optional evaluation configuration (enable_evaluation, enable_ged, etc.)
            reference_graph: Optional reference graph for evaluation (list of Triple objects)

        Returns:
            PipelineResult containing all stage outputs and metadata, including optional evaluation
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

            judgment_result = self._execute_judgment_stage(triple_result.triples, config_options)
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

            # Stage 3.5: Optional Graph Quality Evaluation
            # Check if evaluation is requested and reference graph is available
            evaluation_enabled = (evaluation_config and
                                evaluation_config.get('enable_evaluation', False) and
                                reference_graph is not None)

            if evaluation_enabled:
                self.detailed_logger.log_info("PIPELINE", "Starting Optional Stage: Graph Quality Evaluation", {
                    "stage": "evaluation",
                    "reference_graph_size": len(reference_graph),
                    "predicted_graph_size": len([t for t, approved in zip(triple_result.triples, judgment_result.judgments) if approved])
                })

                if progress_callback:
                    progress_callback(3, "Running graph quality evaluation...")

                evaluation_result = self._execute_evaluation_stage(
                    triple_result.triples,
                    judgment_result.judgments,
                    reference_graph,
                    evaluation_config
                )

                result.evaluation_result = evaluation_result
                result.evaluation_enabled = True
                result.reference_graph_info = {
                    "size": len(reference_graph),
                    "format": "Triple objects"
                }

                if evaluation_result and evaluation_result.success:
                    self.detailed_logger.log_info("EVALUATION", "Graph evaluation completed successfully", {
                        "overall_score": evaluation_result.metrics.get_overall_score(),
                        "processing_time": evaluation_result.processing_time
                    })
                else:
                    self.detailed_logger.log_warning("EVALUATION", "Graph evaluation failed or incomplete", {
                        "error": evaluation_result.error if evaluation_result else "Unknown error"
                    })
            else:
                self.detailed_logger.log_info("PIPELINE", "Graph evaluation skipped", {
                    "evaluation_config_provided": evaluation_config is not None,
                    "reference_graph_provided": reference_graph is not None,
                    "evaluation_enabled": evaluation_config.get('enable_evaluation', False) if evaluation_config else False
                })

            # Stage 4: Graph Conversion for Visualization
            self.detailed_logger.log_info("PIPELINE", "Starting Stage 4: Graph Conversion", {
                "stage": "graph_conversion",
                "approved_triples": approved_count
            })

            try:
                # Convert judgment results to multiple graph formats for different viewers
                self.detailed_logger.log_info("GRAPH", "Generating multiple graph formats...")

                # 1. Plotly format (backward compatibility)
                graph_data = create_graph_from_judgment_result(triple_result, judgment_result)

                # 2. Pyvis format (primary interactive viewer)
                pyvis_data = create_pyvis_graph_from_judgment_result(triple_result, judgment_result)

                # 3. kgGenShows format (for other projects)
                kgshows_data = create_kgshows_graph_from_judgment_result(triple_result, judgment_result)

                # Validate the generated graph data
                is_valid, validation_errors = validate_graph_data(graph_data)
                if not is_valid:
                    self.detailed_logger.log_error("GRAPH", "Graph data validation failed", {
                        "validation_errors": validation_errors[:3],  # First 3 errors
                        "total_errors": len(validation_errors)
                    })
                    # Continue with invalid data but log the issues
                else:
                    self.detailed_logger.log_info("GRAPH", "Graph data validation passed")

                # Generate graph statistics
                graph_stats = get_graph_statistics(graph_data)
                self.detailed_logger.log_info("GRAPH", "Multi-format graph conversion completed successfully", {
                    "plotly_nodes": graph_stats["nodes_count"],
                    "plotly_edges": graph_stats["edges_count"],
                    "pyvis_nodes": pyvis_data["metadata"]["nodes_count"],
                    "pyvis_edges": pyvis_data["metadata"]["edges_count"],
                    "kgshows_entities": len(kgshows_data["entities"]),
                    "kgshows_relationships": len(kgshows_data["relationships"]),
                    "validation_status": graph_stats["validation_status"]
                })

                # Store all formats in result
                result.graph_data = graph_data      # Plotly format
                result.pyvis_data = pyvis_data      # Pyvis format
                result.kgshows_data = kgshows_data  # kgGenShows format

                # Save all graph formats to storage if possible
                try:
                    # Save Plotly format
                    saved_graph_path = self.storage_manager.save_graph_data(graph_data)

                    # Save Pyvis format
                    saved_pyvis_path = self.storage_manager.save_pyvis_data(pyvis_data)

                    # Save kgGenShows format
                    saved_kgshows_path = self.storage_manager.save_kgshows_data(kgshows_data)

                    self.detailed_logger.log_info("STORAGE", "All graph formats saved", {
                        "plotly_path": saved_graph_path,
                        "pyvis_path": saved_pyvis_path,
                        "kgshows_path": saved_kgshows_path
                    })
                except Exception as e:
                    self.detailed_logger.log_error("STORAGE", f"Failed to save graph data: {e}")
                    # Don't fail the pipeline for graph storage issues

            except Exception as e:
                # Graph conversion failure shouldn't fail the entire pipeline
                self.detailed_logger.log_error("GRAPH", f"Graph conversion failed: {e}", {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                # Create empty graph data structure
                result.graph_data = {
                    "nodes": [],
                    "edges": [],
                    "entities": [],
                    "relationships": [],
                    "metadata": {
                        "conversion_error": str(e),
                        "source_type": "streamlit_pipeline"
                    }
                }

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
                "stats": result.stats,
                "graph_data_available": result.graph_data is not None and len(result.graph_data.get("nodes", [])) > 0
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
    
    def _execute_judgment_stage(self, triples: List[Triple], config_options=None) -> JudgmentResult:
        """
        Execute the graph judgment stage.

        Args:
            triples: List of triples to judge
            config_options: Optional configuration options from UI

        Returns:
            JudgmentResult containing judgment outcomes
        """
        # Check if explanations are enabled in config
        enable_explanations = False
        if config_options and isinstance(config_options, dict):
            enable_explanations = config_options.get('enable_explanations', False)

        def judgment_execution():
            if enable_explanations:
                # Get detailed explanations and convert to JudgmentResult
                dict_result = judge_triples_with_explanations(triples, include_reasoning=True)
                return JudgmentResult(
                    judgments=dict_result["judgments"],
                    confidence=dict_result["confidence"],
                    explanations=dict_result["explanations"],
                    success=dict_result["success"],
                    processing_time=dict_result["processing_time"],
                    error=dict_result.get("error")
                )
            else:
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

    def _execute_evaluation_stage(self, triples: List[Triple], judgments: List[bool],
                                reference_graph: List[Triple], evaluation_config: Dict[str, Any]) -> Optional[EvaluationResult]:
        """
        Execute the optional graph quality evaluation stage.

        Args:
            triples: List of generated triples
            judgments: List of judgment results (True/False for each triple)
            reference_graph: Reference graph for evaluation
            evaluation_config: Evaluation configuration options

        Returns:
            EvaluationResult containing evaluation metrics or None if failed
        """
        try:
            # Import evaluation modules with graceful fallback
            try:
                from ..eval.graph_evaluator import GraphEvaluator
            except ImportError:
                # Fallback import for direct execution
                import sys
                import os
                eval_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval')
                sys.path.insert(0, eval_path)
                from graph_evaluator import GraphEvaluator

            # Create predicted graph from approved triples only
            predicted_graph = [triple for triple, approved in zip(triples, judgments) if approved]

            if not predicted_graph:
                self.detailed_logger.log_warning("EVALUATION", "No approved triples for evaluation")
                return None

            # Configure evaluator based on evaluation config
            enable_ged = evaluation_config.get('enable_ged', False)
            enable_bert_score = evaluation_config.get('enable_bert_score', True)
            max_evaluation_time = evaluation_config.get('max_evaluation_time', 30.0)

            evaluator = GraphEvaluator(
                enable_ged=enable_ged,
                enable_bert_score=enable_bert_score,
                max_evaluation_time=max_evaluation_time
            )

            # Run evaluation
            evaluation_result = evaluator.evaluate_graph(predicted_graph, reference_graph)

            self.detailed_logger.log_info("EVALUATION", "Evaluation completed", {
                "success": evaluation_result.success,
                "predicted_triples": len(predicted_graph),
                "reference_triples": len(reference_graph),
                "processing_time": evaluation_result.processing_time
            })

            return evaluation_result

        except Exception as e:
            error_msg = f"Evaluation stage failed: {str(e)}"
            self.detailed_logger.log_error("EVALUATION", error_msg, {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

            # Return failed evaluation result instead of None for better error tracking
            from ..core.models import GraphMetrics
            empty_metrics = GraphMetrics(
                triple_match_f1=0.0, graph_match_accuracy=0.0,
                g_bleu_precision=0.0, g_bleu_recall=0.0, g_bleu_f1=0.0,
                g_rouge_precision=0.0, g_rouge_recall=0.0, g_rouge_f1=0.0,
                g_bert_precision=0.0, g_bert_recall=0.0, g_bert_f1=0.0
            )

            return EvaluationResult(
                metrics=empty_metrics,
                metadata={"error_type": "evaluation_stage_error"},
                success=False,
                processing_time=0.0,
                error=error_msg
            )

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

        # Add evaluation statistics if available
        if result.evaluation_result and result.evaluation_enabled:
            stats.update({
                'evaluation_enabled': True,
                'evaluation_success': result.evaluation_result.success,
                'evaluation_time': result.evaluation_result.processing_time,
                'reference_graph_size': result.reference_graph_info.get('size', 0) if result.reference_graph_info else 0
            })

            if result.evaluation_result.success:
                stats.update({
                    'overall_evaluation_score': result.evaluation_result.metrics.get_overall_score(),
                    'triple_match_f1': result.evaluation_result.metrics.triple_match_f1,
                    'graph_match_accuracy': result.evaluation_result.metrics.graph_match_accuracy,
                    'g_bert_f1': result.evaluation_result.metrics.g_bert_f1
                })
        else:
            stats['evaluation_enabled'] = False

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


def run_full_pipeline(input_text: str, progress_callback=None, evaluation_config=None, reference_graph=None) -> PipelineResult:
    """
    Convenience function to run the complete pipeline with optional evaluation.

    Args:
        input_text: The raw Chinese text to process
        progress_callback: Optional progress callback function
        evaluation_config: Optional evaluation configuration
        reference_graph: Optional reference graph for evaluation

    Returns:
        PipelineResult with all stage outputs and optional evaluation
    """
    orchestrator = PipelineOrchestrator()
    return orchestrator.run_pipeline(input_text, progress_callback, evaluation_config=evaluation_config, reference_graph=reference_graph)