"""
Graph Quality Evaluator for GraphJudge Streamlit Pipeline.

This module provides comprehensive graph quality assessment based on proven metrics
from graph_evaluation/metrics/eval.py. It implements multiple evaluation dimensions
including exact matching, text similarity, semantic similarity, and structural distance.

The evaluator is designed for real-time use during pipeline execution and batch
evaluation for research analysis.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from ..core.models import Triple, GraphMetrics, EvaluationResult
    from ..utils.error_handling import ErrorHandler, ErrorType, safe_execute
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import Triple, GraphMetrics, EvaluationResult
    from utils.error_handling import ErrorHandler, ErrorType, safe_execute


class GraphEvaluator:
    """
    Comprehensive graph quality evaluator implementing multiple assessment metrics.

    This class provides a unified interface for evaluating knowledge graphs using
    various complementary metrics adapted from graph_evaluation/metrics/eval.py.

    Features:
    - Multiple evaluation dimensions (exact, text, semantic, structural)
    - Real-time and batch evaluation modes
    - Comprehensive error handling and reporting
    - Performance optimization for streamlit integration
    """

    def __init__(self,
                 enable_ged: bool = False,
                 enable_bert_score: bool = True,
                 max_evaluation_time: float = 30.0):
        """
        Initialize the graph evaluator with configuration options.

        Args:
            enable_ged: Whether to compute Graph Edit Distance (expensive)
            enable_bert_score: Whether to compute G-BertScore (requires transformers)
            max_evaluation_time: Maximum time allowed for evaluation in seconds
        """
        self.enable_ged = enable_ged
        self.enable_bert_score = enable_bert_score
        self.max_evaluation_time = max_evaluation_time
        self.error_handler = ErrorHandler()

        # Initialize metric modules (lazy loading for performance)
        self._exact_matching = None
        self._text_similarity = None
        self._semantic_similarity = None
        self._structural_distance = None

        # Track evaluation statistics
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0

        logging.info(f"GraphEvaluator initialized with GED={enable_ged}, BertScore={enable_bert_score}")

    def evaluate_graph(self,
                      predicted_graph: List[Triple],
                      reference_graph: List[Triple],
                      enable_ged_override: Optional[bool] = None) -> EvaluationResult:
        """
        Evaluate a predicted graph against a reference graph using multiple metrics.

        This is the main evaluation interface providing comprehensive graph quality
        assessment across multiple dimensions.

        Args:
            predicted_graph: List of Triple objects representing the predicted graph
            reference_graph: List of Triple objects representing the reference graph
            enable_ged_override: Override default GED setting for this evaluation

        Returns:
            EvaluationResult containing comprehensive metrics and metadata
        """
        start_time = time.time()

        # Validate inputs
        validation_result = self._validate_inputs(predicted_graph, reference_graph)
        if not validation_result[0]:
            return EvaluationResult(
                metrics=self._get_empty_metrics(),
                metadata={"error_type": "validation_error"},
                success=False,
                processing_time=time.time() - start_time,
                error=validation_result[1]
            )

        try:
            # Convert graphs to evaluation format
            pred_graph_data = self._convert_graph_format(predicted_graph)
            ref_graph_data = self._convert_graph_format(reference_graph)

            # Compute all evaluation metrics
            metrics_result = self._compute_all_metrics(
                pred_graph_data,
                ref_graph_data,
                enable_ged_override or self.enable_ged
            )

            if not metrics_result[0]:
                return EvaluationResult(
                    metrics=self._get_empty_metrics(),
                    metadata={"error_type": "computation_error"},
                    success=False,
                    processing_time=time.time() - start_time,
                    error=metrics_result[1]
                )

            metrics = metrics_result[1]
            processing_time = time.time() - start_time

            # Update statistics
            self.evaluation_count += 1
            self.total_evaluation_time += processing_time

            # Create comprehensive metadata
            metadata = {
                "predicted_graph_size": len(predicted_graph),
                "reference_graph_size": len(reference_graph),
                "evaluation_settings": {
                    "ged_enabled": enable_ged_override or self.enable_ged,
                    "bert_score_enabled": self.enable_bert_score
                },
                "performance": {
                    "processing_time": processing_time,
                    "evaluation_count": self.evaluation_count,
                    "average_time": self.total_evaluation_time / self.evaluation_count
                }
            }

            # Generate graph statistics
            pred_info = self._get_graph_info(predicted_graph)
            ref_info = self._get_graph_info(reference_graph)

            return EvaluationResult(
                metrics=metrics,
                metadata=metadata,
                success=True,
                processing_time=processing_time,
                predicted_graph_info=pred_info,
                reference_graph_info=ref_info
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Evaluation failed: {str(e)}"
            logging.error(f"Graph evaluation error: {error_msg}")

            return EvaluationResult(
                metrics=self._get_empty_metrics(),
                metadata={"error_type": "computation_error", "exception": str(e)},
                success=False,
                processing_time=processing_time,
                error=error_msg
            )

    def evaluate_batch(self,
                      graph_pairs: List[Tuple[List[Triple], List[Triple]]],
                      show_progress: bool = True) -> List[EvaluationResult]:
        """
        Evaluate multiple graph pairs in batch mode.

        Args:
            graph_pairs: List of (predicted_graph, reference_graph) tuples
            show_progress: Whether to show progress bar

        Returns:
            List of EvaluationResult objects
        """
        results = []

        iterator = graph_pairs
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(graph_pairs, desc="Evaluating graphs")
            except ImportError:
                # Fallback to simple enumeration
                iterator = graph_pairs

        for pred_graph, ref_graph in iterator:
            result = self.evaluate_graph(pred_graph, ref_graph)
            results.append(result)

        return results

    def _validate_inputs(self, predicted_graph: List[Triple], reference_graph: List[Triple]) -> Tuple[bool, Optional[str]]:
        """Validate input graphs for evaluation."""
        if not isinstance(predicted_graph, list):
            return False, "Predicted graph must be a list of Triple objects"

        if not isinstance(reference_graph, list):
            return False, "Reference graph must be a list of Triple objects"

        if len(predicted_graph) == 0:
            return False, "Predicted graph cannot be empty"

        if len(reference_graph) == 0:
            return False, "Reference graph cannot be empty"

        # Validate Triple objects
        for i, triple in enumerate(predicted_graph):
            if not isinstance(triple, Triple):
                return False, f"Predicted graph item {i} is not a Triple object"

        for i, triple in enumerate(reference_graph):
            if not isinstance(triple, Triple):
                return False, f"Reference graph item {i} is not a Triple object"

        return True, None

    def _convert_graph_format(self, graph: List[Triple]) -> List[List[str]]:
        """Convert Triple objects to the format expected by evaluation metrics."""
        return [[t.subject, t.predicate, t.object] for t in graph]

    def _compute_all_metrics(self,
                           pred_graph: List[List[str]],
                           ref_graph: List[List[str]],
                           enable_ged: bool) -> Tuple[bool, Optional[GraphMetrics]]:
        """Compute all evaluation metrics for the given graphs."""
        try:
            # Import metric modules (lazy loading)
            if self._exact_matching is None:
                from .metrics.exact_matching import get_triple_match_f1, get_graph_match_accuracy
                self._exact_matching = (get_triple_match_f1, get_graph_match_accuracy)

            if self._text_similarity is None:
                from .metrics.text_similarity import get_bleu_rouge_scores
                self._text_similarity = get_bleu_rouge_scores

            if self.enable_bert_score and self._semantic_similarity is None:
                from .metrics.semantic_similarity import get_bert_score
                self._semantic_similarity = get_bert_score

            if enable_ged and self._structural_distance is None:
                from .metrics.structural_distance import get_graph_edit_distance
                self._structural_distance = get_graph_edit_distance

            # Compute exact matching metrics
            triple_f1 = self._exact_matching[0]([ref_graph], [pred_graph])
            graph_accuracy = self._exact_matching[1]([pred_graph], [ref_graph])

            # Compute text similarity metrics
            text_scores = self._text_similarity([ref_graph], [pred_graph])

            # Compute semantic similarity metrics
            bert_scores = None
            if self.enable_bert_score and self._semantic_similarity:
                bert_scores = self._semantic_similarity([ref_graph], [pred_graph])

            # Compute structural distance metrics
            ged_score = None
            if enable_ged and self._structural_distance:
                ged_score = self._structural_distance(ref_graph, pred_graph)

            # Create GraphMetrics object
            metrics = GraphMetrics(
                triple_match_f1=float(triple_f1),
                graph_match_accuracy=float(graph_accuracy),
                g_bleu_precision=float(text_scores['bleu']['precision']),
                g_bleu_recall=float(text_scores['bleu']['recall']),
                g_bleu_f1=float(text_scores['bleu']['f1']),
                g_rouge_precision=float(text_scores['rouge']['precision']),
                g_rouge_recall=float(text_scores['rouge']['recall']),
                g_rouge_f1=float(text_scores['rouge']['f1']),
                g_bert_precision=float(bert_scores['precision']) if bert_scores else 0.0,
                g_bert_recall=float(bert_scores['recall']) if bert_scores else 0.0,
                g_bert_f1=float(bert_scores['f1']) if bert_scores else 0.0,
                graph_edit_distance=float(ged_score) if ged_score is not None else None
            )

            return True, metrics

        except Exception as e:
            logging.error(f"Metric computation failed: {str(e)}")
            return False, f"Failed to compute metrics: {str(e)}"

    def _get_empty_metrics(self) -> GraphMetrics:
        """Get empty metrics object for error cases."""
        return GraphMetrics(
            triple_match_f1=0.0,
            graph_match_accuracy=0.0,
            g_bleu_precision=0.0,
            g_bleu_recall=0.0,
            g_bleu_f1=0.0,
            g_rouge_precision=0.0,
            g_rouge_recall=0.0,
            g_rouge_f1=0.0,
            g_bert_precision=0.0,
            g_bert_recall=0.0,
            g_bert_f1=0.0,
            graph_edit_distance=None
        )

    def _get_graph_info(self, graph: List[Triple]) -> Dict[str, Any]:
        """Generate statistical information about a graph."""
        if not graph:
            return {"size": 0, "unique_subjects": 0, "unique_predicates": 0, "unique_objects": 0}

        subjects = set(t.subject for t in graph)
        predicates = set(t.predicate for t in graph)
        objects = set(t.object for t in graph)

        return {
            "size": len(graph),
            "unique_subjects": len(subjects),
            "unique_predicates": len(predicates),
            "unique_objects": len(objects),
            "subject_examples": list(subjects)[:5],
            "predicate_examples": list(predicates)[:5],
            "object_examples": list(objects)[:5]
        }

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics for monitoring and debugging."""
        return {
            "evaluation_count": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": self.total_evaluation_time / max(1, self.evaluation_count),
            "settings": {
                "ged_enabled": self.enable_ged,
                "bert_score_enabled": self.enable_bert_score,
                "max_evaluation_time": self.max_evaluation_time
            }
        }


# Convenience functions for direct usage
def evaluate_graph(predicted_graph: List[Triple],
                  reference_graph: List[Triple],
                  enable_ged: bool = False) -> EvaluationResult:
    """
    Convenience function for single graph evaluation.

    Args:
        predicted_graph: List of Triple objects representing the predicted graph
        reference_graph: List of Triple objects representing the reference graph
        enable_ged: Whether to compute Graph Edit Distance (expensive)

    Returns:
        EvaluationResult containing comprehensive metrics and metadata
    """
    evaluator = GraphEvaluator(enable_ged=enable_ged)
    return evaluator.evaluate_graph(predicted_graph, reference_graph)


def evaluate_batch(graph_pairs: List[Tuple[List[Triple], List[Triple]]],
                  enable_ged: bool = False,
                  show_progress: bool = True) -> List[EvaluationResult]:
    """
    Convenience function for batch graph evaluation.

    Args:
        graph_pairs: List of (predicted_graph, reference_graph) tuples
        enable_ged: Whether to compute Graph Edit Distance (expensive)
        show_progress: Whether to show progress bar

    Returns:
        List of EvaluationResult objects
    """
    evaluator = GraphEvaluator(enable_ged=enable_ged)
    return evaluator.evaluate_batch(graph_pairs, show_progress)