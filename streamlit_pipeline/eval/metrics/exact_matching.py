"""
Exact matching metrics for graph evaluation.

This module implements exact triple matching and graph isomorphism metrics
adapted from graph_evaluation/metrics/graph_matching.py for use in the
streamlit_pipeline evaluation system.
"""

import logging
import sys
import os
from typing import List, Set, Tuple
from collections import Counter

# Ensure proper Unicode handling for Windows console
if os.name == 'nt':  # Windows
    try:
        # Set console to UTF-8 mode if possible
        os.system('chcp 65001 > nul 2>&1')
    except:
        pass


def _validate_triple_structure(triple: List[str]) -> bool:
    """Validate that a triple has the expected structure [subject, predicate, object]."""
    return isinstance(triple, list) and len(triple) >= 3 and all(isinstance(elem, str) for elem in triple[:3])


def _safe_convert_to_tuple(triple: List[str]) -> Tuple[str, str, str]:
    """Safely convert a triple to tuple, handling malformed triples and Unicode."""
    if not _validate_triple_structure(triple):
        try:
            # Safe logging that handles Unicode properly
            triple_repr = repr(triple)[:100]  # Limit length and use repr to avoid Unicode issues
            logging.warning(f"Malformed triple encountered: {triple_repr}, using placeholder values")
        except Exception:
            logging.warning("Malformed triple encountered (display error), using placeholder values")

        # Use placeholder values for malformed triples to avoid crashes
        subject = str(triple[0]) if len(triple) > 0 and triple[0] is not None else "INVALID_SUBJECT"
        predicate = str(triple[1]) if len(triple) > 1 and triple[1] is not None else "INVALID_PREDICATE"
        object_val = str(triple[2]) if len(triple) > 2 and triple[2] is not None else "INVALID_OBJECT"
        return (subject, predicate, object_val)

    # Ensure all elements are properly handled as strings
    try:
        return tuple(str(elem) for elem in triple[:3])
    except Exception as e:
        logging.warning(f"Unicode conversion error in triple: {e}")
        return ("INVALID_SUBJECT", "INVALID_PREDICATE", "INVALID_OBJECT")


def get_triple_match_f1(reference_graphs: List[List[List[str]]],
                       predicted_graphs: List[List[List[str]]]) -> float:
    """
    Compute F1 score for exact triple matching between reference and predicted graphs.

    This metric measures how well the predicted triples match the reference triples
    at the exact string level, providing a strict accuracy assessment.

    Now handles unequal graph counts by evaluating available pairs and providing
    meaningful precision/recall/F1 scores regardless of triple count differences.
    This aligns with standard information retrieval evaluation practices.

    Args:
        reference_graphs: List of reference graphs, each graph is a list of triples
        predicted_graphs: List of predicted graphs, each graph is a list of triples

    Returns:
        float: F1 score for exact triple matching (0.0 to 1.0)
    """
    if not reference_graphs or not predicted_graphs:
        return 0.0

    if len(reference_graphs) != len(predicted_graphs):
        logging.warning(f"Graph count mismatch: {len(reference_graphs)} reference vs {len(predicted_graphs)} predicted - continuing with available pairs")
        # Use minimum count to evaluate available pairs rather than failing
        min_count = min(len(reference_graphs), len(predicted_graphs))
        reference_graphs = reference_graphs[:min_count]
        predicted_graphs = predicted_graphs[:min_count]

        if min_count == 0:
            return 0.0

    total_precision_scores = []
    total_recall_scores = []

    for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
        if not ref_graph or not pred_graph:
            total_precision_scores.append(0.0)
            total_recall_scores.append(0.0)
            continue

        # Convert triples to tuples for set operations with validation
        ref_triples = set(_safe_convert_to_tuple(triple) for triple in ref_graph)
        pred_triples = set(_safe_convert_to_tuple(triple) for triple in pred_graph)

        # Calculate intersection once for efficiency
        intersection = ref_triples.intersection(pred_triples)

        # Calculate precision and recall
        precision = len(intersection) / len(pred_triples) if pred_triples else 0.0
        recall = len(intersection) / len(ref_triples) if ref_triples else 0.0

        total_precision_scores.append(precision)
        total_recall_scores.append(recall)

    # Calculate average precision and recall
    avg_precision = sum(total_precision_scores) / len(total_precision_scores) if total_precision_scores else 0.0
    avg_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0.0

    # Calculate F1 score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0

    return f1_score


def get_graph_match_accuracy(predicted_graphs: List[List[List[str]]],
                           reference_graphs: List[List[List[str]]]) -> float:
    """
    Compute graph-level matching accuracy using exact set comparison.

    This metric measures how many graphs match exactly at the set level,
    providing a strict graph-level accuracy assessment.

    Now handles unequal graph counts by evaluating available pairs rather than
    failing completely when graph counts differ.

    Args:
        predicted_graphs: List of predicted graphs, each graph is a list of triples
        reference_graphs: List of reference graphs, each graph is a list of triples

    Returns:
        float: Graph matching accuracy (0.0 to 1.0)
    """
    if not reference_graphs or not predicted_graphs:
        return 0.0

    if len(reference_graphs) != len(predicted_graphs):
        logging.warning(f"Graph count mismatch: {len(reference_graphs)} reference vs {len(predicted_graphs)} predicted - continuing with available pairs")
        # Use minimum count to evaluate available pairs rather than failing
        min_count = min(len(reference_graphs), len(predicted_graphs))
        reference_graphs = reference_graphs[:min_count]
        predicted_graphs = predicted_graphs[:min_count]

        if min_count == 0:
            return 0.0

    exact_matches = 0

    for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
        # Convert to sets of tuples for exact comparison with validation
        ref_set = set(_safe_convert_to_tuple(triple) for triple in ref_graph)
        pred_set = set(_safe_convert_to_tuple(triple) for triple in pred_graph)

        if ref_set == pred_set:
            exact_matches += 1

    accuracy = exact_matches / len(reference_graphs) if reference_graphs else 0.0
    return accuracy


def get_triple_precision_recall(reference_graph: List[List[str]],
                               predicted_graph: List[List[str]]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for a single graph pair.

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples

    Returns:
        Tuple of (precision, recall, f1) scores
    """
    if not reference_graph or not predicted_graph:
        return 0.0, 0.0, 0.0

    # Convert triples to tuples for set operations with validation
    ref_triples = set(_safe_convert_to_tuple(triple) for triple in reference_graph)
    pred_triples = set(_safe_convert_to_tuple(triple) for triple in predicted_graph)

    # Calculate intersection
    intersection = ref_triples.intersection(pred_triples)

    # Calculate precision
    precision = len(intersection) / len(pred_triples) if pred_triples else 0.0

    # Calculate recall
    recall = len(intersection) / len(ref_triples) if ref_triples else 0.0

    # Calculate F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def analyze_triple_differences(reference_graph: List[List[str]],
                             predicted_graph: List[List[str]]) -> dict:
    """
    Analyze differences between reference and predicted graphs.

    Provides comprehensive diagnostics including coverage, precision, recall,
    and generation behavior analysis regardless of graph size differences.

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples

    Returns:
        Dict containing detailed difference analysis
    """
    ref_triples = set(_safe_convert_to_tuple(triple) for triple in reference_graph)
    pred_triples = set(_safe_convert_to_tuple(triple) for triple in predicted_graph)

    intersection = ref_triples.intersection(pred_triples)
    missing_triples = ref_triples - pred_triples  # In reference but not predicted
    extra_triples = pred_triples - ref_triples    # In predicted but not reference

    # Calculate metrics
    precision = len(intersection) / len(pred_triples) if pred_triples else 0.0
    recall = len(intersection) / len(ref_triples) if ref_triples else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate diagnostic ratios
    coverage = recall  # Same as recall - what % of reference was found
    over_generation_ratio = len(extra_triples) / len(ref_triples) if ref_triples else 0.0
    under_generation_ratio = len(missing_triples) / len(ref_triples) if ref_triples else 0.0

    # Size comparison
    size_ratio = len(pred_triples) / len(ref_triples) if ref_triples else float('inf') if pred_triples else 1.0

    # Determine generation behavior
    if size_ratio > 1.2:
        generation_behavior = "over_generated"
    elif size_ratio < 0.8:
        generation_behavior = "under_generated"
    else:
        generation_behavior = "appropriately_sized"

    return {
        "total_reference": len(ref_triples),
        "total_predicted": len(pred_triples),
        "correct_matches": len(intersection),
        "missing_triples": len(missing_triples),
        "extra_triples": len(extra_triples),
        "missing_examples": list(missing_triples)[:5],
        "extra_examples": list(extra_triples)[:5],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
        "over_generation_ratio": over_generation_ratio,
        "under_generation_ratio": under_generation_ratio,
        "size_ratio": size_ratio,
        "generation_behavior": generation_behavior
    }


def get_evaluation_diagnostics(reference_graphs: List[List[List[str]]],
                              predicted_graphs: List[List[List[str]]]) -> dict:
    """
    Get comprehensive evaluation diagnostics for graph comparison.

    Provides insights into evaluation behavior when graph counts differ,
    helping understand generation patterns and evaluation coverage.

    Args:
        reference_graphs: List of reference graphs
        predicted_graphs: List of predicted graphs

    Returns:
        Dict containing evaluation diagnostics and metadata
    """
    diagnostics = {
        "graph_counts": {
            "reference": len(reference_graphs),
            "predicted": len(predicted_graphs),
            "evaluated_pairs": min(len(reference_graphs), len(predicted_graphs))
        },
        "count_mismatch": len(reference_graphs) != len(predicted_graphs),
        "coverage_stats": {
            "reference_coverage": min(len(predicted_graphs), len(reference_graphs)) / len(reference_graphs) if reference_graphs else 0.0,
            "prediction_coverage": min(len(predicted_graphs), len(reference_graphs)) / len(predicted_graphs) if predicted_graphs else 0.0
        }
    }

    if diagnostics["count_mismatch"]:
        if len(predicted_graphs) > len(reference_graphs):
            diagnostics["mismatch_type"] = "over_generation"
            diagnostics["excess_predictions"] = len(predicted_graphs) - len(reference_graphs)
        else:
            diagnostics["mismatch_type"] = "under_generation"
            diagnostics["missing_predictions"] = len(reference_graphs) - len(predicted_graphs)
    else:
        diagnostics["mismatch_type"] = "equal_counts"

    return diagnostics