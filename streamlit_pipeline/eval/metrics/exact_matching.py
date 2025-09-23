"""
Exact matching metrics for graph evaluation.

This module implements exact triple matching and graph isomorphism metrics
adapted from graph_evaluation/metrics/graph_matching.py for use in the
streamlit_pipeline evaluation system.
"""

import logging
from typing import List, Set, Tuple
from collections import Counter


def get_triple_match_f1(reference_graphs: List[List[List[str]]],
                       predicted_graphs: List[List[List[str]]]) -> float:
    """
    Compute F1 score for exact triple matching between reference and predicted graphs.

    This metric measures how well the predicted triples match the reference triples
    at the exact string level, providing a strict accuracy assessment.

    Args:
        reference_graphs: List of reference graphs, each graph is a list of triples
        predicted_graphs: List of predicted graphs, each graph is a list of triples

    Returns:
        float: F1 score for exact triple matching (0.0 to 1.0)
    """
    if not reference_graphs or not predicted_graphs:
        return 0.0

    if len(reference_graphs) != len(predicted_graphs):
        logging.warning(f"Graph count mismatch: {len(reference_graphs)} reference vs {len(predicted_graphs)} predicted")
        return 0.0

    total_precision_scores = []
    total_recall_scores = []

    for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
        if not ref_graph or not pred_graph:
            total_precision_scores.append(0.0)
            total_recall_scores.append(0.0)
            continue

        # Convert triples to tuples for set operations
        ref_triples = set(tuple(triple) for triple in ref_graph)
        pred_triples = set(tuple(triple) for triple in pred_graph)

        # Calculate precision and recall
        if pred_triples:
            intersection = ref_triples.intersection(pred_triples)
            precision = len(intersection) / len(pred_triples)
        else:
            precision = 0.0

        if ref_triples:
            intersection = ref_triples.intersection(pred_triples)
            recall = len(intersection) / len(ref_triples)
        else:
            recall = 0.0

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

    Args:
        predicted_graphs: List of predicted graphs, each graph is a list of triples
        reference_graphs: List of reference graphs, each graph is a list of triples

    Returns:
        float: Graph matching accuracy (0.0 to 1.0)
    """
    if not reference_graphs or not predicted_graphs:
        return 0.0

    if len(reference_graphs) != len(predicted_graphs):
        logging.warning(f"Graph count mismatch: {len(reference_graphs)} reference vs {len(predicted_graphs)} predicted")
        return 0.0

    exact_matches = 0

    for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
        # Convert to sets of tuples for exact comparison
        ref_set = set(tuple(triple) for triple in ref_graph)
        pred_set = set(tuple(triple) for triple in pred_graph)

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

    # Convert triples to tuples for set operations
    ref_triples = set(tuple(triple) for triple in reference_graph)
    pred_triples = set(tuple(triple) for triple in predicted_graph)

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

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples

    Returns:
        Dict containing detailed difference analysis
    """
    ref_triples = set(tuple(triple) for triple in reference_graph)
    pred_triples = set(tuple(triple) for triple in predicted_graph)

    intersection = ref_triples.intersection(pred_triples)
    missing_triples = ref_triples - pred_triples  # In reference but not predicted
    extra_triples = pred_triples - ref_triples    # In predicted but not reference

    return {
        "total_reference": len(ref_triples),
        "total_predicted": len(pred_triples),
        "correct_matches": len(intersection),
        "missing_triples": len(missing_triples),
        "extra_triples": len(extra_triples),
        "missing_examples": list(missing_triples)[:5],
        "extra_examples": list(extra_triples)[:5],
        "precision": len(intersection) / len(pred_triples) if pred_triples else 0.0,
        "recall": len(intersection) / len(ref_triples) if ref_triples else 0.0
    }