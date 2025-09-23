"""
Semantic similarity metrics for graph evaluation.

This module implements G-BertScore metrics adapted from
graph_evaluation/metrics/graph_matching.py for use in the
streamlit_pipeline evaluation system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

# Optional dependencies with graceful fallbacks
try:
    from bert_score import score as score_bert
    BERT_SCORE_AVAILABLE = True
    logging.info("BertScore library available for semantic similarity evaluation")
except ImportError:
    BERT_SCORE_AVAILABLE = False
    score_bert = None
    logging.warning("BertScore not available, semantic similarity will use word overlap fallback")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def get_bert_score(reference_graphs: List[List[List[str]]],
                  predicted_graphs: List[List[List[str]]],
                  model_type: str = "bert-base-uncased") -> Dict[str, float]:
    """
    Compute G-BertScore for semantic similarity between graphs.

    G-BertScore adapts BertScore for graph evaluation by treating graph edges
    as text sequences and computing semantic similarity using BERT embeddings.

    Args:
        reference_graphs: List of reference graphs, each graph is a list of triples
        predicted_graphs: List of predicted graphs, each graph is a list of triples
        model_type: BERT model type to use for embeddings

    Returns:
        Dict containing precision, recall, and F1 scores
    """
    if not reference_graphs or not predicted_graphs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(reference_graphs) != len(predicted_graphs):
        logging.warning(f"Graph count mismatch: {len(reference_graphs)} reference vs {len(predicted_graphs)} predicted")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if BERT_SCORE_AVAILABLE:
        return _compute_bert_score_actual(reference_graphs, predicted_graphs, model_type)
    else:
        logging.warning("BertScore not available, using word overlap fallback")
        return _compute_semantic_fallback(reference_graphs, predicted_graphs)


def _compute_bert_score_actual(reference_graphs: List[List[List[str]]],
                              predicted_graphs: List[List[List[str]]],
                              model_type: str) -> Dict[str, float]:
    """Compute actual BertScore using the bert_score library."""
    try:
        all_precision_scores = []
        all_recall_scores = []
        all_f1_scores = []

        for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
            if not ref_graph or not pred_graph:
                all_precision_scores.append(0.0)
                all_recall_scores.append(0.0)
                all_f1_scores.append(0.0)
                continue

            # Convert graphs to text sequences
            ref_sequences = _graph_to_text_sequences(ref_graph)
            pred_sequences = _graph_to_text_sequences(pred_graph)

            if not ref_sequences or not pred_sequences:
                all_precision_scores.append(0.0)
                all_recall_scores.append(0.0)
                all_f1_scores.append(0.0)
                continue

            # Compute BertScore for each predicted sequence against all reference sequences
            graph_precision = []
            graph_recall = []
            graph_f1 = []

            for pred_seq in pred_sequences:
                best_precision = 0.0
                best_recall = 0.0
                best_f1 = 0.0

                # Find best match against all reference sequences
                if pred_seq.strip():  # Skip empty sequences
                    try:
                        # Compute BertScore for this prediction against all references
                        P, R, F1 = score_bert([pred_seq], ref_sequences, model_type=model_type, verbose=False)

                        # Take maximum scores (best match)
                        best_precision = float(P.max())
                        best_recall = float(R.max())
                        best_f1 = float(F1.max())

                    except Exception as e:
                        logging.warning(f"BertScore computation failed for sequence: {e}")
                        best_precision = best_recall = best_f1 = 0.0

                graph_precision.append(best_precision)
                graph_recall.append(best_recall)
                graph_f1.append(best_f1)

            # Average scores for this graph
            avg_precision = sum(graph_precision) / len(graph_precision) if graph_precision else 0.0
            avg_recall = sum(graph_recall) / len(graph_recall) if graph_recall else 0.0
            avg_f1 = sum(graph_f1) / len(graph_f1) if graph_f1 else 0.0

            all_precision_scores.append(avg_precision)
            all_recall_scores.append(avg_recall)
            all_f1_scores.append(avg_f1)

        # Average across all graphs
        final_precision = sum(all_precision_scores) / len(all_precision_scores) if all_precision_scores else 0.0
        final_recall = sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
        final_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0

        return {
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1
        }

    except Exception as e:
        logging.error(f"BertScore computation failed: {e}")
        return _compute_semantic_fallback(reference_graphs, predicted_graphs)


def _compute_semantic_fallback(reference_graphs: List[List[List[str]]],
                              predicted_graphs: List[List[List[str]]]) -> Dict[str, float]:
    """Compute semantic similarity using word overlap as fallback."""
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []

    for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
        if not ref_graph or not pred_graph:
            all_precision_scores.append(0.0)
            all_recall_scores.append(0.0)
            all_f1_scores.append(0.0)
            continue

        # Convert to text and extract words
        ref_sequences = _graph_to_text_sequences(ref_graph)
        pred_sequences = _graph_to_text_sequences(pred_graph)

        ref_words = set()
        for seq in ref_sequences:
            ref_words.update(seq.lower().split())

        pred_words = set()
        for seq in pred_sequences:
            pred_words.update(seq.lower().split())

        if not pred_words or not ref_words:
            all_precision_scores.append(0.0)
            all_recall_scores.append(0.0)
            all_f1_scores.append(0.0)
            continue

        # Calculate word overlap
        intersection = ref_words.intersection(pred_words)

        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(ref_words)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        all_precision_scores.append(precision)
        all_recall_scores.append(recall)
        all_f1_scores.append(f1)

    # Average across all graphs
    final_precision = sum(all_precision_scores) / len(all_precision_scores) if all_precision_scores else 0.0
    final_recall = sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
    final_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0

    return {
        "precision": final_precision,
        "recall": final_recall,
        "f1": final_f1
    }


def _graph_to_text_sequences(graph: List[List[str]]) -> List[str]:
    """Convert graph triples to text sequences for semantic similarity computation."""
    sequences = []
    for triple in graph:
        if len(triple) >= 3:
            # Create natural language representation of the triple
            edge_text = f"{triple[0]} {triple[1]} {triple[2]}"
            sequences.append(edge_text.strip())
    return sequences


def analyze_semantic_similarity(reference_graph: List[List[str]],
                               predicted_graph: List[List[str]],
                               model_type: str = "bert-base-uncased") -> Dict[str, Any]:
    """
    Analyze semantic similarity between reference and predicted graphs.

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples
        model_type: BERT model type to use

    Returns:
        Dict containing detailed semantic similarity analysis
    """
    result = get_bert_score([reference_graph], [predicted_graph], model_type)

    # Additional analysis
    ref_sequences = _graph_to_text_sequences(reference_graph)
    pred_sequences = _graph_to_text_sequences(predicted_graph)

    analysis = {
        "bert_scores": result,
        "bert_score_available": BERT_SCORE_AVAILABLE,
        "model_type": model_type if BERT_SCORE_AVAILABLE else "word_overlap_fallback",
        "sequence_analysis": {
            "reference_sequences": len(ref_sequences),
            "predicted_sequences": len(pred_sequences),
            "avg_ref_length": sum(len(seq.split()) for seq in ref_sequences) / len(ref_sequences) if ref_sequences else 0.0,
            "avg_pred_length": sum(len(seq.split()) for seq in pred_sequences) / len(pred_sequences) if pred_sequences else 0.0
        }
    }

    # Add vocabulary analysis for fallback mode
    if not BERT_SCORE_AVAILABLE:
        ref_words = set()
        for seq in ref_sequences:
            ref_words.update(seq.lower().split())

        pred_words = set()
        for seq in pred_sequences:
            pred_words.update(seq.lower().split())

        vocab_intersection = ref_words.intersection(pred_words)

        analysis["fallback_analysis"] = {
            "reference_vocab_size": len(ref_words),
            "predicted_vocab_size": len(pred_words),
            "shared_vocab_size": len(vocab_intersection),
            "vocab_overlap_ratio": len(vocab_intersection) / len(ref_words.union(pred_words)) if ref_words.union(pred_words) else 0.0
        }

    return analysis


def check_bert_score_availability() -> Dict[str, Any]:
    """
    Check BertScore availability and configuration.

    Returns:
        Dict containing availability status and configuration info
    """
    info = {
        "bert_score_available": BERT_SCORE_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "fallback_mode": not BERT_SCORE_AVAILABLE
    }

    if BERT_SCORE_AVAILABLE:
        try:
            # Test a simple computation to verify everything works
            test_ref = ["the cat sits on the mat"]
            test_pred = ["a cat is sitting on a mat"]
            P, R, F1 = score_bert(test_pred, test_ref, model_type="bert-base-uncased", verbose=False)
            info["test_computation"] = "success"
            info["available_models"] = ["bert-base-uncased", "roberta-base", "microsoft/deberta-base"]
        except Exception as e:
            info["test_computation"] = f"failed: {str(e)}"
            info["error"] = str(e)

    if TORCH_AVAILABLE and torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

    return info