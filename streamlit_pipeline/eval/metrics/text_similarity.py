"""
Text similarity metrics for graph evaluation.

This module implements G-BLEU and G-ROUGE metrics adapted from
graph_evaluation/metrics/graph_matching.py for use in the
streamlit_pipeline evaluation system.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from collections import Counter


def _validate_triple_structure(triple: List[str]) -> bool:
    """Validate that a triple has the expected structure [subject, predicate, object]."""
    return isinstance(triple, list) and len(triple) >= 3 and all(isinstance(elem, str) for elem in triple[:3])


def _safe_triple_to_text(triple: List[str]) -> str:
    """Safely convert a triple to text sequence, handling malformed triples and Unicode."""
    if not _validate_triple_structure(triple):
        try:
            triple_repr = repr(triple)[:100]
            logging.warning(f"Malformed triple encountered in text conversion: {triple_repr}")
        except Exception:
            logging.warning("Malformed triple encountered in text conversion (display error)")

        # Use available elements or placeholders
        subject = str(triple[0]) if len(triple) > 0 and triple[0] is not None else "INVALID_SUBJECT"
        predicate = str(triple[1]) if len(triple) > 1 and triple[1] is not None else "INVALID_PREDICATE"
        object_val = str(triple[2]) if len(triple) > 2 and triple[2] is not None else "INVALID_OBJECT"

        try:
            return f"{subject} {predicate} {object_val}".lower().strip()
        except Exception as e:
            logging.warning(f"Unicode error in text conversion: {e}")
            return "invalid_subject invalid_predicate invalid_object"

    try:
        # Ensure proper Unicode handling
        subject = str(triple[0]).strip()
        predicate = str(triple[1]).strip()
        object_val = str(triple[2]).strip()
        return f"{subject} {predicate} {object_val}".lower().strip()
    except Exception as e:
        logging.warning(f"Unicode error in triple text conversion: {e}")
        return "invalid_subject invalid_predicate invalid_object"

# Optional dependencies with graceful fallbacks
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using simplified BLEU implementation")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge_score not available, using simplified ROUGE implementation")


def get_bleu_rouge_scores(reference_graphs: List[List[List[str]]],
                         predicted_graphs: List[List[List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Compute G-BLEU and G-ROUGE scores for graph edge text similarity.

    These metrics adapt traditional text similarity measures for graph evaluation
    by treating graph edges as text sequences for comparison.

    Now handles unequal graph counts by evaluating available pairs and providing
    meaningful BLEU/ROUGE scores regardless of triple count differences.

    Args:
        reference_graphs: List of reference graphs, each graph is a list of triples
        predicted_graphs: List of predicted graphs, each graph is a list of triples

    Returns:
        Dict containing BLEU and ROUGE scores with precision, recall, and F1
    """
    if not reference_graphs or not predicted_graphs:
        return _get_empty_scores()

    if len(reference_graphs) != len(predicted_graphs):
        logging.warning(f"Graph count mismatch: {len(reference_graphs)} reference vs {len(predicted_graphs)} predicted - continuing with available pairs")
        # Use minimum count to evaluate available pairs rather than failing
        min_count = min(len(reference_graphs), len(predicted_graphs))
        reference_graphs = reference_graphs[:min_count]
        predicted_graphs = predicted_graphs[:min_count]

        if min_count == 0:
            return _get_empty_scores()

    bleu_scores = []
    rouge_scores = []

    for ref_graph, pred_graph in zip(reference_graphs, predicted_graphs):
        if not ref_graph or not pred_graph:
            bleu_scores.append((0.0, 0.0, 0.0))
            rouge_scores.append((0.0, 0.0, 0.0))
            continue

        # Convert graphs to text sequences
        ref_sequences = _graph_to_text_sequences(ref_graph)
        pred_sequences = _graph_to_text_sequences(pred_graph)

        # Compute BLEU scores
        bleu_result = _compute_bleu_scores(ref_sequences, pred_sequences)
        bleu_scores.append(bleu_result)

        # Compute ROUGE scores
        rouge_result = _compute_rouge_scores(ref_sequences, pred_sequences)
        rouge_scores.append(rouge_result)

    # Average scores across all graphs
    avg_bleu = _average_scores(bleu_scores)
    avg_rouge = _average_scores(rouge_scores)

    return {
        "bleu": {
            "precision": avg_bleu[0],
            "recall": avg_bleu[1],
            "f1": avg_bleu[2]
        },
        "rouge": {
            "precision": avg_rouge[0],
            "recall": avg_rouge[1],
            "f1": avg_rouge[2]
        }
    }


def _graph_to_text_sequences(graph: List[List[str]]) -> List[str]:
    """Convert graph triples to text sequences for similarity computation with validation."""
    sequences = []
    for triple in graph:
        # Use safe conversion that handles malformed triples
        edge_text = _safe_triple_to_text(triple)
        sequences.append(edge_text)
    return sequences


def _compute_bleu_scores(reference_sequences: List[str],
                        predicted_sequences: List[str]) -> tuple:
    """Compute BLEU precision, recall, and F1 scores."""
    if not reference_sequences or not predicted_sequences:
        return 0.0, 0.0, 0.0

    if NLTK_AVAILABLE:
        return _compute_nltk_bleu(reference_sequences, predicted_sequences)
    else:
        return _compute_simple_bleu(reference_sequences, predicted_sequences)


def _compute_nltk_bleu(reference_sequences: List[str],
                      predicted_sequences: List[str]) -> tuple:
    """Compute BLEU using NLTK implementation."""
    try:
        smoothing = SmoothingFunction().method1

        # Tokenize sequences
        ref_tokens = [seq.split() for seq in reference_sequences]
        pred_tokens = [seq.split() for seq in predicted_sequences]

        # Compute BLEU for each predicted sequence against all references
        bleu_scores = []
        for pred_seq in pred_tokens:
            if pred_seq:  # Skip empty sequences
                score = sentence_bleu(ref_tokens, pred_seq, smoothing_function=smoothing)
                bleu_scores.append(score)

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        # For precision/recall/F1, use BLEU as precision approximation
        precision = avg_bleu
        recall = avg_bleu  # Simplified: BLEU approximates both precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    except Exception as e:
        logging.warning(f"NLTK BLEU computation failed: {e}")
        return _compute_simple_bleu(reference_sequences, predicted_sequences)


def _compute_simple_bleu(reference_sequences: List[str],
                        predicted_sequences: List[str]) -> tuple:
    """Compute simplified BLEU-like scores using n-gram overlap."""
    # Convert to sets of words for overlap calculation
    ref_words = set()
    for seq in reference_sequences:
        ref_words.update(seq.split())

    pred_words = set()
    for seq in predicted_sequences:
        pred_words.update(seq.split())

    if not pred_words or not ref_words:
        return 0.0, 0.0, 0.0

    # Calculate word-level overlap
    intersection = ref_words.intersection(pred_words)

    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(ref_words)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def _compute_rouge_scores(reference_sequences: List[str],
                         predicted_sequences: List[str]) -> tuple:
    """Compute ROUGE precision, recall, and F1 scores."""
    if not reference_sequences or not predicted_sequences:
        return 0.0, 0.0, 0.0

    if ROUGE_AVAILABLE:
        return _compute_rouge_scorer(reference_sequences, predicted_sequences)
    else:
        return _compute_simple_rouge(reference_sequences, predicted_sequences)


def _compute_rouge_scorer(reference_sequences: List[str],
                         predicted_sequences: List[str]) -> tuple:
    """Compute ROUGE using rouge_score library."""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred_seq in predicted_sequences:
            best_precision = 0.0
            best_recall = 0.0
            best_f1 = 0.0

            # Find best scores against all reference sequences
            for ref_seq in reference_sequences:
                scores = scorer.score(ref_seq, pred_seq)
                rouge1_score = scores['rouge1']

                if rouge1_score.precision > best_precision:
                    best_precision = rouge1_score.precision
                if rouge1_score.recall > best_recall:
                    best_recall = rouge1_score.recall
                if rouge1_score.fmeasure > best_f1:
                    best_f1 = rouge1_score.fmeasure

            precision_scores.append(best_precision)
            recall_scores.append(best_recall)
            f1_scores.append(best_f1)

        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return avg_precision, avg_recall, avg_f1

    except Exception as e:
        logging.warning(f"ROUGE computation failed: {e}")
        return _compute_simple_rouge(reference_sequences, predicted_sequences)


def _compute_simple_rouge(reference_sequences: List[str],
                         predicted_sequences: List[str]) -> tuple:
    """Compute simplified ROUGE-like scores using word overlap."""
    # This is a simplified version similar to the BLEU implementation
    # Real ROUGE is more sophisticated with different n-gram strategies
    return _compute_simple_bleu(reference_sequences, predicted_sequences)


def _average_scores(score_list: List[tuple]) -> tuple:
    """Average precision, recall, F1 scores across multiple evaluations."""
    if not score_list:
        return 0.0, 0.0, 0.0

    precisions = [score[0] for score in score_list]
    recalls = [score[1] for score in score_list]
    f1s = [score[2] for score in score_list]

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)

    return avg_precision, avg_recall, avg_f1


def _get_empty_scores() -> Dict[str, Dict[str, float]]:
    """Return empty scores for error cases."""
    return {
        "bleu": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    }


# Additional utility functions for analysis
def analyze_text_similarity(reference_graph: List[List[str]],
                           predicted_graph: List[List[str]]) -> Dict[str, Any]:
    """
    Analyze text similarity between reference and predicted graphs.

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples

    Returns:
        Dict containing detailed text similarity analysis
    """
    ref_sequences = _graph_to_text_sequences(reference_graph)
    pred_sequences = _graph_to_text_sequences(predicted_graph)

    bleu_result = _compute_bleu_scores(ref_sequences, pred_sequences)
    rouge_result = _compute_rouge_scores(ref_sequences, pred_sequences)

    # Analyze vocabulary overlap
    ref_vocab = set()
    for seq in ref_sequences:
        ref_vocab.update(seq.split())

    pred_vocab = set()
    for seq in pred_sequences:
        pred_vocab.update(seq.split())

    vocab_intersection = ref_vocab.intersection(pred_vocab)

    return {
        "bleu_scores": {"precision": bleu_result[0], "recall": bleu_result[1], "f1": bleu_result[2]},
        "rouge_scores": {"precision": rouge_result[0], "recall": rouge_result[1], "f1": rouge_result[2]},
        "vocabulary_analysis": {
            "reference_vocab_size": len(ref_vocab),
            "predicted_vocab_size": len(pred_vocab),
            "shared_vocab_size": len(vocab_intersection),
            "vocab_overlap_ratio": len(vocab_intersection) / len(ref_vocab.union(pred_vocab)) if ref_vocab.union(pred_vocab) else 0.0
        },
        "sequence_analysis": {
            "reference_sequences": len(ref_sequences),
            "predicted_sequences": len(pred_sequences),
            "avg_ref_length": sum(len(seq.split()) for seq in ref_sequences) / len(ref_sequences) if ref_sequences else 0.0,
            "avg_pred_length": sum(len(seq.split()) for seq in pred_sequences) / len(pred_sequences) if pred_sequences else 0.0
        }
    }