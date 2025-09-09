"""
Graph Evaluation Metrics Suite

This script provides a comprehensive evaluation framework for comparing predicted semantic graphs
against gold standard reference graphs. It implements multiple graph comparison metrics to assess
the quality of automatically generated knowledge graphs.

Main functionality:
1. Loads predicted and gold standard graphs from files
2. Evaluates graphs using multiple complementary metrics:
   - Triple Match F1: Measures exact triple matching performance
   - Graph Match Accuracy: Evaluates structural graph isomorphism
   - G-BLEU/G-ROUGE: Applies text similarity metrics to graph edges
   - G-BertScore: Uses semantic similarity for graph edge comparison
   - Graph Edit Distance (GED): Measures minimum edit operations needed

The evaluation provides a multi-faceted assessment of graph generation quality,
covering both structural accuracy and semantic similarity aspects.
"""

import ast  # For safely parsing string representations of Python literals
import argparse  # Command line argument parsing
import json  # For JSON output functionality
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure this file's directory is on sys.path so sibling imports work reliably
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

# Import custom graph matching functions for various evaluation metrics
from graph_matching import split_to_edges, get_tokens, get_bleu_rouge, get_bert_score, get_ged, get_triple_match_f1, get_graph_match_accuracy
from tqdm import tqdm  # Progress bar for long-running operations


# Preprocessing functionality has been moved to eval_preDataProcess.py
# Use eval_preDataProcess.py to convert CSV files to pred.txt and gold.txt formats


def _ensure_pred_txt_path(pred_file_path: str) -> str:
    """
    Ensure the prediction file path is a .txt file usable by this evaluator.

    Behavior:
    - If input is a .txt file path, return it unchanged.
    - If input is a .csv file path, convert it to a .pred.txt file in the same directory
      using the csv_to_predtxt tool, then return the new .pred.txt path.
    - Raise FileNotFoundError if the input path does not exist.
    - Raise ValueError for unsupported file extensions.
    """
    in_path = Path(pred_file_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {in_path}")

    if in_path.suffix.lower() == ".txt":
        return str(in_path)

    if in_path.suffix.lower() == ".csv":
        # Compute output path: same directory, name '<stem>.pred.txt'
        out_path = in_path.parent / f"{in_path.stem}.pred.txt"

        # Import converter from tools dynamically
        # Try multiple relative locations to be robust in different run contexts
        candidate_tools_dirs = [
            Path(__file__).resolve().parents[3] / "tools",
            Path(__file__).resolve().parents[2] / "tools",
            Path(__file__).resolve().parents[4] / "Miscellaneous" / "KgGen" / "GraphJudge" / "tools",
        ]
        convert_csv_to_pred_txt = None
        for tools_dir in candidate_tools_dirs:
            if tools_dir.exists():
                sys.path.insert(0, str(tools_dir))
                try:
                    from csv_to_predtxt import convert_csv_to_pred_txt as _conv  # type: ignore
                    convert_csv_to_pred_txt = _conv
                    break
                except Exception:
                    continue
        if convert_csv_to_pred_txt is None:
            raise ModuleNotFoundError("Could not locate 'csv_to_predtxt' module in expected tools directories")

        convert_csv_to_pred_txt(in_path, out_path)
        return str(out_path)

    raise ValueError("Unsupported prediction file extension. Use .txt or .csv")


def _collect_evaluation_results(
    gold_graphs: list,
    pred_graphs: list,
    triple_match_f1: float,
    graph_match_accuracy: float,
    precisions_rouge: Any,
    recalls_rouge: Any,
    f1s_rouge: Any,
    precisions_bleu: Any,
    recalls_bleu: Any,
    f1s_bleu: Any,
    precisions_BS: Any,
    recalls_BS: Any,
    f1s_BS: Any,
    overall_ged: Optional[float] = None
) -> Dict[str, Any]:
    """
    Collect all evaluation results into a structured dictionary.
    
    Args:
        gold_graphs: List of gold standard graphs
        pred_graphs: List of predicted graphs
        All metric results from various evaluation functions
        overall_ged: Optional GED score
        
    Returns:
        Dict containing all evaluation results in structured format
    """
    num_graphs = len(gold_graphs)
    
    results = {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_graphs": num_graphs,
            "evaluation_metrics": [
                "triple_match_f1",
                "graph_match_accuracy", 
                "g_bleu",
                "g_rouge",
                "g_bert_score"
            ]
        },
        "exact_matching_metrics": {
            "triple_match_f1": {
                "score": round(float(triple_match_f1), 4),
                "description": "Exact triple matching F1 score"
            },
            "graph_match_accuracy": {
                "score": round(float(graph_match_accuracy), 4),
                "description": "Structural graph isomorphism accuracy"
            }
        },
        "text_similarity_metrics": {
            "g_bleu": {
                "precision": round(float(precisions_bleu.sum() / num_graphs), 4),
                "recall": round(float(recalls_bleu.sum() / num_graphs), 4),
                "f1": round(float(f1s_bleu.sum() / num_graphs), 4),
                "description": "Graph-adapted BLEU scores using optimal edge matching"
            },
            "g_rouge": {
                "precision": round(float(precisions_rouge.sum() / num_graphs), 4),
                "recall": round(float(recalls_rouge.sum() / num_graphs), 4),
                "f1": round(float(f1s_rouge.sum() / num_graphs), 4),
                "description": "Graph-adapted ROUGE scores using optimal edge matching"
            }
        },
        "semantic_similarity_metrics": {
            "g_bert_score": {
                "precision": round(float(precisions_BS.sum() / num_graphs), 4),
                "recall": round(float(recalls_BS.sum() / num_graphs), 4),
                "f1": round(float(f1s_BS.sum() / num_graphs), 4),
                "description": "Graph-adapted BERTScore using semantic embeddings"
            }
        }
    }
    
    if overall_ged is not None:
        results["structural_distance_metrics"] = {
            "graph_edit_distance": {
                "average_ged": round(float(overall_ged / num_graphs), 4),
                "description": "Average minimum edit operations to transform predicted to gold graph"
            }
        }
    
    return results


def _output_results(results: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """
    Output evaluation results to console and optionally to JSON file.
    
    Args:
        results: Dictionary containing all evaluation results
        output_file: Optional path to JSON output file
    """
    # Console output (maintain backward compatibility)
    print("\n" + "="*60)
    print("GRAPH EVALUATION RESULTS")
    print("="*60)
    
    # Exact matching metrics
    exact_metrics = results["exact_matching_metrics"]
    print(f'\nExact Matching Metrics:')
    print(f'  Triple Match F1 Score: {exact_metrics["triple_match_f1"]["score"]:.4f}')
    print(f'  Graph Match Accuracy: {exact_metrics["graph_match_accuracy"]["score"]:.4f}')
    
    # Text similarity metrics
    text_metrics = results["text_similarity_metrics"]
    print(f'\nText Similarity Metrics:')
    
    g_bleu = text_metrics["g_bleu"]
    print(f'  G-BLEU Precision: {g_bleu["precision"]:.4f}')
    print(f'  G-BLEU Recall: {g_bleu["recall"]:.4f}')
    print(f'  G-BLEU F1: {g_bleu["f1"]:.4f}')
    
    g_rouge = text_metrics["g_rouge"]
    print(f'  G-ROUGE Precision: {g_rouge["precision"]:.4f}')
    print(f'  G-ROUGE Recall: {g_rouge["recall"]:.4f}')
    print(f'  G-ROUGE F1: {g_rouge["f1"]:.4f}')
    
    # Semantic similarity metrics
    semantic_metrics = results["semantic_similarity_metrics"]
    g_bert = semantic_metrics["g_bert_score"]
    print(f'\nSemantic Similarity Metrics:')
    print(f'  G-BERTScore Precision: {g_bert["precision"]:.4f}')
    print(f'  G-BERTScore Recall: {g_bert["recall"]:.4f}')
    print(f'  G-BERTScore F1: {g_bert["f1"]:.4f}')
    
    # GED if available
    if "structural_distance_metrics" in results:
        ged_metric = results["structural_distance_metrics"]["graph_edit_distance"]
        print(f'\nStructural Distance Metrics:')
        print(f'  Graph Edit Distance (GED): {ged_metric["average_ged"]:.4f}')
    
    print(f'\nTotal graphs evaluated: {results["evaluation_metadata"]["total_graphs"]}')
    print("="*60)
    
    # JSON output if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Detailed results saved to: {output_path}")
        print(f"   Timestamp: {results['evaluation_metadata']['timestamp']}")
        print(f"   Format: Structured JSON with metadata and descriptions")

if __name__ == '__main__':
    # Set up command line argument parsing for input files
    parser = argparse.ArgumentParser(
        description="Graph Evaluation Metrics Suite - Compare predicted vs gold graphs"
    )
    parser.add_argument("--pred_file", default=None, type=str, required=True,
                       help="Path to file containing predicted graphs (.txt format only)")
    parser.add_argument("--gold_file", default=None, type=str, required=True,
                       help="Path to file containing gold standard reference graphs (.txt format only)")
    parser.add_argument("--out", "--output", type=str, default=None,
                       help="Optional path to save detailed JSON results (e.g., results.json)")
    parser.add_argument("--enable-ged", action="store_true", default=False,
                       help="Enable Graph Edit Distance calculation (computationally expensive)")

    args = parser.parse_args()

    # Validate file extensions (preprocessing should be done separately)
    pred_path = Path(args.pred_file)
    gold_path = Path(args.gold_file)
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
        
    if pred_path.suffix.lower() != ".txt":
        raise ValueError(f"Prediction file must be .txt format. Use eval_preDataProcess.py to convert CSV files.")
    if gold_path.suffix.lower() != ".txt":
        raise ValueError(f"Gold file must be .txt format. Use eval_preDataProcess.py to convert CSV files.")
    
    # Load gold standard (reference) graphs from file
    print("Loading gold standard graphs...")
    gold_graphs = []
    with open(args.gold_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            # Use ast.literal_eval for safe parsing of Python literal structures
            gold_graphs.append(ast.literal_eval(line.strip()))
		
    # Load predicted graphs from file with error handling
    print("Loading predicted graphs...")
    pred_graphs = []
    malformed_count = 0
    with open(args.pred_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            try:
                # Attempt to parse the predicted graph
                pred_graphs.append(ast.literal_eval(line.strip()))
            except:
                # If parsing fails, use a null placeholder triple
                # This ensures evaluation can continue despite malformed predictions
                pred_graphs.append([["Null", "Null", "Null"]])
                malformed_count += 1
    
    print(f"Loaded {len(pred_graphs)} predictions ({malformed_count} malformed)")
    print(f"Loaded {len(gold_graphs)} gold standards")
    
    # Ensure equal number of predictions and gold standards for fair comparison
    if len(gold_graphs) != len(pred_graphs):
        raise ValueError(
            f"File length mismatch: gold_file has {len(gold_graphs)} graphs, "
            f"pred_file has {len(pred_graphs)} graphs. Files must have equal length."
        )

    # === EVALUATION METRICS ===
    print("\nComputing evaluation metrics...")
    
    # 1. Triple Match F1 Score: Measures exact matching of individual triples
    print("  Computing Triple Match F1...")
    triple_match_f1 = get_triple_match_f1(gold_graphs, pred_graphs)  

    # 2. Graph Match Accuracy: Evaluates structural graph isomorphism
    print("  Computing Graph Match Accuracy...")
    graph_match_accuracy = get_graph_match_accuracy(pred_graphs, gold_graphs)

    # === TEXT-BASED GRAPH METRICS ===
    # Convert graphs to edge-based representations for text similarity metrics
    print("  Converting graphs to edge representations...")
    gold_edges = split_to_edges(gold_graphs)
    pred_edges = split_to_edges(pred_graphs)
	
    # Tokenize edges for text-based similarity computation
    print("  Tokenizing edges...")
    gold_tokens, pred_tokens = get_tokens(gold_edges, pred_edges)
    
    # 3. G-BLEU and G-ROUGE: Apply text similarity metrics to graph edges
    print("  Computing G-BLEU and G-ROUGE scores...")
    precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges)
    
    # 4. G-BertScore: Use pre-trained BERT for semantic similarity of graph edges
    print("  Computing G-BERTScore...")
    precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)

    # 5. Graph Edit Distance (GED) - Optional expensive computation
    overall_ged = None
    if args.enable_ged:
        print("  Computing Graph Edit Distance (this may take a while)...")
        overall_ged = 0.0
        for i in tqdm(range(len(gold_graphs)), desc="Computing GED"):
            ged = get_ged(gold_graphs[i], pred_graphs[i])
            overall_ged += ged
    
    # === COLLECT AND OUTPUT RESULTS ===
    print("\nCollecting results...")
    results = _collect_evaluation_results(
        gold_graphs=gold_graphs,
        pred_graphs=pred_graphs,
        triple_match_f1=triple_match_f1,
        graph_match_accuracy=graph_match_accuracy,
        precisions_rouge=precisions_rouge,
        recalls_rouge=recalls_rouge,
        f1s_rouge=f1s_rouge,
        precisions_bleu=precisions_bleu,
        recalls_bleu=recalls_bleu,
        f1s_bleu=f1s_bleu,
        precisions_BS=precisions_BS,
        recalls_BS=recalls_BS,
        f1s_BS=f1s_BS,
        overall_ged=overall_ged
    )
    
    # Output results to console and optionally to JSON file
    _output_results(results, args.out)
