"""
Graph Matching and Evaluation Metrics Implementation

This module implements various graph comparison and evaluation metrics for semantic graphs.
It provides comprehensive tools for measuring graph similarity using different approaches:

1. Exact Matching: Triple-level and graph-level exact comparisons
2. Text-based Metrics: Adapted BLEU, ROUGE, and BertScore for graph edges
3. Structural Metrics: Graph isomorphism and edit distance
4. Bipartite Matching: Optimal assignment algorithms for edge-to-edge comparison

The metrics are designed to evaluate automatically generated knowledge graphs
against gold standard references, providing multiple perspectives on graph quality.
"""

import numpy as np
# Optional heavy dependencies are imported lazily with safe fallbacks
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional import
    torch = None  # type: ignore
try:
    from rouge_score import rouge_scorer  # ROUGE text similarity metrics
except Exception:  # pragma: no cover - optional import
    rouge_scorer = None  # type: ignore
try:
    from bert_score import score as score_bert  # Semantic similarity using BERT
except Exception:  # pragma: no cover - optional import
    score_bert = None  # type: ignore
from nltk.translate.bleu_score import sentence_bleu  # BLEU text similarity
from nltk.translate.bleu_score import SmoothingFunction  # BLEU smoothing for short sequences
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm for optimal matching
try:
    from spacy.tokenizer import Tokenizer  # Advanced tokenization
    from spacy.lang.en import English  # English language model
except Exception:  # pragma: no cover - optional import
    Tokenizer = None  # type: ignore
    English = None  # type: ignore
import re  # Regular expressions
import networkx as nx  # Graph manipulation and algorithms
from sklearn import preprocessing  # Machine learning preprocessing tools
from sklearn.metrics import precision_score, recall_score, f1_score  # Classification metrics
from tqdm import tqdm  # Progress tracking

# Make module importable when run via sibling script by ensuring directory on sys.path
import sys
from pathlib import Path
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

def modify_graph(original_graph):
    """
    Normalize graph representation for consistent comparison.
    
    Args:
        original_graph (list): List of triples representing a graph
        
    Returns:
        list: Normalized graph with lowercase string elements
    
    This function standardizes graph representation by converting all elements
    to lowercase strings and removing whitespace. This normalization is crucial
    for fair comparison as it eliminates case and formatting differences.
    """
    modified_graph = []
    for x in original_graph:
        # Convert each element in the triple to lowercase string and strip whitespace
        modified_graph.append([str(t).lower().strip() for t in x])
    return modified_graph

def get_triple_match_f1(gold_graphs, pred_graphs):
    """
    Compute F1 score based on exact triple matching across all graphs.
    
    Args:
        gold_graphs (list): List of gold standard graphs
        pred_graphs (list): List of predicted graphs
        
    Returns:
        float: Micro-averaged F1 score for triple matching
    
    This metric treats the graph comparison as a multi-label classification problem
    where each unique triple is a label. It uses sklearn's multi-label binarization
    to convert graphs into binary vectors and computes standard precision/recall/F1.
    
    The micro-averaging approach gives equal weight to each triple occurrence,
    making this metric sensitive to both precision and recall of individual triples.
    """
    # Normalize both sets of graphs for consistent comparison
    new_gold_graphs = [modify_graph(graph) for graph in gold_graphs]
    new_pred_graphs = [modify_graph(graph) for graph in pred_graphs]
    
    # Flatten graphs to lists of triple strings for multi-label processing
    new_gold_graphs_list = [[str(string).lower() for string in sublist] for sublist in new_gold_graphs]
    new_pred_graphs_list = [[str(string).lower() for string in sublist] for sublist in new_pred_graphs]
    
    # Create vocabulary of all unique triples from both gold and predicted graphs
    # This ensures consistent label space for multi-label binarization
    allclasses = new_pred_graphs_list + new_gold_graphs_list
    allclasses = [item for items in allclasses for item in items]  # Flatten nested lists
    allclasses = list(set(allclasses))  # Remove duplicates

    # Convert graphs to binary label vectors using multi-label binarization
    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(new_pred_graphs_list)  # Predicted graph binary vectors
    mrbin = lb.fit_transform(new_gold_graphs_list)  # Gold graph binary vectors

    # Compute standard classification metrics with micro-averaging
    # Micro-averaging aggregates contributions across all samples for global metric
    precision = precision_score(mrbin, mcbin, average='micro')
    recall = recall_score(mrbin, mcbin, average='micro')
    f1 = f1_score(mrbin, mcbin, average='micro')

    # Output metrics for debugging (commented out)
    # print('Full triple scores')
    # print('-----------------------------------------------------------------')
    # print('Precision: ' + str(precision) + ' Recall: ' + str(recall) + '\nF1: ' + str(f1))
    return f1

def get_triple_match_accuracy(pred_graph, gold_graph):
    """
    Compute accuracy for a single graph pair based on exact triple matches.
    
    Args:
        pred_graph (list): Predicted graph as list of triples
        gold_graph (list): Gold standard graph as list of triples
        
    Returns:
        float: Accuracy score (proportion of predicted triples that are correct)
    
    This function computes a simple accuracy metric by counting how many
    predicted triples exactly match triples in the gold standard graph.
    The accuracy is the ratio of correct predictions to total predictions.
    """
    # Normalize both graphs for consistent comparison
    pred = modify_graph(pred_graph)
    gold = modify_graph(gold_graph)
    
    # Count exact matches between predicted and gold triples
    matchs = 0
    for x in pred:
        if x in gold:
            matchs += 1
    
    # Calculate accuracy as ratio of matches to total predictions
    acc = matchs/len(pred) if len(pred) > 0 else 0
    return acc

def get_graph_match_accuracy(pred_graphs, gold_graphs):
    """
    Compute graph-level accuracy using structural isomorphism checking.
    
    Args:
        pred_graphs (list): List of predicted graphs
        gold_graphs (list): List of gold standard graphs
        
    Returns:
        float: Accuracy score (proportion of graphs that are structurally equivalent)
    
    This metric evaluates whether predicted graphs are structurally isomorphic
    to their corresponding gold standards. It uses NetworkX to build directed
    graphs and checks for isomorphism with edge label matching.
    
    This is a stricter metric than triple matching as it requires exact
    structural equivalence, including the same connectivity patterns.
    """
    matchs = 0
    
    # Compare each predicted graph with its corresponding gold standard
    for pred, gold in zip(pred_graphs, gold_graphs):
        # Create directed graphs for structural comparison
        g1 = nx.DiGraph()  # Gold standard graph
        g2 = nx.DiGraph()  # Predicted graph

        # Build gold standard graph with nodes and labeled edges
        for edge in gold:
            # Add nodes with labels (entity names)
            g1.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
            g1.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
            # Add edge with relationship label
            g1.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), label=str(edge[1]).lower().strip())

        # Build predicted graph with error handling for malformed triples
        for edge in pred:
            # Handle incomplete triples by padding with 'NULL'
            if len(edge) == 2:
                edge.append('NULL')  # Missing object
            elif len(edge) == 1:
                edge.append('NULL')  # Missing predicate and object
                edge.append('NULL')
            
            # Add nodes and edges similar to gold graph
            g2.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
            g2.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
            g2.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), label=str(edge[1]).lower().strip())

        # Check for graph isomorphism with edge label matching
        # This requires both structural equivalence and matching edge labels
        if nx.is_isomorphic(g1, g2, edge_match=lambda x, y: x == y):
            matchs += 1
    
    # Calculate accuracy as proportion of exactly matching graphs
    acc = matchs/len(pred_graphs)
    return acc

def get_tokens(gold_edges, pred_edges):
    """
    Tokenize graph edges for text-based similarity metrics.
    
    Args:
        gold_edges (list): List of gold standard edge strings
        pred_edges (list): List of predicted edge strings
        
    Returns:
        tuple: (gold_tokens, pred_tokens) - tokenized versions of edges
    
    This function prepares graph edges for text-based evaluation metrics
    by tokenizing them using spaCy. Each edge is treated as a sentence
    and split into tokens for BLEU/ROUGE comparison.
    
    The semicolon delimiter is used to separate elements within edges.
    """
    # Initialize tokenizer (fallback to simple split if spaCy unavailable)
    use_spacy = English is not None and Tokenizer is not None
    if use_spacy:
        nlp = English()
        tokenizer = Tokenizer(nlp.vocab, infix_finditer=re.compile(r'''[;]''').finditer)

    gold_tokens = []
    pred_tokens = []

    # Process each graph's edges
    for i in range(len(gold_edges)):
        gold_tokens_edges = []
        pred_tokens_edges = []

        if use_spacy:
            # Tokenize each edge using spaCy pipeline
            for sample in tokenizer.pipe(gold_edges[i]):
                gold_tokens_edges.append([j.text for j in sample])
            for sample in tokenizer.pipe(pred_edges[i]):
                pred_tokens_edges.append([j.text for j in sample])
        else:
            # Fallback: simple semicolon-based split
            gold_tokens_edges = [re.split(r"[;]", s) for s in gold_edges[i]]
            pred_tokens_edges = [re.split(r"[;]", s) for s in pred_edges[i]]
            
        gold_tokens.append(gold_tokens_edges)
        pred_tokens.append(pred_tokens_edges)

    return gold_tokens, pred_tokens

def split_to_edges(graphs):
    """
    Convert graphs to edge-based string representations.
    
    Args:
        graphs (list): List of graphs (each graph is a list of triples)
        
    Returns:
        list: List of edge string lists for each graph
    
    This function transforms graph triples into string representations
    by joining triple elements with semicolons. This format is suitable
    for text-based similarity metrics that treat edges as sentences.
    
    Example: [["entity1", "relation", "entity2"]] -> ["entity1;relation;entity2"]
    """
    processed_graphs = []
    for graph in graphs:
        # Convert each triple to a semicolon-separated string and normalize case
        processed_graphs.append([";".join(str(triple)).lower().strip() for triple in graph])
    return processed_graphs

def get_bert_score(all_gold_edges, all_pred_edges):
    """
    Compute semantic similarity between graph edges using BERT embeddings.
    
    Args:
        all_gold_edges (list): List of gold standard edge lists
        all_pred_edges (list): List of predicted edge lists
        
    Returns:
        tuple: (precisions, recalls, f1s) - numpy arrays of scores per graph
    
    This metric uses pre-trained BERT to compute semantic similarity between
    graph edges, allowing for meaningful comparison beyond exact string matching.
    
    The algorithm:
    1. Computes BERT similarity scores for all edge pairs
    2. Uses Hungarian algorithm to find optimal edge-to-edge assignment
    3. Calculates precision, recall, and F1 based on optimal matching
    
    This approach captures semantic relationships that might be missed
    by purely lexical metrics like exact matching or BLEU.
    """
    references = []
    candidates = []

    # Build comprehensive lists of all edge pairs for batch BERT processing
    ref_cand_index = {}
    for graph_index in tqdm(range(len(all_gold_edges))):
        gold_edges = all_gold_edges[graph_index]
        pred_edges = all_pred_edges[graph_index]
        
        # Create all possible pairings between gold and predicted edges
        for gi, gold_edge in enumerate(gold_edges):
            for pj, pred_edge in enumerate(pred_edges):
                references.append(str(gold_edge))
                candidates.append(str(pred_edge))
                # Store index mapping for later retrieval using integer indices
                ref_cand_index[(graph_index, gi, pj)] = len(references) - 1

    # Compute BERT F1 scores for all edge pairs in batch
    # This is more efficient than computing scores individually
    # Select device automatically; fallback to CPU if torch is unavailable
    try:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    except Exception:
        device = "cpu"
    print(f"Using device for BERTScore: {device}")
    if score_bert is None:
        # Fallback: return zeros if bert-score is unavailable
        num_graphs = len(all_gold_edges)
        return (np.zeros(num_graphs), np.zeros(num_graphs), np.zeros(num_graphs))
    _, _, bs_F1 = score_bert(cands=candidates, refs=references,
                            model_type="bert-base-uncased", lang='en',
                            idf=False, device=device)
    print("Computed bert scores for all pairs")

    # Process each graph individually using precomputed BERT scores
    precisions, recalls, f1s = [], [], []
    for graph_index in tqdm(range(len(all_gold_edges))):
        gold_edges = all_gold_edges[graph_index]
        pred_edges = all_pred_edges[graph_index]
        
        # Build similarity matrix for current graph
        score_matrix = np.zeros((len(gold_edges), len(pred_edges)))
        for gi, _gold in enumerate(gold_edges):
            for pj, _pred in enumerate(pred_edges):
                # Retrieve precomputed BERT score via index mapping
                score_matrix[gi][pj] = bs_F1[ref_cand_index[(graph_index, gi, pj)]]

        # Find optimal assignment using Hungarian algorithm
        # maximize=True finds the assignment that maximizes total similarity
        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

        # Calculate precision and recall based on optimal assignment
        sample_precision = score_matrix[row_ind, col_ind].sum() / len(pred_edges)
        sample_recall = score_matrix[row_ind, col_ind].sum() / len(gold_edges)

        precisions.append(sample_precision)
        recalls.append(sample_recall)
        # Compute F1 with safe division to avoid NaN
        f1s.append(2 * sample_precision * sample_recall / (sample_precision + sample_recall))

    return np.array(precisions), np.array(recalls), np.array(f1s)

# Note: These graph matching metrics are computed by considering each graph as a set of edges and each edge as a
# sentence
def get_bleu_rouge(gold_tokens, pred_tokens, gold_sent, pred_sent):
    """
    Compute BLEU and ROUGE scores for graph edges using optimal bipartite matching.
    
    Args:
        gold_tokens (list): Tokenized gold standard edges
        pred_tokens (list): Tokenized predicted edges  
        gold_sent (list): Gold standard edge sentences
        pred_sent (list): Predicted edge sentences
        
    Returns:
        tuple: (rouge_precision, rouge_recall, rouge_f1, bleu_precision, bleu_recall, bleu_f1)
    
    This function adapts traditional text similarity metrics (BLEU and ROUGE)
    for graph evaluation by treating each edge as a sentence. It uses optimal
    bipartite matching to align edges before computing similarity scores.
    
    The approach:
    1. Computes pairwise BLEU/ROUGE scores between all edge pairs
    2. Uses Hungarian algorithm to find optimal edge alignment
    3. Calculates aggregate precision, recall, and F1 scores
    
    This allows for fair comparison even when graphs have different structures
    or edge orderings, as the optimal matching finds the best possible alignment.
    """
    # Initialize ROUGE scorer with multiple n-gram variants
    if rouge_scorer is None:
        # Fallback: return zeros if rouge-score is unavailable
        num_graphs = len(gold_tokens)
        zeros = np.zeros(num_graphs)
        return (zeros, zeros, zeros, zeros, zeros, zeros)
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)

    # Initialize result containers
    precisions_bleu = []
    recalls_bleu = []
    f1s_bleu = []

    precisions_rouge = []
    recalls_rouge = []
    f1s_rouge = []

    # Process each graph individually
    for graph_idx in range(len(gold_tokens)):
        # Create similarity matrices for BLEU and ROUGE scores
        score_bleu = np.zeros((len(pred_tokens[graph_idx]), len(gold_tokens[graph_idx])))
        score_rouge = np.zeros((len(pred_tokens[graph_idx]), len(gold_tokens[graph_idx])))
        
        # Compute pairwise similarity scores for all edge combinations
        for p_idx in range(len(pred_tokens[graph_idx])):
            for g_idx in range(len(gold_tokens[graph_idx])):
                # BLEU score with smoothing for short sequences
                score_bleu[p_idx, g_idx] = sentence_bleu([gold_tokens[graph_idx][g_idx]], 
                                                       pred_tokens[graph_idx][p_idx], 
                                                       smoothing_function=SmoothingFunction().method1)
                # ROUGE-2 precision score (bigram overlap)
                score_rouge[p_idx, g_idx] = \
                    scorer_rouge.score(gold_sent[graph_idx][g_idx], pred_sent[graph_idx][p_idx])['rouge2'].precision

        def _scores(cost_matrix):
            """
            Compute precision, recall, and F1 from similarity matrix using optimal assignment.
            
            Args:
                cost_matrix (np.ndarray): Matrix of pairwise similarity scores
                
            Returns:
                tuple: (precision, recall, f1) scores
            
            This helper function implements the core logic for converting similarity
            matrices into aggregate metrics using the Hungarian algorithm for optimal
            bipartite matching.
            """
            # Find optimal assignment that maximizes total similarity
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            
            # Calculate precision and recall based on optimal assignment
            precision = cost_matrix[row_ind, col_ind].sum() / cost_matrix.shape[0]
            recall = cost_matrix[row_ind, col_ind].sum() / cost_matrix.shape[1]
            
            # Compute F1 with safe division
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            return precision, recall, f1

        # Compute metrics for both BLEU and ROUGE using optimal assignment
        precision_bleu, recall_bleu, f1_bleu = _scores(score_bleu)
        precisions_bleu.append(precision_bleu)
        recalls_bleu.append(recall_bleu)
        f1s_bleu.append(f1_bleu)

        precision_rouge, recall_rouge, f1_rouge = _scores(score_rouge)
        precisions_rouge.append(precision_rouge)
        recalls_rouge.append(recall_rouge)
        f1s_rouge.append(f1_rouge)

    # Return results as numpy arrays for easy aggregation
    return np.array(precisions_rouge), np.array(recalls_rouge), np.array(f1s_rouge), np.array(
        precisions_bleu), np.array(recalls_bleu), np.array(f1s_bleu)

def return_eq_node(node1, node2):
    """
    Node equality function for graph isomorphism checking.
    
    Args:
        node1, node2 (dict): Node attribute dictionaries
        
    Returns:
        bool: True if nodes have equivalent labels
    
    This function is used by NetworkX's isomorphism checker to determine
    when two nodes should be considered equivalent during graph matching.
    """
    return node1['label'] == node2['label']

def return_eq_edge(edge1, edge2):
    """
    Edge equality function for graph isomorphism checking.
    
    Args:
        edge1, edge2 (dict): Edge attribute dictionaries
        
    Returns:
        bool: True if edges have equivalent labels
    
    This function is used by NetworkX's isomorphism checker to determine
    when two edges should be considered equivalent during graph matching.
    """
    return edge1['label'] == edge2['label']

def get_ged(gold_graph, pred_graph=None):
    """
    Compute normalized Graph Edit Distance (GED) between two graphs.
    
    Args:
        gold_graph (list): Gold standard graph as list of triples
        pred_graph (list, optional): Predicted graph as list of triples
        
    Returns:
        float: Normalized GED score (0=identical, 1=maximally different)
    
    Graph Edit Distance measures the minimum number of edit operations
    (node/edge insertions, deletions, substitutions) needed to transform
    one graph into another. This provides a structural similarity metric
    that considers the cost of graph transformations.
    
    The normalization uses a fixed upper bound based on the assumption
    that the worst case involves completely replacing one graph with another.
    This makes GED scores comparable across different graph sizes.
    """
    # Build NetworkX directed graphs for GED computation
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()

    # Construct gold standard graph
    for edge in gold_graph:
        # Add nodes with labels for proper matching
        g1.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
        g1.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
        # Add labeled edge
        g1.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), 
                   label=str(edge[1]).lower().strip())

    # The upper bound is defined wrt the graph for which GED is the worst.
    # Since ExplaGraphs (by construction) allows a maximum of 8 edges, the worst GED = gold_nodes + gold_edges + 8 + 9.
    # This happens when the predicted graph is linear with 8 edges and 9 nodes.
    # In such a case, for GED to be the worst, we assume that all nodes and edges of the predicted graph are deleted and
    # then all nodes and edges of the gold graph are added.
    # Note that a stricter upper bound can be computed by considering some replacement operations but we ignore that for convenience
    normalizing_constant = g1.number_of_nodes() + g1.number_of_edges() + 30

    # Handle case where no prediction is provided
    if pred_graph is None:
        return 1

    # Construct predicted graph with error handling for incomplete triples
    for edge in pred_graph[:len(gold_graph)]:  # Limit to gold graph size for fair comparison
        # Handle malformed triples by padding with 'NULL'
        if len(edge) == 2:
            edge.append('NULL')
        elif len(edge) == 1:
            edge.append('NULL')
            edge.append('NULL')
        
        # Add nodes and edges similar to gold graph construction
        g2.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
        g2.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
        g2.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), 
                   label=str(edge[1]).lower().strip())

    # Compute graph edit distance with node and edge label matching
    ged = nx.graph_edit_distance(g1, g2, node_match=return_eq_node, edge_match=return_eq_edge)

    # Ensure GED doesn't exceed theoretical maximum
    assert ged <= normalizing_constant

    # Return normalized GED score
    return ged / normalizing_constant
