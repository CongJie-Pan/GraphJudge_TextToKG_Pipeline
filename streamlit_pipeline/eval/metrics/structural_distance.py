"""
Structural distance metrics for graph evaluation.

This module implements Graph Edit Distance (GED) metrics adapted from
graph_evaluation/metrics/graph_matching.py for use in the
streamlit_pipeline evaluation system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

# Optional dependencies with graceful fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    logging.info("NetworkX available for graph structural analysis")
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    logging.warning("NetworkX not available, structural distance will use simplified implementation")


def get_graph_edit_distance(reference_graph: List[List[str]],
                           predicted_graph: List[List[str]],
                           normalize: bool = True) -> Optional[float]:
    """
    Compute Graph Edit Distance (GED) between reference and predicted graphs.

    GED measures the minimum number of edit operations (insertions, deletions,
    substitutions) needed to transform one graph into another.

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples
        normalize: Whether to normalize the distance by graph size

    Returns:
        Optional[float]: Normalized GED score (0.0 to 1.0) or None if computation fails
    """
    if not reference_graph or not predicted_graph:
        return None

    if NETWORKX_AVAILABLE:
        return _compute_networkx_ged(reference_graph, predicted_graph, normalize)
    else:
        logging.warning("NetworkX not available, using simplified GED approximation")
        return _compute_simple_ged(reference_graph, predicted_graph, normalize)


def _compute_networkx_ged(reference_graph: List[List[str]],
                         predicted_graph: List[List[str]],
                         normalize: bool) -> Optional[float]:
    """Compute GED using NetworkX implementation."""
    try:
        # Convert to NetworkX graphs
        ref_nx = _triples_to_networkx(reference_graph)
        pred_nx = _triples_to_networkx(predicted_graph)

        # Compute graph edit distance
        # Note: NetworkX GED can be computationally expensive for large graphs
        try:
            # Use optimize implementation for better performance
            ged = nx.graph_edit_distance(ref_nx, pred_nx, timeout=10)  # 10 second timeout

            if ged is None:
                logging.warning("GED computation timed out, using approximation")
                return _compute_simple_ged(reference_graph, predicted_graph, normalize)

            if normalize:
                # Normalize by the size of the larger graph
                max_size = max(len(reference_graph), len(predicted_graph))
                if max_size > 0:
                    ged = ged / max_size
                else:
                    ged = 0.0

            return float(ged)

        except nx.NetworkXError as e:
            logging.warning(f"NetworkX GED computation failed: {e}")
            return _compute_simple_ged(reference_graph, predicted_graph, normalize)

    except Exception as e:
        logging.error(f"NetworkX GED computation error: {e}")
        return _compute_simple_ged(reference_graph, predicted_graph, normalize)


def _compute_simple_ged(reference_graph: List[List[str]],
                       predicted_graph: List[List[str]],
                       normalize: bool) -> float:
    """Compute simplified GED approximation using set operations."""
    # Convert triples to sets for comparison
    ref_set = set(tuple(triple) for triple in reference_graph)
    pred_set = set(tuple(triple) for triple in predicted_graph)

    # Calculate symmetric difference (approximates edit distance)
    symmetric_diff = ref_set.symmetric_difference(pred_set)
    edit_distance = len(symmetric_diff)

    if normalize:
        # Normalize by the size of the larger graph
        max_size = max(len(ref_set), len(pred_set))
        if max_size > 0:
            edit_distance = edit_distance / max_size
        else:
            edit_distance = 0.0

    return float(edit_distance)


def _triples_to_networkx(graph: List[List[str]]) -> 'nx.DiGraph':
    """Convert list of triples to NetworkX directed graph."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX not available")

    G = nx.DiGraph()

    for triple in graph:
        if len(triple) >= 3:
            subject, predicate, obj = triple[0], triple[1], triple[2]
            # Add edge with predicate as edge attribute
            G.add_edge(subject, obj, predicate=predicate)

    return G


def analyze_structural_differences(reference_graph: List[List[str]],
                                 predicted_graph: List[List[str]]) -> Dict[str, Any]:
    """
    Analyze structural differences between reference and predicted graphs.

    Args:
        reference_graph: Reference graph as list of triples
        predicted_graph: Predicted graph as list of triples

    Returns:
        Dict containing detailed structural analysis
    """
    analysis = {
        "networkx_available": NETWORKX_AVAILABLE,
        "basic_statistics": _get_basic_graph_stats(reference_graph, predicted_graph)
    }

    # Compute GED if possible
    ged_result = get_graph_edit_distance(reference_graph, predicted_graph, normalize=True)
    if ged_result is not None:
        analysis["graph_edit_distance"] = {
            "normalized_ged": ged_result,
            "computation_method": "networkx" if NETWORKX_AVAILABLE else "simplified"
        }

    # Add NetworkX-specific analysis if available
    if NETWORKX_AVAILABLE:
        nx_analysis = _analyze_networkx_properties(reference_graph, predicted_graph)
        analysis.update(nx_analysis)

    # Add basic structural comparison
    structural_comparison = _compare_graph_structures(reference_graph, predicted_graph)
    analysis["structural_comparison"] = structural_comparison

    return analysis


def _get_basic_graph_stats(reference_graph: List[List[str]],
                          predicted_graph: List[List[str]]) -> Dict[str, Any]:
    """Get basic statistical information about both graphs."""
    def get_stats(graph):
        if not graph:
            return {"nodes": 0, "edges": 0, "unique_predicates": 0}

        nodes = set()
        predicates = set()

        for triple in graph:
            if len(triple) >= 3:
                nodes.add(triple[0])  # subject
                nodes.add(triple[2])  # object
                predicates.add(triple[1])  # predicate

        return {
            "nodes": len(nodes),
            "edges": len(graph),
            "unique_predicates": len(predicates)
        }

    ref_stats = get_stats(reference_graph)
    pred_stats = get_stats(predicted_graph)

    return {
        "reference": ref_stats,
        "predicted": pred_stats,
        "size_difference": {
            "nodes_diff": abs(ref_stats["nodes"] - pred_stats["nodes"]),
            "edges_diff": abs(ref_stats["edges"] - pred_stats["edges"]),
            "predicates_diff": abs(ref_stats["unique_predicates"] - pred_stats["unique_predicates"])
        }
    }


def _analyze_networkx_properties(reference_graph: List[List[str]],
                               predicted_graph: List[List[str]]) -> Dict[str, Any]:
    """Analyze graph properties using NetworkX."""
    try:
        ref_nx = _triples_to_networkx(reference_graph)
        pred_nx = _triples_to_networkx(predicted_graph)

        ref_props = _get_networkx_properties(ref_nx)
        pred_props = _get_networkx_properties(pred_nx)

        return {
            "networkx_properties": {
                "reference": ref_props,
                "predicted": pred_props,
                "property_differences": {
                    "density_diff": abs(ref_props["density"] - pred_props["density"]),
                    "components_diff": abs(ref_props["connected_components"] - pred_props["connected_components"]),
                    "avg_degree_diff": abs(ref_props["average_degree"] - pred_props["average_degree"])
                }
            }
        }

    except Exception as e:
        logging.warning(f"NetworkX property analysis failed: {e}")
        return {"networkx_properties": {"error": str(e)}}


def _get_networkx_properties(G: 'nx.DiGraph') -> Dict[str, float]:
    """Get NetworkX graph properties."""
    props = {}

    try:
        # Basic properties
        props["nodes"] = G.number_of_nodes()
        props["edges"] = G.number_of_edges()
        props["density"] = nx.density(G)

        # Degree statistics
        if G.number_of_nodes() > 0:
            degrees = [d for n, d in G.degree()]
            props["average_degree"] = sum(degrees) / len(degrees)
            props["max_degree"] = max(degrees) if degrees else 0
        else:
            props["average_degree"] = 0.0
            props["max_degree"] = 0

        # Connected components (for underlying undirected graph)
        undirected = G.to_undirected()
        props["connected_components"] = nx.number_connected_components(undirected)

        # Clustering (if not too large)
        if G.number_of_nodes() < 1000:  # Avoid expensive computation for large graphs
            props["clustering_coefficient"] = nx.average_clustering(undirected)
        else:
            props["clustering_coefficient"] = 0.0

    except Exception as e:
        logging.warning(f"Error computing graph properties: {e}")
        props["error"] = str(e)

    return props


def _compare_graph_structures(reference_graph: List[List[str]],
                            predicted_graph: List[List[str]]) -> Dict[str, Any]:
    """Compare basic structural properties of the graphs."""
    # Node sets
    ref_nodes = set()
    pred_nodes = set()

    # Predicate sets
    ref_predicates = set()
    pred_predicates = set()

    # Edge patterns (subject-predicate pairs)
    ref_patterns = set()
    pred_patterns = set()

    for triple in reference_graph:
        if len(triple) >= 3:
            ref_nodes.update([triple[0], triple[2]])
            ref_predicates.add(triple[1])
            ref_patterns.add((triple[0], triple[1]))

    for triple in predicted_graph:
        if len(triple) >= 3:
            pred_nodes.update([triple[0], triple[2]])
            pred_predicates.add(triple[1])
            pred_patterns.add((triple[0], triple[1]))

    # Calculate overlaps
    node_intersection = ref_nodes.intersection(pred_nodes)
    predicate_intersection = ref_predicates.intersection(pred_predicates)
    pattern_intersection = ref_patterns.intersection(pred_patterns)

    return {
        "node_overlap": {
            "intersection_size": len(node_intersection),
            "reference_unique": len(ref_nodes - pred_nodes),
            "predicted_unique": len(pred_nodes - ref_nodes),
            "overlap_ratio": len(node_intersection) / len(ref_nodes.union(pred_nodes)) if ref_nodes.union(pred_nodes) else 0.0
        },
        "predicate_overlap": {
            "intersection_size": len(predicate_intersection),
            "reference_unique": len(ref_predicates - pred_predicates),
            "predicted_unique": len(pred_predicates - ref_predicates),
            "overlap_ratio": len(predicate_intersection) / len(ref_predicates.union(pred_predicates)) if ref_predicates.union(pred_predicates) else 0.0
        },
        "pattern_overlap": {
            "intersection_size": len(pattern_intersection),
            "reference_unique": len(ref_patterns - pred_patterns),
            "predicted_unique": len(pred_patterns - ref_patterns),
            "overlap_ratio": len(pattern_intersection) / len(ref_patterns.union(pred_patterns)) if ref_patterns.union(pred_patterns) else 0.0
        }
    }


def check_structural_analysis_availability() -> Dict[str, Any]:
    """
    Check availability of structural analysis dependencies.

    Returns:
        Dict containing availability status and configuration info
    """
    info = {
        "networkx_available": NETWORKX_AVAILABLE,
        "ged_computation": "networkx" if NETWORKX_AVAILABLE else "simplified"
    }

    if NETWORKX_AVAILABLE:
        try:
            # Test NetworkX functionality
            G = nx.DiGraph()
            G.add_edge("A", "B", predicate="test")
            info["networkx_version"] = nx.__version__
            info["test_computation"] = "success"
        except Exception as e:
            info["test_computation"] = f"failed: {str(e)}"
            info["error"] = str(e)

    return info