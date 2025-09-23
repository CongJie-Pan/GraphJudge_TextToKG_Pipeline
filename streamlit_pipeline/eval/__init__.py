"""
Graph Quality Evaluation System for GraphJudge Streamlit Pipeline.

This package provides comprehensive graph quality assessment capabilities
based on proven metrics from graph_evaluation/metrics/eval.py.

Main Components:
- GraphEvaluator: Main evaluation engine
- metrics: Modular metric implementations
- graph_converter: Format conversion utilities
- report_generator: Report generation and export

Usage:
    from streamlit_pipeline.eval import GraphEvaluator
    from streamlit_pipeline.core.models import Triple

    evaluator = GraphEvaluator()
    predicted_graph = [Triple("subject", "predicate", "object")]
    reference_graph = [Triple("subject", "predicate", "object")]

    result = evaluator.evaluate_graph(predicted_graph, reference_graph)
    print(result.export_summary())
"""

from .graph_evaluator import GraphEvaluator

__all__ = ['GraphEvaluator']