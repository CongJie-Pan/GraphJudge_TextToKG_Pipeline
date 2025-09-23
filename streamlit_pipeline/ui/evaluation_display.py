"""
Evaluation Display Components for GraphJudge Streamlit Application.

This module provides comprehensive UI components for displaying graph quality
evaluation results, metrics visualizations, and comparative analysis tools.
Components follow the design patterns from spec.md Section 12 and integrate
with the evaluation system architecture.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import io
import base64

# Optional visualization library imports with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None

try:
    # Try absolute import first (for package installation)
    from streamlit_pipeline.core.models import GraphMetrics, EvaluationResult, Triple
    from streamlit_pipeline.utils.i18n import get_text
    from streamlit_pipeline.eval.graph_evaluator import GraphEvaluator
except ImportError:
    # Fallback to relative imports (for direct execution)
    from ..core.models import GraphMetrics, EvaluationResult, Triple
    from ..utils.i18n import get_text
    from ..eval.graph_evaluator import GraphEvaluator


def display_evaluation_dashboard(evaluation_result: EvaluationResult, show_detailed: bool = True):
    """
    Display comprehensive evaluation dashboard with metrics and visualizations.

    Args:
        evaluation_result: Complete evaluation results with metrics
        show_detailed: Whether to show detailed metric breakdowns
    """
    if not evaluation_result.success:
        st.error(f"{get_text('evaluation.error_title')}: {evaluation_result.error}")
        return

    st.markdown(f"## {get_text('evaluation.dashboard_title')}")

    # Display overview metrics
    _display_metrics_overview(evaluation_result.metrics)

    # Display detailed metrics breakdown
    if show_detailed:
        _display_detailed_metrics(evaluation_result.metrics)

    # Display metadata and processing info
    _display_evaluation_metadata(evaluation_result)

    # Display visualization charts
    if PLOTLY_AVAILABLE:
        _display_metrics_charts(evaluation_result.metrics)
    else:
        st.warning(get_text('evaluation.plotly_not_available'))


def _display_metrics_overview(metrics: GraphMetrics):
    """Display overview metrics in a card layout."""
    st.markdown(f"### {get_text('evaluation.overview_title')}")

    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            get_text('evaluation.triple_match_f1'),
            f"{metrics.triple_match_f1:.3f}",
            delta=_get_quality_indicator(metrics.triple_match_f1)
        )

    with col2:
        st.metric(
            get_text('evaluation.graph_accuracy'),
            f"{metrics.graph_match_accuracy:.3f}",
            delta=_get_quality_indicator(metrics.graph_match_accuracy)
        )

    with col3:
        st.metric(
            get_text('evaluation.g_bleu_f1'),
            f"{metrics.g_bleu_f1:.3f}",
            delta=_get_quality_indicator(metrics.g_bleu_f1)
        )

    with col4:
        st.metric(
            get_text('evaluation.g_bert_f1'),
            f"{metrics.g_bert_f1:.3f}",
            delta=_get_quality_indicator(metrics.g_bert_f1)
        )


def _display_detailed_metrics(metrics: GraphMetrics):
    """Display detailed breakdown of all metrics."""
    st.markdown(f"### {get_text('evaluation.detailed_title')}")

    # Create expandable sections for each metric category
    with st.expander(get_text('evaluation.exact_matching_title'), expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('evaluation.triple_match_f1'), f"{metrics.triple_match_f1:.4f}")
        with col2:
            st.metric(get_text('evaluation.graph_accuracy'), f"{metrics.graph_match_accuracy:.4f}")

    with st.expander(get_text('evaluation.text_similarity_title'), expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**G-BLEU**")
            st.metric(get_text('evaluation.precision'), f"{metrics.g_bleu_precision:.4f}")
            st.metric(get_text('evaluation.recall'), f"{metrics.g_bleu_recall:.4f}")
            st.metric(get_text('evaluation.f1_score'), f"{metrics.g_bleu_f1:.4f}")

        with col2:
            st.markdown("**G-ROUGE**")
            st.metric(get_text('evaluation.precision'), f"{metrics.g_rouge_precision:.4f}")
            st.metric(get_text('evaluation.recall'), f"{metrics.g_rouge_recall:.4f}")
            st.metric(get_text('evaluation.f1_score'), f"{metrics.g_rouge_f1:.4f}")

        with col3:
            st.markdown("**G-BertScore**")
            st.metric(get_text('evaluation.precision'), f"{metrics.g_bert_precision:.4f}")
            st.metric(get_text('evaluation.recall'), f"{metrics.g_bert_recall:.4f}")
            st.metric(get_text('evaluation.f1_score'), f"{metrics.g_bert_f1:.4f}")

    if metrics.graph_edit_distance is not None:
        with st.expander(get_text('evaluation.structural_similarity_title'), expanded=False):
            st.metric(
                get_text('evaluation.graph_edit_distance'),
                f"{metrics.graph_edit_distance:.4f}",
                help=get_text('evaluation.ged_help')
            )
    else:
        st.info(get_text('evaluation.ged_not_available'))


def _display_evaluation_metadata(evaluation_result: EvaluationResult):
    """Display evaluation metadata and processing information."""
    st.markdown(f"### {get_text('evaluation.metadata_title')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Processing Information**")
        st.text(f"Processing Time: {evaluation_result.processing_time:.3f}s")
        st.text(f"Success: {'Yes' if evaluation_result.success else 'No'}")

        if evaluation_result.metadata:
            if 'evaluation_time' in evaluation_result.metadata:
                st.text(f"Evaluation Time: {evaluation_result.metadata['evaluation_time']}")
            if 'config' in evaluation_result.metadata:
                config = evaluation_result.metadata['config']
                st.text(f"BERT Score Enabled: {'Yes' if config.get('enable_bert_score', False) else 'No'}")
                st.text(f"GED Enabled: {'Yes' if config.get('enable_ged', False) else 'No'}")

    with col2:
        st.markdown("**Graph Information**")
        if evaluation_result.reference_graph_info:
            ref_info = evaluation_result.reference_graph_info
            st.text(f"Reference Graph Size: {ref_info.get('size', 'N/A')} triples")
            st.text(f"Unique Subjects: {ref_info.get('unique_subjects', 'N/A')}")
            st.text(f"Unique Predicates: {ref_info.get('unique_predicates', 'N/A')}")

        if evaluation_result.predicted_graph_info:
            pred_info = evaluation_result.predicted_graph_info
            st.text(f"Predicted Graph Size: {pred_info.get('size', 'N/A')} triples")


def _display_metrics_charts(metrics: GraphMetrics):
    """Display interactive charts for metrics visualization."""
    st.markdown(f"### {get_text('evaluation.visualization_title')}")

    # Create metrics comparison radar chart
    _create_radar_chart(metrics)

    # Create precision/recall comparison
    _create_precision_recall_chart(metrics)


def _create_radar_chart(metrics: GraphMetrics):
    """Create radar chart for overall metrics comparison."""
    if not PLOTLY_AVAILABLE:
        return

    # Define metrics for radar chart
    metric_names = [
        'Triple Match F1',
        'Graph Accuracy',
        'G-BLEU F1',
        'G-ROUGE F1',
        'G-BertScore F1'
    ]

    metric_values = [
        metrics.triple_match_f1,
        metrics.graph_match_accuracy,
        metrics.g_bleu_f1,
        metrics.g_rouge_f1,
        metrics.g_bert_f1
    ]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=metric_values,
        theta=metric_names,
        fill='toself',
        name='Graph Quality Metrics',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title=get_text('evaluation.radar_chart_title'),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def _create_precision_recall_chart(metrics: GraphMetrics):
    """Create precision/recall comparison chart."""
    if not PLOTLY_AVAILABLE:
        return

    # Prepare data for precision/recall comparison
    categories = ['G-BLEU', 'G-ROUGE', 'G-BertScore']
    precision_values = [metrics.g_bleu_precision, metrics.g_rouge_precision, metrics.g_bert_precision]
    recall_values = [metrics.g_bleu_recall, metrics.g_rouge_recall, metrics.g_bert_recall]
    f1_values = [metrics.g_bleu_f1, metrics.g_rouge_f1, metrics.g_bert_f1]

    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(get_text('evaluation.precision_recall_title'), get_text('evaluation.f1_comparison_title')),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Precision/Recall bar chart
    fig.add_trace(
        go.Bar(name='Precision', x=categories, y=precision_values, marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Recall', x=categories, y=recall_values, marker_color='lightcoral'),
        row=1, col=1
    )

    # F1 Score line chart
    fig.add_trace(
        go.Scatter(name='F1 Score', x=categories, y=f1_values, mode='lines+markers',
                  marker_color='green', line=dict(width=3)),
        row=1, col=2
    )

    fig.update_layout(
        title=get_text('evaluation.metrics_comparison_title'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def display_comparative_analysis(evaluation_results: List[Tuple[str, EvaluationResult]]):
    """
    Display comparative analysis of multiple evaluation results.

    Args:
        evaluation_results: List of (label, evaluation_result) tuples for comparison
    """
    if len(evaluation_results) < 2:
        st.warning(get_text('evaluation.comparison_insufficient_data'))
        return

    st.markdown(f"### {get_text('evaluation.comparative_analysis_title')}")

    # Create comparison table
    _create_comparison_table(evaluation_results)

    # Create comparison charts
    if PLOTLY_AVAILABLE:
        _create_comparison_charts(evaluation_results)


def _create_comparison_table(evaluation_results: List[Tuple[str, EvaluationResult]]):
    """Create comparison table for multiple evaluation results."""

    # Prepare data for comparison table
    comparison_data = []
    for label, result in evaluation_results:
        if result.success:
            metrics = result.metrics
            comparison_data.append({
                'Run': label,
                'Triple Match F1': f"{metrics.triple_match_f1:.4f}",
                'Graph Accuracy': f"{metrics.graph_match_accuracy:.4f}",
                'G-BLEU F1': f"{metrics.g_bleu_f1:.4f}",
                'G-ROUGE F1': f"{metrics.g_rouge_f1:.4f}",
                'G-BertScore F1': f"{metrics.g_bert_f1:.4f}",
                'Processing Time': f"{result.processing_time:.3f}s"
            })
        else:
            comparison_data.append({
                'Run': label,
                'Triple Match F1': 'Error',
                'Graph Accuracy': 'Error',
                'G-BLEU F1': 'Error',
                'G-ROUGE F1': 'Error',
                'G-BertScore F1': 'Error',
                'Processing Time': f"{result.processing_time:.3f}s"
            })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)


def _create_comparison_charts(evaluation_results: List[Tuple[str, EvaluationResult]]):
    """Create comparison charts for multiple evaluation results."""
    if not PLOTLY_AVAILABLE:
        return

    # Filter successful results
    successful_results = [(label, result) for label, result in evaluation_results if result.success]

    if len(successful_results) < 2:
        st.warning(get_text('evaluation.comparison_no_successful_results'))
        return

    # Prepare data for comparison
    labels = [label for label, _ in successful_results]

    # Extract metrics
    triple_f1 = [result.metrics.triple_match_f1 for _, result in successful_results]
    graph_acc = [result.metrics.graph_match_accuracy for _, result in successful_results]
    bleu_f1 = [result.metrics.g_bleu_f1 for _, result in successful_results]
    rouge_f1 = [result.metrics.g_rouge_f1 for _, result in successful_results]
    bert_f1 = [result.metrics.g_bert_f1 for _, result in successful_results]

    # Create comparison bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(name='Triple Match F1', x=labels, y=triple_f1))
    fig.add_trace(go.Bar(name='Graph Accuracy', x=labels, y=graph_acc))
    fig.add_trace(go.Bar(name='G-BLEU F1', x=labels, y=bleu_f1))
    fig.add_trace(go.Bar(name='G-ROUGE F1', x=labels, y=rouge_f1))
    fig.add_trace(go.Bar(name='G-BertScore F1', x=labels, y=bert_f1))

    fig.update_layout(
        title=get_text('evaluation.comparison_chart_title'),
        xaxis_title=get_text('evaluation.evaluation_runs'),
        yaxis_title=get_text('evaluation.metric_scores'),
        barmode='group',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def display_evaluation_export_options(evaluation_result: EvaluationResult, filename_prefix: str = "evaluation"):
    """
    Display export options for evaluation results.

    Args:
        evaluation_result: Evaluation results to export
        filename_prefix: Prefix for exported filenames
    """
    st.markdown(f"### {get_text('evaluation.export_title')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(get_text('evaluation.export_json')):
            _export_as_json(evaluation_result, filename_prefix)

    with col2:
        if st.button(get_text('evaluation.export_csv')):
            _export_as_csv(evaluation_result, filename_prefix)

    with col3:
        if st.button(get_text('evaluation.export_summary')):
            _export_summary_report(evaluation_result, filename_prefix)


def _export_as_json(evaluation_result: EvaluationResult, filename_prefix: str):
    """Export evaluation result as JSON file."""

    # Convert evaluation result to dictionary
    export_data = {
        'success': evaluation_result.success,
        'processing_time': evaluation_result.processing_time,
        'error': evaluation_result.error,
        'metrics': {
            'triple_match_f1': evaluation_result.metrics.triple_match_f1,
            'graph_match_accuracy': evaluation_result.metrics.graph_match_accuracy,
            'g_bleu_precision': evaluation_result.metrics.g_bleu_precision,
            'g_bleu_recall': evaluation_result.metrics.g_bleu_recall,
            'g_bleu_f1': evaluation_result.metrics.g_bleu_f1,
            'g_rouge_precision': evaluation_result.metrics.g_rouge_precision,
            'g_rouge_recall': evaluation_result.metrics.g_rouge_recall,
            'g_rouge_f1': evaluation_result.metrics.g_rouge_f1,
            'g_bert_precision': evaluation_result.metrics.g_bert_precision,
            'g_bert_recall': evaluation_result.metrics.g_bert_recall,
            'g_bert_f1': evaluation_result.metrics.g_bert_f1,
            'graph_edit_distance': evaluation_result.metrics.graph_edit_distance
        },
        'metadata': evaluation_result.metadata,
        'reference_graph_info': evaluation_result.reference_graph_info,
        'predicted_graph_info': evaluation_result.predicted_graph_info,
        'export_timestamp': datetime.now().isoformat()
    }

    # Create download link
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode('utf-8')).decode()
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{get_text("evaluation.download_json")}</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success(get_text('evaluation.export_success', format='JSON'))


def _export_as_csv(evaluation_result: EvaluationResult, filename_prefix: str):
    """Export evaluation metrics as CSV file."""

    # Create DataFrame with metrics
    metrics_data = {
        'Metric': [
            'Triple Match F1',
            'Graph Match Accuracy',
            'G-BLEU Precision',
            'G-BLEU Recall',
            'G-BLEU F1',
            'G-ROUGE Precision',
            'G-ROUGE Recall',
            'G-ROUGE F1',
            'G-BertScore Precision',
            'G-BertScore Recall',
            'G-BertScore F1',
            'Graph Edit Distance'
        ],
        'Value': [
            evaluation_result.metrics.triple_match_f1,
            evaluation_result.metrics.graph_match_accuracy,
            evaluation_result.metrics.g_bleu_precision,
            evaluation_result.metrics.g_bleu_recall,
            evaluation_result.metrics.g_bleu_f1,
            evaluation_result.metrics.g_rouge_precision,
            evaluation_result.metrics.g_rouge_recall,
            evaluation_result.metrics.g_rouge_f1,
            evaluation_result.metrics.g_bert_precision,
            evaluation_result.metrics.g_bert_recall,
            evaluation_result.metrics.g_bert_f1,
            evaluation_result.metrics.graph_edit_distance if evaluation_result.metrics.graph_edit_distance is not None else 'N/A'
        ]
    }

    df = pd.DataFrame(metrics_data)

    # Create download link
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    b64 = base64.b64encode(csv_str.encode('utf-8')).decode()
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">{get_text("evaluation.download_csv")}</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success(get_text('evaluation.export_success', format='CSV'))


def _export_summary_report(evaluation_result: EvaluationResult, filename_prefix: str):
    """Export evaluation summary report as text file."""

    # Create summary report
    report_lines = [
        f"Graph Quality Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"=== EVALUATION SUMMARY ===",
        f"Success: {'Yes' if evaluation_result.success else 'No'}",
        f"Processing Time: {evaluation_result.processing_time:.3f} seconds",
        f"",
        f"=== EXACT MATCHING METRICS ===",
        f"Triple Match F1: {evaluation_result.metrics.triple_match_f1:.4f}",
        f"Graph Match Accuracy: {evaluation_result.metrics.graph_match_accuracy:.4f}",
        f"",
        f"=== TEXT SIMILARITY METRICS ===",
        f"G-BLEU Precision: {evaluation_result.metrics.g_bleu_precision:.4f}",
        f"G-BLEU Recall: {evaluation_result.metrics.g_bleu_recall:.4f}",
        f"G-BLEU F1: {evaluation_result.metrics.g_bleu_f1:.4f}",
        f"",
        f"G-ROUGE Precision: {evaluation_result.metrics.g_rouge_precision:.4f}",
        f"G-ROUGE Recall: {evaluation_result.metrics.g_rouge_recall:.4f}",
        f"G-ROUGE F1: {evaluation_result.metrics.g_rouge_f1:.4f}",
        f"",
        f"=== SEMANTIC SIMILARITY METRICS ===",
        f"G-BertScore Precision: {evaluation_result.metrics.g_bert_precision:.4f}",
        f"G-BertScore Recall: {evaluation_result.metrics.g_bert_recall:.4f}",
        f"G-BertScore F1: {evaluation_result.metrics.g_bert_f1:.4f}",
        f""
    ]

    if evaluation_result.metrics.graph_edit_distance is not None:
        report_lines.extend([
            f"=== STRUCTURAL SIMILARITY METRICS ===",
            f"Graph Edit Distance: {evaluation_result.metrics.graph_edit_distance:.4f}",
            f""
        ])

    if evaluation_result.reference_graph_info:
        report_lines.extend([
            f"=== REFERENCE GRAPH INFO ===",
            f"Size: {evaluation_result.reference_graph_info.get('size', 'N/A')} triples",
            f"Unique Subjects: {evaluation_result.reference_graph_info.get('unique_subjects', 'N/A')}",
            f"Unique Predicates: {evaluation_result.reference_graph_info.get('unique_predicates', 'N/A')}",
            f"Unique Objects: {evaluation_result.reference_graph_info.get('unique_objects', 'N/A')}",
            f""
        ])

    if evaluation_result.predicted_graph_info:
        report_lines.extend([
            f"=== PREDICTED GRAPH INFO ===",
            f"Size: {evaluation_result.predicted_graph_info.get('size', 'N/A')} triples",
            f""
        ])

    if evaluation_result.error:
        report_lines.extend([
            f"=== ERROR INFORMATION ===",
            f"Error: {evaluation_result.error}",
            f""
        ])

    # Create download link
    report_text = '\n'.join(report_lines)
    b64 = base64.b64encode(report_text.encode('utf-8')).decode()
    filename = f"{filename_prefix}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{get_text("evaluation.download_summary")}</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success(get_text('evaluation.export_success', format='Summary'))


def _get_quality_indicator(score: float) -> str:
    """
    Get quality indicator for a metric score.

    Args:
        score: Metric score (0.0 to 1.0)

    Returns:
        Quality indicator string
    """
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Poor"


def display_evaluation_configuration():
    """Display evaluation configuration options in sidebar or main area."""
    st.markdown(f"### {get_text('evaluation.configuration_title')}")

    # Evaluation options
    enable_evaluation = st.checkbox(
        get_text('evaluation.enable_evaluation'),
        value=False,
        help=get_text('evaluation.enable_evaluation_help')
    )

    if enable_evaluation:
        # Advanced options
        with st.expander(get_text('evaluation.advanced_options'), expanded=False):
            enable_bert_score = st.checkbox(
                get_text('evaluation.enable_bert_score'),
                value=True,
                help=get_text('evaluation.bert_score_help')
            )

            enable_ged = st.checkbox(
                get_text('evaluation.enable_ged'),
                value=False,
                help=get_text('evaluation.ged_help')
            )

            max_eval_time = st.slider(
                get_text('evaluation.max_eval_time'),
                min_value=10.0,
                max_value=120.0,
                value=30.0,
                step=5.0,
                help=get_text('evaluation.max_eval_time_help')
            )

        return {
            'enable_evaluation': enable_evaluation,
            'enable_bert_score': enable_bert_score,
            'enable_ged': enable_ged,
            'max_evaluation_time': max_eval_time
        }

    return {'enable_evaluation': False}


def display_reference_graph_upload():
    """
    Display reference graph upload interface.

    Returns:
        Uploaded reference graph as list of Triple objects, or None
    """
    st.markdown(f"### {get_text('evaluation.reference_graph_title')}")

    # File upload
    uploaded_file = st.file_uploader(
        get_text('evaluation.upload_reference_graph'),
        type=['json', 'csv', 'txt'],
        help=get_text('evaluation.reference_graph_help')
    )

    if uploaded_file is not None:
        # Import reference graph manager
        try:
            from streamlit_pipeline.utils.reference_graph_manager import upload_reference_graph
        except ImportError:
            from ..utils.reference_graph_manager import upload_reference_graph

        # Process uploaded file
        success, triples, error = upload_reference_graph(uploaded_file)

        if success and triples:
            st.success(f"{get_text('evaluation.reference_graph_success')} {len(triples)} triples")

            # Display basic statistics
            with st.expander(get_text('evaluation.reference_graph_stats'), expanded=False):
                try:
                    from streamlit_pipeline.utils.reference_graph_manager import get_reference_graph_stats
                except ImportError:
                    from ..utils.reference_graph_manager import get_reference_graph_stats

                stats = get_reference_graph_stats(triples)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Triples", stats['size'])
                    st.metric("Unique Subjects", stats['unique_subjects'])

                with col2:
                    st.metric("Unique Predicates", stats['unique_predicates'])
                    st.metric("Unique Objects", stats['unique_objects'])

                # Display examples
                if stats.get('subject_examples'):
                    st.text("Subject Examples:")
                    st.text(", ".join(stats['subject_examples']))

                if stats.get('predicate_examples'):
                    st.text("Predicate Examples:")
                    st.text(", ".join(stats['predicate_examples']))

            return triples
        else:
            st.error(f"{get_text('evaluation.reference_graph_error')}: {error}")

    return None