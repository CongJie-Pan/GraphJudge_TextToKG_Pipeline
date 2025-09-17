"""
Result Display Functions for GraphJudge Streamlit Application.

This module provides specialized display functions for different types of
pipeline results and data visualizations. It complements the components
module with focused display logic for complex data structures.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

# Optional visualization library imports with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None
    components = None

from ..core.models import EntityResult, TripleResult, JudgmentResult, Triple
from ..core.pipeline import PipelineResult


def display_final_results(pipeline_result: PipelineResult):
    """
    Display the final consolidated results from all pipeline stages.

    Args:
        pipeline_result: Complete pipeline execution results
    """
    st.markdown("# 🏆 Final Results")
    
    if not pipeline_result.success:
        st.error(f"Pipeline failed at {pipeline_result.error_stage}: {pipeline_result.error}")
        return
    
    # Get the approved triples
    if (pipeline_result.judgment_result and 
        pipeline_result.triple_result and 
        pipeline_result.judgment_result.judgments):
        
        approved_triples = [
            triple for triple, approved in zip(
                pipeline_result.triple_result.triples,
                pipeline_result.judgment_result.judgments
            ) if approved
        ]
        
        rejected_triples = [
            triple for triple, approved in zip(
                pipeline_result.triple_result.triples,
                pipeline_result.judgment_result.judgments
            ) if not approved
        ]
        
        # Summary metrics
        total_triples = len(pipeline_result.triple_result.triples)
        approved_count = len(approved_triples)
        rejection_rate = (total_triples - approved_count) / total_triples if total_triples > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "✅ Approved Triples",
                approved_count,
                delta=f"{(1-rejection_rate)*100:.1f}% approval rate"
            )

        with col2:
            st.metric(
                "❌ Rejected Triples",
                total_triples - approved_count,
                delta=f"{rejection_rate*100:.1f}% rejection rate"
            )

        with col3:
            avg_confidence = (
                sum(pipeline_result.judgment_result.confidence) /
                len(pipeline_result.judgment_result.confidence)
                if pipeline_result.judgment_result.confidence else 0
            )
            st.metric(
                "🎯 Average Confidence",
                f"{avg_confidence:.3f}"
            )

        with col4:
            st.metric(
                "⏱️ Total Processing Time",
                f"{pipeline_result.total_time:.1f}s"
            )
        
        # Display approved triples as the final knowledge graph
        if approved_triples:
            st.markdown("## 🧠 Final Knowledge Graph")
            st.markdown(f"After AI judgment, the following **{len(approved_triples)}** knowledge triples were deemed accurate:")
            
            # Store pipeline result in session state for graph visualization
            st.session_state.pipeline_result = pipeline_result
            display_final_knowledge_graph(approved_triples, pipeline_result.judgment_result, pipeline_result.graph_data)
            
            # Export options
            st.markdown("### 📤 Export Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📄 Export as JSON", key="export_final_json"):
                    export_final_results_json(approved_triples, pipeline_result)

            with col2:
                if st.button("📊 Export as CSV", key="export_final_csv"):
                    export_final_results_csv(approved_triples, pipeline_result)

            with col3:
                if st.button("📋 Generate Report", key="generate_report"):
                    display_analysis_report(pipeline_result)
        
        else:
            st.warning("⚠️ No triples passed AI judgment. You may need to adjust the input text or check the processing logic.")

            # Show rejected triples for reference
            if rejected_triples:
                with st.expander("🔍 View Rejected Triples"):
                    display_rejected_triples_analysis(rejected_triples, pipeline_result.judgment_result)


def display_final_knowledge_graph(triples: List[Triple], judgment_result: JudgmentResult, graph_data: Optional[Dict[str, Any]] = None):
    """
    Display the final approved knowledge graph in an attractive format.

    Args:
        triples: List of approved triples
        judgment_result: Judgment results for confidence scores
        graph_data: Pre-processed graph data from pipeline conversion
    """
    # Create a beautiful table format
    final_data = []
    for i, triple in enumerate(triples):
        confidence_idx = None
        # Find the original index of this triple for confidence
        if judgment_result.confidence:
            confidence_idx = i
        
        confidence = (judgment_result.confidence[confidence_idx] 
                     if confidence_idx is not None and confidence_idx < len(judgment_result.confidence)
                     else 0.0)
        
        # Create a formatted entry
        final_data.append({
            "#": i + 1,
            "Knowledge Triple": f"【{triple.subject}】 → {triple.predicate} → 【{triple.object}】",
            "Subject": triple.subject,
            "Relation": triple.predicate,
            "Object": triple.object,
            "AI Confidence": f"{confidence:.3f}" if confidence > 0 else "N/A",
            "Quality Grade": get_quality_grade(confidence) if confidence > 0 else "Not Rated"
        })
    
    df = pd.DataFrame(final_data)
    
    # Display with custom styling
    st.markdown("### 📋 Knowledge Triple Details")
    
    # Interactive data table with selection
    selected_indices = st.dataframe(
        df[["#", "Knowledge Triple", "AI Confidence", "Quality Grade"]],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "Knowledge Triple": st.column_config.TextColumn(
                "Knowledge Triple",
                help="Click to view detailed information",
                width="large"
            ),
            "AI Confidence": st.column_config.ProgressColumn(
                "AI Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.3f"
            ),
            "Quality Grade": st.column_config.TextColumn(
                "Quality Grade",
                help="Quality rating based on confidence"
            )
        }
    )
    
    # Show knowledge graph visualization
    if len(triples) > 0:
        st.markdown("### 🕸️ Interactive Knowledge Graph")

        # Check if we have pipeline result with pyvis data
        if hasattr(st.session_state, 'pipeline_result') and hasattr(st.session_state.pipeline_result, 'pyvis_data'):
            pyvis_data = st.session_state.pipeline_result.pyvis_data
        else:
            pyvis_data = None

        # Use Pyvis viewer as primary option
        if pyvis_data and PYVIS_AVAILABLE:
            display_pyvis_knowledge_graph(pyvis_data, triples)
        else:
            # Fallback to Plotly or text display
            st.info("💡 Using fallback visualization (Pyvis data not available)")
            create_enhanced_knowledge_graph(triples, graph_data)


def display_rejected_triples_analysis(rejected_triples: List[Triple], judgment_result: JudgmentResult):
    """
    Display analysis of rejected triples to help users understand the filtering.

    Args:
        rejected_triples: List of rejected triples
        judgment_result: Judgment results with explanations
    """
    st.markdown("#### Rejected Triples Analysis")
    
    rejection_data = []
    explanation_idx = 0
    
    for triple in rejected_triples:
        # Find explanation if available
        explanation = None
        if (judgment_result.explanations and 
            explanation_idx < len(judgment_result.explanations)):
            explanation = judgment_result.explanations[explanation_idx]
        
        rejection_data.append({
            "Triple": f"{triple.subject} - {triple.predicate} - {triple.object}",
            "Possible Reason": explanation or "AI judged this relationship as insufficiently accurate or relevant",
            "Suggestion": get_rejection_suggestion(triple, explanation)
        })
        explanation_idx += 1
    
    df = pd.DataFrame(rejection_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def create_enhanced_knowledge_graph(triples: List[Triple], graph_data: Optional[Dict[str, Any]] = None):
    """
    Create an enhanced interactive knowledge graph visualization.

    Args:
        triples: List of triples to visualize (fallback if no graph_data)
        graph_data: Pre-processed graph data from pipeline with nodes and edges
    """
    if not PLOTLY_AVAILABLE:
        st.error("📊 Interactive graph visualization requires Plotly library")
        st.info("💡 Install with: `pip install plotly>=5.0.0`")
        _display_text_based_graph(triples)
        return

    try:
        # Use processed graph data if available, otherwise create from triples
        if graph_data and graph_data.get("nodes") and graph_data.get("edges"):
            nodes = graph_data["nodes"]
            edges = graph_data["edges"]

            st.success(f"🎨 Visualizing knowledge graph: {len(nodes)} entities, {len(edges)} relationships")

            # Display summary metrics from graph data
            if "report" in graph_data and "summary" in graph_data["report"]:
                summary = graph_data["report"]["summary"]
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Entities", summary.get("entities", len(nodes)))
                with col2:
                    st.metric("Relationships", summary.get("relationships", len(edges)))
                with col3:
                    if "approved_triples" in summary:
                        st.metric("Approved", summary["approved_triples"])
                with col4:
                    if "average_confidence" in summary:
                        st.metric("Avg Confidence", f"{summary['average_confidence']:.3f}")

            # Create visualization using pre-processed data
            fig = _create_plotly_graph_from_data(nodes, edges)

        else:
            # Fallback: create graph data from triples
            st.info("🔄 Creating graph from triples (graph data not available)")
            nodes, edges = _convert_triples_to_graph_data(triples)
            fig = _create_plotly_graph_from_data(nodes, edges)

        # Display the interactive graph
        st.plotly_chart(fig, use_container_width=True)

        # Add export options
        _display_graph_export_options(graph_data or {"nodes": nodes, "edges": edges})

    except Exception as e:
        st.error(f"❌ Graph visualization failed: {str(e)}")
        st.info("📋 Displaying text-based relationship view instead:")
        _display_text_based_graph(triples)


def _create_plotly_graph_from_data(nodes: List[Dict], edges: List[Dict]):
    """Create a Plotly figure from nodes and edges data."""
    import math

    # Limit visualization size for performance
    if len(nodes) > 20:
        st.warning(f"⚠️ Large graph detected ({len(nodes)} entities). Showing first 20 for performance.")
        nodes = nodes[:20]
        node_ids = {node["id"] for node in nodes}
        edges = [edge for edge in edges
                if edge["source"] in node_ids and edge["target"] in node_ids]

    # Create circular layout for nodes
    positions = {}
    n = len(nodes)

    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n if n > 1 else 0
        radius = 3
        positions[node["id"]] = {
            'x': radius * math.cos(angle),
            'y': radius * math.sin(angle)
        }

    # Create Plotly figure
    fig = go.Figure()

    # Add edges (relationships)
    for edge in edges:
        source_pos = positions.get(edge["source"])
        target_pos = positions.get(edge["target"])

        if source_pos and target_pos:
            # Draw edge line
            edge_width = edge.get("width", 2)
            edge_color = edge.get("color", "rgba(100, 100, 100, 0.6)")

            fig.add_trace(go.Scatter(
                x=[source_pos['x'], target_pos['x'], None],
                y=[source_pos['y'], target_pos['y'], None],
                mode='lines',
                line=dict(
                    width=edge_width,
                    color=edge_color
                ),
                hoverinfo='none',
                showlegend=False,
                name='edge'
            ))

            # Add relationship label
            mid_x = (source_pos['x'] + target_pos['x']) / 2
            mid_y = (source_pos['y'] + target_pos['y']) / 2

            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=edge["label"],
                showarrow=False,
                font=dict(size=9, color='darkblue'),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightblue",
                borderwidth=1,
                borderpad=2
            )

    # Add nodes (entities)
    for node in nodes:
        pos = positions[node["id"]]
        node_size = node.get("size", 30)
        node_color = node.get("color", "#4ECDC4")

        fig.add_trace(go.Scatter(
            x=[pos['x']],
            y=[pos['y']],
            mode='markers+text',
            text=[node["label"]],
            textposition='middle center',
            textfont=dict(size=min(10, max(8, node_size // 4)), color='white', family="Arial Black"),
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            hoverinfo='text',
            hovertext=f"Entity: {node['label']}<br>Connections: {node.get('size', 0) // 2}",
            showlegend=False,
            name='node'
        ))

    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': "🕸️ Knowledge Graph Network Visualization",
            'x': 0.5,
            'font': {'size': 18, 'color': 'darkblue'}
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40, l=20, r=20, t=50),
        annotations=[dict(
            text="💡 Hover over nodes and edges for details | Node size = relationship count | Edge thickness = confidence",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            xanchor='center', yanchor='bottom',
            font=dict(size=11, color='gray')
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        height=600
    )

    return fig


def _convert_triples_to_graph_data(triples: List[Triple]) -> Tuple[List[Dict], List[Dict]]:
    """Convert triples to nodes and edges data structure."""
    from collections import defaultdict

    entity_connections = defaultdict(int)
    edges = []

    # Process triples to create edges and count entity connections
    for triple in triples:
        entity_connections[triple.subject] += 1
        entity_connections[triple.object] += 1

        edges.append({
            "source": triple.subject,
            "target": triple.object,
            "label": triple.predicate,
            "weight": triple.confidence or 0.5,
            "width": max(2, int((triple.confidence or 0.5) * 5)),
            "color": _get_confidence_color(triple.confidence or 0.5)
        })

    # Create nodes
    nodes = []
    for entity, connections in entity_connections.items():
        nodes.append({
            "id": entity,
            "label": entity,
            "size": min(15 + connections * 3, 50),
            "color": _get_node_color_by_connections(connections)
        })

    return nodes, edges


def _get_confidence_color(confidence: float) -> str:
    """Get edge color based on confidence score."""
    if confidence >= 0.8:
        return "rgba(46, 204, 64, 0.7)"  # Green for high confidence
    elif confidence >= 0.6:
        return "rgba(255, 133, 27, 0.7)"  # Orange for medium confidence
    else:
        return "rgba(170, 170, 170, 0.7)"  # Gray for low confidence


def _get_node_color_by_connections(connections: int) -> str:
    """Get node color based on connection count."""
    if connections >= 5:
        return "#FF6B6B"  # Red for highly connected
    elif connections >= 3:
        return "#4ECDC4"  # Teal for moderately connected
    else:
        return "#45B7D1"  # Blue for less connected


def _display_text_based_graph(triples: List[Triple]):
    """Display a text-based representation of the graph when Plotly is not available."""
    if not triples:
        st.info("No relationships to display")
        return

    st.markdown("**📋 Text-based relationship display:**")

    # Group relationships by predicate for better organization
    from collections import defaultdict
    grouped_relations = defaultdict(list)

    for triple in triples:
        grouped_relations[triple.predicate].append((triple.subject, triple.object))

    # Display grouped relationships
    for predicate, relations in grouped_relations.items():
        with st.expander(f"🔗 {predicate} ({len(relations)} relationships)"):
            for i, (subject, obj) in enumerate(relations[:20], 1):  # Limit display
                confidence = ""
                if hasattr(triples[i-1], 'confidence') and triples[i-1].confidence:
                    confidence = f" (confidence: {triples[i-1].confidence:.3f})"
                st.text(f"{i}. {subject} → {predicate} → {obj}{confidence}")

            if len(relations) > 20:
                st.info(f"... and {len(relations) - 20} more relationships")


def _display_graph_export_options(graph_data: Dict[str, Any]):
    """Display export options for the graph data."""
    if not graph_data:
        return

    with st.expander("📤 Export Graph Data"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Download as JSON"):
                import json
                json_str = json.dumps(graph_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 Download JSON",
                    data=json_str,
                    file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📊 Show Raw Data"):
                st.json(graph_data)


def export_final_results_json(triples: List[Triple], pipeline_result: PipelineResult):
    """Export final results as JSON format."""
    export_data = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "total_processing_time": pipeline_result.total_time,
            "pipeline_success": pipeline_result.success,
            "total_triples_generated": len(pipeline_result.triple_result.triples) if pipeline_result.triple_result else 0,
            "approved_triples_count": len(triples),
            "approval_rate": len(triples) / len(pipeline_result.triple_result.triples) if pipeline_result.triple_result and pipeline_result.triple_result.triples else 0
        },
        "knowledge_graph": [
            {
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "confidence": triple.confidence or 0.0
            }
            for triple in triples
        ],
        "processing_stats": pipeline_result.stats or {}
    }
    
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📁 Download JSON File",
        data=json_str,
        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def export_final_results_csv(triples: List[Triple], pipeline_result: PipelineResult):
    """Export final results as CSV format."""
    csv_data = []
    
    for i, triple in enumerate(triples):
        csv_data.append({
            "#": i + 1,
            "Subject": triple.subject,
            "Predicate": triple.predicate,
            "Object": triple.object,
            "Confidence": triple.confidence or 0.0,
            "Quality Grade": get_quality_grade(triple.confidence or 0.0)
        })
    
    df = pd.DataFrame(csv_data)
    csv_string = df.to_csv(index=False, encoding='utf-8')
    
    st.download_button(
        label="📊 Download CSV File",
        data=csv_string,
        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def display_analysis_report(pipeline_result: PipelineResult):
    """Display a comprehensive analysis report."""
    st.markdown("## 📊 Analysis Report")
    
    # Generate report content
    report_sections = []
    
    # Executive Summary
    if pipeline_result.success:
        approved_count = sum(pipeline_result.judgment_result.judgments) if pipeline_result.judgment_result else 0
        total_count = len(pipeline_result.triple_result.triples) if pipeline_result.triple_result else 0
        approval_rate = approved_count / total_count if total_count > 0 else 0
        
        report_sections.append(f"""
        ### 📋 Executive Summary

        - **Overall Status**: ✅ Successfully completed
        - **Processing Time**: {pipeline_result.total_time:.2f} seconds
        - **Knowledge Extraction**: Successfully extracted {approved_count} high-quality knowledge triples from input text
        - **Quality Rating**: {approval_rate:.1%} of generated triples passed AI quality checks
        """)
    
    # Stage Analysis
    if pipeline_result.entity_result:
        entities_count = len(pipeline_result.entity_result.entities)
        report_sections.append(f"""
        ### 🔍 Entity Extraction Analysis

        - **Entity Count**: {entities_count} entities
        - **Processing Time**: {pipeline_result.entity_result.processing_time:.2f} seconds
        - **Efficiency**: {entities_count/pipeline_result.entity_result.processing_time:.1f} entities/second
        """)
    
    if pipeline_result.triple_result:
        triples_count = len(pipeline_result.triple_result.triples)
        report_sections.append(f"""
        ### 🔗 Triple Generation Analysis

        - **Generated Count**: {triples_count} triples
        - **Processing Time**: {pipeline_result.triple_result.processing_time:.2f} seconds
        - **Generation Efficiency**: {triples_count/pipeline_result.triple_result.processing_time:.1f} triples/second
        """)
    
    
    # Display report
    for section in report_sections:
        st.markdown(section)
    
    # Recommendations
    st.markdown("""
    ### 💡 Recommendations

    1. **High Quality Results**: Triples with confidence >0.8 can be used directly
    2. **Manual Review**: Recommend manual review for results with confidence 0.5-0.8
    3. **Result Optimization**: For more high-quality results, try adjusting the expression of input text
    """)


def get_quality_grade(confidence: float) -> str:
    """Convert confidence score to quality grade."""
    if confidence >= 0.9:
        return "🏆 Excellent"
    elif confidence >= 0.8:
        return "🥇 Good"
    elif confidence >= 0.6:
        return "🥈 Average"
    elif confidence >= 0.4:
        return "🥉 Fair"
    else:
        return "⚠️ Needs Improvement"


def get_rejection_suggestion(triple: Triple, explanation: Optional[str]) -> str:
    """Generate suggestion for rejected triples."""
    if explanation and "inaccurate" in explanation.lower():
        return "Check if the relationship between subject and object is correctly expressed"
    elif explanation and "irrelevant" in explanation.lower():
        return "Confirm whether this relationship is relevant to the topic"
    elif explanation and "vague" in explanation.lower():
        return "Try using more explicit expressions"
    else:
        return "Re-examine the expression or context of this relationship"


def display_comparison_view(current_result: PipelineResult, previous_results: List[PipelineResult]):
    """
    Display comparison between current and previous results.

    Args:
        current_result: Current pipeline result
        previous_results: List of previous results for comparison
    """
    if not previous_results:
        return

    st.markdown("## 📈 Historical Comparison")
    
    # Create comparison metrics
    comparison_data = []
    for i, result in enumerate([current_result] + previous_results[:4]):  # Current + last 4
        if result.success and result.stats:
            comparison_data.append({
                "Run": "Current" if i == 0 else f"History-{i}",
                "Total Time": result.total_time,
                "Entity Count": result.stats.get('entity_count', 0),
                "Triple Count": result.stats.get('triple_count', 0),
                "Approved Count": result.stats.get('approved_triples', 0),
                "Approval Rate": result.stats.get('approval_rate', 0)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Performance trends
        if len(comparison_data) > 1 and PLOTLY_AVAILABLE:
            fig = px.line(
                df,
                x="Run",
                y=["Total Time", "Approval Rate"],
                title="Performance Trends",
                labels={"value": "Value", "variable": "Metric"}
            )
            st.plotly_chart(fig, use_container_width=True)
        elif len(comparison_data) > 1:
            st.info("📊 Trend chart requires Plotly library: `pip install plotly`")


def create_pyvis_knowledge_graph(pyvis_data: Dict[str, Any], height: int = 600) -> Optional[str]:
    """
    Create an interactive Pyvis network visualization from pyvis_data.

    Args:
        pyvis_data: Processed Pyvis format data with nodes and edges
        height: Height of the visualization in pixels

    Returns:
        HTML string of the visualization, or None if failed
    """
    if not PYVIS_AVAILABLE:
        st.error("🌐 Pyvis network visualization requires Pyvis library")
        st.info("💡 Install with: `pip install pyvis>=0.3.2`")
        return None

    try:
        # Create Network instance
        net = Network(
            height=f"{height}px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            select_menu=True,
            filter_menu=True,
            directed=True,
            cdn_resources="in_line"
        )

        # Configure physics for better layout
        physics_options = pyvis_data.get("physics", {
            "enabled": True,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09
            }
        })
        net.set_options(f"""
        var options = {{
          "physics": {json.dumps(physics_options)},
          "interaction": {{
            "hover": true,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": false,
            "hideNodesOnDrag": false
          }},
          "layout": {{
            "improvedLayout": true
          }}
        }}
        """)

        # Add nodes
        nodes = pyvis_data.get("nodes", [])
        for node in nodes:
            net.add_node(
                str(node["id"]),
                label=node.get("label", str(node["id"])) ,
                color=node.get("color", "#1f77b4"),
                size=node.get("size", 25),
                title=node.get("title", node.get("label", str(node["id"]))),
                font=node.get("font", {"size": 12})
            )

        # Add edges
        edges = pyvis_data.get("edges", [])
        for edge in edges:
            net.add_edge(
                str(edge.get("from")),
                str(edge.get("to")),
                label=edge.get("label", ""),
                color=edge.get("color", "#848484"),
                width=edge.get("width", 1),
                title=edge.get("title", edge.get("label", "")),
                arrows=edge.get("arrows", "to")
            )

        # Generate HTML
        html = net.generate_html()

        return html

    except Exception as e:
        st.error(f"❌ Pyvis visualization failed: {str(e)}")
        return None


def display_pyvis_knowledge_graph(pyvis_data: Optional[Dict[str, Any]],
                                 triples: Optional[List[Triple]] = None,
                                 height: int = 600):
    """
    Display interactive Pyvis knowledge graph with fallback options.

    Args:
        pyvis_data: Pre-processed Pyvis graph data
        triples: Fallback triples for text display
        height: Height of visualization in pixels
    """
    if not pyvis_data:
        st.warning("🤷‍♂️ No graph data available for visualization")
        if triples:
            _display_text_based_graph(triples)
        return

    # Display metrics from Pyvis data
    metadata = pyvis_data.get("metadata", {})
    nodes_count = metadata.get("nodes_count", len(pyvis_data.get("nodes", [])))
    edges_count = metadata.get("edges_count", len(pyvis_data.get("edges", [])))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔹 Entities", nodes_count)
    with col2:
        st.metric("🔗 Relationships", edges_count)
    with col3:
        physics_status = "✅ Enabled" if metadata.get("physics_enabled", True) else "❌ Disabled"
        st.metric("⚡ Physics", physics_status)

    # Viewer selection
    viewer_option = st.selectbox(
        "🎨 Choose Visualization:",
        ["Interactive Network (Pyvis)", "Simple Text Display"],
        index=0
    )

    if viewer_option == "Interactive Network (Pyvis)":
        if PYVIS_AVAILABLE:
            # Generate Pyvis visualization
            html = create_pyvis_knowledge_graph(pyvis_data, height)
            if html:
                st.success(f"🌐 Interactive knowledge graph loaded: {nodes_count} entities, {edges_count} relationships")

                # Display in Streamlit
                components.html(html, height=height + 50, scrolling=True)

                # Display export information
                st.info("💡 **Interaction tips:** Drag nodes to rearrange, scroll to zoom, hover for details")

                # Add download option
                if st.button("📥 Download HTML"):
                    st.download_button(
                        label="Save Interactive Graph",
                        data=html,
                        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            else:
                st.error("Failed to generate Pyvis visualization")
                if triples:
                    _display_text_based_graph(triples)
        else:
            st.error("🚫 Pyvis library not available")
            if triples:
                _display_text_based_graph(triples)
    else:
        # Text-based display
        if triples:
            _display_text_based_graph(triples)
        else:
            # Convert from pyvis data to basic text display
            entities = set()
            relationships = []

            for edge in pyvis_data.get("edges", []):
                entities.add(edge["from"])
                entities.add(edge["to"])
                rel_text = f"{edge['from']} → {edge.get('label', 'related to')} → {edge['to']}"
                relationships.append(rel_text)

            st.markdown("### 📝 Text-Based Knowledge Graph")
            st.write(f"**Entities:** {', '.join(sorted(entities))}")
            st.markdown("**Relationships:**")
            for i, rel in enumerate(relationships, 1):
                st.write(f"{i}. {rel}")
