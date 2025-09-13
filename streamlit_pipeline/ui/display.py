"""
Result Display Functions for GraphJudge Streamlit Application.

This module provides specialized display functions for different types of
pipeline results and data visualizations. It complements the components
module with focused display logic for complex data structures.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Optional plotly import with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

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
            
            display_final_knowledge_graph(approved_triples, pipeline_result.judgment_result)
            
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


def display_final_knowledge_graph(triples: List[Triple], judgment_result: JudgmentResult):
    """
    Display the final approved knowledge graph in an attractive format.

    Args:
        triples: List of approved triples
        judgment_result: Judgment results for confidence scores
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
    if len(triples) > 1:
        st.markdown("### 🕸️ Relationship Network Graph")
        if PLOTLY_AVAILABLE:
            create_enhanced_knowledge_graph(triples)
        else:
            st.info("📊 Network graph requires Plotly library: `pip install plotly`")
            st.text("Text-based relationship display:")
            for i, triple in enumerate(triples[:15], 1):
                st.text(f"{i}. {triple.subject} → {triple.predicate} → {triple.object}")


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


def create_enhanced_knowledge_graph(triples: List[Triple]):
    """
    Create an enhanced interactive knowledge graph visualization.
    
    Args:
        triples: List of triples to visualize
    """
    if not PLOTLY_AVAILABLE:
        st.error("Plotly库未安装，无法显示图形可视化")
        return
    
    try:
        # Extract entities and relationships
        entities = set()
        relationships = []
        
        for triple in triples:
            entities.add(triple.subject)
            entities.add(triple.object)
            relationships.append({
                'source': triple.subject,
                'target': triple.object,
                'relation': triple.predicate,
                'confidence': triple.confidence or 0.5
            })
        
        entities = list(entities)
        
        # Limit visualization size for performance
        if len(entities) > 15:
            st.warning(f"Too many entities ({len(entities)}), showing relationships for the first 15 entities")
            entities = entities[:15]
            relationships = [r for r in relationships 
                           if r['source'] in entities and r['target'] in entities]
        
        # Create network graph using Plotly
        # Position entities using a simple circular layout
        import math
        positions = {}
        n = len(entities)
        
        for i, entity in enumerate(entities):
            angle = 2 * math.pi * i / n
            radius = 3
            positions[entity] = {
                'x': radius * math.cos(angle),
                'y': radius * math.sin(angle)
            }
        
        # Create the visualization
        fig = go.Figure()
        
        # Add edges
        for rel in relationships:
            source_pos = positions.get(rel['source'])
            target_pos = positions.get(rel['target'])
            
            if source_pos and target_pos:
                # Draw edge
                fig.add_trace(go.Scatter(
                    x=[source_pos['x'], target_pos['x'], None],
                    y=[source_pos['y'], target_pos['y'], None],
                    mode='lines',
                    line=dict(
                        width=2 + rel['confidence'] * 3,  # Thickness based on confidence
                        color=f"rgba(100, 100, 100, {0.3 + rel['confidence'] * 0.7})"
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Add relationship label
                mid_x = (source_pos['x'] + target_pos['x']) / 2
                mid_y = (source_pos['y'] + target_pos['y']) / 2
                
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    text=rel['relation'],
                    showarrow=False,
                    font=dict(size=10, color='blue'),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="blue",
                    borderwidth=1,
                    borderpad=2
                )
        
        # Add nodes
        for entity in entities:
            pos = positions[entity]
            fig.add_trace(go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='markers+text',
                text=[entity],
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                marker=dict(
                    size=40,
                    color='lightblue',
                    line=dict(width=3, color='darkblue')
                ),
                hoverinfo='text',
                hovertext=entity,
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Knowledge Graph Network Visualization",
                'x': 0.5,
                'font': {'size': 16}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Nodes: Entities | Edges: Relations | Line thickness: AI confidence",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color='gray')
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Visualization generation failed: {str(e)}")
        st.info("You can still view the table format results above")


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
    
    # Quality Analysis
    if pipeline_result.judgment_result:
        high_quality = sum(1 for c in pipeline_result.judgment_result.confidence if c > 0.8)
        medium_quality = sum(1 for c in pipeline_result.judgment_result.confidence if 0.5 <= c <= 0.8)
        low_quality = sum(1 for c in pipeline_result.judgment_result.confidence if c < 0.5)
        
        report_sections.append(f"""
        ### ⚖️ Quality Analysis

        - **High Quality** (>0.8): {high_quality} items
        - **Medium Quality** (0.5-0.8): {medium_quality} items
        - **Needs Improvement** (<0.5): {low_quality} items
        - **Average Confidence**: {sum(pipeline_result.judgment_result.confidence)/len(pipeline_result.judgment_result.confidence):.3f}
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