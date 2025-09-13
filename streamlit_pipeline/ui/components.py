"""
Reusable UI Components for GraphJudge Streamlit Application.

This module provides reusable Streamlit components for displaying pipeline
results, progress indicators, and interactive elements. Components follow
the design patterns from spec.md Section 5 and integrate with the error
handling system.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

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
from .error_display import display_success_message, display_processing_stats


def validate_text_file(uploaded_file) -> tuple[str, str, bool]:
    """
    Validate and read uploaded text file with proper encoding detection.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        tuple: (file_content, encoding_used, success_flag)
    """
    if uploaded_file is None:
        return "", "", False

    # Check file size (limit to 10MB for safety)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        st.error(f"❌ File too large. Maximum size allowed: {max_size / (1024*1024):.1f}MB")
        return "", "", False

    # Check if file is empty
    if uploaded_file.size == 0:
        st.error("❌ File is empty. Please upload a file with content.")
        return "", "", False

    # Try multiple encodings for Chinese text files
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'big5', 'utf-16', 'utf-16le', 'utf-16be']

    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            file_content = uploaded_file.read().decode(encoding)

            # Basic validation: check if content has Chinese characters
            chinese_char_count = sum(1 for char in file_content if '\u4e00' <= char <= '\u9fff')
            total_chars = len(file_content.strip())

            if total_chars == 0:
                continue  # Try next encoding

            # If at least 10% of characters are Chinese, consider it valid
            if chinese_char_count / total_chars >= 0.1 or total_chars < 100:
                return file_content.strip(), encoding, True

        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            st.error(f"❌ Error reading file with {encoding}: {str(e)}")
            continue

    return "", "", False


def display_input_section() -> str:
    """
    Display the main input section for file upload or text entry.

    Returns:
        The input text from the user (either uploaded or typed)
    """
    st.markdown("## 📝 Input Text")
    st.markdown("Please upload a Chinese text file (.txt) or enter text directly:")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "✏️ Type Text"])

    input_text = ""

    with tab1:
        # File upload section
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt'],
            help="Upload a .txt file containing Chinese text for analysis",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Use the validation helper function
            file_content, used_encoding, success = validate_text_file(uploaded_file)

            if success:
                input_text = file_content
                st.success(f"✅ File uploaded successfully! (Encoding: {used_encoding})")

                # Show file info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("File Name", uploaded_file.name)
                with col2:
                    st.metric("File Size", f"{uploaded_file.size:,} bytes")
                with col3:
                    st.metric("Encoding Used", used_encoding)
                with col4:
                    # Count Chinese characters
                    chinese_chars = sum(1 for char in file_content if '\u4e00' <= char <= '\u9fff')
                    st.metric("Chinese Chars", f"{chinese_chars:,}")

                # Show preview of file content
                with st.expander("📋 File Content Preview"):
                    preview_text = input_text[:500] + "..." if len(input_text) > 500 else input_text
                    st.text_area(
                        "Preview",
                        value=preview_text,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed"
                    )

                    # Show additional file analysis
                    if len(input_text) > 500:
                        st.info(f"📄 Showing first 500 characters of {len(input_text):,} total characters")

            else:
                st.error("❌ Failed to read the file. Please ensure it's a valid Chinese text file with proper encoding.")

    with tab2:
        # Text area for manual input
        manual_text = st.text_area(
            "Text Input",
            height=200,
            placeholder="Please enter your Chinese text here. Example: 红楼梦是清代作家曹雪芹创作的章回体长篇小说...",
            help="Supports Chinese classical literature texts, model is optimized for Chinese",
            label_visibility="collapsed"
        )

        if manual_text:
            input_text = manual_text.strip()

    # Input statistics (show for both file upload and manual input)
    if input_text:
        st.markdown("### 📊 Text Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Character Count", len(input_text))
        with col2:
            st.metric("Line Count", len(input_text.split('\n')))
        with col3:
            st.metric("Paragraph Count", len([p for p in input_text.split('\n\n') if p.strip()]))
        with col4:
            # Estimate reading time (assuming 300 characters per minute for Chinese)
            reading_time = len(input_text) / 300
            st.metric("Est. Reading Time", f"{reading_time:.1f} min")

    return input_text


def display_entity_results(entity_result: EntityResult):
    """
    Display entity extraction results in a user-friendly format.

    Args:
        entity_result: The EntityResult to display
    """
    st.markdown("## 🔍 Entity Extraction Results")
    
    # Success indicator and timing
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Extraction Status", "✅ Success" if entity_result.success else "❌ Failed")
    with col2:
        st.metric("Processing Time", f"{entity_result.processing_time:.2f}s")
    with col3:
        st.metric("Entity Count", len(entity_result.entities))
    
    if entity_result.entities:
        # Display entities as tags
        st.markdown("### 🏷️ Extracted Entities")
        
        # Create entity tags with colors
        entity_html = ""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF"]
        
        for i, entity in enumerate(entity_result.entities):
            color = colors[i % len(colors)]
            entity_html += f'<span style="background-color: {color}; color: white; padding: 0.2rem 0.5rem; margin: 0.1rem; border-radius: 0.3rem; font-size: 0.9rem;">{entity}</span> '
        
        st.markdown(entity_html, unsafe_allow_html=True)
        
        # Show denoised text comparison if different
        if entity_result.denoised_text != st.session_state.get('original_input', ''):
            with st.expander("📋 Processed Text"):
                st.text_area(
                    "Denoised Text",
                    value=entity_result.denoised_text,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
    else:
        st.warning("⚠️ No entities found. Please check if the input text contains recognizable entities.")


def display_triple_results(triple_result: TripleResult, show_validation: bool = True):
    """
    Display triple generation results with interactive features.

    Args:
        triple_result: The TripleResult to display
        show_validation: Whether to show validation information
    """
    st.markdown("## 🔗 Knowledge Triple Generation Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Generation Status", "✅ Success" if triple_result.success else "❌ Failed")
    with col2:
        st.metric("Processing Time", f"{triple_result.processing_time:.2f}s")
    with col3:
        st.metric("Triple Count", len(triple_result.triples))
    with col4:
        # Calculate average confidence if available
        confidences = [t.confidence for t in triple_result.triples if t.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        st.metric("Average Confidence", f"{avg_confidence:.2f}" if avg_confidence > 0 else "N/A")
    
    if triple_result.triples:
        # Create DataFrame for display
        triple_data = []
        for i, triple in enumerate(triple_result.triples):
            triple_data.append({
                "#": i + 1,
                "Subject": triple.subject,
                "Predicate": triple.predicate,
                "Object": triple.object,
                "Confidence": f"{triple.confidence:.3f}" if triple.confidence else "N/A"
            })
        
        df = pd.DataFrame(triple_data)
        
        # Interactive table with selection
        st.markdown("### 📊 Generated Knowledge Triples")

        # Table display options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Click rows to view detailed information:")
        with col2:
            export_format = st.selectbox("Export Format", ["JSON", "CSV"], key="triple_export")
        
        # Display the table
        selected_rows = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
        
        # Export functionality
        if st.button("📥 Export Triples", key="export_triples"):
            if export_format == "JSON":
                export_data = [asdict(triple) for triple in triple_result.triples]
                st.download_button(
                    "Download JSON File",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"triples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                csv_data = df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    "Download CSV File",
                    data=csv_data,
                    file_name=f"triples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Show validation information
        if show_validation and triple_result.metadata:
            with st.expander("🔍 Quality Analysis"):
                display_triple_quality_analysis(triple_result)
        
        # Visualization
        if len(triple_result.triples) > 1:
            with st.expander("📈 Relationship Visualization"):
                if PLOTLY_AVAILABLE:
                    display_knowledge_graph_viz(triple_result.triples)
                else:
                    st.info("📊 Visualization requires Plotly library: `pip install plotly`")
                    st.text("Text-based relationship display:")
                    for i, triple in enumerate(triple_result.triples[:10], 1):
                        st.text(f"{i}. {triple.subject} → {triple.predicate} → {triple.object}")
    else:
        st.warning("⚠️ No triples generated. Please check the entity extraction results.")


def display_judgment_results(judgment_result: JudgmentResult, triples: List[Triple]):
    """
    Display graph judgment results with approval/rejection analysis.
    
    Args:
        judgment_result: The JudgmentResult to display
        triples: The original triples that were judged
    """
    st.markdown("## ⚖️ 图判断结果 (Graph Judgment Results)")
    
    # Summary metrics
    approved = sum(1 for j in judgment_result.judgments if j)
    rejected = len(judgment_result.judgments) - approved
    approval_rate = approved / len(judgment_result.judgments) if judgment_result.judgments else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("判断状态", "✅ 完成" if judgment_result.success else "❌ 失败")
    with col2:
        st.metric("处理时间", f"{judgment_result.processing_time:.2f}s")
    with col3:
        st.metric("通过数量", approved, delta=f"{approval_rate:.1%}")
    with col4:
        st.metric("拒绝数量", rejected)
    with col5:
        avg_confidence = sum(judgment_result.confidence) / len(judgment_result.confidence) if judgment_result.confidence else 0
        st.metric("平均置信度", f"{avg_confidence:.3f}")
    
    if judgment_result.judgments:
        # Create combined results
        results_data = []
        for i, (triple, judgment, confidence) in enumerate(zip(triples, judgment_result.judgments, judgment_result.confidence or [0] * len(triples))):
            status_emoji = "✅" if judgment else "❌"
            status_text = "通过" if judgment else "拒绝"
            
            results_data.append({
                "序号": i + 1,
                "状态": f"{status_emoji} {status_text}",
                "主语": triple.subject,
                "谓语": triple.predicate,
                "宾语": triple.object,
                "置信度": f"{confidence:.3f}" if confidence > 0 else "N/A"
            })
        
        df = pd.DataFrame(results_data)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_option = st.selectbox(
                "显示筛选",
                ["全部", "仅通过", "仅拒绝"],
                key="judgment_filter"
            )
        with col2:
            sort_by = st.selectbox(
                "排序方式",
                ["序号", "置信度", "状态"],
                key="judgment_sort"
            )
        
        # Apply filters
        filtered_df = df.copy()
        if filter_option == "仅通过":
            filtered_df = filtered_df[filtered_df['状态'].str.contains("通过")]
        elif filter_option == "仅拒绝":
            filtered_df = filtered_df[filtered_df['状态'].str.contains("拒绝")]
        
        # Apply sorting
        if sort_by == "置信度":
            # Convert confidence to numeric for sorting
            filtered_df['置信度_数值'] = filtered_df['置信度'].apply(
                lambda x: float(x) if x != "N/A" else 0
            )
            filtered_df = filtered_df.sort_values('置信度_数值', ascending=False)
            filtered_df = filtered_df.drop('置信度_数值', axis=1)
        
        st.markdown(f"### 📋 判断结果详情 (共 {len(filtered_df)} 条)")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Explanations if available
        if judgment_result.explanations and any(judgment_result.explanations):
            with st.expander("💭 判断理由 (Judgment Explanations)"):
                for i, (triple, explanation) in enumerate(zip(triples, judgment_result.explanations)):
                    if explanation:
                        judgment_status = "✅ 通过" if judgment_result.judgments[i] else "❌ 拒绝"
                        st.markdown(f"**{i+1}. {triple.subject} - {triple.predicate} - {triple.object}** ({judgment_status})")
                        st.markdown(f"> {explanation}")
                        st.markdown("---")
        
        # Visualization
        if len(judgment_result.judgments) > 1:
            if PLOTLY_AVAILABLE:
                display_judgment_analysis(judgment_result, triples)
            else:
                st.info("📊 Chart analysis requires Plotly library: `pip install plotly`")


def display_triple_quality_analysis(triple_result: TripleResult):
    """
    Display quality analysis for generated triples.

    Args:
        triple_result: The TripleResult containing metadata
    """
    metadata = triple_result.metadata

    if not metadata:
        st.info("No quality analysis data available")
        return
    
    # Quality metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Structural Quality**")
        if 'avg_subject_length' in metadata:
            st.metric("Average Subject Length", f"{metadata['avg_subject_length']:.1f}")
        if 'avg_predicate_length' in metadata:
            st.metric("Average Predicate Length", f"{metadata['avg_predicate_length']:.1f}")

    with col2:
        st.markdown("**Content Quality**")
        if 'unique_subjects' in metadata:
            st.metric("Unique Subjects", metadata['unique_subjects'])
        if 'unique_predicates' in metadata:
            st.metric("Unique Predicates", metadata['unique_predicates'])

    with col3:
        st.markdown("**Processing Info**")
        if 'chunks_processed' in metadata:
            st.metric("Chunks Processed", metadata['chunks_processed'])
        if 'validation_score' in metadata:
            st.metric("Validation Score", f"{metadata['validation_score']:.2f}")


def display_knowledge_graph_viz(triples: List[Triple]):
    """
    Create an interactive knowledge graph visualization.
    
    Args:
        triples: List of triples to visualize
    """
    if not PLOTLY_AVAILABLE:
        st.error("Plotly库未安装，无法显示图形可视化")
        return
    
    # Extract unique nodes
    nodes = set()
    edges = []
    
    for triple in triples:
        nodes.add(triple.subject)
        nodes.add(triple.object)
        edges.append((triple.subject, triple.object, triple.predicate))
    
    if len(nodes) > 20:
        st.warning("⚠️ Too many nodes, showing only the first 20 triples in the relationship graph")
        edges = edges[:20]
    
    # Create network visualization using Plotly
    # This is a simplified version - in production you might want to use NetworkX + Plotly
    node_list = list(nodes)[:20]  # Limit nodes for better visualization
    
    # Create a simple force-directed layout simulation
    import math
    import random
    
    # Position nodes in a circle for simplicity
    positions = {}
    angle_step = 2 * math.pi / len(node_list)
    radius = 2
    
    for i, node in enumerate(node_list):
        angle = i * angle_step
        positions[node] = (
            radius * math.cos(angle) + random.uniform(-0.2, 0.2),
            radius * math.sin(angle) + random.uniform(-0.2, 0.2)
        )
    
    # Create edges for the plot
    edge_x = []
    edge_y = []
    edge_text = []
    
    for subject, obj, predicate in edges[:15]:  # Limit edges
        if subject in positions and obj in positions:
            x0, y0 = positions[subject]
            x1, y1 = positions[obj]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Add predicate as midpoint text
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            edge_text.append((mid_x, mid_y, predicate))
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add edge labels
    for x, y, text in edge_text:
        fig.add_annotation(
            x=x, y=y,
            text=text,
            showarrow=False,
            font=dict(size=8, color='blue'),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1
        )
    
    # Add nodes
    node_x = [positions[node][0] for node in node_list if node in positions]
    node_y = [positions[node][1] for node in node_list if node in positions]
    node_names = [node for node in node_list if node in positions]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_names,
        textposition="middle center",
        hoverinfo='text',
        marker=dict(
            size=30,
            color='lightblue',
            line=dict(width=2, color='blue')
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Knowledge Graph Visualization",
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Nodes represent entities, edges represent relationships",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(size=12, color='gray')
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_judgment_analysis(judgment_result: JudgmentResult, triples: List[Triple]):
    """
    Display detailed analysis of judgment results.
    
    Args:
        judgment_result: The judgment results
        triples: The original triples
    """
    if not PLOTLY_AVAILABLE:
        st.warning("Chart analysis requires Plotly library support")
        return

    st.markdown("### 📊 Judgment Results Analysis")
    
    # Create pie chart for approval/rejection
    approved = sum(judgment_result.judgments)
    rejected = len(judgment_result.judgments) - approved
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=[approved, rejected],
            names=['Approved', 'Rejected'],
            title="Approval Rate Analysis",
            color_discrete_map={'Approved': '#00CC96', 'Rejected': '#EF553B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if judgment_result.confidence and any(c > 0 for c in judgment_result.confidence):
            fig_hist = px.histogram(
                x=judgment_result.confidence,
                title="Confidence Distribution",
                labels={'x': 'Confidence', 'y': 'Count'},
                nbins=10
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No confidence distribution data available")


def display_pipeline_summary(pipeline_result: PipelineResult):
    """
    Display a comprehensive summary of the entire pipeline run.

    Args:
        pipeline_result: The complete pipeline result
    """
    st.markdown("## 📈 Pipeline Execution Summary")
    
    # Overall success indicator
    if pipeline_result.success:
        st.success(f"🎉 Pipeline completed successfully! Total time: {pipeline_result.total_time:.2f} seconds")
    else:
        st.error(f"❌ Pipeline failed at {pipeline_result.error_stage} stage: {pipeline_result.error}")
    
    # Display processing statistics
    if pipeline_result.stats:
        display_processing_stats(pipeline_result.stats)
    
    # Stage-by-stage summary
    with st.expander("🔍 Detailed Stage Information"):
        stages = [
            ("Entity Extraction", pipeline_result.entity_result),
            ("Triple Generation", pipeline_result.triple_result),
            ("Graph Judgment", pipeline_result.judgment_result)
        ]
        
        for stage_name, result in stages:
            if result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**{stage_name}**")
                with col2:
                    status = "✅ Success" if result.success else "❌ Failed"
                    st.markdown(f"Status: {status}")
                with col3:
                    st.markdown(f"Time: {result.processing_time:.2f}s")
            else:
                st.markdown(f"**{stage_name}**: Not executed")


def create_sidebar_controls() -> Dict[str, Any]:
    """
    Create sidebar controls for pipeline configuration.

    Returns:
        Dictionary of configuration options
    """
    st.sidebar.markdown("## ⚙️ Configuration Options")
    
    # API configuration
    st.sidebar.markdown("### 🔌 API Settings")
    api_timeout = st.sidebar.slider("API Timeout (seconds)", 30, 300, 60)
    max_retries = st.sidebar.slider("Max Retries", 1, 5, 3)
    
    # Processing options
    st.sidebar.markdown("### 🔄 Processing Options")
    enable_explanations = st.sidebar.checkbox("Enable Judgment Explanations", value=True)
    batch_size = st.sidebar.slider("Batch Size", 1, 20, 10)
    
    # Display options
    st.sidebar.markdown("### 🎨 Display Options")
    show_technical_details = st.sidebar.checkbox("Show Technical Details", value=False)
    auto_scroll = st.sidebar.checkbox("Auto-scroll to Results", value=True)
    
    # Debug options
    if st.sidebar.checkbox("Debug Mode"):
        st.sidebar.markdown("### 🐛 Debug Options")
        log_level = st.sidebar.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
        show_timing = st.sidebar.checkbox("Show Detailed Timing", value=True)
    else:
        log_level = "INFO"
        show_timing = False
    
    return {
        'api_timeout': api_timeout,
        'max_retries': max_retries,
        'enable_explanations': enable_explanations,
        'batch_size': batch_size,
        'show_technical_details': show_technical_details,
        'auto_scroll': auto_scroll,
        'log_level': log_level,
        'show_timing': show_timing
    }


# Helper function to safely convert dataclass to dict
def asdict(obj):
    """Convert dataclass to dictionary, handling nested objects."""
    if hasattr(obj, '__dataclass_fields__'):
        return {field.name: asdict(getattr(obj, field.name)) 
                for field in obj.__dataclass_fields__.values()}
    elif isinstance(obj, list):
        return [asdict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: asdict(value) for key, value in obj.items()}
    else:
        return obj