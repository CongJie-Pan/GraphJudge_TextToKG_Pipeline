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


def display_input_section() -> str:
    """
    Display the main input section for text entry.
    
    Returns:
        The input text from the user
    """
    st.markdown("## 📝 输入文本 (Input Text)")
    st.markdown("请输入您要分析的中文文本：")
    
    # Text area for input
    input_text = st.text_area(
        "Text Input",
        height=200,
        placeholder="请在此输入您的中文文本。例如：红楼梦是清代作家曹雪芹创作的章回体长篇小说...",
        help="支持中文古典文学文本，模型针对中文进行了优化",
        label_visibility="collapsed"
    )
    
    # Input statistics
    if input_text:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("字符数", len(input_text))
        with col2:
            st.metric("行数", len(input_text.split('\n')))
        with col3:
            st.metric("段落数", len([p for p in input_text.split('\n\n') if p.strip()]))
    
    return input_text.strip()


def display_entity_results(entity_result: EntityResult):
    """
    Display entity extraction results in a user-friendly format.
    
    Args:
        entity_result: The EntityResult to display
    """
    st.markdown("## 🔍 实体提取结果 (Entity Extraction Results)")
    
    # Success indicator and timing
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("提取状态", "✅ 成功" if entity_result.success else "❌ 失败")
    with col2:
        st.metric("处理时间", f"{entity_result.processing_time:.2f}s")
    with col3:
        st.metric("实体数量", len(entity_result.entities))
    
    if entity_result.entities:
        # Display entities as tags
        st.markdown("### 🏷️ 提取的实体")
        
        # Create entity tags with colors
        entity_html = ""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF"]
        
        for i, entity in enumerate(entity_result.entities):
            color = colors[i % len(colors)]
            entity_html += f'<span style="background-color: {color}; color: white; padding: 0.2rem 0.5rem; margin: 0.1rem; border-radius: 0.3rem; font-size: 0.9rem;">{entity}</span> '
        
        st.markdown(entity_html, unsafe_allow_html=True)
        
        # Show denoised text comparison if different
        if entity_result.denoised_text != st.session_state.get('original_input', ''):
            with st.expander("📋 处理后的文本 (Processed Text)"):
                st.text_area(
                    "Denoised Text",
                    value=entity_result.denoised_text,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
    else:
        st.warning("⚠️ 未找到任何实体，请检查输入文本是否包含可识别的实体。")


def display_triple_results(triple_result: TripleResult, show_validation: bool = True):
    """
    Display triple generation results with interactive features.
    
    Args:
        triple_result: The TripleResult to display
        show_validation: Whether to show validation information
    """
    st.markdown("## 🔗 关系三元组生成结果 (Knowledge Triple Results)")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("生成状态", "✅ 成功" if triple_result.success else "❌ 失败")
    with col2:
        st.metric("处理时间", f"{triple_result.processing_time:.2f}s")
    with col3:
        st.metric("三元组数量", len(triple_result.triples))
    with col4:
        # Calculate average confidence if available
        confidences = [t.confidence for t in triple_result.triples if t.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        st.metric("平均置信度", f"{avg_confidence:.2f}" if avg_confidence > 0 else "N/A")
    
    if triple_result.triples:
        # Create DataFrame for display
        triple_data = []
        for i, triple in enumerate(triple_result.triples):
            triple_data.append({
                "序号": i + 1,
                "主语 (Subject)": triple.subject,
                "谓语 (Predicate)": triple.predicate,
                "宾语 (Object)": triple.object,
                "置信度": f"{triple.confidence:.3f}" if triple.confidence else "N/A"
            })
        
        df = pd.DataFrame(triple_data)
        
        # Interactive table with selection
        st.markdown("### 📊 生成的知识三元组")
        
        # Table display options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("点击行可查看详细信息：")
        with col2:
            export_format = st.selectbox("导出格式", ["JSON", "CSV"], key="triple_export")
        
        # Display the table
        selected_rows = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
        
        # Export functionality
        if st.button("📥 导出三元组", key="export_triples"):
            if export_format == "JSON":
                export_data = [asdict(triple) for triple in triple_result.triples]
                st.download_button(
                    "下载 JSON 文件",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"triples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                csv_data = df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    "下载 CSV 文件", 
                    data=csv_data,
                    file_name=f"triples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Show validation information
        if show_validation and triple_result.metadata:
            with st.expander("🔍 质量分析 (Quality Analysis)"):
                display_triple_quality_analysis(triple_result)
        
        # Visualization
        if len(triple_result.triples) > 1:
            with st.expander("📈 关系可视化 (Relationship Visualization)"):
                if PLOTLY_AVAILABLE:
                    display_knowledge_graph_viz(triple_result.triples)
                else:
                    st.info("📊 可视化功能需要安装 Plotly 库: `pip install plotly`")
                    st.text("文本形式的关系展示:")
                    for i, triple in enumerate(triple_result.triples[:10], 1):
                        st.text(f"{i}. {triple.subject} → {triple.predicate} → {triple.object}")
    else:
        st.warning("⚠️ 未生成任何三元组，请检查实体提取结果。")


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
                st.info("📊 图表分析功能需要安装 Plotly 库: `pip install plotly`")


def display_triple_quality_analysis(triple_result: TripleResult):
    """
    Display quality analysis for generated triples.
    
    Args:
        triple_result: The TripleResult containing metadata
    """
    metadata = triple_result.metadata
    
    if not metadata:
        st.info("无质量分析数据")
        return
    
    # Quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**结构质量**")
        if 'avg_subject_length' in metadata:
            st.metric("平均主语长度", f"{metadata['avg_subject_length']:.1f}")
        if 'avg_predicate_length' in metadata:
            st.metric("平均谓语长度", f"{metadata['avg_predicate_length']:.1f}")
    
    with col2:
        st.markdown("**内容质量**")
        if 'unique_subjects' in metadata:
            st.metric("唯一主语数", metadata['unique_subjects'])
        if 'unique_predicates' in metadata:
            st.metric("唯一谓语数", metadata['unique_predicates'])
    
    with col3:
        st.markdown("**处理信息**")
        if 'chunks_processed' in metadata:
            st.metric("处理片段数", metadata['chunks_processed'])
        if 'validation_score' in metadata:
            st.metric("验证评分", f"{metadata['validation_score']:.2f}")


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
        st.warning("⚠️ 节点数量较多，仅显示前20个三元组的关系图")
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
        title="知识图谱可视化",
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="节点代表实体，边代表关系",
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
        st.warning("图表分析功能需要 Plotly 库支持")
        return
    
    st.markdown("### 📊 判断结果分析")
    
    # Create pie chart for approval/rejection
    approved = sum(judgment_result.judgments)
    rejected = len(judgment_result.judgments) - approved
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=[approved, rejected],
            names=['通过', '拒绝'],
            title="通过率分析",
            color_discrete_map={'通过': '#00CC96', '拒绝': '#EF553B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if judgment_result.confidence and any(c > 0 for c in judgment_result.confidence):
            fig_hist = px.histogram(
                x=judgment_result.confidence,
                title="置信度分布",
                labels={'x': '置信度', 'y': '数量'},
                nbins=10
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("暂无置信度分布数据")


def display_pipeline_summary(pipeline_result: PipelineResult):
    """
    Display a comprehensive summary of the entire pipeline run.
    
    Args:
        pipeline_result: The complete pipeline result
    """
    st.markdown("## 📈 流水线运行总结")
    
    # Overall success indicator
    if pipeline_result.success:
        st.success(f"🎉 流水线成功完成！总耗时：{pipeline_result.total_time:.2f}秒")
    else:
        st.error(f"❌ 流水线在 {pipeline_result.error_stage} 阶段失败：{pipeline_result.error}")
    
    # Display processing statistics
    if pipeline_result.stats:
        display_processing_stats(pipeline_result.stats)
    
    # Stage-by-stage summary
    with st.expander("🔍 各阶段详细信息"):
        stages = [
            ("实体提取", pipeline_result.entity_result),
            ("三元组生成", pipeline_result.triple_result),
            ("图判断", pipeline_result.judgment_result)
        ]
        
        for stage_name, result in stages:
            if result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**{stage_name}**")
                with col2:
                    status = "✅ 成功" if result.success else "❌ 失败"
                    st.markdown(f"状态：{status}")
                with col3:
                    st.markdown(f"耗时：{result.processing_time:.2f}s")
            else:
                st.markdown(f"**{stage_name}**：未执行")


def create_sidebar_controls() -> Dict[str, Any]:
    """
    Create sidebar controls for pipeline configuration.
    
    Returns:
        Dictionary of configuration options
    """
    st.sidebar.markdown("## ⚙️ 配置选项")
    
    # API configuration
    st.sidebar.markdown("### 🔌 API 设置")
    api_timeout = st.sidebar.slider("API 超时时间 (秒)", 30, 300, 60)
    max_retries = st.sidebar.slider("最大重试次数", 1, 5, 3)
    
    # Processing options
    st.sidebar.markdown("### 🔄 处理选项")
    enable_explanations = st.sidebar.checkbox("启用判断解释", value=True)
    batch_size = st.sidebar.slider("批处理大小", 1, 20, 10)
    
    # Display options
    st.sidebar.markdown("### 🎨 显示选项")
    show_technical_details = st.sidebar.checkbox("显示技术细节", value=False)
    auto_scroll = st.sidebar.checkbox("自动滚动到结果", value=True)
    
    # Debug options
    if st.sidebar.checkbox("调试模式"):
        st.sidebar.markdown("### 🐛 调试选项")
        log_level = st.sidebar.selectbox("日志级别", ["INFO", "DEBUG", "WARNING", "ERROR"])
        show_timing = st.sidebar.checkbox("显示详细计时", value=True)
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