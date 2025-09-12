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
    st.markdown("# 🏆 最终结果 (Final Results)")
    
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
                "✅ 通过的三元组",
                approved_count,
                delta=f"{(1-rejection_rate)*100:.1f}% 通过率"
            )
        
        with col2:
            st.metric(
                "❌ 被拒绝的",
                total_triples - approved_count,
                delta=f"{rejection_rate*100:.1f}% 拒绝率"
            )
        
        with col3:
            avg_confidence = (
                sum(pipeline_result.judgment_result.confidence) / 
                len(pipeline_result.judgment_result.confidence)
                if pipeline_result.judgment_result.confidence else 0
            )
            st.metric(
                "🎯 平均置信度",
                f"{avg_confidence:.3f}"
            )
        
        with col4:
            st.metric(
                "⏱️ 总处理时间",
                f"{pipeline_result.total_time:.1f}s"
            )
        
        # Display approved triples as the final knowledge graph
        if approved_triples:
            st.markdown("## 🧠 最终知识图谱")
            st.markdown(f"经过AI判断后，以下 **{len(approved_triples)}** 个知识三元组被认为是准确的：")
            
            display_final_knowledge_graph(approved_triples, pipeline_result.judgment_result)
            
            # Export options
            st.markdown("### 📤 导出选项")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 导出为 JSON", key="export_final_json"):
                    export_final_results_json(approved_triples, pipeline_result)
            
            with col2:
                if st.button("📊 导出为 CSV", key="export_final_csv"):
                    export_final_results_csv(approved_triples, pipeline_result)
            
            with col3:
                if st.button("📋 生成报告", key="generate_report"):
                    display_analysis_report(pipeline_result)
        
        else:
            st.warning("⚠️ 没有三元组通过AI判断。您可能需要调整输入文本或检查处理逻辑。")
            
            # Show rejected triples for reference
            if rejected_triples:
                with st.expander("🔍 查看被拒绝的三元组"):
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
            "序号": i + 1,
            "知识三元组": f"【{triple.subject}】 → {triple.predicate} → 【{triple.object}】",
            "主语": triple.subject,
            "关系": triple.predicate,
            "宾语": triple.object,
            "AI置信度": f"{confidence:.3f}" if confidence > 0 else "N/A",
            "质量等级": get_quality_grade(confidence) if confidence > 0 else "未评级"
        })
    
    df = pd.DataFrame(final_data)
    
    # Display with custom styling
    st.markdown("### 📋 知识三元组详情")
    
    # Interactive data table with selection
    selected_indices = st.dataframe(
        df[["序号", "知识三元组", "AI置信度", "质量等级"]],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "知识三元组": st.column_config.TextColumn(
                "知识三元组",
                help="点击查看详细信息",
                width="large"
            ),
            "AI置信度": st.column_config.ProgressColumn(
                "AI置信度",
                min_value=0.0,
                max_value=1.0,
                format="%.3f"
            ),
            "质量等级": st.column_config.TextColumn(
                "质量等级",
                help="基于置信度的质量评级"
            )
        }
    )
    
    # Show knowledge graph visualization
    if len(triples) > 1:
        st.markdown("### 🕸️ 关系网络图")
        if PLOTLY_AVAILABLE:
            create_enhanced_knowledge_graph(triples)
        else:
            st.info("📊 网络图需要安装 Plotly 库: `pip install plotly`")
            st.text("文本形式的关系展示:")
            for i, triple in enumerate(triples[:15], 1):
                st.text(f"{i}. {triple.subject} → {triple.predicate} → {triple.object}")


def display_rejected_triples_analysis(rejected_triples: List[Triple], judgment_result: JudgmentResult):
    """
    Display analysis of rejected triples to help users understand the filtering.
    
    Args:
        rejected_triples: List of rejected triples
        judgment_result: Judgment results with explanations
    """
    st.markdown("#### 被拒绝的三元组分析")
    
    rejection_data = []
    explanation_idx = 0
    
    for triple in rejected_triples:
        # Find explanation if available
        explanation = None
        if (judgment_result.explanations and 
            explanation_idx < len(judgment_result.explanations)):
            explanation = judgment_result.explanations[explanation_idx]
        
        rejection_data.append({
            "三元组": f"{triple.subject} - {triple.predicate} - {triple.object}",
            "可能原因": explanation or "AI判断该关系不够准确或相关",
            "建议": get_rejection_suggestion(triple, explanation)
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
            st.warning(f"实体数量较多({len(entities)}个)，显示前15个实体的关系图")
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
                'text': "知识图谱网络可视化",
                'x': 0.5,
                'font': {'size': 16}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="节点：实体 | 边：关系 | 线条粗细：AI置信度",
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
        st.error(f"可视化生成失败: {str(e)}")
        st.info("您仍可以查看上方的表格形式结果")


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
        label="📁 下载 JSON 文件",
        data=json_str,
        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def export_final_results_csv(triples: List[Triple], pipeline_result: PipelineResult):
    """Export final results as CSV format."""
    csv_data = []
    
    for i, triple in enumerate(triples):
        csv_data.append({
            "序号": i + 1,
            "主语": triple.subject,
            "谓语": triple.predicate,
            "宾语": triple.object,
            "置信度": triple.confidence or 0.0,
            "质量等级": get_quality_grade(triple.confidence or 0.0)
        })
    
    df = pd.DataFrame(csv_data)
    csv_string = df.to_csv(index=False, encoding='utf-8')
    
    st.download_button(
        label="📊 下载 CSV 文件", 
        data=csv_string,
        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def display_analysis_report(pipeline_result: PipelineResult):
    """Display a comprehensive analysis report."""
    st.markdown("## 📊 分析报告")
    
    # Generate report content
    report_sections = []
    
    # Executive Summary
    if pipeline_result.success:
        approved_count = sum(pipeline_result.judgment_result.judgments) if pipeline_result.judgment_result else 0
        total_count = len(pipeline_result.triple_result.triples) if pipeline_result.triple_result else 0
        approval_rate = approved_count / total_count if total_count > 0 else 0
        
        report_sections.append(f"""
        ### 📋 执行摘要
        
        - **总体状态**: ✅ 成功完成
        - **处理时间**: {pipeline_result.total_time:.2f} 秒
        - **知识提取**: 从输入文本中成功提取了 {approved_count} 个高质量知识三元组
        - **质量评级**: {approval_rate:.1%} 的生成三元组通过了AI质量检查
        """)
    
    # Stage Analysis
    if pipeline_result.entity_result:
        entities_count = len(pipeline_result.entity_result.entities)
        report_sections.append(f"""
        ### 🔍 实体提取分析
        
        - **实体数量**: {entities_count} 个
        - **处理时间**: {pipeline_result.entity_result.processing_time:.2f} 秒
        - **效率**: {entities_count/pipeline_result.entity_result.processing_time:.1f} 实体/秒
        """)
    
    if pipeline_result.triple_result:
        triples_count = len(pipeline_result.triple_result.triples)
        report_sections.append(f"""
        ### 🔗 三元组生成分析
        
        - **生成数量**: {triples_count} 个三元组
        - **处理时间**: {pipeline_result.triple_result.processing_time:.2f} 秒
        - **生成效率**: {triples_count/pipeline_result.triple_result.processing_time:.1f} 三元组/秒
        """)
    
    # Quality Analysis
    if pipeline_result.judgment_result:
        high_quality = sum(1 for c in pipeline_result.judgment_result.confidence if c > 0.8)
        medium_quality = sum(1 for c in pipeline_result.judgment_result.confidence if 0.5 <= c <= 0.8)
        low_quality = sum(1 for c in pipeline_result.judgment_result.confidence if c < 0.5)
        
        report_sections.append(f"""
        ### ⚖️ 质量分析
        
        - **高质量** (>0.8): {high_quality} 个
        - **中等质量** (0.5-0.8): {medium_quality} 个
        - **待改进** (<0.5): {low_quality} 个
        - **平均置信度**: {sum(pipeline_result.judgment_result.confidence)/len(pipeline_result.judgment_result.confidence):.3f}
        """)
    
    # Display report
    for section in report_sections:
        st.markdown(section)
    
    # Recommendations
    st.markdown("""
    ### 💡 建议
    
    1. **高质量结果**: 置信度超过0.8的三元组可以直接使用
    2. **人工审核**: 建议对置信度0.5-0.8的结果进行人工检查
    3. **结果优化**: 如需更多高质量结果，可尝试调整输入文本的表述方式
    """)


def get_quality_grade(confidence: float) -> str:
    """Convert confidence score to quality grade."""
    if confidence >= 0.9:
        return "🏆 优秀"
    elif confidence >= 0.8:
        return "🥇 良好"
    elif confidence >= 0.6:
        return "🥈 中等"
    elif confidence >= 0.4:
        return "🥉 一般"
    else:
        return "⚠️ 待改进"


def get_rejection_suggestion(triple: Triple, explanation: Optional[str]) -> str:
    """Generate suggestion for rejected triples."""
    if explanation and "不准确" in explanation:
        return "检查主语和宾语的关系是否正确表述"
    elif explanation and "不相关" in explanation:
        return "确认该关系是否与主题相关"
    elif explanation and "模糊" in explanation:
        return "尝试使用更明确的表述"
    else:
        return "重新审视该关系的表述方式或上下文"


def display_comparison_view(current_result: PipelineResult, previous_results: List[PipelineResult]):
    """
    Display comparison between current and previous results.
    
    Args:
        current_result: Current pipeline result
        previous_results: List of previous results for comparison
    """
    if not previous_results:
        return
    
    st.markdown("## 📈 历史对比")
    
    # Create comparison metrics
    comparison_data = []
    for i, result in enumerate([current_result] + previous_results[:4]):  # Current + last 4
        if result.success and result.stats:
            comparison_data.append({
                "运行": "当前" if i == 0 else f"历史-{i}",
                "总时间": result.total_time,
                "实体数": result.stats.get('entity_count', 0),
                "三元组数": result.stats.get('triple_count', 0),
                "通过数": result.stats.get('approved_triples', 0),
                "通过率": result.stats.get('approval_rate', 0)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Performance trends
        if len(comparison_data) > 1 and PLOTLY_AVAILABLE:
            fig = px.line(
                df, 
                x="运行", 
                y=["总时间", "通过率"],
                title="性能趋势",
                labels={"value": "数值", "variable": "指标"}
            )
            st.plotly_chart(fig, use_container_width=True)
        elif len(comparison_data) > 1:
            st.info("📊 趋势图需要安装 Plotly 库: `pip install plotly`")