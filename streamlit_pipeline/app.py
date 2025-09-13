"""
GraphJudge Streamlit Application - Main Entry Point

This is the main Streamlit application for the GraphJudge Text-to-KG pipeline.
It provides a user-friendly web interface for the three-stage processing:
Entity Extraction → Triple Generation → Graph Judgment

Following spec.md user flows (Section 5) and system architecture (Section 6).
"""

import streamlit as st
import logging
import traceback
import time
from datetime import datetime
from typing import Optional, Dict, Any

# Set up path for imports from within the package
import sys
import os

# Add parent directory to path to enable absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Core pipeline imports
from streamlit_pipeline.core.pipeline import PipelineOrchestrator, PipelineResult
from streamlit_pipeline.core import config

# UI components
from streamlit_pipeline.ui.components import (
    display_input_section, display_entity_results, display_triple_results,
    display_judgment_results, display_pipeline_summary, create_sidebar_controls
)
from streamlit_pipeline.ui.display import display_final_results, display_comparison_view
from streamlit_pipeline.ui.error_display import (
    PipelineProgressDisplay, display_error_card, ErrorRecoveryHelper
)

# Utilities
from streamlit_pipeline.utils.error_handling import ErrorHandler, ErrorInfo, ErrorType, StreamlitLogger
from streamlit_pipeline.utils.api_client import get_api_client
from streamlit_pipeline.utils.session_state import get_session_manager, store_pipeline_result
from streamlit_pipeline.utils.state_persistence import persist_pipeline_result, get_persistence_manager
from streamlit_pipeline.utils.state_cleanup import get_cleanup_manager, check_and_run_cleanup


# Configure Streamlit page
st.set_page_config(
    page_title="GraphJudge - 智能知识图谱构建",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


class GraphJudgeApp:
    """
    Main application class for the GraphJudge Streamlit interface.
    
    Manages session state, coordinates pipeline execution, and handles
    the overall user experience according to spec.md requirements.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.config = config.get_model_config()
        self.error_handler = ErrorHandler()
        self.orchestrator = PipelineOrchestrator()
        self.progress_display = PipelineProgressDisplay()
        
        # Initialize enhanced session state management
        self.session_manager = get_session_manager()
        self.persistence_manager = get_persistence_manager()
        self.cleanup_manager = get_cleanup_manager()
        
        # Initialize session state (now handled by session manager)
        self._initialize_session_state()
        
        # Set up logging
        self._setup_logging()
        
        # Schedule automatic cleanup
        self.cleanup_manager.schedule_automatic_cleanup(interval_minutes=30)
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables using enhanced session manager."""
        # Session state is now handled by the SessionStateManager
        # This method ensures compatibility with the enhanced system
        
        # Check and run scheduled cleanup
        check_and_run_cleanup()
        
        # Ensure backward compatibility with existing UI code
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = self.session_manager.get_pipeline_results()
        
        if 'current_result' not in st.session_state:
            st.session_state.current_result = self.session_manager.get_current_result()
        
        if 'processing' not in st.session_state:
            st.session_state.processing = self.session_manager.is_processing()
        
        if 'run_count' not in st.session_state:
            metadata = self.session_manager.get_session_metadata()
            st.session_state.run_count = metadata.run_count
        
        if 'config_options' not in st.session_state:
            st.session_state.config_options = self.session_manager.get_ui_state('config_options', {})
    
    def _setup_logging(self):
        """Set up logging for the application."""
        if 'logger' not in st.session_state:
            st.session_state.logger = StreamlitLogger()
        
        # Configure logging level
        log_level = st.session_state.config_options.get('log_level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_level))
    
    def run(self):
        """Main application entry point."""
        try:
            # Application header
            self._render_header()
            
            # Sidebar configuration
            self._render_sidebar()
            
            # Main content area
            if st.session_state.processing:
                self._render_processing_view()
            else:
                self._render_main_interface()
            
            # Footer
            self._render_footer()
            
        except Exception as e:
            st.error("Application error occurred")
            st.exception(e)
            
            # Log the error
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_error(
                    f"Application error: {str(e)}",
                    {"traceback": traceback.format_exc()}
                )
    
    def _render_header(self):
        """Render the application header."""
        st.title("🧠 GraphJudge - 智能知识图谱构建系统")
        st.markdown("""
        **GraphJudge** 是一个基于大语言模型的智能知识图谱构建系统。通过三阶段处理流程，
        从中文文本中提取实体、生成知识三元组，并使用AI进行质量判断。
        """)
        
        # Quick stats if we have results
        if st.session_state.pipeline_results:
            self._render_quick_stats()
        
        st.markdown("---")
    
    def _render_quick_stats(self):
        """Render quick statistics from previous runs."""
        recent_results = st.session_state.pipeline_results[-5:]  # Last 5 runs
        successful_runs = [r for r in recent_results if r.success]
        
        if successful_runs:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总运行次数", len(st.session_state.pipeline_results))
            
            with col2:
                avg_time = sum(r.total_time for r in successful_runs) / len(successful_runs)
                st.metric("平均处理时间", f"{avg_time:.1f}s")
            
            with col3:
                total_triples = sum(
                    len(r.triple_result.triples) if r.triple_result else 0
                    for r in successful_runs
                )
                st.metric("累计生成三元组", total_triples)
            
            with col4:
                if successful_runs and successful_runs[-1].stats:
                    approval_rate = successful_runs[-1].stats.get('approval_rate', 0)
                    st.metric("最近通过率", f"{approval_rate:.1%}")
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        # Get configuration options
        st.session_state.config_options = create_sidebar_controls()
        
        # API Status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔌 API状态检查")
        
        if st.sidebar.button("测试API连接", key="test_apis"):
            with st.sidebar.spinner("测试中..."):
                self._test_api_connections()
        
        # Application info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ 关于")
        st.sidebar.info("""
        **版本**: 2.0  
        **模型**: GPT-5-mini + Perplexity  
        **最适合**: 中文古典文学文本  
        **开发**: GraphJudge Research Team
        """)
        
        # Clear results option with enhanced cleanup
        if st.session_state.pipeline_results:
            st.sidebar.markdown("---")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("🗑️ 清除结果", key="clear_results"):
                    self.session_manager.reset_pipeline_data()
                    st.rerun()
            
            with col2:
                if st.button("🧹 完整清理", key="full_cleanup"):
                    self.cleanup_manager.force_complete_cleanup()
                    st.rerun()
        
        # Session statistics in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 会话统计")
        metadata = self.session_manager.get_session_metadata()
        cache_stats = self.session_manager.get_cache_stats()
        
        st.sidebar.text(f"运行次数: {metadata.run_count}")
        st.sidebar.text(f"成功次数: {metadata.successful_runs}")
        if metadata.run_count > 0:
            success_rate = metadata.successful_runs / metadata.run_count
            st.sidebar.text(f"成功率: {success_rate:.1%}")
        
        st.sidebar.text(f"缓存命中率: {cache_stats.hit_rate:.1%}")
        st.sidebar.text(f"缓存大小: {cache_stats.total_size_bytes / 1024 / 1024:.1f} MB")
    
    def _test_api_connections(self):
        """Test API connections and display status."""
        try:
            # Simple API test - try to get the API client
            api_client = get_api_client()
            st.sidebar.success("✅ API配置: 正常加载")
            
            # Test basic configuration
            from streamlit_pipeline.core.config import get_api_config
            try:
                api_key, api_base = get_api_config(load_env=True)
                if api_key:
                    st.sidebar.success("✅ API密钥: 已配置")
                else:
                    st.sidebar.error("❌ API密钥: 未配置")
            except Exception as e:
                st.sidebar.error(f"❌ API配置错误: {str(e)}")
                    
        except Exception as e:
            st.sidebar.error(f"API测试失败: {str(e)}")
    
    def _render_main_interface(self):
        """Render the main interface for input and results."""
        # Input section
        input_text = display_input_section()
        
        # Processing controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            process_button = st.button(
                "🚀 开始处理 (Start Processing)",
                disabled=not input_text.strip(),
                type="primary",
                help="点击开始三阶段知识图谱构建流程"
            )
        
        with col2:
            if st.session_state.current_result:
                st.button("📊 查看详细结果", key="view_details", on_click=self._show_detailed_results)
        
        with col3:
            if st.session_state.pipeline_results:
                st.button("📈 历史对比", key="show_comparison", on_click=self._show_comparison)
        
        # Process the input if button clicked
        if process_button and input_text.strip():
            self._start_processing(input_text.strip())
        
        # Display current results
        if st.session_state.current_result:
            self._render_results_section(st.session_state.current_result)
    
    def _render_processing_view(self):
        """Render the processing view with progress indicators."""
        st.markdown("## 🔄 处理中...")
        
        # This would typically be handled by the progress callback
        # For now, show a static processing message
        st.info("Pipeline is processing your input. This may take a few minutes...")
        
        # Add a cancel button
        if st.button("❌ 取消处理", key="cancel_processing"):
            st.session_state.processing = False
            st.rerun()
    
    def _start_processing(self, input_text: str):
        """
        Start the pipeline processing with enhanced session management.
        
        Args:
            input_text: The input text to process
        """
        # Set processing state using session manager
        self.session_manager.set_processing_state(True, 0)
        st.session_state.processing = True
        st.session_state.original_input = input_text
        
        # Store input text in session manager for potential recovery
        self.session_manager.set_ui_state('temp_input', input_text)
        
        # Initialize progress display
        progress_container = st.empty()
        status_container = st.empty()
        
        try:
            # Progress callback function with enhanced tracking
            def progress_callback(stage: int, message: str):
                # Update session manager progress data
                self.session_manager.update_progress_data(
                    stage, message, 
                    timestamp=time.time(),
                    input_length=len(input_text)
                )
                
                with progress_container.container():
                    # Update progress bar
                    progress = (stage + 1) / 4  # 4 total stages (including completion)
                    st.progress(progress, text=message)
                
                with status_container.container():
                    stage_names = ["🔍 实体提取", "🔗 三元组生成", "⚖️ 图判断", "✅ 完成"]
                    if stage < len(stage_names):
                        st.info(f"当前阶段: {stage_names[stage]}")
            
            # Run the pipeline
            start_time = time.time()
            result = self.orchestrator.run_pipeline(input_text, progress_callback)
            end_time = time.time()
            
            # Store results using enhanced session management
            result.total_time = end_time - start_time
            
            # Store result in session manager (automatically handles history and metadata)
            self.session_manager.set_current_result(result)
            
            # Persist large results for performance
            persist_pipeline_result(f"run_{self.session_manager.get_session_metadata().run_count}", result)
            
            # Update backward compatibility variables
            st.session_state.current_result = result
            st.session_state.pipeline_results = self.session_manager.get_pipeline_results()
            st.session_state.run_count = self.session_manager.get_session_metadata().run_count
            
            # Clear processing state using session manager
            self.session_manager.set_processing_state(False)
            st.session_state.processing = False
            
            # Log the completion
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_info(
                    f"Pipeline completed in {result.total_time:.2f}s",
                    {"success": result.success, "stage_reached": result.stage_reached}
                )
            
            # Show results
            progress_container.empty()
            status_container.empty()
            
            if result.success:
                st.success(f"🎉 处理完成！总耗时: {result.total_time:.2f} 秒")
                st.balloons()
            else:
                st.error(f"❌ 处理失败: {result.error}")
            
            st.rerun()
            
        except Exception as e:
            st.session_state.processing = False
            error_msg = f"Processing failed: {str(e)}"
            st.error(error_msg)
            
            # Create error info for display
            error_info = ErrorInfo(
                error_type=ErrorType.PROCESSING,
                message="流水线处理过程中发生错误",
                technical_details=str(e),
                stage="pipeline_execution"
            )
            
            display_error_card(error_info)
            
            # Log the error
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_error(error_msg, {"traceback": traceback.format_exc()})
    
    def _render_results_section(self, result: PipelineResult):
        """
        Render the results section based on pipeline results.
        
        Args:
            result: The pipeline result to display
        """
        st.markdown("---")
        
        if result.success:
            # Show final results prominently
            display_final_results(result)
            
            # Detailed results in expandable sections
            with st.expander("🔍 查看各阶段详细结果", expanded=False):
                if result.entity_result:
                    display_entity_results(result.entity_result)
                    st.markdown("---")
                
                if result.triple_result:
                    display_triple_results(result.triple_result)
                    st.markdown("---")
                
                if result.judgment_result and result.triple_result:
                    display_judgment_results(result.judgment_result, result.triple_result.triples)
            
            # Pipeline summary
            with st.expander("📊 运行总结", expanded=False):
                display_pipeline_summary(result)
        
        else:
            # Show error information
            st.error(f"Pipeline failed at stage: {result.error_stage}")
            st.error(f"Error: {result.error}")
            
            # Show partial results if available
            if result.entity_result and result.entity_result.success:
                with st.expander("🔍 实体提取结果 (部分完成)"):
                    display_entity_results(result.entity_result)
            
            if result.triple_result and result.triple_result.success:
                with st.expander("🔗 三元组生成结果 (部分完成)"):
                    display_triple_results(result.triple_result)
            
            # Recovery suggestions
            error_info = ErrorInfo(
                error_type=ErrorType.PROCESSING,
                message=result.error or "Unknown error occurred",
                stage=result.error_stage
            )
            ErrorRecoveryHelper.display_recovery_options(error_info)
    
    def _show_detailed_results(self):
        """Show detailed results in a dedicated section."""
        if st.session_state.current_result:
            st.markdown("## 📋 详细结果分析")
            self._render_results_section(st.session_state.current_result)
    
    def _show_comparison(self):
        """Show comparison with historical results."""
        if st.session_state.current_result and st.session_state.pipeline_results:
            st.markdown("## 📈 历史对比分析")
            display_comparison_view(
                st.session_state.current_result,
                st.session_state.pipeline_results[:-1]  # Exclude current result
            )
    
    def _render_footer(self):
        """Render the application footer."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**GraphJudge v2.0**")
            st.caption("Powered by GPT-5-mini & Perplexity")
        
        with col2:
            if st.session_state.current_result:
                st.markdown(f"**运行时间**: {st.session_state.current_result.total_time:.2f}s")
                st.caption(f"处理于: {datetime.now().strftime('%H:%M:%S')}")
        
        with col3:
            st.markdown("**状态**: 就绪")
            st.caption("Ready for next processing")


def main():
    """Main application entry point."""
    try:
        app = GraphJudgeApp()
        app.run()
    except Exception as e:
        st.error("Failed to initialize application")
        st.exception(e)


if __name__ == "__main__":
    main()