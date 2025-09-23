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
from streamlit_pipeline.ui.evaluation_display import (
    display_evaluation_dashboard, display_evaluation_configuration,
    display_reference_graph_upload, display_evaluation_export_options
)
from streamlit_pipeline.ui.simple_progress import (
    SimpleProgressTracker, display_simple_processing_summary,
    display_entity_processing, display_triple_generation_processing,
    display_graph_judgment_processing
)

# Utilities
from streamlit_pipeline.utils.error_handling import ErrorHandler, ErrorInfo, ErrorType, ErrorSeverity, StreamlitLogger
from streamlit_pipeline.utils.api_client import get_api_client
from streamlit_pipeline.utils.session_state import get_session_manager, store_pipeline_result
from streamlit_pipeline.utils.state_persistence import persist_pipeline_result, get_persistence_manager
from streamlit_pipeline.utils.state_cleanup import get_cleanup_manager, check_and_run_cleanup
from streamlit_pipeline.utils.i18n import get_text


# Early i18n initialization for page config
from streamlit_pipeline.utils.i18n import get_current_language, get_i18n_manager

# Initialize i18n system early to ensure proper language loading
def init_i18n_for_page_config():
    """Initialize i18n system early for page configuration."""
    try:
        # Ensure session state exists with proper default
        if 'current_language' not in st.session_state:
            st.session_state.current_language = 'en'

        # Initialize the i18n manager with better error handling
        manager = get_i18n_manager()
        if manager and manager.translations:
            return manager

        # If manager failed, try to initialize manually
        from streamlit_pipeline.utils.i18n import I18nManager
        backup_manager = I18nManager()
        return backup_manager

    except Exception as e:
        # Last resort: return None but log the issue
        print(f"I18n initialization failed: {e}")
        return None

def get_dynamic_page_title():
    """Get page title based on current language setting with proper i18n fallback."""
    try:
        # Try to initialize i18n system
        manager = init_i18n_for_page_config()
        if manager and manager.translations:
            # Get current language, defaulting to 'en'
            current_lang = st.session_state.get('current_language', 'en')

            # Ensure the language exists in translations
            if current_lang in manager.translations:
                app_translations = manager.translations[current_lang].get('app', {})
                if 'page_title' in app_translations:
                    return app_translations['page_title']

            # Fallback to English if current language not found
            if 'en' in manager.translations:
                en_app = manager.translations['en'].get('app', {})
                if 'page_title' in en_app:
                    return en_app['page_title']

    except Exception as e:
        print(f"Page title localization failed: {e}")

    # Ultimate fallback - still avoid hardcoding by using default English
    return "GraphJudge - Intelligent Knowledge Graph Construction"

# Configure Streamlit page with dynamic title
st.set_page_config(
    page_title=get_dynamic_page_title(),
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
        # Initialize i18n system first to ensure language support
        self._initialize_i18n()

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

    def _initialize_i18n(self):
        """Initialize the i18n system with proper session state setup and error handling."""
        try:
            # Ensure current_language is set in session state with validation
            if 'current_language' not in st.session_state:
                st.session_state.current_language = 'en'  # Default to English

            # Validate the current language setting
            valid_languages = ['en', 'zh_CN', 'zh_TW']
            if st.session_state.current_language not in valid_languages:
                st.session_state.current_language = 'en'  # Reset to default if invalid

            # Initialize the i18n manager with error handling
            manager = get_i18n_manager()
            if manager and manager.translations:
                # Verify translations are loaded for current language
                current_lang = st.session_state.current_language
                if current_lang not in manager.translations:
                    # Fallback to English if current language not available
                    st.session_state.current_language = 'en'
                    st.warning(f"Language '{current_lang}' not available, using English")
            else:
                # Manager initialization failed, try manual initialization
                from streamlit_pipeline.utils.i18n import I18nManager
                backup_manager = I18nManager()
                if not backup_manager.translations:
                    st.warning("Translation system initialized with limited functionality")

        except Exception as e:
            # Critical error in i18n initialization
            st.error(f"Failed to initialize internationalization system: {e}")
            # Ensure we have a fallback language setting
            st.session_state.current_language = 'en'

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
            st.error(get_text('errors.app_error'))
            st.exception(e)
            
            # Log the error
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_error(
                    f"Application error: {str(e)}",
                    {"traceback": traceback.format_exc()}
                )
    
    def _render_header(self):
        """Render the application header with proper language support."""
        try:
            # Ensure the current language is properly set
            current_lang = st.session_state.get('current_language', 'en')

            # Get localized text with error handling
            title_text = get_text('app.title')
            desc_text = get_text('app.description')
            getting_started_text = get_text('app.getting_started')

            # Debug info (can be removed in production)
            if st.session_state.get('debug_mode', False):
                st.sidebar.text(f"Current Language: {current_lang}")
                st.sidebar.text(f"Title: {title_text[:30]}...")

            # Render header elements
            st.title(title_text)
            st.markdown(f"""
            {desc_text}

            {getting_started_text}
            """)

            # Quick stats if we have results
            if st.session_state.pipeline_results:
                self._render_quick_stats()

            st.markdown("---")

        except Exception as e:
            # Fallback to English if there's an error
            st.title("🧠 GraphJudge - Intelligent Knowledge Graph Construction System")
            st.markdown("""
            **GraphJudge** is an intelligent knowledge graph construction system based on large language models.
            Through a three-stage processing pipeline, it extracts entities from Chinese text, generates
            knowledge triples, and uses AI for quality assessment.

            💡 **Getting Started**: Upload a Chinese text file (.txt) or paste text directly to begin analysis.
            """)

            if st.session_state.get('debug_mode', False):
                st.error(f"Header rendering error: {e}")

            st.markdown("---")
    
    def _render_quick_stats(self):
        """Render quick statistics from previous runs."""
        recent_results = st.session_state.pipeline_results[-5:]  # Last 5 runs
        successful_runs = [r for r in recent_results if r.success]
        
        if successful_runs:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(get_text('metrics.total_runs'), len(st.session_state.pipeline_results))

            with col2:
                avg_time = sum(r.total_time for r in successful_runs) / len(successful_runs)
                st.metric(get_text('metrics.avg_processing_time'), get_text('metrics.time_seconds', time=f"{avg_time:.1f}"))

            with col3:
                total_triples = sum(
                    len(r.triple_result.triples) if r.triple_result else 0
                    for r in successful_runs
                )
                st.metric(get_text('metrics.total_triples'), total_triples)

            with col4:
                if successful_runs and successful_runs[-1].stats:
                    approval_rate = successful_runs[-1].stats.get('approval_rate', 0)
                    st.metric(get_text('metrics.approval_rate'), get_text('metrics.percentage', rate=f"{approval_rate:.1%}"))
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        # Get configuration options
        st.session_state.config_options = create_sidebar_controls()

        # Evaluation configuration section
        st.sidebar.markdown("---")
        evaluation_config = display_evaluation_configuration()
        st.session_state.evaluation_config = evaluation_config

        # Reference graph upload section (only if evaluation is enabled)
        if evaluation_config.get('enable_evaluation', False):
            st.sidebar.markdown("---")
            reference_graph = display_reference_graph_upload()
            st.session_state.reference_graph = reference_graph

        # API Status
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {get_text('sidebar.api_status')}")

        if st.sidebar.button(get_text('sidebar.test_connection'), key="test_apis"):
            with st.spinner(get_text('sidebar.testing_apis')):
                self._test_api_connections()

        # Application info
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {get_text('sidebar.about')}")
        st.sidebar.info(f"""
        {get_text('app.version')}
        {get_text('app.models')}
        {get_text('app.best_for')}
        {get_text('app.developed_by')}
        """)

        # Clear results option with enhanced cleanup
        if st.session_state.pipeline_results:
            st.sidebar.markdown("---")
            col1, col2 = st.sidebar.columns(2)

            with col1:
                if st.button(get_text('sidebar.clear_results'), key="clear_results"):
                    self.session_manager.reset_pipeline_data()
                    st.rerun()

            with col2:
                if st.button(get_text('sidebar.full_cleanup'), key="full_cleanup"):
                    self.cleanup_manager.force_complete_cleanup()
                    st.rerun()

        # Session statistics in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {get_text('sidebar.session_stats')}")
        metadata = self.session_manager.get_session_metadata()
        cache_stats = self.session_manager.get_cache_stats()

        st.sidebar.text(get_text('sidebar.run_count', count=metadata.run_count))
        st.sidebar.text(get_text('sidebar.successful_runs', count=metadata.successful_runs))
        if metadata.run_count > 0:
            success_rate = metadata.successful_runs / metadata.run_count
            st.sidebar.text(get_text('sidebar.success_rate', rate=f"{success_rate:.1%}"))

        st.sidebar.text(get_text('sidebar.cache_hit_rate', rate=f"{cache_stats.hit_rate:.1%}"))
        st.sidebar.text(get_text('sidebar.cache_size', size=f"{cache_stats.total_size_bytes / 1024 / 1024:.1f}"))
    
    def _test_api_connections(self):
        """Test API connections and display status."""
        try:
            # Get API client
            api_client = get_api_client()
            st.sidebar.success(get_text('sidebar.api_config_loaded'))

            # Test basic configuration
            from streamlit_pipeline.core.config import get_api_config
            try:
                api_key, api_base = get_api_config(load_env=True)
                if api_key:
                    st.sidebar.success(get_text('sidebar.api_key_configured'))

                    # Perform actual API connection test
                    st.sidebar.info(get_text('sidebar.testing_actual_api'))
                    test_results = api_client.test_api_connection()

                    # Display test results for each model
                    for model_name, result in test_results.items():
                        if result["status"] == "success":
                            st.sidebar.success(get_text('sidebar.api_connection_successful', model=model_name))
                            st.sidebar.caption(f"Model: {result['model']}")
                        else:
                            st.sidebar.error(f"❌ {model_name}: {result['error']}")
                            st.sidebar.caption(f"Model: {result['model']}")

                else:
                    st.sidebar.error(get_text('sidebar.api_key_not_configured'))
            except Exception as e:
                st.sidebar.error(get_text('errors.api_config_error', error=str(e)))

        except Exception as e:
            st.sidebar.error(f"API Test Failed: {str(e)}")
    
    def _render_main_interface(self):
        """Render the main interface for input and results."""
        # Input section
        input_text = display_input_section()
        
        # Processing controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            process_button = st.button(
                f"🚀 {get_text('buttons.start_processing')}",
                disabled=not input_text.strip(),
                type="primary",
                help="Click to start the three-stage knowledge graph construction pipeline"
            )
        
        with col2:
            if st.session_state.current_result:
                st.button(f"📊 {get_text('buttons.view_details')}", key="view_details", on_click=self._show_detailed_results)

        with col3:
            if st.session_state.pipeline_results:
                st.button(f"📈 {get_text('buttons.historical_comparison')}", key="show_comparison", on_click=self._show_comparison)
        
        # Process the input if button clicked
        if process_button and input_text.strip():
            self._start_processing(input_text.strip())
        
        # Display current results
        if st.session_state.current_result:
            self._render_results_section(st.session_state.current_result)
    
    def _render_processing_view(self):
        """Render the processing view with progress indicators."""
        st.markdown(f"## {get_text('processing.title')}")

        # This would typically be handled by the progress callback
        # For now, show a static processing message
        st.info(get_text('processing.processing_input'))

        # Add a cancel button
        if st.button(get_text('processing.cancel_processing'), key="cancel_processing"):
            st.session_state.processing = False
            st.rerun()
    
    def _start_processing(self, input_text: str):
        """
        Start the pipeline processing with enhanced session management and detailed progress tracking.

        Args:
            input_text: The input text to process
        """
        # Set processing state using session manager
        self.session_manager.set_processing_state(True, 0)
        st.session_state.processing = True
        st.session_state.original_input = input_text

        # Store input text in session manager for potential recovery
        self.session_manager.set_ui_state('temp_input', input_text)

        # Initialize simple progress tracking
        simple_tracker = SimpleProgressTracker()
        progress_container, status_container = simple_tracker.initialize_display()

        try:
            # Simple progress callback function
            def progress_callback(stage: int, message: str):
                # Update session manager progress data
                self.session_manager.update_progress_data(
                    stage, message,
                    timestamp=time.time(),
                    input_length=len(input_text)
                )

                # Update simple progress tracker based on stage
                if stage == 0:  # Entity Extraction
                    simple_tracker.start_phase(
                        "🔍 Entity Extraction & Text Denoising",
                        "Extracting entities from Chinese text"
                    )
                elif stage == 1:  # Triple Generation
                    simple_tracker.finish_phase(True, "Entity extraction completed")
                    simple_tracker.start_phase(
                        "🔗 Triple Generation",
                        "Generating knowledge graph triples"
                    )
                elif stage == 2:  # Graph Judgment
                    simple_tracker.finish_phase(True, "Triple generation completed")
                    simple_tracker.start_phase(
                        "⚖️ Graph Judgment",
                        "Validating triples with Perplexity AI"
                    )

            # Run the pipeline with detailed tracking and evaluation support
            start_time = time.time()

            # Get evaluation configuration and reference graph if available
            evaluation_config = st.session_state.get('evaluation_config', {})
            reference_graph = st.session_state.get('reference_graph', None)

            result = self.orchestrator.run_pipeline(
                input_text,
                progress_callback,
                st.session_state.config_options,
                evaluation_config=evaluation_config,
                reference_graph=reference_graph
            )
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
            
            # Complete simple tracking
            if result.success:
                simple_tracker.finish_phase(True, "All pipeline phases completed successfully!")
            else:
                simple_tracker.finish_phase(False, f"Pipeline failed: {result.error}")

            # Clear processing state using session manager
            self.session_manager.set_processing_state(False)
            st.session_state.processing = False

            # Log the completion
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_info(
                    f"Pipeline completed in {result.total_time:.2f}s",
                    {"success": result.success, "stage_reached": result.stage_reached}
                )

            # Clear progress displays
            simple_tracker.clear_display()

            if result.success:
                st.success(get_text('processing.processing_complete_time', time=f"{result.total_time:.2f}"))
                st.balloons()

                # Display simple processing summary
                display_simple_processing_summary(simple_tracker, result)
            else:
                st.error(get_text('processing.processing_failed_prefix', error=result.error))

                # Still show simple processing summary for debugging
                display_simple_processing_summary(simple_tracker, result)

            st.rerun()
            
        except Exception as e:
            st.session_state.processing = False
            error_msg = f"Processing failed: {str(e)}"
            st.error(error_msg)
            
            # Create error info for display
            error_info = ErrorInfo(
                error_type=ErrorType.PROCESSING,
                severity=ErrorSeverity.HIGH,
                message="An error occurred during pipeline processing",
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

            # Show evaluation results if available
            if hasattr(result, 'evaluation_result') and result.evaluation_result:
                st.markdown("---")
                display_evaluation_dashboard(result.evaluation_result, show_detailed=True)

                # Export options for evaluation results
                display_evaluation_export_options(result.evaluation_result, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Detailed results in expandable sections
            with st.expander("🔍 View Detailed Results by Stage", expanded=False):
                if result.entity_result:
                    display_entity_results(result.entity_result, show_expanders=False)
                    st.markdown("---")

                if result.triple_result:
                    display_triple_results(result.triple_result, show_expanders=False)
                    st.markdown("---")

                if result.judgment_result and result.triple_result:
                    display_judgment_results(result.judgment_result, result.triple_result.triples, show_expanders=False)

            # Pipeline summary
            with st.expander("📊 Execution Summary", expanded=False):
                display_pipeline_summary(result, show_expanders=False)
        
        else:
            # Show error information
            st.error(get_text('errors.pipeline_failed_stage_prefix', stage=result.error_stage))
            st.error(get_text('errors.generic_error_prefix', error=result.error))
            
            # Show partial results if available
            if result.entity_result and result.entity_result.success:
                with st.expander("🔍 Entity Extraction Results (Partial)"):
                    display_entity_results(result.entity_result, show_expanders=False)

            if result.triple_result and result.triple_result.success:
                with st.expander("🔗 Triple Generation Results (Partial)"):
                    display_triple_results(result.triple_result, show_expanders=False)
            
            # Recovery suggestions
            error_info = ErrorInfo(
                error_type=ErrorType.PROCESSING,
                severity=ErrorSeverity.HIGH,
                message=result.error or "Unknown error occurred",
                stage=result.error_stage
            )
            ErrorRecoveryHelper.display_recovery_options(error_info)
    
    def _show_detailed_results(self):
        """Show detailed results in a dedicated section."""
        if st.session_state.current_result:
            st.markdown(f"## {get_text('results.detailed_analysis')}")
            self._render_results_section(st.session_state.current_result)

    def _show_comparison(self):
        """Show comparison with historical results."""
        if st.session_state.current_result and st.session_state.pipeline_results:
            st.markdown(f"## {get_text('results.historical_comparison')}")
            display_comparison_view(
                st.session_state.current_result,
                st.session_state.pipeline_results[:-1]  # Exclude current result
            )
    
    def _render_footer(self):
        """Render the application footer."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(get_text('app.version_footer'))
            st.caption(get_text('app.powered_by'))
        
        with col2:
            if st.session_state.current_result:
                st.markdown(f"{get_text('app.runtime_label')}: {st.session_state.current_result.total_time:.2f}s")
                st.caption(get_text('app.processed_at', time=datetime.now().strftime('%H:%M:%S')))
        
        with col3:
            st.markdown(get_text('app.status_ready'))
            st.caption(get_text('app.ready_for_next'))


def main():
    """Main application entry point."""
    try:
        app = GraphJudgeApp()
        app.run()
    except Exception as e:
        st.error(get_text('errors.app_init_failed'))
        st.exception(e)


if __name__ == "__main__":
    main()