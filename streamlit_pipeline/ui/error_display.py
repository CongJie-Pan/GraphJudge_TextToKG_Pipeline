"""
User-Friendly Error Display Components for GraphJudge Streamlit Pipeline.

This module provides Streamlit components for displaying errors, warnings,
and progress information in a user-friendly way. It integrates with the
error handling system to provide clear feedback without technical jargon.

Following spec.md Section 10 and Section 3 (FR-I2) requirements.
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
try:
    # Try absolute import first (for package installation)
    from streamlit_pipeline.utils.error_handling import ErrorInfo, ErrorType, ErrorSeverity, ProgressTracker
except ImportError:
    # Fallback to relative imports (for direct execution)
    from ..utils.error_handling import ErrorInfo, ErrorType, ErrorSeverity, ProgressTracker


def display_error_card(error_info: ErrorInfo, show_details: bool = False):
    """
    Display a comprehensive error card with user-friendly messaging.
    
    Args:
        error_info: The error information to display
        show_details: Whether to show technical details by default
    """
    # Choose appropriate emoji and color based on severity
    severity_config = {
        ErrorSeverity.CRITICAL: {"emoji": "🚨", "color": "red", "label": "Critical Error"},
        ErrorSeverity.HIGH: {"emoji": "❌", "color": "red", "label": "Error"}, 
        ErrorSeverity.MEDIUM: {"emoji": "⚠️", "color": "orange", "label": "Warning"},
        ErrorSeverity.LOW: {"emoji": "ℹ️", "color": "blue", "label": "Notice"}
    }
    
    config = severity_config.get(error_info.severity, severity_config[ErrorSeverity.MEDIUM])
    
    # Main error message
    with st.container():
        st.markdown(f"""
        <div style="padding: 1rem; border-left: 4px solid {config['color']}; background-color: rgba(255, 0, 0, 0.1); margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0;">{config['emoji']} {config['label']}</h4>
            <p style="margin: 0; font-size: 1.1rem;">{error_info.message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stage information
        if error_info.stage:
            st.caption(f"📍 Error occurred in: **{error_info.stage.replace('_', ' ').title()}**")
        
        # Timestamp
        time_str = error_info.timestamp.strftime("%H:%M:%S")
        st.caption(f"🕒 Time: {time_str}")
        
        # Suggestions
        if error_info.suggestions:
            st.markdown("**💡 What you can do:**")
            for i, suggestion in enumerate(error_info.suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        
        # Retry information
        if error_info.retry_possible:
            st.success("🔄 This operation can be retried - try clicking the button again!")
        
        # Partial results information
        if error_info.partial_results:
            st.info("📊 Some results may still be available despite this error.")
        
        # Technical details (collapsible)
        if error_info.technical_details:
            with st.expander("🔧 Technical Details (for debugging)"):
                st.code(error_info.technical_details, language="text")


def display_success_message(message: str, details: Optional[Dict[str, Any]] = None):
    """
    Display a success message with optional details.
    
    Args:
        message: The success message to display
        details: Optional additional details (e.g., timing, counts)
    """
    st.success(f"✅ {message}")
    
    if details:
        cols = st.columns(len(details))
        for i, (key, value) in enumerate(details.items()):
            with cols[i]:
                st.metric(key.replace('_', ' ').title(), value)


def display_progress_card(stage_name: str, message: str, progress: float = None):
    """
    Display a progress card for the current operation.
    
    Args:
        stage_name: Name of the current stage
        message: Current status message
        progress: Optional progress value (0.0 to 1.0)
    """
    with st.container():
        st.markdown(f"""
        <div style="padding: 1rem; border-left: 4px solid #1f77b4; background-color: rgba(31, 119, 180, 0.1); margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0;">🔄 {stage_name}</h4>
            <p style="margin: 0;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if progress is not None:
            st.progress(progress)


class PipelineProgressDisplay:
    """
    Enhanced progress display for the three-stage pipeline.
    
    Provides visual feedback about the current stage and overall progress,
    with support for error states and partial completion.
    """
    
    def __init__(self):
        """Initialize the progress display."""
        self.stages = [
            {"name": "Entity Extraction", "emoji": "🔍", "description": "Finding entities in your text"},
            {"name": "Triple Generation", "emoji": "🔗", "description": "Creating knowledge relationships"}, 
            {"name": "Graph Judgment", "emoji": "⚖️", "description": "Evaluating relationship quality"}
        ]
        self.current_stage = -1
        self.error_stage = None
        
        # UI components
        self.progress_container = None
        self.status_container = None
        self.timing_container = None
        
    def initialize_display(self):
        """Set up the progress display components in Streamlit."""
        self.progress_container = st.empty()
        self.status_container = st.empty() 
        self.timing_container = st.empty()
        
        # Show initial state
        self._render_progress()
    
    def start_stage(self, stage_index: int, custom_message: str = None):
        """
        Start a new stage of the pipeline.
        
        Args:
            stage_index: Index of the stage (0-2)
            custom_message: Optional custom message for this stage
        """
        self.current_stage = stage_index
        stage = self.stages[stage_index]
        
        message = custom_message or stage["description"]
        
        with self.status_container.container():
            display_progress_card(
                stage["name"], 
                f"{stage['emoji']} {message}",
                (stage_index + 0.5) / len(self.stages)
            )
        
        self._render_progress()
    
    def complete_stage(self, stage_index: int, success_message: str = None, 
                      timing_info: Dict[str, Any] = None):
        """
        Mark a stage as completed.
        
        Args:
            stage_index: Index of the completed stage
            success_message: Optional success message
            timing_info: Optional timing information
        """
        stage = self.stages[stage_index]
        message = success_message or f"{stage['name']} completed successfully"
        
        self.current_stage = stage_index
        self._render_progress()
        
        # Show success message
        if success_message:
            display_success_message(success_message, timing_info)
    
    def error_in_stage(self, stage_index: int, error_info: ErrorInfo):
        """
        Mark a stage as failed with error information.
        
        Args:
            stage_index: Index of the failed stage
            error_info: Error information to display
        """
        self.error_stage = stage_index
        self.current_stage = stage_index
        
        # Update progress display to show error
        self._render_progress()
        
        # Show error details
        display_error_card(error_info)
    
    def complete_pipeline(self, total_time: float, summary_stats: Dict[str, Any]):
        """
        Mark the entire pipeline as completed.
        
        Args:
            total_time: Total processing time in seconds
            summary_stats: Summary statistics for the run
        """
        self.current_stage = len(self.stages)
        self._render_progress()
        
        # Show completion message
        display_success_message(
            "Pipeline completed successfully! 🎉",
            {
                "Total Time": f"{total_time:.1f}s",
                **summary_stats
            }
        )
    
    def _render_progress(self):
        """Render the current progress state."""
        if not self.progress_container:
            return
        
        with self.progress_container.container():
            # Stage indicator
            cols = st.columns(len(self.stages))
            
            for i, stage in enumerate(self.stages):
                with cols[i]:
                    if i == self.error_stage:
                        # Error state
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 2rem;">❌</div>
                            <div style="font-weight: bold; color: red;">{stage['name']}</div>
                            <div style="font-size: 0.8rem; color: red;">Failed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif i < self.current_stage:
                        # Completed state
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 2rem;">✅</div>
                            <div style="font-weight: bold; color: green;">{stage['name']}</div>
                            <div style="font-size: 0.8rem; color: green;">Completed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif i == self.current_stage:
                        # Current/active state
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 2rem;">{stage['emoji']}</div>
                            <div style="font-weight: bold; color: blue;">{stage['name']}</div>
                            <div style="font-size: 0.8rem; color: blue;">Processing...</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Pending state
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 2rem;">⏳</div>
                            <div style="font-weight: bold; color: gray;">{stage['name']}</div>
                            <div style="font-size: 0.8rem; color: gray;">Pending</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Overall progress bar
            if self.current_stage >= 0:
                progress = min(self.current_stage / len(self.stages), 1.0)
                st.progress(progress)


def display_processing_stats(stats: Dict[str, Any]):
    """
    Display processing statistics in a clean format.
    
    Args:
        stats: Dictionary containing processing statistics
    """
    st.markdown("### 📊 Processing Statistics")
    
    # Create columns for different stat categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Timing**")
        if 'total_time' in stats:
            st.metric("Total Time", f"{stats['total_time']:.1f}s")
        if 'entity_time' in stats:
            st.metric("Entity Extraction", f"{stats['entity_time']:.1f}s")
    
    with col2:
        st.markdown("**Content**") 
        if 'input_length' in stats:
            st.metric("Input Length", f"{stats['input_length']} chars")
        if 'entity_count' in stats:
            st.metric("Entities Found", stats['entity_count'])
    
    with col3:
        st.markdown("**Results**")
        if 'triple_count' in stats:
            st.metric("Triples Generated", stats['triple_count'])
        if 'judgment_count' in stats:
            st.metric("Triples Judged", stats['judgment_count'])


def display_api_status_indicator(api_name: str, status: str, response_time: float = None):
    """
    Display API status indicator.
    
    Args:
        api_name: Name of the API service
        status: Status string ("healthy", "slow", "error")
        response_time: Optional response time in seconds
    """
    status_config = {
        "healthy": {"emoji": "🟢", "color": "green"},
        "slow": {"emoji": "🟡", "color": "orange"},
        "error": {"emoji": "🔴", "color": "red"}
    }
    
    config = status_config.get(status, status_config["error"])
    
    with st.sidebar:
        st.markdown(f"""
        **API Status: {api_name}**
        
        {config['emoji']} Status: <span style="color: {config['color']}">{status.title()}</span>
        """, unsafe_allow_html=True)
        
        if response_time:
            st.caption(f"Response time: {response_time:.2f}s")


class ErrorRecoveryHelper:
    """
    Helper class for providing error recovery suggestions and actions.
    """
    
    @staticmethod
    def suggest_recovery_actions(error_info: ErrorInfo) -> List[Dict[str, Any]]:
        """
        Suggest specific recovery actions based on error type.
        
        Args:
            error_info: The error information
            
        Returns:
            List of recovery action dictionaries
        """
        actions = []
        
        if error_info.error_type == ErrorType.API_RATE_LIMIT:
            actions.append({
                "action": "Wait and Retry",
                "description": "Wait 30 seconds and try again", 
                "automatic": True,
                "wait_time": 30
            })
        
        elif error_info.error_type == ErrorType.INPUT_VALIDATION:
            actions.append({
                "action": "Edit Input",
                "description": "Modify your input text and try again",
                "automatic": False
            })
        
        elif error_info.error_type == ErrorType.API_SERVER:
            actions.append({
                "action": "Retry Later",
                "description": "The service may be temporarily unavailable",
                "automatic": False
            })
        
        elif error_info.error_type == ErrorType.PROCESSING:
            actions.append({
                "action": "Simplify Input",
                "description": "Try with shorter or simpler text",
                "automatic": False
            })
            
        return actions
    
    @staticmethod
    def display_recovery_options(error_info: ErrorInfo):
        """Display interactive recovery options in Streamlit."""
        actions = ErrorRecoveryHelper.suggest_recovery_actions(error_info)
        
        if not actions:
            return
        
        st.markdown("### 🛠️ Recovery Options")
        
        for action in actions:
            with st.expander(f"💡 {action['action']}"):
                st.write(action['description'])
                
                if action.get('automatic'):
                    if st.button(f"Execute {action['action']}", key=f"recovery_{action['action']}"):
                        if action.get('wait_time'):
                            with st.spinner(f"Waiting {action['wait_time']} seconds..."):
                                st.time.sleep(action['wait_time'])
                            st.rerun()
                else:
                    st.info("Manual action required - please make changes and try again.")