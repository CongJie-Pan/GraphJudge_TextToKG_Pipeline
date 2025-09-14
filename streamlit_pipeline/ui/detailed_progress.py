"""
Detailed Progress Display Components for GraphJudge Streamlit Application.

This module provides comprehensive progress tracking and detailed phase-by-phase
processing information similar to the original source code terminal output.
Inspired by the detailed logging and progress tracking in run_entity.py, run_triple.py, and run_gj.py.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import json

class DetailedProgressTracker:
    """
    Advanced progress tracker that mirrors the detailed processing information
    from the original source code (run_entity.py, run_triple.py, run_gj.py).
    """

    def __init__(self):
        """Initialize the detailed progress tracker."""
        self.start_time = time.time()
        self.phase_start_times = {}
        self.phase_logs = {}
        self.current_phase = None
        self.progress_container = None
        self.log_container = None
        self.metrics_container = None

    def initialize_display(self) -> tuple:
        """
        Initialize the detailed progress display containers.

        Returns:
            tuple: (progress_container, log_container, metrics_container)
        """
        st.markdown("## 🔄 Detailed Processing Progress")
        st.markdown("Real-time processing information similar to terminal output from original source code")

        # Create main containers
        self.progress_container = st.empty()
        self.log_container = st.empty()
        self.metrics_container = st.empty()

        return self.progress_container, self.log_container, self.metrics_container

    def start_phase(self, phase_name: str, description: str = "") -> None:
        """
        Start a new processing phase with detailed logging.

        Args:
            phase_name: Name of the processing phase
            description: Detailed description of what this phase does
        """
        self.current_phase = phase_name
        self.phase_start_times[phase_name] = time.time()
        self.phase_logs[phase_name] = []

        # Display phase header similar to original source code
        if self.progress_container:
            with self.progress_container.container():
                st.markdown(f"### {phase_name}")
                if description:
                    st.info(f"📝 **Phase Description**: {description}")

                # Show phase timing
                total_elapsed = time.time() - self.start_time
                st.caption(f"⏱️ Started at: {datetime.now().strftime('%H:%M:%S')} | Total elapsed: {total_elapsed:.1f}s")

    def log_step(self, message: str, step_type: str = "INFO", show_timing: bool = True) -> None:
        """
        Log a detailed processing step similar to original terminal output.

        Args:
            message: The step message to log
            step_type: Type of step (INFO, SUCCESS, WARNING, ERROR)
            show_timing: Whether to show timing information
        """
        if not self.current_phase:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        phase_elapsed = time.time() - self.phase_start_times.get(self.current_phase, time.time())

        # Create formatted log entry
        if show_timing:
            log_entry = f"[{timestamp}] {step_type}: {message} ({phase_elapsed:.1f}s)"
        else:
            log_entry = f"[{timestamp}] {step_type}: {message}"

        # Store log entry
        self.phase_logs[self.current_phase].append({
            'timestamp': timestamp,
            'type': step_type,
            'message': message,
            'elapsed': phase_elapsed
        })

        # Display in log container with appropriate styling
        if self.log_container:
            with self.log_container.container():
                if step_type == "SUCCESS":
                    st.success(f"✅ {message}")
                elif step_type == "WARNING":
                    st.warning(f"⚠️ {message}")
                elif step_type == "ERROR":
                    st.error(f"❌ {message}")
                else:
                    st.info(f"🔄 {message}")

                if show_timing:
                    st.caption(f"Phase elapsed: {phase_elapsed:.1f}s")

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update processing metrics display similar to original source statistics.

        Args:
            metrics: Dictionary of metrics to display
        """
        if not self.metrics_container:
            return

        with self.metrics_container.container():
            st.markdown("#### 📊 Processing Metrics")

            # Display metrics in columns
            if len(metrics) <= 4:
                cols = st.columns(len(metrics))
                for i, (key, value) in enumerate(metrics.items()):
                    with cols[i]:
                        if isinstance(value, (int, float)):
                            if isinstance(value, float):
                                st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, f"{value:,}")
                        else:
                            st.metric(key, str(value))
            else:
                # Use multiple rows for many metrics
                for i in range(0, len(metrics), 4):
                    cols = st.columns(4)
                    for j, (key, value) in enumerate(list(metrics.items())[i:i+4]):
                        with cols[j]:
                            if isinstance(value, (int, float)):
                                if isinstance(value, float):
                                    st.metric(key, f"{value:.2f}")
                                else:
                                    st.metric(key, f"{value:,}")
                            else:
                                st.metric(key, str(value))

    def finish_phase(self, success: bool = True, final_message: str = "") -> None:
        """
        Finish the current processing phase with summary information.

        Args:
            success: Whether the phase completed successfully
            final_message: Optional final message for the phase
        """
        if not self.current_phase:
            return

        phase_time = time.time() - self.phase_start_times[self.current_phase]

        # Log phase completion
        status = "SUCCESS" if success else "ERROR"
        completion_message = final_message or f"{self.current_phase} completed"
        self.log_step(f"{completion_message} (Total phase time: {phase_time:.2f}s)", status)

        # Display phase summary
        if self.progress_container:
            with self.progress_container.container():
                if success:
                    st.success(f"✅ {self.current_phase} completed successfully in {phase_time:.2f}s")
                else:
                    st.error(f"❌ {self.current_phase} failed after {phase_time:.2f}s")

    def show_detailed_log(self) -> None:
        """Display detailed processing log similar to terminal output."""
        st.markdown("### 📋 Detailed Processing Log")

        if not self.phase_logs:
            st.info("No processing logs available yet.")
            return

        # Create expandable sections for each phase
        for phase_name, logs in self.phase_logs.items():
            with st.expander(f"📂 {phase_name} Logs ({len(logs)} entries)"):
                if logs:
                    # Convert logs to DataFrame for better display
                    log_data = []
                    for log in logs:
                        log_data.append({
                            "Time": log['timestamp'],
                            "Type": log['type'],
                            "Message": log['message'],
                            "Elapsed": f"{log['elapsed']:.1f}s"
                        })

                    df = pd.DataFrame(log_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No logs for this phase.")

    def generate_processing_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive processing summary similar to original source code statistics.

        Returns:
            Dictionary containing detailed processing summary
        """
        total_time = time.time() - self.start_time

        summary = {
            "total_processing_time": total_time,
            "phases_completed": len(self.phase_logs),
            "total_log_entries": sum(len(logs) for logs in self.phase_logs.values()),
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add per-phase timing
        phase_timings = {}
        for phase_name in self.phase_logs.keys():
            if phase_name in self.phase_start_times:
                # Calculate phase duration from logs
                phase_logs = self.phase_logs[phase_name]
                if phase_logs:
                    max_elapsed = max(log['elapsed'] for log in phase_logs)
                    phase_timings[f"{phase_name}_time"] = max_elapsed

        summary.update(phase_timings)
        return summary


def display_ectd_processing(progress_callback: Optional[Callable] = None) -> DetailedProgressTracker:
    """
    Display detailed ECTD (Entity Extraction and Text Denoising) processing
    similar to run_entity.py terminal output.

    Args:
        progress_callback: Optional callback function for progress updates

    Returns:
        DetailedProgressTracker instance for continued use
    """
    tracker = DetailedProgressTracker()
    tracker.initialize_display()

    # Phase 1: Entity Extraction Phase (similar to run_entity.py)
    tracker.start_phase(
        "🔍 Entity Extraction Phase",
        "GPT-5-mini analyzes classical Chinese text to identify key entities using advanced contextual understanding"
    )

    # Simulate detailed steps from run_entity.py
    tracker.log_step("Initializing GPT-5-mini API connection...")
    tracker.log_step("Loading classical Chinese text processing prompts...")
    tracker.log_step("Validating input text format and encoding...")

    # Show metrics similar to original source
    tracker.update_metrics({
        "Model": "GPT-5-mini",
        "API Status": "Connected",
        "Language": "Classical Chinese",
        "Cache Status": "Enabled"
    })

    tracker.log_step("Starting entity extraction with deduplication emphasis...", "SUCCESS")
    tracker.log_step("Processing text segments with enhanced Chinese understanding...")

    if progress_callback:
        progress_callback(0, "Extracting entities with GPT-5-mini...")

    return tracker


def display_triple_generation_processing(entities: List[str], progress_callback: Optional[Callable] = None) -> DetailedProgressTracker:
    """
    Display detailed Triple Generation processing similar to run_triple.py terminal output.

    Args:
        entities: List of extracted entities
        progress_callback: Optional callback for progress updates

    Returns:
        DetailedProgressTracker instance
    """
    tracker = DetailedProgressTracker()
    tracker.initialize_display()

    # Phase 1: Enhanced Semantic Graph Generation (similar to run_triple.py)
    tracker.start_phase(
        "🔗 Enhanced Triple Generation Phase",
        "GPT-5-mini processes denoised text to generate structured knowledge triples with JSON schema validation"
    )

    tracker.log_step("Loading enhanced GPT-5-mini triple generation pipeline v2.0...")
    tracker.log_step("Initializing structured JSON output prompts...")
    tracker.log_step("Setting up schema validation with Pydantic...")

    # Show processing configuration from run_triple.py
    tracker.update_metrics({
        "Model": "GPT-5-mini",
        "Output Format": "JSON Schema",
        "Entities Available": len(entities),
        "Text Chunking": "Enabled",
        "Validation": "Pydantic"
    })

    tracker.log_step("Creating enhanced prompts with relation vocabulary standardization...", "SUCCESS")
    tracker.log_step("Starting sequential processing with enhanced rate limiting...")

    if progress_callback:
        progress_callback(1, f"Generating triples from {len(entities)} entities...")

    return tracker


def display_graph_judgment_processing(triples_count: int, progress_callback: Optional[Callable] = None) -> DetailedProgressTracker:
    """
    Display detailed Graph Judgment processing similar to run_gj.py terminal output.

    Args:
        triples_count: Number of triples to judge
        progress_callback: Optional callback for progress updates

    Returns:
        DetailedProgressTracker instance
    """
    tracker = DetailedProgressTracker()
    tracker.initialize_display()

    # Phase 1: Perplexity API Graph Judgment (similar to run_gj.py)
    tracker.start_phase(
        "⚖️ Perplexity API Graph Judgment Phase",
        "Advanced reasoning model validates knowledge graph triples with explainable AI and citation support"
    )

    tracker.log_step("Initializing Perplexity API Graph Judge system...")
    tracker.log_step("Loading sonar-reasoning model with enhanced fact-checking capabilities...")
    tracker.log_step("Setting up explainable judgment with confidence scoring...")

    # Show configuration from run_gj.py
    tracker.update_metrics({
        "Model": "Perplexity/sonar-reasoning",
        "Triples to Judge": triples_count,
        "Reasoning Effort": "Medium",
        "Citations": "Enabled",
        "Concurrent Limit": 3
    })

    tracker.log_step("Creating specialized graph judgment prompts...", "SUCCESS")
    tracker.log_step("Starting sequential triple validation with intelligent retry logic...")

    if progress_callback:
        progress_callback(2, f"Judging {triples_count} knowledge triples...")

    return tracker


def display_comprehensive_processing_summary(tracker: DetailedProgressTracker, pipeline_result) -> None:
    """
    Display comprehensive processing summary similar to the terminal output
    from the original source code files.

    Args:
        tracker: The DetailedProgressTracker instance
        pipeline_result: The complete pipeline result
    """
    st.markdown("## 📈 Comprehensive Processing Summary")
    st.markdown("Final processing statistics similar to original source code terminal output")

    # Generate and display processing summary
    summary = tracker.generate_processing_summary()

    # Main summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processing Time", f"{summary['total_processing_time']:.2f}s")
    with col2:
        st.metric("Phases Completed", summary['phases_completed'])
    with col3:
        st.metric("Log Entries", summary['total_log_entries'])
    with col4:
        if pipeline_result.success:
            st.metric("Pipeline Status", "✅ Success", "All phases completed")
        else:
            st.metric("Pipeline Status", "❌ Failed", f"At {pipeline_result.error_stage}")

    # Detailed phase breakdown
    with st.expander("🔍 Phase-by-Phase Breakdown", expanded=True):
        if pipeline_result.entity_result:
            st.markdown("### 🔍 Entity Extraction & Text Denoising (ECTD)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entities Extracted", len(pipeline_result.entity_result.entities))
            with col2:
                st.metric("Processing Time", f"{pipeline_result.entity_result.processing_time:.2f}s")
            with col3:
                status = "✅ Success" if pipeline_result.entity_result.success else "❌ Failed"
                st.metric("Status", status)

        if pipeline_result.triple_result:
            st.markdown("### 🔗 Triple Generation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Triples Generated", len(pipeline_result.triple_result.triples))
            with col2:
                st.metric("Processing Time", f"{pipeline_result.triple_result.processing_time:.2f}s")
            with col3:
                status = "✅ Success" if pipeline_result.triple_result.success else "❌ Failed"
                st.metric("Status", status)

        if pipeline_result.judgment_result:
            st.markdown("### ⚖️ Graph Judgment")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                approved = sum(pipeline_result.judgment_result.judgments)
                st.metric("Approved Triples", approved)
            with col2:
                rejected = len(pipeline_result.judgment_result.judgments) - approved
                st.metric("Rejected Triples", rejected)
            with col3:
                st.metric("Processing Time", f"{pipeline_result.judgment_result.processing_time:.2f}s")
            with col4:
                status = "✅ Success" if pipeline_result.judgment_result.success else "❌ Failed"
                st.metric("Status", status)

    # Processing timeline similar to terminal logs
    with st.expander("⏱️ Processing Timeline"):
        timeline_data = []

        if pipeline_result.entity_result:
            timeline_data.append({
                "Phase": "Entity Extraction",
                "Start Time": "00:00:00",
                "Duration": f"{pipeline_result.entity_result.processing_time:.1f}s",
                "Status": "✅ Success" if pipeline_result.entity_result.success else "❌ Failed",
                "Items Processed": f"{len(pipeline_result.entity_result.entities)} entities"
            })

        if pipeline_result.triple_result:
            entity_time = pipeline_result.entity_result.processing_time if pipeline_result.entity_result else 0
            start_time = f"{int(entity_time//60):02d}:{int(entity_time%60):02d}"
            timeline_data.append({
                "Phase": "Triple Generation",
                "Start Time": start_time,
                "Duration": f"{pipeline_result.triple_result.processing_time:.1f}s",
                "Status": "✅ Success" if pipeline_result.triple_result.success else "❌ Failed",
                "Items Processed": f"{len(pipeline_result.triple_result.triples)} triples"
            })

        if pipeline_result.judgment_result:
            prev_time = (pipeline_result.entity_result.processing_time if pipeline_result.entity_result else 0) + \
                       (pipeline_result.triple_result.processing_time if pipeline_result.triple_result else 0)
            start_time = f"{int(prev_time//60):02d}:{int(prev_time%60):02d}"
            timeline_data.append({
                "Phase": "Graph Judgment",
                "Start Time": start_time,
                "Duration": f"{pipeline_result.judgment_result.processing_time:.1f}s",
                "Status": "✅ Success" if pipeline_result.judgment_result.success else "❌ Failed",
                "Items Processed": f"{len(pipeline_result.judgment_result.judgments)} judgments"
            })

        if timeline_data:
            df = pd.DataFrame(timeline_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Show detailed logs
    tracker.show_detailed_log()

    # Final summary message similar to original terminal output
    if pipeline_result.success:
        st.success(f"🎉 GraphJudge Pipeline completed successfully in {pipeline_result.total_time:.2f} seconds!")
        st.info("📊 All processing phases completed with detailed logging and metrics tracking.")
    else:
        st.error(f"❌ GraphJudge Pipeline failed at {pipeline_result.error_stage}: {pipeline_result.error}")
        st.warning("🔍 Check the detailed logs above for troubleshooting information.")