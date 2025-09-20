"""
Simple progress tracker for real-time item-level progress display.

This module provides a simplified progress tracking system that focuses on
showing actual processing counts (e.g., "1/27 entities") instead of verbose logs.
"""

import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
import streamlit as st

try:
    from streamlit_pipeline.utils.i18n import get_text
except ImportError:
    from ..utils.i18n import get_text


class SimpleProgressTracker:
    """
    Simple progress tracker that shows actual item counts and processing progress.

    Replaces the verbose DetailedProgressTracker with clean, focused progress display.
    """

    def __init__(self):
        """Initialize the simple progress tracker."""
        self.start_time = time.time()
        self.current_phase = None
        self.phase_start_time = None
        self.progress_container = None
        self.status_container = None

    def initialize_display(self) -> tuple:
        """
        Initialize the simple progress display containers.

        Returns:
            tuple: (progress_container, status_container)
        """
        st.markdown("## 🔄 Processing Progress")

        # Create simple containers for progress display
        self.progress_container = st.empty()
        self.status_container = st.empty()

        return self.progress_container, self.status_container

    def start_phase(self, phase_name: str, description: str = "") -> None:
        """
        Start a new processing phase.

        Args:
            phase_name: Name of the processing phase
            description: Brief description of the phase
        """
        self.current_phase = phase_name
        self.phase_start_time = time.time()

        # Simple phase display
        if self.progress_container:
            with self.progress_container.container():
                st.info(f"🔄 **{phase_name}**")
                if description:
                    st.caption(description)

    def update_progress(self, current: int, total: int, item_name: str = "items") -> None:
        """
        Update progress with actual item counts.

        Args:
            current: Current number of processed items
            total: Total number of items to process
            item_name: Name of items being processed (e.g., "entities", "triples")
        """
        if not self.current_phase or not self.status_container:
            return

        progress_percent = current / total if total > 0 else 0

        with self.status_container.container():
            # Progress bar
            st.progress(progress_percent)

            # Item count display
            st.markdown(f"**Processing {item_name}: {current}/{total}**")

            # Timing information
            if self.phase_start_time:
                elapsed = time.time() - self.phase_start_time
                if current > 0:
                    avg_time = elapsed / current
                    estimated_remaining = avg_time * (total - current)
                    st.caption(f"⏱️ Elapsed: {elapsed:.1f}s | Estimated remaining: {estimated_remaining:.1f}s")
                else:
                    st.caption(f"⏱️ Elapsed: {elapsed:.1f}s")

    def finish_phase(self, success: bool = True, final_message: str = "") -> None:
        """
        Finish the current processing phase.

        Args:
            success: Whether the phase completed successfully
            final_message: Optional final message for the phase
        """
        if not self.current_phase:
            return

        phase_time = time.time() - self.phase_start_time if self.phase_start_time else 0

        # Simple completion display
        if self.progress_container:
            with self.progress_container.container():
                if success:
                    st.success(f"✅ {self.current_phase} completed in {phase_time:.2f}s")
                else:
                    st.error(f"❌ {self.current_phase} failed after {phase_time:.2f}s")

                if final_message:
                    st.caption(final_message)

        # Clear status container
        if self.status_container:
            self.status_container.empty()

    def clear_display(self) -> None:
        """Clear all progress displays."""
        if self.progress_container:
            self.progress_container.empty()
        if self.status_container:
            self.status_container.empty()


def display_entity_processing(text_length: int, progress_callback: Optional[Callable] = None) -> SimpleProgressTracker:
    """
    Display entity extraction progress with real-time updates.

    Args:
        text_length: Length of text being processed
        progress_callback: Optional callback for progress updates

    Returns:
        SimpleProgressTracker instance for continued use
    """
    tracker = SimpleProgressTracker()
    progress_container, status_container = tracker.initialize_display()

    tracker.start_phase(
        "🔍 Entity Extraction & Text Denoising",
        "Extracting entities from Chinese text using GPT-5-mini"
    )

    return tracker


def display_triple_generation_processing(entities_count: int, progress_callback: Optional[Callable] = None) -> SimpleProgressTracker:
    """
    Display triple generation progress with real-time updates.

    Args:
        entities_count: Number of entities to process
        progress_callback: Optional callback for progress updates

    Returns:
        SimpleProgressTracker instance
    """
    tracker = SimpleProgressTracker()
    progress_container, status_container = tracker.initialize_display()

    tracker.start_phase(
        "🔗 Triple Generation",
        f"Generating knowledge triples from {entities_count} entities"
    )

    return tracker


def display_graph_judgment_processing(triples_count: int, progress_callback: Optional[Callable] = None) -> SimpleProgressTracker:
    """
    Display graph judgment progress with real-time updates.

    Args:
        triples_count: Number of triples to judge
        progress_callback: Optional callback for progress updates

    Returns:
        SimpleProgressTracker instance
    """
    tracker = SimpleProgressTracker()
    progress_container, status_container = tracker.initialize_display()

    tracker.start_phase(
        "⚖️ Graph Judgment",
        f"Validating {triples_count} triples using Perplexity AI"
    )

    return tracker


def display_simple_processing_summary(tracker: SimpleProgressTracker, pipeline_result) -> None:
    """
    Display a simple processing summary focused on key metrics.

    Args:
        tracker: The SimpleProgressTracker instance
        pipeline_result: The pipeline result to summarize
    """
    if not pipeline_result:
        return

    st.markdown(f"## {get_text('processing.processing_complete')}")

    # Simple summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if pipeline_result.entity_result:
            st.metric("Entities Found", len(pipeline_result.entity_result.entities))

    with col2:
        if pipeline_result.triple_result:
            st.metric("Triples Generated", len(pipeline_result.triple_result.triples))

    with col3:
        if pipeline_result.judgment_result:
            approved = sum(pipeline_result.judgment_result.judgments)
            st.metric("Triples Approved", approved)

    with col4:
        total_time = getattr(pipeline_result, 'total_time', 0)
        st.metric("Total Time", f"{total_time:.1f}s")

    # Simple success message
    if pipeline_result.success:
        st.success("🎉 Knowledge graph construction completed successfully!")
    else:
        st.error(f"❌ Processing failed: {pipeline_result.error}")