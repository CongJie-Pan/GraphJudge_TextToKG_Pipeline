"""
Demo Script for Enhanced Streamlit Pipeline with Detailed Processing Visualization

This script demonstrates the enhanced streamlit pipeline with detailed phase-by-phase
processing information, similar to the original source code terminal output.

Run this with: streamlit run demo_enhanced_pipeline.py
"""

import streamlit as st
import time
from datetime import datetime

# Import our enhanced components
from ui.detailed_progress import (
    DetailedProgressTracker, display_comprehensive_processing_summary,
    display_ectd_processing, display_triple_generation_processing,
    display_graph_judgment_processing
)
from utils.i18n import get_text

from ui.components import (
    display_input_section, display_entity_results, display_triple_results,
    display_judgment_results
)

# Mock data for demonstration
from core.models import EntityResult, TripleResult, JudgmentResult, Triple, PipelineState
from core.pipeline import PipelineResult

st.set_page_config(
    page_title=get_text('demo.title'),
    page_icon="🧠",
    layout="wide"
)

st.title(get_text('demo.title'))
st.markdown(get_text('demo.description'))

# Demo modes
demo_mode = st.selectbox(
    "Select Demo Mode",
    ["Interactive Demo", "Entity Extraction Demo", "Triple Generation Demo", "Graph Judgment Demo", "Full Pipeline Demo"]
)

if demo_mode == "Interactive Demo":
    st.markdown("## Interactive Demo")
    st.markdown("Upload text or enter text to see the enhanced processing display:")

    # Use our enhanced input section
    input_text = display_input_section()

    if input_text:
        st.markdown("### Processing Preview")
        st.info("In the real pipeline, this would show detailed processing phases like the original source code.")

        # Show what the detailed processing would look like
        with st.expander("🔬 Enhanced Processing Phases (Preview)", expanded=True):
            st.markdown("#### Phase 1: Entity Extraction & Text Denoising (ECTD)")
            st.code("""
[12:34:56] INFO: Initializing GPT-5-mini API connection...
[12:34:57] INFO: Loading enhanced Chinese text processing prompts...
[12:34:58] SUCCESS: Starting entity extraction with deduplication emphasis...
[12:35:02] INFO: Processing text segments with enhanced Chinese understanding...
[12:35:08] SUCCESS: Entity extraction completed - Found 15 entities
            """, language="text")

            st.markdown("#### Phase 2: Triple Generation")
            st.code("""
[12:35:09] INFO: Loading enhanced triple generation pipeline v2.0...
[12:35:10] INFO: Setting up structured JSON output prompts...
[12:35:11] SUCCESS: Starting relation extraction with schema validation...
[12:35:18] INFO: Processing with GPT-5-mini sequential mode...
[12:35:25] SUCCESS: Generated 23 validated triples
            """, language="text")

            st.markdown("#### Phase 3: Graph Judgment")
            st.code("""
[12:35:26] INFO: Initializing Perplexity API Graph Judge system...
[12:35:27] INFO: Loading sonar-reasoning model with fact-checking capabilities...
[12:35:28] SUCCESS: Starting triple validation with confidence scoring...
[12:35:35] INFO: Processing 23 triples with explainable AI...
[12:35:42] SUCCESS: Graph judgment completed - Approved: 18, Rejected: 5
            """, language="text")

elif demo_mode == "Entity Extraction Demo":
    st.markdown("## Entity Extraction Demo")

    # Create mock entity result
    mock_entities = ["甄士隱", "封氏", "姑蘇城", "葫蘆廟", "賈雨村", "胡州", "鄉宦"]
    mock_denoised = "甄士隱是姑蘇城內的鄉宦。甄士隱的妻子是封氏。賈雨村是胡州人氏，也是詩書仕宦之族。"

    entity_result = EntityResult(
        entities=mock_entities,
        denoised_text=mock_denoised,
        success=True,
        processing_time=5.2
    )

    # Store original input for comparison
    st.session_state.original_input = "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"

    display_entity_results(entity_result)

elif demo_mode == "Triple Generation Demo":
    st.markdown("## Triple Generation Demo")

    # Create mock triple results
    mock_triples = [
        Triple(subject="甄士隱", predicate="職業", object="鄉宦"),
        Triple(subject="甄士隱", predicate="妻子", object="封氏"),
        Triple(subject="甄士隱", predicate="地點", object="姑蘇城"),
        Triple(subject="賈雨村", predicate="出身", object="胡州"),
        Triple(subject="封氏", predicate="性格", object="賢淑"),
    ]

    triple_result = TripleResult(
        triples=mock_triples,
        metadata={"chunks_processed": 3, "validation_score": 0.89},
        success=True,
        processing_time=8.7
    )

    display_triple_results(triple_result)

elif demo_mode == "Graph Judgment Demo":
    st.markdown("## Graph Judgment Demo")

    # Create mock judgment results
    mock_triples = [
        Triple(subject="甄士隱", predicate="職業", object="鄉宦"),
        Triple(subject="甄士隱", predicate="妻子", object="封氏"),
        Triple(subject="甄士隱", predicate="地點", object="姑蘇城"),
        Triple(subject="賈雨村", predicate="出身", object="胡州"),
        Triple(subject="封氏", predicate="性格", object="賢淑"),
    ]

    judgment_result = JudgmentResult(
        judgments=[True, True, True, False, True],
        confidence=[0.92, 0.88, 0.85, 0.65, 0.90],
        explanations=[
            "這是正確的，甄士隱確實是鄉宦身份",
            "根據原文，封氏確實是甄士隱的妻子",
            "甄士隱住在姑蘇城內是正確的",
            "這個關係表述不夠準確",
            "封氏的性格描述與原文相符"
        ],
        success=True,
        processing_time=12.3
    )

    display_judgment_results(judgment_result, mock_triples)

elif demo_mode == "Full Pipeline Demo":
    st.markdown("## Full Pipeline Demo")
    st.markdown("Complete pipeline processing simulation with detailed tracking")

    if st.button("🚀 Run Full Pipeline Demo"):
        # Initialize detailed progress tracker
        tracker = DetailedProgressTracker()
        progress_container, log_container, metrics_container = tracker.initialize_display()

        # Simulate ECTD Phase
        tracker.start_phase(
            "🔍 Entity Extraction & Text Denoising (ECTD)",
            "GPT-5-mini analyzes classical Chinese text to identify key entities and denoise content"
        )

        tracker.log_step("Initializing GPT-5-mini API connection...")
        time.sleep(1)
        tracker.log_step("Loading enhanced Chinese text processing prompts...")
        time.sleep(1)
        tracker.log_step("Starting entity extraction with deduplication emphasis...", "SUCCESS")

        tracker.update_metrics({
            "Model": "GPT-5-mini",
            "Language": "Classical Chinese",
            "Phase": "Entity Extraction",
            "Entities Found": 7
        })

        time.sleep(2)
        tracker.finish_phase(True, "Entity extraction completed successfully")

        # Simulate Triple Generation Phase
        tracker.start_phase(
            "🔗 Enhanced Triple Generation",
            "GPT-5-mini processes denoised text to generate structured knowledge triples with JSON validation"
        )

        tracker.log_step("Loading enhanced triple generation pipeline v2.0...")
        time.sleep(1)
        tracker.log_step("Setting up structured JSON output prompts...")
        time.sleep(1)
        tracker.log_step("Starting relation extraction with schema validation...", "SUCCESS")

        tracker.update_metrics({
            "Model": "GPT-5-mini",
            "Output Format": "JSON Schema",
            "Phase": "Triple Generation",
            "Triples Generated": 5
        })

        time.sleep(3)
        tracker.finish_phase(True, "Triple generation completed successfully")

        # Simulate Graph Judgment Phase
        tracker.start_phase(
            "⚖️ Perplexity API Graph Judgment",
            "Advanced reasoning model validates knowledge graph triples with explainable AI"
        )

        tracker.log_step("Initializing Perplexity API Graph Judge system...")
        time.sleep(1)
        tracker.log_step("Loading sonar-reasoning model with fact-checking capabilities...")
        time.sleep(1)
        tracker.log_step("Starting triple validation with confidence scoring...", "SUCCESS")

        tracker.update_metrics({
            "Model": "Perplexity/sonar-reasoning",
            "Reasoning": "Advanced",
            "Phase": "Graph Judgment",
            "Approved": 4,
            "Rejected": 1
        })

        time.sleep(3)
        tracker.finish_phase(True, "Graph judgment completed successfully")

        # Create mock pipeline result for summary
        mock_entity_result = EntityResult(
            entities=["甄士隱", "封氏", "姑蘇城", "葫蘆廟", "賈雨村", "胡州", "鄉宦"],
            denoised_text="甄士隱是姑蘇城內的鄉宦。甄士隱的妻子是封氏。",
            success=True,
            processing_time=5.2
        )

        mock_triple_result = TripleResult(
            triples=[
                Triple(subject="甄士隱", predicate="職業", object="鄉宦"),
                Triple(subject="甄士隱", predicate="妻子", object="封氏"),
                Triple(subject="甄士隱", predicate="地點", object="姑蘇城"),
                Triple(subject="賈雨村", predicate="出身", object="胡州"),
                Triple(subject="封氏", predicate="性格", object="賢淑"),
            ],
            metadata={"chunks_processed": 3},
            success=True,
            processing_time=8.7
        )

        mock_judgment_result = JudgmentResult(
            judgments=[True, True, True, False, True],
            confidence=[0.92, 0.88, 0.85, 0.65, 0.90],
            explanations=None,
            success=True,
            processing_time=12.3
        )

        mock_pipeline_result = PipelineResult(
            success=True,
            stage_reached=3,
            total_time=26.2,
            entity_result=mock_entity_result,
            triple_result=mock_triple_result,
            judgment_result=mock_judgment_result
        )

        # Display comprehensive summary
        display_comprehensive_processing_summary(tracker, mock_pipeline_result)

        st.success("🎉 Full pipeline demo completed!")

# Footer
st.markdown("---")
st.markdown("**Enhanced GraphJudge Pipeline** - Detailed processing visualization inspired by original source code")
st.caption(f"Demo generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")