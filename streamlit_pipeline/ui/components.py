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
import re

# Optional plotly import with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    # Try absolute import first (for package installation)
    from streamlit_pipeline.core.models import EntityResult, TripleResult, JudgmentResult, Triple
    from streamlit_pipeline.core.pipeline import PipelineResult
except ImportError:
    # Fallback to relative imports (for direct execution)
    from ..core.models import EntityResult, TripleResult, JudgmentResult, Triple
    from ..core.pipeline import PipelineResult
from .error_display import display_success_message, display_processing_stats

# Import i18n functionality
try:
    from streamlit_pipeline.utils.i18n import get_text, get_supported_languages, get_current_language, set_language
except ImportError:
    from ..utils.i18n import get_text, get_supported_languages, get_current_language, set_language


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
        st.error(get_text('errors.file_too_large', size=f"{max_size / (1024*1024):.1f}"))
        return "", "", False

    # Check if file is empty
    if uploaded_file.size == 0:
        st.error(get_text('errors.file_empty'))
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
            st.error(get_text('errors.file_read_error', encoding=encoding, error=str(e)))
            continue

    return "", "", False


def display_input_section() -> str:
    """
    Display the main input section for file upload or text entry.

    Returns:
        The input text from the user (either uploaded or typed)
    """
    st.markdown(f"## {get_text('input.title')}")
    st.markdown(get_text('input.description'))

    # Create tabs for different input methods
    tab1, tab2 = st.tabs([get_text('input.tab_file'), get_text('input.tab_text')])

    input_text = ""

    with tab1:
        # File upload section
        # NOTE: Streamlit's st.file_uploader() widget contains hardcoded English text
        # (like "Drag and drop file here") that cannot be localized. This is a limitation
        # of the Streamlit framework itself, not our application.
        uploaded_file = st.file_uploader(
            get_text('input.choose_file'),
            type=['txt'],
            help=get_text('input.upload_help'),
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Use the validation helper function
            file_content, used_encoding, success = validate_text_file(uploaded_file)

            if success:
                input_text = file_content
                st.success(get_text('file_upload.success_message', encoding=used_encoding))

                # Show file info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(get_text('file_upload.file_name'), uploaded_file.name)
                with col2:
                    st.metric(get_text('file_upload.file_size'), f"{uploaded_file.size:,} {get_text('file_upload.bytes')}")
                with col3:
                    st.metric(get_text('file_upload.encoding_used'), used_encoding)
                with col4:
                    # Count Chinese characters
                    chinese_chars = sum(1 for char in file_content if '\u4e00' <= char <= '\u9fff')
                    st.metric(get_text('file_upload.chinese_chars'), f"{chinese_chars:,}")

                # Show preview of file content
                with st.expander(get_text('file_upload.content_preview')):
                    preview_text = input_text[:500] + "..." if len(input_text) > 500 else input_text
                    st.text_area(
                        get_text('file_upload.preview'),
                        value=preview_text,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed"
                    )

                    # Show additional file analysis
                    if len(input_text) > 500:
                        st.info(get_text('components.showing_first_chars', count=500, total=len(input_text)))

            else:
                st.error(get_text('input.file_error'))

    with tab2:
        # Text area for manual input
        manual_text = st.text_area(
            get_text('input.tab_text'),
            height=200,
            placeholder=get_text('input.text_placeholder'),
            help=get_text('input.text_help'),
            label_visibility="collapsed"
        )

        if manual_text:
            input_text = manual_text.strip()

    # Input statistics (show for both file upload and manual input)
    if input_text:
        st.markdown(f"### {get_text('input.text_stats')}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(get_text('input.character_count'), len(input_text))
        with col2:
            st.metric(get_text('input.word_count'), len(input_text.split('\n')))
        with col3:
            st.metric(get_text('input.paragraph_count'), len([p for p in input_text.split('\n\n') if p.strip()]))
        with col4:
            # Estimate reading time (assuming 300 characters per minute for Chinese)
            reading_time = len(input_text) / 300
            st.metric(get_text('input.reading_time'), f"{reading_time:.1f} {get_text('input.reading_time_unit')}")

    return input_text


def display_entity_results(entity_result: EntityResult, show_expanders: bool = True):
    """
    Display detailed entity extraction results with enhanced processing information.

    Args:
        entity_result: The EntityResult to display
        show_expanders: Whether to create expanders for detailed sections
    """
    st.markdown(f"## {get_text('entity.title')}")

    if entity_result.success:
        # Simple success indicator
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "✅ Success")
        with col2:
            st.metric("Entities Found", len(entity_result.entities))
        with col3:
            st.metric("Processing Time", f"{entity_result.processing_time:.2f}s")

        # Detailed processing phases display
        st.markdown(f"### {get_text('entity.processing_phases')}")
        st.markdown(f"#### {get_text('entity.phase1_title')}")
        st.info(get_text('entity.phase1_description'))

        # Entity categorization and display
        if entity_result.entities:
            st.markdown(f"##### {get_text('entity.extracted_entities')}")

            # Enhanced entity display with categorization
            entity_categories = {
                "👤 人物 (Characters)": [],
                "🏛️ 地點 (Locations)": [],
                "📚 物品 (Objects)": [],
                "💭 概念 (Concepts)": []
            }

            # Smart categorization logic with Chinese character patterns
            for entity in entity_result.entities:
                entity_str = str(entity)
                if any(char in entity_str for char in ['氏', '公', '君', '先生', '夫人', '姓', '名']):
                    entity_categories["👤 人物 (Characters)"].append(entity_str)
                elif any(char in entity_str for char in ['城', '府', '廟', '巷', '街', '州', '門']):
                    entity_categories["🏛️ 地點 (Locations)"].append(entity_str)
                elif any(char in entity_str for char in ['書', '廟', '房', '園', '院', '館']):
                    entity_categories["📚 物品 (Objects)"].append(entity_str)
                else:
                    entity_categories["💭 概念 (Concepts)"].append(entity_str)

            # Display categorized entities with enhanced UI
            cols = st.columns(2)
            col_idx = 0
            for category, entities in entity_categories.items():
                if entities:
                    with cols[col_idx % 2]:
                        st.markdown(f"**{category}**")
                        # Create colorful entity tags
                        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3"]
                        entity_html = ""
                        for i, entity in enumerate(entities[:8]):  # Show first 8 per category
                            color = colors[i % len(colors)]
                            entity_html += f'<span style="background-color: {color}; color: white; padding: 0.25rem 0.6rem; margin: 0.15rem; border-radius: 0.4rem; font-size: 0.85rem; display: inline-block;">{entity}</span> '

                        st.markdown(entity_html, unsafe_allow_html=True)
                        if len(entities) > 8:
                            st.caption(f"... and {len(entities) - 8} more {category.split(' ')[1]}")

                    col_idx += 1

        st.markdown(f"#### {get_text('entity.phase2_title')}")
        st.info(get_text('entity.phase2_description'))

        # Denoising comparison with enhanced display
        original_input = st.session_state.get('original_input', '')
        if entity_result.denoised_text and original_input:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(get_text('ui.original_input_text'))
                st.text_area(
                    get_text('components.raw_input'),
                    original_input[:400] + ("..." if len(original_input) > 400 else ""),
                    height=140,
                    disabled=True,
                    key="original_entity_text"
                )
                st.caption(get_text('components.total_length', length=len(original_input)))

            with col2:
                st.markdown(get_text('ui.denoised_structured_text'))
                st.text_area(
                    get_text('components.processed_output'),
                    entity_result.denoised_text[:400] + ("..." if len(entity_result.denoised_text) > 400 else ""),
                    height=140,
                    disabled=True,
                    key="denoised_entity_text"
                )
                st.caption(get_text('components.processed_length', length=len(entity_result.denoised_text)))


    else:
        # Enhanced error display
        st.error(f"❌ Entity extraction failed: {entity_result.error}")

        # Error analysis and troubleshooting
        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('metrics.processing_time'), f"{entity_result.processing_time:.2f}s", get_text('components.before_failure'))
        with col2:
            st.metric(get_text('components.stage'), get_text('components.entity_extraction'), get_text('components.failed_at_stage'))

        if show_expanders:
            with st.expander(get_text('entity.error_analysis')):
                _display_entity_error_analysis_content(entity_result)
        else:
            st.markdown(f"### {get_text('entity.error_analysis')}")
            _display_entity_error_analysis_content(entity_result)


def _display_entity_error_analysis_content(entity_result: EntityResult):
    """Helper function to display entity error analysis content."""
    st.markdown(get_text('ui.possible_causes_solutions'))

    error_suggestions = [
        get_text('error_suggestions.api_connectivity'),
        get_text('error_suggestions.input_format'),
        get_text('error_suggestions.rate_limiting'),
        get_text('error_suggestions.text_length'),
        get_text('error_suggestions.model_availability')
    ]

    for suggestion in error_suggestions:
        st.markdown(f"- {suggestion}")

    if hasattr(entity_result, 'technical_details') and entity_result.technical_details:
        st.markdown(get_text('ui.technical_details'))
        st.code(entity_result.technical_details, language="text")


def _display_triple_phases_content(triple_result: TripleResult):
    """Helper function to display triple generation phases content."""
    st.markdown(f"### {get_text('triple.phase1_title')}")
    st.info(get_text('triple.phase1_description'))

    # Processing statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(get_text('ui.text_processing'))
        if hasattr(triple_result, 'metadata') and triple_result.metadata:
            chunks_processed = triple_result.metadata.get('chunks_processed', 1)
            st.metric(get_text('components.text_chunks'), chunks_processed, get_text('components.processed_segments'))
        else:
            st.metric(get_text('components.processing_method'), get_text('components.sequential'), get_text('components.text_analysis'))
    with col2:
        st.markdown(get_text('ui.relation_discovery'))
        if triple_result.triples:
            unique_relations = len(set(t.predicate for t in triple_result.triples))
            st.metric(get_text('metrics.relation_types'), unique_relations, get_text('metrics.discovered_patterns'))
    with col3:
        st.markdown(get_text('ui.quality_enhancement'))
        st.metric(get_text('metrics.schema_validation'), get_text('components.json_format'), get_text('metrics.structured_output'))

    st.markdown(f"### {get_text('triple.phase2_title')}")
    st.info(get_text('triple.phase2_description'))





def _display_triple_error_analysis_content(triple_result: TripleResult):
    """Helper function to display triple error analysis content."""
    st.markdown(get_text('ui.common_issues_solutions'))
    error_solutions = [
        get_text('error_suggestions.entity_quality'),
        get_text('error_suggestions.text_format'),
        get_text('error_suggestions.json_parsing'),
        get_text('error_suggestions.rate_limits'),
        get_text('error_suggestions.relation_complexity')
    ]

    for solution in error_solutions:
        st.markdown(f"- {solution}")

    if hasattr(triple_result, 'technical_details') and triple_result.technical_details:
        st.markdown(get_text('ui.technical_details'))
        st.code(triple_result.technical_details, language="json")




def _display_judgment_explanations_content(triples, judgment_result):
    """Helper function to display judgment explanations content with English labels."""
    for i, (triple, explanation) in enumerate(zip(triples, judgment_result.explanations)):
        if explanation:
            judgment_status = get_text('judgment.approved') if judgment_result.judgments[i] else get_text('judgment.rejected')
            status_icon = "✅" if judgment_result.judgments[i] else "❌"

            # Create a concise header for the expander
            triple_summary = f"{triple.subject} - {triple.predicate} - {triple.object}"
            if len(triple_summary) > 45:
                triple_summary = triple_summary[:42] + "..."

            expander_header = f"{status_icon} {i+1}. {triple_summary}"

            # Create collapsible expander for each explanation
            with st.expander(expander_header, expanded=False):
                # Display full triple information with enhanced formatting
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(get_text('citations.complete_triple', subject=triple.subject, predicate=triple.predicate, object=triple.object))
                with col2:
                    st.markdown(f"**{get_text('judgment.status')}:** {judgment_status}")


                # Display evidence sources if available
                if hasattr(judgment_result, 'metadata') and judgment_result.metadata:
                    evidence_sources = []
                    if i < len(judgment_result.explanations):
                        explanation_obj = judgment_result.explanations[i]
                        if isinstance(explanation_obj, dict) and 'evidence_sources' in explanation_obj:
                            evidence_sources = explanation_obj['evidence_sources']

                    if evidence_sources:
                        # Translate evidence sources using localization
                        source_translations = {
                            'historical_records': get_text('citations.historical_records'),
                            'literary_works': get_text('citations.literary_works'),
                            'general_knowledge': get_text('citations.general_knowledge'),
                            'domain_expertise': get_text('citations.domain_expertise')
                        }
                        translated_sources = [source_translations.get(src, src) for src in evidence_sources]
                        st.markdown(get_text('citations.reference_sources', sources=', '.join(translated_sources)))

                # Display actual citations if available
                actual_citations = []
                if i < len(judgment_result.explanations):
                    explanation_obj = judgment_result.explanations[i]
                    if isinstance(explanation_obj, dict) and 'actual_citations' in explanation_obj:
                        actual_citations = explanation_obj['actual_citations']

                if actual_citations:
                    st.markdown(get_text('ui.source_citations'))
                    for idx, citation in enumerate(actual_citations, 1):
                        # Display each citation as a clickable link
                        if citation and citation.strip():
                            citation_text = citation.strip()

                            # Validate and format URL properly
                            try:
                                # Check if it's already a valid URL
                                if citation_text.startswith(('http://', 'https://')):
                                    formatted_citation = citation_text
                                elif citation_text.startswith('www.'):
                                    formatted_citation = f"https://{citation_text}"
                                elif '.' in citation_text and not citation_text.startswith('file://'):
                                    # Looks like a domain, add https
                                    formatted_citation = f"https://{citation_text}"
                                else:
                                    # Not a URL, display as text only
                                    formatted_citation = None

                                # Escape special markdown characters in citation text
                                safe_citation = citation_text.replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')

                                if formatted_citation:
                                    # Create clickable link with citation number
                                    st.markdown(f"[{idx}. {safe_citation}]({formatted_citation})")
                                else:
                                    # Display as non-clickable text
                                    st.markdown(f"{idx}. {safe_citation}")

                            except Exception as url_error:
                                # If URL processing fails, display as text
                                safe_citation = citation_text.replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')
                                st.markdown(f"{idx}. {safe_citation}")
                                print(f"DEBUG: URL validation error for citation '{citation_text}': {url_error}")

                st.markdown(get_text('ui.ai_judgment_explanation'))

                # Extract and format Traditional Chinese reasoning
                reasoning_text = ""
                if isinstance(explanation, dict):
                    # If explanation is a dict with reasoning key
                    reasoning_text = explanation.get('reasoning', str(explanation))
                else:
                    # If explanation is a string (from reasoning field)
                    reasoning_text = str(explanation)

                # Clean and format the reasoning text for Traditional Chinese
                if reasoning_text:
                    # Remove any structural formatting from the response parsing
                    cleaned_reasoning = reasoning_text.strip()

                    # Remove common prefixes that might remain
                    prefixes_to_remove = [
                        'Detailed Reasoning:',
                        'reasoning:',
                        '詳細說明：',  # Keep for backward compatibility with existing data
                        'Error during processing:',
                        'Error parsing response:'
                    ]

                    for prefix in prefixes_to_remove:
                        if cleaned_reasoning.startswith(prefix):
                            cleaned_reasoning = cleaned_reasoning[len(prefix):].strip()

                    # Format the Traditional Chinese explanation with proper line breaks
                    # Split by sentence-ending punctuation common in Chinese
                    sentences = re.split(r'([。！？])', cleaned_reasoning)
                    formatted_sentences = []

                    temp_sentence = ""
                    for part in sentences:
                        temp_sentence += part
                        if part in ['。', '！', '？']:
                            if temp_sentence.strip():
                                formatted_sentences.append(temp_sentence.strip())
                            temp_sentence = ""

                    # Add any remaining text
                    if temp_sentence.strip():
                        formatted_sentences.append(temp_sentence.strip())

                    # Display formatted sentences with proper spacing
                    if formatted_sentences:
                        for idx, sentence in enumerate(formatted_sentences, 1):
                            if sentence:
                                st.markdown(f"**{idx}.** {sentence}")
                                if idx < len(formatted_sentences):  # Add spacing between sentences
                                    st.markdown("")
                    else:
                        # Fallback to display original text if parsing fails
                        st.markdown(cleaned_reasoning)
                else:
                    st.markdown(get_text('ui.no_detailed_explanation'))


def _display_pipeline_stage_details_content(pipeline_result):
    """Helper function to display pipeline stage details content."""
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


def display_triple_results(triple_result: TripleResult, show_validation: bool = True, show_expanders: bool = True):
    """
    Display comprehensive triple generation results with detailed processing information.

    Args:
        triple_result: The TripleResult to display
        show_validation: Whether to show validation details
        show_expanders: Whether to create expanders for detailed sections
    """
    st.markdown(f"## {get_text('triple.title')}")

    if triple_result.success:
        # Enhanced summary metrics with detailed information
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(get_text('status.status'), get_text('status.success'), get_text('metrics.generation_complete'))
        with col2:
            st.metric(get_text('metrics.processing_time'), f"{triple_result.processing_time:.2f}s", "GPT-5-mini API")
        with col3:
            st.metric(get_text('metrics.triple_count'), len(triple_result.triples), get_text('metrics.knowledge_relations'))
        with col5:
            # Count unique subjects and objects
            if triple_result.triples:
                unique_entities = set()
                for triple in triple_result.triples:
                    unique_entities.add(triple.subject)
                    unique_entities.add(triple.object)
                st.metric(get_text('metrics.unique_entities'), len(unique_entities), get_text('metrics.graph_nodes'))

        # Detailed processing phases display
        if show_expanders:
            with st.expander(get_text('triple.detailed_phases'), expanded=True):
                _display_triple_phases_content(triple_result)
        else:
            st.markdown(f"### {get_text('triple.detailed_phases')}")
            _display_triple_phases_content(triple_result)


    else:
        # Enhanced error display for triple generation
        st.error(f"❌ Triple generation failed: {triple_result.error}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('metrics.processing_time'), f"{triple_result.processing_time:.2f}s", get_text('components.before_failure'))
        with col2:
            st.metric(get_text('components.stage'), get_text('progress.triple_generation'), get_text('components.failed_at_stage'))

        if show_expanders:
            with st.expander(get_text('triple.error_analysis')):
                _display_triple_error_analysis_content(triple_result)
        else:
            st.markdown(f"### {get_text('triple.error_analysis')}")
            _display_triple_error_analysis_content(triple_result)
    
    if triple_result.triples:
        # Create DataFrame for display
        triple_data = []
        for i, triple in enumerate(triple_result.triples):
            triple_data.append({
                "#": i + 1,
                get_text('judgment.subject'): triple.subject,
                get_text('judgment.relation'): triple.predicate,
                get_text('judgment.object'): triple.object,
            })
        
        df = pd.DataFrame(triple_data)
        
        # Interactive table with selection
        st.markdown(f"### {get_text('triple.generated_triples')}")

        # Table display options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(get_text('ui.click_rows_details'))
        with col2:
            export_format = st.selectbox(get_text('buttons.export_format'), ["JSON", "CSV"], key="triple_export")
        
        # Display the table
        selected_rows = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
        
        # Export functionality
        if st.button(get_text('buttons.export_triples'), key="export_triples"):
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

    else:
        st.warning(get_text('ui.no_triples_generated'))


def display_judgment_results(judgment_result: JudgmentResult, triples: List[Triple], show_expanders: bool = True):
    """
    Display graph judgment results with approval/rejection analysis.

    Args:
        judgment_result: The JudgmentResult to display
        triples: The original triples that were judged
        show_expanders: Whether to create expanders for detailed sections
    """
    st.markdown(f"## {get_text('judgment.title')}")
    
    # Summary metrics
    approved = sum(1 for j in judgment_result.judgments if j)
    rejected = len(judgment_result.judgments) - approved
    approval_rate = approved / len(judgment_result.judgments) if judgment_result.judgments else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(get_text('metrics.judgment_status'), get_text('status.complete') if judgment_result.success else get_text('status.failed'))
    with col2:
        st.metric(get_text('metrics.processing_time'), f"{judgment_result.processing_time:.2f}s")
    with col3:
        st.metric(get_text('metrics.approved'), approved, delta=f"{approval_rate:.1%}")
    with col4:
        st.metric(get_text('metrics.rejected'), rejected)
    
    if judgment_result.judgments:
        # Create combined results
        results_data = []
        for i, (triple, judgment) in enumerate(zip(triples, judgment_result.judgments)):
            status_emoji = "✅" if judgment else "❌"
            status_text = get_text('judgment.approved') if judgment else get_text('judgment.rejected')

            results_data.append({
                "#": i + 1,
                get_text('judgment.status'): f"{status_emoji} {status_text}",
                get_text('judgment.subject'): triple.subject,
                get_text('judgment.relation'): triple.predicate,
                get_text('judgment.object'): triple.object,
            })
        
        df = pd.DataFrame(results_data)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_option = st.selectbox(
                get_text('metrics.filter_display'),
                [get_text('metrics.all_items'), get_text('metrics.approved_only'), get_text('metrics.rejected_only')],
                key="judgment_filter"
            )
        with col2:
            sort_by = st.selectbox(
                get_text('metrics.sort_by'),
                ["#", get_text('judgment.status')],
                key="judgment_sort"
            )
        
        # Apply filters
        filtered_df = df.copy()
        if filter_option == get_text('metrics.approved_only'):
            filtered_df = filtered_df[filtered_df[get_text('judgment.status')].str.contains(get_text('judgment.approved'))]
        elif filter_option == get_text('metrics.rejected_only'):
            filtered_df = filtered_df[filtered_df[get_text('judgment.status')].str.contains(get_text('judgment.rejected'))]

        # Apply sorting
        if sort_by == "Status":
            filtered_df = filtered_df.sort_values('Status', ascending=True)

        st.markdown(f"### 📋 Judgment Results Details ({len(filtered_df)} items)")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Explanations if available
        if judgment_result.explanations and any(judgment_result.explanations):
            if show_expanders:
                with st.expander("💭 Judgment Explanations"):
                    _display_judgment_explanations_content(triples, judgment_result)
            else:
                st.markdown("### 💭 Judgment Explanations")
                _display_judgment_explanations_content(triples, judgment_result)
        
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
    
    # Updated layout with modern Plotly syntax
    fig.update_layout(
        title=dict(
            text="Knowledge Graph Visualization",
            font=dict(size=16)
        ),
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
        st.info("Additional analytics will be displayed here in future versions")


def display_pipeline_summary(pipeline_result: PipelineResult, show_expanders: bool = True):
    """
    Display a comprehensive summary of the entire pipeline run.

    Args:
        pipeline_result: The complete pipeline result
        show_expanders: Whether to create expanders for detailed sections
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
    if show_expanders:
        with st.expander("🔍 Detailed Stage Information"):
            _display_pipeline_stage_details_content(pipeline_result)
    else:
        st.markdown("### 🔍 Detailed Stage Information")
        _display_pipeline_stage_details_content(pipeline_result)


def create_sidebar_controls() -> Dict[str, Any]:
    """
    Create sidebar controls for pipeline configuration.

    Returns:
        Dictionary of configuration options
    """
    # Language selection at the top
    st.sidebar.markdown(f"### {get_text('sidebar.language_selection')}")

    # Get supported languages and current selection
    supported_langs = get_supported_languages()
    current_lang = get_current_language()

    # Create language options for display
    lang_options = list(supported_langs.keys())
    lang_labels = [f"{code} - {name}" for code, name in supported_langs.items()]

    # Get current index
    try:
        current_index = lang_options.index(current_lang)
    except ValueError:
        current_index = 0  # Default to first option if current not found

    # Language selector
    selected_lang_index = st.sidebar.selectbox(
        "Select Language",
        range(len(lang_options)),
        index=current_index,
        format_func=lambda x: lang_labels[x],
        key="language_selector",
        label_visibility="collapsed"
    )

    # Update language if changed
    selected_lang = lang_options[selected_lang_index]
    if selected_lang != current_lang:
        # Store the language change in session state for persistence
        st.session_state.language = selected_lang
        set_language(selected_lang)
        # Force complete page refresh to update all text
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {get_text('sidebar.configuration')}")
    
    # API configuration
    st.sidebar.markdown(f"### {get_text('sidebar.api_settings')}")
    api_timeout = st.sidebar.slider(get_text('sidebar.api_timeout'), 30, 300, 60)
    max_retries = st.sidebar.slider(get_text('sidebar.max_retries'), 1, 5, 3)

    # Processing options
    st.sidebar.markdown(f"### {get_text('sidebar.processing_options')}")
    enable_explanations = st.sidebar.checkbox(get_text('sidebar.enable_explanations'), value=True)
    batch_size = st.sidebar.slider(get_text('sidebar.batch_size'), 1, 20, 10)

    # Display options
    st.sidebar.markdown(f"### {get_text('sidebar.display_options')}")
    show_technical_details = st.sidebar.checkbox(get_text('sidebar.show_technical_details'), value=False)
    auto_scroll = st.sidebar.checkbox(get_text('sidebar.auto_scroll'), value=True)

    # Debug options
    if st.sidebar.checkbox(get_text('sidebar.debug_mode')):
        st.sidebar.markdown(f"### {get_text('sidebar.debug_options')}")
        log_level = st.sidebar.selectbox(get_text('sidebar.log_level'), ["INFO", "DEBUG", "WARNING", "ERROR"])
        show_timing = st.sidebar.checkbox(get_text('sidebar.show_timing'), value=True)
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