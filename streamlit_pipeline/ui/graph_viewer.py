"""
Graph Viewer Component for GraphJudge Streamlit Application.

This module provides a standalone graph viewer that allows users to upload
and visualize existing knowledge graph files in multiple formats.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    # Try absolute import first (for package installation)
    from streamlit_pipeline.core.models import Triple
    from streamlit_pipeline.utils.reference_graph_manager import ReferenceGraphManager
    from streamlit_pipeline.core.graph_converter import (
        create_pyvis_graph_from_triples,
        create_graph_from_triples,
        create_kgshows_graph_from_triples
    )
    from streamlit_pipeline.ui.display import display_pyvis_knowledge_graph
    from streamlit_pipeline.utils.i18n import get_text
except ImportError:
    # Fallback to relative imports (for direct execution)
    from ..core.models import Triple
    from ..utils.reference_graph_manager import ReferenceGraphManager
    from ..core.graph_converter import (
        create_pyvis_graph_from_triples,
        create_graph_from_triples,
        create_kgshows_graph_from_triples
    )
    from .display import display_pyvis_knowledge_graph
    from ..utils.i18n import get_text


def display_graph_viewer_tab():
    """
    Display standalone graph viewer for uploading and visualizing graphs.

    Features:
    - File upload (JSON, CSV, TXT)
    - Parse graph data using ReferenceGraphManager
    - Display using multi-format viewer (Pyvis, Plotly, kgGenShows, Text)
    """
    st.markdown(f"## {get_text('graph_viewer.title')}")
    st.markdown(get_text('graph_viewer.description'))
    st.markdown("---")

    # File upload section
    st.markdown(f"### {get_text('graph_viewer.upload_section_title')}")

    uploaded_file = st.file_uploader(
        get_text('graph_viewer.choose_file'),
        type=['json', 'csv', 'txt'],
        help=get_text('graph_viewer.upload_help'),
        key="graph_viewer_file_upload"
    )

    if uploaded_file is not None:
        # Display file information
        _display_file_info(uploaded_file)

        # Parse the uploaded file
        with st.spinner(get_text('graph_viewer.conversion_info')):
            success, triples, error_msg = _parse_graph_file(uploaded_file)

        if success and triples:
            # Display success message
            st.success(get_text('graph_viewer.parse_success', count=len(triples)))

            # Display graph statistics
            _display_graph_statistics(triples)

            st.markdown("---")
            st.markdown(f"### {get_text('graph_viewer.visualization_section')}")

            # Convert triples to visualization formats
            pyvis_data, graph_data, kgshows_data = _convert_to_visualization_formats(triples)

            # Display multi-format graph viewer
            display_pyvis_knowledge_graph(
                pyvis_data=pyvis_data,
                triples=triples,
                graph_data=graph_data,
                kgshows_data=kgshows_data,
                height=600
            )

        elif error_msg:
            # Display error message
            st.error(get_text('graph_viewer.parse_error', error=error_msg))
        else:
            st.warning(get_text('graph_viewer.no_file_uploaded'))

    else:
        # Show instructions when no file is uploaded
        st.info(get_text('graph_viewer.no_file_uploaded'))

        # Display example formats
        with st.expander("📚 View Supported File Formats"):
            _display_format_examples()


def _display_file_info(uploaded_file):
    """
    Display information about the uploaded file.

    Args:
        uploaded_file: Streamlit uploaded file object
    """
    st.markdown(f"#### {get_text('graph_viewer.file_info_title')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("File Name", uploaded_file.name)

    with col2:
        file_size_kb = uploaded_file.size / 1024
        st.metric("File Size", f"{file_size_kb:.2f} KB")

    with col3:
        file_type = uploaded_file.name.split('.')[-1].upper()
        st.metric("File Type", file_type)


def _parse_graph_file(uploaded_file) -> tuple[bool, Optional[List[Triple]], Optional[str]]:
    """
    Parse uploaded graph file using ReferenceGraphManager.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (success, triples_list, error_message)
    """
    try:
        # Initialize reference graph manager
        manager = ReferenceGraphManager()

        # Upload and parse the file
        success, triples, error_msg = manager.upload_reference_graph(
            uploaded_file,
            file_format="auto"
        )

        return success, triples, error_msg

    except Exception as e:
        return False, None, str(e)


def _display_graph_statistics(triples: List[Triple]):
    """
    Display statistics about the loaded graph.

    Args:
        triples: List of Triple objects
    """
    # Calculate statistics
    unique_entities = set()
    unique_relations = set()

    for triple in triples:
        unique_entities.add(triple.subject)
        unique_entities.add(triple.object)
        unique_relations.add(triple.predicate)

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            get_text('graph_viewer.triples_count'),
            len(triples)
        )

    with col2:
        st.metric(
            get_text('graph_viewer.entities_count'),
            len(unique_entities)
        )

    with col3:
        st.metric(
            "Relation Types",
            len(unique_relations)
        )


def _convert_to_visualization_formats(triples: List[Triple]) -> tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]]
]:
    """
    Convert triples to multiple visualization formats.

    Args:
        triples: List of Triple objects

    Returns:
        Tuple of (pyvis_data, graph_data, kgshows_data)
    """
    try:
        # Convert to Pyvis format
        pyvis_data = create_pyvis_graph_from_triples(triples)
    except Exception as e:
        st.warning(f"Pyvis conversion failed: {e}")
        pyvis_data = None

    try:
        # Convert to Plotly format
        graph_data = create_graph_from_triples(triples)
    except Exception as e:
        st.warning(f"Plotly conversion failed: {e}")
        graph_data = None

    try:
        # Convert to kgGenShows format
        kgshows_data = create_kgshows_graph_from_triples(triples)
    except Exception as e:
        st.warning(f"kgGenShows conversion failed: {e}")
        kgshows_data = None

    return pyvis_data, graph_data, kgshows_data


def _display_format_examples():
    """Display examples of supported file formats."""
    st.markdown("### JSON Format Example")
    st.code('''
{
  "triples": [
    {
      "subject": "曹雪芹",
      "predicate": "創作",
      "object": "紅樓夢"
    },
    {
      "subject": "紅樓夢",
      "predicate": "是",
      "object": "章回體長篇小說"
    }
  ]
}
''', language="json")

    st.markdown("### CSV Format Example")
    st.code('''
subject,predicate,object
曹雪芹,創作,紅樓夢
紅樓夢,是,章回體長篇小說
''', language="csv")

    st.markdown("### TXT Format Example")
    st.code('''
曹雪芹 - 創作 - 紅樓夢
紅樓夢 - 是 - 章回體長篇小說
''', language="text")
