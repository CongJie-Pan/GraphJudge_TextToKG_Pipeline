"""
Unit tests for Pyvis inline rendering behavior.

These tests verify that the Streamlit viewer builds a Pyvis HTML string
with inline resources (no external CDN references) and that basic data
structures render without raising exceptions.
"""

from typing import Dict, Any

import pytest


def _sample_pyvis_data() -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": "A", "label": "Alice", "color": "#1f77b4", "size": 20},
            {"id": "B", "label": "Bob", "color": "#2ca02c", "size": 20},
        ],
        "edges": [
            {"from": "A", "to": "B", "label": "knows", "width": 2, "color": "#888888"}
        ],
        "physics": {"enabled": True},
        "metadata": {"nodes_count": 2, "edges_count": 1},
    }


@pytest.mark.unit
def test_create_pyvis_inline_html():
    try:
        # Import lazily to avoid hard dependency if pyvis/streamlit absent in some envs
        from streamlit_pipeline.ui.display import create_pyvis_knowledge_graph
    except Exception as e:  # pragma: no cover - if import fails due to env
        pytest.skip(f"Skipping test due to import error: {e}")

    html = create_pyvis_knowledge_graph(_sample_pyvis_data(), height=300)
    assert isinstance(html, str) and len(html) > 200
    # Inline resources should avoid external script/link tags (comments may contain URLs)
    forbidden_patterns = [
        '<script src="http',
        '<script src="https',
        '<link rel="stylesheet" href="http',
        '<link rel="stylesheet" href="https',
    ]
    lower_html = html.lower()
    assert not any(p in lower_html for p in forbidden_patterns)
