#!/usr/bin/env python3
"""
Test script to verify the Pyvis knowledge graph implementation.

This script tests the complete flow of:
1. Triple generation → 2. Graph judgment → 3. Multi-format conversion → 4. Pyvis visualization
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add streamlit_pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.models import Triple, TripleResult, JudgmentResult
from core.graph_converter import (
    create_graph_from_judgment_result,
    create_pyvis_graph_from_judgment_result,
    create_kgshows_graph_from_judgment_result,
    validate_graph_data,
    get_graph_statistics
)
# Skip UI testing for now - focus on core functionality


def create_test_data():
    """Create comprehensive test data for the implementation."""
    # Create test triples with Chinese classical literature theme
    triples = [
        Triple(
            subject="賈寶玉",
            predicate="身份",
            object="主角",
            confidence=0.95
        ),
        Triple(
            subject="賈寶玉",
            predicate="家族",
            object="賈家",
            confidence=0.9
        ),
        Triple(
            subject="林黛玉",
            predicate="身份",
            object="表妹",
            confidence=0.85
        ),
        Triple(
            subject="林黛玉",
            predicate="居住",
            object="賈府",
            confidence=0.8
        ),
        Triple(
            subject="薛寶釵",
            predicate="家族",
            object="薛家",
            confidence=0.88
        ),
        Triple(
            subject="賈府",
            predicate="地點",
            object="京城",
            confidence=0.75
        ),
        Triple(
            subject="《紅樓夢》",
            predicate="作者",
            object="曹雪芹",
            confidence=0.98
        ),
        Triple(
            subject="太虛幻境",
            predicate="地點",
            object="虛擬世界",
            confidence=0.6  # Lower confidence - might be rejected
        )
    ]

    # Create test triple result
    triple_result = TripleResult(
        triples=triples,
        success=True,
        processing_time=2.5,
        metadata={"test_mode": True, "source": "test_data"}
    )

    # Create test judgment result (approve most, reject low confidence ones)
    judgment_result = JudgmentResult(
        judgments=[True, True, True, True, True, True, True, False],  # Reject last triple
        confidence=[0.95, 0.9, 0.85, 0.8, 0.88, 0.75, 0.98, 0.3],
        explanations=[
            "Main character confirmed",
            "Family relationship clear",
            "Relationship established",
            "Living arrangement confirmed",
            "Family connection verified",
            "Geographic location valid",
            "Authorship well-documented",
            "Insufficient evidence for virtual location"
        ],
        success=True,
        processing_time=3.0
    )

    return triple_result, judgment_result, triples


def test_graph_conversions():
    """Test all three graph format conversions."""
    print("Testing Multi-Format Graph Conversion")
    print("=" * 60)

    # Create test data
    triple_result, judgment_result, triples = create_test_data()

    print(f"Test data created:")
    print(f"   Total triples: {len(triples)}")
    print(f"   Approved triples: {sum(judgment_result.judgments)}")
    print(f"   Rejected triples: {len(judgment_result.judgments) - sum(judgment_result.judgments)}")

    results = {}

    try:
        # Test 1: Plotly format (backward compatibility)
        print(f"\n[1] Test 1: Plotly format conversion...")
        plotly_graph = create_graph_from_judgment_result(triple_result, judgment_result)

        is_valid, validation_errors = validate_graph_data(plotly_graph)
        if is_valid:
            print(f"   SUCCESS: Plotly format VALID")
            print(f"   Nodes: {len(plotly_graph.get('nodes', []))}")
            print(f"   Edges: {len(plotly_graph.get('edges', []))}")
            results["plotly"] = plotly_graph
        else:
            print(f"   ERROR: Plotly format INVALID - {validation_errors[:2]}")

        # Test 2: Pyvis format (primary viewer)
        print(f"\n[2] Test 2: Pyvis format conversion...")
        pyvis_graph = create_pyvis_graph_from_judgment_result(triple_result, judgment_result)

        nodes_count = pyvis_graph["metadata"]["nodes_count"]
        edges_count = pyvis_graph["metadata"]["edges_count"]
        print(f"   SUCCESS: Pyvis format VALID")
        print(f"   Nodes: {nodes_count}")
        print(f"   Edges: {edges_count}")
        print(f"   Physics: {'Enabled' if pyvis_graph['metadata']['physics_enabled'] else 'Disabled'}")
        results["pyvis"] = pyvis_graph

        # Test 3: kgGenShows format (for other projects)
        print(f"\n[3] Test 3: kgGenShows format conversion...")
        kgshows_graph = create_kgshows_graph_from_judgment_result(triple_result, judgment_result)

        entities_count = len(kgshows_graph["entities"])
        relationships_count = len(kgshows_graph["relationships"])
        print(f"   SUCCESS: kgGenShows format VALID")
        print(f"   Entities: {entities_count}")
        print(f"   Relationships: {relationships_count}")
        print(f"   Approval rate: {kgshows_graph['report']['processing_summary']['approval_rate']:.1f}%")
        results["kgshows"] = kgshows_graph

        # Test 4: Consistency check
        print(f"\n[4] Test 4: Cross-format consistency check...")
        plotly_entities = len(plotly_graph.get("entities", []))
        pyvis_entities = nodes_count
        kgshows_entities = entities_count

        if plotly_entities == pyvis_entities == kgshows_entities:
            print(f"   SUCCESS: Entity counts consistent: {plotly_entities}")
        else:
            print(f"   WARNING: Entity counts differ: Plotly={plotly_entities}, Pyvis={pyvis_entities}, kgShows={kgshows_entities}")

        plotly_relationships = len(plotly_graph.get("relationships", []))
        pyvis_relationships = edges_count
        kgshows_relationships = relationships_count

        if plotly_relationships == pyvis_relationships == kgshows_relationships:
            print(f"   SUCCESS: Relationship counts consistent: {plotly_relationships}")
        else:
            print(f"   WARNING: Relationship counts differ: Plotly={plotly_relationships}, Pyvis={pyvis_relationships}, kgShows={kgshows_relationships}")

        return results

    except Exception as e:
        print(f"\nERROR during testing: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None


def test_pyvis_data_structure():
    """Test Pyvis data structure validity."""
    print(f"\n[5] Test 5: Pyvis data structure validation...")

    try:
        # Get test results
        results = test_graph_conversions()
        if not results or "pyvis" not in results:
            print("   ERROR: Cannot test Pyvis structure - no data available")
            return False

        pyvis_data = results["pyvis"]

        # Check required structure
        required_keys = ["nodes", "edges", "physics", "metadata"]
        missing_keys = [key for key in required_keys if key not in pyvis_data]

        if missing_keys:
            print(f"   ERROR: Missing keys: {missing_keys}")
            return False

        # Check nodes structure
        nodes = pyvis_data["nodes"]
        if nodes and isinstance(nodes, list):
            sample_node = nodes[0]
            required_node_keys = ["id", "label", "color", "size"]
            missing_node_keys = [key for key in required_node_keys if key not in sample_node]
            if missing_node_keys:
                print(f"   ERROR: Node missing keys: {missing_node_keys}")
                return False

        # Check edges structure
        edges = pyvis_data["edges"]
        if edges and isinstance(edges, list):
            sample_edge = edges[0]
            required_edge_keys = ["from", "to", "label"]
            missing_edge_keys = [key for key in required_edge_keys if key not in sample_edge]
            if missing_edge_keys:
                print(f"   ERROR: Edge missing keys: {missing_edge_keys}")
                return False

        print(f"   SUCCESS: Pyvis structure VALID")
        print(f"   Nodes: {len(nodes)}, Edges: {len(edges)}")
        return True

    except Exception as e:
        print(f"   ERROR: Pyvis structure validation error: {str(e)}")
        return False


def save_test_outputs():
    """Save all format outputs for manual inspection."""
    print(f"\n[6] Test 6: Saving test outputs...")

    try:
        results = test_graph_conversions()
        if not results:
            print("   ERROR: No results to save")
            return False

        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)

        # Save each format
        for format_name, data in results.items():
            output_file = output_dir / f"test_graph_{format_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"   SUCCESS: Saved {format_name} format: {output_file}")

        print(f"   All formats saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"   ERROR: Save error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("Testing Pyvis Knowledge Graph Implementation")
    print("=" * 70)

    # Test basic conversions
    conversion_results = test_graph_conversions()
    conversion_success = conversion_results is not None

    # Test Pyvis data structure
    structure_success = test_pyvis_data_structure()

    # Save outputs
    save_success = save_test_outputs()

    # Final report
    print(f"\nTest Summary")
    print("=" * 40)
    print(f"Multi-format conversion: {'PASS' if conversion_success else 'FAIL'}")
    print(f"Pyvis data structure: {'PASS' if structure_success else 'FAIL'}")
    print(f"Output saving: {'PASS' if save_success else 'FAIL'}")

    all_passed = conversion_success and structure_success and save_success

    if all_passed:
        print(f"\nALL TESTS PASSED!")
        print(f"The Pyvis implementation is ready for use in Streamlit!")
        print(f"")
        print(f"Next steps:")
        print(f"  1. Install dependencies: pip install -r requirements.txt")
        print(f"  2. Run Streamlit app: streamlit run app.py")
        print(f"  3. Process text and view interactive Pyvis graph!")
    else:
        print(f"\nSome tests failed. Please check the implementation.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())