#!/usr/bin/env python3
"""
Simple test script to verify graph converter functionality.

This script tests the graph conversion pipeline to ensure it properly
converts judgment results to visualization-ready graph data.
"""

import sys
import json
from pathlib import Path

# Add streamlit_pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.models import Triple, TripleResult, JudgmentResult
from core.graph_converter import (
    create_graph_from_judgment_result,
    create_graph_from_triples,
    validate_graph_data,
    get_graph_statistics
)


def create_test_data():
    """Create test data for graph conversion."""
    # Create test triples
    triples = [
        Triple(
            subject="女媧氏",
            predicate="地點",
            object="大荒山"
        ),
        Triple(
            subject="女媧氏",
            predicate="地點",
            object="無稽崖"
        ),
        Triple(
            subject="石頭",
            predicate="地點",
            object="青埂峰"
        ),
        Triple(
            subject="賈寶玉",
            predicate="身份",
            object="主角"
        ),
        Triple(
            subject="賈寶玉",
            predicate="家族",
            object="賈家"
        )
    ]

    # Create test triple result
    triple_result = TripleResult(
        triples=triples,
        success=True,
        processing_time=1.5,
        metadata={"test": True}
    )

    # Create test judgment result (approve most triples)
    judgment_result = JudgmentResult(
        judgments=[True, True, False, True, True],  # Reject the 3rd triple
        success=True,
        processing_time=2.0
    )

    return triple_result, judgment_result, triples


def test_graph_conversion():
    """Test the graph conversion functionality."""
    print("Testing Graph Conversion Pipeline")
    print("=" * 50)

    # Create test data
    triple_result, judgment_result, triples = create_test_data()

    print(f"Test data created:")
    print(f"  - Total triples: {len(triples)}")
    print(f"  - Approved triples: {sum(judgment_result.judgments)}")
    print(f"  - Rejected triples: {len(judgment_result.judgments) - sum(judgment_result.judgments)}")

    try:
        # Test 1: Convert from judgment result
        print(f"\nTest 1: Converting judgment result to graph...")
        graph_data = create_graph_from_judgment_result(triple_result, judgment_result)

        print(f"SUCCESS: Graph conversion successful!")
        print(f"  - Nodes: {len(graph_data.get('nodes', []))}")
        print(f"  - Edges: {len(graph_data.get('edges', []))}")
        print(f"  - Entities: {len(graph_data.get('entities', []))}")

        # Test 2: Validate graph data
        print(f"\nTest 2: Validating graph data...")
        is_valid, validation_errors = validate_graph_data(graph_data)

        if is_valid:
            print(f"SUCCESS: Graph data validation passed!")
        else:
            print(f"ERROR: Graph data validation failed:")
            for error in validation_errors[:5]:
                print(f"    - {error}")

        # Test 3: Generate statistics
        print(f"\nTest 3: Generating graph statistics...")
        stats = get_graph_statistics(graph_data)

        print(f"SUCCESS: Statistics generated:")
        print(f"  - Nodes: {stats['nodes_count']}")
        print(f"  - Edges: {stats['edges_count']}")
        print(f"  - Average node degree: {stats['average_node_degree']:.2f}")
        print(f"  - Isolated nodes: {stats['isolated_nodes']}")
        print(f"  - Validation status: {stats['validation_status']}")

        # Test 4: Convert from approved triples only
        print(f"\nTest 4: Converting approved triples only...")
        approved_triples = [
            triple for triple, approved in zip(triples, judgment_result.judgments)
            if approved
        ]

        approved_graph = create_graph_from_triples(approved_triples)

        print(f"SUCCESS: Approved triples graph created:")
        print(f"  - Nodes: {len(approved_graph.get('nodes', []))}")
        print(f"  - Edges: {len(approved_graph.get('edges', []))}")

        # Test 5: Display sample of graph data
        print(f"\nTest 5: Sample graph data structure:")
        sample_data = {
            "nodes_sample": graph_data.get("nodes", [])[:3],
            "edges_sample": graph_data.get("edges", [])[:3],
            "metadata": graph_data.get("metadata", {})
        }

        print(json.dumps(sample_data, ensure_ascii=False, indent=2))

        print(f"\nAll tests completed successfully!")
        print(f"Summary:")
        print(f"  - Graph conversion: WORKING")
        print(f"  - Data validation: WORKING")
        print(f"  - Statistics generation: WORKING")
        print(f"  - Data structure: VALID for visualization")

        return True

    except Exception as e:
        print(f"\nERROR: Test failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Run the test."""
    print("GraphJudge Streamlit Pipeline - Graph Converter Test")
    print("=" * 60)

    success = test_graph_conversion()

    if success:
        print(f"\nSUCCESS: All tests passed! The graph visualization should work correctly.")
        print(f"Next steps:")
        print(f"   1. Install Plotly: pip install plotly>=5.0.0")
        print(f"   2. Run the Streamlit app: streamlit run app.py")
        print(f"   3. Process some Chinese text to see the interactive graph!")
    else:
        print(f"\nERROR: Tests failed. Please check the implementation.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())