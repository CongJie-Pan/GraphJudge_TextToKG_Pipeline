"""
Graph Converter Module for GraphJudge Streamlit Pipeline.

This module converts judgment results into JSON graph format suitable for
visualization and further processing. It adapts the conversion logic from
the main GraphJudge system for use in the Streamlit application.

Author: Claude Code
Date: 2025-09-16
Purpose: Convert judgment results to graph JSON format for visualization
"""

import re
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

from .models import Triple, JudgmentResult, TripleResult


class TripleExtractor:
    """
    Extract and parse knowledge graph triplets from judgment results.

    This class handles the parsing of triples and their conversion to
    a standardized format for graph visualization.
    """

    def __init__(self):
        """Initialize the extractor with pattern matching rules."""
        # Common predicate patterns in Chinese and English
        self.action_patterns = [
            '行為', '地點', '作品', '作者', '職業', '妻子', '女兒',
            '父親', '岳丈', '主人', '談論內容', '囑咐', '包含',
            '認為', '幫助', '贈送', '知道', '抱', '看見', '聽見',
            '創作', '居住', '喜歡', '討厭', '擁有', '屬於'
        ]

    def extract_triple_from_text(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Extract subject, predicate, object from text that might contain triple information.

        Args:
            text (str): Text that might contain triple information

        Returns:
            Optional[Tuple[str, str, str]]: (subject, predicate, object) or None if parsing fails
        """
        try:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text.strip())

            # Handle "Is this true:" format from judgment prompts
            if "Is this true:" in text:
                content = text.split("Is this true:")[-1].strip()
                content = content.rstrip("?").strip()

                # Try to split by known action patterns
                for pattern in self.action_patterns:
                    if pattern in content:
                        parts = content.split(pattern, 1)
                        if len(parts) == 2:
                            subject = parts[0].strip()
                            obj = parts[1].strip()
                            return (subject, pattern, obj)

                # If no pattern matched, try simple split
                parts = content.split()
                if len(parts) >= 3:
                    # Take first as subject, last as object, middle as predicate
                    subject = parts[0]
                    obj = parts[-1]
                    predicate = ' '.join(parts[1:-1])
                    return (subject, predicate, obj)

            # If no "Is this true:" format, try direct parsing
            for pattern in self.action_patterns:
                if pattern in text:
                    parts = text.split(pattern, 1)
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        obj = parts[1].strip()
                        return (subject, pattern, obj)

            return None

        except Exception as e:
            print(f"Error extracting triple from text '{text}': {str(e)}")
            return None


class GraphConverter:
    """
    Convert judgment results to JSON knowledge graph format.

    This class processes judgment results and converts approved triplets
    into a JSON format suitable for visualization and analysis.
    """

    def __init__(self):
        """Initialize the converter."""
        self.extractor = TripleExtractor()

        # Data storage
        self.entities: Set[str] = set()
        self.relationships: List[str] = []
        self.valid_triplets: List[Tuple[str, str, str]] = []

        # Statistics
        self.stats = {
            'total_triples': 0,
            'approved_triples': 0,
            'rejected_triples': 0,
            'unique_entities': 0,
            'unique_relationships': 0,
        }

    def convert_judgment_result_to_graph(self,
                                       triple_result: TripleResult,
                                       judgment_result: JudgmentResult) -> Dict[str, Any]:
        """
        Convert judgment results to graph JSON format.

        Args:
            triple_result: Result containing generated triples
            judgment_result: Result containing judgment decisions

        Returns:
            Dict[str, Any]: Graph data in JSON format
        """
        self._reset_data()

        if not triple_result.success or not judgment_result.success:
            return self._create_error_graph("Triple generation or judgment failed")

        if not triple_result.triples or not judgment_result.judgments:
            return self._create_empty_graph()

        # Process each triple with its judgment
        for i, triple in enumerate(triple_result.triples):
            self.stats['total_triples'] += 1

            # Get judgment for this triple
            if i < len(judgment_result.judgments):
                is_approved = judgment_result.judgments[i]

                if is_approved:
                    self.stats['approved_triples'] += 1
                    self._add_approved_triple(triple)
                else:
                    self.stats['rejected_triples'] += 1

        # Update final statistics
        self._update_final_stats()

        return self._generate_graph_json()

    def convert_triples_to_graph(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Convert a list of approved triples to graph JSON format.

        Args:
            triples: List of approved triples

        Returns:
            Dict[str, Any]: Graph data in JSON format
        """
        self._reset_data()

        if not triples:
            return self._create_empty_graph()

        # Process each triple
        for triple in triples:
            self.stats['total_triples'] += 1
            self.stats['approved_triples'] += 1
            self._add_approved_triple(triple)

        # Update final statistics
        self._update_final_stats()

        return self._generate_graph_json()

    def _reset_data(self):
        """Reset internal data structures."""
        self.entities.clear()
        self.relationships.clear()
        self.valid_triplets.clear()

        self.stats = {
            'total_triples': 0,
            'approved_triples': 0,
            'rejected_triples': 0,
            'unique_entities': 0,
            'unique_relationships': 0,
        }

    def _add_approved_triple(self, triple: Triple):
        """Add an approved triple to the graph data."""
        # Add entities
        self.entities.add(triple.subject)
        self.entities.add(triple.object)

        # Store the triple
        self.valid_triplets.append((triple.subject, triple.predicate, triple.object))

        # Add relationship in visualization format
        relationship = f"{triple.subject} - {triple.predicate} - {triple.object}"
        self.relationships.append(relationship)

    def _update_final_stats(self):
        """Update final statistics."""
        self.stats['unique_entities'] = len(self.entities)
        self.stats['unique_relationships'] = len(self.relationships)


    def _generate_graph_json(self) -> Dict[str, Any]:
        """Generate the complete graph JSON structure."""
        # Sort entities and relationships for consistent output
        sorted_entities = sorted(list(self.entities))

        # Create nodes data for visualization
        nodes = []
        for i, entity in enumerate(sorted_entities):
            # Count how many relationships this entity participates in
            entity_count = sum(1 for rel in self.valid_triplets
                             if rel[0] == entity or rel[2] == entity)

            nodes.append({
                "id": entity,
                "label": entity,
                "size": min(10 + entity_count * 2, 30),  # Size based on connections
                "color": self._get_entity_color(entity_count)
            })

        # Create edges data for visualization
        edges = []
        for subject, predicate, obj in self.valid_triplets:
            edges.append({
                "source": subject,
                "target": obj,
                "label": predicate,
                "weight": 0.5,
                "color": "#1f77b4",  # Standard blue color
                "width": 3  # Fixed width
            })

        # Create report section
        report = {
            "summary": {
                "entities": len(sorted_entities),
                "relationships": len(self.relationships),
                "total_evaluated": self.stats['total_triples'],
                "approved_triples": self.stats['approved_triples'],
                "rejected_triples": self.stats['rejected_triples'],
                "processing_timestamp": datetime.now().isoformat()
            },
            "processing_summary": {
                "total_evaluated": self.stats['total_triples'],
                "approved_extracted": self.stats['approved_triples'],
                "approval_rate": (round(self.stats['approved_triples'] / max(1, self.stats['total_triples']) * 100, 2)
                                if self.stats['total_triples'] > 0 else 0)
            },
            "quality_metrics": {
                "entity_count": len(sorted_entities),
                "relationship_count": len(self.relationships),
                "data_source": "Streamlit GraphJudge pipeline",
                "filter_criteria": "Only approved relationships included"
            }
        }

        # Create the complete graph JSON
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "entities": sorted_entities,
            "relationships": self.relationships,
            "report": report,
            "metadata": {
                "source_type": "streamlit_pipeline",
                "conversion_timestamp": datetime.now().isoformat(),
                "converter_version": "1.0.0",
                "visualization_ready": True
            }
        }

        return graph_data

    def convert_to_pyvis_format(self) -> Dict[str, Any]:
        """
        Convert current graph data to Pyvis format.

        Returns:
            Dict containing nodes and edges for Pyvis network visualization
        """
        # Sort entities for consistent output
        sorted_entities = sorted(list(self.entities))

        # Create nodes for Pyvis
        nodes = []
        for entity in sorted_entities:
            # Count connections for this entity
            connection_count = sum(1 for rel in self.valid_triplets
                                 if rel[0] == entity or rel[2] == entity)

            # Determine entity category and color
            entity_type, color = self._categorize_entity(entity)

            nodes.append({
                "id": entity,
                "label": entity,
                "color": color,
                "size": min(15 + connection_count * 3, 50),  # Size based on connections
                "title": f"{entity}\n类型: {entity_type}\n连接数: {connection_count}",  # Hover tooltip
                "font": {"size": 12},
                "borderWidth": 2
            })

        # Create edges for Pyvis
        edges = []
        for subject, predicate, obj in self.valid_triplets:
            edges.append({
                "from": subject,
                "to": obj,
                "label": predicate,
                "width": 3,  # Fixed width
                "color": "#1f77b4",  # Standard blue color
                "title": f"{subject} → {predicate} → {obj}",
                "arrows": "to",
                "smooth": {"type": "continuous"}
            })

        # Prepare Pyvis format data
        pyvis_data = {
            "nodes": nodes,
            "edges": edges,
            "physics": {
                "enabled": True,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                }
            },
            "metadata": {
                "format": "pyvis",
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "conversion_timestamp": datetime.now().isoformat(),
                "physics_enabled": True
            }
        }

        return pyvis_data

    def convert_to_kgshows_format(self) -> Dict[str, Any]:
        """
        Convert current graph data to kgGenShows format.

        Returns:
            Dict containing entities and relationships arrays for kgGenShows viewer
        """
        # Sort entities for consistent output
        sorted_entities = sorted(list(self.entities))

        # Create relationships in kgGenShows format: "subject - predicate - object"
        relationships = []
        for subject, predicate, obj in self.valid_triplets:
            relationship = f"{subject} - {predicate} - {obj}"
            relationships.append(relationship)

        # Create kgGenShows format data
        kgshows_data = {
            "entities": sorted_entities,
            "relationships": relationships,
            "report": {
                "summary": {
                    "entities": len(sorted_entities),
                    "relationships": len(relationships),
                    "total_evaluated": self.stats['total_triples'],
                    "approved_triples": self.stats['approved_triples'],
                    "rejected_triples": self.stats['rejected_triples'],
                        "processing_timestamp": datetime.now().isoformat()
                },
                "processing_summary": {
                    "total_evaluated": self.stats['total_triples'],
                    "approved_extracted": self.stats['approved_triples'],
                    "approval_rate": (round(self.stats['approved_triples'] / max(1, self.stats['total_triples']) * 100, 2)
                                    if self.stats['total_triples'] > 0 else 0)
                },
                "quality_metrics": {
                    "entity_count": len(sorted_entities),
                    "relationship_count": len(relationships),
                    "data_source": "Streamlit GraphJudge pipeline",
                    "filter_criteria": "Only approved relationships included"
                }
            },
            "metadata": {
                "format": "kgshows",
                "source_type": "streamlit_pipeline",
                "conversion_timestamp": datetime.now().isoformat(),
                "converter_version": "1.1.0",
                "kgshows_compatible": True
            }
        }

        return kgshows_data

    def _categorize_entity(self, entity: str) -> tuple[str, str]:
        """
        Categorize entity and return type with color (similar to kgGenShows).

        Args:
            entity: Entity name to categorize

        Returns:
            Tuple of (category_name, color_hex)
        """
        entity_lower = entity.lower()

        # Character names (orange)
        character_indicators = ['賈', '林', '薛', '王', '史', '鳳', '寶玉', '黛玉', '寶釵', '鳳姐']
        if any(indicator in entity for indicator in character_indicators):
            return ("人物", "#ff7f0e")

        # Places (green)
        place_indicators = ['府', '園', '院', '廳', '房', '山', '河', '京城', '廟', '宮']
        if any(indicator in entity for indicator in place_indicators):
            return ("地点", "#2ca02c")

        # Literary works (purple)
        work_indicators = ['記', '夢', '歌', '詩', '曲', '《', '》', '書']
        if any(indicator in entity for indicator in work_indicators):
            return ("作品", "#9467bd")

        # Concepts/Events (red)
        concept_indicators = ['功名', '富貴', '情', '愛', '恩', '緣', '命', '運']
        if any(indicator in entity for indicator in concept_indicators):
            return ("概念", "#d62728")

        # Default: other (blue)
        return ("其他", "#1f77b4")

    def _get_entity_color(self, connection_count: int) -> str:
        """Get color for entity based on connection count."""
        if connection_count >= 5:
            return "#FF6B6B"  # Red for highly connected
        elif connection_count >= 3:
            return "#4ECDC4"  # Teal for moderately connected
        else:
            return "#45B7D1"  # Blue for less connected


    def _create_error_graph(self, error_message: str) -> Dict[str, Any]:
        """Create an error graph structure."""
        return {
            "nodes": [],
            "edges": [],
            "entities": [],
            "relationships": [],
            "report": {
                "summary": {
                    "entities": 0,
                    "relationships": 0,
                    "error": error_message
                }
            },
            "metadata": {
                "source_type": "streamlit_pipeline",
                "conversion_timestamp": datetime.now().isoformat(),
                "error": error_message
            }
        }

    def _create_empty_graph(self) -> Dict[str, Any]:
        """Create an empty graph structure."""
        return {
            "nodes": [],
            "edges": [],
            "entities": [],
            "relationships": [],
            "report": {
                "summary": {
                    "entities": 0,
                    "relationships": 0,
                    "total_evaluated": 0,
                    "approved_triples": 0,
                    "rejected_triples": 0
                }
            },
            "metadata": {
                "source_type": "streamlit_pipeline",
                "conversion_timestamp": datetime.now().isoformat(),
                "empty_result": True
            }
        }


def create_graph_from_judgment_result(triple_result: TripleResult,
                                    judgment_result: JudgmentResult) -> Dict[str, Any]:
    """
    Convenience function to create graph from judgment result.

    Args:
        triple_result: Result containing generated triples
        judgment_result: Result containing judgment decisions

    Returns:
        Dict[str, Any]: Graph data in JSON format
    """
    converter = GraphConverter()
    return converter.convert_judgment_result_to_graph(triple_result, judgment_result)


def create_graph_from_triples(triples: List[Triple]) -> Dict[str, Any]:
    """
    Convenience function to create graph from list of triples.

    Args:
        triples: List of approved triples

    Returns:
        Dict[str, Any]: Graph data in JSON format
    """
    converter = GraphConverter()
    return converter.convert_triples_to_graph(triples)


def validate_graph_data(graph_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate graph data structure for visualization compatibility.

    Args:
        graph_data: Graph data dictionary to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Check required top-level keys
    required_keys = ["nodes", "edges", "entities", "relationships", "metadata"]
    for key in required_keys:
        if key not in graph_data:
            errors.append(f"Missing required key: {key}")

    # Validate nodes structure
    if "nodes" in graph_data:
        nodes = graph_data["nodes"]
        if not isinstance(nodes, list):
            errors.append("nodes must be a list")
        else:
            required_node_keys = ["id", "label"]
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"Node {i} must be a dictionary")
                    continue

                for key in required_node_keys:
                    if key not in node:
                        errors.append(f"Node {i} missing required key: {key}")

    # Validate edges structure
    if "edges" in graph_data:
        edges = graph_data["edges"]
        if not isinstance(edges, list):
            errors.append("edges must be a list")
        else:
            required_edge_keys = ["source", "target", "label"]
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"Edge {i} must be a dictionary")
                    continue

                for key in required_edge_keys:
                    if key not in edge:
                        errors.append(f"Edge {i} missing required key: {key}")

    # Validate metadata structure
    if "metadata" in graph_data:
        metadata = graph_data["metadata"]
        if not isinstance(metadata, dict):
            errors.append("metadata must be a dictionary")
        else:
            required_metadata_keys = ["source_type", "conversion_timestamp"]
            for key in required_metadata_keys:
                if key not in metadata:
                    errors.append(f"Metadata missing required key: {key}")

    # Cross-validation: ensure edge references exist in nodes
    if "nodes" in graph_data and "edges" in graph_data and len(errors) == 0:
        node_ids = {node.get("id") for node in graph_data["nodes"] if isinstance(node, dict)}

        for i, edge in enumerate(graph_data["edges"]):
            if isinstance(edge, dict):
                source = edge.get("source")
                target = edge.get("target")

                if source and source not in node_ids:
                    errors.append(f"Edge {i} references non-existent source node: {source}")

                if target and target not in node_ids:
                    errors.append(f"Edge {i} references non-existent target node: {target}")

    is_valid = len(errors) == 0
    return is_valid, errors


def create_pyvis_graph_from_judgment_result(triple_result: TripleResult,
                                          judgment_result: JudgmentResult) -> Dict[str, Any]:
    """
    Convenience function to create Pyvis graph from judgment result.

    Args:
        triple_result: Result containing generated triples
        judgment_result: Result containing judgment decisions

    Returns:
        Dict[str, Any]: Pyvis graph data format
    """
    converter = GraphConverter()
    converter.convert_judgment_result_to_graph(triple_result, judgment_result)
    return converter.convert_to_pyvis_format()


def create_pyvis_graph_from_triples(triples: List[Triple]) -> Dict[str, Any]:
    """
    Convenience function to create Pyvis graph from list of triples.

    Args:
        triples: List of approved triples

    Returns:
        Dict[str, Any]: Pyvis graph data format
    """
    converter = GraphConverter()
    converter.convert_triples_to_graph(triples)
    return converter.convert_to_pyvis_format()


def create_kgshows_graph_from_judgment_result(triple_result: TripleResult,
                                            judgment_result: JudgmentResult) -> Dict[str, Any]:
    """
    Convenience function to create kgGenShows graph from judgment result.

    Args:
        triple_result: Result containing generated triples
        judgment_result: Result containing judgment decisions

    Returns:
        Dict[str, Any]: kgGenShows graph data format
    """
    converter = GraphConverter()
    converter.convert_judgment_result_to_graph(triple_result, judgment_result)
    return converter.convert_to_kgshows_format()


def create_kgshows_graph_from_triples(triples: List[Triple]) -> Dict[str, Any]:
    """
    Convenience function to create kgGenShows graph from list of triples.

    Args:
        triples: List of approved triples

    Returns:
        Dict[str, Any]: kgGenShows graph data format
    """
    converter = GraphConverter()
    converter.convert_triples_to_graph(triples)
    return converter.convert_to_kgshows_format()


def get_graph_statistics(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate statistics about the graph data.

    Args:
        graph_data: Graph data dictionary

    Returns:
        Dictionary containing graph statistics
    """
    stats = {
        "nodes_count": 0,
        "edges_count": 0,
        "entities_count": 0,
        "relationships_count": 0,
        "isolated_nodes": 0,
        "average_node_degree": 0.0,
        "max_node_degree": 0,
        "validation_status": "unknown"
    }

    try:
        # Basic counts
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        stats["nodes_count"] = len(nodes)
        stats["edges_count"] = len(edges)
        stats["entities_count"] = len(graph_data.get("entities", []))
        stats["relationships_count"] = len(graph_data.get("relationships", []))

        # Node degree analysis
        if nodes and edges:
            from collections import defaultdict
            node_degrees = defaultdict(int)

            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get("source")
                    target = edge.get("target")
                    if source:
                        node_degrees[source] += 1
                    if target:
                        node_degrees[target] += 1

            if node_degrees:
                degrees = list(node_degrees.values())
                stats["average_node_degree"] = sum(degrees) / len(degrees)
                stats["max_node_degree"] = max(degrees)
                stats["isolated_nodes"] = len(nodes) - len(node_degrees)

        # Validate the graph data
        is_valid, validation_errors = validate_graph_data(graph_data)
        stats["validation_status"] = "valid" if is_valid else "invalid"
        if not is_valid:
            stats["validation_errors"] = validation_errors[:5]  # First 5 errors

    except Exception as e:
        stats["validation_status"] = "error"
        stats["error_message"] = str(e)

    return stats