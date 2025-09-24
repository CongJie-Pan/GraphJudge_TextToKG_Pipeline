"""
Reference Graph Management for GraphJudge Evaluation System.

This module provides utilities for uploading, validating, and managing reference graphs
for evaluation purposes. It supports multiple file formats and provides conversion
utilities for seamless integration with the evaluation system.
"""

import os
import json
import csv
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta

try:
    from ..core.models import Triple
    from ..core.config import get_evaluation_config
    from .error_handling import ErrorHandler, ErrorType, safe_execute
except ImportError:
    # For direct execution or testing
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import Triple
    from core.config import get_evaluation_config
    from utils.error_handling import ErrorHandler, ErrorType, safe_execute


class ReferenceGraphManager:
    """
    Manager for reference graph upload, validation, and storage.

    Handles multiple file formats and provides validation utilities
    for ensuring reference graphs are compatible with evaluation system.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the reference graph manager.

        Args:
            temp_dir: Optional custom temporary directory for file storage
        """
        self.error_handler = ErrorHandler()
        self.logger = logging.getLogger(__name__)
        self.config = get_evaluation_config()

        # Set up temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "streamlit_pipeline" / "reference_graphs"

        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Track uploaded files for cleanup
        self.uploaded_files = {}

    def upload_reference_graph(self, uploaded_file, file_format: str = "auto") -> Tuple[bool, Optional[List[Triple]], Optional[str]]:
        """
        Upload and process a reference graph file.

        Args:
            uploaded_file: Streamlit uploaded file object or file path
            file_format: Format of the file ("json", "csv", "txt", "auto")

        Returns:
            Tuple of (success, triples_list, error_message)
        """
        try:
            # Handle different input types
            if hasattr(uploaded_file, 'read'):
                # Streamlit uploaded file
                file_content = uploaded_file.read()
                filename = uploaded_file.name

                # Decode if bytes
                if isinstance(file_content, bytes):
                    file_content = file_content.decode('utf-8')
            elif isinstance(uploaded_file, (str, Path)):
                # File path
                with open(uploaded_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                filename = os.path.basename(uploaded_file)
            else:
                return False, None, "Invalid file input type"

            # Auto-detect format if needed
            if file_format == "auto":
                file_format = self._detect_file_format(filename, file_content)

            # Validate file size
            if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
                return False, None, "File too large. Maximum size is 10MB."

            # Parse file content based on format
            triples = self._parse_file_content(file_content, file_format)

            if not triples:
                return False, None, f"No valid triples found in {file_format} file"

            # Validate graph size
            if len(triples) > self.config['reference_graph_max_size']:
                return False, None, f"Reference graph too large. Maximum {self.config['reference_graph_max_size']} triples allowed."

            # Validate triple format
            validation_result = self._validate_triples(triples)
            if not validation_result[0]:
                return False, None, validation_result[1]

            # Store file for future reference
            self._store_uploaded_file(filename, file_content, triples)

            self.logger.info(f"Successfully uploaded reference graph: {len(triples)} triples from {filename}")

            return True, triples, None

        except Exception as e:
            error_msg = f"Failed to upload reference graph: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _detect_file_format(self, filename: str, content: str) -> str:
        """Auto-detect file format based on filename and content."""
        filename_lower = filename.lower()

        if filename_lower.endswith('.json'):
            return "json"
        elif filename_lower.endswith('.csv'):
            return "csv"
        elif filename_lower.endswith('.txt'):
            return "txt"

        # Content-based detection
        content_stripped = content.strip()

        if content_stripped.startswith('{') or content_stripped.startswith('['):
            return "json"
        elif ',' in content_stripped and '\n' in content_stripped:
            # Check if it looks like CSV
            lines = content_stripped.split('\n')[:5]  # Check first 5 lines
            if all(',' in line for line in lines if line.strip()):
                return "csv"

        # Default to txt
        return "txt"

    def _parse_file_content(self, content: str, file_format: str) -> List[Triple]:
        """Parse file content based on format and return list of Triple objects."""
        triples = []

        try:
            if file_format == "json":
                triples = self._parse_json_format(content)
            elif file_format == "csv":
                triples = self._parse_csv_format(content)
            elif file_format == "txt":
                triples = self._parse_txt_format(content)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        except Exception as e:
            self.logger.error(f"Failed to parse {file_format} content: {str(e)}")
            raise

        return triples

    def _parse_json_format(self, content: str) -> List[Triple]:
        """Parse JSON format reference graph."""
        data = json.loads(content)
        triples = []

        if isinstance(data, list):
            # List of triples
            for item in data:
                if isinstance(item, list) and len(item) >= 3:
                    # [subject, predicate, object] format
                    triples.append(Triple(str(item[0]), str(item[1]), str(item[2])))
                elif isinstance(item, dict):
                    # GraphJudge approval format: {"triple": {...}, "judgment": true, "approved": true, ...}
                    if 'triple' in item and isinstance(item['triple'], dict):
                        # Check if this is an approved triple (prefer 'approved' field, fallback to 'judgment')
                        is_approved = item.get('approved', item.get('judgment', True))

                        if is_approved:
                            triple_data = item['triple']
                            if all(key in triple_data for key in ['subject', 'predicate', 'object']):
                                # Create Triple object with optional source_text
                                source_text = triple_data.get('source_text')
                                metadata = triple_data.get('metadata', {})

                                triples.append(Triple(
                                    subject=str(triple_data['subject']),
                                    predicate=str(triple_data['predicate']),
                                    object=str(triple_data['object']),
                                    source_text=source_text,
                                    metadata=metadata
                                ))
                    # {"subject": "", "predicate": "", "object": ""} format
                    elif all(key in item for key in ['subject', 'predicate', 'object']):
                        triples.append(Triple(str(item['subject']), str(item['predicate']), str(item['object'])))
                    elif all(key in item for key in ['s', 'p', 'o']):
                        triples.append(Triple(str(item['s']), str(item['p']), str(item['o'])))

        elif isinstance(data, dict):
            # Handle various dictionary formats
            if 'triples' in data:
                # {"triples": [...]}
                return self._parse_json_format(json.dumps(data['triples']))
            elif 'graph' in data:
                # {"graph": {...}}
                if 'edges' in data['graph']:
                    for edge in data['graph']['edges']:
                        if isinstance(edge, dict) and all(key in edge for key in ['source', 'target', 'label']):
                            triples.append(Triple(str(edge['source']), str(edge['label']), str(edge['target'])))

        return triples

    def _parse_csv_format(self, content: str) -> List[Triple]:
        """Parse CSV format reference graph."""
        triples = []

        # Try different CSV dialects
        for delimiter in [',', '\t', ';']:
            try:
                reader = csv.reader(content.strip().split('\n'), delimiter=delimiter)
                temp_triples = []

                for row_idx, row in enumerate(reader):
                    if len(row) >= 3:
                        # Skip header row if it looks like headers
                        if row_idx == 0 and any(header.lower() in ['subject', 'predicate', 'object', 'relation'] for header in row):
                            continue
                        temp_triples.append(Triple(str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip()))

                # Use the delimiter that gives us the most triples
                if len(temp_triples) > len(triples):
                    triples = temp_triples

            except Exception:
                continue

        return triples

    def _parse_txt_format(self, content: str) -> List[Triple]:
        """Parse text format reference graph."""
        triples = []

        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try different separators
            for separator in ['\t', '|', ',', ' ']:
                parts = line.split(separator)
                if len(parts) >= 3:
                    # Clean up parts
                    parts = [part.strip().strip('"').strip("'") for part in parts[:3]]
                    if all(parts):  # Ensure no empty parts
                        triples.append(Triple(parts[0], parts[1], parts[2]))
                        break

        return triples

    def _validate_triples(self, triples: List[Triple]) -> Tuple[bool, Optional[str]]:
        """Validate that triples are properly formatted."""
        if not triples:
            return False, "No triples found"

        for i, triple in enumerate(triples):
            if not isinstance(triple, Triple):
                return False, f"Item {i} is not a valid Triple object"

            if not all([triple.subject, triple.predicate, triple.object]):
                return False, f"Triple {i} has empty subject, predicate, or object"

            # Check for reasonable length limits
            if any(len(str(field)) > 500 for field in [triple.subject, triple.predicate, triple.object]):
                return False, f"Triple {i} has excessively long fields (>500 characters)"

        return True, None

    def _store_uploaded_file(self, filename: str, content: str, triples: List[Triple]):
        """Store uploaded file information for cleanup and reference."""
        timestamp = datetime.now()
        file_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{filename}"

        # Store file content to temporary location
        temp_file_path = self.temp_dir / file_id

        try:
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.uploaded_files[file_id] = {
                'original_filename': filename,
                'file_path': str(temp_file_path),
                'upload_time': timestamp,
                'triple_count': len(triples),
                'file_size': len(content)
            }

        except Exception as e:
            self.logger.warning(f"Failed to store uploaded file {filename}: {str(e)}")

    def convert_to_evaluation_format(self, triples: List[Triple]) -> List[List[str]]:
        """
        Convert Triple objects to the format expected by evaluation metrics.

        Args:
            triples: List of Triple objects

        Returns:
            List of [subject, predicate, object] string lists
        """
        return [[t.subject, t.predicate, t.object] for t in triples]

    def get_graph_statistics(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Generate statistics about a reference graph.

        Args:
            triples: List of Triple objects

        Returns:
            Dictionary containing graph statistics
        """
        if not triples:
            return {"size": 0, "subjects": 0, "predicates": 0, "objects": 0}

        subjects = set(t.subject for t in triples)
        predicates = set(t.predicate for t in triples)
        objects = set(t.object for t in triples)

        return {
            "size": len(triples),
            "unique_subjects": len(subjects),
            "unique_predicates": len(predicates),
            "unique_objects": len(objects),
            "subject_examples": list(subjects)[:5],
            "predicate_examples": list(predicates)[:5],
            "object_examples": list(objects)[:5],
            "most_common_predicates": self._get_most_common_predicates(triples, top_k=5)
        }

    def _get_most_common_predicates(self, triples: List[Triple], top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most common predicates in the graph."""
        from collections import Counter
        predicate_counts = Counter(t.predicate for t in triples)
        return predicate_counts.most_common(top_k)

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old uploaded files."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        files_to_remove = []

        for file_id, file_info in self.uploaded_files.items():
            if file_info['upload_time'] < cutoff_time:
                try:
                    file_path = Path(file_info['file_path'])
                    if file_path.exists():
                        file_path.unlink()
                    files_to_remove.append(file_id)
                except Exception as e:
                    self.logger.warning(f"Failed to remove old file {file_id}: {str(e)}")

        for file_id in files_to_remove:
            del self.uploaded_files[file_id]

        if files_to_remove:
            self.logger.info(f"Cleaned up {len(files_to_remove)} old reference graph files")

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.config['supported_formats']

    def get_upload_summary(self) -> Dict[str, Any]:
        """Get summary of uploaded reference graphs."""
        return {
            "total_files": len(self.uploaded_files),
            "files": [
                {
                    "id": file_id,
                    "filename": info['original_filename'],
                    "upload_time": info['upload_time'].isoformat(),
                    "triple_count": info['triple_count'],
                    "file_size": info['file_size']
                }
                for file_id, info in self.uploaded_files.items()
            ]
        }


# Convenience functions for easy integration
def upload_reference_graph(uploaded_file, file_format: str = "auto") -> Tuple[bool, Optional[List[Triple]], Optional[str]]:
    """
    Convenience function to upload a reference graph.

    Args:
        uploaded_file: File to upload
        file_format: Format of the file

    Returns:
        Tuple of (success, triples_list, error_message)
    """
    manager = ReferenceGraphManager()
    return manager.upload_reference_graph(uploaded_file, file_format)


def validate_reference_graph(triples: List[Triple]) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate a reference graph.

    Args:
        triples: List of Triple objects to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    manager = ReferenceGraphManager()
    return manager._validate_triples(triples)


def get_reference_graph_stats(triples: List[Triple]) -> Dict[str, Any]:
    """
    Convenience function to get reference graph statistics.

    Args:
        triples: List of Triple objects

    Returns:
        Dictionary containing graph statistics
    """
    manager = ReferenceGraphManager()
    return manager.get_graph_statistics(triples)