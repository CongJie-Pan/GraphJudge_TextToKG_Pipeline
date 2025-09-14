"""
Storage Manager for GraphJudge Streamlit Pipeline.

This module handles persistent storage of pipeline results across iterations,
ensuring that each phase's output is saved to the appropriate iteration folder
and can be retrieved by subsequent phases.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from ..core.models import EntityResult, TripleResult, JudgmentResult, Triple
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import EntityResult, TripleResult, JudgmentResult, Triple


class StorageManager:
    """
    Manages persistent storage of pipeline results in iteration folders.

    Each pipeline run creates a new iteration folder (iteration_1, iteration_2, etc.)
    containing the outputs from all three phases: ECTD, Triple Generation, and Graph Judgment.
    """

    def __init__(self, base_path: str = None):
        """Initialize storage manager with base path."""
        if base_path is None:
            # Default to streamlit_pipeline/datasets
            current_dir = Path(__file__).parent.parent
            base_path = current_dir / "datasets"

        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Current iteration info
        self.current_iteration = None
        self.current_iteration_path = None

    def get_next_iteration_number(self) -> int:
        """
        Find the next available iteration number.

        Returns:
            Next iteration number (1, 2, 3, etc.)
        """
        existing_iterations = glob.glob(str(self.base_path / "iteration_*"))

        if not existing_iterations:
            return 1

        # Extract numbers from iteration folders
        numbers = []
        for path in existing_iterations:
            folder_name = os.path.basename(path)
            if folder_name.startswith("iteration_"):
                try:
                    num = int(folder_name.split("_")[1])
                    numbers.append(num)
                except (ValueError, IndexError):
                    continue

        return max(numbers, default=0) + 1

    def create_new_iteration(self, input_text: str = None) -> str:
        """
        Create a new iteration folder for the current pipeline run.

        Args:
            input_text: Optional input text for this iteration

        Returns:
            Path to the created iteration folder
        """
        iteration_num = self.get_next_iteration_number()
        iteration_folder = f"iteration_{iteration_num}"
        iteration_path = self.base_path / iteration_folder

        # Create iteration folder
        iteration_path.mkdir(exist_ok=True)

        # Create phase subfolders
        (iteration_path / "ectd").mkdir(exist_ok=True)
        (iteration_path / "triples").mkdir(exist_ok=True)
        (iteration_path / "judgment").mkdir(exist_ok=True)

        # Store iteration metadata
        metadata = {
            "iteration_number": iteration_num,
            "created_at": datetime.now().isoformat(),
            "input_text": input_text[:200] + "..." if input_text and len(input_text) > 200 else input_text,
            "input_length": len(input_text) if input_text else 0,
            "phases_completed": []
        }

        with open(iteration_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Set current iteration
        self.current_iteration = iteration_num
        self.current_iteration_path = iteration_path

        print(f"[STORAGE] Created iteration_{iteration_num} folder")
        return str(iteration_path)

    def save_entity_result(self, entity_result: EntityResult) -> str:
        """
        Save ECTD phase results to the current iteration folder.

        Args:
            entity_result: EntityResult object containing entities and denoised text

        Returns:
            Path to saved file
        """
        if not self.current_iteration_path:
            raise ValueError("No current iteration. Call create_new_iteration() first.")

        ectd_folder = self.current_iteration_path / "ectd"

        # Save entities as JSON
        entities_data = {
            "entities": entity_result.entities,
            "denoised_text": entity_result.denoised_text,
            "success": entity_result.success,
            "processing_time": entity_result.processing_time,
            "error": entity_result.error,
            "timestamp": datetime.now().isoformat()
        }

        entities_file = ectd_folder / "entities.json"
        with open(entities_file, "w", encoding="utf-8") as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)

        # Save denoised text separately for easy reading
        denoised_file = ectd_folder / "denoised_text.txt"
        with open(denoised_file, "w", encoding="utf-8") as f:
            f.write(entity_result.denoised_text)

        # Update metadata
        self._update_metadata("ectd", {"entities_count": len(entity_result.entities)})

        print(f"[STORAGE] Saved ECTD results: {len(entity_result.entities)} entities")
        return str(entities_file)

    def save_triple_result(self, triple_result: TripleResult) -> str:
        """
        Save Triple Generation phase results to the current iteration folder.

        Args:
            triple_result: TripleResult object containing generated triples

        Returns:
            Path to saved file
        """
        if not self.current_iteration_path:
            raise ValueError("No current iteration. Call create_new_iteration() first.")

        triples_folder = self.current_iteration_path / "triples"

        # Convert triples to serializable format
        triples_data = []
        for triple in triple_result.triples:
            triples_data.append({
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "source_text": triple.source_text,
                "metadata": triple.metadata
            })

        # Save triples as JSON
        result_data = {
            "triples": triples_data,
            "metadata": triple_result.metadata,
            "success": triple_result.success,
            "processing_time": triple_result.processing_time,
            "error": triple_result.error,
            "timestamp": datetime.now().isoformat()
        }

        triples_file = triples_folder / "triples.json"
        with open(triples_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # Save triples in human-readable format
        readable_file = triples_folder / "triples_readable.txt"
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write(f"Generated Triples ({len(triple_result.triples)} total)\n")
            f.write("=" * 50 + "\n\n")

            for i, triple in enumerate(triple_result.triples, 1):
                f.write(f"{i}. {triple.subject} -> {triple.predicate} -> {triple.object}\n")

        # Update metadata
        self._update_metadata("triples", {"triples_count": len(triple_result.triples)})

        print(f"[STORAGE] Saved Triple results: {len(triple_result.triples)} triples")
        return str(triples_file)

    def save_judgment_result(self, judgment_result: JudgmentResult, original_triples: List[Triple]) -> str:
        """
        Save Graph Judgment phase results to the current iteration folder.

        Args:
            judgment_result: JudgmentResult object containing judgments
            original_triples: Original triples that were judged

        Returns:
            Path to saved file
        """
        if not self.current_iteration_path:
            raise ValueError("No current iteration. Call create_new_iteration() first.")

        judgment_folder = self.current_iteration_path / "judgment"

        # Combine judgments with original triples
        judged_triples = []
        for i, (triple, judgment) in enumerate(zip(original_triples, judgment_result.judgments)):
            confidence = judgment_result.confidence[i] if i < len(judgment_result.confidence) else 0.0

            judged_triples.append({
                "triple": {
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "source_text": triple.source_text
                },
                "judgment": judgment,
                "confidence": confidence,
                "approved": bool(judgment)
            })

        # Save judgment results
        result_data = {
            "judged_triples": judged_triples,
            "explanations": judgment_result.explanations,
            "success": judgment_result.success,
            "processing_time": judgment_result.processing_time,
            "error": judgment_result.error,
            "summary": {
                "total_triples": len(original_triples),
                "approved_triples": sum(1 for j in judgment_result.judgments if j),
                "rejected_triples": sum(1 for j in judgment_result.judgments if not j),
                "approval_rate": sum(1 for j in judgment_result.judgments if j) / len(judgment_result.judgments) if judgment_result.judgments else 0
            },
            "timestamp": datetime.now().isoformat()
        }

        judgment_file = judgment_folder / "judgment.json"
        with open(judgment_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # Save approved triples separately
        approved_triples = [jt for jt in judged_triples if jt["approved"]]
        approved_file = judgment_folder / "approved_triples.json"
        with open(approved_file, "w", encoding="utf-8") as f:
            json.dump(approved_triples, f, ensure_ascii=False, indent=2)

        # Update metadata
        self._update_metadata("judgment", {
            "approved_triples": len(approved_triples),
            "total_triples": len(original_triples),
            "approval_rate": result_data["summary"]["approval_rate"]
        })

        print(f"[STORAGE] Saved Judgment results: {len(approved_triples)}/{len(original_triples)} approved")
        return str(judgment_file)

    def load_entity_result(self, iteration_path: str = None) -> Optional[EntityResult]:
        """
        Load ECTD results from an iteration folder.

        Args:
            iteration_path: Optional path to iteration folder (uses current if not provided)

        Returns:
            EntityResult object or None if not found
        """
        if iteration_path:
            folder_path = Path(iteration_path)
        else:
            folder_path = self.current_iteration_path

        if not folder_path:
            return None

        entities_file = folder_path / "ectd" / "entities.json"
        if not entities_file.exists():
            return None

        try:
            with open(entities_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return EntityResult(
                entities=data["entities"],
                denoised_text=data["denoised_text"],
                success=data["success"],
                processing_time=data["processing_time"],
                error=data["error"]
            )
        except Exception as e:
            print(f"[STORAGE] Error loading entity result: {e}")
            return None

    def load_triple_result(self, iteration_path: str = None) -> Optional[TripleResult]:
        """
        Load Triple Generation results from an iteration folder.

        Args:
            iteration_path: Optional path to iteration folder (uses current if not provided)

        Returns:
            TripleResult object or None if not found
        """
        if iteration_path:
            folder_path = Path(iteration_path)
        else:
            folder_path = self.current_iteration_path

        if not folder_path:
            return None

        triples_file = folder_path / "triples" / "triples.json"
        if not triples_file.exists():
            return None

        try:
            with open(triples_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct Triple objects
            triples = []
            for triple_data in data["triples"]:
                triple = Triple(
                    subject=triple_data["subject"],
                    predicate=triple_data["predicate"],
                    object=triple_data["object"],
                    source_text=triple_data.get("source_text", ""),
                    metadata=triple_data.get("metadata", {})
                )
                triples.append(triple)

            return TripleResult(
                triples=triples,
                metadata=data["metadata"],
                success=data["success"],
                processing_time=data["processing_time"],
                error=data["error"]
            )
        except Exception as e:
            print(f"[STORAGE] Error loading triple result: {e}")
            return None

    def _update_metadata(self, phase: str, phase_data: Dict[str, Any]):
        """Update iteration metadata with phase completion info."""
        if not self.current_iteration_path:
            return

        metadata_file = self.current_iteration_path / "metadata.json"

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Add phase to completed phases
            if phase not in metadata["phases_completed"]:
                metadata["phases_completed"].append(phase)

            # Add phase-specific data
            if "phase_data" not in metadata:
                metadata["phase_data"] = {}
            metadata["phase_data"][phase] = phase_data
            metadata["phase_data"][phase]["completed_at"] = datetime.now().isoformat()

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[STORAGE] Error updating metadata: {e}")

    def get_iteration_summary(self, iteration_path: str = None) -> Dict[str, Any]:
        """
        Get summary information about an iteration.

        Args:
            iteration_path: Optional path to iteration folder (uses current if not provided)

        Returns:
            Dictionary containing iteration summary
        """
        if iteration_path:
            folder_path = Path(iteration_path)
        else:
            folder_path = self.current_iteration_path

        if not folder_path:
            return {}

        metadata_file = folder_path / "metadata.json"
        if not metadata_file.exists():
            return {}

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """
    Get or create the global storage manager instance.

    Returns:
        StorageManager instance
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


def create_new_pipeline_iteration(input_text: str) -> str:
    """
    Convenience function to create a new pipeline iteration.

    Args:
        input_text: Input text for this pipeline run

    Returns:
        Path to created iteration folder
    """
    storage_manager = get_storage_manager()
    return storage_manager.create_new_iteration(input_text)


def save_phase_result(phase: str, result: Any, additional_data: Any = None) -> str:
    """
    Convenience function to save phase results.

    Args:
        phase: Phase name ("ectd", "triples", "judgment")
        result: Phase result object
        additional_data: Additional data for some phases

    Returns:
        Path to saved file
    """
    storage_manager = get_storage_manager()

    if phase == "ectd":
        return storage_manager.save_entity_result(result)
    elif phase == "triples":
        return storage_manager.save_triple_result(result)
    elif phase == "judgment":
        return storage_manager.save_judgment_result(result, additional_data)
    else:
        raise ValueError(f"Unknown phase: {phase}")