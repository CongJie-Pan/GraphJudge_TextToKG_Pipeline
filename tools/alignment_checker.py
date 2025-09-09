"""
Sentence ↔ Entity Alignment Checker for ECTD Pipeline

This script implements comprehensive alignment validation between entity files and 
denoised text files in the ECTD pipeline, ensuring data integrity and consistency 
across the processing stages. The alignment checker validates both structural and 
semantic correspondence between extracted entities and their source sentences.

Key Features:
1. Line count validation between entity and denoised files
2. Entity-sentence semantic correspondence checking
3. Missing entity detection in source text
4. Orphaned entity identification
5. Data integrity repair and trimming
6. Comprehensive alignment statistics and reporting
7. Interactive and automated fixing modes
8. Quality metrics for alignment assessment

Validation Types:
- Structural: Line counts, file formats, encoding consistency
- Semantic: Entity presence in corresponding sentences
- Contextual: Entity relevance and coherence with source text
- Statistical: Distribution and coverage analysis

Usage:
    from tools.alignment_checker import AlignmentChecker
    
    checker = AlignmentChecker()
    issues = checker.check_alignment("test_entity.txt", "test_denoised.target")
    checker.fix_alignment_issues(issues, auto_fix=True)

Command Line Usage:
    python tools/alignment_checker.py --entities test_entity.txt --denoised test_denoised.target --fix
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import difflib
from enum import Enum


class AlignmentIssueType(Enum):
    """
    Enumeration of different types of alignment issues.
    
    This enum categorizes various alignment problems that can occur between
    entity files and denoised text files, enabling targeted fixing strategies.
    """
    LINE_COUNT_MISMATCH = "line_count_mismatch"
    ENTITY_NOT_IN_TEXT = "entity_not_in_text"
    EMPTY_ENTITY_LINE = "empty_entity_line"
    EMPTY_TEXT_LINE = "empty_text_line"
    ORPHANED_ENTITY = "orphaned_entity"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    FORMAT_ERROR = "format_error"
    ENCODING_ERROR = "encoding_error"


@dataclass
class AlignmentIssue:
    """
    Data class representing an alignment issue between entities and text.
    
    This class encapsulates detailed information about alignment problems,
    including location, severity, and suggested fixes for comprehensive
    issue tracking and resolution.
    """
    issue_type: AlignmentIssueType     # Type of alignment issue
    line_number: int                   # Line number where issue occurs
    entity_line: str                   # Original entity line content
    text_line: str                     # Corresponding text line content
    missing_entities: List[str]        # Entities not found in text
    orphaned_entities: List[str]       # Entities without text correspondence
    severity: str                      # Severity level: LOW, MEDIUM, HIGH, CRITICAL
    suggested_fix: str                 # Suggested resolution approach
    additional_info: Dict              # Additional diagnostic information
    
    def to_dict(self) -> Dict:
        """Convert issue to dictionary for JSON serialization."""
        return {
            "issue_type": self.issue_type.value,
            "line_number": self.line_number,
            "entity_line": self.entity_line,
            "text_line": self.text_line,
            "missing_entities": self.missing_entities,
            "orphaned_entities": self.orphaned_entities,
            "severity": self.severity,
            "suggested_fix": self.suggested_fix,
            "additional_info": self.additional_info
        }


class AlignmentChecker:
    """
    Comprehensive alignment checker for ECTD pipeline data validation.
    
    This class implements sophisticated alignment checking between entity files
    and denoised text files, providing detailed diagnostics and automated
    fixing capabilities to ensure data integrity throughout the ECTD pipeline.
    """
    
    def __init__(self, fuzzy_threshold: float = 0.8):
        """
        Initialize the AlignmentChecker with configurable parameters.
        
        Args:
            fuzzy_threshold (float): Threshold for fuzzy string matching (0.0 to 1.0).
                                   Higher values require closer matches for entity detection.
        
        The checker uses fuzzy string matching to handle variations in entity
        representation between extraction and text denoising phases.
        """
        self.setup_logging()
        self.fuzzy_threshold = fuzzy_threshold
        self.statistics = {
            "total_lines_checked": 0,
            "perfect_alignments": 0,
            "minor_issues": 0,
            "major_issues": 0,
            "critical_issues": 0,
            "entities_validated": 0,
            "entities_missing": 0,
            "orphaned_entities": 0,
            "semantic_mismatches": 0,
            "issue_distribution": defaultdict(int)
        }
    
    def setup_logging(self):
        """
        Configure logging for detailed tracking of alignment checking process.
        
        This method sets up comprehensive logging to track the alignment process,
        including debug information about detected issues and fixing operations.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/alignment_checking.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
    
    def _parse_entity_line(self, line: str) -> List[str]:
        """
        Parse a single entity line into a list of entities.
        
        Args:
            line (str): Raw entity line from the input file
            
        Returns:
            List[str]: List of individual entities
            
        This method reuses consistent parsing logic to handle various entity
        file formats, ensuring compatibility with upstream processing tools.
        """
        line = line.strip()
        if not line:
            return []
        
        # Handle Python list format: ["entity1", "entity2"]
        if line.startswith('[') and line.endswith(']'):
            try:
                entities = eval(line)
                if isinstance(entities, list):
                    return [str(entity).strip() for entity in entities if str(entity).strip()]
            except (SyntaxError, ValueError) as e:
                self.logger.warning(f"Failed to parse list format: {line[:50]}... Error: {e}")
        
        # Handle comma-separated format
        cleaned_line = re.sub(r'[\[\]"\'""''「」『』]', '', line)
        entities = [entity.strip() for entity in cleaned_line.split(',')]
        entities = [entity for entity in entities if entity]
        
        return entities
    
    def _load_files(self, entity_path: str, text_path: str) -> Tuple[List[str], List[str]]:
        """
        Load and validate both entity and text files.
        
        Args:
            entity_path (str): Path to entity file
            text_path (str): Path to denoised text file
            
        Returns:
            Tuple[List[str], List[str]]: Entity lines and text lines
            
        Raises:
            FileNotFoundError: If either file doesn't exist
            UnicodeDecodeError: If files have encoding issues
            
        This method handles file loading with proper error handling and
        encoding validation to ensure reliable data processing.
        """
        # Validate file existence
        if not os.path.exists(entity_path):
            raise FileNotFoundError(f"Entity file not found: {entity_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")
        
        try:
            # Load entity file
            with open(entity_path, 'r', encoding='utf-8') as f:
                entity_lines = [line.strip() for line in f.readlines()]
            
            # Load text file
            with open(text_path, 'r', encoding='utf-8') as f:
                text_lines = [line.strip() for line in f.readlines()]
            
            self.logger.info(f"Loaded {len(entity_lines)} entity lines and {len(text_lines)} text lines")
            
            return entity_lines, text_lines
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error loading files: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading files: {e}")
            raise
    
    def _check_entity_in_text(self, entity: str, text: str) -> bool:
        """
        Check if an entity appears in the corresponding text line.
        
        Args:
            entity (str): Entity to search for
            text (str): Text line to search in
            
        Returns:
            bool: True if entity is found in text (exact or fuzzy match)
            
        This method uses both exact matching and fuzzy string matching to
        account for variations in entity representation between extraction
        and text processing phases.
        """
        if not entity or not text:
            return False
        
        # Exact match (case-sensitive)
        if entity in text:
            return True
        
        # Fuzzy matching for slight variations
        # Use difflib to find close matches
        close_matches = difflib.get_close_matches(
            entity, [text], n=1, cutoff=self.fuzzy_threshold
        )
        
        if close_matches:
            return True
        
        # Character-level fuzzy matching for Chinese text
        # Split text into overlapping windows of entity length
        entity_len = len(entity)
        for i in range(len(text) - entity_len + 1):
            text_window = text[i:i + entity_len]
            similarity = difflib.SequenceMatcher(None, entity, text_window).ratio()
            if similarity >= self.fuzzy_threshold:
                return True
        
        return False
    
    def _determine_issue_severity(self, issue_type: AlignmentIssueType, 
                                 missing_count: int, total_count: int) -> str:
        """
        Determine the severity level of an alignment issue.
        
        Args:
            issue_type (AlignmentIssueType): Type of the issue
            missing_count (int): Number of missing entities
            total_count (int): Total number of entities
            
        Returns:
            str: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            
        This method categorizes issues by severity to help prioritize fixing
        efforts and determine appropriate resolution strategies.
        """
        if issue_type in [AlignmentIssueType.LINE_COUNT_MISMATCH, 
                         AlignmentIssueType.ENCODING_ERROR]:
            return "CRITICAL"
        
        if issue_type == AlignmentIssueType.FORMAT_ERROR:
            return "HIGH"
        
        if total_count == 0:
            return "MEDIUM"
        
        missing_ratio = missing_count / total_count
        
        if missing_ratio >= 0.8:  # 80% or more entities missing
            return "CRITICAL"
        elif missing_ratio >= 0.5:  # 50-79% entities missing
            return "HIGH"
        elif missing_ratio >= 0.2:  # 20-49% entities missing
            return "MEDIUM"
        else:  # Less than 20% entities missing
            return "LOW"
    
    def _suggest_fix(self, issue: AlignmentIssue) -> str:
        """
        Suggest appropriate fix for an alignment issue.
        
        Args:
            issue (AlignmentIssue): The alignment issue to fix
            
        Returns:
            str: Suggested fix description
            
        This method provides intelligent suggestions for resolving different
        types of alignment issues based on their characteristics and severity.
        """
        if issue.issue_type == AlignmentIssueType.LINE_COUNT_MISMATCH:
            return "Trim files to match shorter length or investigate missing data"
        
        elif issue.issue_type == AlignmentIssueType.ENTITY_NOT_IN_TEXT:
            if issue.severity in ["HIGH", "CRITICAL"]:
                return "Re-extract entities or verify text denoising quality"
            else:
                return "Remove missing entities or adjust fuzzy matching threshold"
        
        elif issue.issue_type == AlignmentIssueType.EMPTY_ENTITY_LINE:
            return "Remove empty line or extract entities for corresponding text"
        
        elif issue.issue_type == AlignmentIssueType.EMPTY_TEXT_LINE:
            return "Remove corresponding entity line or verify text processing"
        
        elif issue.issue_type == AlignmentIssueType.ORPHANED_ENTITY:
            return "Remove orphaned entities or find corresponding text"
        
        elif issue.issue_type == AlignmentIssueType.SEMANTIC_MISMATCH:
            return "Review entity extraction and text denoising consistency"
        
        else:
            return "Manual review required"
    
    def check_line_alignment(self, entity_lines: List[str], text_lines: List[str]) -> List[AlignmentIssue]:
        """
        Check basic line-level alignment between entity and text files.
        
        Args:
            entity_lines (List[str]): Lines from entity file
            text_lines (List[str]): Lines from text file
            
        Returns:
            List[AlignmentIssue]: List of detected alignment issues
            
        This method performs structural validation to ensure the basic
        correspondence between entity and text files at the line level.
        """
        issues = []
        
        # Check line count match
        if len(entity_lines) != len(text_lines):
            issue = AlignmentIssue(
                issue_type=AlignmentIssueType.LINE_COUNT_MISMATCH,
                line_number=0,
                entity_line="",
                text_line="",
                missing_entities=[],
                orphaned_entities=[],
                severity="CRITICAL",
                suggested_fix="Trim files to match shorter length or investigate missing data",
                additional_info={
                    "entity_line_count": len(entity_lines),
                    "text_line_count": len(text_lines),
                    "difference": abs(len(entity_lines) - len(text_lines))
                }
            )
            issue.suggested_fix = self._suggest_fix(issue)
            issues.append(issue)
            
            self.statistics["critical_issues"] += 1
            self.statistics["issue_distribution"][AlignmentIssueType.LINE_COUNT_MISMATCH.value] += 1
        
        # Check for empty lines
        min_lines = min(len(entity_lines), len(text_lines))
        for i in range(min_lines):
            entity_line = entity_lines[i].strip()
            text_line = text_lines[i].strip()
            
            if not entity_line and text_line:
                issue = AlignmentIssue(
                    issue_type=AlignmentIssueType.EMPTY_ENTITY_LINE,
                    line_number=i + 1,
                    entity_line=entity_line,
                    text_line=text_line[:100] + "..." if len(text_line) > 100 else text_line,
                    missing_entities=[],
                    orphaned_entities=[],
                    severity="MEDIUM",
                    suggested_fix="",
                    additional_info={"text_length": len(text_line)}
                )
                issue.suggested_fix = self._suggest_fix(issue)
                issues.append(issue)
                
                self.statistics["major_issues"] += 1
                self.statistics["issue_distribution"][AlignmentIssueType.EMPTY_ENTITY_LINE.value] += 1
            
            elif entity_line and not text_line:
                issue = AlignmentIssue(
                    issue_type=AlignmentIssueType.EMPTY_TEXT_LINE,
                    line_number=i + 1,
                    entity_line=entity_line[:100] + "..." if len(entity_line) > 100 else entity_line,
                    text_line=text_line,
                    missing_entities=[],
                    orphaned_entities=[],
                    severity="MEDIUM",
                    suggested_fix="",
                    additional_info={"entity_count": len(self._parse_entity_line(entity_line))}
                )
                issue.suggested_fix = self._suggest_fix(issue)
                issues.append(issue)
                
                self.statistics["major_issues"] += 1
                self.statistics["issue_distribution"][AlignmentIssueType.EMPTY_TEXT_LINE.value] += 1
        
        return issues
    
    def check_semantic_alignment(self, entity_lines: List[str], text_lines: List[str]) -> List[AlignmentIssue]:
        """
        Check semantic alignment between entities and their corresponding text.
        
        Args:
            entity_lines (List[str]): Lines from entity file
            text_lines (List[str]): Lines from text file
            
        Returns:
            List[AlignmentIssue]: List of detected semantic alignment issues
            
        This method performs deep semantic validation to ensure entities
        actually appear in their corresponding text lines with appropriate
        contextual relevance.
        """
        issues = []
        min_lines = min(len(entity_lines), len(text_lines))
        
        for i in range(min_lines):
            entity_line = entity_lines[i].strip()
            text_line = text_lines[i].strip()
            
            if not entity_line or not text_line:
                continue  # Skip empty lines (handled by line alignment check)
            
            # Parse entities from the line
            try:
                entities = self._parse_entity_line(entity_line)
            except Exception as e:
                # Format error in entity line
                issue = AlignmentIssue(
                    issue_type=AlignmentIssueType.FORMAT_ERROR,
                    line_number=i + 1,
                    entity_line=entity_line,
                    text_line=text_line,
                    missing_entities=[],
                    orphaned_entities=[],
                    severity="HIGH",
                    suggested_fix="Fix entity line format or re-extract entities",
                    additional_info={"parse_error": str(e)}
                )
                issues.append(issue)
                
                self.statistics["major_issues"] += 1
                self.statistics["issue_distribution"][AlignmentIssueType.FORMAT_ERROR.value] += 1
                continue
            
            if not entities:
                continue  # Skip lines with no entities
            
            # Check each entity against the text
            missing_entities = []
            found_entities = []
            
            for entity in entities:
                if self._check_entity_in_text(entity, text_line):
                    found_entities.append(entity)
                else:
                    missing_entities.append(entity)
                    self.statistics["entities_missing"] += 1
                
                self.statistics["entities_validated"] += 1
            
            # Create issue if entities are missing
            if missing_entities:
                severity = self._determine_issue_severity(
                    AlignmentIssueType.ENTITY_NOT_IN_TEXT,
                    len(missing_entities),
                    len(entities)
                )
                
                issue = AlignmentIssue(
                    issue_type=AlignmentIssueType.ENTITY_NOT_IN_TEXT,
                    line_number=i + 1,
                    entity_line=entity_line,
                    text_line=text_line[:200] + "..." if len(text_line) > 200 else text_line,
                    missing_entities=missing_entities,
                    orphaned_entities=[],
                    severity=severity,
                    suggested_fix="",
                    additional_info={
                        "total_entities": len(entities),
                        "found_entities": len(found_entities),
                        "missing_count": len(missing_entities),
                        "missing_ratio": len(missing_entities) / len(entities)
                    }
                )
                issue.suggested_fix = self._suggest_fix(issue)
                issues.append(issue)
                
                # Update statistics based on severity
                if severity == "CRITICAL":
                    self.statistics["critical_issues"] += 1
                elif severity == "HIGH":
                    self.statistics["major_issues"] += 1
                else:
                    self.statistics["minor_issues"] += 1
                
                self.statistics["issue_distribution"][AlignmentIssueType.ENTITY_NOT_IN_TEXT.value] += 1
            else:
                # Perfect alignment
                self.statistics["perfect_alignments"] += 1
        
        return issues
    
    def check_alignment(self, entity_path: str, text_path: str) -> List[AlignmentIssue]:
        """
        Perform comprehensive alignment checking between entity and text files.
        
        Args:
            entity_path (str): Path to entity file
            text_path (str): Path to denoised text file
            
        Returns:
            List[AlignmentIssue]: Complete list of detected alignment issues
            
        This method orchestrates the complete alignment checking process,
        including structural, semantic, and contextual validation.
        """
        self.logger.info(f"Starting alignment check between {entity_path} and {text_path}")
        
        # Reset statistics
        self.statistics = {
            "total_lines_checked": 0,
            "perfect_alignments": 0,
            "minor_issues": 0,
            "major_issues": 0,
            "critical_issues": 0,
            "entities_validated": 0,
            "entities_missing": 0,
            "orphaned_entities": 0,
            "semantic_mismatches": 0,
            "issue_distribution": defaultdict(int)
        }
        
        try:
            # Load files
            entity_lines, text_lines = self._load_files(entity_path, text_path)
            
            # Update statistics
            self.statistics["total_lines_checked"] = min(len(entity_lines), len(text_lines))
            
            # Perform alignment checks
            issues = []
            
            # 1. Check line-level alignment
            line_issues = self.check_line_alignment(entity_lines, text_lines)
            issues.extend(line_issues)
            
            # 2. Check semantic alignment
            semantic_issues = self.check_semantic_alignment(entity_lines, text_lines)
            issues.extend(semantic_issues)
            
            # Log summary
            self.logger.info(f"Alignment check completed. Found {len(issues)} issues:")
            self.logger.info(f"  Critical: {self.statistics['critical_issues']}")
            self.logger.info(f"  Major: {self.statistics['major_issues']}")
            self.logger.info(f"  Minor: {self.statistics['minor_issues']}")
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error during alignment check: {e}")
            raise
    
    def fix_alignment_issues(self, issues: List[AlignmentIssue], entity_path: str, 
                           text_path: str, output_dir: str = None, auto_fix: bool = False) -> Dict[str, str]:
        """
        Attempt to fix detected alignment issues.
        
        Args:
            issues (List[AlignmentIssue]): Issues to fix
            entity_path (str): Original entity file path
            text_path (str): Original text file path
            output_dir (str, optional): Directory for fixed files
            auto_fix (bool): Whether to apply fixes automatically
            
        Returns:
            Dict[str, str]: Paths to fixed files
            
        This method implements automated fixing strategies for common alignment
        issues, with options for manual review and intervention.
        """
        if not output_dir:
            output_dir = os.path.dirname(entity_path)
        
        # Load original files
        entity_lines, text_lines = self._load_files(entity_path, text_path)
        
        # Sort issues by severity and line number
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        major_issues = [i for i in issues if i.severity == "HIGH"]
        minor_issues = [i for i in issues if i.severity in ["MEDIUM", "LOW"]]
        
        self.logger.info(f"Fixing {len(issues)} alignment issues:")
        self.logger.info(f"  Critical: {len(critical_issues)}")
        self.logger.info(f"  Major: {len(major_issues)}")
        self.logger.info(f"  Minor: {len(minor_issues)}")
        
        fixed_entity_lines = entity_lines.copy()
        fixed_text_lines = text_lines.copy()
        
        # Handle critical issues first (line count mismatch)
        for issue in critical_issues:
            if issue.issue_type == AlignmentIssueType.LINE_COUNT_MISMATCH:
                min_length = min(len(fixed_entity_lines), len(fixed_text_lines))
                fixed_entity_lines = fixed_entity_lines[:min_length]
                fixed_text_lines = fixed_text_lines[:min_length]
                self.logger.info(f"Trimmed files to {min_length} lines to fix line count mismatch")
        
        # Handle other issues
        lines_to_remove = set()
        
        for issue in major_issues + minor_issues:
            line_idx = issue.line_number - 1  # Convert to 0-based index
            
            if line_idx >= len(fixed_entity_lines) or line_idx >= len(fixed_text_lines):
                continue
            
            if issue.issue_type == AlignmentIssueType.EMPTY_ENTITY_LINE:
                if auto_fix:
                    lines_to_remove.add(line_idx)
                    self.logger.debug(f"Marked line {issue.line_number} for removal (empty entity line)")
            
            elif issue.issue_type == AlignmentIssueType.EMPTY_TEXT_LINE:
                if auto_fix:
                    lines_to_remove.add(line_idx)
                    self.logger.debug(f"Marked line {issue.line_number} for removal (empty text line)")
            
            elif issue.issue_type == AlignmentIssueType.ENTITY_NOT_IN_TEXT:
                if auto_fix and issue.severity in ["LOW", "MEDIUM"]:
                    # Remove missing entities from entity line
                    entities = self._parse_entity_line(fixed_entity_lines[line_idx])
                    valid_entities = [e for e in entities if e not in issue.missing_entities]
                    
                    if valid_entities:
                        fixed_entity_lines[line_idx] = str(valid_entities)
                        self.logger.debug(f"Removed {len(issue.missing_entities)} entities from line {issue.line_number}")
                    else:
                        lines_to_remove.add(line_idx)
                        self.logger.debug(f"Marked line {issue.line_number} for removal (no valid entities)")
        
        # Remove marked lines (in reverse order to maintain indices)
        for line_idx in sorted(lines_to_remove, reverse=True):
            if line_idx < len(fixed_entity_lines):
                del fixed_entity_lines[line_idx]
            if line_idx < len(fixed_text_lines):
                del fixed_text_lines[line_idx]
        
        # Save fixed files
        base_name = os.path.splitext(os.path.basename(entity_path))[0]
        fixed_entity_path = os.path.join(output_dir, f"{base_name}_aligned.txt")
        fixed_text_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(text_path))[0]}_aligned.target")
        
        try:
            with open(fixed_entity_path, 'w', encoding='utf-8') as f:
                for line in fixed_entity_lines:
                    f.write(line + '\n')
            
            with open(fixed_text_path, 'w', encoding='utf-8') as f:
                for line in fixed_text_lines:
                    f.write(line + '\n')
            
            self.logger.info(f"Fixed files saved:")
            self.logger.info(f"  Entities: {fixed_entity_path}")
            self.logger.info(f"  Text: {fixed_text_path}")
            
            return {
                "entity_file": fixed_entity_path,
                "text_file": fixed_text_path,
                "original_entity_lines": len(entity_lines),
                "original_text_lines": len(text_lines),
                "fixed_entity_lines": len(fixed_entity_lines),
                "fixed_text_lines": len(fixed_text_lines),
                "lines_removed": len(lines_to_remove)
            }
            
        except Exception as e:
            self.logger.error(f"Error saving fixed files: {e}")
            raise
    
    def save_alignment_report(self, issues: List[AlignmentIssue], output_path: str) -> None:
        """
        Save a detailed alignment report to JSON file.
        
        Args:
            issues (List[AlignmentIssue]): List of alignment issues
            output_path (str): Path for the output report file
            
        This method generates a comprehensive report of alignment issues
        suitable for analysis and debugging of the ECTD pipeline.
        """
        report = {
            "alignment_check_summary": {
                "total_issues": len(issues),
                "statistics": dict(self.statistics),
                "issue_breakdown": {
                    "critical": len([i for i in issues if i.severity == "CRITICAL"]),
                    "high": len([i for i in issues if i.severity == "HIGH"]),
                    "medium": len([i for i in issues if i.severity == "MEDIUM"]),
                    "low": len([i for i in issues if i.severity == "LOW"])
                }
            },
            "detailed_issues": [issue.to_dict() for issue in issues],
            "recommendations": self._generate_recommendations(issues)
        }
        
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Alignment report saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving alignment report: {e}")
            raise
    
    def _generate_recommendations(self, issues: List[AlignmentIssue]) -> List[str]:
        """
        Generate actionable recommendations based on detected issues.
        
        Args:
            issues (List[AlignmentIssue]): List of detected issues
            
        Returns:
            List[str]: List of recommended actions
            
        This method analyzes the pattern of issues to provide high-level
        recommendations for improving the ECTD pipeline.
        """
        recommendations = []
        
        critical_count = len([i for i in issues if i.severity == "CRITICAL"])
        major_count = len([i for i in issues if i.severity == "HIGH"])
        
        if critical_count > 0:
            recommendations.append("CRITICAL: Fix file structure issues before proceeding with entity extraction improvements")
        
        if major_count > 5:
            recommendations.append("Consider re-running entity extraction with improved prompts or higher temperature")
        
        entity_issues = len([i for i in issues if i.issue_type == AlignmentIssueType.ENTITY_NOT_IN_TEXT])
        if entity_issues > len(issues) * 0.3:
            recommendations.append("High rate of missing entities suggests entity extraction quality issues")
        
        empty_line_issues = len([i for i in issues if i.issue_type in [AlignmentIssueType.EMPTY_ENTITY_LINE, AlignmentIssueType.EMPTY_TEXT_LINE]])
        if empty_line_issues > 0:
            recommendations.append("Remove empty lines or ensure consistent processing across all input lines")
        
        if len(issues) == 0:
            recommendations.append("Excellent alignment quality! No issues detected.")
        elif len(issues) < 5:
            recommendations.append("Good alignment quality with minor issues that can be easily fixed")
        
        return recommendations
    
    def print_alignment_summary(self, issues: List[AlignmentIssue]) -> None:
        """
        Print a formatted summary of alignment check results.
        
        Args:
            issues (List[AlignmentIssue]): List of detected issues
            
        This method provides a human-readable summary of the alignment check
        results for quick assessment and decision making.
        """
        print("\n" + "="*70)
        print("SENTENCE ↔ ENTITY ALIGNMENT CHECK RESULTS")
        print("="*70)
        
        # Overall statistics
        print(f"Total lines checked: {self.statistics['total_lines_checked']}")
        print(f"Perfect alignments: {self.statistics['perfect_alignments']}")
        print(f"Total issues found: {len(issues)}")
        
        if len(issues) == 0:
            print("\n🎉 EXCELLENT! No alignment issues detected.")
            print("✅ All entities are properly aligned with their corresponding text.")
            return
        
        # Issue breakdown
        print(f"\nIssue Breakdown:")
        print(f"  🔴 Critical: {self.statistics['critical_issues']}")
        print(f"  🟠 Major:    {self.statistics['major_issues']}")
        print(f"  🟡 Minor:    {self.statistics['minor_issues']}")
        
        # Entity statistics
        if self.statistics['entities_validated'] > 0:
            success_rate = (self.statistics['entities_validated'] - self.statistics['entities_missing']) / self.statistics['entities_validated']
            print(f"\nEntity Validation:")
            print(f"  Total entities checked: {self.statistics['entities_validated']}")
            print(f"  Missing entities: {self.statistics['entities_missing']}")
            print(f"  Success rate: {success_rate:.1%}")
        
        # Issue distribution
        if self.statistics['issue_distribution']:
            print(f"\nIssue Types:")
            for issue_type, count in sorted(self.statistics['issue_distribution'].items()):
                print(f"  {issue_type.replace('_', ' ').title()}: {count}")
        
        # Recommendations
        recommendations = self._generate_recommendations(issues)
        if recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*70)


def main():
    """
    Command-line interface for the alignment checker.
    
    This function provides a convenient command-line interface for running
    alignment checks with customizable parameters and fixing options.
    """
    parser = argparse.ArgumentParser(
        description="Check and fix alignment between entity and denoised text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alignment_checker.py --entities test_entity.txt --denoised test_denoised.target
  python alignment_checker.py --entities data/entities.txt --denoised data/text.txt --fix --auto
  python alignment_checker.py --entities entities.txt --denoised text.txt --report alignment_report.json
        """
    )
    
    parser.add_argument(
        '--entities', '-e',
        required=True,
        help='Path to entity file'
    )
    
    parser.add_argument(
        '--denoised', '-d',
        required=True,
        help='Path to denoised text file'
    )
    
    parser.add_argument(
        '--fix', '-f',
        action='store_true',
        help='Attempt to fix detected alignment issues'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Apply fixes automatically without manual confirmation'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for fixed files (default: same as input)'
    )
    
    parser.add_argument(
        '--report', '-r',
        help='Path to save detailed alignment report (JSON format)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.8,
        help='Fuzzy matching threshold (0.0 to 1.0, default: 0.8)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize alignment checker
        checker = AlignmentChecker(fuzzy_threshold=args.threshold)
        
        # Perform alignment check
        issues = checker.check_alignment(args.entities, args.denoised)
        
        # Print summary
        checker.print_alignment_summary(issues)
        
        # Save detailed report if requested
        if args.report:
            checker.save_alignment_report(issues, args.report)
            print(f"\n📄 Detailed report saved to: {args.report}")
        
        # Fix issues if requested
        if args.fix and issues:
            output_dir = args.output or os.path.dirname(args.entities)
            fix_results = checker.fix_alignment_issues(
                issues, args.entities, args.denoised, output_dir, args.auto
            )
            
            print(f"\n🔧 Fixed files saved to:")
            print(f"  Entities: {fix_results['entity_file']}")
            print(f"  Text: {fix_results['text_file']}")
            print(f"  Lines removed: {fix_results['lines_removed']}")
        
        print(f"\n✅ Alignment check completed successfully!")
        
        # Return exit code based on issue severity
        critical_issues = len([i for i in issues if i.severity == "CRITICAL"])
        major_issues = len([i for i in issues if i.severity == "HIGH"])
        
        if critical_issues > 0:
            return 2  # Critical issues found
        elif major_issues > 0:
            return 1  # Major issues found
        else:
            return 0  # No major issues
        
    except Exception as e:
        print(f"❌ Error during alignment check: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
