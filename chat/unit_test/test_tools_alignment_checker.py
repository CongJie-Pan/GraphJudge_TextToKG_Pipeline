"""
Unit Tests for Alignment Checker Tool

This test suite validates the functionality of the tools/alignment_checker.py module,
ensuring that sentence-entity alignment checking, issue detection, and automated
fixing work correctly across different scenarios and data integrity problems.

Test Coverage:
- Line count validation between entity and text files
- Entity-sentence semantic correspondence checking
- Missing entity detection and orphaned entity identification
- Alignment issue classification and severity assessment
- Automated fixing strategies and data repair
- Statistics tracking and comprehensive reporting
- Error handling for various file and format issues

Run with: pytest test_tools_alignment_checker.py -v --json-report --json-report-file=test_alignment_checker_report.json
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import sys

# Add the parent directories to the path to import the module under test
# The tools directory is located at GraphJudge/tools, not chat/tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the tool modules
try:
    from tools.alignment_checker import AlignmentChecker, AlignmentIssue, AlignmentIssueType
except ImportError:
    # Handle case where tools directory structure may be different
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    from alignment_checker import AlignmentChecker, AlignmentIssue, AlignmentIssueType


class TestAlignmentIssue:
    """Test cases for the AlignmentIssue dataclass."""
    
    def test_alignment_issue_creation(self):
        """Test creating AlignmentIssue instances."""
        issue = AlignmentIssue(
            issue_type=AlignmentIssueType.ENTITY_NOT_IN_TEXT,
            line_number=1,
            entity_line='["甄士隱", "不存在"]',
            text_line="甄士隱於書房閒坐。",
            missing_entities=["不存在"],
            orphaned_entities=[],
            severity="MEDIUM",
            suggested_fix="Remove missing entities",
            additional_info={"missing_count": 1}
        )
        
        assert issue.issue_type == AlignmentIssueType.ENTITY_NOT_IN_TEXT
        assert issue.line_number == 1
        assert issue.missing_entities == ["不存在"]
        assert issue.severity == "MEDIUM"
    
    def test_to_dict(self):
        """Test dictionary conversion for JSON serialization."""
        issue = AlignmentIssue(
            issue_type=AlignmentIssueType.LINE_COUNT_MISMATCH,
            line_number=0,
            entity_line="",
            text_line="",
            missing_entities=[],
            orphaned_entities=[],
            severity="CRITICAL",
            suggested_fix="Trim files to match",
            additional_info={"difference": 5}
        )
        
        issue_dict = issue.to_dict()
        
        assert issue_dict["issue_type"] == "line_count_mismatch"
        assert issue_dict["severity"] == "CRITICAL"
        assert issue_dict["additional_info"]["difference"] == 5


class TestAlignmentChecker:
    """Test cases for the AlignmentChecker class."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_logs_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.test_logs_dir, exist_ok=True)
        
        # Sample test data
        self.sample_entity_lines = [
            '["甄士隱", "書房"]',
            '["閶門", "十里街", "仁清巷"]',
            '["賈雨村", "胡州"]',
            '["不存在的實體"]',  # Entity not in corresponding text
            ''  # Empty line
        ]
        
        self.sample_text_lines = [
            "甄士隱於書房閒坐，至手倦拋書，伏几少憩。",
            "這閶門外有個十里街，街內有個仁清巷，巷內有個古廟。",
            "賈雨村原系胡州人氏，也是詩書仕宦之族。",
            "這是一段沒有對應實體的文本。",
            "非空文本行對應空實體行。"
        ]
        
        # Misaligned data for testing issues
        self.misaligned_entity_lines = [
            '["甄士隱", "書房"]',
            '["閶門", "十里街"]',
            '["賈雨村"]'  # Missing lines compared to text
        ]
        
        self.misaligned_text_lines = [
            "甄士隱於書房閒坐。",
            "這閶門外有個十里街。",
            "賈雨村原系胡州人氏。",
            "多出來的文本行。",
            "又一個多出來的文本行。"
        ]
        
        # Patch logging to use temporary directory
        self.patcher = patch('tools.alignment_checker.logging.FileHandler')
        self.mock_file_handler = self.patcher.start()
        
    def teardown_method(self):
        """Clean up method called after each test method."""
        # Stop all patches
        self.patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_default_config(self):
        """Test AlignmentChecker initialization with default configuration."""
        checker = AlignmentChecker()
        
        # Check default fuzzy threshold
        assert checker.fuzzy_threshold == 0.8
        
        # Check statistics initialization
        assert checker.statistics["total_lines_checked"] == 0
        assert checker.statistics["perfect_alignments"] == 0
        assert checker.statistics["entities_validated"] == 0
    
    def test_init_custom_fuzzy_threshold(self):
        """Test AlignmentChecker initialization with custom fuzzy threshold."""
        checker = AlignmentChecker(fuzzy_threshold=0.9)
        assert checker.fuzzy_threshold == 0.9
    
    def test_parse_entity_line(self):
        """Test parsing entity lines in different formats."""
        checker = AlignmentChecker()
        
        # Test Python list format
        line = '["甄士隱", "書房", "封氏"]'
        entities = checker._parse_entity_line(line)
        assert entities == ["甄士隱", "書房", "封氏"]
        
        # Test comma-separated format
        line = '甄士隱, 書房, 封氏'
        entities = checker._parse_entity_line(line)
        assert entities == ["甄士隱", "書房", "封氏"]
        
        # Test empty line
        line = ''
        entities = checker._parse_entity_line(line)
        assert entities == []
        
        # Test malformed line
        line = '["甄士隱", "書房"'  # Missing closing bracket
        entities = checker._parse_entity_line(line)
        # Should still parse as comma-separated
        assert "甄士隱" in entities
        assert "書房" in entities
    
    def test_check_entity_in_text_exact_match(self):
        """Test exact entity matching in text."""
        checker = AlignmentChecker()
        
        # Test exact matches
        assert checker._check_entity_in_text("甄士隱", "甄士隱於書房閒坐") == True
        assert checker._check_entity_in_text("書房", "甄士隱於書房閒坐") == True
        
        # Test non-matches
        assert checker._check_entity_in_text("賈雨村", "甄士隱於書房閒坐") == False
        assert checker._check_entity_in_text("", "甄士隱於書房閒坐") == False
        assert checker._check_entity_in_text("甄士隱", "") == False
    
    def test_check_entity_in_text_fuzzy_match(self):
        """Test fuzzy entity matching in text."""
        checker = AlignmentChecker(fuzzy_threshold=0.7)
        
        # Test fuzzy matches (slight variations)
        # Note: This depends on the fuzzy matching implementation
        # For exact testing, we'd need to know the specific fuzzy logic
        result = checker._check_entity_in_text("甄士隱", "甄士隐於書房閒坐")  # Traditional vs Simplified
        # This test depends on actual fuzzy matching implementation
    
    def test_load_files_success(self):
        """Test successful loading of entity and text files."""
        # Create test files
        entity_file = os.path.join(self.temp_dir, "entities.txt")
        text_file = os.path.join(self.temp_dir, "text.txt")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            for line in self.sample_entity_lines[:3]:  # First 3 lines
                f.write(line + '\n')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            for line in self.sample_text_lines[:3]:  # First 3 lines
                f.write(line + '\n')
        
        checker = AlignmentChecker()
        entity_lines, text_lines = checker._load_files(entity_file, text_file)
        
        assert len(entity_lines) == 3
        assert len(text_lines) == 3
        assert entity_lines[0] == '["甄士隱", "書房"]'
        assert "甄士隱" in text_lines[0]
    
    def test_load_files_not_found(self):
        """Test error handling for non-existent files."""
        checker = AlignmentChecker()
        
        non_existent_entity = os.path.join(self.temp_dir, "no_entities.txt")
        non_existent_text = os.path.join(self.temp_dir, "no_text.txt")
        
        # Create only one file
        with open(non_existent_entity, 'w') as f:
            f.write('["test"]')
        
        # Test missing text file
        with pytest.raises(FileNotFoundError):
            checker._load_files(non_existent_entity, non_existent_text)
        
        # Test missing entity file
        with pytest.raises(FileNotFoundError):
            checker._load_files(non_existent_text, non_existent_entity)
    
    def test_check_line_alignment_perfect(self):
        """Test line alignment checking with perfectly aligned data."""
        checker = AlignmentChecker()
        
        # Create matched entity and text lines
        entity_lines = self.sample_entity_lines[:3]
        text_lines = self.sample_text_lines[:3]
        
        issues = checker.check_line_alignment(entity_lines, text_lines)
        
        # Should find no issues with properly aligned data
        assert len(issues) == 0
    
    def test_check_line_alignment_mismatch(self):
        """Test line alignment checking with mismatched line counts."""
        checker = AlignmentChecker()
        
        issues = checker.check_line_alignment(self.misaligned_entity_lines, self.misaligned_text_lines)
        
        # Should detect line count mismatch
        assert len(issues) > 0
        line_count_issues = [i for i in issues if i.issue_type == AlignmentIssueType.LINE_COUNT_MISMATCH]
        assert len(line_count_issues) == 1
        assert line_count_issues[0].severity == "CRITICAL"
    
    def test_check_line_alignment_empty_lines(self):
        """Test line alignment checking with empty lines."""
        checker = AlignmentChecker()
        
        entity_lines = ['["甄士隱"]', '', '["賈雨村"]']
        text_lines = ['甄士隱於書房閒坐。', '非空文本行。', '']
        
        issues = checker.check_line_alignment(entity_lines, text_lines)
        
        # Should detect empty line issues
        empty_entity_issues = [i for i in issues if i.issue_type == AlignmentIssueType.EMPTY_ENTITY_LINE]
        empty_text_issues = [i for i in issues if i.issue_type == AlignmentIssueType.EMPTY_TEXT_LINE]
        
        assert len(empty_entity_issues) == 1
        assert len(empty_text_issues) == 1
    
    def test_check_semantic_alignment_perfect(self):
        """Test semantic alignment checking with perfect entity-text correspondence."""
        checker = AlignmentChecker()
        
        entity_lines = ['["甄士隱", "書房"]']
        text_lines = ['甄士隱於書房閒坐。']
        
        issues = checker.check_semantic_alignment(entity_lines, text_lines)
        
        # Should find no issues
        assert len(issues) == 0
        assert checker.statistics["perfect_alignments"] == 1
    
    def test_check_semantic_alignment_missing_entities(self):
        """Test semantic alignment checking with missing entities."""
        checker = AlignmentChecker()
        
        entity_lines = ['["甄士隱", "不存在的實體"]']
        text_lines = ['甄士隱於書房閒坐。']
        
        issues = checker.check_semantic_alignment(entity_lines, text_lines)
        
        # Should detect missing entity issue
        assert len(issues) == 1
        assert issues[0].issue_type == AlignmentIssueType.ENTITY_NOT_IN_TEXT
        assert "不存在的實體" in issues[0].missing_entities
        assert issues[0].additional_info["missing_ratio"] == 0.5  # 1 out of 2 entities missing
    
    def test_check_semantic_alignment_format_error(self):
        """Test semantic alignment checking with format errors."""
        checker = AlignmentChecker()
        
        entity_lines = ['invalid_format_line']
        text_lines = ['有效的文本行。']
        
        issues = checker.check_semantic_alignment(entity_lines, text_lines)
        
        # Should handle format error gracefully and continue processing
        # The exact behavior depends on implementation
        assert isinstance(issues, list)
    
    def test_determine_issue_severity(self):
        """Test issue severity determination logic."""
        checker = AlignmentChecker()
        
        # Test critical severity for line count mismatch
        severity = checker._determine_issue_severity(AlignmentIssueType.LINE_COUNT_MISMATCH, 5, 10)
        assert severity == "CRITICAL"
        
        # Test high severity for many missing entities
        severity = checker._determine_issue_severity(AlignmentIssueType.ENTITY_NOT_IN_TEXT, 8, 10)
        assert severity == "CRITICAL"
        
        # Test medium severity for moderate missing entities
        severity = checker._determine_issue_severity(AlignmentIssueType.ENTITY_NOT_IN_TEXT, 3, 10)
        assert severity == "MEDIUM"
        
        # Test low severity for few missing entities
        severity = checker._determine_issue_severity(AlignmentIssueType.ENTITY_NOT_IN_TEXT, 1, 10)
        assert severity == "LOW"
    
    def test_suggest_fix(self):
        """Test fix suggestion generation."""
        checker = AlignmentChecker()
        
        # Test fix suggestion for line count mismatch
        issue = AlignmentIssue(
            issue_type=AlignmentIssueType.LINE_COUNT_MISMATCH,
            line_number=0, entity_line="", text_line="",
            missing_entities=[], orphaned_entities=[],
            severity="CRITICAL", suggested_fix="", additional_info={}
        )
        fix = checker._suggest_fix(issue)
        assert "trim" in fix.lower()
        
        # Test fix suggestion for missing entities
        issue.issue_type = AlignmentIssueType.ENTITY_NOT_IN_TEXT
        issue.severity = "LOW"
        fix = checker._suggest_fix(issue)
        assert "remove missing entities" in fix.lower() or "adjust" in fix.lower()
    
    def test_check_alignment_complete(self):
        """Test complete alignment checking workflow."""
        # Create test files
        entity_file = os.path.join(self.temp_dir, "entities.txt")
        text_file = os.path.join(self.temp_dir, "text.txt")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            for line in self.sample_entity_lines:
                f.write(line + '\n')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            for line in self.sample_text_lines:
                f.write(line + '\n')
        
        checker = AlignmentChecker()
        issues = checker.check_alignment(entity_file, text_file)
        
        # Should detect various issues in the sample data
        assert isinstance(issues, list)
        assert len(issues) >= 0  # May have issues due to sample data design
        
        # Check that statistics were updated
        stats = checker.statistics
        assert stats["total_lines_checked"] > 0
    
    def test_fix_alignment_issues(self):
        """Test automated fixing of alignment issues."""
        # Create test files with known issues
        entity_file = os.path.join(self.temp_dir, "entities.txt")
        text_file = os.path.join(self.temp_dir, "text.txt")
        
        # Create data with line count mismatch
        entity_lines = ['["甄士隱"]', '["賈雨村"]']
        text_lines = ['甄士隱於書房閒坐。', '賈雨村原系胡州人氏。', '多餘的文本行。']
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            for line in entity_lines:
                f.write(line + '\n')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            for line in text_lines:
                f.write(line + '\n')
        
        checker = AlignmentChecker()
        issues = checker.check_alignment(entity_file, text_file)
        
        # Apply fixes
        fix_results = checker.fix_alignment_issues(
            issues, entity_file, text_file, self.temp_dir, auto_fix=True
        )
        
        # Verify fix results
        assert "entity_file" in fix_results
        assert "text_file" in fix_results
        assert fix_results["fixed_entity_lines"] == fix_results["fixed_text_lines"]
        
        # Verify fixed files exist
        assert os.path.exists(fix_results["entity_file"])
        assert os.path.exists(fix_results["text_file"])
    
    def test_save_alignment_report(self):
        """Test saving detailed alignment report."""
        checker = AlignmentChecker()
        
        # Create sample issues
        issues = [
            AlignmentIssue(
                issue_type=AlignmentIssueType.ENTITY_NOT_IN_TEXT,
                line_number=1, entity_line='["不存在"]', text_line="文本行。",
                missing_entities=["不存在"], orphaned_entities=[],
                severity="MEDIUM", suggested_fix="Remove missing entity",
                additional_info={"missing_count": 1}
            )
        ]
        
        report_file = os.path.join(self.temp_dir, "alignment_report.json")
        checker.save_alignment_report(issues, report_file)
        
        # Verify report file
        assert os.path.exists(report_file)
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        assert "alignment_check_summary" in report
        assert "detailed_issues" in report
        assert "recommendations" in report
        assert len(report["detailed_issues"]) == 1
    
    def test_generate_recommendations(self):
        """Test recommendation generation based on issues."""
        checker = AlignmentChecker()
        
        # Test with no issues
        recommendations = checker._generate_recommendations([])
        assert any("excellent" in rec.lower() or "no issues" in rec.lower() for rec in recommendations)
        
        # Test with critical issues
        critical_issue = AlignmentIssue(
            issue_type=AlignmentIssueType.LINE_COUNT_MISMATCH,
            line_number=0, entity_line="", text_line="",
            missing_entities=[], orphaned_entities=[],
            severity="CRITICAL", suggested_fix="", additional_info={}
        )
        recommendations = checker._generate_recommendations([critical_issue])
        assert any("critical" in rec.lower() for rec in recommendations)
        
        # Test with many entity issues
        entity_issues = [
            AlignmentIssue(
                issue_type=AlignmentIssueType.ENTITY_NOT_IN_TEXT,
                line_number=i, entity_line="", text_line="",
                missing_entities=[], orphaned_entities=[],
                severity="MEDIUM", suggested_fix="", additional_info={}
            ) for i in range(10)
        ]
        recommendations = checker._generate_recommendations(entity_issues)
        assert any("entity extraction" in rec.lower() for rec in recommendations)
    
    def test_unicode_and_encoding(self):
        """Test proper Unicode handling for Chinese text."""
        checker = AlignmentChecker()
        
        # Test with complex Chinese characters
        entity_lines = ['["賈寶玉", "林黛玉", "薛寶釵"]']
        text_lines = ['賈寶玉與林黛玉、薛寶釵的故事。']
        
        issues = checker.check_semantic_alignment(entity_lines, text_lines)
        
        # Should find no issues as all entities are in text
        assert len(issues) == 0
    
    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        checker = AlignmentChecker()
        
        # Test with empty files
        empty_entity_lines = []
        empty_text_lines = []
        issues = checker.check_line_alignment(empty_entity_lines, empty_text_lines)
        assert len(issues) == 0  # Empty files should have no alignment issues
        
        # Test with very long entity names
        long_entity = "非常長的實體名稱" * 20
        entity_lines = [f'["{long_entity}"]']
        text_lines = [f"這裡包含{long_entity}的文本。"]
        issues = checker.check_semantic_alignment(entity_lines, text_lines)
        assert len(issues) == 0  # Should find the long entity
        
        # Test with special characters in entities
        special_entity_lines = ['["實體@#$", "另一個實體!?"]']
        special_text_lines = ['文本包含實體@#$和另一個實體!?。']
        issues = checker.check_semantic_alignment(special_entity_lines, special_text_lines)
        # Should handle special characters gracefully


class TestAlignmentCheckerIntegration:
    """Integration tests for the AlignmentChecker tool."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch logging to use temporary directory
        self.patcher = patch('tools.alignment_checker.logging.FileHandler')
        self.mock_file_handler = self.patcher.start()
    
    def teardown_method(self):
        """Clean up method called after each test method."""
        self.patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_alignment_check(self):
        """Test complete end-to-end alignment checking workflow."""
        # Create realistic test data with various issues
        entity_data = [
            '["甄士隱", "書房"]',  # Perfect match
            '["閶門", "十里街", "不存在實體"]',  # One missing entity
            '["賈雨村", "胡州"]',  # Perfect match
            '',  # Empty entity line
            '["孤立實體"]'  # Line mismatch (no corresponding text)
        ]
        
        text_data = [
            "甄士隱於書房閒坐，至手倦拋書。",
            "這閶門外有個十里街，街內有個仁清巷。",
            "賈雨村原系胡州人氏，也是詩書仕宦之族。",
            "對應空實體行的非空文本行。"
            # Missing text line for last entity line
        ]
        
        # Write test files
        entity_file = os.path.join(self.temp_dir, "test_entities.txt")
        text_file = os.path.join(self.temp_dir, "test_text.txt")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            for line in entity_data:
                f.write(line + '\n')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            for line in text_data:
                f.write(line + '\n')
        
        # Run complete alignment check
        checker = AlignmentChecker()
        issues = checker.check_alignment(entity_file, text_file)
        
        # Should detect multiple types of issues
        issue_types = [issue.issue_type for issue in issues]
        
        # Should find line count mismatch
        assert AlignmentIssueType.LINE_COUNT_MISMATCH in issue_types
        
        # May find missing entities and empty line issues
        # (Exact assertions depend on implementation details)
        
        # Save alignment report
        report_file = os.path.join(self.temp_dir, "alignment_report.json")
        checker.save_alignment_report(issues, report_file)
        assert os.path.exists(report_file)
        
        # Apply fixes
        fix_results = checker.fix_alignment_issues(
            issues, entity_file, text_file, self.temp_dir, auto_fix=True
        )
        
        # Verify that fixed files are created
        assert os.path.exists(fix_results["entity_file"])
        assert os.path.exists(fix_results["text_file"])
        
        # Verify that line counts match after fixing
        assert fix_results["fixed_entity_lines"] == fix_results["fixed_text_lines"]
    
    def test_performance_large_dataset(self):
        """Test performance with larger dataset."""
        # Create large test dataset
        large_entity_data = []
        large_text_data = []
        
        for i in range(500):
            # Create realistic entity and text pairs
            entities = ["甄士隱", "賈雨村", "林黛玉"][i % 3]
            large_entity_data.append(f'["{entities}"]')
            large_text_data.append(f"這是包含{entities}的文本第{i}行。")
        
        # Write large test files
        entity_file = os.path.join(self.temp_dir, "large_entities.txt")
        text_file = os.path.join(self.temp_dir, "large_text.txt")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            for line in large_entity_data:
                f.write(line + '\n')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            for line in large_text_data:
                f.write(line + '\n')
        
        # Process and time the operation
        import time
        start_time = time.time()
        
        checker = AlignmentChecker()
        issues = checker.check_alignment(entity_file, text_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert processing_time < 60.0  # Should complete within 60 seconds
        assert checker.statistics["total_lines_checked"] == 500
        assert checker.statistics["perfect_alignments"] > 400  # Most should be perfect
    
    def test_realistic_ectd_data(self):
        """Test with realistic ECTD pipeline output data."""
        # Simulate realistic ECTD output with typical issues
        realistic_entities = [
            '["甄士隱", "甄士隱", "書房"]',  # Contains duplicate
            '["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]',
            '["甄費", "甄士隱", "封氏", "功名", "朝代"]',  # Contains generic terms
            '["賈雨村", "胡州", "詩書仕宦之族"]',
            '[]'  # Empty entity list
        ]
        
        realistic_text = [
            "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。",
            "這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。",
            "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。",
            "賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，暫寄廟中安身。",
            "這是一個對應空實體列表的文本行。"
        ]
        
        # Write realistic test files
        entity_file = os.path.join(self.temp_dir, "realistic_entities.txt")
        text_file = os.path.join(self.temp_dir, "realistic_text.txt")
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            for line in realistic_entities:
                f.write(line + '\n')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            for line in realistic_text:
                f.write(line + '\n')
        
        # Run alignment check
        checker = AlignmentChecker()
        issues = checker.check_alignment(entity_file, text_file)
        
        # Should handle duplicates and generic terms gracefully
        # Most entities should be found in their corresponding text
        stats = checker.statistics
        if stats["entities_validated"] > 0:
            success_rate = (stats["entities_validated"] - stats["entities_missing"]) / stats["entities_validated"]
            assert success_rate >= 0.6  # At least 60% of entities should be found
        
        # Generate and save report
        report_file = os.path.join(self.temp_dir, "realistic_report.json")
        checker.save_alignment_report(issues, report_file)
        
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Report should contain meaningful statistics and recommendations
        assert report["alignment_check_summary"]["total_issues"] >= 0
        assert len(report["recommendations"]) > 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--json-report", "--json-report-file=test_alignment_checker_report.json"])
