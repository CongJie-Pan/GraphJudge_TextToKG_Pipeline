"""
Unit tests for the gold_label_bootstrapping module.

Tests the GoldLabelBootstrapper class functionality including
triple loading, source text processing, RapidFuzz matching,
LLM semantic evaluation, and result saving.
"""

import pytest
import os
import json
import csv
import tempfile
from unittest.mock import patch, mock_open, MagicMock, AsyncMock

from ..gold_label_bootstrapping import GoldLabelBootstrapper
from ..data_structures import TripleData, BootstrapResult, BootstrapStatistics
from .conftest import PerplexityTestBase, mock_async_perplexity_judge


class TestGoldLabelBootstrapperInitialization(PerplexityTestBase):
    """Test GoldLabelBootstrapper initialization."""
    
    def test_initialization(self, mock_async_perplexity_judge):
        """Test bootstrapper initialization."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        assert bootstrapper.graph_judge == mock_async_perplexity_judge
        assert isinstance(bootstrapper.rapidfuzz_available, bool)
    
    def test_check_rapidfuzz_availability_true(self):
        """Test RapidFuzz availability check when available."""
        from ..gold_label_bootstrapping import GoldLabelBootstrapper
        bootstrapper = GoldLabelBootstrapper(MagicMock())
        
        # Mock the import helper method to simulate successful import
        with patch.object(bootstrapper, '_import_rapidfuzz', return_value=True):
            assert bootstrapper._check_rapidfuzz_availability() == True
    
    def test_check_rapidfuzz_availability_false(self):
        """Test RapidFuzz availability check when not available."""
        from ..gold_label_bootstrapping import GoldLabelBootstrapper
        bootstrapper = GoldLabelBootstrapper(MagicMock())
        
        # Mock the import helper method to simulate ImportError
        with patch.object(bootstrapper, '_import_rapidfuzz', return_value=False):
            assert bootstrapper._check_rapidfuzz_availability() == False


class TestTripleLoading(PerplexityTestBase):
    """Test triple loading functionality."""
    
    def test_load_triples_from_file_json_format(self, mock_async_perplexity_judge):
        """Test loading triples from JSON-like format."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        json_content = """
according to the text content [["作者", "創作", "石頭記"], ["女媧", "補石", "補天"]]
some description [["賈寶玉", "喜歡", "林黛玉"]]
"""
        
        with patch('builtins.open', mock_open(read_data=json_content)):
            with patch('os.path.exists', return_value=True):
                triples = bootstrapper.load_triples_from_file("test_file.txt")
                
                assert len(triples) >= 2  # Should find at least 2 triples
                if len(triples) > 0:
                    assert isinstance(triples[0], TripleData)
                    assert triples[0].subject in ["作者", "賈寶玉", "女媧"]
                    assert triples[0].predicate in ["創作", "喜歡", "補石"]
    
    def test_load_triples_from_file_simple_format(self, mock_async_perplexity_judge):
        """Test loading triples from simple format."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        simple_content = """
作者 創作 石頭記
賈寶玉 喜歡 林黛玉
女媧 補石 補天
"""
        
        with patch('builtins.open', mock_open(read_data=simple_content)):
            with patch('os.path.exists', return_value=True):
                triples = bootstrapper.load_triples_from_file("test_file.txt")
                
                assert len(triples) == 3
                assert triples[0].subject == "作者"
                assert triples[0].predicate == "創作"
                assert triples[0].object == "石頭記"
                assert triples[1].subject == "賈寶玉"
                assert triples[2].object == "補天"
    
    def test_load_triples_from_file_missing_file(self, mock_async_perplexity_judge):
        """Test loading triples from missing file."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        with patch('os.path.exists', return_value=False):
            triples = bootstrapper.load_triples_from_file("missing_file.txt")
            assert len(triples) == 0
    
    def test_load_triples_from_file_malformed_json(self, mock_async_perplexity_judge):
        """Test loading triples with malformed JSON."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        malformed_content = """
according to text [["incomplete", "json"
another line [["valid", "triple", "here"]]
"""
        
        with patch('builtins.open', mock_open(read_data=malformed_content)):
            with patch('os.path.exists', return_value=True):
                triples = bootstrapper.load_triples_from_file("test_file.txt")
                
                # Should still load valid triples despite malformed ones
                assert len(triples) >= 1
                valid_triple = next((t for t in triples if t.subject == "valid"), None)
                assert valid_triple is not None


class TestSourceTextLoading(PerplexityTestBase):
    """Test source text loading functionality."""
    
    def test_load_source_text_success(self, mock_async_perplexity_judge):
        """Test successful source text loading."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        source_content = """
賈寶玉是一個重要人物，深深地喜歡林黛玉，這是大家都知道的事實。
曹雪芹花費多年創作紅樓夢這部大作。
這是一個沒有相關的句子。
"""
        
        with patch('builtins.open', mock_open(read_data=source_content)):
            with patch('os.path.exists', return_value=True):
                lines = bootstrapper.load_source_text("test_source.txt")
                
                assert len(lines) == 3
                assert "賈寶玉" in lines[0]
                assert "曹雪芹" in lines[1]
                assert "紅樓夢" in lines[1]
    
    def test_load_source_text_with_limit(self, mock_async_perplexity_judge):
        """Test source text loading with line limit."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create content that exceeds the limit
        large_content = "\n".join([f"Line {i}" for i in range(1500)])
        
        with patch('builtins.open', mock_open(read_data=large_content)):
            with patch('os.path.exists', return_value=True):
                lines = bootstrapper.load_source_text("test_source.txt")
                
                # Should be limited to max_source_lines (1000)
                assert len(lines) <= 1000
    
    def test_load_source_text_missing_file(self, mock_async_perplexity_judge):
        """Test loading source text from missing file."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        with patch('os.path.exists', return_value=False):
            lines = bootstrapper.load_source_text("missing_file.txt")
            assert len(lines) == 0


class TestStage1RapidFuzzMatching(PerplexityTestBase):
    """Test Stage 1 RapidFuzz matching functionality."""
    
    def test_stage1_rapidfuzz_matching_with_rapidfuzz(self, mock_async_perplexity_judge):
        """Test Stage 1 matching when RapidFuzz is available."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        bootstrapper.rapidfuzz_available = True
        
        triples = [
            TripleData("賈寶玉", "喜歡", "林黛玉"),
            TripleData("曹雪芹", "創作", "紅樓夢"),
            TripleData("錯誤", "陳述", "測試")
        ]
        
        source_lines = [
            "賈寶玉深深地喜歡林黛玉，這是大家都知道的事實。",
            "曹雪芹花費多年創作紅樓夢這部大作。",
            "這是一個沒有相關的句子。"
        ]
        
        with patch('rapidfuzz.fuzz.partial_ratio') as mock_fuzz:
            # Mock similarity scores for all comparisons (3 triples × 3 source_lines = 9 comparisons)
            # Each triple will be compared against all source lines, best score will be kept
            mock_fuzz.side_effect = [
                90, 10, 5,   # Triple 1: high match with line 1, low with others
                15, 85, 10,  # Triple 2: high match with line 2, low with others
                5, 10, 30    # Triple 3: low match with all lines
            ]
            
            results = bootstrapper.stage1_rapidfuzz_matching(triples, source_lines)
            
            assert len(results) == 3
            
            # Check high similarity cases (should be auto-confirmed)
            high_sim_results = [r for r in results if r.fuzzy_score >= 0.8]
            assert len(high_sim_results) >= 2
            
            # Check low similarity case (should be uncertain)
            low_sim_results = [r for r in results if r.fuzzy_score < 0.8]
            assert len(low_sim_results) >= 1
            assert low_sim_results[0].auto_expected is None
    
    def test_stage1_rapidfuzz_matching_without_rapidfuzz(self, mock_async_perplexity_judge):
        """Test Stage 1 matching when RapidFuzz is not available."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        bootstrapper.rapidfuzz_available = False
        
        triples = [TripleData("Test", "Subject", "Object")]
        source_lines = ["Test line"]
        
        results = bootstrapper.stage1_rapidfuzz_matching(triples, source_lines)
        
        assert len(results) == 1
        # Should use mock scoring
        assert 0.0 <= results[0].fuzzy_score <= 1.0
    
    def test_stage1_bootstrap_result_structure(self, mock_async_perplexity_judge):
        """Test that Stage 1 results have proper structure."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        triples = [TripleData("Subject", "Predicate", "Object")]
        source_lines = ["Source line"]
        
        results = bootstrapper.stage1_rapidfuzz_matching(triples, source_lines)
        
        assert len(results) == 1
        result = results[0]
        
        assert isinstance(result, BootstrapResult)
        assert isinstance(result.triple, TripleData)
        assert isinstance(result.source_idx, int)
        assert 0.0 <= result.fuzzy_score <= 1.0
        assert result.auto_expected in [None, True, False]
        assert result.llm_evaluation is None  # Should be None in Stage 1
        assert isinstance(result.note, str)


class TestStage2LLMEvaluation(PerplexityTestBase):
    """Test Stage 2 LLM semantic evaluation functionality."""
    
    @pytest.mark.asyncio
    async def test_stage2_llm_semantic_evaluation_success(self, mock_async_perplexity_judge):
        """Test successful Stage 2 LLM evaluation."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create uncertain results that need LLM evaluation
        triple = TripleData("賈寶玉", "喜歡", "林黛玉")
        uncertain_result = BootstrapResult(
            triple=triple,
            source_idx=0,
            fuzzy_score=0.6,  # Below threshold
            auto_expected=None,  # Uncertain
            llm_evaluation=None,
            expected=None,
            note="Low similarity, requires semantic evaluation"
        )
        
        source_lines = [
            "賈寶玉深深地喜歡林黛玉，這是大家都知道的事實。",
        ]
        
        # Ensure judge is in mock mode for testing
        mock_async_perplexity_judge.is_mock = True
        
        results = await bootstrapper.stage2_llm_semantic_evaluation([uncertain_result], source_lines)
        
        assert len(results) == 1
        result = results[0]
        
        # Check that LLM evaluation was performed
        assert result.llm_evaluation is not None
        assert result.auto_expected is not None
        assert result.llm_evaluation in ["Yes", "No"]
        assert result.expected == result.auto_expected
    
    @pytest.mark.asyncio
    async def test_stage2_llm_evaluation_empty_list(self, mock_async_perplexity_judge):
        """Test Stage 2 evaluation with empty uncertain results."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        results = await bootstrapper.stage2_llm_semantic_evaluation([], [])
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_stage2_llm_evaluation_with_context(self, mock_async_perplexity_judge):
        """Test Stage 2 evaluation uses source context properly."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        triple = TripleData("Test", "Subject", "Object")
        uncertain_result = BootstrapResult(
            triple=triple,
            source_idx=1,  # Middle of source lines
            fuzzy_score=0.5,
            auto_expected=None,
            llm_evaluation=None,
            expected=None,
            note="Test"
        )
        
        source_lines = [
            "Line 0",
            "Line 1 - target line",
            "Line 2",
            "Line 3"
        ]
        
        mock_async_perplexity_judge.is_mock = True
        
        results = await bootstrapper.stage2_llm_semantic_evaluation([uncertain_result], source_lines)
        
        assert len(results) == 1
        result = results[0]
        assert result.llm_evaluation is not None
    
    @pytest.mark.asyncio
    async def test_stage2_llm_evaluation_error_handling(self, mock_async_perplexity_judge):
        """Test Stage 2 evaluation error handling."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Mock judge to raise an error
        mock_async_perplexity_judge.judge_graph_triple = AsyncMock(side_effect=Exception("API Error"))
        mock_async_perplexity_judge.is_mock = False
        
        triple = TripleData("Test", "Subject", "Object")
        uncertain_result = BootstrapResult(
            triple=triple,
            source_idx=0,
            fuzzy_score=0.5,
            auto_expected=None,
            llm_evaluation=None,
            expected=None,
            note="Test"
        )
        
        results = await bootstrapper.stage2_llm_semantic_evaluation([uncertain_result], ["Test line"])
        
        assert len(results) == 1
        result = results[0]
        assert "error" in result.note.lower()


class TestSamplingAndSaving(PerplexityTestBase):
    """Test sampling and saving functionality."""
    
    def test_sample_uncertain_cases(self, mock_async_perplexity_judge):
        """Test sampling uncertain cases for manual review."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create test results with some Stage 2 cases
        results = []
        for i in range(10):
            triple = TripleData(f"主題{i}", f"關係{i}", f"客體{i}")
            result = BootstrapResult(
                triple=triple,
                source_idx=i,
                fuzzy_score=0.6,
                auto_expected=True,  # Simulated LLM evaluation result
                llm_evaluation="Yes",  # Mark as Stage 2 case
                expected=True,
                note="LLM semantic evaluation: Yes"
            )
            results.append(result)
        
        # Test sampling
        sampled_results = bootstrapper.sample_uncertain_cases(results)
        
        assert len(sampled_results) == len(results)
        
        # Check that some cases were marked for manual review
        manual_review_cases = [r for r in sampled_results if r.expected is None and "MANUAL REVIEW" in r.note]
        expected_sample_size = max(1, int(len(results) * 0.15))  # 15% sample rate
        
        # Allow some flexibility in sample size due to randomness
        assert 0 <= len(manual_review_cases) <= expected_sample_size + 1
    
    def test_sample_uncertain_cases_no_stage2(self, mock_async_perplexity_judge):
        """Test sampling when there are no Stage 2 cases."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create results without Stage 2 cases (no llm_evaluation)
        results = [
            BootstrapResult(
                triple=TripleData("Test", "Subject", "Object"),
                source_idx=0,
                fuzzy_score=0.9,
                auto_expected=True,
                llm_evaluation=None,  # No Stage 2 evaluation
                expected=True,
                note="High similarity"
            )
        ]
        
        sampled_results = bootstrapper.sample_uncertain_cases(results)
        assert len(sampled_results) == 1
        # Should not mark for manual review if no Stage 2 cases
        assert sampled_results[0].expected is not None
    
    def test_save_bootstrap_results(self, mock_async_perplexity_judge):
        """Test saving bootstrap results to CSV."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create test results
        triple1 = TripleData("賈寶玉", "喜歡", "林黛玉")
        triple2 = TripleData("曹雪芹", "創作", "紅樓夢")
        
        results = [
            BootstrapResult(
                triple=triple1,
                source_idx=0,
                fuzzy_score=0.95,
                auto_expected=True,
                llm_evaluation=None,
                expected=True,
                note="High similarity (≥0.8) with source"
            ),
            BootstrapResult(
                triple=triple2,
                source_idx=1,
                fuzzy_score=0.75,
                auto_expected=False,
                llm_evaluation="No",
                expected=None,
                note="LLM semantic evaluation: No | SAMPLED FOR MANUAL REVIEW"
            )
        ]
        
        # Use temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            success = bootstrapper.save_bootstrap_results(results, tmp_filename)
            assert success == True
            
            # Verify the CSV content
            with open(tmp_filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 2
                
                # Check first row
                assert rows[0]['subject'] == "賈寶玉"
                assert rows[0]['predicate'] == "喜歡"
                assert rows[0]['object'] == "林黛玉"
                assert rows[0]['auto_expected'] == "True"
                
                # Check second row
                assert rows[1]['subject'] == "曹雪芹"
                assert rows[1]['expected'] == ""  # None/empty for manual review
                assert "MANUAL REVIEW" in rows[1]['note']
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_calculate_statistics(self, mock_async_perplexity_judge):
        """Test bootstrap statistics calculation."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create test results with known distribution
        results = []
        
        # 60 auto-confirmed
        for i in range(60):
            results.append(BootstrapResult(
                triple=TripleData(f"S{i}", "P", "O"),
                source_idx=i,
                fuzzy_score=0.9,
                auto_expected=True,
                llm_evaluation=None,
                expected=True,
                note="High similarity"
            ))
        
        # 25 auto-rejected
        for i in range(25):
            results.append(BootstrapResult(
                triple=TripleData(f"S{i+60}", "P", "O"),
                source_idx=i,
                fuzzy_score=0.3,
                auto_expected=False,
                llm_evaluation="No",
                expected=False,
                note="LLM evaluation: No"
            ))
        
        # 15 manual review
        for i in range(15):
            results.append(BootstrapResult(
                triple=TripleData(f"S{i+85}", "P", "O"),
                source_idx=i,
                fuzzy_score=0.7,
                auto_expected=True,
                llm_evaluation="Yes",
                expected=None,  # Manual review
                note="Sampled for manual review"
            ))
        
        stats = bootstrapper.calculate_statistics(results)
        
        assert isinstance(stats, BootstrapStatistics)
        assert stats.total_triples == 100
        assert stats.auto_confirmed == 60
        assert stats.auto_rejected == 25
        assert stats.manual_review == 15
        assert stats.coverage_percentage == 85.0


class TestBootstrapIntegration(PerplexityTestBase):
    """Test complete bootstrap integration."""
    
    @pytest.mark.asyncio
    async def test_bootstrap_gold_labels_integration(self, mock_async_perplexity_judge):
        """Test complete gold label bootstrapping integration."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Ensure mock mode for testing
        mock_async_perplexity_judge.is_mock = True
        
        # Create mock input files
        triples_content = """
according to the text content [["賈寶玉", "喜歡", "林黛玉"], ["曹雪芹", "創作", "紅樓夢"]]
another line [["女媧", "補石", "補天"]]
"""
        
        source_content = """
賈寶玉深深地喜歡林黛玉，這是眾所皆知的事實。
曹雪芹用盡心血創作了這部不朽的紅樓夢。
女媧氏煉五色石以補蒼天的神話傳說廣為流傳。
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary input files
            triples_file = os.path.join(temp_dir, "test_triples.txt")
            source_file = os.path.join(temp_dir, "test_source.txt")
            output_file = os.path.join(temp_dir, "test_output.csv")
            
            with open(triples_file, 'w', encoding='utf-8') as f:
                f.write(triples_content)
            
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(source_content)
            
            # Run the bootstrap process
            success = await bootstrapper.bootstrap_gold_labels(triples_file, source_file, output_file)
            
            assert success == True
            assert os.path.exists(output_file)
            
            # Verify the output CSV
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) >= 2  # Should have processed multiple triples
                
                # Check that required columns are present
                expected_columns = ['subject', 'predicate', 'object', 'source_idx', 
                                  'fuzzy_score', 'auto_expected', 'expected', 'note']
                for col in expected_columns:
                    assert col in reader.fieldnames
    
    @pytest.mark.asyncio
    async def test_bootstrap_gold_labels_missing_files(self, mock_async_perplexity_judge):
        """Test bootstrap process with missing input files."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Test with missing triples file
        success = await bootstrapper.bootstrap_gold_labels(
            "missing_triples.txt", "missing_source.txt", "output.csv"
        )
        assert success == False
    
    @pytest.mark.asyncio
    async def test_bootstrap_gold_labels_empty_files(self, mock_async_perplexity_judge):
        """Test bootstrap process with empty input files."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty files
            triples_file = os.path.join(temp_dir, "empty_triples.txt")
            source_file = os.path.join(temp_dir, "empty_source.txt")
            output_file = os.path.join(temp_dir, "output.csv")
            
            with open(triples_file, 'w') as f:
                f.write("")
            
            with open(source_file, 'w') as f:
                f.write("")
            
            success = await bootstrapper.bootstrap_gold_labels(triples_file, source_file, output_file)
            assert success == False
