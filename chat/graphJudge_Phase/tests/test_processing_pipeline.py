"""
Unit tests for the processing_pipeline module.

Tests the ProcessingPipeline class functionality including
batch processing, result collection, output generation,
and statistics calculation.
"""

import pytest
import asyncio
import os
import csv
import json
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

from ..processing_pipeline import ProcessingPipeline
from ..data_structures import ProcessingStatistics, ExplainableJudgment
from .conftest import PerplexityTestBase, create_mock_dataset, create_test_files


class TestProcessingPipelineInitialization(PerplexityTestBase):
    """Test ProcessingPipeline initialization."""
    
    def test_initialization(self, mock_async_perplexity_judge):
        """Test pipeline initialization."""
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        assert pipeline.graph_judge == mock_async_perplexity_judge
    
    def test_generate_reasoning_file_path_auto(self):
        """Test automatic reasoning file path generation."""
        mock_judge = MagicMock()
        pipeline = ProcessingPipeline(mock_judge)
        
        csv_path = "/path/to/pred_instructions_context_perplexity_itr2.csv"
        expected_reasoning_path = "/path/to/pred_instructions_context_perplexity_itr2_reasoning.json"
        
        result = pipeline.generate_reasoning_file_path(csv_path)
        
        # Normalize paths for cross-platform comparison
        result_normalized = result.replace('\\', '/')
        expected_normalized = expected_reasoning_path.replace('\\', '/')
        
        assert result_normalized == expected_normalized
    
    def test_generate_reasoning_file_path_custom(self):
        """Test custom reasoning file path."""
        mock_judge = MagicMock()
        pipeline = ProcessingPipeline(mock_judge)
        
        csv_path = "/path/to/output.csv"
        custom_path = "/custom/path/reasoning.json"
        
        result = pipeline.generate_reasoning_file_path(csv_path, custom_path)
        assert result == custom_path


class TestStandardProcessing(PerplexityTestBase):
    """Test standard processing mode."""
    
    @pytest.mark.asyncio
    async def test_process_instructions_standard_mode(self, mock_async_perplexity_judge):
        """Test standard instruction processing."""
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        # Create test dataset
        test_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Apple Founded by Steve Jobs ?", "input": "", "output": ""},
            {"instruction": "Is this true: Microsoft Founded by Mark Zuckerberg ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=False
                )
                
                # Verify statistics
                assert isinstance(stats, ProcessingStatistics)
                assert stats.total_instructions == 3
                assert stats.successful_responses >= 0
                assert stats.error_responses >= 0
                assert stats.success_rate >= 0.0
                
                # Verify output file
                assert os.path.exists(output_file)
                
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # Check header
                    assert rows[0] == ["prompt", "generated"]
                    
                    # Check data rows
                    assert len(rows) == 4  # Header + 3 data rows
    
    @pytest.mark.asyncio
    async def test_process_instructions_with_errors(self, mock_async_perplexity_judge):
        """Test instruction processing with some errors."""
        # Configure mock to raise errors for some calls
        mock_async_perplexity_judge.judge_graph_triple = AsyncMock(side_effect=[
            "Yes",  # First call succeeds
            Exception("API Error"),  # Second call fails
            "No"   # Third call succeeds
        ])
        
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: Statement 1 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Statement 2 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Statement 3 ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=False
                )
                
                # Should have processed all instructions despite errors
                assert stats.total_instructions == 3
                assert stats.error_responses > 0
                assert stats.successful_responses > 0


class TestExplainableProcessing(PerplexityTestBase):
    """Test explainable processing mode."""
    
    @pytest.mark.asyncio
    async def test_process_instructions_explainable_mode(self, mock_async_perplexity_judge):
        """Test explainable instruction processing."""
        # Configure mock for explainable judgment
        mock_explainable_judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.9,
            reasoning="這是一個測試推理",
            evidence_sources=["test_source"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=1.0
        )
        
        mock_async_perplexity_judge.judge_graph_triple_with_explanation = AsyncMock(
            return_value=mock_explainable_judgment
        )
        
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            reasoning_file = output_file.replace('.csv', '_reasoning.json')
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=True,
                    reasoning_file_path=reasoning_file
                )
                
                # Verify statistics
                assert isinstance(stats, ProcessingStatistics)
                assert stats.total_instructions == 1
                assert stats.avg_confidence > 0.0
                
                # Verify main CSV file
                assert os.path.exists(output_file)
                
                # Verify reasoning file
                assert os.path.exists(reasoning_file)
                
                with open(reasoning_file, 'r', encoding='utf-8') as f:
                    reasoning_data = json.load(f)
                    assert len(reasoning_data) == 1
                    assert reasoning_data[0]["judgment"] == "Yes"
                    assert reasoning_data[0]["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_process_instructions_explainable_with_errors(self, mock_async_perplexity_judge):
        """Test explainable processing with some errors."""
        # Configure mock to return error judgment
        error_judgment = ExplainableJudgment(
            judgment="No",
            confidence=0.0,
            reasoning="Error during processing",
            evidence_sources=[],
            alternative_suggestions=[],
            error_type="processing_error",
            processing_time=0.0
        )
        
        mock_async_perplexity_judge.judge_graph_triple_with_explanation = AsyncMock(
            return_value=error_judgment
        )
        
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: Error Statement ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=True
                )
                
                # Should handle error gracefully
                assert stats.total_instructions == 1
                assert stats.unique_error_types > 0


class TestConcurrencyControl(PerplexityTestBase):
    """Test concurrency control and rate limiting."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_limit(self, mock_async_perplexity_judge):
        """Test that concurrent processing respects limits."""
        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0
        
        async def mock_judge_with_tracking(instruction, input_text=None):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            concurrent_calls -= 1
            return "Yes"
        
        mock_async_perplexity_judge.judge_graph_triple = mock_judge_with_tracking
        
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        # Create multiple test items
        test_data = [
            {"instruction": f"Is this true: Statement {i} ?", "input": "", "output": ""}
            for i in range(10)
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=False
                )
                
                # Should have processed all items
                assert stats.total_instructions == 10
                
                # Should respect concurrency limit (default is 3)
                assert max_concurrent <= 3
    
    @pytest.mark.asyncio
    async def test_rate_limiting_delays(self, mock_async_perplexity_judge):
        """Test that rate limiting introduces appropriate delays."""
        call_times = []
        
        async def mock_judge_with_timing(instruction, input_text=None):
            import time
            call_times.append(time.time())
            return "Yes"
        
        mock_async_perplexity_judge.judge_graph_triple = mock_judge_with_timing
        
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: Statement 1 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Statement 2 ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=False
                )
                
                # Should have made calls (timing verification is approximate)
                assert len(call_times) == 2


class TestStatisticsCalculation(PerplexityTestBase):
    """Test statistics calculation."""
    
    def test_calculate_statistics_all_success(self):
        """Test statistics calculation with all successful responses."""
        mock_judge = MagicMock()
        pipeline = ProcessingPipeline(mock_judge)
        
        responses = ["Yes", "No", "Yes", "Yes", "No"]
        reasoning_results = [
            {"confidence": 0.9, "error_type": None},
            {"confidence": 0.8, "error_type": None},
            {"confidence": 0.95, "error_type": None},
            {"confidence": 0.85, "error_type": None},
            {"confidence": 0.75, "error_type": None}
        ]
        
        stats = pipeline._calculate_statistics(responses, reasoning_results)
        
        assert stats.total_instructions == 5
        assert stats.successful_responses == 5
        assert stats.error_responses == 0
        assert stats.yes_judgments == 3
        assert stats.no_judgments == 2
        assert stats.success_rate == 100.0
        assert stats.positive_rate == 60.0
        assert stats.avg_confidence == 0.85
        assert stats.unique_error_types == 0
    
    def test_calculate_statistics_with_errors(self):
        """Test statistics calculation with some errors."""
        mock_judge = MagicMock()
        pipeline = ProcessingPipeline(mock_judge)
        
        responses = ["Yes", "Error: Failed", "No", "Error: Timeout", "Yes"]
        reasoning_results = [
            {"confidence": 0.9, "error_type": None},
            {"confidence": 0.0, "error_type": "api_error"},
            {"confidence": 0.8, "error_type": None},
            {"confidence": 0.0, "error_type": "timeout_error"},
            {"confidence": 0.85, "error_type": None}
        ]
        
        stats = pipeline._calculate_statistics(responses, reasoning_results)
        
        assert stats.total_instructions == 5
        assert stats.successful_responses == 3
        assert stats.error_responses == 2
        assert stats.yes_judgments == 2
        assert stats.no_judgments == 1
        assert stats.success_rate == 60.0
        assert stats.positive_rate == pytest.approx(66.67, rel=1e-2)
        assert stats.unique_error_types == 2
    
    def test_calculate_statistics_standard_mode(self):
        """Test statistics calculation for standard mode (no reasoning data)."""
        mock_judge = MagicMock()
        pipeline = ProcessingPipeline(mock_judge)
        
        responses = ["Yes", "No", "Yes"]
        reasoning_results = []  # Empty for standard mode
        
        stats = pipeline._calculate_statistics(responses, reasoning_results)
        
        assert stats.total_instructions == 3
        assert stats.successful_responses == 3
        assert stats.yes_judgments == 2
        assert stats.no_judgments == 1
        assert stats.avg_confidence == 0.0  # No confidence data in standard mode
        assert stats.unique_error_types == 0


class TestFileOperations(PerplexityTestBase):
    """Test file operations in processing pipeline."""
    
    def test_save_csv_results(self, mock_async_perplexity_judge):
        """Test saving CSV results."""
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: Statement 1 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Statement 2 ?", "input": "", "output": ""}
        ]
        responses = ["Yes", "No"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.csv")
            
            pipeline._save_csv_results(test_data, responses, output_file)
            
            # Verify file exists and has correct content
            assert os.path.exists(output_file)
            
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                assert rows[0] == ["prompt", "generated"]
                assert len(rows) == 3  # Header + 2 data rows
                assert "Statement 1" in rows[1][0]
                assert rows[1][1] == "Yes"
                assert "Statement 2" in rows[2][0]
                assert rows[2][1] == "No"
    
    def test_save_reasoning_results(self, mock_async_perplexity_judge):
        """Test saving reasoning results."""
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        reasoning_results = [
            {
                "index": 0,
                "prompt": "Is this true: Test Statement ?",
                "judgment": "Yes",
                "confidence": 0.9,
                "reasoning": "Test reasoning",
                "evidence_sources": ["test"],
                "alternative_suggestions": [],
                "error_type": None,
                "processing_time": 1.0
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            reasoning_file = os.path.join(temp_dir, "reasoning.json")
            
            result = pipeline._save_reasoning_results(reasoning_results, reasoning_file)
            
            assert result is True
            assert os.path.exists(reasoning_file)
            
            with open(reasoning_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                assert len(loaded_data) == 1
                assert loaded_data[0]["judgment"] == "Yes"
                assert loaded_data[0]["confidence"] == 0.9


class TestProcessingPipelineIntegration(PerplexityTestBase):
    """Test integration scenarios for processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_standard_processing(self, mock_async_perplexity_judge):
        """Test complete end-to-end standard processing."""
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Apple Founded by Steve Jobs ?", "input": "", "output": ""},
            {"instruction": "Is this true: Microsoft Founded by Mark Zuckerberg ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=False
                )
                
                # Verify complete processing
                assert stats.total_instructions == 3
                assert os.path.exists(output_file)
                
                # Verify CSV structure
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    assert len(rows) == 3
                    
                    for row in rows:
                        assert "prompt" in row
                        assert "generated" in row
                        assert row["generated"] in ["Yes", "No"] or "Error" in row["generated"]
    
    @pytest.mark.asyncio
    async def test_end_to_end_explainable_processing(self, mock_async_perplexity_judge):
        """Test complete end-to-end explainable processing."""
        # Configure comprehensive mock for explainable mode
        mock_explainable_judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.9,
            reasoning="詳細的推理解釋",
            evidence_sources=["domain_knowledge", "historical_records"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=1.2
        )
        
        mock_async_perplexity_judge.judge_graph_triple_with_explanation = AsyncMock(
            return_value=mock_explainable_judgment
        )
        
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""}
        ]
        mock_dataset = create_mock_dataset(test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=mock_dataset,
                    explainable_mode=True
                )
                
                # Verify dual output
                assert os.path.exists(output_file)
                
                reasoning_file = pipeline.generate_reasoning_file_path(output_file)
                assert os.path.exists(reasoning_file)
                
                # Verify reasoning file content
                with open(reasoning_file, 'r', encoding='utf-8') as f:
                    reasoning_data = json.load(f)
                    assert len(reasoning_data) == 1
                    assert reasoning_data[0]["confidence"] == 0.9
                    assert "推理解釋" in reasoning_data[0]["reasoning"]
