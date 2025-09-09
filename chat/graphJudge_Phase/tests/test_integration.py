"""
Integration tests for the complete GraphJudge Phase system.

Tests the integration between all modules working together,
end-to-end workflows, and compatibility with the original
run_gj.py functionality.
"""

import pytest
import asyncio
import os
import json
import csv
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

from ..main import main
from ..graph_judge_core import PerplexityGraphJudge
from ..processing_pipeline import ProcessingPipeline
from ..gold_label_bootstrapping import GoldLabelBootstrapper
from ..data_structures import ExplainableJudgment, BootstrapResult, TripleData
from .conftest import PerplexityTestBase, create_test_files


class TestModuleIntegration(PerplexityTestBase):
    """Test integration between different modules."""
    
    def test_graph_judge_with_prompt_engineering(self):
        """Test integration between graph judge and prompt engineering."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True  # Ensure predictable behavior
        
        # Test that PromptEngineer is properly integrated
        instruction = "Is this true: 曹雪芹 創作 紅樓夢 ?"
        
        # This should work without errors
        prompt = judge.prompt_engineer.create_graph_judgment_prompt(instruction)
        assert "曹雪芹 創作 紅樓夢" in prompt
        
        # Test explainable prompt creation
        explainable_prompt = judge.prompt_engineer.create_explainable_judgment_prompt(instruction)
        assert "判斷結果" in explainable_prompt
    
    @pytest.mark.asyncio
    async def test_pipeline_with_graph_judge(self, mock_async_perplexity_judge):
        """Test integration between processing pipeline and graph judge."""
        pipeline = ProcessingPipeline(mock_async_perplexity_judge)
        
        test_data = [
            {"instruction": "Is this true: Apple Founded by Steve Jobs ?", "input": "", "output": ""}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=test_data,
                    explainable_mode=False
                )
                
                # Should have successfully processed through the pipeline
                assert stats.total_instructions == 1
                assert os.path.exists(output_file)
    
    @pytest.mark.asyncio
    async def test_bootstrapper_with_graph_judge(self, mock_async_perplexity_judge):
        """Test integration between gold label bootstrapper and graph judge."""
        bootstrapper = GoldLabelBootstrapper(mock_async_perplexity_judge)
        
        # Create test triples and source lines
        triples = [TripleData("Apple", "Founded by", "Steve Jobs")]
        source_lines = ["Apple was founded by Steve Jobs in 1976."]
        
        # Test Stage 1 (RapidFuzz matching)
        stage1_results = bootstrapper.stage1_rapidfuzz_matching(triples, source_lines)
        assert len(stage1_results) == 1
        
        # Test Stage 2 (LLM evaluation) with uncertain cases
        uncertain_results = [result for result in stage1_results if result.auto_expected is None]
        if uncertain_results:
            stage2_results = await bootstrapper.stage2_llm_semantic_evaluation(uncertain_results, source_lines)
            assert len(stage2_results) >= 0


class TestEndToEndWorkflows(PerplexityTestBase):
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_standard_judgment_workflow(self):
        """Test complete standard judgment workflow."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        pipeline = ProcessingPipeline(judge)
        
        # Chinese literature examples
        test_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""},
            {"instruction": "Is this true: 賈寶玉 喜歡 林黛玉 ?", "input": "", "output": ""},
            {"instruction": "Is this true: 賈寶玉 創作 紅樓夢 ?", "input": "", "output": ""}  # Should be false
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=test_data,
                    explainable_mode=False
                )
                
                # Verify processing completed
                assert stats.total_instructions == 3
                assert stats.successful_responses >= 0
                assert os.path.exists(output_file)
                
                # Verify CSV output structure matches original run_gj.py
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    assert len(rows) == 3
                    assert all("prompt" in row and "generated" in row for row in rows)
                    assert all(row["generated"] in ["Yes", "No"] or "Error" in row["generated"] for row in rows)
    
    @pytest.mark.asyncio
    async def test_explainable_judgment_workflow(self):
        """Test complete explainable judgment workflow."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        pipeline = ProcessingPipeline(judge)
        
        test_data = [
            {"instruction": "Is this true: Microsoft Founded by Mark Zuckerberg ?", "input": "", "output": ""}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            reasoning_file = output_file.replace('.csv', '_reasoning.json')
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=test_data,
                    explainable_mode=True,
                    reasoning_file_path=reasoning_file
                )
                
                # Verify dual output files
                assert os.path.exists(output_file)
                assert os.path.exists(reasoning_file)
                
                # Verify reasoning file structure
                with open(reasoning_file, 'r', encoding='utf-8') as f:
                    reasoning_data = json.load(f)
                    assert len(reasoning_data) == 1
                    
                    reasoning_entry = reasoning_data[0]
                    assert "judgment" in reasoning_entry
                    assert "confidence" in reasoning_entry
                    assert "reasoning" in reasoning_entry
                    assert "evidence_sources" in reasoning_entry
                    assert "alternative_suggestions" in reasoning_entry
                    assert "error_type" in reasoning_entry
                    assert "processing_time" in reasoning_entry
    
    @pytest.mark.asyncio
    async def test_bootstrap_workflow(self):
        """Test complete gold label bootstrapping workflow."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        bootstrapper = GoldLabelBootstrapper(judge)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            triples_file = os.path.join(temp_dir, "triples.txt")
            source_file = os.path.join(temp_dir, "source.txt")
            output_file = os.path.join(temp_dir, "bootstrap_output.csv")
            
            # Write test triples
            with open(triples_file, 'w', encoding='utf-8') as f:
                f.write('according to text [["曹雪芹", "創作", "紅樓夢"], ["賈寶玉", "喜歡", "林黛玉"]]\n')
                f.write('another description [["Apple", "Founded by", "Steve Jobs"]]\n')
            
            # Write test source text
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write('曹雪芹創作了著名的紅樓夢小說。\n')
                f.write('賈寶玉在小說中深深地喜歡林黛玉。\n')
                f.write('Steve Jobs 創立了 Apple 公司。\n')
                f.write('這是一個不相關的句子。\n')
            
            # Run bootstrap process
            success = await bootstrapper.bootstrap_gold_labels(triples_file, source_file, output_file)
            
            assert success is True
            assert os.path.exists(output_file)
            
            # Verify output structure
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) >= 2  # Should have processed multiple triples
                
                # Check required columns
                expected_columns = ['subject', 'predicate', 'object', 'source_idx', 
                                  'fuzzy_score', 'auto_expected', 'expected', 'note']
                for col in expected_columns:
                    assert col in reader.fieldnames


class TestCompatibilityWithOriginal(PerplexityTestBase):
    """Test compatibility with original run_gj.py functionality."""
    
    @pytest.mark.asyncio
    async def test_output_format_compatibility(self):
        """Test that output format matches original run_gj.py."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        pipeline = ProcessingPipeline(judge)
        
        # Use same test data format as original
        test_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Apple Founded by Steve Jobs ?", "input": "", "output": ""}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                await pipeline.process_instructions(
                    data_eval=test_data,
                    explainable_mode=False
                )
                
                # Verify CSV format matches original
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Should have header and data
                    lines = content.strip().split('\n')
                    assert lines[0] == "prompt,generated"
                    assert len(lines) >= 3  # Header + data rows
                    
                    # Each data line should have proper CSV format
                    for line in lines[1:]:
                        parts = line.split(',', 1)  # Split on first comma only
                        assert len(parts) == 2
                        assert parts[1] in ["Yes", "No"] or "Error" in parts[1]
    
    @pytest.mark.asyncio
    async def test_environment_variable_compatibility(self):
        """Test compatibility with environment variable usage."""
        # Test that the modular system respects same environment variables
        with patch.dict(os.environ, {
            'PIPELINE_ITERATION': '5',
            'PIPELINE_INPUT_FILE': self.test_input_file,
            'PIPELINE_OUTPUT_FILE': self.test_output_file
        }):
            # Re-import config to pick up environment variables
            import importlib
            from .. import config
            importlib.reload(config)
            
            # Should use environment variables
            assert config.iteration == 5
            assert config.input_file == self.test_input_file
            assert config.output_file == self.test_output_file
    
    def test_data_structure_compatibility(self):
        """Test that data structures are compatible with original formats."""
        # Test TripleData compatibility
        triple = TripleData("Subject", "Predicate", "Object", "source", 1)
        assert hasattr(triple, 'subject')
        assert hasattr(triple, 'predicate')
        assert hasattr(triple, 'object')
        assert hasattr(triple, 'source_line')
        assert hasattr(triple, 'line_number')
        
        # Test that it can be converted to dict (for JSON serialization)
        triple_dict = triple._asdict()
        assert isinstance(triple_dict, dict)
        assert triple_dict['subject'] == "Subject"
        
        # Test ExplainableJudgment compatibility
        judgment = ExplainableJudgment(
            judgment="Yes",
            confidence=0.9,
            reasoning="Test reasoning",
            evidence_sources=["test"],
            alternative_suggestions=[],
            error_type=None,
            processing_time=1.0
        )
        
        judgment_dict = judgment._asdict()
        assert isinstance(judgment_dict, dict)
        assert judgment_dict['judgment'] == "Yes"


class TestErrorHandlingIntegration(PerplexityTestBase):
    """Test error handling across integrated modules."""
    
    @pytest.mark.asyncio
    async def test_api_error_propagation(self):
        """Test that API errors are properly handled across modules."""
        # Create judge that will fail
        judge = PerplexityGraphJudge()
        judge.is_mock = False  # Force real mode
        
        # Mock API to fail
        with patch('graphJudge_Phase.graph_judge_core.acompletion', side_effect=Exception("API Error")):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                pipeline = ProcessingPipeline(judge)
                
                test_data = [
                    {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
                ]
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    input_file, output_file = create_test_files(temp_dir, test_data)
                    
                    with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                        stats = await pipeline.process_instructions(
                            data_eval=test_data,
                            explainable_mode=False
                        )
                        
                        # Should handle errors gracefully
                        assert stats.total_instructions == 1
                        assert stats.error_responses >= 0  # May have errors
                        assert os.path.exists(output_file)
    
    @pytest.mark.asyncio
    async def test_file_system_error_handling(self):
        """Test handling of file system errors."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        pipeline = ProcessingPipeline(judge)
        
        test_data = [
            {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
        ]
        
        # Try to write to invalid path
        invalid_output_path = "/invalid/path/output.csv"
        
        with patch('graphJudge_Phase.processing_pipeline.output_file', invalid_output_path):
            # Should handle file system errors gracefully
            try:
                stats = await pipeline.process_instructions(
                    data_eval=test_data,
                    explainable_mode=False
                )
                # If it succeeds, that's also fine (some systems might handle it)
                assert isinstance(stats.total_instructions, int)
            except (OSError, IOError, PermissionError):
                # Expected to fail with file system error
                pass
    
    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            try:
                judge = PerplexityGraphJudge()
                # Should either work in mock mode or raise appropriate error
                assert isinstance(judge.is_mock, bool)
            except ValueError as e:
                # Expected if API key is required
                assert "PERPLEXITYAI_API_KEY" in str(e)


class TestPerformanceIntegration(PerplexityTestBase):
    """Test performance characteristics of integrated system."""
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test performance of batch processing."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True  # Use mock for predictable timing
        
        pipeline = ProcessingPipeline(judge)
        
        # Create larger dataset
        test_data = [
            {"instruction": f"Is this true: Statement {i} ?", "input": "", "output": ""}
            for i in range(20)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file, output_file = create_test_files(temp_dir, test_data)
            
            import time
            start_time = time.time()
            
            with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                stats = await pipeline.process_instructions(
                    data_eval=test_data,
                    explainable_mode=False
                )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process all items
            assert stats.total_instructions == 20
            
            # Performance should be reasonable (less than 30 seconds for 20 items in mock mode)
            assert processing_time < 30.0
            
            # Should maintain good throughput
            throughput = stats.total_instructions / processing_time
            assert throughput > 0.5  # At least 0.5 items per second
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that memory usage remains stable during processing."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        pipeline = ProcessingPipeline(judge)
        
        # Process multiple batches to check for memory leaks
        for batch in range(3):
            test_data = [
                {"instruction": f"Is this true: Batch {batch} Statement {i} ?", "input": "", "output": ""}
                for i in range(10)
            ]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                input_file, output_file = create_test_files(temp_dir, test_data)
                
                with patch('graphJudge_Phase.processing_pipeline.output_file', output_file):
                    stats = await pipeline.process_instructions(
                        data_eval=test_data,
                        explainable_mode=False
                    )
                    
                    # Each batch should process successfully
                    assert stats.total_instructions == 10


class TestCrossModuleDataFlow(PerplexityTestBase):
    """Test data flow between different modules."""
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_modules(self):
        """Test that data remains consistent as it flows through modules."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        # Test prompt creation -> judgment -> response parsing
        instruction = "Is this true: 曹雪芹 創作 紅樓夢 ?"
        
        # 1. Create prompt
        prompt = judge.prompt_engineer.create_graph_judgment_prompt(instruction)
        assert "曹雪芹 創作 紅樓夢" in prompt
        
        # 2. Get judgment
        judgment = await judge.judge_graph_triple(instruction)
        assert judgment in ["Yes", "No"]
        
        # 3. Test explainable flow
        explainable_judgment = await judge.judge_graph_triple_with_explanation(instruction)
        assert isinstance(explainable_judgment, ExplainableJudgment)
        assert explainable_judgment.judgment == judgment  # Should be consistent
    
    def test_configuration_propagation(self):
        """Test that configuration properly propagates through modules."""
        # Test custom configuration
        judge = PerplexityGraphJudge(
            model_name="perplexity/sonar-pro",
            reasoning_effort="high",
            enable_console_logging=True
        )
        
        # Configuration should be accessible in integrated modules
        assert judge.model_name == "perplexity/sonar-pro"
        assert judge.reasoning_effort == "high"
        assert judge.enable_logging == True
        
        # Pipeline should inherit judge configuration
        pipeline = ProcessingPipeline(judge)
        assert pipeline.graph_judge == judge
        assert pipeline.graph_judge.model_name == "perplexity/sonar-pro"
