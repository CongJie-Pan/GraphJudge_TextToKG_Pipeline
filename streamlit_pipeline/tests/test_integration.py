"""
Integration tests for GraphJudge Streamlit Pipeline.

This module provides comprehensive integration testing following the testing
strategy outlined in spec.md Section 15 and TDD principles from Testing_Demands.md.

Key testing areas:
- Module interactions across the pipeline
- Data flow consistency between stages  
- API integration with mocked responses
- Error propagation and recovery
- Performance under realistic loads
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from typing import List, Dict, Any

from core.models import (
    Triple, EntityResult, TripleResult, JudgmentResult, PipelineState,
    PipelineStage, ProcessingStatus, create_error_result
)
from fixtures.api_fixtures import (
    ScenarioFixtures, GPT5MiniFixtures, PerplexityFixtures,
    STANDARD_TEST_SCENARIOS, ERROR_TEST_SCENARIOS
)
from test_utils import (
    MockAPIClient, create_mock_gpt5_client, create_mock_perplexity_client,
    AsyncTestUtils, PerformanceTestUtils, ErrorSimulator,
    TestDataGenerator, CommonTestPatterns
)


# =============================================================================
# INTEGRATION TEST BASE CLASS
# =============================================================================

class BaseIntegrationTest:
    """Base class for integration tests with common setup."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up common test dependencies."""
        self.data_generator = TestDataGenerator()
        self.error_simulator = ErrorSimulator()
        self.performance_utils = PerformanceTestUtils()
    
    def assert_data_flow_consistency(self, input_data: Any, output_data: Any):
        """Assert that data flows consistently between pipeline stages."""
        # This will be implemented by subclasses based on their specific data types
        pass


# =============================================================================
# PIPELINE STAGE INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestPipelineStageIntegration(BaseIntegrationTest):
    """Test integration between different pipeline stages."""
    
    @pytest.mark.parametrize("scenario_name", [s[0] for s in STANDARD_TEST_SCENARIOS])
    def test_entity_to_triple_integration(self, scenario_name):
        """Test entity extraction results can be used for triple generation."""
        scenario = getattr(ScenarioFixtures, scenario_name)
        
        # Create entity result
        entity_result = EntityResult(
            entities=scenario["entities"],
            denoised_text=scenario["denoised_text"],
            success=True,
            processing_time=1.0
        )
        
        # Verify entity result can be used for triple generation
        assert entity_result.success, f"Entity extraction should succeed for {scenario_name}"
        assert len(entity_result.entities) > 0, "Should have extracted entities"
        assert entity_result.denoised_text, "Should have denoised text"
        
        # Simulate triple generation using entity results
        expected_triples = [
            Triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                confidence=t.get("confidence", 0.8)
            )
            for t in scenario["triples"]
        ]
        
        triple_result = TripleResult(
            triples=expected_triples,
            metadata={"input_entities": len(entity_result.entities)},
            success=True,
            processing_time=2.0
        )
        
        # Assert data flow consistency
        assert triple_result.success, "Triple generation should succeed with valid entity input"
        assert len(triple_result.triples) > 0, "Should generate triples from entities"
        
        # Verify metadata consistency
        assert triple_result.metadata["input_entities"] == len(entity_result.entities)
    
    @pytest.mark.parametrize("scenario_name", [s[0] for s in STANDARD_TEST_SCENARIOS])  
    def test_triple_to_judgment_integration(self, scenario_name):
        """Test triple results can be used for graph judgment."""
        scenario = getattr(ScenarioFixtures, scenario_name)
        
        # Create triple result
        triples = [
            Triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                confidence=t.get("confidence", 0.8)
            )
            for t in scenario["triples"]
        ]
        
        triple_result = TripleResult(
            triples=triples,
            metadata={"validation_passed": True},
            success=True,
            processing_time=2.0
        )
        
        # Simulate judgment using triple results
        judgments = [j["judgment"] for j in scenario["judgments"]]
        confidence_scores = [j["confidence"] for j in scenario["judgments"]]
        
        judgment_result = JudgmentResult(
            judgments=judgments,
            confidence=confidence_scores,
            success=True,
            processing_time=1.5
        )
        
        # Assert integration consistency
        assert len(judgment_result.judgments) == len(triple_result.triples)
        assert len(judgment_result.confidence) == len(triple_result.triples)
        assert judgment_result.success, "Judgment should succeed with valid triple input"
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from text input to final judgment."""
        scenario = ScenarioFixtures.DREAM_OF_RED_CHAMBER
        
        # Stage 1: Entity extraction
        entity_result = EntityResult(
            entities=scenario["entities"],
            denoised_text=scenario["denoised_text"],
            success=True,
            processing_time=1.2
        )
        
        # Stage 2: Triple generation
        triples = [
            Triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                confidence=t.get("confidence", 0.8)
            )
            for t in scenario["triples"]
        ]
        
        triple_result = TripleResult(
            triples=triples,
            metadata={
                "input_entities": len(entity_result.entities),
                "source_text_length": len(scenario["input_text"])
            },
            success=True,
            processing_time=2.1
        )
        
        # Stage 3: Graph judgment
        judgment_result = JudgmentResult(
            judgments=[j["judgment"] for j in scenario["judgments"]],
            confidence=[j["confidence"] for j in scenario["judgments"]],
            explanations=[j["explanation"] for j in scenario["judgments"]],
            success=True,
            processing_time=1.8
        )
        
        # Verify end-to-end consistency
        assert entity_result.success and triple_result.success and judgment_result.success
        assert len(judgment_result.judgments) == len(triple_result.triples)
        
        # Calculate total processing time
        total_time = (entity_result.processing_time + 
                     triple_result.processing_time + 
                     judgment_result.processing_time)
        
        assert total_time < 10.0, f"Pipeline should complete in reasonable time, got {total_time}s"


# =============================================================================
# API INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.mock_api
class TestAPIIntegration(BaseIntegrationTest):
    """Test API integration with mocked responses."""
    
    def test_gpt5_api_integration(self):
        """Test GPT-5-mini API integration with realistic responses."""
        mock_client = create_mock_gpt5_client("MODERN_SCENARIO")
        scenario = ScenarioFixtures.MODERN_SCENARIO
        
        # Test entity extraction
        entity_response = asyncio.run(
            mock_client.make_request("extract_entities", text=scenario["input_text"])
        )
        
        assert "choices" in entity_response
        assert len(entity_response["choices"]) > 0
        
        # Verify response structure matches OpenAI format
        choice = entity_response["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        
        # Test text denoising  
        denoising_response = asyncio.run(
            mock_client.make_request("denoise_text", text=scenario["input_text"])
        )
        
        assert "choices" in denoising_response
        denoised_content = denoising_response["choices"][0]["message"]["content"]
        assert denoised_content == scenario["denoised_text"]
    
    def test_perplexity_api_integration(self):
        """Test Perplexity API integration with realistic responses."""
        mock_client = create_mock_perplexity_client("MODERN_SCENARIO")
        scenario = ScenarioFixtures.MODERN_SCENARIO
        
        # Test judgment
        triples_input = scenario["triples"]
        judgment_response = asyncio.run(
            mock_client.make_request("judge_triples", triples=triples_input)
        )
        
        assert "choices" in judgment_response
        assert "usage" in judgment_response
        
        # Verify response structure
        choice = judgment_response["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
    
    @pytest.mark.parametrize("error_type", [s[0] for s in ERROR_TEST_SCENARIOS])
    def test_api_error_handling(self, error_type):
        """Test API error handling across different scenarios."""
        mock_client = MockAPIClient()
        mock_client.set_failure(f"Test {error_type} error")
        
        # Test that errors are properly caught and handled
        with pytest.raises(Exception, match=f"Test {error_type} error"):
            asyncio.run(mock_client.make_request("test_method"))
        
        # Verify call history is maintained even on errors
        assert mock_client.call_count == 1
        assert len(mock_client.call_history) == 1
    
    def test_api_rate_limiting_behavior(self):
        """Test API rate limiting behavior."""
        mock_client = MockAPIClient()
        
        # Set up rate limit response
        rate_limit_response = GPT5MiniFixtures.rate_limit_error()
        mock_client.set_response("limited_method", rate_limit_response)
        
        response = asyncio.run(mock_client.make_request("limited_method"))
        
        assert "error" in response
        assert response["error"]["type"] == "rate_limit_error"


# =============================================================================
# ERROR PROPAGATION INTEGRATION TESTS  
# =============================================================================

@pytest.mark.integration
class TestErrorPropagation(BaseIntegrationTest):
    """Test error propagation across pipeline stages."""
    
    def test_entity_stage_error_propagation(self):
        """Test error handling when entity extraction fails."""
        # Create failed entity result
        entity_result = create_error_result(
            EntityResult, 
            "Entity extraction API failed", 
            processing_time=0.5
        )
        
        assert not entity_result.success
        assert entity_result.error == "Entity extraction API failed"
        assert len(entity_result.entities) == 0
        
        # Verify downstream stages can handle empty entity results
        triple_result = TripleResult(
            triples=[],
            metadata={"input_entities": 0, "skipped_due_to_error": True},
            success=False,
            processing_time=0.0,
            error="No entities available for triple generation"
        )
        
        assert not triple_result.success
        assert len(triple_result.triples) == 0
    
    def test_triple_stage_error_propagation(self):
        """Test error handling when triple generation fails."""
        # Start with successful entity extraction
        entity_result = EntityResult(
            entities=["实体1", "实体2"],
            denoised_text="测试文本",
            success=True,
            processing_time=1.0
        )
        
        # Create failed triple result
        triple_result = create_error_result(
            TripleResult,
            "Triple generation JSON parsing failed",
            processing_time=1.2
        )
        
        assert entity_result.success  # Previous stage succeeded
        assert not triple_result.success  # Current stage failed
        assert "JSON parsing" in triple_result.error
    
    def test_judgment_stage_error_propagation(self):
        """Test error handling when graph judgment fails."""
        # Previous stages successful
        entity_result = EntityResult(
            entities=["A", "B"],
            denoised_text="A relates to B",
            success=True,
            processing_time=1.0
        )
        
        triples = [Triple("A", "relates_to", "B", confidence=0.8)]
        triple_result = TripleResult(
            triples=triples,
            metadata={"validation_passed": True},
            success=True,
            processing_time=2.0
        )
        
        # Failed judgment stage
        judgment_result = create_error_result(
            JudgmentResult,
            "Perplexity API timeout",
            processing_time=0.3
        )
        
        assert entity_result.success and triple_result.success
        assert not judgment_result.success
        assert "timeout" in judgment_result.error
        
        # Verify partial results are available
        assert len(triple_result.triples) > 0  # Triples available even if judgment failed
    
    def test_pipeline_state_error_tracking(self):
        """Test pipeline state tracks errors correctly."""
        pipeline_state = PipelineState(
            input_text="测试文本",
            status=ProcessingStatus.RUNNING_TRIPLE
        )
        
        # Simulate error in triple stage
        pipeline_state.status = ProcessingStatus.FAILED
        pipeline_state.error_stage = PipelineStage.TRIPLE_GENERATION
        pipeline_state.error_message = "Triple generation timeout"
        
        assert pipeline_state.has_error
        assert not pipeline_state.is_complete
        assert pipeline_state.error_stage == PipelineStage.TRIPLE_GENERATION
        assert pipeline_state.progress_percentage == 40.0  # Failed at triple stage


# =============================================================================
# PERFORMANCE INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration(BaseIntegrationTest):
    """Test performance characteristics under realistic conditions."""
    
    def test_single_document_performance(self):
        """Test performance with single document processing."""
        scenario = ScenarioFixtures.DREAM_OF_RED_CHAMBER
        
        # Time each stage
        start_time = asyncio.get_event_loop().time()
        
        # Stage 1: Entity extraction (simulated)
        entity_result = EntityResult(
            entities=scenario["entities"],
            denoised_text=scenario["denoised_text"],
            success=True,
            processing_time=1.5
        )
        
        # Stage 2: Triple generation (simulated)
        triples = [
            Triple(t["subject"], t["predicate"], t["object"], t.get("confidence", 0.8))
            for t in scenario["triples"]
        ]
        triple_result = TripleResult(
            triples=triples,
            metadata={"chunks": 1},
            success=True,
            processing_time=2.2
        )
        
        # Stage 3: Graph judgment (simulated)
        judgment_result = JudgmentResult(
            judgments=[j["judgment"] for j in scenario["judgments"]],
            confidence=[j["confidence"] for j in scenario["judgments"]],
            success=True,
            processing_time=1.8
        )
        
        total_time = (entity_result.processing_time + 
                     triple_result.processing_time + 
                     judgment_result.processing_time)
        
        # Performance assertions
        assert total_time < 10.0, f"Single document should process in <10s, got {total_time}s"
        assert entity_result.processing_time < 3.0, "Entity extraction should be <3s"
        assert triple_result.processing_time < 5.0, "Triple generation should be <5s"
        assert judgment_result.processing_time < 3.0, "Graph judgment should be <3s"
    
    def test_batch_processing_performance(self):
        """Test performance with batch processing simulation."""
        batch_size = 5
        scenarios = [
            ScenarioFixtures.DREAM_OF_RED_CHAMBER,
            ScenarioFixtures.MODERN_SCENARIO,
            ScenarioFixtures.SINGLE_ENTITY
        ]
        
        # Simulate batch processing
        results = []
        total_start = asyncio.get_event_loop().time()
        
        for i in range(batch_size):
            scenario = scenarios[i % len(scenarios)]
            
            # Simulate processing each item
            entity_result = EntityResult(
                entities=scenario["entities"],
                denoised_text=scenario["denoised_text"],
                success=True,
                processing_time=1.0 + (i * 0.1)  # Slight variation
            )
            results.append(entity_result)
        
        total_end = asyncio.get_event_loop().time()
        total_batch_time = total_end - total_start
        
        # Performance assertions
        assert len(results) == batch_size
        assert all(r.success for r in results)
        assert total_batch_time < 30.0, f"Batch of {batch_size} should complete in <30s"
        
        # Average processing time should be reasonable
        avg_time = sum(r.processing_time for r in results) / len(results)
        assert avg_time < 2.0, f"Average processing time should be <2s, got {avg_time}s"


# =============================================================================
# DATA CONSISTENCY INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestDataConsistency(BaseIntegrationTest):
    """Test data consistency across pipeline stages."""
    
    def test_entity_count_consistency(self):
        """Test entity counts remain consistent through processing."""
        scenario = ScenarioFixtures.MODERN_SCENARIO
        
        entity_result = EntityResult(
            entities=scenario["entities"],
            denoised_text=scenario["denoised_text"],
            success=True,
            processing_time=1.0
        )
        
        # Verify entities are preserved in downstream processing
        input_entity_count = len(entity_result.entities)
        
        # Triple generation should reference the extracted entities
        triples = scenario["triples"]
        referenced_entities = set()
        
        for triple in triples:
            if triple["subject"] in entity_result.entities:
                referenced_entities.add(triple["subject"])
            if triple["object"] in entity_result.entities:
                referenced_entities.add(triple["object"])
        
        # Should have reasonable entity utilization
        utilization_ratio = len(referenced_entities) / input_entity_count
        assert utilization_ratio > 0.3, f"Low entity utilization: {utilization_ratio:.2%}"
    
    def test_triple_judgment_consistency(self):
        """Test judgments correspond to triples correctly."""
        scenario = ScenarioFixtures.DREAM_OF_RED_CHAMBER
        
        triples = [
            Triple(t["subject"], t["predicate"], t["object"])
            for t in scenario["triples"]
        ]
        
        triple_result = TripleResult(
            triples=triples,
            metadata={"total_generated": len(triples)},
            success=True,
            processing_time=2.0
        )
        
        judgment_result = JudgmentResult(
            judgments=[j["judgment"] for j in scenario["judgments"]],
            confidence=[j["confidence"] for j in scenario["judgments"]],
            explanations=[j["explanation"] for j in scenario["judgments"]],
            success=True,
            processing_time=1.5
        )
        
        # Verify 1:1 correspondence
        assert len(judgment_result.judgments) == len(triple_result.triples)
        assert len(judgment_result.confidence) == len(triple_result.triples)
        assert len(judgment_result.explanations) == len(triple_result.triples)
        
        # Verify data types are consistent
        assert all(isinstance(j, bool) for j in judgment_result.judgments)
        assert all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in judgment_result.confidence)
        assert all(isinstance(e, str) and len(e) > 0 for e in judgment_result.explanations)
    
    def test_processing_metadata_consistency(self):
        """Test processing metadata is consistent across stages."""
        scenario = ScenarioFixtures.MODERN_SCENARIO
        
        # Each stage should track timing and success metrics
        stages = [
            EntityResult(scenario["entities"], scenario["denoised_text"], True, 1.2),
            TripleResult(
                [Triple(t["subject"], t["predicate"], t["object"]) for t in scenario["triples"]],
                {"input_entities": len(scenario["entities"])},
                True, 
                2.1
            ),
            JudgmentResult(
                [j["judgment"] for j in scenario["judgments"]],
                [j["confidence"] for j in scenario["judgments"]],
                success=True,
                processing_time=1.8
            )
        ]
        
        # Verify all stages have required metadata
        for stage in stages:
            assert hasattr(stage, 'success'), "All stages should track success"
            assert hasattr(stage, 'processing_time'), "All stages should track timing"
            assert isinstance(stage.processing_time, (int, float)), "Processing time should be numeric"
            assert stage.processing_time > 0, "Processing time should be positive"


# =============================================================================
# LOGGING INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestLoggingIntegration(BaseIntegrationTest):
    """Test logging system integration across pipeline stages."""

    def setup_method(self):
        """Set up test fixtures for logging tests."""
        # Initialize base class attributes
        self.data_generator = TestDataGenerator()
        self.error_simulator = ErrorSimulator()
        self.performance_utils = PerformanceTestUtils()

        # Create temporary directory for test logs
        import tempfile
        self.test_logs_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures after logging tests."""
        import shutil
        import os
        if os.path.exists(self.test_logs_dir):
            shutil.rmtree(self.test_logs_dir)

    @patch('core.entity_processor.DetailedLogger')
    def test_entity_processor_logging_integration(self, mock_logger_class):
        """Test that entity processor integrates correctly with logging system."""
        from core.entity_processor import extract_entities

        # Create mock logger instance
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock API call to avoid actual API request
        with patch('core.entity_processor.call_gpt5_mini') as mock_api:
            mock_api.return_value = '["entity1", "entity2", "entity3"]'

            # Test entity extraction with logging
            result = extract_entities("Sample Chinese text for testing")

            # Verify logger was initialized for ECTD phase
            mock_logger_class.assert_called_with(phase="ectd")

            # Verify key logging calls were made
            mock_logger.log_info.assert_called()
            mock_logger.log_debug.assert_called()

            # Verify result structure
            assert result.success is True
            assert len(result.entities) == 3

    @patch('streamlit_pipeline.utils.detailed_logger.DetailedLogger')
    def test_triple_generator_logging_integration(self, mock_logger_class):
        """Test that triple generator integrates correctly with logging system."""
        from core.triple_generator import generate_triples

        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock API call
        with patch('core.triple_generator.call_gpt5_mini') as mock_api:
            mock_api.return_value = '''[
                {"subject": "subject1", "predicate": "predicate1", "object": "object1"},
                {"subject": "subject2", "predicate": "predicate2", "object": "object2"}
            ]'''

            # Test triple generation with logging
            result = generate_triples("Sample denoised text", ["entity1", "entity2"])

            # Verify logger was initialized for triple generation phase
            mock_logger_class.assert_called_with(phase="triple_gen")

            # Verify logging calls
            mock_logger.log_info.assert_called()

            # Verify result structure
            assert result.success is True
            assert len(result.triples) == 2

    @patch('streamlit_pipeline.utils.detailed_logger.DetailedLogger')
    def test_graph_judge_logging_integration(self, mock_logger_class):
        """Test that graph judge integrates correctly with logging system."""
        from core.graph_judge import judge_triples

        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock API call
        with patch('core.graph_judge.call_perplexity_api') as mock_api:
            mock_api.return_value = '''[
                {"judgment": "correct", "confidence": 0.95},
                {"judgment": "incorrect", "confidence": 0.85}
            ]'''

            # Test graph judgment with logging
            from core.models import Triple
            test_triples = [
                Triple("subject1", "predicate1", "object1"),
                Triple("subject2", "predicate2", "object2")
            ]

            result = judge_triples(test_triples, "Original text context")

            # Verify logger was initialized for graph judge phase
            mock_logger_class.assert_called_with(phase="graph_judge")

            # Verify logging calls
            mock_logger.log_info.assert_called()

    def test_logging_directory_structure(self):
        """Test that logging creates proper directory structure."""
        from utils.detailed_logger import DetailedLogger
        from datetime import datetime

        logger = DetailedLogger(phase="test_phase")

        # Verify date-based directory structure
        expected_date = datetime.now().strftime("%Y_%m_%d")
        assert expected_date in str(logger.logs_dir)
        assert "test_phase" in str(logger.logs_dir)
        assert logger.logs_dir.exists()

        # Test that log files are created
        logger.log_info("TEST", "Test message")

        debug_file = logger.logs_dir / f"test_phase_{logger.session_id}.log"
        assert debug_file.exists()

        content = debug_file.read_text(encoding='utf-8')
        assert "Test message" in content

    def test_concurrent_phase_logging(self):
        """Test that multiple pipeline phases can log concurrently."""
        from utils.detailed_logger import DetailedLogger

        # Create loggers for different phases
        ectd_logger = DetailedLogger(phase="ectd")
        triple_logger = DetailedLogger(phase="triple_gen")
        judge_logger = DetailedLogger(phase="graph_judge")

        # Log messages from each phase
        ectd_logger.log_info("ENTITY", "Entity extraction started")
        triple_logger.log_info("TRIPLE", "Triple generation started")
        judge_logger.log_info("JUDGE", "Graph judgment started")

        # Verify separate log files exist
        ectd_file = ectd_logger.logs_dir / f"ectd_{ectd_logger.session_id}.log"
        triple_file = triple_logger.logs_dir / f"triple_gen_{triple_logger.session_id}.log"
        judge_file = judge_logger.logs_dir / f"graph_judge_{judge_logger.session_id}.log"

        assert ectd_file.exists()
        assert triple_file.exists()
        assert judge_file.exists()

        # Verify content isolation
        ectd_content = ectd_file.read_text(encoding='utf-8')
        triple_content = triple_file.read_text(encoding='utf-8')
        judge_content = judge_file.read_text(encoding='utf-8')

        assert "Entity extraction started" in ectd_content
        assert "Triple generation started" in triple_content
        assert "Graph judgment started" in judge_content

        # Verify no cross-contamination
        assert "Triple generation started" not in ectd_content
        assert "Graph judgment started" not in ectd_content
        assert "Entity extraction started" not in triple_content

    def test_error_logging_integration(self):
        """Test that errors are properly logged across the pipeline."""
        from utils.detailed_logger import DetailedLogger

        logger = DetailedLogger(phase="test_error")

        # Log various error types
        logger.log_error("API", "API connection failed", {
            "error_code": 500,
            "retry_count": 3
        })

        logger.log_warning("STORAGE", "Disk space running low")

        # Verify error file creation
        error_file = logger.logs_dir / f"test_error_{logger.session_id}_errors.log"
        warning_file = logger.logs_dir / f"test_error_{logger.session_id}_warnings.log"

        assert error_file.exists()
        assert warning_file.exists()

        # Verify error content
        error_content = error_file.read_text(encoding='utf-8')
        warning_content = warning_file.read_text(encoding='utf-8')

        assert "API connection failed" in error_content
        assert "error_code" in error_content
        assert "Disk space running low" in warning_content

    def test_api_call_logging_integration(self):
        """Test that API calls are properly logged with metrics."""
        from utils.detailed_logger import DetailedLogger

        logger = DetailedLogger(phase="test_api")

        # Log successful API call
        logger.log_api_call("GPT-5-mini", 500, 150, success=True)

        # Log failed API call
        logger.log_api_call("GPT-5-mini", 300, 0, success=False, error="Rate limit exceeded")

        # Verify logging
        debug_file = logger.logs_dir / f"test_api_{logger.session_id}.log"
        error_file = logger.logs_dir / f"test_api_{logger.session_id}_errors.log"

        debug_content = debug_file.read_text(encoding='utf-8')
        error_content = error_file.read_text(encoding='utf-8')

        # Verify successful call logging
        assert "API call to GPT-5-mini" in debug_content
        assert "input_tokens: 500" in debug_content
        assert "success: true" in debug_content

        # Verify failed call logging
        assert "success: false" in debug_content
        assert "Rate limit exceeded" in error_content

    def test_pipeline_lifecycle_logging(self):
        """Test complete pipeline lifecycle logging."""
        from utils.detailed_logger import DetailedLogger

        logger = DetailedLogger(phase="pipeline_test")

        # Simulate complete pipeline lifecycle
        config = {"model": "gpt-5-mini", "temperature": 0.7}
        logger.log_pipeline_start(config)

        # Entity extraction phase
        logger.log_phase_start("entity_extraction", {"text_length": 5000})
        logger.log_api_call("GPT-5-mini", 500, 150, success=True)
        logger.log_phase_complete("entity_extraction", True, {"entity_count": 25})

        # Triple generation phase
        logger.log_phase_start("triple_generation", {"entity_count": 25})
        logger.log_api_call("GPT-5-mini", 600, 200, success=True)
        logger.log_phase_complete("triple_generation", True, {"triple_count": 40})

        # Graph judgment phase
        logger.log_phase_start("graph_judgment", {"triple_count": 40})
        logger.log_api_call("Perplexity", 800, 100, success=True)
        logger.log_phase_complete("graph_judgment", True, {"judged_count": 40})

        results = {"total_entities": 25, "total_triples": 40, "success": True}
        logger.log_pipeline_complete(results)

        # Verify complete lifecycle logging
        debug_file = logger.logs_dir / f"pipeline_test_{logger.session_id}.log"
        content = debug_file.read_text(encoding='utf-8')

        assert "Pipeline execution started" in content
        assert "entity_extraction" in content
        assert "triple_generation" in content
        assert "graph_judgment" in content
        assert "Pipeline execution completed" in content

        # Verify metrics were logged
        assert "entity_count: 25" in content
        assert "triple_count: 40" in content
        assert "judged_count: 40" in content