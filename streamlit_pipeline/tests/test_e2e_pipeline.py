"""
End-to-End Pipeline Integration Tests for GraphJudge Streamlit Pipeline.

This module provides comprehensive end-to-end testing of the complete pipeline
workflow, following testing strategy from spec.md Section 15 and requirements
from docs/Testing_Demands.md.

Test Coverage:
- Complete pipeline execution (Entity → Triple → Graph Judge)
- Data flow integrity across all stages
- Error recovery and partial failure scenarios
- Performance under realistic loads
- Configuration management and API integration
- Session state management and cleanup
"""

import pytest
import asyncio
import time
import json
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import core pipeline components
from streamlit_pipeline.core.pipeline import PipelineOrchestrator, PipelineResult
from streamlit_pipeline.core.entity_processor import extract_entities
from streamlit_pipeline.core.triple_generator import generate_triples
from streamlit_pipeline.core.graph_judge import judge_triples
from streamlit_pipeline.core.models import (
    Triple, EntityResult, TripleResult, JudgmentResult, PipelineState,
    PipelineStage, ProcessingStatus
)

# Import utilities and fixtures
from streamlit_pipeline.utils.api_client import get_api_client
from streamlit_pipeline.utils.error_handling import ErrorHandler, ErrorType
from streamlit_pipeline.utils.session_state import get_session_manager
from streamlit_pipeline.utils.state_persistence import get_persistence_manager
from streamlit_pipeline.utils.state_cleanup import get_cleanup_manager


@pytest.fixture
def sample_chinese_text():
    """Provide sample Chinese text for testing."""
    return """
    林黛玉是賈府的親戚，從江南來到榮國府。她聰明伶俐，才情出眾，
    深受賈母喜愛。賈寶玉初見林黛玉時，覺得這個妹妹似曾相識。
    二人因緣分深厚，漸生情愫。賈府是金陵四大家族之一，門第顯赫。
    """


@pytest.fixture
def mock_successful_api_responses():
    """Provide mock API responses for successful pipeline execution."""
    return {
        'entity_extraction': {
            'entities': ['林黛玉', '賈府', '榮國府', '賈母', '賈寶玉', '金陵四大家族'],
            'denoised_text': '林黛玉來到榮國府，深受賈母喜愛。賈寶玉初見林黛玉。賈府是金陵四大家族之一。'
        },
        'triple_generation': [
            {'subject': '林黛玉', 'predicate': '來到', 'object': '榮國府'},
            {'subject': '林黛玉', 'predicate': '受到喜愛', 'object': '賈母'},
            {'subject': '賈寶玉', 'predicate': '初見', 'object': '林黛玉'},
            {'subject': '賈府', 'predicate': '屬於', 'object': '金陵四大家族'}
        ],
        'graph_judgment': {
            'judgments': [True, True, True, True],
            'explanations': [
                '林黛玉確實來到榮國府居住',
                '賈母確實喜愛林黛玉',
                '賈寶玉與林黛玉確實初次相見',
                '賈府確實是金陵四大家族之一'
            ]
        }
    }


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteE2EPipeline:
    """Comprehensive end-to-end pipeline testing."""
    
    def test_complete_pipeline_success_flow(self, sample_chinese_text, mock_successful_api_responses, unified_api_mock, mock_environment_with_api_keys):
        """Test complete successful pipeline execution from start to finish."""
        
        # Mock all API calls to return successful responses
        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.core.pipeline.get_api_client') as mock_pipeline_api_client, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_graph_judge_api_client:
            
            # Configure mock API responses - need to return entity list format for first call, text for second call
            call_count = 0
            original_entities = mock_successful_api_responses['entity_extraction']['entities']
            original_denoised = mock_successful_api_responses['entity_extraction']['denoised_text']
            
            def mock_entity_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # First call is entity extraction
                    return str(original_entities)  # Return list as string
                else:  # Second call is text denoising
                    return original_denoised
            
            mock_entity_api.side_effect = mock_entity_response
            
            # Mock API clients for triple generation and graph judgment
            mock_triple_client = Mock()
            mock_graph_judge_client = Mock()
            mock_pipeline_api_client.return_value = mock_triple_client
            mock_graph_judge_api_client.return_value = mock_graph_judge_client
            
            # Configure triple generation client - use call_gpt5_mini method
            triples_json = {
                "triples": [
                    [triple["subject"], triple["predicate"], triple["object"]]
                    for triple in mock_successful_api_responses['triple_generation']
                ]
            }
            mock_triple_client.call_gpt5_mini.return_value = f"```json\n{json.dumps(triples_json)}\n```"
            
            # Configure graph judgment client - use call_perplexity method
            judgment_parts = []
            for i, (judgment, explanation) in enumerate(
                zip(mock_successful_api_responses['graph_judgment']['judgments'],
                    mock_successful_api_responses['graph_judgment']['explanations'])
            ):
                result = "Yes" if judgment else "No"
                judgment_parts.append(f"Triple {i+1}: {result} - {explanation}")

            mock_graph_judge_client.call_perplexity.return_value = "\n".join(judgment_parts)
            
            # Execute complete pipeline
            orchestrator = PipelineOrchestrator()
            progress_updates = []
            
            def progress_callback(stage: int, message: str):
                progress_updates.append((stage, message))
            
            result = orchestrator.run_pipeline(sample_chinese_text, progress_callback)
            
            # Verify successful completion
            assert result.success, f"Pipeline should succeed, but got error: {result.error}"
            assert result.stage_reached == 3, "Should complete all 3 stages"
            assert result.total_time >= 0, "Should have non-negative processing time"
            
            # Verify entity extraction results
            assert result.entity_result is not None, "Should have entity extraction results"
            assert result.entity_result.success, "Entity extraction should succeed"
            assert len(result.entity_result.entities) == 6, "Should extract 6 entities"
            assert '林黛玉' in result.entity_result.entities, "Should extract 林黛玉"
            assert '賈府' in result.entity_result.entities, "Should extract 賈府"
            
            # Verify triple generation results
            assert result.triple_result is not None, "Should have triple generation results"
            assert result.triple_result.success, "Triple generation should succeed"
            assert len(result.triple_result.triples) == 4, "Should generate 4 triples"
            
            # Verify specific triples
            triple_subjects = [t.subject for t in result.triple_result.triples]
            assert '林黛玉' in triple_subjects, "Should have triples about 林黛玉"
            assert '賈寶玉' in triple_subjects, "Should have triples about 賈寶玉"
            
            # Verify graph judgment results
            assert result.judgment_result is not None, "Should have graph judgment results"
            assert result.judgment_result.success, "Graph judgment should succeed"
            assert len(result.judgment_result.judgments) == 4, "Should judge 4 triples"
            assert all(result.judgment_result.judgments), "All triples should be approved"
            # Note: explanations may be None for binary judgment mode, which is fine
            
            # Verify progress tracking
            assert len(progress_updates) >= 3, "Should have progress updates for each stage"
            stages_reported = [update[0] for update in progress_updates]
            assert 0 in stages_reported, "Should report entity extraction stage"
            assert 1 in stages_reported, "Should report triple generation stage"
            assert 2 in stages_reported, "Should report graph judgment stage"
            
            # Verify statistics
            assert result.stats is not None, "Should have statistics"
            assert result.stats['entity_count'] == 6, "Should track entity count"
            assert result.stats['triple_count'] == 4, "Should track triple count"
            assert result.stats['approval_rate'] == 1.0, "Should have 100% approval rate"
    
    def test_pipeline_partial_failure_recovery(self, sample_chinese_text):
        """Test pipeline behavior with partial failures and recovery."""
        
        # Test entity extraction failure
        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api:
            mock_entity_api.side_effect = Exception("API Error")
            
            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_pipeline(sample_chinese_text)
            
            assert not result.success, "Pipeline should fail when entity extraction fails"
            assert result.stage_reached == 0, "Should fail at entity extraction stage"
            assert result.error_stage == "entity_extraction", "Should identify correct error stage"
            assert result.entity_result is not None, "Should have entity result with error"
            assert not result.entity_result.success, "Entity result should indicate failure"
        
        # Test triple generation failure with successful entity extraction
        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_api_client:
            
            # Successful entity extraction
            mock_entity_api.return_value = f"""
Extracted Entities: 林黛玉, 賈府

Denoised Text:
林黛玉來到賈府。
            """.strip()
            
            # Failed triple generation - mock api client
            mock_client = Mock()
            mock_client.complete.side_effect = Exception("Triple API Error")
            mock_api_client.return_value = mock_client
            
            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_pipeline(sample_chinese_text)
            
            assert not result.success, "Pipeline should fail when triple generation fails"
            assert result.stage_reached == 1, "Should reach triple generation stage"
            assert result.error_stage == "triple_generation", "Should identify correct error stage"
            assert result.entity_result.success, "Entity extraction should still be successful"
            assert result.triple_result is not None, "Should have triple result with error"
            assert not result.triple_result.success, "Triple result should indicate failure"
    
    def test_pipeline_performance_benchmarking(self, sample_chinese_text, mock_successful_api_responses):
        """Test pipeline performance under realistic conditions."""

        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_api_client, \
             patch('streamlit_pipeline.core.pipeline.get_api_client') as mock_pipeline_api_client:
            
            # Configure entity extraction mock
            mock_entity_api.return_value = f"""
Extracted Entities: {', '.join(mock_successful_api_responses['entity_extraction']['entities'])}

Denoised Text:
{mock_successful_api_responses['entity_extraction']['denoised_text']}
            """.strip()
            
            # Configure API client mock for triple generation and graph judgment
            mock_client = Mock()
            mock_api_client.return_value = mock_client
            
            def mock_complete(prompt, **kwargs):
                if 'JSON' in prompt or 'triple' in prompt.lower():
                    return self._create_triple_response(mock_successful_api_responses['triple_generation'])
                else:
                    return self._create_judgment_response(mock_successful_api_responses['graph_judgment'])
            
            # Set up API client methods - call_gpt5_mini returns string directly
            def mock_call_gpt5_mini(prompt, **kwargs):
                """Mock for APIClient.call_gpt5_mini method."""
                if 'JSON' in prompt or 'triple' in prompt.lower() or '任務：分析古典中文文本' in prompt:
                    # Triple generation response - return JSON string directly
                    triples_json = {
                        "triples": [
                            [triple["subject"], triple["predicate"], triple["object"]]
                            for triple in mock_successful_api_responses['triple_generation']
                        ]
                    }
                    return f"```json\n{json.dumps(triples_json)}\n```"
                else:
                    # Graph judgment response - return text directly
                    judgment_parts = []
                    for i, (judgment, explanation) in enumerate(
                        zip(mock_successful_api_responses['graph_judgment']['judgments'],
                            mock_successful_api_responses['graph_judgment']['explanations'])
                    ):
                        result = "Yes" if judgment else "No"
                        judgment_parts.append(f"Triple {i+1}: {result} - {explanation}")

                    return "\n".join(judgment_parts)

            mock_client.call_gpt5_mini = mock_call_gpt5_mini
            mock_client.complete.side_effect = mock_complete

            # Also set up the pipeline API client mock
            mock_pipeline_api_client.return_value = mock_client

            # Run performance test
            orchestrator = PipelineOrchestrator()
            start_time = time.time()
            result = orchestrator.run_pipeline(sample_chinese_text)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Performance assertions
            assert result.success, "Performance test should complete successfully"
            assert total_time < 30.0, f"Pipeline should complete in under 30 seconds, took {total_time:.2f}s"
            assert result.total_time >= 0, "Should track processing time"
            
            # Verify stage timing breakdown
            assert result.entity_result.processing_time >= 0, "Entity processing time should be tracked"
            assert result.triple_result.processing_time >= 0, "Triple processing time should be tracked"
            assert result.judgment_result.processing_time >= 0, "Judgment processing time should be tracked"
            
            # Performance metrics
            total_stage_time = (
                result.entity_result.processing_time +
                result.triple_result.processing_time +
                result.judgment_result.processing_time
            )
            
            # Allow for some overhead in total time vs sum of stage times
            assert total_stage_time <= result.total_time * 1.1, "Stage times should be reasonable vs total time"
    
    def test_pipeline_session_state_integration(self, sample_chinese_text, mock_successful_api_responses):
        """Test pipeline integration with session state management."""

        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_api_client, \
             patch('streamlit_pipeline.core.pipeline.get_api_client') as mock_pipeline_api_client:

            # Configure mocks
            self._setup_successful_mocks_unified(
                mock_entity_api, mock_api_client, mock_successful_api_responses
            )

            # Also set up the pipeline API client mock
            mock_pipeline_api_client.return_value = mock_api_client.return_value
            
            # Get session manager
            session_manager = get_session_manager()
            initial_run_count = session_manager.get_session_metadata().run_count
            
            # Execute pipeline
            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_pipeline(sample_chinese_text)
            
            # Verify session state updates
            assert result.success, "Pipeline should succeed for session state test"
            
            # Store result in session manager
            session_manager.set_current_result(result)
            
            # Verify session state
            current_result = session_manager.get_current_result()
            assert current_result is not None, "Should store current result"
            assert current_result.success, "Stored result should be successful"
            
            # Verify metadata updates
            metadata = session_manager.get_session_metadata()
            assert metadata.run_count > initial_run_count, "Run count should increase"
            assert metadata.successful_runs > 0, "Should track successful runs"
            
            # Verify pipeline results history
            pipeline_results = session_manager.get_pipeline_results()
            assert len(pipeline_results) > 0, "Should store pipeline results"
            assert pipeline_results[-1].success, "Latest result should be successful"
    
    def test_pipeline_error_handling_and_recovery(self, sample_chinese_text):
        """Test comprehensive error handling and recovery scenarios."""
        
        error_handler = ErrorHandler()
        
        # Test invalid input handling
        orchestrator = PipelineOrchestrator()
        
        # Empty input
        result = orchestrator.run_pipeline("")
        assert not result.success, "Should fail with empty input"
        assert result.error_stage == "input_validation", "Should identify input validation error"
        
        # Whitespace only input
        result = orchestrator.run_pipeline("   \n\t   ")
        assert not result.success, "Should fail with whitespace-only input"
        assert result.error_stage == "input_validation", "Should identify input validation error"
        
        # Test API error scenarios
        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_api:
            # Simulate different API error types
            error_scenarios = [
                (Exception("Rate limit exceeded"), "Rate limit"),
                (Exception("Authentication failed"), "Authentication"),
                (Exception("Server error"), "Server error"),
                (Exception("Timeout"), "Timeout")
            ]
            
            for error, description in error_scenarios:
                mock_client = Mock()
                mock_client.complete.side_effect = error
                mock_api.return_value = mock_client
                
                result = orchestrator.run_pipeline(sample_chinese_text)
                assert not result.success, f"Should fail for {description} error"
                assert result.error is not None, f"Should have error message for {description}"
                assert result.stage_reached == 0, f"Should fail at stage 0 for {description}"
    
    def test_pipeline_data_flow_integrity(self, sample_chinese_text, mock_successful_api_responses):
        """Test data integrity and consistency across pipeline stages."""

        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_api_client, \
             patch('streamlit_pipeline.core.pipeline.get_api_client') as mock_pipeline_api_client:

            # Configure mocks
            self._setup_successful_mocks_unified(
                mock_entity_api, mock_api_client, mock_successful_api_responses
            )

            # Also set up the pipeline API client mock
            mock_pipeline_api_client.return_value = mock_api_client.return_value
            
            # Execute pipeline
            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_pipeline(sample_chinese_text)
            
            assert result.success, "Pipeline should succeed for data integrity test"
            
            # Verify data consistency between stages
            entities = result.entity_result.entities
            triples = result.triple_result.triples
            judgments = result.judgment_result.judgments
            
            # All triple subjects/objects should reference extracted entities
            all_triple_entities = set()
            for triple in triples:
                all_triple_entities.add(triple.subject)
                all_triple_entities.add(triple.object)
            
            extracted_entities = set(entities)
            
            # Most triple entities should be from extracted entities (some may be implicit)
            entity_overlap = all_triple_entities.intersection(extracted_entities)
            overlap_ratio = len(entity_overlap) / len(all_triple_entities) if all_triple_entities else 0
            assert overlap_ratio >= 0.5, f"At least 50% of triple entities should be extracted entities, got {overlap_ratio:.2%}"
            
            # Number of judgments should match number of triples
            assert len(judgments) == len(triples), "Should have one judgment per triple"
            
            # Verify data types and constraints
            for triple in triples:
                assert isinstance(triple.subject, str) and triple.subject.strip(), "Triple subjects should be non-empty strings"
                assert isinstance(triple.predicate, str) and triple.predicate.strip(), "Triple predicates should be non-empty strings"
                assert isinstance(triple.object, str) and triple.object.strip(), "Triple objects should be non-empty strings"
    
    # Helper methods
    
    def _setup_successful_mocks_unified(self, mock_entity_api, mock_api_client, responses):
        """Helper to set up successful mock API responses with unified approach."""
        # Entity extraction mock - need to handle both entity extraction and text denoising calls
        call_count = 0
        original_entities = responses['entity_extraction']['entities']
        original_denoised = responses['entity_extraction']['denoised_text']
        
        def mock_entity_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call is entity extraction
                return str(original_entities)  # Return list as string
            else:  # Second call is text denoising
                return original_denoised
        
        mock_entity_api.side_effect = mock_entity_response
        
        # API client mock for triple generation and graph judgment
        mock_client = Mock()
        mock_api_client.return_value = mock_client
        
        def mock_complete(prompt, **kwargs):
            if 'JSON' in prompt or 'triple' in prompt.lower():
                # Triple generation response
                triples_json = {
                    "triples": [
                        {
                            "subject": triple["subject"],
                            "predicate": triple["predicate"],
                            "object": triple["object"]
                        }
                        for triple in responses['triple_generation']
                    ]
                }
                response = Mock()
                response.choices = [Mock()]
                response.choices[0].message = Mock()
                response.choices[0].message.content = f"```json\n{json.dumps(triples_json)}\n```"
                return response
            else:
                # Graph judgment response
                judgment_parts = []
                for i, (judgment, confidence, explanation) in enumerate(
                    zip(responses['graph_judgment']['judgments'], 
                        responses['graph_judgment']['confidence'], 
                        responses['graph_judgment']['explanations'])
                ):
                    result = "ACCEPT" if judgment else "REJECT"
                    judgment_parts.append(f"Triple {i+1}: {result} (Confidence: {confidence:.2f}) - {explanation}")
                
                response = Mock()
                response.choices = [Mock()]
                response.choices[0].message = Mock()
                response.choices[0].message.content = "\n".join(judgment_parts)
                return response

        # Set up API client methods - call_gpt5_mini returns string directly
        def mock_call_gpt5_mini(prompt, **kwargs):
            """Mock for APIClient.call_gpt5_mini method."""
            if 'JSON' in prompt or 'triple' in prompt.lower() or '任務：分析古典中文文本' in prompt:
                # Triple generation response - return JSON string directly
                triples_json = {
                    "triples": [
                        [triple["subject"], triple["predicate"], triple["object"]]
                        for triple in responses['triple_generation']
                    ]
                }
                return f"```json\n{json.dumps(triples_json)}\n```"
            else:
                # Graph judgment response - return text directly
                judgment_parts = []
                for i, (judgment, explanation) in enumerate(
                    zip(responses['graph_judgment']['judgments'],
                        responses['graph_judgment']['explanations'])
                ):
                    result = "ACCEPT" if judgment else "REJECT"
                    judgment_parts.append(f"Triple {i+1}: {result} - {explanation}")

                return "\n".join(judgment_parts)

        mock_client.call_gpt5_mini = mock_call_gpt5_mini
        mock_client.complete.side_effect = mock_complete
    
    def _create_entity_response(self, data: Dict[str, Any]) -> Mock:
        """Create mock entity extraction API response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = f"""
Extracted Entities: {', '.join(data['entities'])}

Denoised Text:
{data['denoised_text']}
        """.strip()
        return response
    
    def _create_triple_response(self, data: List[Dict[str, Any]]) -> Mock:
        """Create mock triple generation API response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        
        # Create JSON response format
        triples_json = {
            "triples": [
                {
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"],
                    "confidence": triple.get("confidence", 0.8)
                }
                for triple in data
            ]
        }
        
        response.choices[0].message.content = f"```json\n{json.dumps(triples_json)}\n```"
        return response
    
    def _create_judgment_response(self, data: Dict[str, Any]) -> Mock:
        """Create mock graph judgment API response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        
        # Create judgment response
        judgment_parts = []
        for i, (judgment, explanation) in enumerate(
            zip(data['judgments'], data['explanations'])
        ):
            result = "ACCEPT" if judgment else "REJECT"
            judgment_parts.append(f"Triple {i+1}: {result} - {explanation}")
        
        response.choices[0].message.content = "\n".join(judgment_parts)
        return response


@pytest.mark.integration
@pytest.mark.performance
class TestPipelinePerformance:
    """Performance-focused integration tests."""
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_concurrent_pipeline_execution(self, sample_chinese_text, mock_successful_api_responses):
        """Test pipeline performance under concurrent load."""

        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.utils.api_client.call_gpt5_mini') as mock_entity_api_utils, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_api_client, \
             patch('streamlit_pipeline.core.pipeline.get_api_client') as mock_pipeline_api_client, \
             patch('streamlit_pipeline.core.triple_generator.call_gpt5_mini') as mock_triple_api, \
             patch('streamlit_pipeline.utils.api_client.get_api_client') as mock_utils_api_client:

            # Set up mocks
            self._setup_performance_mocks_unified(mock_entity_api, mock_api_client, mock_successful_api_responses)

            # Set up the utils entity API mock with the same behavior
            mock_entity_api_utils.side_effect = mock_entity_api.side_effect

            # Also set up the pipeline API client mock
            mock_pipeline_api_client.return_value = mock_api_client.return_value
            mock_utils_api_client.return_value = mock_api_client.return_value

            # Set up triple generator API mock with proper JSON format
            mock_triple_api.return_value = '''{
                "triples": [
                    ["subject1", "predicate1", "object1"],
                    ["subject2", "predicate2", "object2"]
                ]
            }'''
            
            # Run multiple pipelines concurrently
            import concurrent.futures
            
            def run_single_pipeline():
                orchestrator = PipelineOrchestrator()
                return orchestrator.run_pipeline(sample_chinese_text)
            
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_single_pipeline) for _ in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all pipelines succeeded
            assert all(result.success for result in results), "All concurrent pipelines should succeed"
            assert len(results) == 3, "Should have 3 results"
            
            # Performance check - concurrent execution should not take 3x as long
            assert total_time < 45.0, f"Concurrent execution should complete within 45 seconds, took {total_time:.2f}s"
    
    def test_large_text_processing_performance(self, mock_successful_api_responses):
        """Test pipeline performance with large text inputs."""

        # Create large text input (5000+ characters)
        large_text = """
        林黛玉是賈府的親戚，從江南來到榮國府。她聰明伶俐，才情出眾，深受賈母喜愛。
        賈寶玉初見林黛玉時，覺得這個妹妹似曾相識。二人因緣分深厚，漸生情愫。
        賈府是金陵四大家族之一，門第顯赫。府中人物眾多，關係復雜。
        """ * 50  # Repeat 50 times to create large text

        with patch('streamlit_pipeline.core.entity_processor.call_gpt5_mini') as mock_entity_api, \
             patch('streamlit_pipeline.core.graph_judge.get_api_client') as mock_api_client, \
             patch('streamlit_pipeline.core.pipeline.get_api_client') as mock_pipeline_api_client:
            
            # Set up mocks for large text processing
            self._setup_performance_mocks_unified(mock_entity_api, mock_api_client, mock_successful_api_responses)

            # Also set up the pipeline API client mock
            mock_pipeline_api_client.return_value = mock_api_client.return_value
            
            orchestrator = PipelineOrchestrator()
            start_time = time.time()
            result = orchestrator.run_pipeline(large_text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify successful processing
            assert result.success, "Large text processing should succeed"
            
            # Performance assertions
            assert processing_time < 30.0, f"Large text should process within 30 seconds, took {processing_time:.2f}s"
            
            # Verify text chunking was handled
            assert result.triple_result.metadata.get('text_processing', {}).get('chunks_created', 0) > 1, \
                "Large text should be chunked"
    
    def _setup_performance_mocks_unified(self, mock_entity_api, mock_api_client, responses):
        """Set up mocks with realistic performance characteristics using unified approach."""
        # Entity extraction mock - need to handle both entity extraction and text denoising calls
        call_count = 0
        original_entities = responses['entity_extraction']['entities']
        original_denoised = responses['entity_extraction']['denoised_text']

        def mock_entity_response(prompt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call is entity extraction
                return str(original_entities)  # Return list as string
            else:  # Second call is text denoising
                # For large text tests, return text proportional to input size
                if prompt and len(prompt) > 1000:
                    # Extract the actual text content from the prompt and return meaningful denoised text
                    # For large inputs, repeat the base denoised text to create larger output
                    # Need at least 2000 characters to trigger chunking (1000 tokens * 2 chars/token)
                    multiplier = max(50, len(prompt) // 100)  # Ensure at least 50x for chunking
                    large_denoised = (original_denoised + ' ') * multiplier
                    return large_denoised.strip()
                return original_denoised
        
        mock_entity_api.side_effect = mock_entity_response
        
        # API client mock for triple generation and graph judgment
        mock_client = Mock()
        mock_api_client.return_value = mock_client
        
        def mock_complete(prompt, **kwargs):
            if 'JSON' in prompt or 'triple' in prompt.lower():
                # Triple generation response
                triples_json = {
                    "triples": [
                        {
                            "subject": triple["subject"],
                            "predicate": triple["predicate"],
                            "object": triple["object"]
                        }
                        for triple in responses['triple_generation']
                    ]
                }
                response = Mock()
                response.choices = [Mock()]
                response.choices[0].message = Mock()
                response.choices[0].message.content = f"```json\n{json.dumps(triples_json)}\n```"
                return response
            else:
                # Graph judgment response
                judgment_parts = []
                for i, (judgment, confidence, explanation) in enumerate(
                    zip(responses['graph_judgment']['judgments'], 
                        responses['graph_judgment']['confidence'], 
                        responses['graph_judgment']['explanations'])
                ):
                    result = "ACCEPT" if judgment else "REJECT"
                    judgment_parts.append(f"Triple {i+1}: {result} (Confidence: {confidence:.2f}) - {explanation}")
                
                response = Mock()
                response.choices = [Mock()]
                response.choices[0].message = Mock()
                response.choices[0].message.content = "\n".join(judgment_parts)
                return response

        # Set up API client methods - call_gpt5_mini returns string directly
        def mock_call_gpt5_mini(prompt, **kwargs):
            """Mock for APIClient.call_gpt5_mini method."""
            if 'JSON' in prompt or 'triple' in prompt.lower() or '任務：分析古典中文文本' in prompt:
                # Triple generation response - return JSON string directly
                triples_json = {
                    "triples": [
                        [triple["subject"], triple["predicate"], triple["object"]]
                        for triple in responses['triple_generation']
                    ]
                }
                return f"```json\n{json.dumps(triples_json)}\n```"
            else:
                # Graph judgment response - return text directly
                judgment_parts = []
                for i, (judgment, explanation) in enumerate(
                    zip(responses['graph_judgment']['judgments'],
                        responses['graph_judgment']['explanations'])
                ):
                    result = "ACCEPT" if judgment else "REJECT"
                    judgment_parts.append(f"Triple {i+1}: {result} - {explanation}")

                return "\n".join(judgment_parts)

        mock_client.call_gpt5_mini = mock_call_gpt5_mini
        mock_client.complete.side_effect = mock_complete

    def _create_entity_response(self, data):
        """Create mock entity response for performance tests."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = f"""
Extracted Entities: {', '.join(data['entities'])}

Denoised Text:
{data['denoised_text']}
        """.strip()
        return response
    
    def _create_triple_response(self, data):
        """Create mock triple response for performance tests."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        
        triples_json = {
            "triples": [
                {
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"],
                    "confidence": triple.get("confidence", 0.8)
                }
                for triple in data
            ]
        }
        
        response.choices[0].message.content = f"```json\n{json.dumps(triples_json)}\n```"
        return response
    
    def _create_judgment_response(self, data):
        """Create mock judgment response for performance tests."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        
        judgment_parts = []
        for i, (judgment, explanation) in enumerate(
            zip(data['judgments'], data['explanations'])
        ):
            result = "ACCEPT" if judgment else "REJECT"
            judgment_parts.append(f"Triple {i+1}: {result} - {explanation}")
        
        response.choices[0].message.content = "\n".join(judgment_parts)
        return response