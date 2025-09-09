"""
Unit tests for the graph_judge_core module.

Tests the core PerplexityGraphJudge class functionality,
including initialization, graph judgment, explainable judgment,
streaming, and citation handling.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

from ..graph_judge_core import PerplexityGraphJudge
from ..data_structures import ExplainableJudgment
from .conftest import PerplexityTestBase, MockPerplexityResponse


class TestPerplexityGraphJudgeInitialization(PerplexityTestBase):
    """Test PerplexityGraphJudge initialization."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        judge = PerplexityGraphJudge()
        
        assert judge.model_name == "perplexity/sonar-reasoning"
        assert judge.reasoning_effort == "medium"
        assert judge.enable_logging == False
        assert judge.temperature == 0.2
        assert judge.max_tokens == 2000
        assert isinstance(judge.is_mock, bool)
    
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        judge = PerplexityGraphJudge(
            model_name="perplexity/sonar-pro",
            reasoning_effort="high",
            enable_console_logging=True
        )
        
        assert judge.model_name == "perplexity/sonar-pro"
        assert judge.reasoning_effort == "high"
        assert judge.enable_logging == True
    
    def test_initialization_without_api_key(self):
        """Test initialization failure without API key."""
        # Remove API key from environment
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="PERPLEXITYAI_API_KEY not found"):
                PerplexityGraphJudge()
    
    def test_initialization_with_mock_mode(self):
        """Test initialization in mock mode."""
        with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', False):
            judge = PerplexityGraphJudge()
            assert judge.is_mock == True
    
    def test_initialization_check_perplexity_availability(self):
        """Test _check_perplexity_availability method."""
        judge = PerplexityGraphJudge()
        
        # Test availability check
        with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
            availability = judge._check_perplexity_availability()
            assert availability == True
        
        with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', False):
            availability = judge._check_perplexity_availability()
            assert availability == False


class TestBasicGraphJudgment(PerplexityTestBase):
    """Test basic graph judgment functionality."""
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_mock_mode(self):
        """Test graph triple judgment in mock mode."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True  # Ensure mock mode
        
        # Test known positive cases
        result1 = await judge.judge_graph_triple("Is this true: Apple Founded by Steve Jobs ?")
        assert result1 == "Yes"
        
        result2 = await judge.judge_graph_triple("Is this true: 曹雪芹 創作 紅樓夢 ?")
        assert result2 == "Yes"
        
        # Test known negative case
        result3 = await judge.judge_graph_triple("Is this true: Microsoft Founded by Mark Zuckerberg ?")
        assert result3 == "No"
        
        # Test default case
        result4 = await judge.judge_graph_triple("Is this true: Unknown Statement ?")
        assert result4 == "Yes"  # Default in mock mode
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_input_text(self):
        """Test graph triple judgment with additional input text."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        result = await judge.judge_graph_triple(
            "Is this true: Apple Founded by Steve Jobs ?",
            input_text="Additional context about Apple's founding."
        )
        assert result in ["Yes", "No"]
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_non_mock_mode(self):
        """Test graph triple judgment in non-mock mode with mocked API."""
        mock_response = MockPerplexityResponse(answer="Yes, this is correct.")
        
        with patch('graphJudge_Phase.graph_judge_core.acompletion', return_value=mock_response):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                judge = PerplexityGraphJudge()
                judge.is_mock = False  # Force non-mock mode
                
                result = await judge.judge_graph_triple("Is this true: Apple Founded by Steve Jobs ?")
                assert result == "Yes"
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_retry(self):
        """Test graph triple judgment with retry logic."""
        mock_response = MockPerplexityResponse(answer="No, this is incorrect.")
        
        with patch('graphJudge_Phase.graph_judge_core.acompletion', side_effect=[Exception("API Error"), mock_response]):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                with patch('asyncio.sleep'):  # Speed up test
                    judge = PerplexityGraphJudge()
                    judge.is_mock = False
                    
                    result = await judge.judge_graph_triple("Is this true: Test Statement ?")
                    assert result == "No"
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_max_retries_exceeded(self):
        """Test graph triple judgment when max retries are exceeded."""
        with patch('graphJudge_Phase.graph_judge_core.acompletion', side_effect=Exception("API Error")):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                with patch('asyncio.sleep'):  # Speed up test
                    judge = PerplexityGraphJudge()
                    judge.is_mock = False
                    
                    result = await judge.judge_graph_triple("Is this true: Test Statement ?")
                    assert result == "No"  # Conservative default


class TestExplainableJudgment(PerplexityTestBase):
    """Test explainable graph judgment functionality."""
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_explanation_mock_mode(self):
        """Test explainable graph triple judgment in mock mode."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        # Test positive case
        result1 = await judge.judge_graph_triple_with_explanation("Is this true: 曹雪芹 創作 紅樓夢 ?")
        
        assert isinstance(result1, ExplainableJudgment)
        assert result1.judgment == "Yes"
        assert 0.0 <= result1.confidence <= 1.0
        assert isinstance(result1.reasoning, str)
        assert len(result1.reasoning) > 0
        assert isinstance(result1.evidence_sources, list)
        assert isinstance(result1.alternative_suggestions, list)
        assert result1.error_type is None
        assert isinstance(result1.processing_time, float)
        
        # Test negative case
        result2 = await judge.judge_graph_triple_with_explanation("Is this true: Microsoft Founded by Mark Zuckerberg ?")
        
        assert result2.judgment == "No"
        assert result2.confidence > 0.0
        assert "Microsoft" in result2.reasoning
        assert len(result2.alternative_suggestions) > 0
        assert result2.error_type == "factual_error"
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_explanation_with_citations(self):
        """Test explainable judgment with citation integration."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        result = await judge.judge_graph_triple_with_explanation(
            "Is this true: Apple Founded by Steve Jobs ?",
            include_citations=True
        )
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment in ["Yes", "No"]
        assert isinstance(result.evidence_sources, list)
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_explanation_without_citations(self):
        """Test explainable judgment without citation integration."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        result = await judge.judge_graph_triple_with_explanation(
            "Is this true: Apple Founded by Steve Jobs ?",
            include_citations=False
        )
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment in ["Yes", "No"]
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_explanation_error_handling(self):
        """Test explainable judgment error handling."""
        with patch('graphJudge_Phase.graph_judge_core.acompletion', side_effect=Exception("API Error")):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                with patch('asyncio.sleep'):
                    judge = PerplexityGraphJudge()
                    judge.is_mock = False
                    
                    result = await judge.judge_graph_triple_with_explanation("Is this true: Test ?")
                    
                    assert isinstance(result, ExplainableJudgment)
                    assert result.judgment == "No"
                    assert result.confidence == 0.0
                    assert "Error during processing" in result.reasoning
                    assert result.error_type == "processing_error"


class TestStreamingJudgment(PerplexityTestBase):
    """Test streaming graph judgment functionality."""
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_streaming_mock_mode(self):
        """Test streaming graph triple judgment in mock mode."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        result = await judge.judge_graph_triple_streaming("Is this true: 曹雪芹 創作 紅樓夢 ?")
        assert result in ["Yes", "No"]
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_streaming_with_container(self):
        """Test streaming judgment with stream container."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        mock_container = MagicMock()
        result = await judge.judge_graph_triple_streaming(
            "Is this true: Apple Founded by Steve Jobs ?",
            stream_container=mock_container
        )
        
        assert result in ["Yes", "No"]
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_streaming_non_mock_mode(self):
        """Test streaming judgment in non-mock mode with mocked API."""
        # Create a mock async generator that can be properly iterated
        class MockAsyncGenerator:
            def __init__(self, response_data):
                self.response_data = response_data
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if hasattr(self, '_yielded'):
                    raise StopAsyncIteration
                self._yielded = True
                return self.response_data
        
        mock_response = MockAsyncGenerator(MockPerplexityResponse(answer="Yes"))
        
        with patch('graphJudge_Phase.graph_judge_core.acompletion', return_value=mock_response):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                judge = PerplexityGraphJudge()
                judge.is_mock = False
                
                result = await judge.judge_graph_triple_streaming("Is this true: Test ?")
                assert result in ["Yes", "No"]
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_streaming_error_handling(self):
        """Test streaming judgment error handling."""
        with patch('graphJudge_Phase.graph_judge_core.acompletion', side_effect=Exception("Stream Error")):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                judge = PerplexityGraphJudge()
                judge.is_mock = False
                
                result = await judge.judge_graph_triple_streaming("Is this true: Test ?")
                assert result == "No"  # Conservative default


class TestCitationHandling(PerplexityTestBase):
    """Test citation handling functionality."""
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_citations_mock_mode(self):
        """Test graph judgment with citations in mock mode."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        result = await judge.judge_graph_triple_with_citations("Is this true: 曹雪芹 創作 紅樓夢 ?")
        
        assert isinstance(result, dict)
        assert "judgment" in result
        assert "confidence" in result
        assert "citations" in result
        assert "citation_count" in result
        assert "processing_time" in result
        
        assert result["judgment"] in ["Yes", "No"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["citations"], list)
        assert isinstance(result["citation_count"], int)
        assert isinstance(result["processing_time"], float)
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_citations_non_mock_mode(self):
        """Test graph judgment with citations in non-mock mode."""
        mock_response = MockPerplexityResponse(
            answer="Yes, this is correct.",
            citations=["https://example.com/source1", "https://example.com/source2"]
        )
        
        with patch('graphJudge_Phase.graph_judge_core.acompletion', return_value=mock_response):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                judge = PerplexityGraphJudge()
                judge.is_mock = False
                
                result = await judge.judge_graph_triple_with_citations("Is this true: Test ?")
                
                assert result["judgment"] == "Yes"
                assert result["citation_count"] >= 0
                assert isinstance(result["citations"], list)
    
    @pytest.mark.asyncio
    async def test_judge_graph_triple_with_citations_error_handling(self):
        """Test citation judgment error handling."""
        with patch('graphJudge_Phase.graph_judge_core.acompletion', side_effect=Exception("API Error")):
            with patch('graphJudge_Phase.graph_judge_core.LITELLM_AVAILABLE', True):
                with patch('asyncio.sleep'):
                    judge = PerplexityGraphJudge()
                    judge.is_mock = False
                    
                    result = await judge.judge_graph_triple_with_citations("Is this true: Test ?")
                    
                    assert result["judgment"] == "No"
                    assert result["confidence"] == 0.0
                    assert result["citation_count"] == 0
                    assert "error" in result


class TestMockMethods(PerplexityTestBase):
    """Test mock methods for testing scenarios."""
    
    def test_mock_judgment_logic(self):
        """Test _mock_judgment logic."""
        judge = PerplexityGraphJudge()
        
        # Test known positive cases
        result1 = judge._mock_judgment("Is this true: Apple Founded by Steve Jobs ?")
        assert result1 == "Yes"
        
        result2 = judge._mock_judgment("Is this true: 曹雪芹 創作 紅樓夢 ?")
        assert result2 == "Yes"
        
        # Test known negative case
        result3 = judge._mock_judgment("Is this true: Microsoft Founded by Mark Zuckerberg ?")
        assert result3 == "No"
        
        # Test default case
        result4 = judge._mock_judgment("Is this true: Unknown Statement ?")
        assert result4 == "Yes"
    
    def test_mock_explainable_judgment_positive(self):
        """Test _mock_explainable_judgment for positive cases."""
        judge = PerplexityGraphJudge()
        start_time = time.time()
        
        result = judge._mock_explainable_judgment("Is this true: Apple Founded by Steve Jobs ?", start_time)
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment == "Yes"
        assert result.confidence == 0.95
        assert "歷史事實" in result.reasoning
        assert len(result.alternative_suggestions) == 0
        assert result.error_type is None
    
    def test_mock_explainable_judgment_negative(self):
        """Test _mock_explainable_judgment for negative cases."""
        judge = PerplexityGraphJudge()
        start_time = time.time()
        
        result = judge._mock_explainable_judgment("Is this true: Microsoft Founded by Mark Zuckerberg ?", start_time)
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment == "No"
        assert result.confidence == 0.90
        assert "錯誤" in result.reasoning
        assert len(result.alternative_suggestions) == 2
        assert result.error_type == "factual_error"
    
    def test_mock_explainable_judgment_default(self):
        """Test _mock_explainable_judgment for default cases."""
        judge = PerplexityGraphJudge()
        start_time = time.time()
        
        result = judge._mock_explainable_judgment("Is this true: Unknown Statement ?", start_time)
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment == "Yes"
        assert result.confidence == 0.75
        assert "合理" in result.reasoning
        assert len(result.alternative_suggestions) == 0
        assert result.error_type is None


class TestGraphJudgeCoreIntegration(PerplexityTestBase):
    """Test integration scenarios for graph judge core."""
    
    @pytest.mark.asyncio
    async def test_multiple_judgments_consistency(self):
        """Test consistency across multiple judgments."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        instruction = "Is this true: 曹雪芹 創作 紅樓夢 ?"
        
        # Run multiple judgments
        results = []
        for _ in range(3):
            result = await judge.judge_graph_triple(instruction)
            results.append(result)
        
        # All results should be consistent in mock mode
        assert all(result == results[0] for result in results)
    
    @pytest.mark.asyncio
    async def test_different_judgment_modes_compatibility(self):
        """Test compatibility between different judgment modes."""
        judge = PerplexityGraphJudge()
        judge.is_mock = True
        
        instruction = "Is this true: Apple Founded by Steve Jobs ?"
        
        # Get judgments from different modes
        basic_result = await judge.judge_graph_triple(instruction)
        explainable_result = await judge.judge_graph_triple_with_explanation(instruction)
        streaming_result = await judge.judge_graph_triple_streaming(instruction)
        citation_result = await judge.judge_graph_triple_with_citations(instruction)
        
        # Basic and explainable should agree on binary judgment
        assert basic_result == explainable_result.judgment
        
        # Streaming should also agree
        assert basic_result == streaming_result
        
        # Citation result should also agree
        assert basic_result == citation_result["judgment"]
    
    def test_judge_configuration_consistency(self):
        """Test that judge configuration is maintained consistently."""
        judge = PerplexityGraphJudge(
            model_name="perplexity/sonar-pro",
            reasoning_effort="high",
            enable_console_logging=True
        )
        
        # Configuration should be preserved
        assert judge.model_name == "perplexity/sonar-pro"
        assert judge.reasoning_effort == "high"
        assert judge.enable_logging == True
        
        # Other defaults should be maintained
        assert judge.temperature == 0.2
        assert judge.max_tokens == 2000
