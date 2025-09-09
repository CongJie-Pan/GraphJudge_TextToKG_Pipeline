"""
Common test fixtures and utilities for GraphJudge Phase tests.

This module provides shared fixtures, mock classes, and testing utilities
that are used across multiple test modules in the GraphJudge Phase test suite.
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock
from typing import List, Dict, Any, NamedTuple


class MockPerplexityResponse:
    """Mock Perplexity API response for testing."""
    
    def __init__(self, answer="Test answer", citations=None, response_time=1.0):
        # Ensure answer is never None
        self.answer = answer if answer is not None else "Test answer"
        self.citations = citations or []
        self.response_time = response_time
        # Perplexity API response structure
        self.choices = [type('MockChoice', (), {
            'message': type('MockMessage', (), {'content': self.answer})()
        })()]


class MockCitationInfo:
    """Mock citation information for testing."""
    
    def __init__(self, text_segment="", start_index=0, end_index=0, source_urls=None, source_titles=None):
        self.text_segment = text_segment
        self.start_index = start_index
        self.end_index = end_index
        self.source_urls = source_urls or []
        self.source_titles = source_titles or []


class PerplexityTestBase:
    """Base class for Perplexity API tests with proper setup and teardown."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_file = os.path.join(self.temp_dir, "test_input.json")
        self.test_output_file = os.path.join(self.temp_dir, "test_output.csv")
        
        # Setup environment variables for testing
        self.test_iteration = "3"
        os.environ['PIPELINE_ITERATION'] = self.test_iteration
        os.environ['PIPELINE_INPUT_FILE'] = self.test_input_file
        os.environ['PIPELINE_OUTPUT_FILE'] = self.test_output_file
        # Add Perplexity API key for testing
        os.environ['PERPLEXITYAI_API_KEY'] = 'test_api_key'
        
        # Sample test data for graph judgment
        self.sample_instructions = [
            {
                "instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?",
                "input": "",
                "output": ""
            },
            {
                "instruction": "Is this true: Apple Founded by Steve Jobs ?",
                "input": "",
                "output": ""
            },
            {
                "instruction": "Is this true: 賈寶玉 創作 紅樓夢 ?",
                "input": "",
                "output": ""
            }
        ]
        
        # Write sample data to test file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_instructions, f)
    
    def teardown_method(self):
        """Tear down method called after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temporary files
        shutil.rmtree(self.temp_dir, ignore_errors=True)


@pytest.fixture
def mock_perplexity_response():
    """Pytest fixture for mock Perplexity response."""
    return MockPerplexityResponse(
        answer="Yes, this is correct.",
        citations=["https://example.com/source1", "https://example.com/source2"]
    )


@pytest.fixture
def sample_test_data():
    """Pytest fixture for sample test data."""
    return [
        {
            "instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?",
            "input": "",
            "output": ""
        },
        {
            "instruction": "Is this true: Apple Founded by Steve Jobs ?",
            "input": "",
            "output": ""
        },
        {
            "instruction": "Is this true: Microsoft Founded by Mark Zuckerberg ?",
            "input": "",
            "output": ""
        }
    ]


@pytest.fixture
def temp_directory():
    """Pytest fixture for temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_triple_data():
    """Pytest fixture for mock triple data."""
    from ..data_structures import TripleData
    return [
        TripleData("賈寶玉", "喜歡", "林黛玉", "test line 1", 1),
        TripleData("曹雪芹", "創作", "紅樓夢", "test line 2", 2),
        TripleData("女媧", "補石", "補天", "test line 3", 3)
    ]


@pytest.fixture
def mock_source_lines():
    """Pytest fixture for mock source lines."""
    return [
        "賈寶玉深深地喜歡林黛玉，這是大家都知道的事實。",
        "曹雪芹花費多年創作紅樓夢這部大作。",
        "女媧氏煉五色石以補蒼天的神話傳說廣為流傳。",
        "這是一個沒有相關的句子。"
    ]


@pytest.fixture
def mock_async_perplexity_judge():
    """Pytest fixture for mock async Perplexity judge."""
    mock_judge = MagicMock()
    mock_judge.is_mock = True
    mock_judge.model_name = "perplexity/sonar-reasoning"
    mock_judge.enable_logging = False
    
    # Mock async methods
    async def mock_judge_triple(instruction, input_text=None):
        if "Apple Founded by Steve Jobs" in instruction or "曹雪芹 創作 紅樓夢" in instruction:
            return "Yes"
        elif "Microsoft Founded by Mark Zuckerberg" in instruction:
            return "No"
        else:
            return "Yes"
    
    mock_judge.judge_graph_triple = AsyncMock(side_effect=mock_judge_triple)
    
    return mock_judge


@pytest.fixture
def mock_explainable_judgment():
    """Pytest fixture for mock explainable judgment."""
    from ..data_structures import ExplainableJudgment
    return ExplainableJudgment(
        judgment="Yes",
        confidence=0.95,
        reasoning="這是一個已知的歷史事實，有充分的文獻證據支持。",
        evidence_sources=["domain_knowledge", "historical_records"],
        alternative_suggestions=[],
        error_type=None,
        processing_time=1.2
    )


def create_mock_dataset(data: List[Dict[str, str]]):
    """Create a mock dataset for testing."""
    class MockDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
    
    return MockDataset(data)


def create_test_files(temp_dir: str, sample_data: List[Dict[str, str]]):
    """Create test input and output files."""
    input_file = os.path.join(temp_dir, "test_input.json")
    output_file = os.path.join(temp_dir, "test_output.csv")
    
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    return input_file, output_file


def assert_valid_binary_response(response: str):
    """Assert that a response is a valid binary judgment."""
    assert response in ["Yes", "No"], f"Invalid binary response: {response}"


def assert_valid_explainable_judgment(judgment):
    """Assert that a judgment is a valid explainable judgment."""
    from ..data_structures import ExplainableJudgment
    
    assert isinstance(judgment, ExplainableJudgment)
    assert judgment.judgment in ["Yes", "No"]
    assert 0.0 <= judgment.confidence <= 1.0
    assert isinstance(judgment.reasoning, str)
    assert len(judgment.reasoning) > 0
    assert isinstance(judgment.evidence_sources, list)
    assert isinstance(judgment.alternative_suggestions, list)
    assert isinstance(judgment.processing_time, float)
    assert judgment.processing_time >= 0.0


def assert_valid_triple_data(triple):
    """Assert that triple data is valid."""
    from ..data_structures import TripleData
    
    assert isinstance(triple, TripleData)
    assert isinstance(triple.subject, str)
    assert isinstance(triple.predicate, str)
    assert isinstance(triple.object, str)
    assert len(triple.subject) > 0
    assert len(triple.predicate) > 0
    assert len(triple.object) > 0


def assert_valid_bootstrap_result(result):
    """Assert that bootstrap result is valid."""
    from ..data_structures import BootstrapResult
    
    assert isinstance(result, BootstrapResult)
    assert_valid_triple_data(result.triple)
    assert isinstance(result.source_idx, int)
    assert 0.0 <= result.fuzzy_score <= 1.0
    assert result.auto_expected in [None, True, False]
    assert result.llm_evaluation in [None, "Yes", "No"]
    assert result.expected in [None, True, False]
    assert isinstance(result.note, str)


# Constants for testing
TEST_INSTRUCTIONS = [
    "Is this true: 曹雪芹 創作 紅樓夢 ?",
    "Is this true: Apple Founded by Steve Jobs ?",
    "Is this true: Microsoft Founded by Mark Zuckerberg ?",
    "Is this true: 賈寶玉 創作 紅樓夢 ?",
    "Is this true: Google Founded by Larry Page ?"
]

TEST_CHINESE_LITERATURE = [
    "Is this true: 曹雪芹 創作 紅樓夢 ?",
    "Is this true: 賈寶玉 喜歡 林黛玉 ?",
    "Is this true: 薛寶釵 住在 大觀園 ?",
    "Is this true: 林黛玉 是 賈母外孫女 ?"
]

TEST_TECH_FACTS = [
    "Is this true: Apple Founded by Steve Jobs ?",
    "Is this true: Microsoft Founded by Bill Gates ?",
    "Is this true: Google Founded by Larry Page ?",
    "Is this true: Facebook Founded by Mark Zuckerberg ?"
]
