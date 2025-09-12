"""
Comprehensive API response fixtures for GraphJudge Streamlit Pipeline testing.

This module provides realistic mock responses for GPT-5-mini and Perplexity APIs,
following the testing guidelines in docs/Testing_Demands.md and the testing strategy
in spec.md Section 15.

Key principles:
- Realistic API response structures
- Multiple scenarios (success, error, edge cases) 
- Support for both unit and integration testing
- Easy to extend for new test cases
"""

from typing import Dict, List, Any, Union
from datetime import datetime
import json


# =============================================================================
# GPT-5-MINI API FIXTURES
# =============================================================================

class GPT5MiniFixtures:
    """Mock responses for GPT-5-mini entity extraction and text denoising."""
    
    @staticmethod
    def successful_entity_response(entities: List[str]) -> Dict[str, Any]:
        """Generate a successful GPT-5-mini entity extraction response."""
        return {
            "id": "chatcmpl-test-12345",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(entities, ensure_ascii=False)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 25,
                "total_tokens": 175
            }
        }
    
    @staticmethod
    def successful_denoising_response(denoised_text: str) -> Dict[str, Any]:
        """Generate a successful GPT-5-mini text denoising response."""
        return {
            "id": "chatcmpl-test-67890",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": denoised_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280
            }
        }
    
    @staticmethod
    def rate_limit_error() -> Dict[str, Any]:
        """Generate a rate limit error response."""
        return {
            "error": {
                "message": "Rate limit reached for requests",
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded"
            }
        }
    
    @staticmethod
    def invalid_api_key_error() -> Dict[str, Any]:
        """Generate an invalid API key error response."""
        return {
            "error": {
                "message": "Invalid API key provided",
                "type": "authentication_error", 
                "param": None,
                "code": "invalid_api_key"
            }
        }
    
    @staticmethod
    def malformed_response() -> Dict[str, Any]:
        """Generate a malformed JSON response."""
        return {
            "id": "chatcmpl-malformed-123",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '["entity1", "entity2"'  # Missing closing bracket
                    },
                    "finish_reason": "stop"
                }
            ]
        }


# =============================================================================
# PERPLEXITY API FIXTURES
# =============================================================================

class PerplexityFixtures:
    """Mock responses for Perplexity graph judgment API."""
    
    @staticmethod
    def successful_judgment_response(judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a successful Perplexity judgment response."""
        return {
            "id": "ppl-test-judgment-123",
            "model": "perplexity/sonar-reasoning",
            "created": int(datetime.now().timestamp()),
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 150,
                "total_tokens": 650
            },
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(judgments, ensure_ascii=False)
                    }
                }
            ]
        }
    
    @staticmethod
    def successful_explanation_response(explanations: List[str]) -> Dict[str, Any]:
        """Generate a successful Perplexity response with explanations."""
        return {
            "id": "ppl-test-explanation-456", 
            "model": "perplexity/sonar-reasoning",
            "created": int(datetime.now().timestamp()),
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 300,
                "total_tokens": 1100
            },
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(explanations, ensure_ascii=False)
                    }
                }
            ]
        }
    
    @staticmethod
    def server_error() -> Dict[str, Any]:
        """Generate a server error response."""
        return {
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500
            }
        }
    
    @staticmethod
    def timeout_error() -> Dict[str, Any]:
        """Generate a timeout error response."""
        return {
            "error": {
                "message": "Request timed out",
                "type": "timeout_error", 
                "code": "timeout"
            }
        }


# =============================================================================
# SCENARIO-BASED FIXTURES
# =============================================================================

class ScenarioFixtures:
    """Complete scenario fixtures for integration testing."""
    
    # Classical Chinese Literature Scenarios
    DREAM_OF_RED_CHAMBER = {
        "input_text": "林黛玉初入榮國府，見到賈寶玉，二人一見如故，情投意合。賈母對黛玉疼愛有加，安排她住在碧紗櫥。",
        "entities": ["林黛玉", "榮國府", "賈寶玉", "一見如故", "情投意合", "賈母", "疼愛有加", "碧紗櫥"],
        "denoised_text": "林黛玉初入榮國府，與賈寶玉相識並建立深厚情誼。賈母深愛黛玉，為其安排住所碧紗櫥。",
        "triples": [
            {"subject": "林黛玉", "predicate": "進入", "object": "榮國府", "confidence": 0.95},
            {"subject": "林黛玉", "predicate": "相遇", "object": "賈寶玉", "confidence": 0.92},
            {"subject": "林黛玉", "predicate": "情投意合", "object": "賈寶玉", "confidence": 0.88},
            {"subject": "賈母", "predicate": "疼愛", "object": "林黛玉", "confidence": 0.90},
            {"subject": "林黛玉", "predicate": "居住", "object": "碧紗櫥", "confidence": 0.85}
        ],
        "judgments": [
            {"triple_index": 0, "judgment": True, "confidence": 0.98, "explanation": "文本明確提到林黛玉進入榮國府"},
            {"triple_index": 1, "judgment": True, "confidence": 0.95, "explanation": "文本描述兩人相遇的情景"},
            {"triple_index": 2, "judgment": True, "confidence": 0.88, "explanation": "一見如故暗示二人情投意合"},
            {"triple_index": 3, "judgment": True, "confidence": 0.93, "explanation": "賈母對黛玉疼愛有加有明確描述"},
            {"triple_index": 4, "judgment": True, "confidence": 0.90, "explanation": "文本提到安排她住在碧紗櫥"}
        ]
    }
    
    # Modern Chinese Scenarios
    MODERN_SCENARIO = {
        "input_text": "小王在北京大學學習計算機科學，他的導師是李教授。小王的研究方向是人工智能。",
        "entities": ["小王", "北京大學", "計算機科學", "李教授", "導師", "研究方向", "人工智能"],
        "denoised_text": "小王於北京大學研讀計算機科學，師從李教授，專攻人工智能研究。",
        "triples": [
            {"subject": "小王", "predicate": "就讀於", "object": "北京大學", "confidence": 0.98},
            {"subject": "小王", "predicate": "學習", "object": "計算機科學", "confidence": 0.95},
            {"subject": "李教授", "predicate": "指導", "object": "小王", "confidence": 0.92},
            {"subject": "小王", "predicate": "研究", "object": "人工智能", "confidence": 0.90}
        ],
        "judgments": [
            {"triple_index": 0, "judgment": True, "confidence": 0.99, "explanation": "明確提到小王在北京大學學習"},
            {"triple_index": 1, "judgment": True, "confidence": 0.97, "explanation": "學習計算機科學有直接描述"},
            {"triple_index": 2, "judgment": True, "confidence": 0.94, "explanation": "李教授是他的導師表明指導關係"},
            {"triple_index": 3, "judgment": True, "confidence": 0.92, "explanation": "研究方向是人工智能有明確說明"}
        ]
    }
    
    # Edge Cases and Error Scenarios
    EMPTY_TEXT = {
        "input_text": "",
        "entities": [],
        "denoised_text": "",
        "triples": [],
        "judgments": []
    }
    
    SINGLE_ENTITY = {
        "input_text": "蘋果很甜。",
        "entities": ["蘋果"],
        "denoised_text": "蘋果味道甜美。",
        "triples": [
            {"subject": "蘋果", "predicate": "是", "object": "甜", "confidence": 0.85}
        ],
        "judgments": [
            {"triple_index": 0, "judgment": True, "confidence": 0.88, "explanation": "文本直接描述蘋果很甜"}
        ]
    }
    
    # Error scenarios
    MALFORMED_JSON = {
        "input_text": "測試文本",
        "entities": "Invalid JSON response from API",
        "error_type": "json_decode_error"
    }
    
    RATE_LIMITED = {
        "input_text": "測試文本",
        "error_type": "rate_limit_exceeded",
        "retry_after": 60
    }


# =============================================================================
# UTILITY FUNCTIONS FOR TEST FIXTURES
# =============================================================================

def get_entity_extraction_fixture(scenario_name: str) -> Dict[str, Any]:
    """Get entity extraction fixture for a given scenario."""
    scenario = getattr(ScenarioFixtures, scenario_name, None)
    if not scenario:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    return GPT5MiniFixtures.successful_entity_response(scenario["entities"])


def get_denoising_fixture(scenario_name: str) -> Dict[str, Any]:
    """Get text denoising fixture for a given scenario."""
    scenario = getattr(ScenarioFixtures, scenario_name, None)
    if not scenario:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    return GPT5MiniFixtures.successful_denoising_response(scenario["denoised_text"])


def get_judgment_fixture(scenario_name: str, include_explanations: bool = False) -> Dict[str, Any]:
    """Get graph judgment fixture for a given scenario."""
    scenario = getattr(ScenarioFixtures, scenario_name, None)
    if not scenario:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    if include_explanations:
        explanations = [j["explanation"] for j in scenario["judgments"]]
        return PerplexityFixtures.successful_explanation_response(explanations)
    else:
        judgments = [
            {
                "judgment": j["judgment"],
                "confidence": j["confidence"],
                "triple_index": j["triple_index"]
            }
            for j in scenario["judgments"]
        ]
        return PerplexityFixtures.successful_judgment_response(judgments)


def get_error_fixture(error_type: str) -> Dict[str, Any]:
    """Get error fixture for a given error type."""
    error_fixtures = {
        "rate_limit": GPT5MiniFixtures.rate_limit_error(),
        "invalid_api_key": GPT5MiniFixtures.invalid_api_key_error(),
        "malformed_response": GPT5MiniFixtures.malformed_response(),
        "server_error": PerplexityFixtures.server_error(),
        "timeout": PerplexityFixtures.timeout_error()
    }
    
    return error_fixtures.get(error_type, {})


# =============================================================================
# PYTEST FIXTURES (for use with @pytest.fixture)
# =============================================================================

def create_mock_response(fixture_data: Dict[str, Any], status_code: int = 200):
    """Create a mock HTTP response object."""
    class MockResponse:
        def __init__(self, json_data: Dict[str, Any], status_code: int):
            self.json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = {"content-type": "application/json"}
        
        def json(self):
            return self.json_data
        
        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code} Error")
    
    return MockResponse(fixture_data, status_code)


# Standard test scenarios for parametrized tests
STANDARD_TEST_SCENARIOS = [
    ("DREAM_OF_RED_CHAMBER", "Classical Chinese literature scenario"),
    ("MODERN_SCENARIO", "Modern Chinese text scenario"),
    ("SINGLE_ENTITY", "Simple single entity scenario"),
]

ERROR_TEST_SCENARIOS = [
    ("rate_limit", "API rate limit exceeded"),
    ("invalid_api_key", "Invalid API key error"),
    ("server_error", "Internal server error"),
    ("timeout", "Request timeout error"),
]