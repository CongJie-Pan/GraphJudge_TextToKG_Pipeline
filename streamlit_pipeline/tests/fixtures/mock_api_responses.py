"""
Mock API response fixtures for testing.

This module provides realistic mock responses for GPT-5-mini and Perplexity APIs
to enable comprehensive testing without actual API calls.
"""

# Mock responses for entity extraction
ENTITY_EXTRACTION_RESPONSES = {
    "dream_of_red_chamber": {
        "input": "林黛玉進入榮國府後，與賈寶玉初次相遇，兩人情投意合。",
        "response": '["林黛玉", "榮國府", "賈寶玉", "情投意合"]',
        "expected_entities": ["林黛玉", "榮國府", "賈寶玉", "情投意合"]
    },
    "simple_text": {
        "input": "小明在學校讀書。",
        "response": '["小明", "學校", "讀書"]',
        "expected_entities": ["小明", "學校", "讀書"]
    },
    "empty_entities": {
        "input": "這是一段沒有特定實體的普通文字。",
        "response": '[]',
        "expected_entities": []
    }
}

# Mock responses for text denoising
TEXT_DENOISING_RESPONSES = {
    "dream_of_red_chamber": {
        "input": "林黛玉進入榮國府後，與賈寶玉初次相遇，兩人情投意合。",
        "entities": ["林黛玉", "榮國府", "賈寶玉", "情投意合"],
        "response": "林黛玉初入榮國府，與賈寶玉相遇，二人情意相通。"
    },
    "simple_text": {
        "input": "小明在學校讀書，學習很認真。",
        "entities": ["小明", "學校", "讀書"],
        "response": "小明於學校專心讀書。"
    }
}

# Mock error responses
ERROR_RESPONSES = {
    "api_connection_error": Exception("Connection to API failed"),
    "rate_limit_error": Exception("Rate limit exceeded"),
    "invalid_response": "This is not a valid JSON response",
    "malformed_list": '["entity1", "entity2"'  # Missing closing bracket
}

# Mock responses for triple generation (for future use)
TRIPLE_GENERATION_RESPONSES = {
    "basic_triples": {
        "input_entities": ["林黛玉", "賈寶玉", "榮國府"],
        "input_text": "林黛玉與賈寶玉在榮國府相遇。",
        "response": '''[
            {"subject": "林黛玉", "predicate": "相遇", "object": "賈寶玉"},
            {"subject": "林黛玉", "predicate": "位於", "object": "榮國府"},
            {"subject": "賈寶玉", "predicate": "位於", "object": "榮國府"}
        ]'''
    }
}

# Mock responses for graph judgment (for future use)  
GRAPH_JUDGMENT_RESPONSES = {
    "basic_judgments": {
        "input_triples": [
            {"subject": "林黛玉", "predicate": "相遇", "object": "賈寶玉"},
            {"subject": "林黛玉", "predicate": "位於", "object": "榮國府"}
        ],
        "response": '''[
            {"triple": 0, "judgment": true, "confidence": 0.95},
            {"triple": 1, "judgment": true, "confidence": 0.88}
        ]'''
    }
}