#!/usr/bin/env python3

import sys
sys.path.insert(0, 'streamlit_pipeline')
from streamlit_pipeline.core.triple_generator import validate_response_schema, extract_json_from_response
import json
import re

# Test the exact response format that the mock is returning
triples_data = [
    {'subject': '林黛玉', 'predicate': '來到', 'object': '榮國府', 'confidence': 0.95},
    {'subject': '林黛玉', 'predicate': '受到喜愛', 'object': '賈母', 'confidence': 0.90}
]

triples_json = {
    'triples': [
        {
            'subject': triple['subject'],
            'predicate': triple['predicate'],
            'object': triple['object'],
            'confidence': triple.get('confidence', 0.8)
        }
        for triple in triples_data
    ]
}

response = f"```json\n{json.dumps(triples_json)}\n```"
print('Mock response:')
print(repr(response))

print('\nTesting extract_json_from_response:')
extracted = extract_json_from_response(response)
print('Extracted JSON:', repr(extracted))

print('\nTesting regex pattern manually:')
json_object_pattern = r'\{\s*"triples"\s*:\s*\[.*?\]\s*\}'
matches = re.findall(json_object_pattern, response, re.DOTALL)
print('Regex matches:', matches)

print('\nValidation result:')
result = validate_response_schema(response)
print('Validated data:', result)