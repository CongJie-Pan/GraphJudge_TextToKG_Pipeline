"""
Triple generator module for GraphJudge Streamlit Pipeline.

This module extracts the core triple generation logic from the original
chat/run_triple.py script (~750+ lines) and simplifies it for Streamlit
integration while preserving essential functionality:

- JSON schema validation with Pydantic models
- Text chunking for large inputs  
- Enhanced prompt engineering
- Structured output generation
- Essential error handling

Key simplifications from original:
- Synchronous execution (no asyncio)
- In-memory data passing (no file I/O)
- Simplified error handling (errors as data)
- Reduced complexity (~200-250 lines vs 750+)
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Pydantic imports for schema validation
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from .models import Triple, TripleResult, ProcessingTimer, create_error_result


# Configuration constants
MAX_TOKENS_PER_CHUNK = 1000  # Maximum tokens per text chunk
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context


# Pydantic models for schema validation
if PYDANTIC_AVAILABLE:
    class TripleSchema(BaseModel):
        """Pydantic model for individual triple validation."""
        subject: str
        relation: str
        object: str
        
        class Config:
            str_strip_whitespace = True
    
    class TripleResponse(BaseModel):
        """Pydantic model for GPT-5-mini response validation."""
        triples: List[List[str]]
        
        def validate_structure(self) -> bool:
            """Validate that all triples have exactly 3 components."""
            for triple in self.triples:
                if len(triple) != 3:
                    return False
            return True


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, 
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks to handle large texts.
    
    Uses character-based estimation for Chinese text processing,
    with intelligent boundary detection at sentence endings.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Overlap between chunks for context preservation
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Character-based chunking with Chinese text estimation
    chars_per_token = 2  # Rough estimate for Chinese characters
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap * chars_per_token
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        # Try to break at sentence boundaries for Chinese text
        if end < len(text):
            # Look for Chinese punctuation marks
            for punct in ['。', '！', '？', '；', '，']:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start + max_chars // 2:
                    end = last_punct + 1
                    break
        
        chunks.append(text[start:end])
        
        if end >= len(text):
            break
        
        start = end - overlap_chars
    
    return chunks


def create_enhanced_prompt(text_content: str, entity_list: List[str]) -> str:
    """
    Create enhanced prompt with structured JSON output requirements.
    
    Based on the sophisticated prompt engineering from the original
    run_triple.py, optimized for Chinese text and entity-guided extraction.
    
    Args:
        text_content: Chinese text to process
        entity_list: List of entity names to guide extraction
        
    Returns:
        Formatted prompt string for GPT model
    """
    entities_str = ", ".join(f'"{entity}"' for entity in entity_list)
    
    return f"""
任務：分析古典中文文本，提取實體間的語義關係，輸出標準JSON格式的三元組。

## 輸出格式要求：
```json
{{
  "triples": [
    ["主體", "關係", "客體"],
    ["主體", "關係", "客體"]
  ]
}}
```

## 關係詞規範：
- 使用簡潔的中文關係詞（如："職業", "妻子", "地點", "行為"）
- 避免冗長描述和解釋性詞語
- 確保關係具有明確的語義含義
- 優先使用常見的標準關係類型

## 抽取原則：
1. 重點關注給定實體列表中的實體
2. 提取文本中明確表達的關係
3. 避免推測或推理出的隱含關係
4. 每個三元組必須在原文中有明確依據

## 範例：
輸入文本：「甄士隱是姑蘇城內的鄉宦，妻子是封氏，有一女名英蓮。」
實體列表：["甄士隱", "姑蘇城", "鄉宦", "封氏", "英蓮"]

輸出：
```json
{{
  "triples": [
    ["甄士隱", "地點", "姑蘇城"],
    ["甄士隱", "職業", "鄉宦"],
    ["甄士隱", "妻子", "封氏"],
    ["甄士隱", "女兒", "英蓮"]
  ]
}}
```

## 當前任務：
文本：{text_content}
實體列表：[{entities_str}]

請按照上述格式要求輸出JSON："""


def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON content from GPT response with improved pattern matching.
    
    Uses multiple regex patterns to handle various response formats
    from the GPT model, prioritizing structured JSON objects.
    
    Args:
        response: Raw response string from GPT model
    
    Returns:
        JSON string if found, None otherwise
    """
    if not response:
        return None
        
    response_str = str(response).strip()
    
    # Pattern 1: Look for {"triples": [...]} format (preferred)
    json_object_pattern = r'\{\s*"triples"\s*:\s*\[.*?\]\s*\}'
    matches = re.findall(json_object_pattern, response_str, re.DOTALL)
    if matches:
        return matches[0]
    
    # Pattern 2: Look for [[...]] nested array format
    json_array_pattern = r'\[\[.*?\]\]'
    matches = re.findall(json_array_pattern, response_str, re.DOTALL)
    if matches:
        return matches[0]
    
    # Pattern 3: Look for simple array format
    simple_array_pattern = r'\[.*?\]'
    matches = re.findall(simple_array_pattern, response_str, re.DOTALL)
    if matches:
        # Take the longest match that looks like nested arrays
        longest_match = max(matches, key=len)
        if '[[' in longest_match or longest_match.count('[') > 1:
            return longest_match
    
    return None


def validate_response_schema(response: str) -> Optional[Dict[str, Any]]:
    """
    Validate GPT response against expected JSON schema using Pydantic.
    
    Extracts JSON content and validates structure, converting legacy
    formats to the standardized format when possible.
    
    Args:
        response: Raw response from GPT model
    
    Returns:
        Validated data dictionary or None if validation fails
    """
    if not PYDANTIC_AVAILABLE:
        # Fallback to simple JSON extraction without validation
        json_str = extract_json_from_response(response)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None
    
    try:
        # Extract JSON content from response
        cleaned_json = extract_json_from_response(response)
        if not cleaned_json:
            return None
        
        # Parse JSON
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError:
            return None
        
        # Validate with Pydantic
        if isinstance(data, dict) and 'triples' in data:
            # Standard format: {"triples": [[...]]}
            validated = TripleResponse(**data)
            if validated.validate_structure():
                return data
        elif isinstance(data, list):
            # Legacy format: [[...]] - convert to standard format
            validated = TripleResponse(triples=data)
            if validated.validate_structure():
                return {'triples': data}
        
        return None
        
    except ValidationError:
        return None
    except Exception:
        return None


def parse_triples_from_validated_data(validated_data: Dict[str, Any], 
                                    source_text: str = "") -> List[Triple]:
    """
    Convert validated JSON data to Triple objects.
    
    Args:
        validated_data: Validated JSON data containing triples
        source_text: Original source text for metadata
        
    Returns:
        List of Triple objects
    """
    triples = []
    
    if 'triples' not in validated_data:
        return triples
    
    for triple_data in validated_data['triples']:
        if len(triple_data) == 3:
            subject, predicate, obj = triple_data
            # Clean whitespace and validate non-empty
            subject = str(subject).strip()
            predicate = str(predicate).strip()
            obj = str(obj).strip()
            
            if subject and predicate and obj:
                triple = Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    source_text=source_text[:100] + "..." if len(source_text) > 100 else source_text,
                    metadata={'extracted_from': 'gpt_response'}
                )
                triples.append(triple)
    
    return triples


def generate_triples(entities: List[str], text: str, 
                   api_client=None) -> TripleResult:
    """
    Generate knowledge graph triples from entities and text.
    
    This is the main public interface that replaces the complex async
    pipeline from the original run_triple.py script. It provides clean,
    synchronous processing suitable for Streamlit integration.
    
    Args:
        entities: List of entity names to guide triple extraction
        text: Denoised Chinese text for processing
        api_client: API client instance for GPT model calls (optional for testing)
        
    Returns:
        TripleResult containing extracted triples and metadata
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not text or not text.strip():
            return create_error_result(
                TripleResult, 
                "Input text is empty or invalid",
                time.time() - start_time
            )
        
        if not entities or len(entities) == 0:
            return create_error_result(
                TripleResult,
                "Entity list is empty or invalid", 
                time.time() - start_time
            )
        
        # Text chunking for large inputs
        text_chunks = chunk_text(text)
        all_triples = []
        chunk_info = {
            'total_chunks': len(text_chunks),
            'chunks_processed': 0,
            'chunks_with_triples': 0
        }
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(text_chunks):
            # Create enhanced prompt
            prompt = create_enhanced_prompt(chunk, entities)
            
            # Make API call (if api_client provided)
            if api_client:
                try:
                    response = api_client.call_gpt5_mini(prompt)
                    chunk_info['chunks_processed'] += 1
                    
                    # Validate and parse response
                    validated_data = validate_response_schema(response)
                    if validated_data:
                        chunk_triples = parse_triples_from_validated_data(
                            validated_data, chunk
                        )
                        if chunk_triples:
                            all_triples.extend(chunk_triples)
                            chunk_info['chunks_with_triples'] += 1
                            
                except Exception as e:
                    # Continue processing other chunks on error
                    continue
            else:
                # Mock processing for testing without API client
                chunk_info['chunks_processed'] += 1
        
        # Remove duplicate triples
        unique_triples = []
        seen_triples = set()
        
        for triple in all_triples:
            triple_key = (triple.subject, triple.predicate, triple.object)
            if triple_key not in seen_triples:
                unique_triples.append(triple)
                seen_triples.add(triple_key)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare metadata
        metadata = {
            'text_processing': {
                'original_length': len(text),
                'chunks_created': chunk_info['total_chunks'],
                'chunks_processed': chunk_info['chunks_processed'],
                'chunks_with_triples': chunk_info['chunks_with_triples']
            },
            'extraction_stats': {
                'entities_provided': len(entities),
                'total_triples_extracted': len(all_triples),
                'unique_triples': len(unique_triples),
                'duplicates_removed': len(all_triples) - len(unique_triples)
            },
            'validation': {
                'pydantic_available': PYDANTIC_AVAILABLE,
                'schema_validation': 'enabled' if PYDANTIC_AVAILABLE else 'disabled'
            }
        }
        
        # Calculate success rate
        success_rate = (chunk_info['chunks_processed'] / 
                      max(chunk_info['total_chunks'], 1))
        
        return TripleResult(
            triples=unique_triples,
            metadata=metadata,
            success=success_rate > 0.5,  # Consider successful if >50% chunks processed
            processing_time=processing_time,
            error=None if success_rate > 0.5 else f"Low success rate: {success_rate:.1%}"
        )
        
    except Exception as e:
        return create_error_result(
            TripleResult,
            f"Triple generation failed: {str(e)}",
            time.time() - start_time
        )


def validate_triples_quality(triples: List[Triple]) -> Dict[str, Any]:
    """
    Analyze the quality of extracted triples.
    
    Provides quality metrics for generated triples including
    validation statistics and content analysis.
    
    Args:
        triples: List of Triple objects to analyze
        
    Returns:
        Dictionary containing quality metrics
    """
    if not triples:
        return {
            'total_triples': 0,
            'quality_score': 0.0,
            'issues': ['No triples generated']
        }
    
    quality_metrics = {
        'total_triples': len(triples),
        'empty_fields': 0,
        'short_fields': 0,  # Fields with less than 2 characters
        'long_fields': 0,   # Fields with more than 50 characters
        'valid_triples': 0,
        'issues': []
    }
    
    for triple in triples:
        has_empty_field = False
        # Check for empty fields
        if not triple.subject.strip() or not triple.predicate.strip() or not triple.object.strip():
            quality_metrics['empty_fields'] += 1
            has_empty_field = True
            
        # Only check field lengths and count as valid if no empty fields
        if not has_empty_field:
            # Check field lengths
            for field, name in [(triple.subject, 'subject'), (triple.predicate, 'predicate'), (triple.object, 'object')]:
                if len(field.strip()) < 2:
                    quality_metrics['short_fields'] += 1
                elif len(field.strip()) > 50:
                    quality_metrics['long_fields'] += 1
                    
            quality_metrics['valid_triples'] += 1
    
    # Calculate quality score
    if len(triples) > 0:
        quality_score = quality_metrics['valid_triples'] / len(triples)
    else:
        quality_score = 0.0
    
    quality_metrics['quality_score'] = quality_score
    
    # Add issues
    if quality_metrics['empty_fields'] > 0:
        quality_metrics['issues'].append(f"{quality_metrics['empty_fields']} triples with empty fields")
    if quality_metrics['short_fields'] > 0:
        quality_metrics['issues'].append(f"{quality_metrics['short_fields']} fields too short")
    if quality_metrics['long_fields'] > 0:
        quality_metrics['issues'].append(f"{quality_metrics['long_fields']} fields too long")
    
    return quality_metrics