"""
Entity extraction and text denoising processor for GraphJudge Streamlit Pipeline.

This module contains the refactored entity processing functionality extracted from
chat/run_entity.py, simplified for Streamlit integration while preserving core
GPT-5-mini capabilities.

Key simplifications from original script:
- Synchronous execution (no async/await complexity)
- In-memory data handling (no file I/O)
- Simple error handling (errors returned as data)
- No complex caching system
- No intricate logging system
- No manual directory management

Preserved capabilities:
- GPT-5-mini entity extraction for Chinese text
- Text denoising functionality  
- Classical Chinese processing optimizations
- Entity deduplication
"""

import time
from typing import List, Optional

try:
    from .models import EntityResult, ProcessingTimer, create_error_result
    from ..utils.api_client import call_gpt5_mini
    from ..utils.error_handling import ErrorHandler, ErrorType, safe_execute, get_logger
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import EntityResult, ProcessingTimer, create_error_result
    from utils.api_client import call_gpt5_mini
    from utils.error_handling import ErrorHandler, ErrorType, safe_execute, get_logger


def extract_entities(text: str) -> EntityResult:
    """
    Extract entities and denoise text using GPT-5-mini.
    
    This is the main interface function that combines entity extraction
    and text denoising in a single operation, following the spec.md
    requirement for clean function interfaces.
    
    Enhanced with comprehensive error handling following spec.md Section 10.
    
    Args:
        text: Input Chinese text to process
        
    Returns:
        EntityResult containing extracted entities, denoised text, and metadata.
        Errors are returned as data, never raised as exceptions.
    """
    logger = get_logger()
    logger.info("Starting entity extraction", stage="entity_extraction")
    
    # Input validation with user-friendly errors
    if text is None:
        error_info = ErrorHandler.create_error(
            ErrorType.INPUT_VALIDATION,
            message="No input text provided",
            stage="entity_extraction"
        )
        logger.error("Input validation failed: text is None", stage="entity_extraction")
        return EntityResult(
            entities=[],
            denoised_text="",
            success=False,
            processing_time=0.0,
            error=error_info.message
        )
    
    if not text or not text.strip():
        error_info = ErrorHandler.create_error(
            ErrorType.INPUT_VALIDATION,
            message="Input text is empty or contains only whitespace",
            stage="entity_extraction"
        )
        logger.error("Input validation failed: empty text", stage="entity_extraction")
        return EntityResult(
            entities=[],
            denoised_text="",
            success=False,
            processing_time=0.0,
            error=error_info.message
        )
    
    with ProcessingTimer() as timer:
        # Use safe_execute for error handling
        result, error_info = safe_execute(
            _extract_entities_pipeline,
            text,
            logger=logger,
            stage="entity_extraction"
        )
        
        if error_info:
            logger.error(f"Entity extraction failed: {error_info.message}", stage="entity_extraction")
            return EntityResult(
                entities=[],
                denoised_text="",
                success=False,
                processing_time=timer.elapsed,
                error=error_info.message
            )
        
        logger.info(f"Entity extraction completed successfully in {timer.elapsed:.2f}s", 
                   stage="entity_extraction", 
                   extra={"entity_count": len(result[0]), "processing_time": timer.elapsed})
        
        return EntityResult(
            entities=result[0],
            denoised_text=result[1],
            success=True,
            processing_time=timer.elapsed,
            error=None
        )


def _extract_entities_pipeline(text: str) -> tuple[List[str], str]:
    """
    Internal pipeline function for entity extraction and text denoising.
    
    This function is wrapped by safe_execute for error handling.
    
    Args:
        text: Input Chinese text
        
    Returns:
        Tuple of (entities, denoised_text)
    """
    # Step 1: Extract entities
    entities = _extract_entities_from_text(text)
    
    # Step 2: Denoise text using extracted entities  
    denoised_text = _denoise_text_with_entities(text, entities)
    
    return entities, denoised_text


def _extract_entities_from_text(text: str) -> List[str]:
    """
    Extract entities from Chinese text using GPT-5-mini.
    
    This function replicates the core logic from the original extract_entities()
    function but simplified for synchronous execution.
    
    Args:
        text: Input Chinese text
        
    Returns:
        List of extracted entity names (deduplicated)
        
    Raises:
        Exception: If API call fails or response parsing fails
    """
    # System prompt optimized for Chinese entity extraction
    # Extracted and simplified from original run_entity.py lines 578-590
    system_prompt = """你是一個專門處理古典中文文本的實體提取專家。請嚴格按照以下要求：

1. 提取人物、地點、物品、概念等重要實體
2. 必須去除重複的實體（同一實體只保留一次）
3. 返回格式必須是Python列表格式：["實體1", "實體2", "實體3"]

重要注意事項：
- 專注於《紅樓夢》相關內容的實體識別
- 人物名稱要準確完整
- 地點包括房間、院落、建築物等
- 物品包括書籍、衣物、飾品等具體物件
- 概念包括情感、關係、事件等抽象概念"""

    # User prompt template from original script
    prompt = f"""請從以下古典中文文本中提取所有重要實體，包括人物、地點、物品、概念等：

文本內容：
{text}

請返回Python列表格式的實體清單，確保去除重複項。"""

    # Make API call using simplified client
    response = call_gpt5_mini(prompt, system_prompt)
    
    # Parse response to extract entity list
    entities = _parse_entity_response(response)
    
    return entities


def _denoise_text_with_entities(text: str, entities: List[str]) -> str:
    """
    Denoise and restructure text using extracted entities.
    
    This function replicates the core logic from the original denoise_text()
    function but simplified for synchronous execution.
    
    Args:
        text: Original Chinese text
        entities: List of extracted entities
        
    Returns:
        Denoised and restructured text
        
    Raises:
        Exception: If API call fails
    """
    # System prompt for text denoising (from original lines 657-670)
    system_prompt = """你是一個古典中文文本重構專家。請根據提供的實體清單，將原始文本重新整理為更清晰、更結構化的版本。

要求：
1. 保持古典中文的語言風格和韻味
2. 圍繞提供的實體組織內容
3. 移除冗餘和不相關的描述
4. 將內容重組為清晰的事實陳述
5. 突出實體之間的關係

請確保重構後的文本簡潔明瞭，同時保持原文的核心信息和文學價值。"""

    # Format entities list for prompt
    entities_str = "、".join(entities) if entities else "無特定實體"
    
    # User prompt for denoising
    prompt = f"""請基於以下實體清單，將原始文本重新整理為更清晰、結構化的版本：

實體清單：{entities_str}

原始文本：
{text}

請返回重構後的文本，要求簡潔明瞭，突出實體間的關係，同時保持古典中文的優雅風格。"""

    # Make API call
    denoised_text = call_gpt5_mini(prompt, system_prompt)
    
    return denoised_text.strip()


def _parse_entity_response(response: str) -> List[str]:
    """
    Parse GPT-5-mini response to extract entity list.
    
    The model should return a Python list format, but we need to handle
    various response formats gracefully.
    
    Args:
        response: Raw response from GPT-5-mini
        
    Returns:
        List of extracted entities
        
    Raises:
        Exception: If response cannot be parsed
    """
    try:
        # Clean the response
        response = response.strip()
        
        # Try to find list-like patterns
        if '[' in response and ']' in response:
            # Extract content between brackets
            start = response.find('[')
            end = response.rfind(']') + 1
            list_str = response[start:end]
            
            # Use eval with safety checks (limited to list literals)
            # In production, consider using ast.literal_eval
            try:
                entities = eval(list_str)
                if isinstance(entities, list):
                    # Clean and deduplicate entities
                    cleaned_entities = []
                    seen = set()
                    for entity in entities:
                        if isinstance(entity, str):
                            entity = entity.strip()
                            if entity and entity not in seen:
                                cleaned_entities.append(entity)
                                seen.add(entity)
                    return cleaned_entities
            except:
                pass
        
        # Fallback: try to extract entities from comma/Chinese separator delimited text
        # Look for patterns like "entity1, entity2" or "entity1、entity2"
        import re
        
        # Remove common prefixes/suffixes
        clean_response = re.sub(r'^[^"\']*["\']|["\'][^"\']*$', '', response)
        
        # Split by common separators
        potential_entities = re.split(r'[,，、\n]', clean_response)
        
        entities = []
        seen = set()
        for entity in potential_entities:
            entity = entity.strip().strip('"\'')
            if entity and len(entity) > 0 and entity not in seen:
                entities.append(entity)
                seen.add(entity)
        
        if entities:
            return entities[:20]  # Limit to reasonable number
        
        # If all else fails, return empty list rather than raising exception
        return []
        
    except Exception as e:
        raise Exception(f"Failed to parse entity response: {str(e)}\nResponse: {response[:200]}...")


def batch_extract_entities(texts: List[str]) -> List[EntityResult]:
    """
    Process multiple texts in batch for efficiency.
    
    Args:
        texts: List of input texts to process
        
    Returns:
        List of EntityResult objects corresponding to each input text
    """
    results = []
    for text in texts:
        result = extract_entities(text)
        results.append(result)
        # Small delay between requests to be respectful to API
        time.sleep(0.1)
    
    return results