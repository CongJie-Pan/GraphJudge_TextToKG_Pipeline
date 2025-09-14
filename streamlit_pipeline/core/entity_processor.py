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
    from ..utils.detailed_logger import DetailedLogger
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import EntityResult, ProcessingTimer, create_error_result
    from utils.api_client import call_gpt5_mini
    from utils.error_handling import ErrorHandler, ErrorType, safe_execute, get_logger
    from utils.detailed_logger import DetailedLogger


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
    # Initialize detailed logger for ECTD phase
    detailed_logger = DetailedLogger(phase="ectd")
    detailed_logger.log_info("ENTITY", "Starting entity extraction and text denoising", {
        "text_length": len(text) if text else 0,
        "text_preview": text[:200] + "..." if text and len(text) > 200 else text,
        "text_type": type(text).__name__
    })

    logger = get_logger()
    logger.info("Starting entity extraction", stage="entity_extraction")

    # Input validation with user-friendly errors
    if text is None:
        error_info = ErrorHandler.create_error(
            ErrorType.INPUT_VALIDATION,
            message="No input text provided",
            stage="entity_extraction"
        )
        detailed_logger.log_error("ENTITY", "Input validation failed: text is None", {
            "error_type": "NULL_INPUT",
            "stage": "input_validation"
        })
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
        detailed_logger.log_error("ENTITY", "Input validation failed: empty text", {
            "error_type": "EMPTY_INPUT",
            "text_length": len(text) if text else 0,
            "stage": "input_validation"
        })
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
            detailed_logger.log_error("ENTITY", f"Entity extraction failed: {error_info.message}", {
                "error_message": error_info.message,
                "processing_time": timer.elapsed
            })
            logger.error(f"Entity extraction failed: {error_info.message}", stage="entity_extraction")
            return EntityResult(
                entities=[],
                denoised_text="",
                success=False,
                processing_time=timer.elapsed,
                error=error_info.message
            )

        # Log successful entity extraction
        detailed_logger.log_info("ENTITY", "Entity extraction completed successfully", {
            "entity_count": len(result[0]),
            "entities": result[0][:10],  # First 10 entities for debugging
            "denoised_text_length": len(result[1]) if result[1] else 0,
            "processing_time": timer.elapsed
        })

        logger.info(f"Entity extraction completed successfully in {timer.elapsed:.2f}s",
                   stage="entity_extraction",
                   extra={"entity_count": len(result[0]), "processing_time": timer.elapsed})

        # Log successful entity extraction
        print(f"[ENTITY] Extracted {len(result[0])} entities successfully")

        entity_result = EntityResult(
            entities=result[0],
            denoised_text=result[1],
            success=True,
            processing_time=timer.elapsed,
            error=None
        )

        return entity_result


def _extract_entities_pipeline(text: str) -> tuple[List[str], str]:
    """
    Internal pipeline function for entity extraction and text denoising.

    This function is wrapped by safe_execute for error handling.

    Args:
        text: Input Chinese text

    Returns:
        Tuple of (entities, denoised_text)
    """
    detailed_logger = DetailedLogger(phase="ectd")

    # Step 1: Extract entities
    detailed_logger.log_info("ENTITY", "Starting entity extraction from text")
    entities = _extract_entities_from_text(text)
    detailed_logger.log_info("ENTITY", f"Extracted {len(entities)} entities", {
        "entity_count": len(entities),
        "entities": entities[:5] if entities else []
    })

    # Step 2: Denoise text using extracted entities
    detailed_logger.log_info("ENTITY", "Starting text denoising with extracted entities")
    denoised_text = _denoise_text_with_entities(text, entities)
    detailed_logger.log_info("ENTITY", "Text denoising completed", {
        "original_length": len(text),
        "denoised_length": len(denoised_text) if denoised_text else 0
    })

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
    detailed_logger = DetailedLogger(phase="ectd")
    detailed_logger.log_debug("ENTITY", "Starting API call for entity extraction", {
        "text_length": len(text),
        "text_preview": text[:50] + "..." if len(text) > 50 else text
    })

    # System prompt exactly matching the original run_entity.py implementation
    system_prompt = """你是一個專門處理古典中文文本的實體提取專家。請嚴格按照以下要求：
1. 提取人物、地點、物品、概念等重要實體
2. 必須去除重複的實體（同一實體只保留一次）
3. 返回格式必須是Python列表格式：["實體1", "實體2", "實體3"]
4. 優先提取具體的人名、地名和重要概念
5. 避免提取過於抽象或通用的詞彙
6. 確保每個實體在列表中唯一，無重複
請專注於提取有意義的實體，並確保結果列表中沒有重複項目。"""

    # User prompt exactly matching the original format
    prompt = f"""請從以下古典中文文本中提取重要實體（人物、地點、物品、概念等），去除重複項，按Python列表格式返回：

文本：{text}

請返回格式：["實體1", "實體2", "實體3"]"""

    detailed_logger.log_debug("ENTITY", "Prepared prompts for entity extraction", {
        "system_prompt_length": len(system_prompt),
        "user_prompt_length": len(prompt)
    })

    # Make API call using simplified client
    detailed_logger.log_info("API", "Making API call to GPT-5-mini for entity extraction")
    try:
        response = call_gpt5_mini(prompt, system_prompt)
        detailed_logger.log_api_call("GPT-5-mini", len(prompt), len(response) if response else 0,
                                   success=True)

        detailed_logger.log_debug("ENTITY", "Received API response", {
            "response_length": len(response) if response else 0,
            "response_preview": response[:100] + "..." if response and len(response) > 100 else response
        })
    except Exception as e:
        detailed_logger.log_api_call("GPT-5-mini", len(prompt), 0, success=False, error=str(e))
        raise

    # Parse response to extract entity list
    detailed_logger.log_debug("ENTITY", "Parsing entity response")
    entities = _parse_entity_response(response)

    detailed_logger.log_info("ENTITY", f"Successfully extracted {len(entities)} entities", {
        "entities": entities,
        "entity_count": len(entities)
    })

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
    detailed_logger = DetailedLogger(phase="ectd")
    detailed_logger.log_debug("ENTITY", "Starting text denoising with entities", {
        "text_length": len(text),
        "entity_count": len(entities),
        "entities": entities
    })

    # System prompt exactly matching the original run_entity.py denoising implementation
    system_prompt = """你是一個古典中文文本重構專家。請根據提供的實體清單，將原始文本重新整理為更清晰、更結構化的版本。

要求：
1. 保持古典中文的語言風格和韻味
2. 圍繞提供的實體組織內容
3. 移除冗餘和不相關的描述
4. 將內容重組為清晰的事實陳述
5. 突出實體之間的關係

請確保重構後的文本簡潔明瞭，同時保持原文的核心信息和文學價值。"""

    # Format entities exactly like the original code (as string representation)
    entities_str = str(entities)

    # User prompt exactly matching the original format
    prompt = f"""請基於以下實體清單，將原始文本重新整理為更清晰、結構化的版本：

實體清單：{entities_str}

原始文本：
{text}

請返回重構後的文本，要求簡潔明瞭，突出實體間的關係，同時保持古典中文的優雅風格。"""

    detailed_logger.log_debug("ENTITY", "Prepared prompts for text denoising", {
        "system_prompt_length": len(system_prompt),
        "user_prompt_length": len(prompt),
        "entities_format": entities_str
    })

    # Make API call
    detailed_logger.log_info("API", "Making API call to GPT-5-mini for text denoising")
    try:
        denoised_text = call_gpt5_mini(prompt, system_prompt)
        detailed_logger.log_api_call("GPT-5-mini", len(prompt), len(denoised_text) if denoised_text else 0,
                                   success=True)

        detailed_logger.log_debug("ENTITY", "Received denoised text response", {
            "response_length": len(denoised_text) if denoised_text else 0,
            "response_preview": denoised_text[:100] + "..." if denoised_text and len(denoised_text) > 100 else denoised_text
        })

        cleaned_text = denoised_text.strip()
        detailed_logger.log_info("ENTITY", f"Successfully denoised text", {
            "original_length": len(text),
            "denoised_length": len(cleaned_text)
        })

        return cleaned_text

    except Exception as e:
        detailed_logger.log_api_call("GPT-5-mini", len(prompt), 0, success=False, error=str(e))
        raise


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
    detailed_logger = DetailedLogger(phase="ectd")
    detailed_logger.log_debug("ENTITY", "Starting entity response parsing", {
        "response_length": len(response) if response else 0,
        "response_raw": response[:200] + "..." if response and len(response) > 200 else response
    })

    try:
        # Clean the response
        response = response.strip()
        detailed_logger.log_debug("ENTITY", "Cleaned response", {
            "cleaned_length": len(response),
            "has_brackets": '[' in response and ']' in response
        })

        # Try to find list-like patterns
        if '[' in response and ']' in response:
            # Extract content between brackets
            start = response.find('[')
            end = response.rfind(']') + 1
            list_str = response[start:end]

            detailed_logger.log_debug("ENTITY", "Found bracket pattern", {
                "extracted_list_str": list_str,
                "start_pos": start,
                "end_pos": end
            })

            # Use eval with safety checks (limited to list literals)
            # In production, consider using ast.literal_eval
            try:
                entities = eval(list_str)
                if isinstance(entities, list):
                    detailed_logger.log_debug("ENTITY", "Successfully parsed as Python list", {
                        "raw_entity_count": len(entities),
                        "raw_entities": entities
                    })

                    # Clean and deduplicate entities
                    cleaned_entities = []
                    seen = set()
                    for entity in entities:
                        if isinstance(entity, str):
                            entity = entity.strip()
                            if entity and entity not in seen:
                                cleaned_entities.append(entity)
                                seen.add(entity)

                    detailed_logger.log_info("ENTITY", f"Successfully cleaned and deduplicated entities", {
                        "final_entity_count": len(cleaned_entities),
                        "final_entities": cleaned_entities,
                        "duplicates_removed": len(entities) - len(cleaned_entities)
                    })
                    return cleaned_entities
            except Exception as eval_error:
                detailed_logger.log_warning("ENTITY", "Failed to eval list string", {
                    "eval_error": str(eval_error),
                    "list_str": list_str
                })

        # Fallback: try to extract entities from comma/Chinese separator delimited text
        # Look for patterns like "entity1, entity2" or "entity1、entity2"
        import re

        detailed_logger.log_debug("ENTITY", "Attempting fallback parsing with regex")

        # Remove common prefixes/suffixes
        clean_response = re.sub(r'^[^"\']*["\']|["\'][^"\']*$', '', response)

        # Split by common separators
        potential_entities = re.split(r'[,，、\n]', clean_response)

        detailed_logger.log_debug("ENTITY", "Split response by separators", {
            "potential_entities": potential_entities,
            "split_count": len(potential_entities)
        })

        entities = []
        seen = set()
        for entity in potential_entities:
            entity = entity.strip().strip('"\'')
            if entity and len(entity) > 0 and entity not in seen:
                entities.append(entity)
                seen.add(entity)

        if entities:
            final_entities = entities[:20]  # Limit to reasonable number
            detailed_logger.log_info("ENTITY", f"Successfully parsed entities using fallback method", {
                "parsed_entity_count": len(final_entities),
                "parsed_entities": final_entities,
                "limited_from": len(entities)
            })
            return final_entities

        # If all else fails, return empty list rather than raising exception
        detailed_logger.log_warning("ENTITY", "No entities could be parsed, returning empty list")
        return []

    except Exception as e:
        detailed_logger.log_error("ENTITY", "Failed to parse entity response", {
            "error": str(e),
            "response_preview": response[:200] + "..." if response and len(response) > 200 else response
        })
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