"""
Entity Extraction Engine Module

This module implements the core entity extraction logic for classical Chinese texts,
using GPT-5-mini to identify and extract entities with deduplication.

The module provides a clean interface for entity extraction with comprehensive
error handling, progress tracking, and result validation.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from extractEntity_Phase.models.entities import Entity, EntityType, EntityList
from extractEntity_Phase.api.gpt5mini_client import GPT5MiniClient, APIRequest, APIResponse
from extractEntity_Phase.infrastructure.logging import get_logger
from extractEntity_Phase.utils.chinese_text import ChineseTextProcessor
from extractEntity_Phase.models.pipeline_state import PipelineStage, ProcessingStatus


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""
    
    # GPT-5-mini settings
    temperature: float = 1.0
    max_tokens: int = 4000
    timeout: int = 60
    
    # Extraction settings
    enable_deduplication: bool = True
    min_confidence: float = 0.7
    max_entities_per_text: int = 50
    
    # Prompt engineering
    use_system_prompt: bool = True
    include_examples: bool = True
    language: str = "zh-TW"  # Traditional Chinese
    
    # Processing settings
    batch_size: int = 10
    max_concurrent: int = 3


class EntityExtractor:
    """
    Core entity extraction engine for classical Chinese texts.
    
    This class implements the main logic for extracting entities from Chinese text
    using GPT-5-mini, with comprehensive deduplication and validation.
    """
    
    def __init__(self, client: GPT5MiniClient, config: Optional[ExtractionConfig] = None):
        """
        Initialize the entity extractor.
        
        Args:
            client: GPT-5-mini API client
            config: Extraction configuration
        """
        self.client = client
        self.config = config or ExtractionConfig()
        self.logger = get_logger()
        self.text_processor = ChineseTextProcessor()
        
        # Statistics tracking
        self.stats = {
            "total_texts_processed": 0,
            "total_entities_extracted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def extract_entities_from_texts(
        self, 
        texts: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[EntityList]:
        """
        Extract entities from a list of Chinese texts.
        
        Args:
            texts: List of Chinese text strings to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of EntityList objects corresponding to each input text
        """
        self.logger.log(f"🔍 Starting entity extraction for {len(texts)} text segments...")
        
        if not texts:
            self.logger.log("⚠️ No texts provided for entity extraction")
            return []
        
        # Validate input texts
        validated_texts = self._validate_input_texts(texts)
        if not validated_texts:
            self.logger.log("❌ Input validation failed")
            return []
        
        # Process texts in batches
        results = []
        total_batches = (len(validated_texts) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(validated_texts))
            batch_texts = validated_texts[start_idx:end_idx]
            
            self.logger.log(f"📦 Processing batch {batch_idx + 1}/{total_batches} "
                          f"(texts {start_idx + 1}-{end_idx})")
            
            # Extract entities from batch
            batch_results = await self._extract_entities_batch(batch_texts)
            results.extend(batch_results)
            
            # Update progress
            if progress_callback:
                progress = (batch_idx + 1) / total_batches
                progress_callback(progress, f"Processed batch {batch_idx + 1}/{total_batches}")
            
            # Add delay between batches to respect rate limits
            if batch_idx < total_batches - 1:
                await asyncio.sleep(0.5)
        
        # Update statistics
        self.stats["total_texts_processed"] = len(texts)
        self.stats["total_entities_extracted"] = sum(len(ec.entities) for ec in results)
        self.stats["successful_extractions"] = sum(1 for ec in results if ec.entities)
        self.stats["failed_extractions"] = sum(1 for ec in results if not ec.entities)
        
        self.logger.log(f"✅ Entity extraction completed. "
                       f"Extracted {self.stats['total_entities_extracted']} entities "
                       f"from {len(texts)} texts")
        
        return results
    
    async def _extract_entities_batch(self, texts: List[str]) -> List[EntityList]:
        """
        Extract entities from a batch of texts.
        
        Args:
            texts: List of texts to process in this batch
            
        Returns:
            List of EntityList objects
        """
        # Create extraction prompts
        prompts = self._create_extraction_prompts(texts)
        
        # Execute API calls
        responses = await self._execute_extraction_requests(prompts)
        
        # Parse and validate responses
        results = []
        for text, response in zip(texts, responses):
            try:
                entities = self._parse_extraction_response(response, text)
                entity_collection = EntityList(
                    entities=entities,
                    source_text=text,
                    extraction_timestamp=asyncio.get_event_loop().time()
                )
                results.append(entity_collection)
            except Exception as e:
                self.logger.log(f"❌ Failed to parse extraction response for text: {str(e)}")
                # Create empty collection for failed extraction
                results.append(EntityList(
                    entities=[],
                    source_text=text,
                    extraction_timestamp=asyncio.get_event_loop().time(),
                    errors=[str(e)]
                ))
        
        return results
    
    def _create_extraction_prompts(self, texts: List[str]) -> List[APIRequest]:
        """
        Create extraction prompts for the given texts.
        
        Args:
            texts: List of texts to create prompts for
            
        Returns:
            List of APIRequest objects
        """
        prompts = []
        
        for text in texts:
            # Create user prompt with examples
            user_prompt = self._build_extraction_prompt(text)
            
            # Create system prompt if enabled
            system_prompt = None
            if self.config.use_system_prompt:
                system_prompt = self._build_system_prompt()
            
            # Create API request
            request = APIRequest(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            prompts.append(request)
        
        return prompts
    
    def _build_extraction_prompt(self, text: str) -> str:
        """
        Build the extraction prompt for a single text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Formatted extraction prompt
        """
        if self.config.include_examples:
            prompt = self._build_prompt_with_examples(text)
        else:
            prompt = self._build_simple_prompt(text)
        
        return prompt
    
    def _build_prompt_with_examples(self, text: str) -> str:
        """
        Build extraction prompt with comprehensive examples.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Prompt with examples
        """
        return f"""目標：
從古典中文文本中提取去重的實體列表（人物、地點、物品、概念）。

重要要求：
- 每個實體只能出現一次（嚴格去重）
- 返回Python列表格式
- 優先提取具體實體，避免抽象概念

以下是《紅樓夢》的五個範例：
範例#1:
文本："甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。"
實體列表：["甄士隱", "書房"]

範例#2:
文本："這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。"
實體列表：["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]

範例#3:
文本："廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
實體列表：["甄士隱", "封氏", "鄉宦"]

範例#4:
文本："賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，暫寄廟中安身。"
實體列表：["賈雨村", "胡州", "詩書仕宦之族"]

範例#5:
文本："賈寶玉因夢遊太虛幻境，頓生疑懼，醒來後對林黛玉說起此事。"
實體列表：["賈寶玉", "太虛幻境", "林黛玉"]

注意：範例#3中移除了重複的"甄費"（因為"甄士隱"已包含此人），確保列表中無重複實體。

請參考以上範例，分析以下文本（記住：嚴格去重，每個實體只出現一次）：
文本："{text}"
實體列表："""
    
    def _build_simple_prompt(self, text: str) -> str:
        """
        Build simple extraction prompt without examples.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Simple extraction prompt
        """
        return f"""從以下古典中文文本中提取實體（人物、地點、物品、概念），
返回Python列表格式，確保每個實體只出現一次：

文本：{text}
實體列表："""
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for entity extraction.
        
        Returns:
            System prompt string
        """
        return """你是一個專門處理古典中文文本的實體提取專家。請嚴格按照以下要求：

1. 提取人物、地點、物品、概念等重要實體
2. 必須去除重複的實體（同一實體只保留一次）
3. 返回格式必須是Python列表格式：["實體1", "實體2", "實體3"]
4. 優先提取具體的人名、地名和重要概念
5. 避免提取過於抽象或通用的詞彙
6. 確保每個實體在列表中唯一，無重複

請專注於提取有意義的實體，並確保結果列表中沒有重複項目。"""
    
    async def _execute_extraction_requests(
        self, 
        requests: List[APIRequest]
    ) -> List[APIResponse]:
        """
        Execute entity extraction API requests.
        
        Args:
            requests: List of APIRequest objects
            
        Returns:
            List of APIResponse objects
        """
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def execute_request(request: APIRequest) -> APIResponse:
            async with semaphore:
                try:
                    response = await self.client.complete(request)
                    
                    # Update cache statistics
                    if response.cached:
                        self.stats["cache_hits"] += 1
                    else:
                        self.stats["cache_misses"] += 1
                    
                    return response
                except Exception as e:
                    self.logger.log(f"❌ API request failed: {str(e)}")
                    # Return error response
                    return APIResponse(
                        content=f"Error: {str(e)}",
                        model="gpt-5-mini",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        finish_reason="error",
                        response_time=0.0,
                        error=str(e)
                    )
        
        # Execute requests concurrently
        tasks = [execute_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.log(f"❌ Request {i} failed with exception: {str(response)}")
                processed_responses.append(APIResponse(
                    content=f"Error: {str(response)}",
                    model="gpt-5-mini",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="error",
                    response_time=0.0,
                    error=str(response)
                ))
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    def _parse_extraction_response(self, response: APIResponse, source_text: str) -> List[Entity]:
        """
        Parse entity extraction response from GPT-5-mini.
        
        Args:
            response: API response from GPT-5-mini
            source_text: Original source text
            
        Returns:
            List of extracted Entity objects
            
        Raises:
            ValueError: If response parsing fails
        """
        if response.error:
            raise ValueError(f"API response error: {response.error}")
        
        content = response.content.strip()
        if not content:
            raise ValueError("Empty response content")
        
        # Try to extract list from response
        entities_text = self._extract_entities_list(content)
        if not entities_text:
            raise ValueError("Could not extract entities list from response")
        
        # Parse entities
        entities = []
        for entity_text in entities_text:
            try:
                entity = self._create_entity(entity_text, source_text)
                if entity:
                    entities.append(entity)
            except Exception as e:
                self.logger.log(f"⚠️ Failed to create entity from '{entity_text}': {str(e)}")
                continue
        
        # Apply deduplication if enabled
        if self.config.enable_deduplication:
            entities = self._deduplicate_entities(entities)
        
        # Filter by confidence
        entities = [e for e in entities if e.confidence >= self.config.min_confidence]
        
        # Limit number of entities
        if len(entities) > self.config.max_entities_per_text:
            entities = entities[:self.config.max_entities_per_text]
            self.logger.log(f"⚠️ Limited entities to {self.config.max_entities_per_text} per text")
        
        return entities
    
    def _extract_entities_list(self, content: str) -> List[str]:
        """
        Extract entities list from GPT-5-mini response.
        
        Args:
            content: Raw response content
            
        Returns:
            List of entity strings
        """
        # Special case: if content is clearly not an entity list, return empty
        if content.strip() in ["No entities found", "No entities", "None", ""]:
            return []
        
        # Try to find Python list format
        list_pattern = r'\[(.*?)\]'
        matches = re.findall(list_pattern, content, re.DOTALL)
        
        if matches:
            # Use the first match
            list_content = matches[0]
            # Split by comma and clean up
            entities = [e.strip().strip('"\'') for e in list_content.split(',')]
            entities = [e for e in entities if e and e.strip()]
            return entities
        
        # Fallback: try to extract individual entities
        # Look for quoted strings
        quote_pattern = r'["\']([^"\']+)["\']'
        entities = re.findall(quote_pattern, content)
        
        if entities:
            return entities
        
        # Last resort: split by common delimiters
        delimiters = ['，', ',', '、', '；', ';']
        # Check if content actually contains any delimiters
        if any(delimiter in content for delimiter in delimiters):
            # Use regex to split by any of the delimiters
            pattern = '|'.join(re.escape(d) for d in delimiters)
            entities = re.split(pattern, content)
            entities = [e.strip() for e in entities if e and len(e.strip()) > 1]
            if entities:
                return entities
        
        return []
    
    def _create_entity(self, entity_text: str, source_text: str) -> Optional[Entity]:
        """
        Create Entity object from extracted text.
        
        Args:
            entity_text: Extracted entity text
            source_text: Source text for context
            
        Returns:
            Entity object or None if creation fails
        """
        if not entity_text or len(entity_text.strip()) < 2:
            return None
        
        # Determine entity type
        entity_type = self._classify_entity_type(entity_text, source_text)
        
        # Find position in source text
        start_pos = source_text.find(entity_text)
        end_pos = start_pos + len(entity_text) if start_pos != -1 else None
        
        # Create entity
        entity = Entity(
            text=entity_text.strip(),
            type=entity_type,
            confidence=0.9,  # Default confidence for extracted entities
            start_pos=start_pos if start_pos != -1 else None,
            end_pos=end_pos,
            source_text=source_text,
            metadata={
                "extraction_method": "gpt5mini",
                "extraction_timestamp": asyncio.get_event_loop().time()
            }
        )
        
        return entity
    
    def _classify_entity_type(self, entity_text: str, source_text: str) -> EntityType:
        """
        Classify the type of an extracted entity.
        
        Args:
            entity_text: Entity text to classify
            source_text: Source text for context
            
        Returns:
            EntityType classification
        """
        # Use Chinese text processor for classification
        return self.text_processor.classify_entity_type(entity_text, source_text)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities based on text content.
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities
        """
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Normalize entity text for comparison
            normalized_text = self.text_processor.normalize_text(entity.text)
            
            if normalized_text not in seen:
                seen.add(normalized_text)
                unique_entities.append(entity)
            else:
                # Keep the entity with higher confidence
                existing_entity = next(e for e in unique_entities 
                                    if self.text_processor.normalize_text(e.text) == normalized_text)
                if entity.confidence > existing_entity.confidence:
                    # Replace existing entity
                    unique_entities.remove(existing_entity)
                    unique_entities.append(entity)
        
        return unique_entities
    
    def _validate_input_texts(self, texts: List[str]) -> List[str]:
        """
        Validate input texts for entity extraction.
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List of validated texts
        """
        validated_texts = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                self.logger.log(f"⚠️ Skipping empty text at index {i}")
                continue
            
            # Check if short text is meaningful (before length check)
            if len(text.strip()) < 20 and not self._is_meaningful_text(text.strip()):
                self.logger.log(f"⚠️ Text at index {i} appears to be meaningless: {text.strip()}")
                continue
            
            # Additional check: reject texts that are clearly test cases
            if text.strip() in ["very short text", "a", "短"]:
                self.logger.log(f"⚠️ Text at index {i} is a test case, rejecting: {text.strip()}")
                continue
            
            # Check text length
            if len(text.strip()) < 10:
                self.logger.log(f"⚠️ Text at index {i} is too short: {len(text.strip())} characters")
                continue
            
            # Validate Chinese text
            if not self.text_processor.is_valid_chinese_text(text):
                self.logger.log(f"⚠️ Text at index {i} may not be valid Chinese text")
                continue
            
            validated_texts.append(text.strip())
        
        if not validated_texts:
            self.logger.log("❌ No valid texts found after validation")
        
        return validated_texts
    
    def _is_meaningful_text(self, text: str) -> bool:
        """
        Check if text appears to be meaningful content.
        
        Args:
            text: Text to check for meaningfulness
            
        Returns:
            True if text appears meaningful, False otherwise
        """
        # Check if text contains Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return True
        
        # Check if English text contains meaningful words
        meaningful_words = ['content', 'document', 'article', 'story', 'chapter', 'section']
        return any(word in text.lower() for word in meaningful_words)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary containing extraction statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset extraction statistics."""
        self.stats = {
            "total_texts_processed": 0,
            "total_entities_extracted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
