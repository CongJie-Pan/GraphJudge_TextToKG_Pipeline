"""
Text Denoising Engine Module

This module implements the core text denoising logic for classical Chinese texts,
using GPT-5-mini to restructure text based on extracted entities.

The module provides a clean interface for text denoising with comprehensive
error handling, progress tracking, and result validation.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from extractEntity_Phase.models.entities import Entity, EntityList
from extractEntity_Phase.api.gpt5mini_client import GPT5MiniClient, APIRequest, APIResponse
from extractEntity_Phase.infrastructure.logging import get_logger
from extractEntity_Phase.utils.chinese_text import ChineseTextProcessor
from extractEntity_Phase.models.pipeline_state import PipelineStage, ProcessingStatus


@dataclass
class DenoisingConfig:
    """Configuration for text denoising."""
    
    # GPT-5-mini settings
    temperature: float = 1.0
    max_tokens: int = 4000
    timeout: int = 60
    
    # Denoising settings
    preserve_classical_style: bool = True
    maintain_factual_accuracy: bool = True
    enable_entity_relationships: bool = True
    min_output_length: int = 20
    max_output_length: int = 1000
    
    # Prompt engineering
    use_system_prompt: bool = True
    include_examples: bool = True
    language: str = "zh-TW"  # Traditional Chinese
    
    # Processing settings
    batch_size: int = 10
    max_concurrent: int = 3


class TextDenoiser:
    """
    Core text denoising engine for classical Chinese texts.
    
    This class implements the main logic for denoising and restructuring Chinese text
    using GPT-5-mini, based on extracted entities while preserving classical style.
    """
    
    def __init__(self, client: GPT5MiniClient, config: Optional[DenoisingConfig] = None):
        """
        Initialize the text denoiser.
        
        Args:
            client: GPT-5-mini API client
            config: Denoising configuration
        """
        self.client = client
        self.config = config or DenoisingConfig()
        self.logger = get_logger()
        self.text_processor = ChineseTextProcessor()
        
        # Statistics tracking
        self.stats = {
            "total_texts_processed": 0,
            "total_texts_denoised": 0,
            "successful_denoising": 0,
            "failed_denoising": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_compression_ratio": 0.0
        }
    
    async def denoise_texts(
        self, 
        texts: List[str],
        entities_list: List[Union[List[str], EntityList]],
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Denoise and restructure a list of Chinese texts based on extracted entities.
        
        Args:
            texts: List of original Chinese text strings
            entities_list: List of entity lists or EntityList objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of denoised and restructured texts
        """
        self.logger.log(f"🧹 Starting text denoising for {len(texts)} text segments...")
        
        if not texts or not entities_list:
            self.logger.log("⚠️ No texts or entities provided for denoising")
            return []
        
        if len(texts) != len(entities_list):
            self.logger.log("❌ Mismatch between number of texts and entity lists")
            return []
        
        # Validate inputs
        validated_pairs = self._validate_input_pairs(texts, entities_list)
        if not validated_pairs:
            self.logger.log("❌ Input validation failed")
            return []
        
        # Process texts in batches
        results = []
        total_batches = (len(validated_pairs) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(validated_pairs))
            batch_pairs = validated_pairs[start_idx:end_idx]
            
            self.logger.log(f"📦 Processing denoising batch {batch_idx + 1}/{total_batches} "
                          f"(texts {start_idx + 1}-{end_idx})")
            
            # Denoise batch
            batch_results = await self._denoise_texts_batch(batch_pairs)
            results.extend(batch_results)
            
            # Update progress
            if progress_callback:
                progress = (batch_idx + 1) / total_batches
                progress_callback(progress, f"Denoised batch {batch_idx + 1}/{total_batches}")
            
            # Add delay between batches to respect rate limits
            if batch_idx < total_batches - 1:
                await asyncio.sleep(0.5)
        
        # Update statistics
        self._update_statistics(texts, results)
        
        self.logger.log(f"✅ Text denoising completed. "
                       f"Processed {len(texts)} texts, "
                       f"successful: {self.stats['successful_denoising']}")
        
        return results
    
    async def _denoise_texts_batch(
        self, 
        text_entity_pairs: List[Tuple[str, Union[List[str], EntityList]]]
    ) -> List[str]:
        """
        Denoise a batch of text-entity pairs.
        
        Args:
            text_entity_pairs: List of (text, entities) tuples
            
        Returns:
            List of denoised texts
        """
        # Create denoising prompts
        prompts = self._create_denoising_prompts(text_entity_pairs)
        
        # Execute API calls
        responses = await self._execute_denoising_requests(prompts)
        
        # Parse and validate responses
        results = []
        for (text, entities), response in zip(text_entity_pairs, responses):
            try:
                denoised_text = self._parse_denoising_response(response, text, entities)
                if denoised_text:
                    results.append(denoised_text)
                else:
                    # Fallback to original text if denoising fails
                    results.append(text)
                    self.logger.log(f"⚠️ Denoising failed for text, using original")
            except Exception as e:
                self.logger.log(f"❌ Failed to parse denoising response: {str(e)}")
                # Use original text as fallback
                results.append(text)
        
        return results
    
    def _create_denoising_prompts(
        self, 
        text_entity_pairs: List[Tuple[str, Union[List[str], EntityList]]]
    ) -> List[APIRequest]:
        """
        Create denoising prompts for the given text-entity pairs.
        
        Args:
            text_entity_pairs: List of (text, entities) tuples
            
        Returns:
            List of APIRequest objects
        """
        prompts = []
        
        for text, entities in text_entity_pairs:
            # Convert entities to list of strings if needed
            entity_strings = self._extract_entity_strings(entities)
            
            # Create user prompt
            user_prompt = self._build_denoising_prompt(text, entity_strings)
            
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
    
    def _extract_entity_strings(self, entities: Union[List[str], EntityList]) -> List[str]:
        """
        Extract entity strings from various entity formats.
        
        Args:
            entities: Entity list or EntityList object
            
        Returns:
            List of entity strings
        """
        if isinstance(entities, EntityList):
            return [entity.text for entity in entities.entities]
        elif isinstance(entities, list):
            # Handle list of strings or Entity objects
            entity_strings = []
            for entity in entities:
                if isinstance(entity, str):
                    entity_strings.append(entity)
                elif isinstance(entity, Entity):
                    entity_strings.append(entity.text)
                else:
                    entity_strings.append(str(entity))
            return entity_strings
        else:
            return [str(entities)]
    
    def _build_denoising_prompt(self, text: str, entities: List[str]) -> str:
        """
        Build denoising prompt for a single text-entity pair.
        
        Args:
            text: Original text to denoise
            entities: List of extracted entities
            
        Returns:
            Formatted denoising prompt
        """
        if self.config.include_examples:
            prompt = self._build_prompt_with_examples(text, entities)
        else:
            prompt = self._build_simple_prompt(text, entities)
        
        return prompt
    
    def _build_prompt_with_examples(self, text: str, entities: List[str]) -> str:
        """
        Build denoising prompt with comprehensive examples.
        
        Args:
            text: Original text to denoise
            entities: List of extracted entities
            
        Returns:
            Prompt with examples
        """
        return f"""目標：
基於給定的實體，對古典中文文本進行去噪處理，即移除無關的描述性文字並重組為清晰的事實陳述。

以下是《紅樓夢》的三個範例：
範例#1:
原始文本："廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。家中雖不甚富貴，然本地便也推他為望族了。"
實體：["甄費", "甄士隱", "封氏", "鄉宦"]
去噪文本："甄士隱是一家鄉宦。甄士隱姓甄名費字士隱。甄士隱的妻子是封氏。封氏情性賢淑深明禮義。甄家是本地望族。"

範例#2:
原始文本："賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，父母祖宗根基已盡，人口衰喪，只剩得他一身一口，在家鄉無益，因進京求取功名，再整基業。"
實體：["賈雨村", "胡州", "詩書仕宦之族"]
去噪文本："賈雨村是胡州人氏。賈雨村是詩書仕宦之族。賈雨村生於末世。賈雨村父母祖宗根基已盡。賈雨村進京求取功名。賈雨村想要重整基業。"

範例#3:
原始文本："賈寶玉因夢遊太虛幻境，頓生疑懼，醒來後心中不安，遂將此事告知林黛玉，黛玉聽後亦感驚異。"
實體：["賈寶玉", "太虛幻境", "林黛玉"]
去噪文本："賈寶玉夢遊太虛幻境。賈寶玉夢醒後頓生疑懼。賈寶玉將此事告知林黛玉。林黛玉聽後感到驚異。"

請參考以上範例，處理以下文本：
原始文本：{text}
實體：{entities}
去噪文本："""
    
    def _build_simple_prompt(self, text: str, entities: List[str]) -> str:
        """
        Build simple denoising prompt without examples.
        
        Args:
            text: Original text to denoise
            entities: List of extracted entities
            
        Returns:
            Simple denoising prompt
        """
        return f"""基於給定的實體，對以下古典中文文本進行去噪處理，
移除無關的描述性文字並重組為清晰的事實陳述：

原始文本：{text}
實體：{entities}
去噪文本："""
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for text denoising.
        
        Returns:
            System prompt string
        """
        return """你是一個專門處理古典中文文本的去噪專家。請嚴格按照以下要求：

1. 基於給定的實體，對文本進行去噪處理
2. 移除無關的描述性文字和修飾語
3. 重組為清晰、簡潔的事實陳述
4. 保持古典中文的語言風格和韻味
5. 確保每個陳述都基於給定的實體
6. 避免添加原文中沒有的信息
7. 使用簡潔的句式，每個事實用一句話表達

請專注於提取和重組事實信息，保持文本的準確性和可讀性。"""
    
    async def _execute_denoising_requests(
        self, 
        requests: List[APIRequest]
    ) -> List[APIResponse]:
        """
        Execute text denoising API requests.
        
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
    
    def _parse_denoising_response(
        self, 
        response: APIResponse, 
        original_text: str, 
        entities: Union[List[str], EntityList]
    ) -> Optional[str]:
        """
        Parse text denoising response from GPT-5-mini.
        
        Args:
            response: API response from GPT-5-mini
            original_text: Original source text
            entities: Extracted entities used for denoising
            
        Returns:
            Denoised text string or None if parsing fails
            
        Raises:
            ValueError: If response parsing fails
        """
        if response.error:
            raise ValueError(f"API response error: {response.error}")
        
        content = response.content.strip()
        if not content:
            raise ValueError("Empty response content")
        
        # Clean and validate denoised text
        denoised_text = self._clean_denoised_text(content)
        
        # Validate denoised text quality
        if not self._validate_denoised_text(denoised_text, original_text, entities):
            self.logger.log(f"⚠️ Denoised text validation failed, using original")
            return None
        
        return denoised_text
    
    def _clean_denoised_text(self, content: str) -> str:
        """
        Clean and format denoised text.
        
        Args:
            content: Raw denoised text content
            
        Returns:
            Cleaned denoised text
        """
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            "去噪文本：", "Denoised text:", "Result:", "Answer:", "A:"
        ]
        suffixes_to_remove = [
            "。", ".", "！", "！", "？", "?", "\n", "\r"
        ]
        
        cleaned_text = content.strip()
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Remove trailing punctuation and whitespace
        for suffix in suffixes_to_remove:
            if cleaned_text.endswith(suffix):
                cleaned_text = cleaned_text[:-len(suffix)].strip()
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def _validate_denoised_text(
        self, 
        denoised_text: str, 
        original_text: str, 
        entities: Union[List[str], EntityList]
    ) -> bool:
        """
        Validate the quality of denoised text.
        
        Args:
            denoised_text: Denoised text to validate
            original_text: Original source text
            entities: Extracted entities
            
        Returns:
            True if denoised text is valid, False otherwise
        """
        if not denoised_text or len(denoised_text.strip()) < self.config.min_output_length:
            return False
        
        if len(denoised_text) > self.config.max_output_length:
            return False
        
        # Check if denoised text contains key entities
        entity_strings = self._extract_entity_strings(entities)
        entity_coverage = sum(1 for entity in entity_strings if entity in denoised_text)
        
        if entity_coverage < max(1, len(entity_strings) * 0.5):  # At least 50% entity coverage
            return False
        
        # Check if denoised text is significantly different from original
        if self._calculate_similarity(denoised_text, original_text) > 0.8:
            return False
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple character-based similarity
        chars1 = set(text1)
        chars2 = set(text2)
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def _validate_input_pairs(
        self, 
        texts: List[str], 
        entities_list: List[Union[List[str], EntityList]]
    ) -> List[Tuple[str, Union[List[str], EntityList]]]:
        """
        Validate input text-entity pairs.
        
        Args:
            texts: List of texts to validate
            entities_list: List of entity lists or EntityList objects
            
        Returns:
            List of validated (text, entities) tuples
        """
        validated_pairs = []
        
        for i, (text, entities) in enumerate(zip(texts, entities_list)):
            if not text or not text.strip():
                self.logger.log(f"⚠️ Skipping empty text at index {i}")
                continue
            
            # Check text length
            if len(text.strip()) < 10:
                self.logger.log(f"⚠️ Text at index {i} is too short: {len(text.strip())} characters")
                continue
            
            # Validate Chinese text
            if not self.text_processor.is_valid_chinese_text(text):
                self.logger.log(f"⚠️ Text at index {i} may not be valid Chinese text")
                continue
            
            # Validate entities
            entity_strings = self._extract_entity_strings(entities)
            if not entity_strings:
                self.logger.log(f"⚠️ No valid entities found for text at index {i}")
                continue
            
            validated_pairs.append((text.strip(), entities))
        
        if not validated_pairs:
            self.logger.log("❌ No valid text-entity pairs found after validation")
        
        return validated_pairs
    
    def _update_statistics(self, original_texts: List[str], denoised_texts: List[str]):
        """
        Update denoising statistics.
        
        Args:
            original_texts: List of original texts
            denoised_texts: List of denoised texts
        """
        self.stats["total_texts_processed"] = len(original_texts)
        self.stats["total_texts_denoised"] = len(denoised_texts)
        self.stats["successful_denoising"] = sum(1 for text in denoised_texts if text)
        self.stats["failed_denoising"] = len(original_texts) - self.stats["successful_denoising"]
        
        # Calculate average compression ratio
        if original_texts and denoised_texts:
            total_original_length = sum(len(text) for text in original_texts)
            total_denoised_length = sum(len(text) for text in denoised_texts if text)
            
            if total_original_length > 0:
                self.stats["average_compression_ratio"] = (
                    total_denoised_length / total_original_length
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get denoising statistics.
        
        Returns:
            Dictionary containing denoising statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset denoising statistics."""
        self.stats = {
            "total_texts_processed": 0,
            "total_texts_denoised": 0,
            "successful_denoising": 0,
            "failed_denoising": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_compression_ratio": 0.0
        }
