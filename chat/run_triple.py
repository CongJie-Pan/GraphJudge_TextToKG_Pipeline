"""
Enhanced Entity-Guided Semantic Graph Generation using GPT-5-mini Model

This is an improved version of run_triple.py with the following enhancements:
1. Structured JSON output prompts for better consistency
2. Schema validation using Pydantic
3. Text chunking for large inputs (pagination support)  
4. Integration with post-processor for comprehensive cleaning
5. Improved error handling and logging
6. Multiple output formats with quality metrics

Key improvements addressing the issues from improvement_plan1.md:
- Eliminates ~40% data loss during JSON conversion
- Standardizes relation vocabulary through post-processing
- Provides structured JSON output format
- Supports pagination to avoid truncation
- Includes comprehensive validation and metrics

Usage:
    python run_triple.py
"""

import os
import asyncio
import json
import sys
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from litellm import completion
from tqdm.asyncio import tqdm
from config import get_api_key
from openai_config import (
    OPENAI_RPM_LIMIT, OPENAI_CONCURRENT_LIMIT, OPENAI_RETRY_ATTEMPTS, 
    OPENAI_BASE_DELAY, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, GPT5_MINI_MODEL,
    calculate_rate_limit_delay, get_api_config_summary, track_token_usage, get_token_usage_stats
)

# Add tools directory to path for post-processor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

class TerminalProgressLogger:
    """
    Comprehensive terminal progress logger for GPT-5-mini triple generation pipeline.
    Captures all terminal output and saves it to a detailed progress log file.
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the progress logger.
        
        Args:
            log_dir: Directory to save progress logs (defaults to docs/Iteration_Terminal_Progress)
        """
        if log_dir is None:
            # Default to the docs/Iteration_Terminal_Progress directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, '..', 'docs', 'Iteration_Terminal_Progress')
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"run_triple_progress_{timestamp}.txt"
        self.log_filepath = os.path.join(self.log_dir, self.log_filename)
        
        # Initialize log file with header
        self._write_log_header()
        
        # Store all log entries for summary
        self.log_entries = []
        self.start_time = datetime.datetime.now()
        
        # Statistics tracking
        self.stats = {
            'total_prompts': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'valid_schema_count': 0,
            'invalid_schema_count': 0,
            'unique_triples': 0,
            'duplicates_removed': 0,
            'processing_time': 0,
            'chunked_texts': 0,
            'api_errors': 0
        }
    
    def _write_log_header(self):
        """Write the initial log header."""
        header = f"""# GPT-5-mini Triple Generation v2.0 - Terminal Progress Log
**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Pipeline**: Enhanced Triple Generation (v2.0)
**Status**: IN PROGRESS
**Log File**: {self.log_filename}

## Overview
Enhanced GPT-5-mini semantic graph generation pipeline with the following improvements:
- Structured JSON output prompts for better consistency
- Schema validation using Pydantic
- Text chunking for large inputs (pagination support)
- Integration with post-processor for comprehensive cleaning
- Improved error handling and logging
- Multiple output formats with quality metrics

## Execution Progress
"""
        with open(self.log_filepath, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log(self, message: str, level: str = "INFO", timestamp: bool = True):
        """
        Log a message to both console and file.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
            timestamp: Whether to include timestamp
        """
        # Create timestamp
        if timestamp:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{ts}] {level}: {message}"
        else:
            log_entry = f"{level}: {message}"
        
        # Store for summary
        self.log_entries.append({
            'timestamp': datetime.datetime.now(),
            'level': level,
            'message': message
        })
        
        # Write to file
        with open(self.log_filepath, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
        
        # Also print to console
        print(log_entry)
    
    def log_phase(self, phase_name: str, phase_description: str = ""):
        """Log the start of a new phase."""
        self.log("=" * 60, "PHASE", timestamp=False)
        self.log(f"PHASE: {phase_name}", "PHASE", timestamp=False)
        if phase_description:
            self.log(f"Description: {phase_description}", "PHASE", timestamp=False)
        self.log("=" * 60, "PHASE", timestamp=False)
    
    def log_statistics(self, stats_dict: Dict[str, Any], title: str = "Statistics"):
        """Log statistics in a formatted way."""
        self.log(f"\n📊 {title}:", "STATS", timestamp=False)
        for key, value in stats_dict.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            self.log(f"   {key}: {formatted_value}", "STATS", timestamp=False)
    
    def update_stats(self, **kwargs):
        """Update statistics."""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value
    
    def write_final_summary(self):
        """Write the final summary to the log file."""
        end_time = datetime.datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        summary = f"""

## Final Summary
**End Time**: {end_time.strftime("%Y-%m-%d %H:%M:%S")}
**Total Duration**: {total_duration:.2f} seconds
**Status**: COMPLETED

### Key Statistics
- Total Prompts Processed: {self.stats['total_prompts']}
- Successful API Responses: {self.stats['successful_responses']}
- Failed API Responses: {self.stats['failed_responses']}
- Success Rate: {(self.stats['successful_responses']/max(self.stats['total_prompts'], 1)*100):.1f}%
- Valid Schema Responses: {self.stats['valid_schema_count']}
- Invalid Schema Responses: {self.stats['invalid_schema_count']}
- Unique Triples Generated: {self.stats['unique_triples']}
- Duplicates Removed: {self.stats['duplicates_removed']}
- Text Chunks Created: {self.stats['chunked_texts']}
- API Errors: {self.stats['api_errors']}

### Quality Metrics
- Schema Validation Rate: {(self.stats['valid_schema_count']/max(self.stats['successful_responses'], 1)*100):.1f}%
- Triple Extraction Efficiency: {(self.stats['unique_triples']/max(self.stats['successful_responses'], 1)):.1f} triples per response
- Processing Speed: {(self.stats['total_prompts']/max(total_duration, 0.001)):.2f} prompts/second

### Files Generated
- Main JSON Output: test_instructions_context_gpt5mini_v2.json
- Backup TXT Output: test_generated_graphs_v2.txt
- Enhanced Outputs: enhanced_output/ directory (if post-processor available)
- Progress Log: {self.log_filename}

### Next Steps
1. Review enhanced processing statistics for quality assessment
2. Use enhanced post-processed outputs for Graph Judge evaluation
3. Compare results with Graph Iteration 1 for improvements
4. Analyze relation mapping statistics to expand vocabulary

---
**Log File**: {self.log_filename}
**Generated**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(self.log_filepath, 'a', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n📝 Progress log saved to: {self.log_filepath}")
    
    def get_log_filepath(self) -> str:
        """Get the path to the log file."""
        return self.log_filepath

try:
    from parse_gpt5mini_triples import GPT5MiniTripleParser
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    print("⚠️ Warning: Post-processor not available. Install tools/parse_gpt5mini_triples.py")

# Pydantic imports for schema validation
try:
    from pydantic import BaseModel, ValidationError
    from typing import List as TypingList
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️ Warning: Pydantic not available. Install with: pip install pydantic")

# API configuration - Load OpenAI API key for GPT-5-mini
if __name__ == "__main__":
    # Initialize logger for API configuration
    api_logger = TerminalProgressLogger()
    try:
        api_key = get_api_key()
        os.environ['OPENAI_API_KEY'] = api_key
        api_logger.log(f"✓ OpenAI API key loaded for GPT-5-mini", "SUCCESS")
        get_api_config_summary()
    except ValueError as e:
        api_logger.log(f"✗ API configuration error: {e}", "ERROR")
        api_logger.log("Please ensure OPENAI_API_KEY is set in your .env file", "ERROR")
        exit(1)

# Pydantic models for schema validation
if PYDANTIC_AVAILABLE:
    class Triple(BaseModel):
        """Pydantic model for triple validation."""
        subject: str
        relation: str
        object: str
        
        class Config:
            str_strip_whitespace = True
    
    class TripleResponse(BaseModel):
        """Pydantic model for GPT-5-mini response validation."""
        triples: TypingList[TypingList[str]]
        
        def validate_structure(self) -> bool:
            """Validate that all triples have exactly 3 components."""
            for triple in self.triples:
                if len(triple) != 3:
                    return False
            return True

# Configuration for text chunking and pagination
MAX_TOKENS_PER_CHUNK = 1000  # Maximum tokens per text chunk
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context

# Initialize data containers for text and entity information
text = []  # Will store denoised classical Chinese text data
entity = []  # Will store corresponding entity lists

# Dataset configuration and iteration tracking
dataset = "DreamOf_RedChamber"  # Primary dataset for processing Dream of the Red Chamber
dataset_path = os.environ.get('PIPELINE_DATASET_PATH', f'../datasets/GPT5Mini_result_{dataset}/')  # Path to dataset files with GPT5Mini prefix
Input_Iteration = int(os.environ.get('PIPELINE_INPUT_ITERATION', '2'))  # Iteration number for input files (denoised text and entities)
Graph_Iteration = int(os.environ.get('PIPELINE_GRAPH_ITERATION', '2'))  # Iteration number for enhanced graph generation output

# Load denoised text from previous GPT-5-mini processing stage
if __name__ == "__main__":
    # Initialize logger for data loading
    data_logger = TerminalProgressLogger()
    
    # Use centralized path resolver to ensure consistency with run_entity.py
    from path_resolver import resolve_pipeline_output, load_manifest, log_path_diagnostics
    
    # First try to load from manifest (highest priority)
    input_dir = None
    manifest = None
    
    # Check if PIPELINE_OUTPUT_DIR points to a directory with manifest
    if 'PIPELINE_OUTPUT_DIR' in os.environ:
        potential_dir = os.environ['PIPELINE_OUTPUT_DIR']
        manifest = load_manifest(potential_dir)
        if manifest and manifest.get('stage') == 'ectd':
            input_dir = potential_dir
            data_logger.log(f"📋 Using manifest from PIPELINE_OUTPUT_DIR: {input_dir}", "SUCCESS")
    
    # Fallback to path resolver
    if input_dir is None:
        input_dir = resolve_pipeline_output(Input_Iteration, create=False)
        manifest = load_manifest(input_dir)
        if manifest:
            data_logger.log(f"📋 Found manifest in resolved directory: {input_dir}", "SUCCESS")
        else:
            data_logger.log(f"⚠️ No manifest found, using resolved directory: {input_dir}", "WARNING")
    
    log_path_diagnostics("triple_generation", Input_Iteration, input_dir)
    data_logger.log(f"🔍 Using input directory: {os.path.abspath(input_dir)}")  # Debug: show absolute path
    
    try:
        denoised_file = os.path.join(input_dir, "test_denoised.target")
        with open(denoised_file, 'r', encoding='utf-8') as f:
            text = [l.strip() for l in f.readlines()]
        data_logger.log(f"✓ Loaded {len(text)} denoised text segments from: {denoised_file}", "SUCCESS")
    except FileNotFoundError:
        data_logger.log(f"✗ Error: Could not find denoised text file at {denoised_file}", "ERROR")
        data_logger.log("Please ensure the GPT-5-mini ECTD pipeline has been run successfully", "ERROR")
        exit(1)

    try:
        entity_file = os.path.join(input_dir, "test_entity.txt")
        with open(entity_file, 'r', encoding='utf-8') as f:
            entity = [l.strip() for l in f.readlines()]
        data_logger.log(f"✓ Loaded {len(entity)} entity sets from: {entity_file}", "SUCCESS")
    except FileNotFoundError:
        data_logger.log(f"✗ Error: Could not find entity file at {entity_file}", "ERROR")
        data_logger.log("Please ensure the GPT-5-mini ECTD pipeline has been run successfully", "ERROR")
        exit(1)

    # Validate that we have matching numbers of texts and entities
    if len(text) != len(entity):
        data_logger.log(f"⚠️ Warning: Mismatch between text segments ({len(text)}) and entity sets ({len(entity)})", "WARNING")
        min_size = min(len(text), len(entity))
        text = text[:min_size]
        entity = entity[:min_size]
        data_logger.log(f"Adjusted to process {min_size} matching pairs", "INFO")

def validate_response_schema(response: str) -> Optional[Dict[str, Any]]:
    """
    Validate GPT-5-mini response against expected JSON schema using Pydantic.
    
    Args:
        response (str): Raw response from GPT-5-mini
    
    Returns:
        Dict containing validated triples or None if validation fails
    """
    if not PYDANTIC_AVAILABLE:
        return extract_json_from_response(response)
    
    try:
        # Clean the response to extract JSON
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
            validated = TripleResponse(**data)
            if validated.validate_structure():
                return data
        elif isinstance(data, list):
            # Handle legacy format - convert to new format
            validated = TripleResponse(triples=data)
            if validated.validate_structure():
                return {'triples': data}
        
        return None
        
    except ValidationError as e:
        print(f"⚠️ Schema validation failed: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Response validation error: {e}")
        return None

def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON content from GPT-5-mini response with improved pattern matching.
    
    Args:
        response (str): Raw response from GPT-5-mini
    
    Returns:
        JSON string or None if not found
    """
    response_str = str(response).strip()
    
    # Look for structured JSON format first
    import re
    
    # Pattern 1: Look for {"triples": [...]} format
    json_object_pattern = r'\{\s*"triples"\s*:\s*\[.*?\]\s*\}'
    matches = re.findall(json_object_pattern, response_str, re.DOTALL)
    if matches:
        return matches[0]
    
    # Pattern 2: Look for [[...]] array format
    json_array_pattern = r'\[\[.*?\]\]'
    matches = re.findall(json_array_pattern, response_str, re.DOTALL)
    if matches:
        return matches[0]
    
    # Pattern 3: Look for simple array format
    simple_array_pattern = r'\[.*?\]'
    matches = re.findall(simple_array_pattern, response_str, re.DOTALL)
    if matches:
        # Take the longest match
        longest_match = max(matches, key=len)
        if '[[' in longest_match or longest_match.count('[') > 1:
            return longest_match
    
    return None

def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks to handle large texts.
    
    Args:
        text (str): Input text to chunk
        max_tokens (int): Maximum tokens per chunk
        overlap (int): Overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Simple character-based chunking (can be improved with proper tokenization)
    chars_per_token = 2  # Rough estimate for Chinese text
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap * chars_per_token
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        # Try to break at sentence boundaries
        if end < len(text):
            for punct in ['。', '！', '？', '；']:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start + max_chars // 2:
                    end = last_punct + 1
                    break
        
        chunks.append(text[start:end])
        
        if end >= len(text):
            break
        
        start = end - overlap_chars
    
    return chunks

def create_enhanced_prompt(text_content: str, entity_list: str) -> str:
    """
    Create enhanced prompt with structured JSON output requirements.
    
    Args:
        text_content: Chinese text to process
        entity_list: Entity list string
        
    Returns:
        Formatted prompt string
    """
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
實體列表：{entity_list}

請按照上述格式要求輸出JSON："""

async def openai_api_call(prompt, system_prompt=None, **kwargs) -> str:
    """
    Enhanced asynchronous function to interact with GPT-5-mini via LiteLLM with intelligent rate limiting.
    
    Args:
        prompt (str): The user prompt/question to send to the model
        system_prompt (str, optional): System message to set model behavior  
        **kwargs: Additional parameters for the API call
    
    Returns:
        str: The generated response from the GPT-5-mini model
        
    This function handles communication with GPT-5-mini optimized for OpenAI's higher rate limits.
    """
    # Estimate token usage before making the call
    estimated_tokens = min(len(prompt) // 2, OPENAI_MAX_TOKENS)
    if system_prompt:
        estimated_tokens += len(system_prompt) // 2
    
    # Check token limits before making the call
    if not track_token_usage(estimated_tokens):
        token_stats = get_token_usage_stats()
        print(f"⚠️ Token limit would be exceeded. Current usage: {token_stats['minute_tokens']:,}/min, {token_stats['day_tokens']:,}/day")
        # Wait until next minute if TPM limit would be exceeded
        if token_stats['minute_remaining'] < estimated_tokens:
            print("Waiting for TPM reset...")
            await asyncio.sleep(60)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    max_retries = OPENAI_RETRY_ATTEMPTS
    retry_count = 0
    base_wait_time = calculate_rate_limit_delay()
    
    while retry_count < max_retries:
        try:
            response = completion(
                model=GPT5_MINI_MODEL,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
                max_completion_tokens=OPENAI_MAX_TOKENS,
                **kwargs
            )
            
            # Track actual token usage (approximate)
            response_content = response.choices[0].message.content
            actual_tokens = len(response_content) // 2 + estimated_tokens
            track_token_usage(actual_tokens - estimated_tokens)  # Adjust for actual usage
            
            return response_content
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            print(f"API call attempt {retry_count}/{max_retries} failed: {error_msg}")
            
            # Intelligent retry strategy based on error type
            if "RateLimitError" in error_msg or "RPM" in error_msg or "rate limit" in error_msg.lower():
                # Rate limit error: Progressive delay with jitter
                wait_time = base_wait_time * (1.5 ** retry_count)
                jitter = wait_time * 0.1 * (2 * hash(prompt) % 100 / 100 - 1)  # ±10% jitter
                wait_time = int(wait_time + jitter)
                print(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                
            elif "overloaded" in error_msg.lower() or "busy" in error_msg.lower():
                # Server overload: Longer exponential backoff
                wait_time = min(60 * (2 ** retry_count), 300)  # Cap at 5 minutes
                print(f"Server overloaded. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                
            elif "timeout" in error_msg.lower():
                # Timeout error: Moderate delay
                wait_time = min(10 * (1.5 ** retry_count), 60)
                print(f"Timeout detected. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                
            else:
                # Other errors: Standard exponential backoff
                wait_time = min(5 * (2 ** retry_count), 30)
                print(f"Other error detected. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            
            if retry_count >= max_retries:
                print(f"Failed to get response after {max_retries} attempts")
                return "Error: Could not get response from GPT-5-mini"

async def _run_api(prompts, max_concurrent=1):
    """
    Enhanced API runner optimized for free tier with sequential processing and intelligent rate limiting.
    
    Args:
        prompts (list): List of prompts to process
        max_concurrent (int): Force to 1 for free tier compliance
    
    Returns:
        list: List of responses corresponding to each prompt
    """
    # Force sequential processing for free tier
    max_concurrent = 1
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Track progress and timing
    results = []
    start_time = asyncio.get_event_loop().time()
    
    async def limited_openai_call(prompt, index):
        """Enhanced inner function with progressive delay and comprehensive monitoring."""
        async with semaphore:
            # Progressive delay based on position to spread out requests
            base_delay = calculate_rate_limit_delay()
            progressive_delay = base_delay + (index * 1)  # Add 1 second per request for GPT-5-mini (higher rate limits)
            
            if index > 0:  # Skip delay for first request
                print(f"⏳ Waiting {progressive_delay}s before processing pair {index + 1}/{len(prompts)}")
                await asyncio.sleep(progressive_delay)
            
            # Log token usage stats every 3 requests
            if index % 3 == 0:
                token_stats = get_token_usage_stats()
                print(f"📊 Token usage: {token_stats['minute_tokens']:,}/min ({token_stats['minute_percentage']:.1f}%), {token_stats['day_tokens']:,}/day ({token_stats['day_percentage']:.1f}%)")
            
            print(f"🔄 Processing text-entity pair {index + 1}/{len(prompts)}")
            
            try:
                response = await openai_api_call(prompt)
                response_length = len(str(response))
                print(f"✅ Completed pair {index + 1}/{len(prompts)} - Response length: {response_length} chars")
                return response
            except Exception as e:
                print(f"❌ Failed pair {index + 1}/{len(prompts)} - Error: {str(e)[:100]}...")
                return f"Error: {str(e)}"
    
    print(f"🚀 Starting OpenAI optimized batch processing of {len(prompts)} text-entity pairs...")
    print(f"📊 Configuration: Sequential processing (concurrency=1), RPM limit = {OPENAI_RPM_LIMIT}")
    print(f"⏱️  Base delay between requests: {calculate_rate_limit_delay():.2f} seconds")
    print(f"🔧 Progressive delay: +1s per request for enhanced rate limit compliance")
    print("-" * 80)
    
    # Process prompts sequentially with enhanced monitoring
    with tqdm(total=len(prompts), desc=f"OpenAI GPT-5-mini Graph Generation") as pbar:
        for i, prompt in enumerate(prompts):
            try:
                result = await limited_openai_call(prompt, i)
                results.append(result)
                pbar.update(1)
                
                # Add extra safety pause every 2 requests
                if (i + 1) % 2 == 0 and i < len(prompts) - 1:
                    extra_delay = 5  # Reduced delay for OpenAI's higher rate limits
                    print(f"🛡️  Safety pause: waiting {extra_delay}s after every 2 triple generation requests")
                    await asyncio.sleep(extra_delay)
                    
            except Exception as e:
                print(f"❌ Error processing prompt {i + 1}: {str(e)}")
                results.append(f"Error: {str(e)}")
                pbar.update(1)
    
    elapsed_time = asyncio.get_event_loop().time() - start_time
    print("-" * 80)
    print(f"🎯 OpenAI batch processing completed in {elapsed_time:.1f} seconds!")
    print(f"📈 Average time per request: {elapsed_time/max(len(prompts), 1):.1f}s")
    print(f"📊 Successfully processed {len([r for r in results if not r.startswith('Error:')])} out of {len(prompts)} requests")
    
    return results

async def main():
    """
    Enhanced main execution function with comprehensive improvements.
    """
    # Initialize progress logger
    logger = TerminalProgressLogger()
    
    logger.log_phase("Enhanced GPT-5-mini Semantic Graph Generation Pipeline v2.0", 
                    f"Processing Dream of the Red Chamber - Graph Iteration {Graph_Iteration}")
    logger.log("Features: Structured JSON + Schema Validation + Pagination + Post-Processing", "INFO", timestamp=False)
    
    # Validate input data
    if not text or len(text) == 0:
        logger.log("✗ Error: No denoised text found. Please run GPT-5-mini ECTD pipeline first.", "ERROR")
        return
    
    if not entity or len(entity) == 0:
        logger.log("✗ Error: No entity data found. Please run GPT-5-mini ECTD pipeline first.", "ERROR")
        return
    
    logger.log_phase("PHASE 1: Enhanced Input Data Processing", "Validate and prepare input data")
    
    # Log input data statistics
    input_stats = {
        "Text segments": len(text),
        "Entity sets": len(entity),
        "Processing pairs": min(len(text), len(entity)),
        "Enhanced features": "Chunking, Validation, Post-processing"
    }
    logger.log_statistics(input_stats, "Input Data Validation")
    
    # Show sample data for verification
    logger.log("📋 Sample Data Preview:", "INFO", timestamp=False)
    logger.log(f"   📄 Sample text (first 100 chars): {text[0][:100]}...", "INFO", timestamp=False)
    logger.log(f"   🏷️  Sample entities: {entity[0]}", "INFO", timestamp=False)
    logger.log(f"   📊 Dataset path: {dataset_path}", "INFO", timestamp=False)
    logger.log(f"   🔢 Input iteration: {Input_Iteration}", "INFO", timestamp=False)
    logger.log(f"   🎯 Enhanced graph iteration: {Graph_Iteration}", "INFO", timestamp=False)
    
    prompts = []
    chunk_info = []  # Track chunking information
    
    # Create enhanced prompts with chunking support
    logger.log_phase("PHASE 2: Enhanced Prompt Generation with Pagination", "Create structured prompts with chunking support")
    logger.log(f"📝 Creating enhanced prompts for {len(text)} text-entity pairs...", "INFO")
    
    for i in range(len(text)):
        if (i + 1) % 5 == 0 or i == 0 or i == len(text) - 1:
            logger.log(f"   📄 Processing text {i + 1}/{len(text)}", "INFO")
        
        current_text = text[i]
        current_entities = entity[i]
        
        # Check if text needs chunking
        text_chunks = chunk_text(current_text)
        
        if len(text_chunks) > 1:
            logger.log(f"   🔄 Text {i+1} split into {len(text_chunks)} chunks for better processing", "INFO")
            
            # Create prompts for each chunk
            for j, chunk in enumerate(text_chunks):
                enhanced_prompt = create_enhanced_prompt(chunk, current_entities)
                prompts.append(enhanced_prompt)
                chunk_info.append({
                    'original_index': i,
                    'chunk_index': j,
                    'total_chunks': len(text_chunks),
                    'is_chunked': True
                })
        else:
            # Single prompt for normal-sized text
            enhanced_prompt = create_enhanced_prompt(current_text, current_entities)
            prompts.append(enhanced_prompt)
            chunk_info.append({
                'original_index': i,
                'chunk_index': 0,
                'total_chunks': 1,
                'is_chunked': False
            })
    
    # Update statistics
    chunked_texts = sum(1 for info in chunk_info if info['is_chunked'])
    logger.update_stats(total_prompts=len(prompts), chunked_texts=chunked_texts)
    
    # Log prompt generation results
    prompt_stats = {
        "Total prompts created": len(prompts),
        "Average prompt length": f"{sum(len(p) for p in prompts) / max(len(prompts), 1):.0f} characters",
        "Text chunks created": chunked_texts,
        "Original texts": len(text)
    }
    logger.log_statistics(prompt_stats, "Prompt Generation Results")

    # Create output directory
    # Support output directory override through environment variables for pipeline integration
    output_dir = os.environ.get('PIPELINE_OUTPUT_DIR', dataset_path + f"Graph_Iteration{Graph_Iteration}")
    os.makedirs(output_dir, exist_ok=True)
    logger.log(f"📁 Created enhanced output directory: {output_dir}", "SUCCESS")
    logger.log(f"📊 Using Input Iteration: {Input_Iteration}, Graph Iteration: {Graph_Iteration}", "INFO")

    # Process all prompts with enhanced API handling
    logger.log_phase("PHASE 3: Enhanced Semantic Graph Generation", "Process prompts with GPT-5-mini API")
    
    api_config = {
        "GPT-5-mini model": GPT5_MINI_MODEL,
        "Temperature setting": OPENAI_TEMPERATURE,
        "Max tokens per response": OPENAI_MAX_TOKENS,
        "Total prompts to process": len(prompts)
    }
    logger.log_statistics(api_config, "API Configuration")
    
    start_time = asyncio.get_event_loop().time()
    responses = await _run_api(prompts)
    end_time = asyncio.get_event_loop().time()
    
    processing_time = end_time - start_time
    logger.update_stats(processing_time=processing_time)
    
    # Log processing time statistics
    time_stats = {
        "Total processing time": f"{processing_time:.2f} seconds",
        "Average time per prompt": f"{processing_time/max(len(prompts), 1):.2f} seconds",
        "Processing speed": f"{len(prompts)/max(processing_time, 0.001):.2f} prompts/second"
    }
    logger.log_statistics(time_stats, "Processing Time Statistics")

    # Enhanced response validation with schema checking
    logger.log_phase("PHASE 4: Enhanced Response Analysis & Schema Validation", "Validate and analyze API responses")
    
    successful_generations = sum(1 for r in responses if "Error:" not in str(r))
    failed_generations = len(responses) - successful_generations
    
    # Update statistics
    logger.update_stats(
        successful_responses=successful_generations,
        failed_responses=failed_generations
    )
    
    # Schema validation statistics
    valid_schema_count = 0
    invalid_schema_count = 0
    
    logger.log("🔍 Performing enhanced schema validation on responses...", "INFO")
    validated_responses = []
    
    for i, response in enumerate(responses):
        if "Error:" not in str(response):
            validated = validate_response_schema(str(response))
            if validated:
                valid_schema_count += 1
                validated_responses.append(validated)
            else:
                invalid_schema_count += 1
                validated_responses.append(None)
        else:
            validated_responses.append(None)
    
    # Update schema validation statistics
    logger.update_stats(
        valid_schema_count=valid_schema_count,
        invalid_schema_count=invalid_schema_count
    )
    
    # Log response generation statistics
    response_stats = {
        "Successful generations": f"{successful_generations}/{len(responses)}",
        "Failed generations": f"{failed_generations}/{len(responses)}",
        "Success rate": f"{successful_generations/max(len(responses), 1)*100:.1f}%"
    }
    logger.log_statistics(response_stats, "Response Generation Statistics")
    
    if PYDANTIC_AVAILABLE:
        schema_stats = {
            "Valid schema responses": f"{valid_schema_count}/{successful_generations}",
            "Invalid schema responses": f"{invalid_schema_count}/{successful_generations}",
            "Schema validation rate": f"{valid_schema_count/max(successful_generations, 1)*100:.1f}%"
        }
        logger.log_statistics(schema_stats, "Schema Validation Statistics")
    
    # Analyze response quality
    response_lengths = [len(str(r)) for r in responses if "Error:" not in str(r)]
    if response_lengths:
        avg_response_length = sum(response_lengths) / len(response_lengths)
        min_response_length = min(response_lengths)
        max_response_length = max(response_lengths)
        
        quality_stats = {
            "Average response length": f"{avg_response_length:.0f} characters",
            "Minimum response length": f"{min_response_length} characters",
            "Maximum response length": f"{max_response_length} characters"
        }
        logger.log_statistics(quality_stats, "Response Quality Analysis")

    # Save results with enhanced post-processing
    logger.log_phase("PHASE 5: Enhanced Results Saving & Post-Processing", "Save results and apply post-processing")
    
    # Generate JSON output format like test_instructions_context_gpt5mini.json
    output_file_path = os.path.join(output_dir, "test_instructions_context_gpt5mini_v2.json")
    backup_txt_path = os.path.join(output_dir, "test_generated_graphs_v2.txt")
    
    logger.log(f"📁 Saving enhanced semantic graphs in JSON format to: {output_file_path}", "INFO")
    logger.log(f"📁 Backup TXT format to: {backup_txt_path}", "INFO")
    
    # Save original responses as backup TXT file
    with open(backup_txt_path, "w", encoding='utf-8') as txt_file:
        for i, response in enumerate(responses):
            original_response = str(response).strip().replace('\n', '')
            txt_file.write(original_response + '\n')
    
    # Convert responses to JSON format like test_instructions_context_gpt5mini.json
    json_instructions = []
    
    for i, response in enumerate(responses):
        if "Error:" not in str(response):
            # Extract triples from response and convert to instructions
            validated = validate_response_schema(str(response))
            if validated and 'triples' in validated:
                triples = validated['triples']
                for triple in triples:
                    if len(triple) == 3:
                        subject, relation, obj = triple
                        instruction = {
                            "instruction": f"Is this true: {subject} {relation} {obj} ?",
                            "input": "",
                            "output": ""
                        }
                        json_instructions.append(instruction)
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(responses) - 1:
            logger.log(f"   💾 Processed response {i + 1}/{len(responses)}", "INFO")
    
    # Save JSON instructions file
    with open(output_file_path, "w", encoding='utf-8') as json_file:
        import json
        json.dump(json_instructions, json_file, ensure_ascii=False, indent=2)
    
    logger.log(f"✅ Generated {len(json_instructions)} JSON instructions", "SUCCESS")
    
    # Apply enhanced post-processing with the new parser
    if PARSER_AVAILABLE:
        logger.log_phase("PHASE 6: Advanced Post-Processing & Quality Enhancement", "Apply comprehensive cleaning and validation")
        logger.log("🚀 Running enhanced GPT-5-mini Triple Parser for comprehensive cleaning...", "INFO")
        
        try:
            # Initialize enhanced parser
            parser = GPT5MiniTripleParser()
            
            # Parse the backup TXT file with enhanced processing
            unique_triples = parser.parse_file(backup_txt_path)
            
            if unique_triples:
                # Save multiple enhanced output formats
                post_processed_dir = os.path.join(output_dir, "enhanced_output")
                output_files = parser.save_outputs(unique_triples, post_processed_dir)
                
                logger.log(f"✅ Enhanced post-processing completed successfully!", "SUCCESS")
                logger.log(f"📊 Generated {len(unique_triples)} unique, validated triples", "SUCCESS")
                
                # Update statistics
                logger.update_stats(unique_triples=len(unique_triples))
                
                # Log enhanced output files
                logger.log("📁 Enhanced output files:", "INFO", timestamp=False)
                for format_name, file_path in output_files.items():
                    logger.log(f"   {format_name}: {file_path}", "INFO", timestamp=False)
                
                # Get and display enhanced parser statistics
                parser_stats = parser.get_stats()
                duplicates_removed = parser_stats['parsing_stats'].get('duplicates_removed', 0)
                valid_triples = parser_stats['parsing_stats'].get('valid_triples', 0)
                relation_mappings = len(parser_stats['relation_mapping_stats'])//2
                
                # Update statistics
                logger.update_stats(duplicates_removed=duplicates_removed)
                
                post_processing_stats = {
                    "Schema validation rate": f"{valid_schema_count/max(successful_generations,1)*100:.1f}%",
                    "Duplicates removed": duplicates_removed,
                    "Valid triples extracted": valid_triples,
                    "Relation mappings applied": relation_mappings
                }
                logger.log_statistics(post_processing_stats, "Enhanced Post-Processing Statistics")
                
            else:
                logger.log(f"⚠️ Enhanced post-processing found no valid triples", "WARNING")
                
        except Exception as e:
            logger.log(f"⚠️ Enhanced post-processing failed: {e}", "ERROR")
            logger.log(f"📁 Original TXT output available at: {backup_txt_path}", "INFO")
    else:
        logger.log(f"\n⚠️ Enhanced post-processor not available - using basic output format", "WARNING")
        logger.log(f"📁 JSON output available at: {output_file_path}", "INFO")
        logger.log(f"📁 TXT backup available at: {backup_txt_path}", "INFO")

    # Final summary with enhancement metrics
    logger.log_phase("Final Summary", "Enhanced GPT-5-mini Semantic Graph Generation Completed")
    logger.log(f"📂 Results available in: {output_dir}", "SUCCESS")
    logger.log(f"📁 Main enhanced output: {output_file_path}", "SUCCESS")
    
    if PARSER_AVAILABLE and 'post_processed_dir' in locals():
        logger.log(f"🔧 Enhanced outputs: {post_processed_dir}", "SUCCESS")
        logger.log(f"💡 Recommended: Use enhanced outputs for best quality", "INFO")
    
    logger.log("🔄 Next Steps:", "INFO", timestamp=False)
    logger.log("   1. Review enhanced processing statistics for quality assessment", "INFO", timestamp=False)
    logger.log("   2. Use enhanced post-processed outputs for Graph Judge evaluation", "INFO", timestamp=False)
    logger.log("   3. Compare results with Graph Iteration 1 for improvements", "INFO", timestamp=False)
    logger.log("   4. Analyze relation mapping statistics to expand vocabulary", "INFO", timestamp=False)
    
    # Final enhancement summary
    final_stats = {
        "Successful API responses": f"{successful_generations}/{len(responses)}",
        "Failed API responses": f"{failed_generations}/{len(responses)}",
        "Overall success rate": f"{successful_generations/max(len(responses), 1)*100:.1f}%"
    }
    
    if PYDANTIC_AVAILABLE:
        final_stats["Valid JSON schema"] = f"{valid_schema_count}/{successful_generations}"
    
    logger.log_statistics(final_stats, "Final Enhancement Summary")
    
    # Calculate and display file sizes
    file_size = os.path.getsize(output_file_path)
    file_stats = {
        "Main output file size": f"{file_size:,} bytes ({file_size/1024:.1f} KB)"
    }
    
    if PARSER_AVAILABLE and 'post_processed_dir' in locals():
        try:
            post_processed_size = sum(os.path.getsize(os.path.join(post_processed_dir, f)) 
                                    for f in os.listdir(post_processed_dir) if os.path.isfile(os.path.join(post_processed_dir, f)))
            file_stats["Enhanced files size"] = f"{post_processed_size:,} bytes ({post_processed_size/1024:.1f} KB)"
        except:
            pass
    
    logger.log_statistics(file_stats, "File Size Information")
    
    # Enhancement tips
    if not PYDANTIC_AVAILABLE:
        logger.log("💡 Enhancement Tip: Install pydantic for better schema validation", "INFO")
    if not PARSER_AVAILABLE:
        logger.log("💡 Enhancement Tip: Ensure tools/parse_gpt5mini_triples.py is available for enhanced processing", "INFO")
    
    # Write final summary to log file
    logger.write_final_summary()

def validate_prerequisites():
    """Enhanced prerequisite validation with comprehensive checks."""
    # Check API configuration
    try:
        get_api_key()
    except ValueError as e:
        print(f"✗ API configuration error: {e}")
        return False
    
    # Support environment variable override for pipeline integration - consistent with run_entity.py
    input_dir = resolve_pipeline_output(Input_Iteration, create=False)
    
    # Check input files existence
    denoised_file = os.path.join(input_dir, "test_denoised.target")
    entity_file = os.path.join(input_dir, "test_entity.txt")
    
    if not os.path.exists(denoised_file):
        print(f"✗ Denoised text file not found: {denoised_file}")
        print("Please run the GPT-5-mini ECTD pipeline first")
        return False
    
    if not os.path.exists(entity_file):
        print(f"✗ Entity file not found: {entity_file}")
        print("Please run the GPT-5-mini ECTD pipeline first")
        return False
    
    # Check if output directory can be created
    try:
        output_dir = os.environ.get('PIPELINE_OUTPUT_DIR', dataset_path + f"Graph_Iteration{Graph_Iteration}")
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"✗ Cannot create output directory: {e}")
        return False
    
    return True

# Enhanced entry point with comprehensive validation
if __name__ == "__main__":
    # Initialize logger for entry point
    entry_logger = TerminalProgressLogger()
    entry_logger.log("🚀 Starting Enhanced GPT-5-mini Semantic Graph Generation Pipeline v2.0...", "INFO")
    
    # Enhanced prerequisite validation
    if not validate_prerequisites():
        entry_logger.log("❌ Enhanced prerequisites validation failed. Please fix the issues and try again.", "ERROR")
        exit(1)
    
    # Run the enhanced pipeline
    try:
        asyncio.run(main())
        entry_logger.log("\n✅ Enhanced pipeline execution completed successfully!", "SUCCESS")
    except KeyboardInterrupt:
        entry_logger.log("\n⚠️ Enhanced pipeline interrupted by user", "WARNING")
    except Exception as e:
        entry_logger.log(f"\n❌ Critical error during enhanced pipeline execution: {e}", "ERROR")
        entry_logger.log("Please check your configuration and try again.", "ERROR")
        exit(1)
