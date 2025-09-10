"""
Entity Extraction and Text Denoising Pipeline using GPT-5-mini Model via LiteLLM

This script implements a two-stage text processing pipeline optimized for Chinese text processing:
1. Entity Extraction: Identifies and extracts key entities from classical Chinese text using GPT-5-mini
2. Text Denoising: Restructures and cleans the original text based on extracted entities

The pipeline is specifically designed for Dream of the Red Chamber (紅樓夢) text processing,
leveraging GPT-5-mini's advanced Chinese language understanding and reasoning capabilities.

Key improvements over previous versions:
- Enhanced Chinese text understanding with GPT-5-mini model
- Optimized prompts for classical Chinese literature
- Better entity recognition for Chinese names and locations
- Improved denoising for traditional Chinese narrative style

Main workflow:
1. Extract entities from input text using GPT-5-mini via LiteLLM
2. Use extracted entities to denoise and reformat the original text
3. Support for iterative processing to progressively improve text quality
4. Save intermediate results for analysis and further processing
5. Log all terminal progress to timestamped files for tracking and debugging
"""

import os
import asyncio
import json
import hashlib
import sys
import datetime
from pathlib import Path
from litellm import completion
from tqdm.asyncio import tqdm
from config import get_api_key
from openai_config import (
    OPENAI_RPM_LIMIT, OPENAI_CONCURRENT_LIMIT, OPENAI_RETRY_ATTEMPTS, 
    OPENAI_BASE_DELAY, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, GPT5_MINI_MODEL,
    calculate_rate_limit_delay, get_api_config_summary, track_token_usage, get_token_usage_stats
)

class TerminalLogger:
    """
    Custom logger class that captures all terminal output and saves it to a file.
    
    This class implements a dual-output system that displays messages on the terminal
    while simultaneously writing them to a timestamped log file for permanent record.
    """
    
    def __init__(self, log_dir="../docs/Iteration_Terminal_Progress"):
        """
        Initialize the terminal logger with output file setup.
        
        Args:
            log_dir (str): Directory path where log files will be stored
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.log_dir / f"gpt5mini_entity_iteration_{timestamp}.txt"
        
        # Open log file for writing
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        
        # Write header to log file
        header = f"""
{'='*80}
GPT-5-mini Entity Extraction and Text Denoising Pipeline - Terminal Progress Log
Started at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Log File: {self.log_file_path}
{'='*80}
"""
        self.log_file.write(header)
        self.log_file.flush()
        
        print(f"[LOG] Terminal progress will be logged to: {self.log_file_path}")
    
    def log(self, message):
        """
        Log a message to both terminal and file.
        
        Args:
            message (str): Message to log
        """
        # Print to terminal
        print(message)
        
        # Write to log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_file.write(log_entry)
        self.log_file.flush()  # Ensure immediate write to disk
    
    def close(self):
        """
        Close the log file and write final summary.
        """
        if hasattr(self, 'log_file') and self.log_file:
            footer = f"""
{'='*80}
Pipeline completed at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Log file closed successfully.
{'='*80}
"""
            self.log_file.write(footer)
            self.log_file.close()
            print(f"[LOG] Terminal progress log saved to: {self.log_file_path}")

class NullLogger:
    """
    Null logger implementation for testing and import scenarios.
    
    This class provides the same interface as TerminalLogger but does nothing,
    preventing AttributeError when logger is used in testing environments.
    """
    
    def log(self, message):
        """Log method that does nothing."""
        pass
    
    def close(self):
        """Close method that does nothing."""
        pass

def get_logger():
    """
    Get or create a logger instance.
    
    Returns:
        TerminalLogger or NullLogger: A logger instance appropriate for the current context
    """
    global logger
    if logger is None:
        # In testing or import context, use null logger
        if not (__name__ == "__main__"):
            logger = NullLogger()
        else:
            logger = TerminalLogger()
    return logger

# Global logger instance
logger = None

# API configuration - Load OpenAI API key for GPT-5-mini
# Only initialize if running as main module
if __name__ == "__main__":
    # Initialize logger first
    logger = TerminalLogger()
    
    try:
        api_key = get_api_key()
        os.environ['OPENAI_API_KEY'] = api_key
        get_logger().log(f"✓ OpenAI API key loaded for GPT-5-mini")
        # Log configuration summary using logger instead of print
        config = get_api_config_summary()
        get_logger().log("=== GPT-5-mini API Configuration ===")
        get_logger().log(f"RPM Limit: {config['rpm_limit']}")
        get_logger().log(f"Concurrent Requests: {config['concurrent_limit']}")
        get_logger().log(f"Retry Attempts: {config['retry_attempts']}")
        get_logger().log(f"Base Delay: {config['base_delay']}s")
        get_logger().log(f"Calculated Delay: {config['calculated_delay']}s")
        get_logger().log(f"Temperature: {config['temperature']}")
        get_logger().log(f"Max Tokens: {config['max_tokens']}")
        get_logger().log(f"Model: {config['model']}")
        get_logger().log("=" * 35)
    except ValueError as e:
        get_logger().log(f"✗ API configuration error: {e}")
        get_logger().log("Please ensure OPENAI_API_KEY is set in your .env file")
        get_logger().close()
        exit(1)

# Dataset configuration and path setup
dataset = "DreamOf_RedChamber"  # Primary dataset for processing Dream of the Red Chamber
dataset_path = os.environ.get('PIPELINE_DATASET_PATH', f'../datasets/GPT5mini_result_{dataset}/')  # Path to dataset files with GPT5mini prefix
Iteration = int(os.environ.get('PIPELINE_ITERATION', '1'))  # Current iteration number for iterative processing

# Cache configuration for improved performance and reduced API calls
CACHE_DIR = Path(".cache/gpt5mini_ent")  # Directory for caching GPT-5-mini API responses
CACHE_ENABLED = True  # Enable/disable caching system
cache_hits = 0  # Track cache hit statistics
cache_misses = 0  # Track cache miss statistics

# Input data loading based on iteration number
# For first iteration, use original target file; for subsequent iterations, use previous denoised results
# Only load data if running as main module
text = []
if __name__ == "__main__":
    if Iteration == 1:
        # Load original text data for first iteration from chapter1_raw.txt
        try:
            with open('../datasets/DreamOf_RedChamber/chapter1_raw.txt', 'r', encoding='utf-8') as f:
                # Split into sentences/paragraphs for processing
                content = f.read().strip()
                # Split by double newlines and filter out empty strings and title lines
                text = [line.strip() for line in content.split('\n') 
                        if line.strip() and not line.strip().startswith('第一回') 
                        and not line.strip() == '紅樓夢' and len(line.strip()) > 20]
                get_logger().log(f"✓ Loaded {len(text)} text segments from chapter1_raw.txt")
        except FileNotFoundError:
            get_logger().log("✗ Error: ../datasets/DreamOf_RedChamber/chapter1_raw.txt not found")
            get_logger().log("Please ensure the input file exists before running the pipeline")
            get_logger().close()
            exit(1)
    else:
        # Load previously denoised text for subsequent iterations
        with open(dataset_path + f'Graph_Iteration{Iteration - 1}/test_denoised.target', 'r', encoding='utf-8') as f:
            text = [l.strip() for l in f.readlines()]
        get_logger().log(f"✓ Loaded {len(text)} text segments from previous iteration")

def setup_cache_directory():
    """
    Set up the cache directory structure for storing GPT-5-mini API responses.
    
    This function creates the necessary cache directory and ensures proper
    permissions for reading and writing cached responses to disk.
    """
    if CACHE_ENABLED:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        get_logger().log(f"📁 Cache directory initialized: {CACHE_DIR}")

def generate_cache_key(prompt: str, system_prompt: str = None, **kwargs) -> str:
    """
    Generate a unique cache key for API requests based on prompt content and parameters.
    
    Args:
        prompt (str): User prompt content
        system_prompt (str, optional): System prompt content
        **kwargs: Additional API parameters
        
    Returns:
        str: SHA256 hash as cache key
        
    This function creates a unique identifier for each API request by hashing
    the combination of prompt content and API parameters, ensuring that identical
    requests can be retrieved from cache without making duplicate API calls.
    """
    # Combine all inputs that affect the API response
    cache_content = {
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "model": GPT5_MINI_MODEL,
        **kwargs
    }
    
    # Create deterministic hash from content
    content_str = json.dumps(cache_content, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha256(content_str.encode('utf-8'))
    return hash_obj.hexdigest()

def load_from_cache(cache_key: str) -> str:
    """
    Load cached API response from disk if available.
    
    Args:
        cache_key (str): Unique cache key for the request
        
    Returns:
        str: Cached response content, or None if not found
        
    This function attempts to retrieve a previously cached API response
    from disk, reducing the need for redundant API calls and improving
    performance for repeated requests.
    """
    global cache_hits, cache_misses
    
    if not CACHE_ENABLED:
        cache_misses += 1
        return None
        
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cache_hits += 1
                return cache_data.get('response', None)
    except (json.JSONDecodeError, KeyError, IOError) as e:
        get_logger().log(f"⚠️  Cache read error for key {cache_key[:8]}...: {e}")
    
    cache_misses += 1
    return None

def save_to_cache(cache_key: str, prompt: str, response: str, system_prompt: str = None):
    """
    Save API response to disk cache for future retrieval.
    
    Args:
        cache_key (str): Unique cache key for the request
        prompt (str): Original user prompt
        response (str): API response content
        system_prompt (str, optional): System prompt if used
        
    This function stores successful API responses to disk cache with metadata
    including timestamps and request details for debugging and analysis.
    """
    if not CACHE_ENABLED:
        return
        
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        cache_data = {
            "response": response,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "timestamp": asyncio.get_event_loop().time(),
            "model": GPT5_MINI_MODEL,
            "temperature": OPENAI_TEMPERATURE
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
    except IOError as e:
        get_logger().log(f"⚠️  Cache write error for key {cache_key[:8]}...: {e}")

async def openai_api_call(prompt, system_prompt=None, **kwargs) -> str:
    """
    Enhanced asynchronous function to interact with GPT-5-mini via LiteLLM with intelligent rate limiting.
    
    Args:
        prompt (str): The user prompt/question to send to the model
        system_prompt (str, optional): System message to set model behavior  
        **kwargs: Additional parameters for the API call
    
    Returns:
        str: The generated response from the GPT-5-mini model
        
    This function handles the communication with GPT-5-mini using LiteLLM with:
    - Integrated disk caching to reduce API calls
    - Token usage tracking for TPM/TPD limits
    - Intelligent retry strategies for different error types
    - Enhanced rate limit handling for OpenAI API
    """
    # Generate cache key for this request
    cache_key = generate_cache_key(prompt, system_prompt, **kwargs)
    
    # Try to load from cache first
    cached_response = load_from_cache(cache_key)
    if cached_response:
        get_logger().log(f"📦 Cache hit for key {cache_key[:8]}...")
        return cached_response
    
    get_logger().log(f"🌐 Cache miss, making API call for key {cache_key[:8]}...")
    
    # Estimate token usage before making the call
    estimated_tokens = min(len(prompt) // 2, OPENAI_MAX_TOKENS)  # Rough estimate
    if system_prompt:
        estimated_tokens += len(system_prompt) // 2
    
    # Check token limits before making the call
    if not track_token_usage(estimated_tokens):
        token_stats = get_token_usage_stats()
        get_logger().log(f"⚠️ Token limit would be exceeded. Current usage: {token_stats['minute_tokens']:,}/min, {token_stats['day_tokens']:,}/day")
        # Wait until next minute if TPM limit would be exceeded
        if token_stats['minute_remaining'] < estimated_tokens:
            get_logger().log("Waiting for TPM reset...")
            await asyncio.sleep(60)
    
    # Build the conversation messages in the required format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Enhanced retry logic with intelligent backoff strategies
    max_retries = OPENAI_RETRY_ATTEMPTS
    retry_count = 0
    base_wait_time = calculate_rate_limit_delay()
    
    while retry_count < max_retries:
        try:
            # Make API call with GPT-5-mini model optimized for Chinese text
            response = completion(
                model=GPT5_MINI_MODEL,  # GPT-5-mini for superior Chinese understanding
                messages=messages,
                max_completion_tokens=OPENAI_MAX_TOKENS,  # GPT-5 models use max_completion_tokens
                **kwargs
            )
            
            # Extract the generated content
            response_content = response.choices[0].message.content
            
            # Track actual token usage (approximate)
            actual_tokens = len(response_content) // 2 + estimated_tokens
            track_token_usage(actual_tokens - estimated_tokens)  # Adjust for actual usage
            
            # Save successful response to cache
            save_to_cache(cache_key, prompt, response_content, system_prompt)
            get_logger().log(f"💾 Response cached for key {cache_key[:8]}...")
            
            return response_content
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            get_logger().log(f"API call attempt {retry_count}/{max_retries} failed: {error_msg}")
            
            # Intelligent retry strategy based on error type
            if "RateLimitError" in error_msg or "rate_limit_exceeded" in error_msg.lower() or "rate limit" in error_msg.lower():
                # Rate limit error: Progressive delay with jitter
                wait_time = base_wait_time * (1.5 ** retry_count)
                jitter = wait_time * 0.1 * (2 * hash(cache_key) % 100 / 100 - 1)  # ±10% jitter
                wait_time = int(wait_time + jitter)
                get_logger().log(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                
            elif "overloaded" in error_msg.lower() or "busy" in error_msg.lower() or "capacity" in error_msg.lower():
                # Server overload: Longer exponential backoff
                wait_time = min(60 * (2 ** retry_count), 300)  # Cap at 5 minutes
                get_logger().log(f"Server overloaded. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                
            elif "timeout" in error_msg.lower():
                # Timeout error: Moderate delay
                wait_time = min(10 * (1.5 ** retry_count), 60)
                get_logger().log(f"Timeout detected. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                
            else:
                # Other errors: Standard exponential backoff
                wait_time = min(5 * (2 ** retry_count), 30)
                get_logger().log(f"Other error detected. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            
            if retry_count >= max_retries:
                get_logger().log(f"Failed to get response after {max_retries} attempts")
                return "Error: Could not get response from GPT-5-mini"

async def _run_api(queries, max_concurrent=OPENAI_CONCURRENT_LIMIT):
    """
    Execute multiple API calls with enhanced rate limiting for free tier users.
    
    Args:
        queries (list): List of prompts to process
        max_concurrent (int): Maximum number of concurrent API calls (default: from OPENAI_CONCURRENT_LIMIT)
    
    Returns:
        list: List of responses corresponding to each query
    
    This function implements enhanced rate limiting optimized for free tier:
    - Forced sequential processing (concurrency = 1)
    - Progressive delays to avoid rate limits
    - Token usage monitoring
    """
    # Use configured concurrency limit for better performance
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Track progress and timing
    results = []
    start_time = asyncio.get_event_loop().time()
    
    async def limited_openai_call(query, index):
        """Enhanced inner function with progressive delay and monitoring."""
        async with semaphore:
            # Progressive delay based on position to spread out requests
            base_delay = calculate_rate_limit_delay()
            progressive_delay = base_delay + (index * 0.5)  # Reduced delay for higher RPM
            
            if index > 0:  # Skip delay for first request
                get_logger().log(f"Waiting {progressive_delay}s before request {index + 1}/{len(queries)}")
                await asyncio.sleep(progressive_delay)
            
            # Log token usage stats periodically
            if index % 10 == 0:
                token_stats = get_token_usage_stats()
                get_logger().log(f"Token usage: {token_stats['minute_tokens']:,}/min ({token_stats['minute_percentage']:.1f}%), {token_stats['day_tokens']:,}/day ({token_stats['day_percentage']:.1f}%)")
            
            return await openai_api_call(query)

    # Process queries sequentially with enhanced monitoring
    get_logger().log(f"Starting sequential processing of {len(queries)} queries with enhanced rate limiting...")
    
    with tqdm(total=len(queries), desc=f"Processing with GPT-5-mini ({OPENAI_RPM_LIMIT} RPM limit)") as pbar:
        for i, query in enumerate(queries):
            try:
                result = await limited_openai_call(query, i)
                results.append(result)
                pbar.update(1)
                
                # Add extra delay every 10 requests for safety with higher RPM
                if (i + 1) % 10 == 0 and i < len(queries) - 1:
                    extra_delay = 3
                    get_logger().log(f"Safety pause: waiting {extra_delay}s after every 10 requests")
                    await asyncio.sleep(extra_delay)
                    
            except Exception as e:
                get_logger().log(f"Error processing query {i + 1}: {str(e)}")
                results.append(f"Error: {str(e)}")
                pbar.update(1)
    
    elapsed_time = asyncio.get_event_loop().time() - start_time
    get_logger().log(f"Batch processing completed in {elapsed_time:.1f} seconds")
    
    return results

async def _run_api_with_system(prompt_system_pairs, max_concurrent=OPENAI_CONCURRENT_LIMIT):
    """
    Execute multiple API calls with system prompts and enhanced rate limiting for free tier.
    
    Args:
        prompt_system_pairs (list): List of (user_prompt, system_prompt) tuples
        max_concurrent (int): Maximum number of concurrent API calls (default: from OPENAI_CONCURRENT_LIMIT)
    
    Returns:
        list: List of responses corresponding to each prompt pair
    
    This function implements enhanced rate limiting optimized for free tier with system prompts.
    """
    # Use configured concurrency limit for better performance
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Track progress and timing
    results = []
    start_time = asyncio.get_event_loop().time()

    async def limited_openai_call_with_system(prompt_system_pair, index):
        """Enhanced inner function with progressive delay and system prompt support."""
        async with semaphore:
            # Progressive delay based on position
            base_delay = calculate_rate_limit_delay()
            progressive_delay = base_delay + (index * 0.5)  # Reduced delay for higher RPM
            
            if index > 0: # Skip delay for first request
                get_logger().log(f"Waiting {progressive_delay}s before system prompt request {index + 1}/{len(prompt_system_pairs)}")
                await asyncio.sleep(progressive_delay)
            
            # Log token usage stats periodically
            if index % 10 == 0:
                token_stats = get_token_usage_stats()
                get_logger().log(f"Token usage: {token_stats['minute_tokens']:,}/min ({token_stats['minute_percentage']:.1f}%), {token_stats['day_tokens']:,}/day ({token_stats['day_percentage']:.1f}%)")
            
            user_prompt, system_prompt = prompt_system_pair
            return await openai_api_call(user_prompt, system_prompt=system_prompt)

    # Process prompt pairs sequentially with enhanced monitoring
    get_logger().log(f"Starting sequential processing of {len(prompt_system_pairs)} system prompt queries...")
    
    with tqdm(total=len(prompt_system_pairs), desc=f"Processing with GPT-5-mini + System Prompts ({OPENAI_RPM_LIMIT} RPM limit)") as pbar:
        for i, pair in enumerate(prompt_system_pairs):
            try:
                result = await limited_openai_call_with_system(pair, i)
                results.append(result)
                pbar.update(1)
                
                # Add extra delay every 10 requests for safety with higher RPM
                if (i + 1) % 10 == 0 and i < len(prompt_system_pairs) - 1:
                    extra_delay = 3
                    get_logger().log(f"Safety pause: waiting {extra_delay}s after every 10 system prompt requests")
                    await asyncio.sleep(extra_delay)
                    
            except Exception as e:
                get_logger().log(f"Error processing system prompt query {i + 1}: {str(e)}")
                results.append(f"Error: {str(e)}")
                pbar.update(1)
    
    elapsed_time = asyncio.get_event_loop().time() - start_time
    get_logger().log(f"System prompt batch processing completed in {elapsed_time:.1f} seconds")
    
    return results

async def extract_entities(texts):
    """
    Extract entities from classical Chinese texts using GPT-5-mini's enhanced Chinese understanding.
    
    Args:
        texts (list): List of classical Chinese text strings to process
    
    Returns:
        list: List of entity lists corresponding to each input text
    
    This function leverages GPT-5-mini's advanced Chinese language processing to identify
    key entities (characters, places, organizations, concepts) from classical Chinese text.
    The prompts are optimized for Dream of the Red Chamber content and include contextual
    examples to guide accurate entity extraction with deduplication.
    """
    # System prompt to guide entity extraction behavior with deduplication emphasis
    system_prompt = """你是一個專門處理古典中文文本的實體提取專家。請嚴格按照以下要求：

1. 提取人物、地點、物品、概念等重要實體
2. 必須去除重複的實體（同一實體只保留一次）
3. 返回格式必須是Python列表格式：["實體1", "實體2", "實體3"]
4. 優先提取具體的人名、地名和重要概念
5. 避免提取過於抽象或通用的詞彙
6. 確保每個實體在列表中唯一，無重複

請專注於提取有意義的實體，並確保結果列表中沒有重複項目。"""

    prompts = []
    
    # Create entity extraction prompts optimized for classical Chinese with deduplication emphasis
    for t in texts:
        # Enhanced prompt with more comprehensive examples and deduplication emphasis
        prompt = f"""
目標：
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
文本："{t}"
實體列表："""
        prompts.append((prompt, system_prompt))
    
    # Process all entity extraction prompts concurrently with system prompts
    get_logger().log(f"🔍 Extracting deduplicated entities from {len(texts)} text segments...")
    entities_list = await _run_api_with_system(prompts)
    return entities_list

async def denoise_text(texts, entities_list):
    """
    Denoise and restructure classical Chinese text using GPT-5-mini's advanced understanding.
    
    Args:
        texts (list): List of original classical Chinese text strings
        entities_list (list): List of entity lists corresponding to each text
    
    Returns:
        list: List of denoised and restructured texts
    
    This function leverages GPT-5-mini's deep understanding of Chinese literature to create
    cleaner, more structured versions of classical Chinese text. The denoising process:
    1. Removes irrelevant or redundant descriptive elements
    2. Reformats text into clear, factual statements
    3. Preserves classical Chinese style while improving clarity
    4. Focuses on relationships between identified entities
    """
    prompts = []
    
    # Create denoising prompts optimized for classical Chinese literature
    for t, entities in zip(texts, entities_list):
        # Enhanced prompt with classical Chinese denoising examples
        prompt = f"""
目標：
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
原始文本：{t}
實體：{entities}
去噪文本："""
        prompts.append(prompt)
    
    # Process all denoising prompts concurrently
    get_logger().log(f"🧹 Denoising {len(texts)} text segments...")
    denoised_texts = await _run_api(prompts)
    return denoised_texts

async def main():
    """
    Main execution function that orchestrates the complete GPT-5-mini based ECTD pipeline.
    
    This function:
    1. Sets up caching system for improved performance
    2. Creates necessary output directories with GPT5mini prefix
    3. Extracts entities from input texts using GPT-5-mini and saves them
    4. Loads and validates the extracted entities
    5. Uses GPT-5-mini to denoise the original texts based on entities
    6. Saves the denoised texts for further processing
    7. Provides comprehensive logging and cache statistics
    """
    get_logger().log("=" * 60)
    get_logger().log("🎯 GPT-5-mini ECTD Pipeline - Entity Extraction & Text Denoising")
    get_logger().log(f"📖 Processing Dream of the Red Chamber - Iteration {Iteration}")
    get_logger().log("=" * 60)
    
    # Step 0: Initialize cache system for improved performance
    setup_cache_directory()
    global cache_hits, cache_misses
    cache_hits = 0
    cache_misses = 0
    
    # Step 1: Create output directory if it doesn't exist
    # Use centralized path resolver to ensure consistency with validation
    from path_resolver import resolve_pipeline_output, write_manifest, log_path_diagnostics
    
    output_dir = resolve_pipeline_output(Iteration, create=True)
    log_path_diagnostics("ectd", Iteration, output_dir)
    get_logger().log(f"📁 Resolved output directory: {output_dir}")
    get_logger().log(f"📊 Using Iteration: {Iteration}, Dataset path: {dataset_path}")
    get_logger().log(f"🔍 Full output path: {os.path.abspath(output_dir)}")  # Debug: show absolute path
    
    # Validate input data
    if not text or len(text) == 0:
        get_logger().log("✗ Error: No input text found. Please check the input file.")
        return
    
    # Step 1: Extract entities from all input texts using GPT-5-mini
    get_logger().log(f"🔍 Processing {len(text)} text segments for entity extraction...")
    entities_list = await extract_entities(text)
    
    # Validate entity extraction results
    successful_extractions = sum(1 for e in entities_list if "Error:" not in str(e))
    get_logger().log(f"📊 Entity extraction statistics:")
    get_logger().log(f"   - Successful extractions: {successful_extractions}/{len(entities_list)}")
    get_logger().log(f"   - Success rate: {successful_extractions/len(entities_list)*100:.1f}%")
    
    # Save extracted entities to file for inspection and potential manual correction
    entity_file = os.path.join(output_dir, "test_entity.txt")
    with open(entity_file, "w", encoding='utf-8') as output_file:
        for entities in entities_list:
            # Clean and save each entity list (remove newlines for consistent formatting)
            output_file.write(str(entities).strip().replace('\n', '') + '\n')
        # Ensure data is written to disk immediately
        output_file.flush()
        os.fsync(output_file.fileno())
    get_logger().log(f"✓ Entities saved to: {entity_file}")
    
    # Verify the file was actually written with content
    if os.path.exists(entity_file) and os.path.getsize(entity_file) > 0:
        get_logger().log(f"✓ Entity file verification passed: {os.path.getsize(entity_file)} bytes")
    else:
        get_logger().log(f"❌ Entity file verification failed: file missing or empty")
        raise RuntimeError(f"Failed to write entity file: {entity_file}")
    
    # Save denoised texts to output file for use in subsequent processing steps
    denoised_file = os.path.join(output_dir, "test_denoised.target")
    with open(denoised_file, "w", encoding='utf-8') as output_file:
        for denoised_text in denoised_texts:
            # Clean and save each denoised text (remove excessive newlines)
            output_file.write(str(denoised_text).strip().replace('\n\n', '\n') + '\n')
        # Ensure data is written to disk immediately
        output_file.flush()
        os.fsync(output_file.fileno())
    get_logger().log(f"✓ Denoised texts saved to: {denoised_file}")
    
    # Verify the denoised file was actually written with content
    if os.path.exists(denoised_file) and os.path.getsize(denoised_file) > 0:
        get_logger().log(f"✓ Denoised file verification passed: {os.path.getsize(denoised_file)} bytes")
    else:
        get_logger().log(f"❌ Denoised file verification failed: file missing or empty")
        raise RuntimeError(f"Failed to write denoised file: {denoised_file}")
    
    # Write path manifest for stage validation and next stage consumption
    created_files = ["test_entity.txt", "test_denoised.target"]
    manifest_metadata = {
        "successful_extractions": successful_extractions,
        "total_texts": len(entities_list),
        "successful_denoising": successful_denoising,
        "entity_file_size": os.path.getsize(entity_file),
        "denoised_file_size": os.path.getsize(denoised_file)
    }
    
    try:
        manifest_path = write_manifest(output_dir, "ectd", Iteration, created_files, manifest_metadata)
        get_logger().log(f"✓ Path manifest written to: {manifest_path}")
    except Exception as e:
        get_logger().log(f"⚠️ Warning: Could not write path manifest: {e}")
        # Continue execution - manifest is helpful but not critical
    
    # Step 2: Load the saved entities back from file for validation
    last_extracted_entities = []
    with open(entity_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            last_extracted_entities.append(l.strip())
    get_logger().log(f"✓ Loaded {len(last_extracted_entities)} entity sets for validation")
    
    # Step 3: Denoise texts using GPT-5-mini and the extracted entities
    get_logger().log("🧹 Starting text denoising with GPT-5-mini...")
    denoised_texts = await denoise_text(text, last_extracted_entities)
    
    # Validate denoising results
    successful_denoising = sum(1 for d in denoised_texts if "Error:" not in str(d))
    get_logger().log(f"📊 Text denoising statistics:")
    get_logger().log(f"   - Successful denoising: {successful_denoising}/{len(denoised_texts)}")
    get_logger().log(f"   - Success rate: {successful_denoising/len(denoised_texts)*100:.1f}%")
    
    # Save denoised texts to output file for use in subsequent processing steps
    denoised_file = os.path.join(output_dir, "test_denoised.target")
    with open(denoised_file, "w", encoding='utf-8') as output_file:
        for denoised_text in denoised_texts:
            # Clean and save each denoised text (remove newlines for consistent formatting)
            cleaned_text = str(denoised_text).strip().replace('\n', ' ')
            output_file.write(cleaned_text + '\n')
        # Ensure data is written to disk immediately
        output_file.flush()
        os.fsync(output_file.fileno())
    get_logger().log(f"✓ Denoised texts saved to: {denoised_file}")
    
    # Verify the file was actually written with content
    if os.path.exists(denoised_file) and os.path.getsize(denoised_file) > 0:
        get_logger().log(f"✓ Denoised file verification passed: {os.path.getsize(denoised_file)} bytes")
    else:
        get_logger().log(f"❌ Denoised file verification failed: file missing or empty")
        raise RuntimeError(f"Failed to write denoised file: {denoised_file}")
    
    # Step 4: Generate summary statistics
    avg_entities_per_text = sum(len(str(e).split(',')) for e in entities_list) / len(entities_list)
    avg_text_length = sum(len(t) for t in text) / len(text)
    avg_denoised_length = sum(len(str(d)) for d in denoised_texts) / len(denoised_texts)
    
    get_logger().log("\n📈 Processing Summary:")
    get_logger().log(f"   - Input texts processed: {len(text)}")
    get_logger().log(f"   - Average entities per text: {avg_entities_per_text:.1f}")
    get_logger().log(f"   - Average original text length: {avg_text_length:.1f} characters")
    get_logger().log(f"   - Average denoised text length: {avg_denoised_length:.1f} characters")
    get_logger().log(f"   - Compression ratio: {avg_denoised_length/avg_text_length*100:.1f}%")
    
    # Cache performance statistics
    total_requests = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    get_logger().log("\n📦 Cache Performance:")
    get_logger().log(f"   - Total API requests: {total_requests}")
    get_logger().log(f"   - Cache hits: {cache_hits}")
    get_logger().log(f"   - Cache misses: {cache_misses}")
    get_logger().log(f"   - Cache hit rate: {cache_hit_rate:.1f}%")
    get_logger().log(f"   - API calls saved: {cache_hits}")
    if CACHE_ENABLED:
        get_logger().log(f"   - Cache directory: {CACHE_DIR}")
    else:
        get_logger().log("   - Cache: Disabled")
    
    get_logger().log(f"\n🎉 GPT-5-mini ECTD pipeline completed successfully for Iteration {Iteration}!")
    get_logger().log(f"📂 Results available in: {output_dir}")
    get_logger().log("🔄 You can now run the next iteration or proceed to semantic graph generation.")

def validate_prerequisites():
    """
    Validate that all prerequisites are met before running the pipeline.
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    # Check API configuration
    try:
        get_api_key()
    except ValueError as e:
        if logger:
            get_logger().log(f"✗ API configuration error: {e}")
        return False
    
    # Check input file existence
    if Iteration == 1:
        input_file = '../datasets/DreamOf_RedChamber/chapter1_raw.txt'
        if not os.path.exists(input_file):
            if logger:
                get_logger().log(f"✗ Input file not found: {input_file}")
                get_logger().log("Please ensure the chapter1_raw.txt file exists in the datasets folder")
            return False
    
    # Check if output directory can be created
    try:
        from path_resolver import resolve_pipeline_output
        output_dir = resolve_pipeline_output(Iteration, create=True)
    except Exception as e:
        if logger:
            get_logger().log(f"✗ Cannot create output directory: {e}")
        return False
    
    return True

# Entry point: Execute the complete GPT-5-mini entity extraction and text denoising pipeline
if __name__ == "__main__":
    get_logger().log("[START] Starting GPT-5-mini ECTD Pipeline...")
    
    # Validate prerequisites before starting
    if not validate_prerequisites():
        get_logger().log("[ERROR] Prerequisites validation failed. Please fix the issues and try again.")
        get_logger().close()
        exit(1)
    
    # Run the main pipeline
    try:
        asyncio.run(main())
        get_logger().log("[SUCCESS] Pipeline execution completed successfully!")
        get_logger().close()
    except KeyboardInterrupt:
        get_logger().log("\n⚠️ Pipeline interrupted by user")
        get_logger().close()
    except Exception as e:
        get_logger().log(f"\n[ERROR] Critical error during pipeline execution: {e}")
        get_logger().log("Please check your configuration and try again.")
        get_logger().close()
        exit(1)