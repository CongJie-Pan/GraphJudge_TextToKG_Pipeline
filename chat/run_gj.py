"""
Perplexity API-based Graph Judge Implementation 

This script implements the Graph Judge functionality using Perplexity API with sonar-reasoning model.

1. Uses Perplexity's sonar-reasoning model for accurate fact checking
2. Leverages advanced reasoning capabilities for better judgment accuracy
3. Provides citation sources for graph judgment decisions
4. Better handling of classical Chinese literature knowledge (紅樓夢)
5. More robust error handling and response validation

Compatibility:
- Same JSON input format as existing scripts
- Same CSV output format for evaluation pipeline
- Maintains async processing pattern
- Compatible with existing evaluation metrics
"""

import os
import asyncio
import json
import csv
import re
import sys
import logging
import random
import argparse
from typing import Dict, List, Optional, Tuple, NamedTuple
from pathlib import Path
import tempfile
from datetime import datetime
import time

# Gold Label Bootstrapping dependencies
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
    print("✓ RapidFuzz imported successfully for gold label bootstrapping")
except ImportError:
    print("⚠️ RapidFuzz not available. Install with: pip install rapidfuzz")
    RAPIDFUZZ_AVAILABLE = False
    # Mock fuzz for testing
    class MockFuzz:
        @staticmethod
        def partial_ratio(a, b):
            return 50.0 if a != b else 100.0
    fuzz = MockFuzz()

# Perplexity API imports
try:
    from litellm import completion, acompletion
    from dotenv import load_dotenv
    load_dotenv()
    PERPLEXITY_AVAILABLE = True
    print(f"✓ Perplexity API system imported successfully")
except ImportError as e:
    print(f"⚠️ Failed to import Perplexity API system: {e}")
    print("⚠️ Running in compatibility mode without Perplexity API features")
    print("📦 To enable full functionality, install: pip install litellm python-dotenv")
    PERPLEXITY_AVAILABLE = False
    
    # Create mock classes for testing compatibility
    class MockPerplexityResponse:
        def __init__(self, answer="Mock answer", citations=None, response_time=1.0):
            self.answer = answer if answer is not None else "Mock answer"
            self.citations = citations or []
            self.response_time = response_time
    
    # Mock completion functions for testing
    async def acompletion(*args, **kwargs):
        return MockPerplexityResponse(answer="Yes, this is correct.")
    
    def completion(*args, **kwargs):
        return MockPerplexityResponse(answer="Yes, this is correct.")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️ datasets library not found. Please install: pip install datasets")
    print("⚠️ Running in compatibility mode - dataset operations will be mocked")
    DATASETS_AVAILABLE = False
    
    # Mock load_dataset for testing
    def load_dataset(*args, **kwargs):
        class MockDataset:
            def __init__(self):
                self.data = [
                    {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, key):
                if key == "train":
                    return MockDatasetSplit(self.data)
                return MockDatasetSplit(self.data)
        
        class MockDatasetSplit:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __iter__(self):
                return iter(self.data)
            
            def train_test_split(self, **kwargs):
                return {"test": MockDatasetSplit(self.data)}
        
        return MockDataset()

# Dataset configuration following existing patterns
folder = "KIMI_result_DreamOf_RedChamber"
# Support environment variables for pipeline integration
iteration = int(os.environ.get('PIPELINE_ITERATION', '2'))
input_file = os.environ.get('PIPELINE_INPUT_FILE', f"../datasets/{folder}/Graph_Iteration{iteration}/test_instructions_context_kimi_v2.json")
output_file = os.environ.get('PIPELINE_OUTPUT_FILE', f"../datasets/{folder}/Graph_Iteration{iteration}/pred_instructions_context_perplexity_itr{iteration}.csv")

# Perplexity API configuration
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"  # Default model for graph judgment
PERPLEXITY_CONCURRENT_LIMIT = 3  # Perplexity allows higher concurrency
PERPLEXITY_RETRY_ATTEMPTS = 3
PERPLEXITY_BASE_DELAY = 0.5  # Faster response times
PERPLEXITY_REASONING_EFFORT = "medium"  # For graph judgment accuracy

# Model selection options
PERPLEXITY_MODELS = {
    "sonar-pro": "perplexity/sonar-pro",
    "sonar-reasoning": "perplexity/sonar-reasoning",
    "sonar-reasoning-pro": "perplexity/sonar-reasoning-pro"
}

# Logging configuration
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "iteration2")

# Gold Label Bootstrapping configuration
GOLD_BOOTSTRAP_CONFIG = {
    'fuzzy_threshold': 0.8,      # RapidFuzz similarity threshold
    'sample_rate': 0.15,         # 15% sampling rate for manual review
    'llm_batch_size': 10,        # Batch size for LLM semantic evaluation
    'max_source_lines': 1000,    # Maximum source lines to process
    'random_seed': 42            # For reproducible sampling
}


class TripleData(NamedTuple):
    """Data structure for knowledge graph triples"""
    subject: str
    predicate: str
    object: str
    source_line: str = ""
    line_number: int = 0


class BootstrapResult(NamedTuple):
    """Result of gold label bootstrapping for a single triple"""
    triple: TripleData
    source_idx: int
    fuzzy_score: float
    auto_expected: Optional[bool]  # None for uncertain, True/False for confident
    llm_evaluation: Optional[str]  # LLM's semantic evaluation if performed
    expected: Optional[bool]       # Final expected value (for manual review cases)
    note: str                      # Additional notes or error messages


class ExplainableJudgment(NamedTuple):
    """
    Data structure for explainable graph judgment results
    """
    judgment: str                    # "Yes" or "No" binary decision
    confidence: float                # Confidence score (0.0-1.0)
    reasoning: str                   # Detailed reasoning explanation
    evidence_sources: List[str]      # Sources of evidence used
    alternative_suggestions: List[Dict]  # Alternative suggestions if judgment is "No"
    error_type: Optional[str]        # Error classification (if applicable)
    processing_time: float           # Time taken to process (seconds)


def setup_terminal_logging():
    """
    Set up terminal logging to capture output to a timestamped log file
    
    Returns:
        str: Path to the created log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_gj_log_{timestamp}.txt"
    log_filepath = os.path.join(LOG_DIR, log_filename)
    
    return log_filepath


class TerminalLogger:
    """
    Simple terminal logger that captures output to file
    """
    
    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.original_print = print
        
        # Initialize log file with header
        self.write_to_log("=" * 80)
        self.write_to_log(f"Perplexity API Graph Judge Execution Log")
        self.write_to_log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_to_log("=" * 80)
        
        # Replace print function with logged version
        import builtins
        builtins.print = self.logged_print
        
    def write_to_log(self, message: str):
        """Write a message directly to the log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.log_filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            # If logging fails, at least print to console
            self.original_print(f"Logging error: {e}")
    
    def logged_print(self, *args, **kwargs):
        """Custom print function that logs to file and prints to console"""
        # Convert arguments to string message
        message = ' '.join(str(arg) for arg in args)
        
        # Write to log file
        self.write_to_log(message)
        
        # Print to console using original print
        self.original_print(*args, **kwargs)
    
    def log_message(self, message: str, level: str = "INFO"):
        """Log a message with level indicator"""
        log_entry = f"[{level}] {message}"
        self.write_to_log(log_entry)
        self.original_print(log_entry)
    
    def start_session(self):
        """Start a new logging session"""
        self.log_message("🎯 Perplexity API Graph Judge - Session Started", "INFO")
    
    def end_session(self):
        """End the logging session"""
        self.log_message("🎯 Perplexity API Graph Judge - Session Ended", "INFO")
        self.write_to_log("=" * 80)
        
        # Restore original print function
        import builtins
        builtins.print = self.original_print


class PerplexityGraphJudge:
    """
    Graph Judge adapter that integrates Perplexity API for graph triple validation.
    
    This class replaces the Gemini RAG system with Perplexity's sonar-reasoning capabilities,
    providing enhanced fact-checking capabilities through Perplexity's advanced reasoning.
    """
    
    def __init__(self, model_name: str = PERPLEXITY_MODEL, 
                 reasoning_effort: str = PERPLEXITY_REASONING_EFFORT, 
                 enable_console_logging: bool = False):
        """
        Initialize the Perplexity Graph Judge system
        
        Args:
            model_name (str): Perplexity model to use
            reasoning_effort (str): Reasoning effort level (low/medium/high)
            enable_console_logging (bool): Whether to show console logs
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.enable_logging = enable_console_logging
        self.temperature = 0.2
        self.max_tokens = 2000
        self.is_mock = not PERPLEXITY_AVAILABLE
        
        # Validate API key
        if not os.getenv('PERPLEXITYAI_API_KEY'):
            raise ValueError("PERPLEXITYAI_API_KEY not found in environment variables")
        
        if not PERPLEXITY_AVAILABLE:
            print(f"⚠️ Perplexity Graph Judge running in mock mode")
            return
        
        try:
            print(f"✓ Perplexity Graph Judge initialized with model: {model_name}")
            print(f"✓ Reasoning effort set to: {reasoning_effort}")
        except Exception as e:
            print(f"✗ Failed to initialize Perplexity Graph Judge: {e}")
            print(f"⚠️ Falling back to mock mode")
            self.is_mock = True
    
    def _create_graph_judgment_prompt(self, instruction: str) -> str:
        """
        Create a specialized prompt for graph judgment using Perplexity's capabilities
        
        Args:
            instruction (str): The graph judgment instruction (e.g., "Is this true: Apple Founded by Steve Jobs ?")
            
        Returns:
            str: Optimized prompt for graph triple validation
        """
        # Extract the triple from instruction format
        triple_part = instruction.replace("Is this true: ", "").replace(" ?", "")
        
        # Create enhanced prompt that leverages Perplexity's reasoning capabilities
        prompt = f"""
You are a knowledge graph validation expert. Please evaluate whether the following triple statement is factually correct.

Task Requirements:
1. Output only "Yes" or "No" (no additional text)
2. Base your judgment on reliable information sources
3. For "Dream of the Red Chamber" related content, pay special attention to literary accuracy
4. For real-world facts, refer to the latest authoritative information

Evaluation Rules:
- If the triple is syntactically correct and factually accurate, answer "Yes"
- If the triple is syntactically incorrect or factually wrong, answer "No"

Triple to evaluate:
{triple_part}

Please answer only "Yes" or "No":
""".strip()
        
        return prompt
    
    def _parse_response(self, response) -> str:
        """
        Parse and validate the Perplexity response for graph judgment
        
        Args:
            response: Response from Perplexity API (could be completion response or mock)
            
        Returns:
            str: Cleaned binary response ("Yes" or "No")
        """
        # Handle None response
        if response is None:
            print("Warning: Received None response, defaulting to No")
            return "No"
        
        # Extract the main answer safely from Perplexity response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            answer = str(response.choices[0].message.content).strip()
        elif hasattr(response, 'answer') and response.answer is not None:
            answer = str(response.answer).strip()
        else:
            print("Warning: Response has no valid content, defaulting to No")
            return "No"
        
        # Handle empty answer
        if not answer:
            print("Warning: Empty answer, defaulting to No")
            return "No"
        
        # Clean the response to extract binary judgment
        # Look for explicit Yes/No patterns
        if re.search(r'\byes\b', answer, re.IGNORECASE):
            return "Yes"
        elif re.search(r'\bno\b', answer, re.IGNORECASE):
            return "No"
        elif re.search(r'\b是\b|\b正確\b|\b對\b', answer):
            return "Yes"
        elif re.search(r'\b否\b|\b錯誤\b|\b不對\b|\b不是\b', answer):
            return "No"
        else:
            # If no clear binary response, analyze the content for sentiment
            positive_indicators = ['correct', 'true', 'accurate', 'valid', '正確', '是的', '對的']
            negative_indicators = ['incorrect', 'false', 'wrong', 'invalid', '錯誤', '不對', '否']
            
            answer_lower = answer.lower()
            positive_score = sum(1 for indicator in positive_indicators if indicator in answer_lower)
            negative_score = sum(1 for indicator in negative_indicators if indicator in answer_lower)
            
            if positive_score > negative_score:
                return "Yes"
            elif negative_score > positive_score:
                return "No"
            else:
                # Default to No for ambiguous responses (conservative approach)
                print(f"Warning: Ambiguous response, defaulting to No: {answer[:100]}...")
                return "No"
    
    async def judge_graph_triple(self, instruction: str, input_text: str = None) -> str:
        """
        Judge a graph triple using Perplexity API
        
        Args:
            instruction (str): Graph judgment instruction
            input_text (str): Additional context (optional)
            
        Returns:
            str: Binary judgment result ("Yes" or "No")
        """
        # Handle mock mode
        if self.is_mock:
            # Simple mock logic for testing
            if "Apple Founded by Steve Jobs" in instruction or "曹雪芹 創作 紅樓夢" in instruction:
                return "Yes"
            elif "Microsoft Founded by Mark Zuckerberg" in instruction:
                return "No"
            else:
                return "Yes"  # Default for unknown cases in mock mode
        
        max_retries = PERPLEXITY_RETRY_ATTEMPTS
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create specialized prompt for graph judgment
                prompt = self._create_graph_judgment_prompt(instruction)
                
                # Perplexity API call
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )
                
                # Parse and validate the response
                judgment = self._parse_response(response)
                
                # Extract and log citation information
                citation_summary = self.get_citation_summary(response)
                if citation_summary['has_citations'] and self.enable_logging:
                    print(f"Graph judgment: {judgment} (with {citation_summary['total_citations']} citations)")
                    for citation in citation_summary['citations'][:3]:  # Show first 3 citations
                        print(f"  📚 Citation {citation['number']}: {citation['title']}")
                
                return judgment
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                print(f"Error on attempt {retry_count}/{max_retries}: {error_msg}")
                
                if retry_count >= max_retries:
                    print(f"Failed to get response after {max_retries} attempts")
                    return "No"  # Conservative default
                
                # Wait before retry
                wait_time = PERPLEXITY_BASE_DELAY * (2 ** retry_count)
                await asyncio.sleep(wait_time)

    async def judge_graph_triple_with_explanation(self, instruction: str, input_text: str = None, include_citations: bool = True) -> ExplainableJudgment:
        """
        Judge a graph triple with detailed explainable reasoning
        
        Args:
            instruction (str): Graph judgment instruction
            input_text (str): Additional context (optional)
            
        Returns:
            ExplainableJudgment: Comprehensive judgment with reasoning, confidence, and evidence
        """
        start_time = time.time()
        
        # Handle mock mode
        if self.is_mock:
            # Provide more realistic mock data for testing
            if "Apple Founded by Steve Jobs" in instruction or "曹雪芹 創作 紅樓夢" in instruction:
                return ExplainableJudgment(
                    judgment="Yes",
                    confidence=0.95,
                    reasoning="這是一個已知的歷史事實，有充分的文獻證據支持。",
                    evidence_sources=["domain_knowledge", "historical_records"],
                    alternative_suggestions=[],
                    error_type=None,
                    processing_time=time.time() - start_time
                )
            elif "Microsoft Founded by Mark Zuckerberg" in instruction:
                return ExplainableJudgment(
                    judgment="No",
                    confidence=0.90,
                    reasoning="這個陳述是錯誤的。Microsoft是由Bill Gates和Paul Allen創立的，而不是Mark Zuckerberg。Mark Zuckerberg是Facebook的創始人。",
                    evidence_sources=["domain_knowledge", "tech_history"],
                    alternative_suggestions=[
                        {"subject": "Bill Gates", "relation": "創立", "object": "Microsoft", "confidence": 0.95},
                        {"subject": "Paul Allen", "relation": "共同創立", "object": "Microsoft", "confidence": 0.95}
                    ],
                    error_type="factual_error",
                    processing_time=time.time() - start_time
                )
            else:
                return ExplainableJudgment(
                    judgment="Yes",
                    confidence=0.75,
                    reasoning="基於可用資訊，這個陳述似乎是合理的。",
                    evidence_sources=["general_knowledge"],
                    alternative_suggestions=[],
                    error_type=None,
                    processing_time=time.time() - start_time
                )
        
        max_retries = PERPLEXITY_RETRY_ATTEMPTS
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create specialized prompt for explainable graph judgment
                prompt = self._create_explainable_judgment_prompt(instruction)
                
                # Perplexity API call for explainable judgment
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )
                
                # Parse and validate the response for explainable judgment
                explainable_judgment = self._parse_explainable_response(response)
                
                # Enhance with citation information if requested
                if include_citations:
                    citation_summary = self.get_citation_summary(response)
                    if citation_summary['has_citations']:
                        # Add citation sources to evidence sources
                        enhanced_evidence_sources = list(explainable_judgment.evidence_sources)
                        enhanced_evidence_sources.append(f"perplexity_citations({citation_summary['total_citations']})")
                        
                        # Create enhanced judgment with citation information
                        final_judgment = explainable_judgment._replace(
                            evidence_sources=enhanced_evidence_sources,
                            processing_time=time.time() - start_time
                        )
                    else:
                        final_judgment = explainable_judgment._replace(processing_time=time.time() - start_time)
                else:
                    final_judgment = explainable_judgment._replace(processing_time=time.time() - start_time)
                
                # Extract and log citation information
                citation_summary = self.get_citation_summary(response)
                if citation_summary['has_citations'] and self.enable_logging:
                    print(f"Explainable judgment: {final_judgment.judgment} (confidence: {final_judgment.confidence:.2f}, with {citation_summary['total_citations']} citations)")
                    for citation in citation_summary['citations'][:3]:  # Show first 3 citations
                        print(f"  📚 Citation {citation['number']}: {citation['title']}")
                
                return final_judgment
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                print(f"Error on explainable judgment attempt {retry_count}/{max_retries}: {error_msg}")
                
                if retry_count >= max_retries:
                    print(f"Failed to get explainable response after {max_retries} attempts")
                    # Return conservative default with error information
                    return ExplainableJudgment(
                        judgment="No",
                        confidence=0.0,
                        reasoning=f"Error during processing: {error_msg}",
                        evidence_sources=[],
                        alternative_suggestions=[],
                        error_type="processing_error",
                        processing_time=time.time() - start_time
                    )
                
                # Wait before retry
                wait_time = PERPLEXITY_BASE_DELAY * (2 ** retry_count)
                await asyncio.sleep(wait_time)
    
    def _create_explainable_judgment_prompt(self, instruction: str) -> str:
        """
        Create a specialized prompt for explainable graph judgment
        為可解釋圖判斷創建專門的提示詞
        
        Args:
            instruction (str): The graph judgment instruction
            
        Returns:
            str: Enhanced prompt for explainable triple validation
        """
        # Extract the triple from instruction format
        triple_part = instruction.replace("Is this true: ", "").replace(" ?", "")
        
        # Create comprehensive prompt for explainable reasoning with Chinese elements
        prompt = f"""
你是知識圖譜驗證專家。請對以下三元組語句進行詳細分析並提供結構化的判斷結果。

待評估三元組：{triple_part}

請按以下格式提供完整分析：

1. 判斷結果：[僅回答"Yes"或"No"]

2. 置信度：[0.0到1.0之間的數值，表示你的確定程度]

3. 詳細推理：[解釋你的判斷理由，包括：
   - 三元組語法正確性分析
   - 事實準確性評估
   - 相關背景知識或證據
   - 對於《紅樓夢》內容，請參考原文]

4. 證據來源：[列出支持判斷的證據類型：source_text, domain_knowledge, literary_history等]

5. 錯誤類型：[如果判斷為"No"，請指定錯誤類型：entity_mismatch, factual_error, structural_error, temporal_inconsistency, source_unsupported。如果為"Yes"，回答"None"]

6. 替代建議：[如果判斷為"No"，請提供正確的三元組建議，格式為：subject-relation-object]

請確保你的回應結構化、邏輯清晰，並基於可靠來源。
""".strip()
        
        return prompt
    
    def _parse_explainable_response(self, response) -> ExplainableJudgment:
        """
        Parse Perplexity response for explainable graph judgment
        解析 Perplexity 的可解釋圖判斷回應
        
        Args:
            response: Response from Perplexity API
            
        Returns:
            ExplainableJudgment: Parsed explainable judgment result
        """
        # Handle None response
        if response is None:
            return ExplainableJudgment(
                judgment="No",
                confidence=0.0,
                reasoning="Received empty response, cannot make judgment",
                evidence_sources=[],
                alternative_suggestions=[],
                error_type="response_error",
                processing_time=0.0
            )
        
        # Extract the main answer safely from Perplexity response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            answer = str(response.choices[0].message.content).strip()
        elif hasattr(response, 'answer') and response.answer is not None:
            answer = str(response.answer).strip()
        else:
            return ExplainableJudgment(
                judgment="No",
                confidence=0.0,
                reasoning="Response format error, no valid content",
                evidence_sources=[],
                alternative_suggestions=[],
                error_type="response_error",
                processing_time=0.0
            )
        
        # Handle empty answer
        if not answer:
            return ExplainableJudgment(
                judgment="No",
                confidence=0.0,
                reasoning="收到空的回應內容",
                evidence_sources=[],
                alternative_suggestions=[],
                error_type="response_error",
                processing_time=0.0
            )
        
        try:
            # Parse structured response
            judgment = self._extract_judgment(answer)
            confidence = self._extract_confidence(answer)
            reasoning = self._extract_reasoning(answer)
            evidence_sources = self._extract_evidence_sources(answer)
            error_type = self._extract_error_type(answer)
            alternative_suggestions = self._extract_alternatives(answer)
            
            return ExplainableJudgment(
                judgment=judgment,
                confidence=confidence,
                reasoning=reasoning,
                evidence_sources=evidence_sources,
                alternative_suggestions=alternative_suggestions,
                error_type=error_type,
                processing_time=0.0  # Will be updated by caller
            )
            
        except Exception as e:
            # Fallback parsing if structured parsing fails
            simple_judgment = self._parse_response(response)
            return ExplainableJudgment(
                judgment=simple_judgment,
                confidence=0.5,  # Default moderate confidence
                reasoning=f"無法解析結構化回應，回退到簡單判斷。原始回應：{answer[:200]}...",
                evidence_sources=["response_analysis"],
                alternative_suggestions=[],
                error_type="parsing_error" if simple_judgment == "No" else None,
                processing_time=0.0
            )
    
    def _extract_judgment(self, answer: str) -> str:
        """Extract binary judgment from structured response"""
        # Look for judgment patterns
        if re.search(r'判斷結果[：:\s]*["\']?Yes["\']?', answer, re.IGNORECASE):
            return "Yes"
        elif re.search(r'判斷結果[：:\s]*["\']?No["\']?', answer, re.IGNORECASE):
            return "No"
        elif re.search(r'\byes\b', answer, re.IGNORECASE):
            return "Yes"
        elif re.search(r'\bno\b', answer, re.IGNORECASE):
            return "No"
        else:
            # Default fallback logic
            return self._parse_response(type('MockResponse', (), {'answer': answer})())
    
    def _extract_confidence(self, answer: str) -> float:
        """Extract confidence score from response"""
        # Look for confidence patterns
        confidence_patterns = [
            r'置信度[：:\s]*([0-9]*\.?[0-9]+)',
            r'confidence[：:\s]*([0-9]*\.?[0-9]+)',
            r'確信程度[：:\s]*([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                try:
                    conf = float(match.group(1))
                    return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
                except ValueError:
                    continue
        
        # Default confidence based on judgment certainty indicators
        if any(word in answer.lower() for word in ['確定', '肯定', '明確', 'certain', 'definitely']):
            return 0.9
        elif any(word in answer.lower() for word in ['可能', '似乎', 'likely', 'probably']):
            return 0.7
        else:
            return 0.5  # Default moderate confidence
    
    def _extract_reasoning(self, answer: str) -> str:
        """Extract detailed reasoning from response"""
        reasoning_patterns = [
            r'詳細推理[：:\s]*(.+?)(?=\d+\.|證據來源|錯誤類型|$)',
            r'推理[：:\s]*(.+?)(?=\d+\.|證據來源|錯誤類型|$)',
            r'reasoning[：:\s]*(.+?)(?=\d+\.|evidence|error|$)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 10:  # Ensure substantial content
                    return reasoning
        
        # Fallback: use the full answer as reasoning
        return answer[:500] + "..." if len(answer) > 500 else answer
    
    def _extract_evidence_sources(self, answer: str) -> List[str]:
        """Extract evidence sources from response"""
        evidence_patterns = [
            r'證據來源[：:\s]*(.+?)(?=\d+\.|錯誤類型|替代建議|$)',
            r'evidence[：:\s]*(.+?)(?=\d+\.|error|alternative|$)',
            r'Evidence Sources[：:\s]*(.+?)(?=\d+\.|Error Type|Alternative|$)'
        ]
        
        sources = []
        for pattern in evidence_patterns:
            match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                evidence_text = match.group(1).strip()
                # Parse comma-separated or line-separated sources
                potential_sources = re.split(r'[,，\n]', evidence_text)
                for source in potential_sources:
                    clean_source = source.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                    if clean_source and len(clean_source) > 2:
                        sources.append(clean_source)
        
        return sources if sources else ["general_analysis"]
    
    def _extract_error_type(self, answer: str) -> Optional[str]:
        """Extract error type from response"""
        # Check if the text explicitly states no error type information first
        if re.search(r'沒有錯誤類型|無錯誤類型|no error type', answer, re.IGNORECASE):
            return None
        
        error_patterns = [
            r'錯誤類型[：:\s]*([^，,\n]+)',
            r'error[_\s]type[：:\s]*([^，,\n]+)'
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                error_type = match.group(1).strip().lower()
                if 'none' in error_type or '無' in error_type or 'null' in error_type:
                    return None
                return error_type
        
        return None
    
    def _extract_alternatives(self, answer: str) -> List[Dict]:
        """Extract alternative suggestions from response"""
        alt_patterns = [
            r'替代建議[：:\s]*(.+?)(?=\d+\.|$)',
            r'alternative[：:\s]*(.+?)(?=\d+\.|$)',
            r'Alternative Suggestions[：:\s]*(.+?)(?=\d+\.|$)'
        ]
        
        alternatives = []
        for pattern in alt_patterns:
            match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                alt_text = match.group(1).strip()
                # Parse various formats: "主體-關係-客體", "subject-relation-object"
                triple_matches = re.findall(r'([^-\n]+)-([^-\n]+)-([^-\n，,]+)', alt_text)
                for subj, rel, obj in triple_matches:
                    alternatives.append({
                        "subject": subj.strip(),
                        "relation": rel.strip(),
                        "object": obj.strip(),
                        "confidence": 0.8  # Default confidence for alternatives
                    })
                
                # If no triple format found, try to extract individual suggestions
                if not triple_matches:
                    # Look for simple suggestions like "曹雪芹-作頭號人物"
                    simple_matches = re.findall(r'([^-\n]+)-([^-\n，,]+)', alt_text)
                    for subj, rest in simple_matches:
                        # Try to split the rest into relation and object
                        parts = rest.strip().split()
                        if len(parts) >= 2:
                            alternatives.append({
                                "subject": subj.strip(),
                                "relation": parts[0].strip(),
                                "object": ' '.join(parts[1:]).strip(),
                                "confidence": 0.8
                            })
                        else:
                            # Handle cases like "曹雪芹-作頭號人物" where rest is a single phrase
                            # Treat the entire rest as the relation
                            alternatives.append({
                                "subject": subj.strip(),
                                "relation": rest.strip(),
                                "object": "相關對象",  # Default object for incomplete triples
                                "confidence": 0.8
                            })
        
        return alternatives
    
    def _save_reasoning_file(self, reasoning_results: List[Dict], output_path: str) -> bool:
        """
        Save explainable reasoning results to JSON file
        將可解釋推理結果保存到JSON文件
        
        Args:
            reasoning_results (List[Dict]): List of structured reasoning results
            output_path (str): Path to save the reasoning file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save reasoning data as formatted JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Explainable reasoning results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving reasoning file: {e}")
            return False

    # ==================== Phase 2: Enhanced Features ====================
    
    async def judge_graph_triple_streaming(self, instruction: str, 
                                         stream_container=None) -> str:
        """
        Streaming version of graph triple judgment for real-time processing feedback
        
        Args:
            instruction (str): Graph judgment instruction
            stream_container: Optional container for streaming updates (e.g., Streamlit container)
            
        Returns:
            str: Binary judgment result ("Yes" or "No")
        """
        # Handle mock mode
        if self.is_mock:
            return await self.judge_graph_triple(instruction)
        
        try:
            prompt = self._create_graph_judgment_prompt(instruction)
            
            # Perplexity API call with streaming
            response = await acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                stream=True
            )
            
            full_answer = ""
            chunk_count = 0
            
            # Handle both async iterator and direct response
            try:
                if hasattr(response, '__aiter__'):
                    # Async iterator - streaming response
                    async for chunk in response:
                        chunk_count += 1
                        
                        # Extract content from streaming chunk
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_answer += delta.content
                                
                                # Update stream container if provided (reduced frequency)
                                if stream_container and chunk_count % 5 == 0:
                                    try:
                                        if hasattr(stream_container, 'markdown'):
                                            stream_container.markdown(f"🔄 Processing: {full_answer[:100]}...")
                                        elif hasattr(stream_container, 'write'):
                                            stream_container.write(f"Processing: {full_answer[:100]}...")
                                    except Exception as e:
                                        print(f"Stream update error: {e}")
                        
                        # Small delay to prevent overwhelming
                        await asyncio.sleep(0.02)
                else:
                    # Direct response - non-streaming
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        full_answer = response.choices[0].message.content
                        if stream_container:
                            try:
                                if hasattr(stream_container, 'markdown'):
                                    stream_container.markdown(f"✅ Response received: {full_answer[:100]}...")
                            except Exception as e:
                                print(f"Stream update error: {e}")
            except Exception as stream_error:
                print(f"Streaming error: {stream_error}")
                # Fallback to direct response access
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    full_answer = response.choices[0].message.content
            
            # Parse the complete response
            mock_response = type('MockResponse', (), {
                'choices': [type('MockChoice', (), {
                    'message': type('MockMessage', (), {'content': full_answer})()
                })()]
            })()
            
            return self._parse_response(mock_response)
            
        except Exception as e:
            print(f"Error in streaming judgment: {e}")
            return "No"  # Conservative default
    
    def _extract_citations(self, response) -> List[Dict]:
        """
        Extract citations from Perplexity response
        
        Args:
            response: Response from Perplexity API
            
        Returns:
            List[Dict]: List of citation dictionaries with metadata
        """
        citations = []
        
        try:
            # Primary method: Check for citations directly in response
            if hasattr(response, 'citations') and response.citations:
                for i, citation in enumerate(response.citations):
                    if isinstance(citation, str):  # URL string
                        citations.append({
                            "number": str(i + 1),
                            "title": self._extract_title_from_url(citation),
                            "url": citation,
                            "type": "perplexity_citation",
                            "source": "direct"
                        })
                    elif isinstance(citation, dict):  # Citation object
                        citations.append({
                            "number": str(i + 1),
                            "title": citation.get('title', self._extract_title_from_url(citation.get('url', ''))),
                            "url": citation.get('url', ''),
                            "type": "perplexity_citation",
                            "source": "object"
                        })
            
            # Secondary method: Extract citation references from content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    # Extract citation numbers from content like [1], [2], etc.
                    citation_numbers = re.findall(r'\[(\d+)\]', content)
                    # Remove duplicates while preserving order
                    unique_numbers = list(dict.fromkeys(citation_numbers))
                    
                    for num in unique_numbers:
                        # Check if we already have this citation from direct method
                        existing = any(c['number'] == num for c in citations)
                        if not existing:
                            # Try to find corresponding URL from response citations
                            url = ""
                            if hasattr(response, 'citations') and response.citations:
                                idx = int(num) - 1
                                if 0 <= idx < len(response.citations):
                                    citation_item = response.citations[idx]
                                    url = citation_item if isinstance(citation_item, str) else citation_item.get('url', '')
                            
                            citations.append({
                                "number": num,
                                "title": self._extract_title_from_url(url) if url else f"Reference {num}",
                                "url": url,
                                "type": "perplexity_citation",
                                "source": "content_reference"
                            })
        
        except Exception as e:
            print(f"Error extracting citations: {e}")
        
        # Sort citations by number for consistent ordering
        try:
            citations.sort(key=lambda x: int(x['number']))
        except (ValueError, KeyError):
            pass  # Keep original order if sorting fails
        
        return citations
    
    def _extract_title_from_url(self, url: str) -> str:
        """
        Extract a readable title from a URL
        
        Args:
            url (str): URL to extract title from
            
        Returns:
            str: Extracted title or simplified URL
        """
        try:
            # Remove protocol and domain
            if '://' in url:
                url = url.split('://', 1)[1]
            
            # Remove www. prefix
            if url.startswith('www.'):
                url = url[4:]
            
            # Take the first part of the path
            if '/' in url:
                url = url.split('/', 1)[0]
            
            # Clean up and return
            title = url.replace('-', ' ').replace('_', ' ').title()
            return title[:50] + "..." if len(title) > 50 else title
            
        except Exception:
            return "Unknown Source"
    
    def get_citation_summary(self, response) -> Dict:
        """
        Get a summary of citations from a Perplexity response
        
        Args:
            response: Response from Perplexity API
            
        Returns:
            Dict: Summary of citations with metadata
        """
        citations = self._extract_citations(response)
        
        return {
            "total_citations": len(citations),
            "citations": citations,
            "has_citations": len(citations) > 0,
            "citation_types": list(set(c.get("type", "unknown") for c in citations))
        }
    
    def _clean_html_tags(self, text: str) -> str:
        """
        Clean HTML tags from response text (based on reference implementation)
        
        Args:
            text (str): Text containing HTML tags
            
        Returns:
            str: Cleaned text without HTML tags
        """
        if not text:
            return ""
        
        # Remove common HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    async def judge_graph_triple_with_citations(self, instruction: str, input_text: str = None) -> Dict:
        """
        Get graph judgment with detailed citation information
        
        Args:
            instruction (str): Graph judgment instruction
            input_text (str): Additional context (optional)
            
        Returns:
            Dict: Judgment result with citation details
        """
        start_time = time.time()
        
        # Handle mock mode
        if self.is_mock:
            return {
                "judgment": "Yes",
                "confidence": 0.8,
                "citations": [],
                "citation_count": 0,
                "processing_time": time.time() - start_time
            }
        
        max_retries = PERPLEXITY_RETRY_ATTEMPTS
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                prompt = self._create_graph_judgment_prompt(instruction)
                
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )
                
                judgment = self._parse_response(response)
                citation_summary = self.get_citation_summary(response)
                
                return {
                    "judgment": judgment,
                    "confidence": 0.8 if citation_summary['has_citations'] else 0.6,
                    "citations": citation_summary['citations'],
                    "citation_count": citation_summary['total_citations'],
                    "processing_time": time.time() - start_time
                }
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                                    return {
                    "judgment": "No",
                    "confidence": 0.0,
                    "citations": [],
                    "citation_count": 0,
                    "processing_time": time.time() - start_time,
                    "error": str(e)
                }
                await asyncio.sleep(PERPLEXITY_BASE_DELAY * (2 ** retry_count))

    # ==================== Gold Label Bootstrapping Methods ====================
    
    def _load_triples_from_file(self, triples_file: str) -> List[TripleData]:
        """
        Load and parse triples from the generated graphs file
        從生成的圖文件中加載和解析三元組
        
        Args:
            triples_file (str): Path to the triples file
            
        Returns:
            List[TripleData]: Parsed triples with metadata
        """
        triples = []
        
        if not os.path.exists(triples_file):
            print(f"⚠️ Triples file not found: {triples_file}")
            return triples
        
        try:
            with open(triples_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse different formats of triple representation
                    # Format 1: JSON-like lists in text
                    if '[' in line and ']' in line:
                        try:
                            # Extract JSON-like content
                            json_match = re.search(r'\[(.*?)\]', line)
                            if json_match:
                                json_str = '[' + json_match.group(1) + ']'
                                # Replace single quotes with double quotes for JSON parsing
                                json_str = json_str.replace("'", '"')
                                parsed_triples = json.loads(json_str)
                                
                                for triple_list in parsed_triples:
                                    if isinstance(triple_list, list) and len(triple_list) >= 3:
                                        triple = TripleData(
                                            subject=str(triple_list[0]).strip(),
                                            predicate=str(triple_list[1]).strip(),
                                            object=str(triple_list[2]).strip(),
                                            source_line=line,
                                            line_number=line_num
                                        )
                                        triples.append(triple)
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"⚠️ Failed to parse JSON on line {line_num}: {e}")
                    
                    # Format 2: Simple "S P O" format
                    elif ' ' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            triple = TripleData(
                                subject=parts[0].strip(),
                                predicate=parts[1].strip(),
                                object=' '.join(parts[2:]).strip(),
                                source_line=line,
                                line_number=line_num
                            )
                            triples.append(triple)
                    
            print(f"✓ Loaded {len(triples)} triples from {triples_file}")
            return triples
            
        except Exception as e:
            print(f"✗ Error loading triples from {triples_file}: {e}")
            return []

    def _load_source_text(self, source_file: str) -> List[str]:
        """
        Load source text lines for comparison
        加載源文本行用於比較
        
        Args:
            source_file (str): Path to the source text file
            
        Returns:
            List[str]: List of source text lines
        """
        if not os.path.exists(source_file):
            print(f"⚠️ Source file not found: {source_file}")
            return []
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Limit source lines to prevent memory issues
            max_lines = GOLD_BOOTSTRAP_CONFIG['max_source_lines']
            if len(lines) > max_lines:
                print(f"⚠️ Source file has {len(lines)} lines, limiting to {max_lines}")
                lines = lines[:max_lines]
            
            print(f"✓ Loaded {len(lines)} source text lines from {source_file}")
            return lines
            
        except Exception as e:
            print(f"✗ Error loading source text from {source_file}: {e}")
            return []

    def _stage1_rapidfuzz_matching(self, triples: List[TripleData], source_lines: List[str]) -> List[BootstrapResult]:
        """
        Stage 1: RapidFuzz string similarity matching
        第一階段：RapidFuzz 字符串相似度匹配
        
        Args:
            triples (List[TripleData]): List of triples to evaluate
            source_lines (List[str]): Source text lines for comparison
            
        Returns:
            List[BootstrapResult]: Initial bootstrap results with fuzzy scores
        """
        if not RAPIDFUZZ_AVAILABLE:
            print("⚠️ RapidFuzz not available, using mock scoring")
        
        results = []
        threshold = GOLD_BOOTSTRAP_CONFIG['fuzzy_threshold']
        
        print(f"🔍 Stage 1: Running RapidFuzz matching with threshold {threshold}")
        
        for triple in triples:
            # Create triple string representation
            triple_str = f"{triple.subject} {triple.predicate} {triple.object}"
            
            best_score = 0.0
            best_source_idx = -1
            
            # Compare against all source lines
            for idx, source_line in enumerate(source_lines):
                if not source_line:
                    continue
                
                score = fuzz.partial_ratio(triple_str, source_line) / 100.0
                if score > best_score:
                    best_score = score
                    best_source_idx = idx
            
            # Determine auto_expected based on threshold
            if best_score >= threshold:
                auto_expected = True
                note = f"High similarity (≥{threshold}) with source"
            else:
                auto_expected = None  # Uncertain, needs Stage 2 evaluation
                note = f"Low similarity (<{threshold}), requires semantic evaluation"
            
            result = BootstrapResult(
                triple=triple,
                source_idx=best_source_idx,
                fuzzy_score=best_score,
                auto_expected=auto_expected,
                llm_evaluation=None,
                expected=auto_expected,  # Will be updated in Stage 2 if needed
                note=note
            )
            results.append(result)
        
        confirmed_count = sum(1 for r in results if r.auto_expected == True)
        uncertain_count = sum(1 for r in results if r.auto_expected is None)
        
        print(f"📊 Stage 1 Results: {confirmed_count} confirmed, {uncertain_count} uncertain (need Stage 2)")
        
        return results

    async def _stage2_llm_semantic_evaluation(self, uncertain_results: List[BootstrapResult], 
                                            source_lines: List[str]) -> List[BootstrapResult]:
        """
        Stage 2: LLM semantic evaluation for uncertain cases
        
        Args:
            uncertain_results (List[BootstrapResult]): Results that need semantic evaluation
            source_lines (List[str]): Source text lines for context
            
        Returns:
            List[BootstrapResult]: Updated results with LLM evaluations
        """
        if not uncertain_results:
            print("📊 Stage 2: No uncertain cases to evaluate")
            return uncertain_results
        
        print(f"🧠 Stage 2: LLM semantic evaluation for {len(uncertain_results)} uncertain cases")
        
        updated_results = []
        batch_size = GOLD_BOOTSTRAP_CONFIG['llm_batch_size']
        
        for i in range(0, len(uncertain_results), batch_size):
            batch = uncertain_results[i:i + batch_size]
            
            for result in batch:
                try:
                    triple = result.triple
                    source_context = ""
                    
                    # Get source context around the best matching line
                    if 0 <= result.source_idx < len(source_lines):
                        # Include context lines for better evaluation
                        start_idx = max(0, result.source_idx - 2)
                        end_idx = min(len(source_lines), result.source_idx + 3)
                        source_context = '\n'.join(source_lines[start_idx:end_idx])
                    
                    # Create semantic evaluation prompt
                    evaluation_prompt = f"""
請判斷以下三元組是否可以從給定的源文本中語義推導出來：

三元組：{triple.subject} {triple.predicate} {triple.object}

源文本上下文：
{source_context}

判斷規則：
- 如果源文本明確提到或可以合理推導出此三元組關係，回答 "Yes"
- 如果源文本與此三元組矛盾或完全無關，回答 "No"
- 只需回答 "Yes" 或 "No"，無需解釋

判斷結果："""

                    # Use the existing Perplexity system for evaluation
                    if not self.is_mock:
                        response = self.qa_system.ask_question(evaluation_prompt)
                        llm_evaluation = self._parse_response(response)
                    else:
                        # Mock evaluation for testing
                        llm_evaluation = "Yes" if "創作" in triple.predicate or "喜歡" in triple.predicate else "No"
                    
                    # Update result based on LLM evaluation
                    auto_expected = True if llm_evaluation == "Yes" else False
                    note = f"LLM semantic evaluation: {llm_evaluation}"
                    
                    updated_result = result._replace(
                        auto_expected=auto_expected,
                        llm_evaluation=llm_evaluation,
                        expected=auto_expected,
                        note=note
                    )
                    updated_results.append(updated_result)
                    
                    print(f"🔄 Evaluated: {triple.subject} {triple.predicate} {triple.object} → {llm_evaluation}")
                    
                    # Small delay to respect API limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error evaluating triple {triple.subject} {triple.predicate} {triple.object}: {e}")
                    # Keep original result but mark as error
                    error_result = result._replace(
                        note=f"LLM evaluation error: {str(e)}"
                    )
                    updated_results.append(error_result)
        
        confirmed_by_llm = sum(1 for r in updated_results if r.auto_expected == True)
        rejected_by_llm = sum(1 for r in updated_results if r.auto_expected == False)
        
        print(f"📊 Stage 2 Results: {confirmed_by_llm} confirmed by LLM, {rejected_by_llm} rejected by LLM")
        
        return updated_results

    def _sample_uncertain_cases(self, results: List[BootstrapResult]) -> List[BootstrapResult]:
        """
        Sample uncertain cases for manual review
        
        Args:
            results (List[BootstrapResult]): All bootstrap results
            
        Returns:
            List[BootstrapResult]: Results with sampled cases marked for manual review
        """
        sample_rate = GOLD_BOOTSTRAP_CONFIG['sample_rate']
        random.seed(GOLD_BOOTSTRAP_CONFIG['random_seed'])
        
        # Find cases that were initially uncertain (went through Stage 2)
        stage2_cases = [r for r in results if r.llm_evaluation is not None]
        
        if not stage2_cases:
            print("📊 No Stage 2 cases found for sampling")
            return results
        
        # Sample for manual review
        sample_size = max(1, int(len(stage2_cases) * sample_rate))
        sampled_indices = set(random.sample(range(len(stage2_cases)), 
                                          min(sample_size, len(stage2_cases))))
        
        print(f"📝 Sampling {sample_size} cases from {len(stage2_cases)} Stage 2 cases for manual review")
        
        updated_results = []
        stage2_idx = 0
        
        for result in results:
            if result.llm_evaluation is not None:  # This is a Stage 2 case
                if stage2_idx in sampled_indices:
                    # Mark for manual review
                    updated_result = result._replace(
                        expected=None,  # Clear auto-assignment, needs manual review
                        note=result.note + " | SAMPLED FOR MANUAL REVIEW"
                    )
                    updated_results.append(updated_result)
                else:
                    updated_results.append(result)
                stage2_idx += 1
            else:
                updated_results.append(result)
        
        manual_review_count = sum(1 for r in updated_results if r.expected is None and "MANUAL REVIEW" in r.note)
        print(f"📝 {manual_review_count} cases marked for manual review")
        
        return updated_results

    def _save_bootstrap_results(self, results: List[BootstrapResult], output_file: str) -> bool:
        """
        Save bootstrap results to CSV file
        
        Args:
            results (List[BootstrapResult]): Bootstrap results to save
            output_file (str): Output CSV file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['subject', 'predicate', 'object', 'source_idx', 'fuzzy_score', 
                            'auto_expected', 'expected', 'note']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow({
                        'subject': result.triple.subject,
                        'predicate': result.triple.predicate,
                        'object': result.triple.object,
                        'source_idx': result.source_idx,
                        'fuzzy_score': f"{result.fuzzy_score:.3f}",
                        'auto_expected': result.auto_expected if result.auto_expected is not None else '',
                        'expected': result.expected if result.expected is not None else '',
                        'note': result.note
                    })
            
            print(f"✅ Bootstrap results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving bootstrap results: {e}")
            return False

    async def bootstrap_gold_labels(self, triples_file: str, source_file: str, output_file: str) -> bool:
        """
        Main gold label bootstrapping method
        
        This method implements the two-stage gold label bootstrapping process:
        1. Stage 1: RapidFuzz string similarity matching
        2. Stage 2: LLM semantic evaluation for uncertain cases
        
        Args:
            triples_file (str): Path to the triples file
            source_file (str): Path to the source text file  
            output_file (str): Path to the output CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("=" * 70)
        print("🎯 Gold Label Bootstrapping - Automatic Gold Label Assignment")
        print("=" * 70)
        
        try:
            # Load input data
            print("📚 Loading input data...")
            triples = self._load_triples_from_file(triples_file)
            if not triples:
                print("❌ No triples loaded, aborting")
                return False
            
            source_lines = self._load_source_text(source_file)
            if not source_lines:
                print("❌ No source text loaded, aborting")
                return False
            
            # Stage 1: RapidFuzz matching
            print("\n" + "="*50)
            print("🔍 Stage 1: RapidFuzz String Similarity Matching")
            print("="*50)
            stage1_results = self._stage1_rapidfuzz_matching(triples, source_lines)
            
            # Stage 2: LLM semantic evaluation for uncertain cases
            print("\n" + "="*50)
            print("🧠 Stage 2: LLM Semantic Evaluation")
            print("="*50)
            uncertain_results = [r for r in stage1_results if r.auto_expected is None]
            
            if uncertain_results:
                stage2_results = await self._stage2_llm_semantic_evaluation(uncertain_results, source_lines)
                
                # Merge results
                final_results = []
                stage2_dict = {(r.triple.subject, r.triple.predicate, r.triple.object): r 
                             for r in stage2_results}
                
                for result in stage1_results:
                    key = (result.triple.subject, result.triple.predicate, result.triple.object)
                    if key in stage2_dict:
                        final_results.append(stage2_dict[key])
                    else:
                        final_results.append(result)
            else:
                final_results = stage1_results
            
            # Sample uncertain cases for manual review
            print("\n" + "="*50)
            print("📝 Sampling for Manual Review")
            print("="*50)
            final_results = self._sample_uncertain_cases(final_results)
            
            # Save results
            print("\n" + "="*50)
            print("💾 Saving Results")
            print("="*50)
            success = self._save_bootstrap_results(final_results, output_file)
            
            if success:
                # Print final statistics
                total_triples = len(final_results)
                auto_confirmed = sum(1 for r in final_results if r.auto_expected == True)
                auto_rejected = sum(1 for r in final_results if r.auto_expected == False)
                manual_review = sum(1 for r in final_results if r.expected is None)
                
                print(f"\n📊 Final Bootstrap Statistics:")
                print(f"   - Total triples processed: {total_triples}")
                print(f"   - Auto-confirmed (True): {auto_confirmed} ({auto_confirmed/total_triples*100:.1f}%)")
                print(f"   - Auto-rejected (False): {auto_rejected} ({auto_rejected/total_triples*100:.1f}%)")
                print(f"   - Manual review needed: {manual_review} ({manual_review/total_triples*100:.1f}%)")
                print(f"   - Coverage (auto-labeled): {(auto_confirmed+auto_rejected)/total_triples*100:.1f}%")
                
                print(f"\n🎉 Gold label bootstrapping completed successfully!")
                print(f"📂 Results saved to: {output_file}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error during gold label bootstrapping: {e}")
            return False


# Initialize the global Perplexity Graph Judge instance
try:
    gemini_judge = PerplexityGraphJudge(enable_console_logging=False)
    if PERPLEXITY_AVAILABLE:
        print("✓ Global Perplexity Graph Judge instance initialized")
    else:
        print("⚠️ Global Perplexity Graph Judge instance initialized in mock mode")
except Exception as e:
    print(f"✗ Failed to initialize global Perplexity Graph Judge: {e}")
    gemini_judge = None


# Load the evaluation dataset following existing patterns
try:
    if not DATASETS_AVAILABLE or not os.path.exists(input_file):
        print(f"⚠️ Using mock dataset for testing")
        # Create mock dataset
        mock_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Apple Founded by Steve Jobs ?", "input": "", "output": ""},
            {"instruction": "Is this true: Microsoft Founded by Mark Zuckerberg ?", "input": "", "output": ""}
        ]
        
        class MockDataEval:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __iter__(self):
                return iter(self.data)
            
            def __getitem__(self, index):
                return self.data[index]
        
        data_eval = MockDataEval(mock_data)
        print(f"📊 Using mock dataset: {len(data_eval)} instructions")
    else:
        total_input = load_dataset("json", data_files=input_file)
        
        # Dynamically adjust test_size based on available data
        available_samples = len(total_input["train"])
        print(f"✓ Dataset loaded: {available_samples} total samples")
        
        if available_samples <= 50:
            # For small datasets, use all samples
            data_eval = total_input["train"]
            print(f"✓ Using all {available_samples} samples for evaluation (small dataset)")
        else:
            # For larger datasets, use up to 499 samples or all available (whichever is smaller)
            test_size = min(499, available_samples - 1)  # Ensure test_size < available_samples
            if test_size <= 0:
                # If we can't split, use all samples
                data_eval = total_input["train"]
                print(f"✓ Using all {available_samples} samples for evaluation (cannot split)")
            else:
                data_eval = total_input["train"].train_test_split(
                    test_size=test_size, shuffle=True, seed=42
                )["test"]
                print(f"✓ Using {len(data_eval)} evaluation samples from {available_samples} total")
            
        print(f"📊 Final evaluation dataset size: {len(data_eval)} instructions")
    
except Exception as e:
    print(f"⚠️ Error loading dataset, using mock data: {e}")
    # Fallback to mock data
    mock_data = [
        {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
    ]
    
    class MockDataEval:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
    
    data_eval = MockDataEval(mock_data)

# Load instructions data directly for processing compatibility
try:
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            instructions = json.load(f)
        print(f"✓ Loaded {len(instructions)} instruction entries")
    else:
        print(f"⚠️ Input file not found, using mock instructions")
        instructions = [
            {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
        ]
except Exception as e:
    print(f"⚠️ Error loading instructions, using mock data: {e}")
    instructions = [
        {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
    ]


async def get_perplexity_completion(instruction, input_text=None):
    """
    Get completion from Perplexity API for graph judgment
    
    This function maintains compatibility with the existing pipeline
    while leveraging Perplexity's advanced reasoning capabilities.
    
    Args:
        instruction (str): The instruction/question for classification
        input_text (str, optional): Additional context (if any)
    
    Returns:
        str: The binary judgment result ("Yes" or "No")
    """
    if gemini_judge is None:
        print("✗ Perplexity Graph Judge not initialized")
        return "Error: Graph Judge not available"
    
    return await gemini_judge.judge_graph_triple(instruction, input_text)


def _generate_reasoning_file_path(csv_output_path: str, custom_path: Optional[str] = None) -> str:
    """
    Generate reasoning file path based on CSV output path
    
    Args:
        csv_output_path (str): Path to the main CSV output file
        custom_path (Optional[str]): Custom reasoning file path if specified
        
    Returns:
        str: Path for the reasoning JSON file
    """
    if custom_path:
        return custom_path
    
    # Auto-generate path based on CSV file name
    path_obj = Path(csv_output_path)
    reasoning_filename = path_obj.stem + "_reasoning" + ".json"
    return str(path_obj.parent / reasoning_filename)


async def process_instructions(explainable_mode: bool = False, reasoning_file_path: Optional[str] = None):
    """
    Process instructions with Gemini RAG system for graph judgment
    
    This function orchestrates the entire graph judgment evaluation process:
    1. Creates async tasks for all instruction-input pairs
    2. Executes them with concurrency control (standard or explainable mode)
    3. Collects binary judgment responses and optionally detailed reasoning
    4. Saves results in dual-file format: CSV (compatible) + JSON (explainable)
    
    Args:
        explainable_mode (bool): Whether to enable explainable reasoning mode
        reasoning_file_path (Optional[str]): Custom path for reasoning file
    
    The function leverages Gemini's grounding capabilities for more accurate
    graph validation while maintaining compatibility with existing infrastructure.
    """
    print("🚀 Starting Perplexity API Graph Judge processing...")
    
    if not data_eval or len(data_eval) == 0:
        print("✗ No evaluation data available")
        return
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(PERPLEXITY_CONCURRENT_LIMIT)
    
    async def limited_completion(item, index):
        """Rate-limited wrapper for get_gemini_completion with optional explainable mode"""
        async with semaphore:
            # Add delay between requests to be respectful to API
            await asyncio.sleep(PERPLEXITY_BASE_DELAY)
            
            instruction = item["instruction"]
            input_text = item.get("input", "")
            
            mode_indicator = "🧠" if explainable_mode else "🔄"
            print(f"{mode_indicator} Processing instruction {index + 1}/{len(data_eval)}")
            
            try:
                if explainable_mode:
                    # Get explainable judgment with detailed reasoning
                    explainable_result = await gemini_judge.judge_graph_triple_with_explanation(instruction, input_text)
                    binary_response = explainable_result.judgment
                    
                    # Prepare reasoning data for JSON output
                    reasoning_data = {
                        "index": index,
                        "prompt": instruction,
                        "judgment": explainable_result.judgment,
                        "confidence": explainable_result.confidence,
                        "reasoning": explainable_result.reasoning,
                        "evidence_sources": explainable_result.evidence_sources,
                        "alternative_suggestions": explainable_result.alternative_suggestions,
                        "error_type": explainable_result.error_type,
                        "processing_time": explainable_result.processing_time
                    }
                    
                    print(f"✅ Completed instruction {index + 1}/{len(data_eval)} - Result: {binary_response} (confidence: {explainable_result.confidence:.2f})")
                    return binary_response, reasoning_data
                else:
                    # Standard mode - get simple binary response
                    response = await get_perplexity_completion(instruction, input_text)
                    print(f"✅ Completed instruction {index + 1}/{len(data_eval)} - Result: {response}")
                    return response  # Return just the response, not a tuple
                    
            except Exception as e:
                print(f"❌ Failed instruction {index + 1}/{len(data_eval)} - Error: {str(e)[:100]}...")
                error_response = "Error: Failed to process"
                
                if explainable_mode:
                    # Create error reasoning data
                    error_reasoning = {
                        "index": index,
                        "prompt": instruction,
                        "judgment": error_response,
                        "confidence": 0.0,
                        "reasoning": f"Processing error: {str(e)}",
                        "evidence_sources": [],
                        "alternative_suggestions": [],
                        "error_type": "processing_error",
                        "processing_time": 0.0
                    }
                    return error_response, error_reasoning
                else:
                    return error_response  # Return just the error response, not a tuple
    
    # Create async tasks for all instruction-input pairs
    tasks = [limited_completion(item, i) for i, item in enumerate(data_eval)]

    # Execute all tasks with progress tracking
    mode_name = "Explainable" if explainable_mode else "Standard"
    print(f"📊 Processing {len(tasks)} graph judgment tasks in {mode_name} mode...")
    print(f"📊 Configuration: Max concurrent requests = {PERPLEXITY_CONCURRENT_LIMIT}")
    print(f"⏱️  Base delay between requests: {PERPLEXITY_BASE_DELAY} seconds")
    if explainable_mode:
        reasoning_output_path = _generate_reasoning_file_path(output_file, reasoning_file_path)
        print(f"🧠 Reasoning output: {reasoning_output_path}")
    print("-" * 60)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate responses and reasoning data
    responses = []
    reasoning_results = []
    
    for result in results:
        if isinstance(result, Exception):
            responses.append(f"Error: {str(result)}")
            if explainable_mode:
                reasoning_results.append({
                    "index": len(reasoning_results),
                    "prompt": "Error processing",
                    "judgment": "Error",
                    "confidence": 0.0,
                    "reasoning": f"Exception occurred: {str(result)}",
                    "evidence_sources": [],
                    "alternative_suggestions": [],
                    "error_type": "system_error",
                    "processing_time": 0.0
                })
        else:
            if explainable_mode:
                # In explainable mode, result is a tuple (binary_response, reasoning_data)
                binary_response, reasoning_data = result
                responses.append(binary_response)
                reasoning_results.append(reasoning_data)
            else:
                # In standard mode, result is just the response string
                responses.append(result)

    # Write responses to CSV file (standard format, compatible with existing pipeline)
    print(f"💾 Saving CSV results to {output_file}...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt", "generated"])  # Standard header format

        # Process each response and ensure proper formatting
        successful_responses = 0
        error_responses = 0
        yes_responses = 0
        no_responses = 0
        
        for item, response in zip(data_eval, responses):
            prompt = item["instruction"]
            
            # Clean response and ensure it matches expected format
            cleaned_response = str(response).strip().replace('\n', ' ')
            
            # Count response types for statistics
            if "Error:" not in cleaned_response:
                successful_responses += 1
                if cleaned_response == "Yes":
                    yes_responses += 1
                elif cleaned_response == "No":
                    no_responses += 1
            else:
                error_responses += 1
            
            writer.writerow([prompt, cleaned_response])
    
    # Save reasoning file if in explainable mode
    if explainable_mode and reasoning_results:
        print(f"🧠 Saving explainable reasoning to {reasoning_output_path}...")
        gemini_judge._save_reasoning_file(reasoning_results, reasoning_output_path)
    
    # Print completion statistics
        print(f"✅ Perplexity API Graph Judge processing completed!")
    print(f"📊 Results saved to: {output_file}")
    if explainable_mode:
        print(f"🧠 Reasoning saved to: {reasoning_output_path}")
    print(f"📈 Processing statistics:")
    print(f"   - Successful responses: {successful_responses}")
    print(f"   - Error responses: {error_responses}")
    print(f"   - 'Yes' judgments: {yes_responses}")
    print(f"   - 'No' judgments: {no_responses}")
    print(f"   - Success rate: {successful_responses/len(responses)*100:.1f}%")
    
    # Additional statistics
    if successful_responses > 0:
        positive_rate = yes_responses / successful_responses * 100
        print(f"   - Positive judgment rate: {positive_rate:.1f}%")
    
    # Explainable mode specific statistics
    if explainable_mode and reasoning_results:
        avg_confidence = sum(r.get("confidence", 0.0) for r in reasoning_results if r.get("confidence") is not None) / len(reasoning_results)
        error_types = [r.get("error_type") for r in reasoning_results if r.get("error_type")]
        unique_error_types = set(error_types)
        
        print(f"🧠 Explainable mode statistics:")
        print(f"   - Average confidence: {avg_confidence:.2f}")
        print(f"   - Unique error types: {len(unique_error_types)}")
        if unique_error_types:
            print(f"   - Error types found: {', '.join(unique_error_types)}")


def validate_input_file():
    """
    Validate that the input file exists and has the correct format.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"✗ Input file not found: {input_file}")
        print("Please ensure the file exists or run the data preparation step first.")
        return False
    
    # Validate JSON structure
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print(f"✗ Invalid input file format: expected non-empty list")
            return False
        
        # Check required fields
        required_fields = ["instruction"]
        sample = data[0]
        
        for field in required_fields:
            if field not in sample:
                print(f"✗ Missing required field '{field}' in input data")
                return False
        
        print(f"✓ Input file validation passed: {len(data)} entries")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON format in input file: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating input file: {e}")
        return False


def create_output_directory():
    """
    Ensure the output directory exists before writing results.
    """
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")


def parse_arguments():
    """
    Parse command line arguments for different operation modes
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Perplexity API Graph Judge - Enhanced Knowledge Graph Triple Validation & Gold Label Bootstrapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard graph judgment mode
  python run_gj.py
  
  # Gold label bootstrapping mode
  python run_gj.py --bootstrap \
    --triples-file ../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs.txt \
    --source-file ../datasets/KIMI_result_DreamOf_RedChamber/Iteration1/test_denoised.target \
    --output ../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/gold_bootstrap.csv \
    --threshold 0.8 --sample-rate 0.15
        """
    )
    
    # Operation mode
    parser.add_argument(
        '--bootstrap', 
        action='store_true',
        help='Run gold label bootstrapping instead of graph judgment'
    )
    
    parser.add_argument(
        '--explainable',
        action='store_true',
        help='Enable explainable mode - generate detailed reasoning file alongside CSV output'
    )
    
    # Gold label bootstrapping arguments
    parser.add_argument(
        '--triples-file',
        type=str,
        help='Path to the triples file for bootstrapping (required for --bootstrap)'
    )
    
    parser.add_argument(
        '--source-file',
        type=str,
        help='Path to the source text file for bootstrapping (required for --bootstrap)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (for bootstrapping mode)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='RapidFuzz similarity threshold for Stage 1 (default: 0.8)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=0.15,
        help='Sampling rate for manual review (default: 0.15)'
    )
    
    # Explainable mode arguments
    parser.add_argument(
        '--reasoning-file',
        type=str,
        help='Custom path for reasoning file output (optional, auto-generated if not specified)'
    )
    
    return parser.parse_args()


async def run_gold_label_bootstrapping(args):
    """
    Run the gold label bootstrapping pipeline
    
    Args:
        args: Parsed command line arguments
    """
    # Validate required arguments
    if not args.triples_file:
        print("❌ --triples-file is required for bootstrapping mode")
        sys.exit(1)
    
    if not args.source_file:
        print("❌ --source-file is required for bootstrapping mode")
        sys.exit(1)
    
    if not args.output:
        print("❌ --output is required for bootstrapping mode")
        sys.exit(1)
    
    # Update global configuration
    GOLD_BOOTSTRAP_CONFIG['fuzzy_threshold'] = args.threshold
    GOLD_BOOTSTRAP_CONFIG['sample_rate'] = args.sample_rate
    
    print(f"📊 Bootstrap Configuration:")
    print(f"   - Fuzzy threshold: {args.threshold}")
    print(f"   - Sample rate: {args.sample_rate}")
    print(f"   - Triples file: {args.triples_file}")
    print(f"   - Source file: {args.source_file}")
    print(f"   - Output file: {args.output}")
    
    if gemini_judge is None:
        print("❌ Perplexity Graph Judge initialization failed. Exiting.")
        sys.exit(1)
    
    # Run the bootstrapping process
    success = await gemini_judge.bootstrap_gold_labels(
        triples_file=args.triples_file,
        source_file=args.source_file,
        output_file=args.output
    )
    
    if not success:
        print("❌ Gold label bootstrapping failed.")
        sys.exit(1)


async def run_graph_judgment(explainable_mode: bool = False, reasoning_file_path: Optional[str] = None):
    """
    Run the graph judgment pipeline (standard or explainable mode)
    
    Args:
        explainable_mode (bool): Whether to enable explainable reasoning mode
        reasoning_file_path (Optional[str]): Custom path for reasoning file
    """
    # Pre-flight validation checks
    print("🔍 Running pre-flight checks...")
    
    if not validate_input_file():
        print("❌ Pre-flight validation failed. Exiting.")
        sys.exit(1)
    
    create_output_directory()
    
    # Run the main processing pipeline with specified mode
    await process_instructions(explainable_mode, reasoning_file_path)
    
    print("\n🎉 Graph judgment pipeline completed successfully!")
    print(f"📂 CSV results available at: {output_file}")
    
    if explainable_mode:
        reasoning_path = _generate_reasoning_file_path(output_file, reasoning_file_path)
        print(f"🧠 Reasoning results available at: {reasoning_path}")
        print("\n📋 Next steps:")
        print("   1. Review the generated CSV file for binary judgment results")
        print("   2. Review the reasoning JSON file for detailed explanations")
        print("   3. Run evaluation metrics against ground truth data")
        print("   4. Analyze confidence scores and error patterns")
        print("   5. Use alternative suggestions for data quality improvement")
    else:
        print("\n📋 Next steps:")
        print("   1. Review the generated CSV file for judgment results")
        print("   2. Run evaluation metrics against ground truth data")
        print("   3. Consider using --explainable mode for detailed insights")


if __name__ == "__main__":
    """
    Main execution block with comprehensive error handling and validation.
    
    This section supports two operation modes:
    1. Standard graph judgment pipeline (default)
    2. Gold label bootstrapping (--bootstrap flag)
    
    The execution flow:
    1. Parse command line arguments
    2. Set up terminal logging
    3. Initialize and validate systems
    4. Run the appropriate pipeline based on mode
    5. Handle errors gracefully
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up terminal logging first
    try:
        log_filepath = setup_terminal_logging()
        terminal_logger = TerminalLogger(log_filepath)
        terminal_logger.start_session()
        # Don't use print here since it will be logged anyway
        terminal_logger.original_print(f"📝 Logging to: {log_filepath}")
    except Exception as e:
        print(f"⚠️ Failed to set up logging: {e}")
        print("Continuing without file logging...")
        terminal_logger = None
    
    # Print header based on mode
    if args.bootstrap:
        print("=" * 70)
        print("🎯 Perplexity API Graph Judge - Gold Label Bootstrapping")
        print("=" * 70)
    elif args.explainable:
        print("=" * 70)
        print("🧠 Perplexity API Graph Judge - Enhanced Explainable Knowledge Graph Validation")
        print("=" * 70)
        print("📋 Mode: Dual-output (CSV + Reasoning JSON)")
    else:
        print("=" * 70)
        print("🎯 Perplexity API Graph Judge - Knowledge Graph Triple Validation")
        print("=" * 70)
        print("📋 Mode: Standard (CSV output only)")
    
    # Pre-flight validation checks
    if gemini_judge is None:
        print("❌ Perplexity Graph Judge initialization failed. Exiting.")
        sys.exit(1)
    
    if not PERPLEXITY_AVAILABLE:
        print("⚠️ Running in mock mode - some features may be limited")
        print("📦 For full functionality, install: pip install litellm python-dotenv")
    
    if args.bootstrap and not RAPIDFUZZ_AVAILABLE:
        print("⚠️ RapidFuzz not available - using mock similarity scoring")
        print("📦 For full functionality, install: pip install rapidfuzz")
    
    # Run the appropriate pipeline
    try:
        if args.bootstrap:
            asyncio.run(run_gold_label_bootstrapping(args))
        else:
            # Run graph judgment with explainable mode if specified
            asyncio.run(run_graph_judgment(
                explainable_mode=args.explainable,
                reasoning_file_path=args.reasoning_file
            ))
        
        # End logging session
        if terminal_logger:
            terminal_logger.end_session()
            print(f"📝 Complete log saved to: {log_filepath}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        if terminal_logger:
            terminal_logger.log_message("Processing interrupted by user", "WARNING")
            terminal_logger.end_session()
    except Exception as e:
        print(f"\n❌ Critical error during processing: {e}")
        print("Please check your configuration and try again.")
        if terminal_logger:
            terminal_logger.log_message(f"Critical error: {e}", "ERROR")
            terminal_logger.end_session()
        sys.exit(1)
