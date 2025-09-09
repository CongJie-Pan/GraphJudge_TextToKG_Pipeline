"""
Core GraphJudge implementation using Perplexity API.

This module contains the main PerplexityGraphJudge class that provides
the core functionality for graph triple validation using Perplexity's
sonar-reasoning capabilities.
"""

import os
import asyncio
import time
from typing import Optional, Dict, Any
# Try to import litellm
try:
    from litellm import acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    print("⚠️ litellm not available. Install with: pip install litellm")
    LITELLM_AVAILABLE = False
    
    # Mock acompletion for testing
    async def acompletion(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.choices = [type('MockChoice', (), {
                    'message': type('MockMessage', (), {'content': 'Yes'})()
                })()]
        return MockResponse()
from .config import (
    PERPLEXITY_MODEL, PERPLEXITY_RETRY_ATTEMPTS, PERPLEXITY_BASE_DELAY,
    PERPLEXITY_REASONING_EFFORT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
)
from .data_structures import ExplainableJudgment
from .prompt_engineering import PromptEngineer


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
        Initialize the Perplexity Graph Judge system.
        
        Args:
            model_name (str): Perplexity model to use
            reasoning_effort (str): Reasoning effort level (low/medium/high)
            enable_console_logging (bool): Whether to show console logs
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.enable_logging = enable_console_logging
        self.temperature = DEFAULT_TEMPERATURE
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.is_mock = not (self._check_perplexity_availability() and LITELLM_AVAILABLE)
        
        # Initialize prompt engineer instance for integration testing
        self.prompt_engineer = PromptEngineer()
        
        # Validate API key (only in non-mock mode)
        if not self.is_mock and not os.getenv('PERPLEXITYAI_API_KEY'):
            raise ValueError("PERPLEXITYAI_API_KEY not found in environment variables")
        
        if not self.is_mock:
            print(f"✓ Perplexity Graph Judge initialized with model: {model_name}")
            print(f"✓ Reasoning effort set to: {reasoning_effort}")
        else:
            print(f"⚠️ Perplexity Graph Judge running in mock mode")
    
    def _check_perplexity_availability(self) -> bool:
        """
        Check if Perplexity API is available.
        
        Returns:
            bool: True if Perplexity API is available, False otherwise
        """
        # Check if litellm is available and API key is present
        if not LITELLM_AVAILABLE:
            print(f"⚠️ Failed to import Perplexity API system")
            print("⚠️ Running in compatibility mode without Perplexity API features")
            print("📦 To enable full functionality, install: pip install litellm python-dotenv")
            return False
        
        return True
    
    async def judge_graph_triple(self, instruction: str, input_text: str = None) -> str:
        """
        Judge a graph triple using Perplexity API.
        
        Args:
            instruction (str): Graph judgment instruction
            input_text (str): Additional context (optional)
            
        Returns:
            str: Binary judgment result ("Yes" or "No")
        """
        # Handle mock mode
        if self.is_mock:
            return self._mock_judgment(instruction)
        
        max_retries = PERPLEXITY_RETRY_ATTEMPTS
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create specialized prompt for graph judgment
                prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
                
                # Perplexity API call
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )
                
                # Parse and validate the response
                judgment = PromptEngineer.parse_response(response)
                
                # Extract and log citation information
                citation_summary = PromptEngineer.get_citation_summary(response)
                if citation_summary.has_citations and self.enable_logging:
                    print(f"Graph judgment: {judgment} (with {citation_summary.total_citations} citations)")
                    for citation in citation_summary.citations[:3]:  # Show first 3 citations
                        print(f"  📚 Citation {citation.number}: {citation.title}")
                
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
        Judge a graph triple with detailed explainable reasoning.
        
        Args:
            instruction (str): Graph judgment instruction
            input_text (str): Additional context (optional)
            include_citations (bool): Whether to include citation information
            
        Returns:
            ExplainableJudgment: Comprehensive judgment with reasoning, confidence, and evidence
        """
        start_time = time.time()
        
        # Handle mock mode
        if self.is_mock:
            return self._mock_explainable_judgment(instruction, start_time)
        
        max_retries = PERPLEXITY_RETRY_ATTEMPTS
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create specialized prompt for explainable graph judgment
                prompt = PromptEngineer.create_explainable_judgment_prompt(instruction)
                
                # Perplexity API call for explainable judgment
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )
                
                # Parse and validate the response for explainable judgment
                explainable_judgment = PromptEngineer.parse_explainable_response(response)
                
                # Enhance with citation information if requested
                if include_citations:
                    citation_summary = PromptEngineer.get_citation_summary(response)
                    if citation_summary.has_citations:
                        # Add citation sources to evidence sources
                        enhanced_evidence_sources = list(explainable_judgment.evidence_sources)
                        enhanced_evidence_sources.append(f"perplexity_citations({citation_summary.total_citations})")
                        
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
                citation_summary = PromptEngineer.get_citation_summary(response)
                if citation_summary.has_citations and self.enable_logging:
                    print(f"Explainable judgment: {final_judgment.judgment} (confidence: {final_judgment.confidence:.2f}, with {citation_summary.total_citations} citations)")
                    for citation in citation_summary.citations[:3]:  # Show first 3 citations
                        print(f"  📚 Citation {citation.number}: {citation.title}")
                
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
    
    async def judge_graph_triple_streaming(self, instruction: str, 
                                         stream_container=None) -> str:
        """
        Streaming version of graph triple judgment for real-time processing feedback.
        
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
            prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
            
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
            
            return PromptEngineer.parse_response(mock_response)
            
        except Exception as e:
            print(f"Error in streaming judgment: {e}")
            return "No"  # Conservative default
    
    async def judge_graph_triple_with_citations(self, instruction: str, input_text: str = None) -> Dict[str, Any]:
        """
        Get graph judgment with detailed citation information.
        
        Args:
            instruction (str): Graph judgment instruction
            input_text (str): Additional context (optional)
            
        Returns:
            Dict[str, Any]: Judgment result with citation details
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
                prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
                
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )
                
                judgment = PromptEngineer.parse_response(response)
                citation_summary = PromptEngineer.get_citation_summary(response)
                
                return {
                    "judgment": judgment,
                    "confidence": 0.8 if citation_summary.has_citations else 0.6,
                    "citations": [c._asdict() for c in citation_summary.citations],
                    "citation_count": citation_summary.total_citations,
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
    
    def _mock_judgment(self, instruction: str) -> str:
        """
        Provide mock judgment for testing when Perplexity API is not available.
        
        Args:
            instruction (str): Graph judgment instruction
            
        Returns:
            str: Mock judgment result
        """
        # Simple mock logic for testing
        if "Apple Founded by Steve Jobs" in instruction or "曹雪芹 創作 紅樓夢" in instruction:
            return "Yes"
        elif "Microsoft Founded by Mark Zuckerberg" in instruction:
            return "No"
        else:
            return "Yes"  # Default for unknown cases in mock mode
    
    def _mock_explainable_judgment(self, instruction: str, start_time: float) -> ExplainableJudgment:
        """
        Provide mock explainable judgment for testing.
        
        Args:
            instruction (str): Graph judgment instruction
            start_time (float): Start time for processing
            
        Returns:
            ExplainableJudgment: Mock explainable judgment result
        """
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
