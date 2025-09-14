"""
Simplified Graph Judge module for GraphJudge Streamlit Pipeline.

This module extracts the essential graph judgment functionality from the complex
run_gj.py script (~2200 lines) into a clean, simplified interface suitable for
Streamlit integration.

Key simplifications from original:
- Focus on Perplexity API integration only (defer multi-API complexity)
- Synchronous execution (Streamlit-compatible)
- Clean function interface: judge_triples(triples: List[Triple]) -> JudgmentResult
- Basic explainable reasoning mode
- Simplified error handling
- No file-based I/O or complex retry mechanisms

Maintains core functionality:
- Graph triple validation using Perplexity sonar-reasoning model
- Binary judgment results with confidence scores
- Basic explainable reasoning with evidence sources
- Proper response parsing and validation
"""

import time
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from ..core.models import Triple, JudgmentResult
    from ..utils.api_client import get_api_client
    from ..utils.detailed_logger import DetailedLogger
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.models import Triple, JudgmentResult
    from utils.api_client import get_api_client
    from utils.detailed_logger import DetailedLogger


@dataclass
class ExplainableJudgment:
    """
    Detailed judgment result with explainable reasoning.
    Simplified version of the complex ExplainableJudgment from original script.
    """
    judgment: str  # "Yes" or "No"
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Explanation of the judgment
    evidence_sources: List[str]  # Types of evidence used
    alternative_suggestions: List[Dict[str, Any]]  # Alternative correct triples
    error_type: Optional[str]  # Type of error if judgment is "No"
    processing_time: float  # Time taken for judgment


class GraphJudge:
    """
    Simplified Graph Judge that validates knowledge graph triples using Perplexity API.
    
    This class extracts core functionality from the complex PerplexityGraphJudge
    while maintaining essential features for graph triple validation.
    """
    
    def __init__(self, model_name: str = "perplexity/sonar-reasoning"):
        """
        Initialize the Graph Judge with Perplexity API integration.

        Args:
            model_name: Perplexity model identifier for judgment
        """
        self.model_name = model_name
        self.api_client = get_api_client()
        self.detailed_logger = DetailedLogger(phase="judgment")

        # Configuration from original script
        self.temperature = 1.0  # GPT-5 models only support temperature=1
        self.max_tokens = 2000  # Sufficient for judgment responses
    
    def judge_triples(self, triples: List[Triple]) -> JudgmentResult:
        """
        Judge a list of triples and return binary judgment results.

        This is the main interface function as specified in spec.md Section 8.

        Args:
            triples: List of Triple objects to validate

        Returns:
            JudgmentResult with judgments, confidence scores, and metadata
        """
        self.detailed_logger.log_info("JUDGMENT", "Starting graph judgment process", {
            "triple_count": len(triples),
            "triples": [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in triples[:5]],  # First 5 triples
            "model": self.model_name
        })

        if not triples:
            self.detailed_logger.log_warning("JUDGMENT", "No triples provided for judgment")
            return JudgmentResult(
                judgments=[],
                confidence=[],
                success=True,
                processing_time=0.0
            )

        start_time = time.time()
        judgments = []
        confidence_scores = []
        api_calls = 0

        try:
            for idx, triple in enumerate(triples):
                self.detailed_logger.log_debug("JUDGMENT", f"Processing triple {idx + 1}/{len(triples)}", {
                    "triple_index": idx,
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object
                })

                try:
                    # Convert triple to judgment instruction format
                    instruction = self._create_judgment_instruction(triple)
                    self.detailed_logger.log_debug("JUDGMENT", "Created judgment instruction", {
                        "triple_index": idx,
                        "instruction_length": len(instruction)
                    })

                    # Get binary judgment from Perplexity
                    self.detailed_logger.log_info("API", f"Making judgment API call for triple {idx + 1}")
                    judgment, had_error = self._judge_single_triple_with_error_flag(instruction)

                    binary_judgment = judgment == "Yes"
                    judgments.append(binary_judgment)

                    # Assign confidence based on response clarity and error status
                    if had_error:
                        confidence = 0.0  # Zero confidence for API errors
                        self.detailed_logger.log_warning("JUDGMENT", f"API error for triple {idx + 1}, setting confidence to 0", {
                            "triple_index": idx,
                            "judgment_response": judgment
                        })
                    else:
                        confidence = self._estimate_confidence(judgment)
                        self.detailed_logger.log_debug("JUDGMENT", f"Triple {idx + 1} judged successfully", {
                            "triple_index": idx,
                            "judgment": judgment,
                            "binary_result": binary_judgment,
                            "confidence": confidence
                        })

                    confidence_scores.append(confidence)
                    api_calls += 1

                except Exception as e:
                    # Handle individual triple errors gracefully
                    judgments.append(False)  # Conservative default
                    confidence_scores.append(0.0)  # Zero confidence for errors
                    self.detailed_logger.log_error("JUDGMENT", f"Error judging triple {idx + 1}", {
                        "triple_index": idx,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.object
                    })
            
            processing_time = time.time() - start_time

            # Log final judgment results
            approved_count = sum(1 for j in judgments if j)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            self.detailed_logger.log_info("JUDGMENT", "Graph judgment completed successfully", {
                "total_triples": len(triples),
                "approved_triples": approved_count,
                "rejected_triples": len(triples) - approved_count,
                "approval_rate": approved_count / len(triples) if triples else 0.0,
                "average_confidence": avg_confidence,
                "api_calls_made": api_calls,
                "processing_time": processing_time
            })

            return JudgmentResult(
                judgments=judgments,
                confidence=confidence_scores,
                success=True,
                processing_time=processing_time
            )

        except Exception as e:
            # Handle catastrophic errors
            processing_time = time.time() - start_time

            self.detailed_logger.log_error("JUDGMENT", "Graph judgment failed with catastrophic error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "triples_count": len(triples),
                "processing_time": processing_time
            })

            return JudgmentResult(
                judgments=[False] * len(triples),  # Conservative defaults
                confidence=[0.0] * len(triples),
                success=False,
                processing_time=processing_time,
                error=str(e)
            )
    
    def judge_triples_with_explanations(
        self,
        triples: List[Triple],
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Judge triples with detailed explanations for each judgment.

        This provides the explainable reasoning mode as specified in spec.md FR-GJ5.

        Args:
            triples: List of Triple objects to validate
            include_reasoning: Whether to include detailed reasoning

        Returns:
            Dictionary with judgments and detailed explanations
        """
        self.detailed_logger.log_info("JUDGMENT", "Starting explainable graph judgment process", {
            "triple_count": len(triples),
            "include_reasoning": include_reasoning,
            "triples": [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in triples[:3]]  # First 3 triples
        })

        if not triples:
            self.detailed_logger.log_warning("JUDGMENT", "No triples provided for explainable judgment")
            return {
                "judgments": [],
                "explanations": [],
                "confidence": [],
                "success": True,
                "processing_time": 0.0,
                "metadata": {"total_triples": 0, "api_calls": 0}
            }

        start_time = time.time()
        judgments = []
        explanations = []
        confidence_scores = []
        api_calls = 0

        try:
            for idx, triple in enumerate(triples):
                self.detailed_logger.log_debug("JUDGMENT", f"Processing explainable judgment for triple {idx + 1}/{len(triples)}", {
                    "triple_index": idx,
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "include_reasoning": include_reasoning
                })

                try:
                    # Convert triple to judgment instruction format
                    instruction = self._create_judgment_instruction(triple)

                    if include_reasoning:
                        # Get detailed explanation
                        self.detailed_logger.log_info("API", f"Making explainable judgment API call for triple {idx + 1}")
                        explanation_result = self._judge_with_explanation(instruction)

                        binary_judgment = explanation_result.judgment == "Yes"
                        judgments.append(binary_judgment)
                        confidence_scores.append(explanation_result.confidence)
                        explanations.append({
                            "reasoning": explanation_result.reasoning,
                            "evidence_sources": explanation_result.evidence_sources,
                            "error_type": explanation_result.error_type
                        })

                        self.detailed_logger.log_debug("JUDGMENT", f"Explainable judgment completed for triple {idx + 1}", {
                            "triple_index": idx,
                            "judgment": explanation_result.judgment,
                            "binary_result": binary_judgment,
                            "confidence": explanation_result.confidence,
                            "reasoning_length": len(explanation_result.reasoning) if explanation_result.reasoning else 0,
                            "evidence_sources_count": len(explanation_result.evidence_sources) if explanation_result.evidence_sources else 0
                        })
                    else:
                        # Simple binary judgment
                        self.detailed_logger.log_info("API", f"Making simple judgment API call for triple {idx + 1}")
                        judgment = self._judge_single_triple(instruction)
                        binary_judgment = judgment == "Yes"
                        judgments.append(binary_judgment)
                        confidence = self._estimate_confidence(judgment)
                        confidence_scores.append(confidence)
                        explanations.append({"reasoning": f"Binary judgment: {judgment}"})

                        self.detailed_logger.log_debug("JUDGMENT", f"Simple judgment completed for triple {idx + 1}", {
                            "triple_index": idx,
                            "judgment": judgment,
                            "binary_result": binary_judgment,
                            "confidence": confidence
                        })

                    api_calls += 1

                except Exception as e:
                    # Handle individual triple errors gracefully
                    judgments.append(False)
                    confidence_scores.append(0.0)
                    explanations.append({
                        "reasoning": f"Error during processing: {str(e)}",
                        "error_type": "processing_error"
                    })
                    self.detailed_logger.log_error("JUDGMENT", f"Error judging triple {idx + 1} with explanation", {
                        "triple_index": idx,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.object
                    })

            processing_time = time.time() - start_time

            # Log final explainable judgment results
            approved_count = sum(1 for j in judgments if j)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            self.detailed_logger.log_info("JUDGMENT", "Explainable graph judgment completed successfully", {
                "total_triples": len(triples),
                "approved_triples": approved_count,
                "rejected_triples": len(triples) - approved_count,
                "approval_rate": approved_count / len(triples) if triples else 0.0,
                "average_confidence": avg_confidence,
                "api_calls_made": api_calls,
                "reasoning_enabled": include_reasoning,
                "processing_time": processing_time
            })

            return {
                "judgments": judgments,
                "explanations": explanations,
                "confidence": confidence_scores,
                "success": True,
                "processing_time": processing_time,
                "metadata": {
                    "total_triples": len(triples),
                    "api_calls": api_calls,
                    "model_used": self.model_name,
                    "reasoning_enabled": include_reasoning
                }
            }

        except Exception as e:
            # Handle catastrophic errors
            processing_time = time.time() - start_time

            self.detailed_logger.log_error("JUDGMENT", "Explainable graph judgment failed with catastrophic error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "triples_count": len(triples),
                "include_reasoning": include_reasoning,
                "processing_time": processing_time
            })

            return {
                "judgments": [False] * len(triples),
                "explanations": [{"reasoning": str(e), "error_type": "catastrophic_failure"}] * len(triples),
                "confidence": [0.0] * len(triples),
                "success": False,
                "processing_time": processing_time,
                "error": str(e),
                "metadata": {
                    "total_triples": len(triples),
                    "api_calls": api_calls,
                    "error_type": "catastrophic_failure"
                }
            }
    
    def _create_judgment_instruction(self, triple: Triple) -> str:
        """
        Convert a Triple object to judgment instruction format.
        Based on the original script's instruction format.
        
        Args:
            triple: Triple object with subject, relation, object
            
        Returns:
            Instruction string in format "Is this true: Subject Relation Object ?"
        """
        return f"Is this true: {triple.subject} {triple.predicate} {triple.object} ?"
    
    def _judge_single_triple_with_error_flag(self, instruction: str) -> Tuple[str, bool]:
        """
        Get binary judgment for a single triple with error flag.

        Args:
            instruction: Judgment instruction string

        Returns:
            Tuple of (judgment, had_error) where had_error indicates API failure
        """
        try:
            judgment = self._judge_single_triple(instruction)
            self.detailed_logger.log_debug("JUDGMENT", "Single triple judgment successful", {
                "judgment": judgment,
                "had_error": False
            })
            return judgment, False  # No error
        except Exception as e:
            self.detailed_logger.log_warning("JUDGMENT", "Single triple judgment failed, using conservative default", {
                "error": str(e),
                "error_type": type(e).__name__,
                "default_judgment": "No",
                "had_error": True
            })
            return "No", True  # Error occurred, conservative default
    
    def _judge_single_triple(self, instruction: str) -> str:
        """
        Get binary judgment for a single triple using Perplexity API.
        Simplified version of the original judge_graph_triple method.
        
        Args:
            instruction: Judgment instruction string
            
        Returns:
            Binary judgment ("Yes" or "No")
        """
        # Create specialized prompt for graph judgment
        prompt = self._create_judgment_prompt(instruction)
        
        try:
            # Call Perplexity API through unified client
            response = self.api_client.call_perplexity(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response to extract binary judgment
            return self._parse_binary_response(response)
            
        except Exception as e:
            print(f"Error in API call for instruction '{instruction}': {e}")
            raise  # Re-raise the exception so caller can detect the error
    
    def _judge_with_explanation(self, instruction: str) -> ExplainableJudgment:
        """
        Get detailed judgment with explanation for a single triple.
        Simplified version of the original judge_graph_triple_with_explanation method.
        
        Args:
            instruction: Judgment instruction string
            
        Returns:
            ExplainableJudgment with detailed reasoning
        """
        start_time = time.time()
        
        # Create specialized prompt for explainable judgment
        prompt = self._create_explainable_prompt(instruction)
        
        try:
            # Call Perplexity API through unified client
            response = self.api_client.call_perplexity(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response to extract detailed judgment
            return self._parse_explainable_response(response, start_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error in explainable judgment for instruction '{instruction}': {e}")
            
            return ExplainableJudgment(
                judgment="No",
                confidence=0.0,
                reasoning=f"Error during processing: {str(e)}",
                evidence_sources=[],
                alternative_suggestions=[],
                error_type="processing_error",
                processing_time=processing_time
            )
    
    def _create_judgment_prompt(self, instruction: str) -> str:
        """
        Create specialized prompt for binary graph judgment.
        Based on the original _create_graph_judgment_prompt method.
        
        Args:
            instruction: The judgment instruction
            
        Returns:
            Optimized prompt for binary judgment
        """
        # Extract triple from instruction format
        triple_part = instruction.replace("Is this true: ", "").replace(" ?", "")
        
        prompt = f"""You are a knowledge graph validation expert. Please evaluate whether the following triple statement is factually correct.

Task Requirements:
1. Output only "Yes" or "No" (no additional text)
2. Base your judgment on reliable information sources
3. For "Dream of the Red Chamber" (紅樓夢) related content, pay special attention to literary accuracy
4. For real-world facts, refer to authoritative information

Evaluation Rules:
- If the triple is syntactically correct and factually accurate, answer "Yes"
- If the triple is syntactically incorrect or factually wrong, answer "No"

Triple to evaluate:
{triple_part}

Please answer only "Yes" or "No":"""
        
        return prompt
    
    def _create_explainable_prompt(self, instruction: str) -> str:
        """
        Create specialized prompt for explainable graph judgment.
        Based on the original _create_explainable_judgment_prompt method.
        
        Args:
            instruction: The judgment instruction
            
        Returns:
            Enhanced prompt for detailed reasoning
        """
        # Extract triple from instruction format
        triple_part = instruction.replace("Is this true: ", "").replace(" ?", "")
        
        prompt = f"""You are a knowledge graph validation expert. Please analyze the following triple statement and provide a structured judgment result.

Triple to evaluate: {triple_part}

Please provide a complete analysis in the following format:

1. Judgment: [Answer only "Yes" or "No"]

2. Confidence: [A number between 0.0 and 1.0 indicating your certainty level]

3. Detailed Reasoning: [Explain your judgment, including:
   - Syntactic correctness analysis
   - Factual accuracy assessment  
   - Relevant background knowledge or evidence
   - For 《紅樓夢》(Dream of the Red Chamber) content, refer to original text]

4. Evidence Sources: [List the types of knowledge used, such as: historical_records, literary_works, general_knowledge, domain_expertise]

5. Error Type (if judgment is "No"): [Specify error type: factual_error, syntactic_error, relationship_error, or other]

Please format your response clearly with numbered sections."""
        
        return prompt
    
    def _parse_binary_response(self, response: str) -> str:
        """
        Parse API response to extract binary judgment.
        Based on the original _parse_response method.
        
        Args:
            response: Raw response from Perplexity API
            
        Returns:
            Binary judgment ("Yes" or "No")
        """
        if not response:
            return "No"
        
        response_text = str(response).strip()
        
        # Look for explicit Yes/No patterns
        if re.search(r'\byes\b', response_text, re.IGNORECASE):
            return "Yes"
        elif re.search(r'\bno\b', response_text, re.IGNORECASE):
            return "No"
        elif re.search(r'\b是\b|\b正確\b|\b對\b', response_text):
            return "Yes"
        elif re.search(r'\b否\b|\b錯誤\b|\b不對\b|\b不是\b', response_text):
            return "No"
        else:
            # Analyze content for sentiment if no clear binary response
            # Be more conservative with positive indicators to avoid false positives
            positive_indicators = ['correct', 'accurate', 'valid', '正確', '是的', '對的']
            negative_indicators = ['incorrect', 'false', 'wrong', 'invalid', '錯誤', '不對', '否']
            
            # Add ambiguous phrases that should default to "No"
            ambiguous_indicators = ['could be', 'might be', 'maybe', 'perhaps', 'possibly', 'it depends']
            
            response_lower = response_text.lower()
            
            # Check for ambiguous phrases first - these should default to "No"
            ambiguous_score = sum(1 for indicator in ambiguous_indicators if indicator in response_lower)
            if ambiguous_score > 0:
                print(f"Warning: Ambiguous response, defaulting to No: {response_text[:100]}...")
                return "No"
            
            positive_score = sum(1 for indicator in positive_indicators if indicator in response_lower)
            negative_score = sum(1 for indicator in negative_indicators if indicator in response_lower)
            
            # Only include "true" as positive if it's not in ambiguous context
            if 'true' in response_lower and 'could be true' not in response_lower and 'might be true' not in response_lower:
                positive_score += 1
            
            if positive_score > negative_score:
                return "Yes"
            elif negative_score > positive_score:
                return "No"
            else:
                # Default to No for ambiguous responses (conservative approach)
                print(f"Warning: Ambiguous response, defaulting to No: {response_text[:100]}...")
                return "No"
    
    def _parse_explainable_response(self, response: str, start_time: float) -> ExplainableJudgment:
        """
        Parse API response to extract detailed explainable judgment.
        Simplified version of the original _parse_explainable_response method.
        
        Args:
            response: Raw response from Perplexity API
            start_time: Start time for processing time calculation
            
        Returns:
            ExplainableJudgment with detailed analysis
        """
        processing_time = time.time() - start_time
        
        if not response:
            return ExplainableJudgment(
                judgment="No",
                confidence=0.0,
                reasoning="Empty response from API",
                evidence_sources=[],
                alternative_suggestions=[],
                error_type="empty_response",
                processing_time=processing_time
            )
        
        response_text = str(response).strip()
        
        try:
            # Extract judgment (simplified parsing)
            judgment = self._parse_binary_response(response_text)
            
            # Extract confidence (look for decimal numbers)
            confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.75
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            # Extract reasoning (look for reasoning section or use full response)
            reasoning_match = re.search(r'reasoning[:\s]*(.+?)(?=\d\.|evidence|$)', response_text, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text[:500]
            
            # Extract evidence sources (simplified)
            evidence_sources = []
            if 'historical' in response_text.lower():
                evidence_sources.append('historical_records')
            if 'literary' in response_text.lower() or '紅樓夢' in response_text:
                evidence_sources.append('literary_works')
            if 'domain' in response_text.lower():
                evidence_sources.append('domain_expertise')
            if not evidence_sources:
                evidence_sources.append('general_knowledge')
            
            # Determine error type
            error_type = None
            if judgment == "No":
                if 'factual' in response_text.lower():
                    error_type = "factual_error"
                elif 'syntax' in response_text.lower():
                    error_type = "syntactic_error"
                else:
                    error_type = "validation_error"
            
            return ExplainableJudgment(
                judgment=judgment,
                confidence=confidence,
                reasoning=reasoning,
                evidence_sources=evidence_sources,
                alternative_suggestions=[],  # Simplified - no alternative extraction
                error_type=error_type,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Error parsing explainable response: {e}")
            
            return ExplainableJudgment(
                judgment="No",
                confidence=0.0,
                reasoning=f"Error parsing response: {str(e)}",
                evidence_sources=[],
                alternative_suggestions=[],
                error_type="parsing_error",
                processing_time=processing_time
            )
    
    def _estimate_confidence(self, judgment: str) -> float:
        """
        Estimate confidence score for binary judgment.
        Simplified heuristic when detailed confidence is not available.
        
        Args:
            judgment: Binary judgment ("Yes" or "No")
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic: assume reasonable confidence for clear responses
        if judgment == "Yes":
            return 0.8  # High confidence for positive judgments
        elif judgment == "No":
            return 0.7  # Slightly lower confidence for negative judgments
        else:
            return 0.5  # Low confidence for ambiguous responses


# Convenience functions for direct usage

def judge_triples(triples: List[Triple]) -> JudgmentResult:
    """
    Convenience function for judging triples.
    Matches the interface specified in spec.md Section 8.
    
    Args:
        triples: List of Triple objects to validate
        
    Returns:
        JudgmentResult with binary judgments and confidence scores
    """
    judge = GraphJudge()
    return judge.judge_triples(triples)


def judge_triples_with_explanations(
    triples: List[Triple],
    include_reasoning: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for judging triples with explanations.
    Provides explainable reasoning mode as specified in spec.md FR-GJ5.
    
    Args:
        triples: List of Triple objects to validate
        include_reasoning: Whether to include detailed reasoning
        
    Returns:
        Dictionary with judgments and detailed explanations
    """
    judge = GraphJudge()
    return judge.judge_triples_with_explanations(triples, include_reasoning)