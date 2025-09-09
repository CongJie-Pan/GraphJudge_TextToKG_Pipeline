"""
Prompt engineering for the GraphJudge system.

This module handles all prompt creation and response parsing for the
Perplexity API integration, including both standard and explainable
judgment modes.
"""

import re
from typing import List, Optional, Dict, Any
from .data_structures import ExplainableJudgment, CitationData, CitationSummary


class PromptEngineer:
    """
    Handles prompt creation and response parsing for Perplexity API.
    
    This class provides comprehensive prompt engineering capabilities
    for graph judgment tasks, including both binary and explainable
    judgment modes.
    """
    
    @staticmethod
    def create_graph_judgment_prompt(instruction: str) -> str:
        """
        Create a specialized prompt for graph judgment using Perplexity's capabilities.
        
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
    
    @staticmethod
    def create_explainable_judgment_prompt(instruction: str) -> str:
        """
        Create a specialized prompt for explainable graph judgment.
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
    
    @staticmethod
    def parse_response(response) -> str:
        """
        Parse and validate the Perplexity response for graph judgment.
        
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
            # Use more conservative indicators to avoid false positives
            strong_positive_indicators = ['definitely yes', 'certainly true', 'absolutely correct', 'definitely correct']
            strong_negative_indicators = ['definitely no', 'certainly false', 'absolutely wrong', 'definitely incorrect']
            
            # Check for strong indicators first
            answer_lower = answer.lower()
            for indicator in strong_positive_indicators:
                if indicator in answer_lower:
                    return "Yes"
            for indicator in strong_negative_indicators:
                if indicator in answer_lower:
                    return "No"
            
            # For ambiguous responses with weak indicators, default to No (conservative approach)
            # This includes responses like "Maybe this is correct", "It depends", etc.
            ambiguous_indicators = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'depends', 'unclear', 'uncertain']
            if any(indicator in answer_lower for indicator in ambiguous_indicators):
                print(f"Warning: Ambiguous response detected, defaulting to No: {answer[:100]}...")
                return "No"
            
            # Only use sentiment analysis for clear, confident statements
            positive_indicators = ['correct', 'true', 'accurate', 'valid', '正確', '是的', '對的']
            negative_indicators = ['incorrect', 'false', 'wrong', 'invalid', '錯誤', '不對', '否']
            
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
    
    @staticmethod
    def parse_explainable_response(response) -> ExplainableJudgment:
        """
        Parse Perplexity response for explainable graph judgment.
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
        
        # If the response does not look structured, fallback immediately
        structured_markers = [
            "判斷結果", "置信度", "詳細推理", "證據來源", "錯誤類型", "替代建議", "\n1.", "1. ", "1.", "2. "
        ]
        if not any(marker in answer for marker in structured_markers):
            simple_judgment = PromptEngineer.parse_response(response)
            return ExplainableJudgment(
                judgment=simple_judgment,
                confidence=0.5,
                reasoning=f"無法解析結構化回應，回退到簡單判斷。原始回應：{answer[:200]}...",
                evidence_sources=["general_analysis"],
                alternative_suggestions=[],
                error_type="parsing_error" if simple_judgment == "No" else None,
                processing_time=0.0
            )

        try:
            # Parse structured response
            judgment = PromptEngineer._extract_judgment(answer)
            confidence = PromptEngineer._extract_confidence(answer)
            reasoning = PromptEngineer._extract_reasoning(answer)
            evidence_sources = PromptEngineer._extract_evidence_sources(answer)
            error_type = PromptEngineer._extract_error_type(answer)
            alternative_suggestions = PromptEngineer._extract_alternatives(answer)
            
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
            simple_judgment = PromptEngineer.parse_response(response)
            return ExplainableJudgment(
                judgment=simple_judgment,
                confidence=0.5,  # Default moderate confidence
                reasoning=f"無法解析結構化回應，回退到簡單判斷。原始回應：{answer[:200]}...",
                evidence_sources=["general_analysis"],
                alternative_suggestions=[],
                error_type="parsing_error" if simple_judgment == "No" else None,
                processing_time=0.0
            )
    
    @staticmethod
    def _extract_judgment(answer: str) -> str:
        """Extract binary judgment from structured response."""
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
            return PromptEngineer.parse_response(type('MockResponse', (), {'answer': answer})())
    
    @staticmethod
    def _extract_confidence(answer: str) -> float:
        """Extract confidence score from response."""
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
    
    @staticmethod
    def _extract_reasoning(answer: str) -> str:
        """Extract detailed reasoning from response."""
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
    
    @staticmethod
    def _extract_evidence_sources(answer: str) -> List[str]:
        """Extract evidence sources from response."""
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
    
    @staticmethod
    def _extract_error_type(answer: str) -> Optional[str]:
        """Extract error type from response."""
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
    
    @staticmethod
    def _extract_alternatives(answer: str) -> List[Dict]:
        """Extract alternative suggestions from response."""
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
    
    @staticmethod
    def extract_citations(response) -> List[CitationData]:
        """
        Extract citations from Perplexity response.
        
        Args:
            response: Response from Perplexity API
            
        Returns:
            List[CitationData]: List of citation data
        """
        citations = []
        
        try:
            # Primary method: Check for citations directly in response
            if hasattr(response, 'citations') and response.citations:
                for i, citation in enumerate(response.citations):
                    if isinstance(citation, str):  # URL string
                        citations.append(CitationData(
                            number=str(i + 1),
                            title=PromptEngineer._extract_title_from_url(citation),
                            url=citation,
                            type="perplexity_citation",
                            source="direct"
                        ))
                    elif isinstance(citation, dict):  # Citation object
                        citations.append(CitationData(
                            number=str(i + 1),
                            title=citation.get('title', PromptEngineer._extract_title_from_url(citation.get('url', ''))),
                            url=citation.get('url', ''),
                            type="perplexity_citation",
                            source="object"
                        ))
            
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
                        existing = any(c.number == num for c in citations)
                        if not existing:
                            # Try to find corresponding URL from response citations
                            url = ""
                            if hasattr(response, 'citations') and response.citations:
                                idx = int(num) - 1
                                if 0 <= idx < len(response.citations):
                                    citation_item = response.citations[idx]
                                    url = citation_item if isinstance(citation_item, str) else citation_item.get('url', '')
                            
                            citations.append(CitationData(
                                number=num,
                                title=PromptEngineer._extract_title_from_url(url) if url else f"Reference {num}",
                                url=url,
                                type="perplexity_citation",
                                source="content_reference"
                            ))
        
        except Exception as e:
            print(f"Error extracting citations: {e}")
        
        # Sort citations by number for consistent ordering
        try:
            citations.sort(key=lambda x: int(x.number))
        except (ValueError, KeyError):
            pass  # Keep original order if sorting fails
        
        return citations
    
    @staticmethod
    def _extract_title_from_url(url: str) -> str:
        """
        Extract a readable title from a URL.
        
        Args:
            url (str): URL to extract title from
            
        Returns:
            str: Extracted title or simplified URL
        """
        # Handle None, empty, or invalid URLs
        if not url or not isinstance(url, str):
            return "Unknown Source"
        
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
            
            # If still empty or clearly not a domain-like token, return Unknown
            if not url or '.' not in url:
                return "Unknown Source"
            
            # Clean up and return
            title = url.replace('-', ' ').replace('_', ' ').title()
            return title[:50] + "..." if len(title) > 50 else title
            
        except Exception:
            return "Unknown Source"
    
    @staticmethod
    def get_citation_summary(response) -> CitationSummary:
        """
        Get a summary of citations from a Perplexity response.
        
        Args:
            response: Response from Perplexity API
            
        Returns:
            CitationSummary: Summary of citations with metadata
        """
        citations = PromptEngineer.extract_citations(response)
        
        return CitationSummary(
            total_citations=len(citations),
            citations=citations,
            has_citations=len(citations) > 0,
            citation_types=list(set(c.type for c in citations))
        )
    
    @staticmethod
    def clean_html_tags(text: str) -> str:
        """
        Clean HTML tags from response text.
        
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
