"""
Unit tests for the prompt_engineering module.

Tests the PromptEngineer class functionality including prompt creation,
response parsing, citation extraction, and explainable judgment parsing.
"""

import pytest
import re
from unittest.mock import MagicMock

from ..prompt_engineering import PromptEngineer
from ..data_structures import ExplainableJudgment, CitationData, CitationSummary
from .conftest import MockPerplexityResponse


class TestPromptCreation:
    """Test prompt creation functionality."""
    
    def test_create_graph_judgment_prompt_chinese(self):
        """Test graph judgment prompt creation with Chinese content."""
        instruction = "Is this true: 曹雪芹 創作 紅樓夢 ?"
        prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
        
        # Verify key elements in the prompt
        expected_elements = [
            "knowledge graph validation expert",
            "曹雪芹 創作 紅樓夢",
            "factually correct",
            "Dream of the Red Chamber"
        ]
        
        for element in expected_elements:
            assert element in prompt, f"Missing element: {element}"
        
        # Check that prompt contains Yes/No instruction
        assert ("Yes" in prompt or "No" in prompt), "Prompt should contain Yes/No instruction"
    
    def test_create_graph_judgment_prompt_english(self):
        """Test graph judgment prompt creation with English content."""
        instruction = "Is this true: Apple Founded by Steve Jobs ?"
        prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
        
        expected_elements = [
            "knowledge graph validation expert",
            "Apple Founded by Steve Jobs",
            "factually correct",
            "reliable information sources"
        ]
        
        for element in expected_elements:
            assert element in prompt, f"Missing element: {element}"
    
    def test_create_graph_judgment_prompt_formatting(self):
        """Test that prompt is properly formatted."""
        instruction = "Is this true: Test Statement ?"
        prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
        
        # Check prompt structure
        assert prompt.startswith("You are a knowledge graph validation expert")
        assert "Test Statement" in prompt
        assert prompt.endswith("Please answer only \"Yes\" or \"No\":")
        
        # Check that instruction format is properly extracted
        assert "Is this true: " not in prompt or prompt.count("Is this true: ") == 1
        assert " ?" not in prompt or "Test Statement ?" not in prompt
    
    def test_create_explainable_judgment_prompt_chinese(self):
        """Test explainable judgment prompt creation with Chinese formatting."""
        instruction = "Is this true: 曹雪芹 創作 紅樓夢 ?"
        prompt = PromptEngineer.create_explainable_judgment_prompt(instruction)
        
        # Verify key elements in the explainable prompt
        expected_elements = [
            "知識圖譜驗證專家",
            "曹雪芹 創作 紅樓夢",
            "判斷結果",
            "置信度",
            "詳細推理",
            "證據來源",
            "錯誤類型",
            "替代建議"
        ]
        
        for element in expected_elements:
            assert element in prompt, f"Missing element in explainable prompt: {element}"
    
    def test_create_explainable_judgment_prompt_structure(self):
        """Test explainable judgment prompt structure."""
        instruction = "Is this true: Test Statement ?"
        prompt = PromptEngineer.create_explainable_judgment_prompt(instruction)
        
        # Check for numbered sections
        assert "1. 判斷結果" in prompt
        assert "2. 置信度" in prompt
        assert "3. 詳細推理" in prompt
        assert "4. 證據來源" in prompt
        assert "5. 錯誤類型" in prompt
        assert "6. 替代建議" in prompt
        
        # Check that triple is properly extracted
        assert "Test Statement" in prompt


class TestResponseParsing:
    """Test response parsing functionality."""
    
    def test_parse_response_yes_variations(self):
        """Test parsing various 'Yes' response formats."""
        test_cases = [
            (MockPerplexityResponse("Yes, this is correct."), "Yes"),
            (MockPerplexityResponse("YES"), "Yes"),
            (MockPerplexityResponse("是的，這是正確的"), "Yes"),
            (MockPerplexityResponse("正確"), "Yes"),
            (MockPerplexityResponse("The statement is true."), "Yes"),
            (MockPerplexityResponse("This is accurate and valid."), "Yes")
        ]
        
        for response, expected in test_cases:
            result = PromptEngineer.parse_response(response)
            assert result == expected, f"Failed for response: {response.answer}"
    
    def test_parse_response_no_variations(self):
        """Test parsing various 'No' response formats."""
        test_cases = [
            (MockPerplexityResponse("No, this is incorrect."), "No"),
            (MockPerplexityResponse("NO"), "No"),
            (MockPerplexityResponse("否，這是錯誤的"), "No"),
            (MockPerplexityResponse("錯誤"), "No"),
            (MockPerplexityResponse("The statement is false."), "No"),
            (MockPerplexityResponse("This is wrong and invalid."), "No")
        ]
        
        for response, expected in test_cases:
            result = PromptEngineer.parse_response(response)
            assert result == expected, f"Failed for response: {response.answer}"
    
    def test_parse_response_ambiguous_cases(self):
        """Test parsing ambiguous responses (should default to No)."""
        test_cases = [
            MockPerplexityResponse("Maybe this is correct"),
            MockPerplexityResponse("It depends on the context"),
            MockPerplexityResponse("Unclear information"),
            MockPerplexityResponse(""),
            MockPerplexityResponse(None)
        ]
        
        for response in test_cases:
            result = PromptEngineer.parse_response(response)
            assert result == "No", f"Ambiguous response should default to No: {response.answer}"
    
    def test_parse_response_none_handling(self):
        """Test parsing None response."""
        result = PromptEngineer.parse_response(None)
        assert result == "No"
    
    def test_parse_response_empty_content(self):
        """Test parsing response with empty content."""
        empty_response = MockPerplexityResponse("")
        result = PromptEngineer.parse_response(empty_response)
        assert result == "No"
    
    def test_parse_response_complex_answer(self):
        """Test parsing complex answers with sentiment analysis."""
        positive_response = MockPerplexityResponse(
            "Based on historical records, this statement appears to be correct and accurate."
        )
        result = PromptEngineer.parse_response(positive_response)
        assert result == "Yes"
        
        negative_response = MockPerplexityResponse(
            "This statement contains incorrect information and is demonstrably false."
        )
        result = PromptEngineer.parse_response(negative_response)
        assert result == "No"


class TestExplainableResponseParsing:
    """Test explainable response parsing functionality."""
    
    def test_parse_explainable_response_structured(self):
        """Test parsing structured explainable responses."""
        structured_answer = """
        1. 判斷結果：No
        
        2. 置信度：0.85
        
        3. 詳細推理：此三元組在語義上不正確，Microsoft是由Bill Gates創立的，不是Mark Zuckerberg。
        
        4. 證據來源：domain_knowledge, tech_history
        
        5. 錯誤類型：factual_error
        
        6. 替代建議：Bill Gates-創立-Microsoft
        """
        
        mock_response = MockPerplexityResponse(answer=structured_answer)
        result = PromptEngineer.parse_explainable_response(mock_response)
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment == "No"
        assert result.confidence == 0.85
        assert "語義上" in result.reasoning
        assert "domain_knowledge" in result.evidence_sources
        assert "tech_history" in result.evidence_sources
        assert result.error_type == "factual_error"
        assert len(result.alternative_suggestions) > 0
        assert result.alternative_suggestions[0]["subject"] == "Bill Gates"
    
    def test_parse_explainable_response_none_handling(self):
        """Test parsing None explainable response."""
        result = PromptEngineer.parse_explainable_response(None)
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment == "No"
        assert result.confidence == 0.0
        assert "empty response" in result.reasoning
        assert result.error_type == "response_error"
    
    def test_parse_explainable_response_fallback(self):
        """Test fallback parsing for malformed explainable responses."""
        malformed_answer = "This is just a simple yes answer without structure."
        mock_response = MockPerplexityResponse(answer=malformed_answer)
        
        result = PromptEngineer.parse_explainable_response(mock_response)
        
        assert isinstance(result, ExplainableJudgment)
        assert result.judgment in ["Yes", "No"]
        assert result.confidence == 0.5  # Default moderate confidence
        assert "無法解析結構化回應" in result.reasoning
        assert result.error_type == "parsing_error" or result.error_type is None


class TestExtractMethods:
    """Test individual extract methods for explainable responses."""
    
    def test_extract_judgment(self):
        """Test judgment extraction from responses."""
        test_cases = [
            ("判斷結果：Yes", "Yes"),
            ("判斷結果: No", "No"),
            ("1. 判斷結果：「Yes」", "Yes"),
            ("Simply yes", "Yes"),
            ("Simply no", "No"),
            ("Unclear answer", "No")  # Fallback
        ]
        
        for answer, expected in test_cases:
            result = PromptEngineer._extract_judgment(answer)
            assert result == expected, f"Failed for answer: {answer}"
    
    def test_extract_confidence(self):
        """Test confidence extraction from responses."""
        test_cases = [
            ("置信度：0.95", 0.95),
            ("confidence: 0.8", 0.8),
            ("確信程度：0.75", 0.75),
            ("這個判斷確定", 0.9),  # High confidence keywords
            ("可能正確", 0.7),    # Medium confidence keywords
            ("不確定的推理", 0.5)   # Default confidence
        ]
        
        for answer, expected_min in test_cases:
            result = PromptEngineer._extract_confidence(answer)
            assert 0.0 <= result <= 1.0
            
            # For exact numeric matches
            if "：" in answer or ":" in answer:
                assert abs(result - expected_min) < 0.01
            # For keyword-based confidence, check ranges
            elif "確定" in answer:
                assert result >= 0.8
            elif "可能" in answer:
                assert 0.6 <= result <= 0.8
    
    def test_extract_reasoning(self):
        """Test reasoning extraction from responses."""
        test_answer = """
        1. 判斷結果：No
        2. 置信度：0.85
        3. 詳細推理：這個陳述是錯誤的。Microsoft是由Bill Gates和Paul Allen創立的，而不是Mark Zuckerberg。
        4. 證據來源：domain_knowledge
        """
        
        reasoning = PromptEngineer._extract_reasoning(test_answer)
        
        assert "這個陳述是錯誤的" in reasoning
        assert "Bill Gates" in reasoning
        assert len(reasoning) > 10
    
    def test_extract_evidence_sources(self):
        """Test evidence sources extraction from responses."""
        test_answer = """
        4. 證據來源：domain_knowledge, historical_records, literary_history
        """
        
        sources = PromptEngineer._extract_evidence_sources(test_answer)
        
        assert "domain_knowledge" in sources
        assert "historical_records" in sources
        assert "literary_history" in sources
        assert len(sources) >= 3
    
    def test_extract_error_type(self):
        """Test error type extraction from responses."""
        test_cases = [
            ("錯誤類型：entity_mismatch", "entity_mismatch"),
            ("error type: factual_error", "factual_error"),
            ("錯誤類型：None", None),
            ("錯誤類型：無", None),
            ("沒有錯誤類型資訊", None)
        ]
        
        for answer, expected in test_cases:
            result = PromptEngineer._extract_error_type(answer)
            assert result == expected, f"Failed for answer: {answer}"
    
    def test_extract_alternatives(self):
        """Test alternative suggestions extraction from responses."""
        test_answer = """
        6. 替代建議：Bill Gates-創立-Microsoft, Paul Allen-共同創立-Microsoft
        """
        
        alternatives = PromptEngineer._extract_alternatives(test_answer)
        
        assert len(alternatives) >= 1
        if len(alternatives) > 0:
            assert alternatives[0]["subject"] == "Bill Gates"
            assert alternatives[0]["relation"] == "創立"
            assert alternatives[0]["object"] == "Microsoft"
            assert "confidence" in alternatives[0]


class TestCitationHandling:
    """Test citation extraction and handling functionality."""
    
    def test_extract_citations_direct(self):
        """Test citation extraction from direct response citations."""
        mock_response = MockPerplexityResponse(
            answer="Yes, this is correct.",
            citations=["https://example.com/source1", "https://example.com/source2"]
        )
        
        citations = PromptEngineer.extract_citations(mock_response)
        
        assert len(citations) == 2
        assert citations[0].number == "1"
        assert citations[0].url == "https://example.com/source1"
        assert citations[0].type == "perplexity_citation"
        assert citations[0].source == "direct"
        
        assert citations[1].number == "2"
        assert citations[1].url == "https://example.com/source2"
    
    def test_extract_citations_from_content(self):
        """Test citation extraction from content references."""
        content_with_citations = "This is correct [1] according to sources [2] and research [3]."
        mock_response = MockPerplexityResponse(answer=content_with_citations)
        mock_response.choices[0].message.content = content_with_citations
        
        citations = PromptEngineer.extract_citations(mock_response)
        
        # Should extract citation numbers from content
        citation_numbers = [c.number for c in citations]
        assert "1" in citation_numbers
        assert "2" in citation_numbers
        assert "3" in citation_numbers
    
    def test_extract_citations_empty(self):
        """Test citation extraction with no citations."""
        mock_response = MockPerplexityResponse(answer="Yes, this is correct.")
        citations = PromptEngineer.extract_citations(mock_response)
        assert len(citations) == 0
    
    def test_extract_title_from_url(self):
        """Test URL title extraction functionality."""
        test_cases = [
            ("https://www.wikipedia.org/red_chamber", "Wikipedia"),
            ("http://example.com/test-page", "Example"),
            ("https://zh.wikipedia.org/紅樓夢", "Zh.wikipedia"),
            ("https://perplexity.ai/search", "Perplexity")
        ]
        
        for url, expected_start in test_cases:
            title = PromptEngineer._extract_title_from_url(url)
            assert isinstance(title, str)
            assert len(title) > 0
            # Title should contain expected domain name
            assert expected_start.lower() in title.lower()
    
    def test_extract_title_from_url_error_handling(self):
        """Test URL title extraction error handling."""
        # Test with malformed URLs
        malformed_urls = ["", "not-a-url", "://malformed", None]
        
        for url in malformed_urls:
            title = PromptEngineer._extract_title_from_url(url)
            assert title == "Unknown Source"
    
    def test_get_citation_summary(self):
        """Test citation summary generation."""
        mock_response = MockPerplexityResponse(
            answer="Yes, this is correct.",
            citations=["https://example.com/source1", "https://example.com/source2"]
        )
        
        summary = PromptEngineer.get_citation_summary(mock_response)
        
        assert isinstance(summary, CitationSummary)
        assert summary.total_citations == 2
        assert summary.has_citations == True
        assert len(summary.citations) == 2
        assert "perplexity_citation" in summary.citation_types
    
    def test_get_citation_summary_empty(self):
        """Test citation summary for empty case."""
        mock_response = MockPerplexityResponse(answer="Yes, this is correct.")
        summary = PromptEngineer.get_citation_summary(mock_response)
        
        assert isinstance(summary, CitationSummary)
        assert summary.total_citations == 0
        assert summary.has_citations == False
        assert len(summary.citations) == 0


class TestUtilityMethods:
    """Test utility methods in PromptEngineer."""
    
    def test_clean_html_tags(self):
        """Test HTML tag cleaning functionality."""
        test_cases = [
            ("<p>Simple text</p>", "Simple text"),
            ("<b>Bold</b> and <i>italic</i> text", "Bold and italic text"),
            ("Text with <a href='#'>link</a>", "Text with link"),
            ("No HTML tags", "No HTML tags"),
            ("", ""),
            (None, "")
        ]
        
        for input_text, expected in test_cases:
            result = PromptEngineer.clean_html_tags(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_clean_html_tags_complex(self):
        """Test HTML tag cleaning with complex content."""
        complex_html = """
        <div class="content">
            <h1>Title</h1>
            <p>This is a <strong>test</strong> paragraph with <em>emphasis</em>.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
        """
        
        result = PromptEngineer.clean_html_tags(complex_html)
        
        # Should remove all HTML tags but preserve text content
        assert "<" not in result
        assert ">" not in result
        assert "Title" in result
        assert "test" in result
        assert "paragraph" in result
        assert "emphasis" in result
        assert "Item 1" in result


class TestPromptEngineeringIntegration:
    """Test integration scenarios for prompt engineering."""
    
    def test_full_cycle_basic_judgment(self):
        """Test full cycle: prompt creation -> response parsing."""
        instruction = "Is this true: 曹雪芹 創作 紅樓夢 ?"
        
        # Create prompt
        prompt = PromptEngineer.create_graph_judgment_prompt(instruction)
        assert "曹雪芹 創作 紅樓夢" in prompt
        
        # Simulate response and parse
        mock_response = MockPerplexityResponse("Yes, this is historically accurate.")
        result = PromptEngineer.parse_response(mock_response)
        assert result == "Yes"
    
    def test_full_cycle_explainable_judgment(self):
        """Test full cycle: explainable prompt creation -> explainable parsing."""
        instruction = "Is this true: Microsoft Founded by Mark Zuckerberg ?"
        
        # Create explainable prompt
        prompt = PromptEngineer.create_explainable_judgment_prompt(instruction)
        assert "Microsoft Founded by Mark Zuckerberg" in prompt
        assert "判斷結果" in prompt
        
        # Simulate structured response and parse
        structured_response = """
        1. 判斷結果：No
        2. 置信度：0.9
        3. 詳細推理：這是錯誤的，Microsoft由Bill Gates創立。
        4. 證據來源：tech_history
        5. 錯誤類型：factual_error
        6. 替代建議：Bill Gates-創立-Microsoft
        """
        
        mock_response = MockPerplexityResponse(structured_response)
        result = PromptEngineer.parse_explainable_response(mock_response)
        
        assert result.judgment == "No"
        assert result.confidence == 0.9
        assert result.error_type == "factual_error"
    
    def test_multilingual_content_handling(self):
        """Test handling of multilingual content in prompts and responses."""
        # Chinese instruction
        chinese_instruction = "Is this true: 賈寶玉 喜歡 林黛玉 ?"
        chinese_prompt = PromptEngineer.create_graph_judgment_prompt(chinese_instruction)
        assert "賈寶玉 喜歡 林黛玉" in chinese_prompt
        
        # English instruction
        english_instruction = "Is this true: Apple Founded by Steve Jobs ?"
        english_prompt = PromptEngineer.create_graph_judgment_prompt(english_instruction)
        assert "Apple Founded by Steve Jobs" in english_prompt
        
        # Mixed response parsing
        mixed_response = MockPerplexityResponse("是的，這個陳述是 correct 的。")
        result = PromptEngineer.parse_response(mixed_response)
        assert result == "Yes"
