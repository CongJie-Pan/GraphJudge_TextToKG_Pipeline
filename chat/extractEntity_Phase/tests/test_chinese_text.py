"""
Tests for Chinese text processing utilities.
"""

import pytest
from extractEntity_Phase.utils.chinese_text import (
    ChineseCharacterType, CharacterInfo, ChineseTextProcessor,
    is_chinese_character, is_chinese_text, get_chinese_character_count,
    get_chinese_character_ratio, extract_chinese_characters,
    get_unique_chinese_characters, analyze_character,
    normalize_to_traditional, normalize_to_simplified,
    segment_text, clean_text, get_text_statistics,
    validate_chinese_text
)


class TestChineseCharacterType:
    """Test Chinese character type enumeration."""
    
    def test_character_type_values(self):
        """Test character type enum values."""
        assert ChineseCharacterType.TRADITIONAL.value == "traditional"
        assert ChineseCharacterType.SIMPLIFIED.value == "simplified"
        assert ChineseCharacterType.COMMON.value == "common"
        assert ChineseCharacterType.RARE.value == "rare"
        assert ChineseCharacterType.UNKNOWN.value == "unknown"


class TestCharacterInfo:
    """Test character info dataclass."""
    
    def test_character_info_creation(self):
        """Test character info creation."""
        char_info = CharacterInfo(
            character="賈",
            type=ChineseCharacterType.TRADITIONAL,
            unicode="8B8A",
            frequency=0.85
        )
        assert char_info.character == "賈"
        assert char_info.type == ChineseCharacterType.TRADITIONAL
        assert char_info.unicode == "8B8A"
        assert char_info.frequency == 0.85
    
    def test_character_info_defaults(self):
        """Test character info default values."""
        char_info = CharacterInfo(character="賈")
        assert char_info.type == ChineseCharacterType.UNKNOWN
        assert char_info.unicode == ""
        assert char_info.frequency == 0.0


class TestChineseTextProcessor:
    """Test Chinese text processor class."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = ChineseTextProcessor()
        assert processor is not None
        assert hasattr(processor, '_chinese_pattern')
        assert hasattr(processor, 'TRADITIONAL_CHARACTERS')
    
    def test_is_chinese_character(self):
        """Test Chinese character detection."""
        processor = ChineseTextProcessor()
        
        # Test Chinese characters
        assert processor.is_chinese_character("賈") == True
        assert processor.is_chinese_character("寶") == True
        assert processor.is_chinese_character("玉") == True
        
        # Test non-Chinese characters
        assert processor.is_chinese_character("A") == False
        assert processor.is_chinese_character("1") == False
        assert processor.is_chinese_character("!") == False
        assert processor.is_chinese_character(" ") == False
    
    def test_is_chinese_text(self):
        """Test Chinese text detection."""
        processor = ChineseTextProcessor()
        
        # Test Chinese text
        assert processor.is_chinese_text("賈寶玉") == True
        assert processor.is_chinese_text("紅樓夢") == True
        assert processor.is_chinese_text("大觀園") == True
        
        # Test mixed text
        assert processor.is_chinese_text("賈寶玉123") == False
        assert processor.is_chinese_text("賈寶玉 ABC") == False
        
        # Test non-Chinese text
        assert processor.is_chinese_text("Hello World") == False
        assert processor.is_chinese_text("123456") == False
    
    def test_get_chinese_character_count(self):
        """Test Chinese character counting."""
        processor = ChineseTextProcessor()
        
        # Test pure Chinese text
        assert processor.get_chinese_character_count("賈寶玉") == 3
        assert processor.get_chinese_character_count("紅樓夢") == 3
        
        # Test mixed text
        assert processor.get_chinese_character_count("賈寶玉123") == 3
        assert processor.get_chinese_character_count("賈寶玉 ABC") == 3
        
        # Test empty text
        assert processor.get_chinese_character_count("") == 0
        assert processor.get_chinese_character_count("   ") == 0
    
    def test_get_chinese_character_ratio(self):
        """Test Chinese character ratio calculation."""
        processor = ChineseTextProcessor()
        
        # Test pure Chinese text
        assert processor.get_chinese_character_ratio("賈寶玉") == 1.0
        assert processor.get_chinese_character_ratio("紅樓夢") == 1.0
        
        # Test mixed text
        assert processor.get_chinese_character_ratio("賈寶玉123") == 0.5  # 3/6
        assert processor.get_chinese_character_ratio("賈寶玉 ABC") == 0.5  # 3/6
        
        # Test non-Chinese text
        assert processor.get_chinese_character_ratio("Hello World") == 0.0
        assert processor.get_chinese_character_ratio("123456") == 0.0
        
        # Test empty text
        assert processor.get_chinese_character_ratio("") == 0.0
    
    def test_extract_chinese_characters(self):
        """Test Chinese character extraction."""
        processor = ChineseTextProcessor()
        
        # Test pure Chinese text
        chinese_chars = processor.extract_chinese_characters("賈寶玉")
        assert chinese_chars == ["賈", "寶", "玉"]
        
        # Test mixed text
        chinese_chars = processor.extract_chinese_characters("賈寶玉123ABC")
        assert chinese_chars == ["賈", "寶", "玉"]
        
        # Test empty text
        chinese_chars = processor.extract_chinese_characters("")
        assert chinese_chars == []
    
    def test_get_unique_chinese_characters(self):
        """Test unique Chinese character extraction."""
        processor = ChineseTextProcessor()
        
        # Test text with unique characters
        unique_chars = processor.get_unique_chinese_characters("賈寶玉")
        assert len(unique_chars) == 3
        assert "賈" in unique_chars
        assert "寶" in unique_chars
        assert "玉" in unique_chars
        
        # Test text with duplicate characters
        unique_chars = processor.get_unique_chinese_characters("賈寶玉賈寶玉")
        assert len(unique_chars) == 3  # Should still be 3 unique characters
        
        # Test mixed text
        unique_chars = processor.get_unique_chinese_characters("賈寶玉123ABC")
        assert len(unique_chars) == 3
        assert all(char in ["賈", "寶", "玉"] for char in unique_chars)
    
    def test_analyze_character(self):
        """Test character analysis."""
        processor = ChineseTextProcessor()
        
        # Test Chinese character
        char_info = processor.analyze_character("賈")
        assert isinstance(char_info, CharacterInfo)
        assert char_info.character == "賈"
        assert char_info.type in [ChineseCharacterType.TRADITIONAL, ChineseCharacterType.SIMPLIFIED, ChineseCharacterType.COMMON]
        
        # Test non-Chinese character
        char_info = processor.analyze_character("A")
        assert isinstance(char_info, CharacterInfo)
        assert char_info.character == "A"
        assert char_info.type == ChineseCharacterType.UNKNOWN
    
    def test_normalize_to_traditional(self):
        """Test text normalization to traditional Chinese."""
        processor = ChineseTextProcessor()
        
        # Test simplified to traditional conversion
        traditional = processor.normalize_to_traditional("红楼梦")
        assert "紅" in traditional  # Should convert simplified to traditional
        
        # Test already traditional text
        traditional = processor.normalize_to_traditional("紅樓夢")
        assert traditional == "紅樓夢"  # Should remain unchanged
        
        # Test mixed text
        traditional = processor.normalize_to_traditional("红楼梦ABC123")
        assert "紅" in traditional  # Should convert Chinese characters
        assert "ABC123" in traditional  # Should preserve non-Chinese characters
    
    def test_normalize_to_simplified(self):
        """Test text normalization to simplified Chinese."""
        processor = ChineseTextProcessor()
        
        # Test traditional to simplified conversion
        simplified = processor.normalize_to_simplified("紅樓夢")
        assert "红" in simplified  # Should convert traditional to simplified
        
        # Test already simplified text
        simplified = processor.normalize_to_simplified("红楼梦")
        assert simplified == "红楼梦"  # Should remain unchanged
        
        # Test mixed text
        simplified = processor.normalize_to_simplified("紅樓夢ABC123")
        assert "红" in simplified  # Should convert Chinese characters
        assert "ABC123" in simplified  # Should preserve non-Chinese characters
    
    def test_segment_text(self):
        """Test text segmentation."""
        processor = ChineseTextProcessor()
        
        # Test Chinese text segmentation
        segments = processor.segment_text("賈寶玉在大觀園中讀書")
        assert len(segments) > 0
        assert all(isinstance(segment, str) for segment in segments)
        
        # Test mixed text segmentation
        segments = processor.segment_text("賈寶玉在ABC大觀園中讀書123")
        assert len(segments) > 0
        
        # Test empty text
        segments = processor.segment_text("")
        assert segments == []
    
    def test_clean_text(self):
        """Test text cleaning."""
        processor = ChineseTextProcessor()
        
        # Test text with extra whitespace
        cleaned = processor.clean_text("  賈寶玉  大觀園  ")
        assert cleaned == "賈寶玉 大觀園"  # Should normalize whitespace
        
        # Test text with special characters
        cleaned = processor.clean_text("賈寶玉！@#￥%……&*（）大觀園")
        assert "賈寶玉" in cleaned
        assert "大觀園" in cleaned
        
        # Test empty text
        cleaned = processor.clean_text("")
        assert cleaned == ""
    
    def test_get_text_statistics(self):
        """Test text statistics generation."""
        processor = ChineseTextProcessor()
        
        # Test Chinese text statistics
        stats = processor.get_text_statistics("賈寶玉在大觀園中讀書")
        assert "total_characters" in stats
        assert "chinese_characters" in stats
        assert "non_chinese_characters" in stats
        assert "chinese_ratio" in stats
        assert "unique_chinese_characters" in stats
        
        # Test mixed text statistics
        stats = processor.get_text_statistics("賈寶玉在ABC大觀園中讀書123")
        assert stats["total_characters"] > 0
        assert stats["chinese_characters"] > 0
        assert stats["non_chinese_characters"] > 0
        assert 0.0 <= stats["chinese_ratio"] <= 1.0
    
    def test_validate_chinese_text(self):
        """Test Chinese text validation."""
        processor = ChineseTextProcessor()
        
        # Test valid Chinese text
        is_valid, issues = processor.validate_chinese_text("賈寶玉")
        assert is_valid == True
        assert len(issues) == 0
        
        # Test text with potential issues
        is_valid, issues = processor.validate_chinese_text("賈寶玉123")
        assert is_valid == False
        assert len(issues) > 0
        
        # Test empty text
        is_valid, issues = processor.validate_chinese_text("")
        assert is_valid == False
        assert len(issues) > 0


class TestUtilityFunctions:
    """Test top-level utility functions."""
    
    def test_is_chinese_character_function(self):
        """Test top-level is_chinese_character function."""
        assert is_chinese_character("賈") == True
        assert is_chinese_character("A") == False
        assert is_chinese_character("1") == False
    
    def test_is_chinese_text_function(self):
        """Test top-level is_chinese_text function."""
        assert is_chinese_text("賈寶玉") == True
        assert is_chinese_text("Hello World") == False
        assert is_chinese_text("賈寶玉123") == False
    
    def test_get_chinese_character_count_function(self):
        """Test top-level get_chinese_character_count function."""
        assert get_chinese_character_count("賈寶玉") == 3
        assert get_chinese_character_count("賈寶玉123") == 3
        assert get_chinese_character_count("") == 0
    
    def test_get_chinese_character_ratio_function(self):
        """Test top-level get_chinese_character_ratio function."""
        assert get_chinese_character_ratio("賈寶玉") == 1.0
        assert get_chinese_character_ratio("賈寶玉123") == 0.5
        assert get_chinese_character_ratio("Hello World") == 0.0
    
    def test_extract_chinese_characters_function(self):
        """Test top-level extract_chinese_characters function."""
        chars = extract_chinese_characters("賈寶玉123")
        assert chars == ["賈", "寶", "玉"]
    
    def test_get_unique_chinese_characters_function(self):
        """Test top-level get_unique_chinese_characters function."""
        unique_chars = get_unique_chinese_characters("賈寶玉賈寶玉")
        assert len(unique_chars) == 3
        assert all(char in ["賈", "寶", "玉"] for char in unique_chars)
    
    def test_analyze_character_function(self):
        """Test top-level analyze_character function."""
        char_info = analyze_character("賈")
        assert isinstance(char_info, CharacterInfo)
        assert char_info.character == "賈"
    
    def test_normalize_to_traditional_function(self):
        """Test top-level normalize_to_traditional function."""
        traditional = normalize_to_traditional("红楼梦")
        assert "紅" in traditional
    
    def test_normalize_to_simplified_function(self):
        """Test top-level normalize_to_simplified function."""
        simplified = normalize_to_simplified("紅樓夢")
        assert "红" in simplified
    
    def test_segment_text_function(self):
        """Test top-level segment_text function."""
        segments = segment_text("賈寶玉在大觀園中讀書")
        assert len(segments) > 0
        assert all(isinstance(segment, str) for segment in segments)
    
    def test_clean_text_function(self):
        """Test top-level clean_text function."""
        cleaned = clean_text("  賈寶玉  大觀園  ")
        assert cleaned == "賈寶玉 大觀園"
    
    def test_get_text_statistics_function(self):
        """Test top-level get_text_statistics function."""
        stats = get_text_statistics("賈寶玉在大觀園中讀書")
        assert "total_characters" in stats
        assert "chinese_characters" in stats
    
    def test_validate_chinese_text_function(self):
        """Test top-level validate_chinese_text function."""
        is_valid, issues = validate_chinese_text("賈寶玉")
        assert is_valid == True
        assert len(issues) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_strings(self):
        """Test processing of empty strings."""
        processor = ChineseTextProcessor()
        
        assert processor.is_chinese_text("") == False
        assert processor.get_chinese_character_count("") == 0
        assert processor.get_chinese_character_ratio("") == 0.0
        assert processor.extract_chinese_characters("") == []
        assert processor.get_unique_chinese_characters("") == []
        assert processor.segment_text("") == []
        assert processor.clean_text("") == ""
    
    def test_whitespace_only_strings(self):
        """Test processing of whitespace-only strings."""
        processor = ChineseTextProcessor()
        
        assert processor.is_chinese_text("   ") == False
        assert processor.get_chinese_character_count("   ") == 0
        assert processor.get_chinese_character_ratio("   ") == 0.0
        assert processor.extract_chinese_characters("   ") == []
        assert processor.get_unique_chinese_characters("   ") == []
        assert processor.clean_text("   ") == ""
    
    def test_mixed_content(self):
        """Test processing of mixed content."""
        processor = ChineseTextProcessor()
        
        mixed_text = "賈寶玉123ABC!@#$%^&*()大觀園"
        
        # Should extract only Chinese characters
        chinese_chars = processor.extract_chinese_characters(mixed_text)
        assert all(processor.is_chinese_character(char) for char in chinese_chars)
        
        # Should calculate correct ratio
        ratio = processor.get_chinese_character_ratio(mixed_text)
        assert 0.0 < ratio < 1.0
    
    def test_long_text(self):
        """Test processing of very long text."""
        processor = ChineseTextProcessor()
        
        # Create long text with Chinese characters
        long_text = "賈寶玉" * 1000  # 3000 characters
        
        # Should handle without errors
        count = processor.get_chinese_character_count(long_text)
        assert count == 3000
        
        ratio = processor.get_chinese_character_ratio(long_text)
        assert ratio == 1.0
        
        unique_chars = processor.get_unique_chinese_characters(long_text)
        assert len(unique_chars) == 3  # Only 3 unique characters
    
    def test_special_characters(self):
        """Test processing of special characters."""
        processor = ChineseTextProcessor()
        
        special_text = "賈寶玉！@#￥%……&*（）【】{}「」『』《》〈〉"
        
        # Should extract only Chinese characters
        chinese_chars = processor.extract_chinese_characters(special_text)
        assert len(chinese_chars) == 3  # Only the 3 Chinese characters
        
        # Should clean text appropriately
        cleaned = processor.clean_text(special_text)
        assert "賈寶玉" in cleaned


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_character_type(self):
        """Test handling of invalid character types."""
        processor = ChineseTextProcessor()
        
        # Should handle gracefully
        char_info = processor.analyze_character("")
        assert char_info.type == ChineseCharacterType.UNKNOWN
        
        char_info = processor.analyze_character(" ")
        assert char_info.type == ChineseCharacterType.UNKNOWN
    
    def test_segmentation_edge_cases(self):
        """Test text segmentation edge cases."""
        processor = ChineseTextProcessor()
        
        # Test with very short text
        segments = processor.segment_text("賈")
        assert len(segments) >= 1
        
        # Test with single character
        segments = processor.segment_text("寶")
        assert len(segments) >= 1
    
    def test_validation_edge_cases(self):
        """Test text validation edge cases."""
        processor = ChineseTextProcessor()
        
        # Test with None (should handle gracefully)
        try:
            is_valid, issues = processor.validate_chinese_text(None)
            # If no exception, should return False with issues
            assert is_valid == False
            assert len(issues) > 0
        except (TypeError, AttributeError):
            # Exception is also acceptable
            pass
        
        # Test with very long text
        long_text = "賈" * 10000
        is_valid, issues = processor.validate_chinese_text(long_text)
        assert is_valid == True  # Should be valid if all characters are Chinese
        assert len(issues) == 0

