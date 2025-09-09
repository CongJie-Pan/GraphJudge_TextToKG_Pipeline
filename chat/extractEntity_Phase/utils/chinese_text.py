"""
Chinese Text Processing Utilities Module

This module provides specialized utilities for processing traditional Chinese text,
including character analysis, text normalization, and Chinese-specific text operations.

The module is optimized for classical Chinese literature processing and provides
efficient methods for handling traditional Chinese characters and text structures.
"""

import re
import unicodedata
from typing import List, Dict, Set, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from extractEntity_Phase.models.entities import EntityType


class ChineseCharacterType(str, Enum):
    """Enumeration of Chinese character types."""
    
    TRADITIONAL = "traditional"    # Traditional Chinese characters
    SIMPLIFIED = "simplified"      # Simplified Chinese characters
    COMMON = "common"              # Common characters
    RARE = "rare"                  # Rare characters
    UNKNOWN = "unknown"            # Unknown character type


@dataclass
class CharacterInfo:
    """Information about a Chinese character."""
    
    character: str
    type: ChineseCharacterType = ChineseCharacterType.UNKNOWN
    unicode: str = ""
    frequency: float = 0.0
    radical: Optional[str] = None
    stroke_count: Optional[int] = None
    pinyin: Optional[str] = None
    meaning: Optional[str] = None


class ChineseTextProcessor:
    """
    Chinese text processing utility class.
    
    This class provides comprehensive utilities for analyzing and processing
    traditional Chinese text, including character analysis, text normalization,
    and Chinese-specific text operations.
    """
    
    # Unicode ranges for Chinese characters
    CHINESE_UNICODE_RANGES = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
        (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
    ]
    
    # Common traditional Chinese characters that differ from simplified
    TRADITIONAL_CHARACTERS = {
        '国': '國', '书': '書', '车': '車', '马': '馬', '龙': '龍',
        '鸟': '鳥', '鱼': '魚', '贝': '貝', '见': '見', '长': '長',
        '风': '風', '飞': '飛', '门': '門', '问': '問', '间': '間',
        '关': '關', '开': '開', '进': '進', '远': '遠', '还': '還',
        '这': '這', '那': '那', '时': '時', '东': '東', '西': '西',
        '南': '南', '北': '北', '中': '中', '华': '華', '学': '學',
        '习': '習', '读': '讀', '写': '寫', '说': '說', '话': '話',
        '语': '語', '词': '詞', '句': '句', '文': '文', '字': '字',
        '音': '音', '义': '義', '理': '理', '道': '道', '德': '德',
        '仁': '仁', '礼': '禮', '智': '智', '信': '信', '忠': '忠',
        '孝': '孝', '节': '節', '气': '氣', '神': '神', '仙': '仙',
        '佛': '佛', '禅': '禪', '法': '法', '宝': '寶', '玉': '玉',
        '金': '金', '银': '銀', '铜': '銅', '铁': '鐵', '石': '石',
        '木': '木', '水': '水', '火': '火', '土': '土', '山': '山',
        '川': '川', '江': '江', '河': '河', '海': '海', '天': '天',
        '地': '地', '日': '日', '月': '月', '星': '星', '云': '雲',
        '雨': '雨', '雪': '雪', '雷': '雷', '电': '電', '光': '光',
        '影': '影', '色': '色', '香': '香', '味': '味', '声': '聲',
        '乐': '樂', '舞': '舞', '歌': '歌', '诗': '詩', '赋': '賦',
        '章': '章', '画': '畫', '印': '印', '刻': '刻', '雕': '雕',
        '塑': '塑', '建': '建', '筑': '築', '宫': '宮', '殿': '殿',
        '楼': '樓', '台': '臺', '亭': '亭', '阁': '閣', '轩': '軒',
        '斋': '齋', '堂': '堂', '室': '室', '房': '房', '屋': '屋',
        '舍': '舍', '院': '院', '园': '園', '林': '林', '树': '樹',
        '花': '花', '草': '草', '叶': '葉', '根': '根', '枝': '枝',
        '果': '果', '实': '實', '种': '種', '植': '植', '栽': '栽',
        '养': '養', '护': '護', '保': '保', '卫': '衛', '守': '守',
        '防': '防', '攻': '攻', '战': '戰', '争': '爭', '斗': '鬥',
        '打': '打', '杀': '殺', '死': '死', '生': '生', '活': '活',
        '存': '存', '亡': '亡', '兴': '興', '衰': '衰', '盛': '盛',
        '败': '敗', '成': '成', '功': '功', '名': '名', '利': '利',
        '财': '財', '富': '富', '贵': '貴', '贫': '貧', '贱': '賤',
        '穷': '窮', '达': '達', '通': '通', '顺': '順', '逆': '逆',
        '正': '正', '反': '反', '对': '對', '错': '錯', '是': '是',
        '非': '非', '好': '好', '坏': '壞', '善': '善', '恶': '惡',
        '美': '美', '丑': '醜', '真': '真', '假': '假', '虚': '虛',
        '空': '空', '满': '滿', '多': '多', '少': '少', '大': '大',
        '小': '小', '高': '高', '低': '低', '宽': '寬', '窄': '窄',
        '厚': '厚', '薄': '薄', '深': '深', '浅': '淺', '快': '快',
        '慢': '慢', '早': '早', '晚': '晚', '新': '新', '旧': '舊',
        '古': '古', '今': '今', '前': '前', '后': '後', '左': '左',
        '右': '右', '上': '上', '下': '下', '内': '內', '外': '外',
        '旁': '旁', '边': '邊', '角': '角', '端': '端', '头': '頭',
        '尾': '尾', '首': '首', '末': '末', '始': '始', '终': '終',
        '结': '結', '束': '束', '闭': '閉', '合': '合', '分': '分',
        '离': '離', '聚': '聚', '散': '散', '集': '集', '收': '收',
        '放': '放', '取': '取', '舍': '捨', '得': '得', '失': '失',
        '有': '有', '无': '無', '来': '來', '去': '去', '往': '往',
        '返': '返', '回': '回', '归': '歸', '出': '出', '入': '入',
        '进': '進', '退': '退', '升': '升', '降': '降', '起': '起',
        '落': '落', '翔': '翔', '游': '遊', '泳': '泳', '走': '走',
        '跑': '跑', '跳': '跳', '跃': '躍', '爬': '爬', '行': '行',
        '立': '立', '坐': '坐', '卧': '臥', '睡': '睡', '醒': '醒',
        '想': '想', '思': '思', '念': '念', '记': '記', '忆': '憶',
        '忘': '忘', '知': '知', '识': '識', '懂': '懂', '明': '明',
        '白': '白', '清': '清', '楚': '楚', '确': '確', '定': '定',
        '决': '決', '断': '斷', '判': '判', '别': '別', '区': '區',
        '类': '類', '属': '屬', '纳': '納', '包': '包', '含': '含',
        '括': '括', '总': '總', '计': '計', '算': '算', '数': '數',
        '量': '量', '度': '度', '衡': '衡', '测': '測', '称': '稱',
        '重': '重', '轻': '輕', '红': '紅', '梦': '夢'
    }
    
    def __init__(self):
        """Initialize the Chinese text processor."""
        self._chinese_pattern = self._compile_chinese_pattern()
    
    def _compile_chinese_pattern(self) -> re.Pattern:
        """Compile regex pattern for Chinese characters."""
        # Use proper Unicode ranges for Chinese characters
        pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]'
        return re.compile(pattern, re.UNICODE)
    
    def is_chinese_character(self, char: str) -> bool:
        """
        Check if a character is a Chinese character.
        
        Args:
            char: Single character to check
            
        Returns:
            True if character is Chinese, False otherwise
        """
        if len(char) != 1:
            return False
        
        return bool(self._chinese_pattern.match(char))
    
    def is_chinese_text(self, text: str) -> bool:
        """
        Check if text contains Chinese characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Chinese characters, False otherwise
        """
        if not text:
            return False
        # Check if ALL characters are Chinese (no mixed text allowed)
        chinese_count = self.get_chinese_character_count(text)
        return chinese_count == len(text)
    
    def get_chinese_character_count(self, text: str) -> int:
        """
        Count the number of Chinese characters in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Number of Chinese characters
        """
        return len(self._chinese_pattern.findall(text))
    
    def get_chinese_character_ratio(self, text: str) -> float:
        """
        Calculate the ratio of Chinese characters in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Ratio of Chinese characters (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        chinese_count = 0
        total_count = 0
        
        # Count characters by iterating through the string, ignoring spaces
        for char in text:
            if not char.isspace():  # Skip whitespace characters
                total_count += 1
                if self.is_chinese_character(char):
                    chinese_count += 1
        
        return chinese_count / total_count if total_count > 0 else 0.0
    
    def extract_chinese_characters(self, text: str) -> List[str]:
        """
        Extract all Chinese characters from text.
        
        Args:
            text: Text to extract characters from
            
        Returns:
            List of Chinese characters
        """
        return self._chinese_pattern.findall(text)
    
    def get_unique_chinese_characters(self, text: str) -> List[str]:
        """
        Get unique Chinese characters from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of unique Chinese characters
        """
        return list(set(self.extract_chinese_characters(text)))
    
    def analyze_character(self, char: str) -> CharacterInfo:
        """
        Analyze a Chinese character and provide detailed information.
        
        Args:
            char: Single character to analyze
            
        Returns:
            CharacterInfo object with character details
        """
        if not char or len(char) == 0:
            return CharacterInfo(
                character=char,
                type=ChineseCharacterType.UNKNOWN,
                unicode="U+0000"
        )
        
        # Determine character type (simplified vs traditional)
        char_type = self._determine_character_type(char)
        
        return CharacterInfo(
            character=char,
            type=char_type,
            unicode=f"U+{ord(char):04X}"
        )
    
    def _determine_character_type(self, char: str) -> ChineseCharacterType:
        # Check if character is in traditional character mapping
        if char in self.TRADITIONAL_CHARACTERS.values():
            return ChineseCharacterType.TRADITIONAL
        elif char in self.TRADITIONAL_CHARACTERS:
            return ChineseCharacterType.SIMPLIFIED
        
        # Check if character is actually Chinese
        if self.is_chinese_character(char):
            return ChineseCharacterType.COMMON
        else:
            return ChineseCharacterType.UNKNOWN
    
    def normalize_to_traditional(self, text: str) -> str:
        """
        Convert text to traditional Chinese characters.
        
        Args:
            text: Text to convert
            
        Returns:
            Text with traditional characters
        """
        return self._convert_to_traditional(text)
    
    def normalize_to_simplified(self, text: str) -> str:
        """
        Convert text to simplified Chinese characters.
        
        Args:
            text: Text to convert
            
        Returns:
            Text with simplified characters
        """
        return self._convert_to_simplified(text)
    
    def _convert_to_traditional(self, text: str) -> str:
        """
        Convert simplified characters to traditional.
        
        Args:
            text: Text to convert
            
        Returns:
            Text with traditional characters
        """
        result = text
        for simplified, traditional in self.TRADITIONAL_CHARACTERS.items():
            result = result.replace(simplified, traditional)
        return result
    
    def _convert_to_simplified(self, text: str) -> str:
        """
        Convert traditional characters to simplified.
        
        Args:
            text: Text to convert
            
        Returns:
            Text with simplified characters
        """
        result = text
        for simplified, traditional in self.TRADITIONAL_CHARACTERS.items():
            result = result.replace(traditional, simplified)
        return result
    
    def segment_text(self, text: str, max_length: int = 100) -> List[str]:
        """
        Segment Chinese text into manageable chunks.
        
        Args:
            text: Text to segment
            max_length: Maximum length of each segment
            
        Returns:
            List of text segments
        """
        if not text:
            return []
        if len(text) <= max_length:
            return [text]
        segments = []
        current_segment = ""
        
        # Simple character-based segmentation for Chinese text
        for char in text:
            if len(current_segment) >= max_length:
                segments.append(current_segment)
                current_segment = char
            else:
                current_segment += char
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Chinese text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Chinese characters
        cleaned = ""
        for char in text:
            if self.is_chinese_character(char) or char.isalnum() or char.isspace():
                cleaned += char
        
        return cleaned.strip()
    
    def get_text_statistics(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
                    return {
            "total_characters": 0,
            "chinese_characters": 0,
            "non_chinese_characters": 0,
            "chinese_ratio": 0.0,
            "unique_chinese_characters": 0
        }
        
        # Basic statistics
        total_length = len(text)
        chinese_count = self.get_chinese_character_count(text)
        chinese_ratio = self.get_chinese_character_ratio(text)
        unique_chinese = len(self.get_unique_chinese_characters(text))
        non_chinese_count = total_length - chinese_count
        
        return {
            "total_characters": total_length,
            "chinese_characters": chinese_count,
            "non_chinese_characters": non_chinese_count,
            "chinese_ratio": round(chinese_ratio, 3),
            "unique_chinese_characters": unique_chinese
        }
    
    def classify_entity_type(self, entity_text: str, source_text: str) -> 'EntityType':
        """
        Classify entity type based on text patterns.
        
        Args:
            entity_text: Entity text to classify
            source_text: Source text for context
            
        Returns:
            EntityType classification
        """
        # Import here to avoid circular imports
        from extractEntity_Phase.models.entities import EntityType
        
        # Person indicators
        person_patterns = ['姓', '名', '字', '號', '公', '先生', '夫人', '小姐', '王', '李', '張', '劉', '陳', '楊', '黃', '趙', '周', '吳', '徐', '孫']
        if any(pattern in entity_text for pattern in person_patterns) or any(pattern in source_text for pattern in ['姓', '名字', '稱呼']):
            return EntityType.PERSON
            
        # Location indicators  
        location_patterns = ['府', '宮', '園', '樓', '亭', '閣', '山', '河', '江', '湖', '城', '鎮', '村', '街', '巷', '門', '橋']
        if any(pattern in entity_text for pattern in location_patterns):
            return EntityType.LOCATION
            
        # Organization indicators
        org_patterns = ['府', '朝', '廷', '宗', '族', '家', '門', '派', '會', '社', '團']
        if any(pattern in entity_text for pattern in org_patterns):
            return EntityType.ORGANIZATION
            
        # Object indicators
        object_patterns = ['寶', '玉', '劍', '刀', '琴', '書', '畫', '印', '鏡', '珠', '金', '銀']
        if any(pattern in entity_text for pattern in object_patterns):
            return EntityType.OBJECT
            
        # Time indicators
        time_patterns = ['春', '夏', '秋', '冬', '朝', '夕', '晨', '夜', '年', '月', '日', '時']
        if any(pattern in entity_text for pattern in time_patterns):
            return EntityType.TIME
            
        # Default to OTHER for unclassified entities
        return EntityType.OTHER

    def validate_chinese_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate Chinese text for common issues.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not text:
            issues.append("Text is empty")
            return False, issues
        
        # Check for non-Chinese characters in Chinese text
        chinese_ratio = self.get_chinese_character_ratio(text)
        if chinese_ratio < 1.0:  # Only pure Chinese text is valid
            issues.append(f"Low Chinese character ratio: {chinese_ratio:.1%}")
        
        # Check for excessive whitespace
        if re.search(r'\s{3,}', text):
            issues.append("Excessive whitespace detected")
        
        # Check for control characters
        control_chars = [char for char in text if unicodedata.category(char)[0] == 'C']
        if control_chars:
            issues.append(f"Control characters detected: {len(control_chars)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


# Utility functions for common operations
def is_chinese_character(char: str) -> bool:
    """
    Check if a character is a Chinese character.
    
    Args:
        char: Single character to check
        
    Returns:
        True if character is Chinese, False otherwise
    """
    processor = ChineseTextProcessor()
    return processor.is_chinese_character(char)


def is_chinese_text(text: str) -> bool:
    """
    Check if text contains Chinese characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains Chinese characters
    """
    processor = ChineseTextProcessor()
    return processor.is_chinese_text(text)


def get_chinese_character_count(text: str) -> int:
    """
    Count Chinese characters in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Number of Chinese characters
    """
    processor = ChineseTextProcessor()
    return processor.get_chinese_character_count(text)

def clean_chinese_text(text: str) -> str:
    """
    Clean and normalize Chinese text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    processor = ChineseTextProcessor()
    return processor.clean_text(text)


def get_chinese_character_ratio(text: str) -> float:
    """
    Calculate the ratio of Chinese characters in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Ratio of Chinese characters (0.0 to 1.0)
    """
    processor = ChineseTextProcessor()
    return processor.get_chinese_character_ratio(text)


def extract_chinese_characters(text: str) -> List[str]:
    """
    Extract all Chinese characters from text.
    
    Args:
        text: Text to extract characters from
        
    Returns:
        List of Chinese characters
    """
    processor = ChineseTextProcessor()
    return processor.extract_chinese_characters(text)


def get_unique_chinese_characters(text: str) -> List[str]:
    """
    Get unique Chinese characters from text.
    
    Args:
        text: Text to analyze

    Returns:
        List of unique Chinese characters
    """
    processor = ChineseTextProcessor()
    return processor.get_unique_chinese_characters(text)


def analyze_character(char: str) -> CharacterInfo:
    """
    Analyze a Chinese character and provide detailed information.
    
    Args:
        char: Single character to analyze
        
    Returns:
        CharacterInfo object with character details
    """
    processor = ChineseTextProcessor()
    return processor.analyze_character(char)

def normalize_to_traditional(text: str) -> str:
    """
    Convert text to traditional Chinese characters.
    
    Args:
        text: Text to convert
        
    Returns:
        Text with traditional characters
    """
    processor = ChineseTextProcessor()
    return processor.normalize_to_traditional(text)

def normalize_to_simplified(text: str) -> str:
    """
    Convert text to simplified Chinese characters.
    
    Args:
        text: Text to convert
        
    Returns:
        Text with simplified characters
    """
    processor = ChineseTextProcessor()
    return processor.normalize_to_simplified(text)


def segment_text(text: str, max_length: int = 100) -> List[str]:
    """
    Segment Chinese text into manageable chunks.
    
    Args:
        text: Text to segment
        max_length: Maximum length of each segment
        
    Returns:
        List of text segments
    """
    processor = ChineseTextProcessor()
    return processor.segment_text(text, max_length)


def clean_text(text: str) -> str:
    """
    Clean and normalize Chinese text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    processor = ChineseTextProcessor()
    return processor.clean_text(text)


def get_text_statistics(text: str) -> Dict[str, Union[int, float]]:
    """
    Get comprehensive text statistics.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with text statistics
    """
    processor = ChineseTextProcessor()
    return processor.get_text_statistics(text)


def validate_chinese_text(text: str) -> Tuple[bool, List[str]]:
    """
    Validate Chinese text for common issues.
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    processor = ChineseTextProcessor()
    return processor.validate_chinese_text(text)
