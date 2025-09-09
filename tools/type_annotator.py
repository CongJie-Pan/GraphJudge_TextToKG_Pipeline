"""
Lightweight Type Annotator for ECTD Pipeline

This script implements a lightweight entity type annotation system that assigns
semantic types to extracted entities using rule-based patterns and regex matching.
The annotation system is optimized for classical Chinese literature, particularly
"Dream of the Red Chamber" (紅樓夢), and supports standard entity types.

Key Features:
1. Rule-based type assignment using regex patterns
2. Support for multiple entity types: PERSON, LOCATION, CONCEPT, OBJECT, etc.
3. Classical Chinese name pattern recognition
4. Location and geographical pattern matching
5. Abstract concept and literary device recognition
6. TSV output format for downstream processing
7. Confidence scoring for type assignments
8. Extensible pattern configuration system

Entity Types Supported:
- PERSON: Character names, titles, roles
- LOCATION: Places, geographical locations, buildings
- CONCEPT: Abstract concepts, emotions, ideas
- OBJECT: Physical objects, items, artifacts
- ORGANIZATION: Groups, institutions, family lines
- EVENT: Actions, occurrences, incidents
- TEMPORAL: Time-related entities
- LITERARY: Literary devices, narrative elements

Usage:
    from tools.type_annotator import TypeAnnotator
    
    annotator = TypeAnnotator()
    typed_entities = annotator.annotate_entities_file("test_entity.txt")
    annotator.save_typed_entities(typed_entities, "test_entity_typed.tsv")

Command Line Usage:
    python tools/type_annotator.py --input test_entity.txt --output test_entity_typed.tsv
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EntityAnnotation:
    """
    Data class representing a typed entity annotation.
    
    This class encapsulates all information about an annotated entity,
    including the original text, assigned type, confidence score, and
    matching pattern information for traceability.
    """
    entity: str                    # Original entity text
    entity_type: str              # Assigned type (PERSON, LOCATION, etc.)
    confidence: float             # Confidence score (0.0 to 1.0)
    pattern_matched: str          # Pattern that triggered the assignment
    pattern_category: str         # Category of the matching pattern
    
    def to_tsv_row(self) -> str:
        """Convert annotation to TSV format row."""
        return f"{self.entity}\t{self.entity_type}\t{self.confidence:.3f}\t{self.pattern_matched}\t{self.pattern_category}"


class TypeAnnotator:
    """
    Lightweight entity type annotator using rule-based pattern matching.
    
    This class implements a comprehensive type annotation system that uses
    regex patterns and linguistic rules to assign semantic types to entities
    extracted from classical Chinese text. The system is designed to be
    fast, accurate, and easily extensible with new patterns.
    """
    
    def __init__(self, config_path: str = None, fuzzy_threshold: float = 0.8):
        """
        Initialize the TypeAnnotator with configurable patterns.
        
        Args:
            config_path (str, optional): Path to custom pattern configuration file.
                                       If None, uses default patterns optimized for
                                       classical Chinese literature.
            fuzzy_threshold (float): Threshold for fuzzy matching (0.0 to 1.0).
                                   Default is 0.8 for balanced accuracy.
        
        The annotator uses a hierarchical pattern matching system where
        high-confidence patterns are checked first, followed by medium and
        low confidence patterns. This ensures accurate type assignment while
        maintaining good coverage.
        """
        self.setup_logging()
        self.patterns = self._load_pattern_configuration(config_path)
        self.fuzzy_threshold = fuzzy_threshold
        self.statistics = {
            "total_entities": 0,
            "typed_entities": 0,
            "untyped_entities": 0,
            "type_distribution": defaultdict(int),
            "confidence_distribution": defaultdict(int),
            "pattern_usage": defaultdict(int)
        }
        
    def setup_logging(self):
        """
        Configure logging for detailed tracking of annotation process.
        
        This method sets up comprehensive logging to track the annotation process,
        including debug information about pattern matching and type assignments.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/type_annotation.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
    
    def _load_pattern_configuration(self, config_path: str = None) -> Dict[str, Dict]:
        """
        Load pattern configuration for entity type recognition.
        
        Args:
            config_path (str, optional): Path to custom configuration file
            
        Returns:
            Dict[str, Dict]: Dictionary containing patterns organized by type and confidence
            
        The pattern configuration includes regex patterns for different entity types,
        organized by confidence levels (high, medium, low) to optimize matching
        accuracy and performance.
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default pattern configuration optimized for classical Chinese literature
        return {
            "PERSON": {
                "high_confidence": [
                    # Dream of Red Chamber specific characters (highest priority)
                    (r'^(甄士隱|賈雨村|林黛玉|賈寶玉|王熙鳳|薛寶釵|史湘雲|賈迎春|賈探春|賈惜春|妙玉|李紈|秦可卿|巧姐|劉姥姥|賈政|王夫人|賈母|薛姨媽|趙姨娘|周瑞家的|來旺家的|吳興家的|鄭好時|夏金桂|賈珍|賈璉|薛蟠|香菱|晴雯|襲人|鴛鴦|平兒|紫鵑|雪雁|秋紋|碧痕|麝月|茜雪|佳蕙|四兒|芳官|蕊官|藕官|豆官|葵官|艾官|寶官|玉官|茄官|藥官|小紅|小螺|司棋|待書|繡橘|待月|彩霞|彩雲|彩鳳|玉釧兒|金釧兒|鶯兒|翠墨|翠縷|智能兒|金桂|寶蟾|香菱|甄英蓮|嬌杏|冷子興|賈代化|賈代善|史老太君|王子騰|王仁|薛姨爹|呆霸王|夏金桂|甄寶玉|北靜王|忠順王爺|甄老爺|甄太太)$', 1.0, "hongloumeng_character"),
                    # Character titles and honorifics
                    (r'.*[公王侯伯子男爺娘夫人小姐公子少爺老爺太太奶奶姑娘丫頭書僮小廝]$', 0.9, "title_suffix"),
                    # Religious and mythological figures
                    (r'.*[仙子神君佛祖菩薩羅漢真人道士和尚尼姑]$', 0.9, "religious_title"),
                    # Classical Chinese personal names with family names (more specific)
                    # Only match names that are clearly personal names, not locations or objects
                    (r'^(賈|王|林|薛|史|李|秦|劉|趙|周|吳|鄭|馮|陳|蔣|沈|韓|楊|朱|尤|許|何|呂|施|張|孔|曹|嚴|華|金|魏|陶|姜|戚|謝|鄒|喻|柏|水|竇|章|雲|蘇|潘|葛|奚|范|彭|郎|魯|韋|昌|馬|苗|鳳|花|方|俞|任|袁|柳|酆|鮑|唐|費|廉|岑|雷|賀|倪|湯|滕|殷|羅|畢|郝|鄔|安|常|樂|于|時|傅|皮|卞|齊|康|伍|余|元|卜|顧|孟|平|黃|和|穆|蕭|尹|姚|邵|湛|汪|祁|毛|禹|狄|米|貝|明|臧|計|伏|成|戴|談|宋|茅|龐|熊|紀|舒|屈|項|祝|董|梁|杜|阮|藍|閔|席|季|麻|強|路|婁|危|江|童|顏|郭|梅|盛|刁|鍾|徐|邱|駱|高|夏|蔡|田|樊|胡|凌|霍|虞|萬|支|柯|昝|管|盧|莫|經|房|裘|繆|干|解|應|宗|丁|宣|賁|鄧|鬱|單|杭|洪|包|諸|左|石|崔|吉|鈕|龔|程|嵇|邢|滑|裴|陸|榮|翁|荀|羊|於|惠|甄|曲|家|封|芮|羿|儲|靳|汲|邴|糜|松|井|段|富|巫|烏|焦|巴|弓|牧|隗|山|谷|車|侯|宓|蓬|全|郗|班|仰|秋|仲|伊|宮|寧|仇|欒|暴|甘|鈞|厲|戎|祖|武|符|劉|景|詹|束|龍|葉|幸|司|韶|郜|黎|薊|薄|印|宿|白|懷|蒲|邰|從|鄂|索|咸|籍|賴|卓|藺|屠|蒙|池|喬|陰|鬱|胥|能|蒼|雙|聞|莘|黨|翟|譚|貢|勞|逄|姬|申|扶|堵|冉|宰|酈|雍|卻|璩|桑|桂|濮|牛|壽|通|邊|扈|燕|冀|郟|浦|尚|農|溫|別|莊|晏|柴|瞿|閻|充|慕|連|茹|習|宦|艾|魚|容|向|古|易|慎|戈|廖|庾|終|暨|居|衡|步|都|耿|滿|弘|匡|國|文|寇|廣|祿|闕|東|歐|殳|沃|利|蔚|越|夔|隆|師|鞏|厙|聶|晁|勾|敖|融|冷|訾|辛|闞|那|簡|饒|空|曾|毋|沙|乜|養|鞠|須|豐|巢|關|蒯|相|查|後|荊|紅|游|竺|權|逯|蓋|後|桓|公|万俟|司馬|上官|歐陽|夏侯|諸葛|聞人|東方|赫連|皇甫|尉遲|公羊|澹台|公冶|宗政|濮陽|淳于|單于|太叔|申屠|公孫|仲孫|軒轅|令狐|鍾離|宇文|長孫|慕容|鮮于|閭丘|司徒|司空|丌官|司寇|仉督|子車|顓孫|端木|巫馬|公西|漆雕|樂正|壤駟|公良|拓跋|夾谷|宰父|穀梁|晉楚|閆法|汝鄢|塗欽|段干|百里|東郭|南門|呼延|歸海|羊舌|微生|嶽帥|緱亢|況郈|有琴|梁丘|左丘|東門|西門|商牟|佘佴|伯賞|南宮|墨哈|譙笪|年愛|陽佟|言福)[\u4e00-\u9fff]{1,2}$', 1.0, "chinese_full_name")
                ],
                "medium_confidence": [
                    # Names with common Chinese name characters
                    (r'[\u4e00-\u9fff]*[玉雲梅蘭菊竹松鳳龍虎豹鶴鵬雁燕鶯蝶花草山水河海天地日月星辰春夏秋冬雨雪風雷][\u4e00-\u9fff]*', 0.7, "poetic_name_elements"),
                    # Two or three character names
                    (r'^[\u4e00-\u9fff]{2,3}$', 0.6, "standard_name_length")
                ],
                "low_confidence": [
                    # Single character that could be a name
                    (r'^[\u4e00-\u9fff]$', 0.4, "single_character")
                ]
            },
            "LOCATION": {
                "high_confidence": [
                    # Geographical locations with clear indicators
                    (r'.*[山嶺峰嶽岭峯崖崗丘陵谷川江河湖海洋池塘溪澗瀑泉井]$', 0.95, "geographical_suffix"),
                    (r'.*[城府州縣郡鎮村莊寨堡關門樓閣殿堂廟寺院觀塔亭台榭軒齋房屋宅院園林]$', 0.95, "architectural_suffix"),
                    # Direction-based locations
                    (r'.*[東西南北中上下前後左右內外].*', 0.8, "directional_location"),
                    # Dream of Red Chamber specific locations
                    (r'^(大荒山|無稽崖|青埂峰|太虛幻境|離恨天|赤瑕宮|警幻仙子宮|西方靈河岸|三生石|北邙山|姑蘇|閶門|十里街|仁清巷|葫蘆廟|胡州|神京|大如州)$', 1.0, "hongloumeng_location")
                ],
                "medium_confidence": [
                    # Places with locative particles
                    (r'.*[之於在].*', 0.6, "locative_particle"),
                    # Common place name patterns
                    (r'[\u4e00-\u9fff]*[地方處所場場所]', 0.7, "place_indicator")
                ],
                "low_confidence": [
                    # Potentially geographical terms
                    (r'[\u4e00-\u9fff]*[境界域區]', 0.5, "boundary_territory")
                ]
            },
            "CONCEPT": {
                "high_confidence": [
                    # Abstract philosophical concepts
                    (r'.*[道理義禮仁智信勇恕忠孝悌慈愛恨情慾志氣神魂魄靈夢幻].*', 0.9, "philosophical_concept"),
                    # Emotional and psychological states
                    (r'.*[喜怒哀樂愁憂思慮疑懼驚恐慚愧羞辱榮辱得失成敗禍福吉凶].*', 0.85, "emotional_state"),
                    # Literary and narrative concepts
                    (r'.*[因果緣分命運天意造化玄機奧秘真假虛實夢醒].*', 0.9, "literary_concept")
                ],
                "medium_confidence": [
                    # General abstract nouns
                    (r'.*[事情事物現象現實真相道理原因結果目的意義價值作用影響效果].*', 0.7, "abstract_noun"),
                    # Social and cultural concepts
                    (r'.*[文化傳統習俗禮儀制度規矩法則].*', 0.75, "cultural_concept")
                ],
                "low_confidence": [
                    # Potentially abstract terms
                    (r'.*[性質特點特色特徵].*', 0.5, "quality_trait")
                ]
            },
            "OBJECT": {
                "high_confidence": [
                    # Dream of Red Chamber specific objects (highest priority)
                    (r'^(通靈寶玉|石頭記|紅樓夢|情僧錄|風月寶鑒|金陵十二釵|好了歌)$', 1.0, "hongloumeng_object"),
                    # Specific objects and artifacts
                    (r'.*[玉石珠寶金銀銅鐵器具用品工具書籍文房].*', 0.9, "valuable_object"),
                    (r'.*[衣服帽子鞋襪首飾裝飾品].*', 0.85, "clothing_accessory")
                ],
                "medium_confidence": [
                    # Common objects
                    (r'.*[桌椅床榻几案櫃箱盒瓶壺碗盤].*', 0.75, "furniture_utensil"),
                    # Natural objects
                    (r'.*[花草樹木果實種子根葉枝幹].*', 0.7, "natural_object")
                ],
                "low_confidence": [
                    # Generic object indicators
                    (r'.*[物品東西].*', 0.5, "generic_object")
                ]
            },
            "ORGANIZATION": {
                "high_confidence": [
                    # Family and clan names
                    (r'.*[家族門第世系血脈宗族].*', 0.9, "family_organization"),
                    # Official and institutional terms
                    (r'.*[朝廷官府衙門部門機構組織團體].*', 0.85, "official_organization")
                ],
                "medium_confidence": [
                    # Social groups
                    (r'.*[群眾百姓民眾人民大眾].*', 0.7, "social_group"),
                    # Professional groups
                    (r'.*[商賈工匠農夫學者文人].*', 0.75, "professional_group")
                ],
                "low_confidence": [
                    # Generic group terms
                    (r'.*[眾人大家].*', 0.5, "generic_group")
                ]
            },
            "EVENT": {
                "high_confidence": [
                    # Specific events and actions
                    (r'.*[婚嫁喪葬祭祀慶典儀式].*', 0.9, "ceremonial_event"),
                    # Historical events
                    (r'.*[戰爭戰鬥征伐討伐起義叛亂革命].*', 0.85, "military_event")
                ],
                "medium_confidence": [
                    # General events and activities
                    (r'.*[會議聚會集會宴會筵席].*', 0.75, "social_event"),
                    # Natural events
                    (r'.*[災難災害天災人禍].*', 0.7, "disaster_event")
                ],
                "low_confidence": [
                    # Generic event terms
                    (r'.*[事件事情].*', 0.5, "generic_event")
                ]
            },
            "TEMPORAL": {
                "high_confidence": [
                    # Specific time periods
                    (r'.*[春夏秋冬季節年月日時辰朝暮晨昏].*', 0.9, "time_period"),
                    # Dynasty and era names
                    (r'.*[朝代王朝時代年代世代].*', 0.85, "historical_period")
                ],
                "medium_confidence": [
                    # Time-related terms
                    (r'.*[時候時機時刻時間].*', 0.75, "time_related"),
                    # Sequential time
                    (r'.*[過去現在將來以前以後從前往後].*', 0.7, "temporal_sequence")
                ],
                "low_confidence": [
                    # Generic temporal indicators
                    (r'.*[當時那時此時].*', 0.5, "temporal_indicator")
                ]
            }
        }
    
    def _match_patterns(self, entity: str, entity_type: str) -> Optional[EntityAnnotation]:
        """
        Match an entity against patterns for a specific type.
        
        Args:
            entity (str): Entity text to match
            entity_type (str): Type category to check patterns for
            
        Returns:
            Optional[EntityAnnotation]: Annotation if pattern matches, None otherwise
            
        This method checks an entity against all patterns for a given type,
        starting with high-confidence patterns and working down. The first
        matching pattern determines the annotation.
        """
        if entity_type not in self.patterns:
            return None
        
        # Check patterns in order of confidence: high -> medium -> low
        for confidence_level in ["high_confidence", "medium_confidence", "low_confidence"]:
            if confidence_level not in self.patterns[entity_type]:
                continue
                
            for pattern_data in self.patterns[entity_type][confidence_level]:
                pattern, confidence, pattern_name = pattern_data
                
                if re.search(pattern, entity):
                    self.statistics["pattern_usage"][f"{entity_type}_{pattern_name}"] += 1
                    
                    return EntityAnnotation(
                        entity=entity,
                        entity_type=entity_type,
                        confidence=confidence,
                        pattern_matched=pattern_name,
                        pattern_category=confidence_level
                    )
        
        return None
    
    def annotate_entity(self, entity: str) -> EntityAnnotation:
        """
        Annotate a single entity with its most likely type.
        
        Args:
            entity (str): Entity text to annotate
            
        Returns:
            EntityAnnotation: Annotation with type, confidence, and pattern info
            
        This method attempts to match the entity against all type patterns,
        selecting the match with the highest confidence score. If no patterns
        match, it assigns a default "UNKNOWN" type with low confidence.
        """
        entity = entity.strip()
        if not entity:
            # Update statistics for empty entity
            self.statistics["total_entities"] += 1
            self.statistics["untyped_entities"] += 1
            self.statistics["type_distribution"]["UNKNOWN"] += 1
            self.statistics["confidence_distribution"]["0-9%"] += 1
            
            return EntityAnnotation(
                entity=entity,
                entity_type="UNKNOWN",
                confidence=0.0,
                pattern_matched="empty_entity",
                pattern_category="none"
            )
        
        best_annotation = None
        best_confidence = 0.0
        
        # Try to match against all entity types in priority order
        # Check specific object types first, then PERSON for character names, then other types
        priority_order = ["OBJECT", "PERSON", "LOCATION", "CONCEPT", "ORGANIZATION", "EVENT", "TEMPORAL", "LITERARY"]
        
        for entity_type in priority_order:
            if entity_type in self.patterns:
                annotation = self._match_patterns(entity, entity_type)
                if annotation and annotation.confidence > best_confidence:
                    best_annotation = annotation
                    best_confidence = annotation.confidence
                    # If we found a high-confidence match (0.9+), use it immediately
                    if annotation.confidence >= 0.9:
                        break
        
        # If no pattern matched, assign UNKNOWN type
        if best_annotation is None:
            best_annotation = EntityAnnotation(
                entity=entity,
                entity_type="UNKNOWN",
                confidence=0.1,  # Low confidence for unknown entities
                pattern_matched="no_pattern_match",
                pattern_category="none"
            )
            self.statistics["untyped_entities"] += 1
        else:
            self.statistics["typed_entities"] += 1
        
        # Update statistics
        self.statistics["total_entities"] += 1
        self.statistics["type_distribution"][best_annotation.entity_type] += 1
        confidence_bucket = f"{int(best_annotation.confidence * 10) * 10}-{int(best_annotation.confidence * 10) * 10 + 9}%"
        self.statistics["confidence_distribution"][confidence_bucket] += 1
        
        return best_annotation
    
    def annotate_entity_list(self, entities: List[str]) -> List[EntityAnnotation]:
        """
        Annotate a list of entities with their types.
        
        Args:
            entities (List[str]): List of entity strings to annotate
            
        Returns:
            List[EntityAnnotation]: List of annotations for each entity
            
        This method processes a list of entities in batch, applying type
        annotation to each entity and returning a corresponding list of
        annotations.
        """
        annotations = []
        for entity in entities:
            if entity.strip():  # Skip empty entities
                annotation = self.annotate_entity(entity)
                annotations.append(annotation)
        
        return annotations
    
    def annotate_entities_file(self, input_path: str) -> List[List[EntityAnnotation]]:
        """
        Annotate entities from an entity file.
        
        Args:
            input_path (str): Path to input entity file
            
        Returns:
            List[List[EntityAnnotation]]: List of annotation lists, one per line
            
        This method reads an entity file and applies type annotation to all
        entities, preserving the line structure of the original file.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self.logger.info(f"Starting entity type annotation for file: {input_path}")
        
        annotated_lines = []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse entities from line (assuming same format as clean_entities.py)
                entities = self._parse_entity_line(line)
                if entities:
                    annotations = self.annotate_entity_list(entities)
                    annotated_lines.append(annotations)
                    
                    self.logger.debug(f"Line {line_num}: Annotated {len(annotations)} entities")
            
            self.logger.info(f"Entity annotation completed. Processed {len(lines)} lines, "
                           f"annotated {self.statistics['total_entities']} entities")
            
        except Exception as e:
            self.logger.error(f"Error processing file {input_path}: {e}")
            raise
        
        return annotated_lines
    
    def _parse_entity_line(self, line: str) -> List[str]:
        """
        Parse a single entity line into a list of entities.
        
        Args:
            line (str): Raw entity line from the input file
            
        Returns:
            List[str]: List of individual entities
            
        This method reuses the parsing logic from clean_entities.py to handle
        the same input formats consistently.
        """
        line = line.strip()
        if not line:
            return []
        
        # Handle Python list format: ["entity1", "entity2"]
        if line.startswith('[') and line.endswith(']'):
            try:
                entities = eval(line)
                if isinstance(entities, list):
                    return [str(entity).strip() for entity in entities if str(entity).strip()]
            except (SyntaxError, ValueError):
                pass
        
        # Handle comma-separated format
        cleaned_line = re.sub(r'[\[\]"\'""''「」『』]', '', line)
        entities = [entity.strip() for entity in cleaned_line.split(',')]
        entities = [entity for entity in entities if entity]
        
        return entities
    
    def save_typed_entities(self, annotated_lines: List[List[EntityAnnotation]], 
                          output_path: str) -> None:
        """
        Save annotated entities to a TSV file.
        
        Args:
            annotated_lines (List[List[EntityAnnotation]]): Annotated entities
            output_path (str): Path for the output TSV file
            
        This method saves the annotated entities in TSV format with columns for
        entity text, type, confidence, matching pattern, and pattern category.
        The format is optimized for easy loading by downstream processing tools.
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                   exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write TSV header
                f.write("entity\ttype\tconfidence\tpattern\tcategory\n")
                
                # Write entity annotations
                for line_annotations in annotated_lines:
                    for annotation in line_annotations:
                        f.write(annotation.to_tsv_row() + '\n')
                    # Add empty line to separate original input lines
                    f.write('\n')
            
            self.logger.info(f"Typed entities saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving typed entities to {output_path}: {e}")
            raise
    
    def get_annotation_statistics(self) -> Dict[str, Union[int, Dict]]:
        """
        Get comprehensive statistics about the annotation process.
        
        Returns:
            Dict: Dictionary containing detailed annotation statistics
            
        This method provides insights into the annotation process, including
        type distribution, confidence levels, and pattern usage statistics.
        """
        return {
            "total_entities": self.statistics["total_entities"],
            "typed_entities": self.statistics["typed_entities"],
            "untyped_entities": self.statistics["untyped_entities"],
            "typing_rate": self.statistics["typed_entities"] / max(1, self.statistics["total_entities"]),
            "type_distribution": dict(self.statistics["type_distribution"]),
            "confidence_distribution": dict(self.statistics["confidence_distribution"]),
            "pattern_usage": dict(self.statistics["pattern_usage"])
        }
    
    def print_statistics(self) -> None:
        """
        Print a formatted summary of annotation statistics.
        
        This method provides a human-readable summary of the annotation process,
        including type distribution, confidence metrics, and pattern effectiveness.
        """
        stats = self.get_annotation_statistics()
        
        print("\n" + "="*60)
        print("ENTITY TYPE ANNOTATION STATISTICS")
        print("="*60)
        print(f"Total entities processed: {stats['total_entities']}")
        print(f"Successfully typed: {stats['typed_entities']}")
        print(f"Untyped (UNKNOWN): {stats['untyped_entities']}")
        print(f"Typing success rate: {stats['typing_rate']:.1%}")
        
        print("\nType Distribution:")
        print("-" * 30)
        for entity_type, count in sorted(stats['type_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['total_entities'] * 100
            print(f"  {entity_type:<15}: {count:>4} ({percentage:>5.1f}%)")
        
        print("\nConfidence Distribution:")
        print("-" * 30)
        for conf_range, count in sorted(stats['confidence_distribution'].items()):
            percentage = count / stats['total_entities'] * 100
            print(f"  {conf_range:<15}: {count:>4} ({percentage:>5.1f}%)")
        
        print("\nTop Pattern Usage:")
        print("-" * 30)
        top_patterns = sorted(stats['pattern_usage'].items(), key=lambda x: x[1], reverse=True)[:10]
        for pattern, count in top_patterns:
            print(f"  {pattern:<30}: {count:>4}")
        
        print("="*60)


def main():
    """
    Command-line interface for the type annotator.
    
    This function provides a convenient command-line interface for running
    the entity type annotation process with customizable parameters.
    """
    parser = argparse.ArgumentParser(
        description="Annotate entities with semantic types using rule-based patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python type_annotator.py --input test_entity.txt --output test_entity_typed.tsv
  python type_annotator.py --input data/entities.txt --config custom_patterns.json --verbose
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input entity file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output typed entity TSV file path'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Custom pattern configuration file path'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize annotator with optional custom configuration
        annotator = TypeAnnotator(config_path=args.config)
        
        # Annotate entities from file
        annotated_entities = annotator.annotate_entities_file(args.input)
        
        # Save typed entities
        annotator.save_typed_entities(annotated_entities, args.output)
        
        # Print statistics
        annotator.print_statistics()
        
        print(f"\n✅ Entity type annotation completed successfully!")
        print(f"📄 Input: {args.input}")
        print(f"📄 Output: {args.output}")
        print(f"📊 Format: TSV with type annotations")
        
    except Exception as e:
        print(f"❌ Error during entity type annotation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
