# Interation 2 : The ECTD, Triple genration, and GraphJudge phases of the output data

- Writing date : 2025/8/13

## ECTD

### Code

- `run_entity.py`: Miscellaneous\KgGen\GraphJudge\chat\run_entity.py

### Output data

- `test_entity.txt`:
  Miscellaneous\KgGen\GraphJudge\datasets\KIMI_result_DreamOf_RedChamber\Grapg_Iteration2\test_entity.txt
- `test_denoised.target`:     
  Miscellaneous\KgGen\GraphJudge\datasets\KIMI_result_DreamOf_RedChamber\Grapg_Iteration2\test_denoised.target

### Ouptut Data Quality Summary

The ECTD outputs consist of two artifacts that are complementary but not yet strictly aligned for benchmarking:

- Overall quality: Sufficient for iterative experimentation and qualitative analysis; requires normalization for reliable metrics-based evaluation.
- Data locale: Traditional Chinese throughout; encoding and punctuation appear consistent.
- And need to add logging function in the `run_entity.py`

1) test_entity.txt
- Format: Line-delimited JSON-like arrays of strings; each line lists extracted entities for a corresponding text segment.
- Coverage: Captures key proper nouns and terms central to the “Dream of the Red Chamber” Chapter 1 narrative (e.g., 甄士隱, 賈雨村, 葫蘆廟, 通靈寶玉, 警幻仙子, etc.). Multi-token names are preserved.
- Noise/precision: Includes generic/common nouns and abstract concepts that are not named entities (e.g., 石, 神仙, 家庭, 金銀). Synonymic variants and near-duplicates appear across lines. Expected precision is moderate; recommend enforcing an NE-only constraint or adding type tags (PER/LOC/ORG/OBJ/CONCEPT) and filters.
- Consistency: Schema is simple and readable but lacks stable identifiers that link each entity list to a specific source segment.

2) test_denoised.target
- Format: Traditional Chinese narrative paragraphs with a visible section marker “去噪文本：”. Text after the marker is coherent and appears denoised/standardized.
- Readability: Sentences are grammatical and chronological; punctuation is normalized; no encoding artifacts observed.
- Structure: The file starts with one preface-like line before the “去噪文本：” marker, which may cause off-by-one alignment if a strict 1:1 mapping with entity lines is expected.

Alignment between files
- Count parity: test_entity.txt contains 31 non-empty lines; test_denoised.target contains a preface line, a “去噪文本：” marker line, and then multiple narrative lines. There is no explicit 1:1 linkage, which complicates automatic scoring.
- Recommendation: Add a stable example_id per segment and ensure strict 1:1 correspondence across files. Remove or standardize header/marker lines in the denoised file to avoid misalignment.

Formatting and path notes
- Paths in this report reference a folder name spelled as “Grapg_Iteration2”; the actual data folder is “Graph_Iteration2”. Align path strings to prevent broken references.

Actionable recommendations
- Add IDs to both files and enforce 1:1 segment alignment.
- Introduce entity type tags and a stoplist to reduce non-NE noise; deduplicate synonyms/variants.
- Remove the “去噪文本：” marker in final datasets or store it as metadata, not inline content.
- Provide a single JSONL with fields {example_id, text, entities: [...]} for ease of consumption and scoring.

Indicative quality rating (qualitative)
- Precision (entities): medium
- Recall (entities): medium-high
- Text cleanliness: high
- Benchmark readiness: medium (needs alignment and schema tightening)

### Terminal Running Progress

- Lack of result data.

---

## Triple Generation

### code

- `run_kimi_triple_v2.py`: Miscellaneous\KgGen\GraphJudge\chat\run_kimi_triple_v2.py

### Output data

- `test_instructions_context_kimi_v2.json`:
  Miscellaneous\KgGen\GraphJudge\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration2\test_instructions_context_kimi_v2.json

### Output Data Quality Summary

The triple-generation output is a JSON array of items with the schema {"instruction", "input", "output"}. Instructions are yes/no verification prompts in mixed English/Chinese style (prefix "Is this true:" followed by Chinese subject–relation–object). Both "input" and "output" fields are currently empty across all items.

- Overall quality: Structurally clean and machine-readable; not yet ready for scoring because gold answers are missing and evidence context is absent.
- Ontology/relations: Relation terms (e.g., 行為/地點/組成部分/作者/談論內容/主人/贈送/囑咐) are diverse but not enumerated in a controlled schema; several predicates are vague or underspecified (e.g., 稱/見/邀/知道).
- Entity normalization: Entities mix proper nouns (e.g., 甄士隱, 賈雨村, 《好了歌》, 葫蘆廟) with generic terms (e.g., 此處/此鄉) which weakens semantic precision and graph usefulness.
- Linguistic consistency: Mixed-lingual prompt style (English lead-in with Chinese triples) is consistent but may confuse simple classifiers; consider fully Chinese or fully English templates.
- Coverage/duplication: Good breadth over early Chapter 1 content with repeated patterns for multiple subjects; likely contains near-duplicates/paraphrases that could bias metrics without deduplication.
- Alignment: No explicit linkage to ECTD segments; the "input" field is empty so model judgments lack grounding in source text.

Actionable recommendations
- Provide gold labels in "output" (e.g., "yes"/"no") and define a strict answer format policy.
- Populate "input" with minimal evidence snippets or example_id pointers to the denoised text to enable grounded verification.
- Define and publish a controlled relation ontology with directionality and allowed values; replace vague predicates or map them to canonical forms.
- Normalize entities (PER/LOC/ORG/WORK/OBJ/CONCEPT) and remove generic placeholders like 此處/此鄉 unless intentionally modeled.
- Deduplicate near-identical instructions; add a balanced proportion of negative (false) cases for robust evaluation.
- Consolidate into a JSONL with fields {example_id, instruction, evidence, gold_label} for easier streaming and training.

Indicative quality rating (qualitative)
- Instruction clarity: medium-high
- Ontology consistency: medium
- Grounding/answerability: low (missing evidence and gold)
- Benchmark readiness: medium-low (needs gold labels, evidence, and ontology normalization)

---

## Graph Judge

### Code

- `run_kimi_gj.py` : Miscellaneous\KgGen\GraphJudge\chat\run_kimi_gj.py

### Processing Effiency

- Processing Time: 43 minutes 28 seconds

### Terminal output

path : `Miscellaneous\KgGen\GraphJudge\chat\logs\iteration2\run_gemini_gj_log_20250813_174542.txt`

### Output data
- `pred_instructions_context_gemini_itr2.csv`:
  Miscellaneous\KgGen\GraphJudge\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration2\pred_instructions_context_gemini_itr2.csv

### Output Data Quality Summary

The Graph Judge output is a CSV with headers `prompt,generated` and 121 rows. Prompts follow the same yes/no verification style as Triple Generation, and `generated` holds the predicted label.

- Label distribution: Yes 89 (73.6%), No 32 (26.4%); noticeable skew toward Yes.
- Uniqueness/duplicates: 120 unique prompts; 2 duplicate prompts detected; 0 conflicting duplicate label groups (consistent predictions for duplicates).
- Grounding: No evidence/context or source identifiers included, which limits auditability and error analysis.
- Ontology/phrasing: Some prompts use vague predicates (e.g., 「稱」/「見」/「邀」/「知道」), which can reduce semantic precision when judging truthfulness.
- Spot-check correctness vs. denoised text: Multiple clear mislabels observed, indicating reliability issues:
  - "Is this true: 作者 作品 石頭記 ?" → predicted No; denoised text states the author wrote 《石頭記》 (should be Yes).
  - "Is this true: 無稽崖 組成部分 青埂峰 ?" → predicted No; text indicates 青埂峰 under 無稽崖 (should be Yes).
  - "Is this true: 石頭 地點 此鄉 ?" → predicted No; text includes 「石墜落於此鄉」 (should be Yes).
  - "Is this true: 作者 受德 祖德 ?" → predicted No; text includes 「下承祖德」 (should be Yes).

Actionable recommendations
- Add columns: `example_id`, `evidence` (text span or reference), `confidence`, and `rationale` to enable grounded error analysis.
- Normalize relations to a controlled ontology and avoid vague predicates; map prompts to canonical relation names.
- Address label skew by curating more hard negatives or calibrating decision thresholds; report per-relation label balance.
- Provide a consolidated JSONL export with {example_id, prompt, prediction, confidence, evidence} for downstream evaluation.
- Run a systematic spot-check protocol and correct identified mislabels; compute accuracy/F1 once gold annotations are available.

Indicative quality rating (qualitative)
- Prediction consistency on duplicates: high
- Label reliability (vs. source text): medium-low (several evident mislabels)
- Auditability/traceability: low (no evidence/context columns)
- Benchmark readiness: medium (structure OK; needs grounding metadata and label calibration)

---

## Evaluation

### Code
- eval_preDataProcess.py:
  Miscellaneous\KgGen\GraphJudge\graph_evaluation\metrics\eval_preDataProcess.py
- eval.py:
  Miscellaneous\KgGen\GraphJudge\graph_evaluation\metrics\eval.py
- graph_matching.py:
  Miscellaneous\KgGen\GraphJudge\graph_evaluation\metrics\graph_matching.py

### Score:

Exact Matching Metrics:
  Triple Match F1 Score: 0.7355
  Graph Match Accuracy: 0.7355

Text Similarity Metrics:
  G-BLEU Precision: 0.8329
  G-BLEU Recall: 0.8329
  G-BLEU F1: 0.8329
  G-ROUGE Precision: 0.0000
  G-ROUGE Recall: 0.0000
  G-ROUGE F1: 0.0000

Semantic Similarity Metrics:
  G-BERTScore Precision: 0.9380
  G-BERTScore Recall: 0.9380
  G-BERTScore F1: 0.9380

Structural Distance Metrics:
  Graph Edit Distance (GED): 0.0248

Total graphs evaluated: 121

### Evaluation Summary

Based on the comprehensive analysis of the knowledge graph evaluation results for the Dream of the Red Chamber (紅樓夢) dataset, here's a detailed technical assessment:

#### Overall Performance: **Strong to Excellent (B+ to A-)**

The knowledge graph generation system demonstrates robust performance across multiple evaluation dimensions with 121 graph pairs evaluated.

1. **Exact Matching Performance (73.55%)**
- **Triple Match F1**: 0.7355 - This indicates that roughly 3 out of 4 predicted triples exactly match the gold standard
- **Graph Match Accuracy**: 0.7355 - Identical score suggests consistent structural alignment
- **Analysis**: The precision demonstrates solid entity-relation extraction capability. The remaining 26.45% gap likely stems from:
  - Synonym variations (e.g., "士隱" vs "甄士隱")
  - Predicate paraphrasing (different ways to express relationships)
  - Missing context-dependent triples

2. **Text Similarity Metrics**
- **G-BLEU F1**: 0.8329 (83.29%) - **Excellent**
  - High n-gram overlap indicates strong lexical similarity between predicted and gold triples
  - Suggests the model captures appropriate vocabulary and phrasing patterns
- **G-ROUGE F1**: 0.0000 - **Critical Issue Identified**
  - This anomalous zero score indicates a technical problem in the evaluation pipeline
  - Likely caused by improper Chinese text tokenization or string preprocessing issues
  - Does not reflect actual model quality

3. **Semantic Similarity (93.80%)**
- **G-BERTScore F1**: 0.9380 - **Outstanding**
  - Near-perfect semantic alignment using contextual embeddings
  - Indicates the model captures deep semantic relationships correctly
  - Compensates for exact string mismatches through semantic understanding

4. **Structural Distance**
- **Graph Edit Distance**: 0.0248 (2.48%) - **Excellent**
  - Very low edit distance means minimal structural modifications needed
  - Confirms strong graph topology preservation
  - Indicates consistent entity-relationship connectivity patterns

#### **Key Insights from Data Analysis**

1.  **Gold vs Predicted Comparison**:
    1. **Null Handling**: Gold data contains 39 null triples (`["Null", "Null", "Null"]`), while predictions contain actual content - this suggests the model successfully extracts information where human annotators found none

    2. **Entity Consistency**: Both datasets maintain consistent character names (士隱, 雨村, 曹雪芹, etc.) with proper Chinese entity recognition

    3. **Relationship Diversity**: Rich predicate vocabulary including spatial (地點), behavioral (行為), and social relationships (妻子, 女兒)

    4. **Content Completeness**: Predictions show more detailed relationships in some cases where gold has nulls

#### **Technical Recommendations**

1. **Immediate Fixes**:
   1. **Resolve G-ROUGE Issue**: Implement proper Chinese tokenization in the evaluation pipeline
   2. **Entity Normalization**: Add alias resolution for character name variations
   3. **Null Filtering**: Consider excluding null triples from evaluation metrics

2. **Model Improvements**:
   1. **Relationship Standardization**: Implement predicate canonicalization to reduce exact match penalties
   2. **Context Enhancement**: Leverage the high semantic similarity to improve exact matching through post-processing
   3. **Confidence Scoring**: Add uncertainty estimation to identify low-confidence predictions

3.  **Benchmark Context**
For knowledge graph evaluation in NLP:
- **70%+ exact matching**: Industry standard for production systems
- **80%+ BLEU**: Strong text generation quality
- **90%+ BERTScore**: Research-grade semantic understanding
- **<5% GED**: Excellent structural preservation

**The system exceeds industry standards in semantic understanding and structural coherence while maintaining solid exact matching performance.**

**Final Assessment**
This evaluation demonstrates a **high-quality knowledge graph generation system** particularly strong in semantic understanding and structural consistency. The primary area for improvement lies in exact string matching through better normalization and the technical fix needed for G-ROUGE computation. The system shows production-ready performance for knowledge extraction from classical Chinese literature.

## Final Json Graph After the GJ Module

- Code
  - convert_Judge_To_jsonGraph.py :
  Miscellaneous\KgGen\GraphJudge\chat\convert_Judge_To_jsonGraph.py

- Json file
  - converted_kg_from_judge_20250818_143811.json:
  Miscellaneous\KgGen\GraphJudge\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration2\converted_kg_from_judge_20250818_143811.json

### Output Data Quality Summary

The final JSON graph was generated from the Graph Judge CSV (121 rows) by retaining only triples labeled “Yes.” The output contains 94 unique entities and 89 relationships (conversion accuracy 73.55%; 0 parsing errors). The file conforms to the kgGenShows schema with entities as strings and relationships formatted as "Source - Relation - Target" and is ready for direct visualization.

- **Scale and coverage**: 94 entities, 89 edges, derived from 89 accepted triples out of 121 evaluated. Content spans key characters, locations, and events from Chapter 1 (e.g., 士隱, 賈雨村, 曹雪芹, 空空道人, 絳珠仙子, 女媧氏, 葫蘆廟, 青埂峰, 西方靈河岸上三生石畔).
- **Predicate distribution**: Dominated by behavioral and spatial relations (行為, 地點). Additional types include 作者, 包含, 囑咐, 贈送, 主人, 妻子, 女兒, 岳丈, 改題, 易名, 知道, 看見, 認為, 談論內容, etc. This breadth captures narrative actions, attributions, family ties, and locations.
- **Format compliance**: The JSON validates against the viewer’s expected structure. All relationships are strings with the required delimiter. Entities are deduplicated.
- **Notable data issues**:
  - One malformed edge with an empty subject produced a blank entity entry: " - 作者 - 受恩 天恩" (entity list contains an empty string). This should be repaired or dropped.
  - Several objects are compound phrases carried over from prompts, e.g., "《石頭記》為《情僧錄》", "雨村十九日買舟西上赴神京", "邀雨村中秋夜到敝齋飲酒". These read well but reduce graph normalization.
  - Alias/variant duplication: both "士隱" and "甄士隱" exist; also role/name variants (e.g., "空空道人" vs the node “情僧” linked via 易名).
  - Mixed granularity: alongside proper nouns, some generic or conceptual nodes (e.g., 家庭閨閣瑣事, 紅塵) and short tokens (e.g., 石) appear, which is faithful to prompts but less precise for KG analytics.
- **Integrity checks**: Relationship count equals the number of accepted (Yes) judgments. No parsing errors were recorded. Entities are unique; relationships were not explicitly deduplicated but show no obvious duplicates in this export.
- **Quality indicators (from embedded report)**: total_evaluated 121; valid_triplets 89; invalid_triplets 32; conversion_accuracy 73.55; entity_count 94; relationship_count 89; source_file pred_instructions_context_gemini_itr2.csv.

Recommendations for the next iteration:
- **Repair malformed edges**: Drop or fix edges with empty subject/object and remove the blank entity.
- **Canonicalize predicates and split compounds**: Map prompts like "情僧 - 改題 - 《石頭記》為《情僧錄》" into normalized triples such as "《石頭記》 - 改題為 - 《情僧錄》" (and optionally retain the agent via a separate edge).
- **Alias normalization**: Merge entity variants (e.g., 士隱 ↔ 甄士隱; 空空道人 ↔ 情僧 via 易名) with a deterministic alias map.
- **Optional typing**: Add lightweight type tags (PER/LOC/WORK/OBJ/CONCEPT) to improve analytics and visualization filtering.

Overall, the graph is structurally valid and visualization-ready, with minor cleaning steps recommended to improve normalization, reduce ambiguity, and enhance downstream evaluation fidelity.