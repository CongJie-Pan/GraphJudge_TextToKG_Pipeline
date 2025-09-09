# Improvement Plan – Iteration 1 ➜ Iteration 2

_This document proposes a concrete, step-by-step roadmap to raise the overall data‐quality and throughput of the three-stage pipeline (ECTD → Triple Generation → Graph Judge) described in **Iteration1_Report.md**.  Actions are grouped by phase, followed by cross-cutting engineering work and a suggested timeline._

---
## 0  Bird-eye Overview
| Phase | Current Pain-Points (Iteration 1) | Target State (Iteration 2) |
|-------|----------------------------------|---------------------------|
| ECTD | • duplicate/abstract tokens inside `test_entity.txt`  \• inconsistent spacing + trailing blank in `test_denoised.target`  \• <3 RPM bottleneck | • de-duplicated, type-tagged entity lists  \• fully aligned entity ↔ sentence mapping  \• 5× faster crawl via batch & caching |
| Triple Gen | • mixed Chinese prompt wrappers in `test_generated_graphs.txt`  \• relation synonyms ("地點" vs "位置")  \• missing ~40 % triples in JSON conversion | • auto-cleaned triples in pure TSV/JSONL  \• single controlled relation vocabulary  \• 100 % preservation with paging |
| Graph Judge | • small sample (30/153)  \• high false-negative rate (e.g. `作者 經歷 夢幻` marked **No**)  \• no gold labels | • full coverage  \• ≤10 % FN on held-out set  \• gold `expected` column + automated metrics |

---
## 1  ECTD (Entities + Clean Text Denoising)
1. **Post-Extraction Cleaner**  
   • _Script_: `tools/clean_entities.py`  
   • Remove duplicates _within a line_ (e.g.兩個 `"神瑛侍者"`)  
   • Drop generic categories ("功名", "朝代") via stop-list.
2. **Lightweight Type Annotator**  
   • Simple regex→type table (`人名|LOC|CONCEPT`)  
   • Output `test_entity_typed.tsv` → later boosts relation prompts.
3. **Sentence ↔ Entity Alignment Check**  
   • Ensure `len(test_entity) == len(test_denoised)`; warn & trim.
4. **Prompt Refinement** (run_entity.py)  
   • Add _system_ guideline to return **deduped, comma-separated list**  
   • Slightly raise temperature to 0.3 for recall.
5. **Concurrency + Cache**  
   • Implement disk cache (`.cache/kimi_ent/{hash}.json`) to avoid re-hits.  
   • Raise `KIMI_CONCURRENT_LIMIT` to 3 and compute dynamic `sleep=max(60/RPM,…)`.

_Quick win example_: after cleaner, row–5 will change from 
`["神瑛侍者", "神瑛侍者", "離恨天"]` → `神瑛侍者	LOC:離恨天`.

---
## 2  Triple Generation
1. **Unified Relation Vocabulary**  
   • Create `config/relation_map.json` (e.g. "地點"→"location", "位置"→"location").  
   • Map after model generation.
2. **Structured Output Prompt**  
   • Replace free-form list with explicit JSON:
```json
{"triples": [["主語","謂語","賓語"], …]}
```
3. **Post-processor**  
   • `tools/parse_kimi_triples.py` →  
     a. strip wrapper phrases  
     b. deduplicate triples  
     c. enforce vocabulary map.
4. **Pagination to Avoid Truncation**  
   • Feed text to model in ≤1k-token chunks; merge outputs so no rows lost.
5. **Schema Validation**  
   • Use `pydantic` to fail fast if JSON malformed.

_Example fix_: original duplicate
`("僧", "行為", "來至峰下")` appears twice; script keeps one.

---
## 3  Graph Judge
1. (❌ abandon-unnecessary)**Full Coverage Instruction Builder**  
   - **Abandon Reason**: The function repeative of `run_kimi_triple_v2.py`.
   - **Purpose**: Convert triples into judgment questions and paginate outputs (≤500 items per file) for `chat/run_kimi_gj.py`.
   - **Input**: `datasets/.../Graph_Iteration1/test_generated_graphs.txt` (TSV: `subject[TAB]predicate[TAB]object`)
   - **Output**:
     - `datasets/.../Graph_Iteration1/test_instructions_context_kimi_0001.json` (≤500 items per file)
     - `datasets/.../Graph_Iteration1/test_instructions_context_kimi_manifest.json` (manifest of paginated files)
   - **JSON schema (per item)**: `{ "instruction": "Is this true: S P O?", "input": "", "meta": {"subject":"...","predicate":"...","object":"...","row_id":N} }`
   - **CLI**:
     ```bash
     python tools/build_instructions.py \
       --triples-file ./datasets/.../Graph_Iteration1/test_generated_graphs.txt \
       --output-dir   ./datasets/.../Graph_Iteration1 \
       --page-size    500
     ```
   - **Errors**: Missing files/bad rows are logged to `logs/tools/build_instructions_errors.log` and skipped; all outputs use UTF-8.

2. **Gold Label Bootstrapping** - **After step 5(External RAG Verification) complete, to implement this.**
   - **Purpose**: Automatically assign `expected=True` to obviously true triples using source text, reducing manual labeling.
   - **Input**: 
     - Triples: `datasets/.../Graph_Iteration1/test_generated_graphs.txt`
     - Source: `datasets/.../Iteration1/test_denoised.target`
   - **Rule**: Fuzzy match the string `"S P O"` against source lines (e.g., `rapidfuzz.partial_ratio`). If any line ≥ 0.8 → set `auto_expected=True`, otherwise mark as `uncertain`.
   - **Sampling**: Randomly sample 15% from `uncertain` to create a manual-review list (field `expected` to be filled by humans).
   - **Mechanism**:
    1. **Stage-1 RapidFuzz pass** – Each triple is rendered as plain text ("S P O") and compared against every source sentence using `partial_ratio`. A similarity ≥ 0.8 directly flags the triple as `auto_expected=True`; lower scores send it to the `uncertain` bucket. RapidFuzz works purely on surface string overlap and does **not** understand meaning.
    2. **Stage-2 LLM semantic pass** – Triples in the `uncertain` bucket can be re-evaluated by a more infomation widely large-language model (LLM) or RAG system such as **KIMI** or **Perplexity**. The model judges whether the triple is semantically entailed by the source, capturing synonyms or paraphrases that RapidFuzz misses. This step is only triggered for a small subset to keep runtime and cost manageable.
   - **Rationale**: The two-tier pipeline balances speed and accuracy. RapidFuzz instantly resolves around 80 % of obvious cases at virtually zero cost, while the LLM focuses on the harder ~20 %, ensuring high recall without excessive human labeling.
   - **Illustrative scenario**:
    • The sentence "賈寶玉喜歡林黛玉" appears verbatim in the source – RapidFuzz similarity is high, triple is accepted automatically.
    • The source says "賈寶玉對林黛玉心生愛意" – RapidFuzz similarity is low, but the LLM recognises the synonym "愛意"≈"喜歡" and confirms correctness.
   - **Output**: `datasets/.../Graph_Iteration1/gold_bootstrap.csv` (columns: `subject,predicate,object,source_idx,fuzzy_score,auto_expected,expected,note`)
   - **CLI**:
     ```bash
     python tools/bootstrap_gold_labels.py \
       --triples-file ./datasets/.../Graph_Iteration1/test_generated_graphs.txt \
       --source-file  ./datasets/.../Iteration1/test_denoised.target \
       --output       ./datasets/.../Graph_Iteration1/gold_bootstrap.csv \
       --threshold    0.8 \
       --sample-rate  0.15
     ```
   - **Errors**: Missing files or malformed rows are logged to `logs/tools/bootstrap_errors.log`; the process continues.

3. ✅ **Judge Prompt Upgrade**  
   - **Goal**: Prepend a Chinese one-shot in the prompt to enforce strict Yes/No outputs and reduce parsing noise.
   - **Action**: Update the prompt construction in `chat/run_kimi_gj.py` to include an example (keep English variable names and comments). Example (Chinese one-shot by design):
     ```
     任務：你需要判斷給定三元組陳述是否為事實正確。請僅輸出 Yes 或 No。
     範例：
     問題：這是真的嗎：曹雪芹 創作 紅樓夢？
     答案：Yes
     問題：這是真的嗎：馬克·祖克柏 創作 紅樓夢？
     答案：No
     現在的問題：這是真的嗎：{S P O}
     答案：
     ```
   - **Parsing**: Accept only `^yes$|^no$` (case-insensitive). Treat others as format anomalies for later cleanup.

4. **(❌ abandon-unnecessary) Evaluation Metrics**  
   - **Abandon Reason**: The function repeative of `run_kimi_triple_v2.py`.
   - **Classification (New)**: Add `graph_evaluation/metrics/metrics.py` to compute Accuracy/Precision/Recall/F1 and write `GraphJudge_report.json`.
   - **Inputs**:
     - Model output: `datasets/.../Graph_Iteration1/pred_instructions_context_kimi_itr1.csv` (columns: `prompt,generated`)
     - Gold labels: `datasets/.../Graph_Iteration1/gold_bootstrap.csv` (with `expected`)
   - **CLI**:
     ```bash
     python graph_evaluation/metrics/metrics.py \
       --pred-csv ../datasets/.../Graph_Iteration1/pred_instructions_context_kimi_itr1.csv \
       --gold-csv ../datasets/.../Graph_Iteration1/gold_bootstrap.csv \
       --output   ../datasets/.../Graph_Iteration1/GraphJudge_report.json
     ```
   - **Graph-level (Existing)**: Keep using `graph_evaluation/metrics/eval.py` to produce Triple/Graph/G-BLEU/ROUGE/BertScore:
     ```bash
     python graph_evaluation/metrics/eval.py \
       --pred_file ../datasets/.../Graph_Iteration1/test_generated_graphs_final.txt \
       --gold_file ../datasets/.../test.source
     ```

5. ✅ **External RAG Verification（Optional）**  
   - **Trigger**: `auto_expected=True` or fuzzy ≥ 0.9 while the model says **No**.
   - **Action**: Query Gemini Ground Search/Wikipedia/Perplexity. If retrieved snippets contain both subject and object and the context aligns with predicate synonyms, flag for human review.
   - **Output**: `datasets/.../Graph_Iteration1/rag_flags.tsv` (columns: `subject,predicate,object,fuzzy_score,evidence_url,snippet,status`)
   - **CLI**:
     ```bash
     python tools/rag_verify.py \
       --pred-csv  ./datasets/.../Graph_Iteration1/pred_instructions_context_kimi_itr1.csv \
       --gold-csv  ./datasets/.../Graph_Iteration1/gold_bootstrap.csv \
       --output    ./datasets/.../Graph_Iteration1/rag_flags.tsv
     ```

_Example target_: verify `作者 創作 石頭記` should be **Yes** – add to gold labels and track if model wrong.

---
## 4  Pipeline Orchestration & Logging
1. **Makefile / CLI `run_all.sh`** – sequentially `entity → triple → judge` with auto-resume.
2. **Rich Progress + Time Stats** – move verbose CLI logs into `logs/iteration_2/*.log` to declutter md.
3. **Error Alerts** – send Slack/Email if API retry >3 or accuracy < 70 %.

---
## 5  Timeline & Deliverables
| Week | Deliverable |
|------|-------------|
| 1    | Cleaner scripts + ECTD re-run (entity_typed) |
| 2    | Relation map + Triple post-processor; regenerate triples |
| 3    | Instruction builder with full coverage; gold label sheet (200 rows) |
| 4    | Graph Judge re-run; metrics report ≥70 % acc |
| 5    | Integrate Makefile, caching, logging; write Iteration2_Report.md |

---
## 6  Risk & Mitigation
1. **LLM RPM limits** – continue batching & cache; consider paid tier if wall-clock >24 h.
2. **Label Quality** – double review 10 % sample, use majority vote.
3. **Schema Drift** – unit tests (`tests/validate_json.py`) fail CI when output deviates.

---
### Appendix A – Directory Additions
```
Miscellaneous/KgGen/GraphJudge/
  tools/
    clean_entities.py
    parse_kimi_triples.py
    build_instructions.py
  config/
    relation_map.json
  logs/
    iteration_2/
```

**End of Plan**

