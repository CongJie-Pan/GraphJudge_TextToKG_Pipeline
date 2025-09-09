# Entity Extraction & Text Denoising (ECTD) Implementation Guide  
*Dream of the Red Chamber – Chapter 1*

---

## Table of Contents
1. Introduction  
2. Directory Layout  
3. Prerequisites  
4. Data Preparation – Chapter 1  
5. Configuration  
6. Running the ECTD Pipeline  
7. Inspecting & Interpreting the Output  
8. Iterative Refinement  
9. Troubleshooting  
10. Next Steps  

---

## 1. Introduction
The **Entity Extraction & Text Denoising (ECTD)** stage converts raw narrative text into:  
* a cleaned (denoised) version of the text, and  
* a consolidated list of entities that appear in the text.

These two artefacts become the inputs for the next stage (**Semantic Graph Generation**).  
This guide walks you through executing **ECTD** for *Chapter 1* of *Dream of the Red Chamber* (《紅樓夢》) using the **GraphJudge** repository.

> **Audience**: Engineers new to NLP fine-tuning and Azure services – step-by-step, copy-paste-ready.

---

## 2. Directory Layout
```
project_root/
└─ Miscellaneous/
   └─ KgGen/
      └─ GraphJudge/
         ├─ chat/
         │  ├─ run_chatgpt_entity.py     # ECTD script for gpt-4o-mini model
            ├─ run_entity           # ECTD script for KIMI-K2 model
         │  └─ config.py                # API key helper
         ├─ datasets/
         │  └─ HongLouMeng/
         │     ├─ chapter1_raw.txt      # <-- we will create this
         │     └─ ...                   
         ├─ docs/
         │  └─ ECTD_Implementation_guide.md   # <-- (this file)
         └─ ...
```

---

## 3. Prerequisites
1. **Python ≥ 3.10** (conda or venv recommended)
2. **OpenAI API Key** *or* **Azure OpenAI** credentials  
   * Set environment variables **before running**:
   ```bash
   # Standard OpenAI
   setx OPENAI_API_KEY    "sk-your-key"
   setx OPENAI_API_BASE   "https://api.openai.com/v1"  # optional

   # Azure OpenAI (preferred if your institution provides credits)
   setx AZURE_OPENAI_KEY       "<your-azure-key>"
   setx AZURE_OPENAI_ENDPOINT  "https://<your-resource>.openai.azure.com/"
   ```
3. Install dependencies (only lightweight packages – no CUDA required):
   ```bash
   cd Miscellaneous/KgGen/GraphJudge
   pip install -r chat/requirements.txt  # transformers, tiktoken,  etc.
   ```

---

## 4. Data Preparation – **Chapter 1**
### 4.1 Acquire the Text (UTF-8)
1. Obtain the Chinese source text for Chapter 1.  
   *Public domain sources include Project Gutenberg, Chinese Text Project, etc.*
2. Paste **only Chapter 1** into a new file:
   `Miscellaneous/KgGen/GraphJudge/datasets/DreamOf_RedChamber/chapter1_raw.txt`

### 4.2 Segment into Passages (Recommended)
OpenAI models handle ≈ 8K tokens comfortably.  
If Chapter 1 is very long, split it into *paragraph-level* passages (~500 – 800 characters each).

Below is a **utility script** to auto-split by `\n\n` blank lines. Save as `split_chapter1.py` *(optional)*:
```python
"""Split Dream of the Red Chamber – Chapter 1 into manageable chunks."""
from pathlib import Path

INPUT  = Path('datasets/DreamOf_RedChamber/chapter1_raw.txt')
OUTPUT = Path('datasets/DreamOf_RedChamber/ch1_passages.txt')

chunks = [p.strip() for p in INPUT.read_text(encoding='utf-8').split('\n\n') if p.strip()]

with OUTPUT.open('w', encoding='utf-8') as f:
    for idx, chunk in enumerate(chunks, 1):
        # Write one JSON line per passage for the ECTD script.
        f.write(f"{{\"id\": {idx}, \"text\": \"{chunk}\"}}\n")
print(f"Wrote {len(chunks)} passages → {OUTPUT}")
```
> `run_chatgpt_entity.py` can ingest **either** a plain-text file (one passage per line) **or** the JSON-Lines format above. Choose whichever is easier for you.

---

## 5. Configuration
Open **`chat/run_chatgpt_entity.py`** and confirm/adjust these variables:

```python
# 35-46 lines (approx.)
dataset   = "HongLouMeng"    # ← custom dataset folder name within ./datasets
Iteration = 1                 # Start with 1; increment for each refinement round

# 131+ lines: prompt examples – ensure examples are CHINESE and domain-specific
```

### Why Update the Prompt Examples?
Providing *in-domain* Chinese examples greatly boosts extraction accuracy.  
**Example snippet** already in the repo:
```python
Example#1:
Text: "賈寶玉因夢遊太虛幻境，頓生疑懼。"
List of entities: ["賈寶玉", "太虛幻境"]
```
Add **one extra example** from Chapter 1 to further anchor the model.

---

## 6. Running the ECTD Pipeline
### 6.1 Quick-Start Command

**gpt-4o-mini Model**
```bash
python chat/run_chatgpt_entity.py \
  --input_file   datasets/DreamOf_RedChamber/chapter1_raw.txt \
  --model        gpt-4o-mini \
  --temperature  0.2 \
  --max_tokens   1024
```

**KIMI-K2 Model**
```bash
python chat/run_entity.py \
  --input_file   datasets/DreamOf_RedChamber/chapter1_raw.txt \
  --model        moonshot/kimi-k2-0711-preview \
  --temperature  0.2 \
  --max_tokens   1024
```


Arguments explained:
* **`--input_file`** – path to prepared Chapter 1 passages  
* **`--model`** – recommended `moonshot/kimi-k2-0711-preview` for higher Chinese reasoning accuracy  
  *(legacy example: `gpt-4o-mini-2024-xx` still works as a fallback)*
* **`--temperature`** – 0.2 for deterministic extraction  
* **`--max_tokens`** – response budget per passage

> **Tip**: If you hit Azure rate-limits, add `--concurrency 3` to reduce simultaneous calls.

### 6.2 Behind the Scenes
The script performs **two API calls** per passage:
1. **`extract_entities(prompt)`**  → returns *entity JSON*  
2. **`denoise_text(prompt)`**      → returns *cleaned passage*

All results are cached under:
```
datasets/GPT4o_mini_result_HongLouMeng/Iteration1/
  ├─ entities.jsonl      # One JSON object per passage {id, entities:[..]}
  └─ denoised.txt        # One cleaned passage per line
```

---

## 7. Inspecting & Interpreting the Output
1. **entities.jsonl** – Open it in VS Code or any JSON viewer.
   * Confirm key characters: 賈寶玉、甄士隱、賈母… are correctly extracted.
2. **denoised.txt** – Validate that archaic phrases are kept but noise (chapter headers, irrelevant commentary) is removed.

> **Quality Check**:  
> – Entity list size ≈ 3 – 8 per passage is typical.  
> – If you see empty lists, raise `temperature` slightly to 0.5 or tweak the examples.

---

## 8. Iterative Refinement
ECTD is **iterative** by design:
1. Manually skim *denoised.txt*; remove obvious errors or merge sentences if needed.
2. **Increment `Iteration` → 2** in `run_chatgpt_entity.py`.
3. Set **`--input_file`** to **`denoised.txt`** from Iteration 1.
4. Re-run the script – you will obtain **Iteration2** results with improved consistency.

Repeat until *entity precision & recall* plateau (usually ≤ 3 rounds).

---

## 9. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `OpenAIAuthError` | Env. variables typo | Re-export keys, restart shell |
| `RateLimitError: 429` | Too many parallel calls | Lower `--concurrency`, add exponential backoff |
| Garbled Chinese  | Wrong file encoding | Ensure **UTF-8** when saving raw text |
| Empty entity lists | Prompt examples not Chinese-specific | Add at least *two* in-domain examples |

---

## 10. Next Steps
* Proceed to **Semantic Graph Generation** (`chat/run_chatgpt_triple.py`) using `entities.jsonl` + `denoised.txt` as inputs.
* Store all artefacts in version control or Azure Blob Storage for reproducibility.
* Once Chapters 1-5 are done, batch them and move to **KASFT fine-tuning** (see *KASFT_Implementation_guide.md*).

---

### Appendix A – Minimal End-to-End Example
```python
"""Run ECTD for a *single* paragraph (debugging helper)."""
import json, pathlib, subprocess

sample = "林黛玉隨母身故後，寄居在榮國府。"
out   = pathlib.Path('datasets/DreamOf_RedChamber/tmp.txt')

a = out.open('w', encoding='utf-8'); a.write(sample + '\n'); a.close()

subprocess.run([
    'python', 'chat/run_chatgpt_entity.py',
    '--input_file', str(out),
    '--model', 'moonshot/kimi-k2-0711-preview',  # gpt-4o-mini can also be used as an alternative
    '--temperature', '0.3'
])

print("Entities & denoised text saved under ./datasets/GPT4o_mini_result_HongLouMeng/Iteration1/")
```

---

*Happy Knowledge-Graph Building!*