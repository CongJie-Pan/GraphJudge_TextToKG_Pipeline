# Graph Judge Implementation Guide with KIMI-K2 LiteLLM

## Overview

This guide provides detailed instructions for implementing Graph Judge (GJ) functionality using KIMI-K2 API through LiteLLM (Moonshot AI), directly integrating with the existing GraphJudge infrastructure. This approach bypasses the Knowledge-Augmented Structured Fine-Tuning (KASFT) phase while maintaining compatibility with existing evaluation pipelines.

### Architecture Overview

```
ECTD Data (Ready) → Triple Generation (prepare_KGCom.ipynb) → Graph Judge (KIMI-K2 + LiteLLM) → Existing Evaluation Pipeline
```

**What we're leveraging:** Existing chat scripts, data processing pipelines, and evaluation infrastructure.

**What we're replacing:** The fine-tuned LoRA/BERT models with KIMI-K2 API calls while maintaining output format compatibility.

**What we're integrating with:** The existing `prepare_KGCom.ipynb` workflow, chat script patterns, and evaluation metrics.

## Prerequisites

### 1. Data Requirements
- ECTD processed data must be available in: `Miscellaneous/KgGen/GraphJudge/datasets/GPT4o_mini_result_DreamOf_RedChamber/`
- Integration with existing `prepare_KGCom.ipynb` for triple generation and filtering
- Required files structure:
  ```
  GPT4o_mini_result_DreamOf_RedChamber/
  ├── Iteration1/
  │   ├── test_denoised.target  (denoised text data)
  │   └── test_entity.txt       (extracted entities)
  └── Graph_Iteration1/
      ├── test_generated_graphs.txt  (generated from prepare_KGCom.ipynb)
      └── test_instructions_context_llama2_7b.json  (instruction format)
  ```

### 2. API Requirements
- Moonshot AI API key for KIMI-K2 access
- LiteLLM library installation

### 3. Environment Setup
```bash
# Install required packages for KIMI-K2 integration
pip install litellm python-dotenv datasets pandas

# Install additional packages for comprehensive evaluation
pip install rouge-score bert-score nltk spacy scikit-learn networkx

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Install the huggingface-hub dataset to facilitate access to various datasets hosted on Hugging Face.
# This package is essential for integrating with the Hugging Face ecosystem, allowing for easy dataset loading and manipulation.
pip install datasets huggingface-hub

# Download spaCy models for both English and Chinese
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

## Implementation Steps

### Step 1: Environment Configuration

#### 1.1 Create Environment File
Create a `.env` file in `Miscellaneous/KgGen/GraphJudge/chat/` directory:

```bash
# Moonshot AI API Configuration for KIMI-K2
MOONSHOT_API_KEY=your_moonshot_api_key_here

# Optional: Moonshot API Base URL (defaults to global endpoint)
MOONSHOT_API_BASE=https://api.moonshot.ai/v1

# Optional: OpenAI API (for comparison testing)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

#### 1.2 Update Configuration Module
Modify `chat/config.py` to support Moonshot AI API:

```python
def get_moonshot_api_config() -> str:
    """
    Retrieve Moonshot AI API configuration for KIMI-K2 model.
    
    Returns:
        str: Moonshot API key
        
    Raises:
        ValueError: If MOONSHOT_API_KEY is not set
    """
    # Load from .env file first
    load_env_file()
    
    # Get Moonshot API key
    api_key = os.getenv('MOONSHOT_API_KEY', '').strip()
    
    if not api_key:
        raise ValueError(
            "Moonshot API key not found. Please set MOONSHOT_API_KEY in environment variables or .env file"
        )
    
    return api_key
```

### Step 2: Modify Existing Chat Script for KIMI-K2 Integration

#### 2.1 Create Modified Graph Judge Script
Create `chat/run_kimi_gj.py` based on existing `run_chatgpt_gj.py` patterns:

```python
"""
KIMI-K2-based Graph Judge Implementation using LiteLLM

This script implements the Graph Judge functionality using KIMI-K2 model through LiteLLM,
following the existing chat script patterns from run_chatgpt_gj.py.

Based on the existing GraphJudge infrastructure:
- Uses the same JSON input format as existing scripts
- Follows the async pattern from run_chatgpt_gj.py
- Outputs in the same CSV format for compatibility
- Integrates with existing evaluation pipeline
"""

import os
import asyncio
import json
import csv
import pandas as pd
from tqdm.asyncio import tqdm
from datasets import load_dataset
from litellm import completion
from config import get_moonshot_api_config

# Dataset configuration following existing patterns
folder = "GPT4o_mini_result_DreamOf_RedChamber"
input_file = f"./datasets/{folder}/test_instructions_context_llama2_7b.json"
output_file = f"./datasets/{folder}/pred_instructions_context_kimi_itr1.csv"

# API configuration
api_key = get_moonshot_api_config()
os.environ['MOONSHOT_API_KEY'] = api_key

# Load the evaluation dataset following existing patterns
total_input = load_dataset("json", data_files=input_file)
data_eval = total_input["train"].train_test_split(
    test_size=499, shuffle=True, seed=42
)["test"]

# Load instructions data directly
with open(input_file, "r", encoding="utf-8") as f:
    instructions = json.load(f)

async def get_kimi_completion(instruction, input_text=None):
    """
    Send a prompt to KIMI-K2 and get the generated response.
    Based on get_chatgpt_completion pattern from run_chatgpt_gj.py
    
    Args:
        instruction (str): The instruction/question for classification
        input_text (str, optional): Additional context (if any)
    
    Returns:
        str: The generated response from KIMI-K2
    """
    # Use the advanced prompt format from lora_infer_batch.py
    prompt = f"""Goal:
You need to do the graph judgement task, which means you need to clarify
 the correctness of the given triple.
Attention:
1.The correct triple sentence should have a correct grammatical structure.
2.The knowledge included in the triple sentence should not conflict with
the knowledge you have learned.
3.The answer should be either "Yes, it is true." or "No, it is not true."

Here are two examples:
Example#1:
Question: Is this true: Apple Founded by Mark Zuckerberg ?
Answer: No, it is not true.
Example#2:
Question: Is this true: Mark Zuckerberg Founded Facebook ?
Answer: Yes, it is true.

Refer to the examples and here is the question:
Question: {instruction}
Answer:"""
    
    # Retry loop following existing pattern
    while True:
        try:
            response = completion(
                model="moonshot/kimi-k2-0711-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistent judgment
                max_tokens=200    # Shorter responses for binary classification
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

async def process_instructions():
    """
    Process instructions following the existing async pattern from run_chatgpt_gj.py
    """
    # Create async tasks for all instruction-input pairs
    tasks = []
    for item in data_eval:
        instruction = item["instruction"]
        input_text = item.get("input", "")  # Handle optional input
        tasks.append(get_kimi_completion(instruction, input_text))

    # Execute all tasks concurrently with progress tracking
    responses = await tqdm.gather(*tasks, desc="Processing with KIMI-K2")

    # Write responses to CSV file compatible with existing pipeline
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt", "generated"])  # Standard header format

        for item, response in zip(data_eval, responses):
            prompt = item["instruction"]
            # Clean response and ensure it matches expected format
            cleaned_response = response.strip().replace('\n', ' ')
            writer.writerow([prompt, cleaned_response])

if __name__ == "__main__":
    asyncio.run(process_instructions())
    print(f"✅ KIMI-K2 Graph Judge processing completed!")
    print(f"📊 Results saved to: {output_file}")
```

### Step 3: Integration with Existing Triple Generation

Instead of creating new scripts, we leverage the existing `prepare_KGCom.ipynb` workflow for triple generation and instruction formatting. Modify the second cell of `datasets/prepare_KGCom.ipynb`:

```python
"""
Generate test data for KIMI-K2 Graph Judge integration
Modify the second cell of prepare_KGCom.ipynb to generate JSON instructions
"""

import os
import json
import ast
import pandas as pd
from typing import List, Dict

# Configuration for Dream of Red Chamber dataset
dataset_path = './GPT4o_mini_result_DreamOf_RedChamber/'

# Load the generated triples from the existing pipeline
# This assumes you've already run the first part of prepare_KGCom.ipynb
# and have test_generated_graphs.txt from previous steps

generated_graphs_file = dataset_path + 'Graph_Iteration1/test_generated_graphs.txt'
output_json_file = dataset_path + 'test_instructions_context_llama2_7b.json'

# Read generated triples
triples = []
try:
    with open(generated_graphs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
except FileNotFoundError:
    print(f"Warning: {generated_graphs_file} not found. Please run the first part of prepare_KGCom.ipynb first.")
    triples = []

# Generate instruction format compatible with existing chat scripts
instructions_data = []
for i, (subj, pred, obj) in enumerate(triples):
    # Create instruction in the format expected by chat scripts
    instruction = f"Is this true: {subj} {pred} {obj}?"
    
    instructions_data.append({
        "instruction": instruction,
        "input": "",  # No additional input context for basic triples
        "output": ""  # Will be filled by KIMI-K2
    })

# Save in JSON format compatible with existing scripts
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(instructions_data, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(instructions_data)} instruction entries")
print(f"📁 Saved to: {output_json_file}")
print("📋 Now you can run the KIMI-K2 graph judge script")
```

### Step 4: Execution Workflow

#### 4.1 Integrated Workflow with Existing Infrastructure

Follow this step-by-step workflow that integrates with existing GraphJudge infrastructure:

**Step 4.1.1: Prepare Data using existing notebook**
```bash
# Navigate to the datasets directory
cd Miscellaneous/KgGen/GraphJudge/datasets/

# Run the first cell of prepare_KGCom.ipynb to generate initial triples
jupyter notebook prepare_KGCom.ipynb
# Execute first cell to generate test_generated_graphs.txt

# Run the modified second cell (from Step 3) to generate JSON instructions
# This creates test_instructions_context_llama2_7b.json
```

**Step 4.1.2: Execute KIMI-K2 Graph Judge**
```bash
# Navigate to chat directory
cd ../chat/

# Run the KIMI-K2 graph judge script
python run_kimi_gj.py

# This will process the JSON instructions and output CSV results
```

**Step 4.1.3: Filter Results and Generate Final Graph File**
```bash
# After KIMI-K2 processing, filter the results to generate final graph file
cd ../datasets/
jupyter notebook prepare_KGCom.ipynb
# Run the third cell to filter positive judgments and generate final graph file
```

**Step 4.1.4: Run Comprehensive Graph Evaluation**
```bash
# Navigate to evaluation directory
cd ../graph_evaluation/

# Update eval.sh paths if needed, then run evaluation
bash eval.sh

# Or run directly with custom paths:
python metrics/eval.py \
    --pred_file ../datasets/GPT4o_mini_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs_final.txt \
    --gold_file ../datasets/GPT4o_mini_result_DreamOf_RedChamber/test.source
```

### Step 5: Comprehensive Graph Evaluation System

#### 5.1 Understanding the Evaluation Pipeline

The GraphJudge framework includes a sophisticated evaluation system in `graph_evaluation/` that provides multiple metrics to assess graph quality:

**Available Evaluation Metrics:**
1. **Triple Match F1** - Exact matching of individual triples (precision/recall/F1)
2. **Graph Match Accuracy** - Structural graph isomorphism evaluation  
3. **G-BLEU/G-ROUGE** - Text similarity metrics adapted for graph edges
4. **G-BertScore** - Semantic similarity using BERT for graph edge comparison
5. **Graph Edit Distance (GED)** - Minimum edit operations needed (optional)

#### 5.2 Preparing Data for Evaluation

The evaluation system expects specific input formats:

**Required Files:**
- **Predicted Graph File** (`test_generated_graphs_final.txt`) - Final filtered triples in list format
- **Gold Standard File** (`test.source`) - Reference graphs in the same format

#### 5.3 Converting KIMI-K2 Results to Evaluation Format

Modify the third cell of `prepare_KGCom.ipynb` to generate the correct evaluation format:

```python
"""
Convert KIMI-K2 judgment results to evaluation-ready graph format
"""

import pandas as pd
import ast

# Load KIMI-K2 judgment results
results_file = './GPT4o_mini_result_DreamOf_RedChamber/pred_instructions_context_kimi_itr1.csv'
results_df = pd.read_csv(results_file)

# Load original generated triples
triples_file = './GPT4o_mini_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs.txt'
with open(triples_file, 'r', encoding='utf-8') as f:
    original_triples = [line.strip().split('\t') for line in f if line.strip()]

# Filter triples based on KIMI-K2 judgment
filtered_triples = []
for i, (_, row) in enumerate(results_df.iterrows()):
    if i < len(original_triples):
        decision = row['generated']
        # Keep triples that are judged as true
        if any(keyword in decision.lower() for keyword in ['yes', 'true', '是']):
            filtered_triples.append(original_triples[i])

# Convert to evaluation format (list of lists)
evaluation_graphs = [filtered_triples]  # Single graph for this example

# Save in format expected by evaluation system
output_file = './GPT4o_mini_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs_final.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for graph in evaluation_graphs:
        f.write(str(graph) + '\n')

print(f"✅ Filtered {len(original_triples)} triples down to {len(filtered_triples)} high-quality triples")
print(f"📊 Retention rate: {len(filtered_triples)/len(original_triples)*100:.1f}%")
print(f"📁 Evaluation-ready file saved to: {output_file}")
```

#### 5.4 Running the Evaluation

**Method 1: Using the provided script**
```bash
cd Miscellaneous/KgGen/GraphJudge/graph_evaluation/
bash eval.sh
```

**Method 2: Direct execution with custom paths**
```bash
python metrics/eval.py \
    --pred_file ../datasets/GPT4o_mini_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs_final.txt \
    --gold_file ../datasets/GPT4o_mini_result_DreamOf_RedChamber/test.source
```

#### 5.5 Understanding Evaluation Output

The evaluation system outputs comprehensive metrics:

```
Triple Match F1 Score: 0.8234
Graph Match F1 Score: 0.7456

G-BLEU Precision: 0.7891
G-BLEU Recall: 0.8123
G-BLEU F1: 0.8005

G-Rouge Precision: 0.7765
G-Rouge Recall Score: 0.8234
G-Rouge F1 Score: 0.7993

G-BertScore Precision Score: 0.8456
G-BertScore Recall Score: 0.8234
G-BertScore F1 Score: 0.8344
```

**Metric Interpretation:**
- **Higher scores** indicate better graph quality
- **Triple Match F1** measures exact correctness
- **G-BLEU/G-ROUGE** measure text-level similarity
- **G-BertScore** captures semantic similarity
- **Graph Match** evaluates structural equivalence

## Usage Instructions

### Quick Start

1. **Set up environment:**
   ```bash
   cd Miscellaneous/KgGen/GraphJudge/chat/
   cp env_example.txt .env
   # Edit .env file with your MOONSHOT_API_KEY
   ```

2. **Install dependencies:**
   ```bash
   # Core dependencies for KIMI-K2 integration
   pip install litellm python-dotenv datasets pandas
   
   # Additional dependencies for comprehensive evaluation
   pip install rouge-score bert-score nltk spacy scikit-learn networkx
   
   # Download required language models
   python -c "import nltk; nltk.download('punkt')"
   python -m spacy download en_core_web_sm
   ```

3. **Execute the integrated workflow:**
   ```bash
   # Step 1: Generate instruction data using existing notebook
   cd ../datasets/
   jupyter notebook prepare_KGCom.ipynb
   # Run modified second cell to generate JSON instructions
   
   # Step 2: Run KIMI-K2 Graph Judge
   cd ../chat/
   python run_kimi_gj.py
   
   # Step 3: Filter results and prepare for evaluation
   cd ../datasets/
   jupyter notebook prepare_KGCom.ipynb
   # Run modified third cell to filter results and generate final graph file
   
   # Step 4: Run comprehensive evaluation
   cd ../graph_evaluation/
   bash eval.sh
   ```

### Integration with Existing Workflow

**Key advantage:** This approach integrates seamlessly with existing GraphJudge infrastructure:

1. **Data Processing:** Uses existing `prepare_KGCom.ipynb` workflow
2. **Script Patterns:** Follows existing chat script patterns (`run_chatgpt_gj.py`)
3. **Output Format:** Generates CSV files compatible with existing evaluation
4. **Evaluation Pipeline:** No changes needed to existing evaluation scripts

## Configuration Options

### Model Selection
You can experiment with different Moonshot AI models by modifying the model parameter in `run_kimi_gj.py`:

```python
# In the get_kimi_completion function
response = completion(
    model="moonshot/kimi-k2-0711-preview",        # Latest KIMI model with enhanced reasoning  
    # model="moonshot/moonshot-v1-128k",  # Large context window
    # model="moonshot/moonshot-v1-32k",   # Medium context window  
    # model="moonshot/moonshot-v1-8k",    # Faster and more cost-effective
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    max_tokens=200
)
```

### Prompt Customization
You can modify the prompt template in the `get_kimi_completion` function to better suit different domains or add domain-specific examples for classical Chinese literature.

### Performance Tuning
- **Temperature:** Lower values (0.1-0.3) for more consistent binary classification
- **Max Tokens:** 100-300 tokens usually sufficient for binary responses
- **Batch Size:** Controlled by the data loading (499 samples by default, following existing patterns)

## Expected Output Structure

After running the complete integrated pipeline, you should have:

```
datasets/GPT4o_mini_result_DreamOf_RedChamber/
├── Iteration1/
│   ├── test_denoised.target                    # Original ECTD data
│   └── test_entity.txt                         # Original ECTD entities
├── Graph_Iteration1/
│   ├── test_generated_graphs.txt               # Generated from prepare_KGCom.ipynb
│   └── test_generated_graphs_final.txt         # Filtered graphs ready for evaluation
├── test_instructions_context_llama2_7b.json   # JSON instructions for KIMI-K2
├── pred_instructions_context_kimi_itr1.csv    # KIMI-K2 judgment results (compatible format)
└── test.source                                 # Gold standard reference file (if available)
```

**Key Files:**
- **Input:** `test_instructions_context_llama2_7b.json` - Instruction format compatible with chat scripts
- **Judgment Output:** `pred_instructions_context_kimi_itr1.csv` - KIMI-K2 binary judgments
- **Evaluation Input:** `test_generated_graphs_final.txt` - Filtered graphs in evaluation format
- **Evaluation Script:** `graph_evaluation/eval.sh` - Comprehensive evaluation metrics
- **Compatibility:** All outputs work directly with existing evaluation infrastructure

**Evaluation Results:** The evaluation system generates comprehensive metrics including Triple Match F1, Graph Match Accuracy, G-BLEU, G-ROUGE, and G-BertScore for thorough assessment of graph quality.

## Troubleshooting

### Common Issues

1. **API Key Issues:**
   - Ensure MOONSHOT_API_KEY is correctly set
   - Verify API key has sufficient credits
   - Check API key permissions for KIMI and Moonshot models

2. **Missing Dependencies:**
   ```bash
   pip install --upgrade litellm python-dotenv
   ```

3. **Data Format Issues:**
   - Verify ECTD data is in the expected format
   - Check file paths are correct
   - Ensure UTF-8 encoding for all text files

4. **Performance Issues:**
   - Reduce batch size for large datasets
   - Adjust temperature and max_tokens parameters
   - Consider using moonshot-v1-8k instead of kimi-k2 for faster processing

5. **Evaluation Issues:**
   - Ensure `test.source` gold standard file exists for evaluation
   - Verify `test_generated_graphs_final.txt` contains valid Python list format
   - Check that filtered triples are in the correct format: `[['subject', 'predicate', 'object'], ...]`
   - Install evaluation dependencies: `pip install rouge-score bert-score nltk spacy scikit-learn networkx`

### Debug Mode
Add debug logging by setting environment variable:
```bash
export LITELLM_LOG=DEBUG
python run_kimi_gj.py
```

## Performance Considerations

### Cost Optimization
- Use `moonshot-v1-8k` for most tasks (cheaper)
- Use `kimi-k2` only for complex judgment tasks requiring advanced reasoning
- Implement caching for repeated judgments
- Batch processing to reduce API overhead

### Speed Optimization
- Adjust async batch size based on API rate limits
- Use lower temperature for consistent but faster responses
- Implement retry logic with exponential backoff

## Future Enhancements

1. **Advanced Triple Generation:** Implement more sophisticated NLP techniques for triple extraction
2. **Confidence Calibration:** Fine-tune confidence scoring based on evaluation feedback
3. **Multi-Model Ensemble:** Combine multiple models for better judgment accuracy
4. **Incremental Processing:** Support for processing large datasets in chunks
5. **Performance Monitoring:** Add metrics collection and visualization

## Conclusion

This revised implementation provides a **realistic and practical approach** to integrating KIMI-K2 as a Graph Judge replacement while **leveraging existing GraphJudge infrastructure**. 

### Key Benefits:

1. **Seamless Integration:** Works with existing `prepare_KGCom.ipynb`, chat scripts, and evaluation pipeline
2. **Minimal Changes:** Requires only one new script and minor notebook modifications
3. **Format Compatibility:** Outputs are compatible with existing evaluation infrastructure
4. **Proven Patterns:** Follows established patterns from `run_chatgpt_gj.py` and other existing scripts
5. **No Infrastructure Changes:** Reuses existing data processing and evaluation workflows

### Technical Advantages:

- **No Fine-tuning Required:** Leverages KIMI-K2's pre-trained reasoning capabilities
- **Cost-effective:** API-based approach eliminates training infrastructure needs
- **Scalable:** Easy to extend to other datasets and domains
- **Maintainable:** Follows existing codebase patterns and conventions

This approach demonstrates how external LLM APIs can be integrated into existing research pipelines while maintaining compatibility and minimizing implementation overhead. The pattern established here can be applied to integrate other advanced LLMs into the GraphJudge framework.