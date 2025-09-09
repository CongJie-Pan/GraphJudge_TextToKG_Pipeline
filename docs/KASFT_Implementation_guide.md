# KASFT Implementation Guide: Fine-Tuning on Azure with Cost Analysis

This comprehensive guide walks you through implementing the Knowledge Graph Supervised Fine-Tuning (KASFT) component of the GraphJudge project on Azure, including detailed cost calculations and step-by-step instructions for beginners.

## Table of Contents
1. [Overview](#overview)
2. [Cost Analysis](#cost-analysis)
3. [Azure Setup Options](#azure-setup-options)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Code Examples](#code-examples)
6. [Troubleshooting](#troubleshooting)
7. [Cost Optimization](#cost-optimization)

## Overview

KASFT (Knowledge Graph Supervised Fine-Tuning) is the most computationally intensive part of the GraphJudge pipeline. It involves:

1. **Training Data Generation**: Creating graph judgment training datasets
2. **LoRA Fine-Tuning**: Adapting LLaMA models for graph evaluation
3. **Inference**: Using fine-tuned models to judge and filter knowledge graphs

### Memory Requirements
- **Base Model (LLaMA-3-8B)**: ~16 GB VRAM (FP16)
- **LoRA Training**: Additional ~8-12 GB VRAM
- **System RAM**: 32-64 GB recommended
- **Total VRAM Needed**: 24-28 GB for comfortable training

## Cost Analysis

### Current Exchange Rate (2024)
- **USD to NTD**: ~32.5 (as of late 2024)

### Azure GPU VM Pricing (East Asia Region)

#### Option 1: Standard_NV12ads_A10_v5 (Recommended)
- **Specs**: 12 vCPUs, 110 GB RAM, 1x NVIDIA A10 (24GB VRAM)
- **Cost**: $0.908 USD/hour
- **NTD Cost**: ~NT$29.5/hour

#### Option 2: Standard_NV6ads_A10_v5 (Budget Option)
- **Specs**: 6 vCPUs, 55 GB RAM, 1x NVIDIA A10 (24GB VRAM)
- **Cost**: $0.454 USD/hour  
- **NTD Cost**: ~NT$14.8/hour

#### Option 3: Standard_NC40ads_H100_v5 (High Performance)
- **Specs**: 40 vCPUs, 320 GB RAM, 1x NVIDIA H100 (94GB VRAM)
- **Cost**: $6.98 USD/hour
- **NTD Cost**: ~NT$227/hour

### Estimated Project Costs (NTD)

Based on a typical "Dream of the Red Chamber" knowledge graph project:

#### Small Dataset (~10K triples)
| Task | Duration | VM Type | Cost (NTD) |
|------|----------|---------|------------|
| Data Preparation | 1 hour | NV6ads_A10_v5 | NT$15 |
| LoRA Fine-tuning | 3-4 hours | NV12ads_A10_v5 | NT$120 |
| Inference | 1 hour | NV6ads_A10_v5 | NT$15 |
| **Total** | **6 hours** | **Mixed** | **~NT$150** |

#### Medium Dataset (~50K triples)
| Task | Duration | VM Type | Cost (NTD) |
|------|----------|---------|------------|
| Data Preparation | 2 hours | NV12ads_A10_v5 | NT$60 |
| LoRA Fine-tuning | 6-8 hours | NV12ads_A10_v5 | NT$240 |
| Inference | 2-3 hours | NV12ads_A10_v5 | NT$90 |
| **Total** | **11 hours** | **NV12ads_A10_v5** | **~NT$390** |

#### Large Dataset (~100K+ triples)
| Task | Duration | VM Type | Cost (NTD) |
|------|----------|---------|------------|
| Data Preparation | 3 hours | NV12ads_A10_v5 | NT$90 |
| LoRA Fine-tuning | 12-16 hours | NC40ads_H100_v5 | NT$3,600 |
| Inference | 4-6 hours | NV12ads_A10_v5 | NT$180 |
| **Total** | **20 hours** | **Mixed** | **~NT$3,870** |

### Cost Optimization Strategies

1. **Use Spot Instances**: Save 60-90% with `Standard_NV12ads_A10_v5` spot pricing (~NT$4.4/hour)
2. **4-bit Quantization**: Reduce VRAM usage by 50%, enable smaller VMs
3. **Gradient Accumulation**: Use smaller batch sizes with accumulation
4. **Mixed Precision**: FP16 training reduces memory usage
5. **Scheduled Training**: Run during off-peak hours for potential discounts

## Azure Setup Options

### Option 1: Azure Virtual Machines (Recommended for Beginners)

#### 1.1 Create GPU VM via Azure Portal

1. **Login to Azure Portal**: https://portal.azure.com
2. **Create Virtual Machine**:
   ```
   Resource Group: rg-kg-project
   VM Name: vm-kg-gpu
   Region: East Asia (lowest cost)
   Image: Ubuntu 22.04 LTS
   Size: Standard_NV12ads_A10_v5
   Authentication: SSH public key
   ```

3. **Configure Networking**:
   ```
   Public IP: Yes
   SSH (22): Allow
   HTTP (80): Allow (optional)
   HTTPS (443): Allow (optional)
   ```

#### 1.2 Setup via Azure CLI

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name rg-kg-project --location eastasia

# Create VM
az vm create \
  --resource-group rg-kg-project \
  --name vm-kg-gpu \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NV12ads_A10_v5 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Get VM IP
az vm show \
  --resource-group rg-kg-project \
  --name vm-kg-gpu \
  --show-details \
  --query publicIps \
  --output tsv
```

### Option 2: Azure Machine Learning Compute

#### 2.1 Create ML Workspace

```bash
# Create ML workspace
az ml workspace create \
  --name kg-ml-workspace \
  --resource-group rg-kg-project \
  --location eastasia
```

#### 2.2 Create Compute Instance

```bash
# Create compute instance
az ml compute create \
  --name kg-gpu-compute \
  --type ComputeInstance \
  --size Standard_NV12ads_A10_v5 \
  --workspace-name kg-ml-workspace \
  --resource-group rg-kg-project
```

## Step-by-Step Implementation

### Step 1: Environment Setup

#### 1.1 Connect to VM

```bash
# SSH into the VM
ssh azureuser@<YOUR_VM_IP>

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git python3-pip python3-venv htop nvtop
```

#### 1.2 Install NVIDIA Drivers

```bash
# Install NVIDIA drivers (if not pre-installed)
sudo apt install -y nvidia-driver-535

# Verify installation
nvidia-smi

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 1.3 Clone Repository and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/2025-IM-senior-project.git
cd 2025-IM-senior-project/Miscellaneous/KgGen/GraphJudge

# Create virtual environment
python3 -m venv kasft_env
source kasft_env/bin/activate

# Install PyTorch with CUDA support
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Install additional requirements
pip install transformers==4.37.2
pip install peft==0.8.2
pip install datasets==2.16.1
pip install accelerate==0.27.2
pip install bitsandbytes==0.42.0
pip install scipy scikit-learn
pip install tqdm pandas numpy
pip install jupyter notebook
```

### Step 2: Prepare Training Data

#### 2.1 Process Raw Knowledge Graph

```bash
# Navigate to datasets directory
cd datasets

# Start Jupyter notebook for data preparation
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### 2.2 Run Data Preparation Notebook

Open the notebook `prepare_KGCom.ipynb` and modify it for Chinese text:

```python
# Example modification for Chinese KG data
import pandas as pd
import json

def prepare_chinese_kg_data(kg_file_path, output_path):
    """
    Prepare Chinese knowledge graph data for training
    
    Args:
        kg_file_path: Path to your Chinese KG triples file
        output_path: Output path for prepared training data
    """
    # Load your Chinese KG triples
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        triples = [line.strip().split('\t') for line in f if line.strip()]
    
    # Create positive and negative examples
    training_data = []
    
    for i, (head, relation, tail) in enumerate(triples):
        # Positive example
        positive_example = {
            "text": f"評估這個知識三元組的品質：[{head}, {relation}, {tail}]",
            "label": 1,  # High quality
            "explanation": "這是一個高品質的知識三元組"
        }
        training_data.append(positive_example)
        
        # Create negative examples by corrupting triples
        # Corrupt head
        corrupted_head = corrupt_entity(head, triples)
        negative_example_1 = {
            "text": f"評估這個知識三元組的品質：[{corrupted_head}, {relation}, {tail}]",
            "label": 0,  # Low quality
            "explanation": "這個三元組的實體關係不正確"
        }
        training_data.append(negative_example_1)
        
        # Corrupt relation
        corrupted_relation = corrupt_relation(relation, triples)
        negative_example_2 = {
            "text": f"評估這個知識三元組的品質：[{head}, {corrupted_relation}, {tail}]",
            "label": 0,  # Low quality
            "explanation": "這個三元組的關係類型不正確"
        }
        training_data.append(negative_example_2)
    
    # Save training data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    return training_data

# Usage
training_data = prepare_chinese_kg_data(
    kg_file_path="GPT4o_mini_result_HongLouMeng/Graph_Iteration1/test_generated_graphs.txt",
    output_path="training_data_chinese.json"
)

print(f"Generated {len(training_data)} training examples")
```

### Step 3: LoRA Fine-Tuning

#### 3.1 Modify Fine-Tuning Script

Create an optimized version of `lora_finetune_chinese_context.py`:

```python
#!/usr/bin/env python3
"""
Optimized LoRA fine-tuning script for Chinese knowledge graph evaluation
Includes memory optimization and cost-effective settings
"""

import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb

# =============================================================================
# Configuration
# =============================================================================

# Model settings
BASE_MODEL_PATH = "/data/models/llama-3-8b-Instruct/"  # Update this path
OUTPUT_DIR = "./models/llama3-8b-lora-chinese-kg/"
DATASET_PATH = "./datasets/training_data_chinese.json"

# Training hyperparameters (optimized for cost)
MICRO_BATCH_SIZE = 4        # Reduced for memory efficiency
BATCH_SIZE = 32             # Effective batch size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
NUM_EPOCHS = 3              # Reduced epochs for cost
LEARNING_RATE = 2e-4
MAX_LENGTH = 512            # Reduced sequence length
SAVE_STEPS = 100

# LoRA configuration (optimized)
LORA_CONFIG = LoraConfig(
    r=16,                   # Reduced rank for efficiency
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 4-bit quantization for memory efficiency
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}

# =============================================================================
# Data Processing
# =============================================================================

def load_and_process_data(dataset_path):
    """Load and process the Chinese KG training data"""
    print(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process data for training
    processed_data = []
    for example in data:
        # Create instruction-following format
        instruction = "你是一個專業的知識圖譜評估專家。請評估給定三元組的品質。"
        input_text = example["text"]
        output_text = example["explanation"]
        
        # Format as conversation
        formatted_text = f"### 指令:\n{instruction}\n\n### 輸入:\n{input_text}\n\n### 回應:\n{output_text}"
        
        processed_data.append({
            "text": formatted_text,
            "label": example["label"]
        })
    
    return Dataset.from_list(processed_data)

def tokenize_function(examples, tokenizer):
    """Tokenize the examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# =============================================================================
# Model Setup
# =============================================================================

def setup_model_and_tokenizer():
    """Setup the model and tokenizer with optimizations"""
    print("Setting up model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    print("Loading base model with 4-bit quantization...")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        **QUANTIZATION_CONFIG,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    model = get_peft_model(model, LORA_CONFIG)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

# =============================================================================
# Training
# =============================================================================

def train_model():
    """Main training function"""
    print("Starting LoRA fine-tuning for Chinese KG evaluation...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and process dataset
    dataset = load_and_process_data(DATASET_PATH)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,                          # Mixed precision
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="adamw_bnb_8bit",            # 8-bit optimizer
        dataloader_pin_memory=False,       # Reduce memory usage
        remove_unused_columns=False,
        report_to=None,                    # Disable wandb
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed!")

# =============================================================================
# Memory Monitoring
# =============================================================================

def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print_gpu_utilization()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start training
    try:
        train_model()
        print("\n✅ Training completed successfully!")
        if torch.cuda.is_available():
            print("\nFinal GPU utilization:")
            print_gpu_utilization()
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        if torch.cuda.is_available():
            print("GPU utilization at failure:")
            print_gpu_utilization()
        raise
```

#### 3.2 Run Training

```bash
# Make the script executable
chmod +x lora_finetune_chinese_context.py

# Start training with monitoring
python lora_finetune_chinese_context.py

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

### Step 4: Inference

#### 4.1 Create Inference Script

```python
#!/usr/bin/env python3
"""
Inference script for Chinese KG evaluation using fine-tuned LoRA model
"""

import torch
import json
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import argparse

class ChineseKGEvaluator:
    def __init__(self, base_model_path, lora_weights_path):
        """Initialize the evaluator with model paths"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading base model...")
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        # Load LoRA weights
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def evaluate_triple(self, head, relation, tail):
        """Evaluate a single knowledge triple"""
        # Format the input
        instruction = "你是一個專業的知識圖譜評估專家。請評估給定三元組的品質。"
        input_text = f"評估這個知識三元組的品質：[{head}, {relation}, {tail}]"
        
        prompt = f"### 指令:\n{instruction}\n\n### 輸入:\n{input_text}\n\n### 回應:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part
        generated_text = response[len(prompt):].strip()
        
        return generated_text
    
    def batch_evaluate(self, triples_file, output_file, threshold=0.5):
        """Evaluate multiple triples and filter based on quality"""
        print(f"Processing triples from {triples_file}")
        
        # Load triples
        with open(triples_file, 'r', encoding='utf-8') as f:
            triples = [line.strip().split('\t') for line in f if line.strip()]
        
        high_quality_triples = []
        evaluation_results = []
        
        for i, (head, relation, tail) in enumerate(triples):
            if i % 100 == 0:
                print(f"Processed {i}/{len(triples)} triples")
            
            try:
                evaluation = self.evaluate_triple(head, relation, tail)
                
                # Simple quality scoring based on keywords
                quality_score = self.compute_quality_score(evaluation)
                
                result = {
                    "triple": [head, relation, tail],
                    "evaluation": evaluation,
                    "quality_score": quality_score,
                    "is_high_quality": quality_score >= threshold
                }
                
                evaluation_results.append(result)
                
                if quality_score >= threshold:
                    high_quality_triples.append([head, relation, tail])
                    
            except Exception as e:
                print(f"Error processing triple {[head, relation, tail]}: {str(e)}")
                continue
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # Save filtered high-quality triples
        filtered_file = output_file.replace('.json', '_filtered.txt')
        with open(filtered_file, 'w', encoding='utf-8') as f:
            for triple in high_quality_triples:
                f.write('\t'.join(triple) + '\n')
        
        print(f"Evaluation completed!")
        print(f"Total triples: {len(triples)}")
        print(f"High quality triples: {len(high_quality_triples)}")
        print(f"Quality ratio: {len(high_quality_triples)/len(triples)*100:.2f}%")
        print(f"Results saved to: {output_file}")
        print(f"Filtered triples saved to: {filtered_file}")
    
    def compute_quality_score(self, evaluation_text):
        """Compute quality score based on evaluation text"""
        # Keywords indicating high quality
        positive_keywords = ['高品質', '正確', '合理', '有效', '準確', '良好']
        negative_keywords = ['低品質', '錯誤', '不合理', '無效', '不準確', '不良']
        
        positive_count = sum(1 for kw in positive_keywords if kw in evaluation_text)
        negative_count = sum(1 for kw in negative_keywords if kw in evaluation_text)
        
        # Simple scoring mechanism
        score = (positive_count - negative_count + len(positive_keywords)) / (2 * len(positive_keywords))
        return max(0, min(1, score))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to base model")
    parser.add_argument("--lora_weights", required=True, help="Path to LoRA weights")
    parser.add_argument("--input_file", required=True, help="Input triples file")
    parser.add_argument("--output_file", required=True, help="Output evaluation file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Quality threshold")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ChineseKGEvaluator(args.base_model, args.lora_weights)
    
    # Run batch evaluation
    evaluator.batch_evaluate(args.input_file, args.output_file, args.threshold)

if __name__ == "__main__":
    main()
```

#### 4.2 Run Inference

```bash
# Run inference on your knowledge graph
python lora_infer_chinese.py \
  --base_model "/data/models/llama-3-8b-Instruct/" \
  --lora_weights "./models/llama3-8b-lora-chinese-kg/" \
  --input_file "datasets/GPT4o_mini_result_HongLouMeng/Graph_Iteration1/test_generated_graphs.txt" \
  --output_file "results/evaluation_results.json" \
  --threshold 0.6
```

## Code Examples

### Memory Optimization Techniques

```python
# 1. Gradient Checkpointing
model.gradient_checkpointing_enable()

# 2. 4-bit Quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3. Memory-efficient Optimizer
training_args.optim = "adamw_bnb_8bit"

# 4. Reduced Precision
training_args.fp16 = True

# 5. Batch Size Optimization
effective_batch_size = 32
micro_batch_size = 4
gradient_accumulation_steps = effective_batch_size // micro_batch_size
```

### Monitoring Script

```python
#!/usr/bin/env python3
"""
Real-time monitoring script for training progress and costs
"""

import time
import psutil
import subprocess
import json
from datetime import datetime, timedelta

class TrainingMonitor:
    def __init__(self, hourly_cost_ntd=29.5):
        self.start_time = datetime.now()
        self.hourly_cost = hourly_cost_ntd
        
    def get_gpu_info(self):
        """Get GPU utilization and memory usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line:
                    util, mem_used, mem_total = line.split(', ')
                    gpu_info.append({
                        'utilization': int(util),
                        'memory_used': int(mem_used),
                        'memory_total': int(mem_total),
                        'memory_percent': int(mem_used) / int(mem_total) * 100
                    })
            return gpu_info
        except:
            return []
    
    def get_cost_info(self):
        """Calculate current cost"""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600
        current_cost = hours * self.hourly_cost
        return {
            'elapsed_hours': hours,
            'current_cost_ntd': current_cost,
            'estimated_daily_cost': self.hourly_cost * 24
        }
    
    def print_status(self):
        """Print current training status"""
        cost_info = self.get_cost_info()
        gpu_info = self.get_gpu_info()
        
        print(f"\n=== Training Monitor - {datetime.now().strftime('%H:%M:%S')} ===")
        print(f"⏱️  Elapsed: {cost_info['elapsed_hours']:.2f} hours")
        print(f"💰 Current Cost: NT${cost_info['current_cost_ntd']:.2f}")
        print(f"📊 Est. Daily Cost: NT${cost_info['estimated_daily_cost']:.2f}")
        
        for i, gpu in enumerate(gpu_info):
            print(f"🖥️  GPU {i}: {gpu['utilization']}% util, {gpu['memory_percent']:.1f}% memory ({gpu['memory_used']}/{gpu['memory_total']} MB)")
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"🧠 CPU: {cpu_percent}% | RAM: {memory.percent}%")
        print("=" * 50)

def monitor_training(interval=60):
    """Monitor training with specified interval"""
    monitor = TrainingMonitor()
    
    print("🚀 Starting training monitor...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            monitor.print_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n📊 Final Training Summary:")
        cost_info = monitor.get_cost_info()
        print(f"Total Time: {cost_info['elapsed_hours']:.2f} hours")
        print(f"Total Cost: NT${cost_info['current_cost_ntd']:.2f}")

if __name__ == "__main__":
    monitor_training()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
MICRO_BATCH_SIZE = 2  # Instead of 4

# Solution 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use 4-bit quantization
load_in_4bit = True

# Solution 4: Reduce sequence length
MAX_LENGTH = 256  # Instead of 512
```

#### 2. Model Loading Errors

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Check model path
import os
if not os.path.exists(BASE_MODEL_PATH):
    print(f"Model path does not exist: {BASE_MODEL_PATH}")
```

#### 3. Training Instability

```python
# Reduce learning rate
LEARNING_RATE = 1e-4  # Instead of 2e-4

# Add warmup steps
warmup_steps = 100

# Use cosine scheduler
lr_scheduler_type = "cosine"
```

#### 4. Cost Control

```bash
# Set up automatic shutdown
sudo crontab -e

# Add line to shutdown after 4 hours
0 */4 * * * /sbin/shutdown -h now

# Monitor costs
python monitoring_script.py
```

### Performance Optimization

#### 1. Batch Size Tuning

```python
def find_optimal_batch_size(model, tokenizer, start_size=1):
    """Find the largest batch size that fits in memory"""
    batch_size = start_size
    while True:
        try:
            # Create dummy batch
            dummy_input = tokenizer(
                ["Test input"] * batch_size,
                return_tensors="pt",
                padding=True,
                max_length=512
            )
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(**dummy_input)
            
            print(f"Batch size {batch_size} works")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_size = batch_size // 2
                print(f"Optimal batch size: {optimal_size}")
                return optimal_size
            else:
                raise e
```

#### 2. Memory Profiling

```python
import torch.profiler

def profile_training_step(model, batch):
    """Profile a single training step"""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

## Cost Optimization

### 1. Spot Instance Strategy

```bash
# Create spot instance with 70% discount
az vm create \
  --resource-group rg-kg-project \
  --name vm-kg-spot \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NV12ads_A10_v5 \
  --priority Spot \
  --max-price 0.30 \
  --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys
```

### 2. Automated Cost Monitoring

```python
#!/usr/bin/env python3
"""
Cost monitoring and automatic shutdown script
"""

import subprocess
import time
from datetime import datetime, timedelta

class CostMonitor:
    def __init__(self, max_cost_ntd=500, check_interval=300):
        self.max_cost = max_cost_ntd
        self.check_interval = check_interval
        self.start_time = datetime.now()
        self.hourly_rate = 29.5  # NT$ per hour for NV12ads_A10_v5
        
    def get_current_cost(self):
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        return elapsed_hours * self.hourly_rate
    
    def check_and_shutdown(self):
        current_cost = self.get_current_cost()
        print(f"Current cost: NT${current_cost:.2f} / NT${self.max_cost}")
        
        if current_cost >= self.max_cost:
            print("⚠️  Maximum cost reached! Shutting down...")
            # Save current work before shutdown
            subprocess.run(["pkill", "-f", "python.*lora_finetune"])
            time.sleep(10)
            subprocess.run(["sudo", "shutdown", "-h", "now"])
    
    def monitor(self):
        print(f"🔍 Cost monitoring started. Max budget: NT${self.max_cost}")
        
        while True:
            self.check_and_shutdown()
            time.sleep(self.check_interval)

if __name__ == "__main__":
    monitor = CostMonitor(max_cost_ntd=500)  # Set your budget here
    monitor.monitor()
```

### 3. Training Checkpoint Strategy

```python
# Save checkpoints frequently to avoid losing progress
training_args = TrainingArguments(
    save_steps=50,                    # Save every 50 steps
    save_total_limit=3,               # Keep only last 3 checkpoints
    load_best_model_at_end=True,
    save_strategy="steps",
    resume_from_checkpoint=True,      # Resume if interrupted
)

# Manual checkpoint saving
def save_emergency_checkpoint(model, tokenizer, output_dir, step):
    """Save emergency checkpoint"""
    checkpoint_dir = f"{output_dir}/emergency_checkpoint_{step}"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Emergency checkpoint saved to {checkpoint_dir}")
```

## Final Tips for Beginners

### 1. Start Small
- Begin with a small dataset (1K triples) to test the pipeline
- Use shorter training time (1 epoch) for initial testing
- Monitor costs closely

### 2. Use Version Control
```bash
# Save your work frequently
git add .
git commit -m "Training checkpoint at step X"
git push origin main
```

### 3. Document Everything
```bash
# Keep a training log
echo "$(date): Started training with batch_size=$BATCH_SIZE, lr=$LEARNING_RATE" >> training_log.txt
```

### 4. Plan Your Budget
- Small experiment: NT$100-200
- Medium dataset: NT$300-500  
- Large scale: NT$1000+
- Always set a maximum budget limit

### 5. Clean Up Resources
```bash
# Always delete resources when done
az vm delete --resource-group rg-kg-project --name vm-kg-gpu --yes
az group delete --name rg-kg-project --yes
```

This comprehensive guide should help you successfully implement KASFT fine-tuning on Azure while managing costs effectively. Remember to start small, monitor closely, and scale up gradually as you gain experience with the platform.