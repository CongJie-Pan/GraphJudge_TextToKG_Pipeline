"""
LoRA Fine-tuning for Graph Judgment on REBEL Dataset with Context

This script implements Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA)
for fine-tuning LLaMA-3-8B-Instruct model on graph judgment tasks using the REBEL dataset.
The REBEL dataset focuses on relation extraction from Wikipedia text and is well-suited for
knowledge graph construction and evaluation tasks.

Main functionality:
1. Loads instruction-response dataset specifically designed for REBEL relation extraction
2. Applies LoRA adapters for efficient fine-tuning on relation extraction knowledge
3. Fine-tunes the model to judge correctness of relationship triples from Wikipedia text
4. Supports distributed training with optimized hyperparameters for text-based relations
5. Implements custom data preprocessing for relation-focused instruction-following format

This variant is optimized for understanding entity relationships extracted from Wikipedia
and other text sources, making it particularly effective for knowledge graphs derived
from large-scale text corpora and web content.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU device for training
import sys
import torch
import torch.nn as nn
# import bitsandbytes as bnb  # For 8-bit optimization (optional)
from datasets import load_dataset  # HuggingFace datasets for data loading
import transformers
# Ensure LLaMA is available in the current transformers installation
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, AutoTokenizer  # LLaMA model and tokenizer
from peft import (  # Parameter-Efficient Fine-Tuning library
    LoraConfig,  # LoRA configuration
    get_peft_model,  # Wrap model with PEFT adapters
    get_peft_model_state_dict,  # Extract PEFT weights
)

# === TRAINING HYPERPARAMETERS ===
# These parameters are optimized for REBEL relation extraction tasks

MICRO_BATCH_SIZE = 8                # Batch size per device (suitable for relation extraction tasks)
BATCH_SIZE = 128                    # Effective batch size through gradient accumulation
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # Steps to accumulate gradients
# EPOCHS = 2                        # Number of training epochs (commented in favor of step-based training)
STEPS = 500                         # Maximum training steps (step-based training for consistency)
LEARNING_RATE = 3e-4                # The Karpathy constant - proven effective for LLM fine-tuning
CUTOFF_LEN = 512                   # Maximum sequence length for input
LORA_R = 8                         # LoRA rank - controls adaptation capacity vs efficiency trade-off
LORA_ALPHA = 16                    # LoRA scaling parameter (typically 2x rank)
LORA_DROPOUT = 0.05                # Dropout rate in LoRA layers for regularization
VAL_SET_SIZE = 2000                # Size of validation set for monitoring training
TARGET_MODULES = [                 # Model modules to apply LoRA to
    "q_proj",                      # Query projection in attention layers
    "v_proj",                      # Value projection in attention layers
]

# === FILE PATHS AND MODEL CONFIG ===
DATA_PATH = "data/rebel_sub_4omini_context/train_instructions_context_llama2_7b.json"  # REBEL training data path
OUTPUT_DIR = "models/llama3-8b-instruct-lora-rebel-sub-context"  # Directory to save fine-tuned model
# base_model_path = "NousResearch/Llama-2-7b-hf"  # Alternative base model
base_model_path = "/data/haoyuhuang/model/llama-3-8b-Instruct/"  # Path to LLaMA-3-8B-Instruct base model

# === DISTRIBUTED TRAINING SETUP ===
# Configure for multi-GPU training using DistributedDataParallel (DDP)
device_map = "auto"  # Automatic device mapping for model layers
world_size = int(os.environ.get("WORLD_SIZE", 1))  # Number of processes in distributed training
ddp = world_size != 1  # Enable DDP if multiple processes detected

if ddp:
    # Configure device mapping for distributed training
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    # Adjust gradient accumulation steps for distributed training
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# === TOKENIZER SETUP ===
# Initialize tokenizer with special configurations for instruction fine-tuning
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
tokenizer.padding_side = "left"     # Left padding for causal language modeling
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
tokenizer.pad_token_id = 0          # Set padding token ID

# === LORA CONFIGURATION ===
# Configure LoRA adapter parameters for efficient fine-tuning
config = LoraConfig(
    r=LORA_R,                       # Rank of the adaptation matrices
    lora_alpha=LORA_ALPHA,          # Scaling parameter for LoRA updates
    target_modules=TARGET_MODULES,   # Which model modules to adapt
    lora_dropout=LORA_DROPOUT,      # Dropout rate for regularization
    bias="none",                    # Whether to adapt bias terms
    task_type="CAUSAL_LM",          # Task type for causal language modeling
)

# === MODEL INITIALIZATION ===
# Load the pre-trained LLaMA model and apply LoRA adapters
model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,      # Use half precision to reduce memory usage
    # device_map=device_map,        # Automatic device placement (commented for manual control)
)

# Apply LoRA adapters to the model
model = get_peft_model(model, config)
model.print_trainable_parameters()  # Display number of trainable vs total parameters
model.config.use_cache = False      # Disable KV cache for training (saves memory)
model.config.pretraining_tp = 1     # Tensor parallelism setting

def generate_and_tokenize_prompt(data_point):
    """
    Process a single data point into tokenized format for training on REBEL relation extraction.
    
    Args:
        data_point (dict): Raw data containing instruction, input, and output
        
    Returns:
        dict: Tokenized inputs with proper labels for causal language modeling
    
    This function implements the instruction-following format used in Alpaca-style
    fine-tuning, specifically adapted for relation extraction tasks. It handles
    entity relationships and contextual information common in REBEL dataset,
    ensuring proper masking for relation judgment training.
    
    The prompt format is optimized for relation extraction understanding:
    - Instruction: Description of the relation judgment task
    - Input: Entity context or additional relationship information (optional)
    - Response: Expected judgment for relation correctness
    """
    # Ensure all inputs are strings to handle potential type issues
    instruction = str(data_point["instruction"])
    input_text = str(data_point["input"]) if data_point["input"] else ""
    output = str(data_point["output"])

    # Build the prompt based on whether input context is provided
    if input_text:
        # Format with both instruction and relation context
        user_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input_text}
### Response:
"""
    else:
        # Format with instruction only
        user_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
"""

    # Tokenize the complete sequence (prompt + response)
    encoded = tokenizer(
        user_prompt + output,
        truncation=True,                # Truncate if exceeds max length
        max_length=CUTOFF_LEN,         # Maximum sequence length
        padding="max_length",          # Pad to maximum length
        return_tensors=None,           # Return as lists, not tensors
    )

    # Tokenize just the prompt to determine where the response starts
    # This is crucial for masking the prompt portion in loss computation
    prompt_encoded = tokenizer(
        user_prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,                 # No padding for prompt-only encoding
        return_tensors=None,
    )

    # Create labels for loss computation
    # Set prompt portion to -100 (ignored in loss) and response portion to actual token IDs
    labels = [-100] * len(prompt_encoded["input_ids"])  # Mask prompt tokens
    labels.extend(encoded["input_ids"][len(prompt_encoded["input_ids"]):])  # Include response tokens
    
    # Ensure labels match the sequence length
    if len(labels) < CUTOFF_LEN:
        # Pad labels with -100 if shorter than max length
        labels.extend([-100] * (CUTOFF_LEN - len(labels)))
    labels = labels[:CUTOFF_LEN]  # Truncate if longer than max length

    return {
        "input_ids": encoded["input_ids"],        # Token IDs for the complete sequence
        "attention_mask": encoded["attention_mask"],  # Attention mask for padding
        "labels": labels                          # Labels for loss computation
    }

if __name__ == "__main__":
    # === DATA LOADING AND PREPROCESSING ===
    # Load the REBEL training dataset from JSON file
    data = load_dataset("json", data_files=DATA_PATH)
    
    # Split data into training and validation sets if validation is desired
    if VAL_SET_SIZE > 0:
        # Create train/validation split with fixed seed for reproducibility
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        # Process training data: shuffle and apply tokenization
        train_data = (
            train_val["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        # Process validation data: shuffle and apply tokenization
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
    else:
        # Use all data for training if no validation set is specified
        train_data = (
            data["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        val_data = None

    # === TRAINING SETUP ===
    # Initialize the HuggingFace Trainer with comprehensive training arguments
    trainer = transformers.Trainer(
        model=model,                    # The LoRA-adapted model
        train_dataset=train_data,       # Processed training data
        eval_dataset=val_data,          # Processed validation data (if available)
        args=transformers.TrainingArguments(
            # Batch size and gradient accumulation settings
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            
            # Learning rate schedule and optimization
            warmup_steps=100,           # Linear warmup steps
            # num_train_epochs=EPOCHS,  # Total epochs (commented in favor of max_steps)
            max_steps=STEPS,            # Maximum training steps
            learning_rate=LEARNING_RATE, # Peak learning rate
            
            # Memory and precision optimizations
            fp16=True,                  # Mixed precision training for memory efficiency
            
            # Logging and monitoring
            logging_steps=5,            # Log metrics every N steps
            logging_strategy="steps",   # Log based on steps, not epochs
            logging_first_step=True,    # Log the first step for debugging
            
            # Evaluation strategy
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            eval_steps=20 if VAL_SET_SIZE > 0 else None,
            
            # Model saving strategy
            save_strategy="steps",      # Save based on steps
            save_steps=20,             # Save checkpoint every N steps
            save_total_limit=3,        # Keep only the latest N checkpoints
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            
            # Output and distributed training settings
            output_dir=OUTPUT_DIR,      # Directory for outputs and checkpoints
            ddp_find_unused_parameters=False if ddp else None,  # DDP optimization
            
            # Optimizer and reporting
            optim="adamw_torch",        # AdamW optimizer implementation
            report_to="wandb",          # Report metrics to Weights & Biases
        ),
        # Data collator for dynamic padding and batching
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8,       # Pad to multiples of 8 for tensor core efficiency
            return_tensors="pt",        # Return PyTorch tensors
            padding=True                # Enable dynamic padding
        ),
    )

    # === MODEL TRAINING ===
    # Start the fine-tuning process
    trainer.train()

    # === MODEL SAVING ===
    # Save the fine-tuned LoRA adapters and tokenizer
    model.save_pretrained(OUTPUT_DIR)      # Save LoRA weights
    tokenizer.save_pretrained(OUTPUT_DIR)  # Save tokenizer configuration
    print("\n Training completed! Model saved to:", OUTPUT_DIR)