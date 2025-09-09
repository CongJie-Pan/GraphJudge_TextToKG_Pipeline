"""
BERT Binary Classifier Fine-tuning for Graph Judgment Tasks

This script fine-tunes a BERT-base model for binary classification of knowledge graph statements.
Unlike generative approaches, this classifier directly predicts whether a given statement
is correct (true) or incorrect (false) using supervised learning.

Main functionality:
1. Loads instruction-response pairs and converts them to binary classification format
2. Fine-tunes BERT-base-uncased for sequence classification
3. Implements proper data preprocessing with label conversion
4. Supports distributed training and validation monitoring
5. Saves the trained classifier for inference

This approach offers faster inference compared to generative models and provides
explicit confidence scores for predictions, making it suitable for applications
requiring quick and reliable graph judgment decisions.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU device for training
import torch
import torch.nn as nn
from datasets import load_dataset  # HuggingFace datasets for data loading
from transformers import (  # HuggingFace transformers for BERT components
    BertTokenizer,                  # BERT tokenizer for text preprocessing
    BertForSequenceClassification,  # BERT model for classification tasks
    TrainingArguments,              # Training configuration
    Trainer                         # High-level training interface
)

# === TRAINING HYPERPARAMETERS ===
# Configuration optimized for binary classification tasks
MICRO_BATCH_SIZE = 16               # Batch size per device (larger than generative models)
BATCH_SIZE = 128                    # Effective batch size through gradient accumulation
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # Steps to accumulate gradients
EPOCHS = 4                          # Number of training epochs (typically fewer than LLM fine-tuning)
LEARNING_RATE = 2e-5                # Standard learning rate for BERT fine-tuning
CUTOFF_LEN = 512                   # Maximum sequence length for BERT input
VAL_SET_SIZE = 2000                # Size of validation set for monitoring training

# === FILE PATHS AND MODEL CONFIGURATION ===
DATA_PATH = "data/rebel_sub_4omini_context/train_instructions_context_llama2_7b.json"  # Training data path
OUTPUT_DIR = "models/bert-base-classifier-rebel-sub"  # Directory to save fine-tuned classifier
base_model_path = "google-bert/bert-base-uncased"    # Base BERT model from HuggingFace

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

# === MODEL INITIALIZATION ===
# Initialize BERT tokenizer and classification model
tokenizer = BertTokenizer.from_pretrained(base_model_path)
model = BertForSequenceClassification.from_pretrained(
    base_model_path,
    num_labels=2,                   # Binary classification: correct vs incorrect
    problem_type="single_label_classification"  # Specify single-label classification
)

def generate_and_tokenize_prompt(data_point):
    """
    Process a single data point into tokenized format for binary classification training.
    
    Args:
        data_point (dict): Raw data containing instruction, input, and output
        
    Returns:
        dict: Tokenized inputs with binary labels for classification
    
    This function converts instruction-response pairs into binary classification format.
    The instruction and input are combined into a single text sequence, and the output
    is converted to a binary label (1 for correct/true, 0 for incorrect/false).
    
    This approach differs from generative training by focusing on classification
    rather than text generation, enabling faster inference and explicit confidence scores.
    """
    # Convert inputs to strings to handle potential type issues
    instruction = str(data_point["instruction"])
    input_text = str(data_point["input"]) if data_point["input"] else ""
    
    # Combine instruction and input into a single text for classification
    if input_text:
        # Format with both instruction and input context
        text = f"Instruction: {instruction} Input: {input_text}"
    else:
        # Format with instruction only
        text = f"Instruction: {instruction}"
    
    # Convert output to binary label based on response content
    # This assumes the output contains indicators of correctness
    # Adjust this logic according to your actual output format
    label = 1 if str(data_point["output"]).lower() in ['true', 'correct', 'yes'] else 0
    
    # Tokenize the combined text using BERT tokenizer
    encoded = tokenizer(
        text,
        truncation=True,                # Truncate if exceeds max length
        max_length=CUTOFF_LEN,         # Maximum sequence length for BERT
        padding="max_length",          # Pad to maximum length for consistent batching
        return_tensors=None,           # Return as lists, not tensors
    )
    
    # Add binary labels to the encoded dictionary
    encoded["labels"] = label
    return encoded

if __name__ == "__main__":
    # === DATA LOADING AND PREPROCESSING ===
    # Load the training dataset from JSON file
    data = load_dataset("json", data_files=DATA_PATH)
    
    # Split data into training and validation sets if validation is desired
    if VAL_SET_SIZE > 0:
        # Create train/validation split with fixed seed for reproducibility
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        # Process training data: shuffle and apply tokenization with label conversion
        train_data = (
            train_val["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        # Process validation data: shuffle and apply tokenization with label conversion
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

    # === TRAINING CONFIGURATION ===
    # Configure training arguments specific to classification tasks
    training_args = TrainingArguments(
        # Batch size and gradient accumulation settings
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Learning rate schedule and optimization
        warmup_steps=100,               # Linear warmup steps
        num_train_epochs=EPOCHS,        # Total training epochs
        learning_rate=LEARNING_RATE,    # Peak learning rate (standard for BERT)
        
        # Memory and precision optimizations
        fp16=True,                      # Mixed precision training for memory efficiency
        
        # Logging and monitoring
        logging_steps=5,                # Log metrics every N steps
        logging_strategy="steps",       # Log based on steps, not epochs
        logging_first_step=True,        # Log the first step for debugging
        
        # Evaluation strategy
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        eval_steps=20 if VAL_SET_SIZE > 0 else None,
        
        # Model saving strategy
        save_strategy="steps",          # Save based on steps
        save_steps=20,                 # Save checkpoint every N steps
        save_total_limit=3,            # Keep only the latest N checkpoints
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        
        # Output and distributed training settings
        output_dir=OUTPUT_DIR,          # Directory for outputs and checkpoints
        ddp_find_unused_parameters=False if ddp else None,  # DDP optimization
        
        # Reporting and logging
        report_to="wandb",              # Report metrics to Weights & Biases
    )

    # === TRAINER INITIALIZATION ===
    # Initialize the HuggingFace Trainer for classification
    trainer = Trainer(
        model=model,                    # BERT classification model
        args=training_args,             # Training configuration
        train_dataset=train_data,       # Processed training data
        eval_dataset=val_data,          # Processed validation data (if available)
    )

    # === MODEL TRAINING ===
    # Start the fine-tuning process
    trainer.train()

    # === MODEL SAVING ===
    # Save the fine-tuned classifier and tokenizer
    model.save_pretrained(OUTPUT_DIR)      # Save model weights and configuration
    tokenizer.save_pretrained(OUTPUT_DIR)  # Save tokenizer configuration
    print("\nTraining completed! Model saved to:", OUTPUT_DIR) 