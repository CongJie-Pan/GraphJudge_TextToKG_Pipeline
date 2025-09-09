"""
BERT Binary Classifier Inference for Graph Judgment Tasks

This script performs inference using a fine-tuned BERT binary classifier for graph judgment tasks.
It loads a trained BERT classification model and evaluates the correctness of knowledge graph
statements, providing both predictions and confidence scores.

Main functionality:
1. Loads a fine-tuned BERT classifier and tokenizer
2. Processes instruction-based prompts for binary classification
3. Generates predictions with confidence scores
4. Supports efficient batch processing for large datasets
5. Outputs predictions in CSV format for analysis

This approach provides faster inference compared to generative models and explicit
confidence scores, making it suitable for applications requiring quick and reliable
graph judgment decisions with uncertainty quantification.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set GPU device for inference
import torch
import argparse  # Command line argument parsing
import pandas as pd  # Data manipulation and CSV operations
from tqdm import tqdm  # Progress tracking for batch processing
from datasets import load_dataset  # HuggingFace datasets for data loading
from transformers import BertTokenizer, BertForSequenceClassification  # BERT components

# === MODEL CONFIGURATION ===
# Define model paths and settings
BASE_MODEL = "google-bert/bert-base-uncased"  # Base BERT model identifier
#WEIGHTS_PATH = "models/bert-base-classifier-scierc"      # Alternative trained model path
WEIGHTS_PATH = "models/bert-base-classifier-rebel-sub"    # Path to fine-tuned classifier

# === DEVICE SETUP ===
# Automatic device detection with CUDA preference
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MODEL LOADING ===
# Initialize tokenizer and model from the fine-tuned checkpoint
tokenizer = BertTokenizer.from_pretrained(WEIGHTS_PATH)  # Load tokenizer configuration
model = BertForSequenceClassification.from_pretrained(
    WEIGHTS_PATH,
    num_labels=2                    # Binary classification setup
).half().cuda()                     # Convert to half precision and move to GPU

def process_input(instruction, input_text=None):
    """
    Process raw instruction and input into formatted text for classification.
    
    Args:
        instruction (str): The main instruction/question for classification
        input_text (str, optional): Additional context or input data
        
    Returns:
        str: Formatted text ready for tokenization and classification
    
    This function formats the input in a way that matches the training format
    used during fine-tuning, ensuring consistent performance during inference.
    """
    if input_text:
        # Format with both instruction and input context
        text = f"Instruction: {instruction} Input: {input_text}"
    else:
        # Format with instruction only
        text = f"Instruction: {instruction}"
    return text

def batch_evaluate(
    instructions,
    max_length=512,     # Maximum sequence length for BERT
    batch_size=32       # Batch size for processing (unused parameter but kept for compatibility)
):
    """
    Perform batch inference on a list of instructions using the BERT classifier.
    
    Args:
        instructions (list): List of instruction strings to classify
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size parameter (kept for compatibility)
        
    Returns:
        list: List of prediction results with confidence scores
    
    This function processes multiple instructions simultaneously for efficient
    inference. It tokenizes the inputs, runs them through the classifier,
    and formats the outputs with confidence scores.
    """
    # Process all input instructions into formatted text
    processed_texts = [process_input(inst) for inst in instructions]
    
    # Tokenize all texts in batch for efficient processing
    inputs = tokenizer(
        processed_texts,
        return_tensors="pt",        # Return PyTorch tensors
        padding=True,               # Pad sequences to the same length
        truncation=True,            # Truncate sequences that exceed max_length
        max_length=max_length       # Maximum sequence length
    )
    
    # Move input tensors to the appropriate device (GPU/CPU)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Perform inference with the classifier model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
    # Process model outputs to get predictions and confidence scores
    predictions = torch.softmax(outputs.logits, dim=-1)  # Convert logits to probabilities
    predicted_labels = torch.argmax(predictions, dim=-1)  # Get predicted class labels
    confidence_scores = torch.max(predictions, dim=-1)[0]  # Get confidence scores (max probability)
    
    # Convert predictions to human-readable format with confidence scores
    results = []
    for label, score in zip(predicted_labels, confidence_scores):
        label = label.item()  # Convert tensor to Python integer
        score = score.item()  # Convert tensor to Python float
        
        # Map binary labels to descriptive text
        if label == 1:
            result = "Yes, it is true."
        else:
            result = "No, it is not true."
        
        # Include confidence score for uncertainty quantification
        results.append(f"{result} (confidence: {score:.3f})")
    
    return results

if __name__ == "__main__":
    # === COMMAND LINE INTERFACE ===
    # Set up argument parser for configurable input/output paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--finput", type=str, 
                       default="data/scierc_4omini_context/test_instructions_context_llama2_7b.json",
                       help="Input JSON file containing test instructions")
    parser.add_argument("--foutput", type=str, 
                       default="data/scierc_4omini_context/pred_instructions_context_bert_classifier.csv",
                       help="Output CSV file for classifier predictions")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing (adjustable based on GPU memory)")
    args = parser.parse_args()
    
    # === DATA LOADING ===
    # Load test data from JSON file using HuggingFace datasets
    total_input = load_dataset("json", data_files=args.finput)
    data_eval = total_input["train"]  # Access the loaded dataset

    # Initialize containers for results
    instruct, pred = [], []
    total_num = len(data_eval)
    batch_size = args.batch_size  # Batch size (adjustable based on GPU memory)
    prompts = data_eval['instruction']  # Extract instruction column
    
    # === BATCH PROCESSING ===
    # Process instructions in batches for efficient inference
    for i in tqdm(range(0, total_num, batch_size), desc="Processing batches"):
        # Extract current batch of prompts
        batch_prompts = prompts[i:i+batch_size]
        tqdm.write(f'{i}/{total_num}')  # Progress indicator
        
        # Generate predictions for the current batch
        batch_responses = batch_evaluate(batch_prompts)
        
        # Process and store results
        for cur_instruct, cur_response in zip(batch_prompts, batch_responses):
            # Clean response by replacing newlines with commas for CSV compatibility
            cur_response = cur_response.replace('\n', ',')
            pred.append(cur_response)
            instruct.append(cur_instruct)
    
    # === OUTPUT GENERATION ===
    # Create output DataFrame and save to CSV
    output = pd.DataFrame({'prompt': instruct, 'generated': pred})
    output.to_csv(args.foutput, header=True, index=False)
    
    print(f"Classification inference completed! Results saved to: {args.foutput}")