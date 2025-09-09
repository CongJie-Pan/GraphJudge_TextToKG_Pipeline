"""
LoRA Fine-tuned Model Batch Inference for Graph Judgment Tasks

This script performs efficient batch inference using a LoRA fine-tuned LLaMA model 
for graph judgment tasks. It provides optimized processing of multiple instructions
simultaneously to improve throughput and reduce inference time.

Main functionality:
1. Loads a base LLaMA model with LoRA adapters for graph judgment
2. Implements both simple and advanced prompt templates for different use cases
3. Supports efficient batch processing with configurable batch sizes
4. Handles padding and attention masks for variable-length sequences
5. Processes large datasets with progress tracking and memory optimization

The batch processing approach significantly improves throughput compared to
single-instance inference, making it suitable for evaluating large test sets
or production environments requiring high processing rates.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU device for inference
import sys
import torch
import argparse  # Command line argument parsing
import pandas as pd  # Data manipulation and CSV operations
from peft import PeftModel  # Parameter-Efficient Fine-Tuning model loading
from tqdm import tqdm  # Progress tracking for batch processing
from datasets import load_dataset  # HuggingFace datasets for data loading
import transformers
# Ensure LLaMA is available in the current transformers installation
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# === MODEL CONFIGURATION ===
# Define base model and LoRA adapter paths
BASE_MODEL = "NousResearch/Llama-2-7b-hf"  # Base LLaMA model path

# === LORA WEIGHTS CONFIGURATION ===
# Different LoRA adapter options for various datasets/tasks
# LORA_WEIGHTS = "models/llama2-7b-lora-wn18rr/"      # WN18RR dataset
# LORA_WEIGHTS = "models/llama2-7b-lora-wn11/"        # WN11 dataset
# LORA_WEIGHTS = "models/llama2-7b-lora-FB13/"        # FB13 dataset

LORA_WEIGHTS = "models/llama2-7b-lora-rebel-sub/"      # REBEL-sub dataset (default)
# LORA_WEIGHTS = "models/llama2-7b-lora-scierc-context/"     # SciERC with context
# LORA_WEIGHTS = "models/llama2-7b-lora-genwiki-context-tmp/" # GenWiki with context
# LORA_WEIGHTS = "models/llama2-7b-lora-genwiki-20250508/"   # GenWiki specific version

# === TOKENIZER SETUP ===
# Initialize tokenizer with specific configurations for batch processing
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'left'  # Left padding for causal language modeling
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
LOAD_8BIT = False  # Flag for 8-bit quantization (disabled for better quality)

# === DEVICE SETUP ===
# Automatic device detection with priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    # Check for Apple Silicon MPS if available
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

# === MODEL LOADING ===
# Load the base LLaMA model with appropriate settings
model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,         # Optional 8-bit quantization
        # torch_dtype=torch.float16,     # Half precision (commented)
        # device_map="auto",             # Automatic device mapping (commented)
    ).half().cuda()  # Convert to half precision and move to GPU

# Apply LoRA adapters to the base model
model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,                   # Path to LoRA adapter weights
        # torch_dtype=torch.float16,     # Half precision (commented)
    ).half().cuda()  # Convert to half precision and move to GPU

# Additional optimization (commented out but available)
# if not LOAD_8BIT:
#     model.half()  # seems to fix bugs for some users.

def generate_prompt(instruction):
    """
    Generate a simple structured prompt for instruction-following inference.
    
    Args:
        instruction (str): The main instruction/question for the model
        
    Returns:
        str: Formatted prompt following the Alpaca instruction format
    
    This function creates a basic prompt template suitable for most
    graph judgment tasks without additional context.
    """
    prompt = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
"""
    return prompt

def generate_advanced_prompt(instruction, input=None):
    """
    Generate an advanced structured prompt specifically designed for graph judgment tasks.
    
    Args:
        instruction (str): The main instruction/question for the model
        input (str, optional): Additional context (not used in this implementation)
        
    Returns:
        str: Formatted prompt with specific guidelines for graph judgment
    
    This function creates a more detailed prompt template that includes:
    - Clear task definition for graph judgment
    - Specific attention points for correctness evaluation
    - Example demonstrations for consistent output format
    - Explicit output format requirements
    """
    prompt = f"""
Goal:
You need to do the graph judgement task, which means you need to clarify
 the correctness of the given triple.
Attention:
1.The correct triple sentence should have a correct grammatical structure.
2.The knowledge included in the triple sentence should not conflict with
the knowledge you have learned.
3.The answer should be either "Yes, it is true." or "No, it is not true."

Here are two examples:
Example#1:
Question: Is this ture: Apple Founded by Mark Zuckerberg ?
Answer: No, it is not true.
Example#2:
Question: Is this true: Mark Zuckerberg Founded Facebook ?
Answer: Yes, it is true.

Refer to the examples and here is the question:
Question: {instruction}
Answer:
"""
    return prompt

# Set model to evaluation mode
model.eval()

def evaluate(
    instruction,
    input=None,
    temperature=0,       # Low temperature for deterministic output
    top_p=0.75,         # Nucleus sampling parameter
    top_k=10,           # Top-k sampling parameter
    num_beams=4,        # Beam search width
    max_new_tokens=64,  # Maximum tokens to generate
    **kwargs,
):
    """
    Generate a response for a single instruction using the fine-tuned model.
    
    Args:
        instruction (str): The instruction/question for the model
        input (str, optional): Additional context (unused in this implementation)
        temperature (float): Sampling temperature (0 = deterministic)
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k sampling limit
        num_beams (int): Number of beams for beam search
        max_new_tokens (int): Maximum new tokens to generate
        **kwargs: Additional generation parameters
        
    Returns:
        str: Generated response text
    
    This function handles single-instance inference and is mainly kept
    for compatibility. The batch_evaluate function is preferred for efficiency.
    """
    # Create formatted prompt
    prompt = generate_prompt(instruction, input)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    
    # Debug output
    print("Input:")
    print(prompt)

    # Generate response using the model
    with torch.no_grad():  # Disable gradient computation for inference
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,  # Return detailed generation info
            output_scores=True,           # Include generation scores
            max_new_tokens=max_new_tokens,
        )
    
    # Extract generated sequence
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    
    # Debug output
    print("Output:")
    print(output)
    
    # Extract only the response portion (after the prompt)
    return output.split("### Response:")[1].strip()

def batch_evaluate(
    instructions,
    temperature=0,       # Low temperature for deterministic output
    top_p=0.75,         # Nucleus sampling parameter
    top_k=10,           # Top-k sampling parameter
    num_beams=4,        # Beam search width
    max_new_tokens=64,  # Maximum tokens to generate
    **kwargs,
):
    """
    Perform efficient batch inference on multiple instructions simultaneously.
    
    Args:
        instructions (list): List of instruction strings to process
        temperature (float): Sampling temperature (0 = deterministic)
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k sampling limit
        num_beams (int): Number of beams for beam search
        max_new_tokens (int): Maximum new tokens to generate
        **kwargs: Additional generation parameters
        
    Returns:
        list: List of generated responses corresponding to input instructions
    
    This function provides the core batch processing capability, enabling
    efficient processing of multiple instructions by leveraging parallel
    computation on GPUs. It handles variable-length inputs through padding
    and attention masks.
    """
    # Generate prompts for all instructions in the batch
    prompts = [generate_prompt(inst) for inst in instructions]
    
    # Tokenize all prompts in batch with padding and truncation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",        # Return PyTorch tensors
        padding=True,               # Pad sequences to the same length
        truncation=True,            # Truncate sequences that exceed max_length
        max_length=512              # Maximum sequence length (adjustable based on needs)
    )
    
    # Move input tensors to the appropriate device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    
    # Generate responses for the entire batch
    with torch.no_grad():  # Disable gradient computation for inference
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,      # Use attention mask for proper padding handling
            generation_config=generation_config,
            return_dict_in_generate=True,       # Return detailed generation info
            output_scores=True,                 # Include generation scores
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,  # Specify padding token ID
        )
    
    # Process and extract responses from generated sequences
    outputs = []
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        
        # Extract response portion if the expected format is present
        if "### Response:" in output:
            outputs.append(output.split("### Response:")[1].strip())
        else:
            # Fallback: use the entire output if format is unexpected
            outputs.append(output)
    
    return outputs

if __name__ == "__main__":
    # === COMMAND LINE INTERFACE ===
    # Set up argument parser for different dataset configurations
    parser = argparse.ArgumentParser()
    
    # Various dataset options (commented out alternatives)
    # parser.add_argument("--finput", type=str, default="data/WN11/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN11/pred_instructions_llama2_7b.csv")
    # parser.add_argument("--finput", type=str, default="data/FB13/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/FB13/pred_instructions_llama2_7b.csv")
    
    # Default configuration for REBEL-sub dataset
    parser.add_argument("--finput", type=str, 
                       default="data/rebel_sub_4omini_context/test_instructions_context_llama2_7b_woECTD.csv",
                       help="Input CSV file containing test instructions")
    parser.add_argument("--foutput", type=str, 
                       default="data/rebel_sub_4omini_context/pred_instructions_context_llama2_7b_woECTD.csv",
                       help="Output CSV file for predictions")
    
    # Additional dataset options (commented)
    # parser.add_argument("--finput", type=str, default="data/WN18RR/test_instructions_llama_merge.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN18RR/pred_instructions_llama2_7b_merge.csv")
    
    args = parser.parse_args()
    
    # === DATA LOADING AND PROCESSING ===
    # Load input data from CSV file
    total_input = pd.read_csv(args.finput, header=0, sep=',')
    instruct, pred = [], []  # Lists to store instructions and predictions
    total_num = len(total_input)
    batch_size = 32  # Batch size (adjustable based on GPU memory constraints)
    
    # Extract prompts from the dataset
    prompts = total_input['prompt'].tolist()
    
    # Process data in batches for efficient inference
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
    
    print(f"Batch inference completed! Results saved to: {args.foutput}")
