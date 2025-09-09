"""
Naive Batch Inference for Graph Judgment Tasks - Base Model Without LoRA

This script performs batch inference using the base LLaMA model without any LoRA adapters.
It serves as a baseline for comparison against fine-tuned models, allowing evaluation
of how much improvement fine-tuning provides over the pre-trained base model.

Main functionality:
1. Uses the base LLaMA-2-7B model without any fine-tuning adaptations
2. Implements simple instruction-following prompts for graph judgment
3. Supports efficient batch processing with configurable batch sizes
4. Processes questions in a specific format for consistency with fine-tuned variants
5. Provides baseline performance metrics for comparison studies

This "naive" approach helps establish the performance floor and demonstrates
the value added by domain-specific fine-tuning on graph judgment tasks.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set GPU device for inference
import sys
import torch
import argparse  # Command line argument parsing
import pandas as pd  # Data manipulation and CSV operations
from peft import PeftModel  # Not used in naive version but imported for consistency
from tqdm import tqdm  # Progress tracking for batch processing
import transformers
# Ensure LLaMA is available in the current transformers installation
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# === MODEL CONFIGURATION ===
# Use base model without any fine-tuning
BASE_MODEL = "NousResearch/Llama-2-7b-hf"  # Base LLaMA model path

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
# Load only the base LLaMA model without any adaptations
model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,         # Optional 8-bit quantization
        # torch_dtype=torch.float16,     # Half precision (commented)
        # device_map="auto",             # Automatic device mapping (commented)
    ).half().cuda()  # Convert to half precision and move to GPU

# Note: No LoRA adapters are loaded in the naive version
# Additional optimization (commented out but available)
# if not LOAD_8BIT:
#     model.half()  # seems to fix bugs for some users.

def generate_prompt(instruction):
    """
    Generate a structured prompt with specific formatting for graph judgment.
    
    Args:
        instruction (str): The main instruction/question for the model
        
    Returns:
        str: Formatted prompt following a consistent format for evaluation
    
    This function creates a specific prompt format that extracts and reformats
    the question portion from the instruction to ensure consistency across
    different evaluation scenarios. It specifically looks for "Is this" patterns
    to maintain uniformity in question formatting.
    """
    prompt = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{'Is this' + instruction.split('Is this')[1]}
### Response:
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
    Generate a response for a single instruction using the base model.
    
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
    
    This function handles single-instance inference using only the base model
    capabilities without any domain-specific fine-tuning. It serves as a baseline
    for measuring the improvement provided by specialized training.
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

    # Generate response using the base model
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
    Perform efficient batch inference on multiple instructions using the base model.
    
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
    
    This function provides baseline batch processing capability using only
    the pre-trained model knowledge. It demonstrates the base model's ability
    to handle graph judgment tasks without any specialized training.
    """
    # Generate prompts for all instructions in the batch
    prompts = [generate_prompt(inst) for inst in instructions]
    
    # Tokenize all prompts in batch with padding and truncation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",        # Return PyTorch tensors
        padding=True,               # Pad sequences to the same length
        truncation=True,            # Truncate sequences that exceed max_length
        max_length=512              # Maximum sequence length
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
    
    # Generate responses for the entire batch using base model
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
    # Set up argument parser for baseline evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument("--finput", type=str, 
                       default="data/rebel_sub_4omini_context/test_instructions_context_llama2_7b_woECTD.csv",
                       help="Input CSV file containing test instructions")
    parser.add_argument("--foutput", type=str, 
                       default="data/rebel_sub_4omini_context/pred_instructions_context_llama2_7b_woECTD_naive.csv",
                       help="Output CSV file for naive baseline predictions")
    args = parser.parse_args()
    
    # === DATA LOADING AND PROCESSING ===
    # Load input data from CSV file
    total_input = pd.read_csv(args.finput, header=0, sep=',')
    instruct, pred = [], []  # Lists to store instructions and predictions
    total_num = len(total_input)
    batch_size = 32  # Batch size (adjustable based on GPU memory)
    
    # Extract prompts from the dataset
    prompts = total_input['prompt'].tolist()
    
    # Process data in batches for efficient baseline inference
    for i in tqdm(range(0, total_num, batch_size), desc="Processing naive baseline"):
        # Extract current batch of prompts
        batch_prompts = prompts[i:i+batch_size]
        tqdm.write(f'{i}/{total_num}')  # Progress indicator
        
        # Generate baseline predictions using the base model
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
    
    print(f"Naive baseline inference completed! Results saved to: {args.foutput}")
