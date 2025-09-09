"""
LoRA Fine-tuned Model Inference for Graph Judgment Tasks

This script performs inference using a LoRA fine-tuned LLaMA model for graph judgment tasks.
The model evaluates the correctness of knowledge graph triples and generates responses
indicating whether statements are true or false.

Main functionality:
1. Loads a base LLaMA model and applies LoRA adapters
2. Processes instruction-based prompts for graph judgment
3. Generates responses using configurable generation parameters
4. Supports batch processing of evaluation data
5. Outputs predictions in CSV format for analysis

The script is designed for evaluating the performance of fine-tuned models
on various knowledge graph benchmarks and datasets.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU device for inference
#os.system("pip install datasets")        # Install dependencies if needed (commented)
#os.system("pip install deepspeed")
#os.system("pip install accelerate")
#os.system("pip install transformers>=4.28.0")
import sys
import torch
import argparse  # Command line argument parsing
import pandas as pd  # Data manipulation and CSV operations
from peft import PeftModel  # Parameter-Efficient Fine-Tuning model loading
from tqdm import tqdm  # Progress tracking for batch processing
import transformers
# Ensure LLaMA is available in the current transformers installation
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# === MODEL CONFIGURATION ===
# Initialize tokenizer from the base model
tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
LOAD_8BIT = False  # Flag for 8-bit quantization (disabled for better quality)
BASE_MODEL = "NousResearch/Llama-2-7b-hf"  # Base LLaMA model path

# === LORA WEIGHTS CONFIGURATION ===
# Different LoRA adapter options for various datasets/tasks
# LORA_WEIGHTS = "models/llama2-7b-lora-wn18rr/"   # WN18RR dataset
# LORA_WEIGHTS = "models/llama2-7b-lora-wn11/"     # WN11 dataset  
# LORA_WEIGHTS = "models/llama2-7b-lora-FB13/"     # FB13 dataset
LORA_WEIGHTS = "models/llama2-7b-lora-gen/"        # General/GenWiki dataset

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

def generate_prompt(instruction, input=None):
    """
    Generate a structured prompt for instruction-following inference.
    
    Args:
        instruction (str): The main instruction/question for the model
        input (str, optional): Additional context or input data
        
    Returns:
        str: Formatted prompt following the Alpaca instruction format
    
    This function creates prompts that match the training format used during
    fine-tuning, ensuring consistent performance during inference.
    """
    if input:
        # Format with both instruction and additional input context
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        # Format with instruction only
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

# Optimization for PyTorch 2.0+ (commented out due to Windows compatibility)
# if not LOAD_8BIT:
#     model.half()  # seems to fix bugs for some users.

# Set model to evaluation mode
model.eval()

# Torch compilation for faster inference (commented due to platform compatibility)
# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)

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
    Generate a response for a given instruction using the fine-tuned model.
    
    Args:
        instruction (str): The instruction/question for the model
        input (str, optional): Additional context
        temperature (float): Sampling temperature (0 = deterministic)
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k sampling limit
        num_beams (int): Number of beams for beam search
        max_new_tokens (int): Maximum new tokens to generate
        **kwargs: Additional generation parameters
        
    Returns:
        str: Generated response text
    
    This function handles the complete inference pipeline from prompt creation
    to response generation and extraction.
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
    
    # Alternative generation approach (commented out)
    # sequences = model.forward(
    #             prompt,
    #             do_sample=True,
    #             top_k=10,
    #             num_return_sequences=1,
    #             eos_token_id=tokenizer.eos_token_id,
    #             max_length=128,
    #             )
    
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

if __name__ == "__main__":
    # === COMMAND LINE INTERFACE ===
    # Set up argument parser for different dataset configurations
    parser = argparse.ArgumentParser()
    
    # Various dataset options (commented out alternatives)
    # parser.add_argument("--finput", type=str, default="data/WN11/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN11/pred_instructions_llama2_7b.csv")
    # parser.add_argument("--finput", type=str, default="data/FB13/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/FB13/pred_instructions_llama2_7b.csv")
    
    # Default configuration for GenWiki dataset
    parser.add_argument("--finput", type=str, 
                       default="data/genwiki_4o/test_instructions_llama2_7b_itr2.csv",
                       help="Input CSV file containing test instructions")
    parser.add_argument("--foutput", type=str, 
                       default="data/genwiki_4o/pred_instructions_llama2_7b_itr2.csv",
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
    
    # Process each instruction in the dataset
    for index, data in tqdm(total_input.iterrows(), desc="Processing instructions"):
        tqdm.write(f'{index}/{total_num}')  # Progress indicator
        
        # Extract instruction from current row
        cur_instruct = data['prompt']
        
        # Generate prediction using the model
        cur_response = evaluate(cur_instruct)
        
        # Clean response by replacing newlines with commas for CSV compatibility
        cur_response = cur_response.replace('\n', ',')
        
        # Store results for output
        # tqdm.write(cur_response)  # Debug output (commented)
        pred.append(cur_response)
        instruct.append(cur_instruct)
    
    # === OUTPUT GENERATION ===
    # Create output DataFrame and save to CSV
    output = pd.DataFrame({'prompt': instruct, 'generated': pred})
    output.to_csv(args.foutput, header=True, index=False)
    
    print(f"Inference completed! Results saved to: {args.foutput}")
