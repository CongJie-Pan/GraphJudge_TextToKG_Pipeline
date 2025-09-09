"""
Backward compatibility wrapper for run_gj.py

This file maintains compatibility with existing CLI and test systems
by providing the same interface as the original run_gj.py file.
"""

import os
import sys
import asyncio
from typing import Optional

# Add the graphJudge_Phase module to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
graphjudge_phase_path = os.path.join(current_dir, "graphJudge_Phase")
if graphjudge_phase_path not in sys.path:
    sys.path.insert(0, graphjudge_phase_path)

# Import from the modularized system
try:
    from graphJudge_Phase import (
        PerplexityGraphJudge,
        GoldLabelBootstrapper,
        ProcessingPipeline,
        TripleData,
        BootstrapResult,
        ExplainableJudgment,
        validate_input_file,
        create_output_directory,
        setup_terminal_logging,
        TerminalLogger,
        PERPLEXITY_MODEL,
        PERPLEXITY_CONCURRENT_LIMIT,
        PERPLEXITY_RETRY_ATTEMPTS,
        PERPLEXITY_BASE_DELAY,
        PERPLEXITY_REASONING_EFFORT,
        PERPLEXITY_MODELS,
        GOLD_BOOTSTRAP_CONFIG,
        LOG_DIR,
        folder,
        iteration,
        input_file,
        output_file,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_TOKENS,
        MAX_RETRY_ATTEMPTS,
        BASE_RETRY_DELAY,
        MAX_RETRY_DELAY,
        DEFAULT_ENCODING,
        CSV_DELIMITER,
        MIN_INSTRUCTION_LENGTH,
        MAX_INSTRUCTION_LENGTH,
        REQUIRED_FIELDS,
        STREAM_UPDATE_FREQUENCY,
        STREAM_CHUNK_DELAY
    )
    print("✓ Successfully imported modularized GraphJudge system")
except ImportError as e:
    print(f"❌ Failed to import modularized GraphJudge system: {e}")
    print("⚠️ Falling back to compatibility mode")
    
    # Create mock classes for compatibility
    class PerplexityGraphJudge:
        def __init__(self, *args, **kwargs):
            print("⚠️ Using mock PerplexityGraphJudge (modularized system not available)")
        
        async def judge_graph_triple(self, *args, **kwargs):
            return "Yes"
        
        async def judge_graph_triple_with_explanation(self, *args, **kwargs):
            from graphJudge_Phase.data_structures import ExplainableJudgment
            return ExplainableJudgment(
                judgment="Yes",
                confidence=0.8,
                reasoning="Mock reasoning",
                evidence_sources=["mock"],
                alternative_suggestions=[],
                error_type=None,
                processing_time=0.1
            )
    
    class GoldLabelBootstrapper:
        def __init__(self, *args, **kwargs):
            print("⚠️ Using mock GoldLabelBootstrapper (modularized system not available)")
        
        async def bootstrap_gold_labels(self, *args, **kwargs):
            return True
    
    class ProcessingPipeline:
        def __init__(self, *args, **kwargs):
            print("⚠️ Using mock ProcessingPipeline (modularized system not available)")
        
        async def process_instructions(self, *args, **kwargs):
            return None

# Re-export all main components for backward compatibility
__all__ = [
    'PerplexityGraphJudge',
    'GoldLabelBootstrapper',
    'ProcessingPipeline',
    'TripleData',
    'BootstrapResult', 
    'ExplainableJudgment',
    'validate_input_file',
    'create_output_directory',
    'setup_terminal_logging',
    'TerminalLogger'
]

# Maintain the same global instance pattern as the original
try:
    gemini_judge = PerplexityGraphJudge(enable_console_logging=False)
    print("✓ Global Perplexity Graph Judge instance initialized")
except Exception as e:
    print(f"✗ Failed to initialize global Perplexity Graph Judge: {e}")
    gemini_judge = None

# Maintain the same dataset loading pattern
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️ datasets library not found. Please install: pip install datasets")
    print("⚠️ Running in compatibility mode - dataset operations will be mocked")
    DATASETS_AVAILABLE = False
    
    # Mock load_dataset for testing
    def load_dataset(*args, **kwargs):
        class MockDataset:
            def __init__(self):
                self.data = [
                    {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, key):
                if key == "train":
                    return MockDatasetSplit(self.data)
                return MockDatasetSplit(self.data)
        
        class MockDatasetSplit:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __iter__(self):
                return iter(self.data)
            
            def train_test_split(self, **kwargs):
                return {"test": MockDatasetSplit(self.data)}
        
        return MockDataset()

# Load the evaluation dataset following existing patterns
try:
    if not DATASETS_AVAILABLE or not os.path.exists(input_file):
        print(f"⚠️ Using mock dataset for testing")
        # Create mock dataset
        mock_data = [
            {"instruction": "Is this true: 曹雪芹 創作 紅樓夢 ?", "input": "", "output": ""},
            {"instruction": "Is this true: Apple Founded by Steve Jobs ?", "input": "", "output": ""},
            {"instruction": "Is this true: Microsoft Founded by Mark Zuckerberg ?", "input": "", "output": ""}
        ]
        
        class MockDataEval:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __iter__(self):
                return iter(self.data)
            
            def __getitem__(self, index):
                return self.data[index]
        
        data_eval = MockDataEval(mock_data)
        print(f"📊 Using mock dataset: {len(data_eval)} instructions")
    else:
        total_input = load_dataset("json", data_files=input_file)
        
        # Dynamically adjust test_size based on available data
        available_samples = len(total_input["train"])
        print(f"✓ Dataset loaded: {available_samples} total samples")
        
        if available_samples <= 50:
            # For small datasets, use all samples
            data_eval = total_input["train"]
            print(f"✓ Using all {available_samples} samples for evaluation (small dataset)")
        else:
            # For larger datasets, use up to 499 samples or all available (whichever is smaller)
            test_size = min(499, available_samples - 1)  # Ensure test_size < available_samples
            if test_size <= 0:
                # If we can't split, use all samples
                data_eval = total_input["train"]
                print(f"✓ Using all {available_samples} samples for evaluation (cannot split)")
            else:
                data_eval = total_input["train"].train_test_split(
                    test_size=test_size, shuffle=True, seed=42
                )["test"]
                print(f"✓ Using {len(data_eval)} evaluation samples from {available_samples} total")
            
        print(f"📊 Final evaluation dataset size: {len(data_eval)} instructions")
    
except Exception as e:
    print(f"⚠️ Error loading dataset, using mock data: {e}")
    # Fallback to mock data
    mock_data = [
        {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
    ]
    
    class MockDataEval:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
    
    data_eval = MockDataEval(mock_data)

# Load instructions data directly for processing compatibility
try:
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            import json
            instructions = json.load(f)
        print(f"✓ Loaded {len(instructions)} instruction entries")
    else:
        print(f"⚠️ Input file not found, using mock instructions")
        instructions = [
            {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
        ]
except Exception as e:
    print(f"⚠️ Error loading instructions, using mock data: {e}")
    instructions = [
        {"instruction": "Is this true: Test Statement ?", "input": "", "output": ""}
    ]

# Maintain the same function signatures for backward compatibility
async def get_perplexity_completion(instruction, input_text=None):
    """
    Get completion from Perplexity API for graph judgment
    
    This function maintains compatibility with the existing pipeline
    while leveraging Perplexity's advanced reasoning capabilities.
    
    Args:
        instruction (str): The instruction/question for classification
        input_text (str, optional): Additional context (if any)
    
    Returns:
        str: The binary judgment result ("Yes" or "No")
    """
    if gemini_judge is None:
        print("✗ Perplexity Graph Judge not initialized")
        return "Error: Graph Judge not available"
    
    return await gemini_judge.judge_graph_triple(instruction, input_text)


def _generate_reasoning_file_path(csv_output_path: str, custom_path: Optional[str] = None) -> str:
    """
    Generate reasoning file path based on CSV output path
    
    Args:
        csv_output_path (str): Path to the main CSV output file
        custom_path (Optional[str]): Custom reasoning file path if specified
        
    Returns:
        str: Path for the reasoning JSON file
    """
    if custom_path:
        return custom_path
    
    # Auto-generate path based on CSV file name
    from pathlib import Path
    path_obj = Path(csv_output_path)
    reasoning_filename = path_obj.stem + "_reasoning" + ".json"
    return str(path_obj.parent / reasoning_filename)


async def process_instructions(explainable_mode: bool = False, reasoning_file_path: Optional[str] = None):
    """
    Process instructions with Perplexity API system for graph judgment
    
    This function orchestrates the entire graph judgment evaluation process:
    1. Creates async tasks for all instruction-input pairs
    2. Executes them with concurrency control (standard or explainable mode)
    3. Collects binary judgment responses and optionally detailed reasoning
    4. Saves results in dual-file format: CSV (compatible) + JSON (explainable)
    
    Args:
        explainable_mode (bool): Whether to enable explainable reasoning mode
        reasoning_file_path (Optional[str]): Custom path for reasoning file
    
    The function leverages Perplexity's grounding capabilities for more accurate
    graph validation while maintaining compatibility with existing infrastructure.
    """
    if gemini_judge is None:
        print("✗ Perplexity Graph Judge not initialized")
        return
    
    # Initialize processing pipeline
    pipeline = ProcessingPipeline(gemini_judge)
    
    # Run the processing pipeline
    await pipeline.process_instructions(
        data_eval=data_eval,
        explainable_mode=explainable_mode,
        reasoning_file_path=reasoning_file_path
    )


def validate_input_file():
    """
    Validate that the input file exists and has the correct format.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    return validate_input_file(input_file)


def create_output_directory():
    """
    Ensure the output directory exists before writing results.
    """
    create_output_directory(output_file)


def parse_arguments():
    """
    Parse command line arguments for different operation modes
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    # Import the argument parser from the modularized system
    from graphJudge_Phase.main import parse_arguments as modularized_parse_arguments
    return modularized_parse_arguments()


async def run_gold_label_bootstrapping(args):
    """
    Run the gold label bootstrapping pipeline
    
    Args:
        args: Parsed command line arguments
    """
    if gemini_judge is None:
        print("❌ Perplexity Graph Judge initialization failed. Exiting.")
        sys.exit(1)
    
    # Initialize bootstrapper
    bootstrapper = GoldLabelBootstrapper(gemini_judge)
    
    # Run the bootstrapping process
    success = await bootstrapper.bootstrap_gold_labels(
        triples_file=args.triples_file,
        source_file=args.source_file,
        output_file=args.output
    )
    
    if not success:
        print("❌ Gold label bootstrapping failed.")
        sys.exit(1)


async def run_graph_judgment(explainable_mode: bool = False, reasoning_file_path: Optional[str] = None):
    """
    Run the graph judgment pipeline (standard or explainable mode)
    
    Args:
        explainable_mode (bool): Whether to enable explainable reasoning mode
        reasoning_file_path (Optional[str]): Custom path for reasoning file
    """
    # Pre-flight validation checks
    print("🔍 Running pre-flight checks...")
    
    if not validate_input_file():
        print("❌ Pre-flight validation failed. Exiting.")
        sys.exit(1)
    
    create_output_directory()
    
    # Run the main processing pipeline with specified mode
    await process_instructions(explainable_mode, reasoning_file_path)
    
    print("\n🎉 Graph judgment pipeline completed successfully!")
    print(f"📂 CSV results available at: {output_file}")
    
    if explainable_mode:
        reasoning_path = _generate_reasoning_file_path(output_file, reasoning_file_path)
        print(f"🧠 Reasoning results available at: {reasoning_path}")
        print("\n📋 Next steps:")
        print("   1. Review the generated CSV file for binary judgment results")
        print("   2. Review the reasoning JSON file for detailed explanations")
        print("   3. Run evaluation metrics against ground truth data")
        print("   4. Analyze confidence scores and error patterns")
        print("   5. Use alternative suggestions for data quality improvement")
    else:
        print("\n📋 Next steps:")
        print("   1. Review the generated CSV file for judgment results")
        print("   2. Run evaluation metrics against ground truth data")
        print("   3. Consider using --explainable mode for detailed insights")


if __name__ == "__main__":
    """
    Main execution block with comprehensive error handling and validation.
    
    This section supports two operation modes:
    1. Standard graph judgment pipeline (default)
    2. Gold label bootstrapping (--bootstrap flag)
    
    The execution flow:
    1. Parse command line arguments
    2. Set up terminal logging
    3. Initialize and validate systems
    4. Run the appropriate pipeline based on mode
    5. Handle errors gracefully
    """
    # Import the main function from the modularized system
    from graphJudge_Phase.main import main as modularized_main
    asyncio.run(modularized_main())
