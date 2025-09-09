"""
Main entry point for the GraphJudge system.

This module provides the command-line interface and main execution logic
for the GraphJudge system, including both standard graph judgment and
gold label bootstrapping modes.
"""

import os
import asyncio
import argparse
import sys
from typing import Optional
from .config import *
from .logging_system import setup_terminal_logging, TerminalLogger
from .graph_judge_core import PerplexityGraphJudge
from .gold_label_bootstrapping import GoldLabelBootstrapper
from .processing_pipeline import ProcessingPipeline
from .utilities import validate_input_file, create_output_directory, setup_environment


def parse_arguments():
    """
    Parse command line arguments for different operation modes.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Perplexity API Graph Judge - Enhanced Knowledge Graph Triple Validation & Gold Label Bootstrapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard graph judgment mode
  python -m graphJudge_Phase.main
  
  # Gold label bootstrapping mode
  python -m graphJudge_Phase.main --bootstrap \
    --triples-file ../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs.txt \
    --source-file ../datasets/KIMI_result_DreamOf_RedChamber/Iteration1/test_denoised.target \
    --output ../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/gold_bootstrap.csv \
    --threshold 0.8 --sample-rate 0.15
        """
    )
    
    # Operation mode
    parser.add_argument(
        '--bootstrap', 
        action='store_true',
        help='Run gold label bootstrapping instead of graph judgment'
    )
    
    parser.add_argument(
        '--explainable',
        action='store_true',
        help='Enable explainable mode - generate detailed reasoning file alongside CSV output'
    )
    
    # Gold label bootstrapping arguments
    parser.add_argument(
        '--triples-file',
        type=str,
        help='Path to the triples file for bootstrapping (required for --bootstrap)'
    )
    
    parser.add_argument(
        '--source-file',
        type=str,
        help='Path to the source text file for bootstrapping (required for --bootstrap)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (for bootstrapping mode)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='RapidFuzz similarity threshold for Stage 1 (default: 0.8)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=0.15,
        help='Sampling rate for manual review (default: 0.15)'
    )
    
    # Explainable mode arguments
    parser.add_argument(
        '--reasoning-file',
        type=str,
        help='Custom path for reasoning file output (optional, auto-generated if not specified)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        choices=list(PERPLEXITY_MODELS.keys()),
        default='sonar-reasoning',
        help='Perplexity model to use (default: sonar-reasoning)'
    )
    
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Reasoning effort level (default: medium)'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable file logging'
    )
    
    return parser.parse_args()


async def run_gold_label_bootstrapping(args):
    """
    Run the gold label bootstrapping pipeline.
    
    Args:
        args: Parsed command line arguments
    """
    # Validate required arguments
    if not args.triples_file:
        print("❌ --triples-file is required for bootstrapping mode")
        sys.exit(1)
    
    if not args.source_file:
        print("❌ --source-file is required for bootstrapping mode")
        sys.exit(1)
    
    if not args.output:
        print("❌ --output is required for bootstrapping mode")
        sys.exit(1)
    
    # Update global configuration
    GOLD_BOOTSTRAP_CONFIG['fuzzy_threshold'] = args.threshold
    GOLD_BOOTSTRAP_CONFIG['sample_rate'] = args.sample_rate
    
    print(f"📊 Bootstrap Configuration:")
    print(f"   - Fuzzy threshold: {args.threshold}")
    print(f"   - Sample rate: {args.sample_rate}")
    print(f"   - Triples file: {args.triples_file}")
    print(f"   - Source file: {args.source_file}")
    print(f"   - Output file: {args.output}")
    
    # Initialize GraphJudge
    try:
        graph_judge = PerplexityGraphJudge(
            model_name=PERPLEXITY_MODELS[args.model],
            reasoning_effort=args.reasoning_effort,
            enable_console_logging=args.verbose
        )
    except Exception as e:
        print(f"❌ Failed to initialize Perplexity Graph Judge: {e}")
        sys.exit(1)
    
    # Initialize bootstrapper
    bootstrapper = GoldLabelBootstrapper(graph_judge)
    
    # Run the bootstrapping process
    success = await bootstrapper.bootstrap_gold_labels(
        triples_file=args.triples_file,
        source_file=args.source_file,
        output_file=args.output
    )
    
    if not success:
        print("❌ Gold label bootstrapping failed.")
        sys.exit(1)


async def run_graph_judgment(args, data_eval):
    """
    Run the graph judgment pipeline (standard or explainable mode).
    
    Args:
        args: Parsed command line arguments
        data_eval: Dataset to process
    """
    # Pre-flight validation checks
    print("🔍 Running pre-flight checks...")
    
    if not validate_input_file():
        print("❌ Pre-flight validation failed. Exiting.")
        sys.exit(1)
    
    create_output_directory(output_file)
    
    # Initialize GraphJudge
    try:
        graph_judge = PerplexityGraphJudge(
            model_name=PERPLEXITY_MODELS[args.model],
            reasoning_effort=args.reasoning_effort,
            enable_console_logging=args.verbose
        )
    except Exception as e:
        print(f"❌ Failed to initialize Perplexity Graph Judge: {e}")
        sys.exit(1)
    
    # Initialize processing pipeline
    pipeline = ProcessingPipeline(graph_judge)
    
    # Run the main processing pipeline with specified mode
    stats = await pipeline.process_instructions(
        data_eval=data_eval,
        explainable_mode=args.explainable,
        reasoning_file_path=args.reasoning_file
    )
    
    print("\n🎉 Graph judgment pipeline completed successfully!")
    print(f"📂 CSV results available at: {output_file}")
    
    if args.explainable:
        reasoning_path = pipeline.generate_reasoning_file_path(output_file, args.reasoning_file)
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


def load_dataset():
    """
    Load the evaluation dataset.
    
    Returns:
        Dataset object for evaluation
    """
    try:
        from datasets import load_dataset
        DATASETS_AVAILABLE = True
    except ImportError:
        print("⚠️ datasets library not found. Please install: pip install datasets")
        print("⚠️ Running in compatibility mode - dataset operations will be mocked")
        DATASETS_AVAILABLE = False
    
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
        
        return data_eval
        
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
        
        return MockDataEval(mock_data)


async def main():
    """
    Main execution function.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up terminal logging first
    terminal_logger = None
    if not args.no_logging:
        try:
            log_filepath = setup_terminal_logging()
            terminal_logger = TerminalLogger(log_filepath)
            terminal_logger.start_session()
            terminal_logger.original_print(f"📝 Logging to: {log_filepath}")
        except Exception as e:
            print(f"⚠️ Failed to set up logging: {e}")
            print("Continuing without file logging...")
    
    # Print header based on mode
    if args.bootstrap:
        print("=" * 70)
        print("🎯 Perplexity API Graph Judge - Gold Label Bootstrapping")
        print("=" * 70)
    elif args.explainable:
        print("=" * 70)
        print("🧠 Perplexity API Graph Judge - Enhanced Explainable Knowledge Graph Validation")
        print("=" * 70)
        print("📋 Mode: Dual-output (CSV + Reasoning JSON)")
    else:
        print("=" * 70)
        print("🎯 Perplexity API Graph Judge - Knowledge Graph Triple Validation")
        print("=" * 70)
        print("📋 Mode: Standard (CSV output only)")
    
    # Environment setup
    if not setup_environment():
        print("❌ Environment setup failed. Exiting.")
        sys.exit(1)
    
    # Run the appropriate pipeline
    try:
        if args.bootstrap:
            await run_gold_label_bootstrapping(args)
        else:
            # Load dataset for graph judgment
            data_eval = load_dataset()
            
            # Run graph judgment with explainable mode if specified
            await run_graph_judgment(args, data_eval)
        
        # End logging session
        if terminal_logger:
            terminal_logger.end_session()
            print(f"📝 Complete log saved to: {terminal_logger.get_log_filepath()}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        if terminal_logger:
            terminal_logger.log_message("Processing interrupted by user", "WARNING")
            terminal_logger.end_session()
    except Exception as e:
        print(f"\n❌ Critical error during processing: {e}")
        print("Please check your configuration and try again.")
        if terminal_logger:
            terminal_logger.log_message(f"Critical error: {e}", "ERROR")
            terminal_logger.end_session()
        sys.exit(1)


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
    asyncio.run(main())
