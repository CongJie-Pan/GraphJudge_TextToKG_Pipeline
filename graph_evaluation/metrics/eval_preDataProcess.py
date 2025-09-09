"""
Graph Evaluation Data Preprocessing Pipeline

This module provides comprehensive preprocessing functionality for converting CSV data
to the pred.txt and gold.txt formats required by the GraphJudge evaluation framework.

Key Features:
1. CSV to pred.txt conversion: Transforms model predictions into evaluation format
2. Pred.txt to gold.txt conversion: Creates ground truth files based on model answers
3. Integrated pipeline: One-stop solution for all preprocessing needs
4. Robust error handling: Graceful handling of malformed data
5. Comprehensive logging: Detailed processing information for debugging

Usage:
    # Command line usage
    python eval_preDataProcess.py --csv input.csv --output_dir examples/
    
    # Programmatic usage
    from eval_preDataProcess import process_csv_to_evaluation_files
    process_csv_to_evaluation_files("input.csv", "output_dir/")

Authors: AI Assistant (Google Engineer Standards)
Created: 2024
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import ast

# Configure logging with detailed format for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for consistent processing
PROMPT_PREFIX = "Is this true:"
NULL_TRIPLE = [["Null", "Null", "Null"]]
SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "gbk"]

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class TripleParsingError(DataProcessingError):
    """Exception raised when triple parsing fails."""
    pass

def parse_triple_from_prompt(prompt: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract (subject, relation, object) from a prompt string.
    
    Expected format: "Is this true: <subject> <relation> <object> ?"
    
    Args:
        prompt: Input prompt string to parse
        
    Returns:
        Tuple of (subject, relation, object) if parsing succeeds, None otherwise
        
    Raises:
        TripleParsingError: If prompt format is invalid
        
    Examples:
        >>> parse_triple_from_prompt("Is this true: 僧 行為 見士隱抱英蓮大哭 ?")
        ('僧', '行為', '見士隱抱英蓮大哭')
    """
    if not isinstance(prompt, str):
        logger.warning(f"Invalid prompt type: {type(prompt)}")
        return None

    text = prompt.strip()

    # Remove English prefix if present (tolerant parsing)
    if text.startswith(PROMPT_PREFIX):
        text = text[len(PROMPT_PREFIX):].strip()

    # Normalize punctuation and whitespace variants
    # Handle both ASCII ? and full-width ？ and collapse all unicode spaces
    if text.endswith("?") or text.endswith("？"):
        text = text[:-1].strip()

    # Replace a wide range of unicode whitespace with a single space
    # Includes NBSP (\u00A0), narrow NBSP (\u202F), en/em/quads (\u2000-\u200B)
    space_class = re.compile(r"[\s\u00A0\u202F\u2000-\u200B]+")
    text = space_class.sub(" ", text).strip()

    # Split into tokens; relation is the penultimate token, object is last
    tokens = text.split(" ")
    if len(tokens) < 3:
        logger.debug(f"Failed to parse triple (too few tokens) from: '{text}'")
        return None

    relation = tokens[-2].strip()
    obj = tokens[-1].strip()
    subject = " ".join(tokens[:-2]).strip()

    if not subject or not relation or not obj:
        logger.debug(
            f"Empty component in triple after normalization: subject='{subject}', relation='{relation}', object='{obj}'"
        )
        return None

    return subject, relation, obj

def normalize_generated_response(value: str) -> Optional[bool]:
    """
    Normalize generated response to boolean value.
    
    Args:
        value: Response string to normalize
        
    Returns:
        True for "Yes", False for "No", None for unknown/invalid responses
        
    Examples:
        >>> normalize_generated_response("Yes")
        True
        >>> normalize_generated_response("no")
        False
        >>> normalize_generated_response("maybe")
        None
    """
    if not isinstance(value, str):
        return None
        
    normalized = value.strip().lower()
    if normalized == "yes":
        return True
    elif normalized == "no":
        return False
    else:
        logger.warning(f"Unknown generated response: '{value}'")
        return None

def convert_csv_to_pred_txt(csv_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Convert CSV file to pred.txt format required by eval.py.
    
    This function processes a CSV file with 'prompt' and 'generated' columns,
    extracting triples from prompts and converting model responses to the
    standardized format expected by the evaluation framework.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path for output pred.txt file
        
    Returns:
        Dictionary containing processing statistics
        
    Raises:
        DataProcessingError: If CSV format is invalid or file processing fails
        FileNotFoundError: If input CSV file doesn't exist
        
    Processing Logic:
        - Extract triple from ALL prompts regardless of Yes/No answer
        - If triple parsing succeeds → [[subject, relation, object]]
        - If parsing fails → [["Null", "Null", "Null"]]
        
    This preserves all original triple extractions before the review process.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Processing statistics
    stats = {
        "total_rows": 0,
        "yes_responses": 0,
        "no_responses": 0,
        "unknown_responses": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "null_placeholders": 0
    }
    
    # Try different encodings for robust file reading
    csv_content = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                csv_content = f.read()
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if csv_content is None:
        raise DataProcessingError(f"Failed to read CSV with any supported encoding: {SUPPORTED_ENCODINGS}")
    
    # Process CSV content
    from io import StringIO
    csv_file = StringIO(csv_content)
    
    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(csv_file)
        
        # Validate CSV format
        if "prompt" not in reader.fieldnames or "generated" not in reader.fieldnames:
            raise DataProcessingError(
                f"CSV must contain 'prompt' and 'generated' columns. Found: {reader.fieldnames}"
            )
        
        for row_idx, row in enumerate(reader, 1):
            stats["total_rows"] += 1
            
            prompt = row.get("prompt", "")
            generated = row.get("generated", "")
            
            # Normalize response
            response_bool = normalize_generated_response(generated)
            
            # Update response statistics
            if response_bool is True:
                stats["yes_responses"] += 1
            elif response_bool is False:
                stats["no_responses"] += 1
            else:
                stats["unknown_responses"] += 1
            
            # Process triple extraction from ALL prompts (regardless of Yes/No answer)
            triple = parse_triple_from_prompt(prompt)
                
            # Generate output line - always try to extract the triple
            if triple is not None:
                line_obj = [[triple[0], triple[1], triple[2]]]
                stats["successful_extractions"] += 1
                logger.debug(f"Row {row_idx}: Extracted triple {triple}")
            else:
                line_obj = NULL_TRIPLE
                stats["failed_extractions"] += 1
                stats["null_placeholders"] += 1
                logger.warning(f"Row {row_idx}: Failed to extract triple from prompt: '{prompt}'")
            
            # Write line with proper JSON formatting for ast.literal_eval compatibility
            f_out.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
    
    logger.info(f"CSV to pred.txt conversion completed: {output_path}")
    logger.info(f"Processing statistics: {stats}")
    
    return stats

def convert_pred_to_gold_txt(pred_path: Path, gold_path: Path, csv_path: Path) -> Dict[str, int]:
    """
    Convert pred.txt to gold.txt based on original CSV responses.
    
    This function creates a gold standard file where:
    - "Yes" responses: Keep the extracted triple as ground truth
    - "No" responses: Mark as [["Null", "Null", "Null"]] (incorrect/false)
    
    This represents the "审核" (review/validation) process where the model
    has judged which triples are correct vs incorrect.
    
    Args:
        pred_path: Path to pred.txt file
        gold_path: Path for output gold.txt file  
        csv_path: Path to original CSV (used to determine true/false labels)
        
    Returns:
        Dictionary containing processing statistics
        
    Raises:
        DataProcessingError: If file formats are inconsistent
        
    Note:
        This function assumes "Yes" responses in the CSV represent ground truth.
        In practice, you should manually review and correct the gold.txt file
        to ensure accuracy against domain knowledge.
    """
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred.txt file not found: {pred_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Ensure output directory exists
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_lines": 0,
        "true_labels": 0,
        "false_labels": 0,
        "format_errors": 0
    }
    
    # Read CSV to get original responses
    csv_responses = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_responses.append(row.get("generated", ""))
    
    # Read pred.txt lines
    pred_lines = []
    with pred_path.open("r", encoding="utf-8") as f:
        pred_lines = [line.strip() for line in f.readlines()]
    
    # Validate line count consistency
    if len(csv_responses) != len(pred_lines):
        raise DataProcessingError(
            f"Line count mismatch: CSV has {len(csv_responses)} rows, "
            f"pred.txt has {len(pred_lines)} lines"
        )
    
    # Generate gold.txt
    with gold_path.open("w", encoding="utf-8", newline="") as f_out:
        for idx, (csv_response, pred_line) in enumerate(zip(csv_responses, pred_lines)):
            stats["total_lines"] += 1
            
            # Determine if this should be treated as ground truth
            response_bool = normalize_generated_response(csv_response)
            
            try:
                # Parse the pred.txt line to get the actual triple
                pred_obj = ast.literal_eval(pred_line)
                
                if response_bool is True:
                    # Keep the actual triple as gold standard
                    gold_obj = pred_obj
                    stats["true_labels"] += 1
                    logger.debug(f"Line {idx+1}: Keeping as gold truth: {pred_obj}")
                else:
                    # Mark as false/null
                    gold_obj = NULL_TRIPLE
                    stats["false_labels"] += 1
                    logger.debug(f"Line {idx+1}: Marking as false: {pred_obj} -> {NULL_TRIPLE}")
                
            except (ValueError, SyntaxError) as e:
                # Handle malformed pred.txt lines
                logger.error(f"Line {idx+1}: Failed to parse pred.txt line: '{pred_line}' - {e}")
                gold_obj = NULL_TRIPLE
                stats["format_errors"] += 1
            
            # Write gold line
            f_out.write(json.dumps(gold_obj, ensure_ascii=False) + "\n")
    
    logger.info(f"Pred.txt to gold.txt conversion completed: {gold_path}")
    logger.info(f"Processing statistics: {stats}")
    
    return stats

def process_csv_to_evaluation_files(
    csv_path: Union[str, Path], 
    output_dir: Union[str, Path],
    pred_filename: str = "pred.txt",
    gold_filename: str = "gold.txt"
) -> Dict[str, any]:
    """
    Complete pipeline: Convert CSV to both pred.txt and gold.txt files.
    
    This is the main entry point for preprocessing CSV data for GraphJudge evaluation.
    It orchestrates the entire conversion process and provides comprehensive statistics.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory for output files
        pred_filename: Name for prediction file (default: "pred.txt")
        gold_filename: Name for gold standard file (default: "gold.txt")
        
    Returns:
        Dictionary containing complete processing statistics and file paths
        
    Example:
        >>> results = process_csv_to_evaluation_files(
        ...     "data.csv", 
        ...     "output/",
        ...     "predictions.txt",
        ...     "gold_standard.txt"
        ... )
        >>> print(f"Created files: {results['output_files']}")
    """
    # Convert paths to Path objects
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    
    logger.info(f"Starting CSV to evaluation files conversion")
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    pred_path = output_dir / pred_filename
    gold_path = output_dir / gold_filename
    
    results = {
        "input_file": str(csv_path),
        "output_dir": str(output_dir),
        "output_files": {
            "pred_txt": str(pred_path),
            "gold_txt": str(gold_path)
        },
        "processing_stats": {}
    }
    
    try:
        # Step 1: Convert CSV to pred.txt
        logger.info("Step 1: Converting CSV to pred.txt...")
        pred_stats = convert_csv_to_pred_txt(csv_path, pred_path)
        results["processing_stats"]["pred_conversion"] = pred_stats
        
        # Step 2: Convert pred.txt to gold.txt
        logger.info("Step 2: Converting pred.txt to gold.txt...")
        gold_stats = convert_pred_to_gold_txt(pred_path, gold_path, csv_path)
        results["processing_stats"]["gold_conversion"] = gold_stats
        
        # Summary statistics
        total_rows = pred_stats["total_rows"]
        success_rate = pred_stats["successful_extractions"] / total_rows if total_rows > 0 else 0
        
        results["summary"] = {
            "total_processed": total_rows,
            "extraction_success_rate": round(success_rate, 4),
            "files_created": 2,
            "ready_for_evaluation": True
        }
        
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total rows processed: {total_rows}")
        logger.info(f"Extraction success rate: {success_rate:.2%}")
        logger.info(f"Pred.txt file: {pred_path}")
        logger.info(f"Gold.txt file: {gold_path}")
        logger.info("\n⚠️  IMPORTANT: Please manually review gold.txt to ensure accuracy!")
        logger.info("   The gold.txt file was generated based on model 'Yes' responses.")
        logger.info("   You should validate it against domain knowledge before evaluation.")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        results["error"] = str(e)
        results["summary"] = {
            "total_processed": 0,
            "extraction_success_rate": 0.0,
            "files_created": 0,
            "ready_for_evaluation": False
        }
        raise
    
    return results

def main() -> None:
    """Command line interface for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Graph Evaluation Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python eval_preDataProcess.py --csv data.csv --output_dir examples/
  
  # Custom filenames
  python eval_preDataProcess.py --csv data.csv --output_dir examples/ \\
    --pred_filename my_pred.txt --gold_filename my_gold.txt
  
  # Verbose logging
  python eval_preDataProcess.py --csv data.csv --output_dir examples/ --verbose

Note: After processing, manually review the gold.txt file for accuracy!
        """
    )
    
    parser.add_argument(
        "--csv", 
        required=True, 
        type=str,
        help="Path to input CSV file with 'prompt' and 'generated' columns"
    )
    
    parser.add_argument(
        "--output_dir", 
        required=True, 
        type=str,
        help="Directory for output pred.txt and gold.txt files"
    )
    
    parser.add_argument(
        "--pred_filename",
        default="pred.txt",
        type=str,
        help="Filename for prediction file (default: pred.txt)"
    )
    
    parser.add_argument(
        "--gold_filename",
        default="gold.txt", 
        type=str,
        help="Filename for gold standard file (default: gold.txt)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Run preprocessing pipeline
        results = process_csv_to_evaluation_files(
            csv_path=args.csv,
            output_dir=args.output_dir,
            pred_filename=args.pred_filename,
            gold_filename=args.gold_filename
        )
        
        print("\n✅ Preprocessing completed successfully!")
        print(f"📁 Output files:")
        print(f"   • Pred.txt: {results['output_files']['pred_txt']}")
        print(f"   • Gold.txt: {results['output_files']['gold_txt']}")
        print(f"📊 Summary: {results['summary']}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
