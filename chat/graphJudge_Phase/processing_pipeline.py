"""
Processing pipeline for the GraphJudge system.

This module orchestrates the graph judgment evaluation process,
including concurrent processing, result collection, and output generation.
"""

import asyncio
import csv
import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from .config import (
    PERPLEXITY_CONCURRENT_LIMIT, PERPLEXITY_BASE_DELAY, 
    output_file, DEFAULT_ENCODING, CSV_DELIMITER
)
from .graph_judge_core import PerplexityGraphJudge
from .data_structures import ExplainableJudgment, ProcessingResult, ProcessingStatistics
from .prompt_engineering import PromptEngineer


class ProcessingPipeline:
    """
    Orchestrates the graph judgment evaluation process.
    
    This class manages the complete processing pipeline including:
    1. Concurrent request handling with rate limiting
    2. Result collection and error handling
    3. Output generation in multiple formats
    4. Statistics calculation and reporting
    """
    
    def __init__(self, graph_judge: PerplexityGraphJudge):
        """
        Initialize the processing pipeline.
        
        Args:
            graph_judge (PerplexityGraphJudge): Graph judge instance for processing
        """
        self.graph_judge = graph_judge
    
    def generate_reasoning_file_path(self, csv_output_path: str, custom_path: Optional[str] = None) -> str:
        """
        Generate reasoning file path based on CSV output path.
        
        Args:
            csv_output_path (str): Path to the main CSV output file
            custom_path (Optional[str]): Custom reasoning file path if specified
            
        Returns:
            str: Path for the reasoning JSON file
        """
        if custom_path:
            return custom_path
        
        # Auto-generate path based on CSV file name
        path_obj = Path(csv_output_path)
        reasoning_filename = path_obj.stem + "_reasoning" + ".json"
        return str(path_obj.parent / reasoning_filename)
    
    def save_reasoning_file(self, reasoning_results: List[Dict], output_path: str) -> bool:
        """
        Save explainable reasoning results to JSON file.
        將可解釋推理結果保存到JSON文件
        
        Args:
            reasoning_results (List[Dict]): List of structured reasoning results
            output_path (str): Path to save the reasoning file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save reasoning data as formatted JSON
            with open(output_path, 'w', encoding=DEFAULT_ENCODING) as f:
                json.dump(reasoning_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Explainable reasoning results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving reasoning file: {e}")
            return False
    
    async def process_instructions(self, data_eval, explainable_mode: bool = False, 
                                 reasoning_file_path: Optional[str] = None) -> ProcessingStatistics:
        """
        Process instructions with Perplexity API system for graph judgment.
        
        This function orchestrates the entire graph judgment evaluation process:
        1. Creates async tasks for all instruction-input pairs
        2. Executes them with concurrency control (standard or explainable mode)
        3. Collects binary judgment responses and optionally detailed reasoning
        4. Saves results in dual-file format: CSV (compatible) + JSON (explainable)
        
        Args:
            data_eval: Dataset to process
            explainable_mode (bool): Whether to enable explainable reasoning mode
            reasoning_file_path (Optional[str]): Custom path for reasoning file
        
        Returns:
            ProcessingStatistics: Processing statistics and results
        """
        print("🚀 Starting Perplexity API Graph Judge processing...")
        
        if not data_eval or len(data_eval) == 0:
            print("✗ No evaluation data available")
            return ProcessingStatistics(
                total_instructions=0,
                successful_responses=0,
                error_responses=0,
                yes_judgments=0,
                no_judgments=0,
                success_rate=0.0,
                positive_rate=0.0,
                avg_confidence=0.0,
                unique_error_types=0
            )
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(PERPLEXITY_CONCURRENT_LIMIT)
        
        async def limited_completion(item, index):
            """Rate-limited wrapper for graph judgment with optional explainable mode"""
            async with semaphore:
                # Add delay between requests to be respectful to API
                await asyncio.sleep(PERPLEXITY_BASE_DELAY)
                
                instruction = item["instruction"]
                input_text = item.get("input", "")
                
                mode_indicator = "🧠" if explainable_mode else "🔄"
                print(f"{mode_indicator} Processing instruction {index + 1}/{len(data_eval)}")
                
                try:
                    if explainable_mode:
                        # Get explainable judgment with detailed reasoning
                        explainable_result = await self.graph_judge.judge_graph_triple_with_explanation(instruction, input_text)
                        binary_response = explainable_result.judgment
                        
                        # Prepare reasoning data for JSON output
                        reasoning_data = {
                            "index": index,
                            "prompt": instruction,
                            "judgment": explainable_result.judgment,
                            "confidence": explainable_result.confidence,
                            "reasoning": explainable_result.reasoning,
                            "evidence_sources": explainable_result.evidence_sources,
                            "alternative_suggestions": explainable_result.alternative_suggestions,
                            "error_type": explainable_result.error_type,
                            "processing_time": explainable_result.processing_time
                        }
                        
                        print(f"✅ Completed instruction {index + 1}/{len(data_eval)} - Result: {binary_response} (confidence: {explainable_result.confidence:.2f})")
                        return binary_response, reasoning_data
                    else:
                        # Standard mode - get simple binary response
                        response = await self.graph_judge.judge_graph_triple(instruction, input_text)
                        print(f"✅ Completed instruction {index + 1}/{len(data_eval)} - Result: {response}")
                        return response  # Return just the response, not a tuple
                        
                except Exception as e:
                    print(f"❌ Failed instruction {index + 1}/{len(data_eval)} - Error: {str(e)[:100]}...")
                    error_response = "Error: Failed to process"
                    
                    if explainable_mode:
                        # Create error reasoning data
                        error_reasoning = {
                            "index": index,
                            "prompt": instruction,
                            "judgment": error_response,
                            "confidence": 0.0,
                            "reasoning": f"Processing error: {str(e)}",
                            "evidence_sources": [],
                            "alternative_suggestions": [],
                            "error_type": "processing_error",
                            "processing_time": 0.0
                        }
                        return error_response, error_reasoning
                    else:
                        return error_response  # Return just the error response, not a tuple
        
        # Create async tasks for all instruction-input pairs
        tasks = [limited_completion(item, i) for i, item in enumerate(data_eval)]

        # Execute all tasks with progress tracking
        mode_name = "Explainable" if explainable_mode else "Standard"
        print(f"📊 Processing {len(tasks)} graph judgment tasks in {mode_name} mode...")
        print(f"📊 Configuration: Max concurrent requests = {PERPLEXITY_CONCURRENT_LIMIT}")
        print(f"⏱️  Base delay between requests: {PERPLEXITY_BASE_DELAY} seconds")
        if explainable_mode:
            reasoning_output_path = self.generate_reasoning_file_path(output_file, reasoning_file_path)
            print(f"🧠 Reasoning output: {reasoning_output_path}")
        print("-" * 60)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate responses and reasoning data
        responses = []
        reasoning_results = []
        
        for result in results:
            if isinstance(result, Exception):
                responses.append(f"Error: {str(result)}")
                if explainable_mode:
                    reasoning_results.append({
                        "index": len(reasoning_results),
                        "prompt": "Error processing",
                        "judgment": "Error",
                        "confidence": 0.0,
                        "reasoning": f"Exception occurred: {str(result)}",
                        "evidence_sources": [],
                        "alternative_suggestions": [],
                        "error_type": "system_error",
                        "processing_time": 0.0
                    })
            else:
                if explainable_mode:
                    # In explainable mode, result is a tuple (binary_response, reasoning_data)
                    binary_response, reasoning_data = result
                    responses.append(binary_response)
                    reasoning_results.append(reasoning_data)
                else:
                    # In standard mode, result is just the response string
                    responses.append(result)

        # Write responses to CSV file (standard format, compatible with existing pipeline)
        print(f"💾 Saving CSV results to {output_file}...")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Created output directory: {output_dir}")
        
        with open(output_file, "w", newline="", encoding=DEFAULT_ENCODING) as csvfile:
            writer = csv.writer(csvfile, delimiter=CSV_DELIMITER)
            writer.writerow(["prompt", "generated"])  # Standard header format

            # Process each response and ensure proper formatting
            successful_responses = 0
            error_responses = 0
            yes_responses = 0
            no_responses = 0
            
            for item, response in zip(data_eval, responses):
                prompt = item["instruction"]
                
                # Clean response and ensure it matches expected format
                cleaned_response = str(response).strip().replace('\n', ' ')
                
                # Count response types for statistics
                if "Error:" not in cleaned_response:
                    successful_responses += 1
                    if cleaned_response == "Yes":
                        yes_responses += 1
                    elif cleaned_response == "No":
                        no_responses += 1
                else:
                    error_responses += 1
                
                writer.writerow([prompt, cleaned_response])
        
        # Save reasoning file if in explainable mode
        if explainable_mode and reasoning_results:
            print(f"🧠 Saving explainable reasoning to {reasoning_output_path}...")
            self.save_reasoning_file(reasoning_results, reasoning_output_path)
        
        # Calculate and print statistics
        stats = self._calculate_statistics(responses, reasoning_results)
        
        print(f"✅ Perplexity API Graph Judge processing completed!")
        print(f"📊 Results saved to: {output_file}")
        if explainable_mode:
            print(f"🧠 Reasoning saved to: {reasoning_output_path}")
        print(f"📈 Processing statistics:")
        print(f"   - Successful responses: {stats.successful_responses}")
        print(f"   - Error responses: {stats.error_responses}")
        print(f"   - 'Yes' judgments: {stats.yes_judgments}")
        print(f"   - 'No' judgments: {stats.no_judgments}")
        print(f"   - Success rate: {stats.success_rate:.1f}%")
        
        if stats.successful_responses > 0:
            print(f"   - Positive judgment rate: {stats.positive_rate:.1f}%")
        
        if explainable_mode and reasoning_results:
            print(f"🧠 Explainable mode statistics:")
            print(f"   - Average confidence: {stats.avg_confidence:.2f}")
            print(f"   - Unique error types: {stats.unique_error_types}")
            if stats.unique_error_types > 0:
                error_types = [r.get("error_type") for r in reasoning_results if r.get("error_type")]
                unique_error_types_list = list(set(error_types))
                print(f"   - Error types found: {', '.join(unique_error_types_list)}")
        
        return stats
    
    def _calculate_statistics(self, responses: List[str], reasoning_results: List[Dict] = None) -> ProcessingStatistics:
        """
        Calculate statistics from responses and reasoning results.
        
        Args:
            responses (List[str]): List of response strings
            reasoning_results (List[Dict], optional): List of reasoning result dictionaries
            
        Returns:
            ProcessingStatistics: Calculated statistics
        """
        total_instructions = len(responses)
        successful_responses = 0
        error_responses = 0
        yes_responses = 0
        no_responses = 0
        
        # Count response types for statistics
        for response in responses:
            cleaned_response = str(response).strip().replace('\n', ' ')
            
            if "Error:" not in cleaned_response:
                successful_responses += 1
                if cleaned_response == "Yes":
                    yes_responses += 1
                elif cleaned_response == "No":
                    no_responses += 1
            else:
                error_responses += 1
        
        # Calculate basic statistics
        success_rate = successful_responses / total_instructions * 100 if total_instructions > 0 else 0
        positive_rate = yes_responses / successful_responses * 100 if successful_responses > 0 else 0
        
        # Calculate reasoning-specific statistics
        avg_confidence = 0.0
        unique_error_types = 0
        if reasoning_results:
            confidences = [r.get("confidence", 0.0) for r in reasoning_results if r.get("confidence") is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            error_types = [r.get("error_type") for r in reasoning_results if r.get("error_type")]
            unique_error_types = len(set(error_types))
        
        return ProcessingStatistics(
            total_instructions=total_instructions,
            successful_responses=successful_responses,
            error_responses=error_responses,
            yes_judgments=yes_responses,
            no_judgments=no_responses,
            success_rate=success_rate,
            positive_rate=positive_rate,
            avg_confidence=avg_confidence,
            unique_error_types=unique_error_types
        )
    
    def _save_csv_results(self, test_data: List[Dict], responses: List[str], output_file: str) -> None:
        """
        Save CSV results to file.
        
        Args:
            test_data (List[Dict]): List of test data items
            responses (List[str]): List of response strings
            output_file (str): Path to output CSV file
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, "w", newline="", encoding=DEFAULT_ENCODING) as csvfile:
            writer = csv.writer(csvfile, delimiter=CSV_DELIMITER)
            writer.writerow(["prompt", "generated"])  # Standard header format
            
            # Process each response and ensure proper formatting
            for item, response in zip(test_data, responses):
                prompt = item["instruction"]
                cleaned_response = str(response).strip().replace('\n', ' ')
                writer.writerow([prompt, cleaned_response])
    
    def _save_reasoning_results(self, reasoning_results: List[Dict], reasoning_file: str) -> bool:
        """
        Save reasoning results to JSON file.
        
        Args:
            reasoning_results (List[Dict]): List of reasoning result dictionaries
            reasoning_file (str): Path to reasoning JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.save_reasoning_file(reasoning_results, reasoning_file)