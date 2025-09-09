#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CLI Pipeline Architecture for Knowledge Graph Generation

This module implements the master CLI controller that orchestrates the complete
knowledge graph generation pipeline with interactive iteration management.

Architecture Components:
1. Master CLI Controller with Interactive Iteration Management
2. Dynamic Configuration Management with Iteration Context  
3. Enhanced Progress Monitoring & Recovery with Iteration Tracking
4. Automated Directory Structure Creation

Usage Examples:
    # Interactive mode - prompts for iteration number
    python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt
    
    # Direct iteration specification (non-interactive)
    python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt --iteration 3
    
    # Individual stage execution with iteration auto-detection
    python cli.py run-ectd --parallel-workers 5
    python cli.py run-triple-generation --batch-size 10
    python cli.py run-graph-judge --explainable
    python cli.py run-evaluation --metrics all
    
    # Pipeline status and monitoring
    python cli.py status
    python cli.py logs --tail 100
    python cli.py cleanup --iteration 3

Author: Engineering Team
Date: 2025-01-15
Version: 1.0.0
"""

import asyncio
import argparse
import json
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add current CLI directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import stage managers (using absolute imports for direct script execution)
from iteration_manager import IterationManager
from config_manager import ConfigManager, PipelineConfig
from stage_manager import StageManager
from pipeline_monitor import PipelineMonitor


# PipelineConfig is imported from config_manager.py to avoid duplication


class KGPipeline:
    """
    Master Knowledge Graph Pipeline Controller
    
    This class orchestrates the complete knowledge graph generation workflow
    with interactive iteration management and automated directory setup.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the KG Pipeline.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.current_iteration = None
        self.iteration_path = None
        
        # Initialize managers
        self.iteration_manager = IterationManager()
        self.config_manager = ConfigManager()
        self.monitor = PipelineMonitor()
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self._create_default_config()
        
        # Initialize stage manager
        self.stage_manager = StageManager(self.config)
        
        print(" Unified CLI Pipeline Architecture - Initialized")
        print(f" Base output path: {self.config.base_output_path}")
    
    def _create_default_config(self) -> PipelineConfig:
        """Create default pipeline configuration."""
        return PipelineConfig(
            ectd_config={
                'model': 'gpt5-mini',  # Updated to use GPT-5-mini as primary model
                'fallback_model': 'kimi-k2',
                'force_primary_model': True,
                'temperature': 0.3,
                'batch_size': 20,
                'cache_enabled': True
            },
            triple_generation_config={
                'output_format': 'json',
                'validation_enabled': True,
                'relation_mapping': './config/relation_map.json'
            },
            graph_judge_phase_config={
                'explainable_mode': True,
                'confidence_threshold': 0.7,
                'evidence_sources': ['source_text', 'domain_knowledge']
            },
            evaluation_config={
                'metrics': ['triple_match_f1', 'graph_match_accuracy', 'g_bleu', 'g_rouge', 'g_bert_score'],
                'gold_standard': './datasets/gold_standard.json'
            },
            pipeline_state_config={
                'enable_state_tracking': True,
                'state_persistence': True,
                'checkpoint_interval': 100
            }
        )
    
    def validate_configuration(self) -> bool:
        """
        Validate the complete pipeline configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        print(f"\n🔍 Validating Complete Pipeline Configuration...")
        
        # Validate configuration structure
        config_valid = self.config.validate_model_configuration()
        
        # Validate stage manager configuration
        stage_valid = self.stage_manager.validate_configuration()
        
        # Log configuration summary
        self.config.log_configuration_summary()
        self.stage_manager.log_configuration_summary()
        
        overall_valid = config_valid and stage_valid
        
        if overall_valid:
            print(f"✅ Complete pipeline configuration validation passed")
        else:
            print(f"❌ Pipeline configuration validation failed")
        
        return overall_valid
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive pipeline status report.
        
        Returns:
            Dict[str, Any]: Complete pipeline status
        """
        report = {
            'pipeline': {
                'current_iteration': self.current_iteration,
                'iteration_path': self.iteration_path,
                'config_valid': False
            },
            'stage_manager': {},
            'configuration': {}
        }
        
        # Get stage manager status
        try:
            report['stage_manager'] = self.stage_manager.get_stage_status_report()
        except Exception as e:
            report['stage_manager'] = {'error': str(e)}
        
        # Get configuration status
        try:
            report['pipeline']['config_valid'] = self.validate_configuration()
        except Exception as e:
            report['configuration'] = {'error': str(e)}
        
        return report
    
    def prompt_iteration_number(self) -> int:
        """
        Interactive prompt for iteration number with suggestions.
        
        Returns:
            int: Selected iteration number
        """
        return self.iteration_manager.prompt_with_suggestions()
    
    def setup_iteration_structure(self, iteration: int) -> str:
        """
        Create and setup iteration directory structure.
        
        Args:
            iteration: Iteration number
            
        Returns:
            str: Path to iteration directory
        """
        print(f"\n Setting up Iteration {iteration} structure...")
        iteration_path = self.iteration_manager.create_iteration_structure(iteration)
        
        # Create iteration-specific configuration
        iteration_config_path = os.path.join(iteration_path, "configs", f"iteration{iteration}_config.yaml")
        self.config_manager.create_iteration_config(iteration_config_path, iteration, self.config)
        
        # Create iteration tracking file
        tracking_file = os.path.join(iteration_path, "iteration_info.json")
        self.iteration_manager.create_iteration_tracking(tracking_file, iteration)
        
        return iteration_path
    
    def checkpoint_progress(self, stage_name: str, iteration: int):
        """
        Save checkpoint progress for recovery.
        
        Args:
            stage_name: Name of completed stage
            iteration: Current iteration number
        """
        self.iteration_manager.update_tracking_status(
            self.iteration_path, stage_name, "completed"
        )
        
        # Save pipeline state
        checkpoint_data = {
            'last_completed_stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'config': self.config.__dict__
        }
        
        checkpoint_file = os.path.join(self.iteration_path, "checkpoint.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        print(f" Checkpoint saved: {stage_name} completed")
    
    async def run_pipeline(self, input_file: str, iteration: Optional[int] = None):
        """
        Run the complete pipeline with iteration management.
        
        Args:
            input_file: Path to input file
            iteration: Iteration number (optional, will prompt if not provided)
        """
        # Step 1: Determine iteration number
        if iteration is None:
            iteration = self.prompt_iteration_number()
        
        self.current_iteration = iteration
        
        # Step 2: Setup iteration directory structure
        self.iteration_path = self.setup_iteration_structure(iteration)
        
        # Step 3: Load iteration-specific configuration
        iteration_config_path = os.path.join(self.iteration_path, "configs", f"iteration{iteration}_config.yaml")
        if os.path.exists(iteration_config_path):
            self.config = self.config_manager.load_config(iteration_config_path)
        
        print(f"\n Starting Iteration {iteration} pipeline execution...")
        print(f" Working directory: {self.iteration_path}")
        print(f" Input file: {input_file}")
        
        # Step 4: Check for existing checkpoint
        checkpoint_file = os.path.join(self.iteration_path, "checkpoint.json")
        start_stage = None
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            last_completed = checkpoint_data.get('last_completed_stage')
            if last_completed:
                print(f" Found checkpoint: last completed stage was '{last_completed}'")
                confirm = input("Do you want to resume from checkpoint? (y/N): ").strip().lower()
                if confirm == 'y':
                    start_stage = last_completed
        
        # Step 5: Execute pipeline stages with iteration context
        self.monitor.start_monitoring(iteration, self.iteration_path)
        
        try:
            await self.stage_manager.run_stages(
                input_file=input_file,
                iteration=iteration,
                iteration_path=self.iteration_path,
                start_from_stage=start_stage
            )
            
            print(f"\n Iteration {iteration} pipeline completed successfully!")
            print(f" Results available in: {self.iteration_path}/results/")
            print(f" Reports available in: {self.iteration_path}/reports/")
            
        except Exception as e:
            print(f"\n Pipeline execution failed: {e}")
            self.monitor.log_error(f"Pipeline failure in iteration {iteration}: {e}")
            raise
        finally:
            self.monitor.stop_monitoring()
    
    async def run_stage(self, stage_name: str, **kwargs):
        """
        Run individual pipeline stage.
        
        Args:
            stage_name: Name of stage to run
            **kwargs: Stage-specific arguments
        """
        # Auto-detect iteration if not specified
        iteration = kwargs.get('iteration')
        if not iteration:
            iteration = self.iteration_manager.get_latest_iteration()
            if not iteration:
                iteration = self.prompt_iteration_number()
        
        # Setup iteration path
        if not self.iteration_path:
            self.iteration_path = self.iteration_manager.get_iteration_path(iteration)
            if not os.path.exists(self.iteration_path):
                self.iteration_path = self.setup_iteration_structure(iteration)
        
        print(f"\n Running stage: {stage_name} (Iteration {iteration})")
        
        # Filter out iteration from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'iteration'}
        
        # Execute the specific stage
        await self.stage_manager.run_single_stage(
            stage_name=stage_name,
            iteration=iteration,
            iteration_path=self.iteration_path,
            **filtered_kwargs
        )
        
        # Update progress
        self.checkpoint_progress(stage_name, iteration)
        
        print(f"[SUCCESS] Stage '{stage_name}' completed successfully!")
    
    def show_status(self):
        """Show pipeline status and recent iterations."""
        print(" Pipeline Status Report")
        print("=" * 50)
        
        # Show existing iterations
        iterations = self.iteration_manager.list_existing_iterations()
        if iterations:
            print(f"Existing iterations: {iterations}")
            
            # Show status of latest iteration
            latest = max(iterations)
            latest_path = self.iteration_manager.get_iteration_path(latest)
            tracking_file = os.path.join(latest_path, "iteration_info.json")
            
            if os.path.exists(tracking_file):
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    tracking_data = json.load(f)
                
                print(f"\n Latest Iteration {latest}:")
                print(f"   Status: {tracking_data.get('status', 'Unknown')}")
                print(f"   Created: {tracking_data.get('created_at', 'Unknown')}")
                print(f"   Last Updated: {tracking_data.get('last_updated', 'Unknown')}")
                
                completed_stages = tracking_data.get('stages_completed', [])
                if completed_stages:
                    print(f"   Completed Stages: {len(completed_stages)}")
                    for stage in completed_stages[-3:]:  # Show last 3
                        print(f"     - {stage.get('stage')}: {stage.get('completed_at')}")
        else:
            print("No iterations found.")
        
        # Show configuration
        print(f"\n Current Configuration:")
        print(f"   Base output path: {self.config.base_output_path}")
        print(f"   Parallel workers: {self.config.parallel_workers}")
        print(f"   Error tolerance: {self.config.error_tolerance}")
    
    def show_logs(self, tail: int = 50, iteration: Optional[int] = None):
        """
        Show recent logs.
        
        Args:
            tail: Number of recent log lines to show
            iteration: Specific iteration to show logs for
        """
        if not iteration:
            iterations = self.iteration_manager.list_existing_iterations()
            if iterations:
                iteration = max(iterations)
            else:
                print("No iterations found.")
                return
        
        iteration_path = self.iteration_manager.get_iteration_path(iteration)
        logs_dir = os.path.join(iteration_path, "logs")
        
        if not os.path.exists(logs_dir):
            print(f"No logs found for iteration {iteration}")
            return
        
        print(f" Recent logs for Iteration {iteration} (last {tail} lines):")
        print("=" * 60)
        
        # Find most recent log file
        log_files = []
        for root, dirs, files in os.walk(logs_dir):
            for file in files:
                if file.endswith('.log') or file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    log_files.append((file_path, os.path.getmtime(file_path)))
        
        if log_files:
            # Sort by modification time and get most recent
            log_files.sort(key=lambda x: x[1], reverse=True)
            latest_log = log_files[0][0]
            
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-tail:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"Error reading log file: {e}")
        else:
            print("No log files found.")
    
    def cleanup_iteration(self, iteration: int, confirm: bool = False):
        """
        Clean up iteration directory.
        
        Args:
            iteration: Iteration number to clean up
            confirm: Whether to confirm deletion
        """
        iteration_path = self.iteration_manager.get_iteration_path(iteration)
        
        if not os.path.exists(iteration_path):
            print(f"Iteration {iteration} directory not found.")
            return
        
        if not confirm:
            confirm = input(f"Are you sure you want to delete Iteration {iteration}? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cleanup cancelled.")
                return
        
        import shutil
        try:
            shutil.rmtree(iteration_path)
            print(f" Iteration {iteration} cleaned up successfully.")
        except Exception as e:
            print(f" Error cleaning up iteration {iteration}: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified CLI Pipeline Architecture for Knowledge Graph Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - prompts for iteration number
  python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt
  
  # Direct iteration specification
  python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt --iteration 3
  
  # Individual stages
  python cli.py run-ectd --parallel-workers 5
  python cli.py run-triple-generation --batch-size 10
  python cli.py run-graph-judge --explainable
  
  # Configuration validation and debugging
  python cli.py validate-config
  python cli.py validate-config --config ./config/custom_config.yaml
  python cli.py status-report
  python cli.py status-report --output report.json
  
  # Monitoring
  python cli.py status
  python cli.py logs --tail 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run pipeline command
    pipeline_parser = subparsers.add_parser('run-pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--input', required=True, help='Input file path')
    pipeline_parser.add_argument('--iteration', type=int, help='Iteration number (optional)')
    pipeline_parser.add_argument('--config', help='Configuration file path')
    
    # Run individual stages
    ectd_parser = subparsers.add_parser('run-ectd', help='Run ECTD stage')
    ectd_parser.add_argument('--parallel-workers', type=int, default=5, help='Number of parallel workers')
    ectd_parser.add_argument('--iteration', type=int, help='Iteration number')
    
    triple_parser = subparsers.add_parser('run-triple-generation', help='Run triple generation stage')
    triple_parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    triple_parser.add_argument('--iteration', type=int, help='Iteration number')
    
    judge_parser = subparsers.add_parser('run-graph-judge', help='Run graph judge stage')
    judge_parser.add_argument('--explainable', action='store_true', help='Enable explainable mode')
    judge_parser.add_argument('--iteration', type=int, help='Iteration number')
    
    eval_parser = subparsers.add_parser('run-evaluation', help='Run evaluation stage')
    eval_parser.add_argument('--metrics', default='all', help='Metrics to compute')
    eval_parser.add_argument('--iteration', type=int, help='Iteration number')
    
    # Monitoring commands
    subparsers.add_parser('status', help='Show pipeline status')
    
    # Configuration validation command
    validate_parser = subparsers.add_parser('validate-config', help='Validate pipeline configuration')
    validate_parser.add_argument('--config', help='Configuration file path to validate')
    
    # Detailed status report command
    report_parser = subparsers.add_parser('status-report', help='Generate detailed status report')
    report_parser.add_argument('--output', help='Output file for status report (JSON format)')
    
    logs_parser = subparsers.add_parser('logs', help='Show recent logs')
    logs_parser.add_argument('--tail', type=int, default=50, help='Number of recent lines to show')
    logs_parser.add_argument('--iteration', type=int, help='Specific iteration to show logs for')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up iteration directory')
    cleanup_parser.add_argument('--iteration', type=int, required=True, help='Iteration to clean up')
    cleanup_parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize pipeline
    config_path = getattr(args, 'config', None)
    pipeline = KGPipeline(config_path)
    
    try:
        if args.command == 'run-pipeline':
            await pipeline.run_pipeline(args.input, args.iteration)
        
        elif args.command == 'run-ectd':
            await pipeline.run_stage('ectd', 
                                   parallel_workers=args.parallel_workers,
                                   iteration=args.iteration)
        
        elif args.command == 'run-triple-generation':
            await pipeline.run_stage('triple_generation',
                                   batch_size=args.batch_size,
                                   iteration=args.iteration)
        
        elif args.command == 'run-graph-judge':
            await pipeline.run_stage('graph_judge',
                                   explainable=args.explainable,
                                   iteration=args.iteration)
        
        elif args.command == 'run-evaluation':
            await pipeline.run_stage('evaluation',
                                   metrics=args.metrics,
                                   iteration=args.iteration)
        
        elif args.command == 'status':
            pipeline.show_status()
        
        elif args.command == 'validate-config':
            # Validate configuration with detailed reporting
            config_path = getattr(args, 'config', None)
            if config_path:
                # Load and validate specific config file
                test_config = pipeline.config_manager.load_config(config_path)
                valid = test_config.validate_model_configuration() if hasattr(test_config, 'validate_model_configuration') else True
                print(f"Configuration file validation: {'✅ PASSED' if valid else '❌ FAILED'}")
            else:
                # Validate current pipeline configuration
                valid = pipeline.validate_configuration()
                print(f"Pipeline configuration validation: {'✅ PASSED' if valid else '❌ FAILED'}")
        
        elif args.command == 'status-report':
            # Generate detailed status report
            report = pipeline.get_status_report()
            
            if args.output:
                # Save to file
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Status report saved to: {args.output}")
            else:
                # Print to console
                print(f"\n📊 Detailed Pipeline Status Report:")
                print(json.dumps(report, indent=2, ensure_ascii=False))
        
        elif args.command == 'logs':
            pipeline.show_logs(args.tail, args.iteration)
        
        elif args.command == 'cleanup':
            pipeline.cleanup_iteration(args.iteration, args.confirm)
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
