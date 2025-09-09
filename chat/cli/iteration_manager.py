#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteration Manager for Unified CLI Pipeline Architecture

This module handles iteration directory creation, tracking, and management
for the knowledge graph generation pipeline.

Features:
- Interactive iteration number prompting with suggestions
- Automatic directory structure creation
- Iteration tracking and status management
- Checkpoint and recovery support

Author: Engineering Team  
Date: 2025-01-15
Version: 1.0.0
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class IterationManager:
    """
    Manages iteration lifecycle for the KG pipeline.
    
    This class handles creating, tracking, and managing iterations including
    directory structure setup, progress tracking, and recovery mechanisms.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the iteration manager.
        
        Args:
            base_path: Base path for iterations (defaults to standard location)
        """
        if base_path is None:
            # Default to the standard iteration report location
            current_dir = Path(__file__).parent
            # Go up to GraphJudge directory (2 levels: cli -> chat -> GraphJudge)
            graphjudge_dir = current_dir.parent.parent
            self.base_path = graphjudge_dir / "docs" / "Iteration_Report"
        else:
            self.base_path = Path(base_path)
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        print(f" Iteration Manager initialized: {self.base_path}")
    
    def list_existing_iterations(self) -> List[int]:
        """
        List all existing iteration directories.
        
        Returns:
            List of iteration numbers found
        """
        iterations = []
        
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir() and item.name.startswith("Iteration"):
                    try:
                        iteration_num = int(item.name.replace("Iteration", ""))
                        iterations.append(iteration_num)
                    except ValueError:
                        continue
        
        return sorted(iterations)
    
    def prompt_with_suggestions(self) -> int:
        """
        Prompt user with existing iterations and suggestions.
        
        Returns:
            Selected iteration number
        """
        existing_iterations = self.list_existing_iterations()
        
        print(f"\n Existing iterations found: {existing_iterations}")
        
        if existing_iterations:
            suggested_next = max(existing_iterations) + 1
            print(f" Suggested next iteration: {suggested_next}")
            
            while True:
                try:
                    response = input(f"Enter iteration number (or press Enter for {suggested_next}): ").strip()
                    
                    if not response:  # User pressed Enter
                        return suggested_next
                    
                    iteration_num = int(response)
                    if iteration_num > 0:
                        if iteration_num in existing_iterations:
                            confirm = input(f" Iteration {iteration_num} already exists. Continue? (y/N): ").strip().lower()
                            if confirm == 'y':
                                return iteration_num
                            else:
                                continue
                        return iteration_num
                    else:
                        print(" Iteration number must be positive.")
                except ValueError:
                    print(" Please enter a valid integer.")
                except KeyboardInterrupt:
                    print("\n Operation cancelled by user.")
                    sys.exit(0)
        else:
            print(" No existing iterations found. Starting fresh.")
            while True:
                try:
                    iteration = input("Enter iteration number (default: 1): ").strip()
                    if not iteration:
                        return 1
                    iteration_num = int(iteration)
                    if iteration_num > 0:
                        return iteration_num
                    else:
                        print(" Iteration number must be positive.")
                except ValueError:
                    print(" Please enter a valid integer.")
                except KeyboardInterrupt:
                    print("\n Operation cancelled by user.")
                    sys.exit(0)
    
    def create_iteration_structure(self, iteration: int) -> str:
        """
        Create comprehensive iteration directory structure.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Path to the created iteration directory
        """
        iteration_path = self.base_path / f"Iteration{iteration}"
        
        # Directory structure based on improvement_plan2.md specifications
        directory_structure = {
            "results": [
                "ectd", "triple_generation", "graph_judge", "evaluation"
            ],
            "logs": [
                "pipeline", "stages", "errors", "performance"
            ],
            "configs": [],
            "reports": [
                "summary", "analysis", "comparison"
            ],
            "analysis": [
                "charts", "statistics", "error_analysis"
            ],
            "backups": []
        }
        
        # Create main iteration directory
        iteration_path.mkdir(parents=True, exist_ok=True)
        print(f" Created main directory: {iteration_path}")
        
        # Create subdirectories
        for main_dir, subdirs in directory_structure.items():
            main_path = iteration_path / main_dir
            main_path.mkdir(exist_ok=True)
            print(f"  +-- {main_dir}/")
            
            for subdir in subdirs:
                sub_path = main_path / subdir
                sub_path.mkdir(exist_ok=True)
                print(f"  |   +-- {subdir}/")
        
        return str(iteration_path)
    
    def create_iteration_tracking(self, tracking_path: str, iteration: int):
        """
        Create iteration tracking information file.
        
        Args:
            tracking_path: Path to tracking file
            iteration: Iteration number
        """
        tracking_info = {
            "iteration": iteration,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "stages_completed": [],
            "last_updated": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "base_path": f"./Iteration{iteration}",
            "directory_structure": {
                "results": ["ectd", "triple_generation", "graph_judge", "evaluation"],
                "logs": ["pipeline", "stages", "errors", "performance"],
                "configs": [],
                "reports": ["summary", "analysis", "comparison"],
                "analysis": ["charts", "statistics", "error_analysis"],
                "backups": []
            }
        }
        
        with open(tracking_path, 'w', encoding='utf-8') as f:
            json.dump(tracking_info, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Created iteration tracking: {tracking_path}")
    
    def update_tracking_status(self, iteration_path: str, stage_name: str, status: str):
        """
        Update iteration tracking with stage completion status.
        
        Args:
            iteration_path: Path to iteration directory
            stage_name: Name of the stage
            status: Status to update
        """
        tracking_file = os.path.join(iteration_path, "iteration_info.json")
        
        if not os.path.exists(tracking_file):
            print(f" Tracking file not found: {tracking_file}")
            return
        
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                tracking_info = json.load(f)
            
            # Add stage completion record
            tracking_info["stages_completed"].append({
                "stage": stage_name,
                "status": status,
                "completed_at": datetime.now().isoformat()
            })
            
            # Update general status
            if status == "completed":
                tracking_info["status"] = "in_progress"
            elif status == "failed":
                tracking_info["status"] = "failed"
            
            tracking_info["last_updated"] = datetime.now().isoformat()
            
            # Save updated tracking info
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_info, f, indent=2, ensure_ascii=False)
            
            print(f" Updated tracking: {stage_name} -> {status}")
            
        except Exception as e:
            print(f" Error updating tracking status: {e}")
    
    def get_iteration_path(self, iteration: int) -> str:
        """
        Get path to iteration directory.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Path to iteration directory
        """
        return str(self.base_path / f"Iteration{iteration}")
    
    def get_latest_iteration(self) -> Optional[int]:
        """
        Get the latest iteration number.
        
        Returns:
            Latest iteration number or None if no iterations exist
        """
        iterations = self.list_existing_iterations()
        return max(iterations) if iterations else None
    
    def get_iteration_status(self, iteration: int) -> Dict[str, Any]:
        """
        Get detailed status of an iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Dictionary with iteration status information
        """
        iteration_path = self.get_iteration_path(iteration)
        tracking_file = os.path.join(iteration_path, "iteration_info.json")
        
        if not os.path.exists(tracking_file):
            return {
                "exists": False,
                "error": "Tracking file not found"
            }
        
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                tracking_info = json.load(f)
            
            # Add some computed information
            tracking_info["exists"] = True
            tracking_info["directory_exists"] = os.path.exists(iteration_path)
            
            # Count completed stages
            completed_stages = [s for s in tracking_info.get("stages_completed", []) 
                              if s.get("status") == "completed"]
            tracking_info["completed_stage_count"] = len(completed_stages)
            
            return tracking_info
            
        except Exception as e:
            return {
                "exists": True,
                "error": f"Error reading tracking file: {e}"
            }
    
    def cleanup_iteration(self, iteration: int) -> bool:
        """
        Clean up an iteration directory.
        
        Args:
            iteration: Iteration number to clean up
            
        Returns:
            True if successful, False otherwise
        """
        iteration_path = Path(self.get_iteration_path(iteration))
        
        if not iteration_path.exists():
            print(f" Iteration {iteration} directory not found.")
            return False
        
        try:
            import shutil
            shutil.rmtree(iteration_path)
            print(f" Iteration {iteration} cleaned up successfully.")
            return True
        except Exception as e:
            print(f" Error cleaning up iteration {iteration}: {e}")
            return False
    
    def list_iteration_files(self, iteration: int) -> Dict[str, List[str]]:
        """
        List files in an iteration directory organized by subdirectory.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Dictionary mapping subdirectory names to lists of files
        """
        iteration_path = Path(self.get_iteration_path(iteration))
        
        if not iteration_path.exists():
            return {}
        
        file_listing = {}
        
        for subdir in iteration_path.iterdir():
            if subdir.is_dir():
                files = []
                try:
                    for file_path in subdir.rglob('*'):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(subdir)
                            files.append(str(relative_path))
                    file_listing[subdir.name] = sorted(files)
                except Exception as e:
                    file_listing[subdir.name] = [f"Error: {e}"]
        
        return file_listing
