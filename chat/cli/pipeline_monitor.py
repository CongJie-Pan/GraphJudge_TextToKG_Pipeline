#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Monitor for Unified CLI Pipeline Architecture

This module provides comprehensive monitoring and logging capabilities
for the knowledge graph generation pipeline with real-time progress tracking.

Features:
- Real-time progress tracking and logging
- Performance metrics collection
- Resource usage monitoring
- Error detection and alerting
- Comprehensive reporting

Author: Engineering Team
Date: 2025-01-15
Version: 1.0.0
"""

import json
import os
import psutil
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class StageMetrics:
    """Stage execution metrics."""
    stage_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "pending"
    error_message: Optional[str] = None
    performance_samples: List[PerformanceMetrics] = None
    
    def __post_init__(self):
        if self.performance_samples is None:
            self.performance_samples = []


@dataclass
class PipelineMetrics:
    """Complete pipeline execution metrics."""
    iteration: int
    pipeline_start_time: str
    pipeline_end_time: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    status: str = "running"
    stages: List[StageMetrics] = None
    overall_performance: List[PerformanceMetrics] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = []
        if self.overall_performance is None:
            self.overall_performance = []
        if self.errors is None:
            self.errors = []


class PipelineMonitor:
    """
    Comprehensive pipeline monitoring and logging system.
    
    This class provides real-time monitoring of pipeline execution including
    performance metrics, resource usage, and progress tracking.
    """
    
    def __init__(self):
        """Initialize the pipeline monitor."""
        self.is_monitoring = False
        self.monitor_thread = None
        self.current_metrics = None
        self.iteration_path = None
        self.log_file_path = None
        
        # Performance monitoring settings
        self.monitor_interval = 5.0  # seconds
        self.max_samples = 1000  # maximum performance samples to keep
        
        # Initialize system monitoring
        self.process = psutil.Process()
        self.initial_io = psutil.disk_io_counters()
        self.initial_net = psutil.net_io_counters()
        
        print(" Pipeline Monitor initialized")
    
    def start_monitoring(self, iteration: int, iteration_path: str):
        """
        Start monitoring pipeline execution.
        
        Args:
            iteration: Current iteration number
            iteration_path: Path to iteration directory
        """
        self.iteration_path = iteration_path
        
        # Create monitoring log file
        logs_dir = os.path.join(iteration_path, "logs", "performance")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(logs_dir, f"pipeline_monitor_{timestamp}.log")
        
        # Initialize metrics
        self.current_metrics = PipelineMetrics(
            iteration=iteration,
            pipeline_start_time=datetime.now().isoformat()
        )
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Log startup
        self._log_message("INFO", f"Pipeline monitoring started for Iteration {iteration}")
        self._log_message("INFO", f"Monitoring interval: {self.monitor_interval}s")
        self._log_message("INFO", f"Log file: {self.log_file_path}")
        
        print(f" Started monitoring for Iteration {iteration}")
        print(f" Monitor log: {self.log_file_path}")
    
    def stop_monitoring(self):
        """
        Stop monitoring pipeline execution.
        """
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        # Finalize metrics
        if self.current_metrics:
            self.current_metrics.pipeline_end_time = datetime.now().isoformat()
            
            # Calculate total duration
            start_time = datetime.fromisoformat(self.current_metrics.pipeline_start_time)
            end_time = datetime.fromisoformat(self.current_metrics.pipeline_end_time)
            self.current_metrics.total_duration_seconds = (end_time - start_time).total_seconds()
            
            # Save final metrics
            self._save_metrics()
        
        self._log_message("INFO", "Pipeline monitoring stopped")
        print(" Pipeline monitoring stopped")
    
    def log_stage_start(self, stage_name: str):
        """
        Log the start of a pipeline stage.
        
        Args:
            stage_name: Name of the stage
        """
        if not self.current_metrics:
            return
        
        stage_metrics = StageMetrics(
            stage_name=stage_name,
            start_time=datetime.now().isoformat(),
            status="running"
        )
        
        self.current_metrics.stages.append(stage_metrics)
        
        self._log_message("INFO", f"Stage started: {stage_name}")
        print(f" Monitor: {stage_name} stage started")
    
    def log_stage_end(self, stage_name: str, success: bool, error_message: Optional[str] = None):
        """
        Log the completion of a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            success: Whether the stage completed successfully
            error_message: Error message if the stage failed
        """
        if not self.current_metrics:
            return
        
        # Find the corresponding stage metrics
        stage_metrics = None
        for stage in self.current_metrics.stages:
            if stage.stage_name == stage_name and stage.end_time is None:
                stage_metrics = stage
                break
        
        if not stage_metrics:
            # Create new stage metrics if not found (shouldn't happen normally)
            stage_metrics = StageMetrics(
                stage_name=stage_name,
                start_time=datetime.now().isoformat(),
                status="unknown"
            )
            self.current_metrics.stages.append(stage_metrics)
        
        # Update stage completion info
        stage_metrics.end_time = datetime.now().isoformat()
        stage_metrics.status = "completed" if success else "failed"
        stage_metrics.error_message = error_message
        
        # Calculate duration
        start_time = datetime.fromisoformat(stage_metrics.start_time)
        end_time = datetime.fromisoformat(stage_metrics.end_time)
        stage_metrics.duration_seconds = (end_time - start_time).total_seconds()
        
        # Log completion
        status_msg = "completed successfully" if success else f"failed: {error_message}"
        self._log_message("INFO", f"Stage {stage_name} {status_msg} in {stage_metrics.duration_seconds:.1f}s")
        
        status_prefix = "SUCCESS" if success else "FAILED"
        print(f"[{status_prefix}] Monitor: {stage_name} stage {status_msg}")
        
        # Save intermediate metrics
        self._save_metrics()
    
    def log_error(self, error_message: str):
        """
        Log an error message.
        
        Args:
            error_message: Error message to log
        """
        if self.current_metrics:
            self.current_metrics.errors.append({
                "timestamp": datetime.now().isoformat(),
                "message": error_message
            })
        
        self._log_message("ERROR", error_message)
        print(f" Monitor Error: {error_message}")
    
    def get_current_metrics(self) -> Optional[PipelineMetrics]:
        """
        Get current pipeline metrics.
        
        Returns:
            Current pipeline metrics or None if not monitoring
        """
        return self.current_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.current_metrics or not self.current_metrics.overall_performance:
            return {}
        
        performance_data = self.current_metrics.overall_performance
        
        # Calculate statistics
        cpu_values = [p.cpu_percent for p in performance_data]
        memory_values = [p.memory_percent for p in performance_data]
        memory_mb_values = [p.memory_used_mb for p in performance_data]
        
        summary = {
            "total_samples": len(performance_data),
            "monitoring_duration_minutes": len(performance_data) * self.monitor_interval / 60,
            "cpu_percent": {
                "min": min(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0
            },
            "memory_percent": {
                "min": min(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0
            },
            "memory_used_mb": {
                "min": min(memory_mb_values) if memory_mb_values else 0,
                "max": max(memory_mb_values) if memory_mb_values else 0,
                "avg": sum(memory_mb_values) / len(memory_mb_values) if memory_mb_values else 0
            }
        }
        
        return summary
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                if self.current_metrics:
                    self.current_metrics.overall_performance.append(metrics)
                    
                    # Limit the number of samples to prevent memory issues
                    if len(self.current_metrics.overall_performance) > self.max_samples:
                        self.current_metrics.overall_performance = \
                            self.current_metrics.overall_performance[-self.max_samples:]
                
                # Wait for next interval
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self._log_message("ERROR", f"Monitoring loop error: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            current_io = psutil.disk_io_counters()
            disk_read_mb = (current_io.read_bytes - self.initial_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (current_io.write_bytes - self.initial_io.write_bytes) / (1024 * 1024)
            
            # Network I/O
            current_net = psutil.net_io_counters()
            net_sent_mb = (current_net.bytes_sent - self.initial_net.bytes_sent) / (1024 * 1024)
            net_recv_mb = (current_net.bytes_recv - self.initial_net.bytes_recv) / (1024 * 1024)
            
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb
            )
            
        except Exception as e:
            # Return zero metrics if collection fails
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
    
    def _log_message(self, level: str, message: str):
        """
        Log a message to the monitoring log file.
        
        Args:
            level: Log level (INFO, ERROR, WARNING)
            message: Message to log
        """
        if not self.log_file_path:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f" Failed to write to monitor log: {e}")
    
    def _save_metrics(self):
        """Save current metrics to JSON file."""
        if not self.current_metrics or not self.iteration_path:
            return
        
        try:
            # Create metrics directory
            metrics_dir = os.path.join(self.iteration_path, "analysis", "statistics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Save metrics to JSON file
            metrics_file = os.path.join(metrics_dir, "pipeline_metrics.json")
            
            # Convert to dictionary for JSON serialization
            metrics_dict = asdict(self.current_metrics)
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self._log_message("ERROR", f"Failed to save metrics: {e}")
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Formatted performance report as string
        """
        if not self.current_metrics:
            return "No metrics available"
        
        summary = self.get_performance_summary()
        
        report = []
        report.append("=" * 60)
        report.append(" PIPELINE PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Basic info
        report.append(f"Iteration: {self.current_metrics.iteration}")
        report.append(f"Start Time: {self.current_metrics.pipeline_start_time}")
        if self.current_metrics.pipeline_end_time:
            report.append(f"End Time: {self.current_metrics.pipeline_end_time}")
            report.append(f"Total Duration: {self.current_metrics.total_duration_seconds:.1f}s")
        
        # Stage summary
        report.append(f"\n STAGE SUMMARY:")
        for stage in self.current_metrics.stages:
            status_prefix = "[DONE]" if stage.status == "completed" else "[FAIL]" if stage.status == "failed" else "[RUNNING]"
            duration_str = f" ({stage.duration_seconds:.1f}s)" if stage.duration_seconds else ""
            report.append(f"  {status_prefix} {stage.stage_name}: {stage.status}{duration_str}")
        
        # Performance summary
        if summary:
            report.append(f"\nPERFORMANCE SUMMARY:")
            report.append(f"  Monitoring Duration: {summary['monitoring_duration_minutes']:.1f} minutes")
            report.append(f"  CPU Usage: {summary['cpu_percent']['avg']:.1f}% (min: {summary['cpu_percent']['min']:.1f}%, max: {summary['cpu_percent']['max']:.1f}%)")
            report.append(f"  Memory Usage: {summary['memory_percent']['avg']:.1f}% (min: {summary['memory_percent']['min']:.1f}%, max: {summary['memory_percent']['max']:.1f}%)")
            report.append(f"  Average Memory: {summary['memory_used_mb']['avg']:.1f} MB")
        
        # Errors
        if self.current_metrics.errors:
            report.append(f"\n ERRORS ({len(self.current_metrics.errors)}):")
            for error in self.current_metrics.errors[-5:]:  # Show last 5 errors
                report.append(f"  - {error['timestamp']}: {error['message']}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
