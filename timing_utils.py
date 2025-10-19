"""
Timing utilities for measuring operation performance in the Summarizer workflow.
Provides decorators, context managers, and reporting functionality.
"""

import time
import functools
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import logging

# Global timing storage
_timing_data = {
    "operations": [],
    "start_time": None,
    "end_time": None,
    "total_duration": 0
}

def reset_timing_data():
    """Reset all timing data for a new run."""
    global _timing_data
    _timing_data = {
        "operations": [],
        "start_time": None,
        "end_time": None,
        "total_duration": 0
    }

def start_timing():
    """Start timing for the entire workflow."""
    global _timing_data
    _timing_data["start_time"] = time.time()
    _timing_data["operations"] = []

def end_timing():
    """End timing for the entire workflow."""
    global _timing_data
    _timing_data["end_time"] = time.time()
    if _timing_data["start_time"]:
        _timing_data["total_duration"] = _timing_data["end_time"] - _timing_data["start_time"]

def time_operation(operation_name: str, category: str = "general", details: Optional[Dict[str, Any]] = None):
    """
    Decorator to time a function or method.
    
    Args:
        operation_name: Name of the operation being timed
        category: Category for grouping operations (e.g., "fetch", "process", "ai", "team_analysis")
        details: Additional details to store with the timing data
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                # Store timing data
                timing_entry = {
                    "operation": operation_name,
                    "category": category,
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": "success",
                    "details": details or {}
                }
                _timing_data["operations"].append(timing_entry)
                
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                # Store timing data for failed operation
                timing_entry = {
                    "operation": operation_name,
                    "category": category,
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": "error",
                    "error": str(e),
                    "details": details or {}
                }
                _timing_data["operations"].append(timing_entry)
                
                raise
        return wrapper
    return decorator

@contextmanager
def time_context(operation_name: str, category: str = "general", details: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing code blocks.
    
    Args:
        operation_name: Name of the operation being timed
        category: Category for grouping operations
        details: Additional details to store with the timing data
    """
    start_time = time.time()
    try:
        yield
        end_time = time.time()
        duration = end_time - start_time
        
        # Store timing data
        timing_entry = {
            "operation": operation_name,
            "category": category,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "status": "success",
            "details": details or {}
        }
        _timing_data["operations"].append(timing_entry)
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        # Store timing data for failed operation
        timing_entry = {
            "operation": operation_name,
            "category": category,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "status": "error",
            "error": str(e),
            "details": details or {}
        }
        _timing_data["operations"].append(timing_entry)
        
        raise

def get_timing_summary() -> Dict[str, Any]:
    """Get a comprehensive timing summary of all operations."""
    if not _timing_data["operations"]:
        return {"message": "No timing data available"}
    
    # Calculate category summaries
    category_times = {}
    category_counts = {}
    
    for op in _timing_data["operations"]:
        category = op["category"]
        duration = op["duration"]
        
        if category not in category_times:
            category_times[category] = 0
            category_counts[category] = 0
        
        category_times[category] += duration
        category_counts[category] += 1
    
    # Sort operations by duration (longest first)
    sorted_operations = sorted(_timing_data["operations"], key=lambda x: x["duration"], reverse=True)
    
    # Calculate total time
    total_time = sum(op["duration"] for op in _timing_data["operations"])
    
    summary = {
        "total_workflow_duration": _timing_data["total_duration"],
        "total_operations_time": total_time,
        "operation_count": len(_timing_data["operations"]),
        "category_breakdown": {
            category: {
                "total_time": category_times[category],
                "operation_count": category_counts[category],
                "average_time": category_times[category] / category_counts[category]
            }
            for category in category_times
        },
        "slowest_operations": sorted_operations[:10],  # Top 10 slowest operations
        "all_operations": sorted_operations
    }
    
    return summary

def print_timing_summary():
    """Print a formatted timing summary to console."""
    summary = get_timing_summary()
    
    if "message" in summary:
        print("No timing data available")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE TIMING SUMMARY")
    print("="*80)
    
    # Overall timing
    print(f"Total Workflow Duration: {summary['total_workflow_duration']:.2f} seconds")
    print(f"Total Operations Time: {summary['total_operations_time']:.2f} seconds")
    print(f"Number of Operations: {summary['operation_count']}")
    print()
    
    # Category breakdown
    print("CATEGORY BREAKDOWN:")
    print("-" * 40)
    for category, data in summary['category_breakdown'].items():
        print(f"{category:20} | {data['total_time']:8.2f}s | {data['operation_count']:3d} ops | avg: {data['average_time']:6.2f}s")
    print()
    
    # Slowest operations
    print("SLOWEST OPERATIONS:")
    print("-" * 60)
    for i, op in enumerate(summary['slowest_operations'][:10], 1):
        status_icon = "✅" if op['status'] == 'success' else "❌"
        print(f"{i:2d}. {status_icon} {op['operation']:30} | {op['duration']:8.2f}s | {op['category']}")
        if op['status'] == 'error' and 'error' in op:
            print(f"    Error: {op['error']}")
    print()
    
    # Detailed breakdown by category
    print("DETAILED BREAKDOWN BY CATEGORY:")
    print("-" * 60)
    for category in sorted(summary['category_breakdown'].keys()):
        category_ops = [op for op in summary['all_operations'] if op['category'] == category]
        category_ops.sort(key=lambda x: x['duration'], reverse=True)
        
        print(f"\n{category.upper()}:")
        for op in category_ops:
            status_icon = "✅" if op['status'] == 'success' else "❌"
            print(f"  {status_icon} {op['operation']:25} | {op['duration']:6.2f}s")
            if op['details']:
                for key, value in op['details'].items():
                    print(f"    {key}: {value}")

def save_timing_report(filename: Optional[str] = None):
    """Append timing details to single comprehensive text log file."""
    if not filename:
        filename = "logs/timing_operations.log"
    
    # Ensure logs directory exists
    import os
    os.makedirs("logs", exist_ok=True)
    
    summary = get_timing_summary()
    
    if "message" in summary:
        return filename
    
    # Prepare timing entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write to log file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TIMING SESSION: {timestamp}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Operations: {summary['operation_count']}\n")
        f.write(f"Total Operations Time: {summary['total_operations_time']:.2f}s\n")
        f.write(f"Workflow Duration: {summary['total_workflow_duration']:.2f}s\n")
        f.write(f"\n")
        
        # Category breakdown
        f.write(f"CATEGORY BREAKDOWN:\n")
        f.write(f"{'-'*50}\n")
        for category, data in summary['category_breakdown'].items():
            f.write(f"{category:<20} | {data['total_time']:>8.2f}s | {data['operation_count']:>3} ops | avg: {data['average_time']:>6.2f}s\n")
        f.write(f"\n")
        
        # Slowest operations
        f.write(f"SLOWEST OPERATIONS:\n")
        f.write(f"{'-'*70}\n")
        for i, op in enumerate(summary['slowest_operations'][:15], 1):
            status_icon = "✅" if op['status'] == 'success' else "❌"
            f.write(f"{i:2d}. {status_icon} {op['operation']:<35} | {op['duration']:>8.2f}s | {op['category']}\n")
        f.write(f"\n")
        
        # Detailed operations by category
        f.write(f"DETAILED BREAKDOWN BY CATEGORY:\n")
        f.write(f"{'-'*70}\n")
        for category in sorted(summary['category_breakdown'].keys()):
            category_ops = [op for op in summary['all_operations'] if op['category'] == category]
            category_ops.sort(key=lambda x: x['duration'], reverse=True)
            
            f.write(f"\n{category.upper()}:\n")
            for op in category_ops:
                status_icon = "✅" if op['status'] == 'success' else "❌"
                f.write(f"  {status_icon} {op['operation']:<35} | {op['duration']:>8.2f}s\n")
                if op['details']:
                    for key, value in op['details'].items():
                        f.write(f"    {key}: {value}\n")
        f.write(f"\n")
    
    print(f"Timing details appended to: {filename}")
    return filename

def log_timing_entry(operation_name: str, duration: float, category: str = "general", 
                    status: str = "success", details: Optional[Dict[str, Any]] = None):
    """
    Manually log a timing entry (useful for operations that can't be decorated).
    
    Args:
        operation_name: Name of the operation
        duration: Duration in seconds
        category: Category for grouping
        status: "success" or "error"
        details: Additional details
    """
    timing_entry = {
        "operation": operation_name,
        "category": category,
        "duration": duration,
        "start_time": time.time() - duration,
        "end_time": time.time(),
        "status": status,
        "details": details or {}
    }
    _timing_data["operations"].append(timing_entry)

# Convenience functions for common operations
def time_llm_call(operation_name: str, model_name: str, input_tokens: int, output_tokens: int):
    """Time an LLM API call with token information."""
    return time_context(
        operation_name, 
        "llm_call", 
        {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    )

def time_team_analysis(operation_name: str, teams_detected: int, analysis_type: str):
    """Time a team analysis operation."""
    return time_context(
        operation_name,
        "team_analysis",
        {
            "teams_detected": teams_detected,
            "analysis_type": analysis_type
        }
    )

def time_memory_operation(operation_name: str, operation_type: str, memory_size: int = 0):
    """Time a memory operation."""
    return time_context(
        operation_name,
        "memory",
        {
            "operation_type": operation_type,
            "memory_size": memory_size
        }
    )
