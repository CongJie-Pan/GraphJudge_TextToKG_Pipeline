#!/usr/bin/env python3
"""
KIMI-K2 ECTD Pipeline with Rate Limit Fix

This script runs the KIMI-K2 entity extraction and text denoising pipeline
with the rate limiting fixes applied to handle 3 RPM API limits.

Usage:
    python run_with_rate_limit_fix.py
"""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to run the KIMI-K2 ECTD pipeline with rate limiting.
    """
    print("🚀 Starting KIMI-K2 ECTD Pipeline with Rate Limit Fix")
    print("=" * 60)
    
    try:
        # Import and run the main pipeline
        from run_entity import main as run_pipeline
        
        # Run the pipeline
        asyncio.run(run_pipeline())
        
        print("\n✅ Pipeline completed successfully!")
        print("📊 Check the output files in the datasets directory")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 