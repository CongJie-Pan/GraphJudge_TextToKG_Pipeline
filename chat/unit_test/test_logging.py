#!/usr/bin/env python3
"""
Simple test script to verify the terminal logging functionality
簡單測試腳本以驗證終端日誌功能
"""

import os
import sys
from datetime import datetime

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import our logging functionality
from run_gj import setup_terminal_logging, TerminalLogger

def test_logging():
    """Test the logging functionality"""
    print("🧪 Testing terminal logging functionality...")
    
    try:
        # Set up logging
        log_filepath = setup_terminal_logging()
        terminal_logger = TerminalLogger(log_filepath)
        
        print("✓ Terminal logger initialized successfully")
        print(f"📝 Log file: {log_filepath}")
        
        # Test different types of output
        print("Testing regular print statements...")
        print("🎯 This is a test message")
        print("📊 Processing some data...")
        print("✅ Task completed successfully")
        
        # Test logging specific messages
        terminal_logger.log_message("This is an info message", "INFO")
        terminal_logger.log_message("This is a warning message", "WARNING")
        terminal_logger.log_message("This is an error message", "ERROR")
        
        # End the session
        terminal_logger.end_session()
        
        print(f"\n✅ Logging test completed successfully!")
        print(f"📂 Check the log file: {log_filepath}")
        
        # Verify the log file exists and has content
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"📄 Log file size: {len(content)} characters")
            print("📄 Log file contents preview:")
            print("-" * 50)
            lines = content.split('\n')
            for i, line in enumerate(lines[:10]):  # Show first 10 lines
                print(f"{i+1:2d}: {line}")
            if len(lines) > 10:
                print(f"... and {len(lines) - 10} more lines")
            print("-" * 50)
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logging()
