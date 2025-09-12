#!/usr/bin/env python3
"""
Startup script for GraphJudge Streamlit Application.

This script provides a convenient way to launch the Streamlit application
with proper path configuration and error handling.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main startup function."""
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    parent_dir = script_dir.parent
    
    # Add parent to Python path for imports
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Change to the streamlit_pipeline directory
    os.chdir(script_dir)
    
    print("🧠 Starting GraphJudge Streamlit Application...")
    print(f"Working directory: {script_dir}")
    print("=" * 50)
    
    try:
        # Check if streamlit is available
        subprocess.run([sys.executable, "-c", "import streamlit"], 
                      check=True, capture_output=True)
        
        # Launch the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except subprocess.CalledProcessError:
        print("❌ Error: Streamlit is not installed.")
        print("Please install dependencies: pip install -r requirements.txt")
        return 1
    except FileNotFoundError:
        print("❌ Error: app.py not found in current directory.")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())