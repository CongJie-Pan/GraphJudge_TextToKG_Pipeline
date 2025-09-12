#!/usr/bin/env python3
"""
Optional Dependency Installation Script for GraphJudge Streamlit Application.

This script helps users install optional dependencies for enhanced features.
"""

import subprocess
import sys
from pathlib import Path

def install_plotly():
    """Install plotly for enhanced visualizations."""
    print("🎨 Installing Plotly for enhanced visualizations...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.0.0"])
        print("✅ Plotly installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Plotly: {e}")
        return False

def check_plotly():
    """Check if plotly is available."""
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__} is already installed")
        return True
    except ImportError:
        print("❌ Plotly is not installed")
        return False

def main():
    """Main installation function."""
    print("🔧 GraphJudge Optional Dependencies Installer")
    print("=" * 50)
    
    print("Checking current installation status:")
    plotly_available = check_plotly()
    
    if not plotly_available:
        print("\n📊 Enhanced Visualization Features:")
        print("- Interactive knowledge graph visualizations")
        print("- Statistical charts and plots")
        print("- Performance trend analysis")
        
        response = input("\nWould you like to install Plotly for enhanced visualizations? (y/N): ")
        if response.lower() in ['y', 'yes']:
            success = install_plotly()
            if success:
                print("\n🎉 Installation complete!")
                print("You can now enjoy full visualization features in the GraphJudge app.")
            else:
                print("\n⚠️  Installation failed.")
                print("You can still use the application with basic table displays.")
        else:
            print("\n📝 Plotly not installed.")
            print("The application will work with basic table displays.")
    else:
        print("\n🎉 All optional dependencies are already installed!")
    
    print("\n🚀 You can now run the application with: streamlit run app.py")

if __name__ == "__main__":
    main()