#!/usr/bin/env python3
"""
Multi-Cloud AI Inference Optimizer Launcher
============================================

This script launches the multi-cloud dashboard that compares AWS, Azure, GCP, and Oracle Cloud.

Usage:
    python run_multi_cloud.py
    streamlit run app/multi_cloud_dashboard.py
"""

import subprocess
import sys
import os

def main():
    """Launch the multi-cloud dashboard."""
    print("🚀 Launching Multi-Cloud AI Inference Optimizer...")
    print("☁️  Comparing AWS, Azure, GCP, and Oracle Cloud")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app/multi_cloud_dashboard.py"):
        print("❌ Error: multi_cloud_dashboard.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    try:
        # Launch the dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/multi_cloud_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 