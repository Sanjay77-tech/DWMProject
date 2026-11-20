"""
Convenience Runner Script for UPI Scam and Fraud Detection System
Allows running: python run.py
"""

import os
import sys
import subprocess

def main():
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    print("==================================================================")
    print("🛡️  Starting UPI Fraud & Scam Detection Engine (Streamlit GUI)...")
    print(f"Target App: {app_path}")
    print("==================================================================")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] Application gracefully stopped.")

if __name__ == "__main__":
    main()
