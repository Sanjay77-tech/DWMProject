"""
UPI Sentinel AI - Application Runner
Launches the FastAPI backend and React frontend.

Usage:
    python run.py          # Unified mode: Serves React application + FastAPI API on http://localhost:8000
    python run.py --dev    # Development mode: Launches FastAPI backend (8000) and Vite HMR (5173) in parallel
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_unified():
    print("==================================================================")
    print("🛡️  Starting UPI Sentinel AI (React + FastAPI Engine)...")
    print("   Serving at: http://localhost:8000")
    print("==================================================================")
    
    # Check if frontend is built
    dist_dir = BASE_DIR / "frontend" / "dist"
    if not dist_dir.exists():
        print("[INFO] Building React production bundle first...")
        subprocess.run(["npm.cmd", "run", "build"], cwd=str(BASE_DIR / "frontend"), shell=True)

    # Open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    import threading
    t = threading.Thread(target=open_browser)
    t.daemon = True
    t.start()

    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

def run_dev():
    print("==================================================================")
    print("🛡️  Starting UPI Sentinel AI in Full-Stack DEV Mode...")
    print("   Backend:  http://localhost:8000")
    print("   Frontend: http://localhost:5173")
    print("==================================================================")

    # 1. Start FastAPI backend in subprocess
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(BASE_DIR)
    )

    time.sleep(1.5)

    # 2. Open browser to Vite frontend
    webbrowser.open("http://localhost:5173")

    # 3. Start Vite dev server in foreground
    try:
        subprocess.run(["npm.cmd", "run", "dev"], cwd=str(BASE_DIR / "frontend"), shell=True)
    except KeyboardInterrupt:
        pass
    finally:
        backend_proc.terminate()
        print("\n[INFO] Development servers terminated.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dev":
        run_dev()
    else:
        run_unified()
