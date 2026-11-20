"""
FastAPI Application Entrypoint for UPI Fraud & Scam Detection Engine.
Powers the React frontend with asynchronous ML, XAI, and behavioral analysis APIs.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routers import dataset, dashboard, prediction, analytics, training, history

app = FastAPI(
    title="UPI Sentinel AI API",
    description="Intelligent UPI Fraud & Scam Detection Backend Engine",
    version="2.0.0"
)

# CORS configuration for modern frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular API routers
app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(history.router, prefix="/api/history", tags=["History"])

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "UPI Sentinel AI (FastAPI)",
        "version": "2.0.0"
    }

# Mount React static build if available
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api"):
            return {"error": "Not Found"}
        target_file = frontend_dist / full_path
        if target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(frontend_dist / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
