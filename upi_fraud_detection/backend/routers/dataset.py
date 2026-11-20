"""
Dataset Router: Ingestion, Automated Schema Inspection, and Sample Downloads.
"""

import os
import io
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from typing import Dict, Any, Optional

from backend.state import state
from src.data_loader import load_dataset
from src.preprocessing import clean_transaction_data, build_preprocessing_pipeline, prepare_feature_matrices, load_preprocessor
from src.feature_engineering import extract_transaction_features
from src.anomaly_detection import predict_anomaly, load_anomaly_model
from src.classification import predict_supervised, load_classifier
from src.pipeline_manager import get_model_paths, ensure_baseline_models_exist

router = APIRouter()

def process_and_activate(df_loaded: pd.DataFrame, meta: Dict[str, Any], source_label: str, filename: Optional[str] = None):
    """Executes cleaning, feature engineering, and baseline scoring."""
    ensure_baseline_models_exist()
    paths = get_model_paths()
    
    df_clean = clean_transaction_data(df_loaded)
    df_feat = extract_transaction_features(df_clean)
    
    iso = load_anomaly_model(paths['isolation_forest'])
    rf = load_classifier(paths['random_forest'])
    preprocessor = load_preprocessor(paths['preprocessor'])
    
    X_scaled, _, _, _ = prepare_feature_matrices(df_feat, preprocessor, fit=False)
    is_ano, ano_scores = predict_anomaly(iso, X_scaled, 'isolation_forest')
    _, rf_probs = predict_supervised(rf, X_scaled)
    
    df_feat['anomaly_flag'] = is_ano
    df_feat['anomaly_score'] = ano_scores
    df_feat['supervised_fraud_prob'] = np.round(rf_probs * 100, 2)
    risk_scores = np.round((rf_probs * 50.0) + (ano_scores * 0.50), 1)
    df_feat['risk_score'] = risk_scores
    
    state.current_df = df_feat
    state.metadata = meta
    state.active_source_label = source_label
    state.active_filename = filename

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")
        
    try:
        content = await file.read()
        csv_file = io.StringIO(content.decode('utf-8', errors='replace'))
        df_loaded, meta = load_dataset(uploaded_file=csv_file)
        
        if len(df_loaded) == 0:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
            
        source_label = f"Custom: {file.filename} ({len(df_loaded):,} records)"
        meta['filename'] = file.filename
        
        process_and_activate(df_loaded, meta, source_label, filename=file.filename)
        
        # Prepare 10 preview records
        preview_cols = [c for c in ['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount', 'transaction_type', 'risk_score', 'supervised_fraud_prob', 'anomaly_score', 'is_fraud'] if c in state.current_df.columns]
        preview = state.current_df[preview_cols].head(10).to_dict(orient='records')
        
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "inspection": meta,
            "preview": preview,
            "total_records": len(state.current_df),
            "columns": list(state.current_df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV dataset: {str(e)}")

@router.post("/load-benchmark")
async def load_benchmark_dataset():
    try:
        df_bench, meta = load_dataset()
        source_label = f"Curated UPI Benchmark ({len(df_bench):,} records)"
        meta['filename'] = "upi_benchmark_transactions.csv"
        
        process_and_activate(df_bench, meta, source_label, filename="upi_benchmark_transactions.csv")
        
        preview_cols = [c for c in ['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount', 'transaction_type', 'risk_score', 'supervised_fraud_prob', 'anomaly_score', 'is_fraud'] if c in state.current_df.columns]
        preview = state.current_df[preview_cols].head(10).to_dict(orient='records')
        
        return {
            "status": "success",
            "message": "Curated UPI Benchmark dataset activated.",
            "inspection": meta,
            "preview": preview,
            "total_records": len(state.current_df),
            "columns": list(state.current_df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading benchmark: {str(e)}")

@router.post("/reset")
async def reset_dataset():
    state.reset_dataset()
    return {"status": "success", "message": "Dataset unloaded."}

@router.get("/status")
async def get_dataset_status():
    if not state.is_dataset_loaded():
        return {
            "is_loaded": False,
            "source_label": state.active_source_label,
            "filename": None,
            "total_records": 0
        }
    return {
        "is_loaded": True,
        "source_label": state.active_source_label,
        "filename": state.active_filename,
        "total_records": len(state.current_df),
        "total_columns": len(state.current_df.columns),
        "inspection": state.metadata
    }

@router.get("/download-sample/{sample_type}")
async def download_sample(sample_type: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if sample_type == "custom":
        path = os.path.join(root_dir, 'data', 'custom_sample_upload.csv')
        filename = "custom_sample_upload.csv"
    elif sample_type == "benchmark":
        path = os.path.join(root_dir, 'data', 'raw', 'upi_benchmark_transactions.csv')
        filename = "upi_benchmark_transactions.csv"
    else:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
        
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File does not exist on disk.")
    return FileResponse(path, media_type='text/csv', filename=filename)

@router.get("/export-scored")
async def export_scored_dataset():
    if not state.is_dataset_loaded():
        raise HTTPException(status_code=400, detail="No active dataset to export.")
    csv_str = state.current_df.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=upi_fraud_scored_dataset.csv"}
    )
