"""
Data Preprocessing Pipeline Module for UPI Scam and Fraud Detection
Implements duplicate removal, missing value imputation, chronological train-val-test splitting,
and Scikit-learn Pipeline / ColumnTransformer scaling to prevent data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Standard numerical feature set used for modeling
MODEL_FEATURE_COLUMNS = [
    'amount',
    'user_avg_amount',
    'amount_deviation',
    'amount_ratio_avg',
    'hour_of_day',
    'is_unusual_hour',
    'tx_count_5m',
    'tx_count_30m',
    'tx_count_24h',
    'time_since_prev_tx_min',
    'is_new_recipient',
    'is_device_changed',
    'is_location_changed',
    'failed_attempts',
    'balance_error',
    'account_drain_ratio',
    'is_transfer',
    'is_cash_out'
]

def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw/extracted dataset:
    - Removes duplicate transactions
    - Filters out invalid transaction amounts (e.g. amount <= 0)
    - Replaces infinite values with NaN
    """
    df = df.copy()
    
    # Remove exact duplicate rows or duplicate transaction IDs
    if 'transaction_id' in df.columns:
        df = df.drop_duplicates(subset=['transaction_id'])
    else:
        df = df.drop_duplicates()
        
    # Check invalid amounts
    if 'amount' in df.columns:
        df = df[df['amount'] > 0]
        
    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df.reset_index(drop=True)

def build_preprocessing_pipeline(feature_cols: Optional[List[str]] = None) -> ColumnTransformer:
    """
    Constructs a robust Scikit-learn ColumnTransformer with median imputation
    and RobustScaler (resilient to extreme financial transaction outliers).
    """
    cols = feature_cols if feature_cols is not None else MODEL_FEATURE_COLUMNS
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cols)
        ],
        remainder='drop'
    )
    return preprocessor

def split_chronological(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits transaction data strictly chronologically to evaluate real-world 
    future transaction detection without lookahead bias.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:train_end].copy().reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    df_test = df.iloc[val_end:].copy().reset_index(drop=True)
    
    return df_train, df_val, df_test

def prepare_feature_matrices(
    df: pd.DataFrame, 
    preprocessor: Optional[ColumnTransformer] = None,
    fit: bool = False,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], ColumnTransformer, List[str]]:
    """
    Transforms dataframe into scaled NumPy matrix X and target y (if available).
    Ensures that missing columns in custom datasets are gracefully zero-filled.
    """
    cols = feature_cols if feature_cols is not None else MODEL_FEATURE_COLUMNS
    
    # Ensure all required feature columns exist in df
    df_sub = df.copy()
    for c in cols:
        if c not in df_sub.columns:
            df_sub[c] = 0.0
            
    if preprocessor is None:
        preprocessor = build_preprocessing_pipeline(cols)
        fit = True
        
    if fit:
        X_scaled = preprocessor.fit_transform(df_sub)
    else:
        X_scaled = preprocessor.transform(df_sub)
        
    y = df['is_fraud'].values if 'is_fraud' in df.columns else None
    
    return X_scaled, y, preprocessor, cols

def save_preprocessor(preprocessor: ColumnTransformer, save_path: str):
    """Saves fitted preprocessor pipeline to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

def load_preprocessor(load_path: str) -> ColumnTransformer:
    """Loads fitted preprocessor pipeline from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Preprocessor file not found at: {load_path}")
    return joblib.load(load_path)
