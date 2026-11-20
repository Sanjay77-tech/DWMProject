"""
Global In-Memory State for UPI Fraud Detection Backend
Maintains the session state for the uploaded dataset, metadata, training jobs, and audit history.
"""

from typing import Optional, Dict, Any, List
import pandas as pd

class AppState:
    def __init__(self):
        self.current_df: Optional[pd.DataFrame] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.active_source_label: str = "None (Awaiting Ingestion)"
        self.active_filename: Optional[str] = None
        self.evaluated_history: List[Dict[str, Any]] = []
        self.training_jobs: Dict[str, Dict[str, Any]] = {}

    def is_dataset_loaded(self) -> bool:
        return self.current_df is not None and len(self.current_df) > 0

    def reset_dataset(self):
        self.current_df = None
        self.metadata = None
        self.active_source_label = "None (Awaiting Ingestion)"
        self.active_filename = None

state = AppState()
