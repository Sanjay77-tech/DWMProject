"""
Model Registry Module for UPI Sentinel AI.
Manages persistent model artifacts, paired preprocessors, metadata schemas,
active model selection, and model versioning.
"""

import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
REGISTRY_FILE = os.path.join(MODELS_DIR, 'model_registry.json')
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, 'active_model.json')

def get_models_dir() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR

def sanitize_serializable(obj: Any) -> Any:
    """Recursively converts NumPy types and floats into JSON-serializable primitives."""
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): sanitize_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_serializable(v) for v in obj]
    return obj

class ModelRegistry:
    def __init__(self):
        get_models_dir()
        self._ensure_registry_file()

    def _ensure_registry_file(self):
        if not os.path.exists(REGISTRY_FILE):
            initial_data = {
                "active_model_id": "random_forest",
                "models": {}
            }
            with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)

    def load_registry(self) -> Dict[str, Any]:
        self._ensure_registry_file()
        try:
            with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"active_model_id": "random_forest", "models": {}}

    def save_registry(self, data: Dict[str, Any]):
        with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(sanitize_serializable(data), f, indent=2)

    def register_model(
        self,
        model_id: str,
        algorithm_name: str,
        paradigm: str,
        model_obj: Any,
        preprocessor_obj: Any,
        feature_names: List[str],
        hyperparameters: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        set_active: bool = True
    ) -> Dict[str, Any]:
        """
        Saves the model artifact, its matched preprocessor pipeline, and its metadata.
        Optionally sets it as the active model.
        """
        models_dir = get_models_dir()
        model_filename = f"{model_id}.pkl"
        preprocessor_filename = f"{model_id}_preprocessor.pkl"

        model_path = os.path.join(models_dir, model_filename)
        preprocessor_path = os.path.join(models_dir, preprocessor_filename)

        # 1. Save model artifact
        joblib.dump(model_obj, model_path)

        # 2. Save matched preprocessor
        joblib.dump(preprocessor_obj, preprocessor_path)

        # 3. Create metadata entry
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "model_id": model_id,
            "algorithm": algorithm_name,
            "paradigm": paradigm,
            "model_file": model_filename,
            "preprocessor_file": preprocessor_filename,
            "trained_at": now_str,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "hyperparameters": sanitize_serializable(hyperparameters),
            "metrics": sanitize_serializable(metrics) if metrics else {},
            "version": f"v_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        # 4. Update registry
        reg = self.load_registry()
        reg["models"][model_id] = entry
        if set_active:
            reg["active_model_id"] = model_id
            self._save_active_meta(entry)

        self.save_registry(reg)
        return entry

    def _save_active_meta(self, entry: Dict[str, Any]):
        with open(ACTIVE_MODEL_FILE, 'w', encoding='utf-8') as f:
            json.dump(sanitize_serializable(entry), f, indent=2)

    def get_active_model_id(self) -> str:
        reg = self.load_registry()
        return reg.get("active_model_id", "random_forest")

    def set_active_model(self, model_id: str) -> bool:
        """Switches the active model if its files exist."""
        reg = self.load_registry()
        if model_id not in reg["models"]:
            return False

        entry = reg["models"][model_id]
        models_dir = get_models_dir()
        model_path = os.path.join(models_dir, entry["model_file"])
        prep_path = os.path.join(models_dir, entry["preprocessor_file"])

        if not (os.path.exists(model_path) and os.path.exists(prep_path)):
            return False

        reg["active_model_id"] = model_id
        self.save_registry(reg)
        self._save_active_meta(entry)
        return True

    def load_model_and_preprocessor(self, model_id: Optional[str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Loads the specified model (or active model) and its exactly paired preprocessor.
        Raises FileNotFoundError if artifacts are missing.
        """
        reg = self.load_registry()
        target_id = model_id if model_id and model_id in reg["models"] else reg.get("active_model_id")

        if not target_id or target_id not in reg["models"]:
            # Fallback to random_forest if present
            if "random_forest" in reg["models"]:
                target_id = "random_forest"
            else:
                raise FileNotFoundError(f"Model '{target_id}' not found in registry. Please train a model first.")

        entry = reg["models"][target_id]
        models_dir = get_models_dir()
        model_path = os.path.join(models_dir, entry["model_file"])
        prep_path = os.path.join(models_dir, entry["preprocessor_file"])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}. Retraining required.")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Matched preprocessor file missing: {prep_path}. Retraining required.")

        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

        return model, preprocessor, entry

    def list_models(self) -> List[Dict[str, Any]]:
        reg = self.load_registry()
        active_id = reg.get("active_model_id")
        result = []
        models_dir = get_models_dir()

        for mid, meta in reg.get("models", {}).items():
            model_path = os.path.join(models_dir, meta.get("model_file", ""))
            prep_path = os.path.join(models_dir, meta.get("preprocessor_file", ""))
            exists = os.path.exists(model_path) and os.path.exists(prep_path)

            item = dict(meta)
            item["is_active"] = (mid == active_id)
            item["files_exist"] = exists
            result.append(item)

        return result

# Global registry instance
registry = ModelRegistry()
