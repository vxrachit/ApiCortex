"""Model artifact loading utilities.

Supports both pickle and joblib-serialized model files with automatic fallback.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib


def load_model(model_path: Path) -> Any:
    """Load XGBoost model from disk.
    
    Attempts to load as pickle first (faster), falls back to joblib
    for compatibility with different serialization formats.
    
    Args:
        model_path: Path to model artifact file.
        
    Returns:
        Loaded model object.
        
    Raises:
        FileNotFoundError: If model artifact does not exist.
        Exception: If both pickle and joblib loading fail.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    try:
        with model_path.open("rb") as file_handle:
            return pickle.load(file_handle)
    except Exception:
        
        return joblib.load(model_path)
