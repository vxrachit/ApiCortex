"""SHAP-based model explainability for feature importance.

Wraps TreeExplainer to compute feature contributions to predictions,
gracefully handling import/execution errors by logging warnings.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


class ShapExplainer:
    """Computes SHAP feature importance explanations for model predictions.
    
    Lazy-loads SHAP library on first explanation request. Returns top-K features
    ranked by absolute contribution. Handles errors gracefully by logging and
    returning empty explanation list.
    """
    def __init__(self, model: Any, enabled: bool, top_k: int, logger: logging.Logger) -> None:
        """Initialize explainer wrapper.
        
        Args:
            model: XGBoost model object with tree structure.
            enabled: Whether to enable SHAP explanations (allows cheap disable).
            top_k: Number of top features to return per prediction.
            logger: Logger for error/warning messages.
        """
        self._model = model
        self._enabled = enabled
        self._top_k = top_k
        self._logger = logger
        self._explainer: Any | None = None

    def explain(self, feature_frame: pd.DataFrame) -> list[dict[str, float]]:
        """Compute SHAP explanations for model prediction.
        
        Returns top-K features by absolute SHAP value (feature importance).
        Returns empty list if disabled or SHAP library unavailable, logging warning.
        
        Args:
            feature_frame: DataFrame with single row of feature values.
            
        Returns:
            List of dicts {feature: str, contribution: float, abs_contribution: float}
            sorted by abs_contribution descending. Empty list if disabled/unavailable.
        """
        if not self._enabled:
            return []
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self._model)

            shap_values = self._explainer.shap_values(feature_frame)
            if isinstance(shap_values, list):
                # Multi-class output, take the last class (failure prediction)
                values = np.asarray(shap_values[-1])[0]
            else:
                values = np.asarray(shap_values)[0]

            # Get top features by absolute contribution
            abs_values = np.abs(values)
            top_indices = np.argsort(abs_values)[::-1][: self._top_k]
            features = []
            for idx in top_indices:
                feature_name = str(feature_frame.columns[idx])
                contribution = float(values[idx])
                abs_contribution = float(abs_values[idx])
                features.append({
                    "feature": feature_name,
                    "contribution": contribution,
                    "abs_contribution": abs_contribution,
                })
            return features
        except Exception as exc:
            self._logger.warning("SHAP explainability unavailable", extra={"extra": {"error": str(exc)}})
            return []
