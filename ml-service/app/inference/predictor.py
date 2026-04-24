"""Risk scoring inference engine.

Generates API failure predictions with confidence scores and optional
feature importance explanations using SHAP values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.explainability.shap_explainer import ShapExplainer
from app.features.feature_engineering import FEATURE_COLUMNS


@dataclass
class PredictionResult:
    """Result of a single API failure prediction.
    
    Attributes:
        risk_score: Probability of API failure (0.0 to 1.0).
        prediction: Human-readable risk category (normal/degraded/high_failure_risk).
        confidence: How certain the prediction is (0.5 to 1.0).
        top_features: Most impactful features for this prediction (if SHAP enabled).
    """
    risk_score: float
    prediction: str
    confidence: float
    top_features: list[dict[str, float]]


class Predictor:
    """Wrapper around trained XGBoost model for failure risk inference.
    
    Handles feature ordering, probability normalization, confidence calculation,
    and optional SHAP-based feature importance explanation.
    """
    def __init__(self, model: Any, explainer: ShapExplainer | None = None) -> None:
        """Initialize predictor with trained model and optional explainer.
        
        Args:
            model: Trained XGBoost model with predict_proba method.
            explainer: Optional ShapExplainer for feature importance (if None, no explanations).
        """
        self._model = model
        self._explainer = explainer
        model_features = getattr(model, "feature_names_in_", None)
        self._feature_order = list(model_features) if model_features is not None else list(FEATURE_COLUMNS)

    def predict(self, features: dict[str, float]) -> PredictionResult:
        """Generate failure risk prediction for given features.
        
        Args:
            features: Dict mapping feature names to float values.
            
        Returns:
            PredictionResult with risk_score, prediction category, confidence, and optional top_features.
        """
        frame = pd.DataFrame([{name: float(features.get(name, 0.0)) for name in self._feature_order}])

        risk_score = self._predict_probability(frame)
        prediction = self._label_for_score(risk_score)
        confidence = float(max(risk_score, 1.0 - risk_score))

        top_features: list[dict[str, float]] = []
        if self._explainer is not None:
            top_features = self._explainer.explain(frame)

        return PredictionResult(
            risk_score=risk_score,
            prediction=prediction,
            confidence=confidence,
            top_features=top_features,
        )

    def _predict_probability(self, frame: pd.DataFrame) -> float:
        """Extract failure probability from model output, handling both classification modes.
        
        Returns:
            Probability value clipped to [0.0, 1.0] range.
        """
        if hasattr(self._model, "predict_proba"):
            output = self._model.predict_proba(frame)
            probability = float(output[0][1])
        else:
            output = self._model.predict(frame)
            probability = float(np.asarray(output)[0])
        return float(np.clip(probability, 0.0, 1.0))

    @staticmethod
    def _label_for_score(score: float) -> str:
        """Map probability score to human-readable risk category.
        
        Args:
            score: Probability value [0.0, 1.0].
            
        Returns:
            One of: 'normal' (< 0.4), 'degraded' (0.4-0.7), 'high_failure_risk' (>= 0.7).
        """
        if score < 0.4:
            return "normal"
        if score < 0.7:
            return "degraded"
        return "high_failure_risk"
