"""
model_loader.py — MaternalGuard ML Model Service
--------------------------------------------------
DHRUV: This file owns model lifecycle (load, predict).
Replace the stub in predict() with your real XGBoost + SHAP logic.

Expected model file: backend/model/model.pkl
Expected input:      pandas DataFrame with columns = FEATURE_ORDER
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_ORDER: List[str] = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BloodGlucose",
    "BodyTemp",
    "HeartRate",
]

MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"

# ─── Exceptions ───────────────────────────────────────────────────────────────


class ModelNotReadyError(RuntimeError):
    """Raised when a prediction is requested but no model is loaded."""


# ─── Model Service ───────────────────────────────────────────────────────────


class ModelService:
    """
    Wrapper around the XGBoost model.

    Lifecycle:
        1. load_model()  is called at app startup (lifespan).
        2. predict(df)   is called per request.

    If model.pkl is absent at startup, the service degrades gracefully:
        - /health returns  model_loaded: false
        - /predict returns HTTP 503
        - /demo  is always available (hardcoded, no model needed)
    """

    def __init__(self) -> None:
        self._model = None
        self._model_loaded = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Attempt to load model.pkl from disk.
        Logs a warning (not an error) if file is absent — app continues running.
        """
        if not MODEL_PATH.exists():
            logger.warning(
                "model.pkl not found at %s. "
                "/predict will return 503 until the model is placed here.",
                MODEL_PATH,
            )
            return

        try:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            self._model_loaded = True
            logger.info("Model loaded successfully from %s", MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load model: %s", exc)
            # Still don't crash — degrade gracefully.

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features) -> Dict[str, Any]:
        """
        Run inference and return a dict matching PredictionResponse.

        Args:
            features: pandas DataFrame with columns matching FEATURE_ORDER.

        Returns:
            dict with keys: risk_level, confidence, top_reasons, referral, referral_color

        Raises:
            ModelNotReadyError: if model.pkl was not loaded at startup.
        """
        if not self._model_loaded:
            raise ModelNotReadyError(
                "Model is not loaded. Place model.pkl in backend/model/ and restart."
            )

        # ── DHRUV: replace everything below with real XGBoost + SHAP logic ──

        # 1. Get raw probabilities from the model
        #    proba = self._model.predict_proba(features)   # shape: (1, 3) → [Low, Mid, High]
        #    risk_index = proba.argmax(axis=1)[0]          # 0=Low, 1=Mid, 2=High
        #    confidence = float(proba[0][risk_index])

        # 2. Map index → label
        #    risk_labels = ["Low", "Mid", "High"]
        #    risk_level  = risk_labels[risk_index]

        # 3. Referral decision
        #    if risk_level == "High":
        #        referral       = "Send to PHC immediately"
        #        referral_color = "red"
        #    elif risk_level == "Mid":
        #        referral       = "Monitor closely — follow up in 48 h"
        #        referral_color = "yellow"
        #    else:
        #        referral       = "No referral needed"
        #        referral_color = "green"

        # 4. SHAP top-3 reasons
        #    explainer   = shap.TreeExplainer(self._model)
        #    shap_values = explainer.shap_values(features)
        #    top_indices = abs(shap_values[risk_index][0]).argsort()[::-1][:3]
        #    top_reasons = [FEATURE_ORDER[i] for i in top_indices]

        # Stub — remove when real logic is added
        raise ModelNotReadyError(
            "predict() stub: real XGBoost + SHAP logic not yet implemented."
        )


# Singleton used by main.py
model_service = ModelService()
