"""
main.py — MaternalGuard FastAPI Backend
----------------------------------------
Endpoints:
    GET  /health   → liveness check, reports whether model is loaded
    GET  /demo     → hardcoded High-risk response (no model needed, useful for frontend dev)
    POST /predict  → real inference via model_loader.ModelService
                     returns HTTP 503 if model.pkl is not present

CORS: all origins allowed (frontend runs on a different port during dev)

DHRUV: your work lives in model_loader.py → ModelService.predict()
"""

from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import FEATURE_ORDER, ModelNotReadyError, model_service


# ─── Pydantic Models ──────────────────────────────────────────────────────────


class PatientVitals(BaseModel):
    """
    Six vitals collected by the ASHA worker.
    Field ranges validated by Pydantic — request is rejected (422) if out of range.
    """
    Age: int = Field(ge=10, le=70, description="Patient age in years (10–70)")
    SystolicBP: int = Field(ge=70, le=200, description="Systolic blood pressure mmHg (70–200)")
    DiastolicBP: int = Field(ge=40, le=150, description="Diastolic blood pressure mmHg (40–150)")
    BloodGlucose: float = Field(ge=6.0, le=20.0, description="Blood glucose mmol/L (6–20)")
    BodyTemp: float = Field(ge=35.0, le=42.0, description="Body temperature °C (35–42)")
    HeartRate: int = Field(ge=40, le=150, description="Heart rate bpm (40–150)")


class PredictionResponse(BaseModel):
    """JSON shape returned by both /demo and /predict — must stay in sync."""
    risk_level: str          # "Low" | "Mid" | "High"
    confidence: float        # 0.0–1.0
    top_reasons: List[str]   # top 3 SHAP feature names / descriptions
    referral: str            # human-readable referral instruction
    referral_color: str      # "green" | "yellow" | "red"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ─── Helpers ─────────────────────────────────────────────────────────────────


def build_features(vitals: PatientVitals) -> pd.DataFrame:
    """Convert validated Pydantic model → DataFrame with the exact column order the model expects."""
    row = {feature: getattr(vitals, feature) for feature in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model at startup. If model.pkl is absent, app still starts — /predict returns 503."""
    model_service.load_model()
    yield


# ─── App ─────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="MaternalGuard API",
    description="AI-powered maternal health risk screener for ASHA workers",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins so the React frontend (localhost:5173) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health_check() -> HealthResponse:
    """Liveness probe. Returns model_loaded: false when model.pkl is absent."""
    return HealthResponse(status="ok", model_loaded=model_service.model_loaded)


@app.get("/demo", response_model=PredictionResponse, tags=["demo"])
def demo_prediction() -> PredictionResponse:
    """
    Hardcoded High-risk response — no model required.
    FRONTEND: use this endpoint to build and test your UI before model is ready.
    """
    return PredictionResponse(
        risk_level="High",
        confidence=0.87,
        top_reasons=[
            "Blood glucose is critically elevated",
            "Systolic BP above safe threshold for age",
            "Age is a contributing risk factor",
        ],
        referral="Send to PHC immediately",
        referral_color="red",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict_risk(vitals: PatientVitals) -> PredictionResponse:
    """
    Run XGBoost inference on patient vitals.
    Returns HTTP 503 if model.pkl has not been loaded yet.

    DHRUV: real logic goes into model_loader.py → ModelService.predict()
    """
    features = build_features(vitals)

    try:
        prediction = model_service.predict(features)
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed — check server logs.") from exc

    return PredictionResponse(**prediction)
