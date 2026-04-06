from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import ModelNotReadyError, model_service


class PatientVitals(BaseModel):
    Age: int = Field(ge=5, le=65)
    SystolicBP: int = Field(ge=70, le=200)
    DiastolicBP: int = Field(ge=40, le=150)
    BloodGlucose: float = Field(ge=6, le=20)
    BodyTemp: float = Field(ge=35, le=42)
    HeartRate: int = Field(ge=40, le=150)


class PredictionResponse(BaseModel):
    risk_level: str
    confidence: float
    top_reasons: List[str]
    referral: str
    referral_color: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load_model()
    yield


app = FastAPI(
    title="MaternalGuard API",
    description="AI-powered maternal health risk screener for ASHA workers",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_service.model_loaded)


@app.get("/demo", response_model=PredictionResponse)
def demo_prediction() -> PredictionResponse:
    return PredictionResponse(
        risk_level="High",
        confidence=0.87,
        top_reasons=[
            "Blood glucose is critically elevated",
            "Systolic BP above safe threshold",
            "Age is a contributing risk factor",
        ],
        referral="Send to PHC immediately",
        referral_color="red",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(vitals: PatientVitals) -> PredictionResponse:
    try:
        prediction = model_service.predict(
            age=vitals.Age,
            systolic_bp=vitals.SystolicBP,
            diastolic_bp=vitals.DiastolicBP,
            blood_glucose=vitals.BloodGlucose,
            body_temp=vitals.BodyTemp,
            heart_rate=vitals.HeartRate,
        )
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return PredictionResponse(**prediction)
