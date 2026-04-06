from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MaternalGuard API", description="AI-powered maternal health risk screener")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # FRONTEND: replace this with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientVitals(BaseModel):
    Age: int
    SystolicBP: int
    DiastolicBP: int
    BloodGlucose: float
    BodyTemp: float
    HeartRate: int

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_risk(vitals: PatientVitals):
    """
    Predicts maternal health risk based on 6 vitals.
    """
    logger.info(f"Received vitals: {vitals}")
    
    # DHRUV: load model and replace this with real prediction logic
    # 1. Load XGBoost model from backend/model/model.pkl
    # 2. Preprocess data as needed (pandas dataframe)
    # 3. Model prediction
    # 4. Generate SHAP values for top 3 reasons

    # Placeholder hardcoded response
    return {
        "risk_level": "Low",  # Possible values: "Low", "Mid", "High"
        "shap_reasons": [
            "Normal Blood Glucose levels",
            "Optimal Systolic BP for age",
            "Stable Heart Rate"
        ],
        "referral_needed": False,
        "message": "Patient vitals are within normal range."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
