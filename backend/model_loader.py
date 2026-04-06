import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


EXPECTED_FEATURE_COLUMNS: List[str] = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
    "PulsePressure",
    "MAP",
    "BPRatio",
    "AgeGlucose",
]
RISK_LABELS = {0: "Low", 1: "Mid", 2: "High"}
REFERRAL_MAP = {
    "High": ("Send to PHC immediately", "red"),
    "Mid": ("Monitor closely, revisit in 48 hours", "amber"),
    "Low": ("No immediate action needed", "green"),
}
REASON_PHRASES = {
    "Age": {
        "high": "Age is a high-risk factor",
        "low": "Age is within safe range",
    },
    "SystolicBP": {
        "high": "Systolic BP is above safe threshold",
        "low": "Systolic BP is normal",
    },
    "DiastolicBP": {
        "high": "Diastolic BP is elevated",
        "low": "Diastolic BP is normal",
    },
    "BS": {
        "high": "Blood glucose is critically elevated",
        "low": "Blood glucose is normal",
    },
    "BodyTemp": {
        "high": "Body temperature is abnormally high",
        "low": "Body temperature is normal",
    },
    "HeartRate": {
        "high": "Heart rate is dangerously elevated",
        "low": "Heart rate is normal",
    },
    "PulsePressure": {
        "high": "Pulse pressure indicates cardiovascular stress",
        "low": "Pulse pressure is normal",
    },
    "MAP": {
        "high": "Mean arterial pressure is critically high",
        "low": "Mean arterial pressure is normal",
    },
    "BPRatio": {
        "high": "Blood pressure ratio indicates hypertension risk",
        "low": "Blood pressure ratio is normal",
    },
    "AgeGlucose": {
        "high": "Combined age and glucose risk is elevated",
        "low": "Age-glucose interaction is normal",
    },
}
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "maternal_risk_xgb_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
EXPLAINER_PATH = MODEL_DIR / "maternal_risk_shap_explainer.pkl"


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is requested before the model artifacts are available."""


class MaternalRiskModelService:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
        explainer_path: Path = EXPLAINER_PATH,
    ):
        self.model_path = model_path
        self.feature_columns_path = feature_columns_path
        self.explainer_path = explainer_path
        self.model = None
        self.feature_columns: Optional[List[str]] = None
        self.explainer = None

    @property
    def model_loaded(self) -> bool:
        return (
            self.model is not None
            and self.feature_columns is not None
            and self.explainer is not None
        )

    def load_model(self) -> bool:
        if not (
            self.model_path.exists()
            and self.feature_columns_path.exists()
            and self.explainer_path.exists()
        ):
            self.model = None
            self.feature_columns = None
            self.explainer = None
            return False

        try:
            with self.model_path.open("rb") as model_file:
                self.model = pickle.load(model_file)
            with self.feature_columns_path.open("rb") as feature_columns_file:
                self.feature_columns = list(pickle.load(feature_columns_file))
            with self.explainer_path.open("rb") as explainer_file:
                self.explainer = pickle.load(explainer_file)
        except Exception:
            self.model = None
            self.feature_columns = None
            self.explainer = None
            return False

        if self.feature_columns != EXPECTED_FEATURE_COLUMNS:
            self.model = None
            self.feature_columns = None
            self.explainer = None
            return False

        return True

    def predict(
        self,
        age: int,
        systolic_bp: int,
        diastolic_bp: int,
        blood_glucose: float,
        body_temp: float,
        heart_rate: int,
    ) -> Dict[str, Any]:
        if not self.model_loaded or not self.feature_columns:
            raise ModelNotReadyError("Model not trained yet")

        features = self._build_features(
            age=age,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            blood_glucose=blood_glucose,
            body_temp=body_temp,
            heart_rate=heart_rate,
        )

        predicted_class = int(np.asarray(self.model.predict(features))[0])
        confidence = self._get_confidence(features, predicted_class)
        top_reasons = self._build_top_reasons(features, predicted_class)
        risk_level = self._get_risk_label(predicted_class)
        referral, referral_color = REFERRAL_MAP[risk_level]

        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "top_reasons": top_reasons,
            "referral": referral,
            "referral_color": referral_color,
        }

    def _build_features(
        self,
        age: int,
        systolic_bp: int,
        diastolic_bp: int,
        blood_glucose: float,
        body_temp: float,
        heart_rate: int,
    ) -> pd.DataFrame:
        pulse_pressure = systolic_bp - diastolic_bp
        map_val = diastolic_bp + (pulse_pressure / 3)
        bp_ratio = systolic_bp / diastolic_bp
        age_glucose = age * blood_glucose

        features = pd.DataFrame(
            [[
                age,
                systolic_bp,
                diastolic_bp,
                blood_glucose,
                body_temp,
                heart_rate,
                pulse_pressure,
                map_val,
                bp_ratio,
                age_glucose,
            ]],
            columns=self.feature_columns,
        )

        return features

    def _get_confidence(self, features: pd.DataFrame, predicted_class: int) -> float:
        probabilities = np.asarray(self.model.predict_proba(features))[0]
        classes = np.asarray(getattr(self.model, "classes_", np.arange(len(probabilities))))
        class_index = np.where(classes == predicted_class)[0]
        target_index = int(class_index[0]) if class_index.size else int(np.argmax(probabilities))
        return round(float(probabilities[target_index]), 2)

    def _build_top_reasons(self, features: pd.DataFrame, predicted_class: int) -> List[str]:
        shap_values = self.explainer.shap_values(features)
        class_contributions = self._extract_class_contributions(shap_values, predicted_class)

        ranked_features = sorted(
            (
                (feature_name, float(class_contributions[index]))
                for index, feature_name in enumerate(self.feature_columns or [])
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        reasons: List[str] = []
        for feature_name, shap_value in ranked_features[:3]:
            direction = "high" if shap_value > 0 else "low"
            reasons.append(REASON_PHRASES[feature_name][direction])

        return reasons

    def _extract_class_contributions(self, shap_values: Any, predicted_class: int) -> np.ndarray:
        if isinstance(shap_values, list):
            return np.asarray(shap_values[predicted_class])[0]

        values = np.asarray(getattr(shap_values, "values", shap_values))

        if values.ndim == 2:
            return values[0]

        if values.ndim == 3:
            if values.shape[0] == 1 and values.shape[1] == len(self.feature_columns or []):
                return values[0, :, predicted_class]
            if values.shape[0] == 1 and values.shape[1] == len(RISK_LABELS):
                return values[0, predicted_class, :]
            if values.shape[0] == len(RISK_LABELS) and values.shape[1] == 1:
                return values[predicted_class, 0, :]

        raise ValueError("Unexpected SHAP value shape")

    def _get_risk_label(self, predicted_class: int) -> str:
        if predicted_class not in RISK_LABELS:
            raise ValueError("Unsupported model output")
        return RISK_LABELS[predicted_class]


model_service = MaternalRiskModelService()
