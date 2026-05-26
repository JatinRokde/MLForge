from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
import joblib

try:
    model = joblib.load("./severity_model.pkl")
    scaler = joblib.load("./scaler.pkl")

except Exception as e:
    raise Exception(f"Model loading failed: {e}")


# FastAPI App
app = FastAPI(
    title="Telecom Incident Severity Prediction API",
    description="Ordinal Logistic Regression Inference API"
)


# Severity Label Mapping
SEVERITY_MAPPING = {
    1: "Informational",
    2: "Minor",
    3: "Moderate",
    4: "Major",
    5: "Critical"
}

# Request Schema
class IncidentFeatures(BaseModel):
    cpu_usage_percent: float = Field(..., ge=0, le=100)
    memory_usage_percent: float = Field(..., ge=0, le=100)
    packet_drop_rate_percent: float = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    restart_count: int = Field(..., ge=0)
    build_failure_count: int = Field(..., ge=0)
    crash_frequency_per_hour: float = Field(..., ge=0)
    alarm_density_per_minute: float = Field(..., ge=0)
    affected_services_count: int = Field(..., ge=1)
    active_users_impacted: int = Field(..., ge=0)


# Response Schema
class PredictionResponse(BaseModel):
    predicted_severity_level: int
    predicted_severity_label: str


# Health Check
@app.get("/health")
def health():
    return {
        "status": "running"
    }    


# Prediction Endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: IncidentFeatures):

    try:
        # Create DataFrame
        features = pd.DataFrame([{
            "cpu_usage_percent":
                data.cpu_usage_percent,

            "memory_usage_percent":
                data.memory_usage_percent,

            "packet_drop_rate_percent":
                data.packet_drop_rate_percent,

            "latency_ms":
                data.latency_ms,

            "error_count":
                data.error_count,

            "restart_count":
                data.restart_count,

            "build_failure_count":
                data.build_failure_count,

            "crash_frequency_per_hour":
                data.crash_frequency_per_hour,

            "alarm_density_per_minute":
                data.alarm_density_per_minute,

            "affected_services_count":
                data.affected_services_count,

            "active_users_impacted":
                data.active_users_impacted
        }])

        # Feature Scaling
        scaled_features = scaler.transform(features)

        # Prediction
        predicted_class = model.predict(
            scaled_features
        )[0]

        # Response
        return PredictionResponse(
            predicted_severity_level=int(predicted_class),
            predicted_severity_label=
                SEVERITY_MAPPING[int(predicted_class)]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )