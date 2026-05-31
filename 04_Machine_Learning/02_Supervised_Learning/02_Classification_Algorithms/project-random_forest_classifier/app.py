from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI app
app = FastAPI(
    title="Telecom Network Fault Prediction API",
    version="1.0.0",
    description="Random Forest based Network Failure Prediction API"
)



model = joblib.load( "./random_forest_model.pkl")

feature_columns = joblib.load("./feature_columns.pkl")


# Request Schema
class FaultPredictionRequest(BaseModel):
    cpu_usage: float
    memory_usage: float
    temperature: float
    packet_loss: float
    latency: float
    active_users: int
    handover_failure_rate: float
    critical_alarm_count: int
    major_alarm_count: int
    power_voltage: float
    network_throughput: float


# Response Schema
class FaultPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    failure_probability: float
    healthy_probability: float
    risk_level: str
    class_probabilities: Dict[str, float]


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Predict Endpoint
@app.post(
    "/predict",
    response_model=FaultPredictionResponse
)
def predict(
    request: FaultPredictionRequest
):

    input_df = pd.DataFrame(
        [
            request.model_dump()
        ]
    )

    input_df = input_df[
        feature_columns
    ]

    prediction = int(
        model.predict(input_df)[0]
    )

    probabilities = model.predict_proba(
        input_df
    )[0]

    healthy_probability = float(
        probabilities[0]
    )

    failure_probability = float(
        probabilities[1]
    )

    prediction_label = (
        "Failure Expected"
        if prediction == 1
        else "Healthy"
    )

    if failure_probability >= 0.80:
        risk_level = "Critical"

    elif failure_probability >= 0.60:
        risk_level = "High"

    elif failure_probability >= 0.30:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return FaultPredictionResponse(
        prediction=prediction,
        prediction_label=prediction_label,

        failure_probability=round(
            failure_probability,
            4
        ),

        healthy_probability=round(
            healthy_probability,
            4
        ),

        risk_level=risk_level,

        class_probabilities={
            "healthy": round(
                healthy_probability,
                4
            ),
            "failure": round(
                failure_probability,
                4
            )
        }
    )