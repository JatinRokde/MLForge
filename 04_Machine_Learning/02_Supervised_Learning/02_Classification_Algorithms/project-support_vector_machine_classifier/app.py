from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np

from scipy.sparse import hstack


# FastAPI
app = FastAPI(
    title="Incident Severity Prediction API",
    version="1.0.0"
)


MODEL_PATH = "./svm_model.pkl"
TFIDF_PATH = "./tfidf_vectorizer.pkl"
SCALER_PATH = "./scaler.pkl"
LABEL_ENCODER_PATH = "./label_encoder.pkl"

model = joblib.load(MODEL_PATH)

tfidf = joblib.load(TFIDF_PATH)

scaler = joblib.load(SCALER_PATH)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

# Request Schema
class IncidentRequest(BaseModel):
    incident_title: str
    affected_users: int
    cpu_usage: float
    error_rate: float


# Response Schema
class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_severity: str


# Health Endpoint
@app.get("/health")

def health():

    return {
        "status": "healthy"
    }


# Prediction Enpoint
@app.post(
    "/predict",
    response_model=PredictionResponse
)

def predict(payload: IncidentRequest):
    # IF-IDF Features
    text_features = tfidf.transform(
        [
            payload.incident_title
        ]
    )

    # Numerical Features
    numerical_features = scaler.transform(
        [[
            payload.affected_users,
            payload.cpu_usage,
            payload.error_rate
        ]]
    )

    # Combine Features
    final_features = hstack(
        [
            text_features,
            numerical_features
        ]
    )

    # Prediction
    prediction = model.predict(
        final_features
    )[0]

    severity = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return PredictionResponse(
        predicted_class_id=int(prediction),
        predicted_severity=severity
    )