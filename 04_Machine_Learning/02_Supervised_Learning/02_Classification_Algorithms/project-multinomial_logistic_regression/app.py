from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pandas as pd
import joblib

model = joblib.load(
    "multinomial_logistic_regression_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

app = FastAPI(
    title="5G Network Issue Classification API",
    description="Multinomial Logistic Regression Inference API"
)

class_map = {
    0: "Normal Session",
    1: "Congestion Issue",
    2: "Signal Weakness",
    3: "Packet Loss / Interference"
}

class NetworkInput(BaseModel):
    session_duration_sec: float
    avg_latency_ms: float
    packet_loss_percent: float
    signal_strength_dbm: float
    throughput_mbps: float
    jitter_ms: float
    handover_count: int
    connected_users: int
    cpu_utilization_percent: float


@app.get("/")
def home():
    return {
        "message": "5G Network Issue Classification API Running"
    }

@app.post("/predict")

def predict(data: NetworkInput):
    input_data = pd.DataFrame([{
        "session_duration_sec": data.session_duration_sec,
        "avg_latency_ms": data.avg_latency_ms,
        "packet_loss_percent": data.packet_loss_percent,
        "signal_strength_dbm": data.signal_strength_dbm,
        "throughput_mbps": data.throughput_mbps,
        "jitter_ms": data.jitter_ms,
        "handover_count": data.handover_count,
        "connected_users": data.connected_users,
        "cpu_utilization_percent": data.cpu_utilization_percent
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    probability_output = {
        class_map[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    return {
        "predicted_class": int(prediction),
        "predicted_label": class_map[prediction],
        "probabilities": probability_output
    }
