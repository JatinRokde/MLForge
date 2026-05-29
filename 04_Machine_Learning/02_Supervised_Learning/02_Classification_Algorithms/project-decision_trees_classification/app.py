from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# FastAPI app

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Decision Tree based customer churn prediction service"
)

model = joblib.load(
    "./decision_tree_pipeline.pkl"
)

feature_importance = joblib.load(
    "./feature_importance.pkl"
)

# Request Schema
class CustomerRequest(BaseModel):
    age: int = Field(..., example=32)
    tenure_months: int = Field(..., example=5)
    contract_type: str = Field(..., example="Monthly")
    plan_type: str = Field(..., example="Premium")
    monthly_charges: float = Field(..., example=999)
    avg_call_minutes: float = Field(..., example=350)
    avg_sms_count: int = Field(..., example=100)
    avg_data_usage_gb: float = Field(..., example=18)
    support_tickets: int = Field(..., example=5)
    complaint_count: int = Field(..., example=3)
    avg_resolution_time_hours: float = Field(
        ...,
        example=24
    )
    late_payments: int = Field(..., example=2)
    payment_method: str = Field(..., example="UPI")
    auto_renewal: str = Field(..., example="No")
    complaint_rate: float = Field(..., example=0.60)
    usage_drop_pct: float = Field(..., example=40)
    customer_lifetime_value: float = Field(
        ...,
        example=5000
    )


class BatchRequest(BaseModel):
    customers: List[CustomerRequest]


# Response Schema
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float


# Health Endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Model Info

@app.get("/model-info")
def model_info():

    tree = model.named_steps["model"]

    return {
        "algorithm": "Decision Tree Classifier",
        "criterion": tree.criterion,
        "max_depth": tree.get_depth(),
        "nodes": tree.tree_.node_count
    }


# Feature Importance
@app.get("/feature-importance")
def get_feature_importance():

    return (
        feature_importance
        .head(20)
        .to_dict(orient="records")
    )


# Single Prediction
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(payload: CustomerRequest):

    df = pd.DataFrame(
        [payload.model_dump()]
    )

    prediction = int(
        model.predict(df)[0]
    )

    probability = float(
        model.predict_proba(df)[0][1]
    )

    label = (
        "Churn"
        if prediction == 1
        else "No Churn"
    )

    return PredictionResponse(
        churn_prediction=prediction,
        churn_label=label,
        churn_probability=round(
            probability,
            4
        )
    )


# Batch Prediction
@app.post("/batch-predict")
def batch_predict(
    payload: BatchRequest
):

    rows = [
        customer.model_dump()
        for customer in payload.customers
    ]

    df = pd.DataFrame(rows)

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)

    results = []

    for pred, prob in zip(
        predictions,
        probabilities
    ):

        results.append(
            {
                "prediction": int(pred),
                "label": (
                    "Churn"
                    if pred == 1
                    else "No Churn"
                ),
                "probability": round(
                    float(prob[1]),
                    4
                )
            }
        )

    return {
        "total_customers": len(results),
        "results": results
    }