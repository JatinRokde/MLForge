import logging
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "churn_logistic_pipeline.pkl"

try:
    logger.info("Loading model...")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")

# FastAPI App
app = FastAPI(
    title="Churn Prediction API",
    description="Binary Logistic Regression Model for Customer Churn"
)

# Schema
class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0, description="Customer tenure in months")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]

    @field_validator("tenure")
    @classmethod
    def validate_tenure(cls, v):
        if v > 100:
            raise ValueError("Tenure seems unrealistic")
        return v
    
class PredictResponse(BaseModel):
    churn_probability: float
    prediction: int
    threshold: float

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True
    }

# Prediction Endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: ChurnRequest, threshold: float = 0.5):
    try:
        request_data = request.model_dump()
        logger.info(f"Incoming request: {request_data}")

        # Convert input to DF
        input_df = pd.DataFrame([request_data])

        # Model prediction
        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= threshold)

        response = {
            "churn_probability": round(float(prob), 4),
            "prediction": pred,
            "threshold": threshold
        }

        logger.info(f"Prediction response: {response}")

        return response

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during prediction"
        )