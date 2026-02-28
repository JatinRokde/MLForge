from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path
import numpy as np

# Load Model and Scaler
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "house_prices_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

# Initialize FastAPI
app = FastAPI(title="House Price Prediction API")

# Input Schema
class HouseInput(BaseModel):
    size_sqft: float = Field(..., gt=200, lt=10000, description="Size in square feet")
    bedrooms: int = Field(..., ge=1, le=15, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=15, description="Number of bathrooms")
    age_years: int = Field(..., ge=0, le=150, description="Age of house in years")
    distance_city_km: float = Field(..., ge=0, le=100, description="Distance from city center (km)")
    school_rating: int = Field(..., ge=1, le=10, description="Nearby school rating (1-10)")

# Output Schema
class HouseOutput(BaseModel):
    predicted_price: float

@app.get("/")
def health():
    return {"message": "House Price Prediction API is Up."}

# Prediction Endpoint
@app.post("/predict", response_model=HouseOutput)
def predict(data: HouseInput):
    input_data = np.array([[
        data.size_sqft,
        data.bedrooms,
        data.bathrooms,
        data.age_years,
        data.distance_city_km,
        data.school_rating
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return HouseOutput(
        predicted_price=round(float(prediction[0]), 2)
    )