from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("delivery_time_model.pkl")

app = FastAPI(title="Delivery Time Prediction API")

class DistanceInput(BaseModel):
    distance: float

class DeliveryTimeOutput(BaseModel):
    delivery_time: float

@app.get("/")
def health():
    return {"message" : "Delivery time prediction API is Up."}

@app.post("/predict")
def predict(data: DistanceInput):
    distance = np.array([[data.distance]])
    prediction_time = model.predict(distance)

    return {"Expected Delivery Time" : round(float(prediction_time[0]), 2)}