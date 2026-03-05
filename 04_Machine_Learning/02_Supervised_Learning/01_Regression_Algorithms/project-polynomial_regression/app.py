from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pipeline model
model = joblib.load("cpu_temp_model.pkl")

app = FastAPI(title="CPU Temperature Prediction API")

class CPULoadInput(BaseModel):
    cpu_load_percent: float

class TemperatureOutput(BaseModel):
    predicted_cpu_temperature: float

@app.get("/")
def health():
    return {"message": "CPU Temperature Prediction API is Up."}

@app.post("/predict", response_model=TemperatureOutput)
def predict(data: CPULoadInput):
    X = np.array([[data.cpu_load_percent]])
    prediction = model.predict(X)
    return {
        "predicted_cpu_temperature": round(float(prediction[0]), 2)
    }