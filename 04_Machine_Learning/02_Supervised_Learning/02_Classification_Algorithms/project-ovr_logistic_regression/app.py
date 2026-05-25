from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

import joblib

# Load Model
MODEL_PATH = './ovr_support_ticket_model.pkl'

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    raise Exception(f"Failed to load model; {e}")


# FastAPI app
app = FastAPI(
    title="OvR Support Ticket Classification API",
    description="One-Vs-Rest Logistic Regression API"
)


# Request Schemas
class TicketRequest(BaseModel):
    ticket_text: str

class BatchTicketRequest(BaseModel):
    tickets: List[str]


# Response Schemas
class PredictionResponse(BaseModel):
    predicted_department: str
    probabilities: Dict[str, float]


# Utility Function
def format_probabilities(classes, probabilites):
    result = {}

    for cls, prob in zip(classes, probabilites):
        result[cls] = round(float(prob), 4)

    return result


# Root Endpoint
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "One-vs-Rest Logistic Regression",
        "message": "Support Ticket Classification API"
    }


# Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Single Prediction Endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_ticket(request: TicketRequest):
    try:
        # Input text
        ticket = request.ticket_text

        # Predict class
        prediction = model.predict([ticket])[0]

        # Predict probabilities
        probabilities = model.predict_proba([ticket])[0]

        # Get class names
        classes = model.classes_

        # Format probabilities
        probability_dict = format_probabilities(
            classes, probabilities
        )

        return {
            "predicted_department": prediction,
            "probabilities": probability_dict
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
# Batch Prediction Endpoint
@app.post("/predict-batch")
def predict_batch(request: BatchTicketRequest):

    try:

        tickets = request.tickets

        # Predict classes
        predictions = model.predict(tickets)

        # Predict probabilities
        probabilities = model.predict_proba(tickets)

        results = []

        for i in range(len(tickets)):

            probability_dict = {}

            for cls, prob in zip(
                model.classes_,
                probabilities[i]
            ):
                probability_dict[cls] = round(float(prob), 4)

            results.append({
                "ticket": tickets[i],
                "prediction": predictions[i],
                "probabilities": probability_dict
            })

        return {
            "total_tickets": len(results),
            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    

# Top-K predictions endpoint
@app.post("/predict-top-k")
def predict_top_k(
    request: TicketRequest,
    k: int = 3
):
    try:
        ticket = request.ticket_text

        probabilites = model.predict_proba([ticket])[0]

        classes = model.classes_

        # Combine class + probability
        class_probs = list(zip(classes, probabilites))

        # Sort descending
        class_probs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Top K
        top_k = class_probs[:k]

        return {
            "ticket": ticket,
            "top-k-predictions": [
                {
                    "department": cls,
                    "probability": round(float(prob), 4)
                }
                for cls, prob in top_k
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
# Confidence Threshold Endpoint
@app.post("/predict-with-threshold")
def predict_with_threshold(
    request: TicketRequest,
    threshold: float = 0.60
):
    try:
        ticket = request.ticket_text

        probabilities = model.predict_proba([ticket])[0]

        max_probability = max(probabilities)

        predicted_class = model.classes_[
            probabilities.argmax()
        ]

        if max_probability < threshold:

            return {
                "status": "LOW_CONFIDENCE",
                "message": "Route to human agent",
                "max_probability": round(float(max_probability), 4)
            }
        
        return {
            "status": "SUCCESS",
            "prediction": predicted_class,
            "confidence": round(float(max_probability), 4)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )