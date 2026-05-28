from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np

app = FastAPI(
    title="Naive Bayes Ticket Classifier API",
    description="Production-Style Ticket Classification API"
)

# Load Model
model = joblib.load(
    "./naive_bayes_ticket_classifier.pkl"
)

# Request Schema
class TicketRequest(BaseModel):
    ticket: str

class BatchTicketRequest(BaseModel):
    tickets: list[str]


# Response Schema
class PredictionResponse(BaseModel):
    ticket: str
    predicted_category: str
    confidence: float
    class_probabilities: dict


# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: TicketRequest):
    ticket_text = request.ticket

    prediction = model.predict(
        [ticket_text]
    )[0]

    # Probability Scores
    probabilities = model.predict_proba(
        [ticket_text]
    )[0]

    # Class Labels
    classes = model.classes_

    class_probabilities = {
        class_name: round(float(prob), 4)
        for class_name, prob in zip(
            classes,
            probabilities
        )
    }

    confidence = round(
        float(np.max(probabilities)),
        4
    )

    return {
        "ticket": ticket_text,
        "predicted_category": prediction,
        "confidence": confidence,
        "class_probabilities": class_probabilities
    }


@app.post("/batch-predict")
def batch_predict(request: BatchTicketRequest):
    predictions = model.predict(
        request.tickets
    )

    probabilities = model.predict_proba(
        request.tickets
    )

    results = []

    for ticket, prediction, probability in zip(
        request.tickets,
        predictions,
        probabilities
    ):

        results.append({
            "ticket": ticket,
            "predicted_category": prediction,
            "confidence": round(
                float(np.max(probability)),
                4
            )
        })

    return {
        "total_tickets": len(results),
        "results": results
    }