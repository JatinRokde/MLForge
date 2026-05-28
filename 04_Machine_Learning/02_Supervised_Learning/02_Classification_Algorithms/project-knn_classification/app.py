from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib


app = FastAPI(
    title="KNN Recommendation API"
)


# Load Artifacts
knn_model = joblib.load("./knn_model.pkl")
scaler = joblib.load("./scaler.pkl")
feature_columns = joblib.load("./feature_columns.pkl")


# Load Dataset

df = pd.read_csv(
    "../data/knn_recommendation_dataset.csv"
)


# Feature Matrix
X = df[feature_columns]

X_scaled = scaler.transform(X)


class RecommendationRequest(BaseModel):
    product_index: int
    top_k: int = 5


@app.get("/")
def home():
    return {
        "message": "Production KNN Recommendation API Running"
    }


@app.post("/recommend")
def recommend_products(request: RecommendationRequest):

    product_index = request.product_index
    top_k = request.top_k

    if product_index >= len(df):
        return {
            "error": "Invalid product index"
        }

    query_vector = X_scaled[product_index].reshape(1, -1)

    distances, indices = knn_model.kneighbors(
        query_vector,
        n_neighbors=top_k + 1
    )

    recommendations = []

    for idx, dist in zip(indices[0][1:], distances[0][1:]):

        row = df.iloc[idx]

        recommendations.append({
            "product_id": row["product_id"],
            "category": row["category"],
            "brand": row["brand"],
            "price": row["price"],
            "avg_rating": row["avg_rating"],
            "similarity_score": round(1 - float(dist), 4)
        })

    return {
        "input_product": {
            "product_id": df.iloc[product_index]["product_id"],
            "category": df.iloc[product_index]["category"],
            "brand": df.iloc[product_index]["brand"]
        },
        "recommendations": recommendations
    }