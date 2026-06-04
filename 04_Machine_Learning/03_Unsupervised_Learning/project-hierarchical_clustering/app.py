from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np

import joblib

scaler = joblib.load(
    "./scaler.pkl"
)

model = joblib.load(
    "./customer_segmentation.pkl"
)


app = FastAPI(
    title="Customer Segmentation API"
)


class CustomerFeatures(BaseModel):

    Annual_Spend: float
    Order_Count: int
    Avg_Order_Value: float
    Days_Since_Last_Purchase: int
    Product_Category_Count: int
    Mobile_App_Usage: int
    Website_Visits: int
    Wishlist_Items: int
    Coupon_Usage_Rate: float
    Return_Rate: float


cluster_mapping = {
    0: "Premium Customers",
    1: "Frequent Shoppers",
    2: "Regular Customers"
}


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):

    df = pd.DataFrame(
        [customer.dict()]
    )

    X_scaled = scaler.transform(df)

    cluster = int(
        model.predict(X_scaled)[0]
    )

    cluster_name = cluster_mapping.get(
        cluster,
        "Unknown"
    )

    return {
        "cluster_id": cluster,
        "cluster_name": cluster_name
    }