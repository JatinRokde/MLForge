from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib


app = FastAPI(
    title="Customer Segmentation API",
)


kmeans_model = joblib.load(
    "./kmeans_model.pkl"
)

scaler = joblib.load(
    "./scaler.pkl"
)

feature_columns = joblib.load(
    "./feature_columns.pkl"
)


# SEGMENT MAPPING
CLUSTER_SEGMENTS = {
    0: "Premium Customer",
    1: "Regular Customer",
    2: "Potential Growth Customer",
    3: "Loyal Customer",
    4: "Engaged Customer"
}


# REQUEST MODEL
class CustomerRequest(BaseModel):
    Age: int
    Annual_Income: float
    Total_Orders: int
    Avg_Order_Value: float
    Days_Since_Last_Purchase: int
    App_Sessions_Per_Month: int
    Website_Visits_Per_Month: int
    Wishlist_Count: int
    Cart_Additions_Per_Month: int


@app.get("/")
def health():

    return {
        "status": "running",
        "model": "KMeans Customer Segmentation"
    }


# PREDICT
@app.post("/predict")
def predict(request: CustomerRequest):

    input_data = pd.DataFrame(
        [{
            "Age": request.Age,
            "Annual_Income": request.Annual_Income,
            "Total_Orders": request.Total_Orders,
            "Avg_Order_Value": request.Avg_Order_Value,
            "Days_Since_Last_Purchase":
                request.Days_Since_Last_Purchase,
            "App_Sessions_Per_Month":
                request.App_Sessions_Per_Month,
            "Website_Visits_Per_Month":
                request.Website_Visits_Per_Month,
            "Wishlist_Count":
                request.Wishlist_Count,
            "Cart_Additions_Per_Month":
                request.Cart_Additions_Per_Month
        }]
    )

    input_data = input_data[
        feature_columns
    ]

    scaled_data = scaler.transform(
        input_data
    )

    cluster = int(
        kmeans_model.predict(
            scaled_data
        )[0]
    )

    segment_name = CLUSTER_SEGMENTS.get(
        cluster,
        "Unknown Segment"
    )

    return {
        "cluster_id": cluster,
        "segment_name": segment_name,
        "customer_data": input_data.to_dict(
            orient="records"
        )[0]
    }