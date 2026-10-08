from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_fraud


# Create the FastAPI application
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting whether a credit card transaction is fraudulent.",
    version="1.0.0"
)


class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    transaction_data = transaction.model_dump()

    import pandas as pd

    transaction_df = pd.DataFrame([transaction_data])

    probability, prediction = predict_fraud(transaction_df)

    return {
        "fraud_probability": round(probability, 4),
        "prediction": int(prediction),
        "decision": "FRAUD" if prediction == 1 else "LEGITIMATE"
    }

