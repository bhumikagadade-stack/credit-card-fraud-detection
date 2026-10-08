

import pandas as pd
from pathlib import Path

from predict import predict_fraud


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the dataset
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

# Find a known fraudulent transaction
fraud_index = df[df["Class"] == 1].index[0]

transaction = df.drop(columns=["Class"]).loc[[fraud_index]]

# Make prediction
probability, prediction = predict_fraud(transaction)

print("Prediction Test")
print("----------------")
print(f"Fraud probability : {probability:.4f}")
print(
    f"Prediction        : "
    f"{'FRAUD' if prediction == 1 else 'LEGITIMATE'}"
)