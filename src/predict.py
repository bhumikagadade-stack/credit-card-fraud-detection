import joblib
import pandas as pd
from pathlib import Path


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model
MODEL_PATH = BASE_DIR / "models" / "random_forest_fraud_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "fraud_threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


def predict_fraud(transaction):
    """
    Predict whether a credit card transaction is fraudulent.

    Parameters:
        transaction (pd.DataFrame):
            Transaction containing the same features used during training.

    Returns:
        probability (float):
            Predicted probability of fraud.

        prediction (int):
            1 = Fraud
            0 = Legitimate
    """

    probability = model.predict_proba(transaction)[0, 1]

    prediction = int(probability >= threshold)

    return probability, prediction