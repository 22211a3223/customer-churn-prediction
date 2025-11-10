import pandas as pd

def preprocess_input(df):
    # Example: convert numeric columns
    for col in ["age", "tenure", "clv", "feedback_length", "churn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing values
    df = df.fillna(0)
    return df