from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import numpy as np

# -----------------------------
# Flask app initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load the model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
model = joblib.load(model_path)
print("Model loaded successfully!")

# -----------------------------
# Preprocess input function
# -----------------------------
def preprocess_input(data_dict):
    """
    Convert input dictionary to DataFrame with correct columns for the model.
    Automatically computes feedback_length.
    """
    # Compute feedback_length
    data_dict['feedback_length'] = len(data_dict.get('feedback_text', ''))

    # Ensure columns match training order
    columns = [
        'age',
        'tenure',
        'plan_type',
        'feedback_text',
        'sentiment',
        'voice_emotion',
        'clv',
        'retention_strategy',
        'feedback_length'
    ]

    # Create single-row DataFrame
    df = pd.DataFrame([data_dict], columns=columns)
    return df

# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Extract form data
        data = {
            'age': float(request.form.get('age', 0)),
            'tenure': float(request.form.get('tenure', 0)),
            'plan_type': request.form.get('plan_type', ''),
            'feedback_text': request.form.get('feedback_text', ''),
            'sentiment': float(request.form.get('sentiment', 0)),
            'voice_emotion': request.form.get('voice_emotion', ''),
            'clv': float(request.form.get('clv', 0)),
            'retention_strategy': request.form.get('retention_strategy', '')
        }

        # Preprocess for model
        features = preprocess_input(data)

        # Predict
        pred_prob = model.predict_proba(features)[0][1]
        pred_class = model.predict(features)[0]

        prediction = {
            'churn_prob': round(pred_prob * 100, 2),
            'churn': 'Yes' if pred_class == 1 else 'No'
        }

    return render_template('index.html', prediction=prediction)

# -----------------------------
# Run the app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
