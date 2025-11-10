import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: Load vectorizer for feedback text if you used one during training
# vectorizer = joblib.load('app_folder/tfidf_vectorizer.pkl')

def preprocess_tabular(X):
    if isinstance(X, dict):
        X = pd.DataFrame([X])

    # Numeric columns
    num_cols = ['age', 'tenure', 'sentiment', 'clv']
    for col in num_cols:
        if col not in X.columns:
            X[col] = 0

    # Text vectorization (optional)
    # feedback_tfidf = vectorizer.transform(X['feedback_text']).toarray()
    # X = X.drop('feedback_text', axis=1)

    # For now, drop text or keep simple flag
    X['feedback_length'] = X['feedback_text'].apply(len)
    X = X.drop(columns=['feedback_text'])

    # Categorical encoding
    cat_cols = ['plan_type', 'voice_emotion', 'retention_strategy']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X