import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("bot_detection_data.csv")

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["Cleaned_Tweet"] = df["Tweet"].astype(str).apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["Cleaned_Tweet"]).toarray()
X_behavior = df[["Retweet Count", "Mention Count", "Follower Count"]].values
X = np.hstack((X_text, X_behavior))

y = df["Bot Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bot Detection Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC Score:", auc_score)

# Save Model
import joblib
joblib.dump(rf_model, "bot_detection_model/random_forest_model.pkl")
joblib.dump(vectorizer, "bot_detection_model/tfidf_vectorizer.pkl")

# Flask API Endpoint
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['tweet']
    cleaned_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([cleaned_input]).toarray()
    prediction = rf_model.predict(input_vector)[0]
    return jsonify({'bot_label': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)