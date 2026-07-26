"""
Notebook 01: Chatbot Intent Model Training
Dataset: Custom intents.json with 30 retail FAQ intents
Model: TF-IDF + Logistic Regression
Metrics: Accuracy, Classification Report
"""

import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Setup paths ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODELS_DIR = os.path.join(ROOT, 'app', 'models')
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(MODELS_DIR, exist_ok=True)

from data.preprocess import preprocess_text

print("=" * 60)
print("Notebook 01: Chatbot Intent Model Training")
print("=" * 60)

# --- Load intents ---
print("\n[1] Loading intents.json...")
with open(os.path.join(DATA_DIR, 'intents.json'), 'r') as f:
    data = json.load(f)

X, y = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(preprocess_text(pattern))
        y.append(intent['tag'])

print(f"    Loaded {len(X)} training samples across {len(data['intents'])} intents")

# --- Train/Test Split ---
print("\n[2] Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

# --- Model ---
print("\n[3] Training TF-IDF + Logistic Regression pipeline...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, C=2.0))
])
model.fit(X_train, y_train)
print("    Training complete.")

# --- Evaluation ---
print("\n[4] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n    ✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred))

# --- Save ---
print("\n[5] Saving model...")
model_path = os.path.join(MODELS_DIR, 'chatbot_model.pkl')
joblib.dump(model, model_path)
print(f"    ✅ Saved chatbot_model.pkl ({os.path.getsize(model_path) // 1024} KB)")
print("\n" + "=" * 60)
print("Chatbot model training complete!")
print("=" * 60)
