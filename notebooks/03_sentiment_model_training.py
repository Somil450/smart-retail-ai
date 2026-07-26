"""
Notebook 03: Sentiment Model Training
Dataset: NLTK Movie Reviews corpus (built-in, no download required)
Model: TF-IDF + Logistic Regression
Metrics: Accuracy, Classification Report, Confusion Matrix
"""

import os
import sys
import nltk
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# --- Setup paths ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODELS_DIR = os.path.join(ROOT, 'app', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

from data.preprocess import preprocess_text, download_nltk_data

# --- Download data ---
print("=" * 60)
print("Notebook 03: Sentiment Model Training")
print("=" * 60)

download_nltk_data()
nltk.download('movie_reviews', quiet=True)

from nltk.corpus import movie_reviews

# --- Load data ---
print("\n[1] Loading NLTK Movie Reviews dataset...")
documents = [
    (' '.join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

import random
random.shuffle(documents)

texts, labels = zip(*documents)
print(f"    Loaded {len(texts)} reviews ({labels.count('pos')} positive, {labels.count('neg')} negative)")

# --- Preprocess ---
print("\n[2] Preprocessing text (tokenize, remove stopwords, lemmatize)...")
cleaned_texts = [preprocess_text(t) for t in texts]
print(f"    Done. Sample: '{cleaned_texts[0][:80]}...'")

# --- Train/Test Split ---
print("\n[3] Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# --- Model Pipeline ---
print("\n[4] Training TF-IDF + Logistic Regression pipeline...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])
model.fit(X_train, y_train)
print("    Training complete.")

# --- Evaluation ---
print("\n[5] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n    ✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=['neg', 'pos'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Neg', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
plt.title(f'Sentiment Model Confusion Matrix\nAccuracy: {acc:.2%}')
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, 'sentiment_confusion_matrix.png')
plt.savefig(cm_path)
print(f"\n    Saved confusion matrix → {cm_path}")

# --- Save model ---
print("\n[6] Saving model...")
model_path = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
joblib.dump(model, model_path)
print(f"    ✅ Saved sentiment_model.pkl ({os.path.getsize(model_path) // 1024} KB)")
print("\n" + "=" * 60)
print("Training complete! Sentiment model is ready.")
print("=" * 60)
