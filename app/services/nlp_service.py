import os
import sys
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.preprocess import preprocess_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_model.pkl')

sentiment_model = None

def load_models():
    global sentiment_model
    try:
        sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
        print("[NLP Service] Sentiment model loaded.")
    except Exception as e:
        print(f"[NLP Service] Error loading sentiment model: {e}")

def analyze_sentiment(text: str) -> dict:
    if sentiment_model is None:
        return {"error": "Sentiment model not loaded"}

    cleaned = preprocess_text(text)

    prediction = sentiment_model.predict([cleaned])[0]
    probabilities = sentiment_model.predict_proba([cleaned])[0]
    confidence = float(max(probabilities))

    # Map movie_reviews labels to retail-friendly labels
    label_map = {'pos': 'Positive', 'neg': 'Negative', 'Positive': 'Positive',
                 'Negative': 'Negative', 'Neutral': 'Neutral'}
    sentiment_label = label_map.get(prediction, prediction.capitalize())

    return {
        "text": text,
        "sentiment": sentiment_label,
        "confidence": round(confidence, 2)
    }

load_models()
