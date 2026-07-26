import os
import sys
import json
import joblib
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.preprocess import preprocess_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CHATBOT_MODEL_PATH = os.path.join(MODELS_DIR, 'chatbot_model.pkl')
INTENTS_PATH = os.path.join(DATA_DIR, 'intents.json')

chatbot_model = None
intents_responses = {}

CONFIDENCE_THRESHOLD = 0.3

def load_models():
    global chatbot_model, intents_responses
    try:
        chatbot_model = joblib.load(CHATBOT_MODEL_PATH)
        with open(INTENTS_PATH, 'r') as f:
            data = json.load(f)
        for intent in data['intents']:
            intents_responses[intent['tag']] = intent['responses']
        print("[Chatbot Service] Chatbot model and intents loaded.")
    except Exception as e:
        print(f"[Chatbot Service] Error: {e}")

def get_chatbot_response(message: str) -> dict:
    if chatbot_model is None:
        return {"error": "Chatbot model not loaded"}

    cleaned = preprocess_text(message)
    predicted_intent = chatbot_model.predict([cleaned])[0]

    # Get confidence using predict_proba
    try:
        proba = chatbot_model.predict_proba([cleaned])[0]
        confidence = float(max(proba))
    except Exception:
        confidence = 1.0

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_intent = "unknown"
        reply = "I'm not sure how to help with that. Could you rephrase?"
    else:
        responses = intents_responses.get(predicted_intent, ["I'm not sure how to help with that."])
        reply = random.choice(responses)

    return {
        "message": message,
        "intent": predicted_intent,
        "reply": reply
    }

load_models()
