import os
import sys
import pickle
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FACE_DB_PATH = os.path.join(MODELS_DIR, 'face_db.pkl')
FACE_LBPH_PATH = os.path.join(MODELS_DIR, 'face_lbph.yml')
PRODUCT_MODEL_PATH = os.path.join(MODELS_DIR, 'product_classifier.pkl')
CATEGORIES_PATH = os.path.join(MODELS_DIR, 'product_categories.json')

face_recognizer = None
face_db = {}
product_model = None
product_categories = ['bags', 'clothing', 'shoes']  # default

FACE_CONFIDENCE_THRESHOLD = 80  # Lower = stricter match in LBPH


def load_models():
    global face_recognizer, face_db, product_model, product_categories

    # Load face recognizer
    try:
        import cv2
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(FACE_LBPH_PATH)
        with open(FACE_DB_PATH, 'rb') as f:
            face_db = pickle.load(f)
        print("[CV Service] Face recognizer loaded.")
    except Exception as e:
        print(f"[CV Service] Face recognizer not loaded: {e}")

    # Load product classifier (sklearn Random Forest)
    try:
        import joblib
        product_model = joblib.load(PRODUCT_MODEL_PATH)
        with open(CATEGORIES_PATH, 'r') as f:
            product_categories = json.load(f)['categories']
        print("[CV Service] Product classifier loaded.")
    except Exception as e:
        print(f"[CV Service] Product classifier not loaded: {e}")


def recognize_face(image_bytes: bytes) -> dict:
    try:
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return {"error": "Could not decode image"}

        img_resized = cv2.resize(img, (100, 100))

        if face_recognizer is not None:
            label_pred, confidence = face_recognizer.predict(img_resized)
            if confidence < FACE_CONFIDENCE_THRESHOLD:
                customer_id = face_db.get(label_pred, f"customer_{label_pred:03d}")
                status = "returning_customer"
            else:
                customer_id = "unknown_visitor"
                status = "new_customer"
        else:
            # Fallback if model not loaded
            customer_id = "unknown_visitor"
            status = "new_customer"
            confidence = 999

        return {
            "customer_id": customer_id,
            "status": status,
            "message": f"Face recognition complete (confidence: {confidence:.1f})"
        }
    except Exception as e:
        return {"error": str(e)}


def classify_product(image_bytes: bytes) -> dict:
    try:
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return {"error": "Could not decode image"}

        # Preprocess to 28x28 grayscale as Fashion-MNIST expects, then flatten
        img_resized = cv2.resize(img, (28, 28)).astype('float32') / 255.0
        img_input = img_resized.reshape(1, -1)  # flatten to 784 features

        if product_model is not None:
            proba = product_model.predict_proba(img_input)[0]
            class_idx = int(np.argmax(proba))
            confidence = float(proba[class_idx])
            category = product_categories[class_idx]
        else:
            category = "unknown"
            confidence = 0.0

        return {
            "category": category,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}


load_models()
