"""
train_models.py — Master training orchestrator
Runs all four training notebooks in sequence to generate real model files.
Run from the project root:  python notebooks/train_models.py
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(ROOT, 'notebooks')

scripts = [
    ("01 - Chatbot",             os.path.join(NOTEBOOKS, '01_chatbot_training.py')),
    ("02 - Face Recognition",    os.path.join(NOTEBOOKS, '02_face_recognition_setup.py')),
    ("03 - Sentiment Analysis",  os.path.join(NOTEBOOKS, '03_sentiment_model_training.py')),
    ("04 - Product Classifier",  os.path.join(NOTEBOOKS, '04_image_classifier_training.py')),
]

print("=" * 60)
print("Smart Retail AI — Model Training Pipeline")
print("=" * 60)

for name, script in scripts:
    print(f"\n>>> Running: {name}")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)
    print(f"[DONE] {name} completed successfully.")

print("\n" + "=" * 60)
print("All models trained and saved to app/models/")
print("You can now start the server:  uvicorn app.main:app --reload")
print("=" * 60)
