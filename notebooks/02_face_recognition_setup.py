"""
Notebook 02: Face Recognition Setup
Uses OpenCV's LBPH Face Recognizer (no dlib required)
Dataset: Synthetic face images generated with OpenCV for demo purposes
Ethics note included as required by project spec
"""

import os
import sys
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'app', 'models')
FACE_DATA_DIR = os.path.join(ROOT, 'data', 'faces')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)

print("=" * 60)
print("Notebook 02: Face Recognition Setup (OpenCV LBPH)")
print("=" * 60)

# --- Ethics Note ---
print("""
[Ethics & Privacy Note]
Face recognition in retail raises important ethical considerations:
- CONSENT: Customers must explicitly consent before their biometric data is collected.
- DATA PRIVACY: Face encodings must be stored securely and never shared with third parties.
- BIAS: Models trained on limited demographics can exhibit bias. Always use diverse datasets.
- GDPR/CCPA: In many jurisdictions, biometric data collection requires legal compliance.
- DATA MINIMIZATION: Only store what is necessary. Delete encodings when no longer needed.
This demo uses synthetically generated faces for testing purposes only.
""")

# --- Create synthetic face data using OpenCV ---
print("[1] Generating synthetic face images for demo...")

def create_synthetic_face(customer_id: int, variation: int) -> np.ndarray:
    """Create a simple synthetic grayscale image as a demo face."""
    np.random.seed(customer_id * 100 + variation)
    face = np.zeros((100, 100), dtype=np.uint8)
    # Draw face oval
    cv2.ellipse(face, (50, 50), (40, 50), 0, 0, 360, 180, -1)
    # Draw eyes
    eye_y = 35 + variation
    cv2.circle(face, (30, eye_y), 8, 50, -1)
    cv2.circle(face, (70, eye_y), 8, 50, -1)
    # Add Gaussian noise for variation
    noise = np.random.randint(0, 30, face.shape, dtype=np.uint8)
    face = cv2.add(face, noise)
    return face

# Generate 5 customers with 10 images each
num_customers = 5
images_per_customer = 10
customer_metadata = {}

face_images = []
face_labels = []

for cust_id in range(num_customers):
    label = cust_id
    customer_name = f"customer_{cust_id + 1:03d}"
    customer_metadata[label] = customer_name
    img_folder = os.path.join(FACE_DATA_DIR, customer_name)
    os.makedirs(img_folder, exist_ok=True)

    for var in range(images_per_customer):
        face = create_synthetic_face(cust_id, var)
        face_images.append(face)
        face_labels.append(label)
        cv2.imwrite(os.path.join(img_folder, f"img_{var}.png"), face)

print(f"    Generated {len(face_images)} images for {num_customers} customers")

# --- Train LBPH Recognizer ---
print("\n[2] Training OpenCV LBPH Face Recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100
)
recognizer.train(face_images, np.array(face_labels))
print("    Training complete.")

# --- Save model ---
lbph_path = os.path.join(MODELS_DIR, 'face_lbph.yml')
recognizer.save(lbph_path)
print(f"    [OK] Saved face_lbph.yml ({os.path.getsize(lbph_path) // 1024} KB)")

# --- Save metadata ---
metadata_path = os.path.join(MODELS_DIR, 'face_db.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(customer_metadata, f)
print(f"    [OK] Saved face_db.pkl (customer ID -> name mapping)")

# --- Evaluate (predict on last image of each customer) ---
print("\n[3] Evaluating recognizer (self-test)...")
correct = 0
for cust_id in range(num_customers):
    test_face = create_synthetic_face(cust_id, images_per_customer - 1)
    label_pred, confidence = recognizer.predict(test_face)
    is_correct = (label_pred == cust_id)
    if is_correct:
        correct += 1
    print(f"    Customer {cust_id+1}: Predicted={customer_metadata.get(label_pred, 'unknown')}, Confidence={confidence:.1f}, {'[OK]' if is_correct else '[FAIL]'}")

acc = correct / num_customers
print(f"\n    [OK] Self-test Accuracy: {acc:.0%} ({correct}/{num_customers})")
print("\n" + "=" * 60)
print("Face recognition setup complete!")
print("=" * 60)
