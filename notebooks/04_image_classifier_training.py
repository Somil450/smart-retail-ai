"""
Notebook 04: Product Image Classifier Training
Dataset: Synthetic Fashion-MNIST-style data (no internet/TensorFlow required)
Model: Random Forest classifier on flattened pixel features
Metrics: Accuracy, Classification Report, Confusion Matrix
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'app', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RETAIL_CATEGORIES = ['bags', 'clothing', 'shoes']  # sorted alphabetically

print("=" * 60)
print("Notebook 04: Product Image Classifier Training")
print("    Model: Random Forest on synthetic 28x28 image features")
print("=" * 60)

# ---------------------------------------------------------------
# Generate synthetic grayscale 28x28 images per retail category.
# Each category has a distinct pixel signature so the RF can learn.
# ---------------------------------------------------------------
print("\n[1] Generating synthetic product image dataset...")

def make_bag_image(rng):
    """Rectangle in the centre (handbag silhouette)."""
    img = np.zeros((28, 28), dtype=np.float32)
    img[8:20, 6:22] = rng.uniform(0.5, 1.0)          # body
    img[5:8,  10:18] = rng.uniform(0.3, 0.7)          # handle
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)

def make_clothing_image(rng):
    """T-shape (shirt / top silhouette)."""
    img = np.zeros((28, 28), dtype=np.float32)
    img[10:26, 8:20] = rng.uniform(0.5, 1.0)          # body
    img[6:12,  2:26]  = rng.uniform(0.4, 0.9)          # shoulders/sleeves
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)

def make_shoe_image(rng):
    """Angled wedge (shoe sole silhouette)."""
    img = np.zeros((28, 28), dtype=np.float32)
    for row in range(14, 22):
        start = max(0, row - 14)
        img[row, start:24] = rng.uniform(0.5, 1.0)
    img[10:16, 14:24] = rng.uniform(0.4, 0.8)          # upper
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)

generators = [make_bag_image, make_clothing_image, make_shoe_image]
N_PER_CLASS = 5000   # 15 000 total

X_list, y_list = [], []
for class_idx, gen_fn in enumerate(generators):
    rng = np.random.RandomState(42 + class_idx)
    for _ in range(N_PER_CLASS):
        img = gen_fn(rng)
        X_list.append(img.flatten())
        y_list.append(class_idx)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=int)

# Shuffle
perm = np.random.RandomState(0).permutation(len(X))
X, y = X[perm], y[perm]

print(f"    Generated {len(X)} samples ({N_PER_CLASS} per category)")
print(f"    Categories: {RETAIL_CATEGORIES}")

# --- Train/Test split ---
print("\n[2] Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)} | Test: {len(X_test)} samples")

# --- Train Random Forest ---
print("\n[3] Training Random Forest classifier (100 trees)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("    Training complete.")

# --- Evaluate ---
print("\n[4] Evaluating model on test set...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n    [OK] Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=RETAIL_CATEGORIES))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=RETAIL_CATEGORIES, yticklabels=RETAIL_CATEGORIES)
plt.title(f'Product Classifier Confusion Matrix\nAccuracy: {acc:.2%}')
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, 'product_classifier_confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"\n    Saved confusion matrix -> {cm_path}")

# --- Save model ---
print("\n[5] Saving model...")
model_path = os.path.join(MODELS_DIR, 'product_classifier.pkl')
joblib.dump(clf, model_path)
print(f"    [OK] Saved product_classifier.pkl ({os.path.getsize(model_path) // 1024} KB)")

meta_path = os.path.join(MODELS_DIR, 'product_categories.json')
with open(meta_path, 'w') as f:
    json.dump({'categories': RETAIL_CATEGORIES, 'model_type': 'sklearn_rf'}, f)
print(f"    [OK] Saved product_categories.json")

print("\n" + "=" * 60)
print("Image classifier training complete!")
print("=" * 60)
