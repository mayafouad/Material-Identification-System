import joblib
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from cnn_feature_extractor import CNNFeatureExtractor
from utils import CLASSES


# ---------------------------
# 🔥 FIXED DATASET DIRECTORY
# ---------------------------
# This points to: Material-Identification-System/augmented_dataset
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/knn_cnn.pkl"
SCALER_PATH = Path(__file__).resolve().parents[1] / "models/scaler_knn_cnn.pkl"


def load_dataset(dataset_dir, extractor):
    X, y = [], []

    print(f"\n[INFO] Loading dataset from: {dataset_dir}")
    print(f"Exists? {dataset_dir.exists()}")

    for label, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name.lower()  # ensure lowercase compatibility
        if not class_dir.exists():
            print(f"[WARN] Missing folder: {class_dir} — skipping this class.")
            continue

        image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        print(f"[INFO] Loading {class_name} ({len(image_paths)} images)")

        for img_path in tqdm(image_paths):
            try:
                feat = extractor.extract(str(img_path))
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"[WARN] Failed to process {img_path}: {e}")

    return np.array(X), np.array(y)


def train_knn_cnn(k=5):
    print("\n===================================")
    print("     TRAINING KNN + CNN FEATURES   ")
    print("===================================\n")

    # Using CNN feature extractor to extract deep features from images
    extractor = CNNFeatureExtractor()

    print("[STEP] Loading CNN features from dataset...")
    X, y = load_dataset(DATASET_DIR, extractor)

    print(f"\n[INFO] Loaded feature matrix: {X.shape}")
    if X.shape[0] == 0:
        print("\n❌ ERROR: No images found. Check dataset directory.")
        return

    print("\n[STEP] Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[STEP] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[STEP] Training KNN (Euclidean distance)...")
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)

    print("\n[STEP] Evaluating KNN...")
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n⭐ Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n[STEP] Saving model and scaler...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n✔ Model saved to:  {MODEL_PATH}")
    print(f"✔ Scaler saved to: {SCALER_PATH}")
    print("\nTraining complete!")

def preprocess_image(img_path, extractor = CNNFeatureExtractor()):
    feat = extractor.extract(str(img_path))
    return feat

def predict_material(image_path):
    UNKNOWN_THRESHOLD = 0.4
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = preprocess_image(str(image_path))
    features_scaled = scaler.transform([features])
    
    probs = knn.predict_proba(features_scaled)[0]
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]

    if best_prob < UNKNOWN_THRESHOLD:
        return "Unknown", float(1-best_prob)

    return CLASSES[best_idx], float(best_prob)



if __name__ == "__main__":
    #train_knn_cnn()
    pred, prob = predict_material("images.jpg")
    print(f"Prediction: {pred} (confidence: {prob:.2f})")
