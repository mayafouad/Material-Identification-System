import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from knn_cnn import load_dataset
from src.cnn_feature_extractor import CNNFeatureExtractor
from src.utils import CLASSES
import joblib

DATASET_DIR = Path(r"C:\Users\ahmed\Downloads\test")
KNN_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/knn_cnn.pkl"
KNN_SCALER_PATH = Path(__file__).resolve().parents[1] / "models/scaler_knn_cnn.pkl"
SVM_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/svm_cnn.pkl"
SVM_SCALER_PATH = Path(__file__).resolve().parents[1] / "models/scaler_cnn.pkl"
CONFIDENCE_THRESHOLD = 0.4


def test_knn_model():
    print("\n" + "=" * 50)
    print("  TESTING KNN MODEL ON EXTERNAL DATASET")
    print("=" * 50)

    # load model
    knn = joblib.load(KNN_MODEL_PATH)
    scaler = joblib.load(KNN_SCALER_PATH)
    print(f"✓ Model loaded")

    # load test data using existing function
    print("\nLoading test dataset...")
    extractor = CNNFeatureExtractor()
    X_test, y_test = load_dataset(DATASET_DIR, extractor)

    if len(X_test) == 0:
        print("No images found.")
        return

    print(f"\nTest samples: {len(y_test)}")

    # scale and predict
    X_test_scaled = scaler.transform(X_test)

    print("\nPredicting...")
    y_pred = []
    
    for features in tqdm(X_test_scaled):
        features = features.reshape(1, -1)
        
        probs = knn.predict_proba(features)[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]
        
        # Apply 0.4 threshold rejection
        if confidence < CONFIDENCE_THRESHOLD:
            y_pred.append(6)  # 'unknown' class
        else:
            y_pred.append(best_idx)

    y_pred = np.array(y_pred)

    # results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  KNN ACCURACY: {accuracy:.2%}")
    print(f"{'=' * 50}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    unknown_count = np.sum(y_pred == 6)
    print(f"\nSamples classified as 'unknown': {unknown_count}/{len(y_pred)}")


def test_svm_model():
    print("\n" + "=" * 50)
    print("  TESTING SVM MODEL ON EXTERNAL DATASET")
    print("=" * 50)

    # load model
    svm = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SVM_SCALER_PATH)
    print(f"✓ Model loaded")

    # load test data
    print("\nLoading test dataset...")
    extractor = CNNFeatureExtractor()
    X_test, y_test = load_dataset(DATASET_DIR, extractor)

    if len(X_test) == 0:
        print("No images found.")
        return

    print(f"\nTest samples: {len(y_test)}")

    # scale and predict
    X_test_scaled = scaler.transform(X_test)

    print("\nPredicting...")
    y_pred = []

    for features in tqdm(X_test_scaled):
        features = features.reshape(1, -1)

        probs = svm.predict_proba(features)[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        # Apply 0.4 threshold rejection
        if confidence < CONFIDENCE_THRESHOLD:
            y_pred.append(6)  # 'unknown' class
        else:
            y_pred.append(best_idx)
    
    y_pred = np.array(y_pred)

    # results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  SVM ACCURACY: {accuracy:.2%}")
    print(f"{'=' * 50}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    unknown_count = np.sum(y_pred == 6)
    print(f"\nSamples classified as 'unknown': {unknown_count}/{len(y_pred)}")


if __name__ == "__main__":
    test_knn_model()
    test_svm_model()
