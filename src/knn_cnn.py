import joblib
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from src.cnn_feature_extractor import CNNFeatureExtractor
from data_augmentation import DataAugmentor
from src.utils import CLASSES, load_dataset


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/knn_cnn.pkl"
SCALER_PATH = Path(__file__).resolve().parents[1] / "models/scaler_knn_cnn.pkl"

data_augmentor = DataAugmentor()

def train_knn_cnn(k=5, increase_percent=40):
    print("\n===================================")
    print("     TRAINING KNN + CNN FEATURES   ")
    print("===================================\n")

    # Using CNN feature extractor to extract deep features from images
    extractor = CNNFeatureExtractor()

    print("[STEP] Loading CNN features from dataset...")
    X, y, paths = load_dataset(DATASET_DIR, extractor)

    print(f"\n[INFO] Loaded feature matrix: {X.shape}")
    if X.shape[0] == 0:
        print("\n❌ ERROR: No images found. Check dataset directory.")
        return

    print("\n[STEP] Train/Test split...")
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.2, random_state=53, stratify=y
    )
    print(f"       Train: {len(X_train)} | Test: {len(X_test)}")

    # Augment training data only
    X_aug, y_aug = data_augmentor.augment_training_data(paths_train, y_train, extractor, increase_percent)
    
    # Combine original + augmented training data
    X_train = np.vstack([X_train, X_aug])
    y_train = np.concatenate([y_train, y_aug])
    print(f"\n[INFO] After augmentation - Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[STEP] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[STEP] Training KNN (Euclidean distance)...")
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)

    print("\n[STEP] Evaluating KNN...")
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
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
    train_knn_cnn(k=5, increase_percent=40)
    # pred, prob = predict_material("images.jpg")
    # print(f"Prediction: {pred} (confidence: {prob:.2f})")


