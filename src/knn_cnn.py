import joblib
import numpy as np
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cnn_feature_extractor import CNNFeatureExtractor
from data_augmentation import DataAugmentor
from utils import CLASSES, load_image, load_dataset_paths


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models/knn_cnn.pkl"
SCALER_PATH = BASE_DIR / "models/scaler_knn_cnn.pkl"


# =========================
# Feature builder
# =========================
def build_features_from_paths(paths, labels, extractor, augmentor=None):
    X, y = [], []

    for path, label in zip(paths, labels):
        img = load_image(path)
        if img is None:
            continue

        # original
        feat = extractor.extract(img)
        X.append(feat)
        y.append(label)

        # augmented (TRAIN ONLY)
        if augmentor is not None:
            aug_img = augmentor.augment(img)
            aug_feat = extractor.extract(aug_img)
            X.append(aug_feat)
            y.append(label)

    return np.array(X), np.array(y)


# =========================
# Training
# =========================
def train_knn_cnn(
    k=5,
    train_ratio=0.7,
    val_ratio=0.15,
    increase_percent=40,
    random_state=42
):
    print("\n==========================================")
    print(" KNN + CNN (Split → Augment → Extract)")
    print("==========================================\n")

    extractor = CNNFeatureExtractor()
    augmentor = DataAugmentor(increase_percent=increase_percent)

    # 1️⃣ Load paths only
    paths, labels = load_dataset_paths(DATASET_DIR)

    # 2️⃣ Train / Temp split
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=random_state
    )

    # 3️⃣ Val / Test split
    val_fraction = val_ratio / (1.0 - train_ratio)
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp,
        y_temp,
        test_size=(1.0 - val_fraction),
        stratify=y_temp,
        random_state=random_state
    )

    print(f"[INFO] Train: {len(p_train)} | Val: {len(p_val)} | Test: {len(p_test)}")

    # 4️⃣ Build features
    print("[STEP] Building TRAIN features (with augmentation)...")
    X_train, y_train = build_features_from_paths(
        p_train, y_train, extractor, augmentor
    )

    print("[STEP] Building VAL features (no augmentation)...")
    X_val, y_val = build_features_from_paths(
        p_val, y_val, extractor
    )

    print("[STEP] Building TEST features (no augmentation)...")
    X_test, y_test = build_features_from_paths(
        p_test, y_test, extractor
    )

    # 5️⃣ Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 6️⃣ Train KNN
    print("[STEP] Training KNN...")
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean",
        weights="distance"
    )
    knn.fit(X_train, y_train)

    # 7️⃣ Validation
    print("\n[VALIDATION RESULTS]")
    val_preds = knn.predict(X_val)
    print("Val Accuracy:", accuracy_score(y_val, val_preds))
    print(classification_report(y_val, val_preds, target_names=CLASSES))

    # 8️⃣ Test
    print("\n[FINAL TEST RESULTS]")
    test_preds = knn.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    # 9️⃣ Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n✔ Model saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")


# =========================
# Prediction with unknown handling
# =========================
def predict_material(image_path, threshold=0.4):
    extractor = CNNFeatureExtractor()
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    img = load_image(image_path)
    feat = extractor.extract(img)
    feat = scaler.transform([feat])

    probs = knn.predict_proba(feat)[0]
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]

    if best_prob < threshold:
        return "Unknown", float(1 - best_prob)

    return CLASSES[best_idx], float(best_prob)


if __name__ == "__main__":
    train_knn_cnn(k=5)
