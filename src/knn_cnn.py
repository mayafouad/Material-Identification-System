import joblib
import numpy as np
from pathlib import Path
from math import ceil

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
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "knn_cnn.pkl"
SCALER_PATH = MODEL_DIR / "scaler_knn_cnn.pkl"

MODEL_DIR.mkdir(exist_ok=True)


# =========================
# Feature Builder
# =========================
def build_features_from_paths(
    paths,
    labels,
    extractor,
    augmentor=None
):
    """
    Build CNN features from image paths.
    Applies proportional augmentation ONLY if augmentor is provided.
    """
    X, y = [], []

    for cls in np.unique(labels):
        cls_paths = paths[labels == cls]

        imgs = [load_image(p) for p in cls_paths]
        imgs = [img for img in imgs if img is not None]

        if len(imgs) == 0:
            continue

        # -------- Original images --------
        for img in imgs:
            X.append(extractor.extract(img))
            y.append(cls)

        # -------- Augmented images (TRAIN ONLY) --------
        if augmentor is not None:
            num_aug = ceil(len(imgs) * augmentor.increase_percent / 100)

            for i in range(num_aug):
                base_img = imgs[i % len(imgs)]
                aug_img = augmentor.augment(base_img)
                X.append(extractor.extract(aug_img))
                y.append(cls)

    return np.array(X), np.array(y)


# =========================
# Training
# =========================
def train_knn_cnn(
    k=5,
    test_ratio=0.2,
    increase_percent=40,
    random_state=42
):
    print("\n==========================================")
    print(" KNN + CNN (Train/Test → Augment → Extract)")
    print("==========================================\n")

    extractor = CNNFeatureExtractor()
    augmentor = DataAugmentor(increase_percent=increase_percent)

    # =================================================
    # 1️⃣ Load paths ONLY
    # =================================================
    paths, labels = load_dataset_paths(DATASET_DIR)

    if len(paths) == 0:
        raise RuntimeError("No images found in dataset directory")

    # =================================================
    # 2️⃣ Train / Test split (BEFORE augmentation)
    # =================================================
    p_train, p_test, y_train, y_test = train_test_split(
        paths,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state
    )

    print(f"[INFO] Train: {len(p_train)} | Test: {len(p_test)}")

    # =================================================
    # 3️⃣ Augment TRAIN only → Extract features
    # =================================================
    print("[STEP] Building TRAIN features (with augmentation)...")
    X_train, y_train = build_features_from_paths(
        p_train, y_train, extractor, augmentor
    )

    print("[STEP] Building TEST features (no augmentation)...")
    X_test, y_test = build_features_from_paths(
        p_test, y_test, extractor, augmentor=None
    )

    # =================================================
    # 4️⃣ Scale (fit on TRAIN only)
    # =================================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =================================================
    # 5️⃣ Train KNN
    # =================================================
    print("[STEP] Training KNN...")
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean",
        weights="distance"
    )
    knn.fit(X_train, y_train)

    # =================================================
    # 6️⃣ Final Test
    # =================================================
    print("\n[FINAL TEST RESULTS]")
    test_preds = knn.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    # =================================================
    # 7️⃣ Save
    # =================================================
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n✔ Model saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")


# =========================
# Inference with Unknown Handling
# =========================
def predict_material(image_path, unknown_threshold=0.4):
    extractor = CNNFeatureExtractor()
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    img = load_image(image_path)
    feat = extractor.extract(img)
    feat_scaled = scaler.transform([feat])

    probs = knn.predict_proba(feat_scaled)[0]
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]

    if best_prob < unknown_threshold:
        return "Unknown", float(1 - best_prob)

    return CLASSES[best_idx], float(best_prob)


if __name__ == "__main__":
    train_knn_cnn(
        k=5,
        test_ratio=0.2,
        increase_percent=40
    )
