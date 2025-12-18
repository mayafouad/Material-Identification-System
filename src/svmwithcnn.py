import joblib
import numpy as np
from pathlib import Path
from math import ceil

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cnn_feature_extractor import CNNFeatureExtractor
from data_augmentation import DataAugmentor
from utils import CLASSES, load_image, load_dataset_paths


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "svm_cnn.pkl"
SCALER_PATH = MODEL_DIR / "scaler_cnn.pkl"

MODEL_DIR.mkdir(exist_ok=True)
# =========================


def build_features_from_paths(
    paths,
    labels,
    extractor,
    augmentor=None
):
    X, y = [], []

    for cls in np.unique(labels):
        cls_paths = paths[labels == cls]

        imgs = [load_image(p) for p in cls_paths]
        imgs = [img for img in imgs if img is not None]

        if len(imgs) == 0:
            continue

        # ---- ORIGINAL IMAGES ----
        feats = np.array([extractor.extract(img) for img in imgs])
        X.append(feats)
        y.extend([cls] * len(feats))

        # ---- AUGMENTED IMAGES (TRAIN ONLY) ----
        if augmentor is not None:
            num_aug = ceil(len(imgs) * augmentor.increase_percent / 100)
            aug_feats = []

            for i in range(num_aug):
                aug_img = augmentor.augment(imgs[i % len(imgs)])
                feat = extractor.extract(aug_img)
                aug_feats.append(feat)

            aug_feats = np.array(aug_feats)
            X.append(aug_feats)
            y.extend([cls] * len(aug_feats))

    return np.vstack(X), np.array(y)



def train_svm_cnn(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    increase_percent=40,
    random_state=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    print("\n==========================================")
    print(" SVM + CNN (Split → Augment → Extract)")
    print("==========================================\n")

    extractor = CNNFeatureExtractor()
    augmentor = DataAugmentor(increase_percent=increase_percent)

    # =================================================
    # 1️⃣ LOAD PATHS ONLY
    # =================================================
    paths, labels = load_dataset_paths(DATASET_DIR)

    if len(paths) == 0:
        raise RuntimeError("No images found in dataset directory")

    # =================================================
    # 2️⃣ SPLIT (BEFORE ANY AUGMENTATION OR CNN)
    # =================================================
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=random_state
    )

    val_fraction = val_ratio / (val_ratio + test_ratio)

    p_val, p_test, y_val, y_test = train_test_split(
        p_temp,
        y_temp,
        test_size=(1.0 - val_fraction),
        stratify=y_temp,
        random_state=random_state
    )

    print(f"[INFO] Train: {len(p_train)} | Val: {len(p_val)} | Test: {len(p_test)}")

    # =================================================
    # 3️⃣ AUGMENT (TRAIN ONLY) → EXTRACT FEATURES
    # =================================================
    print("[STEP] Building TRAIN features (with augmentation)...")
    X_train, y_train = build_features_from_paths(
        p_train, y_train, extractor, augmentor
    )

    print("[STEP] Building VAL features (no augmentation)...")
    X_val, y_val = build_features_from_paths(
        p_val, y_val, extractor, augmentor=None
    )

    print("[STEP] Building TEST features (no augmentation)...")
    X_test, y_test = build_features_from_paths(
        p_test, y_test, extractor, augmentor=None
    )

    # =================================================
    # 4️⃣ SCALE (FIT ON TRAIN ONLY)
    # =================================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # =================================================
    # 5️⃣ TRAIN SVM
    # =================================================
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )

    print("[STEP] Training SVM...")
    svm.fit(X_train, y_train)

    # =================================================
    # 6️⃣ VALIDATION
    # =================================================
    print("\n[VALIDATION RESULTS]")
    val_preds = svm.predict(X_val)
    print("Val Accuracy:", accuracy_score(y_val, val_preds))
    print(classification_report(y_val, val_preds, target_names=CLASSES))

    # =================================================
    # 7️⃣ FINAL TEST (ONCE)
    # =================================================
    print("\n[FINAL TEST RESULTS]")
    test_preds = svm.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    # =================================================
    # 8️⃣ SAVE
    # =================================================
    joblib.dump(svm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n✔ Model saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")


# =========================
# Inference
# =========================
def predict_material(image_path, unknown_threshold=0.4):
    extractor = CNNFeatureExtractor()
    svm = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    img = load_image(image_path)
    feat = extractor.extract(img)

    feat_scaled = scaler.transform([feat])
    probs = svm.predict_proba(feat_scaled)[0]

    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]

    if best_prob < unknown_threshold:
        return "Unknown", float(1 - best_prob)

    return CLASSES[best_idx], float(best_prob)


if __name__ == "__main__":
    train_svm_cnn(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        increase_percent=40
    )
