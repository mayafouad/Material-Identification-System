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


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "svm_cnn.pkl"
SCALER_PATH = MODEL_DIR / "scaler_cnn.pkl"

MODEL_DIR.mkdir(exist_ok=True)


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

        for img in imgs:
            X.append(extractor.extract(img))
            y.append(cls)

        # Augmenting Training set ONLY
        if augmentor is not None:
            num_aug = ceil(len(imgs) * augmentor.increase_percent / 100)

            for i in range(num_aug):
                base_img = imgs[i % len(imgs)]
                aug_img = augmentor.augment(base_img)
                X.append(extractor.extract(aug_img))
                y.append(cls)

    return np.array(X), np.array(y)


def train_svm_cnn(
    test_ratio=0.2,
    increase_percent=40,
    random_state=42
):
    print("\n==========================================")
    print(" SVM + CNN (Train/Test → Augment → Extract)")
    print("==========================================\n")

    extractor = CNNFeatureExtractor()
    augmentor = DataAugmentor(increase_percent=increase_percent)

    paths, labels = load_dataset_paths(DATASET_DIR)

    if len(paths) == 0:
        raise RuntimeError("No images found in dataset directory")

    p_train, p_test, y_train, y_test = train_test_split(
        paths,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state
    )

    print(f"[INFO] Train: {len(p_train)} | Test: {len(p_test)}")

    print("[STEP] Building TRAIN features (with augmentation)...")
    X_train, y_train = build_features_from_paths(
        p_train, y_train, extractor, augmentor
    )

    print("[STEP] Building TEST features (no augmentation)...")
    X_test, y_test = build_features_from_paths(
        p_test, y_test, extractor, augmentor=None
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[STEP] Training SVM...")
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )
    svm.fit(X_train, y_train)

    train_preds = svm.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    print("\n[TRAINING RESULTS]")
    print("Train Accuracy:", train_acc)

    test_preds = svm.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print("\n[FINAL TEST RESULTS]")
    print("Test Accuracy:", test_acc)
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))


    joblib.dump(svm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n✔ Model saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")


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
        test_ratio=0.2,
        increase_percent=40
    )
