import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import CLASSES

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FEATURES_PATH = DATA_DIR / "features.npz"
MODEL_PATH = MODEL_DIR / "svm_cnn.pkl"
SCALER_PATH = MODEL_DIR / "scaler_svm_cnn.pkl"

MODEL_DIR.mkdir(exist_ok=True)


def train_svm():
    print("\n==========================================")
    print(" SVM TRAINING (Using Precomputed Features)")
    print("==========================================\n")

    data = np.load(FEATURES_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    print(f"[INFO] Train: {X_train.shape}")
    print(f"[INFO] Val  : {X_val.shape}")
    print(f"[INFO] Test : {X_test.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    svm = SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        probability=True
    )

    print("[STEP] Training SVM...")
    svm.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    print("\n[TRAIN RESULTS]")
    train_preds = svm.predict(X_train)
    print("Train Accuracy:", accuracy_score(y_train, train_preds))

    print("\n[VALIDATION RESULTS]")
    val_preds = svm.predict(X_val)
    print("Val Accuracy:", accuracy_score(y_val, val_preds))
    print(classification_report(y_val, val_preds, target_names=CLASSES))

    print("\n[FINAL TEST RESULTS]")
    test_preds = svm.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    joblib.dump(svm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n✔ Model saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")


if __name__ == "__main__":
    train_svm()