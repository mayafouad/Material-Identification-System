import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.feature_extraction import FeatureExtractor
CLASSES = ['glass', 'plastic']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Assuming FeatureExtractor class is already defined above

def build_dataset(data_dir, extractor):
    X, y = [], []
    data_dir = Path(data_dir)

    for cls in CLASSES:
        cls_path = data_dir / cls
        if not cls_path.exists():
            continue

        for img_path in tqdm(list(cls_path.glob("*.jpg")), desc=f"Processing {cls}"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Skipping corrupted image: {img_path}")
                continue

            # Extract features
            features = extractor.extract_features(img)
            X.append(features)
            y.append(CLASS_TO_IDX[cls])

    return np.array(X), np.array(y)


def train_knn(X, y, k=5):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    return knn


if __name__ == "__main__":

    extractor = FeatureExtractor()
    data_dir = "sample"  

    print("Building dataset...")
    X, y = build_dataset(data_dir, extractor)

    print("Training KNN...")
    knn_model = train_knn(X, y, k=2)  

