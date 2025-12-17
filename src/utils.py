import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Class definitions
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


def get_class_counts(dir):
    dir = Path(dir)
    counts = {}

    for cls in CLASSES:
        cls_path = dir / cls
        if cls_path.exists():
            jpg_count = len(list(cls_path.glob('*.jpg')))
            counts[cls] = jpg_count
        else:
            counts[cls] = 0

    return counts


def load_image(image_path, color_mode='rgb'):
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[WARN] Skipping corrupted image: {image_path}")
        return None

    if color_mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def load_dataset(dataset_dir, extractor):
    X, y, paths = [], [], []

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
                paths.append(img_path)  # Keep track of paths for augmentation
            except Exception as e:
                print(f"[WARN] Failed to process {img_path}: {e}")

    return np.array(X), np.array(y), paths
