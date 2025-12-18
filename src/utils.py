import cv2
import numpy as np
from pathlib import Path

# =========================
# Class definitions
# =========================
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


# =========================
# Dataset utilities
# =========================
def get_class_counts(dataset_dir):
    """
    Count number of images per class.
    """
    dataset_dir = Path(dataset_dir)
    counts = {}

    for cls in CLASSES:
        cls_path = dataset_dir / cls
        if cls_path.exists():
            counts[cls] = len(list(cls_path.glob("*.jpg")))
        else:
            counts[cls] = 0

    return counts


def load_image(image_path, color_mode="rgb"):
    """
    Load a single image from disk.

    Args:
        image_path: Path or str
        color_mode: "rgb" or "gray"

    Returns:
        np.ndarray or None
    """
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[WARN] Skipping corrupted image: {image_path}")
        return None

    if color_mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def load_dataset_paths(dataset_dir):
    """
    Load dataset as image paths + labels ONLY.

    This function does NOT:
    - extract CNN features
    - apply augmentation
    - touch training logic

    Returns:
        paths: np.ndarray of Path
        labels: np.ndarray of int
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    paths = []
    labels = []

    print(f"\n[INFO] Loading dataset paths from: {dataset_dir}")

    for label, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            print(f"[WARN] Missing folder: {class_dir}")
            continue

        image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"[INFO] {class_name}: {len(image_paths)} images")

        for img_path in image_paths:
            paths.append(img_path)
            labels.append(label)

    return np.array(paths), np.array(labels)
