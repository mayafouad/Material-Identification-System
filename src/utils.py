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





