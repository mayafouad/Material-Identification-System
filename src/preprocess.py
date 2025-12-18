import numpy as np
from pathlib import Path
from math import ceil
from sklearn.model_selection import train_test_split

from cnn_feature_extractor import CNNFeatureExtractor
from data_augmentation import DataAugmentor
from utils import load_image, load_dataset_paths



BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)


def build_features(paths, labels, extractor, augmentor=None):
    X, y = [], []

    for img_path, label in zip(paths, labels):
        img = load_image(img_path)
        if img is None:
            continue

        # original
        X.append(extractor.extract(img))
        y.append(label)

        # augmentation (TRAIN only)
        if augmentor is not None:
            num_aug = ceil(augmentor.increase_percent / 100)
            for _ in range(num_aug):
                aug_img = augmentor.augment(img)
                X.append(extractor.extract(aug_img))
                y.append(label)

    return np.array(X), np.array(y)


def main(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    increase_percent=40,
    random_state=42
):
    extractor = CNNFeatureExtractor()
    augmentor = DataAugmentor(increase_percent=increase_percent)

    paths, labels = load_dataset_paths(DATASET_DIR)

    # Split
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(1 - train_ratio),
        stratify=labels, random_state=random_state
    )

    val_fraction = val_ratio / (val_ratio + test_ratio)
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp, test_size=(1 - val_fraction),
        stratify=y_temp, random_state=random_state
    )

    print("Building TRAIN features...")
    X_train, y_train = build_features(p_train, y_train, extractor, augmentor)

    print("Building VAL features...")
    X_val, y_val = build_features(p_val, y_val, extractor)

    print("Building TEST features...")
    X_test, y_test = build_features(p_test, y_test, extractor)

    # Save
    np.savez(OUT_DIR / "train.npz", X=X_train, y=y_train)
    np.savez(OUT_DIR / "val.npz",   X=X_val,   y=y_val)
    np.savez(OUT_DIR / "test.npz",  X=X_test,  y=y_test)

    print("✔ Preprocessing complete")


if __name__ == "__main__":
    main()
