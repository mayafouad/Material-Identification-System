import numpy as np
from pathlib import Path
from math import ceil
from sklearn.model_selection import train_test_split

from cnn_feature_extractor import CNNFeatureExtractor
from data_augmentation import DataAugmentor
from utils import load_image, load_dataset_paths, CLASSES


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES_PATH = OUTPUT_DIR / "features.npz"


def build_features(paths, labels, extractor, augmentor=None):
    X, y = [], []

    for cls in np.unique(labels):
        cls_paths = paths[labels == cls]

        # Load images
        imgs = [load_image(p) for p in cls_paths]
        imgs = [img for img in imgs if img is not None]

        if len(imgs) == 0:
            continue

        imgs_prepared = [
            extractor._prepare_image(img) for img in imgs
        ]

        feats = extractor.model.predict(
            np.stack(imgs_prepared),
            verbose=0
        )

        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)

        X.append(feats)
        y.extend([cls] * len(feats))

        if augmentor is not None:
            num_aug = ceil(len(imgs) * augmentor.increase_percent / 100)
            aug_imgs = []

            for i in range(num_aug):
                aug_imgs.append(
                    augmentor.augment(imgs[i % len(imgs)])
                )

            aug_prepared = [
                extractor._prepare_image(img) for img in aug_imgs
            ]

            aug_feats = extractor.model.predict(
                np.stack(aug_prepared),
                verbose=0
            )

            aug_feats /= (np.linalg.norm(aug_feats, axis=1, keepdims=True) + 1e-6)

            X.append(aug_feats)
            y.extend([cls] * len(aug_feats))

    return np.vstack(X), np.array(y)


def main(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    increase_percent=40,
    random_state=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6


    extractor = CNNFeatureExtractor()
    augmentor = DataAugmentor(increase_percent=increase_percent)

    paths, labels = load_dataset_paths(DATASET_DIR)

    if len(paths) == 0:
        raise RuntimeError("No images found in dataset directory")

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

    print(f"Train images: {len(p_train)}")
    print(f"Val images  : {len(p_val)}")
    print(f"Test images : {len(p_test)}")

    print("\nBuilding TRAIN features (with augmentation)...")
    X_train, y_train = build_features(
        p_train, y_train, extractor, augmentor
    )

    print("\nBuilding VAL features (no augmentation)...")
    X_val, y_val = build_features(
        p_val, y_val, extractor, augmentor=None
    )

    print("\nBuilding TEST features (no augmentation)...")
    X_test, y_test = build_features(
        p_test, y_test, extractor, augmentor=None
    )

    np.savez_compressed(
        FEATURES_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=np.array(CLASSES)
    )

    print("\nPreprocessing complete")
    print(f"Saved to: {FEATURES_PATH}")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_val  : {X_val.shape}")
    print(f"  - X_test : {X_test.shape}")


if __name__ == "__main__":
    main(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        increase_percent=40
    )