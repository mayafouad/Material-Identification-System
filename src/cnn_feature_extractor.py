import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input
)


class CNNFeatureExtractor:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size

        self.model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None")

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img.shape}")

        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = preprocess_input(img)

        return img

    def extract(self, img: np.ndarray) -> np.ndarray:
        img = self._prepare_image(img)
        img = np.expand_dims(img, axis=0)
        features = self.model.predict(img, verbose=0)[0]
        features = features.astype(np.float32)
        features /= (np.linalg.norm(features) + 1e-6)
        return features

    def extract_batch(self, imgs: np.ndarray) -> np.ndarray:

        if len(imgs) == 0:
            raise ValueError("Empty image batch")

        imgs = np.stack([self._prepare_image(img) for img in imgs])

        features = self.model.predict(imgs, verbose=0)

        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        features = features / norms

        return features.astype(np.float32)
