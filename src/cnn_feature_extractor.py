import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input
)


class CNNFeatureExtractor:
    def __init__(self):
        # EfficientNetB0 supports ANY input shape
        self.model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg"   # ALWAYS gives 1280-dim vector
        )

    def extract(self, img_input):
        """
        Args:
            img_input: Either a file path (str) or an RGB numpy array
            
        Returns:
            Normalized 1280-dim feature vector
        """
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Could not load image: {img_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_input  # Already an RGB array

        x = preprocess_input(img.astype(np.float32))
        x = np.expand_dims(x, axis=0)

        # EfficientNet handles variable shapes smoothly
        features = self.model.predict(x, verbose=0)[0]

        # Normalize
        features = features.astype(np.float32)
        features /= (np.linalg.norm(features) + 1e-6)

        return features
