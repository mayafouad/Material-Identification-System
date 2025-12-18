"""
test.py - Hidden Dataset Evaluation Script
===========================================
This file is used to evaluate your model on the hidden test dataset.

The function 'predict' takes:
  - dataFilePath: Path to folder containing test images
  - bestModelPath: Path to your trained model file

Returns:
  - List of predictions (class IDs: 0-5, or 6 for unknown)

Usage by evaluators:
    from test import predict
    predictions = predict("path/to/test/images", "path/to/model.pkl")
"""

import os
import cv2
import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Enable GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input
)

# ============================================================================
# CONSTANTS (Matching your utils.py)
# ============================================================================
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Unknown rejection threshold (matching your training scripts)
UNKNOWN_THRESHOLD = 0.4


# ============================================================================
# CNN FEATURE EXTRACTOR (Matching your cnn_feature_extractor.py)
# ============================================================================
class CNNFeatureExtractor:
    """
    Feature extractor using EfficientNetB0.
    Matches your cnn_feature_extractor.py implementation.
    """

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

    def extract_batch(self, imgs):
        """
        Extract features from a batch of images efficiently.
        Args:
            imgs: numpy array of shape (batch_size, H, W, 3), RGB, dtype=float32
        Returns:
            Normalized features of shape (batch_size, 1280)
        """
        # Preprocess batch
        imgs = preprocess_input(imgs.astype(np.float32))

        # EfficientNet predicts features for the whole batch at once
        features = self.model.predict(imgs, verbose=0)

        # Normalize each feature vector individually
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        features = features / norms

        return features.astype(np.float32)


# ============================================================================
# PREDICTION FUNCTION (MAIN INTERFACE)
# ============================================================================
def predict(dataFilePath, bestModelPath):
    """
    Predict classes for all images in the given folder.

    Args:
        dataFilePath (str): Path to folder containing test images
        bestModelPath (str): Path to trained model file (.pkl)

    Returns:
        list: List of predicted class IDs (0-5 for known classes, 6 for unknown)
              Classes: 0=glass, 1=paper, 2=cardboard, 3=plastic, 4=metal, 5=trash, 6=unknown
    """

    print(f"[INFO] Loading model from: {bestModelPath}")
    print(f"[INFO] Loading images from: {dataFilePath}")

    # Load model (classifier only, as per your training scripts)
    classifier = joblib.load(bestModelPath)
    print(f"[INFO] Model loaded: {type(classifier).__name__}")

    # Load scaler from same directory
    model_dir = Path(bestModelPath).parent
    model_name = Path(bestModelPath).stem

    # Determine scaler path based on model name
    if 'knn' in model_name.lower():
        scaler_path = model_dir / 'scaler_knn_cnn.pkl'
    elif 'svm' in model_name.lower():
        scaler_path = model_dir / 'scaler_cnn.pkl'
    else:
        # Try both
        scaler_path = model_dir / 'scaler_cnn.pkl'
        if not scaler_path.exists():
            scaler_path = model_dir / 'scaler_knn_cnn.pkl'

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found. Expected at: {scaler_path}\n"
            f"Make sure scaler file is in the same directory as the model."
        )

    scaler = joblib.load(scaler_path)
    print(f"[INFO] Scaler loaded from: {scaler_path}")

    # Initialize feature extractor
    print("[INFO] Loading EfficientNetB0 feature extractor...")
    extractor = CNNFeatureExtractor()
    print("[INFO] Feature extractor ready")

    # Get all image files
    data_path = Path(dataFilePath)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()  # Use set to avoid duplicates

    for ext in image_extensions:
        image_files.update(data_path.glob(f'*{ext}'))
        image_files.update(data_path.glob(f'*{ext.upper()}'))

    # Convert to list and sort for consistent ordering
    image_files = sorted(list(image_files), key=lambda x: x.name)

    print(f"[INFO] Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print(f"[WARNING] No images found in {dataFilePath}")
        return []

    # Extract features for all images
    print("[INFO] Extracting features...")
    features_list = []
    valid_indices = []  # Track which images were successfully processed

    for i, img_path in enumerate(image_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[INFO] Processing image {i+1}/{len(image_files)}: {img_path.name}")

        try:
            features = extractor.extract(str(img_path))
            features_list.append(features)
            valid_indices.append(i)
        except Exception as e:
            print(f"[WARN] Failed to process {img_path.name}: {e}")
            # For corrupted images, predict as unknown (class 6)
            continue

    if len(features_list) == 0:
        print("[ERROR] No valid images could be processed")
        return [6] * len(image_files)  # All unknown

    features_array = np.array(features_list)

    # Scale features
    print("[INFO] Scaling features...")
    features_scaled = scaler.transform(features_array)

    # Make predictions
    print("[INFO] Making predictions...")
    predictions = [6] * len(image_files)  # Initialize all as unknown

    for idx, features in enumerate(features_scaled):
        features_2d = features.reshape(1, -1)

        # Get probabilities
        probs = classifier.predict_proba(features_2d)[0]
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]

        # Apply unknown threshold (matching your training scripts)
        if best_prob < UNKNOWN_THRESHOLD:
            pred_class = 6  # Unknown
        else:
            pred_class = best_idx

        # Map back to original image index
        original_idx = valid_indices[idx]
        predictions[original_idx] = int(pred_class)

    print(f"[INFO] Predictions complete: {len(predictions)} images classified")

    # Print class distribution
    print(f"[INFO] Class distribution: ", end="")
    for cls in range(7):
        count = predictions.count(cls)
        if count > 0:
            class_name = CLASSES[cls] if cls < 6 else 'unknown'
            print(f"{class_name}={count} ", end="")
    print()

    return predictions


# ============================================================================
# MAIN (for testing)
# ============================================================================
if __name__ == "__main__":
    """
    Test the predict function locally.
    
    Usage:
        python test.py
        
    Note: Update the paths below to match your setup
    """

    import sys

    # Example usage - modify these paths for your testing
    if len(sys.argv) >= 3:
        # Command line arguments
        test_data_path = sys.argv[1]
        model_path = sys.argv[2]
    else:
        # Default paths - UPDATE THESE
        test_data_path = "../testing/"  # Folder containing test images
        model_path = "../models/svm_cnn.pkl"       # Your best model file

    # Check if paths exist
    if not Path(test_data_path).exists():
        print(f"[ERROR] Test data path does not exist: {test_data_path}")
        print("[TIP] Update the 'test_data_path' variable in the script")
        print("[USAGE] python test.py <test_images_folder> <model_path>")
        exit(1)

    if not Path(model_path).exists():
        print(f"[ERROR] Model path does not exist: {model_path}")
        print("[TIP] Update the 'model_path' variable in the script")
        print("[USAGE] python test.py <test_images_folder> <model_path>")
        exit(1)

    # Run prediction
    print("\n" + "="*70)
    print("TESTING PREDICT FUNCTION")
    print("="*70 + "\n")

    try:
        predictions = predict(test_data_path, model_path)

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Total predictions: {len(predictions)}")
        print(f"Predictions: {predictions[:20]}{'...' if len(predictions) > 20 else ''}")
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
