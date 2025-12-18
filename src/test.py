import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from cnn_feature_extractor import CNNFeatureExtractor
from utils import CLASSES, IDX_TO_CLASS ,load_image

UNKNOWN_THRESHOLD = 0.6
def predict(dataFilePath, bestModelPath):
    
    print(f"Loading model from: {bestModelPath}")
    print(f"Loading images from: {dataFilePath}")

    classifier = joblib.load(bestModelPath)
    print(f"Model loaded: {type(classifier).__name__}")
    model_dir = Path(bestModelPath).parent
    model_name = Path(bestModelPath).stem

    if 'knn' in model_name.lower():
        scaler_path = model_dir / 'scaler_knn_cnn.pkl'
    elif 'svm' in model_name.lower():
        scaler_path = model_dir / 'scaler_svm_cnn.pkl'
    else:
        scaler_path = model_dir / 'scaler_svm_cnn.pkl'
        if not scaler_path.exists():
            scaler_path = model_dir / 'scaler_knn_cnn.pkl'

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found. Expected at: {scaler_path}\n"
            f"Make sure scaler file is in the same directory as the model."
        )

    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")

    extractor = CNNFeatureExtractor()
    print("Loaded feature extractor")

    data_path = Path(dataFilePath)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()

    for ext in image_extensions:
        image_files.update(data_path.glob(f'*{ext}'))
        image_files.update(data_path.glob(f'*{ext.upper()}'))

    # Convert to list and sort for consistent ordering
    image_files = list(image_files)

    print(f"Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print(f"[WARNING] No images found in {dataFilePath}")
        return []

    print("Extracting features...")
    features_list = []
    valid_indices = []  # Track processed images

    for i, img_path in enumerate(image_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")

        try:
            img = load_image(str(img_path))
            features = extractor.extract(img)
            features_list.append(features)
            valid_indices.append(i)
        except Exception as e:
            print(f"[WARNING] Failed to process {img_path.name}: {e}")
            # Set corrupted images as unknown
            continue

    if len(features_list) == 0:
        print("[ERROR] No valid images could be processed")
        return [6] * len(image_files)  # All unknown

    features_array = np.array(features_list)

    print("Scaling features...")
    features_scaled = scaler.transform(features_array)

    print("Predicting...")
    predictions = [6] * len(image_files)  # Initialize all as unknown

    for idx, features in enumerate(features_scaled):
        features_2d = features.reshape(1, -1)

        # Get probabilities
        probs = classifier.predict_proba(features_2d)[0]
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]

        original_idx = valid_indices[idx]
        img_name = image_files[original_idx].name

        print(f"Image {idx+1}/{len(image_files)}: {img_name} : Best class={IDX_TO_CLASS[best_idx]}, Probability={best_prob:.4f}")

        if best_prob < UNKNOWN_THRESHOLD:
            pred_class = 6  # Unknown
        else:
            pred_class = best_idx

        # Map back to original image index
        original_idx = valid_indices[idx]
        predictions[original_idx] = int(pred_class)

    print(f"Predictions complete: {len(predictions)} images classified")

    print(f"Class distribution: ", end="")
    for cls in range(7):
        count = predictions.count(cls)
        if count > 0:
            class_name = CLASSES[cls] if cls < 6 else 'unknown'
            print(f"{class_name}={count} ", end="")
    print()

    return predictions
