def predict(dataFilePath, bestModelPath):

    # imports
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import cv2
    import joblib
    from pathlib import Path
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB0,
        preprocess_input
    )

    # constants
    CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
    UNKNOWN_CLASS = 6
    UNKNOWN_THRESHOLD = 0.5
    INPUT_SIZE = (224, 224)

    if bestModelPath is None:
        bestModelPath = Path(__file__).parent / "models" / "svm_cnn.pkl"

    print(f"Loading model from: {bestModelPath}")
    print(f"Loading images from: {dataFilePath}")


    # load model and scaler
    classifier = joblib.load(bestModelPath)
    print(f"Model loaded: {type(classifier).__name__}")

    model_dir = Path(bestModelPath).parent
    model_name = Path(bestModelPath).stem.lower()

    if 'knn' in model_name:
        scaler_path = model_dir / 'scaler_knn_cnn.pkl'
    else:
        scaler_path = model_dir / 'scaler_svm_cnn.pkl'

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")


    cnn_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    def prepare_image(img):
        if img is None:
            raise ValueError("Image is None")

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img.shape}")

        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return img

    def extract_features(img):
        img = prepare_image(img)
        img = np.expand_dims(img, axis=0)
        features = cnn_model.predict(img, verbose=0)[0]
        features = features.astype(np.float32)
        features /= (np.linalg.norm(features) + 1e-6)
        return features


    # Load images
    data_path = Path(dataFilePath)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))

    image_files = sorted(set(image_files))
    print(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        print("[WARNING] No images found")
        return []


    # Feature extraction

    features_list = []
    valid_indices = []

    print("Extracting features...")
    for i, img_path in enumerate(image_files):
        if i == 0 or (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Corrupted image")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features = extract_features(img)

            features_list.append(features)
            valid_indices.append(i)

        except Exception as e:
            print(f"[WARNING] Failed on {img_path.name}: {e}")

    if len(features_list) == 0:
        print("[ERROR] No valid images processed")
        return [UNKNOWN_CLASS] * len(image_files)

    features_array = np.array(features_list)


    print("Scaling features...")
    features_scaled = scaler.transform(features_array)

    predictions = [UNKNOWN_CLASS] * len(image_files)

    print("Predicting...")
    for i, feat in enumerate(features_scaled):
        probs = classifier.predict_proba(feat.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        best_prob = probs[best_idx]

        original_idx = valid_indices[i]
        img_name = image_files[original_idx].name

        print(
            f"{img_name} → "
            f"{IDX_TO_CLASS.get(best_idx, 'unknown')} "
            f"(prob={best_prob:.4f})"
        )

        if best_prob >= UNKNOWN_THRESHOLD:
            predictions[original_idx] = best_idx

    print("Prediction summary:")
    for cls in range(7):
        count = predictions.count(cls)
        if count > 0:
            name = CLASSES[cls] if cls < 6 else "unknown"
            print(f"{name}: {count}")

    return predictions
