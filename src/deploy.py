import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)} device(s) available")
    else:
        print("No GPU detected, using CPU")
except Exception as e:
    print(f"[WARN] GPU config error: {e}")

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

COLORS = {
    'glass': (255, 191, 0),      # Deep Sky Blue
    'paper': (255, 255, 255),    # White
    'cardboard': (42, 107, 175), # Brown/Tan
    'plastic': (0, 255, 255),    # Yellow
    'metal': (192, 192, 192),    # Silver
    'trash': (128, 128, 128),    # Gray
    'unknown': (0, 0, 255),      # Red
}

MATERIAL_INFO = {
    'glass': {'icon': '🍾', 'tip': 'Recyclable - Glass Bin'},
    'paper': {'icon': '📄', 'tip': 'Recyclable - Paper Bin'},
    'cardboard': {'icon': '📦', 'tip': 'Recyclable - Paper/Cardboard Bin'},
    'plastic': {'icon': '🥤', 'tip': 'Recyclable - Plastic Bin'},
    'metal': {'icon': '🥫', 'tip': 'Recyclable - Metal Bin'},
    'trash': {'icon': '🗑️', 'tip': 'Non-Recyclable - General Waste'},
    'unknown': {'icon': '❓', 'tip': 'Cannot identify - Check manually'},
}


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR
if not (SCRIPT_DIR / "models").exists():
    if (SCRIPT_DIR.parent / "models").exists():
        PROJECT_ROOT = SCRIPT_DIR.parent
    else:
        PROJECT_ROOT = Path.cwd()

MODELS_DIR = PROJECT_ROOT / "models"
CAPTURES_DIR = PROJECT_ROOT / "captures"

MODELS_DIR.mkdir(exist_ok=True)
CAPTURES_DIR.mkdir(exist_ok=True)

SVM_MODEL_PATH = MODELS_DIR / "svm_cnn.pkl"
SVM_SCALER_PATH = MODELS_DIR / "scaler_svm_cnn.pkl"
KNN_MODEL_PATH = MODELS_DIR / "knn_cnn.pkl"
KNN_SCALER_PATH = MODELS_DIR / "scaler_knn_cnn.pkl"

SVM_CONFIDENCE_THRESHOLD = 0.5
KNN_CONFIDENCE_THRESHOLD = 0.7


class RealTimeFeatureExtractor:


    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        print("Loading EfficientNetB0 model...")

        try:
            self.model = EfficientNetB0(
                include_top=False,
                weights="imagenet",
                pooling="avg",
                input_shape=(input_size[0], input_size[1], 3)
            )

            dummy_input = np.zeros((1, input_size[0], input_size[1], 3), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            print("Feature extractor ready!")

        except Exception as e:
            print(f"[ERROR] Failed to load EfficientNetB0: {e}")
            print("[TIP] Make sure TensorFlow is properly installed:")
            print("      pip install tensorflow")
            raise

    def extract_from_frame(self, frame):

        try:
            img = cv2.resize(frame, self.input_size)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            x = preprocess_input(img.astype(np.float32))
            x = np.expand_dims(x, axis=0)

            features = self.model.predict(x, verbose=0)[0]

            features = features.astype(np.float32)
            features /= (np.linalg.norm(features) + 1e-6)

            return features

        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return np.zeros(1280, dtype=np.float32)


class MaterialClassifier:

    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.distance_threshold = None
        self.load_model(model_type)

    def load_model(self, model_type):
        self.model_type = model_type

        if model_type == 'svm':
            if not SVM_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"SVM model not found at {SVM_MODEL_PATH}\n"
                    f"Please train the model first or place the model files in: {MODELS_DIR}"
                )

            if not SVM_SCALER_PATH.exists():
                raise FileNotFoundError(
                    f"SVM scaler not found at {SVM_SCALER_PATH}\n"
                    f"Please train the model first or place the scaler file in: {MODELS_DIR}"
                )

            print(f"Loading SVM model from {SVM_MODEL_PATH}")
            self.model = joblib.load(SVM_MODEL_PATH)
            self.scaler = joblib.load(SVM_SCALER_PATH)
            self.distance_threshold = None

        elif model_type == 'knn':
            if not KNN_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"KNN model not found at {KNN_MODEL_PATH}\n"
                    f"Please train the model first or place the model file in: {MODELS_DIR}"
                )

            print(f"Loading KNN model from {KNN_MODEL_PATH}")
            model_data = joblib.load(KNN_MODEL_PATH)

            if isinstance(model_data, dict):
                self.model = model_data['knn']
                self.scaler = model_data.get('scaler')
                self.distance_threshold = model_data.get('distance_threshold', None)
            else:
                self.model = model_data
                self.scaler = None
                self.distance_threshold = None

            if self.scaler is None:
                if KNN_SCALER_PATH.exists():
                    self.scaler = joblib.load(KNN_SCALER_PATH)
                elif SVM_SCALER_PATH.exists():
                    print("[WARN] KNN scaler not found, using SVM scaler")
                    self.scaler = joblib.load(SVM_SCALER_PATH)
                else:
                    raise FileNotFoundError(
                        f"No scaler found. Please ensure scaler file exists in: {MODELS_DIR}"
                    )

        print(f"{model_type.upper()} model loaded successfully!")

    def predict(self, features):

        try:
            features_scaled = self.scaler.transform([features])

            if self.model_type == 'svm':
                return self._predict_svm(features_scaled)
            else:
                return self._predict_knn(features_scaled)

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return 'unknown', 0.0, True

    def _predict_svm(self, features_scaled):
        probs = self.model.predict_proba(features_scaled)[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        if confidence < SVM_CONFIDENCE_THRESHOLD:
            return 'unknown', confidence, True

        class_name = CLASSES[best_idx] if best_idx < len(CLASSES) else 'unknown'
        return class_name, confidence, False

    def _predict_knn(self, features_scaled):
        probs = self.model.predict_proba(features_scaled)[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        distances, _ = self.model.kneighbors(features_scaled)
        mean_distance = np.mean(distances[0])

        if confidence < KNN_CONFIDENCE_THRESHOLD:
            return 'unknown', confidence, True

        if self.distance_threshold and mean_distance > self.distance_threshold:
            return 'unknown', confidence, True

        class_name = CLASSES[best_idx] if best_idx < len(CLASSES) else 'unknown'
        return class_name, confidence, False


class RealTimeApp:

    def __init__(self, model_type='svm', camera_id=0):
        self.camera_id = camera_id
        self.running = False

        self.fps = 0
        self.frame_times = []
        self.inference_time = 0

        self.prediction_history = []
        self.history_size = 5

        print("\n" + "=" * 50)
        print("  MATERIAL STREAM IDENTIFICATION SYSTEM")
        print("  Real-Time Deployment")
        print("=" * 50 + "\n")

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Models directory: {MODELS_DIR}")
        print(f"Captures directory: {CAPTURES_DIR}\n")

        self.extractor = RealTimeFeatureExtractor()
        self.classifier = MaterialClassifier(model_type)

    def smooth_prediction(self, prediction, confidence):
        self.prediction_history.append((prediction, confidence))

        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        votes = {}
        for pred, conf in self.prediction_history:
            if pred not in votes:
                votes[pred] = 0
            votes[pred] += conf

        best_pred = max(votes.keys(), key=lambda x: votes[x])
        avg_conf = votes[best_pred] / len([p for p, c in self.prediction_history if p == best_pred])

        return best_pred, avg_conf

    def draw_ui(self, frame, prediction, confidence, is_unknown):
        h, w = frame.shape[:2]

        # Create semi-transparent overlay for info panel
        overlay = frame.copy()

        # Top info bar
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)

        # Bottom status bar
        cv2.rectangle(overlay, (0, h - 100), (w, h), (30, 30, 30), -1)

        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Get color for prediction
        color = COLORS.get(prediction, (255, 255, 255))

        # Draw title
        title = "MATERIAL IDENTIFICATION SYSTEM"
        cv2.putText(frame, title, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw model type and FPS
        model_info = f"Model: {self.classifier.model_type.upper()} | FPS: {self.fps:.1f} | Inference: {self.inference_time * 1000:.0f}ms"
        cv2.putText(frame, model_info, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw center guide rectangle
        center_x, center_y = w // 2, h // 2
        guide_size = min(w, h) // 3
        guide_color = color if not is_unknown else (0, 0, 255)
        cv2.rectangle(frame,
                      (center_x - guide_size, center_y - guide_size),
                      (center_x + guide_size, center_y + guide_size),
                      guide_color, 3)

        # Draw corner accents
        accent_len = 30
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            x = center_x + dx * guide_size
            y = center_y + dy * guide_size
            cv2.line(frame, (x, y), (x - dx * accent_len, y), guide_color, 5)
            cv2.line(frame, (x, y), (x, y - dy * accent_len), guide_color, 5)

        # Draw prediction result
        result_text = f"DETECTED: {prediction.upper()}"
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2

        # Draw text shadow
        cv2.putText(frame, result_text, (text_x + 2, h - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(frame, result_text, (text_x, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw confidence bar
        bar_width = 300
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 35

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)

        # Filled portion
        fill_width = int(bar_width * confidence)
        bar_color = (0, 255, 0) if confidence > 0.7 else ((0, 255, 255) if confidence > 0.5 else (0, 0, 255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

        # Confidence text
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        cv2.putText(frame, conf_text, (bar_x + bar_width + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw disposal tip
        tip = MATERIAL_INFO.get(prediction, {}).get('tip', '')
        tip_size = cv2.getTextSize(tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, tip, ((w - tip_size[0]) // 2, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw controls hint
        controls = "Q: Quit | S: Switch Model | SPACE: Capture"
        cv2.putText(frame, controls, (w - 350, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def capture_frame(self, frame, prediction, confidence):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{prediction}_{confidence:.2f}_{timestamp}.jpg"
        filepath = CAPTURES_DIR / filename
        cv2.imwrite(str(filepath), frame)
        print(f"Frame captured: {filepath}")
        return filepath

    def run(self):
        print("\nStarting camera...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}!")
            print("[TIP] Try different camera IDs (0, 1, 2) or check camera connection.")
            print("[TIP] Use --camera flag to specify different camera: python deploy.py --camera 1")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        print("Camera initialized successfully!")
        print("\n" + "-" * 40)
        print("CONTROLS:")
        print("  Q or ESC  - Quit application")
        print("  S         - Switch between SVM/KNN")
        print("  SPACE     - Capture current frame")
        print("-" * 40 + "\n")

        self.running = True
        frame_count = 0
        start_time = time.time()

        window_name = "Material Identification System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        try:
            while self.running:
                ret, frame = cap.read()

                if not ret:
                    print("[WARN] Failed to grab frame")
                    continue

                inference_start = time.time()

                features = self.extractor.extract_from_frame(frame)
                prediction, confidence, is_unknown = self.classifier.predict(features)

                self.inference_time = time.time() - inference_start

                prediction, confidence = self.smooth_prediction(prediction, confidence)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0.5:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

                display_frame = self.draw_ui(frame.copy(), prediction, confidence, is_unknown)
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\nQuit requested")
                    self.running = False

                elif key == ord('s'):
                    new_model = 'knn' if self.classifier.model_type == 'svm' else 'svm'
                    print(f"\nSwitching to {new_model.upper()} model...")
                    try:
                        self.classifier.load_model(new_model)
                        self.prediction_history.clear()
                    except FileNotFoundError as e:
                        print(f"[ERROR] {e}")

                elif key == ord(' '):
                    self.capture_frame(frame, prediction, confidence)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            print("Application closed")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['svm', 'knn'],
        default='svm',
        help='Model type to use for classification (default: svm)'
    )

    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )

    args = parser.parse_args()

    try:
        app = RealTimeApp(model_type=args.model, camera_id=args.camera)
        app.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\n[TIP] Make sure to train the models first or place model files in:")
        print(f"      {MODELS_DIR}")
        print("\nRequired files:")
        print("  - svm_cnn.pkl (SVM model)")
        print("  - scaler_cnn.pkl (SVM scaler)")
        print("  - knn_cnn.pkl (KNN model)")
        print("  - scaler_knn_cnn.pkl (KNN scaler)")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
