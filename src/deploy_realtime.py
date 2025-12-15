import cv2
import numpy as np
import joblib
import time
from collections import deque
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input
from cnn_feature_extractor import CNNFeatureExtractor
from utils import CLASSES

# Material class configuration
CLASS_NAMES = {i: cls.capitalize() for i, cls in enumerate(CLASSES)}
CLASS_NAMES[6] = "Unknown"

CLASS_COLORS = {
    0: (255, 191, 0),  # Glass - Cyan/Gold
    1: (255, 255, 255),  # Paper - White
    2: (139, 69, 19),  # Cardboard - Brown
    3: (0, 0, 255),  # Plastic - Red
    4: (192, 192, 192),  # Metal - Silver
    5: (0, 128, 0),  # Trash - Green
    6: (128, 128, 128)  # Unknown - Gray
}


class RealtimeClassifier:
    def __init__(self, model_path, scaler_path, confidence_threshold=0.7, debug_mode=False):
        """
        Initialize the real-time classifier

        Args:
            model_path: Path to saved model file (.pkl)
            scaler_path: Path to saved scaler file (.pkl)
            confidence_threshold: Minimum confidence for classification
            debug_mode: Show all class probabilities
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.extractor = CNNFeatureExtractor()
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.prediction_history = deque(maxlen=5)

        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")

    def extract_features(self, frame):
        """Extract features from frame using CNN extractor"""
        # Convert frame to RGB (OpenCV uses BGR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess for EfficientNet (same as in cnn_feature_extractor.py)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(img_rgb.astype(np.float32))
        x = np.expand_dims(x, axis=0)

        # Extract features directly
        features = self.extractor.model.predict(x, verbose=0)[0]

        # Normalize (same as in your extractor)
        features = features.astype(np.float32)
        features /= (np.linalg.norm(features) + 1e-6)

        return features

    def predict_with_confidence(self, features):
        """Make prediction and estimate confidence"""
        # Scale features
        features_scaled = self.scaler.transform([features])

        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            # Debug: print all probabilities
            if self.debug_mode:
                print("\n--- Class Probabilities ---")
                for i, prob in enumerate(probabilities):
                    print(f"{CLASS_NAMES[i]:12s}: {prob:.3f}")
                print(f"Predicted: {CLASS_NAMES[prediction]} ({confidence:.3f})")
        else:
            confidence = 1.0

        # Apply rejection mechanism for "Unknown" class
        if confidence < self.confidence_threshold:
            prediction = 6  # Unknown class
            confidence = 1 - confidence

        return prediction, confidence

    def smooth_prediction(self, prediction):
        """Apply temporal smoothing to reduce flickering"""
        self.prediction_history.append(prediction)

        # Return most common prediction in history
        if len(self.prediction_history) >= 3:
            return max(set(self.prediction_history), key=self.prediction_history.count)
        return prediction

    def draw_results(self, frame, prediction, confidence, fps):
        """Draw classification results on frame"""
        h, w = frame.shape[:2]

        # Create overlay
        overlay = frame.copy()

        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Get class info
        class_name = CLASS_NAMES[prediction]
        class_color = CLASS_COLORS[prediction]

        # Draw class name
        cv2.putText(frame, f"Material: {class_name}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, class_color, 3)

        # Draw confidence
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw colored border
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), class_color, 5)

        # Add instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def run(self, camera_id=0, display_size=(1280, 720)):
        """
        Run real-time classification

        Args:
            camera_id: Camera device ID (0 for default webcam)
            display_size: Window display size (width, height)
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"❌ ERROR: Could not open camera {camera_id}")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "=" * 60)
        print("  🎥 REAL-TIME CLASSIFICATION ACTIVE")
        print("=" * 60)
        print("\nControls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save current frame")
        print(f"  • Confidence threshold: {self.confidence_threshold}")
        print("\n")

        frame_count = 0
        start_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("❌ ERROR: Failed to capture frame")
                break

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            try:
                # Extract features
                features = self.extract_features(frame)

                # Make prediction
                prediction, confidence = self.predict_with_confidence(features)

                # Smooth prediction
                smooth_pred = self.smooth_prediction(prediction)

                # Draw results
                display_frame = self.draw_results(frame, smooth_pred, confidence, fps)

            except Exception as e:
                print(f"⚠ Warning: Classification error - {e}")
                display_frame = frame
                cv2.putText(display_frame, f"Error: {str(e)}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Resize for display
            display_frame = cv2.resize(display_frame, display_size)

            # Show frame
            cv2.imshow('Material Stream Identification System', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Shutting down...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Frame saved as {filename}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Real-time classification stopped.\n")


def main():
    """Main function to run the deployment system"""

    # Set paths relative to project root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Choose your model type: "knn" or "svm"
    MODEL_TYPE = "knn"  # ⚠️ CHANGE THIS TO "svm" IF USING SVM

    if MODEL_TYPE == "knn":
        MODEL_PATH = PROJECT_ROOT / "models" / "knn_cnn.pkl"
        SCALER_PATH = PROJECT_ROOT / "models" / "scaler_knn_cnn.pkl"
    else:  # svm
        MODEL_PATH = PROJECT_ROOT / "models" / "svm_cnn.pkl"
        SCALER_PATH = PROJECT_ROOT / "models" / "scaler_cnn.pkl"

    # Check if files exist
    if not MODEL_PATH.exists():
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please train your model first by running:")
        print(f"  python knn_cnn.py  (for KNN)")
        print(f"  python svm_with_cnn.py  (for SVM)")
        return

    if not SCALER_PATH.exists():
        print(f"\n❌ ERROR: Scaler not found at {SCALER_PATH}")
        print("Scaler should be saved during training.")
        return

    # Configuration
    CAMERA_ID = 0  # 0 for default webcam, 1 for external camera
    CONFIDENCE_THRESHOLD = 0.5  # LOWERED from 0.7 for better detection
    DEBUG_MODE = True  # Show prediction probabilities

    print("\n" + "=" * 60)
    print("  MATERIAL STREAM IDENTIFICATION SYSTEM")
    print("  Real-Time Deployment")
    print("=" * 60)
    print(f"\n📊 Configuration:")
    print(f"  • Model type: {MODEL_TYPE.upper()}")
    print(f"  • Model path: {MODEL_PATH.name}")
    print(f"  • Scaler path: {SCALER_PATH.name}")
    print(f"  • Camera ID: {CAMERA_ID}")
    print(f"  • Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\n⏳ Loading model and scaler...\n")

    try:
        # Initialize classifier
        classifier = RealtimeClassifier(
            model_path=str(MODEL_PATH),
            scaler_path=str(SCALER_PATH),
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        # Run real-time classification
        classifier.run(camera_id=CAMERA_ID)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure model and scaler files exist.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()