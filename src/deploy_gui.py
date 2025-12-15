import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input
from cnn_feature_extractor import CNNFeatureExtractor
from utils import CLASSES


class MSIDeploymentGUI:
    """GUI-based Material Stream Identification System Deployment"""

    def __init__(self, root):
        self.root = root
        self.root.title("Material Stream Identification System - Real-Time Deployment")
        self.root.geometry("1400x800")

        # State variables
        self.model = None
        self.scaler = None
        self.extractor = CNNFeatureExtractor()
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.prediction_log = []

        # Class information from utils
        self.class_names = {i: cls.capitalize() for i, cls in enumerate(CLASSES)}
        self.class_names[6] = "Unknown"

        self.class_colors = {
            0: "#FFD700", 1: "#FFFFFF", 2: "#8B4513",
            3: "#FF0000", 4: "#C0C0C0", 5: "#008000", 6: "#808080"
        }

        self.setup_ui()

    def setup_ui(self):
        """Setup the GUI layout"""

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Camera feed
        left_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding="10")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(left_frame,
                                     text="Camera feed will appear here\n\nLoad a model and start the camera to begin")
        self.video_label.pack(expand=True)

        # Right panel - Controls and Stats
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Model Configuration
        config_frame = ttk.LabelFrame(right_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=5)

        ttk.Button(config_frame, text="📁 Load Model & Scaler", command=self.load_model).pack(fill=tk.X, pady=2)

        ttk.Label(config_frame, text="Camera ID:").pack(pady=(10, 0))
        self.camera_id_var = tk.StringVar(value="0")
        ttk.Entry(config_frame, textvariable=self.camera_id_var, width=10).pack()

        ttk.Label(config_frame, text="Confidence Threshold:").pack(pady=(10, 0))
        self.confidence_var = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(config_frame, from_=0.0, to=1.0,
                                    variable=self.confidence_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X)
        self.threshold_label = ttk.Label(config_frame, text="0.70")
        self.threshold_label.pack()

        # Update threshold label when slider moves
        def update_threshold_label(val):
            self.threshold_label.config(text=f"{float(val):.2f}")

        threshold_scale.config(command=update_threshold_label)

        # Control Buttons
        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(control_frame, text="▶ Start Camera",
                                    command=self.start_camera)
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop Camera",
                                   command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        ttk.Button(control_frame, text="📸 Capture Frame",
                   command=self.capture_frame).pack(fill=tk.X, pady=2)

        ttk.Button(control_frame, text="💾 Export Log",
                   command=self.export_log).pack(fill=tk.X, pady=2)

        # Current Prediction Display
        pred_frame = ttk.LabelFrame(right_frame, text="Current Prediction", padding="10")
        pred_frame.pack(fill=tk.X, pady=5)

        self.pred_class_label = ttk.Label(pred_frame, text="Class: N/A",
                                          font=("Arial", 14, "bold"))
        self.pred_class_label.pack()

        self.pred_conf_label = ttk.Label(pred_frame, text="Confidence: N/A",
                                         font=("Arial", 12))
        self.pred_conf_label.pack()

        self.pred_color_canvas = tk.Canvas(pred_frame, height=30, bg="gray")
        self.pred_color_canvas.pack(fill=tk.X, pady=5)

        # Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)

        self.fps_label = ttk.Label(stats_frame, text="FPS: 0.0")
        self.fps_label.pack()

        self.frame_count_label = ttk.Label(stats_frame, text="Frames: 0")
        self.frame_count_label.pack()

        self.model_status_label = ttk.Label(stats_frame, text="Model: Not Loaded ❌",
                                            foreground="red")
        self.model_status_label.pack()

        # Prediction History
        history_frame = ttk.LabelFrame(right_frame, text="Prediction History", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.history_text = tk.Text(history_frame, height=10, width=30, wrap=tk.WORD)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(history_frame, command=self.history_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.config(yscrollcommand=scrollbar.set)

    def load_model(self):
        """Load the trained model and scaler"""
        model_path = filedialog.askopenfilename(
            title="Select Model File (knn_cnn.pkl or svm_cnn.pkl)",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if model_path:
            try:
                # Load model
                self.model = joblib.load(model_path)
                model_name = Path(model_path).stem

                # Determine scaler name based on model name
                if "knn" in model_name.lower():
                    scaler_name = "scaler_knn_cnn.pkl"
                elif "svm" in model_name.lower():
                    scaler_name = "scaler_cnn.pkl"
                else:
                    scaler_name = "scaler.pkl"

                scaler_path = Path(model_path).parent / scaler_name

                # Load scaler
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                    status_msg = f"Model: {model_name} ✓"
                    self.model_status_label.config(text=status_msg, foreground="green")
                    messagebox.showinfo("Success",
                                        f"✓ Model loaded: {model_name}\n✓ Scaler loaded: {scaler_name}")
                else:
                    messagebox.showerror("Error",
                                         f"Scaler not found!\n\nExpected: {scaler_path}\n\nPlease ensure the scaler file exists in the same directory as the model.")
                    return

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def start_camera(self):
        """Start the camera feed and classification"""
        if self.model is None or self.scaler is None:
            messagebox.showwarning("Warning", "Please load both model and scaler first!")
            return

        try:
            camera_id = int(self.camera_id_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid camera ID! Please enter a number.")
            return

        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {camera_id}")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Start video thread
        self.video_thread = threading.Thread(target=self.update_video, daemon=True)
        self.video_thread.start()

    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def update_video(self):
        """Update video feed and perform classification"""
        frame_count = 0
        start_time = time.time()

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame = frame.copy()

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            try:
                # Perform classification
                features = self.extract_features(frame)
                prediction, confidence = self.predict(features)

                # Draw results on frame
                display_frame = self.draw_results(frame, prediction, confidence, fps)

                # Update prediction display
                self.update_prediction_display(prediction, confidence)

                # Log prediction every 10 frames (not 30)
                if frame_count % 10 == 0:
                    self.log_prediction(prediction, confidence)

            except Exception as e:
                display_frame = frame
                print(f"Classification error: {e}")

            # Convert to PhotoImage and display
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (800, 600))
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update statistics
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.frame_count_label.config(text=f"Frames: {frame_count}")

            # Reset counter periodically
            if elapsed > 2.0:
                frame_count = 0
                start_time = time.time()

            time.sleep(0.01)

    def extract_features(self, frame):
        """Extract features from frame"""
        # Convert frame to RGB (OpenCV uses BGR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess for EfficientNet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(img_rgb.astype(np.float32))
        x = np.expand_dims(x, axis=0)

        # Extract features directly
        features = self.extractor.model.predict(x, verbose=0)[0]

        # Normalize
        features = features.astype(np.float32)
        features /= (np.linalg.norm(features) + 1e-6)

        return features

    def predict(self, features):
        """Make prediction"""
        # Scale features
        features_scaled = self.scaler.transform([features])

        # Get prediction
        prediction = self.model.predict(features_scaled)[0]

        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probs)
        else:
            confidence = 1.0

        # Apply unknown rejection
        if confidence < self.confidence_var.get():
            prediction = 6  # Unknown
            confidence = 1 - confidence

        return prediction, confidence

    def draw_results(self, frame, prediction, confidence, fps):
        """Draw classification results on frame"""
        h, w = frame.shape[:2]
        class_name = self.class_names[prediction]

        # Draw background
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)

        # Draw text
        cv2.putText(frame, f"{class_name}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"Conf: {confidence:.2%}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def update_prediction_display(self, prediction, confidence):
        """Update the prediction display panel"""
        class_name = self.class_names[prediction]
        color = self.class_colors[prediction]

        self.pred_class_label.config(text=f"Class: {class_name}")
        self.pred_conf_label.config(text=f"Confidence: {confidence:.2%}")
        self.pred_color_canvas.config(bg=color)

    def log_prediction(self, prediction, confidence):
        """Log prediction to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        class_name = self.class_names[prediction]

        log_entry = f"[{timestamp}] {class_name} ({confidence:.2%})\n"
        self.prediction_log.append(log_entry)

        self.history_text.insert(tk.END, log_entry)
        self.history_text.see(tk.END)

        # Keep only last 100 entries
        if len(self.prediction_log) > 100:
            self.prediction_log.pop(0)

    def capture_frame(self):
        """Capture and save current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Success", f"Frame saved as:\n{filename}")
        else:
            messagebox.showwarning("Warning", "No frame to capture. Start the camera first.")

    def export_log(self):
        """Export prediction log to file"""
        if not self.prediction_log:
            messagebox.showwarning("Warning", "No predictions to export yet.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        if filename:
            with open(filename, 'w') as f:
                f.writelines(self.prediction_log)
            messagebox.showinfo("Success", f"Log exported to:\n{filename}")

    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MSIDeploymentGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()