from pathlib import Path
from test import predict  

data_folder = Path(__file__).resolve().parents[1]/"testData"
model_file = Path(__file__).resolve().parents[1] / "models/knn_cnn.pkl"  

predictions = predict(data_folder, model_file)

from utils import CLASSES

for pred in predictions:
    print(CLASSES[pred] if pred < 6  else "Unknown")
