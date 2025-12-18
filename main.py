from test import predict
from pathlib import Path
data_folder =  Path(__file__).parent /"testData"
model_file = Path(__file__).parent/ "models/svm_cnn.pkl"

predictions = predict(data_folder, model_file)
