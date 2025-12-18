# Material Identification System (MSI)

An **Automated Material Stream Identification (MSI) System** built using fundamental **Machine Learning (ML)** techniques.  
This project applies supervised learning models to classify and identify material types from image datasets.

---

## 📌 Features
- **Data Preprocessing**: Tools for cleaning, balancing, and augmenting datasets.
- **Duplicate Removal**: Script (`duplicate_remover.py`) to eliminate redundant images.
- **Models Implemented**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - Convolutional Neural Networks (CNN) for feature extraction
- **Evaluation Metrics**:
  - Accuracy
  - Classification reports
  - Overfitting detection (train vs test performance)
- **Extensible Architecture**: Easy to add new models or datasets.
---

## 📂 Repository Structure
Material-Identification-System/ 

├── data/ # Raw and processed data files 

├── dataset/ # Image datasets for training/testing 

├── models/ # Saved ML models (KNN, SVM, etc.) 

├── src/ # Source code for training & evaluation 

├── duplicate_remover.py # Utility script for dataset cleaning 

├── .gitignore # Git ignore rules 

└── README.md # Project documentation

---

## 🚀 Usage
1. Prepare Dataset Place your material images inside the dataset/ folder.
2. Run duplicate_remover.py to clean duplicates.
3. Run preprocess.py to build features
4. tain the model
5. run main or deploy

---

