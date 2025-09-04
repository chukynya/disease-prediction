import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

base_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_directory, "model", "model.pkl")
indices_path = os.path.join(base_directory, "model", "test_indices.npy")
dataset_path = os.path.join(base_directory, "data", "dataset_10k.csv")

if not os.path.exists(model_path):
    raise FileNotFoundError("model/model.pkl tidak ditemukan. Jalankan train.py terlebih dahulu.")
if not os.path.exists(dataset_path):
    raise FileNotFoundError("data/dataset_10k.csv tidak ditemukan.")
if not os.path.exists(indices_path):
    raise FileNotFoundError("model/test_indices.npy tidak ditemukan. Jalankan train.py terlebih dahulu.")

pipeline = joblib.load(model_path)
dataframe = pd.read_csv(dataset_path)
dataframe = dataframe.drop(columns=["GroupID"], errors="ignore")
test_indices = np.load(indices_path)
feature_columns = [c for c in dataframe.columns if c != "Disease"]
target_column = "Disease"
test_indices_array = np.asarray(test_indices)
features_matrix = dataframe.iloc[test_indices_array][feature_columns]
target_vector = dataframe.iloc[test_indices_array][target_column].astype(str)

predicted_target = pipeline.predict(features_matrix)
model_accuracy = accuracy_score(target_vector, predicted_target)
classification_text = classification_report(target_vector, predicted_target, digits=4)
cm_counts = confusion_matrix(target_vector, predicted_target)

print("=== Evaluasi Model (model/model.pkl pada data/dataset_10k.csv) ===")
print("Akurasi:", f"{model_accuracy:.4f}")
print("\nLaporan Klasifikasi:\n", classification_text)
print("Confusion Matrix (angka):\n", cm_counts)

with np.errstate(invalid="ignore"):
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts, row_sums, where=row_sums != 0)
np.set_printoptions(precision=4, suppress=True)
print("Confusion Matrix (ternormalisasi, 4 desimal):\n", cm_norm)
