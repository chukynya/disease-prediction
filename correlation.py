import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_directory, "data", "dataset_10k.csv")
if not os.path.exists(dataset_path):
    raise FileNotFoundError("data/dataset_10k.csv tidak ditemukan")

dataframe = pd.read_csv(dataset_path)
numeric_dataframe = dataframe.select_dtypes(include=[np.number])
if numeric_dataframe.shape[1] == 0:
    raise ValueError("Tidak ada kolom numerik untuk dihitung korelasinya")

correlation_matrix = numeric_dataframe.corr(method="pearson")
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()

output_path = os.path.join(base_directory, "data", "correlation_heatmap.png")
plt.savefig(output_path, dpi=200)
print("saved:", os.path.relpath(output_path, base_directory))
