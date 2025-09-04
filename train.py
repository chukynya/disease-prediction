import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

base_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_directory, "data", "dataset_10k.csv")
if not os.path.exists(dataset_path):
	raise FileNotFoundError("data/dataset_10k.csv tidak ditemukan")

dataframe = pd.read_csv(dataset_path)
dataframe = dataframe.drop(columns=["GroupID"], errors="ignore")
feature_columns = [c for c in dataframe.columns if c != "Disease"]
target_column = "Disease"
features_matrix = dataframe[feature_columns]
target_vector = dataframe[target_column].astype(str)
train_features, test_features, train_target, test_target, train_index, test_index = train_test_split(
	features_matrix,
	target_vector,
	np.arange(len(dataframe)),
	test_size=0.2,
	random_state=42,
	stratify=target_vector,
)

preprocess_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

random_forest_classifier = RandomForestClassifier(
	n_estimators=200,
	random_state=42,
	n_jobs=-1,
	class_weight="balanced_subsample",
)

model_pipeline = Pipeline([
	("preprocess", preprocess_pipeline),
	("rf", random_forest_classifier),
])

hyperparameter_grid = {
	"rf__n_estimators": [100, 200, 300],
	"rf__max_depth": [None, 10, 20],
	"rf__min_samples_split": [2, 5],
	"rf__min_samples_leaf": [1, 2],
	"rf__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
	estimator=model_pipeline,
	param_grid=hyperparameter_grid,
	scoring="accuracy",
	cv=5,
	n_jobs=1,
	refit=True,
)

grid_search.fit(train_features, train_target)
best_pipeline = grid_search.best_estimator_

model_directory = os.path.join(base_directory, "model")
os.makedirs(model_directory, exist_ok=True)
model_path = os.path.join(model_directory, "model.pkl")
joblib.dump(best_pipeline, model_path)
np.save(os.path.join(model_directory, "test_indices.npy"), test_index)
print("saved:", os.path.relpath(model_path, base_directory))

