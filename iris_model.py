# iris_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "iris_rf_model.joblib")
joblib.dump(iris.target_names, "target_names.joblib")
