# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and target names
model = joblib.load("iris_rf_model.joblib")
target_names = joblib.load("target_names.joblib")

app = FastAPI(title="Iris Species Predictor API")

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_species(data: IrisRequest):
    input_features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_features)[0]
    species = target_names[prediction]
    return {"predicted_species": species}
