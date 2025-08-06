# 🌸 Iris Species Predictor API

A simple machine learning API built with FastAPI to predict the species of iris flowers using a Random Forest Classifier trained on the classic Iris dataset.

---

## 📦 Features

- Predict Iris species based on:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- RESTful API endpoint using FastAPI
- Pre-trained model serialized using `joblib`

---

## 🧠 Model Information

- Dataset: `sklearn.datasets.load_iris()`
- Algorithm: `RandomForestClassifier`
- Accuracy: >95% (on basic training split)
