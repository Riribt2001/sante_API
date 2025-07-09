from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Chargement des fichiers
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

classes = ["obstructive", "restrictive", "normal", "unknown"]

class InputData(BaseModel):
    FEV1: float
    FVC: float
    FEV1_FVC: float
    SpO2: int
    BPM: int

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.FEV1, data.FVC, data.FEV1_FVC, data.SpO2, data.BPM]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return {"diagnostic": classes[prediction]}
