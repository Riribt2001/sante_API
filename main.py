from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

app = FastAPI()

# Chargement du modèle et du scaler (assure-toi que ces fichiers sont dans le même dossier)
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Classes de diagnostic
classes = ["obstructive", "restrictive", "normal", "unknown"]

# Modèle de données attendues en entrée
class InputData(BaseModel):
    FEV1: float
    FVC: float
    FEV1_FVC: float
    SpO2: int
    BPM: int

# Route racine (pour tester que l'API fonctionne)
@app.get("/")
def root():
    return {"message": "API is running. Use /predict endpoint."}

# Route de prédiction
@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.FEV1, data.FVC, data.FEV1_FVC, data.SpO2, data.BPM]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return {"diagnostic": classes[prediction]}

# Lancement du serveur en prenant le port défini par Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render donne le port via variable d'environnement
    uvicorn.run("main:app", host="0.0.0.0", port=port)
