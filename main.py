from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn
import sqlalchemy
import databases

# ---------------------
# Configuration base de données SQLite
# ---------------------
DATABASE_URL = "sqlite:///./spirometry_data.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

records = sqlalchemy.Table(
    "records",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("FEV1", sqlalchemy.Float),
    sqlalchemy.Column("FVC", sqlalchemy.Float),
    sqlalchemy.Column("FEV1_FVC", sqlalchemy.Float),
    sqlalchemy.Column("SpO2", sqlalchemy.Integer),
    sqlalchemy.Column("BPM", sqlalchemy.Integer),
    sqlalchemy.Column("diagnostic", sqlalchemy.String),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

# ---------------------
# Application FastAPI
# ---------------------
app = FastAPI()

# Chargement du modèle IA et du scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
classes = ["obstructive", "restrictive", "normal", "unknown"]

# Schéma des données en entrée
class InputData(BaseModel):
    FEV1: float
    FVC: float
    FEV1_FVC: float
    SpO2: int
    BPM: int

# Connexion/déconnexion à la base
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Test de santé
@app.get("/")
def root():
    return {"message": "API is running. Use /predict or /backup."}

# Endpoint prédiction seule
@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.FEV1, data.FVC, data.FEV1_FVC, data.SpO2, data.BPM]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return {"diagnostic": classes[prediction]}

# Endpoint prédiction + sauvegarde dans SQLite
@app.post("/backup")
async def backup(data: InputData):
    X = np.array([[data.FEV1, data.FVC, data.FEV1_FVC, data.SpO2, data.BPM]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    diagnostic = classes[prediction]

    # Insertion dans la base
    query = records.insert().values(
        FEV1=data.FEV1,
        FVC=data.FVC,
        FEV1_FVC=data.FEV1_FVC,
        SpO2=data.SpO2,
        BPM=data.BPM,
        diagnostic=diagnostic
    )
    await database.execute(query)

    return {"status": "Backup réussi ✅", "diagnostic": diagnostic}

# Lancement du serveur compatible Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render fournit le port
    uvicorn.run("main:app", host="0.0.0.0", port=port)

