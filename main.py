from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import sqlalchemy
import databases
import uvicorn

# ----------------------------
# ✅ 1. Connexion PostgreSQL (Render)
# ----------------------------
DATABASE_URL = "postgresql://spirometry_db_user:hDFjcUUGuTC1NcTG47ta1NuuhPkuu6WK@dpg-d23nhkndiees739qnj1g-a.oregon-postgres.render.com/spirometry_db"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# ----------------------------
# ✅ 2. Définition de la table
# ----------------------------
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

# ❌ Ne PAS créer la table automatiquement sur Render
# engine = sqlalchemy.create_engine(DATABASE_URL)
# metadata.create_all(engine)

# ----------------------------
# ✅ 3. Création de l'app
# ----------------------------
app = FastAPI()

# 🔍 Chargement modèle IA + scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
classes = ["obstructive", "restrictive", "normal", "unknown"]

# 📥 Schéma Pydantic
class InputData(BaseModel):
    FEV1: float
    FVC: float
    FEV1_FVC: float
    SpO2: int
    BPM: int

# ----------------------------
# ✅ 4. Connexion DB Render
# ----------------------------
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# ----------------------------
# ✅ 5. Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "API is running. Use /predict or /backup."}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.FEV1, data.FVC, data.FEV1_FVC, data.SpO2, data.BPM]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return {"diagnostic": classes[prediction]}

@app.post("/backup")
async def backup(data: InputData):
    X = np.array([[data.FEV1, data.FVC, data.FEV1_FVC, data.SpO2, data.BPM]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    diagnostic = classes[prediction]

    # 📝 Insertion dans PostgreSQL
    query = records.insert().values(
        FEV1=data.FEV1,
        FVC=data.FVC,
        FEV1_FVC=data.FEV1_FVC,
        SpO2=data.SpO2,
        BPM=data.BPM,
        diagnostic=diagnostic
    )
    await database.execute(query)

    return {"status": "Backup reussi ", "diagnostic": diagnostic}

# ----------------------------
# ✅ 6. Pour exécution locale
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
