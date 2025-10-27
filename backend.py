from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI(title="Mechanical Failure API")
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")


class Sensors(BaseModel):
    sensor_1: float
    sensor_2: float
    sensor_3: float
    operating_temp: float


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train.py first.")
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(s: Sensors):
    model = load_model()
    X = [[s.sensor_1, s.sensor_2, s.sensor_3, s.operating_temp]]
    try:
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"failure_probability": float(prob)}


@app.post("/predict_batch")
def predict_batch(items: List[Sensors]):
    model = load_model()
    X = [[i.sensor_1, i.sensor_2, i.sensor_3, i.operating_temp]
         for i in items]
    try:
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return probs
