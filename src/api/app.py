from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import mlflow.pyfunc
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Iris + Housing ML API")

REQUEST_COUNTER = Counter("request_count", "Total prediction requests")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Names of registered models in MLflow Model Registry
IRIS_MODEL_NAME = os.getenv("IRIS_MODEL_NAME", "IrisClassifier")
IRIS_MODEL_STAGE = os.getenv("IRIS_MODEL_STAGE", "Production")

HOUSING_MODEL_NAME = os.getenv("HOUSING_MODEL_NAME", "HousingRegressor")
HOUSING_MODEL_STAGE = os.getenv("HOUSING_MODEL_STAGE", "Production")

def load_model(name, stage):
    try:
        return mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    except Exception as e:
        print(f"Failed loading model {name} at stage {stage}: {e}")
        return None

iris_model = load_model(IRIS_MODEL_NAME, IRIS_MODEL_STAGE)
housing_model = load_model(HOUSING_MODEL_NAME, HOUSING_MODEL_STAGE)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict/iris")
def predict_iris(data: IrisInput):
    REQUEST_COUNTER.inc()
    if iris_model is None:
        raise HTTPException(status_code=503, detail="Iris model not loaded")
    df = pd.DataFrame([data.dict()])
    pred = iris_model.predict(df)[0]
    return {"prediction": str(pred)}

@app.post("/predict/housing")
def predict_housing(data: HousingInput):
    REQUEST_COUNTER.inc()
    if housing_model is None:
        raise HTTPException(status_code=503, detail="Housing model not loaded")
    df = pd.DataFrame([data.dict()])
    pred = housing_model.predict(df)[0]
    return {"prediction": float(pred)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}
