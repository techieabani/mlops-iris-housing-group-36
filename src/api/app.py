from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, os, joblib, mlflow
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title='Iris + Housing ML API')

REQUEST_COUNTER = Counter('request_count', 'Total prediction requests')

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MODEL_LOCAL_PATH = os.getenv('MODEL_LOCAL_PATH', 'models/model.joblib')

mlflow.set_tracking_uri(MLFLOW_URI)

model = None
# try local joblib first
if os.path.exists(MODEL_LOCAL_PATH):
    try:
        model = joblib.load(MODEL_LOCAL_PATH)
    except Exception as e:
        print('Failed loading local model:', e)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post('/predict/iris')
def predict_iris(inp: IrisInput):
    REQUEST_COUNTER.inc()
    if model is None:
        raise HTTPException(status_code=503, detail='Model is not available')
    df = pd.DataFrame([inp.dict()])
    pred = model.predict(df)[0]
    return {'prediction': str(pred)}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get('/health')
def health():
    return {'status':'ok'}