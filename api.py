from fastapi import FastAPI, status
from pydantic import BaseModel
import uvicorn
import joblib

model_l = joblib.load('model_lgbmc.pkl')
model_x = joblib.load('model_xgb.pkl')

app = FastAPI(
    title="Census to cloud",
    summary="it is based about the salary census",
    version="1.0"
)

@app.get('/health')
def health():
    return "200 OK"

@app.post('/predict')
def predict():
    pass