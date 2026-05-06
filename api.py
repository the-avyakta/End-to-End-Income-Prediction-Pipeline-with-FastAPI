from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import uvicorn
import joblib
import pandas as pd
from config import MODEL_WEIGHT, THRESHOLD


model_l = joblib.load('model_lgbmc.pkl')
model_x = joblib.load('model_xgb.pkl')

app = FastAPI(
    title="Census to cloud",
    summary="it is based about the salary census",
    version="1.0"
)

#validator 
class new_data(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    
    @field_validator('age')
    def age_check(cls, value):
        if (value>150):
            raise ValueError("Incorrect Age")
        return value

    @field_validator('hours_per_week')
    def check_hpw(cls, value):
        if (value>168):
            raise ValueError("Incorrect Hours per week")
        return value


@app.get('/health')
def health():
    return "200 OK"

@app.post('/predict')
def predict(data: new_data):
    try:
        data = pd.DataFrame([{

            "age": data.age,
            "workclass": data.workclass,
            "education": data.education,
            "education.num": data.education_num,
            "marital.status": data.marital_status,
            "occupation": data.occupation,
            "relationship": data.relationship,
            "race": data.race,
            "sex": data.sex,
            "capital.gain": data.capital_gain,
            "capital.loss": data.capital_loss,
            "hours.per.week": data.hours_per_week,
            "native.country": data.native_country

        }])

        prob_l = model_l.predict_proba(data)[:,1]
        prob_x = model_x.predict_proba(data)[:,1]
        final_prob = prob_l*MODEL_WEIGHT + prob_x*(1-MODEL_WEIGHT)
        y_pred = (final_prob>THRESHOLD).astype(int)
        pred = int(y_pred[0])
        


        return "less than 50K" if pred == 0 else "More than 50K"
    
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
        

    
     