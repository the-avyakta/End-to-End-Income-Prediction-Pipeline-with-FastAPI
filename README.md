# End-to-End-Income-Prediction-Pipeline-with-FastAPI

- End-to-end machine learning pipeline for census-based income prediction with data processing, model training, and FastAPI deployment.

## Tech Stack
- numpy
- pandas
- scikit-learn
- uvicorn
- joblib
- xgboost
- lightgbm
- fastapi
- pydantic
- joblib


## Model Evaluation
- Accuracy 0.8314140948871488
- Recall 0.84375

```
Class rep               precision    recall  f1-score   support

                    0       0.94      0.83      0.88      4945
                    1       0.61      0.84      0.71      1568

    accuracy                           0.83      6513
   macro_avg       0.78      0.84      0.79      6513
weighted_avg       0.86      0.83      0.84      6513
```

## Run Locally
- 1st Step : pip install -r requirements.txt
- 2nd Step : python train.py

## Docker 
- docker build -t census-api .
- docker run -p 8000:8000 census-api


--
# Spread Love ❤
