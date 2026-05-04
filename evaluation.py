import joblib
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from train import training

from config import MODEL_WEIGHT,THRESHOLD

X_test, y_test,cv_score,model_l, model_x = training()

lgbmc_prob = model_l.predict_proba(X_test)[:,1]
xgbc_prob = model_x.predict_proba(X_test)[:,1]

final_prob = (lgbmc_prob * MODEL_WEIGHT ) + (xgbc_prob * (1- MODEL_WEIGHT))

# y_pred_l = (lgbmc_prob>prob_weight).astype(int)
y_pred = (final_prob>THRESHOLD).astype(int)


# y_test,y_pred,cv_score,model_l, model_x  = training()

print("Accuracy",accuracy_score(y_test, y_pred))
print("Recall",recall_score(y_test, y_pred))
print("CV Score",cv_score)
print("F1",f1_score(y_test, y_pred))
print("Con mat",confusion_matrix(y_test, y_pred))
print("Class rep",classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred_x))


"""

Accuracy 0.8314140948871488
Recall 0.84375
CV Score [0.87715931 0.86986564 0.8696737  0.86888078 0.87156844]
F1 0.7067307692307694
Con mat [[4092  853]
 [ 245 1323]]
Class rep               precision    recall  f1-score   support

           0       0.94      0.83      0.88      4945
           1       0.61      0.84      0.71      1568

    accuracy                           0.83      6513
   macro avg       0.78      0.84      0.79      6513
weighted avg       0.86      0.83      0.84      6513

"""

joblib.dump(model_x, "model_xgb.pkl")
joblib.dump(model_l,"model_lgbmc.pkl")


"""

grid search weights
or optimize via CV
custom order : in ordinal encoding 

"""