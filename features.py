import pandas as pd

def feature_engineering(X_feat):
    X_feat = X_feat.copy()
    X_feat['age_cat'] = pd.cut(X_feat['age'],bins=[0,15,35,55,100] ,labels=[0,1,2,3] ) #astype(int) has error - can't conver float nan to int - it had some nan i fixed but but accuracy droppped

    X_feat['native.country'] = X_feat['native.country'].str.strip().map({'United-States': 1}).fillna(0)
    # X_feat['is_pvt'] = (X_feat['workclass'].str.strip()=='Private').astype(int)  # """ it is reducing """
   
    X_feat['is_hsgrd'] = (X_feat['education'].str.strip()=='HS-grad').astype(int)
    X_feat['is_hsgrdusa'] = ((X_feat['education'].str.strip()=='HS-grad')& (X_feat['marital.status']=='Married-civ-spouse')).astype(int)
    X_feat['is_white'] = (X_feat['race'].str.strip()=='White').astype(int)
    X_feat['cap_diff'] = X_feat['capital.gain'] + X_feat['capital.loss'] 
    X_feat['is_youngmale'] = ((X_feat['age']<=50)&(X_feat['sex'].str.strip()=='Male')).astype(int)

    return X_feat