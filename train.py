import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from data import load_data
from config import DATA_PATH
from custom_pipeline import build_pipelines

def training():
    X,y = load_data(DATA_PATH)

    model_l, model_x = build_pipelines(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_score = cross_val_score(
    model_l,
    X_train,
    y_train,
    cv=skf,
    scoring='accuracy',
    error_score='raise'
    )


    model_l.fit(X_train, y_train) 

    model_x.fit(X_train, y_train )

    feat_names = model_l.named_steps['columntransformer'].get_feature_names_out()
    feat_imp = model_l.named_steps['lgbmclassifier'].feature_importances_

    # print(pd.Series(feat_imp, index=feat_names).sort_values(ascending=True))

    """
    remainder__is_white                0
    remainder__is_husband              0
    remainder__age_cat                 0
    remainder__is_hsgrd                7
    remainder__is_youngmale           11
    remainder__is_hsgrdusa            12
    remainder__native.country         25
    pipeline__sex                     44
    remainder__cap_diff               64
    pipeline__race                    78
    pipeline__education              117
    pipeline__relationship           141
    pipeline__marital.status         147
    pipeline__workclass              170
    simpleimputer__capital.loss      275
    simpleimputer__education.num     285
    pipeline__occupation             315
    simpleimputer__hours.per.week    332
    simpleimputer__capital.gain      361
    simpleimputer__age               616

    """



    return X_test, y_test,cv_score,model_l, model_x 
