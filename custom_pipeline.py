from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import feature_engineering


def build_pipelines(X):
    X_fe = FunctionTransformer(feature_engineering, validate=False)
    obj_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'education']
    num_cols = X.select_dtypes(['int64']).columns.to_list()
    ord_en = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
    imputer = SimpleImputer(strategy='mean')
    imputer_cleaner = SimpleImputer(missing_values='?',strategy='most_frequent')

    data_clean_pipeline = make_pipeline(
        (imputer_cleaner),
        (ord_en)
    )

    preprocessing = make_column_transformer(
        (data_clean_pipeline, obj_cols),
        (imputer, num_cols),    
        remainder='passthrough'
    )

    lgbmc = LGBMClassifier(
        random_state=42,
        verbose=-1
        )

    model_l = make_pipeline(
        X_fe,
        preprocessing,
        lgbmc
        
    )
    xgbc = XGBClassifier()

    model_x = make_pipeline(
        X_fe,
        preprocessing,
        xgbc
        
    )

    return model_l, model_x