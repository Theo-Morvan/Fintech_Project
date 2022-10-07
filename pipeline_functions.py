from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
from xgboost import XGBClassifier
import ipdb

def full_pipeline():
    cols_one_hot_encoding = ["employment"]
    cols_scaling = ["income"]
    path_through_columns = ["digital1","digital2","digital3"]
    preprocessor = ColumnTransformer([
        ("employment",OneHotEncoder(drop="first"),cols_one_hot_encoding),
        ("income",StandardScaler(),cols_scaling),
        ("digital_columns", "passthrough",path_through_columns)
    ])
    pipeline = Pipeline([
        ("columns_preprocessing",preprocessor)
    ]
    )
    return pipeline

def create_optuna_pipeline(X_train,y_train, X_val, y_val):

    def optuna_objective(trial):
        params = {
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
                "min_child_weight" : trial.suggest_int("min_child_weight", 2, 10),
                "n_estimators": trial.suggest_int("n_estimators",50,150, step=25)
            }
        # minimum child weight, larger the term more conservative the tree.

        sample_weights = compute_sample_weight(class_weight="balanced",y = y_train)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        ipdb.set_trace()
        final_score = recall_score(y_val, preds)
        return final_score

    return optuna_objective