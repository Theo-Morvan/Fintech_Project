from pyclbr import Function
from sklearn.linear_model import LogisticRegression
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
from lightgbm import LGBMClassifier
import ipdb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor

def _log_transform_income(X):
    X = X.copy()
    values = X["income"].values + 1 #on évite les valeurs 0
    log_values = np.log(values)
    X["income"] = log_values
    return X

def full_pipeline():

    log_transformer = FunctionTransformer(_log_transform_income, validate=False)

    cols_one_hot_encoding = ["employment"]
    cols_scaling = ["income"]
    path_through_columns = ["digital3"]
    preprocessor = ColumnTransformer([
        ("employment",OneHotEncoder(drop="first"),cols_one_hot_encoding),
        ("income",StandardScaler(),cols_scaling),
        ("digital_columns", "passthrough",path_through_columns)
    ])
    pipeline = Pipeline([
        ("income_transformer", log_transformer),
        ("columns_preprocessing",preprocessor),
    ]
    )
    return pipeline

def imbalance_correction(y):

    pass

def create_optuna_pipeline_xgboost(X_train,y_train, X_val, y_val):

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
        # ipdb.set_trace()
        final_score = f1_score(y_val, preds)
        return final_score

    return optuna_objective

def create_optuna_pipeline_lightgbm(X_train,y_train, X_val, y_val):

    def optuna_objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            # "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
        }
        sample_weights = compute_sample_weight(class_weight="balanced",y = y_train)
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        # ipdb.set_trace()
        final_score = f1_score(y_val, preds)
        return final_score

    return optuna_objective

def optimal_mix_predictions(preds_1,preds_2,**kwargs):
    if "weight" in list(kwargs.keys()):
        weight = kwargs["weight"]
        value = weight*preds_1 + (1-weight)*preds_2
    else:
        weight = 1/2
        value = weight*preds_1 + (1-weight)*preds_2
    
    if "cutting_threshold" in list(kwargs.keys()):
        final_class = (value>=kwargs["cutting_threshold"])*1
    else:
        final_class = (value>=0.5)*1
    return final_class

def optimal_mix_probas(preds_1,preds_2,**kwargs):
    if "weight" in list(kwargs.keys()):
        weight = kwargs["weight"]
        value = weight*preds_1 + (1-weight)*preds_2
    else:
        weight = 1/2
        value = weight*preds_1 + (1-weight)*preds_2
    return value

def create_complete_pipeline(X_train,y_train, X_val, y_val):
    def optuna_objective(trial):
        params_lgbm = {
        "objective": "binary",
        "metric": "binary_logloss",
        # "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
        }
        params_xgb =  {
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
        sample_weights = compute_sample_weight(class_weight="balanced",y = y_train)
        model_lgbm = LGBMClassifier(**params_lgbm)
        model_xgb = XGBClassifier(**params_xgb)
        estimators = [
            ("lgbm", model_lgbm),
            ("xgboost", model_xgb)
        ]
        model_lgbm.fit(X_train, y_train, sample_weight=sample_weights)
        model_xgb.fit(X_train, y_train, sample_weight=sample_weights)
        preds_lgbm = model_lgbm.predict_proba(X_val)[:,1]
        preds_xgb = model_xgb.predict_proba(X_val)[:,1]
        # ipdb.set_trace()
        params_cuts = {
            "weight": trial.suggest_float("weight",0,1),
            # "cutting_threshold":trial.suggest_float("cutting_threshold",0.,1)
        }
        preds = optimal_mix_predictions(preds_lgbm,preds_xgb,**params_cuts)
        # ipdb.set_trace()
        final_score = f1_score(y_val, preds)

        # final_model = StackingRegressor(estimators=estimators, final_estimator=HistGradientBoostingRegressor())
        return final_score
    
    return optuna_objective

def create_logistic_regression_pipeline(preds_1,preds_2,y_true):

    def optuna_objective(trial):
        params = {
            "l1_ratio":trial.suggest_float("l1_ratio",0,1),
            "penalty":"elasticnet",
            "solver":"saga",
            "class_weight":"balanced",
        }
        model = LogisticRegression(**params)
        X = np.concatenate((preds_1.reshape(-1,1),preds_2.reshape(-1,1)),axis=1)
        model.fit(X, y_true)

        preds = model.predict(X)
        score = f1_score(y_true,preds)
        return score
    return optuna_objective

liste_lgbm = ["lambda_l1",
    "lambda_l2",
    "num_leaves",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "min_child_samples",
    'learning_rate'
]
liste_xgb = [
    "lambda",
    "alpha",
    "subsample",
    "colsample_bytree",
    "max_depth",
    "min_child_weight",
    "n_estimators"
]
liste_weights = ["weight","cutting_threshold"]