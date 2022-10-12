from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,FunctionTransformer
from sklearn.model_selection import KFold
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def _log_transform_income(X):
    X = X.copy()
    values = X["income"].values + 1 #on évite les valeurs 0
    log_values = np.log(values)
    X["income"] = log_values
    return X

def income_categories(x):
    category = 0
    if x<=0.:
        category = 0
    elif (0<x) & (x<=17500):
        category = 1
    elif (17500<x) & (x<=24000):
        category = 2
    elif (24000<x) & (x<=37000):
        category = 3
    else:
        category = 4
    return category

def _adding_category_column(X):
    X = X.copy()
    X["category_income"] = X["income"].apply(income_categories)
    return X

def full_pipeline():

    log_transformer = FunctionTransformer(_log_transform_income, validate=False)
    category_creator = FunctionTransformer(_adding_category_column, validate=False)
    cols_one_hot_encoding = ["employment","category_income"]
    cols_scaling = ["income"]
    path_through_columns = ["digital3"]
    preprocessor = ColumnTransformer([
        ("employment",OneHotEncoder(drop="first"),cols_one_hot_encoding),
        ("income",StandardScaler(),cols_scaling),
        ("digital_columns", "passthrough",path_through_columns)
    ])
    pipeline = Pipeline([
        ("income_transformer", log_transformer),
        ("category_income",category_creator),
        ("columns_preprocessing",preprocessor),
    ]
    )
    return pipeline

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

def create_optuna_objective_xgboost_cv(X,y, number_cv):

    def optuna_objective(trial):
        scores = np.zeros(number_cv)
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
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

        sample_weights = compute_sample_weight(class_weight="balanced",y = y)
        
        for num, (train_id, valid_id) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_id], X[valid_id]
            y_train, y_valid = y[train_id], y[valid_id]
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            preds = model.predict(X_valid)
        # ipdb.set_trace()
            score_cv= f1_score(y_valid, preds)
            scores[num] = score_cv
        return np.mean(scores)

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
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
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

def optimal_mix_probas(preds_1,preds_2,preds_3,return_class=False,**kwargs):
    weights = [element for element in kwargs.keys() if "weight" in element].sort()
    
    try:
        len(weights) == 3
    except:
        weights = ["weight_1","weight_2","weight_3"]

    weights_values = [kwargs[element] for element in weights]
    if len(weights)==3:
        print(kwargs)
        value = kwargs[weights[0]]*preds_1+kwargs[weights[1]]*preds_2+kwargs[weights[2]]*preds_3
        value = value/sum(weights_values)
    else:
        weight = 1/3
        value = weight*preds_1 + weight*preds_2 + weight*preds_3
    if return_class:
        value = (value>=0.5)*1
    return value

def create_complete_pipeline(X, y, number_cv):
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
            "max_depth": trial.suggest_int("max_de  pth", 3, 9, step=2),
            "min_child_weight" : trial.suggest_int("min_child_weight", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators",50,150, step=25)
        }

        params_cuts = {
            "weight_1": trial.suggest_float("weight_1",0,1),
            "weight_2": trial.suggest_float("weight_2",0,1),
            "weight_3": trial.suggest_float("weight_3",0,1),
            # "cutting_threshold":trial.suggest_float("cutting_threshold",0.,1)
        }

        

        scores = np.zeros(number_cv)
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        for num, (train_id, valid_id) in enumerate(kf.split(X)):
            X_train, X_valid = np.take(X,train_id,axis=0), np.take(X,valid_id,axis=0)
            y_train, y_valid = np.take(y,train_id,axis=0), np.take(y, valid_id, axis=0)
            model_lgbm = LGBMClassifier(**params_lgbm)
            model_xgb = XGBClassifier(**params_xgb)
            logistic_model = LogisticRegression()
            sample_weights = compute_sample_weight(class_weight="balanced",y = y_train)
            model_lgbm.fit(X_train, y_train, sample_weight=sample_weights)
            model_xgb.fit(X_train, y_train, sample_weight=sample_weights)
            logistic_model.fit(X_train, y_train)
            preds_lgbm = model_lgbm.predict_proba(X_valid)[:,1]
            preds_xgb = model_xgb.predict_proba(X_valid)[:,1]
            preds_logistique = logistic_model.predict_proba(X_valid)[:,1]
        # ipdb.set_trace()

            preds = optimal_mix_probas(preds_lgbm,preds_xgb,preds_logistique,True,**params_cuts)
        # ipdb.set_trace()
            score_recall_cv = f1_score(y_valid, preds)
            scores[num] = score_recall_cv
        # final_model = StackingRegressor(estimators=estimators, final_estimator=HistGradientBoostingRegressor())
        return np.mean(scores)
    
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
liste_weights = ["weight_1","weight_2","weight_3","cutting_threshold"]