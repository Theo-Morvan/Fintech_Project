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
from pipeline_functions import *
from interest_rates import *
columns_test = ["competing1","competing2","rate"]



def compute_financial_results(
    df_past_results, 
    df_new_rates,
    columns_test = ["competing1","competing2","rate"]
):
    df_back_test = df_past_results.merge(df_new_rates,on='id',how="inner")
    df_back_test = df_back_test[~df_back_test["rate"].isnull()]
    df_back_test["winner"] = df_back_test[columns_test].min(axis=1)
    df_clients_won = df_back_test[df_back_test["winner"]==df_back_test["rate"]]
    
    return df_clients_won["profit"].sum()

def optuna_optimal_mix_creation(df_previous, **kwargs):

    def optuna_trial_opti_backtesting(trial):
        keys_kwargs = list(kwargs.keys())
        preds_names = [element for element in keys_kwargs if "preds" in element]
        predictions = {preds_name : kwargs[preds_name] for preds_name in preds_names}
        params_optuna = {
            f"weight_rate_{model}":trial.suggest_float(f"weight_rate_{model}",0,1) for model in predictions.keys()
        }
        params_optuna["margin"] = trial.suggest_float("margin",0,0.2)
        rates = {}

        for model, prediction in predictions.items():

            df_rates = compute_interest_rates(prediction,params_optuna["margin"],df_previous.index)
            rates[model] = df_rates
        len(model)
        params_optuna = {
            f"weight_rate_{model}":trial.suggest_float(f"weight_rate_{model}",0,1) for model in rates.keys()
        }
        sum_weight = 0
        columns_new_df = ["id"]
        df_total = pd.DataFrame(df_previous["id"])
        columns_models = []
        for model in rates.keys():
            rates[model]["weight"] = params_optuna[f"weight_rate_{model}"]
            rates[model]["weighted_rate"] = rates[model]["weight"]*rates[model]["rate"]
            df_total[f"weighted_rate_{model}"]=rates[model]["weighted_rate"]
            sum_weight+= params_optuna[f"weight_rate_{model}"]
            columns_models.append(f"weighted_rate_{model}")
        df_total["rate"] = df_total[columns_models].sum(axis=1)/sum_weight
        df_total.loc[df_total["rate"]>1.] = np.nan
        profit = compute_financial_results(df_previous,df_total[["id","rate"]])
        return profit
    
    return optuna_trial_opti_backtesting
        
            
