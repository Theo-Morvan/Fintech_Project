import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import xgboost
from pipeline_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import ipdb
from interest_rates import *
from backtesting_opti import *

dirname = os.path.dirname
root = os.getcwd()
data_path = os.path.join(dirname(root),"Fintech_Project", 'data')
path = os.path.join(data_path,"PastLoans.csv")
path_new_set = os.path.join(data_path,"NewApplications_3_Round1.csv")
path_proba = os.path.join(root + '/data', 'default_predictions_backtesting_mix_models.csv') #the dataset with the probabilities of default
columns_test = ["competing1","competing2","rate"]

if __name__=="__main__":
    df_new_preds = pd.read_csv(path_new_set,index_col="id")
    df_past_results = pd.read_csv(os.path.join(data_path,"profit_31.csv"))
    probas = pd.read_csv(path_proba)
    probas_xgb = pd.read_csv(os.path.join(root + '/data',"preds_xgb.csv"))
    preds_lgbm = pd.read_csv(os.path.join(root + '/data',"preds_lgbm.csv"))
    preds_logistic = pd.read_csv(os.path.join(root + '/data',"preds_logistic.csv"))
    predictions_mix = {"preds_xgb":probas_xgb,"preds_lgbm":preds_lgbm,"preds_logistic":preds_logistic}
    optuna_pipeline = optuna_optimal_mix_creation(df_past_results,**predictions_mix)
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_pipeline, n_trials=20, n_jobs=-1)
    trial = study.best_trial
    print(trial.params)
    ipdb.set_trace()