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

dirname = os.path.dirname
root = os.getcwd()
data_path = os.path.join(dirname(root),"Fintech_Project", 'data')
path = os.path.join(data_path,"PastLoans.csv")
path_new_set = os.path.join(data_path,"NewApplications_3_Round1.csv")
path_proba = os.path.join(root + '/data', 'default_predictions_backtesting.csv') #the dataset with the probabilities of default
columns_test = ["competing1","competing2","rate"]


if __name__ == "__main__":
    probas = pd.read_csv(path_proba)
    df_new_preds = pd.read_csv(path_new_set,index_col="id")
    index_ids = df_new_preds.index
    df_preds = compute_interest_rates_reject_higher_1(probas, 0.02,index_ids)
    df_past_results = pd.read_csv(os.path.join(data_path,"profit_31.csv"))
    df_back_test = df_past_results.merge(df_preds,on='id',how="inner")
    df_back_test["winner"] = df_back_test[columns_test].min(axis=1)
    df_clients_won = df_back_test[df_back_test["winner"]==df_back_test["rate"]]

    print(f"profits made with this strategy during backtesting : {df_clients_won['profit'].sum()}")

    ipdb.set_trace()
    