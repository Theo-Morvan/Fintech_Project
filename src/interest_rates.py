import numpy as np
import pandas as pd
import os
dirname = os.path.dirname
root = os.getcwd()

data_path = os.path.join(dirname(root), "data")
path = os.path.join(data_path,"PastLoans.csv")
path_new_set = os.path.join(data_path, "NewApplications_3_Round1.csv")
path_proba = os.path.join(root + '/data', 'default_predictions.csv') #the dataset with the probabilities of default
columns_test = ["competing1","competing2","rate"]

def compute_interest_rates(default_probas, a, b, index):
    df_preds = pd.DataFrame(default_probas, columns=["Proba no Default","Proba Default"])
    df_preds["id"] = index

    df_preds["break_even_rate"] = df_preds["Proba Default"]/(1-df_preds["Proba Default"])
    df_preds["rate"] = df_preds.break_even_rate*a + b
    df_preds.loc[df_preds["rate"] > 1, "rate"] = np.nan

    return df_preds

def compute_interest_rates_reject_higher_1(default_probas, M,index):

    df_preds = pd.DataFrame(default_probas, columns=["Proba no Default","Proba Default"])
    # ipdb.set_trace()
    df_preds["id"] = index

    df_preds["break_even_rate"] = df_preds["Proba Default"]/(1-df_preds["Proba Default"])
    df_preds["rate"] = df_preds.break_even_rate + M
    df_preds.loc[df_preds["rate"]>1.] = np.nan
    return df_preds