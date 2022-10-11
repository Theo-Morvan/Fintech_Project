import numpy as np
import pandas as pd

def compute_interest_rates(default_probas, M,index):

    df_preds = pd.DataFrame(default_probas, columns=["Proba no Default","Proba Default"])
    # ipdb.set_trace()
    df_preds["id"] = index

    df_preds["break_even_rate"] = df_preds["Proba Default"]/(1-df_preds["Proba Default"])
    df_preds["rate"] = df_preds.break_even_rate + M

    return df_preds

def compute_interest_rates_reject_higher_1(default_probas, M,index):

    df_preds = pd.DataFrame(default_probas, columns=["Proba no Default","Proba Default"])
    # ipdb.set_trace()
    df_preds["id"] = index

    df_preds["break_even_rate"] = df_preds["Proba Default"]/(1-df_preds["Proba Default"])
    df_preds["rate"] = df_preds.break_even_rate + M
    df_preds.loc[df_preds["rate"]>1.] = np.nan
    return df_preds