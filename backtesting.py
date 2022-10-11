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

dirname = os.path.dirname
root = os.getcwd()
data_path = os.path.join(dirname(root),"Fintech_Project", 'data')
path = os.path.join(data_path,"PastLoans.csv")
path_new_set = os.path.join(data_path,"NewApplications_3_Round1.csv")
path_proba = os.path.join(root + '/data', 'default_predictions.csv') #the dataset with the probabilities of default

if __name__ == "__main__":
    probas = pd.read_csv(path_proba)


    