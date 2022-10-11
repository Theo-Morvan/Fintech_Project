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
data_path = os.path.join(dirname(root), 'data')
path = os.path.join(data_path,"PastLoans.csv")
path_new_set = os.path.join(data_path,"NewApplications_3_Round1.csv")
path_output = os.path.join(root + '/data', 'default_predictions.csv')


path = os.path.join("data","PastLoans.csv")
path_new_set = os.path.join("data","NewApplications_3_Round1.csv")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(path, index_col='id')
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create pipeline
    pipeline = full_pipeline()
    X_train_final = pipeline.fit_transform(X_train)
    X_val_final = pipeline.transform(X_val)
    X_test_final = pipeline.transform(X_test)
    optuna_objective = create_complete_pipeline(X_train_final, y_train, X_val_final, y_val)
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=50, n_jobs=-1)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    params_xgb={}
    params_lgbm={} 
    params_weights = {}
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        if key in liste_lgbm:
            params_lgbm[key]=value
        if key in liste_xgb:
            params_xgb[key]=value
        else:
            params_weights[key]=value

    model_lgbm = LGBMClassifier(**params_lgbm)
    model_xgb = XGBClassifier(**params_xgb)

    sample_weights = compute_sample_weight(class_weight="balanced",y = y_train)
    model_xgb.fit(X_train_final,y_train, sample_weight=sample_weights)
    model_lgbm.fit(X_train_final,y_train, sample_weight=sample_weights)

    preds_lgbm = model_lgbm.predict_proba(X_train_final)[:,1]
    preds_xgb = model_xgb.predict_proba(X_train_final)[:,1]
    optuna_logistic = create_logistic_regression_pipeline(preds_lgbm,preds_xgb, y_train)
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_logistic, n_trials=5, n_jobs=-1)
    print(study.best_trial.params)
    params_logistic = study.best_trial.params
    params_logistic["penalty"]="elasticnet"
    params_logistic["solver"]="saga"
    params_logistic["class_weight"]="balanced"
    model_logistic = LogisticRegression(**study.best_trial.params)
    model_logistic.fit(
        np.concatenate((preds_lgbm.reshape(-1,1),preds_xgb.reshape(-1,1)),axis=1),
        y_train
    )
    model_qda = QuadraticDiscriminantAnalysis()
    model_qda.fit(
        np.concatenate((preds_lgbm.reshape(-1,1),preds_xgb.reshape(-1,1)),axis=1),
        y_train
    )
    preds_lgbm = model_lgbm.predict_proba(X_test_final)[:,1]
    preds_xgb = model_xgb.predict_proba(X_test_final)[:,1]

    
    preds_class = optimal_mix_predictions(preds_lgbm,preds_xgb,**params_weights)
    matrix_confusion = confusion_matrix(y_test, preds_class)

    print('Confusion matrix:')
    print(matrix_confusion)

    # Predict
    df_new_preds = pd.read_csv(path_new_set,index_col="id")
    X_new = df_new_preds.iloc[:,:]
    X_new_scaled = pipeline.transform(X_new)
    preds_lgbm = model_lgbm.predict_proba(X_new_scaled)
    preds_xgb = model_xgb.predict_proba(X_new_scaled)

    
    final_proba = model_logistic.predict_proba(
        np.concatenate((preds_lgbm[:,1].reshape(-1,1),preds_xgb[:,1].reshape(-1,1)),axis=1)
        )
    final_proba = model_qda.predict_proba(
        np.concatenate((preds_lgbm[:,1].reshape(-1,1),preds_xgb[:,1].reshape(-1,1)),axis=1)
        )
    
    predictions = optimal_mix_probas(preds_lgbm,preds_xgb,**params_weights)
    # ipdb.set_trace()
    df_preds = pd.DataFrame(final_proba, columns=["Proba no Default","Proba Default"])
    # ipdb.set_trace()
    df_preds["id"] = df_new_preds.index

    df_preds["break_even_rate"] = df_preds["Proba Default"]/(1-df_preds["Proba Default"])
    df_preds["rate"] = df_preds.break_even_rate + 0.02
    # ipdb.set_trace()
    fig, ax =plt.subplots(1,1,figsize=(12,10))
    sns.distplot(df_preds["rate"])
    plt.show()
    # ipdb.set_trace()
    df_preds[['id', 'rate']].to_csv(path_output, header=True, index=False)

    
