import numpy as np
import pandas as pd
import pickle


def give_data(feature_sel=0):
    """
    Return in order: X_train, X_test, y_train, y_test
    """
    
    common_path = "/home/ugurkap/AppliedNotebooks/santander-customer-transaction-prediction/"
    
    y_train = pd.read_csv(common_path + "smote_train_labels.csv")
    X_test = pd.read_csv(common_path + "X_val.csv", index_col="ID_code")
    y_test = pd.read_csv(common_path + "y_val.csv", index_col="ID_code")

    if feature_sel == 1:
        # KBest
        with open(common_path + "kbest_column_names", "rb") as f:
            columns = pickle.load(f)

        X_train = pd.read_csv(common_path + "X_kbest.csv")
        X_test = X_test[list(columns)]
    elif feature_sel == 2:
        # Selector
        with open(common_path + "selector_column_names", "rb") as f:
            columns = pickle.load(f)
        
        X_train = pd.read_csv(common_path + "X_selector.csv")
        X_test = X_test[list(columns)]
    else:
        X_train = pd.read_csv(common_path + "smote_train.csv")

    return X_train, X_test, y_train, y_test

