import pandas as pd


def give_data(feature_sel=0):
    """
    Return in order: x_train, X_test, y_train, y_test
    """

    common_path = "/Users/<your_username>/Repositories/Applied-AI-Study-Group/Applied AI Study Group #5 - August 2021/Week 4/notebooks/santander-customer-transaction-prediction"

    x_train = pd.read_csv(common_path + "smote_train.csv")
    y_train = pd.read_csv(common_path + "smote_train_labels.csv")
    X_test = pd.read_csv(common_path + "X_val.csv", index_col="ID_code")
    y_test = pd.read_csv(common_path + "y_val.csv", index_col="ID_code")

    return x_train, X_test, y_train, y_test
