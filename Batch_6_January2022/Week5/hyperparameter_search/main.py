import numpy as np
import mlflow

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.ax import AxSearch

from helper_funcs import give_data

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

experiment_name = "LGBM + SMOTETomek + E50 + Ax"

fet_sel_dict = {0: "No Selection", 1: "KBest", 2: "Selector"}

#experiment_name = "Default"


@mlflow_mixin
def LightGBMCallback(env):
    """Assumes that `valid_0` is the target validation score."""
    _, metric, score, _ = env.evaluation_result_list[0]
    mlflow.log_metric(metric, score)


@mlflow_mixin
def objective(config):
    # Get and log parameters
    params = {
        "num_leaves": config["num_leaves"],
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "objective": config["objective"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "tree_learner": config["tree_learner"],
        "subsample": config["subsample"],
        "subsample_freq": config["subsample_freq"],
        "feature_sel": fet_sel_dict[config["feature_sel"]]
    }

    mlflow.log_params(params)

    model = LGBMClassifier(**params, random_state=0)

    X_train, X_test, y_train, y_test = give_data(
        feature_sel=config["feature_sel"])

    model.fit(X_train,
              np.ravel(y_train),
              eval_set=[(X_test, np.ravel(y_test))],
              verbose=False,
              early_stopping_rounds=50,
              callbacks=[LightGBMCallback])

    eval_results = classification_report(np.ravel(y_test),
                                         model.predict(X_test),
                                         output_dict=True)
    eval_results["accuracy"] = accuracy_score(y_test, model.predict(X_test))
    eval_results["auroc"] = roc_auc_score(y_test,
                                          model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("val_auroc", eval_results["auroc"])

    fold_accuracy = eval_results["accuracy"]
    mlflow.log_metric("val_accuracy", fold_accuracy)

    fold_f1 = eval_results["1"]["f1-score"]
    mlflow.log_metric("val_f1-score-1", fold_f1)
    mlflow.log_metric("val_f1-score-0", eval_results["0"]["f1-score"])

    fold_precision = eval_results["1"]["precision"]
    mlflow.log_metric("val_precision", fold_precision)

    fold_recall = eval_results["1"]["recall"]
    mlflow.log_metric("val_recall", fold_recall)

    eval_results_tr = classification_report(np.ravel(y_train),
                                            model.predict(X_train),
                                            output_dict=True)
    eval_results_tr["accuracy"] = accuracy_score(y_train,
                                                 model.predict(X_train))

    fold_accuracy_tr = eval_results_tr["accuracy"]
    mlflow.log_metric("tr_accuracy", fold_accuracy_tr)

    fold_f1_tr = eval_results_tr["1"]["f1-score"]
    mlflow.log_metric("tr_f1-score", fold_f1_tr)

    fold_precision_tr = eval_results_tr["1"]["precision"]
    mlflow.log_metric("tr_precision", fold_precision_tr)

    fold_recall_tr = eval_results_tr["1"]["recall"]
    mlflow.log_metric("tr_recall", fold_recall_tr)

    tune.report(auroc=eval_results["auroc"], done=True)


def tune_fn():
    mlflow.set_experiment(experiment_name=experiment_name)

    optuna_search = OptunaSearch(metric="auroc", mode="max")

    ax_search = AxSearch(metric="auroc", mode="max")

    tune.run(objective,
             name="mlflow_gbdt",
             num_samples=65,
             config={
                 "num_leaves": tune.randint(5, 95),
                 "learning_rate": tune.loguniform(1e-4, 1.0),
                 "n_estimators": tune.randint(100, 100000),
                 "subsample": tune.loguniform(0.01, 1.0),
                 "subsample_freq": tune.randint(1, 5),
                 "objective": "binary",
                 "reg_alpha": tune.loguniform(1e-4, 1.0),
                 "reg_lambda": tune.loguniform(1e-4, 1.0),
                 "tree_learner": "feature",
                 "feature_sel": 0,
                 "mlflow": {
                     "experiment_name": experiment_name,
                     "tracking_uri": mlflow.get_tracking_uri()
                 }
             },
             search_alg=optuna_search)


if __name__ == "__main__":
    tune_fn()

    # "n_components": tune.qrandint(20, 1000, 10),
    # "gamma": tune.uniform(0.1, 2.0),
