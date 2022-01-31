import os
import config
import joblib
import argparse
import pandas as pd
import optuna
import logging
logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler("../logs/hyperopt_lgb.txt", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

from lightgbm import LGBMClassifier
from metrics import ClassificationMetrics

def runfold(fold, model):
    ## read the training data with fold
    df = pd.read_csv(config.TRAINING_PROCESSED_FILE)

    ## Training data is where kfold is not equal to the fold
    df_train = df.loc[df["kfold"] != fold,:].reset_index(drop=True)

    ## Validation data is where kfold is equal to the fold
    df_valid = df.loc[df["kfold"] == fold,:].reset_index(drop=True)

    ## Dropping unnecessary columns
    x_train = df_train.drop(["id", "kfold", "song_popularity"], axis=1).values
    y_train = df_train.song_popularity.values

    ## Dropping unnecessary columns
    x_valid = df_valid.drop(["id", "kfold", "song_popularity"], axis=1).values
    y_valid = df_valid.song_popularity.values

    ## Defining the model
    clf = model
    clf.fit(
        x_train, 
        y_train,
        early_stopping_rounds=433,
        eval_set=[(x_valid, y_valid)],
        verbose=False
        )

    # create predictions for validation samples
    preds = clf.predict_proba(x_valid)

    #calculate the metrics
    accuracy = ClassificationMetrics._auc(y_true=y_valid, y_pred=preds[:,1])
    return accuracy

def runAll(model):
    acc = []
    for fold in range(config.N_FOLDS):
        acc.append(runfold(fold=fold, model=model))
    avg_auc = sum(acc)/len(acc)
    return avg_auc


def objective(trial):
    ## Defining the model
    model = LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        learning_rate=0.01,
        max_depth=trial.suggest_int("max_depth", 1, 7),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.9),
        subsample=trial.suggest_float("subsample", 0.1, 0.9),
        min_child_samples=trial.suggest_int("min_child_samples", 30, 500),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        max_bin=trial.suggest_int("max_bin", 200, 1000),
        min_data_per_group=trial.suggest_int("min_data_per_group", 30, 500),
        subsample_freq=1,
        cat_smooth=96,
        cat_l2=17,
        verbose=-1,
        random_state=42,
        n_estimators=20000
    )
    
    return runAll(model)


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize"
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("Finished.")