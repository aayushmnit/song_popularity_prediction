import os
import config
import joblib
import argparse
import pandas as pd
from metrics import ClassificationMetrics
import model_dispatcher

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
    clf = model_dispatcher.models[model]
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
    print(f"Fold={fold}, Classifier AUC: {accuracy}")

    ## save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.bin"))
    return accuracy

def runAll(model):
    acc = []
    for fold in range(config.N_FOLDS):
        acc.append(runfold(fold=fold, model=model))
    print(f"Mean AUC: {sum(acc)/len(acc)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    runAll(model=args.model)

    
        
    