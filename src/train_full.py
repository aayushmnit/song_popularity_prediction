import os
import config
import joblib
import argparse
import pandas as pd

from metrics import ClassificationMetrics
import model_dispatcher
def run(model):
    ## read the training data with fold
    df = pd.read_csv(config.TRAINING_PROCESSED_FILE)

    ## Dropping unnecessary columns
    x_train = df.drop(["id", "kfold", "song_popularity"], axis=1).values
    y_train = df.song_popularity.values

    ## Defining the model
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_full.bin"))


    testdf = pd.read_csv(config.TESTING_PROCESSED_FILE)
    
    ## Dropping unnecessary columns
    x_sub = testdf.drop(["id", "kfold"], axis=1).values

    preds = clf.predict_proba(x_sub)[:,1]
    testdf["song_popularity"] = preds
    testdf.id = testdf.id.astype(int)
    testdf.loc[:,['id', 'song_popularity']].to_csv(f"{config.SUBMIT_OUTPUT}{model}_full.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    run(model = args.model)
        
    