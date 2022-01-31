import joblib
import config
import argparse
import pandas as pd

def infer(model):
    
    df = pd.read_csv(config.TESTING_PROCESSED_FILE)
    
    ## Dropping unnecessary columns
    x_sub = df.drop(["id", "kfold"], axis=1).values

    for fold in range(5):
        clf = joblib.load(f"{config.MODEL_OUTPUT}/{model}_{fold}.bin")
        # create predictions for validation samples
        preds = clf.predict_proba(x_sub)[:,1]
        df["kfold_{}".format(fold)] = preds
        print(f"Predicting using model {fold}.")
    
    df['id'] = df['id'].astype(int)
    df['song_popularity'] = df.iloc[:,-5:].mean(axis=1)
    df.loc[:,['id', 'song_popularity']].to_csv(f"{config.SUBMIT_OUTPUT}{model}.csv", index=False)
    print("Done.")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    infer(model=args.model)