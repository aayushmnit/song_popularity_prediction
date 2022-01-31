import pandas as pd
import config
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.song_popularity)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold

    df.to_csv("../data/train_folds.csv", index=False)