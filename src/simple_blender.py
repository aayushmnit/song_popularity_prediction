
import config
import pandas as pd

def blend():
    lgbOp = pd.read_csv(f"{config.SUBMIT_OUTPUT}lightgbm_hyperoptimize.csv")
    xgbOp = pd.read_csv(f"{config.SUBMIT_OUTPUT}xgb_hyperoptimize.csv")

    lgbOp['song_popularity'] = lgbOp.song_popularity.rank(ascending=True) / lgbOp.shape[0]
    xgbOp['song_popularity'] = xgbOp.song_popularity.rank(ascending=True) / xgbOp.shape[0]

    lgbOp["song_popularity"] = (lgbOp["song_popularity"] + xgbOp["song_popularity"]) / 2

    lgbOp.to_csv(f"{config.SUBMIT_OUTPUT}blend_hyperoptimize.csv", index=False)

if __name__ == "__main__":
    blend()
    