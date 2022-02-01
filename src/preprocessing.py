import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
import config

def preprocess():
    train_df = pd.read_csv(config.TRAINING_FILE)
    test_df = pd.read_csv(config.TESTING_FILE)
    test_df['kfold'] = -1

    ## Changing milli seconds to minutes
    train_df['song_duration_ms'] = train_df.song_duration_ms / 1000 / 60
    test_df['song_duration_ms'] = test_df.song_duration_ms / 1000 / 60

    valcol = [
                'song_duration_ms', 'acousticness', 'danceability', 'energy',
                'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode',
                'speechiness', 'tempo', 'time_signature', 'audio_valence', 'kfold'
            ]

    cat = ['audio_mode', 'time_signature']
    cont = ['song_duration_ms', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'audio_valence']
    
    col_train = train_df.columns
    col_test = test_df.columns
    
    ## Initiating the imputer
    imputer = IterativeImputer(random_state=0, max_iter=10, initial_strategy='mean')
    train_df = pd.DataFrame(imputer.fit_transform(train_df))
    train_df.columns = col_train
    test_df = pd.DataFrame(imputer.fit_transform(test_df))
    test_df.columns = col_test

    ## Robust Scaler 
    ## https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
    combined_df = pd.concat([train_df.loc[:,cont], test_df.loc[:,cont]], ignore_index=True)

    rl = RobustScaler() 
    combined_df = rl.fit_transform(combined_df)
        
    ## Adding the processed data to original dataframe
    train_df.loc[:,cont] = combined_df[:train_df.shape[0],:]
    test_df.loc[:,cont] = combined_df[train_df.shape[0]:,:] 

    ## One-Hot Encoding
    train_df = pd.concat([train_df, pd.get_dummies(train_df.time_signature)],axis=1).drop(columns=['time_signature'])
    test_df = pd.concat([test_df, pd.get_dummies(test_df.time_signature)],axis=1).drop(columns=['time_signature'])
    
    ## Export
    train_df.to_csv(config.TRAINING_PROCESSED_FILE, index=False)
    test_df.to_csv(config.TESTING_PROCESSED_FILE, index=False)

if __name__ == "__main__":
    preprocess()
    
    
