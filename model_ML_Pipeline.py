
"""
track_id: The Spotify ID for the track
artists: The artists' names who performed the track. If there is more than one artist, they are separated by a ;
album_name: The album name in which the track appears
track_name: Name of the track
popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
duration_ms: The track length in milliseconds
explicit: Whether the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
loudness: The overall loudness of a track in decibels (dB)
mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
track_genre: The genre in which the track belongs
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
from datetime import datetime
import gradio as gr
def main(): #XXXTBD Static Pipeline, hyperparameter optimization is done already.

    df = pd.read_csv(r"D:\Users\hhhjk\pythonProject\spotify_danceability\dataset.csv")
    X, y = spotify_danceability_preprocess(df)
    # base_models(X, y)
    # best_models = hyperparameter_optimization(X, y)
    # best_model = best_models['CatBoost']
    best_model = CatBoostRegressor(depth=8,iterations=750)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_pred = y_guesses(X_train, X_test, y_train, y_test, best_model)

    top_200_getter(y_test)

    gradio_webapp(spotify_userid, client_id, client_secret, best_model)

    app = gr.Interface(fn=gradio_webapp, inputs=["text", "text", "text"], outputs="text")
    app.launch()

    return

def spotify_danceability_preprocess(df):
    df = df[df.track_genre != 'kids']

    df = df[df.track_genre != 'children']

    df = df[df.track_genre != 'study']

    df.track_genre.unique()

    outcome = 'danceability'

    df['time_signature'].unique()

    df.drop("explicit", axis=1, inplace=True)

    df.drop("Unnamed: 0", axis=1, inplace=True)

    df['time_signature'] = df['time_signature'].replace({0: 6, 1: 7})

    df.drop(65900, axis=0, inplace=True)

    duplicated_rows = df[df.duplicated(subset=['track_id'])]
    df = df.drop(duplicated_rows.index)

    duplicated_rows = df[df.duplicated(subset=['track_id', 'artists', "album_name", "track_name"])]
    df = df.drop(duplicated_rows.index)

    duplicated_rows = df[df.duplicated(subset=['track_id', "track_name"])]
    df = df.drop(duplicated_rows.index)

    duplicated_rows = df[df.duplicated(subset=['track_id', 'artists', "album_name"])]
    df = df.drop(duplicated_rows.index)

    duplicated_rows = df[df.duplicated(subset=['popularity', 'duration_ms', "danceability", "energy", "key", "loudness",
                                               "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
                                               "valence", "tempo", "time_signature"])]

    df = df.drop(duplicated_rows.index)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols.remove(outcome)

    for col in num_cols:
        replace_with_thresholds(df, col)

    df.style.set_properties(**{'text-align': 'center'})
    df.head()

    df = pd.get_dummies(df, columns=["key"], drop_first=True)
    df[['key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11']] = \
        df[['key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10',
            'key_11']].astype(int)

    df = pd.get_dummies(df, columns=["time_signature"], drop_first=True)
    df[['time_signature_4', 'time_signature_5', 'time_signature_6', 'time_signature_7']] = \
        df[['time_signature_4', 'time_signature_5', 'time_signature_6', 'time_signature_7']].astype(int)

    model_cols = [col for col in df.columns if col not in cat_but_car]

    # Standartlaştırma
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    temp_df = df.copy()
    temp_df[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=df[num_cols].index)
    df = temp_df.copy()

    y = df[outcome]

    X = df.copy()
    X.drop([outcome], axis=1, inplace=True)

    for col in cat_but_car:
        X.drop([col], axis=1, inplace=True)

    return X, y

def y_guesses(X_train, X_test, y_train, best_model):

    best_model.fit(X_train,y_train)

    y_pred_out = best_catboost_model.predict(X_test)

    return y_pred_out

def top_200_getter(y_test):
    col1 = list(y_test.index)
    col2 = y_pred
    y_pred_ind = pd.DataFrame(col2, index=col1)
    top_200 = y_pred_ind.sort_values(by=0).tail(200).index

    return top_200


def random_tracks_getter(top_200, num_of_tracks, random=True):
    if random:
        if num_of_tracks <= TWO_HUNDRED:
            random_n_tracks_ind = random.sample(list(top_200), num_of_tracks)
    else: #random=False
        if num_of_tracks <= TWO_HUNDRED:
            random_n_tracks_ind = top_200[num_of_tracks]
        else:
            random_n_tracks_ind = top_200

    random_50_tracks = [df_.iloc[indd] for indd in random_n_tracks_ind]
    random_50_tracks_ids = [track['track_id'] for track in random_50_tracks]

    random_tracks = [df_.iloc[indd] for indd in top_50]
    rand_tracks_ids = [track['track_id'] for track in random_tracks]

    return rand_tracks_ids


if __name__ == "__main__":
    print("Pipeline has started!!")
    main()