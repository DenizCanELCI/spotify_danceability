
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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


df_ = pd.read_csv(r"D:\Users\hhhjk\pythonProject\spotify_danceability\dataset.csv")

df = df_.copy()

#######################################################################
# 1. Exploratory Data Analysis
#######################################################################
df.info()

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


def check_df(dataframe, head=5):
    print("#################### Shape ######################")
    print(dataframe.shape)
    print("#################### Types ######################")
    print(dataframe.dtypes)
    print("#################### Head #######################")
    print(dataframe.head(head))
    print("#################### Tail #######################")
    print(dataframe.tail(head))
    print("#################### NA #########################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ##################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)


check_df(df)


def grab_col_names(dataframe, cat_th=13, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols.remove(outcome)



def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=False)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, outcome, col)


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={"size": 12}, linecolor="w", cmap="RdBu")
    plt.show(block=True)


correlation_matrix(df, num_cols)

#######################################################################
# 2. Data Preprocessing & Feature Engineering
#######################################################################


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



####################################################################################################################



for col in num_cols:
    print(col, ":", check_outlier(df, col))


for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, ":", check_outlier(df, col))

####################################################################################################################
#TASK - Encoding & Features


df.style.set_properties(**{'text-align': 'center'})
df.head()

df = pd.get_dummies(df, columns=["key"], drop_first=True)
df[['key_1','key_2','key_3','key_4','key_5','key_6','key_7','key_8','key_9','key_10','key_11']] = \
    df[['key_1','key_2','key_3','key_4','key_5','key_6','key_7','key_8','key_9','key_10','key_11']].astype(int)

df = pd.get_dummies(df, columns=["time_signature"], drop_first=True)
df[['time_signature_4','time_signature_5','time_signature_6','time_signature_7']] = \
    df[['time_signature_4','time_signature_5','time_signature_6','time_signature_7']].astype(int)


model_cols = [col for col in df.columns if col not in cat_but_car]

#######################################################################
# 3. Base Model
#######################################################################


# Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
temp_df = df.copy()
temp_df[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=df[num_cols].index)
df = temp_df.copy()


y = df[outcome]

X = df.copy()
X.drop([outcome], axis=1, inplace=True)

for col in cat_but_car:
    X.drop([col], axis=1,inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def base_models_rootmse(X, y, scoring="neg_root_mean_squared_error"):
    print("Base Models....")
    regressors = [
        ("XGBoost", XGBRegressor()),
        ("LightGBM", LGBMRegressor(verbose=-1)),
        ("CatBoost", CatBoostRegressor(verbose=False))
    ]
    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(np.sqrt(-cv_results['test_score'].mean()), 4)} ({name}) ")
    return



base_models_rootmse(X_train, y_train)
# Base Models_rootmse
#neg_root_mean_squared_error: 0.1063 (XGBoost)
#neg_root_mean_squared_error: 0.1073 (LightGBM)
#neg_root_mean_squared_error: 0.1039 (CatBoost)
########## XGBoost ##########
# neg_root_mean_squared_error (Before):  0.1054
# neg_root_mean_squared_error (After): 0.1042
# XGBoost best params: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 70}
########## LightGBM ##########
# neg_root_mean_squared_error (Before):  0.1069
# neg_root_mean_squared_error (After): 0.1068
# LightGBM best params: {'learning_rate': 0.3, 'max_depth': 6, 'n_estimators': 70}
########## CatBoost ##########
# neg_root_mean_squared_error (Before):  0.1032

#######################################################################
# 4. Automated Hyperparameter Optimization
#######################################################################

xgboost_params = {"learning_rate": [0.1],
                  "max_depth": [10, 12, 14],
                  "n_estimators": range(50, 80, 10)}
lightgbm_params = {'learning_rate': [0.01, 0.1, 0.3],
                   'max_depth': [4, 6, 8],
                   'n_estimators': range(50,80,10)}

# catboost_params = {'eval_metric': ['RMSE','MAPE'],
#                    'iterations': range(500, 1000, 250),
#                    'depth': [4, 6, 8, 12]
# }
catboost_params = {'eval_metric': ['RMSE','MAPE'],
                   'iterations': range(500, 1000, 250),
                   'depth': [4, 6, 8, 12]
}


regressors_hpo = [
    ("XGBoost", XGBRegressor(), xgboost_params),
    ("LightGBM", LGBMRegressor(verbose=-1), lightgbm_params),
    ("CatBoost", CatBoostRegressor(verbose=False), catboost_params)
]


def hyperparameter_optimization_rmse(X, y, cv=4, scoring="neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors_hpo:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before):  {-round(cv_results['test_score'].mean(),4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X,y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {-round(cv_results['test_score'].mean(),4)}")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization_rmse(X_train, y_train, scoring='neg_root_mean_squared_error')

#Hyperparameter Optimization....
########## XGBoost ##########
#neg_root_mean_squared_error (Before):  0.1052
#neg_root_mean_squared_error (After): 0.1042
#XGBoost best params: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 300}


best_xgb_model = XGBRegressor(learning_rate=0.1,max_depth=10,n_estimators=300)
best_catboost_model = CatBoostRegressor(depth=8,eval_metric='RMSE',iterations=750)

best_xgb_model.fit(X_train,y_train)
best_catboost_model.fit(X_train, y_train)
y_pred = best_xgb_model.predict(X_test)
y_pred = best_catboost_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred, squared=False) #rmse hesabı
print("Root Mean Squared Error:", mse)
# Mean Squared Error: 0.10400572469934198
# CatBoost: Root Mean Squared Error: 0.10100101816263003

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:',mae)
# Mean Absolute Error: 0.08132177375680566
# CatBoost: Mean Absolute Error: 0.07879955768575692


#---------------------------------------------Aşağısı çalışma halinde !-----------------------------------
col1 = list(y_test.index)
col2 = y_pred
y_pred_ind = pd.DataFrame(col2, index=col1)

top_200 = y_pred_ind.sort_values(by=0).tail(200).index
y_test.sort_values()

random_50_tracks_ind = random.sample(top_200, 50) #XXXTBD

# y_prd_most = best_xgb_model.predict(pd.DataFrame(X[X.index==66808]))
#
# random_50_tracks = df.iloc[random_50_tracks_ind[1]]
random_50_tracks = [df_.iloc[indd] for indd in random_50_tracks_ind]

random_50_tracks_ids = [track['track_id'] for track in random_50_tracks]

client_id = '1cc97646ee854447944864d5e0eb3ab8' #XXXTBD
client_secret = .env.client_secret #XXXTBD

def spotipy_add_playlist(inp_client_id,
                         inp_client_secret,
                         username_id = 11124005204,
                         inp_scope="playlist-modify-public playlist-modify-private"):
    """
    :param username_id:
    :return:
    """

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=inp_client_id,
                                                   client_secret= inp_client_secret,
                                                   redirect_uri='https://open.spotify.com/',
                                                   scope=inp_scope))

    hour_now = datetime.now().hour
    minute_now = datetime.now().minute
    playlist_name = "ML Dance Playlist "+str(hour_now)+"_"+str(minute_now)+"_time"
    playlist_description = "This is my new Dance playlist"

    playlist = sp.user_playlist_create(user=username_id, name=playlist_name, public=True,
                                       description=playlist_description)

    # Add tracks to the playlist
    # track_uris = ["spotify:track:4iV5W9uYEdYUVa79Axb7Rh", "spotify:track:2takcwOaAZWiXQijPHIx7B"]
    track_uris = random_50_tracks_ids

    sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)

    return

spotipy_add_playlist(client_id, client_secret)

#- GRADIO WEB APP - INWORK --------------------------------------------------------------------------------------

def gradio_webapp(spotify_userid, client_id, client_secret):
    try:
        if __name__ == "__main__":
            main()
        return "Hello " + spotify_userid + (" The ML Dance Playlist has been created: \n\t"+str(playlist_name))
    except:
        return "There was an error, the playlist has not been created."


app = gr.Interface(fn = greet_user, inputs="text", outputs="text")

app.launch()


# - ML PIPELINE ------------------------------------------------------------------------
def main():

    df = pd.read_csv(r"D:\Users\hhhjk\pythonProject\spotify_danceability\dataset.csv")
    X, y = spotify_danceability_preprocess(df)
    # base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    best_model = best_models['CatBoost']

    gradio_webapp(spotify_userid, client_id, client_secret, best_model)

    app = gr.Interface(fn=gradio_webapp, inputs=["text", "text"], outputs="text")
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

if __name__ == "__main__":
    main()

