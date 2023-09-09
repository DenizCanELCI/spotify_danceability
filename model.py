
"""
track_id: The Spotify ID for the track
artists: The artists' names who performed the track. If there is more than one artist, they are separated by a ;
album_name: The album name in which the track appears
track_name: Name of the track
popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
duration_ms: The track length in milliseconds
explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
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
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


df_ = pd.read_csv(r"D:\Users\hhhjk\pythonProject\spotify_danceability\dataset.csv")

"""df_.head()

df_.info()

df_.isnull().sum() #There is one null value



artists             1
album_name          1
track_name          1

df[df["artists"].isnull()]


                    track_id artists album_name track_name  popularity  \
65900  1kR4gIb7nGxHPI3D2ifs59     NaN        NaN        NaN           0   
       duration_ms  explicit  danceability  energy  key  loudness  mode  \
65900            0     False         0.501   0.583    7     -9.46     0   
       speechiness  acousticness  instrumentalness  liveness  valence  \
65900       0.0605          0.69           0.00396    0.0747    0.734   
         tempo  time_signature track_genre  
65900  138.391               4       k-pop  


df_.describe().T"""

df_ = df_.drop(65900, axis=0)


#######################################################################
# 1. Exploratory Data Analysis
#######################################################################

df = df_.copy()
#-----------------------------------------------*************************************************
# TASK - Levitas kaç şarkıda geçiyor bulma
# lst = []
# for i, el in enumerate(list(df['artists'])):
#     # lst.append(str(i))
#     if type(el) == float:
#         lst.append({i,0})
#     elif 'evitas' in el:
#         lst.append(1)
#     else: lst.append(0)
#
# lst.count(float(1))
#
# indexes = [i for i in range(len(lst)) if lst[i] == 1]
#
# df.iloc[40531]

#-----------------------------------------------*************************************************

df = df.drop("Unnamed: 0", axis=1)

# Time_signature için 3,4,5,6,7 değerleri atanmış (ex: 3 4'lük, 4 4'lük vs.) Fakat verisetinde bunlar 3,4,5,0,1 olarak gözüküyor. Bunlar:
# 0 ->6 ve 1 ->7 olarak atandı.

df['time_signature'] = df['time_signature'].replace({0: 6, 1: 7})


outcome = 'danceability'
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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

except_explicit = df.drop("explicit", axis=1) # check_df fonksiyonunda "explicit" değişkeni hata veriyor bu yüzden çıkardım

# check_df(except_explicit) #DE: bende except_explicit için de hata verdi, sadece numeric değerler gönderilebilir


# cat treshold değeri 10 olarak atanmış. Fakat veriseti parametre acıklamalarından da anlaşılacağı üzere key değeri 12 adet. Anahtar
# nota kategorik olarak kullanılmalı

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
df[cat_cols]

df[num_cols]

df[cat_but_car]

df['track_genre'].nunique()

df['track_genre'].unique()

df.info()

check_df(df[num_cols])

# check_df(df[cat_cols])


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

df[cat_cols]

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

df[num_cols]

for col in num_cols:
    num_summary(df, col)


# def target_summary_with_num(dataframe, target, numerical_col):
#     print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
#
# for col in num_cols:
#     target_summary_with_num(df, outcome, col)

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
# df.isnull().any()



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

# num_cols.remove('Unnamed: 0')
num_cols

for col in num_cols:
    check_outlier(df, col)

df[num_cols]

####################################################################################################################
#TASK - Outlier tespiti ve kaldirilmasi
for col in num_cols:
    print(f'col = {col}\tis true? =  {check_outlier(df, col)}')

# Outlier'ları iyileştiriyoruz
for col in num_cols:
    replace_with_thresholds(df, col)


# Tekrar kontrol ediyoruz outlier durumlarını
for col in num_cols:
    print(f'col = {col}\tis true? =  {check_outlier(df, col)}')
#outlier kalmadı!
####################################################################################################################


####################################################################################################################

#TASK - Lof ile outlier tespiti
# from sklearn.neighbors import LocalOutlierFactor
# clf = LocalOutlierFactor(n_neighbors=20)
# lof_out = clf.fit_predict(df[num_cols])
# len(lof_out)
# Counter(lof_out)
#
# df_scores = clf.negative_outlier_factor_
# np.sort(df_scores)[0:5]
#
# scores = pd.DataFrame(np.sort(df_scores))
# scores.plot(stacked=True, xlim=[0, 50], style='.-')
# plt.show()
#
# th = np.sort(df_scores)[3]
#
# df[df_scores < th]
#
# df[df_scores < th].shape
#
# df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
#
# df[df_scores < th].index
#
# df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
#
# df.shape
####################################################################################################################
# TASK - Spotify Web API
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
#
# client_id = '1cc97646ee854447944864d5e0eb3ab8'
# client_secret = 'your_client_secret'
#
# client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
#
# results = sp.search(q='weezer', limit=20)
# for idx, track in enumerate(results['tracks']['items']):
#     print(idx, track['name'])

#
####################################################################################################################
#TASK - Encoding

df.head()

df[cat_cols] # 'explicit', 'mode', 'time_signature'
df['time_signature'].unique() # ordinality var! Bkz açıklama docstring'i.

temp_df = df.copy()
temp_df['explicit'].unique()
temp_df['explicit'] = [1 if el == True else 0 for el in temp_df['explicit']]
# True False yerine 1-0 a çevirdik.

df = temp_df.copy()
df['explicit'].unique() # array([0, 1], dtype=int64)

df.style.set_properties(**{'text-align': 'center'})
df.head()

df['isFlat'] = 1

df.loc[df['key'].isin([0, 2, 4, 5, 7, 9, 11]), 'isFlat'] = 0

df["is4/4"] = 1

df.loc[df['time_signature'].isin([3,5,6,7]), 'is4/4'] = 0

df.columns
cat_but_car
df[cat_but_car]
model_cols = [col for col in df.columns if col not in cat_but_car]
df[model_cols]

####################################################################################################################
####################################################################################################################
#TASK - Encoding

####################################################################################################################

#######################################################################
# 3. Base Model
#######################################################################
from sklearn.ensemble import RandomForestRegressor

# rfr_model = RandomForestRegressor(n_estimators=100,criterion="squared_error",
#                                   max_depth=None,
#                                   min_samples_split=2,
#                                   min_samples_leaf=1,
#                                   min_weight_fraction_leaf=0.0,
#                                   max_features=1.0,
#                                   max_leaf_nodes=None,
#                                   min_impurity_decrease=0.0,
#                                   bootstrap=True,
#                                   oob_score=False,
#                                   random_state=17)
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
df.isna().sum()
from sklearn.preprocessing import StandardScaler
# Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)  # İsimlendirmeleri Düzeltiyoruz

nan_rows = df[df.isna().any(axis=1)].index #Nan satırı buluyoruz
df.iloc[nan_rows-1]
df.drop(nan_rows, inplace=True)
df.isna().sum()

y = df[outcome]
X = df.copy()
X.drop([outcome], axis=1, inplace=True)
for col in cat_but_car:
    X.drop([col], axis=1,inplace=True)

# cols = df.columns[df.eq('Gen Hoshino').any()]

df['track_id']
#----------------------------------------------------------------------------
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/(Y_actual+0.1)))*100
    return mape

lrmodel = LinearRegression()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lrmodel.fit(X_train, y_train)

df[outcome]
y_pred = lrmodel.predict(X_test)

# cv_results = cross_validate(lrmodel, X, y, cv=5, scoring='f1')
mape = MAPE(y_test, y_pred)

# print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# accuracy = lrmodel.score(X_test, y_test)
accuracy_lr = (100 - mape)
print('Accuracy:', round(accuracy_lr, 2), '%.')# Accuracy = 79.71 %.
#----------------------------------------------------------------------------
rfr_model = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0)
rfr_model.fit(X_train, y_train)

y_pred = rfr_model.predict(X_test)


errors = abs(y_pred - y_test)
mape = MAPE(y_test, y_pred) #100 * (errors / y_test)
accuracy = 100 - mape

print('Accuracy:', round(accuracy, 2), '%.') #Accuracy: 79.46 %.
#----------------------------------------------------------------------------

xgbr_model = XGBRegressor()
xgbr_model.fit(X_train, y_train)

xgbr_model.get_params
#
y_pred = xgbr_model.predict(X_test)

errors = abs(y_pred - y_test)
mape = 100 * (errors / (y_test+0.1))
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.') # Accuracy: 86.98 %.
#----------------------------------------------------------------------------
lgbm_model = LGBMRegressor(verbose=-1)
lgbm_model.fit(X_train, y_train)

lgbm_model.get_params()
#{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': None, 'num_leaves': 31, 'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'verbose': -1}
y_pred = lgbm_model.predict(X_test)

errors = abs(y_pred - y_test)
mape = 100 * (errors / (y_test+0.1))
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.') # Accuracy: 85.93 %
#----------------------------------------------------------------------------
ctboost_model = CatBoostRegressor(verbose=False)
ctboost_model.fit(X_train, y_train)

ctboost_model.get_all_params()

y_pred = ctboost_model.predict(X_test)

errors = abs(y_pred - y_test)
mape = 100 * (errors / (y_test+0.1))
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.') # Accuracy: 86.87 %.
#----------------------------------------------------------------------------
def base_models(X, y, scoring="r2"): # scoring="neg_mean_squared_error"
    print("Base Models....")
    regressors = [
        ("LR", LinearRegression()),
        ("KNN", KNeighborsRegressor()),
        # ("SVC", SVR()),
        # ("CART", DecisionTreeRegressor()),
        # ("RF", RandomForestRegressor()),
        # ("Adaboost", AdaBoostRegressor()),
        # ("GBM", GradientBoostingRegressor()),
        ("XGBoost", XGBRegressor()),
        ("LightGBM", LGBMRegressor(verbose=-1)),
        ("CatBoost", CatBoostRegressor(verbose=False))
    ]
    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
        if scoring == 'neg_mean_squared_error':
            print(f"{scoring}: {round(-cv_results['test_score'].mean(), 4)} ({name}) ")
        else:
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
    return
"""
Base Models....
r2: 0.6198 (XGBoost) 
r2: 0.5998 (LightGBM) 
r2: 0.6285 (CatBoost) 
"""

base_models(X, y)
# Base Models....
# r2: 0.32 (LR)
# r2: -0.0882 (KNN)
# r2: 0.6198 (XGBoost)
# r2: 0.5998 (LightGBM)
# r2: 0.6285 (CatBoost)
#######################################################################
# 4. Automated Hyperparameter Optimization
#######################################################################

# cart_params = {"max_depth": range(1, 20),
#                "min_samples_split": range(2, 30)}
xgboost_params = {"learning_rate": [0.1],
                  "max_depth": [10,12,14],
                  "n_estimators": [100,150]}
lightgbm_params = {'learning_rate':[0.01, 0.1, 0.3,],
                   'max_depth':[4,6,8],
                   'n_estimators':[100,150,250]}

ctboost_params = {'eval_metric':['RMSE','MAPE'],
                  'iterations':range(500,1500,500),
                  'depth':[4,6,8,12]
}
regressors_hpo = [
    ("XGBoost", XGBRegressor(), xgboost_params)
    # ("LightGBM", LGBMRegressor(verbose=-1), lightgbm_params)
    # ("CatBoost", CatBoostRegressor(verbose=False), ctboost_params)
]

def hyperparameter_optimization(X, y, cv=4, scoring="r2"): #    >>> scores =>('r2', 'neg_mean_squared_error')
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors_hpo:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X,y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        if scoring == 'neg_mean_squared_error':
            print(f"{scoring} (After): {-round(cv_results['test_score'].mean(), 4)}")
        else:
            print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y, scoring='neg_mean_squared_error')

best_xgb_model = XGBRegressor(learning_rate=0.1,max_depth=10,n_estimators=250)
best_xgb_model = best_models['XGBoost']



 ########## LightGBM ##########
# r2 (Before): 0.6007
# r2 (After): 0.6258
# LightGBM best params: {'learning_rate': 0.3, 'max_depth': 8, 'n_estimators': 250}

########## XGBoost ##########
# r2 (Before): 0.6249
# r2 (After): 0.672
# XGBoost best params: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 250}

########## XGBoost ##########
# neg_mean_squared_error (Before): -0.0112
# neg_mean_squared_error (After): -0.0097
# XGBoost best params: {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 300}

###############################################################################################################
# TASK - Optuna (Opsiyonel)

###############################################################################################################