import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Sckit learn modules and classes
from sklearn.metrics import confusion_matrix, f1_score, \
precision_score, recall_score, classification_report, accuracy_score, \
roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# encoding
# from category_encoders import TargetEncoder
import xgboost as xgb
import os
import warnings
seed=200


# in this part we fix the nas
df = pd.read_csv('new_df_data.csv')
print(df.columns)

# check to see if there are any Nans
print(df.isna().any())

# print the column names who have Nans
print('columns with empty values')
nans_cols = df.columns[df.isnull().any()].to_list()
print(nans_cols)


nans_cols.remove('Goalie')
df[nans_cols]

# index = df[df['shotType'].isna()]['shotType'].index

shot_type_frequent = df['shotType'].value_counts().idxmax()
df['shotType'] = df['shotType'].fillna(shot_type_frequent)
df[['x_last_event', 'y_last_event', 'distance_from_last_event', 'speed']] = df[['x_last_event', 'y_last_event', 'distance_from_last_event', 'speed']].fillna(0)


df['team'] = None

index = df[df['strength'].isna()]['strength'].index
index_away = (df['away_team_name'] == df['team_name'])
index_home = (df['home_team_name'] == df['team_name'])
df.loc[index_away, 'team'] = 'away'
df.loc[index_home, 'team'] = 'home'

even = df['num_player_home'] == df['num_player_away']
comp_home = (df['num_player_home'][~even] > df['num_player_away'][~even])
comp_away = (df['num_player_home'][~even] < df['num_player_away'][~even])

# only keep those who are bigger
# remove those who are smaller or equal

away = df['team'] == 'away'
home = df['team'] == 'home'
df.loc[even,'strength'] = 'Even'
df.loc[away & comp_home,'strength'] = 'Power Play'
df.loc[away & ~comp_home,'strength'] = 'Short Handed'


df.loc[home & comp_away,'strength'] = 'Power Play'
df.loc[home & ~comp_away,'strength'] = 'Short Handed'


desired_columns = ['season', 'emptyNet', 'game_seconds', 'period', 'periodTime', 'x_coordinate', 'y_coordinate',
                   'shot_distance', 'shot_angle', 'shotType', 'strength', 'last_event_type',
                   'x_last_event', 'y_last_event', 'time_from_last_event', 'distance_from_last_event',
                   'is_rebound', 'change_in_shot_angle', 'speed',
                   'home_team_name', 'away_team_name', 'team_name', 'home_goal', 'away_goal',
                   'num_player_home',
                   'num_player_away', 'time_power_play',
                   'is_goal']


# for preventing data leakage from now on we need to split the dataset into train, val, and test
# before any further encoding and transformation
Test = df[df['season'] == 2020][desired_columns[1:]]
train_val = df[df['season'] < 2020]
X = train_val[desired_columns[1:-1]]
y = train_val['is_goal']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

# rebuild the dataframe
y_train_df = pd.DataFrame({'is_goal': y_train.to_numpy()}).reset_index(drop=True)
Train = pd.concat([X_train.reset_index(drop=True), y_train_df], axis=1, ignore_index=True)
Train.columns = desired_columns[1:]
# rebuild the dataframe
y_val_df = pd.DataFrame({'is_goal': y_val.to_numpy()}).reset_index(drop=True)
Val = pd.concat([X_val.reset_index(drop=True), y_val_df], axis=1, ignore_index=True)
Val.columns = desired_columns[1:]


# fill nans in the shotType column
# Fit and transform the shotType column
imputer =  SimpleImputer(strategy='constant', fill_value='missing')
Train[['shotType']] = imputer.fit_transform(Train[['shotType']])
# Transform shotType column using the same encoder in Val and Test
Val[['shotType']] = imputer.transform(Val[['shotType']])
Test[['shotType']] = imputer.transform(Test[['shotType']])


# fill nans in cols 'x_last_event', 'y_last_event', 'distance_from_last_event', 'speed'
numerical_nans_cols = ['x_last_event', 'y_last_event', 'distance_from_last_event', 'speed']

for index, col in enumerate(numerical_nans_cols):
    imputer = SimpleImputer(strategy='median', fill_value='missing')
    Train[[col]] = imputer.fit_transform(Train[[col]])
    # Transform shotType column using the same encoder in Val and Test
    Val[[col]] = imputer.transform(Val[[col]])
    Test[[col]] = imputer.transform(Test[[col]])


# df.to_csv('df_na_fixed.csv')

Train.to_csv('Train_imp.csv')
Test.to_csv('Test_imp.csv')
Val.to_csv('Val_imp.csv')






