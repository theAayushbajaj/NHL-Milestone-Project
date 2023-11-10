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
# encoding
# from category_encoders import TargetEncoder
import xgboost as xgb
import os
import warnings


# in this part we fix the nas
df = pd.read_csv('new_df_data.csv')
print(df.columns)

# check to see if there are any Nans
print(df.isna().any())

# print the column names who have Nans
print('columns with empty values')
nans_cols = df.columns[df.isnull().any()].to_list()
print(nans_cols)


# nans_cols.remove('Goalie')
# df[nans_cols]

index = df[df['shotType'].isna()]['shotType'].index

df['shotType'] = df['shotType'].fillna('Wrist Shot')
df[['x_last_event', 'y_last_event', 'distance_from_last_event', 'speed']]= df[['x_last_event', 'y_last_event', 'distance_from_last_event', 'speed']].fillna(0)


df['team'] = None
index = df[df['strength'].isna()]['strength'].index
index_away = (df['away_team_name'] == df['team_name'])
index_home = (df['home_team_name'] == df['team_name'])
df.loc[index_away, 'team'] = 'away'
df.loc[index_home, 'team'] = 'home'

even = df['num_player_home'] == df['num_player_away']
comp = df['num_player_home'] > df['num_player_away']

away = df['team'] == 'away'
home = df['team'] == 'home'
df.loc[even,'strength'] = 'Even'
df.loc[away & ~comp,'strength'] = 'Power Play'
df.loc[away & comp,'strength'] = 'Short Handed'
df.loc[home & comp,'strength'] = 'Power Play'
df.loc[home & ~comp,'strength'] = 'Short Handed'


df.to_csv('df_na_fixed.csv')






