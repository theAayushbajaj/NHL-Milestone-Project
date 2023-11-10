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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# encoding
import category_encoders as ce
import xgboost as xgb
import os
import warnings
import utils_ar_pa as ut

seed = 200

# in this part we encode the cols
df = pd.read_csv('df_na_fixed.csv', index_col=0)
print(df.columns)

# check to see if there are any Nans
print(df.isna().any())
# the only empty col is name of the Goalie which is not a predictor

desired_columns = ['season', 'emptyNet', 'game_seconds', 'period', 'x_coordinate', 'y_coordinate',
                  'shot_distance', 'shot_angle', 'shotType', 'strength', 'last_event_type',
                  'x_last_event', 'y_last_event', 'time_from_last_event', 'distance_from_last_event',
                  'is_rebound', 'change_in_shot_angle', 'speed', 'is_goal']

data = df[desired_columns].copy()

col_types = data[desired_columns].dtypes
print(col_types)
cats =  data.select_dtypes(include=['object']).columns
bools = data.select_dtypes(include=['bool']).columns
number = data.select_dtypes(include=['int64', 'float64']).columns
print('cats', cats)
print('bools', bools)
print('number', number)

# encode is_rebound into 1 or 0
label_encoder = LabelEncoder()
data['is_rebound'] = label_encoder.fit_transform(data['is_rebound'])

# encode 'strength' into 0, 1, 2
custom_order = ['Short Handed', 'Even', 'Power Play']
label_encoder_order = LabelEncoder()
label_encoder_order.fit(custom_order)
data['strength'] = label_encoder_order.transform(data['strength'])

# for preventing data leakage from now on we need to split the dataset into train, val, and test
# before any further encoding and transformation
Test = data[data['season'] == 2020][desired_columns[1:]]
train_val = data[data['season'] < 2020]
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

# encode shotType using TargetEncoder
# the reason for this choice is that the number of categories are high
encoder = ce.TargetEncoder()
# Fit and transform the shotType column
Train['shotType'] = encoder.fit_transform(Train['shotType'], Train['is_goal'])
# Transform shotType column using the same encoder in Val and Test
Val['shotType'] = encoder.transform(Val['shotType'])
Test['shotType'] = encoder.transform(Test['shotType'])

# encode last_event_type using TargetEncoder
# the reason for this choice is that the number of categories are high
encoder_last_event = ce.TargetEncoder()
# Fit and transform the shotType column
Train['last_event_type'] = encoder_last_event.fit_transform(Train['last_event_type'], Train['is_goal'])
# Transform shotType column using the same encoder in Val and Test
Val['last_event_type'] = encoder_last_event.transform(Val['last_event_type'])
Test['last_event_type'] = encoder_last_event.transform(Test['last_event_type'])


# Here we have chosen the columns who have different metrics
# and we will scale all of them
# note that for preventing data leakage, we use the fit_transform method on Train
# but we use the transform method for Test and Val

columns_metrics = ['game_seconds', 'x_coordinate',
                   'y_coordinate', 'shot_distance',
                   'shot_angle', 'x_last_event',
                   'y_last_event', 'time_from_last_event',
                   'distance_from_last_event',
                   'change_in_shot_angle',
                   'speed']
for col in columns_metrics:
    scaler = StandardScaler()
    Train[col] = scaler.fit_transform(Train[[col]])
    Test[col] = scaler.transform(Test[[col]])
    Val[col] = scaler.transform(Val[[col]])

# now our data is ready for plugging into our model
# lets save the data here for the next part
Train.to_csv('Train.csv')
Val.to_csv('Val.csv')
Test.to_csv('Test.csv')
