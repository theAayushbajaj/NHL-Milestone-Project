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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
# encoding
import category_encoders as ce
from category_encoders.binary import BinaryEncoder
from category_encoders.target_encoder import TargetEncoder
import xgboost as xgb
import os
import warnings
from sklearn.impute import SimpleImputer
import utils_ar_pa as ut

seed = 200

# in this part we encode the cols
Train = pd.read_csv('Train_imp.csv', index_col=0)
Test = pd.read_csv('Test_imp.csv', index_col=0)
Val = pd.read_csv('Val_imp.csv', index_col=0)


# encode is_rebound into 1 or 0
label_encoder = LabelEncoder()
Train['is_rebound'] = label_encoder.fit_transform(Train['is_rebound'])
Test['is_rebound'] = label_encoder.transform(Test['is_rebound'])
Val['is_rebound'] = label_encoder.transform(Val['is_rebound'])

# encode 'strength' into 0, 1, 2
custom_order = ['Short Handed', 'Even', 'Power Play']
label_encoder_order = LabelEncoder()
label_encoder_order.fit(custom_order)
Train['strength'] = label_encoder.fit_transform(Train['strength'])
Test['strength'] = label_encoder.transform(Test['strength'])
Val['strength'] = label_encoder.transform(Val['strength'])

# encode the shotType column
# Fit and transform the shotType column
binary_encoder = TargetEncoder(cols=['shotType', 'last_event_type', 'team_name',
                                     'home_team_name',  'away_team_name'],
                                return_df=True,
                                min_samples_leaf = 20,
                                smoothing = 10,
                                hierarchy = None)

Train.iloc[:, :-1] = binary_encoder.fit_transform(Train.iloc[:, :-1], Train.iloc[:, -1])
Val.iloc[:, :-1] = binary_encoder.transform(Val.iloc[:, :-1])
Test.iloc[:, :-1]= binary_encoder.transform(Test.iloc[:, :-1])


# transform the periodTime into seconds
Train['periodTime'] = Train['periodTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
Test['periodTime'] = Test['periodTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
Val['periodTime'] = Val['periodTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))



# Here we have chosen the columns who have different metrics
# and we will scale all of them
# note that for preventing data leakage, we use the fit_transform method on Train
# but we use the transform method for Test and Val

columns_metrics = ['game_seconds', 'periodTime', 'x_coordinate',
                   'y_coordinate', 'shot_distance',
                   'shot_angle', 'x_last_event',
                   'y_last_event', 'time_from_last_event',
                   'distance_from_last_event',
                   'change_in_shot_angle',
                   'speed', 'time_power_play']

for col in columns_metrics:
    scaler = MinMaxScaler()
    Train[col] = scaler.fit_transform(Train[[col]])
    Test[col] = scaler.transform(Test[[col]])
    Val[col] = scaler.transform(Val[[col]])

# now our data is ready for plugging into our model
# lets save the data here for the next part
Train.to_csv('Train_tg.csv')
Val.to_csv('Val_tg.csv')
Test.to_csv('Test_tg.csv')

