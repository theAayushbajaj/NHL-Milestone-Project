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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.tree import DecisionTreeClassifier
import shap
seed = 200

##################################################################################
data = pd.read_csv('df_na_fixed.csv', index_col=0)

desired_columns = ['season', 'emptyNet', 'game_seconds', 'period', 'x_coordinate', 'y_coordinate',
                  'shot_distance', 'shot_angle', 'shotType', 'strength', 'last_event_type',
                  'x_last_event', 'y_last_event', 'time_from_last_event', 'distance_from_last_event',
                  'is_rebound', 'change_in_shot_angle', 'speed', 'is_goal']

# encode is_rebound into 1 or 0
label_encoder = LabelEncoder()
data['is_rebound'] = label_encoder.fit_transform(data['is_rebound'])

# encode 'strength' into 0, 1, 2
custom_order = ['Short Handed', 'Even', 'Power Play']
label_encoder_order = LabelEncoder()
label_encoder_order.fit(custom_order)
data['strength'] = label_encoder_order.transform(data['strength'])

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
##################################################################################
X_train = Train[desired_columns[1:-1]]
y_train = Train[desired_columns[-1]]
X_val = Val[desired_columns[1:-1]]
y_val = Val[desired_columns[-1]]
model = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.7,
                            colsample_bytree=1,  reg_lambda=0.65,
                            seed=seed)
model.fit(X_train, y_train)
explainer = shap.Explainer(model, X_train)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, plot_type='bar')