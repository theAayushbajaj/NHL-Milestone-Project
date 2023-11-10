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
# Method 1: mutual information
print('Method 1-Mutual Information')
X_train = Train[desired_columns[1:-1]]
y_train = Train[desired_columns[-1]]

# calculate mutual information
selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X_train, y_train)

# get the scores
feature_scores = selector.scores_
feature_names = X_train.columns

# create a dataframe to show the selected features and their corresponding scores (importance)
feature_scores_df = pd.DataFrame({'Feature': feature_names, 'Mutual_Information_Score': feature_scores})
feature_scores_df = feature_scores_df.sort_values(by='Mutual_Information_Score', ascending=False)
print(feature_scores_df)

model = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.7,
                            colsample_bytree=1,  reg_lambda=0.65,
                            seed=seed)
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Feature_Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Feature_Importance', ascending=False)


k = 10 # we have 17 features
selected_features_mutual_info = feature_scores_df['Feature'].head(k).tolist()
print("Selected Features:", selected_features_mutual_info)


##################################################################################
# Method 2: wrapper method
print('Method 2-Wrapper Method')
model = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.7,
                            colsample_bytree=1,  reg_lambda=0.65,
                            seed=seed)
n_features_to_select = 10
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)
selected_features_wrapper = X_train.columns[rfe.support_].tolist()
feature_ranking = rfe.ranking_
feature_importance_df_wrapper = pd.DataFrame({'Feature': X_train.columns, 'Feature_Importance': feature_ranking})
feature_importance_df_wrapper = feature_importance_df_wrapper.sort_values(by='Feature_Importance')
print(feature_importance_df_wrapper)
print("Selected Features:", selected_features_wrapper)


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
sns.set(font_scale=1)
sns.barplot(data=feature_importances_df, x='Feature_Importance', y='Feature', ax=axes[0])
axes[0].set_title('Mutual Information \nFeature Importance')
axes[0].set_xlabel('Mutual Information Score (Higher is Better)')
axes[0].set_ylabel('Feature')

sns.barplot(data=feature_importance_df_wrapper, x='Feature_Importance', y='Feature', ax=axes[1])
axes[1].set_title('RFE Feature Ranking')
axes[1].set_xlabel('Ranking (Lower is Better)')
axes[1].set_ylabel('Feature')
plt.suptitle('Comparison of two method for Feature Importance: (1) Mutual Information \n and (2) Wrapper Method',
             x=0.57, y=0.98)
plt.tight_layout()
plt.savefig('rank_importance.png', dpi=200)
plt.show()









