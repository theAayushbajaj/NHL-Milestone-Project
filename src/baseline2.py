from comet_ml import Experiment
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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import os
import warnings
import utils_ar_pa as ut
import joblib
#####################################
# hide warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")
######################################
# my_key = os.environ.get("comet_key")
# print('Experiment Started!')
# # Create an experiment with your api key
# experiment = Experiment(
#     api_key=my_key,
#     project_name="baseline model",
#     workspace="2nd milestone",
#     log_code=True,
#     auto_param_logging=True,
#     auto_metric_logging=True)
#######################################
# Set display options to show all columns
pd.set_option('display.max_columns', None)
# set seed
seed = 200
#########################################
df_base = pd.read_csv('baseline_model_data.csv')

# adding space to the white space in column names
cols_rename_base = ut.renamer(df_base)
df_base.rename(columns=cols_rename_base, inplace=True)
print(df_base.columns)

# filter season 2020 for test and the remaining goes to train
df_base_test = df_base[df_base.season == 2020]
df_base_train = df_base[df_base.season < 2020]

target = 'is_goal'

features_list = [['shot_distance'], ['shot_angle'],
                 ['shot_distance', 'shot_angle']]

name = 'logistic_regression'
goal_rate_list = []
cumulative_goals_list = []
prob_true_list = []
prob_pred_list = []
fpr_list = []
tpr_list = []
thresholds_list = []
# model_list = []


for features in features_list:
    train, val, test = ut.feature_selector(df_base_train, df_base_test, features, target,
                                           random_state=seed)
    model = LogisticRegression(random_state=seed)

    goal_rate, cumulative_goals, prob_true, prob_pred, fpr, tpr, thresholds, model = ut.plot_calibration_cal_paul(model, features, target, val, train, name)

    goal_rate_list.append(goal_rate)
    cumulative_goals_list.append(cumulative_goals)
    prob_true_list.append(prob_true)
    prob_pred_list.append(prob_pred)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thresholds_list.append(thresholds)

from utils_ar_pa import RandomModel_paul
model = RandomModel_paul()
goal_rate, cumulative_goals, prob_true, prob_pred, fpr, tpr, thresholds, model = ut.plot_calibration_cal_paul(model, features, target, val, train, name)

goal_rate_list.append(goal_rate)
cumulative_goals_list.append(cumulative_goals)
prob_true_list.append(prob_true)
prob_pred_list.append(prob_pred)
fpr_list.append(fpr)
tpr_list.append(tpr)
thresholds_list.append(thresholds)

name = 'baseline'
ut.plots(goal_rate_list, cumulative_goals_list, prob_true_list,
          prob_pred_list, fpr_list, tpr_list, thresholds_list, name)
