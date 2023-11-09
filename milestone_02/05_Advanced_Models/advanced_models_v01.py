#%%

import numpy as np
import pandas as pd
import warnings
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from comet_ml import Experiment
import joblib

#%%
# print current working directory
import os
print(os.getcwd())

#%%

# change working directory to parent
os.chdir('../')
print(os.getcwd())

#%%
import functions_mlstn_02 as _fct

#%%
# reset working directory
os.chdir('./05_Advanced_Models')
print(os.getcwd())







#%%

# load data
data = pd.read_csv('../data_for_remaining_tasks/df_data.csv')
data

# %%
# load the data used in baseline_models (part 3)
data_baseline = pd.read_csv('../03_baseline/baseline_model_data.csv')
# %%

train_base, val_base, test_base = _fct.split_train_val_test(data_baseline)

# %%
