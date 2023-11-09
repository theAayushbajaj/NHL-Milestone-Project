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

def split_train_val_test(data):
    """
    Splits the input dataframe into training, validation, and test sets based on the season column.

    Args:
    - df_data: pandas DataFrame containing the data to be split

    Returns:
    - train: pandas DataFrame containing the training data (seasons 16-19)
    - val: pandas DataFrame containing the validation data (season 19)
    - test: pandas DataFrame containing the test data (season 20)
    """
    # season 16 to 19 as training data
    train = data[data['season'] < 2020]
    # season 20 as test data
    test = data[data['season'] == 2020]

    # validation set as last year of training
    val_index = train['season'] == 2019
    val = train[val_index]
    train = train[~val_index]
    return train, val, test


# %%
