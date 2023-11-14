#%%
# Basic libraries
import numpy as np
import pandas as pd
import warnings

# Preprocessing and model selection
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import shap
from sklearn.base import clone
# Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    f1_score, classification_report, recall_score,
    precision_recall_curve
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameter optimization
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Comet.ml for experiment tracking
from comet_ml import Experiment

# Utilities
import utils

# Joblib for model persistence
import joblib

#%%
# get api key from text file
COMET_API_KEY = open('comet_api_key.txt').read().strip()

# Create an experiment with your api key
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="Baseline model",
    workspace="2nd milestone",
    log_code=True
)
experiment.log_code(file_name='03_baseline_models.py')
#%%
def baseline_question1_2(experiment):
    '''

    '''
    #%%
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    #%%

    #X_train = train_base['shot_distance']
    #y_train = train_base['is_goal']
    #X_val = val_base['shot_distance']
    #y_val = val_base['is_goal']

    #%%

    features = ['shot_distance']
    target = ['is_goal']
    #%%
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('logistic_regression', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_base[features], train_base[target])

    #%%
    model_reg_filename = f"03_baseline_models.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Baseline Logistic", "calibration_curve"], 
                                 experiment = experiment)
    #%%


#%%
def baseline_question3(experiment):
    '''

    '''
    #%%
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    #%%
    # 1)

    features = ['shot_distance']
    target = ['is_goal']
    #%%
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('logistic_regression', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_base[features], train_base[target])

    #%%
    model_reg_filename = f"03_baseline_models.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Baseline Logistic", "calibration_curve"], 
                                 experiment = experiment)
    #%%
    # 2)

    features = ['shot_angle']
    target = ['is_goal']
    #%%
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('logistic_regression', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_base[features], train_base[target])

    #%%
    model_reg_filename = f"03_baseline_models.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Baseline Logistic", "calibration_curve"], 
                                 experiment = experiment)
    #%%
    # 3)

    features = ['shot_angle', 'shot_distance']
    target = ['is_goal']
    #%%
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('logistic_regression', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_base[features], train_base[target])

    #%%
    model_reg_filename = f"03_baseline_models.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Baseline Logistic", "calibration_curve"], 
                                 experiment = experiment)    
    #%% 
     
    # 4)

    features = ['shot_distance']
    target = ['is_goal']
    #%%
    # Create the pipeline
    class RandomModel:
        def __init__(self):
            pass
        
        def predict_proba(self, X):
            col_1 = np.random.uniform(size= len(X))
            col_2 = 1 - col_1
            return np.column_stack((col_1, col_2))
        
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.random.choice([0,1], size=len(X))

    # pipeline with random model
    pipeline_random = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('random_model', RandomModel())
    ])

    # Fit the pipeline
    pipeline_random.fit(train_base[features], train_base[target])

    #%%
    model_reg_filename = f"03_baseline_models.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = pipeline_random, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Baseline Logistic", "calibration_curve"], 
                                 experiment = experiment)
    
    #%%
    experiment.end()

    #%%